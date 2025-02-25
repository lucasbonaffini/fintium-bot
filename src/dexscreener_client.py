import aiohttp
import logging
import ssl
import certifi
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class DexScreenerClient:
    def __init__(self):
        self.base_url = "https://api.dexscreener.com/latest"
        self.session = None
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self.ssl_context))
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_token_data(self, token_address: str) -> Optional[Dict]:
        """Fetch comprehensive token data from DexScreener"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self.ssl_context))
                
            url = f"{self.base_url}/tokens/{token_address}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_token_data(data)
                return None
        except Exception as e:
            logging.error(f"DexScreener API error: {str(e)}")
            return None

    async def get_trending_pairs(self, chain: str = 'solana', limit: int = 100) -> List[Dict]:
        """Fetch trending pairs with improved error handling"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self.ssl_context))
                
            url = f"{self.base_url}/dex/pairs/{chain}/trending"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get('pairs', [])
                    return [self._process_pair_data(pair) for pair in pairs[:limit]]
                return []
        except Exception as e:
            logging.error(f"Error fetching trending pairs: {str(e)}")
            return []

    def _process_token_data(self, data: Dict) -> Dict:
        """Process and enrich raw token data"""
        pairs = data.get('pairs', [])
        if not pairs:
            return {}

        main_pair = max(pairs, key=lambda x: float(x.get('liquidity', {}).get('usd', 0)))
        
        return {
            'address': main_pair.get('baseToken', {}).get('address'),
            'name': main_pair.get('baseToken', {}).get('name'),
            'symbol': main_pair.get('baseToken', {}).get('symbol'),
            'price': float(main_pair.get('priceUsd', 0)),
            'price_change_24h': float(main_pair.get('priceChange', {}).get('h24', 0)),
            'volume_24h': float(main_pair.get('volume', {}).get('h24', 0)),
            'liquidity': float(main_pair.get('liquidity', {}).get('usd', 0)),
            'market_cap': float(main_pair.get('marketCap', 0)),
            'created_at': main_pair.get('pairCreatedAt'),
            'holders': main_pair.get('holders', 0),
            'total_supply': main_pair.get('baseToken', {}).get('totalSupply', 0),
            'dex': main_pair.get('dexId'),
            'chain': main_pair.get('chainId', 'solana')
        }

    def _process_pair_data(self, pair: Dict) -> Dict:
        """Process pair data with relevant trading metrics"""
        return {
            'address': pair.get('baseToken', {}).get('address'),
            'name': pair.get('baseToken', {}).get('name'),
            'symbol': pair.get('baseToken', {}).get('symbol'),
            'price': float(pair.get('priceUsd', 0)),
            'price_change_24h': float(pair.get('priceChange', {}).get('h24', 0)),
            'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
            'liquidity': float(pair.get('liquidity', {}).get('usd', 0)),
            'market_cap': float(pair.get('marketCap', 0)),
            'created_at': pair.get('pairCreatedAt'),
            'momentum_score': self._calculate_momentum_score(pair),
            'risk_score': self._calculate_risk_score(pair)
        }

    def _calculate_momentum_score(self, pair: Dict) -> float:
        """Calculate momentum score based on price and volume metrics"""
        try:
            price_change = float(pair.get('priceChange', {}).get('h24', 0))
            volume_change = float(pair.get('volume', {}).get('h24', 0))
            market_cap = float(pair.get('marketCap', 0))
            liquidity = float(pair.get('liquidity', {}).get('usd', 0))
            
            # Weighted scoring
            price_weight = 0.4
            volume_weight = 0.3
            liquidity_weight = 0.3
            
            price_score = min(max(price_change / 100, -1), 1)
            volume_score = min(volume_change / market_cap if market_cap > 0 else 0, 1)
            liquidity_score = min(liquidity / market_cap if market_cap > 0 else 0, 1)
            
            return (price_score * price_weight + 
                   volume_score * volume_weight + 
                   liquidity_score * liquidity_weight)
                   
        except Exception:
            return 0

    def _calculate_risk_score(self, pair: Dict) -> float:
        """Calculate risk score based on various metrics"""
        try:
            market_cap = float(pair.get('marketCap', 0))
            liquidity = float(pair.get('liquidity', {}).get('usd', 0))
            holders = int(pair.get('holders', 0))
            
            # Risk factors
            low_liquidity = liquidity < 50000
            low_holders = holders < 100
            high_price_volatility = abs(float(pair.get('priceChange', {}).get('h24', 0))) > 50
            
            risk_score = 0
            if low_liquidity: risk_score += 0.4
            if low_holders: risk_score += 0.3
            if high_price_volatility: risk_score += 0.3
            
            return min(risk_score, 1.0)
            
        except Exception:
            return 1.0  # Maximum risk for error cases