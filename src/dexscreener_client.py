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
        connector = aiohttp.TCPConnector(ssl=self.ssl_context)
        self.session = aiohttp.ClientSession(connector=connector)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_token_data(self, token_address: str) -> Optional[Dict]:
        """Fetch comprehensive token data from DexScreener"""
        try:
            url = f"{self.base_url}/tokens/{token_address}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_token_data(data)
                return None
        except Exception as e:
            logging.error(f"DexScreener API error: {str(e)}")
            return None

    async def get_trending_pairs(self, chain: str = 'solana') -> List[Dict]:
        """Fetch trending pairs for a specific chain"""
        try:
            url = f"{self.base_url}/dex/pairs/{chain}/trending"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('pairs', [])
                return []
        except Exception as e:
            logging.error(f"Error fetching trending pairs: {str(e)}")
            return []

    async def get_recent_pairs(self, chain: str = 'solana') -> List[Dict]:
        """Fetch recent pairs for a specific chain"""
        try:
            url = f"{self.base_url}/dex/pairs/{chain}/recent"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('pairs', [])
                return []
        except Exception as e:
            logging.error(f"Error fetching recent pairs: {str(e)}")
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
            'created_at': main_pair.get('createTime'),
            'holders': main_pair.get('holders', 0),
            'total_supply': main_pair.get('baseToken', {}).get('totalSupply', 0),
            'chain': main_pair.get('chainId', 'solana')
        }

    async def search_pairs(self, query: str) -> List[Dict]:
        """Search for pairs matching a query"""
        try:
            url = f"{self.base_url}/dex/search"
            params = {"query": query}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('pairs', [])
                return []
        except Exception as e:
            logging.error(f"Error searching pairs: {str(e)}")
            return []