import aiohttp
import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from config import SolscanConfig
import ssl
import certifi

class SolscanAPI:
    """Client to interact with Solana JSON-RPC API for free balance retrieval"""
    
    def __init__(self, config):
        self.config = config
        self.rpc_url = "https://api.mainnet-beta.solana.com"
        self._session = None
        # Create SSL context with certifi certificates
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session



    async def get_account_balance(self, wallet_address: str) -> Optional[float]:
        """Get account SOL balance using Solana JSON-RPC API"""
        try:
            session = await self._get_session()
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getBalance",
                "params": [wallet_address]
            }
            
            async with session.post(self.rpc_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if "result" in data and "value" in data["result"]:
                        # Convert lamports to SOL (1 SOL = 1e9 lamports)
                        return float(data["result"]["value"]) / 1e9
                    else:
                        logging.error(f"Unexpected response format: {data}")
                else:
                    logging.error(f"Error response from Solana API: {response.status}")
                    return None
                
        except Exception as e:
            logging.error(f"Error getting account balance: {str(e)}")
            return None

    async def get_token_holdings(self, wallet_address: str) -> List[Dict]:
        """Get SPL token balances using free Solana API"""
        try:
            session = await self._get_session()
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [
                    wallet_address,
                    {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                    {"encoding": "jsonParsed"}
                ]
            }
            
            async with session.post(self.rpc_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if "result" in data and "value" in data["result"]:
                        return data["result"]["value"]
                return []
        except Exception as e:
            logging.error(f"Error getting token holdings: {str(e)}")
            return []

    async def get_account_transactions(self, wallet_address: str, limit: int = 20) -> List[Dict]:
        """Get recent transactions using free Solana API"""
        try:
            session = await self._get_session()
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [
                    wallet_address,
                    {"limit": limit}
                ]
            }
            
            async with session.post(self.rpc_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if "result" in data:
                        return data["result"]
                return []
        except Exception as e:
            logging.error(f"Error getting transactions: {str(e)}")
            return []

    async def get_token_metadata(self, token_address: str) -> Optional[Dict]:
        """Get token metadata"""
        try:
            session = await self._get_session()
            url = f"{self.config.base_url}/token/{token_address}"
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data')
                else:
                    logging.error(f"Error getting token metadata: {response.status}")
                    return None
        except Exception as e:
            logging.error(f"Error getting token metadata: {str(e)}")
            return None

    async def get_token_price(self, token_address: str) -> Optional[float]:
        """Get current token price in USD"""
        try:
            data = await self._make_request(f"token/price?address={token_address}")
            return float(data['data']['priceUsd']) if data and 'data' in data else None
        except Exception as e:
            logging.error(f"Error getting token price: {str(e)}")
            return None

    async def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make authenticated request to Solscan API"""
        for attempt in range(self.config.retries):
            try:
                session = await self._get_session()
                url = f"{self.base_url}/{endpoint}"
                async with session.get(url, params=params, timeout=self.config.timeout) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status in [401, 429]:
                        logging.warning(f"Solscan API error {response.status}, retrying...")
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    else:
                        logging.error(f"Solscan API error: {response.status} - {await response.text()}")
                        return None
            except Exception as e:
                logging.error(f"Request error (attempt {attempt + 1}): {str(e)}")
                await asyncio.sleep(self.config.retry_delay)
        return None

    async def close(self):
        """Close aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
