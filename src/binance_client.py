import logging
import asyncio
import time
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from config import BinanceConfig

class BinanceClient:
    def __init__(self, config: BinanceConfig):
        self.api_key = config.api_key
        self.api_secret = config.api_secret
        self.base_url = config.base_url
        self.testnet = config.testnet
        self.timeout = config.timeout
        self.retry_count = config.retries
        self.retry_delay = config.retry_delay
        self.rate_limit = config.rate_limit
        self.last_request_time = 0
        self.exchange = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize the Binance client"""
        options = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'timeout': self.timeout * 1000,  # ccxt uses ms
            'enableRateLimit': True
        }
        
        if self.testnet:
            options['urls'] = {
                'api': 'https://testnet.binance.vision/api'
            }
        
        self.exchange = ccxt.binance(options)
        self.logger.info("Binance client initialized")
    
    async def close(self):
        """Close the exchange connection"""
        if self.exchange:
            await self.exchange.close()
            self.logger.info("Binance client closed")
    
    async def _execute_with_retry(self, func, *args, **kwargs) -> Any:
        """Execute a function with retry logic"""
        for attempt in range(self.retry_count + 1):
            try:
                # Rate limiting
                now = time.time()
                elapsed = now - self.last_request_time
                if elapsed < 1.0 / self.rate_limit:
                    await asyncio.sleep(1.0 / self.rate_limit - elapsed)
                
                result = await func(*args, **kwargs)
                self.last_request_time = time.time()
                return result
                
            except ccxt.NetworkError as e:
                if attempt < self.retry_count:
                    delay = self.retry_delay * (2 ** attempt)
                    self.logger.warning(f"Network error: {str(e)}. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"Network error after {self.retry_count} retries: {str(e)}")
                    raise
                    
            except ccxt.ExchangeError as e:
                self.logger.error(f"Exchange error: {str(e)}")
                raise
                
            except Exception as e:
                self.logger.error(f"Unexpected error: {str(e)}")
                raise
    
    async def get_exchange_info(self) -> Dict:
        """Get exchange information"""
        return await self._execute_with_retry(self.exchange.fetch_markets)
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker information for a symbol"""
        return await self._execute_with_retry(self.exchange.fetch_ticker, symbol)
    
    async def get_account_balance(self) -> Dict:
        """Get account balance"""
        return await self._execute_with_retry(self.exchange.fetch_balance)
    
    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Get OHLCV data for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            limit: Number of candles to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        ohlcv = await self._execute_with_retry(
            self.exchange.fetch_ohlcv, symbol, timeframe, limit=limit
        )
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    async def create_market_order(self, symbol: str, side: str, amount: float) -> Dict:
        """
        Create a market order
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: Order side ('buy' or 'sell')
            amount: Order amount in base currency
            
        Returns:
            Order details
        """
        return await self._execute_with_retry(
            self.exchange.create_market_order, symbol, side, amount
        )
    
    async def create_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Dict:
        """
        Create a limit order
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: Order side ('buy' or 'sell')
            amount: Order amount in base currency
            price: Order price
            
        Returns:
            Order details
        """
        return await self._execute_with_retry(
            self.exchange.create_limit_order, symbol, side, amount, price
        )
    
    async def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancel an order by ID"""
        return await self._execute_with_retry(
            self.exchange.cancel_order, order_id, symbol
        )
    
    async def get_order(self, order_id: str, symbol: str) -> Dict:
        """Get order details by ID"""
        return await self._execute_with_retry(
            self.exchange.fetch_order, order_id, symbol
        )
    
    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders, optionally filtered by symbol"""
        return await self._execute_with_retry(
            self.exchange.fetch_open_orders, symbol
        )
    
    async def get_closed_orders(self, symbol: str = None, limit: int = 50) -> List[Dict]:
        """Get closed orders, optionally filtered by symbol"""
        return await self._execute_with_retry(
            self.exchange.fetch_closed_orders, symbol, limit=limit
        )
    
    async def get_trading_fees(self, symbol: str = None) -> Dict:
        """Get trading fees, optionally for a specific symbol"""
        return await self._execute_with_retry(
            self.exchange.fetch_trading_fees, symbol
        )
    
    async def get_market_depth(self, symbol: str, limit: int = 20) -> Dict:
        """Get order book / market depth for a symbol"""
        return await self._execute_with_retry(
            self.exchange.fetch_order_book, symbol, limit
        )