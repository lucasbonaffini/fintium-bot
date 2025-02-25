# trading_bot.py
import os
import asyncio
import logging
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from config import Config, DatabaseConfig, FilterConfig, BlacklistConfig, SolscanConfig
from telegram_handler import TelegramHandler
from rugcheck_client import RugCheckClient
from database import Database
from filter_manager import FilterManager
from models import Token, PriceHistory, MarketEvent, Blacklist
from unibot_client import UnibotSolanaClient
from dexscreener_client import DexScreenerClient
from security_manager import SecurityManager
from market_analyzer import MarketAnalyzer
from trade_manager import TradeManager
from sqlalchemy import text

class TradingBot:
    def __init__(self):
        # Load configuration
        self.config = Config()
        
        # Initialize database
        self.db = Database(self.config.db)
        
        # Initialize components
        self.telegram = TelegramHandler(
            os.getenv('TELEGRAM_BOT_TOKEN'),
            os.getenv('TELEGRAM_CHAT_ID')
        )
        self.telegram.set_trading_bot(self)
        
        # Initialize market components
        self.dexscreener = DexScreenerClient()
        self.security_manager = SecurityManager(self.db, self.config)
        self.market_analyzer = MarketAnalyzer(self.config)
        self.rugcheck = RugCheckClient(self.security_manager)
        
        # Initialize Unibot client
        self.unibot = UnibotSolanaClient(
            self.config,
            os.getenv('TELEGRAM_API_ID'),
            os.getenv('TELEGRAM_API_HASH'),
            os.getenv('TELEGRAM_PHONE')
        )
        
        # Initialize trade manager
        self.trade_manager = TradeManager(
            self.unibot,
            self.market_analyzer,
            self.dexscreener,
            self.rugcheck,
            self.security_manager
        )
        
        self.is_running = False
        self.last_market_check = None
        self.market_check_interval = 300  # 5 minutes
        
    async def initialize(self):
        """Initialize all components"""
        try:
            logging.info("Initializing database...")
            await self.db.initialize()
            
            logging.info("Initializing Telegram handler...")
            await self.telegram.initialize()
            
            logging.info("Initializing Unibot client...")
            await self.unibot.initialize()
            
            logging.info("Initializing market analyzer...")
            await self.security_manager.initialize()
            
            logging.info("Bot initialization completed successfully")
            await self.telegram.send_message("✅ Trading Bot initialized successfully!")
            
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}")
            await self.telegram.send_message(f"⚠️ Initialization error: {str(e)}")
            raise
            
    async def run(self):
        """Main bot loop"""
        try:
            self.is_running = True
            logging.info("Starting trading bot main loop...")
            await self.telegram.send_message("🚀 Trading Bot Started - Monitoring Market")
            
            while self.is_running:
                try:
                    # Check current positions
                    logging.info("Checking current positions...")
                    await self.trade_manager.monitor_active_trades()
                    
                    # Discover new opportunities
                    logging.info("Checking new trading pairs...")
                    await self.discover_trading_opportunities()
                    
                    # Market status check
                    await self._check_market_health()
                    
                    # Performance report
                    await self._send_performance_report()
                    
                    await asyncio.sleep(60)  # 1 minute between main cycles
                    
                except Exception as e:
                    logging.error(f"Error in main loop: {str(e)}")
                    await self.telegram.send_message(f"⚠️ Error in operation: {str(e)}")
                    await asyncio.sleep(60)
                    
        except KeyboardInterrupt:
            logging.info("Bot stopped by user")
        except Exception as e:
            logging.error(f"Fatal error in main loop: {str(e)}")
            await self.telegram.send_message(f"🚨 Fatal error: {str(e)}")
        finally:
            await self.cleanup()
            
    async def discover_trading_opportunities(self):
        """Discover and analyze new trading opportunities"""
        try:
            logging.info("Starting token discovery process...")
            
            # Get trending pairs
            trending_pairs = await self.dexscreener.get_trending_pairs()
            if not trending_pairs:
                return
                
            # Process each pair
            for pair_data in trending_pairs:
                if not self.is_running:
                    break
                    
                token_address = pair_data.get('address')
                if not token_address:
                    continue
                    
                # Process trading opportunity
                await self.trade_manager.process_trading_opportunity(pair_data)
                
            logging.info(f"Total unique tokens discovered: {len(trending_pairs)}")
            
            # Log discovered opportunities
            final_tokens = [p for p in trending_pairs if self._meets_initial_criteria(p)]
            logging.info(f"Final tokens after filtering: {len(final_tokens)}")
            
        except Exception as e:
            logging.error(f"Error discovering tokens: {str(e)}")

    def _meets_initial_criteria(self, token_data: Dict) -> bool:
        """Basic criteria check for token filtering"""
        try:
            # Minimum requirements
            min_liquidity = self.config.filters.min_liquidity
            min_market_cap = self.config.filters.min_market_cap
            min_holders = self.config.filters.min_holders
            
            # Check basic metrics
            if token_data.get('liquidity', 0) < min_liquidity:
                return False
            if token_data.get('market_cap', 0) < min_market_cap:
                return False
            if token_data.get('holders', 0) < min_holders:
                return False
                
            # Check name patterns
            token_name = token_data.get('name', '').lower()
            for pattern in self.config.filters.forbidden_names:
                if pattern in token_name:
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Error checking token criteria: {str(e)}")
            return False

    async def _check_market_health(self):
        """Check overall market health and component status"""
        try:
            current_time = datetime.now()
            if (self.last_market_check and 
                (current_time - self.last_market_check).total_seconds() < self.market_check_interval):
                return
                
            self.last_market_check = current_time
            
            # Get wallet status
            wallet_status = await self.unibot.get_wallet_status()
            if not wallet_status:
                raise Exception("Could not get wallet status")
                
            # Check DexScreener API
            dex_status = await self._check_dexscreener_health()
            
            # Check Unibot connection
            unibot_status = await self.unibot.check_connection()
            
            # Check database connection
            db_status = await self.db.check_connection()
            
            # Format status message
            status_msg = "🔍 Market Status Check\n\n"
            status_msg += f"{'✅' if dex_status else '❌'} DexScreener API: {'Operational' if dex_status else 'Error'}\n"
            status_msg += f"{'✅' if unibot_status else '❌'} Unibot Connection: {'Active' if unibot_status else 'Error'}\n"
            status_msg += f"💰 Wallet Status:\n"
            status_msg += f"Balance: {wallet_status.get('sol_balance', 'Error')}\n"
            status_msg += f"Value: {wallet_status.get('total_value_usd', 'Error')}\n"
            status_msg += f"{'✅' if db_status else '❌'} Database Connection: {'Active' if db_status else 'Error'}\n\n"
            status_msg += f"📊 Active Positions: {len(self.trade_manager.active_trades)}\n"
            
            await self.telegram.send_message(status_msg)
            
        except Exception as e:
            logging.error(f"Error in market status check: {str(e)}")
            await self.telegram.send_message(f"⚠️ Market status check error: {str(e)}")

    async def _check_dexscreener_health(self) -> bool:
        """Check DexScreener API health"""
        try:
            # Try to fetch a small number of pairs as health check
            pairs = await self.dexscreener.get_trending_pairs(limit=1)
            return len(pairs) > 0
        except Exception:
            return False

    async def _send_performance_report(self):
        """Send periodic performance report"""
        try:
            # Get trading statistics
            stats = self.trade_manager.get_trading_statistics()
            
            # Format report message
            report = "📊 24h Performance Report\n\n"
            report += f"Total Trades: {stats.get('total_trades', 0)}\n"
            report += f"Profitable Trades: {stats.get('profitable_trades', 0)}\n"
            report += f"Win Rate: {stats.get('win_rate', 0):.2f}%\n"
            report += f"Total PnL: ${stats.get('total_pnl', 0):.2f}\n\n"
            
            # Add top performers
            report += "🔝 Top Performers:\n"
            top_trades = self.trade_manager.get_top_trades(limit=3)
            for trade in top_trades:
                report += f"• {trade['symbol']}: {trade['pnl_percent']:.2f}%\n"
            
            await self.telegram.send_message(report)
            
        except Exception as e:
            logging.error(f"Error sending performance report: {str(e)}")

    async def cleanup(self):
        """Cleanup resources and connections"""
        try:
            self.is_running = False
            
            # Close Unibot connection
            await self.unibot.close()
            
            # Close database connection
            await self.db.close()
            
            # Close other connections
            await self.dexscreener.close()
            
            logging.info("Cleanup completed successfully")
            
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")

    async def handle_command(self, command: str, args: List[str] = None) -> str:
        """Handle bot commands"""
        try:
            if command == 'status':
                await self._check_market_health()
                return "Status check initiated"
                
            elif command == 'stats':
                await self._send_performance_report()
                return "Performance report generated"
                
            elif command == 'stop':
                self.is_running = False
                return "Bot stopping..."
                
            elif command == 'positions':
                positions = self.trade_manager.active_trades
                if not positions:
                    return "No active positions"
                    
                msg = "Current Positions:\n"
                for addr, data in positions.items():
                    msg += f"• {data.get('symbol', 'Unknown')}: Entry ${data.get('entry_price', 0):.4f}\n"
                return msg
                
            return "Unknown command"
            
        except Exception as e:
            logging.error(f"Error handling command: {str(e)}")
            return f"Error executing command: {str(e)}"
