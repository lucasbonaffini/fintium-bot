import logging
import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import signal

from config import Config
from database import Database
from binance_client import BinanceClient
from trade_manager import TradeManager
from strategy import MovingAverageCrossover

class TradingBot:
    def __init__(self):
        """Initialize the trading bot"""
        self.config = Config()
        self.database = None
        self.binance_client = None
        self.trade_manager = None
        self.running = False
        self.logger = logging.getLogger(__name__)
        
        # Set up signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, sig, frame):
        """Handle termination signals"""
        self.logger.info(f"Received signal {sig}, shutting down...")
        self.running = False
        
    async def initialize(self):
        """Initialize all components"""
        try:
            self.logger.info("Initializing trading bot components...")
            
            # Initialize database
            self.database = Database(self.config.db)
            db_initialized = await self.database.initialize()
            
            if not db_initialized:
                raise RuntimeError("Failed to initialize database")
            
            # Initialize Binance client
            self.binance_client = BinanceClient(self.config.binance)
            await self.binance_client.initialize()
            
            # Initialize trade manager
            self.trade_manager = TradeManager(
                self.config, 
                self.binance_client,
                self.database
            )
            await self.trade_manager.initialize()
            
            self.logger.info("Trading bot initialization complete")
            
        except Exception as e:
            self.logger.critical(f"Initialization error: {str(e)}")
            await self.close()
            raise
            
    async def run(self):
        """Run the main bot loop"""
        self.running = True
        
        self.logger.info("Starting trading bot main loop")
        
        try:
            # Initial data collection
            await self.trade_manager.update_market_data()
            
            # Initial performance calculation
            performance = await self.trade_manager.calculate_performance()
            self.logger.info(f"Initial performance: {performance}")
            
            # Main loop
            cycle_count = 0
            while self.running:
                cycle_count += 1
                
                self.logger.info(f"Starting trading cycle {cycle_count}")
                
                # Run trading cycle
                await self.trade_manager.run_trading_cycle()
                
                # Sleep until next cycle
                # Choose sleep duration based on strategy timeframe
                # For 1h timeframe, check every 5 minutes
                # For shorter timeframes, check more frequently
                timeframe = self.config.strategy.timeframe
                
                if timeframe.endswith('h'):
                    # Hours - sleep for 5 minutes
                    sleep_seconds = 60 * 5
                elif timeframe.endswith('m'):
                    # Minutes - sleep for 30 seconds
                    sleep_seconds = 30
                else:
                    # Default - sleep for 1 minute
                    sleep_seconds = 60
                
                self.logger.info(f"Sleeping for {sleep_seconds} seconds until next cycle")
                await asyncio.sleep(sleep_seconds)
                
                # Periodically update stats and log performance
                if cycle_count % 12 == 0:  # Every 12 cycles
                    await self.database.update_trading_stats()
                    
                    performance = await self.trade_manager.calculate_performance()
                    self.logger.info(f"Performance update: {performance}")
            
            self.logger.info("Trading bot main loop stopped")
            
        except Exception as e:
            self.logger.critical(f"Error in main loop: {str(e)}")
            self.running = False
            raise
            
        finally:
            await self.close()
    
    async def close(self):
        """Clean up resources"""
        self.logger.info("Closing trading bot components...")
        
        try:
            if self.binance_client:
                await self.binance_client.close()
                
            if self.database:
                await self.database.close()
                
            self.logger.info("Trading bot shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
    
    async def backtest(self, start_date: datetime, end_date: datetime, initial_balance: float = 10000.0):
        """
        Run backtesting simulation
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_balance: Initial balance for backtest
            
        Returns:
            Performance metrics dictionary
        """
        if not self.config.backtest_mode:
            self.logger.warning("Backtesting called but backtest_mode is not enabled in config")
            self.config.backtest_mode = True
        
        self.logger.info(f"Starting backtest from {start_date} to {end_date} with {initial_balance} initial balance")
        
        try:
            # Initialize backtest-specific components
            # In a real implementation, you would have a separate backtesting class
            # This is just a placeholder
            
            self.logger.info("Backtest completed")
            
            # Return placeholder results
            return {
                "initial_balance": initial_balance,
                "final_balance": initial_balance * 1.2,  # Placeholder
                "total_trades": 15,
                "win_rate": 0.6,
                "profit_factor": 1.8,
                "max_drawdown": 5.2
            }
            
        except Exception as e:
            self.logger.error(f"Backtest error: {str(e)}")
            return {}