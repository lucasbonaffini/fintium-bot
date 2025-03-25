import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from config import Config
from strategy import MovingAverageCrossover
from binance_client import BinanceClient
from database import Database
from models import Symbol, Trade, Candle

class TradeManager:
    def __init__(self, config: Config, binance_client: BinanceClient, database: Database):
        self.config = config
        self.binance_client = binance_client
        self.database = database
        self.strategy = MovingAverageCrossover(config.strategy)
        self.logger = logging.getLogger(__name__)
        self.active_trades = {}  # Dict of active trade IDs to trade objects
        self.trading_pairs = [config.default_pair]  # Start with default pair
        self.trade_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the trade manager"""
        self.logger.info("Initializing trade manager")
        
        # Load active trades from database
        open_trades = await self.database.get_open_trades()
        for trade in open_trades:
            symbol = await self.database.get_symbol_by_name(trade.symbol.symbol)
            if symbol:
                self.active_trades[trade.id] = trade
        
        self.logger.info(f"Loaded {len(self.active_trades)} active trades from database")
        
        # Setup trading pairs symbols in database
        for pair in self.trading_pairs:
            base, quote = pair.split('/')
            symbol = await self.database.get_or_create_symbol(
                symbol=pair,
                exchange="binance",
                base_asset=base,
                quote_asset=quote
            )
            
        self.logger.info("Trade manager initialized")
        
    async def update_market_data(self, timeframe: str = None):
        """
        Update market data for all trading pairs
        
        Args:
            timeframe: Timeframe to update, if None, use strategy timeframe
        """
        if timeframe is None:
            timeframe = self.config.strategy.timeframe
            
        for pair in self.trading_pairs:
            try:
                # Get symbol from database
                symbol = await self.database.get_symbol_by_name(pair)
                if not symbol:
                    self.logger.warning(f"Symbol {pair} not found in database")
                    continue
                
                # Fetch candle data from Binance
                ohlcv_df = await self.binance_client.get_ohlcv(
                    symbol=pair,
                    timeframe=timeframe,
                    limit=200  # Get enough data for indicators
                )
                
                if ohlcv_df.empty:
                    self.logger.warning(f"No OHLCV data returned for {pair}")
                    continue
                
                # Convert to list of dictionaries for database storage
                candles = []
                for timestamp, row in ohlcv_df.iterrows():
                    candles.append({
                        'timestamp': timestamp,
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    })
                
                # Save candles to database
                success = await self.database.save_candles(
                    symbol_id=symbol.id,
                    candles=candles,
                    timeframe=timeframe
                )
                
                if success:
                    self.logger.info(f"Updated {len(candles)} candles for {pair}")
                else:
                    self.logger.warning(f"Failed to update candles for {pair}")
                
            except Exception as e:
                self.logger.error(f"Error updating market data for {pair}: {str(e)}")
                
    async def check_for_signals(self) -> List[Dict]:
        """
        Check for trading signals across all pairs
        
        Returns:
            List of signal dictionaries with symbol, signal_type, and confidence
        """
        signals = []
        
        for pair in self.trading_pairs:
            try:
                # Get symbol from database
                symbol = await self.database.get_symbol_by_name(pair)
                if not symbol:
                    continue
                
                # Get candle data from database
                df = await self.database.get_candles_as_dataframe(
                    symbol_id=symbol.id,
                    timeframe=self.config.strategy.timeframe,
                    limit=200
                )
                
                if df.empty:
                    self.logger.warning(f"No candle data available for {pair}")
                    continue
                
                # Get signal from strategy
                signal_type, confidence = self.strategy.get_latest_signal(df)
                
                # Always log the signal regardless of type for debugging
                self.logger.info(f"Signal for {pair}: {signal_type} with {confidence:.2f} confidence (price: {df.iloc[-1]['close']})")
                
                # Log the last 5 candles and calculated indicators for debugging
                if len(df) >= 5:
                    last_candles = df.tail(5)
                    self.logger.info(f"Last 5 candles for {pair}:\n{last_candles}")
                    
                    # Add calculated indicators if they exist
                    with_indicators = self.strategy.calculate_indicators(df).tail(5)
                    if 'fast_ma' in with_indicators.columns and 'slow_ma' in with_indicators.columns:
                        self.logger.info(f"Indicators for {pair}:\nfast_ma: {with_indicators['fast_ma'].values}\nslow_ma: {with_indicators['slow_ma'].values}")
                        if 'crossover' in with_indicators.columns:
                            self.logger.info(f"Crossover values: {with_indicators['crossover'].values}")
                
                if signal_type != 'HOLD':
                    signals.append({
                        'symbol': symbol,
                        'signal_type': signal_type,
                        'confidence': confidence,
                        'price': df.iloc[-1]['close']
                    })
                    
                    self.logger.info(f"Found {signal_type} signal for {pair} with {confidence:.2f} confidence")
                
            except Exception as e:
                self.logger.error(f"Error checking signals for {pair}: {str(e)}")
                
        return signals
    
    async def execute_signals(self, signals: List[Dict]):
        """
        Execute trading signals
        
        Args:
            signals: List of signal dictionaries
        """
        if not signals:
            return
            
        for signal in signals:
            try:
                symbol = signal['symbol']
                signal_type = signal['signal_type']
                confidence = signal['confidence']
                current_price = signal['price']
                
                # Skip if we already have an active trade for this symbol
                active_for_symbol = [t for t in self.active_trades.values() 
                                     if t.symbol_id == symbol.id]
                
                if active_for_symbol:
                    self.logger.info(f"Skipping {signal_type} signal for {symbol.symbol} - already have active trade")
                    continue
                
                # Check if we've reached maximum active trades
                if len(self.active_trades) >= self.config.strategy.max_active_trades:
                    self.logger.info(f"Skipping {signal_type} signal for {symbol.symbol} - maximum active trades reached")
                    continue
                
                # Check if we've reached maximum trades per day
                # TODO: Implement daily trade limit check from database
                
                # Get account balance
                balance = await self.binance_client.get_account_balance()
                quote_currency = symbol.quote_asset
                available_balance = balance.get(quote_currency, {}).get('free', 0)
                
                # Log the balance information
                self.logger.info(f"Account balance: {balance}")
                self.logger.info(f"Available {quote_currency} balance: {available_balance}")
                
                if quote_currency == 'USDT':
                    # Calculate position size based on risk
                    account_value = available_balance
                    risk_amount = account_value * self.config.strategy.risk_per_trade
                    
                    # For BTC/USDT, convert USDT amount to BTC
                    if symbol.base_asset == 'BTC':
                        position_size = risk_amount / current_price
                    else:
                        position_size = risk_amount
                    
                    # Apply min/max limits
                    position_size = max(position_size, self.config.strategy.min_position_size)
                    position_size = min(position_size, self.config.strategy.max_position_size)
                    
                    # Round to appropriate precision for the symbol
                    # Hardcoded to 6 decimal places for BTC, should ideally come from exchange info
                    position_size = round(position_size, 6)
                    
                    # Skip if balance too low
                    cost = position_size * current_price
                    if cost > available_balance:
                        self.logger.warning(f"Insufficient balance for {signal_type} signal on {symbol.symbol}")
                        self.logger.warning(f"Required cost: {cost}, Available balance: {available_balance}")
                        self.logger.warning(f"Position size: {position_size}, Current price: {current_price}")
                        continue
                    
                    # Calculate take profit and stop loss
                    df = await self.database.get_candles_as_dataframe(
                        symbol_id=symbol.id,
                        timeframe=self.config.strategy.timeframe,
                        limit=50
                    )
                    
                    take_profit_pct = self.config.strategy.take_profit
                    stop_loss_pct = self.config.strategy.stop_loss
                    
                    take_profit, stop_loss = self.strategy.calculate_take_profit_stop_loss(
                        df, current_price, signal_type, take_profit_pct, stop_loss_pct
                    )
                    
                    # Execute the trade
                    async with self.trade_lock:
                        if signal_type == 'BUY':
                            order = await self.binance_client.create_market_order(
                                symbol=symbol.symbol,
                                side='buy',
                                amount=position_size
                            )
                            
                            if order:
                                # Create trade record in database
                                trade = await self.database.create_trade(
                                    symbol_id=symbol.id,
                                    trade_type='BUY',
                                    entry_price=order.get('price', current_price),
                                    quantity=position_size,
                                    entry_time=datetime.utcnow(),
                                    take_profit=take_profit,
                                    stop_loss=stop_loss,
                                    signal_type='CROSSOVER'
                                )
                                
                                if trade:
                                    self.active_trades[trade.id] = trade
                                    self.logger.info(f"Created BUY trade for {symbol.symbol} - {position_size} @ {current_price}")
                            
                        elif signal_type == 'SELL':
                            # For simplicity, we're assuming this is for opening a short position
                            # In reality, you would need margin/futures for shorting
                            self.logger.info(f"SELL signal for {symbol.symbol} - not implemented for spot trading")
                            
                            # If you're using futures, you could implement short entry here
                
            except Exception as e:
                self.logger.error(f"Error executing signal for {signal['symbol'].symbol}: {str(e)}")
    
    async def monitor_active_trades(self):
        """Monitor active trades for exit conditions"""
        if not self.active_trades:
            return
            
        trades_to_check = list(self.active_trades.values())
        
        for trade in trades_to_check:
            try:
                symbol = await self.database.get_symbol_by_name(trade.symbol.symbol)
                if not symbol:
                    continue
                
                # Get latest price data
                ticker = await self.binance_client.get_ticker(symbol.symbol)
                current_price = ticker.get('last', 0)
                
                if current_price <= 0:
                    continue
                
                # Check for take profit or stop loss
                if trade.trade_type == 'BUY':
                    # For long positions
                    take_profit_hit = trade.take_profit and current_price >= trade.take_profit
                    stop_loss_hit = trade.stop_loss and current_price <= trade.stop_loss
                else:
                    # For short positions
                    take_profit_hit = trade.take_profit and current_price <= trade.take_profit
                    stop_loss_hit = trade.stop_loss and current_price >= trade.stop_loss
                
                # Check for trend reversal signal
                df = await self.database.get_candles_as_dataframe(
                    symbol_id=symbol.id,
                    timeframe=self.config.strategy.timeframe,
                    limit=200
                )
                
                if not df.empty:
                    signal_type, _ = self.strategy.get_latest_signal(df)
                    
                    # Exit if signal is opposite to our position
                    signal_exit = (trade.trade_type == 'BUY' and signal_type == 'SELL') or \
                                 (trade.trade_type == 'SELL' and signal_type == 'BUY')
                else:
                    signal_exit = False
                
                # Exit the trade if any exit condition is met
                if take_profit_hit or stop_loss_hit or signal_exit:
                    exit_reason = "take profit" if take_profit_hit else \
                                  "stop loss" if stop_loss_hit else \
                                  "signal reversal"
                    
                    self.logger.info(f"Exiting {trade.trade_type} trade {trade.id} due to {exit_reason}")
                    
                    # Execute exit order
                    side = 'sell' if trade.trade_type == 'BUY' else 'buy'
                    
                    order = await self.binance_client.create_market_order(
                        symbol=symbol.symbol,
                        side=side,
                        amount=trade.quantity
                    )
                    
                    if order:
                        # Calculate PnL
                        exit_price = order.get('price', current_price)
                        
                        if trade.trade_type == 'BUY':
                            pnl = (exit_price - trade.entry_price) * trade.quantity
                            pnl_percent = ((exit_price / trade.entry_price) - 1) * 100
                        else:
                            pnl = (trade.entry_price - exit_price) * trade.quantity
                            pnl_percent = ((trade.entry_price / exit_price) - 1) * 100
                        
                        # Update trade in database
                        await self.database.update_trade(
                            trade_id=trade.id,
                            exit_price=exit_price,
                            exit_time=datetime.utcnow(),
                            status="CLOSED",
                            pnl=pnl,
                            pnl_percent=pnl_percent
                        )
                        
                        # Remove from active trades
                        if trade.id in self.active_trades:
                            del self.active_trades[trade.id]
                            
                        # Update trading stats
                        await self.database.update_trading_stats()
                        
                        self.logger.info(f"Closed trade {trade.id} with PnL: {pnl:.8f} ({pnl_percent:.2f}%)")
                
            except Exception as e:
                self.logger.error(f"Error monitoring trade {trade.id}: {str(e)}")
    
    async def calculate_performance(self):
        """Calculate and return trading performance summary"""
        try:
            # Get performance summary from database
            performance = await self.database.get_performance_summary()
            
            # Add active positions information
            active_positions_value = 0
            
            for trade_id, trade in self.active_trades.items():
                symbol = await self.database.get_symbol_by_name(trade.symbol.symbol)
                if not symbol:
                    continue
                
                ticker = await self.binance_client.get_ticker(symbol.symbol)
                current_price = ticker.get('last', 0)
                
                if current_price > 0:
                    position_value = trade.quantity * current_price
                    active_positions_value += position_value
            
            performance['active_positions'] = len(self.active_trades)
            performance['active_positions_value'] = active_positions_value
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error calculating performance: {str(e)}")
            return {}
    
    async def run_trading_cycle(self):
        """Execute one full trading cycle"""
        try:
            # Update market data
            await self.update_market_data()
            
            # Check for trading signals
            signals = await self.check_for_signals()
            
            # Execute signals if any
            await self.execute_signals(signals)
            
            # Monitor active trades
            await self.monitor_active_trades()
            
            # Calculate performance (for logging/reporting)
            performance = await self.calculate_performance()
            
            # Log summary
            active_count = performance.get('active_positions', 0)
            active_value = performance.get('active_positions_value', 0)
            total_pnl = performance.get('total_pnl', 0)
            
            self.logger.info(f"Trading cycle complete. Active positions: {active_count}, " +
                           f"Value: {active_value:.2f}, Total PnL: {total_pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {str(e)}")