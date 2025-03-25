import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from config import DatabaseConfig
from base import Base
from models import Symbol, Candle, Trade, TradingStats

class Database:
    def __init__(self, config: DatabaseConfig):
        self.host = config.host
        self.port = config.port
        self.database = config.database
        self.user = config.user
        self.password = config.password
        self.engine = None
        self.Session = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize the database connection"""
        try:
            connection_string = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            self.engine = create_engine(connection_string)
            self.Session = sessionmaker(bind=self.engine)
            
            # Create all tables if they don't exist
            Base.metadata.create_all(self.engine)
            
            # Test connection
            await self.check_connection()
            
            self.logger.info("Database connection initialized successfully")
            return True
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            return False
    
    async def check_connection(self) -> bool:
        """Check database connection"""
        try:
            if not self.Session:
                return False
                
            session = self.Session()
            session.execute(text('SELECT 1'))
            session.close()
            return True
                
        except Exception as e:
            self.logger.error(f"Database session error: {str(e)}")
            return False
            
    def get_session(self) -> Session:
        """Get a new database session"""
        if not self.Session:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.Session()
        
    async def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Database connection closed")
            
    # Symbol operations
    
    async def get_or_create_symbol(self, symbol: str, exchange: str, base_asset: str, quote_asset: str) -> Symbol:
        """Get or create a symbol in the database"""
        session = self.get_session()
        try:
            db_symbol = session.query(Symbol).filter_by(symbol=symbol).first()
            
            if not db_symbol:
                db_symbol = Symbol(
                    symbol=symbol,
                    exchange=exchange,
                    base_asset=base_asset,
                    quote_asset=quote_asset
                )
                session.add(db_symbol)
                session.commit()
                self.logger.info(f"Created new symbol record: {symbol}")
            
            return db_symbol
                
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error in get_or_create_symbol: {str(e)}")
            raise
        finally:
            session.close()
            
    async def get_symbol_by_name(self, symbol: str) -> Optional[Symbol]:
        """Get a symbol by name"""
        session = self.get_session()
        try:
            return session.query(Symbol).filter_by(symbol=symbol).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error in get_symbol_by_name: {str(e)}")
            return None
        finally:
            session.close()
    
    # Candle operations
    
    async def save_candles(self, symbol_id: int, candles: List[Dict], timeframe: str) -> bool:
        """
        Save a list of candles to the database
        
        Args:
            symbol_id: Symbol ID
            candles: List of candle dictionaries with keys: timestamp, open, high, low, close, volume
            timeframe: Candle timeframe (e.g., '1m', '5m', '1h')
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session()
        try:
            for candle_data in candles:
                timestamp = candle_data.get('timestamp')
                
                # Check if candle already exists
                existing = session.query(Candle).filter_by(
                    symbol_id=symbol_id,
                    timestamp=timestamp,
                    timeframe=timeframe
                ).first()
                
                if not existing:
                    # Create new candle
                    candle = Candle(
                        symbol_id=symbol_id,
                        timestamp=timestamp,
                        timeframe=timeframe,
                        open=candle_data.get('open'),
                        high=candle_data.get('high'),
                        low=candle_data.get('low'),
                        close=candle_data.get('close'),
                        volume=candle_data.get('volume')
                    )
                    session.add(candle)
                else:
                    # Update existing candle
                    existing.open = candle_data.get('open')
                    existing.high = candle_data.get('high')
                    existing.low = candle_data.get('low')
                    existing.close = candle_data.get('close')
                    existing.volume = candle_data.get('volume')
            
            session.commit()
            return True
            
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error in save_candles: {str(e)}")
            return False
        finally:
            session.close()
    
    async def get_candles(self, symbol_id: int, timeframe: str, limit: int = 100) -> List[Candle]:
        """
        Get candles for a symbol
        
        Args:
            symbol_id: Symbol ID
            timeframe: Candle timeframe
            limit: Maximum number of candles to return
            
        Returns:
            List of Candle objects
        """
        session = self.get_session()
        try:
            candles = session.query(Candle).filter_by(
                symbol_id=symbol_id,
                timeframe=timeframe
            ).order_by(Candle.timestamp.desc()).limit(limit).all()
            
            # Return in ascending order (oldest first)
            return list(reversed(candles))
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error in get_candles: {str(e)}")
            return []
        finally:
            session.close()
    
    async def get_candles_as_dataframe(self, symbol_id: int, timeframe: str, limit: int = 100) -> Any:
        """Get candles as pandas DataFrame"""
        import pandas as pd
        
        session = self.get_session()
        try:
            candles = await self.get_candles(symbol_id, timeframe, limit)
            
            if not candles:
                return pd.DataFrame()
                
            # Convert to DataFrame
            data = {
                'timestamp': [c.timestamp for c in candles],
                'open': [c.open for c in candles],
                'high': [c.high for c in candles],
                'low': [c.low for c in candles],
                'close': [c.close for c in candles],
                'volume': [c.volume for c in candles]
            }
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in get_candles_as_dataframe: {str(e)}")
            return pd.DataFrame()
        finally:
            session.close()
    
    # Trade operations
    
    async def create_trade(self, symbol_id: int, trade_type: str, entry_price: float, 
                    quantity: float, entry_time: datetime, take_profit: float = None, 
                    stop_loss: float = None, signal_type: str = "CROSSOVER") -> Optional[Trade]:
        """Create a new trade record"""
        session = self.get_session()
        try:
            trade = Trade(
                symbol_id=symbol_id,
                trade_type=trade_type,
                entry_price=entry_price,
                quantity=quantity,
                entry_time=entry_time,
                status="OPEN",
                take_profit=take_profit,
                stop_loss=stop_loss,
                signal_type=signal_type
            )
            session.add(trade)
            session.commit()
            self.logger.info(f"Created new trade: {trade_type} {quantity} at {entry_price}")
            return trade
            
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error in create_trade: {str(e)}")
            return None
        finally:
            session.close()
    
    async def update_trade(self, trade_id: int, exit_price: float = None, exit_time: datetime = None, 
                    status: str = None, pnl: float = None, pnl_percent: float = None) -> bool:
        """Update a trade record"""
        session = self.get_session()
        try:
            trade = session.query(Trade).filter_by(id=trade_id).first()
            
            if not trade:
                self.logger.warning(f"Trade not found: {trade_id}")
                return False
                
            if exit_price is not None:
                trade.exit_price = exit_price
                
            if exit_time is not None:
                trade.exit_time = exit_time
                
            if status is not None:
                trade.status = status
                
            if pnl is not None:
                trade.pnl = pnl
                
            if pnl_percent is not None:
                trade.pnl_percent = pnl_percent
                
            session.commit()
            self.logger.info(f"Updated trade {trade_id}: {status}")
            return True
            
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error in update_trade: {str(e)}")
            return False
        finally:
            session.close()
    
    async def get_open_trades(self) -> List[Trade]:
        """Get all open trades"""
        session = self.get_session()
        try:
            return session.query(Trade).filter_by(status="OPEN").all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error in get_open_trades: {str(e)}")
            return []
        finally:
            session.close()
    
    async def get_trade_by_id(self, trade_id: int) -> Optional[Trade]:
        """Get a trade by ID"""
        session = self.get_session()
        try:
            return session.query(Trade).filter_by(id=trade_id).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error in get_trade_by_id: {str(e)}")
            return None
        finally:
            session.close()
    
    async def get_trades_for_symbol(self, symbol_id: int, status: str = None, limit: int = 100) -> List[Trade]:
        """Get trades for a symbol, optionally filtered by status"""
        session = self.get_session()
        try:
            query = session.query(Trade).filter_by(symbol_id=symbol_id)
            
            if status:
                query = query.filter_by(status=status)
                
            return query.order_by(Trade.entry_time.desc()).limit(limit).all()
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error in get_trades_for_symbol: {str(e)}")
            return []
        finally:
            session.close()
    
    # Trading statistics operations
    
    async def update_trading_stats(self, date: datetime = None) -> bool:
        """
        Update trading statistics for a specific date or today
        
        Args:
            date: Date to update stats for (defaults to today)
            
        Returns:
            True if successful, False otherwise
        """
        if not date:
            date = datetime.utcnow().date()
        
        session = self.get_session()
        try:
            # Get all trades closed on the specified date
            start_date = datetime.combine(date, datetime.min.time())
            end_date = datetime.combine(date, datetime.max.time())
            
            trades = session.query(Trade).filter(
                Trade.exit_time.between(start_date, end_date),
                Trade.status == "CLOSED"
            ).all()
            
            if not trades:
                self.logger.info(f"No trades found for {date}")
                return True
                
            # Calculate statistics
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t.pnl > 0)
            losing_trades = sum(1 for t in trades if t.pnl < 0)
            total_pnl = sum(t.pnl for t in trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            profits = [t.pnl for t in trades if t.pnl > 0]
            losses = [t.pnl for t in trades if t.pnl < 0]
            
            avg_profit = sum(profits) / len(profits) if profits else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            largest_profit = max(profits) if profits else 0
            largest_loss = min(losses) if losses else 0
            
            # Update or create stats record
            stats = session.query(TradingStats).filter(
                func.date(TradingStats.date) == date
            ).first()
            
            if not stats:
                stats = TradingStats(
                    date=start_date,
                    total_trades=total_trades,
                    winning_trades=winning_trades,
                    losing_trades=losing_trades,
                    total_pnl=total_pnl,
                    win_rate=win_rate,
                    avg_profit=avg_profit,
                    avg_loss=avg_loss,
                    largest_profit=largest_profit,
                    largest_loss=largest_loss
                )
                session.add(stats)
            else:
                stats.total_trades = total_trades
                stats.winning_trades = winning_trades
                stats.losing_trades = losing_trades
                stats.total_pnl = total_pnl
                stats.win_rate = win_rate
                stats.avg_profit = avg_profit
                stats.avg_loss = avg_loss
                stats.largest_profit = largest_profit
                stats.largest_loss = largest_loss
            
            session.commit()
            self.logger.info(f"Updated trading stats for {date}")
            return True
            
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error in update_trading_stats: {str(e)}")
            return False
        finally:
            session.close()
    
    async def get_trading_stats(self, days: int = 30) -> List[TradingStats]:
        """Get trading statistics for the last N days"""
        session = self.get_session()
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            return session.query(TradingStats).filter(
                TradingStats.date.between(start_date, end_date)
            ).order_by(TradingStats.date.desc()).all()
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error in get_trading_stats: {str(e)}")
            return []
        finally:
            session.close()
            
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of trading performance"""
        session = self.get_session()
        try:
            # Overall statistics
            all_closed_trades = session.query(Trade).filter_by(status="CLOSED").all()
            
            if not all_closed_trades:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0,
                    "total_pnl": 0,
                    "avg_profit": 0,
                    "avg_loss": 0,
                    "largest_profit": 0,
                    "largest_loss": 0,
                    "profit_factor": 0,
                    "average_trade": 0
                }
                
            total_trades = len(all_closed_trades)
            winning_trades = sum(1 for t in all_closed_trades if t.pnl > 0)
            losing_trades = sum(1 for t in all_closed_trades if t.pnl < 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            total_pnl = sum(t.pnl for t in all_closed_trades)
            
            profits = [t.pnl for t in all_closed_trades if t.pnl > 0]
            losses = [t.pnl for t in all_closed_trades if t.pnl < 0]
            
            avg_profit = sum(profits) / len(profits) if profits else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            largest_profit = max(profits) if profits else 0
            largest_loss = min(losses) if losses else 0
            
            # Additional metrics
            total_profit = sum(profits)
            total_loss = abs(sum(losses)) if losses else 1  # Avoid division by zero
            profit_factor = total_profit / total_loss if total_loss else 0
            average_trade = total_pnl / total_trades if total_trades > 0 else 0
            
            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
                "largest_profit": largest_profit,
                "largest_loss": largest_loss,
                "profit_factor": profit_factor,
                "average_trade": average_trade
            }
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error in get_performance_summary: {str(e)}")
            return {}
        finally:
            session.close()