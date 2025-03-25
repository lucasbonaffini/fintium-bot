# models.py
from sqlalchemy import Boolean, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from base import Base

class Symbol(Base):
    __tablename__ = 'symbols'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, unique=True, nullable=False)
    exchange = Column(String)
    base_asset = Column(String)  # BTC, ETH, etc.
    quote_asset = Column(String)  # USDT, USD, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    candles = relationship("Candle", back_populates="symbol")
    trades = relationship("Trade", back_populates="symbol")

class Candle(Base):
    __tablename__ = 'candles'
    
    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'))
    timestamp = Column(DateTime, index=True)
    timeframe = Column(String)  # 1m, 5m, 15m, 1h, etc.
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    
    symbol = relationship("Symbol", back_populates="candles")

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'))
    trade_type = Column(String)  # BUY, SELL
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Float)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime, nullable=True)
    pnl = Column(Float, nullable=True)
    pnl_percent = Column(Float, nullable=True)
    status = Column(String)  # OPEN, CLOSED, CANCELED
    take_profit = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    signal_type = Column(String)  # CROSSOVER, MANUAL, etc.
    
    symbol = relationship("Symbol", back_populates="trades")

class TradingStats(Base):
    __tablename__ = 'trading_stats'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.utcnow, index=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    avg_profit = Column(Float, default=0.0)
    avg_loss = Column(Float, default=0.0)
    largest_profit = Column(Float, default=0.0)
    largest_loss = Column(Float, default=0.0)