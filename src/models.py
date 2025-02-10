# models.py
from sqlalchemy import Boolean, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from base import Base

class Token(Base):
    __tablename__ = 'tokens'
    
    id = Column(Integer, primary_key=True)
    address = Column(String, unique=True, nullable=False)
    symbol = Column(String)
    name = Column(String)
    chain = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    
    price_history = relationship("PriceHistory", back_populates="token")
    market_events = relationship("MarketEvent", back_populates="token")

class PriceHistory(Base):
    __tablename__ = 'price_history'
    
    id = Column(Integer, primary_key=True)
    token_id = Column(Integer, ForeignKey('tokens.id'))
    timestamp = Column(DateTime)
    price = Column(Float)
    volume_24h = Column(Float)
    liquidity = Column(Float)
    price_change_24h = Column(Float)
    holders = Column(Integer)
    market_cap = Column(Float)
    
    token = relationship("Token", back_populates="price_history")

class MarketEvent(Base):
    __tablename__ = 'market_events'
    
    id = Column(Integer, primary_key=True)
    token_id = Column(Integer, ForeignKey('tokens.id'))
    event_type = Column(String)
    timestamp = Column(DateTime)
    details = Column(Text)
    
    token = relationship("Token", back_populates="market_events")

class Blacklist(Base):
    __tablename__ = 'blacklist'
    
    address = Column(String(42), primary_key=True)
    type = Column(String(20))
    reason = Column(Text)
    listed_at = Column(DateTime, default=datetime.utcnow)
    confirmed = Column(Boolean, default=False)