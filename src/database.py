from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from models import Base, Token, PriceHistory, MarketEvent, Blacklist

class Database:
    def __init__(self, config):
        self.config = config
        self.engine = None
        self.SessionLocal = None

    async def initialize(self):
        """Initialize database connection and create tables"""
        try:
            # Create database URL
            db_url = f"postgresql://{self.config.user}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
            
            # Create engine
            self.engine = create_engine(db_url)
            
            # Create tables
            Base.metadata.create_all(self.engine)
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Test connection
            await self.check_connection()
            logging.info("Database connection established successfully")
            
        except Exception as e:
            logging.error(f"Database initialization error: {str(e)}")
            raise

    async def check_connection(self) -> bool:
        """Check database connection"""
        try:
            if not self.SessionLocal:
                return False
                
            with self.SessionLocal() as session:
                session.execute(text('SELECT 1'))
                return True
                
        except Exception as e:
            logging.error(f"Database session error: {str(e)}")
            return False

    async def save_token_data(self, token_data: Dict):
        """Save or update token data"""
        try:
            with self.SessionLocal() as session:
                # Check if token exists
                token = session.query(Token).filter(
                    Token.address == token_data['address']
                ).first()
                
                if not token:
                    token = Token(
                        address=token_data['address'],
                        symbol=token_data.get('symbol'),
                        name=token_data.get('name'),
                        chain=token_data.get('chain', 'solana'),
                        created_at=datetime.now()
                    )
                    session.add(token)
                
                # Update price history
                price_history = PriceHistory(
                    token_id=token.id,
                    timestamp=datetime.now(),
                    price=token_data.get('price', 0),
                    volume_24h=token_data.get('volume_24h', 0),
                    liquidity=token_data.get('liquidity', 0),
                    price_change_24h=token_data.get('price_change_24h', 0),
                    holders=token_data.get('holders', 0),
                    market_cap=token_data.get('market_cap', 0)
                )
                session.add(price_history)
                
                # Add market event if significant change
                if abs(token_data.get('price_change_24h', 0)) > 20:
                    event = MarketEvent(
                        token_id=token.id,
                        event_type='PRICE_MOVEMENT',
                        timestamp=datetime.now(),
                        details=f"Price changed by {token_data.get('price_change_24h')}%"
                    )
                    session.add(event)
                
                session.commit()
                
        except Exception as e:
            logging.error(f"Error saving token data: {str(e)}")
            session.rollback()

    async def get_token_history(self, token_address: str, hours: int = 24) -> List[Dict]:
        """Get token price history"""
        try:
            with self.SessionLocal() as session:
                token = session.query(Token).filter(
                    Token.address == token_address
                ).first()
                
                if not token:
                    return []
                
                history = session.query(PriceHistory).filter(
                    PriceHistory.token_id == token.id,
                    PriceHistory.timestamp >= datetime.now() - timedelta(hours=hours)
                ).order_by(PriceHistory.timestamp.asc()).all()
                
                return [
                    {
                        'timestamp': h.timestamp,
                        'price': h.price,
                        'volume_24h': h.volume_24h,
                        'liquidity': h.liquidity
                    }
                    for h in history
                ]
                
        except Exception as e:
            logging.error(f"Error getting token history: {str(e)}")
            return []

    async def save_trade(self, trade_data: Dict):
        """Save trade to database"""
        try:
            with self.SessionLocal() as session:
                token = session.query(Token).filter(
                    Token.address == trade_data['token_address']
                ).first()
                
                if not token:
                    return
                
                event = MarketEvent(
                    token_id=token.id,
                    event_type=trade_data['type'],
                    timestamp=datetime.now(),
                    details=f"Trade executed at {trade_data.get('price')} USD"
                )
                session.add(event)
                session.commit()
                
        except Exception as e:
            logging.error(f"Error saving trade: {str(e)}")
            session.rollback()

    async def close(self):
        """Close database connection"""
        try:
            if self.engine:
                self.engine.dispose()
                
        except Exception as e:
            logging.error(f"Error closing database: {str(e)}")