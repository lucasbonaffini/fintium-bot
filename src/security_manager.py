# security_manager.py
import os
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import aiohttp
from models import Blacklist, Token, PriceHistory
from sqlalchemy import func
import asyncio

class SecurityManager:
    def __init__(self, db, config):
        self.db = db
        self.config = config
        self.model = IsolationForest(n_estimators=100, contamination=0.01)
        self.historical_data = self._load_historical_data()
        self.goplus_base_url = "https://api.gopluslabs.io/api/v1"
        self.goplus_api_key = os.getenv('GOPLUS_API_KEY', '')  # Si tienes API key
        
    async def initialize(self):
        """Initialize the SecurityManager asynchronously"""
        self.historical_data = await self._load_historical_data()   

    async def _load_historical_data(self) -> pd.DataFrame:
        """Load historical data for anomaly detection"""
        try:
            async with self.db.get_session() as session:
                query = """
                    SELECT price, volume_24h, liquidity 
                    FROM price_history 
                    ORDER BY timestamp DESC 
                    LIMIT 100000
                """
                df = pd.read_sql(query, session.bind)
                return df
        except Exception as e:
            logging.error(f"Error loading historical data: {str(e)}")
            return pd.DataFrame()

    async def check_token_security(self, token_address: str, chain: str = "solana") -> Tuple[bool, str]:
        """Comprehensive security check using multiple sources"""
        try:
            # Check local blacklist first
            is_blacklisted, reason = await self.is_blacklisted(token_address)
            if is_blacklisted:
                return False, f"Token blacklisted: {reason}"

            # Check with GoPlus
            goplus_safe = await self.check_goplus_security(token_address, chain)
            if not goplus_safe:
                return False, "Failed GoPlus security checks"

            # Check for anomalies
            token_data = await self.get_token_data(token_address)
            if token_data and self.detect_anomalies(token_data):
                return False, "Anomalous behavior detected"

            return True, "Passed all security checks"

        except Exception as e:
            logging.error(f"Error in security check: {str(e)}")
            return False, f"Error during security check: {str(e)}"

    async def check_goplus_security(self, token_address: str, chain: str) -> bool:
        """Check token security using GoPlus API"""
        try:
            headers = {"Authorization": f"Bearer {self.goplus_api_key}"} if self.goplus_api_key else {}
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.goplus_base_url}/token_security/{chain}"
                params = {"contract_addresses": token_address}
                
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status != 200:
                        return False
                        
                    data = await response.json()
                    if not data.get('success'):
                        return False

                    token_data = data.get('result', {}).get(token_address.lower(), {})
                    
                    # Analyze security indicators
                    if self._analyze_goplus_indicators(token_data):
                        await self.add_to_blacklist(
                            token_address,
                            'coin',
                            'Failed GoPlus security checks'
                        )
                        return False

                    return True

        except Exception as e:
            logging.error(f"Error checking GoPlus security: {str(e)}")
            return False

    def _analyze_goplus_indicators(self, token_data: Dict) -> bool:
        """Analyze security indicators from GoPlus"""
        try:
            risk_indicators = [
                token_data.get('is_honeypot', False),
                token_data.get('is_proxy', True),
                token_data.get('is_blacklisted', False),
                token_data.get('is_mintable', True),
                token_data.get('owner_change_balance', True),
                float(token_data.get('holder_count', 0)) < 100,
                float(token_data.get('total_supply', 0)) == 0,
                token_data.get('cannot_sell_all', True)
            ]
            
            return any(risk_indicators)

        except Exception as e:
            logging.error(f"Error analyzing GoPlus indicators: {str(e)}")
            return True  # Conservative approach: return risky if analysis fails

    async def add_to_blacklist(self, address: str, type: str, reason: str):
        """Add address to blacklist with reason"""
        try:
            with self.db.get_session() as session:
                blacklist_entry = Blacklist(
                    address=address,
                    type=type,
                    reason=reason,
                    listed_at=datetime.utcnow()
                )
                session.merge(blacklist_entry)
                session.commit()
                logging.info(f"Added {type} to blacklist: {address}")
        except Exception as e:
            logging.error(f"Error adding to blacklist: {str(e)}")

    async def is_blacklisted(self, address: str) -> Tuple[bool, Optional[str]]:
        """Check if address is blacklisted"""
        try:
            with self.db.get_session() as session:
                entry = session.query(Blacklist).\
                    filter(Blacklist.address == address).\
                    first()
                if entry:
                    return True, entry.reason
                return False, None
        except Exception as e:
            logging.error(f"Error checking blacklist: {str(e)}")
            return False, None

    def detect_anomalies(self, token_data: Dict) -> bool:
        """Detect anomalies in token metrics"""
        try:
            features = pd.DataFrame([{
                'price': token_data['price'],
                'volume_24h': token_data['volume_24h'],
                'liquidity': token_data['liquidity']
            }])
            
            features = np.log1p(features)
            anomaly_score = self.model.decision_function(features)
            return anomaly_score[0] < -0.5
            
        except Exception as e:
            logging.error(f"Error in anomaly detection: {str(e)}")
            return False

    async def detect_blacklist_pattern(self, token_data: Dict) -> bool:
        """Detect patterns matching known blacklist characteristics"""
        try:
            with self.db.get_session() as session:
                similar_coins = session.query(Blacklist).\
                    filter(Blacklist.type == 'coin').\
                    filter(func.similarity(Blacklist.address, token_data['address']) > 0.8).\
                    count()
                
                similar_devs = session.query(Blacklist).\
                    filter(Blacklist.type == 'dev').\
                    filter(func.similarity(Blacklist.address, token_data.get('creator_address', '')) > 0.8).\
                    count()
                
                return similar_coins > 0 or similar_devs > 0
                
        except Exception as e:
            logging.error(f"Error detecting blacklist pattern: {str(e)}")
            return False

    async def refresh_security_data(self):
        """Refresh security data from all sources"""
        try:
            await self.refresh_goplus_data()
            await self.update_anomaly_model()
            await self.cleanup_old_entries()
            
            stats = await self.get_blacklist_stats()
            logging.info(f"Security data refreshed. Current stats: {stats}")
            
        except Exception as e:
            logging.error(f"Error refreshing security data: {str(e)}")

    async def refresh_goplus_data(self):
        """Refresh token data from GoPlus"""
        try:
            with self.db.get_session() as session:
                # Get recent tokens to check
                recent_tokens = session.query(Token).\
                    filter(Token.updated_at >= datetime.utcnow() - timedelta(days=1)).\
                    all()

                for token in recent_tokens:
                    await self.check_goplus_security(token.address, token.chain)
                    await asyncio.sleep(1)  # Rate limiting

        except Exception as e:
            logging.error(f"Error refreshing GoPlus data: {str(e)}")

    async def update_anomaly_model(self):
        """Update anomaly detection model with new data"""
        try:
            new_data = self._load_historical_data()
            if not new_data.empty:
                self.model.fit(np.log1p(new_data))
                logging.info("Anomaly detection model updated")
        except Exception as e:
            logging.error(f"Error updating anomaly model: {str(e)}")

    async def cleanup_old_entries(self):
        """Clean up old blacklist entries"""
        try:
            with self.db.get_session() as session:
                # Remove entries older than 30 days if they haven't been confirmed
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                session.query(Blacklist).\
                    filter(Blacklist.listed_at < cutoff_date).\
                    filter(Blacklist.confirmed.is_(False)).\
                    delete()
                session.commit()
        except Exception as e:
            logging.error(f"Error cleaning up old entries: {str(e)}")

    async def get_blacklist_stats(self) -> Dict:
        """Get statistics about blacklisted addresses"""
        try:
            with self.db.get_session() as session:
                total_coins = session.query(Blacklist).\
                    filter(Blacklist.type == 'coin').\
                    count()
                total_devs = session.query(Blacklist).\
                    filter(Blacklist.type == 'dev').\
                    count()
                recent = session.query(Blacklist).\
                    filter(Blacklist.listed_at >= datetime.utcnow() - timedelta(days=1)).\
                    count()
                
                return {
                    'total_coins': total_coins,
                    'total_devs': total_devs,
                    'recent_additions': recent
                }
        except Exception as e:
            logging.error(f"Error getting blacklist stats: {str(e)}")
            return {}

    async def get_token_data(self, token_address: str) -> Optional[Dict]:
        """Get token data from database"""
        try:
            with self.db.get_session() as session:
                token = session.query(Token).\
                    filter(Token.address == token_address).\
                    first()
                
                if not token:
                    return None
                
                latest_price = session.query(PriceHistory).\
                    filter(PriceHistory.token_id == token.id).\
                    order_by(PriceHistory.timestamp.desc()).\
                    first()
                
                if not latest_price:
                    return None
                
                return {
                    'address': token.address,
                    'price': latest_price.price,
                    'volume_24h': latest_price.volume_24h,
                    'liquidity': latest_price.liquidity
                }
                
        except Exception as e:
            logging.error(f"Error getting token data: {str(e)}")
            return None
