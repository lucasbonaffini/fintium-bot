# trading_bot.py
import os
import asyncio
import logging
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from sqlalchemy import func
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
from sqlalchemy import text


class MarketAnalyzer:
    def __init__(self):
        self.volatility_threshold = 20  # 20% price change
        self.volume_spike_threshold = 3  # 3x average volume
        self.significant_mcap_threshold = 1_000_000  # $1M market cap
        self.min_holders = 1000
        self.min_liquidity = 500_000  # $500k liquidity
        self.model = IsolationForest(n_estimators=100, contamination=0.01)
        self.historical_data = pd.DataFrame()

    async def initialize_model(self, db_session):
        """Initialize anomaly detection model with historical data"""
        query = """
            SELECT price, volume_24h, liquidity 
            FROM price_history 
            ORDER BY timestamp DESC 
            LIMIT 100000
        """
        self.historical_data = pd.read_sql(query, db_session.bind)
        if not self.historical_data.empty:
            self.model.fit(np.log1p(self.historical_data))

    def analyze_price_movement(self, token_data: Dict) -> Dict:
        """Analyze significant price movements"""
        price_change = token_data.get('price_change_24h', 0)
        
        return {
            'is_volatile': abs(price_change) >= self.volatility_threshold,
            'direction': 'up' if price_change > 0 else 'down',
            'magnitude': abs(price_change),
            'is_anomaly': self.detect_price_anomaly(token_data)
        }

    def detect_price_anomaly(self, token_data: Dict) -> bool:
        """Detect anomalous price movements using machine learning"""
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

    def analyze_liquidity_changes(self, current: float, previous: float) -> Dict:
        """Analyze liquidity changes"""
        if not previous or previous == 0:
            return {'significant_change': False, 'direction': 'none', 'percentage': 0}
            
        change_pct = ((current - previous) / previous) * 100
        return {
            'significant_change': abs(change_pct) > 20,
            'direction': 'increase' if change_pct > 0 else 'decrease',
            'percentage': change_pct,
            'is_suspicious': abs(change_pct) > 50  # Flag very large changes
        }

    def detect_manipulation_patterns(self, token_data: Dict) -> bool:
        """Detect potential market manipulation patterns"""
        try:
            # Price manipulation check
            price_analysis = self.analyze_price_movement(token_data)
            
            # Liquidity manipulation check
            liquidity_history = token_data.get('liquidity_history', [])
            if liquidity_history:
                liquidity_changes = [
                    self.analyze_liquidity_changes(b, a) 
                    for a, b in zip(liquidity_history[:-1], liquidity_history[1:])
                ]
                suspicious_changes = [
                    change for change in liquidity_changes 
                    if change['is_suspicious']
                ]
                
                return (
                    price_analysis['is_anomaly'] or 
                    len(suspicious_changes) > 2  # Multiple suspicious liquidity changes
                )
            
            return price_analysis['is_anomaly']
            
        except Exception as e:
            logging.error(f"Error detecting manipulation: {str(e)}")
            return False

class TradingBot:
    def __init__(self):
        # Initialize configurations
        self.config = Config (
            db=DatabaseConfig(password=os.getenv('DB_PASSWORD')),
            filters=FilterConfig(),
            blacklists=BlacklistConfig(),
            solscan=SolscanConfig(
                api_key=os.getenv('SOLSCAN_API_KEY', ''),
                base_url=os.getenv('SOLSCAN_BASE_URL', 'https://pro-api.solscan.io'),
                rate_limit=float(os.getenv('SOLSCAN_RATE_LIMIT', '1.0'))
            )
        )
        
        # Initialize components
        self.db = Database(self.config.db)
        self.security_manager = SecurityManager(self.db, self.config)
        self.telegram = TelegramHandler(
            token=os.getenv('TELEGRAM_BOT_TOKEN'),
            chat_id=os.getenv('TELEGRAM_CHAT_ID')
        )
        
        self.rugcheck = RugCheckClient(self.security_manager)
        self.filter_manager = FilterManager(self.config)
        self.market_analyzer = MarketAnalyzer()
        self.positions = {}
        
        # Unibot client for Solana trading
        self.unibot = UnibotSolanaClient(
            config=self.config,
            api_id=os.getenv('TELEGRAM_API_ID'),
            api_hash=os.getenv('TELEGRAM_API_HASH'),
            phone=os.getenv('TELEGRAM_PHONE')
        )
        self.default_dex = os.getenv('DEFAULT_DEX', 'raydium')
        
        # Control de ciclo principal
        self.running = False
        self.last_health_check = datetime.now()
        self.health_check_interval = 300  # 5 minutos
        


    async def initialize(self):
        """Initialize all components with proper error handling"""
        try:
            # Inicializar base de datos
            logging.info("Initializing database...")
            self.db.create_tables()
            
            # Inicializar Telegram
            logging.info("Initializing Telegram handler...")
            self.telegram.set_trading_bot(self) 
            await self.telegram.initialize()
            
            # Inicializar Unibot
            logging.info("Initializing Unibot client...")
            await self.unibot.initialize()
            
            # Inicializar Market Analyzer
            logging.info("Initializing market analyzer...")
            with self.db.get_session() as session:
                await self.market_analyzer.initialize_model(session)
            
            # Verificar conexión con Unibot
            if await self.unibot.check_connection():
                await self.telegram.send_message("✅ Trading Bot initialized successfully!")
            else:
                raise Exception("Failed to verify Unibot connection")
            
            self.running = True
            logging.info("Bot initialization completed successfully")
            
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}")
            await self.telegram.send_message(f"⚠️ Initialization error: {str(e)}")
            raise
        
        
    async def discover_tokens(self) -> List[str]:
        """Enhanced token discovery with detailed logging"""
        try:
            discovered_tokens = set()
            logging.info("Starting token discovery process...")
            
            async with DexScreenerClient() as client:
                # Get and analyze trending pairs
                trending = await client.get_trending_pairs()
                if trending:
                    logging.info(f"Found {len(trending)} trending pairs")
                    for pair in trending:
                        if pair.get('chainId') == 'solana':
                            token = pair.get('baseToken', {}).get('address')
                            if token:
                                discovered_tokens.add(token)
                                logging.info(f"Added trending token: {token} ({pair.get('baseToken', {}).get('symbol')})")

                # Get and analyze recent pairs
                recent = await client.get_recent_pairs('solana')
                if recent:
                    logging.info(f"Found {len(recent)} recent pairs")
                    for pair in recent:
                        token = pair.get('baseToken', {}).get('address')
                        if token:
                            discovered_tokens.add(token)
                            logging.info(f"Added recent token: {token} ({pair.get('baseToken', {}).get('symbol')})")

                logging.info(f"Total unique tokens discovered: {len(discovered_tokens)}")

                # Process tokens in parallel
                filtered_tokens = []
                for token in discovered_tokens:
                    # Get token data
                    token_data = await self.get_token_data(token)
                    if not token_data:
                        logging.info(f"Skipping token {token} - Could not get data")
                        continue

                    # Apply filters
                    if await self._passes_initial_filters(token_data):
                        logging.info(f"Token passed initial filters: {token} ({token_data.get('symbol')})")
                        
                        # Check safety
                        if await self.check_token_safety(token):
                            logging.info(f"Token passed safety checks: {token}")
                            filtered_tokens.append(token)
                        else:
                            logging.info(f"Token failed safety checks: {token}")
                    else:
                        logging.info(f"Token failed initial filters: {token}")

                logging.info(f"Final tokens after filtering: {len(filtered_tokens)}")
                return filtered_tokens

        except Exception as e:
            logging.error(f"Error discovering tokens: {str(e)}")
            return []
        
    async def _advanced_scam_check(self, token_data: Dict) -> bool:
        """Enhanced scam detection with multiple indicators"""
        try:
            # Check for suspicious patterns
            name = token_data.get('name', '').lower()
            symbol = token_data.get('symbol', '').lower()
            
            # Red flags in name/symbol
            suspicious_patterns = [
                r'scam', r'rug', r'safe', r'moon', r'elon', r'doge',
                r'shib', r'inu', r'pump', r'dump', r'airdrop', r'free',
                r'test', r'presale', r'pre-sale', r'pre sale'
            ]
            
            if any(re.search(pattern, name) or re.search(pattern, symbol) 
                for pattern in suspicious_patterns):
                logging.warning(f"Suspicious name/symbol detected for {name} ({symbol})")
                return False

            # Check market metrics
            try:
                liquidity = float(token_data.get('liquidity', 0))
                market_cap = float(token_data.get('market_cap', 0))
                holders = int(token_data.get('holders', 0))
                
                if liquidity < 5000:
                    logging.warning(f"Low liquidity detected: {liquidity}")
                    return False
                
                if market_cap < 10000:
                    logging.warning(f"Low market cap detected: {market_cap}")
                    return False
                
                if holders < 50:
                    logging.warning(f"Low holder count detected: {holders}")
                    return False
            except (ValueError, TypeError) as e:
                logging.error(f"Error parsing market metrics: {str(e)}")
                return False

            # Check ownership concentration
            try:
                largest_holder_pct = float(token_data.get('largest_holder_percentage', 100))
                if largest_holder_pct > 15:
                    logging.warning(f"High ownership concentration detected: {largest_holder_pct}%")
                    return False
            except (ValueError, TypeError) as e:
                logging.error(f"Error parsing holder percentage: {str(e)}")
                return False

            # Check contract verification
            if not token_data.get('is_contract_verified', False):
                logging.warning("Contract not verified")
                return False

            # Check trading patterns
            if hasattr(self, 'market_analyzer') and self.market_analyzer.detect_manipulation_patterns(token_data):
                logging.warning("Market manipulation patterns detected")
                return False

            logging.info(f"Token {name} ({symbol}) passed all scam checks")
            return True

        except Exception as e:
            logging.error(f"Error in advanced scam check: {str(e)}")
            return False


    async def _passes_initial_filters(self, token_data: Dict) -> bool:
        """Apply initial filters to token data"""
        try:
            min_liquidity = float(os.getenv('MIN_LIQUIDITY', '5000'))
            min_holders = int(os.getenv('MIN_HOLDERS', '100'))
            max_holder_pct = float(os.getenv('MAX_HOLDER_PERCENTAGE', '15.0'))

            return all([
                float(token_data.get('liquidity', 0)) >= min_liquidity,
                int(token_data.get('holders', 0)) >= min_holders,
                float(token_data.get('largest_holder_percentage', 0)) <= max_holder_pct,
                not self.filter_manager.has_suspicious_name(token_data.get('name', '')),
                await self.check_token_safety(token_data.get('address'))
            ])
        except Exception as e:
            logging.error(f"Error in filters: {str(e)}")
            return False  

    async def check_token_safety(self, token_address: str) -> bool:
        """Enhanced token safety check"""
        try:
            # Check blacklists first
            if self.filter_manager.is_token_blacklisted(token_address):
                logging.warning(f"Token {token_address} is blacklisted")
                return False

            # Check RugCheck
            is_safe, message = await self.rugcheck.analyze_token_safety(token_address)
            if not is_safe:
                logging.warning(f"Token {token_address} failed safety check: {message}")
                await self.telegram.send_message(
                    f"🚫 Safety Check Failed\n"
                    f"Token: {token_address}\n"
                    f"Reason: {message}"
                )
                return False

            # Get token data for advanced checks
            token_data = await self.get_token_data(token_address)
            if not token_data:
                return False

            # Check for manipulation patterns
            token_data = await self.get_token_data(token_address)
            if token_data and self.market_analyzer.detect_manipulation_patterns(token_data):
                logging.warning(f"Manipulation patterns detected for {token_address}")
                return False

            return True

        except Exception as e:
            logging.error(f"Error in token safety check: {str(e)}")
            return False


    async def get_token_data(self, token_address: str) -> Optional[Dict]:
        """Fetch and enrich token data"""
        try:
            async with DexScreenerClient() as client:
                token_data = await client.get_token_data(token_address)
                if token_data:
                    # Enrich data with historical information
                    token_data['liquidity_history'] = await self.get_liquidity_history(token_address)
                    token_data['volume_history'] = await self.get_volume_history(token_address)
                    await self.store_token_data(token_data)
                return token_data
        except Exception as e:
            logging.error(f"Error fetching token data: {str(e)}")
            return None

    async def get_liquidity_history(self, token_address: str, days: int = 7) -> List[float]:
        """Get historical liquidity data"""
        with self.db.get_session() as session:
            history = session.query(PriceHistory.liquidity).\
                join(Token).\
                filter(Token.address == token_address).\
                filter(PriceHistory.timestamp >= datetime.utcnow() - timedelta(days=days)).\
                order_by(PriceHistory.timestamp.asc()).\
                all()
            return [h[0] for h in history]

    async def get_volume_history(self, token_address: str, days: int = 7) -> List[float]:
        """Get historical volume data"""
        with self.db.get_session() as session:
            history = session.query(PriceHistory.volume_24h).\
                join(Token).\
                filter(Token.address == token_address).\
                filter(PriceHistory.timestamp >= datetime.utcnow() - timedelta(days=days)).\
                order_by(PriceHistory.timestamp.asc()).\
                all()
            return [h[0] for h in history]

    async def store_token_data(self, token_data: Dict):
        """Store token data with enhanced tracking"""
        try:
            with self.db.get_session() as session:
                # Store basic token info
                token = Token(
                    address=token_data['address'],
                    symbol=token_data['symbol'],
                    name=token_data['name'],
                    chain=token_data.get('chain', 'ethereum'),
                    updated_at=datetime.utcnow()
                )

                # Store price history with enhanced metrics
                price_history = PriceHistory(
                    token=token,
                    timestamp=datetime.utcnow(),
                    price=token_data['price'],
                    volume_24h=token_data['volume_24h'],
                    liquidity=token_data['liquidity'],
                    price_change_24h=token_data.get('price_change_24h', 0),
                    holders=token_data.get('holders', 0),
                    market_cap=token_data.get('market_cap', 0)
                )

                # Track significant events
                if self._is_significant_event(token_data):
                    event = MarketEvent(
                        token=token,
                        event_type=self._determine_event_type(token_data),
                        timestamp=datetime.utcnow(),
                        details=self._create_event_details(token_data)
                    )
                    session.add(event)

                session.merge(token)
                session.add(price_history)
                session.commit()

        except Exception as e:
            logging.error(f"Error storing token data: {str(e)}")

    def _is_significant_event(self, token_data: Dict) -> bool:
        """Determine if current state represents a significant event"""
        return any([
            abs(token_data.get('price_change_24h', 0)) >= 20,  # Significant price move
            token_data.get('volume_24h', 0) >= 1_000_000,      # High volume
            token_data.get('holders', 0) >= 10_000,            # Significant holder base
            token_data.get('liquidity', 0) >= 1_000_000        # High liquidity
        ])

    def _determine_event_type(self, token_data: Dict) -> str:
        """Classify the type of significant event"""
        if token_data.get('price_change_24h', 0) >= 20:
            return 'PRICE_SURGE'
        elif token_data.get('price_change_24h', 0) <= -20:
            return 'PRICE_DROP'
        elif token_data.get('volume_24h', 0) >= 1_000_000:
            return 'HIGH_VOLUME'
        elif token_data.get('holders', 0) >= 10_000:
            return 'HOLDER_MILESTONE'
        return 'SIGNIFICANT_CHANGE'

    def _create_event_details(self, token_data: Dict) -> str:
        """Create detailed description of the event"""
        details = []
        if 'price_change_24h' in token_data:
            details.append(f"Price Change: {token_data['price_change_24h']:.2f}%")
        if 'volume_24h' in token_data:
            details.append(f"Volume: ${token_data['volume_24h']:,.2f}")
        if 'holders' in token_data:
            details.append(f"Holders: {token_data['holders']:,}")
        return " | ".join(details)
    
    
    def calculate_position_size(self, token_data: Dict) -> float:
        """Enhanced position size calculation with advanced risk management"""
        try:
            # Base metrics
            liquidity = float(token_data.get('liquidity', 0))
            market_cap = float(token_data.get('market_cap', 0))
            volatility = abs(token_data.get('price_change_24h', 0))
            volume_24h = float(token_data.get('volume_24h', 0))
            
            # Enhanced risk factors
            volatility_factor = max(0.2, 1 - (volatility / 100))
            liquidity_factor = min(1.0, volume_24h / liquidity) if liquidity > 0 else 0.2
            market_cap_factor = min(1.0, market_cap / 1_000_000)  # Scale based on $1M market cap
            
            # Calculate base position with multi-factor approach
            base_position = min(
                liquidity * 0.01,  # 1% of liquidity
                market_cap * 0.001,  # 0.1% of market cap
                volume_24h * 0.05   # 5% of 24h volume - nuevo factor importante
            )

            # Apply weighted risk factors
            risk_adjusted_position = base_position * (
                volatility_factor * 0.4 +    # 40% peso a volatilidad
                liquidity_factor * 0.4 +     # 40% peso a liquidez
                market_cap_factor * 0.2      # 20% peso a market cap
            )

            # Additional risk controls
            if self.market_analyzer.detect_manipulation_patterns(token_data):
                risk_adjusted_position *= 0.5

            # Time-based position adjustment
            hour = datetime.now().hour
            if 0 <= hour < 8:  # Reduced positions during low liquidity hours
                risk_adjusted_position *= 0.7

            # Apply position limits
            MIN_POSITION = 0.1  # Minimum 0.1 SOL
            MAX_POSITION = 1.0  # Maximum 1.0 SOL
            
            final_position = max(MIN_POSITION, min(risk_adjusted_position, MAX_POSITION))

            # Log position calculation details
            logging.info(f"""
            Position Calculation for {token_data.get('symbol')}:
            Base Position: {base_position:.3f} SOL
            Risk Adjusted: {risk_adjusted_position:.3f} SOL
            Final Position: {final_position:.3f} SOL
            Risk Factors: Vol={volatility_factor:.2f}, Liq={liquidity_factor:.2f}, MC={market_cap_factor:.2f}
            """)

            return final_position

        except Exception as e:
            logging.error(f"Position size calculation error: {str(e)}")
            return 0
    

    async def should_trade(self, token_data: Dict) -> Tuple[bool, str]:
        """Advanced trading decision logic with comprehensive analysis"""
        try:
            # 1. Initial Market Health Check
            market_health = await self._check_market_health(token_data)
            if not market_health[0]:
                return market_health

            # 2. Risk Assessment
            risk_level = self._assess_risk_level(token_data)
            if risk_level > 0.8:  # 80% risk threshold
                return False, f"Risk level too high: {risk_level:.2f}"

            # 3. Enhanced Market Analysis
            price_analysis = self.market_analyzer.analyze_price_movement(token_data)
            volume_analysis = await self._analyze_volume_patterns(token_data)
            technical_analysis = await self._perform_technical_analysis(token_data)

            # 4. Trading Scenarios with Weighted Scores
            scenario_scores = [
                # Momentum Trading Scenario (Weight: 0.4)
                (
                    self._evaluate_momentum_scenario(price_analysis, volume_analysis),
                    "Strong momentum with volume confirmation",
                    0.4
                ),
                
                # CEX Listing Scenario (Weight: 0.3)
                (
                    self._evaluate_cex_scenario(token_data, volume_analysis),
                    "High probability CEX listing signals",
                    0.3
                ),
                
                # Technical Breakout Scenario (Weight: 0.3)
                (
                    self._evaluate_breakout_scenario(technical_analysis),
                    "Confirmed technical breakout with support",
                    0.3
                )
            ]

            # Calculate total weighted score
            total_score = sum(score * weight for score, _, weight in scenario_scores)
            
            # Log detailed analysis
            self._log_trading_analysis(token_data, scenario_scores, total_score)

            # Decision making with threshold
            if total_score >= 0.7:  # 70% confidence threshold
                # Find the highest scoring scenario for reason
                best_scenario = max(scenario_scores, key=lambda x: x[0] * x[2])
                return True, f"Trade triggered: {best_scenario[1]} (Score: {total_score:.2f})"

            return False, f"Insufficient conviction (Score: {total_score:.2f})"

        except Exception as e:
            logging.error(f"Trade decision error: {str(e)}")
            return False, f"Error in analysis: {str(e)}"
        
    async def check_market_status(self):
        """Check overall market status and trading conditions"""
        try:
            status_msg = "🔍 Market Status Check\n\n"
            alerts = []
            
            # 1. Check DexScreener API status
            try:
                async with DexScreenerClient() as client:
                    pairs = await client.get_trending_pairs()
                    status_msg += "✅ DexScreener API: Operational\n"
            except Exception as e:
                status_msg += "❌ DexScreener API: Error\n"
                alerts.append(f"DexScreener API issue: {str(e)}")

            # 2. Check Unibot connection
            try:
                if await self.unibot.check_connection():
                    status_msg += "✅ Unibot Connection: Active\n"
                else:
                    status_msg += "❌ Unibot Connection: Error\n"
                    alerts.append("Unibot connection issues detected")
            except Exception as e:
                status_msg += "❌ Unibot Connection: Error\n"
                alerts.append(f"Unibot error: {str(e)}")

            # 3. Check wallet balance using Solscan
            try:
                wallet_status = await self.unibot.get_wallet_status()
                if wallet_status:
                    status_msg += (
                        "💰 Wallet Status:\n"
                        f"Balance: {wallet_status['sol_balance']}\n"
                        f"Value: {wallet_status['total_value_usd']}\n"
                    )
                    if wallet_status['sol_balance'] == "Error":
                        alerts.append("Could not fetch wallet balance")
            except Exception as e:
                status_msg += "❌ Wallet Status: Error\n"
                alerts.append(f"Error fetching wallet status: {str(e)}")

            # 4. Check database connection
            try:
                with self.db.get_session() as session:
                    session.execute(text("SELECT 1"))
                    status_msg += "✅ Database Connection: Active\n"
            except Exception as e:
                status_msg += "❌ Database Connection: Error\n"
                alerts.append(f"Database connection issues: {str(e)}")

            # 5. Check active positions
            try:
                success, response = await self.unibot._queue_command("/positions")
                if success:
                    positions = self._parse_positions_response(response)
                    status_msg += f"\n📊 Active Positions: {len(positions)}\n"
                    if positions:
                        for token, data in positions.items():
                            status_msg += f"• {token}: {data.get('amount', 0)} tokens\n"
                else:
                    status_msg += "\n❌ Unable to fetch positions\n"
                    alerts.append("Position check failed")
            except Exception as e:
                status_msg += "\n❌ Position Check Error\n"
                alerts.append(f"Position check error: {str(e)}")

            # Add alerts section if any
            if alerts:
                status_msg += "\n⚠️ Alerts:\n"
                for alert in alerts:
                    status_msg += f"- {alert}\n"

            # Send status message
            await self.telegram.send_message(status_msg)

            return len(alerts) == 0

        except Exception as e:
            error_msg = f"Error in market status check: {str(e)}"
            logging.error(error_msg)
            await self.telegram.send_message(f"🚨 {error_msg}")
            return False
        
    async def get_market_metrics(self) -> Dict:
        """Collect key market metrics for monitoring"""
        try:
            metrics = {
                'timestamp': datetime.now(),
                'total_liquidity': 0,
                'total_volume': 0,
                'active_tokens': 0,
                'potential_opportunities': 0,
                'risk_level': 'LOW'
            }

            # Collect metrics from DexScreener
            async with DexScreenerClient() as client:
                pairs = await client.get_trending_pairs()
                if pairs:
                    # Calculate total liquidity and volume
                    metrics['total_liquidity'] = sum(
                        float(pair.get('liquidity', {}).get('usd', 0))
                        for pair in pairs
                    )
                    metrics['total_volume'] = sum(
                        float(pair.get('volume', {}).get('h24', 0))
                        for pair in pairs
                    )
                    metrics['active_tokens'] = len(pairs)

                    # Count potential opportunities
                    metrics['potential_opportunities'] = len([
                        pair for pair in pairs
                        if self._is_potential_opportunity(pair)
                    ])

                    # Assess risk level
                    risk_score = self._calculate_market_risk(pairs)
                    metrics['risk_level'] = self._risk_level_from_score(risk_score)

            return metrics

        except Exception as e:
            logging.error(f"Error collecting market metrics: {str(e)}")
            return {}

    def _is_potential_opportunity(self, pair: Dict) -> bool:
        """Check if a pair represents a potential trading opportunity"""
        try:
            # Basic opportunity criteria
            min_liquidity = float(os.getenv('MIN_LIQUIDITY', '5000'))
            min_volume = float(os.getenv('MIN_VOLUME', '10000'))
            
            liquidity = float(pair.get('liquidity', {}).get('usd', 0))
            volume = float(pair.get('volume', {}).get('h24', 0))
            price_change = float(pair.get('priceChange', {}).get('h24', 0))
            
            return all([
                liquidity >= min_liquidity,
                volume >= min_volume,
                abs(price_change) >= 5,  # At least 5% price movement
                abs(price_change) <= 50  # Not too volatile
            ])
        except Exception as e:
            logging.error(f"Error checking opportunity: {str(e)}")
            return False

    def _calculate_market_risk(self, pairs: List[Dict]) -> float:
        """Calculate overall market risk score"""
        try:
            risk_factors = []
            
            for pair in pairs:
                # Individual pair risk factors
                liquidity = float(pair.get('liquidity', {}).get('usd', 0))
                volume = float(pair.get('volume', {}).get('h24', 0))
                price_change = float(pair.get('priceChange', {}).get('h24', 0))
                
                pair_risk = 0.0
                # Liquidity risk
                if liquidity < 10000:
                    pair_risk += 0.3
                elif liquidity < 50000:
                    pair_risk += 0.2
                
                # Volatility risk
                if abs(price_change) > 50:
                    pair_risk += 0.3
                elif abs(price_change) > 25:
                    pair_risk += 0.2
                
                # Volume risk
                if volume < 5000:
                    pair_risk += 0.2
                
                risk_factors.append(pair_risk)
            
            # Return average risk score
            return sum(risk_factors) / len(risk_factors) if risk_factors else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating market risk: {str(e)}")
            return 1.0  # Return maximum risk on error

    def _risk_level_from_score(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score < 0.3:
            return 'LOW'
        elif risk_score < 0.6:
            return 'MEDIUM'
        else:
            return 'HIGH'

    async def _check_market_health(self, token_data: Dict) -> Tuple[bool, str]:
        """Comprehensive market health check"""
        try:
            # Basic requirements with detailed validation
            liquidity = float(token_data.get('liquidity', 0))
            market_cap = float(token_data.get('market_cap', 0))
            
            if liquidity < self.config.filters.min_liquidity:
                return False, f"Insufficient liquidity: {liquidity:.2f} < {self.config.filters.min_liquidity}"
            
            if market_cap < self.config.filters.min_market_cap:
                return False, f"Market cap too low: {market_cap:.2f} < {self.config.filters.min_market_cap}"

            # Manipulation check
            if self.market_analyzer.detect_manipulation_patterns(token_data):
                return False, "Manipulation patterns detected"

            # Volatility check
            if token_data.get('price_change_24h', 0) > 100:  # >100% daily change
                return False, "Excessive volatility"

            return True, "Market health checks passed"

        except Exception as e:
            logging.error(f"Market health check error: {str(e)}")
            return False, "Error in market health check"

    def _evaluate_momentum_scenario(self, price_analysis: Dict, volume_analysis: Dict) -> float:
        """Evaluate momentum trading scenario"""
        score = 0.0
        
        # Price momentum factors
        if price_analysis['direction'] == 'up':
            score += 0.4
            if price_analysis['magnitude'] > 20:  # >20% move
                score += 0.2
        
        # Volume confirmation
        if volume_analysis['is_spike']:
            score += 0.3
            if volume_analysis['spike_magnitude'] > 3:  # 3x average
                score += 0.1

        return min(score, 1.0)

    def _evaluate_cex_scenario(self, token_data: Dict, volume_analysis: Dict) -> float:
        """Evaluate CEX listing probability"""
        score = 0.0
        
        # Market cap check
        if token_data.get('market_cap', 0) > 5_000_000:  # $5M+
            score += 0.3
        
        # Volume patterns
        if volume_analysis['is_accumulation']:
            score += 0.3
        
        # Holder metrics
        if token_data.get('holders', 0) > 1000:
            score += 0.2
        
        # Social metrics (if available)
        if token_data.get('social_score', 0) > 70:
            score += 0.2

        return min(score, 1.0)

    def _log_trading_analysis(self, token_data: Dict, scenario_scores: List, total_score: float):
        """Log detailed trading analysis"""
        logging.info(f"""
        Trading Analysis for {token_data.get('symbol')}:
        Market Cap: ${token_data.get('market_cap', 0):,.2f}
        Liquidity: ${token_data.get('liquidity', 0):,.2f}
        24h Volume: ${token_data.get('volume_24h', 0):,.2f}
        
        Scenario Scores:
        {'-' * 40}
        """ + '\n'.join([f"{reason}: {score:.2f} (weight: {weight})" 
                        for score, reason, weight in scenario_scores]) + f"""
        {'-' * 40}
        Total Score: {total_score:.2f}
        """)

    async def execute_trade(self, token_address: str, amount: float, action: str) -> bool:
        """Execute trade with enhanced monitoring"""
        try:
            # Check wallet balance
            sol_balance, token_balances = await self.unibot.get_balance()
            
            if action.lower() == 'buy':
                if sol_balance is None or amount > sol_balance:
                    await self.telegram.send_message(
                        f"⚠️ Insufficient SOL balance\n"
                        f"Required: {amount} SOL\n"
                        f"Available: {sol_balance if sol_balance is not None else 'Unknown'} SOL"
                    )
                    return False           
                
                # Execute buy
                success = await self.unibot.buy_token(
                    token_address,
                    amount,
                    dex=os.getenv('DEFAULT_DEX', 'raydium'),
                    slippage=float(os.getenv('SLIPPAGE_PERCENT', '1.0'))
                )
                
                if success:
                    # Set auto-sell with configured parameters
                    await self.unibot.auto_sell(
                        token_address, 
                        take_profit=float(os.getenv('TAKE_PROFIT_PERCENT', '300.0')),
                        stop_loss=float(os.getenv('STOP_LOSS_PERCENT', '10.0')),
                        dex=os.getenv('DEFAULT_DEX', 'raydium')
                    )
                    
                    await self.telegram.send_message(
                        f"✅ Buy executed successfully\n"
                        f"Token: {token_address}\n"
                        f"Amount: {amount} SOL\n"
                        f"TP: {os.getenv('TAKE_PROFIT_PERCENT')}%\n"
                        f"SL: {os.getenv('STOP_LOSS_PERCENT')}%"
                    )
                    return True
                    
            elif action.lower() == 'sell':
                success = await self.unibot.sell_token(
                    token_address,
                    percentage=100,
                    dex=os.getenv('DEFAULT_DEX', 'raydium'),
                    slippage=float(os.getenv('SLIPPAGE_PERCENT', '1.0'))
                )
                
                if success:
                    await self.telegram.send_message(
                        f"✅ Sell executed successfully\n"
                        f"Token: {token_address}"
                    )
                    return True
            
            return False

        except Exception as e:
            logging.error(f"Trade execution error: {str(e)}")
            await self.telegram.send_message(f"⚠️ Trade error: {str(e)}")
            return False

    async def store_trade_details(self, trade_details: Dict):
        """Store comprehensive trade information"""
        try:
            with self.db.get_session() as session:
                event = MarketEvent(
                    token_id=trade_details['token_address'],
                    event_type=f"TRADE_{trade_details['action'].upper()}",
                    timestamp=trade_details['timestamp'],
                    details=str(trade_details)
                )
                session.add(event)
                session.commit()
        except Exception as e:
            logging.error(f"Error storing trade details: {str(e)}")

    async def update_portfolio_exposure(self):
        """Update and monitor portfolio exposure"""
        exposure, positions = await self.monitor_wallet_exposure()
        
        # Check exposure limits
        if exposure > self.config.filters.max_total_exposure:
            await self.telegram.send_message(
                f"⚠️ High Exposure Warning\n"
                f"Current: ${exposure:,.2f}\n"
                f"Maximum: ${self.config.filters.max_total_exposure:,.2f}\n"
                "Consider taking profits or reducing positions."
            )
            
    async def monitor_wallet_exposure(self):
        """Monitor wallet exposure and positions"""
        try:
            sol_balance, token_balances = await self.unibot.get_balance()
            
            exposure_msg = (
                "📊 Current Portfolio Status:\n\n"
                f"SOL Balance: {sol_balance:.4f} SOL\n\n"
                "Token Holdings:\n"
            )
            
            for token, data in token_balances.items():
                exposure_msg += (
                    f"- {token}:\n"
                    f"  Amount: {data['amount']}\n"
                    f"  Value: ${data['value_usd']:,.2f}\n"
                )
            
            await self.telegram.send_message(exposure_msg)
            
        except Exception as e:
            logging.error(f"Error monitoring wallet: {str(e)}")


    async def process_token(self, token_address: str):
        """Process token with enhanced logging"""
        try:
            logging.info(f"\n{'='*50}")
            logging.info(f"Starting analysis for token: {token_address}")
            
            # Get token data
            token_data = await self.get_token_data(token_address)
            if not token_data:
                logging.info(f"Skipping {token_address} - Could not get token data")
                return

            # Log token metrics
            logging.info(f"""
            Token Metrics:
            Symbol: {token_data.get('symbol')}
            Price: ${token_data.get('price', 0):.8f}
            24h Change: {token_data.get('price_change_24h', 0):.2f}%
            Volume: ${token_data.get('volume_24h', 0):,.2f}
            Liquidity: ${token_data.get('liquidity', 0):,.2f}
            Market Cap: ${token_data.get('market_cap', 0):,.2f}
            """)

            # Make trading decision
            should_trade, reason = await self.should_trade(token_data)
            logging.info(f"Trading decision for {token_data.get('symbol')}: {should_trade} - {reason}")
            
            if should_trade:
                # Calculate position size
                position_size = self.calculate_position_size(token_data)
                logging.info(f"Calculated position size: {position_size} SOL")
                
                if position_size >= 0.1:  # Minimum 0.1 SOL
                    logging.info(f"Executing trade for {token_data.get('symbol')}")
                    # Execute trade
                    success = await self.execute_trade(token_address, position_size, "buy")
                    
                    if success:
                        logging.info(f"Trade executed successfully for {token_data.get('symbol')}")
                        await self.telegram.send_message(
                            f"🎯 New Trade Executed\n"
                            f"Token: {token_data.get('symbol', token_address)}\n"
                            f"Amount: {position_size} SOL\n"
                            f"Reason: {reason}\n"
                            f"Price: ${token_data.get('price', 0):.8f}\n"
                            f"24h Change: {token_data.get('price_change_24h', 0):.2f}%"
                        )
                    else:
                        logging.warning(f"Trade execution failed for {token_data.get('symbol')}")
                else:
                    logging.info(f"Position size too small for {token_data.get('symbol')}: {position_size} SOL")
            
            logging.info(f"{'='*50}\n")

        except Exception as e:
            logging.error(f"Error processing token {token_address}: {str(e)}")
            await self.telegram.send_message(f"⚠️ Error processing token: {str(e)}")

    async def monitor_position(self, token_address: str, position_size: float, entry_price: float):
        """Monitor active position and manage risk"""
        try:
            monitoring = True
            while monitoring:
                token_data = await self.get_token_data(token_address)
                if not token_data:
                    continue

                current_price = token_data['price']
                pnl_percent = ((current_price - entry_price) / entry_price) * 100

                # Check exit conditions
                exit_signal = await self.check_exit_signals(token_data, pnl_percent)
                if exit_signal:
                    await self.execute_trade(token_address, position_size, "sell")
                    monitoring = False
                    
                await asyncio.sleep(60)  # Check every minute

        except Exception as e:
            logging.error(f"Position monitoring error: {str(e)}")

    async def check_exit_signals(self, token_data: Dict, pnl_percent: float) -> bool:
        """Enhanced exit signal detection"""
        try:
            # Stop loss check (-10%)
            if pnl_percent <= -10:
                await self.telegram.send_message(
                    f"🛑 Stop Loss Triggered\n"
                    f"Token: {token_data['symbol']}\n"
                    f"PnL: {pnl_percent:.2f}%"
                )
                return True

            # Take profit check (+50%)
            if pnl_percent >= 50:
                await self.telegram.send_message(
                    f"🎯 Take Profit Triggered\n"
                    f"Token: {token_data['symbol']}\n"
                    f"PnL: {pnl_percent:.2f}%"
                )
                return True

            # Technical reversal check
            if self.detect_trend_reversal(token_data):
                await self.telegram.send_message(
                    f"↩️ Trend Reversal Detected\n"
                    f"Token: {token_data['symbol']}"
                )
                return True

            # Volume dry up check
            if self.is_volume_drying_up(token_data):
                await self.telegram.send_message(
                    f"📉 Volume Drying Up\n"
                    f"Token: {token_data['symbol']}"
                )
                return True

            # Manipulation check
            if self.market_analyzer.detect_manipulation_patterns(token_data):
                await self.telegram.send_message(
                    f"⚠️ Manipulation Detected\n"
                    f"Token: {token_data['symbol']}"
                )
                return True

            return False

        except Exception as e:
            logging.error(f"Error checking exit signals: {str(e)}")
            return False

    async def generate_performance_report(self):
        """Generate detailed performance report"""
        try:
            with self.db.get_session() as session:
                # Get all trades from the last 24 hours
                yesterday = datetime.utcnow() - timedelta(days=1)
                trades = session.query(MarketEvent).\
                    filter(MarketEvent.timestamp >= yesterday).\
                    filter(MarketEvent.event_type.in_(['TRADE_BUY', 'TRADE_SELL'])).\
                    all()

                # Calculate performance metrics
                total_trades = len(trades)
                profitable_trades = len([t for t in trades if float(t.details.get('pnl_percent', 0)) > 0])
                total_pnl = sum(float(t.details.get('pnl_amount', 0)) for t in trades)

                report = (
                    "📊 24h Performance Report\n\n"
                    f"Total Trades: {total_trades}\n"
                    f"Profitable Trades: {profitable_trades}\n"
                    f"Win Rate: {(profitable_trades/total_trades*100 if total_trades else 0):.2f}%\n"
                    f"Total PnL: ${total_pnl:,.2f}\n\n"
                    "🔝 Top Performers:\n"
                )

                # Add top performing trades
                top_trades = sorted(
                    trades, 
                    key=lambda x: float(x.details.get('pnl_percent', 0)), 
                    reverse=True
                )[:3]

                for trade in top_trades:
                    report += (
                        f"• {trade.token.symbol}: "
                        f"{float(trade.details.get('pnl_percent', 0)):.2f}%\n"
                    )

                await self.telegram.send_message(report)

        except Exception as e:
            logging.error(f"Error generating performance report: {str(e)}")
            
    async def _perform_health_check(self):
        """Perform periodic health checks"""
        try:
            if (datetime.now() - self.last_health_check).seconds >= self.health_check_interval:
                # Verificar conexión con Unibot
                unibot_status = await self.unibot.check_connection()
                
                # Verificar balance de wallet
                balance, _ = await self.unibot.get_balance()
                
                # Verificar conexión a base de datos
                db_status = self._check_database_connection()
                
                status_message = (
                    "🔄 System Health Check:\n"
                    f"Unibot Connection: {'✅' if unibot_status else '❌'}\n"
                    f"Wallet Balance: {balance if balance is not None else '❌'} SOL\n"
                    f"Database Connection: {'✅' if db_status else '❌'}"
                )
                
                await self.telegram.send_message(status_message)
                self.last_health_check = datetime.now()
                
        except Exception as e:
            logging.error(f"Health check error: {str(e)}")
            
    async def monitor_positions(self):
        """Monitor and manage active trading positions"""
        try:
            # Get current positions from Unibot
            success, response = await self.unibot._queue_command("/positions")
            if not success:
                logging.error("Failed to fetch positions")
                return

            # Extract positions from response
            current_positions = self._parse_positions_response(response)
            
            for token_address, position_data in current_positions.items():
                try:
                    # Get current token data from DexScreener
                    token_data = await self.get_token_data(token_address)
                    if not token_data:
                        continue

                    # Calculate current PnL
                    entry_price = position_data.get('entry_price', 0)
                    current_price = token_data.get('price', 0)
                    if entry_price and current_price:
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                        
                        # Check exit conditions
                        should_exit = await self._check_exit_conditions(
                            token_address,
                            token_data,
                            pnl_percent,
                            position_data
                        )

                        if should_exit:
                            # Execute sell order
                            success = await self.execute_trade(
                                token_address,
                                position_data.get('amount', 0),
                                "sell"
                            )
                            
                            if success:
                                await self.telegram.send_message(
                                    f"🔄 Position Closed\n"
                                    f"Token: {token_data.get('symbol')}\n"
                                    f"PnL: {pnl_percent:.2f}%"
                                )

                except Exception as e:
                    logging.error(f"Error monitoring position {token_address}: {str(e)}")

        except Exception as e:
            logging.error(f"Error in position monitoring: {str(e)}")

    def _parse_positions_response(self, response: str) -> Dict:
        """Parse positions from Unibot response"""
        positions = {}
        try:
            # Example response format:
            # Token: ABC
            # Amount: 100
            # Entry: $1.23
            lines = response.split('\n')
            current_token = None
            
            for line in lines:
                if ':' not in line:
                    continue
                    
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if 'token' in key:
                    current_token = value
                    positions[current_token] = {}
                elif current_token:
                    if 'amount' in key:
                        positions[current_token]['amount'] = float(value.replace(',', ''))
                    elif 'entry' in key:
                        positions[current_token]['entry_price'] = float(value.replace('$', '').replace(',', ''))

            return positions
        except Exception as e:
            logging.error(f"Error parsing positions: {str(e)}")
            return {}

    async def _check_exit_conditions(
        self,
        token_address: str,
        token_data: Dict,
        pnl_percent: float,
        position_data: Dict
    ) -> bool:
        """Check if position should be exited"""
        try:
            # Take profit check
            if pnl_percent >= self.config.filters.take_profit:
                logging.info(f"Take profit triggered for {token_address}")
                return True

            # Stop loss check
            if pnl_percent <= -self.config.filters.stop_loss:
                logging.info(f"Stop loss triggered for {token_address}")
                return True

            # Volume dry up check
            if token_data.get('volume_24h', 0) < position_data.get('entry_volume', 0) * 0.5:
                logging.info(f"Volume dry up detected for {token_address}")
                return True

            # Manipulation check
            if self.market_analyzer.detect_manipulation_patterns(token_data):
                logging.info(f"Manipulation detected for {token_address}")
                return True

            return False

        except Exception as e:
            logging.error(f"Error checking exit conditions: {str(e)}")
            return False
            
    async def _generate_reports(self):
        """Generate performance and status reports"""
        try:
            await self.generate_performance_report()
            await self.monitor_wallet_exposure()
            await self.send_security_status()
        except Exception as e:
            logging.error(f"Error generating reports: {str(e)}")
            
    def _check_database_connection(self) -> bool:
        """Verify database connection is active"""
        try:
            with self.db.get_session() as session:
                session.execute("SELECT 1")
                return True
        except Exception:
            return False
        
    async def get_bot_statistics(self) -> str:
        """Get detailed bot statistics"""
        try:
            stats_msg = "📊 Bot Statistics Report\n\n"
            
            # Get runtime stats
            uptime = datetime.now() - self.start_time
            stats_msg += f"⏱️ Uptime: {uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m\n\n"
            
            # Get trading stats for last 24h
            with self.db.get_session() as session:
                yesterday = datetime.utcnow() - timedelta(days=1)
                
                # Count analyzed tokens
                analyzed_count = session.query(Token).\
                    filter(Token.updated_at >= yesterday).\
                    count()
                
                # Count trades
                trades = session.query(MarketEvent).\
                    filter(MarketEvent.timestamp >= yesterday).\
                    filter(MarketEvent.event_type.in_(['TRADE_BUY', 'TRADE_SELL'])).\
                    all()
                
                total_trades = len(trades)
                profitable_trades = len([t for t in trades if float(t.details.get('pnl_percent', 0)) > 0])
                
                stats_msg += f"📈 Last 24h Activity:\n"
                stats_msg += f"Tokens Analyzed: {analyzed_count}\n"
                stats_msg += f"Total Trades: {total_trades}\n"
                stats_msg += f"Profitable Trades: {profitable_trades}\n"
                if total_trades > 0:
                    win_rate = (profitable_trades/total_trades) * 100
                    stats_msg += f"Win Rate: {win_rate:.1f}%\n"
            
            # Get current positions
            success, response = await self.unibot._queue_command("/positions")
            if success:
                current_positions = self._parse_positions_response(response)
                if current_positions:
                    stats_msg += f"\n💼 Current Positions ({len(current_positions)}):\n"
                    for token, data in current_positions.items():
                        stats_msg += f"• {token}: {data.get('amount')} tokens\n"
                else:
                    stats_msg += "\n💼 No active positions\n"
            
            # Get wallet status
            wallet_status = await self.unibot.get_wallet_status()
            stats_msg += f"\n💰 Wallet Status:\n"
            stats_msg += f"Balance: {wallet_status['sol_balance']}\n"
            stats_msg += f"Value: {wallet_status['total_value_usd']}\n"
            
            return stats_msg
            
        except Exception as e:
            logging.error(f"Error getting bot statistics: {str(e)}")
            return "Error generating statistics"


    async def run(self):
        """Enhanced main bot loop with comprehensive monitoring"""
        try:
            # Verificar inicialización
            if not hasattr(self, 'unibot') or not self.unibot.initialized:
                await self.initialize()

            logging.info("Starting trading bot main loop...")
            await self.telegram.send_message("🚀 Trading Bot Started - Monitoring Market")
            
            while True:
                try:
                    # 1. Verificar estado del bot
                    if not self.unibot.initialized:
                        logging.error("Unibot not initialized, attempting to reconnect...")
                        await self.initialize()
                        continue

                    # 2. Actualizar posiciones actuales
                    logging.info("Checking current positions...")
                    await self.unibot.client.send_message(self.unibot.unibot_username, "/positions")
                    await asyncio.sleep(2)  # Esperar respuesta

                    # 3. Verificar nuevos pares de trading
                    logging.info("Checking new trading pairs...")
                    await self.unibot.client.send_message(self.unibot.unibot_username, "New Pairs")
                    await asyncio.sleep(2)  # Esperar respuesta

                    # 4. Procesar cualquier señal de trading
                    try:
                        # Descubrir tokens para analizar
                        tokens = await self.discover_tokens()
                        if tokens:
                            logging.info(f"Found {len(tokens)} potential tokens to analyze")
                            
                            for token_address in tokens:
                                # Obtener datos del token
                                token_data = await self.get_token_data(token_address)
                                if not token_data:
                                    continue
                                    
                                # Verificar si cumple criterios de trading
                                should_trade, reason = await self.should_trade(token_data)
                                if should_trade:
                                    logging.info(f"Trade signal detected for {token_address}: {reason}")
                                    
                                    # Calcular tamaño de posición
                                    position_size = self.calculate_position_size(token_data)
                                    if position_size >= 0.1:  # Mínimo 0.1 SOL
                                        # Ejecutar trade
                                        success = await self.execute_trade(
                                            token_address,
                                            position_size,
                                            "buy"
                                        )
                                        if success:
                                            await self.telegram.send_message(
                                                f"🎯 New Trade Executed\n"
                                                f"Token: {token_data.get('symbol', token_address)}\n"
                                                f"Amount: {position_size} SOL\n"
                                                f"Reason: {reason}"
                                            )

                                # Rate limiting entre análisis
                                await asyncio.sleep(1)
                    except Exception as e:
                        logging.error(f"Error processing trading signals: {str(e)}")

                    # 5. Monitorear posiciones existentes
                    await self.monitor_positions()

                    # 6. Generar reportes periódicos
                    current_hour = datetime.now().hour
                    if not hasattr(self, 'last_report_hour') or self.last_report_hour != current_hour:
                        await self.generate_performance_report()
                        self.last_report_hour = current_hour

                    # 7. Health check
                    if not hasattr(self, 'last_health_check') or \
                    (datetime.now() - self.last_health_check).seconds > 300:  # Cada 5 minutos
                        await self.check_market_status()
                        self.last_health_check = datetime.now()

                    # Esperar antes del siguiente ciclo
                    await asyncio.sleep(60)  # 1 minuto entre ciclos principales

                except Exception as e:
                    logging.error(f"Error in main loop: {str(e)}")
                    await self.telegram.send_message(
                        f"⚠️ Error in main loop: {str(e)}\n"
                        "Bot will continue after a brief pause."
                    )
                    await asyncio.sleep(60)

        except Exception as e:
            logging.critical(f"Fatal error in run: {str(e)}")
            await self.telegram.send_message(f"🚨 Fatal error: {str(e)}")
            raise
        finally:
            # Asegurar limpieza apropiada
            if hasattr(self, 'unibot'):
                await self.unibot.close()
