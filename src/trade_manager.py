import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import asyncio
from market_analyzer import MarketAnalyzer
from dexscreener_client import DexScreenerClient
from rugcheck_client import RugCheckClient
from security_manager import SecurityManager
from unibot_client import UnibotSolanaClient

class TradeManager:
    def __init__(
        self,
        unibot: UnibotSolanaClient,
        analyzer: MarketAnalyzer,
        dexscreener: DexScreenerClient,
        rugcheck: RugCheckClient,
        security: SecurityManager
    ):
        self.unibot = unibot
        self.analyzer = analyzer
        self.dexscreener = dexscreener
        self.rugcheck = rugcheck
        self.security = security
        self.active_trades = {}
        self.trade_history = []
        
    async def process_trading_opportunity(self, token_data: Dict) -> bool:
        """Process a potential trading opportunity"""
        try:
            token_address = token_data.get('address')
            
            # Skip if already trading
            if token_address in self.active_trades:
                return False
                
            # Security checks
            security_passed, security_msg = await self.security.check_token_security(token_address)
            if not security_passed:
                logging.info(f"Security check failed for {token_address}: {security_msg}")
                return False
                
            # Rugcheck analysis
            rugcheck_passed, rugcheck_msg = await self.rugcheck.analyze_token_safety(token_address)
            if not rugcheck_passed:
                logging.info(f"Rugcheck failed for {token_address}: {rugcheck_msg}")
                return False
            
            # Get wallet status
            wallet_status = await self.unibot.get_wallet_status()
            if not wallet_status:
                logging.error("Could not get wallet status")
                return False
                
            # Market analysis
            should_trade, confidence, reason = self.analyzer.analyze_token(token_data)
            if not should_trade:
                logging.info(f"Analysis rejected trade for {token_address}: {reason}")
                return False
            
            # Calculate position size
            position_size = self.analyzer.calculate_position_size(
                token_data,
                float(wallet_status.get('sol_balance', '0').split()[0])
            )
            
            # Validate trade parameters
            params_valid, params_msg = self.analyzer.validate_trade_parameters(token_data, position_size)
            if not params_valid:
                logging.info(f"Invalid trade parameters for {token_address}: {params_msg}")
                return False
            
            # Execute trade
            success = await self.execute_trade(token_address, token_data, position_size)
            if success:
                self.active_trades[token_address] = {
                    'entry_time': datetime.now(),
                    'entry_price': token_data.get('price'),
                    'position_size': position_size,
                    'confidence': confidence
                }
                logging.info(f"Successfully entered trade for {token_address}")
                
            return success
            
        except Exception as e:
            logging.error(f"Error processing trading opportunity: {str(e)}")
            return False
            
    async def execute_trade(self, token_address: str, token_data: Dict, position_size: float) -> bool:
        """Execute trade with proper position management"""
        try:
            # Get trade levels
            tp_levels = self.analyzer.get_take_profit_levels(token_data)
            sl_level = self.analyzer.get_stop_loss_level(token_data)
            
            # Execute buy order
            buy_success = await self.unibot.buy_token(token_address, position_size)
            if not buy_success:
                logging.error(f"Buy order failed for {token_address}")
                return False
            
            # Set take profit and stop loss
            await self.unibot.set_auto_sell(
                token_address,
                take_profit=tp_levels[0],  # First TP level
                stop_loss=sl_level
            )
            
            # Record trade
            self.trade_history.append({
                'token_address': token_address,
                'type': 'BUY',
                'time': datetime.now(),
                'price': token_data.get('price'),
                'size': position_size,
                'tp_levels': tp_levels,
                'sl_level': sl_level
            })
            
            return True
            
        except Exception as e:
            logging.error(f"Error executing trade: {str(e)}")
            return False
            
    async def monitor_active_trades(self):
        """Monitor and manage active trades"""
        try:
            for token_address, trade_data in list(self.active_trades.items()):
                # Get current token data
                token_data = await self.dexscreener.get_token_data(token_address)
                if not token_data:
                    continue
                    
                current_price = token_data.get('price', 0)
                entry_price = trade_data['entry_price']
                
                # Calculate current PnL
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
                
                # Check for exit conditions
                if await self.should_exit_trade(token_address, token_data, trade_data, pnl_percent):
                    # Execute sell
                    if await self.unibot.sell_token(token_address):
                        del self.active_trades[token_address]
                        
                        # Record trade
                        self.trade_history.append({
                            'token_address': token_address,
                            'type': 'SELL',
                            'time': datetime.now(),
                            'price': current_price,
                            'pnl_percent': pnl_percent
                        })
                        
        except Exception as e:
            logging.error(f"Error monitoring trades: {str(e)}")
            
    async def should_exit_trade(self, token_address: str, current_data: Dict, trade_data: Dict, pnl_percent: float) -> bool:
        """Determine if we should exit a trade"""
        try:
            # Check security status
            security_status, _ = await self.security.check_token_security(token_address)
            if not security_status:
                logging.info(f"Security check triggered exit for {token_address}")
                return True
            
            # Time-based exit (24h max hold time)
            time_in_trade = datetime.now() - trade_data['entry_time']
            if time_in_trade.total_seconds() > 24 * 3600:
                logging.info(f"Time-based exit for {token_address}")
                return True
            
            # Trend reversal exit
            _, confidence, _ = self.analyzer.analyze_token(current_data)
            if confidence < 0.3 and pnl_percent > 0:
                logging.info(f"Trend reversal exit for {token_address}")
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error checking exit conditions: {str(e)}")
            return False
            
    def get_trading_statistics(self) -> Dict:
        """Get trading statistics"""
        try:
            completed_trades = [t for t in self.trade_history if t['type'] == 'SELL']
            
            total_trades = len(completed_trades)
            if total_trades == 0:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'avg_pnl': 0,
                    'active_trades': len(self.active_trades)
                }
            
            profitable_trades = len([t for t in completed_trades if t.get('pnl_percent', 0) > 0])
            win_rate = (profitable_trades / total_trades) * 100
            avg_pnl = sum(t.get('pnl_percent', 0) for t in completed_trades) / total_trades
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'active_trades': len(self.active_trades)
            }
            
        except Exception as e:
            logging.error(f"Error calculating statistics: {str(e)}")
            return {}