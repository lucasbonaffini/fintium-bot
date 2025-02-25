import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from config import Config

class MarketAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.min_liquidity = config.filters.min_liquidity
        self.min_market_cap = config.filters.min_market_cap
        self.min_holders = config.filters.min_holders
        self.min_price_change = 5.0  # Minimum price change to consider
        self.min_volume = 10000  # Minimum 24h volume in USD
        
    def analyze_token(self, token_data: Dict) -> Tuple[bool, float, str]:
        """
        Analyze token for trading opportunity
        Returns: (should_trade, confidence_score, reason)
        """
        try:
            # Basic filters
            if token_data.get('liquidity', 0) < self.min_liquidity:
                return False, 0, "Insufficient liquidity"
                
            if token_data.get('market_cap', 0) < self.min_market_cap:
                return False, 0, "Market cap too low"
                
            if token_data.get('holders', 0) < self.min_holders:
                return False, 0, "Too few holders"

            # Calculate trading signals
            momentum_signal = self._calculate_momentum_signal(token_data)
            volume_signal = self._calculate_volume_signal(token_data)
            trend_signal = self._calculate_trend_signal(token_data)
            
            # Combine signals
            signals = [
                (momentum_signal, 0.4),  # 40% weight
                (volume_signal, 0.3),    # 30% weight
                (trend_signal, 0.3)      # 30% weight
            ]
            
            confidence_score = sum(signal * weight for signal, weight in signals)
            
            # Trading decision
            if confidence_score >= 0.7:
                return True, confidence_score, "Strong buy signal"
            elif confidence_score >= 0.5:
                return True, confidence_score, "Moderate buy signal"
            
            return False, confidence_score, "Insufficient trading signals"
            
        except Exception as e:
            logging.error(f"Error analyzing token: {str(e)}")
            return False, 0, f"Analysis error: {str(e)}"

    def _calculate_momentum_signal(self, token_data: Dict) -> float:
        """Calculate momentum signal from price and volume changes"""
        try:
            price_change = float(token_data.get('price_change_24h', 0))
            volume_change = float(token_data.get('volume_24h', 0))
            market_cap = float(token_data.get('market_cap', 0))
            
            # Normalize price change to [-1, 1]
            norm_price_change = np.tanh(price_change / 100)
            
            # Volume/Market Cap ratio (normalized)
            volume_cap_ratio = min(volume_change / market_cap if market_cap > 0 else 0, 1)
            
            # Combined momentum score
            momentum = (norm_price_change + volume_cap_ratio) / 2
            
            return max(min(momentum, 1.0), 0.0)
            
        except Exception as e:
            logging.error(f"Error calculating momentum: {str(e)}")
            return 0

    def _calculate_volume_signal(self, token_data: Dict) -> float:
        """Calculate volume-based signal"""
        try:
            volume_24h = float(token_data.get('volume_24h', 0))
            liquidity = float(token_data.get('liquidity', 0))
            
            if liquidity == 0:
                return 0
                
            # Volume/Liquidity ratio
            vol_liq_ratio = min(volume_24h / liquidity, 3.0) / 3.0
            
            # Minimum volume threshold
            if volume_24h < self.min_volume:
                return 0
                
            return vol_liq_ratio
            
        except Exception as e:
            logging.error(f"Error calculating volume signal: {str(e)}")
            return 0

    def _calculate_trend_signal(self, token_data: Dict) -> float:
        """Calculate trend signal"""
        try:
            price_change = float(token_data.get('price_change_24h', 0))
            
            # Simple trend based on price change
            if abs(price_change) < self.min_price_change:
                return 0
                
            # Normalize to [0, 1]
            if price_change > 0:
                return min(price_change / 100, 1.0)
            
            return 0
            
        except Exception as e:
            logging.error(f"Error calculating trend signal: {str(e)}")
            return 0

    def calculate_position_size(self, token_data: Dict, wallet_balance: float) -> float:
        """Calculate optimal position size based on risk metrics"""
        try:
            # Basic risk checks
            if token_data.get('liquidity', 0) < self.min_liquidity:
                return 0
                
            # Calculate base position size (1% of liquidity)
            base_size = token_data.get('liquidity', 0) * self.config.filters.risk_multiplier
            
            # Adjust for volatility
            if self.config.filters.volatility_adjust:
                price_change = abs(float(token_data.get('price_change_24h', 0)))
                volatility_factor = max(1 - (price_change / 100), 0.3)
                base_size *= volatility_factor
            
            # Apply size limits
            max_size = min(
                self.config.filters.max_position_size,
                wallet_balance * 0.1  # Max 10% of wallet
            )
            
            position_size = min(base_size, max_size)
            position_size = max(position_size, self.config.filters.min_position_size)
            
            return position_size
            
        except Exception as e:
            logging.error(f"Error calculating position size: {str(e)}")
            return 0

    def get_take_profit_levels(self, token_data: Dict) -> List[float]:
        """Calculate take profit levels based on volatility"""
        try:
            price_change = abs(float(token_data.get('price_change_24h', 0)))
            base_tp = self.config.filters.take_profit
            
            # Adjust TP based on volatility
            if price_change > 50:
                levels = [base_tp * 0.5, base_tp * 0.75, base_tp]
            else:
                levels = [base_tp * 0.33, base_tp * 0.66, base_tp]
                
            return levels
            
        except Exception as e:
            logging.error(f"Error calculating TP levels: {str(e)}")
            return [self.config.filters.take_profit]

    def get_stop_loss_level(self, token_data: Dict) -> float:
        """Calculate stop loss level based on volatility"""
        try:
            price_change = abs(float(token_data.get('price_change_24h', 0)))
            base_sl = self.config.filters.stop_loss
            
            # Adjust stop loss based on volatility
            if price_change > 50:
                return base_sl * 1.5  # Wider stop for volatile tokens
            
            return base_sl
            
        except Exception as e:
            logging.error(f"Error calculating SL level: {str(e)}")
            return self.config.filters.stop_loss

    def validate_trade_parameters(self, token_data: Dict, position_size: float) -> Tuple[bool, str]:
        """Validate trade parameters before execution"""
        try:
            liquidity = float(token_data.get('liquidity', 0))
            
            # Check position size vs liquidity
            if position_size > liquidity * 0.01:  # Max 1% of liquidity
                return False, "Position size too large for liquidity"
            
            # Check market impact
            market_impact = (position_size / liquidity) * 100
            if market_impact > self.config.filters.max_market_impact:
                return False, f"Market impact too high: {market_impact:.2f}%"
            
            # Verify minimum trade requirements
            if position_size < self.config.filters.min_position_size:
                return False, "Position size below minimum"
                
            return True, "Trade parameters valid"
            
        except Exception as e:
            logging.error(f"Error validating trade parameters: {str(e)}")
            return False, f"Validation error: {str(e)}"