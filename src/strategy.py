import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Technical analysis library
import ta
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from config import TradingStrategyConfig

class MovingAverageCrossover:
    """
    Moving Average Crossover Strategy Implementation
    
    This strategy generates buy signals when the fast moving average crosses above
    the slow moving average, and sell signals when it crosses below.
    """
    
    def __init__(self, config: TradingStrategyConfig):
        """Initialize the strategy with configuration"""
        self.fast_period = config.fast_ma_period
        self.slow_period = config.slow_ma_period
        self.ma_type = config.ma_type.upper()
        self.timeframe = config.timeframe
        self.min_volume = config.min_volume
        self.logger = logging.getLogger(__name__)
        
        # Validate MA type
        if self.ma_type not in ["SMA", "EMA", "WMA"]:
            self.logger.warning(f"Invalid MA type: {self.ma_type}. Falling back to EMA.")
            self.ma_type = "EMA"
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the required technical indicators for the strategy
        
        Args:
            df: DataFrame with OHLCV data (open, high, low, close, volume)
            
        Returns:
            DataFrame with added indicator columns
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate moving averages based on the selected type
        if self.ma_type == "SMA":
            result['fast_ma'] = SMAIndicator(close=result['close'], window=self.fast_period).sma_indicator()
            result['slow_ma'] = SMAIndicator(close=result['close'], window=self.slow_period).sma_indicator()
        elif self.ma_type == "EMA":
            result['fast_ma'] = EMAIndicator(close=result['close'], window=self.fast_period).ema_indicator()
            result['slow_ma'] = EMAIndicator(close=result['close'], window=self.slow_period).ema_indicator()
        else:  # WMA (weighted MA)
            # Weighted MA calculation (WMA not available in ta library)
            weights_fast = np.arange(1, self.fast_period + 1)
            weights_slow = np.arange(1, self.slow_period + 1)
            
            result['fast_ma'] = result['close'].rolling(self.fast_period).apply(
                lambda x: np.sum(weights_fast * x) / weights_fast.sum(), raw=True
            )
            result['slow_ma'] = result['close'].rolling(self.slow_period).apply(
                lambda x: np.sum(weights_slow * x) / weights_slow.sum(), raw=True
            )
        
        # Calculate the crossover signal
        result['signal'] = 0
        result.loc[result['fast_ma'] > result['slow_ma'], 'signal'] = 1
        result.loc[result['fast_ma'] < result['slow_ma'], 'signal'] = -1
        
        # Calculate crossovers (signal changes)
        result['crossover'] = result['signal'].diff()
        
        # Add RSI for additional confirmation
        result['rsi'] = RSIIndicator(close=result['close'], window=14).rsi()
        
        # Add Bollinger Bands for volatility assessment
        bollinger = BollingerBands(close=result['close'], window=20, window_dev=2)
        result['bb_upper'] = bollinger.bollinger_hband()
        result['bb_middle'] = bollinger.bollinger_mavg()
        result['bb_lower'] = bollinger.bollinger_lband()
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the strategy
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added signal columns
        """
        # Calculate indicators
        result = self.calculate_indicators(df)
        
        # Clean up NaN values that might appear during calculation
        result = result.dropna()
        
        # Generate trading signals
        result['buy_signal'] = (result['crossover'] > 0) & (result['volume'] >= self.min_volume)
        result['sell_signal'] = (result['crossover'] < 0) & (result['volume'] >= self.min_volume)
        
        # Additional RSI filter (optional)
        # Buy only if RSI < 70 (not overbought)
        # Sell only if RSI > 30 (not oversold)
        result.loc[result['rsi'] > 70, 'buy_signal'] = False
        result.loc[result['rsi'] < 30, 'sell_signal'] = False
        
        return result
    
    def get_latest_signal(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Get the latest trading signal from the data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple containing (signal_type, confidence)
            signal_type can be: 'BUY', 'SELL', or 'HOLD'
            confidence is a value between 0 and 1
        """
        signals_df = self.generate_signals(df)
        
        if signals_df.empty:
            return 'HOLD', 0.0
        
        # Get the most recent signals
        latest = signals_df.iloc[-1]
        
        if latest['buy_signal']:
            # Calculate confidence based on RSI and volume
            rsi_factor = max(0, min(1, (70 - latest['rsi']) / 40))
            volume_factor = min(1, latest['volume'] / (self.min_volume * 2))
            confidence = 0.5 + (rsi_factor * 0.25) + (volume_factor * 0.25)
            return 'BUY', confidence
        
        elif latest['sell_signal']:
            # Calculate confidence based on RSI and volume
            rsi_factor = max(0, min(1, (latest['rsi'] - 30) / 40))
            volume_factor = min(1, latest['volume'] / (self.min_volume * 2))
            confidence = 0.5 + (rsi_factor * 0.25) + (volume_factor * 0.25)
            return 'SELL', confidence
        
        return 'HOLD', 0.0
    
    def calculate_take_profit_stop_loss(self, 
                                       df: pd.DataFrame, 
                                       entry_price: float, 
                                       side: str,
                                       take_profit_pct: float = None,
                                       stop_loss_pct: float = None) -> Tuple[float, float]:
        """
        Calculate take profit and stop loss levels based on entry price and volatility
        
        Args:
            df: DataFrame with OHLCV data
            entry_price: Entry price for the position
            side: Position side ('BUY' or 'SELL')
            take_profit_pct: Optional take profit percentage
            stop_loss_pct: Optional stop loss percentage
            
        Returns:
            Tuple containing (take_profit_price, stop_loss_price)
        """
        signals_df = self.calculate_indicators(df)
        
        if signals_df.empty:
            return 0.0, 0.0
        
        # If percentages are provided, use them
        if take_profit_pct is not None and stop_loss_pct is not None:
            if side == 'BUY':
                take_profit = entry_price * (1 + take_profit_pct / 100)
                stop_loss = entry_price * (1 - stop_loss_pct / 100)
            else:  # SELL
                take_profit = entry_price * (1 - take_profit_pct / 100)
                stop_loss = entry_price * (1 + stop_loss_pct / 100)
                
            return take_profit, stop_loss
        
        # Calculate historical volatility for dynamic TP/SL
        # Using Average True Range (ATR) for the last 14 periods
        signals_df['true_range'] = np.maximum(
            signals_df['high'] - signals_df['low'],
            np.maximum(
                abs(signals_df['high'] - signals_df['close'].shift()),
                abs(signals_df['low'] - signals_df['close'].shift())
            )
        )
        
        atr = signals_df['true_range'].rolling(14).mean().iloc[-1]
        
        # Use ATR to set dynamic TP/SL
        if side == 'BUY':
            take_profit = entry_price + (atr * 3)  # 3x ATR for take profit
            stop_loss = entry_price - (atr * 1.5)  # 1.5x ATR for stop loss
        else:  # SELL
            take_profit = entry_price - (atr * 3)
            stop_loss = entry_price + (atr * 1.5)
            
        return take_profit, stop_loss
    
    def generate_chart(self, df: pd.DataFrame, title: str = 'Moving Average Crossover Strategy') -> str:
        """
        Generate a chart showing price, moving averages, and signals
        
        Args:
            df: DataFrame with OHLCV and indicator data
            title: Chart title
            
        Returns:
            Base64 encoded string of the chart image
        """
        signals_df = self.generate_signals(df)
        
        if signals_df.empty:
            return ""
        
        # Create the chart
        plt.figure(figsize=(12, 8))
        
        # Plot price and moving averages
        plt.subplot(2, 1, 1)
        plt.plot(signals_df.index, signals_df['close'], label='Price', color='black', alpha=0.75)
        plt.plot(signals_df.index, signals_df['fast_ma'], label=f'{self.ma_type}{self.fast_period}', color='blue')
        plt.plot(signals_df.index, signals_df['slow_ma'], label=f'{self.ma_type}{self.slow_period}', color='red')
        
        # Plot Bollinger Bands
        plt.plot(signals_df.index, signals_df['bb_upper'], 'g--', alpha=0.3)
        plt.plot(signals_df.index, signals_df['bb_middle'], 'g-', alpha=0.3)
        plt.plot(signals_df.index, signals_df['bb_lower'], 'g--', alpha=0.3)
        
        # Plot buy and sell signals
        buy_signals = signals_df[signals_df['buy_signal']]
        sell_signals = signals_df[signals_df['sell_signal']]
        
        plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
        plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot RSI
        plt.subplot(2, 1, 2)
        plt.plot(signals_df.index, signals_df['rsi'], label='RSI', color='purple')
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.3)
        plt.title('RSI Indicator')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode('utf-8')