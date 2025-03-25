from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import os

@dataclass
class TradingStrategyConfig:
    # Moving Average Configuration
    fast_ma_period: int = 20       # Fast moving average period (short-term)
    slow_ma_period: int = 50       # Slow moving average period (long-term)
    ma_type: str = "EMA"           # Type of moving average: "SMA", "EMA", "WMA"
    timeframe: str = "1h"          # Timeframe for analysis
    
    # Position sizing and risk management
    min_position_size: float = 0.00001  # Minimum BTC position size (reduced even more for testing)
    max_position_size: float = 0.1    # Maximum BTC position size
    risk_per_trade: float = 0.03      # 3% risk per trade (increased slightly)
    take_profit: float = 3.0          # 3% take profit
    stop_loss: float = 1.5            # 1.5% stop loss
    
    # Additional parameters
    max_active_trades: int = 3        # Maximum concurrent trades
    min_volume: float = 100          # Minimum volume for BTC/USDT trading

@dataclass
class DatabaseConfig:
    host: str = field(default_factory=lambda: os.getenv('DB_HOST', 'localhost'))
    port: int = field(default_factory=lambda: int(os.getenv('DB_PORT', '5432')))
    database: str = field(default_factory=lambda: os.getenv('DB_NAME', 'trading_bot'))
    user: str = field(default_factory=lambda: os.getenv('DB_USER', 'admin'))
    password: str = field(default_factory=lambda: os.getenv('DB_PASSWORD', ''))

@dataclass
class BinanceConfig:
    api_key: str = field(default_factory=lambda: os.getenv('BINANCE_API_KEY', ''))
    api_secret: str = field(default_factory=lambda: os.getenv('BINANCE_API_SECRET', ''))
    base_url: str = "https://api.binance.com"
    testnet: bool = False  # Set to True to use testnet
    rate_limit: float = 10.0  # Requests per second
    timeout: int = 30
    cache_duration: int = 60  # 1 minute cache for responses
    retries: int = 3  # Number of retry attempts
    retry_delay: int = 1  # Seconds between retries

@dataclass
class Config:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    strategy: TradingStrategyConfig = field(default_factory=TradingStrategyConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    max_total_exposure: float = 0.5     # Maximum 0.5 BTC total exposure
    default_pair: str = field(default_factory=lambda: os.getenv('DEFAULT_PAIR', 'BTC/USDT'))  # Default trading pair
    max_trades_per_day: int = 5         # Maximum number of trades per day
    backtest_mode: bool = field(default_factory=lambda: os.getenv('BACKTEST_MODE', 'false').lower() == 'true')  # Enable/disable backtesting mode
    slippage_percent: float = 0.1       # Default slippage percentage