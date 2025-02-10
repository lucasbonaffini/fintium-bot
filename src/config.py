from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import os

@dataclass
class FilterConfig:
    min_liquidity: float = 10000  # Minimum liquidity in USD
    min_market_cap: float = 50000  # Minimum market cap in USD
    min_holders: int = 100  # Minimum number of holders
    max_holder_percentage: float = 15.0  # Maximum percentage a single holder can have
    min_age_hours: int = 24  # Minimum token age in hours
    # Trading parameters
    min_position_size: float = 0.1    # Mínimo 0.1 SOL
    max_position_size: float = 1.0    # Máximo 1 SOL
    max_market_impact: float = 1.0    # Máximo 1% de impacto
    risk_multiplier: float = 0.01     # 1% de la liquidez como base
    volatility_adjust: bool = True    # Ajustar por volatilidad
    take_profit: float = 300.0        # 300% take profit
    stop_loss: float = 10.0           # Slippage máximo permitido
    forbidden_names: Set[str] = field(default_factory=lambda: {
        "test", "scam", "rug", "honeypot", "squid"
    })
    suspicious_patterns: Set[str] = field(default_factory=lambda: {
        r"^test\d*$",
        r".*copyr.*",
        r".*©.*",
    })

@dataclass
class BlacklistConfig:
    # Known malicious token addresses
    blacklisted_tokens: Set[str] = field(default_factory=set)
    # Known malicious developer addresses
    blacklisted_developers: Set[str] = field(default_factory=set)
    # Blacklisted contract patterns
    blacklisted_contract_patterns: Set[str] = field(default_factory=lambda: {
        r".*honeypot.*",
        r".*rug.*",
    })
    # Load blacklists from files
    blacklist_files: Dict[str, str] = field(default_factory=lambda: {
        'tokens': 'blacklists/tokens.txt',
        'developers': 'blacklists/developers.txt',
        'contracts': 'blacklists/contracts.txt'
    })

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "dexscreener"
    user: str = "admin"
    password: str = ""

@dataclass
class DexScreenerConfig:
    base_url: str = "https://api.dexscreener.com/latest"
    rate_limit: float = 1.0
    timeout: int = 30
    
@dataclass
class SolscanConfig:
    api_key: str = field(default_factory=lambda: os.getenv('SOLSCAN_API_KEY', ''))
    base_url: str = field(default_factory=lambda: os.getenv('SOLSCAN_BASE_URL', 'https://public-api.solscan.io'))
    rate_limit: float = field(default_factory=lambda: float(os.getenv('SOLSCAN_RATE_LIMIT', '1.0')))
    timeout: int = 30
    cache_duration: int = 300  # 5 minutes cache for responses
    retries: int = 3  # Number of retry attempts
    retry_delay: int = 2  # Seconds between retries

@dataclass
class Config:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    blacklists: BlacklistConfig = field(default_factory=BlacklistConfig)
    solscan: SolscanConfig = field(default_factory=SolscanConfig)
    log_level: str = "INFO"
    max_total_exposure: float = 100000  # Máxima exposición total permitida
    max_position_size: float = 10000    # Tamaño máximo de posición individual
    default_gas_gwei: int = 50          # Gas por defecto en GWEI
    slippage_percent: float = 1.0       # Slippage por defecto
    take_profit_percent: float = 50.0   # Take profit por defecto
    stop_loss_percent: float = 10.0     # Stop loss por defecto