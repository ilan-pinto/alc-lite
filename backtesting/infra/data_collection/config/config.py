"""
Configuration settings for data collection and database connections.
"""

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5433"))
    database: str = os.getenv("DB_NAME", "options_arbitrage")
    user: str = os.getenv("DB_USER", "trading_user")
    password: str = os.getenv("DB_PASSWORD", "secure_trading_password")
    min_pool_size: int = 5
    max_pool_size: int = 20

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_connection_string(self) -> str:
        """Generate async PostgreSQL connection string."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class CollectionConfig:
    """Data collection configuration."""

    # Collection intervals
    tick_interval_ms: int = 100  # Milliseconds between tick collection
    batch_size: int = 1000  # Number of records to batch before insert
    flush_interval_seconds: int = 5  # Force flush interval

    # Market data subscription settings
    market_data_type: str = "DELAYED"  # DELAYED, REALTIME, or FROZEN
    include_greeks: bool = True
    generic_tick_list: str = "106"  # IB generic tick list for Greeks

    # Options chain filters
    expiry_range_days: int = 60  # Days ahead to collect options data
    strike_range_percent: float = (
        0.20  # Percentage range around ATM (e.g., 0.20 = ±20%)
    )
    min_volume: int = 0  # Minimum volume filter
    min_open_interest: int = 0  # Minimum open interest filter

    # Symbols to track
    default_symbols: List[str] = None

    # Data quality settings
    max_spread_percent: float = 0.10  # Maximum bid-ask spread as % of mid price
    stale_data_threshold_seconds: int = (
        300  # Mark data as stale after this many seconds
    )

    # Performance settings
    max_concurrent_contracts: int = 100  # Maximum contracts to track simultaneously
    request_throttle_ms: int = 50  # Milliseconds between IB API requests

    def __post_init__(self):
        if self.default_symbols is None:
            self.default_symbols = [
                "SPY",
                "QQQ",
                "IWM",
                "DIA",  # Major ETFs
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",  # Tech giants
                "TSLA",
                "META",
                "NVDA",
                "AMD",  # High volume tech
                "JPM",
                "BAC",
                "GS",
                "WFC",  # Financials
                "XOM",
                "CVX",
                "COP",
                "SLB",  # Energy
            ]


@dataclass
class HistoricalConfig:
    """Historical data loading configuration."""

    # IB Historical data settings
    bar_size: str = (
        "1 min"  # 1 secs, 5 secs, 10 secs, 15 secs, 30 secs, 1 min, 2 mins, 3 mins, 5 mins, 10 mins, 15 mins, 20 mins, 30 mins, 1 hour, 2 hours, 3 hours, 4 hours, 8 hours, 1 day, 1W, 1M
    )
    duration: str = "1 D"  # Format: <integer> <unit> where unit is S|D|W|M|Y
    what_to_show: str = (
        "MIDPOINT"  # TRADES, MIDPOINT, BID, ASK, BID_ASK, HISTORICAL_VOLATILITY, OPTION_IMPLIED_VOLATILITY
    )
    use_rth: bool = True  # Regular trading hours only

    # Backfill settings
    max_days_per_request: int = 30  # Maximum days to request at once
    request_delay_seconds: float = 0.5  # Delay between historical requests
    retry_attempts: int = 3  # Number of retries for failed requests
    retry_delay_seconds: float = 10.0  # Delay between retries

    # Data processing
    validate_data: bool = True  # Run validation on historical data
    fill_missing_data: bool = True  # Interpolate missing data points
    adjust_for_splits: bool = True  # Adjust historical data for splits


# Global configuration instances
db_config = DatabaseConfig()
collection_config = CollectionConfig()
historical_config = HistoricalConfig()
