"""
Base data source adapter for flexible data loading.

Provides a common interface for different data sources to support
both live and historical data loading for backtesting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd


class DataFrequency(Enum):
    """Supported data frequencies."""

    TICK = "tick"
    SECOND = "1sec"
    MINUTE = "1min"
    FIVE_MINUTES = "5min"
    FIFTEEN_MINUTES = "15min"
    HOUR = "1hour"
    DAILY = "1day"
    WEEKLY = "1week"
    MONTHLY = "1month"


class DataType(Enum):
    """Types of market data."""

    TRADES = "trades"
    QUOTES = "quotes"
    MIDPOINT = "midpoint"
    BID_ASK = "bid_ask"
    VOLUME = "volume"
    GREEKS = "greeks"
    IMPLIED_VOL = "implied_vol"


@dataclass
class MarketDataRequest:
    """Request for market data."""

    symbol: str
    start_date: date
    end_date: date
    data_type: DataType = DataType.TRADES
    frequency: DataFrequency = DataFrequency.MINUTE
    include_extended_hours: bool = False
    adjust_for_splits: bool = True
    adjust_for_dividends: bool = False

    # Option-specific parameters
    strike: Optional[float] = None
    expiry: Optional[date] = None
    option_type: Optional[str] = None  # 'C' or 'P'

    # Quality filters
    min_volume: int = 0
    max_spread_percent: Optional[float] = None


@dataclass
class MarketDataPoint:
    """Single market data point."""

    timestamp: datetime
    symbol: str
    price: float
    volume: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None

    # Option-specific data
    strike: Optional[float] = None
    expiry: Optional[date] = None
    option_type: Optional[str] = None

    # Greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    implied_vol: Optional[float] = None

    # Metadata
    data_source: Optional[str] = None
    data_quality: Optional[float] = None


class DataSourceAdapter(ABC):
    """
    Abstract base class for data source adapters.

    All data sources (IB, CSV, APIs) should implement this interface
    to provide consistent access to market data for backtesting.
    """

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self._connected = False
        self._capabilities = set()

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the data source.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the data source."""
        pass

    @abstractmethod
    async def get_stock_data(self, request: MarketDataRequest) -> List[MarketDataPoint]:
        """
        Get stock/ETF historical data.

        Args:
            request: Data request parameters

        Returns:
            List of market data points
        """
        pass

    @abstractmethod
    async def get_option_data(
        self, request: MarketDataRequest
    ) -> List[MarketDataPoint]:
        """
        Get option historical data.

        Args:
            request: Option data request parameters

        Returns:
            List of option market data points
        """
        pass

    @abstractmethod
    async def get_option_chain(
        self, symbol: str, expiry: date, date_range: Tuple[date, date] = None
    ) -> List[Dict[str, Any]]:
        """
        Get option chain for a symbol and expiry.

        Args:
            symbol: Underlying symbol
            expiry: Option expiry date
            date_range: Date range for historical chain data

        Returns:
            List of option contract details
        """
        pass

    async def get_vix_data(
        self,
        start_date: date,
        end_date: date,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> List[MarketDataPoint]:
        """
        Get VIX data for correlation analysis.

        Args:
            start_date: Start date
            end_date: End date
            frequency: Data frequency

        Returns:
            List of VIX data points
        """
        # Default implementation - to be overridden by specific adapters
        raise NotImplementedError(f"{self.name} adapter does not support VIX data")

    async def get_earnings_dates(
        self, symbol: str, start_date: date, end_date: date
    ) -> List[date]:
        """
        Get earnings announcement dates.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            List of earnings dates
        """
        # Default implementation
        return []

    async def get_dividend_dates(
        self, symbol: str, start_date: date, end_date: date
    ) -> List[Tuple[date, float]]:
        """
        Get dividend ex-dates and amounts.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            List of (ex_date, amount) tuples
        """
        # Default implementation
        return []

    def supports_data_type(self, data_type: DataType) -> bool:
        """Check if adapter supports a specific data type."""
        return data_type in self._capabilities

    def supports_frequency(self, frequency: DataFrequency) -> bool:
        """Check if adapter supports a specific frequency."""
        return frequency in getattr(self, "_supported_frequencies", [])

    @property
    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        return self._connected

    @property
    def capabilities(self) -> set:
        """Get adapter capabilities."""
        return self._capabilities.copy()

    def to_dataframe(
        self, data_points: List[MarketDataPoint], include_greeks: bool = True
    ) -> pd.DataFrame:
        """
        Convert market data points to pandas DataFrame.

        Args:
            data_points: List of market data points
            include_greeks: Whether to include Greeks columns

        Returns:
            DataFrame with market data
        """
        if not data_points:
            return pd.DataFrame()

        # Base columns
        data = []
        for point in data_points:
            row = {
                "timestamp": point.timestamp,
                "symbol": point.symbol,
                "price": point.price,
                "volume": point.volume,
                "bid": point.bid,
                "ask": point.ask,
                "open": point.open,
                "high": point.high,
                "low": point.low,
                "close": point.close,
            }

            # Option-specific columns
            if point.strike is not None:
                row.update(
                    {
                        "strike": point.strike,
                        "expiry": point.expiry,
                        "option_type": point.option_type,
                    }
                )

            # Greeks
            if include_greeks:
                row.update(
                    {
                        "delta": point.delta,
                        "gamma": point.gamma,
                        "theta": point.theta,
                        "vega": point.vega,
                        "rho": point.rho,
                        "implied_vol": point.implied_vol,
                    }
                )

            # Metadata
            row.update(
                {
                    "data_source": point.data_source,
                    "data_quality": point.data_quality,
                }
            )

            data.append(row)

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    async def validate_data_availability(
        self, symbol: str, start_date: date, end_date: date
    ) -> Dict[str, Any]:
        """
        Validate data availability for a symbol and date range.

        Args:
            symbol: Symbol to check
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with availability information
        """
        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "stock_data_available": False,
            "option_data_available": False,
            "data_quality_score": 0.0,
            "missing_dates": [],
            "notes": f"Data availability check not implemented for {self.name}",
        }

    async def get_data_quality_metrics(
        self, symbol: str, start_date: date, end_date: date
    ) -> Dict[str, float]:
        """
        Get data quality metrics for a symbol and date range.

        Returns:
            Dictionary with quality metrics (completeness, consistency, etc.)
        """
        return {
            "completeness": 0.0,
            "consistency": 0.0,
            "timeliness": 0.0,
            "accuracy": 0.0,
            "overall_score": 0.0,
        }


class DataSourceManager:
    """
    Manager for multiple data sources with failover support.

    Allows using multiple data sources in priority order for redundancy
    and data quality improvement.
    """

    def __init__(self):
        self.sources: List[DataSourceAdapter] = []
        self.primary_source: Optional[DataSourceAdapter] = None
        self._connection_status = {}

    def add_source(
        self, adapter: DataSourceAdapter, priority: int = 0, is_primary: bool = False
    ):
        """
        Add a data source adapter.

        Args:
            adapter: Data source adapter
            priority: Priority (higher = more preferred)
            is_primary: Whether this is the primary source
        """
        self.sources.append(adapter)
        self.sources.sort(key=lambda x: getattr(x, "priority", 0), reverse=True)

        if is_primary:
            self.primary_source = adapter

        adapter.priority = priority

    async def connect_all(self) -> Dict[str, bool]:
        """
        Connect to all data sources.

        Returns:
            Dictionary mapping source names to connection status
        """
        results = {}

        for source in self.sources:
            try:
                connected = await source.connect()
                results[source.name] = connected
                self._connection_status[source.name] = connected
            except Exception as e:
                results[source.name] = False
                self._connection_status[source.name] = False
                print(f"Failed to connect to {source.name}: {e}")

        return results

    async def get_best_stock_data(
        self, request: MarketDataRequest, max_attempts: int = 3
    ) -> Tuple[List[MarketDataPoint], str]:
        """
        Get stock data from the best available source.

        Args:
            request: Data request
            max_attempts: Maximum number of sources to try

        Returns:
            Tuple of (data_points, source_name)
        """
        attempts = 0

        for source in self.sources:
            if attempts >= max_attempts:
                break

            if not source.is_connected:
                continue

            try:
                data = await source.get_stock_data(request)
                if data:  # Success
                    return data, source.name
            except Exception as e:
                print(f"Failed to get data from {source.name}: {e}")

            attempts += 1

        return [], "none"

    async def disconnect_all(self):
        """Disconnect from all data sources."""
        for source in self.sources:
            try:
                await source.disconnect()
                self._connection_status[source.name] = False
            except Exception as e:
                print(f"Error disconnecting from {source.name}: {e}")

    def get_connection_status(self) -> Dict[str, bool]:
        """Get connection status of all sources."""
        return self._connection_status.copy()

    def get_available_sources(self) -> List[str]:
        """Get list of connected source names."""
        return [
            name for name, connected in self._connection_status.items() if connected
        ]
