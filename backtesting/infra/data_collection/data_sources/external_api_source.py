"""
External API data source adapter.

Provides access to external data APIs like Alpha Vantage, Yahoo Finance,
Quandl, and other financial data providers for backtesting.
"""

import asyncio
import json
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import logging

from .data_source_adapter import (
    DataFrequency,
    DataSourceAdapter,
    DataType,
    MarketDataPoint,
    MarketDataRequest,
)

logger = logging.getLogger(__name__)


class ExternalAPISource(DataSourceAdapter):
    """
    External API data source adapter.

    Supports multiple API providers:
    - Alpha Vantage (free tier available)
    - Yahoo Finance (via yfinance-like endpoints)
    - Quandl (NASDAQ Data Link)
    - Federal Reserve Economic Data (FRED)
    """

    def __init__(
        self,
        provider: str = "alphavantage",
        api_key: Optional[str] = None,
        config: Dict[str, Any] = None,
    ):
        super().__init__(f"External API ({provider})", config)

        self.provider = provider.lower()
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None

        # Rate limiting
        self._request_count = 0
        self._last_reset_time = datetime.now()

        # Provider-specific configurations
        self.provider_config = self._get_provider_config()

        # Set capabilities based on provider
        self._capabilities = self.provider_config.get(
            "capabilities",
            {
                DataType.TRADES,
                DataType.VOLUME,
            },
        )

        self._supported_frequencies = self.provider_config.get(
            "frequencies", [DataFrequency.DAILY]
        )

    def _get_provider_config(self) -> Dict[str, Any]:
        """Get provider-specific configuration."""
        configs = {
            "alphavantage": {
                "base_url": "https://www.alphavantage.co/query",
                "rate_limit": 5,  # requests per minute (free tier)
                "rate_window": 60,  # seconds
                "capabilities": {DataType.TRADES, DataType.VOLUME},
                "frequencies": [
                    DataFrequency.MINUTE,
                    DataFrequency.FIVE_MINUTES,
                    DataFrequency.FIFTEEN_MINUTES,
                    DataFrequency.HOUR,
                    DataFrequency.DAILY,
                    DataFrequency.WEEKLY,
                    DataFrequency.MONTHLY,
                ],
                "requires_api_key": True,
            },
            "yahoo": {
                "base_url": "https://query1.finance.yahoo.com/v8/finance/chart",
                "rate_limit": 60,  # More generous
                "rate_window": 60,
                "capabilities": {DataType.TRADES, DataType.VOLUME},
                "frequencies": [
                    DataFrequency.MINUTE,
                    DataFrequency.FIVE_MINUTES,
                    DataFrequency.HOUR,
                    DataFrequency.DAILY,
                    DataFrequency.WEEKLY,
                    DataFrequency.MONTHLY,
                ],
                "requires_api_key": False,
            },
            "fred": {
                "base_url": "https://api.stlouisfed.org/fred/series/observations",
                "rate_limit": 120,  # requests per minute
                "rate_window": 60,
                "capabilities": {DataType.TRADES},
                "frequencies": [
                    DataFrequency.DAILY,
                    DataFrequency.WEEKLY,
                    DataFrequency.MONTHLY,
                ],
                "requires_api_key": True,
            },
        }

        return configs.get(self.provider, configs["yahoo"])

    async def connect(self) -> bool:
        """Connect to external API (create session)."""
        try:
            # Check API key requirement
            if self.provider_config.get("requires_api_key") and not self.api_key:
                logger.error(f"{self.provider} requires API key")
                return False

            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)

            # Test connection with a simple request
            test_success = await self._test_connection()

            if test_success:
                self._connected = True
                logger.info(f"Connected to {self.provider} API")
                return True
            else:
                await self.session.close()
                self.session = None
                return False

        except Exception as e:
            logger.error(f"Failed to connect to {self.provider} API: {e}")
            self._connected = False
            return False

    async def disconnect(self):
        """Disconnect from external API."""
        if self.session:
            await self.session.close()
            self.session = None

        self._connected = False
        logger.info(f"Disconnected from {self.provider} API")

    async def get_stock_data(self, request: MarketDataRequest) -> List[MarketDataPoint]:
        """Get stock data from external API."""
        if not self.is_connected:
            raise RuntimeError(f"Not connected to {self.provider} API")

        await self._rate_limit()

        try:
            if self.provider == "alphavantage":
                return await self._get_alphavantage_stock_data(request)
            elif self.provider == "yahoo":
                return await self._get_yahoo_stock_data(request)
            elif self.provider == "fred":
                return await self._get_fred_data(request)
            else:
                raise NotImplementedError(f"Provider {self.provider} not implemented")

        except Exception as e:
            logger.error(f"Error getting stock data from {self.provider}: {e}")
            raise

    async def get_option_data(
        self, request: MarketDataRequest
    ) -> List[MarketDataPoint]:
        """Get option data from external API."""
        # Most free APIs don't provide options data
        if self.provider in ["yahoo", "fred"]:
            logger.warning(f"{self.provider} does not provide options data")
            return []

        if self.provider == "alphavantage":
            logger.warning("Alpha Vantage options data requires premium subscription")
            return []

        return []

    async def get_option_chain(
        self, symbol: str, expiry: date, date_range: Tuple[date, date] = None
    ) -> List[Dict[str, Any]]:
        """Get option chain from external API."""
        logger.warning(f"{self.provider} does not provide option chain data")
        return []

    async def get_vix_data(
        self,
        start_date: date,
        end_date: date,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> List[MarketDataPoint]:
        """Get VIX data from external API."""
        vix_request = MarketDataRequest(
            symbol="VIX" if self.provider != "fred" else "VIXCLS",
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
        )

        return await self.get_stock_data(vix_request)

    async def _test_connection(self) -> bool:
        """Test API connection."""
        try:
            if self.provider == "alphavantage":
                params = {
                    "function": "TIME_SERIES_DAILY",
                    "symbol": "SPY",
                    "outputsize": "compact",
                    "apikey": self.api_key,
                }
                async with self.session.get(
                    self.provider_config["base_url"], params=params
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return (
                            "Time Series (Daily)" in data or "Error Message" not in data
                        )

            elif self.provider == "yahoo":
                url = f"{self.provider_config['base_url']}/SPY"
                async with self.session.get(url) as resp:
                    return resp.status == 200

            elif self.provider == "fred":
                params = {
                    "series_id": "VIXCLS",
                    "api_key": self.api_key,
                    "file_type": "json",
                    "limit": "1",
                }
                async with self.session.get(
                    self.provider_config["base_url"], params=params
                ) as resp:
                    return resp.status == 200

            return False

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    async def _get_alphavantage_stock_data(
        self, request: MarketDataRequest
    ) -> List[MarketDataPoint]:
        """Get stock data from Alpha Vantage."""
        # Map frequency to Alpha Vantage function
        function_map = {
            DataFrequency.MINUTE: "TIME_SERIES_INTRADAY",
            DataFrequency.FIVE_MINUTES: "TIME_SERIES_INTRADAY",
            DataFrequency.FIFTEEN_MINUTES: "TIME_SERIES_INTRADAY",
            DataFrequency.HOUR: "TIME_SERIES_INTRADAY",
            DataFrequency.DAILY: "TIME_SERIES_DAILY",
            DataFrequency.WEEKLY: "TIME_SERIES_WEEKLY",
            DataFrequency.MONTHLY: "TIME_SERIES_MONTHLY",
        }

        interval_map = {
            DataFrequency.MINUTE: "1min",
            DataFrequency.FIVE_MINUTES: "5min",
            DataFrequency.FIFTEEN_MINUTES: "15min",
            DataFrequency.HOUR: "60min",
        }

        function = function_map.get(request.frequency, "TIME_SERIES_DAILY")

        params = {
            "function": function,
            "symbol": request.symbol,
            "apikey": self.api_key,
            "outputsize": "full",  # Get more historical data
        }

        # Add interval for intraday data
        if function == "TIME_SERIES_INTRADAY":
            params["interval"] = interval_map.get(request.frequency, "1min")

        async with self.session.get(
            self.provider_config["base_url"], params=params
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Alpha Vantage API error: {resp.status}")

            data = await resp.json()

            # Check for API errors
            if "Error Message" in data:
                raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")

            if "Note" in data:
                logger.warning(f"Alpha Vantage note: {data['Note']}")
                return []

            # Find time series data
            time_series_key = None
            for key in data.keys():
                if "Time Series" in key:
                    time_series_key = key
                    break

            if not time_series_key:
                logger.warning(f"No time series data found for {request.symbol}")
                return []

            time_series = data[time_series_key]

            # Convert to MarketDataPoint objects
            data_points = []
            for date_str, values in time_series.items():
                try:
                    timestamp = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    timestamp = datetime.strptime(date_str, "%Y-%m-%d")

                # Filter by date range
                if not (request.start_date <= timestamp.date() <= request.end_date):
                    continue

                point = MarketDataPoint(
                    timestamp=timestamp,
                    symbol=request.symbol,
                    price=float(values["4. close"]),
                    volume=(
                        int(values.get("5. volume", 0))
                        if values.get("5. volume")
                        else None
                    ),
                    open=float(values["1. open"]),
                    high=float(values["2. high"]),
                    low=float(values["3. low"]),
                    close=float(values["4. close"]),
                    data_source=self.name,
                    data_quality=0.9,  # Alpha Vantage is high quality
                )
                data_points.append(point)

            # Sort by timestamp
            data_points.sort(key=lambda x: x.timestamp)

            logger.debug(
                f"Retrieved {len(data_points)} data points from Alpha Vantage for {request.symbol}"
            )

            return data_points

    async def _get_yahoo_stock_data(
        self, request: MarketDataRequest
    ) -> List[MarketDataPoint]:
        """Get stock data from Yahoo Finance."""
        # Calculate period parameters
        start_ts = int(
            datetime.combine(request.start_date, datetime.min.time()).timestamp()
        )
        end_ts = int(
            datetime.combine(request.end_date, datetime.max.time()).timestamp()
        )

        # Map frequency to Yahoo interval
        interval_map = {
            DataFrequency.MINUTE: "1m",
            DataFrequency.FIVE_MINUTES: "5m",
            DataFrequency.HOUR: "1h",
            DataFrequency.DAILY: "1d",
            DataFrequency.WEEKLY: "1wk",
            DataFrequency.MONTHLY: "1mo",
        }

        interval = interval_map.get(request.frequency, "1d")

        url = f"{self.provider_config['base_url']}/{request.symbol}"
        params = {
            "period1": start_ts,
            "period2": end_ts,
            "interval": interval,
            "events": "history",
        }

        async with self.session.get(url, params=params) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Yahoo Finance API error: {resp.status}")

            data = await resp.json()

            if "chart" not in data or not data["chart"]["result"]:
                logger.warning(
                    f"No data returned from Yahoo Finance for {request.symbol}"
                )
                return []

            result = data["chart"]["result"][0]

            # Extract data arrays
            timestamps = result["timestamp"]
            quotes = result["indicators"]["quote"][0]

            # Convert to MarketDataPoint objects
            data_points = []
            for i, ts in enumerate(timestamps):
                timestamp = datetime.fromtimestamp(ts)

                # Skip if any essential data is None
                if any(
                    quotes[field][i] is None
                    for field in ["open", "high", "low", "close"]
                ):
                    continue

                point = MarketDataPoint(
                    timestamp=timestamp,
                    symbol=request.symbol,
                    price=quotes["close"][i],
                    volume=quotes.get("volume", [None] * len(timestamps))[i],
                    open=quotes["open"][i],
                    high=quotes["high"][i],
                    low=quotes["low"][i],
                    close=quotes["close"][i],
                    data_source=self.name,
                    data_quality=0.8,  # Yahoo data is generally good
                )
                data_points.append(point)

            logger.debug(
                f"Retrieved {len(data_points)} data points from Yahoo Finance for {request.symbol}"
            )

            return data_points

    async def _get_fred_data(self, request: MarketDataRequest) -> List[MarketDataPoint]:
        """Get data from Federal Reserve Economic Data."""
        params = {
            "series_id": request.symbol,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": request.start_date.strftime("%Y-%m-%d"),
            "observation_end": request.end_date.strftime("%Y-%m-%d"),
        }

        async with self.session.get(
            self.provider_config["base_url"], params=params
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"FRED API error: {resp.status}")

            data = await resp.json()

            if "observations" not in data:
                logger.warning(
                    f"No observations found in FRED data for {request.symbol}"
                )
                return []

            # Convert to MarketDataPoint objects
            data_points = []
            for obs in data["observations"]:
                if obs["value"] == ".":  # FRED uses '.' for missing values
                    continue

                timestamp = datetime.strptime(obs["date"], "%Y-%m-%d")
                value = float(obs["value"])

                point = MarketDataPoint(
                    timestamp=timestamp,
                    symbol=request.symbol,
                    price=value,
                    close=value,
                    data_source=self.name,
                    data_quality=1.0,  # FRED data is very high quality
                )
                data_points.append(point)

            logger.debug(
                f"Retrieved {len(data_points)} data points from FRED for {request.symbol}"
            )

            return data_points

    async def _rate_limit(self):
        """Apply rate limiting based on provider limits."""
        current_time = datetime.now()

        # Reset counter if time window has passed
        time_diff = (current_time - self._last_reset_time).total_seconds()
        if time_diff >= self.provider_config["rate_window"]:
            self._request_count = 0
            self._last_reset_time = current_time

        # Check if we need to wait
        if self._request_count >= self.provider_config["rate_limit"]:
            wait_time = self.provider_config["rate_window"] - time_diff
            if wait_time > 0:
                logger.info(
                    f"Rate limiting: waiting {wait_time:.1f}s for {self.provider}"
                )
                await asyncio.sleep(wait_time)
                self._request_count = 0
                self._last_reset_time = datetime.now()

        self._request_count += 1
        await asyncio.sleep(0.1)  # Small delay between requests

    async def validate_data_availability(
        self, symbol: str, start_date: date, end_date: date
    ) -> Dict[str, Any]:
        """Validate data availability through API."""
        if not self.is_connected:
            return {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "stock_data_available": False,
                "option_data_available": False,
                "data_quality_score": 0.0,
                "missing_dates": [],
                "notes": f"Not connected to {self.provider}",
            }

        try:
            # Test with a small date range
            test_request = MarketDataRequest(
                symbol=symbol,
                start_date=end_date - timedelta(days=5),
                end_date=end_date,
                frequency=DataFrequency.DAILY,
            )

            test_data = await self.get_stock_data(test_request)
            stock_available = len(test_data) > 0

            return {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "stock_data_available": stock_available,
                "option_data_available": False,  # Most APIs don't provide options
                "data_quality_score": 0.8 if stock_available else 0.1,
                "missing_dates": [],
                "notes": f"Validated through {self.provider} API",
            }

        except Exception as e:
            logger.error(f"Error validating data for {symbol}: {e}")
            return {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "stock_data_available": False,
                "option_data_available": False,
                "data_quality_score": 0.0,
                "missing_dates": [],
                "notes": f"Validation error: {str(e)}",
            }


# Utility functions for easy API setup


async def create_alphavantage_source(api_key: str) -> ExternalAPISource:
    """Create and connect Alpha Vantage data source."""
    source = ExternalAPISource("alphavantage", api_key)
    connected = await source.connect()
    if not connected:
        raise RuntimeError("Failed to connect to Alpha Vantage")
    return source


async def create_yahoo_source() -> ExternalAPISource:
    """Create and connect Yahoo Finance data source."""
    source = ExternalAPISource("yahoo")
    connected = await source.connect()
    if not connected:
        raise RuntimeError("Failed to connect to Yahoo Finance")
    return source


async def create_fred_source(api_key: str) -> ExternalAPISource:
    """Create and connect FRED data source."""
    source = ExternalAPISource("fred", api_key)
    connected = await source.connect()
    if not connected:
        raise RuntimeError("Failed to connect to FRED")
    return source
