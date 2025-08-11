"""
Interactive Brokers data source adapter.

Provides access to IB historical data through the ib_async library,
optimized for options arbitrage backtesting requirements.
"""

import asyncio
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import logging
from ib_async import IB, BarData, Contract, Option, Stock, Ticker

from .data_source_adapter import (
    DataFrequency,
    DataSourceAdapter,
    DataType,
    MarketDataPoint,
    MarketDataRequest,
)

logger = logging.getLogger(__name__)


class IBDataSource(DataSourceAdapter):
    """
    Interactive Brokers data source adapter.

    Provides historical market data through IB's API with support for:
    - Stock and ETF data
    - Options data with Greeks
    - Multiple data frequencies
    - Real-time and delayed data
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        timeout: int = 30,
        config: Dict[str, Any] = None,
    ):
        super().__init__("Interactive Brokers", config)

        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout

        # IB connection
        self.ib: Optional[IB] = None

        # Rate limiting
        self._last_request_time = 0
        self._request_count = 0
        self._max_requests_per_minute = 50  # Conservative limit

        # Set capabilities
        self._capabilities = {
            DataType.TRADES,
            DataType.QUOTES,
            DataType.MIDPOINT,
            DataType.BID_ASK,
            DataType.VOLUME,
            DataType.GREEKS,
            DataType.IMPLIED_VOL,
        }

        # Supported frequencies
        self._supported_frequencies = [
            DataFrequency.SECOND,
            DataFrequency.MINUTE,
            DataFrequency.FIVE_MINUTES,
            DataFrequency.FIFTEEN_MINUTES,
            DataFrequency.HOUR,
            DataFrequency.DAILY,
            DataFrequency.WEEKLY,
            DataFrequency.MONTHLY,
        ]

    async def connect(self) -> bool:
        """Connect to Interactive Brokers TWS/Gateway."""
        try:
            if self.ib is None:
                self.ib = IB()

            await self.ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.timeout,
            )

            self._connected = True
            logger.info(f"Connected to IB at {self.host}:{self.port}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            self._connected = False
            return False

    async def disconnect(self):
        """Disconnect from Interactive Brokers."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")

    async def get_stock_data(self, request: MarketDataRequest) -> List[MarketDataPoint]:
        """Get stock/ETF historical data from IB."""
        if not self.is_connected:
            raise RuntimeError("Not connected to IB")

        await self._rate_limit()

        try:
            # Create stock contract
            contract = Stock(request.symbol, "SMART", "USD")
            qualified = await self.ib.qualifyContractsAsync(contract)

            if not qualified:
                logger.warning(f"Could not qualify stock contract for {request.symbol}")
                return []

            contract = qualified[0]

            # Convert request parameters to IB format
            duration_str = self._calculate_duration_string(
                request.start_date, request.end_date
            )
            bar_size = self._convert_frequency_to_ib(request.frequency)
            what_to_show = self._convert_data_type_to_ib(request.data_type)

            # Request historical data
            bars = await self.ib.reqHistoricalDataAsync(
                contract=contract,
                endDateTime=datetime.combine(request.end_date, datetime.min.time()),
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=not request.include_extended_hours,
                formatDate=1,
                keepUpToDate=False,
            )

            # Convert to MarketDataPoint objects
            data_points = []
            for bar in bars:
                point = MarketDataPoint(
                    timestamp=bar.date,
                    symbol=request.symbol,
                    price=float(bar.close),
                    volume=int(bar.volume) if bar.volume else None,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    data_source=self.name,
                    data_quality=1.0,  # IB data is generally high quality
                )
                data_points.append(point)

            logger.debug(
                f"Retrieved {len(data_points)} data points for {request.symbol}"
            )

            return data_points

        except Exception as e:
            logger.error(f"Error getting stock data for {request.symbol}: {e}")
            raise

    async def get_option_data(
        self, request: MarketDataRequest
    ) -> List[MarketDataPoint]:
        """Get option historical data from IB."""
        if not self.is_connected:
            raise RuntimeError("Not connected to IB")

        if not all([request.strike, request.expiry, request.option_type]):
            raise ValueError("Option data requires strike, expiry, and option_type")

        await self._rate_limit()

        try:
            # Create option contract
            contract = Option(
                symbol=request.symbol,
                lastTradeDateOrContractMonth=request.expiry.strftime("%Y%m%d"),
                strike=request.strike,
                right=request.option_type,
                exchange="SMART",
            )

            qualified = await self.ib.qualifyContractsAsync(contract)

            if not qualified:
                logger.warning(
                    f"Could not qualify option contract for {request.symbol} "
                    f"{request.strike}{request.option_type} {request.expiry}"
                )
                return []

            contract = qualified[0]

            # Convert request parameters
            duration_str = self._calculate_duration_string(
                request.start_date, request.end_date
            )
            bar_size = self._convert_frequency_to_ib(request.frequency)
            what_to_show = "MIDPOINT"  # Most reliable for options

            # Request historical data
            bars = await self.ib.reqHistoricalDataAsync(
                contract=contract,
                endDateTime=datetime.combine(request.end_date, datetime.min.time()),
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=not request.include_extended_hours,
                formatDate=1,
                keepUpToDate=False,
            )

            # Convert to MarketDataPoint objects with option details
            data_points = []
            for bar in bars:
                # For options, we create realistic bid/ask from midpoint
                mid_price = float(bar.close)
                spread_estimate = max(0.05, mid_price * 0.02)  # 2% spread estimate

                point = MarketDataPoint(
                    timestamp=bar.date,
                    symbol=request.symbol,
                    price=mid_price,
                    volume=int(bar.volume) if bar.volume else None,
                    bid=max(0.01, mid_price - spread_estimate / 2),
                    ask=mid_price + spread_estimate / 2,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    strike=request.strike,
                    expiry=request.expiry,
                    option_type=request.option_type,
                    data_source=self.name,
                    data_quality=0.8,  # Slightly lower for options due to spread estimation
                )
                data_points.append(point)

            logger.debug(
                f"Retrieved {len(data_points)} option data points for "
                f"{request.symbol} {request.strike}{request.option_type} {request.expiry}"
            )

            return data_points

        except Exception as e:
            logger.error(
                f"Error getting option data for {request.symbol} "
                f"{request.strike}{request.option_type} {request.expiry}: {e}"
            )
            raise

    async def get_option_chain(
        self, symbol: str, expiry: date, date_range: Tuple[date, date] = None
    ) -> List[Dict[str, Any]]:
        """Get option chain for a symbol and expiry from IB."""
        if not self.is_connected:
            raise RuntimeError("Not connected to IB")

        await self._rate_limit()

        try:
            # Get underlying contract
            underlying = Stock(symbol, "SMART", "USD")
            qualified = await self.ib.qualifyContractsAsync(underlying)

            if not qualified:
                return []

            underlying = qualified[0]

            # Get option parameters
            chains = await self.ib.reqSecDefOptParamsAsync(
                underlyingSymbol=underlying.symbol,
                futFopExchange="",
                underlyingSecType=underlying.secType,
                underlyingConId=underlying.conId,
            )

            if not chains:
                return []

            # Find the chain that matches our criteria
            target_chain = None
            expiry_str = expiry.strftime("%Y%m%d")

            for chain in chains:
                if expiry_str in chain.expirations:
                    target_chain = chain
                    break

            if not target_chain:
                return []

            # Build option chain data
            option_chain = []

            for strike in target_chain.strikes:
                for right in ["C", "P"]:
                    contract_info = {
                        "symbol": symbol,
                        "expiry": expiry,
                        "strike": strike,
                        "right": right,
                        "exchange": target_chain.exchange,
                        "multiplier": target_chain.multiplier,
                        "tradingClass": target_chain.tradingClass,
                    }
                    option_chain.append(contract_info)

            logger.debug(
                f"Retrieved option chain for {symbol} {expiry}: "
                f"{len(option_chain)} contracts"
            )

            return option_chain

        except Exception as e:
            logger.error(f"Error getting option chain for {symbol} {expiry}: {e}")
            raise

    async def get_vix_data(
        self,
        start_date: date,
        end_date: date,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> List[MarketDataPoint]:
        """Get VIX data from IB."""
        vix_request = MarketDataRequest(
            symbol="VIX",
            start_date=start_date,
            end_date=end_date,
            data_type=DataType.TRADES,
            frequency=frequency,
        )

        return await self.get_stock_data(vix_request)

    def _calculate_duration_string(self, start_date: date, end_date: date) -> str:
        """Calculate IB duration string from date range."""
        delta = end_date - start_date
        days = delta.days

        if days <= 30:
            return f"{days} D"
        elif days <= 365:
            weeks = days // 7
            return f"{weeks} W"
        else:
            years = days // 365
            return f"{years} Y"

    def _convert_frequency_to_ib(self, frequency: DataFrequency) -> str:
        """Convert DataFrequency to IB bar size setting."""
        frequency_map = {
            DataFrequency.SECOND: "1 sec",
            DataFrequency.MINUTE: "1 min",
            DataFrequency.FIVE_MINUTES: "5 mins",
            DataFrequency.FIFTEEN_MINUTES: "15 mins",
            DataFrequency.HOUR: "1 hour",
            DataFrequency.DAILY: "1 day",
            DataFrequency.WEEKLY: "1W",
            DataFrequency.MONTHLY: "1M",
        }

        return frequency_map.get(frequency, "1 min")

    def _convert_data_type_to_ib(self, data_type: DataType) -> str:
        """Convert DataType to IB whatToShow parameter."""
        data_type_map = {
            DataType.TRADES: "TRADES",
            DataType.QUOTES: "BID_ASK",
            DataType.MIDPOINT: "MIDPOINT",
            DataType.BID_ASK: "BID_ASK",
            DataType.VOLUME: "TRADES",
        }

        return data_type_map.get(data_type, "TRADES")

    async def _rate_limit(self):
        """Implement rate limiting for IB API requests."""
        current_time = asyncio.get_event_loop().time()

        # Reset counter if more than a minute has passed
        if current_time - self._last_request_time > 60:
            self._request_count = 0
            self._last_request_time = current_time

        # Check if we need to wait
        if self._request_count >= self._max_requests_per_minute:
            wait_time = 60 - (current_time - self._last_request_time)
            if wait_time > 0:
                logger.info(f"Rate limiting: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self._request_count = 0
                self._last_request_time = asyncio.get_event_loop().time()

        self._request_count += 1
        await asyncio.sleep(0.1)  # Small delay between requests

    async def validate_data_availability(
        self, symbol: str, start_date: date, end_date: date
    ) -> Dict[str, Any]:
        """Validate data availability through IB."""
        if not self.is_connected:
            return {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "stock_data_available": False,
                "option_data_available": False,
                "data_quality_score": 0.0,
                "missing_dates": [],
                "notes": "Not connected to IB",
            }

        try:
            # Test stock data availability
            stock_request = MarketDataRequest(
                symbol=symbol,
                start_date=end_date - timedelta(days=1),  # Test recent data
                end_date=end_date,
                frequency=DataFrequency.DAILY,
            )

            stock_data = await self.get_stock_data(stock_request)
            stock_available = len(stock_data) > 0

            # Test option data availability (basic check)
            option_chains = await self.get_option_chain(
                symbol, end_date + timedelta(days=30)
            )
            option_available = len(option_chains) > 0

            return {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "stock_data_available": stock_available,
                "option_data_available": option_available,
                "data_quality_score": 0.9 if stock_available else 0.1,
                "missing_dates": [],
                "notes": f"Validated through IB API",
            }

        except Exception as e:
            logger.error(f"Error validating data availability for {symbol}: {e}")
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


# Utility function for easy IB connection setup
async def create_ib_data_source(
    host: str = "127.0.0.1", port: int = 7497, client_id: int = 1
) -> IBDataSource:
    """
    Create and connect an IB data source.

    Args:
        host: IB Gateway/TWS host
        port: IB Gateway/TWS port
        client_id: Client ID for connection

    Returns:
        Connected IBDataSource instance
    """
    ib_source = IBDataSource(host=host, port=port, client_id=client_id)

    connected = await ib_source.connect()
    if not connected:
        raise RuntimeError(f"Failed to connect to IB at {host}:{port}")

    return ib_source
