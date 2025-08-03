"""
Historical data loader for Interactive Brokers.
Fetches and stores historical options and stock data for backtesting.
"""

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import logging
import numpy as np
import pandas as pd
from ib_async import IB, BarData, Contract, Option, Stock

from .config import DatabaseConfig, HistoricalConfig
from .validators import DataValidator

logger = logging.getLogger(__name__)


@dataclass
class HistoricalRequest:
    """Represents a historical data request."""

    contract: Contract
    end_date: datetime
    duration: str
    bar_size: str
    what_to_show: str
    use_rth: bool
    contract_id: Optional[int] = None
    underlying_id: Optional[int] = None


class HistoricalDataLoader:
    """
    Loads historical options and stock data from Interactive Brokers.
    Handles rate limiting, error recovery, and data validation.
    """

    def __init__(
        self, db_pool: asyncpg.Pool, ib_connection: IB, config: HistoricalConfig = None
    ):
        self.db_pool = db_pool
        self.ib = ib_connection
        self.config = config or HistoricalConfig()

        # Request management
        self.pending_requests: List[HistoricalRequest] = []
        self.completed_requests = 0
        self.failed_requests = 0

        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        self.request_window = 60  # 60 seconds
        self.max_requests_per_window = 60  # IB limit

        # Initialize validator
        self.validator = DataValidator(db_pool)

    async def load_symbol_history(
        self, symbol: str, start_date: date, end_date: date = None
    ) -> Dict[str, Any]:
        """
        Load complete historical data for a symbol including stock and options.

        Args:
            symbol: Stock symbol
            start_date: Start date for historical data
            end_date: End date (default: today)

        Returns:
            Dictionary with loading statistics
        """
        end_date = end_date or date.today()
        logger.info(f"Loading history for {symbol} from {start_date} to {end_date}")

        stats = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "stock_bars_loaded": 0,
            "option_bars_loaded": 0,
            "errors": 0,
        }

        try:
            # Load stock data first
            stock_bars = await self._load_stock_history(symbol, start_date, end_date)
            stats["stock_bars_loaded"] = stock_bars

            # Get option chains for the period
            option_contracts = await self._get_historical_option_contracts(
                symbol, start_date, end_date
            )

            # Load option data
            for contract_batch in self._batch_contracts(option_contracts, 10):
                batch_bars = await self._load_option_batch_history(
                    contract_batch, start_date, end_date
                )
                stats["option_bars_loaded"] += batch_bars

        except Exception as e:
            logger.error(f"Error loading history for {symbol}: {e}")
            stats["errors"] += 1

        return stats

    async def _load_stock_history(
        self, symbol: str, start_date: date, end_date: date
    ) -> int:
        """Load historical stock data."""
        # Get stock contract
        stock = Stock(symbol, "SMART", "USD")
        qualified = await self.ib.qualifyContractsAsync(stock)

        if not qualified:
            logger.error(f"Could not qualify stock {symbol}")
            return 0

        stock = qualified[0]

        # Get or create underlying record
        underlying_id = await self._ensure_underlying_exists(symbol)
        if not underlying_id:
            return 0

        # Load data in chunks
        total_bars = 0
        current_end = end_date

        while current_end > start_date:
            # Calculate duration
            duration_days = min(
                self.config.max_days_per_request, (current_end - start_date).days
            )

            if duration_days <= 0:
                break

            duration_str = f"{duration_days} D"

            # Create request
            request = HistoricalRequest(
                contract=stock,
                end_date=datetime.combine(current_end, datetime.min.time()),
                duration=duration_str,
                bar_size=self.config.bar_size,
                what_to_show="TRADES",
                use_rth=self.config.use_rth,
                underlying_id=underlying_id,
            )

            # Execute request
            bars = await self._execute_historical_request(request)

            if bars:
                # Store in database
                stored = await self._store_stock_bars(underlying_id, bars)
                total_bars += stored

                # Move to next chunk
                if bars:
                    first_bar_date = bars[0].date.date()
                    current_end = first_bar_date - timedelta(days=1)
                else:
                    break
            else:
                break

            # Rate limiting
            await self._rate_limit_delay()

        logger.info(f"Loaded {total_bars} stock bars for {symbol}")
        return total_bars

    async def _get_historical_option_contracts(
        self, symbol: str, start_date: date, end_date: date
    ) -> List[Tuple[Contract, int]]:
        """Get option contracts that were active during the period."""
        contracts = []

        async with self.db_pool.acquire() as conn:
            # Query contracts that were active during the period
            rows = await conn.fetch(
                """
                SELECT oc.id, oc.ib_con_id, oc.expiration_date,
                       oc.strike_price, oc.option_type
                FROM option_chains oc
                JOIN underlying_securities us ON oc.underlying_id = us.id
                WHERE us.symbol = $1
                  AND oc.expiration_date >= $2
                  AND oc.expiration_date <= $3 + INTERVAL '60 days'
                  AND oc.ib_con_id IS NOT NULL
                ORDER BY oc.expiration_date, oc.strike_price
            """,
                symbol,
                start_date,
                end_date,
            )

            for row in rows:
                # Create option contract
                option = Option(
                    symbol=symbol,
                    lastTradeDateOrContractMonth=row["expiration_date"].strftime(
                        "%Y%m%d"
                    ),
                    strike=float(row["strike_price"]),
                    right=row["option_type"],
                    exchange="SMART",
                )
                option.conId = row["ib_con_id"]

                contracts.append((option, row["id"]))

        return contracts

    async def _load_option_batch_history(
        self, contracts: List[Tuple[Contract, int]], start_date: date, end_date: date
    ) -> int:
        """Load historical data for a batch of option contracts."""
        total_bars = 0

        for contract, contract_id in contracts:
            try:
                # For options, we typically want shorter duration requests
                # due to lower liquidity and data availability
                bars_loaded = await self._load_single_option_history(
                    contract, contract_id, start_date, end_date
                )
                total_bars += bars_loaded

            except Exception as e:
                logger.error(f"Error loading option {contract.conId}: {e}")
                self.failed_requests += 1

            await self._rate_limit_delay()

        return total_bars

    async def _load_single_option_history(
        self, contract: Contract, contract_id: int, start_date: date, end_date: date
    ) -> int:
        """Load history for a single option contract."""
        total_bars = 0
        current_end = end_date

        # Options often have limited historical data
        # Adjust request size accordingly
        while current_end > start_date:
            duration_days = min(5, (current_end - start_date).days)

            if duration_days <= 0:
                break

            duration_str = f"{duration_days} D"

            request = HistoricalRequest(
                contract=contract,
                end_date=datetime.combine(current_end, datetime.min.time()),
                duration=duration_str,
                bar_size="1 min",  # Higher resolution for options
                what_to_show="MIDPOINT",  # More reliable for options
                use_rth=self.config.use_rth,
                contract_id=contract_id,
            )

            bars = await self._execute_historical_request(request)

            if bars:
                stored = await self._store_option_bars(contract_id, bars)
                total_bars += stored

                # Move to next chunk
                first_bar_date = bars[0].date.date()
                current_end = first_bar_date - timedelta(days=1)
            else:
                # No data available, try earlier period
                current_end = current_end - timedelta(days=duration_days)

        return total_bars

    async def _execute_historical_request(
        self, request: HistoricalRequest
    ) -> Optional[List[BarData]]:
        """Execute a historical data request with retry logic."""
        for attempt in range(self.config.retry_attempts):
            try:
                bars = await self.ib.reqHistoricalDataAsync(
                    contract=request.contract,
                    endDateTime=request.end_date,
                    durationStr=request.duration,
                    barSizeSetting=request.bar_size,
                    whatToShow=request.what_to_show,
                    useRTH=request.use_rth,
                    formatDate=1,  # DATE_TIME format
                    keepUpToDate=False,
                )

                self.completed_requests += 1
                return bars

            except Exception as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")

                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)
                else:
                    self.failed_requests += 1
                    raise

        return None

    async def _store_stock_bars(self, underlying_id: int, bars: List[BarData]) -> int:
        """Store stock bars in database."""
        if not bars:
            return 0

        async with self.db_pool.acquire() as conn:
            values = []

            for bar in bars:
                values.append(
                    (
                        bar.date,  # Already datetime
                        underlying_id,
                        float(bar.close),
                        None,  # bid_price
                        None,  # ask_price
                        None,  # bid_size
                        None,  # ask_size
                        int(bar.volume) if bar.volume else None,
                        float(bar.average) if bar.average else None,  # vwap
                        float(bar.open),
                        float(bar.high),
                        float(bar.low),
                        float(bar.close),
                        "HISTORICAL",
                    )
                )

            # Bulk insert with conflict handling
            result = await conn.executemany(
                """
                INSERT INTO stock_data_ticks
                (time, underlying_id, price, bid_price, ask_price,
                 bid_size, ask_size, volume, vwap,
                 open_price, high_price, low_price, close_price, tick_type)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (time, underlying_id) DO NOTHING
            """,
                values,
            )

            # Extract count from result
            count = int(result.split()[-1]) if result else len(values)
            logger.debug(f"Stored {count} stock bars")
            return count

    async def _store_option_bars(self, contract_id: int, bars: List[BarData]) -> int:
        """Store option bars in database."""
        if not bars:
            return 0

        async with self.db_pool.acquire() as conn:
            values = []

            for bar in bars:
                # For options, we store the midpoint as both bid and ask
                # with a small spread for more realistic backtesting
                midpoint = float(bar.close)
                half_spread = midpoint * 0.01  # 1% spread assumption

                values.append(
                    (
                        bar.date,
                        contract_id,
                        max(0.01, midpoint - half_spread),  # bid_price
                        midpoint + half_spread,  # ask_price
                        midpoint,  # last_price
                        100,  # bid_size (assumed)
                        100,  # ask_size (assumed)
                        None,  # last_size
                        int(bar.volume) if bar.volume else None,
                        None,  # open_interest (not available)
                        None,  # Greeks will be calculated separately
                        None,
                        None,
                        None,
                        None,
                        None,
                        "HISTORICAL",
                    )
                )

            # Bulk insert
            result = await conn.executemany(
                """
                INSERT INTO market_data_ticks
                (time, contract_id, bid_price, ask_price, last_price,
                 bid_size, ask_size, last_size, volume, open_interest,
                 delta, gamma, theta, vega, rho, implied_volatility, tick_type)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17)
                ON CONFLICT DO NOTHING
            """,
                values,
            )

            count = int(result.split()[-1]) if result else len(values)
            logger.debug(f"Stored {count} option bars")
            return count

    async def _ensure_underlying_exists(self, symbol: str) -> Optional[int]:
        """Ensure underlying security exists in database."""
        async with self.db_pool.acquire() as conn:
            underlying_id = await conn.fetchval(
                """
                INSERT INTO underlying_securities (symbol, active)
                VALUES ($1, true)
                ON CONFLICT (symbol) DO UPDATE
                SET updated_at = CURRENT_TIMESTAMP
                RETURNING id
            """,
                symbol,
            )
            return underlying_id

    async def _rate_limit_delay(self):
        """Implement rate limiting for IB API requests."""
        current_time = asyncio.get_event_loop().time()

        # Reset counter if window has passed
        if current_time - self.last_request_time > self.request_window:
            self.request_count = 0
            self.last_request_time = current_time

        # Check if we need to wait
        if self.request_count >= self.max_requests_per_window:
            wait_time = self.request_window - (current_time - self.last_request_time)
            if wait_time > 0:
                logger.info(f"Rate limiting: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.last_request_time = asyncio.get_event_loop().time()

        self.request_count += 1
        await asyncio.sleep(self.config.request_delay_seconds)

    def _batch_contracts(self, contracts: List, batch_size: int):
        """Yield batches of contracts."""
        for i in range(0, len(contracts), batch_size):
            yield contracts[i : i + batch_size]

    async def backfill_missing_data(self, symbol: str, days_back: int = 30):
        """
        Identify and backfill missing data for a symbol.

        Args:
            symbol: Stock symbol
            days_back: Number of days to check
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)

        logger.info(f"Checking for missing data: {symbol} last {days_back} days")

        async with self.db_pool.acquire() as conn:
            # Find gaps in stock data
            gaps = await conn.fetch(
                """
                WITH date_series AS (
                    SELECT generate_series(
                        $2::date, $3::date, '1 day'::interval
                    )::date AS trading_date
                ),
                existing_dates AS (
                    SELECT DISTINCT DATE(time) as data_date
                    FROM stock_data_ticks st
                    JOIN underlying_securities us ON st.underlying_id = us.id
                    WHERE us.symbol = $1
                      AND time >= $2
                      AND time < $3 + interval '1 day'
                )
                SELECT ds.trading_date
                FROM date_series ds
                LEFT JOIN existing_dates ed ON ds.trading_date = ed.data_date
                WHERE ed.data_date IS NULL
                  AND EXTRACT(dow FROM ds.trading_date) NOT IN (0, 6)  -- Exclude weekends
                ORDER BY ds.trading_date
            """,
                symbol,
                start_date,
                end_date,
            )

            if gaps:
                logger.info(f"Found {len(gaps)} days with missing data")

                # Load data for missing days
                for gap in gaps:
                    gap_date = gap["trading_date"]
                    await self.load_symbol_history(symbol, gap_date, gap_date)

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        return {
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "pending_requests": len(self.pending_requests),
        }
