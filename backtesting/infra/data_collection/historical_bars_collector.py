#!/usr/bin/env python3
"""
Historical Bars Collector for 5-minute Options Data
Designed for laptop-based collection with intermittent connectivity
Implements rate limiting and smart batching for IB API compliance
"""

import asyncio
import os
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import argparse
import asyncpg
import logging
import pytz
import yaml
from ib_async import IB, BarData, Contract, Option, Stock

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from backtesting.infra.data_collection.config.config import (
    CollectionConfig,
    DatabaseConfig,
)
from backtesting.infra.data_collection.core.contract_utils import ContractFactory


@dataclass
class CollectionStats:
    """Track collection statistics for monitoring."""

    contracts_requested: int = 0
    contracts_successful: int = 0
    bars_collected: int = 0
    bars_updated: int = 0
    bars_skipped: int = 0
    errors: int = 0
    rate_limit_hits: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    error_details: List[Dict[str, Any]] = field(default_factory=list)


class RateLimiter:
    """
    Rate limiter for IB API compliance.
    Ensures minimum 10 seconds between historical data requests.
    """

    def __init__(self, min_interval_seconds: float = 10.0):
        self.min_interval = min_interval_seconds
        self.last_request_times: Dict[str, float] = {}
        self.lock = asyncio.Lock()

    async def acquire(self, key: str = "default"):
        """Wait if necessary to comply with rate limit."""
        async with self.lock:
            now = time.time()
            if key in self.last_request_times:
                elapsed = now - self.last_request_times[key]
                if elapsed < self.min_interval:
                    wait_time = self.min_interval - elapsed
                    logging.debug(
                        f"Rate limiting: waiting {wait_time:.1f}s for key {key}"
                    )
                    await asyncio.sleep(wait_time)

            self.last_request_times[key] = time.time()


class HistoricalBarsCollector:
    """
    Collects 5-minute historical bars for options with rate limit compliance.
    Designed for laptop-based collection with intermittent connectivity.
    """

    def __init__(
        self,
        config_path: str = None,
        connection_timeout: int = 30,
        connection_retries: int = 3,
    ):
        """
        Initialize the historical bars collector.

        Args:
            config_path: Path to configuration file
            connection_timeout: IB Gateway connection timeout in seconds
            connection_retries: Number of connection retry attempts
        """
        self.config_path = config_path or "scheduler/intraday_collection_israel.yaml"
        self.config = self._load_config()

        # Connection parameters
        self.connection_timeout = connection_timeout
        self.connection_retries = connection_retries

        # Timezone setup
        self.israel_tz = pytz.timezone(self.config["timezone"])
        self.et_tz = pytz.timezone("US/Eastern")

        # Initialize logging
        self._setup_logging()

        # Rate limiting (single connection mode)
        rate_config = self.config.get("rate_limits", {})
        self.rate_limiter = RateLimiter(
            min_interval_seconds=1.0 / rate_config.get("max_requests_per_second", 0.067)
        )

        # Single IB connection (simplified)
        self.ib_connection: Optional[IB] = None
        self.single_connection_mode = rate_config.get("single_connection_mode", True)

        # Database
        self.db_pool = None

        # Collection state
        self.collection_run_id = None
        self.stats = CollectionStats()

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(self.config_path)
        if not config_path.exists():
            config_path = Path(__file__).parent.parent.parent.parent / self.config_path

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Configure logging with appropriate formatting."""
        log_config = self.config.get("logging", {})

        # Create logs directory if needed
        Path("logs").mkdir(exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("logs/historical_bars_collector.log"),
            ],
        )

        self.logger = logging.getLogger(__name__)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        sys.exit(0)

    async def initialize_connections(self):
        """Initialize database and single IB connection with retry logic."""
        # Database connection
        db_config = self.config["database"]
        self.db_pool = await asyncpg.create_pool(
            host=db_config["host"],
            port=db_config["port"],
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"],
            min_size=5,
            max_size=db_config.get("pool_size", 10),
        )
        self.logger.info("Database connection established")

        # IB Gateway connection with retry logic
        import ib_async

        self.logger.info(
            f"Connecting to IB Gateway (timeout={self.connection_timeout}s, retries={self.connection_retries})..."
        )
        self.logger.info(f"Using ib_async version: {ib_async.__version__}")

        self.ib_connection = IB()

        # Retry logic for connection with signal handler protection
        last_exception = None
        for attempt in range(self.connection_retries):
            try:
                self.logger.info(
                    f"Connection attempt {attempt + 1}/{self.connection_retries}"
                )

                # Temporarily disable signal handlers during connection to prevent cancellation
                self.logger.debug(
                    "Temporarily disabling signal handlers for connection stability"
                )
                old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
                old_sigterm = signal.signal(signal.SIGTERM, signal.SIG_IGN)

                try:
                    # Try async connection first
                    await self.ib_connection.connectAsync(
                        "127.0.0.1", 7497, clientId=3, timeout=self.connection_timeout
                    )
                    self.logger.info(
                        "✓ IB Gateway connection established successfully (async)"
                    )
                    return  # Success, exit retry loop

                finally:
                    # Always restore signal handlers
                    signal.signal(signal.SIGINT, old_sigint)
                    signal.signal(signal.SIGTERM, old_sigterm)
                    self.logger.debug("Signal handlers restored")

            except asyncio.CancelledError as e:
                last_exception = e
                self.logger.warning(
                    f"Connection was cancelled on attempt {attempt + 1}/{self.connection_retries} - possibly by shell environment or signal"
                )

                # Try synchronous connection as fallback for cancellation issues
                if attempt < self.connection_retries - 1:
                    self.logger.info(
                        "Trying synchronous connection method as fallback..."
                    )
                    try:
                        # Disconnect first to clean up
                        if self.ib_connection.isConnected():
                            self.ib_connection.disconnect()

                        # Create fresh connection object
                        self.ib_connection = IB()

                        # Use synchronous connect method
                        self.ib_connection.connect(
                            "127.0.0.1",
                            7497,
                            clientId=3,
                            timeout=self.connection_timeout,
                        )
                        self.logger.info(
                            "✓ IB Gateway connection established successfully (sync fallback)"
                        )
                        return

                    except Exception as sync_error:
                        self.logger.error(
                            f"Synchronous connection fallback failed: {sync_error}"
                        )
                        last_exception = sync_error

                if attempt < self.connection_retries - 1:
                    wait_time = (attempt + 1) * 5
                    self.logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)

            except asyncio.TimeoutError as e:
                last_exception = e
                self.logger.warning(
                    f"Connection timeout on attempt {attempt + 1}/{self.connection_retries}"
                )
                if attempt < self.connection_retries - 1:
                    wait_time = (attempt + 1) * 5  # Progressive backoff: 5s, 10s, 15s
                    self.logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)

            except Exception as e:
                last_exception = e
                self.logger.error(
                    f"Connection error on attempt {attempt + 1}/{self.connection_retries}: {type(e).__name__}: {e}"
                )
                if attempt < self.connection_retries - 1:
                    wait_time = (attempt + 1) * 5
                    self.logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)

        # If we get here, all retries failed
        self.logger.error(
            f"❌ All {self.connection_retries} connection attempts failed"
        )
        if isinstance(last_exception, asyncio.CancelledError):
            raise RuntimeError(
                f"IB Gateway connection was repeatedly cancelled after {self.connection_retries} attempts. "
                f"This may be caused by shell script signals, system interrupts, or environment issues. "
                f"Try running the script interactively or check for signal conflicts."
            ) from last_exception
        elif isinstance(last_exception, asyncio.TimeoutError):
            raise RuntimeError(
                f"IB Gateway connection timed out after {self.connection_retries} attempts. "
                f"Please ensure TWS/Gateway is running on port 7497 with API enabled."
            ) from last_exception
        else:
            raise RuntimeError(
                f"IB Gateway connection failed after {self.connection_retries} attempts: {type(last_exception).__name__}: {last_exception}"
            ) from last_exception

    async def get_ib_connection(self) -> IB:
        """Get the single IB connection."""
        if not self.ib_connection:
            raise Exception("No IB connection available")
        return self.ib_connection

    async def collect_historical_bars(
        self,
        symbols: List[str],
        collection_type: str = "manual",
        duration: str = None,
        end_datetime: str = "",
    ):
        """
        Main collection method for historical bars.

        Args:
            symbols: List of symbols to collect
            collection_type: Type of collection run
            duration: Duration string (e.g., "1 D", "2 hours")
            end_datetime: End datetime for historical data (empty string for now)
        """
        self.stats = CollectionStats()
        self.logger.info(
            f"Starting {collection_type} collection for symbols: {symbols}"
        )

        try:
            # Initialize connections
            await self.initialize_connections()

            # Record collection start
            self.collection_run_id = await self._record_collection_start(
                collection_type, symbols, duration
            )

            # Collect for each symbol
            for symbol in symbols:
                await self._collect_symbol_bars(symbol, duration, end_datetime)

            # Record completion
            await self._record_collection_complete()

            # Generate summary
            self._log_collection_summary()

        except Exception as e:
            self.logger.error(f"Collection failed: {e}", exc_info=True)
            self.stats.errors += 1
            self.stats.error_details.append(
                {"error": str(e), "timestamp": datetime.now(self.israel_tz)}
            )
            if self.collection_run_id:
                await self._record_collection_error(str(e))
            raise

        finally:
            await self._cleanup()

    async def _collect_symbol_bars(self, symbol: str, duration: str, end_datetime: str):
        """Collect historical bars for stock and all option contracts of a symbol."""
        self.logger.info(f"Collecting bars for {symbol}")

        try:
            # First collect stock bars for the underlying (if enabled)
            collect_stock_data = self.config.get("collection_params", {}).get(
                "collect_stock_data", True
            )
            self.logger.info(f"🔧 Stock data collection enabled: {collect_stock_data}")

            if collect_stock_data:
                self.logger.info(f"🎯 About to call _collect_stock_bars for {symbol}")
                await self._collect_stock_bars([symbol], duration, end_datetime)
                self.logger.info(f"✅ Completed _collect_stock_bars for {symbol}")
            else:
                self.logger.warning(f"⚠️ Stock data collection disabled for {symbol}")

            # Then get option contracts for this symbol
            contracts = await self._get_option_contracts(symbol)
            self.logger.info(f"Found {len(contracts)} option contracts for {symbol}")

            # Collect bars for each contract with rate limiting
            batch_size = self.config["rate_limits"].get("batch_size_per_connection", 20)

            for i in range(0, len(contracts), batch_size):
                batch = contracts[i : i + batch_size]
                await self._collect_contract_batch(batch, duration, end_datetime)

                # Pause between batches
                pause = self.config["rate_limits"].get(
                    "pause_between_batches_seconds", 5
                )
                if i + batch_size < len(contracts):
                    self.logger.debug(f"Pausing {pause}s between batches...")
                    await asyncio.sleep(pause)

        except Exception as e:
            self.logger.error(f"Failed to collect {symbol}: {e}")
            self.stats.errors += 1
            self.stats.error_details.append(
                {
                    "symbol": symbol,
                    "error": str(e),
                    "timestamp": datetime.now(self.israel_tz),
                }
            )

    async def _get_option_contracts(self, symbol: str) -> List[Tuple[Contract, int]]:
        """Get all relevant option contracts for a symbol from database."""
        contracts = []

        async with self.db_pool.acquire() as conn:
            # Get active option contracts
            rows = await conn.fetch(
                """
                SELECT
                    oc.id as contract_id,
                    oc.strike_price,
                    oc.option_type,
                    oc.expiration_date,
                    oc.ib_con_id,
                    oc.exchange,
                    us.symbol
                FROM option_chains oc
                JOIN underlying_securities us ON oc.underlying_id = us.id
                WHERE us.symbol = $1
                  AND oc.expiration_date >= CURRENT_DATE
                  AND oc.expiration_date <= CURRENT_DATE + INTERVAL '60 days'
                ORDER BY oc.expiration_date, oc.strike_price
            """,
                symbol,
            )

            for row in rows:
                # Create IB contract
                option = Option(
                    symbol=row["symbol"],
                    lastTradeDateOrContractMonth=row["expiration_date"].strftime(
                        "%Y%m%d"
                    ),
                    strike=float(row["strike_price"]),
                    right=row["option_type"],
                    exchange=row["exchange"] or "SMART",
                )
                if row["ib_con_id"]:
                    option.conId = row["ib_con_id"]

                contracts.append((option, row["contract_id"]))

        return contracts

    async def _get_stock_contracts(
        self, symbols: List[str]
    ) -> List[Tuple[Contract, int]]:
        """Get stock contracts for underlying symbols."""
        contracts = []

        async with self.db_pool.acquire() as conn:
            # Get underlying securities for the symbols
            rows = await conn.fetch(
                """
                SELECT id, symbol
                FROM underlying_securities
                WHERE symbol = ANY($1::text[])
                  AND active = true
                ORDER BY symbol
            """,
                symbols,
            )

            for row in rows:
                # Create IB stock contract
                stock = Stock(symbol=row["symbol"], exchange="SMART", currency="USD")

                contracts.append((stock, row["id"]))

        return contracts

    async def _collect_stock_bars(
        self, symbols: List[str], duration: str, end_datetime: str
    ):
        """Collect historical bars for stock symbols."""
        self.logger.info(f"🚀 Starting stock bars collection for symbols: {symbols}")
        try:
            # Get stock contracts
            stock_contracts = await self._get_stock_contracts(symbols)
            self.logger.info(f"Found {len(stock_contracts)} stock contracts")

            if not stock_contracts:
                self.logger.warning(
                    f"❌ No stock contracts found for symbols: {symbols}"
                )
                return

            # Get the single IB connection
            ib = await self.get_ib_connection()
            self.logger.info(f"📡 Got IB connection for stock data collection")

            # Collect bars for each stock contract
            self.logger.info(f"🔄 Processing {len(stock_contracts)} stock contracts")
            for i, (stock_contract, underlying_id) in enumerate(stock_contracts):
                self.logger.info(
                    f"📈 Processing stock contract {i+1}/{len(stock_contracts)}: {stock_contract.symbol} (ID: {underlying_id})"
                )
                try:
                    # Rate limit per stock
                    await self.rate_limiter.acquire(f"stock_{underlying_id}")

                    self.stats.contracts_requested += 1

                    self.logger.info(
                        f"🔍 Requesting stock bars for {stock_contract.symbol}"
                    )
                    # Request historical bars for stock
                    bars = await ib.reqHistoricalDataAsync(
                        stock_contract,
                        endDateTime=end_datetime,
                        durationStr=duration,
                        barSizeSetting="5 mins",
                        whatToShow="TRADES",
                        useRTH=True,
                        formatDate=1,
                    )

                    self.logger.info(
                        f"📊 Received {len(bars) if bars else 0} bars for {stock_contract.symbol}"
                    )

                    if bars:
                        # Store stock bars in database
                        self.logger.info(
                            f"💾 Storing {len(bars)} stock bars for {stock_contract.symbol}"
                        )
                        bars_collected = await self._store_stock_bars(
                            underlying_id, bars
                        )
                        self.stats.bars_collected += bars_collected
                        self.stats.contracts_successful += 1

                        self.logger.info(
                            f"✅ Collected {bars_collected} stock bars for {stock_contract.symbol}"
                        )
                    else:
                        self.logger.warning(
                            f"❌ No stock bars returned for {stock_contract.symbol}"
                        )

                except Exception as e:
                    if "No market data permissions" in str(e):
                        self.logger.warning(
                            f"No market data permissions for stock {stock_contract.symbol}"
                        )
                    elif "pacing violation" in str(e).lower():
                        self.logger.warning(
                            f"Rate limit hit for stock {stock_contract.symbol}"
                        )
                        self.stats.rate_limit_hits += 1
                        # Wait extra time before continuing
                        await asyncio.sleep(30)
                    else:
                        self.logger.error(
                            f"Error collecting stock {stock_contract.symbol}: {e}"
                        )
                        self.stats.errors += 1
                        raise

        except Exception as e:
            self.logger.error(f"❌ Failed to collect stock bars: {e}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            import traceback

            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            self.stats.errors += 1
            raise

    async def _collect_contract_batch(
        self, contracts: List[Tuple[Contract, int]], duration: str, end_datetime: str
    ):
        """Collect historical bars for contracts sequentially (single connection)."""
        # Get the single IB connection
        ib = await self.get_ib_connection()

        # Process contracts one at a time (no parallel processing)
        for contract, contract_id in contracts:
            try:
                # Rate limit per contract
                await self.rate_limiter.acquire(f"contract_{contract_id}")

                self.stats.contracts_requested += 1

                # Collect data for this contract
                bars_collected = await self._collect_single_contract(
                    ib, contract, contract_id, duration, end_datetime
                )

                if bars_collected > 0:
                    self.stats.contracts_successful += 1
                    self.logger.debug(
                        f"✅ Collected {bars_collected} bars for contract {contract_id}"
                    )
                else:
                    self.logger.debug(f"⚠️ No bars collected for contract {contract_id}")

            except Exception as e:
                self.logger.error(f"❌ Contract {contract_id} failed: {e}")
                self.stats.errors += 1

                # Add small delay before continuing to next contract
                await asyncio.sleep(2)

    async def _collect_single_contract(
        self,
        ib: IB,
        contract: Contract,
        contract_id: int,
        duration: str,
        end_datetime: str,
    ) -> int:
        """Collect historical bars for a single option contract."""
        bars_collected = 0

        try:
            # Request historical bars
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_datetime,
                durationStr=duration,
                barSizeSetting="5 mins",
                whatToShow="TRADES",  # Can be TRADES, BID_ASK, or OPTION_IMPLIED_VOLATILITY
                useRTH=True,
                formatDate=1,
            )

            if bars:
                # Store bars in database
                bars_collected = await self._store_bars(contract_id, bars)
                self.stats.bars_collected += bars_collected

                self.logger.debug(
                    f"Collected {bars_collected} bars for {contract.symbol} "
                    f"{contract.strike} {contract.right} {contract.lastTradeDateOrContractMonth}"
                )
            else:
                self.logger.debug(f"No bars returned for contract {contract_id}")

        except Exception as e:
            if "No market data permissions" in str(e):
                self.logger.warning(
                    f"No market data permissions for contract {contract_id}"
                )
            elif "pacing violation" in str(e).lower():
                self.logger.warning(f"Rate limit hit for contract {contract_id}")
                self.stats.rate_limit_hits += 1
                # Wait extra time before continuing
                await asyncio.sleep(30)
            else:
                self.logger.error(f"Error collecting contract {contract_id}: {e}")
                raise

        return bars_collected

    async def _store_bars(self, contract_id: int, bars: List[BarData]) -> int:
        """Store historical bars in database with deduplication."""
        if not bars:
            return 0

        stored_count = 0

        async with self.db_pool.acquire() as conn:
            for bar in bars:
                try:
                    # bar.date is already a datetime object
                    if isinstance(bar.date, str):
                        # Legacy handling if it's a string
                        bar_time = datetime.strptime(bar.date, "%Y%m%d %H:%M:%S")
                        bar_time = self.et_tz.localize(bar_time)
                    else:
                        # bar.date is already a datetime object
                        bar_time = bar.date
                        # If it's naive, localize it to ET
                        if bar_time.tzinfo is None:
                            bar_time = self.et_tz.localize(bar_time)
                        elif bar_time.tzinfo != self.et_tz:
                            # Convert to ET if in different timezone
                            bar_time = bar_time.astimezone(self.et_tz)

                    # Insert or update bar
                    await conn.execute(
                        """
                        INSERT INTO option_bars_5min (
                            time, contract_id,
                            open, high, low, close,
                            volume, bar_count,
                            collection_run_id, data_source
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (contract_id, time)
                        DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            bar_count = EXCLUDED.bar_count,
                            collection_run_id = EXCLUDED.collection_run_id,
                            updated_at = CURRENT_TIMESTAMP
                    """,
                        bar_time,
                        contract_id,
                        float(bar.open),
                        float(bar.high),
                        float(bar.low),
                        float(bar.close),
                        int(bar.volume),
                        int(bar.barCount),
                        self.collection_run_id,
                        "TRADES",
                    )
                    stored_count += 1

                except asyncpg.exceptions.UniqueViolationError:
                    # Bar already exists, skip
                    self.stats.bars_skipped += 1
                except Exception as e:
                    self.logger.error(f"Error storing bar: {e}")

        return stored_count

    async def _store_stock_bars(self, underlying_id: int, bars: List[BarData]) -> int:
        """Store historical stock bars in database with deduplication."""
        if not bars:
            self.logger.warning("📦 No bars provided to store")
            return 0

        self.logger.info(
            f"📦 Storing {len(bars)} stock bars for underlying_id {underlying_id}"
        )
        stored_count = 0

        async with self.db_pool.acquire() as conn:
            for i, bar in enumerate(bars):
                self.logger.debug(f"💾 Processing bar {i+1}/{len(bars)}: {bar.date}")
                try:
                    # bar.date is already a datetime object
                    if isinstance(bar.date, str):
                        # Legacy handling if it's a string
                        bar_time = datetime.strptime(bar.date, "%Y%m%d %H:%M:%S")
                        bar_time = self.et_tz.localize(bar_time)
                    else:
                        # bar.date is already a datetime object
                        bar_time = bar.date
                        # If it's naive, localize it to ET
                        if bar_time.tzinfo is None:
                            bar_time = self.et_tz.localize(bar_time)
                        elif bar_time.tzinfo != self.et_tz:
                            # Convert to ET if in different timezone
                            bar_time = bar_time.astimezone(self.et_tz)

                    # Prepare data for insert
                    insert_data = (
                        bar_time,
                        underlying_id,
                        float(bar.close),  # Use close as the primary price
                        float(bar.open),
                        float(bar.high),
                        float(bar.low),
                        float(bar.close),
                        int(bar.volume),
                        (
                            float(bar.average)
                            if hasattr(bar, "average") and bar.average
                            else None
                        ),
                        "TRADES",
                    )

                    self.logger.info(
                        f"💾 About to insert stock bar {i+1}: time={bar_time}, underlying_id={underlying_id}, price={float(bar.close)}"
                    )
                    self.logger.debug(f"Full insert data: {insert_data}")

                    # Insert or update stock bar
                    result = await conn.execute(
                        """
                        INSERT INTO stock_data_ticks (
                            time, underlying_id,
                            price, open_price, high_price, low_price, close_price,
                            volume, vwap,
                            tick_type
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (time, underlying_id)
                        DO UPDATE SET
                            price = EXCLUDED.close_price,
                            open_price = EXCLUDED.open_price,
                            high_price = EXCLUDED.high_price,
                            low_price = EXCLUDED.low_price,
                            close_price = EXCLUDED.close_price,
                            volume = EXCLUDED.volume,
                            vwap = EXCLUDED.vwap
                    """,
                        *insert_data,
                    )

                    self.logger.info(f"💾 Database execute result: {result}")
                    stored_count += 1
                    self.logger.debug(f"✅ Successfully stored stock bar {i+1}")

                except asyncpg.exceptions.UniqueViolationError:
                    # Bar already exists, skip
                    self.stats.bars_skipped += 1
                    self.logger.debug(f"⚠️ Skipped duplicate stock bar {i+1}")
                except Exception as e:
                    self.logger.error(f"❌ Error storing stock bar {i+1}: {e}")
                    self.logger.error(f"Bar data: {bar}")
                    import traceback

                    self.logger.error(f"Stack trace: {traceback.format_exc()}")

        return stored_count

    async def _record_collection_start(
        self, collection_type: str, symbols: List[str], duration: str
    ) -> int:
        """Record collection run start in database."""
        async with self.db_pool.acquire() as conn:
            collection_id = await conn.fetchval(
                """
                INSERT INTO intraday_collection_runs (
                    run_date, run_type, scheduled_time, started_at,
                    symbols_requested, duration_requested, bar_size, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, 'running')
                ON CONFLICT (run_date, run_type)
                DO UPDATE SET
                    started_at = $4,
                    symbols_requested = $5,
                    duration_requested = $6,
                    status = 'running'
                RETURNING id
            """,
                date.today(),
                collection_type,
                datetime.now(self.israel_tz),
                datetime.now(self.israel_tz),
                symbols,
                duration,
                "5 mins",
            )

        self.logger.info(f"Collection run started with ID: {collection_id}")
        return collection_id

    async def _record_collection_complete(self):
        """Record collection run completion."""
        self.stats.end_time = datetime.now()
        duration_seconds = int(
            (self.stats.end_time - self.stats.start_time).total_seconds()
        )

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE intraday_collection_runs
                SET completed_at = $1,
                    status = CASE WHEN errors = 0 THEN 'success' ELSE 'partial' END,
                    contracts_requested = $2,
                    contracts_successful = $3,
                    bars_collected = $4,
                    bars_updated = $5,
                    bars_skipped = $6,
                    errors = $7,
                    rate_limit_hits = $8,
                    duration_seconds = $9,
                    error_details = $10
                WHERE id = $11
            """,
                datetime.now(self.israel_tz),
                self.stats.contracts_requested,
                self.stats.contracts_successful,
                self.stats.bars_collected,
                self.stats.bars_updated,
                self.stats.bars_skipped,
                self.stats.errors,
                self.stats.rate_limit_hits,
                duration_seconds,
                (
                    None
                    if not self.stats.error_details
                    else {"errors": self.stats.error_details}
                ),
                self.collection_run_id,
            )

    async def _record_collection_error(self, error_msg: str):
        """Record collection run error."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE intraday_collection_runs
                SET status = 'failed',
                    error_details = $1,
                    completed_at = $2
                WHERE id = $3
            """,
                {"error": error_msg},
                datetime.now(self.israel_tz),
                self.collection_run_id,
            )

    def _log_collection_summary(self):
        """Log collection summary statistics."""
        duration = (
            (self.stats.end_time - self.stats.start_time).total_seconds()
            if self.stats.end_time
            else 0
        )

        self.logger.info("=" * 60)
        self.logger.info("HISTORICAL BARS COLLECTION SUMMARY")
        self.logger.info(f"Duration: {duration:.1f} seconds")
        self.logger.info(
            f"Contracts: {self.stats.contracts_successful}/{self.stats.contracts_requested}"
        )
        self.logger.info(f"Bars collected: {self.stats.bars_collected}")
        self.logger.info(f"Bars skipped: {self.stats.bars_skipped}")
        self.logger.info(f"Errors: {self.stats.errors}")
        self.logger.info(f"Rate limit hits: {self.stats.rate_limit_hits}")

        if self.stats.bars_collected > 0:
            bars_per_second = (
                self.stats.bars_collected / duration if duration > 0 else 0
            )
            self.logger.info(f"Performance: {bars_per_second:.1f} bars/second")

        self.logger.info("=" * 60)

    async def _cleanup(self):
        """Clean up connections and resources."""
        # Close single IB connection
        if self.ib_connection and self.ib_connection.isConnected():
            try:
                self.ib_connection.disconnect()
                self.logger.info("✅ IB connection disconnected cleanly")
            except Exception as e:
                self.logger.warning(f"Error disconnecting IB: {e}")

        # Close database pool
        if self.db_pool:
            try:
                await self.db_pool.close()
                self.logger.info("✅ Database connections closed")
            except Exception as e:
                self.logger.warning(f"Error closing database pool: {e}")


async def main():
    """Main entry point for historical bars collector."""
    parser = argparse.ArgumentParser(description="Historical 5-minute Bars Collector")
    parser.add_argument(
        "--config",
        default="scheduler/intraday_collection_israel.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY", "PLTR", "TSLA"],
        help="Symbols to collect",
    )
    parser.add_argument(
        "--type",
        default="manual",
        choices=[
            "morning",
            "midday",
            "afternoon",
            "eod",
            "late_night",
            "gap_fill",
            "manual",
        ],
        help="Collection type",
    )
    parser.add_argument(
        "--duration", default="1 D", help='Duration to collect (e.g., "1 D", "2 hours")'
    )
    parser.add_argument(
        "--end-datetime",
        default="",
        help="End datetime for historical data (empty for now)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel collection with multiple client IDs",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--connection-timeout",
        type=int,
        default=30,
        help="IB Gateway connection timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--connection-retries",
        type=int,
        default=3,
        help="Number of connection retry attempts (default: 3)",
    )

    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Create and run collector
    collector = HistoricalBarsCollector(
        config_path=args.config,
        connection_timeout=args.connection_timeout,
        connection_retries=args.connection_retries,
    )

    # Override parallel setting if specified
    if not args.parallel:
        collector.config["rate_limits"]["client_ids"] = [3]

    # Run collection
    await collector.collect_historical_bars(
        symbols=args.symbols,
        collection_type=args.type,
        duration=args.duration,
        end_datetime=args.end_datetime,
    )


if __name__ == "__main__":
    asyncio.run(main())
