#!/usr/bin/env python3
"""
Daily Options Data Collector for Phase 1 Local Implementation
Optimized for Israel timezone with automatic DST handling
"""

import asyncio
import os
import signal
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import argparse
import asyncpg
import logging
import pytz
import yaml
from ib_async import IB, Contract, Option

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from backtesting.infra.data_collection.config.config import (
    CollectionConfig,
    DatabaseConfig,
)
from backtesting.infra.data_collection.core.collector import OptionsDataCollector


class DailyCollector:
    """
    Daily options data collector with Israel timezone support.
    Handles SPY, PLTR, TSLA for Phase 1 implementation.
    """

    def __init__(
        self, config_path: str = None, force_run: bool = False, truncate: bool = False
    ):
        """
        Initialize daily collector with configuration.

        Args:
            config_path: Path to configuration file
            force_run: Force collection even if already done today
            truncate: Truncate today's data before collection
        """
        self.config_path = config_path or "scheduler/daily_schedule_israel.yaml"
        self.force_run = force_run
        self.truncate = truncate
        self.config = self._load_config()

        # Timezone setup
        self.israel_tz = pytz.timezone(self.config["timezone"])
        self.et_tz = pytz.timezone("US/Eastern")

        # Initialize logging
        self._setup_logging()

        # Collection state
        self.db_pool = None
        self.ib = None
        self.collector = None
        self.collection_id = None
        self.stats = {
            "symbols_processed": 0,
            "contracts_collected": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None,
        }

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(self.config_path)
        if not config_path.exists():
            # Try relative to project root
            config_path = Path(__file__).parent.parent.parent.parent / self.config_path

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Configure logging with Israel timezone timestamps."""
        log_config = self.config["logging"]

        # Create logs directory if needed
        Path("logs").mkdir(exist_ok=True)

        # Custom formatter with Israel timezone
        class IsraelFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                ct = datetime.fromtimestamp(record.created, tz=pytz.UTC)
                israel_time = ct.astimezone(pytz.timezone("Asia/Jerusalem"))
                if datefmt:
                    return israel_time.strftime(datefmt)
                return israel_time.strftime("%Y-%m-%d %H:%M:%S %Z")

        # Configure root logger
        formatter = IsraelFormatter(log_config["format"])

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # File handler
        file_handler = logging.FileHandler(log_config["file_path"])
        file_handler.setFormatter(formatter)

        # Error file handler
        error_handler = logging.FileHandler(log_config["error_file_path"])
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)

        # Configure logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, log_config["level"]))
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.addHandler(error_handler)

        self.logger = logger

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        if self.collector:
            asyncio.create_task(self.collector.stop_collection())
        sys.exit(0)

    async def run(self, collection_type: str = "end_of_day"):
        """
        Run daily collection for configured symbols.

        Args:
            collection_type: Type of collection (end_of_day, friday_expiry_check, etc.)
        """
        self.stats["start_time"] = datetime.now(self.israel_tz)
        self.logger.info(
            f"Starting {collection_type} collection at {self.stats['start_time']}"
        )

        try:
            # Initialize connections first
            await self._initialize_connections()

            # Truncate today's data if requested
            if self.truncate and self.force_run:
                await self._truncate_today_data(collection_type)
                self.logger.info(f"Truncated today's data for {collection_type}")

            # Check if collection already done today (unless forced)
            if not self.force_run and await self._check_already_collected(
                collection_type
            ):
                self.logger.info(
                    f"Collection already completed today for {collection_type}"
                )
                return

            # Record collection start
            self.collection_id = await self._record_collection_start(collection_type)

            # Collect data for each symbol
            for symbol in self.config["symbols"]:
                await self._collect_symbol(symbol, collection_type)

            # Record collection completion
            await self._record_collection_complete()

            # Generate summary report
            await self._generate_summary_report()

        except Exception as e:
            self.logger.error(f"Collection failed: {e}", exc_info=True)
            self.stats["errors"] += 1
            if self.collection_id:
                await self._record_collection_error(str(e))
            raise

        finally:
            await self._cleanup()
            self.stats["end_time"] = datetime.now(self.israel_tz)
            duration = (
                self.stats["end_time"] - self.stats["start_time"]
            ).total_seconds()
            self.logger.info(
                f"Collection completed in {duration:.2f} seconds. Stats: {self.stats}"
            )

    async def _initialize_connections(self):
        """Initialize database and IB connections."""
        # Database connection
        db_config = self.config["database"]
        self.db_pool = await asyncpg.create_pool(
            host=db_config["host"],
            port=db_config["port"],
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"],
            min_size=5,
            max_size=db_config["pool_size"],
        )
        self.logger.info("Database connection established")

        # IB Gateway connection
        ib_config = self.config["ib_gateway"]
        self.ib = IB()

        for attempt in range(ib_config["reconnect_attempts"]):
            try:
                await self.ib.connectAsync(
                    ib_config["host"],
                    ib_config["port"],
                    clientId=ib_config["client_id"],
                    timeout=ib_config["timeout_seconds"],
                )
                self.logger.info(f"Connected to IB Gateway on port {ib_config['port']}")
                break
            except Exception as e:
                if attempt < ib_config["reconnect_attempts"] - 1:
                    self.logger.warning(
                        f"IB connection attempt {attempt + 1} failed, retrying..."
                    )
                    await asyncio.sleep(ib_config["reconnect_delay_seconds"])
                else:
                    raise Exception(
                        f"Failed to connect to IB Gateway after {ib_config['reconnect_attempts']} attempts: {e}"
                    )

        # Initialize collector
        collection_config = CollectionConfig()
        collection_config.default_symbols = self.config["symbols"]
        collection_config.expiry_range_days = self.config["collection"][
            "expiry_range_days"
        ]
        collection_config.strike_range_percent = self.config["collection"][
            "strike_range_percent"
        ]
        collection_config.batch_size = self.config["collection"]["batch_size"]
        collection_config.request_throttle_ms = self.config["collection"][
            "request_throttle_ms"
        ]
        collection_config.flush_interval_seconds = self.config["collection"][
            "flush_interval_seconds"
        ]

        self.collector = OptionsDataCollector(self.db_pool, self.ib, collection_config)

    async def _check_already_collected(self, collection_type: str) -> bool:
        """Check if collection already done today."""
        async with self.db_pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM daily_collection_status
                WHERE collection_date = CURRENT_DATE
                AND collection_type = $1
                AND status = 'success'
            """,
                collection_type,
            )
            return count > 0

    async def _truncate_today_data(self, collection_type: str):
        """Truncate today's collection data from database."""
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                # Delete collection status records
                result = await conn.fetch(
                    """
                    DELETE FROM daily_collection_status
                    WHERE collection_date = CURRENT_DATE
                    AND collection_type = $1
                    RETURNING id
                """,
                    collection_type,
                )
                deleted_status = len(result)

                # Optional: Delete tick data (commented out for safety)
                # deleted_ticks = await conn.fetchval("""
                #     DELETE FROM market_data_ticks
                #     WHERE DATE(time) = CURRENT_DATE
                #     RETURNING COUNT(*)
                # """)

                self.logger.info(
                    f"Truncated {deleted_status} collection status records"
                )

    async def _record_collection_start(self, collection_type: str) -> int:
        """Record collection start in database."""
        async with self.db_pool.acquire() as conn:
            if self.force_run:
                # Use ON CONFLICT to update existing record when forcing
                collection_id = await conn.fetchval(
                    """
                    INSERT INTO daily_collection_status
                    (collection_date, collection_time, symbol, collection_type,
                     started_at, status, timezone)
                    VALUES (CURRENT_DATE, $1, 'ALL', $2, $1, 'running', $3)
                    ON CONFLICT (collection_date, symbol, collection_type)
                    DO UPDATE SET
                        collection_time = $1,
                        started_at = $1,
                        status = 'running',
                        completed_at = NULL,
                        records_collected = 0,
                        error_message = NULL
                    RETURNING id
                """,
                    datetime.now(self.israel_tz),
                    collection_type,
                    str(self.israel_tz),
                )
            else:
                collection_id = await conn.fetchval(
                    """
                    INSERT INTO daily_collection_status
                    (collection_date, collection_time, symbol, collection_type,
                     started_at, status, timezone)
                    VALUES (CURRENT_DATE, $1, 'ALL', $2, $1, 'running', $3)
                    RETURNING id
                """,
                    datetime.now(self.israel_tz),
                    collection_type,
                    str(self.israel_tz),
                )

            self.logger.info(f"Collection run started with ID: {collection_id}")
            return collection_id

    async def _collect_symbol(self, symbol: str, collection_type: str):
        """Collect options data for a single symbol."""
        self.logger.info(f"Collecting data for {symbol}")
        start_time = datetime.now(self.israel_tz)

        try:
            # Record symbol collection start
            async with self.db_pool.acquire() as conn:
                symbol_collection_id = await conn.fetchval(
                    """
                    INSERT INTO daily_collection_status
                    (collection_date, collection_time, symbol, collection_type,
                     started_at, status, timezone)
                    VALUES (CURRENT_DATE, $1, $2, $3, $1, 'running', $4)
                    ON CONFLICT (collection_date, symbol, collection_type)
                    DO UPDATE SET started_at = $1, status = 'running'
                    RETURNING id
                """,
                    start_time,
                    symbol,
                    collection_type,
                    str(self.israel_tz),
                )

            # Use the existing collector to get symbol data
            await self.collector._initialize_symbol_contracts(symbol)

            # Get collection stats
            contracts_collected = len(self.collector.active_contracts)

            # Record success
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE daily_collection_status
                    SET completed_at = $1,
                        status = 'success',
                        records_collected = $2,
                        contracts_discovered = $2,
                        execution_time_ms = $3
                    WHERE id = $4
                """,
                    datetime.now(self.israel_tz),
                    contracts_collected,
                    int(
                        (datetime.now(self.israel_tz) - start_time).total_seconds()
                        * 1000
                    ),
                    symbol_collection_id,
                )

            self.stats["symbols_processed"] += 1
            self.stats["contracts_collected"] += contracts_collected
            self.logger.info(
                f"Successfully collected {contracts_collected} contracts for {symbol}"
            )

        except Exception as e:
            self.logger.error(f"Failed to collect {symbol}: {e}")
            self.stats["errors"] += 1

            # Record failure
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE daily_collection_status
                    SET completed_at = $1,
                        status = 'failed',
                        error_message = $2,
                        error_count = error_count + 1
                    WHERE collection_date = CURRENT_DATE
                    AND symbol = $3
                    AND collection_type = $4
                """,
                    datetime.now(self.israel_tz),
                    str(e),
                    symbol,
                    collection_type,
                )

    async def _record_collection_complete(self):
        """Record overall collection completion."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE daily_collection_status
                SET completed_at = $1,
                    status = CASE WHEN error_count = 0 THEN 'success' ELSE 'partial' END,
                    records_collected = $2,
                    contracts_discovered = $2
                WHERE id = $3
            """,
                datetime.now(self.israel_tz),
                self.stats["contracts_collected"],
                self.collection_id,
            )

    async def _record_collection_error(self, error_msg: str):
        """Record collection error."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE daily_collection_status
                SET status = 'failed',
                    error_message = $1,
                    completed_at = $2
                WHERE id = $3
            """,
                error_msg,
                datetime.now(self.israel_tz),
                self.collection_id,
            )

    async def _generate_summary_report(self):
        """Generate and log collection summary report."""
        async with self.db_pool.acquire() as conn:
            # Get today's collection summary
            summary = await conn.fetch(
                """
                SELECT symbol, collection_type, status,
                       records_collected, execution_time_ms, error_message
                FROM daily_collection_status
                WHERE collection_date = CURRENT_DATE
                ORDER BY started_at DESC
            """
            )

            # Get data quality metrics
            quality = await conn.fetchrow(
                """
                SELECT AVG(data_quality_score) as avg_quality,
                       COUNT(DISTINCT symbol) as symbols_collected,
                       SUM(records_collected) as total_records
                FROM daily_collection_status
                WHERE collection_date = CURRENT_DATE
                AND status IN ('success', 'partial')
            """
            )

        self.logger.info("=" * 60)
        self.logger.info("DAILY COLLECTION SUMMARY REPORT")
        self.logger.info(
            f"Date: {date.today()} | Time: {datetime.now(self.israel_tz).strftime('%H:%M:%S %Z')}"
        )
        self.logger.info("-" * 60)

        for row in summary:
            status_emoji = (
                "✅"
                if row["status"] == "success"
                else "❌" if row["status"] == "failed" else "⚠️"
            )
            self.logger.info(
                f"{status_emoji} {row['symbol']}: {row['status']} | "
                f"Records: {row['records_collected'] or 0} | "
                f"Time: {row['execution_time_ms'] or 0}ms"
            )
            if row["error_message"]:
                self.logger.info(f"   Error: {row['error_message'][:100]}")

        self.logger.info("-" * 60)
        self.logger.info(f"Total Symbols: {quality['symbols_collected'] or 0}")
        self.logger.info(f"Total Records: {quality['total_records'] or 0}")
        self.logger.info(
            f"Avg Quality Score: {quality['avg_quality'] or 0:.2f}"
            if quality["avg_quality"]
            else "N/A"
        )
        self.logger.info("=" * 60)

    async def _cleanup(self):
        """Clean up connections and resources."""
        if self.collector:
            await self.collector.stop_collection()

        if self.ib:
            self.ib.disconnect()
            self.logger.info("Disconnected from IB Gateway")

        if self.db_pool:
            await self.db_pool.close()
            self.logger.info("Database connections closed")


async def main():
    """Main entry point for daily collector."""
    parser = argparse.ArgumentParser(description="Daily Options Data Collector")
    parser.add_argument(
        "--config",
        default="scheduler/daily_schedule_israel.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--symbols", nargs="+", help="Symbols to collect (overrides config)"
    )
    parser.add_argument(
        "--type",
        default="end_of_day",
        choices=["end_of_day", "friday_expiry_check", "morning_check"],
        help="Collection type",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force collection even if already done today",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--timezone", default="Asia/Jerusalem", help="Timezone for timestamps"
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate today's data before collection (use with --force)",
    )

    args = parser.parse_args()

    # Set up basic logging before collector initializes
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Create and run collector
    collector = DailyCollector(args.config, args.force, args.truncate)

    # Override symbols if provided
    if args.symbols:
        collector.config["symbols"] = args.symbols

    # Override timezone if provided
    if args.timezone != collector.config["timezone"]:
        collector.config["timezone"] = args.timezone
        collector.israel_tz = pytz.timezone(args.timezone)

    await collector.run(args.type)


if __name__ == "__main__":
    asyncio.run(main())
