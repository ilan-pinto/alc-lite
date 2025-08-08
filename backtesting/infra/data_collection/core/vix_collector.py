"""
VIX Data Collector Module

This module handles real-time and historical VIX data collection from Interactive Brokers
for correlation analysis with arbitrage opportunities. It collects data for VIX, VIX1D,
VIX9D, VIX3M, and VIX6M instruments.

Author: Claude Code Assistant
Created: 2025-08-04
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from datetime import time as dt_time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import logging
import pandas as pd
from ib_async import IB, BarData, Contract, Index, Ticker
from ib_async.util import df

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class VIXCollectionConfig:
    """Configuration for VIX data collection."""

    # Collection intervals
    tick_interval_ms: int = 100  # Milliseconds between tick collections
    batch_size: int = 1000  # Number of records to batch before database insert

    # Data quality settings
    max_bid_ask_spread_pct: float = 5.0  # Maximum allowed bid/ask spread as percentage
    stale_data_threshold_sec: int = 60  # Seconds before data is considered stale

    # VIX specific settings
    min_vix_level: float = 5.0  # Minimum reasonable VIX level
    max_vix_level: float = 200.0  # Maximum reasonable VIX level

    # Market hours (ET)
    market_open: dt_time = field(default_factory=lambda: dt_time(9, 30))
    market_close: dt_time = field(default_factory=lambda: dt_time(16, 0))

    # Collection preferences
    collect_during_market_hours_only: bool = True
    enable_historical_backfill: bool = True
    historical_days: int = 30  # Days of historical data to collect


class VIXDataCollector:
    """
    VIX data collector for correlation analysis with arbitrage opportunities.

    This class handles:
    - Real-time VIX data streaming from Interactive Brokers
    - Historical VIX data collection and backfill
    - Data validation and quality control
    - Database storage with TimescaleDB optimization
    - VIX term structure calculation and analysis
    """

    # VIX instrument definitions
    VIX_INSTRUMENTS = {
        "VIX": {"name": "CBOE Volatility Index", "maturity_days": 30},
        "VIX1D": {"name": "1-Day Volatility Index", "maturity_days": 1},
        "VIX9D": {"name": "9-Day Volatility Index", "maturity_days": 9},
        "VIX3M": {"name": "3-Month Volatility Index", "maturity_days": 90},
        "VIX6M": {"name": "6-Month Volatility Index", "maturity_days": 180},
    }

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        ib_connection: IB,
        config: VIXCollectionConfig = None,
    ):
        """
        Initialize VIX data collector.

        Args:
            db_pool: AsyncPG database connection pool
            ib_connection: Interactive Brokers connection instance
            config: Collection configuration (uses defaults if None)
        """
        self.db_pool = db_pool
        self.ib = ib_connection
        self.config = config or VIXCollectionConfig()

        # Runtime state
        self.vix_contracts: Dict[str, Contract] = {}
        self.vix_instruments_db: Dict[str, int] = {}  # symbol -> database ID mapping
        self.active_tickers: Dict[str, Ticker] = {}
        self.collection_active = False
        self.data_buffer: List[Dict] = []
        self.last_term_structure: Optional[Dict] = None

        # Statistics
        self.stats = {
            "ticks_collected": 0,
            "ticks_stored": 0,
            "data_quality_issues": 0,
            "term_structures_calculated": 0,
            "collection_start_time": None,
            "last_update_time": None,
        }

    async def initialize(self) -> bool:
        """
        Initialize VIX data collection system.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing VIX data collector...")

            # Ensure VIX instruments are in database
            await self._ensure_vix_instruments_in_db()

            # Create and qualify VIX contracts
            await self._create_vix_contracts()

            # Load instrument ID mappings
            await self._load_instrument_mappings()

            logger.info(
                f"VIX data collector initialized with {len(self.vix_contracts)} instruments"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize VIX data collector: {e}")
            return False

    async def _ensure_vix_instruments_in_db(self) -> None:
        """Ensure VIX instruments are properly defined in the database."""
        async with self.db_pool.acquire() as conn:
            for symbol, info in self.VIX_INSTRUMENTS.items():
                await conn.execute(
                    """
                    INSERT INTO vix_instruments (symbol, name, description, maturity_days)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (symbol) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        maturity_days = EXCLUDED.maturity_days,
                        updated_at = CURRENT_TIMESTAMP
                """,
                    symbol,
                    info["name"],
                    f"{info['name']} for volatility correlation analysis",
                    info["maturity_days"],
                )

    async def _create_vix_contracts(self) -> None:
        """Create and qualify VIX contracts with Interactive Brokers."""
        contracts_to_qualify = []

        # Create contract objects
        for symbol in self.VIX_INSTRUMENTS.keys():
            contract = Index(symbol, "CBOE", "USD")
            self.vix_contracts[symbol] = contract
            contracts_to_qualify.append(contract)

        # Qualify contracts with IB
        try:
            qualified_contracts = await self.ib.qualifyContractsAsync(
                *contracts_to_qualify
            )

            # Update contract references with qualified versions
            for contract in qualified_contracts:
                if contract.symbol in self.vix_contracts:
                    self.vix_contracts[contract.symbol] = contract
                    logger.info(
                        f"Qualified VIX contract: {contract.symbol} (ConId: {contract.conId})"
                    )

                    # Update database with IB contract ID
                    await self._update_contract_id(contract.symbol, contract.conId)

            logger.info(
                f"Successfully qualified {len(qualified_contracts)} VIX contracts"
            )

        except Exception as e:
            logger.error(f"Failed to qualify VIX contracts: {e}")
            raise

    async def _update_contract_id(self, symbol: str, con_id: int) -> None:
        """Update IB contract ID in database."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE vix_instruments
                SET ib_con_id = $1, updated_at = CURRENT_TIMESTAMP
                WHERE symbol = $2
            """,
                con_id,
                symbol,
            )

    async def _load_instrument_mappings(self) -> None:
        """Load VIX instrument ID mappings from database."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT symbol, id FROM vix_instruments WHERE active = true"
            )
            self.vix_instruments_db = {row["symbol"]: row["id"] for row in rows}
            logger.info(
                f"Loaded {len(self.vix_instruments_db)} VIX instrument mappings"
            )

    async def start_real_time_collection(self) -> None:
        """Start real-time VIX data collection."""
        if self.collection_active:
            logger.warning("VIX collection already active")
            return

        try:
            logger.info("Starting real-time VIX data collection...")

            # Request market data for all VIX instruments
            for symbol, contract in self.vix_contracts.items():
                ticker = self.ib.reqMktData(contract, "", False, False)
                self.active_tickers[symbol] = ticker
                logger.info(f"Started market data stream for {symbol}")

            self.collection_active = True
            self.stats["collection_start_time"] = datetime.now()

            # Start collection loop
            asyncio.create_task(self._collection_loop())

            logger.info("Real-time VIX data collection started successfully")

        except Exception as e:
            logger.error(f"Failed to start VIX data collection: {e}")
            raise

    async def stop_real_time_collection(self) -> None:
        """Stop real-time VIX data collection."""
        logger.info("Stopping real-time VIX data collection...")

        self.collection_active = False

        # Cancel market data subscriptions
        for symbol, ticker in self.active_tickers.items():
            self.ib.cancelMktData(ticker.contract)
            logger.info(f"Stopped market data stream for {symbol}")

        self.active_tickers.clear()

        # Flush any remaining data
        await self._flush_data_buffer()

        logger.info("VIX data collection stopped")

    async def _collection_loop(self) -> None:
        """Main collection loop for real-time data."""
        while self.collection_active:
            try:
                # Check if we should collect during current time
                if (
                    self.config.collect_during_market_hours_only
                    and not self._is_market_hours()
                ):
                    await asyncio.sleep(60)  # Check every minute during off-hours
                    continue

                # Collect current VIX data
                await self._collect_current_tick()

                # Calculate and store term structure if we have enough data
                await self._calculate_term_structure()

                # Flush buffer if it's getting full
                if len(self.data_buffer) >= self.config.batch_size:
                    await self._flush_data_buffer()

                # Update statistics
                self.stats["last_update_time"] = datetime.now()

                # Wait for next collection interval
                await asyncio.sleep(self.config.tick_interval_ms / 1000.0)

            except Exception as e:
                logger.error(f"Error in VIX collection loop: {e}")
                await asyncio.sleep(1)  # Brief pause on error

    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours."""
        now = datetime.now().time()
        return self.config.market_open <= now <= self.config.market_close

    async def _collect_current_tick(self) -> None:
        """Collect current tick data from all VIX instruments."""
        timestamp = datetime.now()

        for symbol, ticker in self.active_tickers.items():
            try:
                if not ticker or pd.isna(ticker.marketPrice()):
                    continue

                # Get instrument ID
                instrument_id = self.vix_instruments_db.get(symbol)
                if not instrument_id:
                    continue

                # Extract price data
                last_price = ticker.marketPrice()
                bid_price = ticker.bid if not pd.isna(ticker.bid) else None
                ask_price = ticker.ask if not pd.isna(ticker.ask) else None
                bid_size = ticker.bidSize if not pd.isna(ticker.bidSize) else None
                ask_size = ticker.askSize if not pd.isna(ticker.askSize) else None
                last_size = ticker.lastSize if not pd.isna(ticker.lastSize) else None
                volume = ticker.volume if not pd.isna(ticker.volume) else 0

                # Data quality validation
                quality_score = self._calculate_data_quality(
                    last_price, bid_price, ask_price
                )

                if quality_score < 0.5:  # Skip low quality data
                    self.stats["data_quality_issues"] += 1
                    continue

                # Calculate daily change (simplified - in production would use previous close)
                daily_change = None
                daily_change_pct = None

                # Create tick record
                tick_data = {
                    "time": timestamp,
                    "instrument_id": instrument_id,
                    "last_price": last_price,
                    "bid_price": bid_price,
                    "ask_price": ask_price,
                    "bid_size": bid_size,
                    "ask_size": ask_size,
                    "last_size": last_size,
                    "volume": volume,
                    "daily_change": daily_change,
                    "daily_change_pct": daily_change_pct,
                    "tick_type": "REALTIME",
                    "data_quality_score": quality_score,
                }

                self.data_buffer.append(tick_data)
                self.stats["ticks_collected"] += 1

            except Exception as e:
                logger.error(f"Error collecting tick for {symbol}: {e}")

    def _calculate_data_quality(
        self, last_price: float, bid_price: Optional[float], ask_price: Optional[float]
    ) -> float:
        """
        Calculate data quality score based on various factors.

        Returns:
            float: Quality score between 0 and 1
        """
        score = 1.0

        # Check price reasonableness
        if not (self.config.min_vix_level <= last_price <= self.config.max_vix_level):
            score *= 0.3

        # Check bid/ask spread
        if bid_price and ask_price:
            spread_pct = (ask_price - bid_price) / last_price * 100
            if spread_pct > self.config.max_bid_ask_spread_pct:
                score *= 0.5
        else:
            score *= 0.8  # Penalize missing bid/ask

        return score

    async def _calculate_term_structure(self) -> None:
        """Calculate and store VIX term structure snapshot."""
        # Get latest prices for all VIX instruments
        vix_levels = {}

        for symbol, ticker in self.active_tickers.items():
            if ticker and not pd.isna(ticker.marketPrice()):
                vix_levels[symbol] = ticker.marketPrice()

        # Need at least VIX and one other instrument
        if len(vix_levels) < 2 or "VIX" not in vix_levels:
            return

        try:
            async with self.db_pool.acquire() as conn:
                # Insert term structure snapshot
                await conn.execute(
                    """
                    INSERT INTO vix_term_structure
                    (timestamp, vix_1d, vix_9d, vix_30d, vix_3m, vix_6m)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """,
                    datetime.now(),
                    vix_levels.get("VIX1D"),
                    vix_levels.get("VIX9D"),
                    vix_levels.get("VIX"),  # VIX is 30-day
                    vix_levels.get("VIX3M"),
                    vix_levels.get("VIX6M"),
                )

            self.last_term_structure = vix_levels.copy()
            self.stats["term_structures_calculated"] += 1

        except Exception as e:
            logger.error(f"Error storing term structure: {e}")

    async def _flush_data_buffer(self) -> None:
        """Flush buffered tick data to database."""
        if not self.data_buffer:
            return

        try:
            async with self.db_pool.acquire() as conn:
                # Batch insert tick data
                await conn.executemany(
                    """
                    INSERT INTO vix_data_ticks
                    (time, instrument_id, last_price, bid_price, ask_price,
                     bid_size, ask_size, last_size, volume, daily_change,
                     daily_change_pct, tick_type, data_quality_score)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                    [
                        (
                            tick["time"],
                            tick["instrument_id"],
                            tick["last_price"],
                            tick["bid_price"],
                            tick["ask_price"],
                            tick["bid_size"],
                            tick["ask_size"],
                            tick["last_size"],
                            tick["volume"],
                            tick["daily_change"],
                            tick["daily_change_pct"],
                            tick["tick_type"],
                            tick["data_quality_score"],
                        )
                        for tick in self.data_buffer
                    ],
                )

            records_stored = len(self.data_buffer)
            self.stats["ticks_stored"] += records_stored
            self.data_buffer.clear()

            logger.debug(f"Stored {records_stored} VIX tick records to database")

        except Exception as e:
            logger.error(f"Error flushing VIX data buffer: {e}")

    async def collect_historical_data(self, days_back: int = None) -> Dict[str, int]:
        """
        Collect historical VIX data for backtesting.

        Args:
            days_back: Number of days of historical data to collect

        Returns:
            Dict[str, int]: Count of records collected per instrument
        """
        days_back = days_back or self.config.historical_days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        logger.info(f"Collecting {days_back} days of historical VIX data...")

        collection_stats = {}

        for symbol, contract in self.vix_contracts.items():
            try:
                # Request historical data from IB
                bars = await self.ib.reqHistoricalDataAsync(
                    contract=contract,
                    endDateTime=end_date,
                    durationStr=f"{days_back} D",
                    barSizeSetting="1 min",
                    whatToShow="MIDPOINT",
                    useRTH=True,
                    formatDate=1,
                )

                if not bars:
                    logger.warning(f"No historical data received for {symbol}")
                    continue

                # Convert to database format
                instrument_id = self.vix_instruments_db[symbol]
                historical_records = []

                for bar in bars:
                    historical_records.append(
                        {
                            "time": bar.date,
                            "instrument_id": instrument_id,
                            "open_price": bar.open,
                            "high_price": bar.high,
                            "low_price": bar.low,
                            "last_price": bar.close,  # Use close as last price
                            "volume": bar.volume,
                            "tick_type": "HISTORICAL",
                            "data_quality_score": 1.0,
                        }
                    )

                # Store to database
                await self._store_historical_records(historical_records)
                collection_stats[symbol] = len(historical_records)

                logger.info(
                    f"Collected {len(historical_records)} historical records for {symbol}"
                )

            except Exception as e:
                logger.error(f"Error collecting historical data for {symbol}: {e}")
                collection_stats[symbol] = 0

        logger.info(f"Historical VIX data collection completed: {collection_stats}")
        return collection_stats

    async def _store_historical_records(self, records: List[Dict]) -> None:
        """Store historical VIX records to database."""
        if not records:
            return

        async with self.db_pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO vix_data_ticks
                (time, instrument_id, open_price, high_price, low_price, last_price,
                 volume, tick_type, data_quality_score)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT DO NOTHING
            """,
                [
                    (
                        record["time"],
                        record["instrument_id"],
                        record.get("open_price"),
                        record.get("high_price"),
                        record.get("low_price"),
                        record["last_price"],
                        record.get("volume", 0),
                        record["tick_type"],
                        record["data_quality_score"],
                    )
                    for record in records
                ],
            )

    def get_current_vix_levels(self) -> Dict[str, float]:
        """Get current VIX levels from active tickers."""
        levels = {}
        for symbol, ticker in self.active_tickers.items():
            if ticker and not pd.isna(ticker.marketPrice()):
                levels[symbol] = ticker.marketPrice()
        return levels

    def get_latest_term_structure(self) -> Optional[Dict]:
        """Get the latest VIX term structure."""
        return self.last_term_structure.copy() if self.last_term_structure else None

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        stats = self.stats.copy()
        stats["active_instruments"] = len(self.active_tickers)
        stats["buffer_size"] = len(self.data_buffer)
        stats["collection_active"] = self.collection_active
        return stats

    async def get_vix_correlation_context(self) -> Dict[str, Any]:
        """
        Get current VIX context for correlation analysis.

        Returns:
            Dict containing VIX levels, regime, and term structure info
        """
        current_levels = self.get_current_vix_levels()

        if not current_levels or "VIX" not in current_levels:
            return {}

        vix_level = current_levels["VIX"]

        # Determine volatility regime
        if vix_level < 15:
            regime = "LOW"
        elif vix_level <= 25:
            regime = "MEDIUM"
        elif vix_level <= 40:
            regime = "HIGH"
        else:
            regime = "EXTREME"

        # Determine term structure type
        structure_type = None
        if "VIX3M" in current_levels:
            if current_levels["VIX3M"] > vix_level:
                structure_type = "CONTANGO"
            elif current_levels["VIX3M"] < vix_level:
                structure_type = "BACKWARDATION"
            else:
                structure_type = "FLAT"

        return {
            "vix_level": vix_level,
            "vix_regime": regime,
            "term_structure_type": structure_type,
            "vix_spike_active": vix_level > 30,
            "all_levels": current_levels,
            "timestamp": datetime.now(),
        }


@asynccontextmanager
async def create_vix_collector(
    db_pool: asyncpg.Pool, ib: IB, config: VIXCollectionConfig = None
) -> VIXDataCollector:
    """
    Context manager for creating and managing VIX data collector.

    Usage:
        async with create_vix_collector(db_pool, ib) as collector:
            await collector.start_real_time_collection()
            # ... do work ...
            # collector will be automatically cleaned up
    """
    collector = VIXDataCollector(db_pool, ib, config)

    try:
        success = await collector.initialize()
        if not success:
            raise RuntimeError("Failed to initialize VIX data collector")

        yield collector

    finally:
        if collector.collection_active:
            await collector.stop_real_time_collection()
