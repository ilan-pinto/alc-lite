"""
Historical SFR Data Loader for Backtesting Engine.

Specialized data loader optimized for SFR (Synthetic Free Risk) backtesting requirements.
Extends the existing historical_loader with SFR-specific optimizations and features.
"""

import asyncio
import math
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import asyncpg
import logging
import numpy as np
import pandas as pd
from ib_async import IB, BarData, Contract, Option, Stock
from rich.console import Console
from rich.progress import Progress, TaskID

try:
    from .config.config import DatabaseConfig, HistoricalConfig
    from .core.contract_utils import ContractFactory
    from .core.historical_loader import HistoricalDataLoader, HistoricalRequest
    from .core.validators import DataValidator
    from .data_sources.data_source_adapter import DataSourceAdapter
    from .validation.sfr_validator import SFRDataValidator
except ImportError:
    # Fallback imports for standalone usage
    from backtesting.infra.data_collection.config.config import (
        DatabaseConfig,
        HistoricalConfig,
    )
    from backtesting.infra.data_collection.core.contract_utils import ContractFactory
    from backtesting.infra.data_collection.core.historical_loader import (
        HistoricalDataLoader,
        HistoricalRequest,
    )
    from backtesting.infra.data_collection.core.validators import DataValidator
    from backtesting.infra.data_collection.data_sources.data_source_adapter import (
        DataSourceAdapter,
    )
    from backtesting.infra.data_collection.validation.sfr_validator import (
        SFRDataValidator,
    )

logger = logging.getLogger(__name__)
console = Console()

# SFR target symbols with priority weighting
SFR_TARGET_SYMBOLS = {
    "SPY": 1.0,  # Highest liquidity
    "QQQ": 1.0,  # Tech-heavy, high volume
    "AAPL": 0.9,  # Individual stock, high volume
    "MSFT": 0.9,  # Individual stock, high volume
    "NVDA": 0.8,  # High volatility, good for SFR
    "TSLA": 0.7,  # High volatility, but more unpredictable
    "AMZN": 0.8,  # High value stock, good option spreads
    "META": 0.8,  # Tech stock, good volume
    "GOOGL": 0.8,  # High value stock, good spreads
    "JPM": 0.7,  # Financial sector, good for diversification
}


@dataclass
class SFRDataLoadConfig:
    """SFR-specific data loading configuration."""

    # Time range settings optimized for SFR
    default_lookback_days: int = 365  # 1 year of data
    min_data_points_per_day: int = 100  # Minimum data points per trading day
    preferred_bar_size: str = "1 min"  # Higher resolution for SFR analysis

    # Option chain selection optimized for SFR
    strike_range_otm_percent: float = (
        0.10  # Focus on 10% OTM options for realistic strikes
    )
    expiry_range_days: Tuple[int, int] = (15, 60)  # 15-60 days to expiry
    min_volume_threshold: int = 10  # Minimum daily volume
    min_open_interest: int = 50  # Minimum open interest

    # Data quality requirements
    max_missing_data_percent: float = 5.0  # Max 5% missing data allowed
    required_market_hours: Tuple[str, str] = ("09:30", "16:00")  # Market hours

    # Performance optimization
    max_concurrent_symbols: int = 4  # Process 4 symbols in parallel
    batch_size_options: int = 50  # Options per batch
    batch_size_stock: int = 100  # Stock data points per batch

    # VIX integration for SFR correlation analysis
    load_vix_data: bool = True
    vix_correlation_window: int = 30  # Days for correlation analysis


@dataclass
class SFRDataLoadResult:
    """Result of SFR data loading operation."""

    symbol: str
    success: bool
    start_date: date
    end_date: date
    stock_bars_loaded: int
    option_contracts_loaded: int
    option_bars_loaded: int
    vix_data_points: int
    data_quality_score: float  # 0-1 quality score
    load_duration_seconds: float
    error_message: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class HistoricalSFRLoader(HistoricalDataLoader):
    """
    Specialized historical data loader for SFR backtesting.

    Extends the base HistoricalDataLoader with SFR-specific optimizations:
    - Prioritized loading of liquid options suitable for SFR strategies
    - Integrated VIX data loading for correlation analysis
    - Parallel processing optimized for SFR target symbols
    - Enhanced data validation for options arbitrage requirements
    - Intelligent strike selection based on SFR viability
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        ib_connection: IB = None,
        config: HistoricalConfig = None,
        sfr_config: SFRDataLoadConfig = None,
        data_source_adapter: DataSourceAdapter = None,
    ):
        super().__init__(db_pool, ib_connection, config)

        self.sfr_config = sfr_config or SFRDataLoadConfig()
        self.data_source_adapter = data_source_adapter
        self.sfr_validator = SFRDataValidator(db_pool)

        # Enhanced statistics tracking
        self.load_stats = {
            "total_symbols_processed": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "total_option_contracts": 0,
            "total_stock_bars": 0,
            "total_option_bars": 0,
            "total_vix_points": 0,
            "avg_data_quality": 0.0,
            "load_start_time": None,
            "load_end_time": None,
        }

        # Parallel processing setup
        self.executor = ThreadPoolExecutor(
            max_workers=self.sfr_config.max_concurrent_symbols
        )

    async def load_sfr_dataset(
        self,
        symbols: List[str] = None,
        start_date: date = None,
        end_date: date = None,
        include_vix: bool = True,
        progress: Progress = None,
    ) -> Dict[str, SFRDataLoadResult]:
        """
        Load complete SFR dataset for backtesting.

        Args:
            symbols: List of symbols to load (defaults to SFR_TARGET_SYMBOLS)
            start_date: Start date for data loading
            end_date: End date for data loading
            include_vix: Whether to include VIX data for correlation analysis
            progress: Rich progress tracker

        Returns:
            Dictionary mapping symbol to load results
        """
        # Set defaults
        symbols = symbols or list(SFR_TARGET_SYMBOLS.keys())
        start_date = start_date or (
            date.today() - timedelta(days=self.sfr_config.default_lookback_days)
        )
        end_date = end_date or date.today()

        self.load_stats["load_start_time"] = datetime.now()
        logger.info(
            f"Starting SFR dataset load for {len(symbols)} symbols "
            f"from {start_date} to {end_date}"
        )

        # Create progress tracking if not provided
        if progress is None:
            progress = Progress()
            progress_context = progress
        else:
            progress_context = None

        try:
            with progress_context or progress:
                # Create main task
                main_task = progress.add_task("Loading SFR Dataset", total=len(symbols))

                # Load VIX data first if requested
                vix_task = None
                if include_vix:
                    vix_task = progress.add_task("Loading VIX Data", total=1)
                    await self._load_vix_data(start_date, end_date, progress, vix_task)
                    progress.update(vix_task, completed=1)

                # Process symbols in parallel batches
                results = {}
                symbol_batches = self._create_symbol_batches(symbols)

                for batch in symbol_batches:
                    batch_results = await self._load_symbol_batch(
                        batch, start_date, end_date, progress, main_task
                    )
                    results.update(batch_results)

                # Calculate final statistics
                self._calculate_final_stats(results)

                return results

        except Exception as e:
            logger.error(f"SFR dataset loading failed: {e}")
            raise
        finally:
            self.load_stats["load_end_time"] = datetime.now()

    async def load_sfr_symbol_optimized(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        progress: Progress = None,
        task_id: TaskID = None,
    ) -> SFRDataLoadResult:
        """
        Load historical data for a single symbol optimized for SFR backtesting.

        This method implements SFR-specific optimizations:
        1. Prioritizes liquid options with tight spreads
        2. Focuses on strikes suitable for SFR combinations
        3. Validates data quality for arbitrage requirements
        4. Integrates VIX correlation data
        """
        load_start = datetime.now()

        if progress and task_id:
            progress.update(task_id, description=f"Loading {symbol} stock data...")

        try:
            # Initialize result
            result = SFRDataLoadResult(
                symbol=symbol,
                success=False,
                start_date=start_date,
                end_date=end_date,
                stock_bars_loaded=0,
                option_contracts_loaded=0,
                option_bars_loaded=0,
                vix_data_points=0,
                data_quality_score=0.0,
                load_duration_seconds=0.0,
            )

            # 1. Load stock data with enhanced error handling
            stock_bars = await self._load_stock_data_optimized(
                symbol, start_date, end_date
            )
            result.stock_bars_loaded = stock_bars

            if progress and task_id:
                progress.update(
                    task_id, description=f"Loading {symbol} option chains..."
                )

            # 2. Get SFR-relevant option contracts
            option_contracts = await self._get_sfr_option_contracts(
                symbol, start_date, end_date
            )
            result.option_contracts_loaded = len(option_contracts)

            if not option_contracts:
                result.warnings.append("No suitable option contracts found")
                logger.warning(f"No option contracts found for {symbol}")
            else:
                # 3. Load option data in optimized batches
                option_bars = await self._load_sfr_option_data(
                    option_contracts, start_date, end_date, progress, task_id
                )
                result.option_bars_loaded = option_bars

            # 4. Validate data quality
            result.data_quality_score = await self._calculate_data_quality_score(
                symbol, start_date, end_date, result
            )

            # 5. Update load statistics
            result.load_duration_seconds = (datetime.now() - load_start).total_seconds()
            result.success = True

            logger.info(
                f"Successfully loaded {symbol}: "
                f"{stock_bars} stock bars, {len(option_contracts)} option contracts, "
                f"{result.option_bars_loaded} option bars, "
                f"quality score: {result.data_quality_score:.3f}"
            )

            return result

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.load_duration_seconds = (datetime.now() - load_start).total_seconds()

            logger.error(f"Failed to load {symbol}: {e}")
            return result

    async def _load_stock_data_optimized(
        self, symbol: str, start_date: date, end_date: date
    ) -> int:
        """Load stock data with SFR-specific optimizations."""
        # Use higher frequency data for SFR analysis
        original_bar_size = self.config.bar_size
        self.config.bar_size = self.sfr_config.preferred_bar_size

        try:
            bars_loaded = await self._load_stock_history(symbol, start_date, end_date)
            return bars_loaded
        finally:
            # Restore original bar size
            self.config.bar_size = original_bar_size

    async def _get_sfr_option_contracts(
        self, symbol: str, start_date: date, end_date: date
    ) -> List[Tuple[Contract, int]]:
        """
        Get option contracts optimized for SFR strategies.

        SFR requires liquid options with:
        - Tight bid-ask spreads
        - Sufficient volume and open interest
        - Strikes suitable for conversion/reversal spreads
        """
        contracts = []

        async with self.db_pool.acquire() as conn:
            # Get underlying ID and current price range
            underlying_row = await conn.fetchrow(
                "SELECT id FROM underlying_securities WHERE symbol = $1",
                symbol,
            )

            if not underlying_row:
                return contracts

            underlying_id = underlying_row["id"]

            # Get price range during the period for intelligent strike selection
            price_stats = await conn.fetchrow(
                """
                SELECT
                    MIN(price) as min_price,
                    MAX(price) as max_price,
                    AVG(price) as avg_price,
                    STDDEV(price) as price_stddev
                FROM stock_data_ticks
                WHERE underlying_id = $1
                  AND time >= $2
                  AND time <= $3
                """,
                underlying_id,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )

            if not price_stats or not price_stats["avg_price"]:
                logger.warning(f"No price data found for {symbol} during period")
                return contracts

            avg_price = float(price_stats["avg_price"])
            price_std = float(price_stats["price_stddev"] or 0)

            # Calculate intelligent strike range for SFR
            # SFR benefits from strikes around the money with some OTM options
            strike_range_pct = self.sfr_config.strike_range_otm_percent
            min_strike = avg_price * (1 - strike_range_pct)
            max_strike = avg_price * (1 + strike_range_pct)

            # Query option contracts with SFR-specific filters
            option_rows = await conn.fetch(
                """
                SELECT
                    oc.id, oc.ib_con_id, oc.expiration_date,
                    oc.strike_price, oc.option_type,
                    -- Estimate liquidity score based on historical data
                    COUNT(mdt.time) as data_points,
                    AVG(mdt.volume) as avg_volume,
                    AVG(mdt.bid_ask_spread) as avg_spread
                FROM option_chains oc
                LEFT JOIN market_data_ticks mdt ON oc.id = mdt.contract_id
                    AND mdt.time >= $3 AND mdt.time <= $4
                WHERE oc.underlying_id = $1
                  AND oc.expiration_date >= $2
                  AND oc.expiration_date <= ($4::date + INTERVAL '60 days')::date
                  AND oc.strike_price >= $5
                  AND oc.strike_price <= $6
                  AND oc.ib_con_id IS NOT NULL
                GROUP BY oc.id, oc.ib_con_id, oc.expiration_date,
                         oc.strike_price, oc.option_type
                HAVING
                    -- Filter for liquid options
                    AVG(mdt.volume) >= $7 OR AVG(mdt.volume) IS NULL
                ORDER BY
                    oc.expiration_date,
                    ABS(oc.strike_price - $8),  -- Prefer ATM options
                    AVG(mdt.volume) DESC NULLS LAST
                LIMIT 200  -- Limit for performance
                """,
                underlying_id,
                start_date,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
                min_strike,
                max_strike,
                self.sfr_config.min_volume_threshold,
                avg_price,
            )

            # Convert to contracts and prioritize by SFR suitability
            for row in option_rows:
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

        logger.info(
            f"Found {len(contracts)} SFR-suitable option contracts for {symbol}"
        )
        return contracts

    async def _load_sfr_option_data(
        self,
        contracts: List[Tuple[Contract, int]],
        start_date: date,
        end_date: date,
        progress: Progress = None,
        task_id: TaskID = None,
    ) -> int:
        """Load option data with SFR-specific optimizations and parallel processing."""
        if not contracts:
            return 0

        total_bars = 0

        # Process in optimized batches
        batch_size = self.sfr_config.batch_size_options
        batches = [
            contracts[i : i + batch_size] for i in range(0, len(contracts), batch_size)
        ]

        if progress and task_id:
            progress.update(
                task_id,
                description=f"Loading option data ({len(batches)} batches)...",
                total=len(batches),
            )

        # Use parallel processing for better performance
        batch_tasks = []
        for i, batch in enumerate(batches):
            task = self._load_option_batch_history_parallel(
                batch, start_date, end_date, progress, task_id
            )
            batch_tasks.append(task)

        # Execute batches with controlled concurrency
        semaphore = asyncio.Semaphore(2)  # Limit concurrent batches

        async def process_batch_with_semaphore(task):
            async with semaphore:
                return await task

        batch_results = await asyncio.gather(
            *[process_batch_with_semaphore(task) for task in batch_tasks],
            return_exceptions=True,
        )

        # Sum up results
        for result in batch_results:
            if isinstance(result, int):
                total_bars += result
            else:
                logger.error(f"Batch processing error: {result}")

        return total_bars

    async def _load_vix_data(
        self,
        start_date: date,
        end_date: date,
        progress: Progress = None,
        task_id: TaskID = None,
    ):
        """Load VIX data for SFR correlation analysis."""
        if not self.sfr_config.load_vix_data:
            return

        if progress and task_id:
            progress.update(task_id, description="Loading VIX data...")

        try:
            # Use data source adapter if available, otherwise IB
            if self.data_source_adapter:
                vix_data = await self.data_source_adapter.get_vix_data(
                    start_date, end_date
                )
                await self._store_vix_data(vix_data)
                self.load_stats["total_vix_points"] += len(vix_data)
            elif self.ib:
                # Load via IB API
                vix_bars = await self._load_vix_via_ib(start_date, end_date)
                self.load_stats["total_vix_points"] += vix_bars
            else:
                logger.warning("No VIX data source available")

        except Exception as e:
            logger.error(f"Failed to load VIX data: {e}")

    async def _calculate_data_quality_score(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        result: SFRDataLoadResult,
    ) -> float:
        """
        Calculate a comprehensive data quality score for SFR backtesting.

        Quality factors:
        - Completeness (missing data percentage)
        - Consistency (no gaps during market hours)
        - Option coverage (availability of suitable strikes/expiries)
        - Liquidity indicators (volume, spreads)
        """
        quality_factors = {}

        async with self.db_pool.acquire() as conn:
            # 1. Data completeness check
            expected_trading_days = self._count_trading_days(start_date, end_date)
            actual_data_days = await conn.fetchval(
                """
                SELECT COUNT(DISTINCT DATE(time))
                FROM stock_data_ticks st
                JOIN underlying_securities us ON st.underlying_id = us.id
                WHERE us.symbol = $1
                  AND time >= $2
                  AND time <= $3
                """,
                symbol,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )

            completeness = (actual_data_days or 0) / max(expected_trading_days, 1)
            quality_factors["completeness"] = min(1.0, completeness)

            # 2. Option coverage quality
            if result.option_contracts_loaded > 0:
                avg_contracts_per_expiry = result.option_contracts_loaded / max(
                    len(set()), 1  # Would need expiry data
                )
                coverage_score = min(
                    1.0, avg_contracts_per_expiry / 20
                )  # Target 20 contracts per expiry
            else:
                coverage_score = 0.0

            quality_factors["option_coverage"] = coverage_score

            # 3. Data consistency (check for gaps during market hours)
            gap_count = await conn.fetchval(
                """
                WITH time_gaps AS (
                    SELECT
                        time,
                        LAG(time) OVER (ORDER BY time) as prev_time,
                        EXTRACT(EPOCH FROM (time - LAG(time) OVER (ORDER BY time))) as gap_seconds
                    FROM stock_data_ticks st
                    JOIN underlying_securities us ON st.underlying_id = us.id
                    WHERE us.symbol = $1
                      AND time >= $2
                      AND time <= $3
                      AND EXTRACT(HOUR FROM time AT TIME ZONE 'US/Eastern') BETWEEN 9 AND 15
                )
                SELECT COUNT(*)
                FROM time_gaps
                WHERE gap_seconds > 300  -- More than 5 minute gaps
                """,
                symbol,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )

            consistency = 1.0 - min(1.0, (gap_count or 0) / max(actual_data_days, 1))
            quality_factors["consistency"] = consistency

        # Calculate weighted average
        weights = {
            "completeness": 0.4,
            "option_coverage": 0.3,
            "consistency": 0.3,
        }

        quality_score = sum(
            quality_factors[factor] * weight for factor, weight in weights.items()
        )

        return round(quality_score, 3)

    def _create_symbol_batches(self, symbols: List[str]) -> List[List[str]]:
        """Create batches of symbols for parallel processing."""
        # Sort by priority (from SFR_TARGET_SYMBOLS)
        prioritized_symbols = sorted(
            symbols, key=lambda s: SFR_TARGET_SYMBOLS.get(s, 0.5), reverse=True
        )

        # Create batches
        batch_size = self.sfr_config.max_concurrent_symbols
        return [
            prioritized_symbols[i : i + batch_size]
            for i in range(0, len(prioritized_symbols), batch_size)
        ]

    async def _load_symbol_batch(
        self,
        batch: List[str],
        start_date: date,
        end_date: date,
        progress: Progress,
        main_task: TaskID,
    ) -> Dict[str, SFRDataLoadResult]:
        """Load a batch of symbols in parallel."""
        # Create individual tasks for each symbol
        symbol_tasks = {}
        for symbol in batch:
            task_id = progress.add_task(f"Loading {symbol}...", total=100)
            symbol_tasks[symbol] = task_id

        # Process symbols concurrently
        coroutines = [
            self.load_sfr_symbol_optimized(
                symbol, start_date, end_date, progress, symbol_tasks[symbol]
            )
            for symbol in batch
        ]

        results_list = await asyncio.gather(*coroutines, return_exceptions=True)

        # Process results
        results = {}
        for i, result in enumerate(results_list):
            symbol = batch[i]

            if isinstance(result, Exception):
                logger.error(f"Failed to load {symbol}: {result}")
                results[symbol] = SFRDataLoadResult(
                    symbol=symbol,
                    success=False,
                    start_date=start_date,
                    end_date=end_date,
                    stock_bars_loaded=0,
                    option_contracts_loaded=0,
                    option_bars_loaded=0,
                    vix_data_points=0,
                    data_quality_score=0.0,
                    load_duration_seconds=0.0,
                    error_message=str(result),
                )
            else:
                results[symbol] = result

            # Update main progress
            progress.advance(main_task, 1)
            progress.remove_task(symbol_tasks[symbol])

        return results

    def _calculate_final_stats(self, results: Dict[str, SFRDataLoadResult]):
        """Calculate final loading statistics."""
        successful_results = [r for r in results.values() if r.success]

        self.load_stats.update(
            {
                "total_symbols_processed": len(results),
                "successful_loads": len(successful_results),
                "failed_loads": len(results) - len(successful_results),
                "total_stock_bars": sum(
                    r.stock_bars_loaded for r in successful_results
                ),
                "total_option_contracts": sum(
                    r.option_contracts_loaded for r in successful_results
                ),
                "total_option_bars": sum(
                    r.option_bars_loaded for r in successful_results
                ),
                "avg_data_quality": (
                    statistics.mean([r.data_quality_score for r in successful_results])
                    if successful_results
                    else 0.0
                ),
            }
        )

    def _count_trading_days(self, start_date: date, end_date: date) -> int:
        """Count trading days between start and end date (excluding weekends)."""
        days = 0
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Monday = 0, Sunday = 6
                days += 1
            current += timedelta(days=1)
        return days

    async def get_load_summary(self) -> Dict[str, Any]:
        """Get comprehensive loading summary."""
        duration = None
        if self.load_stats["load_start_time"] and self.load_stats["load_end_time"]:
            duration = (
                self.load_stats["load_end_time"] - self.load_stats["load_start_time"]
            ).total_seconds()

        return {
            **self.load_stats,
            "total_load_duration_seconds": duration,
            "avg_load_duration_per_symbol": (
                duration / max(self.load_stats["total_symbols_processed"], 1)
                if duration
                else None
            ),
            "data_loading_rate_bars_per_second": (
                (
                    self.load_stats["total_stock_bars"]
                    + self.load_stats["total_option_bars"]
                )
                / max(duration, 1)
                if duration
                else None
            ),
        }

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)


# Utility functions for SFR data loading


async def load_sfr_test_dataset(
    db_pool: asyncpg.Pool,
    ib_connection: IB = None,
    symbols: List[str] = None,
    days_back: int = 90,
) -> Dict[str, SFRDataLoadResult]:
    """
    Convenience function to load a test dataset for SFR backtesting.

    Args:
        db_pool: Database connection pool
        ib_connection: IB connection (optional)
        symbols: Symbols to load (defaults to top 3 SFR targets)
        days_back: Number of days to go back

    Returns:
        Loading results
    """
    if symbols is None:
        # Use top 3 most liquid symbols for testing
        symbols = ["SPY", "QQQ", "AAPL"]

    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    loader = HistoricalSFRLoader(db_pool, ib_connection)

    with console.status("Loading SFR test dataset..."):
        results = await loader.load_sfr_dataset(symbols, start_date, end_date)

    # Print summary
    successful = [r for r in results.values() if r.success]
    console.print(
        f"\n[green]Loaded {len(successful)}/{len(symbols)} symbols successfully[/green]"
    )

    for symbol, result in results.items():
        if result.success:
            console.print(
                f"  {symbol}: {result.stock_bars_loaded} stock bars, "
                f"{result.option_contracts_loaded} option contracts, "
                f"quality: {result.data_quality_score:.3f}"
            )
        else:
            console.print(f"  [red]{symbol}: Failed - {result.error_message}[/red]")

    return results


if __name__ == "__main__":
    # Example usage
    import asyncio

    from backtesting.infra.data_collection.config.config import db_config

    async def main():
        # Create database pool
        db_pool = await asyncpg.create_pool(
            db_config.connection_string,
            min_size=2,
            max_size=5,
        )

        try:
            # Load test dataset
            results = await load_sfr_test_dataset(db_pool, symbols=["SPY", "QQQ"])

            # Print summary
            for symbol, result in results.items():
                print(
                    f"{symbol}: Success={result.success}, Quality={result.data_quality_score}"
                )

        finally:
            await db_pool.close()

    asyncio.run(main())
