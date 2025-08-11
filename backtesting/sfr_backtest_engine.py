"""
Comprehensive SFR (Synthetic Free Risk) Backtesting Engine

This module implements a production-ready backtesting engine for SFR arbitrage strategies,
adapted from the live trading logic in /modules/Arbitrage/SFR.py.

Key Features:
- Historical data loading and processing
- SFR opportunity detection using live trading logic
- Realistic execution modeling with multiple slippage models
- Commission and fee calculations
- Performance analytics and risk metrics
- VIX correlation analysis integration
- Comprehensive logging and error handling
- Configurable time periods and parameters

Author: Claude Code
Date: 2025-08-09
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import asyncpg
import logging
import numpy as np
import pandas as pd
from scipy import stats

# Backtesting infrastructure imports
from backtesting.infra.data_collection.config.config import DatabaseConfig

logger = logging.getLogger(__name__)


class SlippageModel(Enum):
    """Slippage calculation models for execution simulation."""

    NONE = "NONE"
    LINEAR = "LINEAR"
    SQUARE_ROOT = "SQUARE_ROOT"
    IMPACT = "IMPACT"


class VixRegime(Enum):
    """VIX volatility regimes for market environment filtering."""

    LOW = "LOW"  # VIX < 15
    MEDIUM = "MEDIUM"  # VIX 15-25
    HIGH = "HIGH"  # VIX 25-35
    EXTREME = "EXTREME"  # VIX > 35


class OpportunityQuality(Enum):
    """Quality classification for SFR opportunities."""

    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"
    UNKNOWN = "UNKNOWN"


class ExecutionDifficulty(Enum):
    """Execution difficulty classification."""

    EASY = "EASY"
    MODERATE = "MODERATE"
    DIFFICULT = "DIFFICULT"
    VERY_DIFFICULT = "VERY_DIFFICULT"
    UNKNOWN = "UNKNOWN"


class RejectionReason(Enum):
    """Reasons for opportunity rejection."""

    INVALID_STRIKE_SPREAD = "INVALID_STRIKE_SPREAD"
    EXPIRY_OUT_OF_RANGE = "EXPIRY_OUT_OF_RANGE"
    POOR_MONEYNESS = "POOR_MONEYNESS"
    MISSING_MARKET_DATA = "MISSING_MARKET_DATA"
    INVALID_CONTRACT_DATA = "INVALID_CONTRACT_DATA"
    BID_ASK_SPREAD_TOO_WIDE = "BID_ASK_SPREAD_TOO_WIDE"
    ARBITRAGE_CONDITION_NOT_MET = "ARBITRAGE_CONDITION_NOT_MET"
    NET_CREDIT_NEGATIVE = "NET_CREDIT_NEGATIVE"
    PROFIT_TARGET_NOT_MET = "PROFIT_TARGET_NOT_MET"
    PRICE_LIMIT_EXCEEDED = "PRICE_LIMIT_EXCEEDED"
    INVALID_STRIKE_COMBINATION = "INVALID_STRIKE_COMBINATION"
    INSUFFICIENT_LIQUIDITY = "INSUFFICIENT_LIQUIDITY"


@dataclass
class SFRBacktestConfig:
    """Configuration for SFR backtesting parameters."""

    # Basic strategy parameters
    profit_target: float = 0.50  # Minimum profit target percentage
    cost_limit: float = 120.0  # Maximum cost limit in dollars
    volume_limit: int = 100  # Minimum option volume threshold
    quantity: int = 1  # Number of contracts per trade

    # Strike selection parameters
    call_strike_range_days: int = 25  # Strike range below stock price
    put_strike_range_days: int = 25  # Strike range below stock price
    expiry_min_days: int = 19  # Minimum days to expiration
    expiry_max_days: int = 45  # Maximum days to expiration
    max_strike_combinations: int = 4  # Max strike pairs to test per expiry
    max_expiry_options: int = 8  # Max expiries to test per symbol

    # Risk management parameters
    max_bid_ask_spread_call: float = 20.0  # Maximum call bid-ask spread
    max_bid_ask_spread_put: float = 20.0  # Maximum put bid-ask spread
    combo_buffer_percent: float = 0.00  # Buffer for combo limit price
    data_timeout_seconds: int = 45  # Market data collection timeout

    # Execution simulation parameters
    slippage_model: SlippageModel = SlippageModel.LINEAR
    base_slippage_bps: int = 2  # Base slippage in basis points
    liquidity_penalty_factor: float = 1.0  # Penalty for low liquidity
    commission_per_contract: float = 1.00  # Commission cost per contract

    # Market environment filters
    vix_regime_filter: Optional[VixRegime] = None  # Filter by VIX regime
    min_vix_level: Optional[float] = None  # Minimum VIX level to trade
    max_vix_level: Optional[float] = None  # Maximum VIX level to trade
    exclude_vix_spikes: bool = False  # Exclude trading during VIX spikes


@dataclass
class ExpiryOption:
    """Data class to hold option contract information for a specific expiry."""

    expiry: str
    expiry_date: date
    call_strike: float
    put_strike: float
    call_contract_id: Optional[int] = None
    put_contract_id: Optional[int] = None


@dataclass
class MarketData:
    """Market data snapshot for backtesting."""

    timestamp: datetime
    stock_price: float
    call_bid: Optional[float] = None
    call_ask: Optional[float] = None
    call_last: Optional[float] = None
    call_volume: Optional[int] = None
    put_bid: Optional[float] = None
    put_ask: Optional[float] = None
    put_last: Optional[float] = None
    put_volume: Optional[int] = None
    vix_level: Optional[float] = None
    vix_regime: Optional[VixRegime] = None


@dataclass
class SFROpportunity:
    """Represents a discovered SFR arbitrage opportunity."""

    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    underlying_id: Optional[int] = None
    expiry_option: Optional[ExpiryOption] = None
    market_data: Optional[MarketData] = None

    # Calculated metrics
    net_credit: float = 0.0
    spread: float = 0.0
    min_profit: float = 0.0
    max_profit: float = 0.0
    min_roi: float = 0.0
    max_roi: float = 0.0

    # Risk metrics
    call_moneyness: float = 0.0
    put_moneyness: float = 0.0
    call_bid_ask_spread: float = 0.0
    put_bid_ask_spread: float = 0.0
    days_to_expiry: int = 0

    # Classification
    opportunity_quality: OpportunityQuality = OpportunityQuality.UNKNOWN
    execution_difficulty: ExecutionDifficulty = ExecutionDifficulty.UNKNOWN
    liquidity_score: float = 0.0

    # Qualification results
    quick_viability_check: bool = False
    viability_rejection_reason: Optional[str] = None
    conditions_check: bool = False
    conditions_rejection_reason: Optional[str] = None

    # Execution simulation
    simulated_execution: bool = False
    execution_timestamp: Optional[datetime] = None
    combo_limit_price: Optional[float] = None
    estimated_slippage: float = 0.0
    estimated_commission: float = 0.0


@dataclass
class SimulatedTrade:
    """Represents a simulated SFR trade execution."""

    trade_id: str = field(default_factory=lambda: str(uuid4()))
    opportunity: Optional[SFROpportunity] = None
    execution_timestamp: datetime = field(default_factory=datetime.now)
    quantity: int = 1

    # Individual leg execution details
    stock_execution_price: float = 0.0
    stock_execution_time: datetime = field(default_factory=datetime.now)
    stock_slippage: float = 0.0

    call_execution_price: float = 0.0
    call_execution_time: datetime = field(default_factory=datetime.now)
    call_slippage: float = 0.0

    put_execution_price: float = 0.0
    put_execution_time: datetime = field(default_factory=datetime.now)
    put_slippage: float = 0.0

    # Combined trade metrics
    total_execution_time_ms: int = 0
    combo_net_credit: float = 0.0
    total_slippage: float = 0.0
    total_commission: float = 0.0

    # Realized profit calculations
    realized_min_profit: float = 0.0
    realized_max_profit: float = 0.0
    realized_min_roi: float = 0.0
    realized_max_roi: float = 0.0

    # Trade status
    execution_status: str = "FILLED"
    execution_quality: str = "GOOD"
    failure_reason: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance analytics for backtesting results."""

    # Opportunity discovery metrics
    total_opportunities_found: int = 0
    opportunities_per_day: float = 0.0
    opportunities_by_quality: Dict[str, int] = field(default_factory=dict)
    avg_opportunity_quality_score: float = 0.0

    # Execution simulation results
    total_simulated_trades: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    execution_success_rate: float = 0.0

    # Profitability metrics
    total_gross_profit: float = 0.0
    total_net_profit: float = 0.0
    total_commissions_paid: float = 0.0
    total_slippage_cost: float = 0.0
    avg_profit_per_trade: float = 0.0
    median_profit_per_trade: float = 0.0

    # ROI statistics
    avg_min_roi: float = 0.0
    median_min_roi: float = 0.0
    best_min_roi: float = 0.0
    worst_min_roi: float = 0.0
    roi_standard_deviation: float = 0.0

    # Risk metrics
    max_single_trade_loss: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0


class SFRBacktestEngine:
    """
    Main SFR backtesting engine implementing the SFR arbitrage logic
    adapted from the live trading system for historical backtesting.
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        config: SFRBacktestConfig = None,
        db_config: DatabaseConfig = None,
    ):
        """
        Initialize the SFR backtesting engine.

        Args:
            db_pool: Database connection pool
            config: SFR backtesting configuration
            db_config: Database configuration
        """
        self.db_pool = db_pool
        self.config = config or SFRBacktestConfig()
        self.db_config = db_config or DatabaseConfig()

        # State management
        self.backtest_run_id: Optional[int] = None
        self.underlying_ids: Dict[str, int] = {}
        self.opportunities: List[SFROpportunity] = []
        self.trades: List[SimulatedTrade] = []
        self.rejections: List[Dict[str, Any]] = []

        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Target symbols for backtesting
        self.target_symbols = [
            "SPY",
            "QQQ",
            "AAPL",
            "MSFT",
            "NVDA",
            "TSLA",
            "AMZN",
            "META",
            "GOOGL",
            "JPM",
        ]

        logger.info(f"SFR Backtest Engine initialized with config: {self.config}")

    async def run_backtest(
        self,
        start_date: date,
        end_date: date,
        symbols: Optional[List[str]] = None,
        time_period_years: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive SFR backtesting for specified time period.

        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting
            symbols: List of symbols to test (default: target symbols)
            time_period_years: Alternative way to specify period (1, 3, 5, 10 years)

        Returns:
            Dictionary with comprehensive backtesting results
        """
        try:
            # Reset engine state for fresh backtest run
            logger.info("🔄 RESETTING ENGINE STATE...")
            self.backtest_run_id = None
            self.underlying_ids.clear()
            self.opportunities.clear()
            self.trades.clear()
            self.rejections.clear()
            self.performance_metrics = PerformanceMetrics()
            self.start_time = None
            self.end_time = None
            logger.info("   ✅ Engine state reset - ready for fresh backtest")

            logger.info("=" * 80)
            logger.info("🚀 STARTING SFR BACKTEST ENGINE")
            logger.info("=" * 80)

            # Enhanced input validation and logging
            logger.info("📋 PARAMETER VALIDATION:")
            logger.info(f"   start_date: {start_date} (type: {type(start_date)})")
            logger.info(f"   end_date: {end_date} (type: {type(end_date)})")
            logger.info(f"   symbols: {symbols} (type: {type(symbols)})")
            logger.info(f"   time_period_years: {time_period_years}")

            # Validate inputs
            if not isinstance(start_date, date) or not isinstance(end_date, date):
                raise ValueError(f"start_date and end_date must be date objects")

            if start_date >= end_date:
                raise ValueError(
                    f"start_date ({start_date}) must be before end_date ({end_date})"
                )

            # Adjust dates if time_period_years is specified
            if time_period_years:
                logger.info(f"⏰ Adjusting dates for {time_period_years} year period")
                end_date = date.today()
                start_date = end_date - timedelta(days=time_period_years * 365)
                logger.info(f"   Adjusted: {start_date} to {end_date}")

            # Use default symbols if not specified
            symbols = symbols or self.target_symbols
            logger.info(f"🎯 Final symbols list: {symbols} (count: {len(symbols)})")

            # Calculate and log date range info
            self.start_time = datetime.combine(start_date, datetime.min.time())
            self.end_time = datetime.combine(end_date, datetime.max.time())
            total_days = (end_date - start_date).days
            trading_days = sum(
                1
                for i in range(total_days)
                if (start_date + timedelta(days=i)).weekday() < 5
            )

            logger.info("📊 BACKTEST SCOPE:")
            logger.info(f"   Period: {start_date} to {end_date}")
            logger.info(f"   Total days: {total_days}")
            logger.info(f"   Trading days: {trading_days}")
            logger.info(f"   Expected operations: {len(symbols) * trading_days}")

            # Log configuration
            logger.info("⚙️ CONFIGURATION:")
            logger.info(f"   Profit target: {self.config.profit_target}%")
            logger.info(f"   Cost limit: ${self.config.cost_limit}")
            logger.info(f"   Volume limit: {self.config.volume_limit}")
            logger.info(f"   Slippage model: {self.config.slippage_model.value}")
            logger.info(
                f"   Commission per contract: ${self.config.commission_per_contract}"
            )

            # Database connection check
            logger.info("💾 DATABASE CONNECTION CHECK:")
            if hasattr(self, "db_pool") and self.db_pool:
                logger.info(
                    f"   Pool status: {'CLOSED' if self.db_pool.is_closing() else 'OPEN'}"
                )
                try:
                    async with self.db_pool.acquire() as conn:
                        test_result = await conn.fetchval("SELECT 1")
                        logger.info(
                            f"   Connection test: {'✅ SUCCESS' if test_result == 1 else '❌ FAILED'}"
                        )
                except Exception as e:
                    logger.error(f"   Connection test: ❌ FAILED - {e}")
                    raise
            else:
                logger.error("   ❌ No database pool available")
                raise RuntimeError("Database pool not available")

            logger.info("🔄 STARTING MAIN EXECUTION PHASES...")

            # Create backtest run record
            logger.info("📝 Phase 1: Creating backtest run record...")
            try:
                self.backtest_run_id = await self._create_backtest_run(
                    start_date, end_date
                )
                logger.info(f"   ✅ Created backtest run ID: {self.backtest_run_id}")
            except Exception as e:
                logger.error(f"   ❌ Failed to create backtest run: {e}")
                raise

            # Load underlying security IDs
            logger.info("🔍 Phase 2: Loading underlying security IDs...")
            try:
                await self._load_underlying_ids(symbols)
                loaded_symbols = list(self.underlying_ids.keys())
                logger.info(
                    f"   ✅ Loaded {len(loaded_symbols)} symbols: {loaded_symbols}"
                )
                if len(loaded_symbols) != len(symbols):
                    missing = set(symbols) - set(loaded_symbols)
                    logger.warning(f"   ⚠️  Missing symbols: {missing}")
            except Exception as e:
                logger.error(f"   ❌ Failed to load underlying IDs: {e}")
                raise

            # Main backtesting loop
            logger.info("🔁 Phase 3: Main backtesting loop...")
            total_opportunities = 0
            total_trades = 0
            symbol_count = 0

            for symbol in symbols:
                symbol_count += 1
                if symbol not in self.underlying_ids:
                    logger.warning(
                        f"[{symbol_count}/{len(symbols)}] Skipping {symbol} - no underlying ID found"
                    )
                    continue

                logger.info(
                    f"[{symbol_count}/{len(symbols)}] Processing symbol: {symbol}"
                )
                symbol_start_time = datetime.now()

                # Process each trading day
                symbol_opportunities = 0
                symbol_trades = 0
                days_processed = 0

                current_date = start_date
                while current_date <= end_date:
                    # Skip weekends
                    if current_date.weekday() < 5:  # Monday=0, Friday=4
                        try:
                            day_results = await self._process_trading_day(
                                symbol, current_date
                            )
                            symbol_opportunities += day_results["opportunities_found"]
                            symbol_trades += day_results["trades_executed"]
                            days_processed += 1

                            # Log progress every 10 days or if there were results
                            if (
                                days_processed % 10 == 0
                                or day_results["opportunities_found"] > 0
                            ):
                                logger.debug(
                                    f"   {symbol} {current_date}: {day_results['opportunities_found']} opportunities, {day_results['trades_executed']} trades"
                                )

                        except Exception as e:
                            logger.error(
                                f"   ❌ Error processing {symbol} on {current_date}: {e}"
                            )
                            # Continue processing other days

                    current_date += timedelta(days=1)

                symbol_duration = (datetime.now() - symbol_start_time).total_seconds()
                logger.info(
                    f"   ✅ {symbol} completed in {symbol_duration:.2f}s: "
                    f"{symbol_opportunities} opportunities, {symbol_trades} trades "
                    f"({days_processed} trading days)"
                )

                total_opportunities += symbol_opportunities
                total_trades += symbol_trades

            logger.info("📊 Phase 4: Calculating performance analytics...")
            try:
                await self._calculate_performance_analytics()
                logger.info("   ✅ Performance analytics calculated")
            except Exception as e:
                logger.error(f"   ❌ Failed to calculate analytics: {e}")
                raise

            logger.info("💾 Phase 5: Storing results in database...")
            try:
                await self._store_results()
                logger.info(
                    f"   ✅ Stored {len(self.opportunities)} opportunities, {len(self.trades)} trades"
                )
            except Exception as e:
                logger.error(f"   ❌ Failed to store results: {e}")
                raise

            logger.info("📋 Phase 6: Generating results summary...")
            try:
                results = await self._generate_results_summary()
                logger.info("   ✅ Results summary generated")
            except Exception as e:
                logger.error(f"   ❌ Failed to generate summary: {e}")
                raise

            # Final summary
            logger.info("=" * 80)
            logger.info("🎉 BACKTEST COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"📊 FINAL RESULTS:")
            logger.info(f"   Total opportunities found: {total_opportunities}")
            logger.info(f"   Total trades executed: {total_trades}")
            logger.info(
                f"   Success rate: {(total_trades / max(total_opportunities, 1)) * 100:.2f}%"
            )
            logger.info(f"   Backtest run ID: {self.backtest_run_id}")
            logger.info("=" * 80)

            return results

        except Exception as e:
            logger.error("=" * 80)
            logger.error("❌ BACKTEST FAILED")
            logger.error("=" * 80)
            logger.error(f"Error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback

            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.error("=" * 80)
            raise

    async def _create_backtest_run(self, start_date: date, end_date: date) -> int:
        """Create backtest run record in database."""
        logger.debug("📝 Creating backtest run record in database...")
        try:
            async with self.db_pool.acquire() as conn:
                # First create generic backtest run
                # Prepare parameters as JSON
                parameters = {
                    "profit_target": self.config.profit_target,
                    "cost_limit": self.config.cost_limit,
                    "volume_limit": self.config.volume_limit,
                    "quantity": self.config.quantity,
                    "slippage_model": self.config.slippage_model.value,
                    "commission_per_contract": self.config.commission_per_contract,
                }

                logger.debug(
                    f"   Creating generic backtest run for period {start_date} to {end_date}"
                )
                generic_run_id = await conn.fetchval(
                    """
                    INSERT INTO backtest_runs
                    (strategy_type, start_date, end_date, parameters, status)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                    """,
                    "SFR",
                    start_date,
                    end_date,
                    json.dumps(parameters),  # Convert dict to JSON string
                    "RUNNING",
                )
                logger.debug(f"   ✅ Generic run created with ID: {generic_run_id}")

                # Create SFR-specific backtest run
                logger.debug(f"   Creating SFR-specific backtest run...")
                sfr_run_id = await conn.fetchval(
                    """
                    INSERT INTO sfr_backtest_runs (
                        backtest_run_id, profit_target, cost_limit, volume_limit, quantity,
                        call_strike_range_days, put_strike_range_days, expiry_min_days, expiry_max_days,
                        max_strike_combinations, max_expiry_options, max_bid_ask_spread_call,
                        max_bid_ask_spread_put, combo_buffer_percent, data_timeout_seconds,
                        slippage_model, base_slippage_bps, liquidity_penalty_factor,
                        commission_per_contract, vix_regime_filter, min_vix_level, max_vix_level,
                        exclude_vix_spikes
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                            $16, $17, $18, $19, $20, $21, $22, $23)
                    RETURNING id
                    """,
                    generic_run_id,
                    self.config.profit_target,
                    self.config.cost_limit,
                    self.config.volume_limit,
                    self.config.quantity,
                    self.config.call_strike_range_days,
                    self.config.put_strike_range_days,
                    self.config.expiry_min_days,
                    self.config.expiry_max_days,
                    self.config.max_strike_combinations,
                    self.config.max_expiry_options,
                    self.config.max_bid_ask_spread_call,
                    self.config.max_bid_ask_spread_put,
                    self.config.combo_buffer_percent,
                    self.config.data_timeout_seconds,
                    self.config.slippage_model.value,
                    self.config.base_slippage_bps,
                    self.config.liquidity_penalty_factor,
                    self.config.commission_per_contract,
                    (
                        self.config.vix_regime_filter.value
                        if self.config.vix_regime_filter
                        else None
                    ),
                    self.config.min_vix_level,
                    self.config.max_vix_level,
                    self.config.exclude_vix_spikes,
                )

                logger.info(f"✅ Created SFR backtest run with ID: {sfr_run_id}")
                return sfr_run_id

        except Exception as e:
            logger.error(f"❌ Failed to create backtest run: {e}")
            import traceback

            logger.debug(f"   Traceback: {traceback.format_exc()}")
            raise

    async def _load_underlying_ids(self, symbols: List[str]) -> None:
        """Load underlying security IDs from database."""
        logger.debug(f"🔍 Loading underlying IDs for {len(symbols)} symbols...")
        async with self.db_pool.acquire() as conn:
            for symbol in symbols:
                try:
                    underlying_id = await conn.fetchval(
                        "SELECT id FROM underlying_securities WHERE symbol = $1", symbol
                    )
                    if underlying_id:
                        self.underlying_ids[symbol] = underlying_id
                        logger.debug(f"   ✅ {symbol}: ID {underlying_id}")
                    else:
                        logger.warning(f"   ❌ {symbol}: No underlying ID found")
                except Exception as e:
                    logger.error(f"   ❌ {symbol}: Database error - {e}")

        logger.debug(
            f"   📊 Successfully loaded {len(self.underlying_ids)}/{len(symbols)} symbols"
        )

    async def _process_trading_day(
        self, symbol: str, trading_date: date
    ) -> Dict[str, int]:
        """
        Process a single trading day for SFR opportunity scanning.

        Args:
            symbol: Stock symbol to process
            trading_date: Date to process

        Returns:
            Dictionary with opportunities_found and trades_executed counts
        """
        try:
            underlying_id = self.underlying_ids[symbol]
            opportunities_found = 0
            trades_executed = 0

            logger.debug(f"📅 Processing {symbol} on {trading_date}")

            # Get stock price data for the day
            stock_data = await self._get_stock_data(underlying_id, trading_date)
            if not stock_data:
                logger.debug(
                    f"   📊 No stock data found for {symbol} on {trading_date}"
                )
                return {"opportunities_found": 0, "trades_executed": 0}

            logger.debug(f"   📈 Stock price: ${stock_data['price']:.2f}")

            # Get VIX data for market environment context
            vix_data = await self._get_vix_data(trading_date)
            if vix_data:
                logger.debug(
                    f"   🌪️  VIX: {vix_data['vix_level']:.2f} ({vix_data['vix_regime'].value})"
                )

            # Skip if VIX filtering is enabled and conditions not met
            if not self._check_vix_filter(vix_data):
                logger.debug(
                    f"   ⏭️  Skipping {symbol} on {trading_date} due to VIX filter"
                )
                return {"opportunities_found": 0, "trades_executed": 0}

            # Get available option chains for the day
            expiry_options = await self._get_expiry_options(
                underlying_id, stock_data["price"], trading_date
            )

            if not expiry_options:
                logger.debug(
                    f"   📋 No valid expiry options for {symbol} on {trading_date}"
                )
                return {"opportunities_found": 0, "trades_executed": 0}

            logger.debug(f"   🔍 Found {len(expiry_options)} expiry options to analyze")

            # Process each expiry option
            for i, expiry_option in enumerate(expiry_options):
                logger.debug(
                    f"   [{i+1}/{len(expiry_options)}] Analyzing expiry {expiry_option.expiry}"
                )

                # Get option market data
                option_data = await self._get_option_data(expiry_option, trading_date)

                if not option_data:
                    logger.debug(
                        f"      ❌ No option data for expiry {expiry_option.expiry}"
                    )
                    continue

                # Create market data snapshot
                market_data = MarketData(
                    timestamp=datetime.combine(trading_date, datetime.min.time()),
                    stock_price=stock_data["price"],
                    call_bid=option_data["call_bid"],
                    call_ask=option_data["call_ask"],
                    call_last=option_data["call_last"],
                    call_volume=option_data["call_volume"],
                    put_bid=option_data["put_bid"],
                    put_ask=option_data["put_ask"],
                    put_last=option_data["put_last"],
                    put_volume=option_data["put_volume"],
                    vix_level=vix_data["vix_level"] if vix_data else None,
                    vix_regime=vix_data["vix_regime"] if vix_data else None,
                )

                # Check for SFR opportunity using live trading logic
                opportunity = await self._check_sfr_opportunity(
                    underlying_id, expiry_option, market_data
                )

                if opportunity:
                    self.opportunities.append(opportunity)
                    opportunities_found += 1
                    logger.debug(
                        f"      ✅ Found opportunity: min_roi={opportunity.min_roi:.2f}%, quality={opportunity.opportunity_quality.value}"
                    )

                    # Simulate trade execution if opportunity passes conditions
                    if opportunity.conditions_check:
                        trade = await self._simulate_trade_execution(opportunity)
                        if trade:
                            self.trades.append(trade)
                            trades_executed += 1
                            logger.debug(
                                f"      💰 Trade executed: profit=${trade.realized_min_profit:.2f}"
                            )
                    else:
                        logger.debug(
                            f"      ⚠️  Opportunity failed conditions: {opportunity.conditions_rejection_reason}"
                        )
                else:
                    logger.debug(
                        f"      ❌ No opportunity found for expiry {expiry_option.expiry}"
                    )

            if opportunities_found > 0 or trades_executed > 0:
                logger.debug(
                    f"   📊 {symbol} {trading_date} summary: {opportunities_found} opportunities, {trades_executed} trades"
                )

            return {
                "opportunities_found": opportunities_found,
                "trades_executed": trades_executed,
            }

        except Exception as e:
            logger.error(f"❌ Error processing {symbol} on {trading_date}: {e}")
            import traceback

            logger.debug(f"   Traceback: {traceback.format_exc()}")
            return {"opportunities_found": 0, "trades_executed": 0}

    async def _get_stock_data(
        self, underlying_id: int, trading_date: date
    ) -> Optional[Dict[str, Any]]:
        """Get stock price data for a specific day."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT price, volume, open_price, high_price, low_price, close_price
                FROM stock_data_ticks
                WHERE underlying_id = $1
                  AND DATE(time) = $2
                  AND tick_type = 'HISTORICAL'
                ORDER BY time DESC
                LIMIT 1
                """,
                underlying_id,
                trading_date,
            )

            if row:
                return {
                    "price": float(row["price"]),
                    "volume": row["volume"],
                    "open": float(row["open_price"]) if row["open_price"] else None,
                    "high": float(row["high_price"]) if row["high_price"] else None,
                    "low": float(row["low_price"]) if row["low_price"] else None,
                    "close": float(row["close_price"]) if row["close_price"] else None,
                }

            return None

    async def _get_vix_data(self, trading_date: date) -> Optional[Dict[str, Any]]:
        """Get VIX data for market environment context."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    close_price as vix_close,
                    CASE
                        WHEN close_price < 15 THEN 'LOW'
                        WHEN close_price < 25 THEN 'MEDIUM'
                        WHEN close_price < 40 THEN 'HIGH'
                        ELSE 'EXTREME'
                    END as vix_regime
                FROM vix_data_ticks
                WHERE instrument_id = 1 -- VIX instrument
                  AND DATE(time) = $1
                ORDER BY time DESC
                LIMIT 1
                """,
                trading_date,
            )

            if row:
                vix_level = float(row["vix_close"])

                # Determine VIX regime if not stored
                vix_regime = row["vix_regime"]
                if not vix_regime:
                    if vix_level < 15:
                        vix_regime = VixRegime.LOW.value
                    elif vix_level < 25:
                        vix_regime = VixRegime.MEDIUM.value
                    elif vix_level < 35:
                        vix_regime = VixRegime.HIGH.value
                    else:
                        vix_regime = VixRegime.EXTREME.value

                return {"vix_level": vix_level, "vix_regime": VixRegime(vix_regime)}

            return None

    def _check_vix_filter(self, vix_data: Optional[Dict[str, Any]]) -> bool:
        """Check if VIX conditions meet filtering criteria."""
        if not vix_data:
            return True  # No VIX data, allow trading

        vix_level = vix_data["vix_level"]
        vix_regime = vix_data["vix_regime"]

        # Check VIX regime filter
        if (
            self.config.vix_regime_filter
            and vix_regime != self.config.vix_regime_filter
        ):
            return False

        # Check VIX level bounds
        if self.config.min_vix_level and vix_level < self.config.min_vix_level:
            return False

        if self.config.max_vix_level and vix_level > self.config.max_vix_level:
            return False

        # Check VIX spike exclusion (>20% daily change)
        if self.config.exclude_vix_spikes:
            # This would require previous day VIX data - simplified for now
            if vix_level > 40:  # Simple spike detection
                return False

        return True

    async def _get_expiry_options(
        self, underlying_id: int, stock_price: float, trading_date: date
    ) -> List[ExpiryOption]:
        """Get valid expiry options for SFR scanning."""
        async with self.db_pool.acquire() as conn:
            # Get option contracts within expiry and strike range
            rows = await conn.fetch(
                """
                SELECT DISTINCT
                    call_oc.expiration_date,
                    call_oc.id as call_contract_id,
                    call_oc.strike_price as call_strike,
                    put_oc.id as put_contract_id,
                    put_oc.strike_price as put_strike
                FROM option_chains call_oc
                JOIN option_chains put_oc ON (
                    call_oc.underlying_id = put_oc.underlying_id
                    AND call_oc.expiration_date = put_oc.expiration_date
                )
                WHERE call_oc.underlying_id = $1
                  AND call_oc.option_type = 'C'
                  AND put_oc.option_type = 'P'
                  AND call_oc.expiration_date >= $2::date + INTERVAL '%s days'
                  AND call_oc.expiration_date <= $2::date + INTERVAL '%s days'
                  AND call_oc.strike_price > put_oc.strike_price
                  AND call_oc.strike_price <= $3::numeric + $4::numeric
                  AND put_oc.strike_price >= $3::numeric - $4::numeric
                  AND put_oc.strike_price <= $3::numeric
                ORDER BY call_oc.expiration_date, call_oc.strike_price
                LIMIT $5
                """
                % (self.config.expiry_min_days, self.config.expiry_max_days),
                underlying_id,
                trading_date,
                stock_price,
                self.config.call_strike_range_days,  # Strike range in dollars
                self.config.max_expiry_options * self.config.max_strike_combinations,
            )

            expiry_options = []
            for row in rows:
                expiry_date = row["expiration_date"]
                expiry_str = expiry_date.strftime("%Y%m%d")

                expiry_option = ExpiryOption(
                    expiry=expiry_str,
                    expiry_date=expiry_date,
                    call_strike=float(row["call_strike"]),
                    put_strike=float(row["put_strike"]),
                    call_contract_id=row["call_contract_id"],
                    put_contract_id=row["put_contract_id"],
                )
                expiry_options.append(expiry_option)

            return expiry_options

    async def _get_option_data(
        self, expiry_option: ExpiryOption, trading_date: date
    ) -> Optional[Dict[str, Any]]:
        """Get option market data for both call and put contracts."""
        async with self.db_pool.acquire() as conn:
            # Get call data
            call_data = await conn.fetchrow(
                """
                SELECT bid_price, ask_price, last_price, volume
                FROM market_data_ticks
                WHERE contract_id = $1
                  AND DATE(time) = $2
                  AND tick_type = 'HISTORICAL'
                ORDER BY time DESC
                LIMIT 1
                """,
                expiry_option.call_contract_id,
                trading_date,
            )

            # Get put data
            put_data = await conn.fetchrow(
                """
                SELECT bid_price, ask_price, last_price, volume
                FROM market_data_ticks
                WHERE contract_id = $1
                  AND DATE(time) = $2
                  AND tick_type = 'HISTORICAL'
                ORDER BY time DESC
                LIMIT 1
                """,
                expiry_option.put_contract_id,
                trading_date,
            )

            if not call_data or not put_data:
                return None

            return {
                "call_bid": (
                    float(call_data["bid_price"]) if call_data["bid_price"] else None
                ),
                "call_ask": (
                    float(call_data["ask_price"]) if call_data["ask_price"] else None
                ),
                "call_last": (
                    float(call_data["last_price"]) if call_data["last_price"] else None
                ),
                "call_volume": call_data["volume"] or 0,
                "put_bid": (
                    float(put_data["bid_price"]) if put_data["bid_price"] else None
                ),
                "put_ask": (
                    float(put_data["ask_price"]) if put_data["ask_price"] else None
                ),
                "put_last": (
                    float(put_data["last_price"]) if put_data["last_price"] else None
                ),
                "put_volume": put_data["volume"] or 0,
            }

    async def _check_sfr_opportunity(
        self, underlying_id: int, expiry_option: ExpiryOption, market_data: MarketData
    ) -> Optional[SFROpportunity]:
        """
        Check for SFR arbitrage opportunity using live trading logic.

        This method implements the core SFR detection logic from the live system,
        adapted for historical backtesting.
        """
        try:
            # Initialize opportunity
            opportunity = SFROpportunity(
                underlying_id=underlying_id,
                expiry_option=expiry_option,
                market_data=market_data,
                timestamp=market_data.timestamp,
            )

            # Quick viability check (from SFRExecutor.quick_viability_check)
            viable, rejection_reason = self._quick_viability_check(
                expiry_option, market_data.stock_price
            )

            opportunity.quick_viability_check = viable
            opportunity.viability_rejection_reason = rejection_reason

            if not viable:
                await self._log_rejection(
                    underlying_id,
                    "QUICK_VIABILITY",
                    rejection_reason,
                    expiry_option,
                    market_data,
                )
                return opportunity

            # Get prices for calculation
            call_price = self._get_option_price(
                market_data.call_bid,
                market_data.call_ask,
                market_data.call_last,
                "SELL",
            )
            put_price = self._get_option_price(
                market_data.put_bid, market_data.put_ask, market_data.put_last, "BUY"
            )

            if call_price is None or put_price is None:
                await self._log_rejection(
                    underlying_id,
                    "MARKET_DATA_MISSING",
                    "invalid_option_prices",
                    expiry_option,
                    market_data,
                )
                return opportunity

            # Check bid-ask spreads
            call_spread = self._calculate_bid_ask_spread(
                market_data.call_bid, market_data.call_ask
            )
            put_spread = self._calculate_bid_ask_spread(
                market_data.put_bid, market_data.put_ask
            )

            if (
                call_spread > self.config.max_bid_ask_spread_call
                or put_spread > self.config.max_bid_ask_spread_put
            ):
                await self._log_rejection(
                    underlying_id,
                    "SPREAD_TOO_WIDE",
                    f"call_spread_{call_spread:.2f}_put_spread_{put_spread:.2f}",
                    expiry_option,
                    market_data,
                )
                return opportunity

            # Calculate SFR metrics (from SFRExecutor.calc_price_and_build_order_for_expiry)
            net_credit = call_price - put_price
            spread = market_data.stock_price - expiry_option.put_strike
            min_profit = net_credit - spread
            max_profit = (
                expiry_option.call_strike - expiry_option.put_strike
            ) + net_credit
            min_roi = (
                (min_profit / (market_data.stock_price + net_credit)) * 100
                if (market_data.stock_price + net_credit) != 0
                else 0
            )
            max_roi = (
                (max_profit / (market_data.stock_price + net_credit)) * 100
                if (market_data.stock_price + net_credit) != 0
                else 0
            )

            # Calculate risk metrics
            call_moneyness = expiry_option.call_strike / market_data.stock_price
            put_moneyness = expiry_option.put_strike / market_data.stock_price
            days_to_expiry = (
                expiry_option.expiry_date - market_data.timestamp.date()
            ).days

            # Update opportunity with calculated metrics
            opportunity.net_credit = net_credit
            opportunity.spread = spread
            opportunity.min_profit = min_profit
            opportunity.max_profit = max_profit
            opportunity.min_roi = min_roi
            opportunity.max_roi = max_roi
            opportunity.call_moneyness = call_moneyness
            opportunity.put_moneyness = put_moneyness
            opportunity.call_bid_ask_spread = call_spread
            opportunity.put_bid_ask_spread = put_spread
            opportunity.days_to_expiry = days_to_expiry

            # Calculate combo limit price
            combo_limit_price = self._calculate_combo_limit_price(
                market_data.stock_price,
                call_price,
                put_price,
                self.config.combo_buffer_percent,
            )
            opportunity.combo_limit_price = combo_limit_price

            # Check conditions (from SFRExecutor.check_conditions)
            conditions_met, conditions_rejection = self._check_conditions(
                market_data.stock_price,
                expiry_option.put_strike,
                combo_limit_price,
                net_credit,
                min_roi,
                min_profit,
            )

            opportunity.conditions_check = conditions_met
            opportunity.conditions_rejection_reason = conditions_rejection

            if not conditions_met:
                await self._log_rejection(
                    underlying_id,
                    "CONDITIONS_CHECK",
                    conditions_rejection,
                    expiry_option,
                    market_data,
                )

            # Calculate quality scores
            opportunity.liquidity_score = self._calculate_liquidity_score(
                market_data.call_volume, market_data.put_volume, call_spread, put_spread
            )
            opportunity.opportunity_quality = self._classify_opportunity_quality(
                min_roi,
                opportunity.liquidity_score,
                call_spread,
                put_spread,
                days_to_expiry,
            )
            opportunity.execution_difficulty = self._classify_execution_difficulty(
                opportunity.liquidity_score, call_spread, put_spread
            )

            return opportunity

        except Exception as e:
            logger.error(f"Error checking SFR opportunity: {e}")
            return None

    def _quick_viability_check(
        self, expiry_option: ExpiryOption, stock_price: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Fast pre-filtering to eliminate non-viable opportunities early.
        Adapted from SFRExecutor.quick_viability_check.
        """
        # Quick strike spread check
        strike_spread = expiry_option.call_strike - expiry_option.put_strike
        if strike_spread < 1.0 or strike_spread > 50.0:
            return False, "invalid_strike_spread"

        # Quick time to expiry check
        days_to_expiry = (expiry_option.expiry_date - date.today()).days
        if (
            days_to_expiry < self.config.expiry_min_days
            or days_to_expiry > self.config.expiry_max_days
        ):
            return False, "expiry_out_of_range"

        # Quick moneyness check
        call_moneyness = expiry_option.call_strike / stock_price
        put_moneyness = expiry_option.put_strike / stock_price

        if (
            call_moneyness < 0.90
            or call_moneyness > 1.15
            or put_moneyness < 0.80
            or put_moneyness > 1.10
        ):
            return False, "poor_moneyness"

        return True, None

    def _get_option_price(
        self,
        bid: Optional[float],
        ask: Optional[float],
        last: Optional[float],
        side: str,
    ) -> Optional[float]:
        """Get appropriate option price for bid/ask side."""
        if side == "SELL":  # Selling option (calls in conversion)
            return bid if bid is not None else last
        else:  # Buying option (puts in conversion)
            return ask if ask is not None else last

    def _calculate_bid_ask_spread(
        self, bid: Optional[float], ask: Optional[float]
    ) -> float:
        """Calculate bid-ask spread."""
        if bid is not None and ask is not None:
            return abs(ask - bid)
        return float("inf")

    def _check_conditions(
        self,
        stock_price: float,
        put_strike: float,
        lmt_price: float,
        net_credit: float,
        min_roi: float,
        min_profit: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check SFR conditions for arbitrage opportunity.
        Adapted from SFRExecutor.check_conditions.
        """
        spread = stock_price - put_strike

        # For conversion arbitrage: min_profit = net_credit - spread
        if spread >= net_credit:
            return False, "arbitrage_condition_not_met"

        if min_profit <= 0:
            return False, "arbitrage_condition_not_met"

        if net_credit < 0:
            return False, "net_credit_negative"

        if self.config.profit_target > min_roi:
            return False, "profit_target_not_met"

        if np.isnan(lmt_price) or lmt_price > self.config.cost_limit:
            return False, "price_limit_exceeded"

        return True, None

    def _calculate_combo_limit_price(
        self,
        stock_price: float,
        call_price: float,
        put_price: float,
        buffer_percent: float,
    ) -> float:
        """Calculate combo limit price with buffer."""
        base_price = stock_price + call_price - put_price
        buffer = base_price * buffer_percent
        return base_price + buffer

    def _calculate_liquidity_score(
        self,
        call_volume: Optional[int],
        put_volume: Optional[int],
        call_spread: float,
        put_spread: float,
    ) -> float:
        """Calculate composite liquidity score (0-1)."""
        # Volume component (0-0.5)
        total_volume = (call_volume or 0) + (put_volume or 0)
        volume_score = min(0.5, total_volume / 200.0)  # Normalize to 200 volume = 0.5

        # Spread component (0-0.5) - tighter spreads are better
        avg_spread = (call_spread + put_spread) / 2.0
        spread_score = max(0, 0.5 - (avg_spread / 10.0))  # Normalize to $10 spread = 0

        return volume_score + spread_score

    def _classify_opportunity_quality(
        self,
        min_roi: float,
        liquidity_score: float,
        call_spread: float,
        put_spread: float,
        days_to_expiry: int,
    ) -> OpportunityQuality:
        """Classify opportunity quality based on multiple factors."""
        # Calculate composite score
        roi_component = min(0.4, max(0, min_roi / 5.0 * 0.4))
        liquidity_component = liquidity_score * 0.3
        spread_component = max(0, 0.2 - ((call_spread + put_spread) / 40.0 * 0.2))
        time_component = 0.1 if 25 <= days_to_expiry <= 35 else 0.05

        quality_score = (
            roi_component + liquidity_component + spread_component + time_component
        )

        if quality_score >= 0.8:
            return OpportunityQuality.EXCELLENT
        elif quality_score >= 0.6:
            return OpportunityQuality.GOOD
        elif quality_score >= 0.4:
            return OpportunityQuality.FAIR
        else:
            return OpportunityQuality.POOR

    def _classify_execution_difficulty(
        self, liquidity_score: float, call_spread: float, put_spread: float
    ) -> ExecutionDifficulty:
        """Classify execution difficulty based on liquidity and spreads."""
        if liquidity_score >= 0.7 and call_spread <= 2.0 and put_spread <= 2.0:
            return ExecutionDifficulty.EASY
        elif liquidity_score >= 0.5 and call_spread <= 5.0 and put_spread <= 5.0:
            return ExecutionDifficulty.MODERATE
        elif liquidity_score >= 0.3 and call_spread <= 10.0 and put_spread <= 10.0:
            return ExecutionDifficulty.DIFFICULT
        else:
            return ExecutionDifficulty.VERY_DIFFICULT

    async def _simulate_trade_execution(
        self, opportunity: SFROpportunity
    ) -> Optional[SimulatedTrade]:
        """
        Simulate realistic trade execution with slippage and commission modeling.
        """
        try:
            if not opportunity.conditions_check:
                return None

            trade = SimulatedTrade(
                opportunity=opportunity,
                execution_timestamp=opportunity.timestamp,
                quantity=self.config.quantity,
            )

            # Simulate individual leg execution with slippage
            stock_price = opportunity.market_data.stock_price
            call_price = self._get_option_price(
                opportunity.market_data.call_bid,
                opportunity.market_data.call_ask,
                opportunity.market_data.call_last,
                "SELL",
            )
            put_price = self._get_option_price(
                opportunity.market_data.put_bid,
                opportunity.market_data.put_ask,
                opportunity.market_data.put_last,
                "BUY",
            )

            # Calculate slippage for each leg
            stock_slippage = self._calculate_slippage(
                stock_price, opportunity.liquidity_score, "BUY"
            )
            call_slippage = self._calculate_slippage(
                call_price, opportunity.liquidity_score, "SELL"
            )
            put_slippage = self._calculate_slippage(
                put_price, opportunity.liquidity_score, "BUY"
            )

            # Apply slippage to execution prices
            trade.stock_execution_price = stock_price + stock_slippage
            trade.stock_slippage = stock_slippage
            trade.stock_execution_time = opportunity.timestamp

            trade.call_execution_price = call_price + call_slippage
            trade.call_slippage = call_slippage
            trade.call_execution_time = opportunity.timestamp + timedelta(
                milliseconds=50
            )

            trade.put_execution_price = put_price + put_slippage
            trade.put_slippage = put_slippage
            trade.put_execution_time = opportunity.timestamp + timedelta(
                milliseconds=100
            )

            # Calculate execution metrics
            trade.total_execution_time_ms = 100  # Simulated 100ms execution
            trade.combo_net_credit = (
                trade.call_execution_price - trade.put_execution_price
            )
            trade.total_slippage = (
                abs(stock_slippage) + abs(call_slippage) + abs(put_slippage)
            )
            trade.total_commission = (
                self.config.commission_per_contract * 3 * self.config.quantity
            )  # 3 legs

            # Calculate realized profits (after slippage and commissions)
            execution_spread = (
                trade.stock_execution_price - opportunity.expiry_option.put_strike
            )
            trade.realized_min_profit = (
                trade.combo_net_credit - execution_spread - trade.total_commission
            )
            trade.realized_max_profit = (
                (
                    opportunity.expiry_option.call_strike
                    - opportunity.expiry_option.put_strike
                )
                + trade.combo_net_credit
                - trade.total_commission
            )

            # Calculate realized ROI
            capital_required = trade.stock_execution_price + trade.combo_net_credit
            if capital_required != 0:
                trade.realized_min_roi = (
                    trade.realized_min_profit / capital_required
                ) * 100
                trade.realized_max_roi = (
                    trade.realized_max_profit / capital_required
                ) * 100

            # Determine execution quality
            slippage_impact = trade.total_slippage / stock_price * 100  # As percentage
            if slippage_impact <= 0.05:  # <= 5 bps
                trade.execution_quality = "EXCELLENT"
            elif slippage_impact <= 0.10:  # <= 10 bps
                trade.execution_quality = "GOOD"
            elif slippage_impact <= 0.20:  # <= 20 bps
                trade.execution_quality = "FAIR"
            else:
                trade.execution_quality = "POOR"

            # Mark opportunity as executed
            opportunity.simulated_execution = True
            opportunity.execution_timestamp = trade.execution_timestamp
            opportunity.estimated_slippage = trade.total_slippage
            opportunity.estimated_commission = trade.total_commission

            return trade

        except Exception as e:
            logger.error(f"Error simulating trade execution: {e}")
            return None

    def _calculate_slippage(
        self, price: float, liquidity_score: float, side: str
    ) -> float:
        """
        Calculate slippage based on model, price, liquidity, and side.

        Args:
            price: Contract price
            liquidity_score: Liquidity score (0-1)
            side: "BUY" or "SELL"
        """
        if self.config.slippage_model == SlippageModel.NONE:
            return 0.0

        # Base slippage in dollars
        base_slippage = price * (self.config.base_slippage_bps / 10000.0)

        # Liquidity penalty (higher penalty for lower liquidity)
        liquidity_penalty = (
            1.0 - liquidity_score
        ) * self.config.liquidity_penalty_factor

        if self.config.slippage_model == SlippageModel.LINEAR:
            slippage = base_slippage * (1.0 + liquidity_penalty)
        elif self.config.slippage_model == SlippageModel.SQUARE_ROOT:
            slippage = base_slippage * np.sqrt(1.0 + liquidity_penalty)
        elif self.config.slippage_model == SlippageModel.IMPACT:
            # Market impact model - higher impact for larger trades
            impact_factor = 1.0 + (
                self.config.quantity / 10.0
            )  # More impact for larger quantities
            slippage = base_slippage * impact_factor * (1.0 + liquidity_penalty)
        else:
            slippage = base_slippage

        # Apply directional slippage (adverse price movement)
        if side == "BUY":
            return slippage  # Pay more when buying
        else:
            return -slippage  # Receive less when selling

    async def _log_rejection(
        self,
        underlying_id: int,
        rejection_stage: str,
        rejection_reason: str,
        expiry_option: ExpiryOption,
        market_data: MarketData,
    ) -> None:
        """Log rejection for analysis and strategy optimization."""
        rejection_record = {
            "underlying_id": underlying_id,
            "rejection_timestamp": market_data.timestamp,
            "rejection_stage": rejection_stage,
            "rejection_reason": rejection_reason,
            "expiry_date": expiry_option.expiry_date,
            "call_strike": expiry_option.call_strike,
            "put_strike": expiry_option.put_strike,
            "stock_price": market_data.stock_price,
            "vix_level_at_rejection": market_data.vix_level,
            "vix_regime_at_rejection": (
                market_data.vix_regime.value if market_data.vix_regime else None
            ),
        }

        self.rejections.append(rejection_record)

    async def _calculate_performance_analytics(self) -> None:
        """Calculate comprehensive performance analytics."""
        try:
            # Basic counts
            self.performance_metrics.total_opportunities_found = len(self.opportunities)
            self.performance_metrics.total_simulated_trades = len(self.trades)

            successful_trades = [
                t for t in self.trades if t.execution_status == "FILLED"
            ]
            self.performance_metrics.successful_executions = len(successful_trades)
            self.performance_metrics.failed_executions = len(self.trades) - len(
                successful_trades
            )

            if len(self.trades) > 0:
                self.performance_metrics.execution_success_rate = len(
                    successful_trades
                ) / len(self.trades)

            # Calculate opportunities per day
            if self.start_time and self.end_time:
                total_days = (self.end_time - self.start_time).days
                if total_days > 0:
                    self.performance_metrics.opportunities_per_day = (
                        len(self.opportunities) / total_days
                    )

            # Profitability metrics
            if successful_trades:
                profits = [t.realized_min_profit for t in successful_trades]
                commissions = [t.total_commission for t in successful_trades]
                slippage_costs = [
                    abs(t.total_slippage) * t.opportunity.market_data.stock_price
                    for t in successful_trades
                ]

                self.performance_metrics.total_gross_profit = sum(
                    t.realized_min_profit + t.total_commission
                    for t in successful_trades
                )
                self.performance_metrics.total_net_profit = sum(profits)
                self.performance_metrics.total_commissions_paid = sum(commissions)
                self.performance_metrics.total_slippage_cost = sum(slippage_costs)
                self.performance_metrics.avg_profit_per_trade = np.mean(profits)
                self.performance_metrics.median_profit_per_trade = np.median(profits)

            # ROI statistics
            if successful_trades:
                rois = [t.realized_min_roi for t in successful_trades]
                self.performance_metrics.avg_min_roi = np.mean(rois)
                self.performance_metrics.median_min_roi = np.median(rois)
                self.performance_metrics.best_min_roi = np.max(rois)
                self.performance_metrics.worst_min_roi = np.min(rois)
                self.performance_metrics.roi_standard_deviation = np.std(rois)

            # Risk metrics
            if successful_trades:
                losses = [
                    t.realized_min_profit
                    for t in successful_trades
                    if t.realized_min_profit < 0
                ]
                if losses:
                    self.performance_metrics.max_single_trade_loss = min(losses)

                # Calculate drawdown
                cumulative_profits = np.cumsum(
                    [t.realized_min_profit for t in successful_trades]
                )
                running_max = np.maximum.accumulate(cumulative_profits)
                drawdown = cumulative_profits - running_max

                if len(drawdown) > 0:
                    self.performance_metrics.max_drawdown = min(drawdown)
                    if max(running_max) > 0:
                        self.performance_metrics.max_drawdown_percent = (
                            abs(self.performance_metrics.max_drawdown)
                            / max(running_max)
                            * 100
                        )

                # Sharpe ratio (simplified - assuming risk-free rate of 2%)
                if self.performance_metrics.roi_standard_deviation > 0:
                    excess_return = (
                        self.performance_metrics.avg_min_roi - 2.0
                    )  # Risk-free rate
                    self.performance_metrics.sharpe_ratio = (
                        excess_return / self.performance_metrics.roi_standard_deviation
                    )

            # Quality breakdown
            quality_counts = {}
            for opp in self.opportunities:
                quality = opp.opportunity_quality.value
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
            self.performance_metrics.opportunities_by_quality = quality_counts

            logger.info("Performance analytics calculated successfully")

        except Exception as e:
            logger.error(f"Error calculating performance analytics: {e}")

    async def _store_results(self) -> None:
        """Store all backtesting results to database."""
        try:
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    # Store opportunities
                    for opportunity in self.opportunities:
                        opportunity.id = await conn.fetchval(
                            """
                            INSERT INTO sfr_opportunities (
                                backtest_run_id, underlying_id, discovery_timestamp,
                                expiry_date, call_strike, put_strike,
                                call_contract_id, put_contract_id,
                                stock_price, call_bid, call_ask, call_last, call_volume,
                                put_bid, put_ask, put_last, put_volume,
                                net_credit, min_profit, max_profit, min_roi,
                                call_delta, call_gamma, call_theta, call_vega, call_iv,
                                put_delta, put_gamma, put_theta, put_vega, put_iv,
                                opportunity_quality, execution_difficulty, liquidity_score,
                                quick_viability_check, viability_rejection_reason,
                                conditions_check, conditions_rejection_reason,
                                simulated_execution, execution_timestamp,
                                combo_limit_price, estimated_slippage, estimated_commission
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                                    $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25,
                                    $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37,
                                    $38, $39, $40, $41, $42, $43)
                            RETURNING id
                            """,
                            self.backtest_run_id,
                            opportunity.underlying_id,
                            opportunity.timestamp,
                            opportunity.expiry_option.expiry_date,
                            opportunity.expiry_option.call_strike,
                            opportunity.expiry_option.put_strike,
                            opportunity.expiry_option.call_contract_id,
                            opportunity.expiry_option.put_contract_id,
                            opportunity.market_data.stock_price,
                            opportunity.market_data.call_bid,
                            opportunity.market_data.call_ask,
                            opportunity.market_data.call_last,
                            opportunity.market_data.call_volume,
                            opportunity.market_data.put_bid,
                            opportunity.market_data.put_ask,
                            opportunity.market_data.put_last,
                            opportunity.market_data.put_volume,
                            opportunity.net_credit,
                            opportunity.min_profit,
                            opportunity.max_profit,
                            opportunity.min_roi,
                            None,
                            None,
                            None,
                            None,
                            None,  # Greeks (not implemented yet)
                            None,
                            None,
                            None,
                            None,
                            None,  # Put Greeks
                            opportunity.opportunity_quality.value,
                            opportunity.execution_difficulty.value,
                            opportunity.liquidity_score,
                            opportunity.quick_viability_check,
                            opportunity.viability_rejection_reason,
                            opportunity.conditions_check,
                            opportunity.conditions_rejection_reason,
                            opportunity.simulated_execution,
                            opportunity.execution_timestamp,
                            opportunity.combo_limit_price,
                            opportunity.estimated_slippage,
                            opportunity.estimated_commission,
                        )

                    # Store simulated trades
                    for trade in self.trades:
                        await conn.execute(
                            """
                            INSERT INTO sfr_simulated_trades (
                                opportunity_id, backtest_run_id, trade_id,
                                execution_timestamp, quantity,
                                stock_execution_price, stock_execution_time, stock_slippage,
                                call_execution_price, call_execution_time, call_slippage,
                                put_execution_price, put_execution_time, put_slippage,
                                total_execution_time_ms, combo_net_credit,
                                total_slippage, total_commission,
                                realized_min_profit, realized_max_profit,
                                realized_min_roi, realized_max_roi,
                                execution_status, execution_quality, failure_reason,
                                vix_level_at_execution, vix_regime_at_execution
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                                    $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25,
                                    $26, $27)
                            """,
                            trade.opportunity.id,
                            self.backtest_run_id,
                            trade.trade_id,
                            trade.execution_timestamp,
                            trade.quantity,
                            trade.stock_execution_price,
                            trade.stock_execution_time,
                            trade.stock_slippage,
                            trade.call_execution_price,
                            trade.call_execution_time,
                            trade.call_slippage,
                            trade.put_execution_price,
                            trade.put_execution_time,
                            trade.put_slippage,
                            trade.total_execution_time_ms,
                            trade.combo_net_credit,
                            trade.total_slippage,
                            trade.total_commission,
                            trade.realized_min_profit,
                            trade.realized_max_profit,
                            trade.realized_min_roi,
                            trade.realized_max_roi,
                            trade.execution_status,
                            trade.execution_quality,
                            trade.failure_reason,
                            trade.opportunity.market_data.vix_level,
                            (
                                trade.opportunity.market_data.vix_regime.value
                                if trade.opportunity.market_data.vix_regime
                                else None
                            ),
                        )

                    # Store rejection log
                    for rejection in self.rejections:
                        await conn.execute(
                            """
                            INSERT INTO sfr_rejection_log (
                                backtest_run_id, underlying_id, rejection_timestamp,
                                rejection_stage, rejection_reason,
                                expiry_date, call_strike, put_strike, stock_price,
                                vix_level_at_rejection, vix_regime_at_rejection
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                            """,
                            self.backtest_run_id,
                            rejection["underlying_id"],
                            rejection["rejection_timestamp"],
                            rejection["rejection_stage"],
                            rejection["rejection_reason"],
                            rejection.get("expiry_date"),
                            rejection.get("call_strike"),
                            rejection.get("put_strike"),
                            rejection.get("stock_price"),
                            rejection.get("vix_level_at_rejection"),
                            rejection.get("vix_regime_at_rejection"),
                        )

                    # Store performance analytics
                    await conn.execute(
                        """
                        INSERT INTO sfr_performance_analytics (
                            backtest_run_id, total_opportunities_found, opportunities_per_day,
                            opportunities_by_quality, total_simulated_trades,
                            successful_executions, failed_executions, execution_success_rate,
                            total_gross_profit, total_net_profit, total_commissions_paid,
                            total_slippage_cost, avg_profit_per_trade, median_profit_per_trade,
                            avg_min_roi, median_min_roi, best_min_roi, worst_min_roi,
                            roi_standard_deviation, max_single_trade_loss, max_drawdown,
                            max_drawdown_percent, sharpe_ratio
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                                $14, $15, $16, $17, $18, $19, $20, $21, $22, $23)
                        """,
                        self.backtest_run_id,
                        self.performance_metrics.total_opportunities_found,
                        self.performance_metrics.opportunities_per_day,
                        (
                            json.dumps(
                                self.performance_metrics.opportunities_by_quality
                            )
                            if self.performance_metrics.opportunities_by_quality
                            else "{}"
                        ),
                        self.performance_metrics.total_simulated_trades,
                        self.performance_metrics.successful_executions,
                        self.performance_metrics.failed_executions,
                        self.performance_metrics.execution_success_rate,
                        self.performance_metrics.total_gross_profit,
                        self.performance_metrics.total_net_profit,
                        self.performance_metrics.total_commissions_paid,
                        self.performance_metrics.total_slippage_cost,
                        self.performance_metrics.avg_profit_per_trade,
                        self.performance_metrics.median_profit_per_trade,
                        self.performance_metrics.avg_min_roi,
                        self.performance_metrics.median_min_roi,
                        self.performance_metrics.best_min_roi,
                        self.performance_metrics.worst_min_roi,
                        self.performance_metrics.roi_standard_deviation,
                        self.performance_metrics.max_single_trade_loss,
                        self.performance_metrics.max_drawdown,
                        self.performance_metrics.max_drawdown_percent,
                        self.performance_metrics.sharpe_ratio,
                    )

                    # Update backtest run status
                    await conn.execute(
                        """
                        UPDATE backtest_runs
                        SET status = 'COMPLETED', completed_at = CURRENT_TIMESTAMP
                        WHERE id = (
                            SELECT backtest_run_id FROM sfr_backtest_runs
                            WHERE id = $1
                        )
                        """,
                        self.backtest_run_id,
                    )

            logger.info(
                f"Stored {len(self.opportunities)} opportunities, {len(self.trades)} trades, {len(self.rejections)} rejections"
            )

        except Exception as e:
            logger.error(f"Error storing results: {e}")
            raise

    async def _generate_results_summary(self) -> Dict[str, Any]:
        """Generate comprehensive results summary."""
        return {
            "backtest_run_id": self.backtest_run_id,
            "config": {
                "profit_target": self.config.profit_target,
                "cost_limit": self.config.cost_limit,
                "slippage_model": self.config.slippage_model.value,
                "commission_per_contract": self.config.commission_per_contract,
            },
            "period": {
                "start_date": self.start_time.date() if self.start_time else None,
                "end_date": self.end_time.date() if self.end_time else None,
                "total_days": (
                    (self.end_time - self.start_time).days
                    if self.start_time and self.end_time
                    else None
                ),
            },
            "opportunities": {
                "total_found": self.performance_metrics.total_opportunities_found,
                "per_day": self.performance_metrics.opportunities_per_day,
                "by_quality": self.performance_metrics.opportunities_by_quality,
            },
            "trades": {
                "total_simulated": self.performance_metrics.total_simulated_trades,
                "successful": self.performance_metrics.successful_executions,
                "failed": self.performance_metrics.failed_executions,
                "success_rate": f"{self.performance_metrics.execution_success_rate:.2%}",
            },
            "profitability": {
                "total_gross_profit": self.performance_metrics.total_gross_profit,
                "total_net_profit": self.performance_metrics.total_net_profit,
                "total_commissions": self.performance_metrics.total_commissions_paid,
                "total_slippage_cost": self.performance_metrics.total_slippage_cost,
                "avg_profit_per_trade": self.performance_metrics.avg_profit_per_trade,
                "median_profit_per_trade": self.performance_metrics.median_profit_per_trade,
            },
            "roi_metrics": {
                "avg_min_roi": f"{self.performance_metrics.avg_min_roi:.2f}%",
                "median_min_roi": f"{self.performance_metrics.median_min_roi:.2f}%",
                "best_min_roi": f"{self.performance_metrics.best_min_roi:.2f}%",
                "worst_min_roi": f"{self.performance_metrics.worst_min_roi:.2f}%",
                "roi_std_dev": f"{self.performance_metrics.roi_standard_deviation:.2f}%",
            },
            "risk_metrics": {
                "max_single_loss": self.performance_metrics.max_single_trade_loss,
                "max_drawdown": self.performance_metrics.max_drawdown,
                "max_drawdown_percent": f"{self.performance_metrics.max_drawdown_percent:.2f}%",
                "sharpe_ratio": self.performance_metrics.sharpe_ratio,
            },
            "symbol_coverage": list(self.underlying_ids.keys()),
            "rejections": {"total": len(self.rejections)},
        }

    async def generate_performance_report(self, backtest_run_id: int) -> Dict[str, Any]:
        """Generate detailed performance report from stored results."""
        async with self.db_pool.acquire() as conn:
            # Get summary performance data
            summary = await conn.fetchrow(
                "SELECT * FROM get_sfr_performance_summary($1)", backtest_run_id
            )

            # Get VIX performance breakdown
            vix_performance = await conn.fetch(
                "SELECT * FROM analyze_sfr_vix_performance($1)", backtest_run_id
            )

            # Get rejection analysis
            rejections = await conn.fetch(
                """
                SELECT rejection_stage, rejection_reason, COUNT(*) as count
                FROM sfr_rejection_log
                WHERE backtest_run_id = $1
                GROUP BY rejection_stage, rejection_reason
                ORDER BY count DESC
                """,
                backtest_run_id,
            )

            return {
                "summary": dict(summary) if summary else {},
                "vix_analysis": [dict(row) for row in vix_performance],
                "rejection_analysis": [dict(row) for row in rejections],
            }


# Example usage and configuration presets
class SFRBacktestConfigs:
    """Pre-defined configuration sets for different backtesting scenarios."""

    @classmethod
    def conservative_config(cls) -> SFRBacktestConfig:
        """Conservative configuration with tight risk controls."""
        return SFRBacktestConfig(
            profit_target=0.75,
            cost_limit=100.0,
            max_bid_ask_spread_call=10.0,
            max_bid_ask_spread_put=10.0,
            slippage_model=SlippageModel.LINEAR,
            base_slippage_bps=3,
            commission_per_contract=1.50,
        )

    @classmethod
    def aggressive_config(cls) -> SFRBacktestConfig:
        """Aggressive configuration for higher volume trading."""
        return SFRBacktestConfig(
            profit_target=0.25,
            cost_limit=200.0,
            max_bid_ask_spread_call=30.0,
            max_bid_ask_spread_put=30.0,
            slippage_model=SlippageModel.IMPACT,
            base_slippage_bps=4,
            commission_per_contract=0.75,
        )

    @classmethod
    def low_vix_config(cls) -> SFRBacktestConfig:
        """Configuration optimized for low VIX environments."""
        return SFRBacktestConfig(
            profit_target=0.50,
            cost_limit=150.0,
            vix_regime_filter=VixRegime.LOW,
            max_vix_level=20.0,
            exclude_vix_spikes=True,
        )

    @classmethod
    def high_vix_config(cls) -> SFRBacktestConfig:
        """Configuration optimized for high VIX environments."""
        return SFRBacktestConfig(
            profit_target=1.00,
            cost_limit=250.0,
            vix_regime_filter=VixRegime.HIGH,
            min_vix_level=25.0,
            slippage_model=SlippageModel.IMPACT,
            base_slippage_bps=5,
        )


if __name__ == "__main__":
    # Example usage
    import asyncpg

    from backtesting.infra.data_collection.config.config import DatabaseConfig

    async def run_example():
        db_config = DatabaseConfig()
        db_pool = await asyncpg.create_pool(db_config.connection_string)

        # Conservative 1-year backtest
        config = SFRBacktestConfigs.conservative_config()
        engine = SFRBacktestEngine(db_pool, config)

        results = await engine.run_backtest(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            symbols=["SPY", "QQQ", "AAPL", "MSFT"],
        )

        print("Backtesting Results:")
        print(f"Total Opportunities: {results['opportunities']['total_found']}")
        print(f"Total Trades: {results['trades']['total_simulated']}")
        print(f"Success Rate: {results['trades']['success_rate']}")
        print(f"Net Profit: ${results['profitability']['total_net_profit']:.2f}")
        print(f"Average ROI: {results['roi_metrics']['avg_min_roi']}")
        print(f"Sharpe Ratio: {results['risk_metrics']['sharpe_ratio']:.2f}")

        await db_pool.close()

    # Run example
    # asyncio.run(run_example())
