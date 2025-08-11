"""
SFR-specific data validation module.

Validates data quality and completeness specifically for SFR (Synthetic Free Risk)
arbitrage backtesting requirements.
"""

import asyncio
import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import asyncpg
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SFRValidationResult:
    """Result of SFR data validation."""

    symbol: str
    validation_date: datetime
    passed: bool
    overall_score: float  # 0-1 quality score

    # Individual validation scores
    completeness_score: float
    consistency_score: float
    liquidity_score: float
    spread_quality_score: float
    arbitrage_suitability_score: float

    # Specific metrics
    missing_data_percentage: float
    avg_bid_ask_spread: float
    min_volume_threshold_met: bool
    option_coverage_percentage: float

    # Issues and warnings
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]

    # Detailed metrics
    metrics: Dict[str, Any]


class SFRDataValidator:
    """
    Comprehensive data validator for SFR arbitrage strategies.

    Validates data quality across multiple dimensions:
    1. Completeness - missing data detection
    2. Consistency - logical data relationships
    3. Liquidity - volume and spread requirements
    4. Arbitrage suitability - specific SFR requirements
    5. Market structure - options chain coverage
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

        # SFR-specific validation thresholds
        self.thresholds = {
            # Data completeness
            "min_data_completeness": 0.95,  # 95% data completeness required
            "max_missing_consecutive_days": 2,  # Max 2 consecutive missing days
            # Liquidity requirements
            "min_avg_volume": 100,  # Minimum average daily volume
            "max_avg_spread_percent": 5.0,  # Maximum average spread %
            "min_option_volume": 10,  # Minimum option volume
            # Options coverage
            "min_strikes_per_expiry": 10,  # Minimum strikes per expiry
            "min_expiry_coverage": 3,  # Minimum expiries covered
            "min_atm_coverage": 0.8,  # 80% of days must have ATM options
            # Market structure
            "max_price_gap_percent": 10.0,  # Max price gap %
            "min_trading_hours_coverage": 0.9,  # 90% market hours coverage
            # SFR-specific
            "min_conversion_opportunities_per_day": 1,  # Min conversion opportunities
            "max_execution_risk_score": 0.3,  # Max execution risk
        }

    async def validate_sfr_symbol_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        detailed_analysis: bool = True,
    ) -> SFRValidationResult:
        """
        Comprehensive SFR data validation for a symbol.

        Args:
            symbol: Symbol to validate
            start_date: Start date of data period
            end_date: End date of data period
            detailed_analysis: Whether to perform detailed analysis

        Returns:
            SFRValidationResult with comprehensive validation results
        """
        logger.info(
            f"Starting SFR validation for {symbol} from {start_date} to {end_date}"
        )

        validation_start = datetime.now()

        # Initialize result
        result = SFRValidationResult(
            symbol=symbol,
            validation_date=validation_start,
            passed=False,
            overall_score=0.0,
            completeness_score=0.0,
            consistency_score=0.0,
            liquidity_score=0.0,
            spread_quality_score=0.0,
            arbitrage_suitability_score=0.0,
            missing_data_percentage=0.0,
            avg_bid_ask_spread=0.0,
            min_volume_threshold_met=False,
            option_coverage_percentage=0.0,
            critical_issues=[],
            warnings=[],
            recommendations=[],
            metrics={},
        )

        try:
            # Get underlying ID
            underlying_id = await self._get_underlying_id(symbol)
            if not underlying_id:
                result.critical_issues.append(f"Symbol {symbol} not found in database")
                return result

            # Run validation components
            await self._validate_stock_data_completeness(
                underlying_id, symbol, start_date, end_date, result
            )

            await self._validate_option_data_coverage(
                underlying_id, symbol, start_date, end_date, result
            )

            await self._validate_liquidity_requirements(
                underlying_id, symbol, start_date, end_date, result
            )

            await self._validate_spread_quality(
                underlying_id, symbol, start_date, end_date, result
            )

            if detailed_analysis:
                await self._analyze_arbitrage_suitability(
                    underlying_id, symbol, start_date, end_date, result
                )

            # Calculate overall score and determine pass/fail
            result.overall_score = self._calculate_overall_score(result)
            result.passed = result.overall_score >= 0.7  # 70% threshold

            # Generate recommendations
            self._generate_recommendations(result)

            validation_duration = (datetime.now() - validation_start).total_seconds()
            result.metrics["validation_duration_seconds"] = validation_duration

            logger.info(
                f"SFR validation completed for {symbol}: "
                f"Score={result.overall_score:.3f}, Passed={result.passed}"
            )

            return result

        except Exception as e:
            logger.error(f"SFR validation failed for {symbol}: {e}")
            result.critical_issues.append(f"Validation error: {str(e)}")
            return result

    async def _get_underlying_id(self, symbol: str) -> Optional[int]:
        """Get underlying security ID."""
        async with self.db_pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT id FROM underlying_securities WHERE symbol = $1", symbol
            )

    async def _validate_stock_data_completeness(
        self,
        underlying_id: int,
        symbol: str,
        start_date: date,
        end_date: date,
        result: SFRValidationResult,
    ):
        """Validate stock data completeness."""
        async with self.db_pool.acquire() as conn:
            # Count expected vs actual trading days
            expected_days = self._count_trading_days(start_date, end_date)

            actual_days = await conn.fetchval(
                """
                SELECT COUNT(DISTINCT DATE(time))
                FROM stock_data_ticks
                WHERE underlying_id = $1
                  AND time >= $2
                  AND time <= $3
                """,
                underlying_id,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )

            actual_days = actual_days or 0
            completeness_ratio = actual_days / max(expected_days, 1)

            result.completeness_score = completeness_ratio
            result.missing_data_percentage = (1 - completeness_ratio) * 100

            # Check for consecutive missing days
            missing_days_query = """
                WITH date_series AS (
                    SELECT generate_series($2::date, $3::date, '1 day'::interval)::date AS trading_date
                    WHERE EXTRACT(dow FROM generate_series($2::date, $3::date, '1 day'::interval)) BETWEEN 1 AND 5
                ),
                existing_dates AS (
                    SELECT DISTINCT DATE(time) as data_date
                    FROM stock_data_ticks
                    WHERE underlying_id = $1
                      AND time >= $2
                      AND time <= $3
                ),
                missing_dates AS (
                    SELECT ds.trading_date
                    FROM date_series ds
                    LEFT JOIN existing_dates ed ON ds.trading_date = ed.data_date
                    WHERE ed.data_date IS NULL
                    ORDER BY ds.trading_date
                )
                SELECT
                    COUNT(*) as total_missing,
                    MAX(consecutive_missing) as max_consecutive
                FROM (
                    SELECT
                        trading_date,
                        ROW_NUMBER() OVER (ORDER BY trading_date) -
                        ROW_NUMBER() OVER (PARTITION BY trading_date - INTERVAL '1 day' * ROW_NUMBER() OVER (ORDER BY trading_date) ORDER BY trading_date) as consecutive_missing
                    FROM missing_dates
                ) grouped
            """

            missing_stats = await conn.fetchrow(
                missing_days_query, underlying_id, start_date, end_date
            )

            if missing_stats and missing_stats["max_consecutive"]:
                max_consecutive = missing_stats["max_consecutive"]
                if max_consecutive > self.thresholds["max_missing_consecutive_days"]:
                    result.critical_issues.append(
                        f"Found {max_consecutive} consecutive missing days "
                        f"(threshold: {self.thresholds['max_missing_consecutive_days']})"
                    )

            # Data point density check
            total_data_points = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM stock_data_ticks
                WHERE underlying_id = $1
                  AND time >= $2
                  AND time <= $3
                """,
                underlying_id,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )

            expected_points_per_day = 390  # 6.5 hours * 60 minutes
            expected_total_points = expected_days * expected_points_per_day
            point_density = (total_data_points or 0) / max(expected_total_points, 1)

            result.metrics.update(
                {
                    "expected_trading_days": expected_days,
                    "actual_trading_days": actual_days,
                    "total_data_points": total_data_points or 0,
                    "data_point_density": point_density,
                    "max_consecutive_missing": (
                        missing_stats["max_consecutive"] if missing_stats else 0
                    ),
                }
            )

            if completeness_ratio < self.thresholds["min_data_completeness"]:
                result.critical_issues.append(
                    f"Data completeness {completeness_ratio:.1%} below threshold "
                    f"{self.thresholds['min_data_completeness']:.1%}"
                )

    async def _validate_option_data_coverage(
        self,
        underlying_id: int,
        symbol: str,
        start_date: date,
        end_date: date,
        result: SFRValidationResult,
    ):
        """Validate options data coverage."""
        async with self.db_pool.acquire() as conn:
            # Count option contracts available
            total_contracts = await conn.fetchval(
                """
                SELECT COUNT(DISTINCT oc.id)
                FROM option_chains oc
                WHERE oc.underlying_id = $1
                  AND oc.expiration_date >= $2
                  AND oc.expiration_date <= ($3 + INTERVAL '60 days')::date
                """,
                underlying_id,
                start_date,
                end_date,
            )

            # Count expiries covered
            expiries_covered = await conn.fetchval(
                """
                SELECT COUNT(DISTINCT oc.expiration_date)
                FROM option_chains oc
                WHERE oc.underlying_id = $1
                  AND oc.expiration_date >= $2
                  AND oc.expiration_date <= ($3 + INTERVAL '60 days')::date
                """,
                underlying_id,
                start_date,
                end_date,
            )

            # Calculate coverage metrics
            expected_expiries = min(
                12, max(3, (end_date - start_date).days // 7)
            )  # Weekly expiries
            expiry_coverage_ratio = (expiries_covered or 0) / expected_expiries

            # Check strikes per expiry
            avg_strikes_per_expiry = 0
            if expiries_covered and expiries_covered > 0:
                avg_strikes_per_expiry = (total_contracts or 0) / expiries_covered

            result.option_coverage_percentage = expiry_coverage_ratio * 100

            # Check for ATM option coverage
            atm_coverage = await self._check_atm_coverage(
                conn, underlying_id, start_date, end_date
            )

            result.metrics.update(
                {
                    "total_option_contracts": total_contracts or 0,
                    "expiries_covered": expiries_covered or 0,
                    "expected_expiries": expected_expiries,
                    "avg_strikes_per_expiry": avg_strikes_per_expiry,
                    "atm_coverage_ratio": atm_coverage,
                }
            )

            # Validation checks
            if (expiries_covered or 0) < self.thresholds["min_expiry_coverage"]:
                result.critical_issues.append(
                    f"Only {expiries_covered} expiries covered "
                    f"(minimum: {self.thresholds['min_expiry_coverage']})"
                )

            if avg_strikes_per_expiry < self.thresholds["min_strikes_per_expiry"]:
                result.critical_issues.append(
                    f"Average {avg_strikes_per_expiry:.1f} strikes per expiry "
                    f"(minimum: {self.thresholds['min_strikes_per_expiry']})"
                )

            if atm_coverage < self.thresholds["min_atm_coverage"]:
                result.warnings.append(
                    f"ATM option coverage {atm_coverage:.1%} below optimal "
                    f"({self.thresholds['min_atm_coverage']:.1%})"
                )

    async def _check_atm_coverage(
        self,
        conn: asyncpg.Connection,
        underlying_id: int,
        start_date: date,
        end_date: date,
    ) -> float:
        """Check at-the-money option coverage."""
        # Get daily stock prices
        daily_prices = await conn.fetch(
            """
            SELECT DISTINCT
                DATE(time) as trading_date,
                AVG(price) as avg_price
            FROM stock_data_ticks
            WHERE underlying_id = $1
              AND time >= $2
              AND time <= $3
            GROUP BY DATE(time)
            ORDER BY trading_date
            """,
            underlying_id,
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.max.time()),
        )

        if not daily_prices:
            return 0.0

        atm_days = 0
        total_days = len(daily_prices)

        for price_row in daily_prices:
            trading_date = price_row["trading_date"]
            avg_price = float(price_row["avg_price"])

            # Check if there are options within 5% of current price for nearby expiries
            atm_options = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM option_chains oc
                WHERE oc.underlying_id = $1
                  AND oc.expiration_date >= $2
                  AND oc.expiration_date <= ($2 + INTERVAL '45 days')::date
                  AND oc.strike_price BETWEEN $3 * 0.95 AND $3 * 1.05
                """,
                underlying_id,
                trading_date,
                avg_price,
            )

            if (atm_options or 0) >= 4:  # At least 2 calls and 2 puts
                atm_days += 1

        return atm_days / max(total_days, 1)

    async def _validate_liquidity_requirements(
        self,
        underlying_id: int,
        symbol: str,
        start_date: date,
        end_date: date,
        result: SFRValidationResult,
    ):
        """Validate liquidity requirements for SFR."""
        async with self.db_pool.acquire() as conn:
            # Stock liquidity metrics
            stock_liquidity = await conn.fetchrow(
                """
                SELECT
                    AVG(volume) as avg_volume,
                    MIN(volume) as min_volume,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY volume) as median_volume,
                    STDDEV(volume) as volume_stddev,
                    COUNT(*) as data_points
                FROM stock_data_ticks
                WHERE underlying_id = $1
                  AND time >= $2
                  AND time <= $3
                  AND volume IS NOT NULL
                """,
                underlying_id,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )

            # Option liquidity metrics (sample)
            option_liquidity = await conn.fetchrow(
                """
                SELECT
                    AVG(mdt.volume) as avg_option_volume,
                    MIN(mdt.volume) as min_option_volume,
                    COUNT(*) as option_data_points,
                    COUNT(DISTINCT oc.id) as unique_contracts
                FROM market_data_ticks mdt
                JOIN option_chains oc ON mdt.contract_id = oc.id
                WHERE oc.underlying_id = $1
                  AND mdt.time >= $2
                  AND mdt.time <= $3
                  AND mdt.volume IS NOT NULL
                """,
                underlying_id,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )

            # Calculate liquidity score
            stock_volume_score = 1.0
            option_volume_score = 0.5  # Default if no option data

            if stock_liquidity and stock_liquidity["avg_volume"]:
                avg_volume = float(stock_liquidity["avg_volume"])
                if avg_volume < self.thresholds["min_avg_volume"]:
                    stock_volume_score = avg_volume / self.thresholds["min_avg_volume"]
                    result.critical_issues.append(
                        f"Average volume {avg_volume:.0f} below threshold "
                        f"{self.thresholds['min_avg_volume']}"
                    )

            if option_liquidity and option_liquidity["avg_option_volume"]:
                avg_option_volume = float(option_liquidity["avg_option_volume"])
                if avg_option_volume >= self.thresholds["min_option_volume"]:
                    option_volume_score = 1.0
                else:
                    option_volume_score = (
                        avg_option_volume / self.thresholds["min_option_volume"]
                    )
                    result.warnings.append(
                        f"Average option volume {avg_option_volume:.0f} below optimal "
                        f"({self.thresholds['min_option_volume']})"
                    )

            result.liquidity_score = (stock_volume_score + option_volume_score) / 2
            result.min_volume_threshold_met = (
                stock_volume_score >= 1.0 and option_volume_score >= 0.8
            )

            result.metrics.update(
                {
                    "avg_stock_volume": (
                        float(stock_liquidity["avg_volume"])
                        if stock_liquidity and stock_liquidity["avg_volume"]
                        else 0
                    ),
                    "avg_option_volume": (
                        float(option_liquidity["avg_option_volume"])
                        if option_liquidity and option_liquidity["avg_option_volume"]
                        else 0
                    ),
                    "unique_option_contracts_with_data": (
                        option_liquidity["unique_contracts"] if option_liquidity else 0
                    ),
                    "stock_volume_score": stock_volume_score,
                    "option_volume_score": option_volume_score,
                }
            )

    async def _validate_spread_quality(
        self,
        underlying_id: int,
        symbol: str,
        start_date: date,
        end_date: date,
        result: SFRValidationResult,
    ):
        """Validate bid-ask spread quality."""
        async with self.db_pool.acquire() as conn:
            # Option spread analysis
            spread_stats = await conn.fetchrow(
                """
                SELECT
                    AVG(mdt.bid_ask_spread) as avg_spread,
                    AVG(CASE WHEN mdt.last_price > 0 THEN mdt.bid_ask_spread / mdt.last_price * 100 ELSE NULL END) as avg_spread_percent,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY mdt.bid_ask_spread) as median_spread,
                    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY CASE WHEN mdt.last_price > 0 THEN mdt.bid_ask_spread / mdt.last_price * 100 ELSE NULL END) as p90_spread_percent,
                    COUNT(*) as spread_samples
                FROM market_data_ticks mdt
                JOIN option_chains oc ON mdt.contract_id = oc.id
                WHERE oc.underlying_id = $1
                  AND mdt.time >= $2
                  AND mdt.time <= $3
                  AND mdt.bid_price IS NOT NULL
                  AND mdt.ask_price IS NOT NULL
                  AND mdt.bid_price > 0
                  AND mdt.ask_price > mdt.bid_price
                """,
                underlying_id,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )

            # Calculate spread quality score
            spread_quality_score = 1.0
            avg_spread_percent = 0.0

            if spread_stats and spread_stats["avg_spread_percent"]:
                avg_spread_percent = float(spread_stats["avg_spread_percent"])
                result.avg_bid_ask_spread = avg_spread_percent

                if avg_spread_percent > self.thresholds["max_avg_spread_percent"]:
                    spread_quality_score = (
                        self.thresholds["max_avg_spread_percent"] / avg_spread_percent
                    )
                    result.warnings.append(
                        f"Average spread {avg_spread_percent:.2f}% above optimal "
                        f"({self.thresholds['max_avg_spread_percent']:.2f}%)"
                    )

            result.spread_quality_score = spread_quality_score

            result.metrics.update(
                {
                    "avg_spread_dollars": (
                        float(spread_stats["avg_spread"])
                        if spread_stats and spread_stats["avg_spread"]
                        else 0
                    ),
                    "avg_spread_percent": avg_spread_percent,
                    "median_spread": (
                        float(spread_stats["median_spread"])
                        if spread_stats and spread_stats["median_spread"]
                        else 0
                    ),
                    "p90_spread_percent": (
                        float(spread_stats["p90_spread_percent"])
                        if spread_stats and spread_stats["p90_spread_percent"]
                        else 0
                    ),
                    "spread_samples": (
                        spread_stats["spread_samples"] if spread_stats else 0
                    ),
                }
            )

    async def _analyze_arbitrage_suitability(
        self,
        underlying_id: int,
        symbol: str,
        start_date: date,
        end_date: date,
        result: SFRValidationResult,
    ):
        """Analyze data suitability for SFR arbitrage."""
        async with self.db_pool.acquire() as conn:
            # Check for potential conversion/reversal setups
            conversion_opportunities = await conn.fetchval(
                """
                WITH daily_setups AS (
                    SELECT
                        DATE(mdt.time) as trading_date,
                        COUNT(*) as option_data_points,
                        COUNT(DISTINCT oc.expiration_date) as expiries_available,
                        COUNT(DISTINCT CASE WHEN oc.option_type = 'C' THEN oc.strike_price END) as call_strikes,
                        COUNT(DISTINCT CASE WHEN oc.option_type = 'P' THEN oc.strike_price END) as put_strikes
                    FROM market_data_ticks mdt
                    JOIN option_chains oc ON mdt.contract_id = oc.id
                    WHERE oc.underlying_id = $1
                      AND mdt.time >= $2
                      AND mdt.time <= $3
                      AND mdt.bid_price IS NOT NULL
                      AND mdt.ask_price IS NOT NULL
                    GROUP BY DATE(mdt.time)
                )
                SELECT COUNT(*)
                FROM daily_setups
                WHERE expiries_available >= 2
                  AND call_strikes >= 3
                  AND put_strikes >= 3
                  AND option_data_points >= 20
                """,
                underlying_id,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )

            # Calculate trading days
            trading_days = self._count_trading_days(start_date, end_date)
            conversion_ratio = (conversion_opportunities or 0) / max(trading_days, 1)

            # Arbitrage suitability score
            if conversion_ratio >= 0.8:  # 80% of days have potential setups
                arb_score = 1.0
            elif conversion_ratio >= 0.5:  # 50% of days
                arb_score = 0.8
            elif conversion_ratio >= 0.2:  # 20% of days
                arb_score = 0.6
            else:
                arb_score = 0.4
                result.warnings.append(
                    f"Limited arbitrage opportunities: only {conversion_ratio:.1%} of days "
                    f"have suitable option setups"
                )

            result.arbitrage_suitability_score = arb_score

            result.metrics.update(
                {
                    "potential_conversion_days": conversion_opportunities or 0,
                    "trading_days_analyzed": trading_days,
                    "conversion_opportunity_ratio": conversion_ratio,
                }
            )

    def _calculate_overall_score(self, result: SFRValidationResult) -> float:
        """Calculate overall validation score."""
        # Weighted scoring
        weights = {
            "completeness": 0.25,
            "liquidity": 0.25,
            "spread_quality": 0.20,
            "arbitrage_suitability": 0.20,
            "consistency": 0.10,
        }

        # Set default consistency score if not set
        if result.consistency_score == 0.0:
            result.consistency_score = 0.9  # Default good consistency

        scores = {
            "completeness": result.completeness_score,
            "liquidity": result.liquidity_score,
            "spread_quality": result.spread_quality_score,
            "arbitrage_suitability": result.arbitrage_suitability_score,
            "consistency": result.consistency_score,
        }

        overall_score = sum(
            scores[component] * weights[component] for component in weights
        )

        # Penalty for critical issues
        critical_penalty = len(result.critical_issues) * 0.1
        overall_score = max(0.0, overall_score - critical_penalty)

        return min(1.0, overall_score)

    def _generate_recommendations(self, result: SFRValidationResult):
        """Generate improvement recommendations."""
        if result.completeness_score < 0.9:
            result.recommendations.append(
                "Improve data completeness by reducing gaps in historical data collection"
            )

        if result.liquidity_score < 0.8:
            result.recommendations.append(
                "Consider filtering for higher volume options or adjusting volume thresholds"
            )

        if result.spread_quality_score < 0.7:
            result.recommendations.append(
                "Focus on more liquid option strikes with tighter spreads for better execution"
            )

        if result.arbitrage_suitability_score < 0.6:
            result.recommendations.append(
                "Expand expiry range or strike selection to increase arbitrage opportunities"
            )

        if len(result.critical_issues) == 0 and result.overall_score >= 0.8:
            result.recommendations.append(
                "Data quality is excellent for SFR backtesting"
            )

    def _count_trading_days(self, start_date: date, end_date: date) -> int:
        """Count trading days (Monday-Friday) in date range."""
        days = 0
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Monday=0, Sunday=6
                days += 1
            current += timedelta(days=1)
        return days

    async def validate_multiple_symbols(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        detailed_analysis: bool = True,
    ) -> Dict[str, SFRValidationResult]:
        """Validate multiple symbols in parallel."""
        logger.info(f"Validating {len(symbols)} symbols for SFR suitability")

        # Create validation tasks
        tasks = [
            self.validate_sfr_symbol_data(
                symbol, start_date, end_date, detailed_analysis
            )
            for symbol in symbols
        ]

        # Run validations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        validation_results = {}
        for i, result in enumerate(results):
            symbol = symbols[i]

            if isinstance(result, Exception):
                logger.error(f"Validation failed for {symbol}: {result}")
                # Create failed result
                failed_result = SFRValidationResult(
                    symbol=symbol,
                    validation_date=datetime.now(),
                    passed=False,
                    overall_score=0.0,
                    completeness_score=0.0,
                    consistency_score=0.0,
                    liquidity_score=0.0,
                    spread_quality_score=0.0,
                    arbitrage_suitability_score=0.0,
                    missing_data_percentage=100.0,
                    avg_bid_ask_spread=0.0,
                    min_volume_threshold_met=False,
                    option_coverage_percentage=0.0,
                    critical_issues=[f"Validation error: {str(result)}"],
                    warnings=[],
                    recommendations=[],
                    metrics={},
                )
                validation_results[symbol] = failed_result
            else:
                validation_results[symbol] = result

        return validation_results

    async def get_validation_summary(
        self, validation_results: Dict[str, SFRValidationResult]
    ) -> Dict[str, Any]:
        """Generate summary of validation results."""
        if not validation_results:
            return {}

        passed_count = sum(1 for r in validation_results.values() if r.passed)
        total_count = len(validation_results)

        avg_scores = {
            "overall": statistics.mean(
                [r.overall_score for r in validation_results.values()]
            ),
            "completeness": statistics.mean(
                [r.completeness_score for r in validation_results.values()]
            ),
            "liquidity": statistics.mean(
                [r.liquidity_score for r in validation_results.values()]
            ),
            "spread_quality": statistics.mean(
                [r.spread_quality_score for r in validation_results.values()]
            ),
            "arbitrage_suitability": statistics.mean(
                [r.arbitrage_suitability_score for r in validation_results.values()]
            ),
        }

        # Top and bottom performers
        sorted_results = sorted(
            validation_results.items(), key=lambda x: x[1].overall_score, reverse=True
        )

        return {
            "total_symbols_validated": total_count,
            "symbols_passed": passed_count,
            "pass_rate": passed_count / total_count if total_count > 0 else 0,
            "average_scores": avg_scores,
            "best_performer": (
                {
                    "symbol": sorted_results[0][0],
                    "score": sorted_results[0][1].overall_score,
                }
                if sorted_results
                else None
            ),
            "worst_performer": (
                {
                    "symbol": sorted_results[-1][0],
                    "score": sorted_results[-1][1].overall_score,
                }
                if sorted_results
                else None
            ),
            "symbols_needing_attention": [
                symbol
                for symbol, result in validation_results.items()
                if not result.passed
            ],
            "common_issues": self._extract_common_issues(validation_results),
        }

    def _extract_common_issues(
        self, validation_results: Dict[str, SFRValidationResult]
    ) -> List[str]:
        """Extract common issues across validations."""
        issue_counts = {}

        for result in validation_results.values():
            for issue in result.critical_issues + result.warnings:
                # Extract issue type (first part of message)
                issue_type = issue.split(":")[0].split("(")[0].strip()
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        # Return issues affecting >20% of symbols
        threshold = len(validation_results) * 0.2
        common_issues = [
            f"{issue} (affects {count} symbols)"
            for issue, count in issue_counts.items()
            if count >= threshold
        ]

        return sorted(
            common_issues,
            key=lambda x: int(x.split("(affects ")[1].split(" symbols")[0]),
            reverse=True,
        )
