"""
Data quality metrics and reporting for historical data collection.

Provides comprehensive data quality assessment and reporting functionality
for backtesting data validation and monitoring.
"""

import json
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncpg
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityMetric:
    """Individual quality metric."""

    name: str
    value: float
    threshold: float
    passed: bool
    weight: float = 1.0
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""

    # Report metadata
    report_id: str
    symbol: str
    data_type: str  # 'stock', 'option', 'mixed'
    period_start: date
    period_end: date
    generated_at: datetime

    # Overall assessment
    overall_quality_score: float
    quality_grade: str  # A, B, C, D, F
    is_suitable_for_backtesting: bool

    # Individual metrics
    completeness_metrics: List[QualityMetric] = field(default_factory=list)
    consistency_metrics: List[QualityMetric] = field(default_factory=list)
    accuracy_metrics: List[QualityMetric] = field(default_factory=list)
    timeliness_metrics: List[QualityMetric] = field(default_factory=list)
    liquidity_metrics: List[QualityMetric] = field(default_factory=list)

    # Summary statistics
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    missing_records: int = 0

    # Issues and recommendations
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Detailed analysis
    time_series_analysis: Dict[str, Any] = field(default_factory=dict)
    statistical_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "report_id": self.report_id,
            "symbol": self.symbol,
            "data_type": self.data_type,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "overall_quality_score": self.overall_quality_score,
            "quality_grade": self.quality_grade,
            "is_suitable_for_backtesting": self.is_suitable_for_backtesting,
            "completeness_metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "passed": m.passed,
                    "weight": m.weight,
                    "description": m.description,
                    "details": m.details,
                }
                for m in self.completeness_metrics
            ],
            "consistency_metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "passed": m.passed,
                    "weight": m.weight,
                    "description": m.description,
                    "details": m.details,
                }
                for m in self.consistency_metrics
            ],
            "accuracy_metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "passed": m.passed,
                    "weight": m.weight,
                    "description": m.description,
                    "details": m.details,
                }
                for m in self.accuracy_metrics
            ],
            "timeliness_metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "passed": m.passed,
                    "weight": m.weight,
                    "description": m.description,
                    "details": m.details,
                }
                for m in self.timeliness_metrics
            ],
            "liquidity_metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "passed": m.passed,
                    "weight": m.weight,
                    "description": m.description,
                    "details": m.details,
                }
                for m in self.liquidity_metrics
            ],
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "missing_records": self.missing_records,
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "time_series_analysis": self.time_series_analysis,
            "statistical_summary": self.statistical_summary,
        }


class DataQualityMetrics:
    """
    Comprehensive data quality metrics calculator and reporter.

    Implements industry-standard data quality dimensions:
    1. Completeness - Missing data assessment
    2. Consistency - Logical consistency validation
    3. Accuracy - Data correctness and outlier detection
    4. Timeliness - Data freshness and temporal consistency
    5. Liquidity - Market liquidity and execution quality
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

        # Quality thresholds for different grades
        self.grade_thresholds = {
            "A": 0.9,  # Excellent quality
            "B": 0.8,  # Good quality
            "C": 0.7,  # Acceptable quality
            "D": 0.6,  # Poor quality
            "F": 0.0,  # Failing quality
        }

        # Backtesting suitability threshold
        self.backtesting_threshold = 0.75

        # Metric weights for overall score calculation
        self.metric_weights = {
            "completeness": 0.25,
            "consistency": 0.20,
            "accuracy": 0.20,
            "timeliness": 0.15,
            "liquidity": 0.20,
        }

    async def generate_comprehensive_quality_report(
        self,
        symbol: str,
        data_type: str,
        start_date: date,
        end_date: date,
        detailed_analysis: bool = True,
    ) -> DataQualityReport:
        """
        Generate comprehensive data quality report.

        Args:
            symbol: Symbol to analyze
            data_type: Type of data ('stock', 'option', 'mixed')
            start_date: Analysis start date
            end_date: Analysis end date
            detailed_analysis: Whether to perform detailed analysis

        Returns:
            DataQualityReport with comprehensive quality assessment
        """
        report_id = f"{symbol}_{data_type}_{start_date}_{end_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Generating quality report {report_id}")

        # Initialize report
        report = DataQualityReport(
            report_id=report_id,
            symbol=symbol,
            data_type=data_type,
            period_start=start_date,
            period_end=end_date,
            generated_at=datetime.now(),
            overall_quality_score=0.0,
            quality_grade="F",
            is_suitable_for_backtesting=False,
        )

        try:
            # Get underlying ID
            underlying_id = await self._get_underlying_id(symbol)
            if not underlying_id:
                report.critical_issues.append(f"Symbol {symbol} not found")
                return report

            # Generate completeness metrics
            await self._calculate_completeness_metrics(
                underlying_id, symbol, start_date, end_date, data_type, report
            )

            # Generate consistency metrics
            await self._calculate_consistency_metrics(
                underlying_id, symbol, start_date, end_date, data_type, report
            )

            # Generate accuracy metrics
            await self._calculate_accuracy_metrics(
                underlying_id, symbol, start_date, end_date, data_type, report
            )

            # Generate timeliness metrics
            await self._calculate_timeliness_metrics(
                underlying_id, symbol, start_date, end_date, data_type, report
            )

            # Generate liquidity metrics
            await self._calculate_liquidity_metrics(
                underlying_id, symbol, start_date, end_date, data_type, report
            )

            if detailed_analysis:
                # Time series analysis
                await self._perform_time_series_analysis(
                    underlying_id, symbol, start_date, end_date, report
                )

                # Statistical summary
                await self._generate_statistical_summary(
                    underlying_id, symbol, start_date, end_date, report
                )

            # Calculate overall quality score and grade
            self._calculate_overall_quality_score(report)
            self._assign_quality_grade(report)
            self._assess_backtesting_suitability(report)

            # Generate recommendations
            self._generate_quality_recommendations(report)

            logger.info(
                f"Quality report completed: {symbol} scored {report.overall_quality_score:.3f} "
                f"(Grade {report.quality_grade})"
            )

            return report

        except Exception as e:
            logger.error(f"Error generating quality report for {symbol}: {e}")
            report.critical_issues.append(f"Report generation error: {str(e)}")
            return report

    async def _get_underlying_id(self, symbol: str) -> Optional[int]:
        """Get underlying security ID."""
        async with self.db_pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT id FROM underlying_securities WHERE symbol = $1", symbol
            )

    async def _calculate_completeness_metrics(
        self,
        underlying_id: int,
        symbol: str,
        start_date: date,
        end_date: date,
        data_type: str,
        report: DataQualityReport,
    ):
        """Calculate data completeness metrics."""
        async with self.db_pool.acquire() as conn:
            # Expected vs actual trading days
            expected_days = self._count_trading_days(start_date, end_date)

            # Stock data completeness
            stock_days = await conn.fetchval(
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

            stock_completeness = (stock_days or 0) / max(expected_days, 1)

            report.completeness_metrics.append(
                QualityMetric(
                    name="stock_data_completeness",
                    value=stock_completeness,
                    threshold=0.95,
                    passed=stock_completeness >= 0.95,
                    weight=1.0,
                    description=f"Stock data available for {stock_days}/{expected_days} trading days",
                    details={"actual_days": stock_days, "expected_days": expected_days},
                )
            )

            # Total record count
            stock_records = await conn.fetchval(
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

            report.total_records += stock_records or 0

            # Option data completeness (if applicable)
            if data_type in ["option", "mixed"]:
                option_contracts = await conn.fetchval(
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

                option_records = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM market_data_ticks mdt
                    JOIN option_chains oc ON mdt.contract_id = oc.id
                    WHERE oc.underlying_id = $1
                      AND mdt.time >= $2
                      AND mdt.time <= $3
                    """,
                    underlying_id,
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.max.time()),
                )

                option_coverage = (option_records or 0) / max(option_contracts or 1, 1)

                report.completeness_metrics.append(
                    QualityMetric(
                        name="option_data_coverage",
                        value=option_coverage,
                        threshold=0.5,  # Lower threshold for options
                        passed=option_coverage >= 0.5,
                        weight=0.8,
                        description=f"Option data coverage: {option_records} records for {option_contracts} contracts",
                        details={
                            "option_records": option_records,
                            "option_contracts": option_contracts,
                        },
                    )
                )

                report.total_records += option_records or 0

            # Data density metric (records per trading day)
            if stock_days and stock_days > 0:
                records_per_day = (stock_records or 0) / stock_days
                expected_records_per_day = 390  # 6.5 hours * 60 minutes

                density_ratio = records_per_day / expected_records_per_day

                report.completeness_metrics.append(
                    QualityMetric(
                        name="data_density",
                        value=density_ratio,
                        threshold=0.8,
                        passed=density_ratio >= 0.8,
                        weight=0.7,
                        description=f"Average {records_per_day:.0f} records per day",
                        details={
                            "records_per_day": records_per_day,
                            "expected_records_per_day": expected_records_per_day,
                        },
                    )
                )

    async def _calculate_consistency_metrics(
        self,
        underlying_id: int,
        symbol: str,
        start_date: date,
        end_date: date,
        data_type: str,
        report: DataQualityReport,
    ):
        """Calculate data consistency metrics."""
        async with self.db_pool.acquire() as conn:
            # Price consistency - OHLC relationships
            inconsistent_ohlc = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM stock_data_ticks
                WHERE underlying_id = $1
                  AND time >= $2
                  AND time <= $3
                  AND open_price IS NOT NULL
                  AND high_price IS NOT NULL
                  AND low_price IS NOT NULL
                  AND close_price IS NOT NULL
                  AND NOT (
                    low_price <= open_price AND open_price <= high_price
                    AND low_price <= close_price AND close_price <= high_price
                  )
                """,
                underlying_id,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )

            total_ohlc_records = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM stock_data_ticks
                WHERE underlying_id = $1
                  AND time >= $2
                  AND time <= $3
                  AND open_price IS NOT NULL
                  AND high_price IS NOT NULL
                  AND low_price IS NOT NULL
                  AND close_price IS NOT NULL
                """,
                underlying_id,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )

            if total_ohlc_records and total_ohlc_records > 0:
                ohlc_consistency_ratio = 1.0 - (
                    (inconsistent_ohlc or 0) / total_ohlc_records
                )

                report.consistency_metrics.append(
                    QualityMetric(
                        name="ohlc_consistency",
                        value=ohlc_consistency_ratio,
                        threshold=0.99,
                        passed=ohlc_consistency_ratio >= 0.99,
                        weight=1.0,
                        description=f"OHLC consistency: {inconsistent_ohlc}/{total_ohlc_records} violations",
                        details={
                            "violations": inconsistent_ohlc,
                            "total_records": total_ohlc_records,
                        },
                    )
                )

            # Option bid-ask consistency
            if data_type in ["option", "mixed"]:
                inconsistent_bid_ask = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM market_data_ticks mdt
                    JOIN option_chains oc ON mdt.contract_id = oc.id
                    WHERE oc.underlying_id = $1
                      AND mdt.time >= $2
                      AND mdt.time <= $3
                      AND mdt.bid_price IS NOT NULL
                      AND mdt.ask_price IS NOT NULL
                      AND mdt.bid_price > mdt.ask_price
                    """,
                    underlying_id,
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.max.time()),
                )

                total_bid_ask_records = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM market_data_ticks mdt
                    JOIN option_chains oc ON mdt.contract_id = oc.id
                    WHERE oc.underlying_id = $1
                      AND mdt.time >= $2
                      AND mdt.time <= $3
                      AND mdt.bid_price IS NOT NULL
                      AND mdt.ask_price IS NOT NULL
                    """,
                    underlying_id,
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.max.time()),
                )

                if total_bid_ask_records and total_bid_ask_records > 0:
                    bid_ask_consistency = 1.0 - (
                        (inconsistent_bid_ask or 0) / total_bid_ask_records
                    )

                    report.consistency_metrics.append(
                        QualityMetric(
                            name="bid_ask_consistency",
                            value=bid_ask_consistency,
                            threshold=0.99,
                            passed=bid_ask_consistency >= 0.99,
                            weight=1.0,
                            description=f"Bid-ask consistency: {inconsistent_bid_ask}/{total_bid_ask_records} violations",
                            details={
                                "violations": inconsistent_bid_ask,
                                "total_records": total_bid_ask_records,
                            },
                        )
                    )

    async def _calculate_accuracy_metrics(
        self,
        underlying_id: int,
        symbol: str,
        start_date: date,
        end_date: date,
        data_type: str,
        report: DataQualityReport,
    ):
        """Calculate data accuracy metrics."""
        async with self.db_pool.acquire() as conn:
            # Price outlier detection
            price_stats = await conn.fetchrow(
                """
                SELECT
                    AVG(price) as mean_price,
                    STDDEV(price) as std_price,
                    COUNT(*) as total_prices
                FROM stock_data_ticks
                WHERE underlying_id = $1
                  AND time >= $2
                  AND time <= $3
                  AND price IS NOT NULL
                """,
                underlying_id,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )

            if price_stats and price_stats["total_prices"] > 10:
                mean_price = float(price_stats["mean_price"])
                std_price = float(price_stats["std_price"] or 0)

                # Count price outliers (more than 4 std deviations)
                price_outliers = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM stock_data_ticks
                    WHERE underlying_id = $1
                      AND time >= $2
                      AND time <= $3
                      AND price IS NOT NULL
                      AND ABS(price - $4) > $5 * 4
                    """,
                    underlying_id,
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.max.time()),
                    mean_price,
                    std_price,
                )

                total_prices = int(price_stats["total_prices"])
                outlier_ratio = (price_outliers or 0) / total_prices
                accuracy_score = 1.0 - outlier_ratio

                report.accuracy_metrics.append(
                    QualityMetric(
                        name="price_accuracy",
                        value=accuracy_score,
                        threshold=0.98,
                        passed=accuracy_score >= 0.98,
                        weight=1.0,
                        description=f"Price accuracy: {price_outliers}/{total_prices} outliers",
                        details={
                            "outliers": price_outliers,
                            "total_prices": total_prices,
                            "mean_price": mean_price,
                            "std_price": std_price,
                        },
                    )
                )

            # Volume reasonableness check
            volume_stats = await conn.fetchrow(
                """
                SELECT
                    AVG(volume) as mean_volume,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY volume) as median_volume,
                    COUNT(*) as total_volumes
                FROM stock_data_ticks
                WHERE underlying_id = $1
                  AND time >= $2
                  AND time <= $3
                  AND volume IS NOT NULL
                  AND volume > 0
                """,
                underlying_id,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )

            if volume_stats and volume_stats["total_volumes"] > 0:
                median_volume = float(volume_stats["median_volume"] or 0)

                # Count extreme volume outliers (more than 50x median)
                volume_outliers = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM stock_data_ticks
                    WHERE underlying_id = $1
                      AND time >= $2
                      AND time <= $3
                      AND volume IS NOT NULL
                      AND volume > $4 * 50
                    """,
                    underlying_id,
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.max.time()),
                    median_volume,
                )

                total_volumes = int(volume_stats["total_volumes"])
                volume_accuracy = 1.0 - ((volume_outliers or 0) / total_volumes)

                report.accuracy_metrics.append(
                    QualityMetric(
                        name="volume_accuracy",
                        value=volume_accuracy,
                        threshold=0.95,
                        passed=volume_accuracy >= 0.95,
                        weight=0.8,
                        description=f"Volume accuracy: {volume_outliers}/{total_volumes} extreme outliers",
                        details={
                            "outliers": volume_outliers,
                            "total_volumes": total_volumes,
                            "median_volume": median_volume,
                        },
                    )
                )

    async def _calculate_timeliness_metrics(
        self,
        underlying_id: int,
        symbol: str,
        start_date: date,
        end_date: date,
        data_type: str,
        report: DataQualityReport,
    ):
        """Calculate data timeliness metrics."""
        async with self.db_pool.acquire() as conn:
            # Temporal gap analysis
            gap_analysis = await conn.fetchrow(
                """
                WITH time_gaps AS (
                    SELECT
                        time,
                        LAG(time) OVER (ORDER BY time) as prev_time,
                        EXTRACT(EPOCH FROM (time - LAG(time) OVER (ORDER BY time))) / 60 as gap_minutes
                    FROM stock_data_ticks
                    WHERE underlying_id = $1
                      AND time >= $2
                      AND time <= $3
                    ORDER BY time
                )
                SELECT
                    COUNT(*) as total_gaps,
                    COUNT(*) FILTER (WHERE gap_minutes > 60) as large_gaps,
                    AVG(gap_minutes) as avg_gap_minutes,
                    MAX(gap_minutes) as max_gap_minutes
                FROM time_gaps
                WHERE prev_time IS NOT NULL
                """,
                underlying_id,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )

            if gap_analysis and gap_analysis["total_gaps"] > 0:
                total_gaps = int(gap_analysis["total_gaps"])
                large_gaps = int(gap_analysis["large_gaps"] or 0)

                timeliness_score = 1.0 - (large_gaps / total_gaps)

                report.timeliness_metrics.append(
                    QualityMetric(
                        name="temporal_consistency",
                        value=timeliness_score,
                        threshold=0.95,
                        passed=timeliness_score >= 0.95,
                        weight=1.0,
                        description=f"Temporal consistency: {large_gaps}/{total_gaps} large gaps",
                        details={
                            "total_gaps": total_gaps,
                            "large_gaps": large_gaps,
                            "avg_gap_minutes": float(
                                gap_analysis["avg_gap_minutes"] or 0
                            ),
                            "max_gap_minutes": float(
                                gap_analysis["max_gap_minutes"] or 0
                            ),
                        },
                    )
                )

            # Data freshness (how recent is the latest data)
            latest_data = await conn.fetchval(
                """
                SELECT MAX(time)
                FROM stock_data_ticks
                WHERE underlying_id = $1
                """,
                underlying_id,
            )

            if latest_data:
                days_since_latest = (datetime.now() - latest_data).days
                freshness_score = max(
                    0.0, 1.0 - (days_since_latest / 30)
                )  # Decay over 30 days

                report.timeliness_metrics.append(
                    QualityMetric(
                        name="data_freshness",
                        value=freshness_score,
                        threshold=0.8,
                        passed=freshness_score >= 0.8,
                        weight=0.5,
                        description=f"Data freshness: {days_since_latest} days old",
                        details={
                            "latest_data": latest_data.isoformat(),
                            "days_old": days_since_latest,
                        },
                    )
                )

    async def _calculate_liquidity_metrics(
        self,
        underlying_id: int,
        symbol: str,
        start_date: date,
        end_date: date,
        data_type: str,
        report: DataQualityReport,
    ):
        """Calculate liquidity-related quality metrics."""
        async with self.db_pool.acquire() as conn:
            # Average volume analysis
            volume_analysis = await conn.fetchrow(
                """
                SELECT
                    AVG(volume) as avg_volume,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY volume) as median_volume,
                    MIN(volume) as min_volume,
                    COUNT(*) FILTER (WHERE volume > 1000) as high_volume_records,
                    COUNT(*) as total_volume_records
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

            if volume_analysis and volume_analysis["total_volume_records"] > 0:
                avg_volume = float(volume_analysis["avg_volume"] or 0)
                high_volume_ratio = (
                    volume_analysis["high_volume_records"] or 0
                ) / volume_analysis["total_volume_records"]

                # Score based on average volume (logarithmic scale)
                volume_score = min(
                    1.0, np.log10(max(avg_volume, 1)) / 6
                )  # Scale to 1M volume = 1.0

                report.liquidity_metrics.append(
                    QualityMetric(
                        name="volume_liquidity",
                        value=volume_score,
                        threshold=0.3,  # Lower threshold for volume
                        passed=volume_score >= 0.3,
                        weight=1.0,
                        description=f"Average volume: {avg_volume:.0f}",
                        details={
                            "avg_volume": avg_volume,
                            "median_volume": float(
                                volume_analysis["median_volume"] or 0
                            ),
                            "high_volume_ratio": high_volume_ratio,
                        },
                    )
                )

            # Option spread analysis (if applicable)
            if data_type in ["option", "mixed"]:
                spread_analysis = await conn.fetchrow(
                    """
                    SELECT
                        AVG(mdt.bid_ask_spread) as avg_spread,
                        AVG(CASE WHEN mdt.last_price > 0 THEN mdt.bid_ask_spread / mdt.last_price * 100 ELSE NULL END) as avg_spread_percent,
                        COUNT(*) as total_spread_records,
                        COUNT(*) FILTER (WHERE mdt.bid_ask_spread <= 0.5) as tight_spread_records
                    FROM market_data_ticks mdt
                    JOIN option_chains oc ON mdt.contract_id = oc.id
                    WHERE oc.underlying_id = $1
                      AND mdt.time >= $2
                      AND mdt.time <= $3
                      AND mdt.bid_price IS NOT NULL
                      AND mdt.ask_price IS NOT NULL
                    """,
                    underlying_id,
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.max.time()),
                )

                if spread_analysis and spread_analysis["total_spread_records"] > 0:
                    avg_spread_percent = float(
                        spread_analysis["avg_spread_percent"] or 10
                    )

                    # Score based on spread tightness (lower is better)
                    spread_score = max(
                        0.0, 1.0 - (avg_spread_percent / 10)
                    )  # 10% spread = 0 score

                    report.liquidity_metrics.append(
                        QualityMetric(
                            name="option_spread_quality",
                            value=spread_score,
                            threshold=0.5,
                            passed=spread_score >= 0.5,
                            weight=0.8,
                            description=f"Average option spread: {avg_spread_percent:.2f}%",
                            details={
                                "avg_spread_percent": avg_spread_percent,
                                "avg_spread_dollars": float(
                                    spread_analysis["avg_spread"] or 0
                                ),
                                "tight_spread_ratio": (
                                    spread_analysis["tight_spread_records"] or 0
                                )
                                / spread_analysis["total_spread_records"],
                            },
                        )
                    )

    async def _perform_time_series_analysis(
        self,
        underlying_id: int,
        symbol: str,
        start_date: date,
        end_date: date,
        report: DataQualityReport,
    ):
        """Perform time series analysis on the data."""
        async with self.db_pool.acquire() as conn:
            # Daily price movements
            daily_stats = await conn.fetch(
                """
                SELECT
                    DATE(time) as trading_date,
                    MIN(price) as daily_low,
                    MAX(price) as daily_high,
                    (array_agg(price ORDER BY time))[1] as daily_open,
                    (array_agg(price ORDER BY time DESC))[1] as daily_close,
                    SUM(volume) as daily_volume
                FROM stock_data_ticks
                WHERE underlying_id = $1
                  AND time >= $2
                  AND time <= $3
                  AND price IS NOT NULL
                GROUP BY DATE(time)
                ORDER BY trading_date
                """,
                underlying_id,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )

            if daily_stats:
                # Calculate daily returns
                daily_returns = []
                prices = [
                    float(row["daily_close"])
                    for row in daily_stats
                    if row["daily_close"]
                ]

                for i in range(1, len(prices)):
                    if prices[i - 1] > 0:
                        daily_return = (prices[i] - prices[i - 1]) / prices[i - 1]
                        daily_returns.append(daily_return)

                if daily_returns:
                    report.time_series_analysis = {
                        "trading_days": len(daily_stats),
                        "daily_return_mean": statistics.mean(daily_returns),
                        "daily_return_std": (
                            statistics.stdev(daily_returns)
                            if len(daily_returns) > 1
                            else 0
                        ),
                        "daily_return_min": min(daily_returns),
                        "daily_return_max": max(daily_returns),
                        "positive_return_days": sum(1 for r in daily_returns if r > 0),
                        "negative_return_days": sum(1 for r in daily_returns if r < 0),
                    }

    async def _generate_statistical_summary(
        self,
        underlying_id: int,
        symbol: str,
        start_date: date,
        end_date: date,
        report: DataQualityReport,
    ):
        """Generate statistical summary of the data."""
        async with self.db_pool.acquire() as conn:
            # Overall statistics
            overall_stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_records,
                    COUNT(DISTINCT DATE(time)) as trading_days,
                    MIN(time) as earliest_record,
                    MAX(time) as latest_record,
                    AVG(price) as avg_price,
                    STDDEV(price) as price_volatility,
                    SUM(volume) as total_volume
                FROM stock_data_ticks
                WHERE underlying_id = $1
                  AND time >= $2
                  AND time <= $3
                """,
                underlying_id,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )

            if overall_stats:
                report.statistical_summary = {
                    "total_records": overall_stats["total_records"] or 0,
                    "trading_days": overall_stats["trading_days"] or 0,
                    "records_per_day": (overall_stats["total_records"] or 0)
                    / max(overall_stats["trading_days"] or 1, 1),
                    "earliest_record": (
                        overall_stats["earliest_record"].isoformat()
                        if overall_stats["earliest_record"]
                        else None
                    ),
                    "latest_record": (
                        overall_stats["latest_record"].isoformat()
                        if overall_stats["latest_record"]
                        else None
                    ),
                    "avg_price": (
                        float(overall_stats["avg_price"])
                        if overall_stats["avg_price"]
                        else 0
                    ),
                    "price_volatility": (
                        float(overall_stats["price_volatility"])
                        if overall_stats["price_volatility"]
                        else 0
                    ),
                    "total_volume": (
                        int(overall_stats["total_volume"])
                        if overall_stats["total_volume"]
                        else 0
                    ),
                }

                # Update report totals
                report.valid_records = overall_stats["total_records"] or 0

    def _calculate_overall_quality_score(self, report: DataQualityReport):
        """Calculate overall quality score from individual metrics."""
        category_scores = {}

        # Calculate category averages
        for category in [
            "completeness",
            "consistency",
            "accuracy",
            "timeliness",
            "liquidity",
        ]:
            metrics = getattr(report, f"{category}_metrics")
            if metrics:
                weighted_sum = sum(m.value * m.weight for m in metrics)
                total_weight = sum(m.weight for m in metrics)
                category_scores[category] = weighted_sum / max(total_weight, 1)
            else:
                category_scores[category] = 0.5  # Default neutral score

        # Calculate weighted overall score
        overall_score = sum(
            category_scores[category] * self.metric_weights[category]
            for category in self.metric_weights
        )

        # Apply penalties for critical issues
        critical_penalty = len(report.critical_issues) * 0.1
        overall_score = max(0.0, overall_score - critical_penalty)

        report.overall_quality_score = min(1.0, overall_score)

    def _assign_quality_grade(self, report: DataQualityReport):
        """Assign quality grade based on overall score."""
        score = report.overall_quality_score

        for grade, threshold in self.grade_thresholds.items():
            if score >= threshold:
                report.quality_grade = grade
                break
        else:
            report.quality_grade = "F"

    def _assess_backtesting_suitability(self, report: DataQualityReport):
        """Assess whether data is suitable for backtesting."""
        # Basic suitability check
        suitable = (
            report.overall_quality_score >= self.backtesting_threshold
            and len(report.critical_issues) == 0
        )

        # Additional checks for specific requirements
        if suitable:
            # Check completeness
            completeness_ok = any(
                m.name == "stock_data_completeness" and m.passed
                for m in report.completeness_metrics
            )

            # Check consistency
            consistency_ok = (
                all(m.passed for m in report.consistency_metrics)
                or len(report.consistency_metrics) == 0
            )

            suitable = suitable and completeness_ok and consistency_ok

        report.is_suitable_for_backtesting = suitable

    def _generate_quality_recommendations(self, report: DataQualityReport):
        """Generate improvement recommendations based on quality metrics."""
        if report.overall_quality_score >= 0.9:
            report.recommendations.append(
                "Excellent data quality - suitable for all backtesting strategies"
            )

        # Completeness recommendations
        for metric in report.completeness_metrics:
            if not metric.passed:
                if metric.name == "stock_data_completeness":
                    report.recommendations.append(
                        "Improve stock data completeness by filling missing trading days"
                    )
                elif metric.name == "option_data_coverage":
                    report.recommendations.append(
                        "Expand option data collection to cover more contracts and expiries"
                    )

        # Consistency recommendations
        for metric in report.consistency_metrics:
            if not metric.passed:
                if metric.name == "ohlc_consistency":
                    report.recommendations.append(
                        "Review and clean OHLC data for logical consistency violations"
                    )
                elif metric.name == "bid_ask_consistency":
                    report.recommendations.append(
                        "Validate option bid-ask data for crossed markets and negative spreads"
                    )

        # Accuracy recommendations
        for metric in report.accuracy_metrics:
            if not metric.passed:
                if metric.name == "price_accuracy":
                    report.recommendations.append(
                        "Implement outlier detection and filtering for price data"
                    )

        # Timeliness recommendations
        for metric in report.timeliness_metrics:
            if not metric.passed:
                if metric.name == "temporal_consistency":
                    report.recommendations.append(
                        "Address temporal gaps in data collection during market hours"
                    )
                elif metric.name == "data_freshness":
                    report.recommendations.append(
                        "Update data collection to include more recent market data"
                    )

        # Liquidity recommendations
        for metric in report.liquidity_metrics:
            if not metric.passed:
                if metric.name == "volume_liquidity":
                    report.recommendations.append(
                        "Consider focusing on more liquid securities for better execution simulation"
                    )
                elif metric.name == "option_spread_quality":
                    report.recommendations.append(
                        "Filter for options with tighter spreads to improve arbitrage accuracy"
                    )

        # Overall recommendations
        if not report.is_suitable_for_backtesting:
            report.recommendations.append(
                "Address critical data quality issues before using for backtesting"
            )

        if report.quality_grade in ["C", "D", "F"]:
            report.recommendations.append(
                "Consider using alternative data sources or improving data collection processes"
            )

    def _count_trading_days(self, start_date: date, end_date: date) -> int:
        """Count trading days between dates."""
        days = 0
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Monday-Friday
                days += 1
            current += timedelta(days=1)
        return days

    async def save_quality_report(
        self, report: DataQualityReport, file_path: Optional[str] = None
    ) -> str:
        """
        Save quality report to JSON file.

        Args:
            report: Quality report to save
            file_path: Optional file path (auto-generated if not provided)

        Returns:
            Path to saved file
        """
        if not file_path:
            file_path = f"quality_report_{report.report_id}.json"

        try:
            with open(file_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

            logger.info(f"Quality report saved to {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error saving quality report: {e}")
            raise
