"""
VIX Correlation Analysis Module

This module provides comprehensive analysis of correlations between VIX volatility levels/regimes
and arbitrage opportunity success rates. It generates insights for optimizing trading strategies
based on volatility market conditions.

Features:
- VIX regime analysis (LOW, MEDIUM, HIGH, EXTREME)
- Term structure correlation (contango vs backwardation)
- Strategy performance by volatility environment
- Statistical correlation metrics and significance testing
- Performance reporting and visualization data

Author: Claude Code Assistant
Created: 2025-08-04
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncpg
import logging
import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)


class VIXRegime(Enum):
    """VIX volatility regime classifications."""

    LOW = "LOW"  # VIX < 15
    MEDIUM = "MEDIUM"  # 15 <= VIX <= 25
    HIGH = "HIGH"  # 25 < VIX <= 40
    EXTREME = "EXTREME"  # VIX > 40


class TermStructureType(Enum):
    """VIX term structure classifications."""

    CONTANGO = "CONTANGO"  # Forward VIX > Current VIX (normal)
    BACKWARDATION = "BACKWARDATION"  # Forward VIX < Current VIX (stress)
    FLAT = "FLAT"  # Forward VIX ≈ Current VIX


@dataclass
class VIXCorrelationMetrics:
    """Metrics for VIX correlation analysis."""

    # Basic correlation metrics
    correlation_coefficient: float
    p_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int

    # Performance by regime
    regime_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Term structure analysis
    structure_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Statistical significance
    is_statistically_significant: bool = field(default=False)
    significance_level: float = field(default=0.05)

    # Additional insights
    optimal_vix_range: Optional[Tuple[float, float]] = None
    worst_vix_range: Optional[Tuple[float, float]] = None
    volatility_timing_score: Optional[float] = None


@dataclass
class StrategyPerformanceByVIX:
    """Performance metrics for a strategy across different VIX conditions."""

    strategy_type: str
    analysis_period_days: int

    # Overall metrics
    total_opportunities: int
    total_successful: int
    overall_success_rate: float
    overall_avg_profit: float

    # VIX regime breakdown
    low_vix_performance: Dict[str, Any] = field(default_factory=dict)
    medium_vix_performance: Dict[str, Any] = field(default_factory=dict)
    high_vix_performance: Dict[str, Any] = field(default_factory=dict)
    extreme_vix_performance: Dict[str, Any] = field(default_factory=dict)

    # Term structure breakdown
    contango_performance: Dict[str, Any] = field(default_factory=dict)
    backwardation_performance: Dict[str, Any] = field(default_factory=dict)
    flat_performance: Dict[str, Any] = field(default_factory=dict)

    # Key insights
    best_vix_regime: Optional[str] = None
    worst_vix_regime: Optional[str] = None
    vix_sensitivity_score: Optional[float] = None


class VIXCorrelationAnalyzer:
    """
    Advanced VIX correlation analyzer for arbitrage strategies.

    This class provides comprehensive analysis of how VIX volatility levels and term structure
    correlate with arbitrage opportunity success rates and profitability.
    """

    def __init__(self, db_pool: asyncpg.Pool):
        """
        Initialize VIX correlation analyzer.

        Args:
            db_pool: AsyncPG database connection pool
        """
        self.db_pool = db_pool
        self.cache = {}  # Simple cache for repeated queries
        self.cache_ttl = 300  # 5 minutes

    async def analyze_strategy_vix_correlation(
        self, strategy_type: str, days_back: int = 30, min_sample_size: int = 10
    ) -> StrategyPerformanceByVIX:
        """
        Analyze VIX correlation for a specific arbitrage strategy.

        Args:
            strategy_type: Strategy type (SFR, SYNTHETIC, BOX, CALENDAR)
            days_back: Number of days to analyze
            min_sample_size: Minimum sample size for statistical significance

        Returns:
            StrategyPerformanceByVIX with comprehensive analysis
        """
        logger.info(
            f"Analyzing VIX correlation for {strategy_type} strategy over {days_back} days"
        )

        start_date = datetime.now() - timedelta(days=days_back)
        end_date = datetime.now()

        # Get raw correlation data
        correlation_data = await self._get_correlation_data(
            strategy_type, start_date, end_date
        )

        if len(correlation_data) < min_sample_size:
            logger.warning(
                f"Insufficient data for {strategy_type}: {len(correlation_data)} records (min: {min_sample_size})"
            )
            return self._create_empty_performance_analysis(strategy_type, days_back)

        # Convert to DataFrame for analysis
        df = pd.DataFrame(correlation_data)

        # Calculate overall performance
        total_opportunities = len(df)
        total_successful = len(df[df["arbitrage_success"] == True])
        overall_success_rate = (
            (total_successful / total_opportunities * 100)
            if total_opportunities > 0
            else 0.0
        )
        overall_avg_profit = (
            df[df["arbitrage_success"] == True]["profit_realized"].mean()
            if total_successful > 0
            else 0.0
        )

        # Analyze by VIX regime
        regime_analysis = self._analyze_by_vix_regime(df)

        # Analyze by term structure
        structure_analysis = self._analyze_by_term_structure(df)

        # Calculate VIX sensitivity score
        vix_sensitivity = self._calculate_vix_sensitivity(df)

        # Identify best and worst regimes
        best_regime, worst_regime = self._identify_optimal_regimes(regime_analysis)

        return StrategyPerformanceByVIX(
            strategy_type=strategy_type,
            analysis_period_days=days_back,
            total_opportunities=total_opportunities,
            total_successful=total_successful,
            overall_success_rate=overall_success_rate,
            overall_avg_profit=(
                float(overall_avg_profit) if not pd.isna(overall_avg_profit) else 0.0
            ),
            low_vix_performance=regime_analysis.get("LOW", {}),
            medium_vix_performance=regime_analysis.get("MEDIUM", {}),
            high_vix_performance=regime_analysis.get("HIGH", {}),
            extreme_vix_performance=regime_analysis.get("EXTREME", {}),
            contango_performance=structure_analysis.get("CONTANGO", {}),
            backwardation_performance=structure_analysis.get("BACKWARDATION", {}),
            flat_performance=structure_analysis.get("FLAT", {}),
            best_vix_regime=best_regime,
            worst_vix_regime=worst_regime,
            vix_sensitivity_score=vix_sensitivity,
        )

    async def _get_correlation_data(
        self, strategy_type: str, start_date: datetime, end_date: datetime
    ) -> List[Dict]:
        """Retrieve correlation data from database."""
        async with self.db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT
                    vac.vix_level,
                    vac.vix_regime,
                    vac.term_structure_type,
                    vac.vix_spike_active,
                    vac.arbitrage_success,
                    vac.execution_speed_ms,
                    vac.profit_realized,
                    vac.correlation_weight,
                    vac.created_at,
                    ao.strategy_type,
                    ao.symbol,
                    ao.theoretical_profit,
                    ao.roi_percent,
                    ao.confidence_score
                FROM vix_arbitrage_correlation vac
                JOIN arbitrage_opportunities ao ON vac.arbitrage_opportunity_id = ao.id
                WHERE ao.strategy_type = $1
                    AND vac.created_at >= $2
                    AND vac.created_at <= $3
                    AND vac.outlier_flag = false
                ORDER BY vac.created_at DESC
            """,
                strategy_type,
                start_date,
                end_date,
            )

            return [dict(row) for row in results]

    def _analyze_by_vix_regime(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by VIX regime."""
        regime_analysis = {}

        for regime in ["LOW", "MEDIUM", "HIGH", "EXTREME"]:
            regime_data = df[df["vix_regime"] == regime]

            if len(regime_data) == 0:
                regime_analysis[regime] = {
                    "sample_size": 0,
                    "success_rate": 0.0,
                    "avg_profit": 0.0,
                    "avg_vix_level": 0.0,
                    "profit_std": 0.0,
                    "avg_execution_speed_ms": 0.0,
                }
                continue

            successful = regime_data[regime_data["arbitrage_success"] == True]

            regime_analysis[regime] = {
                "sample_size": len(regime_data),
                "success_rate": (
                    len(successful) / len(regime_data) * 100
                    if len(regime_data) > 0
                    else 0.0
                ),
                "avg_profit": (
                    float(successful["profit_realized"].mean())
                    if len(successful) > 0
                    else 0.0
                ),
                "avg_vix_level": float(regime_data["vix_level"].mean()),
                "vix_std": float(regime_data["vix_level"].std()),
                "profit_std": (
                    float(successful["profit_realized"].std())
                    if len(successful) > 1
                    else 0.0
                ),
                "avg_execution_speed_ms": (
                    float(regime_data["execution_speed_ms"].mean())
                    if regime_data["execution_speed_ms"].notna().any()
                    else 0.0
                ),
                "median_profit": (
                    float(successful["profit_realized"].median())
                    if len(successful) > 0
                    else 0.0
                ),
                "max_profit": (
                    float(successful["profit_realized"].max())
                    if len(successful) > 0
                    else 0.0
                ),
                "min_profit": (
                    float(successful["profit_realized"].min())
                    if len(successful) > 0
                    else 0.0
                ),
            }

        return regime_analysis

    def _analyze_by_term_structure(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by term structure type."""
        structure_analysis = {}

        for structure in ["CONTANGO", "BACKWARDATION", "FLAT"]:
            structure_data = df[df["term_structure_type"] == structure]

            if len(structure_data) == 0:
                structure_analysis[structure] = {
                    "sample_size": 0,
                    "success_rate": 0.0,
                    "avg_profit": 0.0,
                    "avg_vix_level": 0.0,
                }
                continue

            successful = structure_data[structure_data["arbitrage_success"] == True]

            structure_analysis[structure] = {
                "sample_size": len(structure_data),
                "success_rate": (
                    len(successful) / len(structure_data) * 100
                    if len(structure_data) > 0
                    else 0.0
                ),
                "avg_profit": (
                    float(successful["profit_realized"].mean())
                    if len(successful) > 0
                    else 0.0
                ),
                "avg_vix_level": float(structure_data["vix_level"].mean()),
                "profit_volatility": (
                    float(successful["profit_realized"].std())
                    if len(successful) > 1
                    else 0.0
                ),
            }

        return structure_analysis

    def _calculate_vix_sensitivity(self, df: pd.DataFrame) -> float:
        """Calculate VIX sensitivity score (0-1, higher = more sensitive to VIX)."""
        if len(df) < 10:
            return 0.0

        try:
            # Calculate correlation between VIX level and success
            vix_levels = df["vix_level"].values
            success_values = df["arbitrage_success"].astype(int).values

            correlation, p_value = stats.pearsonr(vix_levels, success_values)

            # Sensitivity score based on correlation strength and significance
            sensitivity = abs(correlation) if p_value < 0.05 else abs(correlation) * 0.5

            return min(max(sensitivity, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating VIX sensitivity: {e}")
            return 0.0

    def _identify_optimal_regimes(
        self, regime_analysis: Dict[str, Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Identify best and worst performing VIX regimes."""
        regimes_with_data = {
            k: v for k, v in regime_analysis.items() if v["sample_size"] > 0
        }

        if not regimes_with_data:
            return None, None

        # Sort by success rate
        sorted_regimes = sorted(
            regimes_with_data.items(), key=lambda x: x[1]["success_rate"], reverse=True
        )

        best_regime = sorted_regimes[0][0] if sorted_regimes else None
        worst_regime = sorted_regimes[-1][0] if sorted_regimes else None

        return best_regime, worst_regime

    def _create_empty_performance_analysis(
        self, strategy_type: str, days_back: int
    ) -> StrategyPerformanceByVIX:
        """Create empty performance analysis for insufficient data."""
        return StrategyPerformanceByVIX(
            strategy_type=strategy_type,
            analysis_period_days=days_back,
            total_opportunities=0,
            total_successful=0,
            overall_success_rate=0.0,
            overall_avg_profit=0.0,
        )

    async def calculate_statistical_significance(
        self, strategy_type: str, days_back: int = 30, significance_level: float = 0.05
    ) -> VIXCorrelationMetrics:
        """
        Calculate statistical significance of VIX correlation.

        Args:
            strategy_type: Strategy type to analyze
            days_back: Number of days to analyze
            significance_level: Statistical significance threshold

        Returns:
            VIXCorrelationMetrics with statistical analysis
        """
        start_date = datetime.now() - timedelta(days=days_back)
        end_date = datetime.now()

        correlation_data = await self._get_correlation_data(
            strategy_type, start_date, end_date
        )

        if len(correlation_data) < 10:
            return VIXCorrelationMetrics(
                correlation_coefficient=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                sample_size=len(correlation_data),
                is_statistically_significant=False,
            )

        df = pd.DataFrame(correlation_data)

        # Calculate Pearson correlation
        vix_levels = df["vix_level"].values
        success_values = df["arbitrage_success"].astype(int).values

        correlation, p_value = stats.pearsonr(vix_levels, success_values)

        # Calculate confidence interval
        n = len(df)
        z_score = stats.norm.ppf(1 - significance_level / 2)
        standard_error = np.sqrt((1 - correlation**2) / (n - 2))
        margin_of_error = z_score * standard_error

        confidence_interval = (
            max(-1, correlation - margin_of_error),
            min(1, correlation + margin_of_error),
        )

        # Additional analyses
        regime_performance = self._analyze_by_vix_regime(df)
        structure_performance = self._analyze_by_term_structure(df)

        # Calculate optimal VIX range
        optimal_range = self._calculate_optimal_vix_range(df)
        worst_range = self._calculate_worst_vix_range(df)

        # Volatility timing score
        timing_score = self._calculate_volatility_timing_score(df)

        return VIXCorrelationMetrics(
            correlation_coefficient=correlation,
            p_value=p_value,
            confidence_interval=confidence_interval,
            sample_size=n,
            regime_performance=regime_performance,
            structure_performance=structure_performance,
            is_statistically_significant=p_value < significance_level,
            significance_level=significance_level,
            optimal_vix_range=optimal_range,
            worst_vix_range=worst_range,
            volatility_timing_score=timing_score,
        )

    def _calculate_optimal_vix_range(
        self, df: pd.DataFrame
    ) -> Optional[Tuple[float, float]]:
        """Calculate optimal VIX range for highest success rates."""
        if len(df) < 20:
            return None

        try:
            # Group by VIX level ranges and find highest success rate
            df["vix_range"] = pd.cut(df["vix_level"], bins=10)
            range_success = df.groupby("vix_range")["arbitrage_success"].agg(
                ["mean", "count"]
            )

            # Filter ranges with sufficient sample size
            significant_ranges = range_success[range_success["count"] >= 5]

            if len(significant_ranges) == 0:
                return None

            # Find range with highest success rate
            best_range = significant_ranges["mean"].idxmax()

            return (float(best_range.left), float(best_range.right))

        except Exception:
            return None

    def _calculate_worst_vix_range(
        self, df: pd.DataFrame
    ) -> Optional[Tuple[float, float]]:
        """Calculate worst VIX range for lowest success rates."""
        if len(df) < 20:
            return None

        try:
            df["vix_range"] = pd.cut(df["vix_level"], bins=10)
            range_success = df.groupby("vix_range")["arbitrage_success"].agg(
                ["mean", "count"]
            )

            significant_ranges = range_success[range_success["count"] >= 5]

            if len(significant_ranges) == 0:
                return None

            worst_range = significant_ranges["mean"].idxmin()

            return (float(worst_range.left), float(worst_range.right))

        except Exception:
            return None

    def _calculate_volatility_timing_score(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate volatility timing score (0-1, higher = better timing)."""
        if len(df) < 10:
            return None

        try:
            # Score based on how well the strategy performs in different VIX regimes
            regime_scores = []

            for regime in ["LOW", "MEDIUM", "HIGH", "EXTREME"]:
                regime_data = df[df["vix_regime"] == regime]
                if len(regime_data) > 0:
                    success_rate = regime_data["arbitrage_success"].mean()
                    regime_scores.append(success_rate)

            if not regime_scores:
                return None

            # Score is higher when strategy performs consistently across regimes
            # or performs exceptionally well in specific regimes
            consistency_score = 1 - np.std(regime_scores)  # Penalize high variance
            performance_score = np.mean(
                regime_scores
            )  # Reward high average performance

            timing_score = consistency_score * 0.3 + performance_score * 0.7

            return min(max(timing_score, 0.0), 1.0)

        except Exception:
            return None

    async def generate_vix_correlation_report(
        self, strategy_types: List[str] = None, days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Generate comprehensive VIX correlation report for multiple strategies.

        Args:
            strategy_types: List of strategy types to analyze (default: all)
            days_back: Number of days to analyze

        Returns:
            Comprehensive correlation report
        """
        if strategy_types is None:
            strategy_types = ["SFR", "SYNTHETIC", "BOX", "CALENDAR"]

        logger.info(
            f"Generating VIX correlation report for {len(strategy_types)} strategies"
        )

        report = {
            "report_generated_at": datetime.now().isoformat(),
            "analysis_period_days": days_back,
            "strategies_analyzed": strategy_types,
            "strategy_analyses": {},
            "cross_strategy_insights": {},
            "recommendations": [],
        }

        # Analyze each strategy
        for strategy in strategy_types:
            try:
                performance = await self.analyze_strategy_vix_correlation(
                    strategy, days_back
                )
                statistical_metrics = await self.calculate_statistical_significance(
                    strategy, days_back
                )

                report["strategy_analyses"][strategy] = {
                    "performance": performance,
                    "statistical_metrics": statistical_metrics,
                }

            except Exception as e:
                logger.error(f"Error analyzing {strategy}: {e}")
                report["strategy_analyses"][strategy] = {"error": str(e)}

        # Generate cross-strategy insights
        report["cross_strategy_insights"] = self._generate_cross_strategy_insights(
            report["strategy_analyses"]
        )

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(
            report["strategy_analyses"]
        )

        logger.info("VIX correlation report generated successfully")
        return report

    def _generate_cross_strategy_insights(
        self, strategy_analyses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate insights across all strategies."""
        insights = {
            "most_vix_sensitive_strategy": None,
            "least_vix_sensitive_strategy": None,
            "best_high_vix_strategy": None,
            "best_low_vix_strategy": None,
            "universal_optimal_regime": None,
        }

        # Find most and least VIX sensitive strategies
        sensitivity_scores = {}
        for strategy, analysis in strategy_analyses.items():
            if "performance" in analysis and hasattr(
                analysis["performance"], "vix_sensitivity_score"
            ):
                if analysis["performance"].vix_sensitivity_score is not None:
                    sensitivity_scores[strategy] = analysis[
                        "performance"
                    ].vix_sensitivity_score

        if sensitivity_scores:
            insights["most_vix_sensitive_strategy"] = max(
                sensitivity_scores, key=sensitivity_scores.get
            )
            insights["least_vix_sensitive_strategy"] = min(
                sensitivity_scores, key=sensitivity_scores.get
            )

        # Find best performers in different regimes
        high_vix_performers = {}
        low_vix_performers = {}

        for strategy, analysis in strategy_analyses.items():
            if "performance" in analysis:
                perf = analysis["performance"]
                if (
                    hasattr(perf, "high_vix_performance")
                    and perf.high_vix_performance.get("success_rate", 0) > 0
                ):
                    high_vix_performers[strategy] = perf.high_vix_performance[
                        "success_rate"
                    ]
                if (
                    hasattr(perf, "low_vix_performance")
                    and perf.low_vix_performance.get("success_rate", 0) > 0
                ):
                    low_vix_performers[strategy] = perf.low_vix_performance[
                        "success_rate"
                    ]

        if high_vix_performers:
            insights["best_high_vix_strategy"] = max(
                high_vix_performers, key=high_vix_performers.get
            )
        if low_vix_performers:
            insights["best_low_vix_strategy"] = max(
                low_vix_performers, key=low_vix_performers.get
            )

        return insights

    def _generate_recommendations(self, strategy_analyses: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        for strategy, analysis in strategy_analyses.items():
            if "performance" not in analysis:
                continue

            perf = analysis["performance"]

            # Regime-based recommendations
            if hasattr(perf, "best_vix_regime") and perf.best_vix_regime:
                recommendations.append(
                    f"Consider increasing {strategy} allocation during {perf.best_vix_regime} VIX regime "
                    f"(success rate: {getattr(perf, f'{perf.best_vix_regime.lower()}_vix_performance', {}).get('success_rate', 0):.1f}%)"
                )

            if hasattr(perf, "worst_vix_regime") and perf.worst_vix_regime:
                recommendations.append(
                    f"Consider reducing {strategy} allocation during {perf.worst_vix_regime} VIX regime "
                    f"(success rate: {getattr(perf, f'{perf.worst_vix_regime.lower()}_vix_performance', {}).get('success_rate', 0):.1f}%)"
                )

            # Sensitivity-based recommendations
            if (
                hasattr(perf, "vix_sensitivity_score")
                and perf.vix_sensitivity_score is not None
            ):
                if perf.vix_sensitivity_score > 0.7:
                    recommendations.append(
                        f"{strategy} is highly VIX-sensitive (score: {perf.vix_sensitivity_score:.2f}). "
                        "Consider VIX-based position sizing."
                    )
                elif perf.vix_sensitivity_score < 0.3:
                    recommendations.append(
                        f"{strategy} is VIX-insensitive (score: {perf.vix_sensitivity_score:.2f}). "
                        "Suitable for consistent allocation across volatility regimes."
                    )

        return recommendations


@asynccontextmanager
async def create_vix_analyzer(db_pool: asyncpg.Pool) -> VIXCorrelationAnalyzer:
    """
    Context manager for creating VIX correlation analyzer.

    Usage:
        async with create_vix_analyzer(db_pool) as analyzer:
            report = await analyzer.generate_vix_correlation_report()
    """
    analyzer = VIXCorrelationAnalyzer(db_pool)
    try:
        yield analyzer
    finally:
        # Cleanup if needed
        analyzer.cache.clear()


# Utility functions for external use
async def quick_vix_analysis(
    db_pool: asyncpg.Pool, strategy_type: str, days_back: int = 7
) -> Dict[str, Any]:
    """
    Quick VIX correlation analysis for a single strategy.

    Args:
        db_pool: Database connection pool
        strategy_type: Strategy to analyze
        days_back: Days to look back

    Returns:
        Simplified analysis results
    """
    async with create_vix_analyzer(db_pool) as analyzer:
        performance = await analyzer.analyze_strategy_vix_correlation(
            strategy_type, days_back
        )

        return {
            "strategy": strategy_type,
            "period_days": days_back,
            "total_opportunities": performance.total_opportunities,
            "success_rate": performance.overall_success_rate,
            "best_vix_regime": performance.best_vix_regime,
            "vix_sensitivity": performance.vix_sensitivity_score,
        }


async def get_optimal_vix_conditions(
    db_pool: asyncpg.Pool, strategy_type: str
) -> Dict[str, Any]:
    """
    Get optimal VIX conditions for a strategy.

    Args:
        db_pool: Database connection pool
        strategy_type: Strategy to analyze

    Returns:
        Optimal VIX conditions
    """
    async with create_vix_analyzer(db_pool) as analyzer:
        metrics = await analyzer.calculate_statistical_significance(strategy_type)

        return {
            "optimal_vix_range": metrics.optimal_vix_range,
            "worst_vix_range": metrics.worst_vix_range,
            "correlation_strength": metrics.correlation_coefficient,
            "statistically_significant": metrics.is_statistically_significant,
            "timing_score": metrics.volatility_timing_score,
        }
