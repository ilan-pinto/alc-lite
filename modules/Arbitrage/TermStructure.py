"""
Term Structure Analysis module for options trading calendar spread opportunities.

This module provides comprehensive analysis of implied volatility term structure
across different expirations to identify optimal calendar spread opportunities.
It detects IV inversions, analyzes volatility skew, and scores opportunities
based on term structure characteristics.

Key Features:
- IV curve construction across multiple expirations
- Term structure inversion detection (front > back month IV)
- Historical IV percentile analysis
- Volatility skew exploitation identification
- Integration with calendar spread opportunity scoring
- Performance-optimized processing with caching
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple, Union

import logging
import numpy as np
from ib_async import Contract, Ticker

from .common import configure_logging, get_logger
from .metrics import RejectionReason, metrics_collector

# Configure logging
logger = get_logger()


@dataclass
class IVDataPoint:
    """Individual implied volatility data point"""

    expiry: str
    days_to_expiry: int
    strike: float
    option_type: str  # 'CALL' or 'PUT'
    iv: float
    price: float
    volume: int
    delta: float
    bid: float
    ask: float
    last_updated: float = field(default_factory=time.time)


@dataclass
class TermStructureCurve:
    """Implied volatility term structure curve"""

    symbol: str
    strike: float
    option_type: str
    curve_points: List[IVDataPoint] = field(default_factory=list)
    curve_time: float = field(default_factory=time.time)

    def get_iv_at_expiry(self, days_to_expiry: int) -> Optional[float]:
        """Get IV for specific days to expiry via interpolation"""
        if not self.curve_points:
            return None

        # Sort by days to expiry
        sorted_points = sorted(self.curve_points, key=lambda x: x.days_to_expiry)

        # Exact match
        for point in sorted_points:
            if point.days_to_expiry == days_to_expiry:
                return point.iv

        # Linear interpolation between adjacent points
        for i in range(len(sorted_points) - 1):
            if (
                sorted_points[i].days_to_expiry
                <= days_to_expiry
                <= sorted_points[i + 1].days_to_expiry
            ):

                x1, y1 = sorted_points[i].days_to_expiry, sorted_points[i].iv
                x2, y2 = sorted_points[i + 1].days_to_expiry, sorted_points[i + 1].iv

                # Linear interpolation
                if x2 != x1:
                    return y1 + (y2 - y1) * (days_to_expiry - x1) / (x2 - x1)
                else:
                    return y1

        return None


@dataclass
class TermStructureInversion:
    """Term structure inversion opportunity"""

    symbol: str
    strike: float
    option_type: str
    front_expiry: str
    back_expiry: str
    front_days: int
    back_days: int
    front_iv: float
    back_iv: float
    iv_differential: float  # front_iv - back_iv (positive = inversion)
    inversion_magnitude: float  # Normalized inversion strength
    confidence_score: float  # Statistical confidence in inversion
    opportunity_score: float  # Overall opportunity attractiveness


@dataclass
class IVPercentileData:
    """Historical IV percentile information"""

    symbol: str
    strike: float
    option_type: str
    current_iv: float
    percentile_rank: float  # 0-100 percentile rank
    historical_mean: float
    historical_std: float
    lookback_days: int
    confidence_level: float


@dataclass
class TermStructureConfig:
    """Configuration for term structure analysis"""

    # IV spread thresholds
    min_iv_spread: float = 2.0  # Minimum IV spread for opportunities (%)
    min_inversion_threshold: float = 10.0  # Minimum inversion magnitude (%)

    # Confidence thresholds
    min_confidence_score: float = (
        0.7  # Minimum confidence for high-quality opportunities
    )
    min_opportunity_score: float = 0.6  # Minimum overall opportunity score

    # Data requirements
    min_data_points: int = 3  # Minimum data points for curve construction
    max_curve_age: float = 300.0  # Maximum age of curve data (seconds)

    # Historical analysis
    iv_percentile_lookback: int = 252  # Trading days for percentile calculation
    min_historical_samples: int = 30  # Minimum samples for percentile analysis

    # Performance optimization
    cache_ttl: float = 60.0  # Cache TTL for IV calculations (seconds)
    max_processing_time: float = 300.0  # Maximum processing time per symbol (ms)


class TermStructureAnalyzer:
    """
    Advanced term structure analyzer for options trading calendar spread opportunities.

    This class provides comprehensive analysis of implied volatility term structure
    to identify optimal calendar spread opportunities. It analyzes IV curves across
    multiple expirations, detects term structure inversions, and integrates with
    existing calendar spread detection systems.

    Key Capabilities:
    1. IV Curve Construction - Build accurate IV curves from option chain data
    2. Inversion Detection - Identify term structure inversions (front > back month)
    3. Percentile Analysis - Compare current IV to historical levels
    4. Opportunity Scoring - Score opportunities based on term structure characteristics
    5. Performance Optimization - Efficient processing with intelligent caching
    """

    def __init__(self, config: Optional[TermStructureConfig] = None) -> None:
        """
        Initialize the term structure analyzer.

        Args:
            config: Configuration for term structure analysis
        """
        self.config = config or TermStructureConfig()

        # Caching for performance optimization
        self.iv_curve_cache: Dict[str, TermStructureCurve] = {}
        self.iv_calculation_cache: Dict[str, Tuple[float, float]] = (
            {}
        )  # (iv, timestamp)
        self.percentile_cache: Dict[str, Tuple[IVPercentileData, float]] = {}

        # Historical data storage (in production, this would connect to a database)
        self.historical_iv_data: Dict[str, List[Tuple[float, float]]] = defaultdict(
            list
        )  # timestamp, iv

        logger.info("Term Structure Analyzer initialized")

    def analyze_term_structure(
        self, symbol: str, options_data: Dict[int, Ticker]
    ) -> Tuple[List[TermStructureCurve], List[TermStructureInversion]]:
        """
        Main term structure analysis method.

        Analyzes the complete options chain to build IV curves and detect
        term structure opportunities and inversions.

        Args:
            symbol: Trading symbol
            options_data: Dictionary mapping contract IDs to ticker data

        Returns:
            Tuple of (IV curves, term structure inversions)
        """
        start_time = time.time()

        try:
            # Build IV curves from options data
            iv_curves = self._build_iv_curves(symbol, options_data)

            if not iv_curves:
                logger.warning(f"No IV curves built for {symbol}")
                return [], []

            # Detect term structure inversions
            inversions = self._detect_term_structure_inversions(iv_curves)

            # Filter and score inversions
            qualified_inversions = self._score_and_filter_inversions(inversions)

            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            if processing_time > self.config.max_processing_time:
                logger.warning(
                    f"Term structure analysis for {symbol} took {processing_time:.1f}ms "
                    f"(limit: {self.config.max_processing_time}ms)"
                )

            logger.info(
                f"Term structure analysis for {symbol}: "
                f"{len(iv_curves)} curves, {len(qualified_inversions)} inversions "
                f"({processing_time:.1f}ms)"
            )

            return iv_curves, qualified_inversions

        except Exception as e:
            logger.error(f"Error in term structure analysis for {symbol}: {str(e)}")
            return [], []

    def _build_iv_curves(
        self, symbol: str, options_data: Dict[int, Ticker]
    ) -> List[TermStructureCurve]:
        """
        Build implied volatility curves from options data.

        Groups options by strike and type, then constructs IV curves
        across different expiration dates.
        """
        # Group options by strike and type
        strike_groups: Dict[Tuple[float, str], List[IVDataPoint]] = defaultdict(list)

        for contract_id, ticker in options_data.items():
            if not ticker or not hasattr(ticker, "contract"):
                continue

            contract = ticker.contract
            if not hasattr(contract, "strike") or not hasattr(contract, "right"):
                continue

            # Calculate IV for this option
            iv = self._calculate_implied_volatility_cached(ticker, contract)
            if iv is None or iv <= 0:
                continue

            # Create data point
            days_to_expiry = self._calculate_days_to_expiry(
                contract.lastTradeDateOrContractMonth
            )

            if days_to_expiry <= 0:
                continue

            data_point = IVDataPoint(
                expiry=contract.lastTradeDateOrContractMonth,
                days_to_expiry=days_to_expiry,
                strike=contract.strike,
                option_type="CALL" if contract.right == "C" else "PUT",
                iv=iv,
                price=(
                    ticker.midpoint()
                    if not np.isnan(ticker.midpoint())
                    else ticker.close
                ),
                volume=getattr(ticker, "volume", 0) or 0,
                delta=self._estimate_delta(ticker, contract),
                bid=ticker.bid if not np.isnan(ticker.bid) else 0.0,
                ask=ticker.ask if not np.isnan(ticker.ask) else 0.0,
            )

            key = (contract.strike, data_point.option_type)
            strike_groups[key].append(data_point)

        # Build curves for each strike/type combination
        iv_curves = []
        for (strike, option_type), data_points in strike_groups.items():
            if len(data_points) < self.config.min_data_points:
                continue

            # Sort by days to expiry
            data_points.sort(key=lambda x: x.days_to_expiry)

            curve = TermStructureCurve(
                symbol=symbol,
                strike=strike,
                option_type=option_type,
                curve_points=data_points,
            )

            iv_curves.append(curve)

            # Cache the curve
            cache_key = f"{symbol}_{strike}_{option_type}"
            self.iv_curve_cache[cache_key] = curve

        return iv_curves

    def _detect_term_structure_inversions(
        self, iv_curves: List[TermStructureCurve]
    ) -> List[TermStructureInversion]:
        """
        Detect term structure inversions across IV curves.

        An inversion occurs when shorter-term options have higher IV
        than longer-term options, creating favorable conditions for
        calendar spreads.
        """
        inversions = []

        for curve in iv_curves:
            if len(curve.curve_points) < 2:
                continue

            # Check all pairs of expiries for inversions
            points = sorted(curve.curve_points, key=lambda x: x.days_to_expiry)

            for i in range(len(points) - 1):
                for j in range(i + 1, len(points)):
                    front_point = points[i]
                    back_point = points[j]

                    # Calculate IV differential (front - back)
                    iv_differential = front_point.iv - back_point.iv

                    # Check for inversion (front IV > back IV)
                    if iv_differential > 0:
                        # Calculate inversion magnitude (normalized)
                        inversion_magnitude = (iv_differential / back_point.iv) * 100

                        if inversion_magnitude >= self.config.min_inversion_threshold:
                            # Calculate confidence score
                            confidence = self._calculate_inversion_confidence(
                                front_point, back_point, curve
                            )

                            # Calculate opportunity score
                            opportunity_score = (
                                self._calculate_inversion_opportunity_score(
                                    front_point,
                                    back_point,
                                    inversion_magnitude,
                                    confidence,
                                )
                            )

                            inversion = TermStructureInversion(
                                symbol=curve.symbol,
                                strike=curve.strike,
                                option_type=curve.option_type,
                                front_expiry=front_point.expiry,
                                back_expiry=back_point.expiry,
                                front_days=front_point.days_to_expiry,
                                back_days=back_point.days_to_expiry,
                                front_iv=front_point.iv,
                                back_iv=back_point.iv,
                                iv_differential=iv_differential,
                                inversion_magnitude=inversion_magnitude,
                                confidence_score=confidence,
                                opportunity_score=opportunity_score,
                            )

                            inversions.append(inversion)

        return inversions

    def _calculate_inversion_confidence(
        self,
        front_point: IVDataPoint,
        back_point: IVDataPoint,
        curve: TermStructureCurve,
    ) -> float:
        """
        Calculate statistical confidence in the term structure inversion.

        Considers factors like data quality, volume, bid-ask spreads,
        and consistency across the curve.
        """
        confidence_factors = []

        # Volume confidence (higher volume = higher confidence)
        front_volume_conf = min(1.0, front_point.volume / 100.0)
        back_volume_conf = min(1.0, back_point.volume / 100.0)
        volume_confidence = (front_volume_conf + back_volume_conf) / 2.0
        confidence_factors.append(volume_confidence * 0.3)

        # Bid-ask spread confidence (tighter spreads = higher confidence)
        front_spread = (
            (front_point.ask - front_point.bid) / front_point.price
            if front_point.price > 0
            else 1.0
        )
        back_spread = (
            (back_point.ask - back_point.bid) / back_point.price
            if back_point.price > 0
            else 1.0
        )

        front_spread_conf = max(0.0, 1.0 - front_spread * 5.0)  # Penalize wide spreads
        back_spread_conf = max(0.0, 1.0 - back_spread * 5.0)
        spread_confidence = (front_spread_conf + back_spread_conf) / 2.0
        confidence_factors.append(spread_confidence * 0.25)

        # Data freshness confidence
        current_time = time.time()
        front_age = current_time - front_point.last_updated
        back_age = current_time - back_point.last_updated

        max_age = 300.0  # 5 minutes
        front_freshness = max(0.0, 1.0 - front_age / max_age)
        back_freshness = max(0.0, 1.0 - back_age / max_age)
        freshness_confidence = (front_freshness + back_freshness) / 2.0
        confidence_factors.append(freshness_confidence * 0.2)

        # Curve consistency confidence (inversion consistent with overall curve shape)
        curve_consistency = self._calculate_curve_consistency(
            curve, front_point, back_point
        )
        confidence_factors.append(curve_consistency * 0.25)

        return sum(confidence_factors)

    def _calculate_curve_consistency(
        self,
        curve: TermStructureCurve,
        front_point: IVDataPoint,
        back_point: IVDataPoint,
    ) -> float:
        """
        Calculate how consistent the inversion is with the overall curve shape.
        """
        if len(curve.curve_points) < 3:
            return 0.5  # Neutral confidence with limited data

        # Calculate average IV slope across the curve
        points = sorted(curve.curve_points, key=lambda x: x.days_to_expiry)
        slopes = []

        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            if p2.days_to_expiry != p1.days_to_expiry:
                slope = (p2.iv - p1.iv) / (p2.days_to_expiry - p1.days_to_expiry)
                slopes.append(slope)

        if not slopes:
            return 0.5

        avg_slope = mean(slopes)

        # Calculate slope for this specific inversion
        if back_point.days_to_expiry != front_point.days_to_expiry:
            inversion_slope = (back_point.iv - front_point.iv) / (
                back_point.days_to_expiry - front_point.days_to_expiry
            )
        else:
            return 0.5

        # If both average and specific slope are negative, it's more consistent
        if avg_slope < 0 and inversion_slope < 0:
            return 0.8  # High consistency
        elif avg_slope > 0 and inversion_slope < 0:
            return 0.3  # Lower consistency (inversion against trend)
        else:
            return 0.6  # Moderate consistency

    def _calculate_inversion_opportunity_score(
        self,
        front_point: IVDataPoint,
        back_point: IVDataPoint,
        inversion_magnitude: float,
        confidence: float,
    ) -> float:
        """
        Calculate overall opportunity score for the term structure inversion.

        Combines inversion magnitude, confidence, and additional trading factors.
        """
        # Inversion magnitude component (0-40%)
        magnitude_score = min(
            1.0, inversion_magnitude / 25.0
        )  # Normalize to 25% inversion

        # Confidence component (0-30%)
        confidence_score = confidence

        # Days to expiry spread component (0-20%)
        days_spread = back_point.days_to_expiry - front_point.days_to_expiry
        optimal_spread = 30  # 30 days is often optimal for calendar spreads
        spread_score = max(
            0.0, 1.0 - abs(days_spread - optimal_spread) / optimal_spread
        )

        # Liquidity component (0-10%)
        avg_volume = (front_point.volume + back_point.volume) / 2.0
        liquidity_score = min(1.0, avg_volume / 50.0)  # Normalize to 50 volume

        # Weighted composite score
        opportunity_score = (
            magnitude_score * 0.4
            + confidence_score * 0.3
            + spread_score * 0.2
            + liquidity_score * 0.1
        )

        return opportunity_score

    def _score_and_filter_inversions(
        self, inversions: List[TermStructureInversion]
    ) -> List[TermStructureInversion]:
        """
        Score and filter inversions based on quality thresholds.
        """
        qualified_inversions = []

        for inversion in inversions:
            # Apply minimum thresholds
            if (
                inversion.confidence_score >= self.config.min_confidence_score
                and inversion.opportunity_score >= self.config.min_opportunity_score
            ):

                qualified_inversions.append(inversion)
            else:
                # Log rejection reason
                if inversion.confidence_score < self.config.min_confidence_score:
                    metrics_collector.add_rejection_reason(
                        RejectionReason.INSUFFICIENT_IV_SPREAD,  # Using closest available reason
                        {
                            "symbol": inversion.symbol,
                            "confidence_score": inversion.confidence_score,
                            "threshold": self.config.min_confidence_score,
                        },
                    )

        # Sort by opportunity score
        qualified_inversions.sort(key=lambda x: x.opportunity_score, reverse=True)

        return qualified_inversions

    def calculate_iv_percentiles(
        self, symbol: str, strike: float, option_type: str, current_iv: float
    ) -> Optional[IVPercentileData]:
        """
        Calculate historical IV percentiles for context.

        In production, this would connect to a historical database.
        For now, it provides a framework for percentile analysis.
        """
        cache_key = f"{symbol}_{strike}_{option_type}"

        # Check cache
        if cache_key in self.percentile_cache:
            percentile_data, cache_time = self.percentile_cache[cache_key]
            if time.time() - cache_time < self.config.cache_ttl:
                return percentile_data

        # Get historical data (placeholder implementation)
        historical_ivs = self._get_historical_iv_data(symbol, strike, option_type)

        if len(historical_ivs) < self.config.min_historical_samples:
            logger.debug(f"Insufficient historical data for {cache_key}")
            return None

        # Calculate percentile rank
        rank = sum(1 for iv in historical_ivs if iv < current_iv) / len(historical_ivs)
        percentile_rank = rank * 100

        # Calculate statistics
        historical_mean = mean(historical_ivs)
        historical_std = stdev(historical_ivs) if len(historical_ivs) > 1 else 0.0

        # Calculate confidence level based on sample size
        confidence_level = min(
            0.95, len(historical_ivs) / 252.0
        )  # Max confidence with 1 year of data

        percentile_data = IVPercentileData(
            symbol=symbol,
            strike=strike,
            option_type=option_type,
            current_iv=current_iv,
            percentile_rank=percentile_rank,
            historical_mean=historical_mean,
            historical_std=historical_std,
            lookback_days=self.config.iv_percentile_lookback,
            confidence_level=confidence_level,
        )

        # Cache the result
        self.percentile_cache[cache_key] = (percentile_data, time.time())

        return percentile_data

    def score_calendar_opportunity(
        self,
        inversion: TermStructureInversion,
        percentile_data: Optional[IVPercentileData] = None,
    ) -> float:
        """
        Score calendar spread opportunity based on term structure analysis.

        Integrates term structure inversion data with percentile analysis
        to provide a comprehensive opportunity score for calendar spreads.
        """
        # Base score from inversion opportunity score
        base_score = inversion.opportunity_score

        # Percentile adjustment
        percentile_adjustment = 0.0
        if percentile_data:
            # Bonus for high IV percentiles (expensive options to sell)
            if percentile_data.percentile_rank > 75:
                percentile_adjustment = (
                    0.1 * (percentile_data.percentile_rank - 75) / 25
                )
            # Penalty for low IV percentiles
            elif percentile_data.percentile_rank < 25:
                percentile_adjustment = (
                    -0.1 * (25 - percentile_data.percentile_rank) / 25
                )

        # Time decay factor (favor shorter front month expiries)
        if inversion.front_days <= 30:
            time_decay_bonus = 0.05
        elif inversion.front_days <= 45:
            time_decay_bonus = 0.02
        else:
            time_decay_bonus = 0.0

        # Inversion strength factor
        if inversion.inversion_magnitude > 20:
            strength_bonus = 0.05
        elif inversion.inversion_magnitude > 15:
            strength_bonus = 0.03
        else:
            strength_bonus = 0.0

        # Calculate final score
        final_score = min(
            1.0, base_score + percentile_adjustment + time_decay_bonus + strength_bonus
        )

        return final_score

    def detect_iv_inversion(
        self, front_iv: float, back_iv: float, front_days: int, back_days: int
    ) -> Tuple[bool, float]:
        """
        Detect IV inversion between two expiration periods.

        Args:
            front_iv: Front month implied volatility
            back_iv: Back month implied volatility
            front_days: Days to front expiry
            back_days: Days to back expiry

        Returns:
            Tuple of (is_inversion, inversion_magnitude)
        """
        if front_days >= back_days:
            return False, 0.0

        if back_iv <= 0:
            return False, 0.0

        # Calculate normalized IVs for fair comparison
        front_iv_normalized = (
            front_iv * np.sqrt(365.0 / front_days) if front_days > 0 else front_iv
        )
        back_iv_normalized = (
            back_iv * np.sqrt(365.0 / back_days) if back_days > 0 else back_iv
        )

        # Check for inversion
        if front_iv_normalized > back_iv_normalized:
            # Handle numerical stability for extreme values
            if (
                back_iv_normalized <= 0
                or not np.isfinite(front_iv_normalized)
                or not np.isfinite(back_iv_normalized)
            ):
                return False, 0.0

            inversion_magnitude = (
                (front_iv_normalized - back_iv_normalized) / back_iv_normalized
            ) * 100

            # Cap magnitude to prevent infinite values
            if not np.isfinite(inversion_magnitude):
                return False, 0.0

            inversion_magnitude = min(inversion_magnitude, 1000.0)  # Cap at 1000%

            if inversion_magnitude >= self.config.min_inversion_threshold:
                return True, inversion_magnitude

        return False, 0.0

    def get_term_structure_summary(
        self, symbol: str
    ) -> Dict[str, Union[int, float, List[Dict]]]:
        """
        Get a summary of term structure analysis for a symbol.

        Returns comprehensive information about IV curves, inversions,
        and opportunity assessments.
        """
        # Get cached curves
        curves = [
            curve
            for key, curve in self.iv_curve_cache.items()
            if key.startswith(symbol)
        ]

        if not curves:
            return {"error": f"No term structure data available for {symbol}"}

        # Analyze inversions across all curves
        inversions = self._detect_term_structure_inversions(curves)
        qualified_inversions = self._score_and_filter_inversions(inversions)

        # Calculate summary statistics
        all_ivs = []
        for curve in curves:
            all_ivs.extend([point.iv for point in curve.curve_points])

        avg_iv = mean(all_ivs) if all_ivs else 0.0
        iv_range = (min(all_ivs), max(all_ivs)) if all_ivs else (0.0, 0.0)

        # Format inversion data
        inversion_summary = []
        for inv in qualified_inversions[:5]:  # Top 5 inversions
            inversion_summary.append(
                {
                    "strike": inv.strike,
                    "option_type": inv.option_type,
                    "front_days": inv.front_days,
                    "back_days": inv.back_days,
                    "iv_differential": round(inv.iv_differential, 2),
                    "inversion_magnitude": round(inv.inversion_magnitude, 2),
                    "opportunity_score": round(inv.opportunity_score, 3),
                }
            )

        return {
            "symbol": symbol,
            "total_curves": len(curves),
            "total_inversions": len(inversions),
            "qualified_inversions": len(qualified_inversions),
            "average_iv": round(avg_iv, 2),
            "iv_range": [round(iv_range[0], 2), round(iv_range[1], 2)],
            "top_inversions": inversion_summary,
            "analysis_timestamp": time.time(),
        }

    # Helper methods

    def _calculate_implied_volatility_cached(
        self, ticker: Ticker, contract: Contract
    ) -> Optional[float]:
        """
        Calculate or retrieve cached implied volatility using IB API data.

        Uses multiple IV sources from IB API with fallback hierarchy:
        1. ticker.impliedVolatility (direct IV from IB)
        2. ticker.modelGreeks.impliedVol (model-based IV)
        3. Average of bid/ask Greeks IV
        4. Fallback to bid-ask spread estimation
        """
        cache_key = f"{contract.conId}_{getattr(ticker, 'time', time.time())}"

        if cache_key in self.iv_calculation_cache:
            iv, cache_time = self.iv_calculation_cache[cache_key]
            if time.time() - cache_time < self.config.cache_ttl:
                return iv

        iv_value = None
        iv_source = "unknown"

        try:
            # Priority 1: Direct implied volatility from IB API
            if (
                hasattr(ticker, "impliedVolatility")
                and ticker.impliedVolatility is not None
            ):
                if (
                    not np.isnan(ticker.impliedVolatility)
                    and ticker.impliedVolatility > 0
                ):
                    iv_value = ticker.impliedVolatility * 100.0  # Convert to percentage
                    iv_source = "direct_iv"

            # Priority 2: Model Greeks implied volatility
            elif hasattr(ticker, "modelGreeks") and ticker.modelGreeks is not None:
                if (
                    hasattr(ticker.modelGreeks, "impliedVol")
                    and ticker.modelGreeks.impliedVol is not None
                ):
                    if (
                        not np.isnan(ticker.modelGreeks.impliedVol)
                        and ticker.modelGreeks.impliedVol > 0
                    ):
                        iv_value = (
                            ticker.modelGreeks.impliedVol * 100.0
                        )  # Convert to percentage
                        iv_source = "model_greeks"

            # Priority 3: Average of bid/ask Greeks IV
            elif (
                hasattr(ticker, "bidGreeks")
                and ticker.bidGreeks is not None
                and hasattr(ticker, "askGreeks")
                and ticker.askGreeks is not None
            ):
                bid_iv = None
                ask_iv = None

                if (
                    hasattr(ticker.bidGreeks, "impliedVol")
                    and ticker.bidGreeks.impliedVol is not None
                    and not np.isnan(ticker.bidGreeks.impliedVol)
                    and ticker.bidGreeks.impliedVol > 0
                ):
                    bid_iv = ticker.bidGreeks.impliedVol * 100.0

                if (
                    hasattr(ticker.askGreeks, "impliedVol")
                    and ticker.askGreeks.impliedVol is not None
                    and not np.isnan(ticker.askGreeks.impliedVol)
                    and ticker.askGreeks.impliedVol > 0
                ):
                    ask_iv = ticker.askGreeks.impliedVol * 100.0

                if bid_iv is not None and ask_iv is not None:
                    iv_value = (bid_iv + ask_iv) / 2.0
                    iv_source = "bid_ask_average"
                elif bid_iv is not None:
                    iv_value = bid_iv
                    iv_source = "bid_greeks"
                elif ask_iv is not None:
                    iv_value = ask_iv
                    iv_source = "ask_greeks"

            # Priority 4: Single Greeks source (bid, ask, or last)
            if iv_value is None:
                for greeks_attr in ["lastGreeks", "bidGreeks", "askGreeks"]:
                    if hasattr(ticker, greeks_attr):
                        greeks = getattr(ticker, greeks_attr)
                        if (
                            greeks is not None
                            and hasattr(greeks, "impliedVol")
                            and greeks.impliedVol is not None
                            and not np.isnan(greeks.impliedVol)
                            and greeks.impliedVol > 0
                        ):
                            iv_value = greeks.impliedVol * 100.0
                            iv_source = greeks_attr.replace("Greeks", "_greeks")
                            break

        except (AttributeError, TypeError) as e:
            logger.debug(f"Error accessing IB IV data for {contract.symbol}: {e}")

        # Fallback to bid-ask spread estimation if no IB IV available
        if iv_value is None:
            if ticker.ask > ticker.bid > 0:
                mid_price = (ticker.ask + ticker.bid) / 2.0
                spread_ratio = (ticker.ask - ticker.bid) / mid_price
                iv_value = min(
                    100.0, max(5.0, spread_ratio * 150.0 + 15.0)
                )  # 5%-100% range
                iv_source = "spread_estimation"
            else:
                iv_value = 20.0  # Default 20% IV
                iv_source = "default"

        # Validate IV is within reasonable range (5% - 200%)
        if iv_value is not None:
            iv_value = max(5.0, min(200.0, iv_value))
        else:
            iv_value = 20.0
            iv_source = "fallback_default"

        # Log IV source for debugging (only at debug level to avoid spam)
        logger.debug(
            f"TermStructure IV for {contract.symbol} strike {contract.strike}: "
            f"{iv_value:.1f}% (source: {iv_source})"
        )

        # Cache the result
        self.iv_calculation_cache[cache_key] = (iv_value, time.time())

        return iv_value

    def _calculate_days_to_expiry(self, expiry_str: str) -> int:
        """Calculate days to expiry from expiry string"""
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y%m%d")
            today = datetime.now().date()
            return (expiry_date.date() - today).days
        except ValueError:
            logger.warning(f"Invalid expiry format: {expiry_str}")
            return 30  # Default assumption

    def _estimate_delta(self, ticker: Ticker, contract: Contract) -> float:
        """Estimate delta for the option (placeholder implementation)"""
        # Simplified delta estimation
        if hasattr(contract, "right"):
            if contract.right == "C":
                return 0.5  # Rough estimate for ATM call
            else:
                return -0.5  # Rough estimate for ATM put
        return 0.0

    def _get_historical_iv_data(
        self, symbol: str, strike: float, option_type: str
    ) -> List[float]:
        """
        Get historical IV data for percentile calculations.

        This is a placeholder implementation. In production, this would
        connect to a historical options database.
        """
        key = f"{symbol}_{strike}_{option_type}"

        # Simulate historical data if not present
        if key not in self.historical_iv_data:
            # Generate synthetic historical data for demonstration
            base_iv = 25.0  # Base IV of 25%
            historical_data = []

            for i in range(self.config.iv_percentile_lookback):
                # Simulate IV with some randomness and mean reversion
                iv = max(5.0, min(80.0, base_iv + np.random.normal(0, 8)))
                timestamp = time.time() - (i * 24 * 3600)  # Daily data
                historical_data.append((timestamp, iv))

            self.historical_iv_data[key] = historical_data

        # Extract IV values
        return [iv for _, iv in self.historical_iv_data[key]]

    def clear_cache(self) -> None:
        """Clear all caches"""
        self.iv_curve_cache.clear()
        self.iv_calculation_cache.clear()
        self.percentile_cache.clear()
        logger.info("Term structure analyzer cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "iv_curves": len(self.iv_curve_cache),
            "iv_calculations": len(self.iv_calculation_cache),
            "percentiles": len(self.percentile_cache),
        }


# Convenience functions for integration with existing calendar spread system


def analyze_symbol_term_structure(
    symbol: str,
    options_data: Dict[int, Ticker],
    config: Optional[TermStructureConfig] = None,
) -> Tuple[List[TermStructureCurve], List[TermStructureInversion]]:
    """
    Convenience function to analyze term structure for a single symbol.

    Args:
        symbol: Trading symbol
        options_data: Options chain data
        config: Analysis configuration

    Returns:
        Tuple of (IV curves, qualified inversions)
    """
    analyzer = TermStructureAnalyzer(config)
    return analyzer.analyze_term_structure(symbol, options_data)


def detect_calendar_spread_opportunities(
    symbol: str,
    options_data: Dict[int, Ticker],
    min_iv_spread: float = 2.0,
    min_confidence: float = 0.7,
) -> List[Dict]:
    """
    Detect calendar spread opportunities using term structure analysis.

    Args:
        symbol: Trading symbol
        options_data: Options chain data
        min_iv_spread: Minimum IV spread threshold
        min_confidence: Minimum confidence threshold

    Returns:
        List of calendar spread opportunity dictionaries
    """
    config = TermStructureConfig(
        min_iv_spread=min_iv_spread, min_confidence_score=min_confidence
    )

    analyzer = TermStructureAnalyzer(config)
    curves, inversions = analyzer.analyze_term_structure(symbol, options_data)

    opportunities = []
    for inversion in inversions:
        # Calculate percentile data
        percentile_data = analyzer.calculate_iv_percentiles(
            symbol, inversion.strike, inversion.option_type, inversion.front_iv
        )

        # Score the opportunity
        calendar_score = analyzer.score_calendar_opportunity(inversion, percentile_data)

        opportunity = {
            "symbol": symbol,
            "strike": inversion.strike,
            "option_type": inversion.option_type,
            "front_expiry": inversion.front_expiry,
            "back_expiry": inversion.back_expiry,
            "front_days": inversion.front_days,
            "back_days": inversion.back_days,
            "iv_spread": inversion.iv_differential,
            "inversion_magnitude": inversion.inversion_magnitude,
            "confidence_score": inversion.confidence_score,
            "calendar_score": calendar_score,
            "iv_percentile": (
                percentile_data.percentile_rank if percentile_data else None
            ),
        }

        opportunities.append(opportunity)

    # Sort by calendar score
    opportunities.sort(key=lambda x: x["calendar_score"], reverse=True)

    return opportunities


# Integration methods for CalendarSpread compatibility


def enhance_calendar_spread_with_term_structure(
    calendar_opportunity,
    options_data: Dict[int, Ticker],
    config: Optional[TermStructureConfig] = None,
):
    """
    Enhance CalendarSpreadOpportunity with advanced term structure analysis.

    Args:
        calendar_opportunity: CalendarSpreadOpportunity instance
        options_data: Options chain data
        config: Term structure configuration

    Returns:
        Enhanced opportunity with term structure metrics
    """
    analyzer = TermStructureAnalyzer(config)

    # Analyze term structure for this symbol
    curves, inversions = analyzer.analyze_term_structure(
        calendar_opportunity.symbol, options_data
    )

    # Find matching inversion for this opportunity
    matching_inversion = None
    for inversion in inversions:
        if (
            inversion.strike == calendar_opportunity.strike
            and inversion.option_type == calendar_opportunity.option_type
            and inversion.front_expiry == calendar_opportunity.front_leg.expiry
            and inversion.back_expiry == calendar_opportunity.back_leg.expiry
        ):
            matching_inversion = inversion
            break

    if matching_inversion:
        # Add term structure metrics to the opportunity
        calendar_opportunity.term_structure_confidence = (
            matching_inversion.confidence_score
        )
        calendar_opportunity.term_structure_magnitude = (
            matching_inversion.inversion_magnitude
        )
        calendar_opportunity.term_structure_opportunity_score = (
            matching_inversion.opportunity_score
        )

        # Calculate enhanced composite score
        original_score = calendar_opportunity.composite_score
        term_structure_bonus = matching_inversion.opportunity_score * 0.15
        calendar_opportunity.composite_score = min(
            1.0, original_score + term_structure_bonus
        )

        logger.info(
            f"Enhanced {calendar_opportunity.symbol} calendar spread with term structure: "
            f"confidence={matching_inversion.confidence_score:.3f}, "
            f"magnitude={matching_inversion.inversion_magnitude:.1f}%, "
            f"score boost={term_structure_bonus:.3f}"
        )

    return calendar_opportunity


def get_optimal_calendar_expiries(
    symbol: str,
    options_data: Dict[int, Ticker],
    target_front_days: int = 30,
    target_back_days: int = 60,
    config: Optional[TermStructureConfig] = None,
) -> List[Tuple[str, str, float]]:
    """
    Get optimal expiry combinations for calendar spreads based on term structure.

    Args:
        symbol: Trading symbol
        options_data: Options chain data
        target_front_days: Target days for front month
        target_back_days: Target days for back month
        config: Term structure configuration

    Returns:
        List of (front_expiry, back_expiry, opportunity_score) tuples
    """
    analyzer = TermStructureAnalyzer(config)
    curves, inversions = analyzer.analyze_term_structure(symbol, options_data)

    # Score expiry combinations
    expiry_scores = []
    for inversion in inversions:
        # Preference for expiries close to targets
        front_days_diff = abs(inversion.front_days - target_front_days)
        back_days_diff = abs(inversion.back_days - target_back_days)

        expiry_score = inversion.opportunity_score * (
            1.0 - (front_days_diff + back_days_diff) / 100.0
        )

        expiry_scores.append(
            (inversion.front_expiry, inversion.back_expiry, max(0.0, expiry_score))
        )

    # Sort by score and return top combinations
    expiry_scores.sort(key=lambda x: x[2], reverse=True)
    return expiry_scores[:10]


def validate_calendar_spread_with_term_structure(
    front_iv: float,
    back_iv: float,
    front_days: int,
    back_days: int,
    min_iv_spread: float = 2.0,
    min_confidence: float = 0.7,
) -> Tuple[bool, Dict[str, Union[float, bool]]]:
    """
    Validate calendar spread using term structure analysis.

    Args:
        front_iv: Front month implied volatility
        back_iv: Back month implied volatility
        front_days: Days to front expiry
        back_days: Days to back expiry
        min_iv_spread: Minimum IV spread threshold
        min_confidence: Minimum confidence threshold

    Returns:
        Tuple of (is_valid, validation_metrics)
    """
    config = TermStructureConfig(
        min_iv_spread=min_iv_spread, min_confidence_score=min_confidence
    )

    analyzer = TermStructureAnalyzer(config)

    # Check for IV inversion
    is_inversion, magnitude = analyzer.detect_iv_inversion(
        front_iv, back_iv, front_days, back_days
    )

    # Calculate basic metrics
    iv_spread = back_iv - front_iv
    iv_spread_pct = (iv_spread / back_iv * 100) if back_iv > 0 else 0.0

    # Validation criteria
    criteria = {
        "has_iv_inversion": is_inversion,
        "iv_spread_sufficient": iv_spread >= min_iv_spread,
        "inversion_magnitude": magnitude,
        "iv_spread_percent": iv_spread_pct,
        "front_iv": front_iv,
        "back_iv": back_iv,
        "normalized_front_iv": (
            front_iv * np.sqrt(365.0 / front_days) if front_days > 0 else front_iv
        ),
        "normalized_back_iv": (
            back_iv * np.sqrt(365.0 / back_days) if back_days > 0 else back_iv
        ),
    }

    # Overall validation
    is_valid = (
        is_inversion
        and iv_spread >= min_iv_spread
        and magnitude >= config.min_inversion_threshold
    )

    return is_valid, criteria


async def analyze_multi_symbol_term_structure(
    symbols: List[str],
    options_data_dict: Dict[str, Dict[int, Ticker]],
    config: Optional[TermStructureConfig] = None,
) -> Dict[str, Dict]:
    """
    Analyze term structure across multiple symbols efficiently.

    Args:
        symbols: List of symbols to analyze
        options_data_dict: Dictionary mapping symbols to options data
        config: Term structure configuration

    Returns:
        Dictionary mapping symbols to term structure analysis results
    """
    analyzer = TermStructureAnalyzer(config)
    results = {}

    for symbol in symbols:
        if symbol not in options_data_dict:
            logger.warning(f"No options data available for {symbol}")
            continue

        try:
            start_time = time.time()

            # Analyze term structure
            curves, inversions = analyzer.analyze_term_structure(
                symbol, options_data_dict[symbol]
            )

            # Get summary
            summary = analyzer.get_term_structure_summary(symbol)

            processing_time = (time.time() - start_time) * 1000

            results[symbol] = {
                "curves": len(curves),
                "inversions": len(inversions),
                "qualified_inversions": len(
                    [
                        inv
                        for inv in inversions
                        if inv.confidence_score >= analyzer.config.min_confidence_score
                    ]
                ),
                "processing_time_ms": processing_time,
                "summary": summary,
                "top_opportunity": (
                    {
                        "strike": inversions[0].strike,
                        "option_type": inversions[0].option_type,
                        "inversion_magnitude": inversions[0].inversion_magnitude,
                        "opportunity_score": inversions[0].opportunity_score,
                    }
                    if inversions
                    else None
                ),
            }

        except Exception as e:
            logger.error(f"Error analyzing term structure for {symbol}: {str(e)}")
            results[symbol] = {"error": str(e)}

    return results
