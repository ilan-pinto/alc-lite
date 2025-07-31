"""
Calendar P&L Calculator module for comprehensive profit/loss analysis of calendar spread positions.

This module provides advanced P&L calculations, breakeven analysis, time decay attribution,
and scenario modeling for calendar spread trading strategies. It integrates with the existing
CalendarSpread and TermStructure implementations to provide comprehensive position analytics.

Key Features:
- Current and projected P&L calculations
- Breakeven point analysis (upside/downside)
- Time decay attribution (theta capture analysis)
- Greeks sensitivity analysis
- Monte Carlo scenario modeling
- Historical performance tracking
- Risk-adjusted return metrics
- Performance optimization with caching

Author: Calendar P&L Analysis System
Version: 1.0.0
"""

import asyncio
import copy
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple, Union

import logging
import numpy as np
from ib_async import Contract, Ticker
from scipy import stats
from scipy.optimize import minimize_scalar

from .CalendarSpread import CalendarSpreadLeg, CalendarSpreadOpportunity
from .common import configure_logging, get_logger
from .metrics import RejectionReason, metrics_collector
from .TermStructure import IVDataPoint, TermStructureAnalyzer

# Configure logging
logger = get_logger()


@dataclass
class CalendarPnLResult:
    """Comprehensive P&L analysis result for calendar spread positions"""

    symbol: str
    strike: float
    option_type: str  # 'CALL' or 'PUT'
    position_size: int

    # Current position data
    initial_debit: float
    current_front_value: float
    current_back_value: float
    current_spread_value: float
    current_pnl: float
    current_pnl_pct: float

    # Time decay analysis
    front_theta: float
    back_theta: float
    net_theta: float
    theta_capture_daily: float
    theta_capture_total: float

    # Maximum profit analysis
    estimated_max_profit: float
    estimated_max_profit_pct: float
    max_profit_price: float
    days_to_max_profit: int

    # Greeks exposure
    net_delta: float
    net_gamma: float
    net_vega: float
    net_rho: float

    # Risk metrics
    breakeven_upside: Optional[float]
    breakeven_downside: Optional[float]
    probability_of_profit: float
    max_loss: float
    risk_reward_ratio: float

    # Timing metrics
    days_in_position: int
    days_to_front_expiry: int
    days_to_back_expiry: int
    time_decay_acceleration: float

    # Performance attribution
    price_pnl: float
    time_pnl: float
    volatility_pnl: float
    other_pnl: float

    # Analysis metadata
    confidence_score: float
    data_quality_score: float
    last_updated: float = field(default_factory=time.time)


@dataclass
class BreakevenPoints:
    """Breakeven analysis for calendar spread positions"""

    upside_breakeven: Optional[float]
    downside_breakeven: Optional[float]
    breakeven_range: Optional[float]
    current_stock_price: float

    # Probability analysis
    prob_above_upside: float
    prob_below_downside: float
    prob_within_range: float

    # Time sensitivity
    breakeven_at_expiry: Tuple[Optional[float], Optional[float]]
    breakeven_halfway: Tuple[Optional[float], Optional[float]]

    # Confidence metrics
    calculation_method: str
    confidence_level: float
    margin_of_error: float


@dataclass
class ThetaAnalysis:
    """Time decay analysis for calendar spread positions"""

    front_theta_daily: float
    back_theta_daily: float
    net_theta_daily: float

    # Theta capture metrics
    theta_capture_rate: float  # Net theta as % of premium collected
    optimal_theta_rate: float  # Theoretical optimal rate
    theta_efficiency: float  # Actual vs optimal theta capture

    # Time decay projections
    projected_theta_7d: float
    projected_theta_14d: float
    projected_theta_30d: float

    # Acceleration analysis
    theta_acceleration: float  # Rate of theta increase
    peak_theta_date: Optional[datetime]

    # Attribution
    front_contribution: float
    back_contribution: float
    cross_gamma_effect: float


@dataclass
class PnLScenario:
    """Individual P&L scenario for stress testing"""

    scenario_name: str
    stock_price: float
    stock_price_change_pct: float
    days_forward: int

    # Market condition changes
    iv_change_front: float  # Percentage change in front month IV
    iv_change_back: float  # Percentage change in back month IV
    interest_rate_change: float

    # Projected values
    projected_front_value: float
    projected_back_value: float
    projected_spread_value: float
    projected_pnl: float
    projected_pnl_pct: float

    # Greeks impact
    delta_pnl: float
    gamma_pnl: float
    theta_pnl: float
    vega_pnl: float
    rho_pnl: float

    # Probability and confidence
    scenario_probability: float
    confidence_interval: Tuple[float, float]


@dataclass
class MonteCarloResults:
    """Monte Carlo simulation results for calendar spread P&L"""

    num_simulations: int
    simulation_days: int

    # P&L distribution
    mean_pnl: float
    median_pnl: float
    std_pnl: float
    min_pnl: float
    max_pnl: float

    # Probability metrics
    prob_profit: float
    prob_max_profit: float
    prob_loss_gt_50pct: float

    # Percentile analysis
    pnl_5th_percentile: float
    pnl_25th_percentile: float
    pnl_75th_percentile: float
    pnl_95th_percentile: float

    # Risk metrics
    expected_shortfall: float  # Expected loss in worst 5% scenarios
    value_at_risk_95: float  # VaR at 95% confidence
    sharpe_ratio: float

    # Distribution characteristics
    skewness: float
    kurtosis: float
    is_normal_distribution: bool


@dataclass
class CalendarPnLConfig:
    """Configuration for calendar P&L calculations"""

    # Calculation parameters
    risk_free_rate: float = 0.05  # 5% annual risk-free rate
    dividend_yield: float = 0.0  # Assume no dividends unless specified

    # Monte Carlo settings
    monte_carlo_simulations: int = 10000
    max_simulation_days: int = 60
    stock_volatility: float = 0.25  # Default 25% annual volatility

    # Greeks calculation settings
    price_bump_size: float = 0.01  # 1% for delta/gamma calculation
    vol_bump_size: float = 0.01  # 1% for vega calculation
    time_bump_size: float = 1.0  # 1 day for theta calculation

    # Breakeven calculation settings
    breakeven_tolerance: float = 0.01  # $0.01 tolerance for breakeven
    max_breakeven_iterations: int = 100

    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: float = 300.0  # 5 minute cache TTL
    enable_parallel_processing: bool = True
    max_workers: int = 4

    # Quality thresholds
    min_data_quality_score: float = 0.7
    min_confidence_score: float = 0.6


class CalendarPnLCalculator:
    """
    Advanced Calendar Spread P&L Calculator.

    This class provides comprehensive profit/loss analysis for calendar spread positions,
    including current P&L, breakeven analysis, time decay attribution, Greeks sensitivity,
    and Monte Carlo scenario modeling.

    Key Capabilities:
    1. Real-time P&L calculation and attribution
    2. Breakeven point analysis with probability distributions
    3. Time decay capture analysis and optimization
    4. Greeks-based sensitivity analysis
    5. Monte Carlo simulation for risk assessment
    6. Performance optimization with intelligent caching
    """

    def __init__(self, config: Optional[CalendarPnLConfig] = None) -> None:
        """
        Initialize the Calendar P&L Calculator.

        Args:
            config: Configuration for P&L calculations
        """
        self.config = config or CalendarPnLConfig()

        # Caching for performance optimization
        self.pnl_cache: Dict[str, Tuple[CalendarPnLResult, float]] = {}
        self.greeks_cache: Dict[str, Tuple[Dict, float]] = {}
        self.breakeven_cache: Dict[str, Tuple[BreakevenPoints, float]] = {}
        self.monte_carlo_cache: Dict[str, Tuple[MonteCarloResults, float]] = {}

        # Thread pool for parallel processing
        self.executor = (
            ThreadPoolExecutor(max_workers=self.config.max_workers)
            if self.config.enable_parallel_processing
            else None
        )

        # Term structure analyzer for IV projections
        self.term_structure_analyzer = TermStructureAnalyzer()

        logger.info("Calendar P&L Calculator initialized")

    def calculate_calendar_pnl(
        self,
        opportunity: CalendarSpreadOpportunity,
        current_stock_price: float,
        position_size: int = 1,
        entry_date: Optional[datetime] = None,
    ) -> CalendarPnLResult:
        """
        Main P&L calculation method for calendar spread positions.

        This method calculates comprehensive P&L metrics including current P&L,
        projected maximum profit, breakeven points, and time decay attribution.

        Args:
            opportunity: Calendar spread opportunity data
            current_stock_price: Current underlying stock price
            position_size: Number of calendar spreads (positive for long positions)
            entry_date: Date when position was entered (defaults to today)

        Returns:
            CalendarPnLResult with comprehensive P&L analysis
        """
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._generate_pnl_cache_key(
                opportunity, current_stock_price, position_size
            )

            if self.config.enable_caching and cache_key in self.pnl_cache:
                cached_result, cache_time = self.pnl_cache[cache_key]
                if time.time() - cache_time < self.config.cache_ttl_seconds:
                    logger.debug(f"Using cached P&L result for {opportunity.symbol}")
                    return cached_result

            # Initialize calculation parameters
            entry_date = entry_date or datetime.now()
            days_in_position = (datetime.now() - entry_date).days

            # Current position values
            current_front_value = opportunity.front_leg.price * 100 * position_size
            current_back_value = opportunity.back_leg.price * 100 * position_size
            current_spread_value = current_back_value - current_front_value

            # P&L calculations
            initial_debit = opportunity.net_debit * 100 * position_size
            current_pnl = current_spread_value - initial_debit
            current_pnl_pct = (
                (current_pnl / initial_debit * 100) if initial_debit > 0 else 0.0
            )

            # Time decay analysis
            theta_analysis = self._calculate_theta_analysis(opportunity, position_size)

            # Maximum profit estimation
            max_profit_analysis = self._calculate_max_profit(
                opportunity, current_stock_price, position_size
            )

            # Greeks calculation
            greeks = self._calculate_position_greeks(
                opportunity, current_stock_price, position_size
            )

            # Breakeven analysis
            breakeven_points = self.calculate_breakeven_points(
                opportunity, current_stock_price, position_size
            )

            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(
                opportunity, current_pnl, initial_debit, breakeven_points
            )

            # P&L attribution
            attribution = self._calculate_pnl_attribution(
                opportunity, current_pnl, days_in_position, current_stock_price
            )

            # Time decay acceleration
            acceleration = self._calculate_time_decay_acceleration(opportunity)

            # Data quality assessment
            data_quality = self._assess_data_quality(opportunity)
            confidence_score = self._calculate_confidence_score(
                opportunity, data_quality
            )

            # Create comprehensive result
            result = CalendarPnLResult(
                symbol=opportunity.symbol,
                strike=opportunity.strike,
                option_type=opportunity.option_type,
                position_size=position_size,
                # Current position data
                initial_debit=initial_debit / 100,  # Convert back to per-contract basis
                current_front_value=current_front_value / 100,
                current_back_value=current_back_value / 100,
                current_spread_value=current_spread_value / 100,
                current_pnl=current_pnl / 100,
                current_pnl_pct=current_pnl_pct,
                # Time decay
                front_theta=opportunity.front_leg.theta,
                back_theta=opportunity.back_leg.theta,
                net_theta=theta_analysis.net_theta_daily,
                theta_capture_daily=theta_analysis.theta_capture_rate,
                theta_capture_total=theta_analysis.projected_theta_30d,
                # Maximum profit
                estimated_max_profit=max_profit_analysis["max_profit"] / 100,
                estimated_max_profit_pct=max_profit_analysis["max_profit_pct"],
                max_profit_price=max_profit_analysis["optimal_price"],
                days_to_max_profit=max_profit_analysis["days_to_optimal"],
                # Greeks
                net_delta=greeks["delta"],
                net_gamma=greeks["gamma"],
                net_vega=greeks["vega"],
                net_rho=greeks["rho"],
                # Risk metrics
                breakeven_upside=breakeven_points.upside_breakeven,
                breakeven_downside=breakeven_points.downside_breakeven,
                probability_of_profit=risk_metrics["prob_profit"],
                max_loss=opportunity.max_loss * position_size,
                risk_reward_ratio=risk_metrics["risk_reward_ratio"],
                # Timing
                days_in_position=days_in_position,
                days_to_front_expiry=opportunity.front_leg.days_to_expiry,
                days_to_back_expiry=opportunity.back_leg.days_to_expiry,
                time_decay_acceleration=acceleration,
                # Attribution
                price_pnl=attribution["price_pnl"] / 100,
                time_pnl=attribution["time_pnl"] / 100,
                volatility_pnl=attribution["volatility_pnl"] / 100,
                other_pnl=attribution["other_pnl"] / 100,
                # Quality metrics
                confidence_score=confidence_score,
                data_quality_score=data_quality,
            )

            # Cache the result
            if self.config.enable_caching:
                self.pnl_cache[cache_key] = (result, time.time())

            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"Calculated P&L for {opportunity.symbol} calendar spread: "
                f"Current P&L: ${result.current_pnl:.2f} ({result.current_pnl_pct:.1f}%), "
                f"Max profit: ${result.estimated_max_profit:.2f} "
                f"({processing_time:.1f}ms)"
            )

            return result

        except Exception as e:
            logger.error(
                f"Error calculating calendar P&L for {opportunity.symbol}: {str(e)}"
            )
            raise

    def calculate_max_profit(
        self,
        opportunity: CalendarSpreadOpportunity,
        current_stock_price: float,
        position_size: int = 1,
    ) -> Dict[str, float]:
        """
        Calculate theoretical maximum profit for calendar spread.

        Maximum profit typically occurs when the stock price equals the strike price
        at front month expiration, allowing the front option to expire worthless
        while the back option retains maximum time value.

        Args:
            opportunity: Calendar spread opportunity
            current_stock_price: Current stock price
            position_size: Position size

        Returns:
            Dictionary with max profit analysis
        """
        try:
            # Time to front expiry in years
            time_to_front = opportunity.front_leg.days_to_expiry / 365.0
            time_to_back = opportunity.back_leg.days_to_expiry / 365.0

            # At front expiry, back option will have remaining time value
            remaining_time = time_to_back - time_to_front

            if remaining_time <= 0:
                logger.warning("Invalid expiry structure for max profit calculation")
                return {
                    "max_profit": 0.0,
                    "max_profit_pct": 0.0,
                    "optimal_price": opportunity.strike,
                    "days_to_optimal": opportunity.front_leg.days_to_expiry,
                }

            # Estimate back option value at front expiry (at strike price)
            # Using simplified Black-Scholes approximation
            strike = opportunity.strike
            back_iv = opportunity.back_leg.iv / 100.0  # Convert percentage to decimal

            # At-the-money time value approximation
            estimated_back_value = self._estimate_option_time_value(
                strike,
                strike,
                remaining_time,
                back_iv,
                opportunity.option_type,
                self.config.risk_free_rate,
            )

            # Front option expires worthless at strike
            front_value_at_expiry = 0.0

            # Net spread value at expiry
            spread_value_at_expiry = estimated_back_value - front_value_at_expiry

            # Maximum profit calculation
            initial_debit = opportunity.net_debit
            max_profit = (spread_value_at_expiry - initial_debit) * position_size
            max_profit_pct = (
                (max_profit / (initial_debit * position_size) * 100)
                if initial_debit > 0
                else 0.0
            )

            return {
                "max_profit": max_profit * 100,  # Convert to dollar terms
                "max_profit_pct": max_profit_pct,
                "optimal_price": strike,
                "days_to_optimal": opportunity.front_leg.days_to_expiry,
                "estimated_back_value": estimated_back_value,
                "spread_value_at_expiry": spread_value_at_expiry,
            }

        except Exception as e:
            logger.error(f"Error calculating max profit: {str(e)}")
            return {
                "max_profit": 0.0,
                "max_profit_pct": 0.0,
                "optimal_price": opportunity.strike,
                "days_to_optimal": opportunity.front_leg.days_to_expiry,
            }

    def calculate_breakeven_points(
        self,
        opportunity: CalendarSpreadOpportunity,
        current_stock_price: float,
        position_size: int = 1,
    ) -> BreakevenPoints:
        """
        Calculate breakeven points for calendar spread position.

        Breakeven occurs where the spread value equals the initial debit paid.
        For calendar spreads, there are typically two breakeven points due to
        the curved P&L profile.

        Args:
            opportunity: Calendar spread opportunity
            current_stock_price: Current stock price
            position_size: Position size

        Returns:
            BreakevenPoints with upside/downside analysis
        """
        try:
            cache_key = f"breakeven_{opportunity.symbol}_{opportunity.strike}_{opportunity.option_type}_{current_stock_price}"

            if self.config.enable_caching and cache_key in self.breakeven_cache:
                cached_result, cache_time = self.breakeven_cache[cache_key]
                if time.time() - cache_time < self.config.cache_ttl_seconds:
                    return cached_result

            # Initial debit is the breakeven spread value we need
            target_spread_value = opportunity.net_debit

            # Search for breakeven points around the strike
            strike = opportunity.strike
            search_range = strike * 0.3  # Search within 30% of strike

            # Find downside breakeven (below strike)
            downside_breakeven = None
            upside_breakeven = None

            # Use optimization to find breakeven points
            def spread_value_function(stock_price):
                """Calculate spread value at given stock price at front expiry"""
                return abs(
                    self._estimate_spread_value_at_expiry(opportunity, stock_price)
                    - target_spread_value
                )

            # Search for downside breakeven
            try:
                downside_result = minimize_scalar(
                    spread_value_function,
                    bounds=(strike - search_range, strike - 0.01),
                    method="bounded",
                )
                if (
                    downside_result.success
                    and downside_result.fun < self.config.breakeven_tolerance
                ):
                    downside_breakeven = downside_result.x
            except:
                pass

            # Search for upside breakeven
            try:
                upside_result = minimize_scalar(
                    spread_value_function,
                    bounds=(strike + 0.01, strike + search_range),
                    method="bounded",
                )
                if (
                    upside_result.success
                    and upside_result.fun < self.config.breakeven_tolerance
                ):
                    upside_breakeven = upside_result.x
            except:
                pass

            # Calculate breakeven range
            breakeven_range = None
            if upside_breakeven and downside_breakeven:
                breakeven_range = upside_breakeven - downside_breakeven

            # Probability analysis (simplified normal distribution assumption)
            prob_above_upside = 0.0
            prob_below_downside = 0.0
            prob_within_range = 0.0

            if upside_breakeven or downside_breakeven:
                # Estimate stock volatility and time to expiry
                time_to_expiry = opportunity.front_leg.days_to_expiry / 365.0
                annual_vol = self.config.stock_volatility

                # Standard deviation of stock price at expiry
                price_std = current_stock_price * annual_vol * np.sqrt(time_to_expiry)

                if upside_breakeven:
                    prob_above_upside = 1 - stats.norm.cdf(
                        (upside_breakeven - current_stock_price) / price_std
                    )

                if downside_breakeven:
                    prob_below_downside = stats.norm.cdf(
                        (downside_breakeven - current_stock_price) / price_std
                    )

                prob_within_range = 1 - prob_above_upside - prob_below_downside

            # Calculate breakeven points at different time periods
            breakeven_at_expiry = (downside_breakeven, upside_breakeven)

            # Estimate breakeven halfway to expiry (simplified)
            halfway_days = opportunity.front_leg.days_to_expiry // 2
            breakeven_halfway = self._estimate_breakeven_at_time(
                opportunity, current_stock_price, halfway_days
            )

            result = BreakevenPoints(
                upside_breakeven=upside_breakeven,
                downside_breakeven=downside_breakeven,
                breakeven_range=breakeven_range,
                current_stock_price=current_stock_price,
                # Probabilities
                prob_above_upside=prob_above_upside,
                prob_below_downside=prob_below_downside,
                prob_within_range=prob_within_range,
                # Time sensitivity
                breakeven_at_expiry=breakeven_at_expiry,
                breakeven_halfway=breakeven_halfway,
                # Confidence
                calculation_method="optimization_search",
                confidence_level=0.8,  # Moderate confidence in numerical method
                margin_of_error=self.config.breakeven_tolerance,
            )

            # Cache the result
            if self.config.enable_caching:
                self.breakeven_cache[cache_key] = (result, time.time())

            logger.debug(
                f"Calculated breakeven points for {opportunity.symbol}: "
                f"Downside: ${downside_breakeven:.2f if downside_breakeven else 'N/A'}, "
                f"Upside: ${upside_breakeven:.2f if upside_breakeven else 'N/A'}"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating breakeven points: {str(e)}")
            return BreakevenPoints(
                upside_breakeven=None,
                downside_breakeven=None,
                breakeven_range=None,
                current_stock_price=current_stock_price,
                prob_above_upside=0.0,
                prob_below_downside=0.0,
                prob_within_range=0.0,
                breakeven_at_expiry=(None, None),
                breakeven_halfway=(None, None),
                calculation_method="error",
                confidence_level=0.0,
                margin_of_error=1.0,
            )

    def calculate_theta_capture(
        self, opportunity: CalendarSpreadOpportunity, position_size: int = 1
    ) -> ThetaAnalysis:
        """
        Calculate comprehensive time decay attribution for calendar spread.

        Analyzes how time decay affects the position, including theta capture
        efficiency and projected time decay over various periods.

        Args:
            opportunity: Calendar spread opportunity
            position_size: Position size

        Returns:
            ThetaAnalysis with comprehensive time decay metrics
        """
        try:
            # Base theta values from the opportunity
            front_theta_daily = opportunity.front_leg.theta * position_size
            back_theta_daily = opportunity.back_leg.theta * position_size
            net_theta_daily = (
                back_theta_daily - front_theta_daily
            )  # We're long back, short front

            # Theta capture metrics
            initial_premium = opportunity.net_debit * position_size
            theta_capture_rate = (
                (net_theta_daily / initial_premium * 100)
                if initial_premium > 0
                else 0.0
            )

            # Optimal theta rate (theoretical maximum based on time to expiry)
            time_to_front = opportunity.front_leg.days_to_expiry
            optimal_daily_decay = (
                initial_premium / time_to_front if time_to_front > 0 else 0.0
            )
            optimal_theta_rate = (
                (optimal_daily_decay / initial_premium * 100)
                if initial_premium > 0
                else 0.0
            )

            # Theta efficiency
            theta_efficiency = (
                (theta_capture_rate / optimal_theta_rate)
                if optimal_theta_rate > 0
                else 0.0
            )

            # Time decay projections
            projected_theta_7d = net_theta_daily * 7
            projected_theta_14d = net_theta_daily * 14
            projected_theta_30d = net_theta_daily * min(
                30, opportunity.front_leg.days_to_expiry
            )

            # Theta acceleration calculation
            # Theta accelerates as expiry approaches (gamma * vol^2 / 2)
            days_to_front = opportunity.front_leg.days_to_expiry
            if days_to_front > 1:
                # Simplified theta acceleration based on time to expiry
                acceleration_factor = (
                    1.0 + (30 - days_to_front) / 30.0 if days_to_front < 30 else 1.0
                )
                theta_acceleration = acceleration_factor - 1.0
            else:
                theta_acceleration = 2.0  # High acceleration very close to expiry

            # Peak theta date estimation
            peak_theta_date = None
            if days_to_front > 7:
                # Peak theta typically occurs 1-2 weeks before expiry
                peak_theta_date = datetime.now() + timedelta(days=days_to_front - 14)

            # Attribution analysis
            # Front contribution (negative for short position)
            front_contribution = -front_theta_daily  # Negative because we're short

            # Back contribution (positive for long position)
            back_contribution = back_theta_daily

            # Cross-gamma effect (interaction between front and back gamma)
            front_gamma = self._estimate_gamma(opportunity.front_leg)
            back_gamma = self._estimate_gamma(opportunity.back_leg)
            cross_gamma_effect = (back_gamma - front_gamma) * 0.1  # Simplified estimate

            return ThetaAnalysis(
                front_theta_daily=front_theta_daily,
                back_theta_daily=back_theta_daily,
                net_theta_daily=net_theta_daily,
                # Capture metrics
                theta_capture_rate=theta_capture_rate,
                optimal_theta_rate=optimal_theta_rate,
                theta_efficiency=min(100.0, max(0.0, theta_efficiency * 100)),
                # Projections
                projected_theta_7d=projected_theta_7d,
                projected_theta_14d=projected_theta_14d,
                projected_theta_30d=projected_theta_30d,
                # Acceleration
                theta_acceleration=theta_acceleration,
                peak_theta_date=peak_theta_date,
                # Attribution
                front_contribution=front_contribution,
                back_contribution=back_contribution,
                cross_gamma_effect=cross_gamma_effect,
            )

        except Exception as e:
            logger.error(f"Error calculating theta analysis: {str(e)}")
            return ThetaAnalysis(
                front_theta_daily=0.0,
                back_theta_daily=0.0,
                net_theta_daily=0.0,
                theta_capture_rate=0.0,
                optimal_theta_rate=0.0,
                theta_efficiency=0.0,
                projected_theta_7d=0.0,
                projected_theta_14d=0.0,
                projected_theta_30d=0.0,
                theta_acceleration=0.0,
                peak_theta_date=None,
                front_contribution=0.0,
                back_contribution=0.0,
                cross_gamma_effect=0.0,
            )

    def model_pnl_scenarios(
        self,
        opportunity: CalendarSpreadOpportunity,
        current_stock_price: float,
        position_size: int = 1,
        custom_scenarios: Optional[List[Dict]] = None,
    ) -> List[PnLScenario]:
        """
        Model P&L across various price and time scenarios.

        Creates a comprehensive set of scenarios to stress-test the calendar
        spread position under different market conditions.

        Args:
            opportunity: Calendar spread opportunity
            current_stock_price: Current stock price
            position_size: Position size
            custom_scenarios: Optional custom scenarios to include

        Returns:
            List of PnLScenario objects with projected outcomes
        """
        try:
            scenarios = []

            # Standard scenario set
            standard_scenarios = [
                # Time decay scenarios (stock stays near current price)
                {
                    "name": "7_days_flat",
                    "price_change": 0.0,
                    "days": 7,
                    "iv_change_front": 0.0,
                    "iv_change_back": 0.0,
                },
                {
                    "name": "14_days_flat",
                    "price_change": 0.0,
                    "days": 14,
                    "iv_change_front": 0.0,
                    "iv_change_back": 0.0,
                },
                {
                    "name": "30_days_flat",
                    "price_change": 0.0,
                    "days": 30,
                    "iv_change_front": 0.0,
                    "iv_change_back": 0.0,
                },
                # Price movement scenarios (1 week forward)
                {
                    "name": "up_5pct_7d",
                    "price_change": 0.05,
                    "days": 7,
                    "iv_change_front": 0.0,
                    "iv_change_back": 0.0,
                },
                {
                    "name": "up_10pct_7d",
                    "price_change": 0.10,
                    "days": 7,
                    "iv_change_front": 0.0,
                    "iv_change_back": 0.0,
                },
                {
                    "name": "down_5pct_7d",
                    "price_change": -0.05,
                    "days": 7,
                    "iv_change_front": 0.0,
                    "iv_change_back": 0.0,
                },
                {
                    "name": "down_10pct_7d",
                    "price_change": -0.10,
                    "days": 7,
                    "iv_change_front": 0.0,
                    "iv_change_back": 0.0,
                },
                # Volatility scenarios
                {
                    "name": "vol_crush_7d",
                    "price_change": 0.0,
                    "days": 7,
                    "iv_change_front": -20.0,
                    "iv_change_back": -20.0,
                },
                {
                    "name": "vol_expansion_7d",
                    "price_change": 0.0,
                    "days": 7,
                    "iv_change_front": 20.0,
                    "iv_change_back": 20.0,
                },
                {
                    "name": "skew_change_7d",
                    "price_change": 0.0,
                    "days": 7,
                    "iv_change_front": -10.0,
                    "iv_change_back": 5.0,
                },
                # Combined scenarios
                {
                    "name": "up_10pct_vol_crush",
                    "price_change": 0.10,
                    "days": 7,
                    "iv_change_front": -15.0,
                    "iv_change_back": -15.0,
                },
                {
                    "name": "down_10pct_vol_expansion",
                    "price_change": -0.10,
                    "days": 7,
                    "iv_change_front": 15.0,
                    "iv_change_back": 15.0,
                },
                # Expiration scenarios
                {
                    "name": "at_strike_expiry",
                    "price_change": (opportunity.strike - current_stock_price)
                    / current_stock_price,
                    "days": opportunity.front_leg.days_to_expiry,
                    "iv_change_front": 0.0,
                    "iv_change_back": 0.0,
                },
            ]

            # Add custom scenarios if provided
            if custom_scenarios:
                standard_scenarios.extend(custom_scenarios)

            # Process each scenario
            for scenario_data in standard_scenarios:
                scenario = self._calculate_scenario_pnl(
                    opportunity, current_stock_price, position_size, scenario_data
                )
                scenarios.append(scenario)

            # Sort scenarios by expected P&L
            scenarios.sort(key=lambda x: x.projected_pnl, reverse=True)

            logger.info(
                f"Generated {len(scenarios)} P&L scenarios for {opportunity.symbol} calendar spread"
            )

            return scenarios

        except Exception as e:
            logger.error(f"Error modeling P&L scenarios: {str(e)}")
            return []

    def run_monte_carlo_simulation(
        self,
        opportunity: CalendarSpreadOpportunity,
        current_stock_price: float,
        position_size: int = 1,
        simulation_days: Optional[int] = None,
    ) -> MonteCarloResults:
        """
        Run Monte Carlo simulation for calendar spread P&L distribution.

        Simulates thousands of possible outcomes based on random stock price
        movements and volatility changes to assess risk and return distribution.

        Args:
            opportunity: Calendar spread opportunity
            current_stock_price: Current stock price
            position_size: Position size
            simulation_days: Days to simulate (defaults to front expiry)

        Returns:
            MonteCarloResults with comprehensive distribution analysis
        """
        try:
            cache_key = f"monte_carlo_{opportunity.symbol}_{opportunity.strike}_{current_stock_price}_{simulation_days}"

            if self.config.enable_caching and cache_key in self.monte_carlo_cache:
                cached_result, cache_time = self.monte_carlo_cache[cache_key]
                if (
                    time.time() - cache_time < self.config.cache_ttl_seconds * 2
                ):  # Longer cache for MC
                    return cached_result

            # Simulation parameters
            simulation_days = simulation_days or opportunity.front_leg.days_to_expiry
            simulation_days = min(simulation_days, self.config.max_simulation_days)
            num_sims = self.config.monte_carlo_simulations

            # Market parameters
            annual_vol = self.config.stock_volatility
            time_step = simulation_days / 365.0

            # Run simulations
            pnl_results = []
            initial_debit = opportunity.net_debit * position_size

            logger.info(
                f"Running {num_sims} Monte Carlo simulations for {opportunity.symbol}"
            )

            # Vectorized simulation for performance
            np.random.seed(42)  # For reproducible results

            # Generate random price paths
            z_price = np.random.normal(0, 1, num_sims)
            z_vol_front = np.random.normal(0, 1, num_sims)
            z_vol_back = np.random.normal(0, 1, num_sims)

            for i in range(num_sims):
                # Simulate stock price movement (geometric Brownian motion)
                drift = (self.config.risk_free_rate - 0.5 * annual_vol**2) * time_step
                diffusion = annual_vol * np.sqrt(time_step) * z_price[i]
                final_stock_price = current_stock_price * np.exp(drift + diffusion)

                # Simulate IV changes (mean-reverting)
                iv_vol = 0.3  # Volatility of volatility
                front_iv_change = z_vol_front[i] * iv_vol * np.sqrt(time_step) * 100
                back_iv_change = z_vol_back[i] * iv_vol * np.sqrt(time_step) * 100

                # Calculate P&L for this simulation
                scenario_data = {
                    "name": f"sim_{i}",
                    "price_change": (final_stock_price - current_stock_price)
                    / current_stock_price,
                    "days": simulation_days,
                    "iv_change_front": front_iv_change,
                    "iv_change_back": back_iv_change,
                }

                scenario = self._calculate_scenario_pnl(
                    opportunity, current_stock_price, position_size, scenario_data
                )

                pnl_results.append(scenario.projected_pnl)

            # Analyze results
            pnl_array = np.array(pnl_results)

            # Basic statistics
            mean_pnl = np.mean(pnl_array)
            median_pnl = np.median(pnl_array)
            std_pnl = np.std(pnl_array)
            min_pnl = np.min(pnl_array)
            max_pnl = np.max(pnl_array)

            # Probability metrics
            prob_profit = np.sum(pnl_array > 0) / num_sims
            prob_max_profit = (
                np.sum(pnl_array > initial_debit * 0.5) / num_sims
            )  # 50% of max possible
            prob_loss_gt_50pct = np.sum(pnl_array < -initial_debit * 0.5) / num_sims

            # Percentiles
            percentiles = np.percentile(pnl_array, [5, 25, 75, 95])

            # Risk metrics
            losses = pnl_array[pnl_array < 0]
            expected_shortfall = np.mean(losses) if len(losses) > 0 else 0.0
            value_at_risk_95 = np.percentile(pnl_array, 5)

            # Sharpe ratio (risk-adjusted return)
            risk_free_daily = self.config.risk_free_rate / 365 * simulation_days
            excess_return = mean_pnl - risk_free_daily
            sharpe_ratio = excess_return / std_pnl if std_pnl > 0 else 0.0

            # Distribution characteristics
            skewness = stats.skew(pnl_array)
            kurtosis = stats.kurtosis(pnl_array)

            # Normality test
            _, p_value = stats.normaltest(pnl_array)
            is_normal = p_value > 0.05

            results = MonteCarloResults(
                num_simulations=num_sims,
                simulation_days=simulation_days,
                # Distribution
                mean_pnl=mean_pnl,
                median_pnl=median_pnl,
                std_pnl=std_pnl,
                min_pnl=min_pnl,
                max_pnl=max_pnl,
                # Probabilities
                prob_profit=prob_profit,
                prob_max_profit=prob_max_profit,
                prob_loss_gt_50pct=prob_loss_gt_50pct,
                # Percentiles
                pnl_5th_percentile=percentiles[0],
                pnl_25th_percentile=percentiles[1],
                pnl_75th_percentile=percentiles[2],
                pnl_95th_percentile=percentiles[3],
                # Risk metrics
                expected_shortfall=expected_shortfall,
                value_at_risk_95=value_at_risk_95,
                sharpe_ratio=sharpe_ratio,
                # Distribution characteristics
                skewness=skewness,
                kurtosis=kurtosis,
                is_normal_distribution=is_normal,
            )

            # Cache results
            if self.config.enable_caching:
                self.monte_carlo_cache[cache_key] = (results, time.time())

            logger.info(
                f"Monte Carlo simulation completed for {opportunity.symbol}: "
                f"Mean P&L: ${mean_pnl:.2f}, Prob of profit: {prob_profit:.1%}, "
                f"VaR(95%): ${value_at_risk_95:.2f}"
            )

            return results

        except Exception as e:
            logger.error(f"Error running Monte Carlo simulation: {str(e)}")
            return MonteCarloResults(
                num_simulations=0,
                simulation_days=0,
                mean_pnl=0.0,
                median_pnl=0.0,
                std_pnl=0.0,
                min_pnl=0.0,
                max_pnl=0.0,
                prob_profit=0.0,
                prob_max_profit=0.0,
                prob_loss_gt_50pct=0.0,
                pnl_5th_percentile=0.0,
                pnl_25th_percentile=0.0,
                pnl_75th_percentile=0.0,
                pnl_95th_percentile=0.0,
                expected_shortfall=0.0,
                value_at_risk_95=0.0,
                sharpe_ratio=0.0,
                skewness=0.0,
                kurtosis=0.0,
                is_normal_distribution=False,
            )

    # Helper methods

    def _generate_pnl_cache_key(
        self, opportunity: CalendarSpreadOpportunity, stock_price: float, size: int
    ) -> str:
        """Generate cache key for P&L calculations"""
        return f"pnl_{opportunity.symbol}_{opportunity.strike}_{opportunity.option_type}_{stock_price:.2f}_{size}"

    def _calculate_theta_analysis(
        self, opportunity: CalendarSpreadOpportunity, position_size: int
    ) -> ThetaAnalysis:
        """Calculate comprehensive theta analysis"""
        return self.calculate_theta_capture(opportunity, position_size)

    def _calculate_max_profit(
        self, opportunity: CalendarSpreadOpportunity, stock_price: float, size: int
    ) -> Dict[str, float]:
        """Calculate maximum profit analysis"""
        return self.calculate_max_profit(opportunity, stock_price, size)

    def _calculate_position_greeks(
        self, opportunity: CalendarSpreadOpportunity, stock_price: float, size: int
    ) -> Dict[str, float]:
        """Calculate position-level Greeks"""
        # Simplified Greeks calculation
        # In production, this would use proper option pricing models

        front_delta = self._estimate_delta(opportunity.front_leg)
        back_delta = self._estimate_delta(opportunity.back_leg)
        net_delta = (back_delta - front_delta) * size

        front_gamma = self._estimate_gamma(opportunity.front_leg)
        back_gamma = self._estimate_gamma(opportunity.back_leg)
        net_gamma = (back_gamma - front_gamma) * size

        front_vega = self._estimate_vega(opportunity.front_leg)
        back_vega = self._estimate_vega(opportunity.back_leg)
        net_vega = (back_vega - front_vega) * size

        # Rho is typically small for calendar spreads
        net_rho = 0.01 * size  # Simplified estimate

        return {
            "delta": net_delta,
            "gamma": net_gamma,
            "vega": net_vega,
            "rho": net_rho,
        }

    def _calculate_risk_metrics(
        self,
        opportunity: CalendarSpreadOpportunity,
        current_pnl: float,
        initial_debit: float,
        breakeven_points: BreakevenPoints,
    ) -> Dict[str, float]:
        """Calculate risk metrics"""
        prob_profit = max(0.0, breakeven_points.prob_within_range)

        max_profit = (
            opportunity.max_profit
            if hasattr(opportunity, "max_profit")
            else initial_debit * 0.5
        )
        max_loss = (
            opportunity.max_loss if hasattr(opportunity, "max_loss") else initial_debit
        )

        risk_reward_ratio = max_profit / max_loss if max_loss > 0 else 0.0

        return {"prob_profit": prob_profit, "risk_reward_ratio": risk_reward_ratio}

    def _calculate_pnl_attribution(
        self,
        opportunity: CalendarSpreadOpportunity,
        current_pnl: float,
        days_in_position: int,
        stock_price: float,
    ) -> Dict[str, float]:
        """Calculate P&L attribution to different factors"""
        # Simplified attribution model
        # In production, this would track historical changes

        total_pnl = current_pnl

        # Time decay component (estimated)
        time_pnl = (
            days_in_position
            * (opportunity.back_leg.theta - opportunity.front_leg.theta)
            * 100
        )

        # Price movement component (estimated from delta)
        delta = self._estimate_delta(opportunity.back_leg) - self._estimate_delta(
            opportunity.front_leg
        )
        # Assume 5% price movement on average
        price_pnl = delta * stock_price * 0.05 * 100

        # Volatility component (residual)
        volatility_pnl = total_pnl - time_pnl - price_pnl

        # Other factors
        other_pnl = 0.0

        return {
            "price_pnl": price_pnl,
            "time_pnl": time_pnl,
            "volatility_pnl": volatility_pnl,
            "other_pnl": other_pnl,
        }

    def _calculate_time_decay_acceleration(
        self, opportunity: CalendarSpreadOpportunity
    ) -> float:
        """Calculate time decay acceleration factor"""
        days_to_front = opportunity.front_leg.days_to_expiry

        if days_to_front > 30:
            return 1.0  # Normal decay
        elif days_to_front > 14:
            return 1.5  # Moderate acceleration
        elif days_to_front > 7:
            return 2.0  # High acceleration
        else:
            return 3.0  # Very high acceleration

    def _assess_data_quality(self, opportunity: CalendarSpreadOpportunity) -> float:
        """Assess quality of market data"""
        quality_factors = []

        # Bid-ask spread quality
        front_spread = (
            opportunity.front_leg.ask - opportunity.front_leg.bid
        ) / opportunity.front_leg.price
        back_spread = (
            opportunity.back_leg.ask - opportunity.back_leg.bid
        ) / opportunity.back_leg.price

        spread_quality = 1.0 - min(1.0, (front_spread + back_spread) / 2.0 * 10)
        quality_factors.append(spread_quality * 0.4)

        # Volume quality
        volume_quality = min(
            1.0, (opportunity.front_leg.volume + opportunity.back_leg.volume) / 100.0
        )
        quality_factors.append(volume_quality * 0.3)

        # IV quality (reasonable range)
        iv_quality = (
            1.0
            if 5 <= opportunity.front_leg.iv <= 100
            and 5 <= opportunity.back_leg.iv <= 100
            else 0.5
        )
        quality_factors.append(iv_quality * 0.3)

        return sum(quality_factors)

    def _calculate_confidence_score(
        self, opportunity: CalendarSpreadOpportunity, data_quality: float
    ) -> float:
        """Calculate confidence score for P&L calculations"""
        confidence_factors = []

        # Data quality component
        confidence_factors.append(data_quality * 0.4)

        # Term structure component
        if (
            hasattr(opportunity, "term_structure_inversion")
            and opportunity.term_structure_inversion
        ):
            confidence_factors.append(0.3)
        else:
            confidence_factors.append(0.1)

        # Liquidity component
        liquidity_score = getattr(opportunity, "combined_liquidity_score", 0.5)
        confidence_factors.append(liquidity_score * 0.2)

        # Time to expiry component (more confidence with reasonable time)
        days_to_front = opportunity.front_leg.days_to_expiry
        if 14 <= days_to_front <= 45:
            confidence_factors.append(0.1)
        else:
            confidence_factors.append(0.05)

        return sum(confidence_factors)

    def _estimate_spread_value_at_expiry(
        self, opportunity: CalendarSpreadOpportunity, stock_price: float
    ) -> float:
        """Estimate spread value at front expiry for given stock price"""
        strike = opportunity.strike

        # Front option value at expiry
        if opportunity.option_type == "CALL":
            front_value = max(0, stock_price - strike)
        else:
            front_value = max(0, strike - stock_price)

        # Back option time value (simplified)
        time_remaining = (
            opportunity.back_leg.days_to_expiry - opportunity.front_leg.days_to_expiry
        ) / 365.0
        back_iv = opportunity.back_leg.iv / 100.0

        back_time_value = self._estimate_option_time_value(
            stock_price,
            strike,
            time_remaining,
            back_iv,
            opportunity.option_type,
            self.config.risk_free_rate,
        )

        # Intrinsic value of back option
        if opportunity.option_type == "CALL":
            back_intrinsic = max(0, stock_price - strike)
        else:
            back_intrinsic = max(0, strike - stock_price)

        back_value = back_intrinsic + back_time_value

        return back_value - front_value

    def _estimate_breakeven_at_time(
        self,
        opportunity: CalendarSpreadOpportunity,
        stock_price: float,
        days_forward: int,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Estimate breakeven points at specific time forward"""
        # Simplified estimation - in production would use full option pricing
        current_downside = getattr(
            self.calculate_breakeven_points(opportunity, stock_price),
            "downside_breakeven",
            None,
        )
        current_upside = getattr(
            self.calculate_breakeven_points(opportunity, stock_price),
            "upside_breakeven",
            None,
        )

        # Adjust for time decay (breakeven range typically widens over time)
        time_factor = 1.1 if days_forward > 7 else 1.0

        adjusted_downside = (
            current_downside * (2 - time_factor) if current_downside else None
        )
        adjusted_upside = current_upside * time_factor if current_upside else None

        return (adjusted_downside, adjusted_upside)

    def _calculate_scenario_pnl(
        self,
        opportunity: CalendarSpreadOpportunity,
        stock_price: float,
        size: int,
        scenario_data: Dict,
    ) -> PnLScenario:
        """Calculate P&L for a specific scenario"""
        try:
            # Extract scenario parameters
            name = scenario_data["name"]
            price_change_pct = scenario_data["price_change"]
            days_forward = scenario_data["days"]
            iv_change_front = scenario_data.get("iv_change_front", 0.0)
            iv_change_back = scenario_data.get("iv_change_back", 0.0)
            rate_change = scenario_data.get("rate_change", 0.0)

            # Calculate future stock price
            future_stock_price = stock_price * (1 + price_change_pct)

            # Estimate future option values (simplified)
            front_days_remaining = max(
                0, opportunity.front_leg.days_to_expiry - days_forward
            )
            back_days_remaining = max(
                0, opportunity.back_leg.days_to_expiry - days_forward
            )

            # Adjust IVs
            new_front_iv = max(5.0, opportunity.front_leg.iv + iv_change_front) / 100.0
            new_back_iv = max(5.0, opportunity.back_leg.iv + iv_change_back) / 100.0

            # Calculate future option values
            if front_days_remaining > 0:
                front_time_value = self._estimate_option_time_value(
                    future_stock_price,
                    opportunity.strike,
                    front_days_remaining / 365.0,
                    new_front_iv,
                    opportunity.option_type,
                    self.config.risk_free_rate + rate_change,
                )
                front_intrinsic = self._calculate_intrinsic_value(
                    future_stock_price, opportunity.strike, opportunity.option_type
                )
                projected_front_value = front_intrinsic + front_time_value
            else:
                projected_front_value = self._calculate_intrinsic_value(
                    future_stock_price, opportunity.strike, opportunity.option_type
                )

            if back_days_remaining > 0:
                back_time_value = self._estimate_option_time_value(
                    future_stock_price,
                    opportunity.strike,
                    back_days_remaining / 365.0,
                    new_back_iv,
                    opportunity.option_type,
                    self.config.risk_free_rate + rate_change,
                )
                back_intrinsic = self._calculate_intrinsic_value(
                    future_stock_price, opportunity.strike, opportunity.option_type
                )
                projected_back_value = back_intrinsic + back_time_value
            else:
                projected_back_value = self._calculate_intrinsic_value(
                    future_stock_price, opportunity.strike, opportunity.option_type
                )

            # Calculate spread value and P&L
            projected_spread_value = projected_back_value - projected_front_value
            initial_debit = opportunity.net_debit
            projected_pnl = (projected_spread_value - initial_debit) * size
            projected_pnl_pct = (
                (projected_pnl / (initial_debit * size) * 100)
                if initial_debit > 0
                else 0.0
            )

            # Greeks impact (simplified)
            greeks = self._calculate_position_greeks(opportunity, stock_price, size)
            price_change = future_stock_price - stock_price

            delta_pnl = greeks["delta"] * price_change * 100
            gamma_pnl = 0.5 * greeks["gamma"] * (price_change**2) * 100
            theta_pnl = (
                -abs(opportunity.front_leg.theta - opportunity.back_leg.theta)
                * days_forward
                * size
                * 100
            )
            vega_pnl = greeks["vega"] * (iv_change_front + iv_change_back) / 2.0
            rho_pnl = greeks["rho"] * rate_change * 100

            # Scenario probability (simplified normal distribution)
            if price_change_pct != 0:
                annual_vol = self.config.stock_volatility
                time_horizon = days_forward / 365.0
                expected_std = annual_vol * np.sqrt(time_horizon)
                z_score = abs(price_change_pct) / expected_std
                scenario_probability = 2 * (1 - stats.norm.cdf(z_score))  # Two-tailed
            else:
                scenario_probability = 0.1  # Assign some probability to flat scenarios

            # Confidence interval (simplified)
            confidence_interval = (projected_pnl * 0.8, projected_pnl * 1.2)

            return PnLScenario(
                scenario_name=name,
                stock_price=future_stock_price,
                stock_price_change_pct=price_change_pct * 100,
                days_forward=days_forward,
                # IV changes
                iv_change_front=iv_change_front,
                iv_change_back=iv_change_back,
                interest_rate_change=rate_change,
                # Projections
                projected_front_value=projected_front_value,
                projected_back_value=projected_back_value,
                projected_spread_value=projected_spread_value,
                projected_pnl=projected_pnl,
                projected_pnl_pct=projected_pnl_pct,
                # Greeks impact
                delta_pnl=delta_pnl,
                gamma_pnl=gamma_pnl,
                theta_pnl=theta_pnl,
                vega_pnl=vega_pnl,
                rho_pnl=rho_pnl,
                # Probability
                scenario_probability=scenario_probability,
                confidence_interval=confidence_interval,
            )

        except Exception as e:
            logger.error(f"Error calculating scenario P&L: {str(e)}")
            return PnLScenario(
                scenario_name=scenario_data.get("name", "error"),
                stock_price=stock_price,
                stock_price_change_pct=0.0,
                days_forward=0,
                iv_change_front=0.0,
                iv_change_back=0.0,
                interest_rate_change=0.0,
                projected_front_value=0.0,
                projected_back_value=0.0,
                projected_spread_value=0.0,
                projected_pnl=0.0,
                projected_pnl_pct=0.0,
                delta_pnl=0.0,
                gamma_pnl=0.0,
                theta_pnl=0.0,
                vega_pnl=0.0,
                rho_pnl=0.0,
                scenario_probability=0.0,
                confidence_interval=(0.0, 0.0),
            )

    def _estimate_option_time_value(
        self,
        stock_price: float,
        strike: float,
        time_to_expiry: float,
        iv: float,
        option_type: str,
        risk_free_rate: float,
    ) -> float:
        """Estimate option time value using simplified Black-Scholes"""
        if time_to_expiry <= 0:
            return 0.0

        try:
            # Simplified Black-Scholes approximation
            d1 = (
                np.log(stock_price / strike)
                + (risk_free_rate + 0.5 * iv**2) * time_to_expiry
            ) / (iv * np.sqrt(time_to_expiry))
            d2 = d1 - iv * np.sqrt(time_to_expiry)

            if option_type.upper() == "CALL":
                time_value = (
                    stock_price * stats.norm.cdf(d1)
                    - strike
                    * np.exp(-risk_free_rate * time_to_expiry)
                    * stats.norm.cdf(d2)
                    - max(0, stock_price - strike)
                )
            else:
                time_value = (
                    strike
                    * np.exp(-risk_free_rate * time_to_expiry)
                    * stats.norm.cdf(-d2)
                    - stock_price * stats.norm.cdf(-d1)
                    - max(0, strike - stock_price)
                )

            return max(0.0, time_value)

        except Exception as e:
            logger.warning(f"Error in Black-Scholes calculation: {str(e)}")
            # Fallback to simple approximation
            atm_factor = 1.0 - abs(stock_price - strike) / strike
            return max(
                0.0, stock_price * iv * np.sqrt(time_to_expiry) * atm_factor * 0.4
            )

    def _calculate_intrinsic_value(
        self, stock_price: float, strike: float, option_type: str
    ) -> float:
        """Calculate intrinsic value of option"""
        if option_type.upper() == "CALL":
            return max(0.0, stock_price - strike)
        else:
            return max(0.0, strike - stock_price)

    def _estimate_delta(self, leg: CalendarSpreadLeg) -> float:
        """Estimate delta for option leg"""
        # Simplified delta estimation
        if leg.right == "C":
            return 0.5  # Rough ATM call delta
        else:
            return -0.5  # Rough ATM put delta

    def _estimate_gamma(self, leg: CalendarSpreadLeg) -> float:
        """Estimate gamma for option leg"""
        # Simplified gamma estimation
        time_factor = max(0.1, leg.days_to_expiry / 30.0)
        return 0.1 / time_factor  # Higher gamma closer to expiry

    def _estimate_vega(self, leg: CalendarSpreadLeg) -> float:
        """Estimate vega for option leg"""
        # Simplified vega estimation
        time_factor = np.sqrt(max(0.1, leg.days_to_expiry / 365.0))
        return leg.price * 0.1 * time_factor

    def clear_cache(self) -> None:
        """Clear all caches"""
        self.pnl_cache.clear()
        self.greeks_cache.clear()
        self.breakeven_cache.clear()
        self.monte_carlo_cache.clear()
        logger.info("Calendar P&L calculator cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "pnl_results": len(self.pnl_cache),
            "greeks": len(self.greeks_cache),
            "breakeven": len(self.breakeven_cache),
            "monte_carlo": len(self.monte_carlo_cache),
        }

    def __del__(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=False)


# Convenience functions for integration


def calculate_calendar_pnl_quick(
    opportunity: CalendarSpreadOpportunity,
    current_stock_price: float,
    position_size: int = 1,
    config: Optional[CalendarPnLConfig] = None,
) -> CalendarPnLResult:
    """
    Quick P&L calculation for calendar spread opportunity.

    Args:
        opportunity: Calendar spread opportunity
        current_stock_price: Current stock price
        position_size: Position size
        config: Optional configuration

    Returns:
        CalendarPnLResult with comprehensive analysis
    """
    calculator = CalendarPnLCalculator(config)
    return calculator.calculate_calendar_pnl(
        opportunity, current_stock_price, position_size
    )


def run_calendar_monte_carlo(
    opportunity: CalendarSpreadOpportunity,
    current_stock_price: float,
    position_size: int = 1,
    simulations: int = 5000,
) -> MonteCarloResults:
    """
    Run Monte Carlo simulation for calendar spread.

    Args:
        opportunity: Calendar spread opportunity
        current_stock_price: Current stock price
        position_size: Position size
        simulations: Number of simulations

    Returns:
        MonteCarloResults with distribution analysis
    """
    config = CalendarPnLConfig(monte_carlo_simulations=simulations)
    calculator = CalendarPnLCalculator(config)
    return calculator.run_monte_carlo_simulation(
        opportunity, current_stock_price, position_size
    )


def analyze_calendar_breakevens(
    opportunity: CalendarSpreadOpportunity,
    current_stock_price: float,
    position_size: int = 1,
) -> BreakevenPoints:
    """
    Analyze breakeven points for calendar spread.

    Args:
        opportunity: Calendar spread opportunity
        current_stock_price: Current stock price
        position_size: Position size

    Returns:
        BreakevenPoints with comprehensive analysis
    """
    calculator = CalendarPnLCalculator()
    return calculator.calculate_breakeven_points(
        opportunity, current_stock_price, position_size
    )


def model_calendar_scenarios(
    opportunity: CalendarSpreadOpportunity,
    current_stock_price: float,
    position_size: int = 1,
    custom_scenarios: Optional[List[Dict]] = None,
) -> List[PnLScenario]:
    """
    Model P&L scenarios for calendar spread.

    Args:
        opportunity: Calendar spread opportunity
        current_stock_price: Current stock price
        position_size: Position size
        custom_scenarios: Optional custom scenarios

    Returns:
        List of PnLScenario objects
    """
    calculator = CalendarPnLCalculator()
    return calculator.model_pnl_scenarios(
        opportunity, current_stock_price, position_size, custom_scenarios
    )


# Integration with existing CalendarSpread system


def enhance_calendar_opportunity_with_pnl(
    opportunity: CalendarSpreadOpportunity,
    current_stock_price: float,
    position_size: int = 1,
) -> CalendarSpreadOpportunity:
    """
    Enhance CalendarSpreadOpportunity with P&L analysis.

    Args:
        opportunity: Original calendar spread opportunity
        current_stock_price: Current stock price
        position_size: Position size

    Returns:
        Enhanced opportunity with P&L metrics
    """
    try:
        calculator = CalendarPnLCalculator()
        pnl_result = calculator.calculate_calendar_pnl(
            opportunity, current_stock_price, position_size
        )

        # Add P&L metrics to opportunity
        opportunity.estimated_max_profit = pnl_result.estimated_max_profit
        opportunity.probability_of_profit = pnl_result.probability_of_profit
        opportunity.theta_capture_daily = pnl_result.theta_capture_daily
        opportunity.breakeven_upside = pnl_result.breakeven_upside
        opportunity.breakeven_downside = pnl_result.breakeven_downside
        opportunity.pnl_confidence_score = pnl_result.confidence_score

        # Enhance composite score with P&L insights
        pnl_score_bonus = pnl_result.confidence_score * 0.1
        opportunity.composite_score = min(
            1.0, opportunity.composite_score + pnl_score_bonus
        )

        logger.info(
            f"Enhanced {opportunity.symbol} calendar spread with P&L analysis: "
            f"Max profit: ${pnl_result.estimated_max_profit:.2f}, "
            f"Prob profit: {pnl_result.probability_of_profit:.1%}"
        )

    except Exception as e:
        logger.error(f"Error enhancing calendar opportunity with P&L: {str(e)}")

    return opportunity


logger.info("Calendar P&L Calculator module loaded successfully")
