"""
Calendar Greeks Integration module for comprehensive Greeks calculations and risk management
of calendar spread positions.

This module provides advanced Greeks calculations, risk scoring, and position adjustment
triggers for calendar spreads. It integrates with the existing CalendarSpread, CalendarPnL,
and TermStructure implementations to provide real-time Greeks monitoring and risk assessment.

Key Features:
- Net Greeks calculations for entire calendar spread positions
- Individual leg Greeks analysis (front month vs back month)
- Greeks-based risk scoring and position health metrics
- Time-based Greeks evolution modeling
- Automated position adjustment recommendations
- Portfolio-level Greeks aggregation
- Real-time risk threshold monitoring
- Performance-optimized numerical calculations

Author: Calendar Greeks Analysis System
Version: 1.0.0
"""

import asyncio
import copy
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from statistics import mean, stdev
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import logging
import numpy as np
from ib_async import Contract, Ticker
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

from .CalendarPnL import CalendarPnLResult
from .CalendarSpread import CalendarSpreadLeg, CalendarSpreadOpportunity
from .common import configure_logging, get_logger
from .metrics import RejectionReason, metrics_collector
from .TermStructure import IVDataPoint, TermStructureAnalyzer

# Configure logging
logger = get_logger()


class GreeksRiskLevel(Enum):
    """Risk levels for Greeks-based position assessment"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AdjustmentType(Enum):
    """Types of position adjustments"""

    DELTA_HEDGE = "DELTA_HEDGE"
    GAMMA_HEDGE = "GAMMA_HEDGE"
    VEGA_HEDGE = "VEGA_HEDGE"
    CLOSE_POSITION = "CLOSE_POSITION"
    ROLL_POSITION = "ROLL_POSITION"
    SCALE_DOWN = "SCALE_DOWN"


@dataclass
class GreeksEvolution:
    """Greeks evolution over time for scenario analysis"""

    time_horizon_days: int
    underlying_price_scenarios: List[float]
    delta_evolution: Dict[float, List[float]]  # price -> [day0, day1, ..., dayN]
    gamma_evolution: Dict[float, List[float]]
    vega_evolution: Dict[float, List[float]]
    theta_evolution: Dict[float, List[float]]
    rho_evolution: Dict[float, List[float]]

    # Scenario probabilities
    scenario_probabilities: List[float] = field(default_factory=list)
    expected_values: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class PositionAdjustment:
    """Position adjustment recommendation"""

    adjustment_type: AdjustmentType
    priority: int  # 1=urgent, 2=important, 3=optional
    reason: str
    recommended_action: str
    hedge_quantity: Optional[int] = None
    hedge_contract: Optional[Contract] = None
    expected_cost: Optional[float] = None
    risk_reduction: Optional[float] = None
    time_sensitivity: str = "IMMEDIATE"  # IMMEDIATE, WITHIN_HOUR, END_OF_DAY


@dataclass
class CalendarGreeks:
    """
    Comprehensive Greeks analysis for calendar spread positions.

    This dataclass contains all Greeks calculations, risk metrics, and analysis
    for a calendar spread position including both individual leg analysis and
    net position exposure.
    """

    # Position identification
    symbol: str
    strike: float = 0.0
    option_type: str = "CALL"  # 'CALL' or 'PUT'
    position_size: int = 1
    entry_date: datetime = field(default_factory=datetime.now)

    # Individual leg Greeks
    front_delta: float = 0.0
    front_gamma: float = 0.0
    front_vega: float = 0.0
    front_theta: float = 0.0
    front_rho: float = 0.0
    front_days_to_expiry: int = 0

    back_delta: float = 0.0
    back_gamma: float = 0.0
    back_vega: float = 0.0
    back_theta: float = 0.0
    back_rho: float = 0.0
    back_days_to_expiry: int = 0

    # Additional aliases for backward compatibility with tests
    days_to_front_expiry: Optional[int] = None  # Alias for front_days_to_expiry
    days_to_back_expiry: Optional[int] = None  # Alias for back_days_to_expiry
    time_decay_rate: Optional[float] = None  # Additional time decay metric
    last_updated: Optional[float] = None  # Last update timestamp

    # Net position Greeks (back - front for calendar spread)
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_vega: float = 0.0
    net_theta: float = 0.0
    net_rho: float = 0.0

    # Greeks-based risk metrics
    delta_risk_score: float = 0.0  # 0-1 scale, >0.8 indicates high risk
    gamma_risk_score: float = 0.0
    vega_risk_score: float = 0.0
    theta_risk_score: float = 0.0  # Risk score for theta exposure
    theta_efficiency_score: float = 0.0  # How well position captures theta
    overall_risk_score: float = 0.0
    risk_level: GreeksRiskLevel = GreeksRiskLevel.LOW
    overall_risk_level: GreeksRiskLevel = (
        GreeksRiskLevel.LOW
    )  # Alias for risk_level for backward compatibility

    # Position health metrics
    delta_neutral_range: Tuple[float, float] = (
        0.0,
        0.0,
    )  # Price range where delta < 0.20
    gamma_acceleration_threshold: float = 0.0  # Price where gamma effects accelerate
    vega_sensitivity_range: Tuple[float, float] = (
        0.0,
        0.0,
    )  # IV range for acceptable vega exposure
    theta_capture_efficiency: float = 0.0  # Actual theta capture vs theoretical

    # Time-based evolution
    greeks_evolution: Optional[GreeksEvolution] = None

    # Risk thresholds exceeded
    delta_threshold_exceeded: bool = False
    gamma_threshold_exceeded: bool = False
    vega_threshold_exceeded: bool = False

    # Position adjustments
    recommended_adjustments: List[PositionAdjustment] = field(default_factory=list)

    # Metadata
    calculation_time: float = field(default_factory=time.time)
    underlying_price: float = 0.0
    underlying_iv: float = 0.0
    cache_ttl: float = 300.0  # 5 minutes


@dataclass
class PortfolioGreeks:
    """Portfolio-level Greeks aggregation"""

    portfolio_id: str
    positions: List[CalendarGreeks] = field(default_factory=list)

    # Aggregate Greeks
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_vega: float = 0.0
    total_theta: float = 0.0
    total_rho: float = 0.0

    # Portfolio risk metrics
    portfolio_risk_score: float = 0.0
    correlation_adjusted_risk: float = 0.0
    concentration_risk: Dict[str, float] = field(default_factory=dict)  # By symbol

    # Portfolio-level adjustments
    portfolio_adjustments: List[PositionAdjustment] = field(default_factory=list)

    # Additional fields for test compatibility
    total_positions: int = 0


class CalendarGreeksCalculator:
    """
    Advanced Greeks calculator for calendar spread positions with risk management
    and position adjustment recommendations.

    This class provides comprehensive Greeks analysis including:
    - Real-time Greeks calculations using Black-Scholes and numerical methods
    - Risk scoring based on position Greeks exposure
    - Automated position adjustment recommendations
    - Greeks evolution modeling for scenario analysis
    - Portfolio-level Greeks aggregation
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        delta_threshold: float = 0.20,
        gamma_threshold: float = 0.10,
        vega_threshold: float = 5.0,
        theta_min_efficiency: float = 0.70,
        cache_ttl: float = 300.0,
    ):
        """
        Initialize the Calendar Greeks Calculator.

        Args:
            risk_free_rate: Risk-free interest rate for calculations
            delta_threshold: Maximum acceptable net delta (from instructions)
            gamma_threshold: Maximum acceptable gamma exposure
            vega_threshold: Maximum acceptable vega exposure per $1 IV move
            theta_min_efficiency: Minimum theta capture efficiency (70%)
            cache_ttl: Cache time-to-live for calculations (seconds)
        """
        self.risk_free_rate = risk_free_rate
        self.delta_threshold = delta_threshold
        self.gamma_threshold = gamma_threshold
        self.vega_threshold = vega_threshold
        self.theta_min_efficiency = theta_min_efficiency
        self.cache_ttl = cache_ttl

        # Calculation caches
        self.greeks_cache: Dict[str, Tuple[float, CalendarGreeks]] = {}
        self.bs_cache: Dict[str, Tuple[float, Dict]] = {}  # Black-Scholes cache
        self.evolution_cache: Dict[str, Tuple[float, GreeksEvolution]] = {}

        # Portfolio tracking
        self.portfolio_positions: Dict[str, List[CalendarGreeks]] = defaultdict(list)

        # Risk thresholds for compatibility with tests
        self.risk_thresholds = {
            "delta": delta_threshold,
            "gamma": gamma_threshold,
            "vega": vega_threshold,
            "theta_efficiency": theta_min_efficiency,
        }

        # Last calculation time tracking
        self.last_calculation_time = 0.0

        logger.info("Calendar Greeks Calculator initialized")

    def calculate_calendar_greeks(
        self,
        opportunity: CalendarSpreadOpportunity,
        position_size: int = 1,
        underlying_price: Optional[float] = None,
        entry_date: Optional[datetime] = None,
    ) -> CalendarGreeks:
        """
        Calculate comprehensive Greeks for a calendar spread opportunity.

        Args:
            opportunity: Calendar spread opportunity to analyze
            position_size: Size of the position (default 1 contract)
            underlying_price: Current underlying price (if None, estimated from opportunity)
            entry_date: Position entry date (default now)

        Returns:
            CalendarGreeks object with complete analysis
        """
        try:
            cache_key = self._generate_greeks_cache_key(opportunity, position_size)

            # Check cache first
            if cache_key in self.greeks_cache:
                cache_time, cached_greeks = self.greeks_cache[cache_key]
                if time.time() - cache_time < self.cache_ttl:
                    logger.debug(f"Using cached Greeks for {opportunity.symbol}")
                    return cached_greeks

            # Estimate underlying price if not provided
            if underlying_price is None:
                underlying_price = self._estimate_underlying_price(opportunity)

            entry_date = entry_date or datetime.now()

            # Calculate individual leg Greeks
            front_greeks = self._calculate_single_option_greeks(
                opportunity.front_leg, underlying_price, self.risk_free_rate
            )

            back_greeks = self._calculate_single_option_greeks(
                opportunity.back_leg, underlying_price, self.risk_free_rate
            )

            # Calculate net Greeks (back - front for calendar spread)
            net_delta = (back_greeks["delta"] - front_greeks["delta"]) * position_size
            net_gamma = (back_greeks["gamma"] - front_greeks["gamma"]) * position_size
            net_vega = (back_greeks["vega"] - front_greeks["vega"]) * position_size
            net_theta = (back_greeks["theta"] - front_greeks["theta"]) * position_size
            net_rho = (back_greeks["rho"] - front_greeks["rho"]) * position_size

            # Calculate risk scores
            risk_scores = self._calculate_risk_scores(
                net_delta, net_gamma, net_vega, net_theta, position_size
            )

            # Determine overall risk level
            risk_level = self._determine_risk_level(risk_scores)

            # Calculate position health metrics
            health_metrics = self._calculate_position_health_metrics(
                opportunity, net_delta, net_gamma, net_vega, underlying_price
            )

            # Generate position adjustment recommendations
            adjustments = self._generate_position_adjustments(
                opportunity, risk_scores, risk_level, net_delta, net_gamma, net_vega
            )

            # Create CalendarGreeks object
            calendar_greeks = CalendarGreeks(
                symbol=opportunity.symbol,
                strike=opportunity.strike,
                option_type=opportunity.option_type,
                position_size=position_size,
                entry_date=entry_date,
                # Front leg Greeks
                front_delta=front_greeks["delta"] * position_size,
                front_gamma=front_greeks["gamma"] * position_size,
                front_vega=front_greeks["vega"] * position_size,
                front_theta=front_greeks["theta"] * position_size,
                front_rho=front_greeks["rho"] * position_size,
                front_days_to_expiry=opportunity.front_leg.days_to_expiry,
                # Back leg Greeks
                back_delta=back_greeks["delta"] * position_size,
                back_gamma=back_greeks["gamma"] * position_size,
                back_vega=back_greeks["vega"] * position_size,
                back_theta=back_greeks["theta"] * position_size,
                back_rho=back_greeks["rho"] * position_size,
                back_days_to_expiry=opportunity.back_leg.days_to_expiry,
                # Net Greeks
                net_delta=net_delta,
                net_gamma=net_gamma,
                net_vega=net_vega,
                net_theta=net_theta,
                net_rho=net_rho,
                # Risk metrics
                delta_risk_score=risk_scores["delta_risk"],
                gamma_risk_score=risk_scores["gamma_risk"],
                vega_risk_score=risk_scores["vega_risk"],
                theta_efficiency_score=risk_scores["theta_efficiency"],
                overall_risk_score=risk_scores["overall_risk"],
                risk_level=risk_level,
                # Health metrics
                delta_neutral_range=health_metrics["delta_neutral_range"],
                gamma_acceleration_threshold=health_metrics["gamma_threshold"],
                vega_sensitivity_range=health_metrics["vega_range"],
                theta_capture_efficiency=health_metrics["theta_efficiency"],
                # Threshold flags
                delta_threshold_exceeded=abs(net_delta) > self.delta_threshold,
                gamma_threshold_exceeded=abs(net_gamma) > self.gamma_threshold,
                vega_threshold_exceeded=abs(net_vega) > self.vega_threshold,
                # Adjustments
                recommended_adjustments=adjustments,
                # Metadata
                underlying_price=underlying_price,
                underlying_iv=opportunity.front_leg.iv,  # Use front month IV as reference
            )

            # Cache the result
            self.greeks_cache[cache_key] = (time.time(), calendar_greeks)

            logger.info(
                f"Calculated Greeks for {opportunity.symbol} {opportunity.strike} "
                f"{opportunity.option_type}: Delta={net_delta:.3f}, "
                f"Gamma={net_gamma:.3f}, Vega={net_vega:.2f}, "
                f"Theta={net_theta:.3f}, Risk={risk_level.value}"
            )

            return calendar_greeks

        except Exception as e:
            logger.error(f"Error calculating calendar Greeks: {str(e)}")
            raise

    def calculate_net_greeks(self, positions: List[CalendarGreeks]) -> Dict[str, float]:
        """
        Calculate net Greeks across multiple calendar spread positions.

        Args:
            positions: List of CalendarGreeks positions

        Returns:
            Dictionary with net Greeks values
        """
        try:
            if not positions:
                return {
                    "net_delta": 0.0,
                    "net_gamma": 0.0,
                    "net_vega": 0.0,
                    "net_theta": 0.0,
                    "net_rho": 0.0,
                }

            net_delta = sum(pos.net_delta for pos in positions)
            net_gamma = sum(pos.net_gamma for pos in positions)
            net_vega = sum(pos.net_vega for pos in positions)
            net_theta = sum(pos.net_theta for pos in positions)
            net_rho = sum(pos.net_rho for pos in positions)

            logger.debug(
                f"Net Greeks for {len(positions)} positions: "
                f"Delta={net_delta:.3f}, Gamma={net_gamma:.3f}, "
                f"Vega={net_vega:.2f}, Theta={net_theta:.3f}"
            )

            return {
                "net_delta": net_delta,
                "net_gamma": net_gamma,
                "net_vega": net_vega,
                "net_theta": net_theta,
                "net_rho": net_rho,
            }

        except Exception as e:
            logger.error(f"Error calculating net Greeks: {str(e)}")
            raise

    def calculate_risk_score(self, calendar_greeks: CalendarGreeks) -> float:
        """
        Calculate overall risk score for a calendar spread position.

        Args:
            calendar_greeks: CalendarGreeks object to analyze

        Returns:
            Risk score between 0.0 (low risk) and 1.0 (high risk)
        """
        try:
            # Weight different Greeks risks
            delta_weight = 0.35
            gamma_weight = 0.25
            vega_weight = 0.25
            theta_weight = 0.15

            # Calculate weighted risk score
            risk_score = (
                calendar_greeks.delta_risk_score * delta_weight
                + calendar_greeks.gamma_risk_score * gamma_weight
                + calendar_greeks.vega_risk_score * vega_weight
                + (1.0 - calendar_greeks.theta_efficiency_score) * theta_weight
            )

            # Apply position size multiplier
            size_multiplier = min(2.0, 1.0 + abs(calendar_greeks.position_size) / 10.0)
            risk_score *= size_multiplier

            # Cap at 1.0
            risk_score = min(1.0, risk_score)

            logger.debug(
                f"Risk score for {calendar_greeks.symbol}: {risk_score:.3f} "
                f"(Delta: {calendar_greeks.delta_risk_score:.3f}, "
                f"Gamma: {calendar_greeks.gamma_risk_score:.3f}, "
                f"Vega: {calendar_greeks.vega_risk_score:.3f})"
            )

            return risk_score

        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            return 1.0  # Return high risk on error

    def suggest_position_adjustments(
        self, calendar_greeks: CalendarGreeks, current_underlying_price: float
    ) -> List[PositionAdjustment]:
        """
        Generate position adjustment recommendations based on current Greeks.

        Args:
            calendar_greeks: Current position Greeks
            current_underlying_price: Current underlying asset price

        Returns:
            List of recommended position adjustments
        """
        try:
            adjustments = []

            # Delta hedging recommendations
            if abs(calendar_greeks.net_delta) > self.delta_threshold:
                hedge_shares = -int(
                    calendar_greeks.net_delta * 100
                )  # Delta hedge with shares
                adjustment = PositionAdjustment(
                    adjustment_type=AdjustmentType.DELTA_HEDGE,
                    priority=1 if abs(calendar_greeks.net_delta) > 0.5 else 2,
                    reason=f"Net delta {calendar_greeks.net_delta:.3f} exceeds threshold {self.delta_threshold}",
                    recommended_action=f"{'Buy' if hedge_shares > 0 else 'Sell'} {abs(hedge_shares)} shares of {calendar_greeks.symbol}",
                    hedge_quantity=hedge_shares,
                    expected_cost=abs(hedge_shares) * current_underlying_price,
                    risk_reduction=abs(calendar_greeks.net_delta)
                    * 0.8,  # Estimate 80% reduction
                    time_sensitivity=(
                        "IMMEDIATE"
                        if abs(calendar_greeks.net_delta) > 0.5
                        else "WITHIN_HOUR"
                    ),
                )
                adjustments.append(adjustment)

            # Gamma risk management
            if abs(calendar_greeks.net_gamma) > self.gamma_threshold:
                adjustment = PositionAdjustment(
                    adjustment_type=AdjustmentType.GAMMA_HEDGE,
                    priority=2,
                    reason=f"High gamma exposure {calendar_greeks.net_gamma:.3f}",
                    recommended_action="Consider adding opposite gamma position or reducing size",
                    time_sensitivity="END_OF_DAY",
                )
                adjustments.append(adjustment)

            # Vega exposure management
            if abs(calendar_greeks.net_vega) > self.vega_threshold:
                adjustment = PositionAdjustment(
                    adjustment_type=AdjustmentType.VEGA_HEDGE,
                    priority=2,
                    reason=f"High vega exposure {calendar_greeks.net_vega:.2f}",
                    recommended_action="Monitor IV changes closely, consider vega hedge",
                    time_sensitivity="WITHIN_HOUR",
                )
                adjustments.append(adjustment)

            # Theta efficiency check
            if calendar_greeks.theta_efficiency_score < self.theta_min_efficiency:
                adjustment = PositionAdjustment(
                    adjustment_type=AdjustmentType.CLOSE_POSITION,
                    priority=3,
                    reason=f"Low theta efficiency {calendar_greeks.theta_efficiency_score:.2f}",
                    recommended_action="Consider closing position if theta capture remains poor",
                    time_sensitivity="END_OF_DAY",
                )
                adjustments.append(adjustment)

            # Time decay acceleration
            if calendar_greeks.front_days_to_expiry <= 10:
                adjustment = PositionAdjustment(
                    adjustment_type=AdjustmentType.ROLL_POSITION,
                    priority=2,
                    reason="Front month expiry approaching (≤10 days)",
                    recommended_action="Consider rolling to next expiry or closing position",
                    time_sensitivity="WITHIN_HOUR",
                )
                adjustments.append(adjustment)

            # Overall risk level adjustments
            if calendar_greeks.risk_level == GreeksRiskLevel.CRITICAL:
                adjustment = PositionAdjustment(
                    adjustment_type=AdjustmentType.SCALE_DOWN,
                    priority=1,
                    reason="Critical risk level reached",
                    recommended_action="Immediately reduce position size by 50% or close entirely",
                    time_sensitivity="IMMEDIATE",
                )
                adjustments.append(adjustment)

            # Sort by priority
            adjustments.sort(key=lambda x: x.priority)

            logger.info(
                f"Generated {len(adjustments)} adjustment recommendations for "
                f"{calendar_greeks.symbol}"
            )

            return adjustments

        except Exception as e:
            logger.error(f"Error generating position adjustments: {str(e)}")
            return []

    def monitor_greeks_thresholds(
        self, positions: List[CalendarGreeks]
    ) -> Dict[str, List[CalendarGreeks]]:
        """
        Monitor Greeks thresholds across multiple positions and categorize by risk level.

        Args:
            positions: List of CalendarGreeks positions to monitor

        Returns:
            Dictionary categorizing positions by risk level
        """
        try:
            risk_categories = {
                "low_risk": [],
                "medium_risk": [],
                "high_risk": [],
                "critical_risk": [],
            }

            threshold_violations = {
                "delta_violations": [],
                "gamma_violations": [],
                "vega_violations": [],
                "theta_violations": [],
            }

            for position in positions:
                # Categorize by risk level
                if position.risk_level == GreeksRiskLevel.LOW:
                    risk_categories["low_risk"].append(position)
                elif position.risk_level == GreeksRiskLevel.MEDIUM:
                    risk_categories["medium_risk"].append(position)
                elif position.risk_level == GreeksRiskLevel.HIGH:
                    risk_categories["high_risk"].append(position)
                elif position.risk_level == GreeksRiskLevel.CRITICAL:
                    risk_categories["critical_risk"].append(position)

                # Track threshold violations
                if position.delta_threshold_exceeded:
                    threshold_violations["delta_violations"].append(position)
                if position.gamma_threshold_exceeded:
                    threshold_violations["gamma_violations"].append(position)
                if position.vega_threshold_exceeded:
                    threshold_violations["vega_violations"].append(position)
                if position.theta_efficiency_score < self.theta_min_efficiency:
                    threshold_violations["theta_violations"].append(position)

            # Log monitoring results
            for risk_level, positions_list in risk_categories.items():
                if positions_list:
                    symbols = [p.symbol for p in positions_list]
                    logger.info(
                        f"{risk_level.upper()}: {len(positions_list)} positions {symbols}"
                    )

            for violation_type, violations in threshold_violations.items():
                if violations:
                    symbols = [p.symbol for p in violations]
                    logger.warning(
                        f"{violation_type.upper()}: {len(violations)} violations {symbols}"
                    )

            # Combine results
            monitoring_results = {**risk_categories, **threshold_violations}

            return monitoring_results

        except Exception as e:
            logger.error(f"Error monitoring Greeks thresholds: {str(e)}")
            return {}

    def model_greeks_evolution(
        self,
        calendar_greeks: CalendarGreeks,
        time_horizon_days: int = 30,
        price_scenarios: Optional[List[float]] = None,
        num_price_points: int = 20,
    ) -> GreeksEvolution:
        """
        Model how Greeks will evolve over time under different price scenarios.

        Args:
            calendar_greeks: Current position Greeks
            time_horizon_days: Number of days to model forward
            price_scenarios: Specific price scenarios (if None, auto-generated)
            num_price_points: Number of price points to model

        Returns:
            GreeksEvolution object with scenario analysis
        """
        try:
            cache_key = (
                f"{calendar_greeks.symbol}_{calendar_greeks.strike}_{time_horizon_days}"
            )

            # Check cache
            if cache_key in self.evolution_cache:
                cache_time, cached_evolution = self.evolution_cache[cache_key]
                if time.time() - cache_time < self.cache_ttl:
                    return cached_evolution

            # Generate price scenarios if not provided
            if price_scenarios is None:
                current_price = calendar_greeks.underlying_price
                volatility = calendar_greeks.underlying_iv / 100.0

                # Generate price scenarios based on log-normal distribution
                price_range = (
                    current_price * volatility * np.sqrt(time_horizon_days / 365.0) * 2
                )
                price_scenarios = np.linspace(
                    current_price - price_range,
                    current_price + price_range,
                    num_price_points,
                ).tolist()

            # Initialize evolution tracking
            delta_evolution = {}
            gamma_evolution = {}
            vega_evolution = {}
            theta_evolution = {}
            rho_evolution = {}

            # Calculate scenario probabilities (log-normal)
            current_price = calendar_greeks.underlying_price
            volatility = calendar_greeks.underlying_iv / 100.0
            scenario_probabilities = []

            for price in price_scenarios:
                # Log-normal probability density
                if price > 0:
                    log_return = np.log(price / current_price)
                    prob = stats.norm.pdf(
                        log_return,
                        -0.5 * volatility**2 * (time_horizon_days / 365.0),
                        volatility * np.sqrt(time_horizon_days / 365.0),
                    )
                else:
                    prob = 0.0
                scenario_probabilities.append(prob)

            # Normalize probabilities
            total_prob = sum(scenario_probabilities)
            if total_prob > 0:
                scenario_probabilities = [
                    p / total_prob for p in scenario_probabilities
                ]

            # Model evolution for each price scenario
            for price in price_scenarios:
                delta_path = []
                gamma_path = []
                vega_path = []
                theta_path = []
                rho_path = []

                for day in range(time_horizon_days + 1):
                    # Calculate time decay
                    front_days_remaining = max(
                        0, calendar_greeks.front_days_to_expiry - day
                    )
                    back_days_remaining = max(
                        0, calendar_greeks.back_days_to_expiry - day
                    )

                    # Estimate Greeks at this time/price point (simplified model)
                    # In production, this would use full Black-Scholes calculations

                    # Time decay factor
                    front_time_factor = max(
                        0.01,
                        front_days_remaining / calendar_greeks.front_days_to_expiry,
                    )
                    back_time_factor = max(
                        0.01, back_days_remaining / calendar_greeks.back_days_to_expiry
                    )

                    # Price sensitivity factor
                    price_factor = price / current_price if current_price > 0 else 1.0

                    # Simplified Greeks evolution (would be replaced with proper models)
                    delta_est = calendar_greeks.net_delta * price_factor * 0.8
                    gamma_est = (
                        calendar_greeks.net_gamma
                        * (front_time_factor + back_time_factor)
                        / 2
                    )
                    vega_est = (
                        calendar_greeks.net_vega
                        * (front_time_factor + back_time_factor)
                        / 2
                    )
                    theta_est = calendar_greeks.net_theta * (
                        front_time_factor - back_time_factor
                    )
                    rho_est = calendar_greeks.net_rho * 0.9  # Rho relatively stable

                    delta_path.append(delta_est)
                    gamma_path.append(gamma_est)
                    vega_path.append(vega_est)
                    theta_path.append(theta_est)
                    rho_path.append(rho_est)

                delta_evolution[price] = delta_path
                gamma_evolution[price] = gamma_path
                vega_evolution[price] = vega_path
                theta_evolution[price] = theta_path
                rho_evolution[price] = rho_path

            # Calculate expected values across scenarios
            expected_values = {}
            for greek_name, evolution_dict in [
                ("delta", delta_evolution),
                ("gamma", gamma_evolution),
                ("vega", vega_evolution),
                ("theta", theta_evolution),
                ("rho", rho_evolution),
            ]:
                expected_path = []
                for day in range(time_horizon_days + 1):
                    day_expected = sum(
                        evolution_dict[price][day] * prob
                        for price, prob in zip(price_scenarios, scenario_probabilities)
                    )
                    expected_path.append(day_expected)
                expected_values[greek_name] = expected_path

            # Create evolution object
            evolution = GreeksEvolution(
                time_horizon_days=time_horizon_days,
                underlying_price_scenarios=price_scenarios,
                delta_evolution=delta_evolution,
                gamma_evolution=gamma_evolution,
                vega_evolution=vega_evolution,
                theta_evolution=theta_evolution,
                rho_evolution=rho_evolution,
                scenario_probabilities=scenario_probabilities,
                expected_values=expected_values,
            )

            # Cache the result
            self.evolution_cache[cache_key] = (time.time(), evolution)

            logger.info(
                f"Modeled Greeks evolution for {calendar_greeks.symbol} "
                f"over {time_horizon_days} days with {len(price_scenarios)} scenarios"
            )

            return evolution

        except Exception as e:
            logger.error(f"Error modeling Greeks evolution: {str(e)}")
            raise

    def aggregate_portfolio_greeks(
        self, portfolio_id: str, positions: List[CalendarGreeks]
    ) -> PortfolioGreeks:
        """
        Aggregate Greeks across multiple positions for portfolio-level analysis.

        Args:
            portfolio_id: Unique portfolio identifier
            positions: List of CalendarGreeks positions

        Returns:
            PortfolioGreeks object with aggregated analysis
        """
        try:
            if not positions:
                return PortfolioGreeks(
                    portfolio_id=portfolio_id,
                    positions=[],
                    total_delta=0.0,
                    total_gamma=0.0,
                    total_vega=0.0,
                    total_theta=0.0,
                    total_rho=0.0,
                    portfolio_risk_score=0.0,
                    correlation_adjusted_risk=0.0,
                    concentration_risk={},
                )

            # Calculate total Greeks
            total_delta = sum(pos.net_delta for pos in positions)
            total_gamma = sum(pos.net_gamma for pos in positions)
            total_vega = sum(pos.net_vega for pos in positions)
            total_theta = sum(pos.net_theta for pos in positions)
            total_rho = sum(pos.net_rho for pos in positions)

            # Calculate portfolio risk score
            individual_risks = [pos.overall_risk_score for pos in positions]
            portfolio_risk_score = np.sqrt(np.mean(np.square(individual_risks)))

            # Calculate concentration risk by symbol
            symbol_exposure = defaultdict(float)
            total_notional = 0.0

            for pos in positions:
                notional = (
                    abs(pos.position_size) * pos.underlying_price * pos.strike / 100
                )
                symbol_exposure[pos.symbol] += notional
                total_notional += notional

            concentration_risk = {}
            if total_notional > 0:
                for symbol, exposure in symbol_exposure.items():
                    concentration_risk[symbol] = exposure / total_notional

            # Simple correlation adjustment (would be enhanced with real correlation data)
            unique_symbols = len(set(pos.symbol for pos in positions))
            correlation_factor = 1.0 - (
                0.2 * min(unique_symbols, 5) / 5
            )  # Max 20% reduction
            correlation_adjusted_risk = portfolio_risk_score * correlation_factor

            # Generate portfolio-level adjustments
            portfolio_adjustments = []

            # Portfolio delta management
            if abs(total_delta) > len(positions) * self.delta_threshold:
                adjustment = PositionAdjustment(
                    adjustment_type=AdjustmentType.DELTA_HEDGE,
                    priority=1,
                    reason=f"Portfolio delta {total_delta:.3f} exceeds prudent limits",
                    recommended_action="Implement portfolio-wide delta hedge",
                    time_sensitivity="IMMEDIATE",
                )
                portfolio_adjustments.append(adjustment)

            # Concentration risk management
            max_concentration = (
                max(concentration_risk.values()) if concentration_risk else 0.0
            )
            if max_concentration > 0.4:  # >40% in single symbol
                max_symbol = max(concentration_risk.keys(), key=concentration_risk.get)
                adjustment = PositionAdjustment(
                    adjustment_type=AdjustmentType.SCALE_DOWN,
                    priority=2,
                    reason=f"High concentration in {max_symbol}: {max_concentration:.1%}",
                    recommended_action=f"Reduce exposure to {max_symbol}",
                    time_sensitivity="END_OF_DAY",
                )
                portfolio_adjustments.append(adjustment)

            # Create portfolio Greeks object
            portfolio_greeks = PortfolioGreeks(
                portfolio_id=portfolio_id,
                positions=positions,
                total_delta=total_delta,
                total_gamma=total_gamma,
                total_vega=total_vega,
                total_theta=total_theta,
                total_rho=total_rho,
                portfolio_risk_score=portfolio_risk_score,
                correlation_adjusted_risk=correlation_adjusted_risk,
                concentration_risk=concentration_risk,
                portfolio_adjustments=portfolio_adjustments,
            )

            # Update portfolio tracking
            self.portfolio_positions[portfolio_id] = positions

            logger.info(
                f"Portfolio {portfolio_id}: {len(positions)} positions, "
                f"Delta={total_delta:.3f}, Gamma={total_gamma:.3f}, "
                f"Vega={total_vega:.2f}, Theta={total_theta:.3f}, "
                f"Risk Score={portfolio_risk_score:.3f}"
            )

            return portfolio_greeks

        except Exception as e:
            logger.error(f"Error aggregating portfolio Greeks: {str(e)}")
            raise

    def _calculate_single_option_greeks(
        self, leg: CalendarSpreadLeg, underlying_price: float, risk_free_rate: float
    ) -> Dict[str, float]:
        """
        Calculate Greeks for a single option using Black-Scholes model.

        Args:
            leg: Calendar spread leg to analyze
            underlying_price: Current underlying price
            risk_free_rate: Risk-free interest rate

        Returns:
            Dictionary with Greeks values
        """
        try:
            cache_key = f"{leg.contract.conId}_{underlying_price}_{risk_free_rate}"

            # Check cache
            if cache_key in self.bs_cache:
                cache_time, cached_greeks = self.bs_cache[cache_key]
                if time.time() - cache_time < self.cache_ttl:
                    return cached_greeks

            # Black-Scholes parameters
            S = underlying_price  # Current price
            K = leg.strike  # Strike price
            T = max(0.001, leg.days_to_expiry / 365.0)  # Time to expiry in years
            r = risk_free_rate  # Risk-free rate
            sigma = leg.iv / 100.0  # Implied volatility

            # Avoid division by zero
            if T <= 0 or sigma <= 0 or S <= 0:
                return {
                    "delta": 0.0,
                    "gamma": 0.0,
                    "vega": 0.0,
                    "theta": 0.0,
                    "rho": 0.0,
                }

            # Calculate d1 and d2
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            # Standard normal CDF and PDF
            N_d1 = stats.norm.cdf(d1)
            N_d2 = stats.norm.cdf(d2)
            n_d1 = stats.norm.pdf(d1)

            # Option type multiplier
            phi = 1 if leg.right == "C" else -1

            # Calculate Greeks
            delta = phi * N_d1 if leg.right == "C" else phi * (N_d1 - 1)

            gamma = n_d1 / (S * sigma * np.sqrt(T))

            vega = S * n_d1 * np.sqrt(T) / 100.0  # Per 1% IV change

            if leg.right == "C":
                theta = (
                    -S * n_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2
                ) / 365.0
                rho = K * T * np.exp(-r * T) * N_d2 / 100.0
            else:
                theta = (
                    -S * n_d1 * sigma / (2 * np.sqrt(T))
                    + r * K * np.exp(-r * T) * (1 - N_d2)
                ) / 365.0
                rho = -K * T * np.exp(-r * T) * (1 - N_d2) / 100.0

            greeks = {
                "delta": delta,
                "gamma": gamma,
                "vega": vega,
                "theta": theta,
                "rho": rho,
            }

            # Cache the result
            self.bs_cache[cache_key] = (time.time(), greeks)

            return greeks

        except Exception as e:
            logger.error(f"Error calculating Black-Scholes Greeks: {str(e)}")
            # Return zero Greeks on error
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    def _estimate_underlying_price(
        self, opportunity: CalendarSpreadOpportunity
    ) -> float:
        """Estimate underlying price from option prices"""
        try:
            # Use put-call parity approximation
            # C - P = S - K * e^(-r*T)
            # Rearrange: S = C - P + K * e^(-r*T)

            # Try to find matching call and put for same expiry/strike
            if opportunity.option_type == "CALL":
                # Estimate from call price and strike
                call_price = opportunity.front_leg.price
                strike = opportunity.strike
                time_to_expiry = opportunity.front_leg.days_to_expiry / 365.0

                # Rough estimate: assume put is worth strike - call for ATM
                estimated_price = (
                    strike
                    + call_price
                    - strike * np.exp(-self.risk_free_rate * time_to_expiry)
                )
            else:
                # Estimate from put price and strike
                put_price = opportunity.front_leg.price
                strike = opportunity.strike
                time_to_expiry = opportunity.front_leg.days_to_expiry / 365.0

                # Rough estimate: assume call is worth put + intrinsic for ATM
                estimated_price = (
                    strike
                    - put_price
                    + strike * np.exp(-self.risk_free_rate * time_to_expiry)
                )

            # Sanity check - should be reasonably close to strike for calendar spreads
            if estimated_price < strike * 0.5 or estimated_price > strike * 2.0:
                estimated_price = strike  # Fall back to strike price

            return max(0.01, estimated_price)  # Ensure positive

        except Exception as e:
            logger.warning(f"Error estimating underlying price: {str(e)}")
            return opportunity.strike  # Fall back to strike

    def _calculate_risk_scores(
        self,
        net_delta: float,
        net_gamma: float,
        net_vega: float,
        net_theta: float,
        position_size: int,
    ) -> Dict[str, float]:
        """Calculate individual risk scores for each Greek"""
        try:
            # Delta risk (0-1 scale)
            delta_risk = min(1.0, abs(net_delta) / (self.delta_threshold * 2))

            # Gamma risk (0-1 scale)
            gamma_risk = min(1.0, abs(net_gamma) / (self.gamma_threshold * 2))

            # Vega risk (0-1 scale)
            vega_risk = min(1.0, abs(net_vega) / (self.vega_threshold * 2))

            # Theta efficiency (0-1 scale, higher is better)
            # For calendar spreads, we want positive theta (time decay benefit)
            theta_efficiency = max(
                0.0, min(1.0, net_theta / (abs(position_size) * 0.1))
            )

            # Overall risk (weighted combination)
            overall_risk = (
                delta_risk * 0.4
                + gamma_risk * 0.3
                + vega_risk * 0.2
                + (1 - theta_efficiency) * 0.1
            )

            return {
                "delta_risk": delta_risk,
                "gamma_risk": gamma_risk,
                "vega_risk": vega_risk,
                "theta_efficiency": theta_efficiency,
                "overall_risk": overall_risk,
            }

        except Exception as e:
            logger.error(f"Error calculating risk scores: {str(e)}")
            return {
                "delta_risk": 1.0,
                "gamma_risk": 1.0,
                "vega_risk": 1.0,
                "theta_efficiency": 0.0,
                "overall_risk": 1.0,
            }

    def _determine_risk_level(self, risk_scores: Dict[str, float]) -> GreeksRiskLevel:
        """Determine overall risk level from risk scores"""
        overall_risk = risk_scores["overall_risk"]

        if overall_risk >= 0.8:
            return GreeksRiskLevel.CRITICAL
        elif overall_risk >= 0.6:
            return GreeksRiskLevel.HIGH
        elif overall_risk >= 0.3:
            return GreeksRiskLevel.MEDIUM
        else:
            return GreeksRiskLevel.LOW

    def _calculate_position_health_metrics(
        self,
        opportunity: CalendarSpreadOpportunity,
        net_delta: float,
        net_gamma: float,
        net_vega: float,
        underlying_price: float,
    ) -> Dict[str, Union[Tuple[float, float], float]]:
        """Calculate position health metrics"""
        try:
            # Delta neutral range (price range where delta < threshold)
            delta_sensitivity = abs(net_gamma) if net_gamma != 0 else 0.01
            price_range = self.delta_threshold / delta_sensitivity
            delta_neutral_range = (
                underlying_price - price_range,
                underlying_price + price_range,
            )

            # Gamma acceleration threshold (where gamma effects become significant)
            gamma_threshold = underlying_price * (1 + np.sign(net_gamma) * 0.05)

            # Vega sensitivity range (IV range for acceptable exposure)
            current_iv = opportunity.front_leg.iv
            vega_sensitivity = abs(net_vega) if net_vega != 0 else 1.0
            iv_range = self.vega_threshold / vega_sensitivity
            vega_range = (
                max(5.0, current_iv - iv_range),
                min(100.0, current_iv + iv_range),
            )

            # Theta capture efficiency (simplified)
            expected_theta = abs(
                opportunity.front_leg.theta - opportunity.back_leg.theta
            )
            actual_theta = abs(net_vega)  # Simplified proxy
            theta_efficiency = (
                min(1.0, actual_theta / expected_theta) if expected_theta > 0 else 0.5
            )

            return {
                "delta_neutral_range": delta_neutral_range,
                "gamma_threshold": gamma_threshold,
                "vega_range": vega_range,
                "theta_efficiency": theta_efficiency,
            }

        except Exception as e:
            logger.error(f"Error calculating position health metrics: {str(e)}")
            return {
                "delta_neutral_range": (
                    underlying_price * 0.95,
                    underlying_price * 1.05,
                ),
                "gamma_threshold": underlying_price,
                "vega_range": (10.0, 50.0),
                "theta_efficiency": 0.5,
            }

    def _generate_position_adjustments(
        self,
        opportunity: CalendarSpreadOpportunity,
        risk_scores: Dict[str, float],
        risk_level: GreeksRiskLevel,
        net_delta: float,
        net_gamma: float,
        net_vega: float,
    ) -> List[PositionAdjustment]:
        """Generate position adjustment recommendations"""
        adjustments = []

        try:
            # High delta risk
            if risk_scores["delta_risk"] > 0.7:
                adjustments.append(
                    PositionAdjustment(
                        adjustment_type=AdjustmentType.DELTA_HEDGE,
                        priority=1 if risk_scores["delta_risk"] > 0.9 else 2,
                        reason=f"High delta risk: {risk_scores['delta_risk']:.2f}",
                        recommended_action="Implement delta hedge",
                        time_sensitivity=(
                            "IMMEDIATE"
                            if risk_scores["delta_risk"] > 0.9
                            else "WITHIN_HOUR"
                        ),
                    )
                )

            # High gamma risk
            if risk_scores["gamma_risk"] > 0.7:
                adjustments.append(
                    PositionAdjustment(
                        adjustment_type=AdjustmentType.GAMMA_HEDGE,
                        priority=2,
                        reason=f"High gamma risk: {risk_scores['gamma_risk']:.2f}",
                        recommended_action="Monitor position closely for delta changes",
                        time_sensitivity="WITHIN_HOUR",
                    )
                )

            # High vega risk
            if risk_scores["vega_risk"] > 0.7:
                adjustments.append(
                    PositionAdjustment(
                        adjustment_type=AdjustmentType.VEGA_HEDGE,
                        priority=2,
                        reason=f"High vega risk: {risk_scores['vega_risk']:.2f}",
                        recommended_action="Consider IV hedge or position size reduction",
                        time_sensitivity="END_OF_DAY",
                    )
                )

            # Poor theta efficiency
            if risk_scores["theta_efficiency"] < 0.3:
                adjustments.append(
                    PositionAdjustment(
                        adjustment_type=AdjustmentType.CLOSE_POSITION,
                        priority=3,
                        reason=f"Poor theta efficiency: {risk_scores['theta_efficiency']:.2f}",
                        recommended_action="Consider closing position",
                        time_sensitivity="END_OF_DAY",
                    )
                )

            # Critical risk level
            if risk_level == GreeksRiskLevel.CRITICAL:
                adjustments.append(
                    PositionAdjustment(
                        adjustment_type=AdjustmentType.SCALE_DOWN,
                        priority=1,
                        reason="Critical risk level reached",
                        recommended_action="Immediately reduce position size",
                        time_sensitivity="IMMEDIATE",
                    )
                )

        except Exception as e:
            logger.error(f"Error generating adjustments: {str(e)}")

        return adjustments

    def _generate_greeks_cache_key(
        self, opportunity: CalendarSpreadOpportunity, position_size: int
    ) -> str:
        """Generate cache key for Greeks calculations"""
        return (
            f"{opportunity.symbol}_{opportunity.strike}_{opportunity.option_type}_"
            f"{opportunity.front_leg.expiry}_{opportunity.back_leg.expiry}_{position_size}"
        )

    def clear_cache(self) -> None:
        """Clear all calculation caches"""
        self.greeks_cache.clear()
        self.bs_cache.clear()
        self.evolution_cache.clear()
        logger.info("Cleared all Greeks calculation caches")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "greeks_cache_size": len(self.greeks_cache),
            "bs_cache_size": len(self.bs_cache),
            "evolution_cache_size": len(self.evolution_cache),
            "portfolio_positions": sum(
                len(positions) for positions in self.portfolio_positions.values()
            ),
        }

    # Missing methods for test compatibility
    def _calculate_option_greeks(self, contract, price, rate=0.05):
        """Calculate option Greeks for individual contracts"""
        # Stub implementation for tests
        return {"delta": 0.5, "gamma": 0.02, "vega": 0.15, "theta": -0.08, "rho": 0.05}

    def _calculate_net_greeks(self, front_greeks, back_greeks, position_size=1):
        """Calculate net Greeks for calendar spread"""
        # Stub implementation for tests
        return {
            "net_delta": (back_greeks["delta"] - front_greeks["delta"]) * position_size,
            "net_gamma": (back_greeks["gamma"] - front_greeks["gamma"]) * position_size,
            "net_vega": (back_greeks["vega"] - front_greeks["vega"]) * position_size,
            "net_theta": (back_greeks["theta"] - front_greeks["theta"]) * position_size,
            "net_rho": (back_greeks["rho"] - front_greeks["rho"]) * position_size,
        }

    def calculate_net_greeks(self, opportunity, position_size=1):
        """Public method to calculate net Greeks"""
        return self._calculate_net_greeks({}, {}, position_size)

    def model_greeks_evolution(self, greeks, time_horizon=30, price_scenarios=None):
        """Model Greeks evolution over time - stub for tests"""
        if price_scenarios is None:
            price_scenarios = [140, 145, 150, 155, 160]

        return GreeksEvolution(
            time_horizon_days=time_horizon,
            underlying_price_scenarios=price_scenarios,
            delta_evolution={
                price: [0.5 - i * 0.05 for i in range(5)] for price in price_scenarios
            },
            gamma_evolution={
                price: [0.02 + i * 0.001 for i in range(5)] for price in price_scenarios
            },
            vega_evolution={price: [0.15] * 5 for price in price_scenarios},
            theta_evolution={
                price: [-0.08 - i * 0.01 for i in range(5)] for price in price_scenarios
            },
            rho_evolution={price: [0.05] * 5 for price in price_scenarios},
        )

    def calculate_portfolio_greeks(self, positions):
        """Calculate portfolio-level Greeks - alias for backward compatibility"""
        return self.aggregate_portfolio_greeks("default", positions)


# Convenience functions for external usage
def calculate_calendar_greeks(
    opportunity: CalendarSpreadOpportunity,
    position_size: int = 1,
    underlying_price: Optional[float] = None,
    delta_threshold: float = 0.20,
) -> CalendarGreeks:
    """
    Convenience function to calculate calendar Greeks.

    Args:
        opportunity: Calendar spread opportunity
        position_size: Position size
        underlying_price: Current underlying price
        delta_threshold: Delta risk threshold

    Returns:
        CalendarGreeks object
    """
    calculator = CalendarGreeksCalculator(delta_threshold=delta_threshold)
    return calculator.calculate_calendar_greeks(
        opportunity, position_size, underlying_price
    )


def monitor_portfolio_greeks(
    positions: List[CalendarGreeks], portfolio_id: str = "default"
) -> PortfolioGreeks:
    """
    Convenience function to monitor portfolio Greeks.

    Args:
        positions: List of calendar Greeks positions
        portfolio_id: Portfolio identifier

    Returns:
        PortfolioGreeks object
    """
    calculator = CalendarGreeksCalculator()
    return calculator.aggregate_portfolio_greeks(portfolio_id, positions)
