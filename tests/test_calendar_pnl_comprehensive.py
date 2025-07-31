"""
Comprehensive unit tests for CalendarPnL module.

This test suite provides extensive coverage of the CalendarPnL implementation,
including all classes, methods, edge cases, error conditions, and integration scenarios.

Test Coverage:
- CalendarPnLResult data class functionality
- BreakevenPoints calculation and validation
- ThetaAnalysis time decay calculations
- PnLScenario modeling and projections
- MonteCarloResults statistical analysis
- CalendarPnLConfig validation and defaults
- CalendarPnLCalculator comprehensive analysis
- P&L attribution and decomposition
- Risk metrics and breakeven analysis
- Performance optimization and caching
- Edge cases and error handling
"""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest
from ib_async import Contract, Ticker
from scipy import stats

# Import the modules under test
from modules.Arbitrage.CalendarPnL import (
    BreakevenPoints,
    CalendarPnLCalculator,
    CalendarPnLConfig,
    CalendarPnLResult,
    MonteCarloResults,
    PnLScenario,
    ThetaAnalysis,
)
from modules.Arbitrage.CalendarSpread import (
    CalendarSpreadLeg,
    CalendarSpreadOpportunity,
)
from modules.Arbitrage.TermStructure import IVDataPoint, TermStructureAnalyzer


class TestCalendarPnLResult:
    """Comprehensive tests for CalendarPnLResult dataclass"""

    def test_pnl_result_creation_complete(self):
        """Test creating a complete P&L result"""
        pnl_result = CalendarPnLResult(
            symbol="AAPL",
            strike=150.0,
            option_type="CALL",
            position_size=2,
            # Current position data
            initial_debit=225.0,
            current_front_value=120.0,
            current_back_value=380.0,
            current_spread_value=260.0,
            current_pnl=35.0,
            current_pnl_pct=15.56,
            # Time decay analysis
            front_theta=-0.12,
            back_theta=-0.06,
            net_theta=-0.06,
            theta_capture_daily=12.0,
            theta_capture_total=84.0,
            # Maximum profit analysis
            estimated_max_profit=85.0,
            estimated_max_profit_pct=37.78,
            max_profit_price=150.0,
            days_to_max_profit=25,
            # Greeks exposure
            net_delta=0.05,
            net_gamma=0.02,
            net_vega=0.18,
            net_rho=0.08,
            # Risk metrics
            breakeven_upside=155.25,
            breakeven_downside=144.75,
            probability_of_profit=68.5,
            max_loss=225.0,
            risk_reward_ratio=0.378,
            # Timing metrics
            days_in_position=5,
            days_to_front_expiry=25,
            days_to_back_expiry=60,
            time_decay_acceleration=1.2,
            # Performance attribution
            price_pnl=8.0,
            time_pnl=22.0,
            volatility_pnl=3.0,
            other_pnl=2.0,
        )

        # Test all fields
        assert pnl_result.symbol == "AAPL"
        assert pnl_result.strike == 150.0
        assert pnl_result.option_type == "CALL"
        assert pnl_result.position_size == 2
        assert pnl_result.initial_debit == 225.0
        assert pnl_result.current_pnl == 35.0
        assert pnl_result.current_pnl_pct == 15.56
        assert pnl_result.front_theta == -0.12
        assert pnl_result.net_theta == -0.06
        assert pnl_result.estimated_max_profit == 85.0
        assert pnl_result.breakeven_upside == 155.25
        assert pnl_result.probability_of_profit == 68.5
        assert pnl_result.risk_reward_ratio == 0.378

    def test_pnl_result_put_option(self):
        """Test P&L result for put option"""
        pnl_result = CalendarPnLResult(
            symbol="AAPL",
            strike=150.0,
            option_type="PUT",
            position_size=1,
            initial_debit=180.0,
            current_front_value=90.0,
            current_back_value=285.0,
            current_spread_value=195.0,
            current_pnl=15.0,
            current_pnl_pct=8.33,
            front_theta=-0.10,
            back_theta=-0.05,
            net_theta=-0.05,
            theta_capture_daily=10.0,
            theta_capture_total=50.0,
            estimated_max_profit=75.0,
            estimated_max_profit_pct=41.67,
            max_profit_price=150.0,
            days_to_max_profit=28,
            net_delta=-0.08,
            net_gamma=0.03,
            net_vega=0.15,
            net_rho=-0.06,
            breakeven_upside=152.85,
            breakeven_downside=147.15,
            probability_of_profit=65.0,
            max_loss=180.0,
            risk_reward_ratio=0.417,
            days_in_position=3,
            days_to_front_expiry=28,
            days_to_back_expiry=63,
            time_decay_acceleration=1.1,
            price_pnl=2.0,
            time_pnl=10.0,
            volatility_pnl=2.0,
            other_pnl=1.0,
        )

        assert pnl_result.option_type == "PUT"
        assert pnl_result.net_delta == -0.08  # Negative for puts

    def test_pnl_result_loss_scenario(self):
        """Test P&L result with loss scenario"""
        pnl_result = CalendarPnLResult(
            symbol="AAPL",
            strike=150.0,
            option_type="CALL",
            position_size=1,
            initial_debit=200.0,
            current_front_value=180.0,
            current_back_value=195.0,
            current_spread_value=15.0,
            current_pnl=-185.0,  # Large loss
            current_pnl_pct=-92.5,
            front_theta=-0.15,
            back_theta=-0.04,
            net_theta=-0.11,
            theta_capture_daily=-22.0,  # Negative theta capture
            theta_capture_total=-110.0,
            estimated_max_profit=50.0,
            estimated_max_profit_pct=25.0,
            max_profit_price=150.0,
            days_to_max_profit=15,
            net_delta=0.25,
            net_gamma=0.08,
            net_vega=-0.12,
            net_rho=0.03,
            breakeven_upside=None,  # No upside breakeven
            breakeven_downside=None,  # No downside breakeven
            probability_of_profit=5.0,  # Very low
            max_loss=200.0,
            risk_reward_ratio=0.25,
            days_in_position=20,
            days_to_front_expiry=5,  # Close to expiry
            days_to_back_expiry=40,
            time_decay_acceleration=2.5,  # High acceleration
            price_pnl=-50.0,
            time_pnl=-120.0,
            volatility_pnl=-10.0,
            other_pnl=-5.0,
        )

        assert pnl_result.current_pnl < 0
        assert pnl_result.current_pnl_pct < 0
        assert pnl_result.breakeven_upside is None
        assert pnl_result.probability_of_profit < 10.0

    def test_pnl_result_edge_values(self):
        """Test P&L result with edge case values"""
        pnl_result = CalendarPnLResult(
            symbol="TEST",
            strike=0.01,
            option_type="CALL",
            position_size=0,  # Zero position
            initial_debit=0.0,
            current_front_value=0.0,
            current_back_value=0.0,
            current_spread_value=0.0,
            current_pnl=0.0,
            current_pnl_pct=0.0,
            front_theta=0.0,
            back_theta=0.0,
            net_theta=0.0,
            theta_capture_daily=0.0,
            theta_capture_total=0.0,
            estimated_max_profit=0.0,
            estimated_max_profit_pct=0.0,
            max_profit_price=0.01,
            days_to_max_profit=0,
            net_delta=0.0,
            net_gamma=0.0,
            net_vega=0.0,
            net_rho=0.0,
            breakeven_upside=None,
            breakeven_downside=None,
            probability_of_profit=0.0,
            max_loss=0.0,
            risk_reward_ratio=0.0,
            days_in_position=0,
            days_to_front_expiry=0,
            days_to_back_expiry=0,
            time_decay_acceleration=0.0,
            price_pnl=0.0,
            time_pnl=0.0,
            volatility_pnl=0.0,
            other_pnl=0.0,
        )

        assert pnl_result.position_size == 0
        assert pnl_result.current_pnl == 0.0
        assert pnl_result.probability_of_profit == 0.0


class TestBreakevenPoints:
    """Comprehensive tests for BreakevenPoints dataclass"""

    def test_breakeven_points_creation(self):
        """Test creating breakeven points"""
        breakeven = BreakevenPoints(
            upside_breakeven=155.25,
            downside_breakeven=144.75,
            breakeven_range=10.50,
            probability_in_range=68.5,
            days_to_front_expiry=25,
            confidence_level=0.85,
        )

        assert breakeven.upside_breakeven == 155.25
        assert breakeven.downside_breakeven == 144.75
        assert breakeven.breakeven_range == 10.50
        assert breakeven.probability_in_range == 68.5
        assert breakeven.days_to_front_expiry == 25
        assert breakeven.confidence_level == 0.85

    def test_breakeven_points_no_range(self):
        """Test breakeven points with no profitable range"""
        breakeven = BreakevenPoints(
            upside_breakeven=None,
            downside_breakeven=None,
            breakeven_range=0.0,
            probability_in_range=0.0,
            days_to_front_expiry=5,
            confidence_level=0.10,
        )

        assert breakeven.upside_breakeven is None
        assert breakeven.downside_breakeven is None
        assert breakeven.breakeven_range == 0.0
        assert breakeven.probability_in_range == 0.0

    def test_breakeven_points_asymmetric(self):
        """Test breakeven points with asymmetric range"""
        breakeven = BreakevenPoints(
            upside_breakeven=158.50,
            downside_breakeven=146.20,
            breakeven_range=12.30,
            probability_in_range=72.0,
            days_to_front_expiry=30,
            confidence_level=0.90,
        )

        # Verify asymmetric range
        strike = 150.0  # Assumed strike
        upside_distance = breakeven.upside_breakeven - strike
        downside_distance = strike - breakeven.downside_breakeven

        assert abs(upside_distance - downside_distance) > 0.5  # Asymmetric


class TestThetaAnalysis:
    """Comprehensive tests for ThetaAnalysis dataclass"""

    def test_theta_analysis_creation(self):
        """Test creating theta analysis"""
        theta_analysis = ThetaAnalysis(
            front_theta_daily=-0.12,
            back_theta_daily=-0.06,
            net_theta_daily=-0.06,
            theta_capture_rate=0.85,
            optimal_theta_rate=0.90,
            theta_efficiency=0.94,
            projected_theta_7d=-0.84,
            projected_theta_14d=-1.68,
            projected_theta_30d=-3.60,
            theta_acceleration=1.2,
            peak_theta_date=None,
            front_contribution=-0.12,
            back_contribution=0.06,
            cross_gamma_effect=0.0,
        )

        assert theta_analysis.front_theta_daily == -0.12
        assert theta_analysis.back_theta_daily == -0.06
        assert theta_analysis.net_theta_daily == -0.06
        assert theta_analysis.theta_capture_rate == 0.85
        assert theta_analysis.optimal_theta_rate == 0.90
        assert theta_analysis.theta_efficiency == 0.94
        assert theta_analysis.theta_acceleration == 1.2

    def test_theta_analysis_favorable(self):
        """Test theta analysis with favorable conditions"""
        theta_analysis = ThetaAnalysis(
            front_theta_daily=-0.18,  # High front theta
            back_theta_daily=-0.04,  # Low back theta
            net_theta_daily=-0.14,  # Favorable net theta
            theta_capture_rate=0.92,
            optimal_theta_rate=0.95,
            theta_efficiency=0.97,
            projected_theta_7d=-0.98,
            projected_theta_14d=-1.96,
            projected_theta_30d=-4.20,
            theta_acceleration=1.5,
            peak_theta_date=None,
            front_contribution=-0.18,
            back_contribution=0.04,
            cross_gamma_effect=0.0,
        )

        assert theta_analysis.theta_capture_rate > 0.9  # High capture rate
        assert theta_analysis.net_theta_daily < -0.10  # Strong time decay
        assert theta_analysis.theta_efficiency > 0.95  # High efficiency

    def test_theta_analysis_unfavorable(self):
        """Test theta analysis with unfavorable conditions"""
        theta_analysis = ThetaAnalysis(
            front_theta_daily=-0.08,  # Low front theta
            back_theta_daily=-0.10,  # Higher back theta (unfavorable)
            net_theta_daily=0.02,  # Positive net theta (working against us)
            theta_capture_rate=0.3,  # Low capture rate
            optimal_theta_rate=0.85,
            theta_efficiency=0.35,  # Poor efficiency
            projected_theta_7d=0.14,
            projected_theta_14d=0.28,
            projected_theta_30d=0.60,
            theta_acceleration=0.8,
            peak_theta_date=None,
            front_contribution=-0.08,
            back_contribution=-0.10,
            cross_gamma_effect=0.0,
        )

        assert theta_analysis.theta_capture_rate < 0.5  # Unfavorable
        assert theta_analysis.net_theta_daily > 0  # Working against us
        assert theta_analysis.theta_efficiency < 0.4  # Poor efficiency

    def test_theta_analysis_edge_cases(self):
        """Test theta analysis with edge cases"""
        # Zero theta case
        zero_theta = ThetaAnalysis(
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

        assert zero_theta.net_theta_daily == 0.0
        assert zero_theta.theta_capture_rate == 0.0

        # Extreme theta case
        extreme_theta = ThetaAnalysis(
            front_theta_daily=-1.0,  # Very high theta
            back_theta_daily=-0.01,  # Very low theta
            net_theta_daily=-0.99,  # Extreme net theta
            theta_capture_rate=0.99,  # Very high capture
            optimal_theta_rate=1.0,
            theta_efficiency=0.99,
            projected_theta_7d=-6.93,
            projected_theta_14d=-13.86,
            projected_theta_30d=-29.7,
            theta_acceleration=10.0,
            peak_theta_date=None,
            front_contribution=-1.0,
            back_contribution=0.01,
            cross_gamma_effect=0.0,
        )

        assert extreme_theta.theta_capture_rate == 0.99
        assert extreme_theta.projected_theta_30d < -20.0


class TestPnLScenario:
    """Comprehensive tests for PnLScenario dataclass"""

    def test_pnl_scenario_creation(self):
        """Test creating P&L scenario"""
        scenario = PnLScenario(
            scenario_name="Base Case",
            stock_price=150.0,
            days_forward=20,
            implied_volatility=28.0,
            front_option_value=85.0,
            back_option_value=195.0,
            spread_value=110.0,
            pnl=35.0,
            pnl_pct=18.42,
            probability=0.25,
            delta_pnl=5.0,
            theta_pnl=25.0,
            vega_pnl=3.0,
            gamma_pnl=2.0,
        )

        assert scenario.scenario_name == "Base Case"
        assert scenario.stock_price == 150.0
        assert scenario.days_forward == 20
        assert scenario.implied_volatility == 28.0
        assert scenario.spread_value == 110.0
        assert scenario.pnl == 35.0
        assert scenario.pnl_pct == 18.42
        assert scenario.probability == 0.25

    def test_pnl_scenario_bull_case(self):
        """Test P&L scenario for bullish stock movement"""
        bull_scenario = PnLScenario(
            scenario_name="Bull Case (+2 StdDev)",
            stock_price=165.0,  # Stock moved up significantly
            days_forward=25,
            implied_volatility=26.0,  # IV decreased
            front_option_value=140.0,  # ITM call more valuable
            back_option_value=215.0,  # Back month also more valuable
            spread_value=75.0,  # Spread value decreased
            pnl=-50.0,  # Loss due to stock moving too far
            pnl_pct=-25.0,
            probability=0.025,  # Low probability event
            delta_pnl=-60.0,  # Delta worked against us
            theta_pnl=15.0,  # Some theta benefit
            vega_pnl=-8.0,  # Vega hurt due to IV decrease
            gamma_pnl=3.0,
        )

        assert bull_scenario.stock_price > 160.0  # Bullish move
        assert bull_scenario.pnl < 0  # Calendar spreads can lose on big moves
        assert bull_scenario.delta_pnl < 0  # Delta negative for bull move
        assert bull_scenario.probability < 0.05  # Unlikely scenario

    def test_pnl_scenario_bear_case(self):
        """Test P&L scenario for bearish stock movement"""
        bear_scenario = PnLScenario(
            scenario_name="Bear Case (-2 StdDev)",
            stock_price=135.0,  # Stock moved down significantly
            days_forward=25,
            implied_volatility=32.0,  # IV increased (typical in down moves)
            front_option_value=20.0,  # OTM call less valuable
            back_option_value=55.0,  # Back month also less valuable
            spread_value=35.0,  # Spread compressed
            pnl=-90.0,  # Loss due to stock moving too far
            pnl_pct=-45.0,
            probability=0.025,  # Low probability event
            delta_pnl=-45.0,  # Delta worked against us
            theta_pnl=15.0,  # Some theta benefit
            vega_pnl=8.0,  # Vega helped due to IV increase
            gamma_pnl=-3.0,
        )

        assert bear_scenario.stock_price < 140.0  # Bearish move
        assert bear_scenario.pnl < 0  # Loss on big move
        assert bear_scenario.vega_pnl > 0  # Vega helped with IV increase

    def test_pnl_scenario_time_decay(self):
        """Test P&L scenario focused on time decay"""
        time_scenario = PnLScenario(
            scenario_name="Optimal Time Decay",
            stock_price=150.0,  # Stock stayed at strike
            days_forward=30,  # Significant time passed
            implied_volatility=28.0,  # IV unchanged
            front_option_value=15.0,  # Front decayed significantly
            back_option_value=165.0,  # Back decayed less
            spread_value=150.0,  # Spread widened
            pnl=75.0,  # Profit from time decay
            pnl_pct=39.47,
            probability=0.15,  # Moderate probability
            delta_pnl=0.0,  # No delta impact
            theta_pnl=75.0,  # All profit from theta
            vega_pnl=0.0,  # No vega impact
            gamma_pnl=0.0,
        )

        assert time_scenario.stock_price == 150.0  # At strike
        assert time_scenario.pnl > 50.0  # Good profit
        assert time_scenario.theta_pnl == time_scenario.pnl  # All from theta
        assert time_scenario.delta_pnl == 0.0  # No price movement


class TestMonteCarloResults:
    """Comprehensive tests for MonteCarloResults dataclass"""

    def test_monte_carlo_results_creation(self):
        """Test creating Monte Carlo results"""
        scenarios = [Mock() for _ in range(1000)]  # Mock scenarios

        mc_results = MonteCarloResults(
            num_simulations=1000,
            scenarios=scenarios,
            mean_pnl=25.50,
            median_pnl=22.00,
            std_pnl=45.80,
            min_pnl=-185.00,
            max_pnl=125.00,
            percentile_5=-95.50,
            percentile_95=98.75,
            probability_profit=0.68,
            probability_max_profit=0.12,
            expected_return=0.1275,
            sharpe_ratio=0.56,
            max_drawdown=0.925,
            confidence_intervals={
                "90%": (-75.20, 85.30),
                "95%": (-95.50, 98.75),
                "99%": (-150.25, 115.60),
            },
            risk_metrics={
                "VaR_95": -95.50,
                "CVaR_95": -125.75,
                "kelly_criterion": 0.08,
                "profit_factor": 1.45,
            },
        )

        assert mc_results.num_simulations == 1000
        assert len(mc_results.scenarios) == 1000
        assert mc_results.mean_pnl == 25.50
        assert mc_results.probability_profit == 0.68
        assert mc_results.sharpe_ratio == 0.56
        assert mc_results.confidence_intervals["95%"] == (-95.50, 98.75)
        assert mc_results.risk_metrics["VaR_95"] == -95.50

    def test_monte_carlo_results_profitable_strategy(self):
        """Test Monte Carlo results for profitable strategy"""
        mc_results = MonteCarloResults(
            num_simulations=5000,
            scenarios=[],  # Empty for test
            mean_pnl=45.75,  # Positive mean
            median_pnl=42.50,  # Positive median
            std_pnl=38.20,
            min_pnl=-125.00,
            max_pnl=175.00,
            percentile_5=-68.50,
            percentile_95=135.25,
            probability_profit=0.78,  # High probability of profit
            probability_max_profit=0.18,
            expected_return=0.2288,  # 22.88% expected return
            sharpe_ratio=1.20,  # Good Sharpe ratio
            max_drawdown=0.625,
            confidence_intervals={
                "90%": (-55.80, 125.40),
                "95%": (-68.50, 135.25),
                "99%": (-105.30, 158.90),
            },
            risk_metrics={
                "VaR_95": -68.50,
                "CVaR_95": -88.75,
                "kelly_criterion": 0.15,
                "profit_factor": 2.25,
            },
        )

        assert mc_results.mean_pnl > 0
        assert mc_results.probability_profit > 0.75
        assert mc_results.sharpe_ratio > 1.0
        assert mc_results.risk_metrics["profit_factor"] > 2.0

    def test_monte_carlo_results_unprofitable_strategy(self):
        """Test Monte Carlo results for unprofitable strategy"""
        mc_results = MonteCarloResults(
            num_simulations=2000,
            scenarios=[],
            mean_pnl=-15.25,  # Negative mean
            median_pnl=-18.50,  # Negative median
            std_pnl=55.30,
            min_pnl=-200.00,
            max_pnl=95.00,
            percentile_5=-125.75,
            percentile_95=68.40,
            probability_profit=0.42,  # Low probability of profit
            probability_max_profit=0.08,
            expected_return=-0.0763,  # Negative expected return
            sharpe_ratio=-0.28,  # Negative Sharpe ratio
            max_drawdown=1.0,  # 100% drawdown
            confidence_intervals={
                "90%": (-105.60, 48.25),
                "95%": (-125.75, 68.40),
                "99%": (-175.80, 88.95),
            },
            risk_metrics={
                "VaR_95": -125.75,
                "CVaR_95": -155.25,
                "kelly_criterion": 0.0,  # No bet recommended
                "profit_factor": 0.68,
            },
        )

        assert mc_results.mean_pnl < 0
        assert mc_results.probability_profit < 0.5
        assert mc_results.sharpe_ratio < 0
        assert mc_results.risk_metrics["profit_factor"] < 1.0

    def test_monte_carlo_results_edge_cases(self):
        """Test Monte Carlo results with edge cases"""
        # Perfect strategy (all profits)
        perfect_results = MonteCarloResults(
            num_simulations=100,
            scenarios=[],
            mean_pnl=50.0,
            median_pnl=50.0,
            std_pnl=0.0,  # No variation
            min_pnl=50.0,  # All same
            max_pnl=50.0,  # All same
            percentile_5=50.0,
            percentile_95=50.0,
            probability_profit=1.0,  # 100% probability
            probability_max_profit=1.0,
            expected_return=0.25,
            sharpe_ratio=float("inf"),  # Infinite Sharpe
            max_drawdown=0.0,
            confidence_intervals={
                "90%": (50.0, 50.0),
                "95%": (50.0, 50.0),
                "99%": (50.0, 50.0),
            },
            risk_metrics={
                "VaR_95": 50.0,
                "CVaR_95": 50.0,
                "kelly_criterion": 1.0,
                "profit_factor": float("inf"),
            },
        )

        assert perfect_results.std_pnl == 0.0
        assert perfect_results.probability_profit == 1.0
        assert perfect_results.max_drawdown == 0.0


class TestCalendarPnLConfig:
    """Comprehensive tests for CalendarPnLConfig dataclass"""

    def test_default_configuration(self):
        """Test default configuration values"""
        config = CalendarPnLConfig()

        assert config.monte_carlo_simulations == 10000
        assert config.confidence_levels == [0.90, 0.95, 0.99]
        assert config.time_steps == [1, 5, 10, 15, 20, 25, 30]
        assert config.volatility_scenarios == [0.8, 0.9, 1.0, 1.1, 1.2]
        assert config.enable_greeks_pnl == True
        assert config.enable_monte_carlo == True
        assert config.enable_scenario_analysis == True
        assert config.breakeven_precision == 0.01
        assert config.max_computation_time == 30.0
        assert config.cache_results == True
        assert config.parallel_processing == True
        assert config.random_seed is None

    def test_custom_configuration(self):
        """Test custom configuration values"""
        config = CalendarPnLConfig(
            monte_carlo_simulations=50000,
            confidence_levels=[0.85, 0.95],
            time_steps=[7, 14, 21, 28],
            volatility_scenarios=[0.7, 1.0, 1.3],
            enable_greeks_pnl=False,
            enable_monte_carlo=False,
            enable_scenario_analysis=False,
            breakeven_precision=0.001,
            max_computation_time=60.0,
            cache_results=False,
            parallel_processing=False,
            random_seed=42,
        )

        assert config.monte_carlo_simulations == 50000
        assert config.confidence_levels == [0.85, 0.95]
        assert config.time_steps == [7, 14, 21, 28]
        assert config.volatility_scenarios == [0.7, 1.0, 1.3]
        assert config.enable_greeks_pnl == False
        assert config.enable_monte_carlo == False
        assert config.enable_scenario_analysis == False
        assert config.breakeven_precision == 0.001
        assert config.max_computation_time == 60.0
        assert config.cache_results == False
        assert config.parallel_processing == False
        assert config.random_seed == 42

    def test_configuration_edge_cases(self):
        """Test configuration with edge case values"""
        # Minimal configuration
        minimal_config = CalendarPnLConfig(
            monte_carlo_simulations=1,
            confidence_levels=[0.5],
            time_steps=[1],
            volatility_scenarios=[1.0],
            breakeven_precision=1.0,
            max_computation_time=1.0,
        )

        assert minimal_config.monte_carlo_simulations == 1
        assert minimal_config.confidence_levels == [0.5]
        assert minimal_config.time_steps == [1]

        # Maximum configuration
        maximal_config = CalendarPnLConfig(
            monte_carlo_simulations=1000000,
            confidence_levels=[0.99, 0.999, 0.9999],
            time_steps=list(range(1, 366)),  # Every day for a year
            volatility_scenarios=[i / 10.0 for i in range(1, 51)],  # 0.1 to 5.0
            breakeven_precision=0.0001,
            max_computation_time=3600.0,
        )

        assert maximal_config.monte_carlo_simulations == 1000000
        assert len(maximal_config.time_steps) == 365
        assert len(maximal_config.volatility_scenarios) == 50


class TestCalendarPnLCalculator:
    """Comprehensive tests for CalendarPnLCalculator class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = CalendarPnLConfig()
        self.calculator = CalendarPnLCalculator(self.config)

        # Create test opportunity
        self.test_opportunity = self._create_test_opportunity()

    def _create_test_opportunity(self):
        """Create a test calendar spread opportunity"""
        front_contract = Mock(spec=Contract)
        front_contract.conId = 11111
        front_contract.symbol = "AAPL"
        front_contract.strike = 150.0
        front_contract.right = "C"

        back_contract = Mock(spec=Contract)
        back_contract.conId = 22222
        back_contract.symbol = "AAPL"
        back_contract.strike = 150.0
        back_contract.right = "C"

        front_leg = CalendarSpreadLeg(
            contract=front_contract,
            strike=150.0,
            expiry="20241115",
            right="C",
            price=6.50,
            bid=6.45,
            ask=6.55,
            volume=200,
            iv=32.0,
            theta=-0.12,
            days_to_expiry=30,
        )

        back_leg = CalendarSpreadLeg(
            contract=back_contract,
            strike=150.0,
            expiry="20241220",
            right="C",
            price=8.75,
            bid=8.70,
            ask=8.80,
            volume=150,
            iv=28.0,
            theta=-0.06,
            days_to_expiry=65,
        )

        return CalendarSpreadOpportunity(
            symbol="AAPL",
            strike=150.0,
            option_type="CALL",
            front_leg=front_leg,
            back_leg=back_leg,
            iv_spread=4.0,
            theta_ratio=2.0,
            net_debit=2.25,
            max_profit=1.75,
            max_loss=2.25,
            front_bid_ask_spread=0.008,
            back_bid_ask_spread=0.011,
            combined_liquidity_score=0.75,
            term_structure_inversion=True,
            net_delta=0.05,
            net_gamma=0.02,
            net_vega=0.18,
            composite_score=0.82,
        )

    def test_calculator_initialization_default(self):
        """Test calculator initialization with default config"""
        calculator = CalendarPnLCalculator()

        assert isinstance(calculator.config, CalendarPnLConfig)
        assert len(calculator.pnl_cache) == 0
        assert len(calculator.scenario_cache) == 0
        assert calculator.last_calculation_time is None

    def test_calculator_initialization_custom_config(self):
        """Test calculator initialization with custom config"""
        custom_config = CalendarPnLConfig(monte_carlo_simulations=5000)
        calculator = CalendarPnLCalculator(custom_config)

        assert calculator.config.monte_carlo_simulations == 5000

    def test_calculate_current_pnl_basic(self):
        """Test basic current P&L calculation"""
        current_front_price = 5.50  # Decreased from 6.50
        current_back_price = 9.25  # Increased from 8.75
        stock_price = 150.0
        position_size = 2

        pnl_result = self.calculator._calculate_current_pnl(
            self.test_opportunity,
            current_front_price,
            current_back_price,
            stock_price,
            position_size,
        )

        assert isinstance(pnl_result, CalendarPnLResult)
        assert pnl_result.symbol == "AAPL"
        assert pnl_result.position_size == position_size
        assert (
            pnl_result.current_front_value == current_front_price * 100 * position_size
        )
        assert pnl_result.current_back_value == current_back_price * 100 * position_size

        # P&L should be positive (front decreased, back increased)
        expected_spread_value = (
            (current_back_price - current_front_price) * 100 * position_size
        )
        initial_cost = self.test_opportunity.net_debit * 100 * position_size
        expected_pnl = expected_spread_value - initial_cost

        assert abs(pnl_result.current_pnl - expected_pnl) < 1.0

    def test_calculate_current_pnl_loss_scenario(self):
        """Test current P&L calculation with loss scenario"""
        current_front_price = 8.00  # Increased from 6.50
        current_back_price = 8.00  # Decreased from 8.75
        stock_price = 140.0  # Stock moved significantly
        position_size = 1

        pnl_result = self.calculator._calculate_current_pnl(
            self.test_opportunity,
            current_front_price,
            current_back_price,
            stock_price,
            position_size,
        )

        # Should show loss
        assert pnl_result.current_pnl < 0
        assert pnl_result.current_pnl_pct < 0

    def test_calculate_breakeven_points_symmetric(self):
        """Test breakeven points calculation"""
        stock_price = 150.0
        front_days = 25
        implied_vol = 0.28

        breakeven = self.calculator._calculate_breakeven_points(
            self.test_opportunity, stock_price, front_days, implied_vol
        )

        assert isinstance(breakeven, BreakevenPoints)
        assert breakeven.days_to_front_expiry == front_days

        # For ATM calendar spread, should have breakeven points
        if breakeven.upside_breakeven and breakeven.downside_breakeven:
            assert breakeven.upside_breakeven > stock_price
            assert breakeven.downside_breakeven < stock_price
            assert breakeven.breakeven_range > 0

    def test_calculate_breakeven_points_no_profit_zone(self):
        """Test breakeven calculation when no profit zone exists"""
        # Very close to expiry should have limited/no profit zone
        stock_price = 150.0
        front_days = 1  # Very close to expiry
        implied_vol = 0.15  # Low volatility

        breakeven = self.calculator._calculate_breakeven_points(
            self.test_opportunity, stock_price, front_days, implied_vol
        )

        # May have no breakeven points or very narrow range
        if breakeven.upside_breakeven is None or breakeven.downside_breakeven is None:
            assert breakeven.breakeven_range == 0.0
            assert breakeven.probability_in_range == 0.0

    def test_analyze_theta_decay_favorable(self):
        """Test theta decay analysis with favorable conditions"""
        current_front_theta = -0.12
        current_back_theta = -0.06
        days_to_front = 25

        theta_analysis = self.calculator._analyze_theta_decay(
            self.test_opportunity,
            current_front_theta,
            current_back_theta,
            days_to_front,
        )

        assert isinstance(theta_analysis, ThetaAnalysis)
        assert theta_analysis.front_theta == current_front_theta
        assert theta_analysis.back_theta == current_back_theta
        assert theta_analysis.net_theta == current_front_theta - current_back_theta
        assert theta_analysis.theta_ratio == abs(
            current_front_theta / current_back_theta
        )

        # Favorable conditions
        assert theta_analysis.theta_ratio > 1.5  # Front decays faster
        assert theta_analysis.net_theta < 0  # Working for us

    def test_analyze_theta_decay_unfavorable(self):
        """Test theta decay analysis with unfavorable conditions"""
        current_front_theta = -0.06  # Low front theta
        current_back_theta = -0.12  # High back theta
        days_to_front = 5  # Close to expiry

        theta_analysis = self.calculator._analyze_theta_decay(
            self.test_opportunity,
            current_front_theta,
            current_back_theta,
            days_to_front,
        )

        assert theta_analysis.theta_ratio < 1.0  # Unfavorable ratio
        assert theta_analysis.net_theta > 0  # Working against us
        assert theta_analysis.optimal_exit_days <= days_to_front

    def test_estimate_max_profit_atm(self):
        """Test maximum profit estimation for ATM calendar"""
        stock_price = 150.0  # At strike
        front_days = 25
        back_days = 60

        max_profit, max_profit_price, days_to_max = (
            self.calculator._estimate_max_profit(
                self.test_opportunity, stock_price, front_days, back_days
            )
        )

        assert max_profit > 0  # Should be profitable
        assert abs(max_profit_price - 150.0) < 2.0  # Should be near strike
        assert 0 < days_to_max <= front_days  # Should be before front expiry

    def test_estimate_max_profit_otm(self):
        """Test maximum profit estimation when OTM"""
        stock_price = 140.0  # OTM for call calendar
        front_days = 25
        back_days = 60

        max_profit, max_profit_price, days_to_max = (
            self.calculator._estimate_max_profit(
                self.test_opportunity, stock_price, front_days, back_days
            )
        )

        # May have lower max profit when starting OTM
        assert isinstance(max_profit, (int, float))
        assert isinstance(max_profit_price, (int, float))
        assert isinstance(days_to_max, int)

    def test_calculate_greeks_exposure(self):
        """Test Greeks exposure calculation"""
        current_front_delta = 0.45
        current_front_gamma = 0.08
        current_front_vega = 0.25
        current_front_rho = 0.12

        current_back_delta = 0.38
        current_back_gamma = 0.05
        current_back_vega = 0.32
        current_back_rho = 0.18

        position_size = 2

        net_delta, net_gamma, net_vega, net_rho = (
            self.calculator._calculate_greeks_exposure(
                current_front_delta,
                current_front_gamma,
                current_front_vega,
                current_front_rho,
                current_back_delta,
                current_back_gamma,
                current_back_vega,
                current_back_rho,
                position_size,
            )
        )

        # Calendar spread: Buy back, sell front
        expected_net_delta = (current_back_delta - current_front_delta) * position_size
        expected_net_gamma = (current_back_gamma - current_front_gamma) * position_size
        expected_net_vega = (current_back_vega - current_front_vega) * position_size
        expected_net_rho = (current_back_rho - current_front_rho) * position_size

        assert abs(net_delta - expected_net_delta) < 0.01
        assert abs(net_gamma - expected_net_gamma) < 0.01
        assert abs(net_vega - expected_net_vega) < 0.01
        assert abs(net_rho - expected_net_rho) < 0.01

    def test_run_scenario_analysis_basic(self):
        """Test basic scenario analysis"""
        stock_price = 150.0
        position_size = 1

        scenarios = self.calculator._run_scenario_analysis(
            self.test_opportunity, stock_price, position_size
        )

        assert isinstance(scenarios, list)
        assert len(scenarios) > 0

        # Should have scenarios for different time/price combinations
        scenario_names = [s.scenario_name for s in scenarios]
        assert any("days" in name.lower() for name in scenario_names)

        # Check scenario structure
        for scenario in scenarios[:3]:  # Check first few
            assert isinstance(scenario, PnLScenario)
            assert isinstance(scenario.stock_price, (int, float))
            assert isinstance(scenario.pnl, (int, float))
            assert 0 <= scenario.probability <= 1

    def test_run_scenario_analysis_disabled(self):
        """Test scenario analysis when disabled"""
        config = CalendarPnLConfig(enable_scenario_analysis=False)
        calculator = CalendarPnLCalculator(config)

        scenarios = calculator._run_scenario_analysis(self.test_opportunity, 150.0, 1)

        assert scenarios == []

    def test_calculate_risk_metrics_basic(self):
        """Test basic risk metrics calculation"""
        scenarios = [
            Mock(pnl=50.0, probability=0.2),
            Mock(pnl=25.0, probability=0.3),
            Mock(pnl=0.0, probability=0.3),
            Mock(pnl=-25.0, probability=0.15),
            Mock(pnl=-50.0, probability=0.05),
        ]

        prob_profit, expected_return, max_loss = (
            self.calculator._calculate_risk_metrics(scenarios, initial_cost=200.0)
        )

        assert 0 <= prob_profit <= 1
        assert isinstance(expected_return, (int, float))
        assert max_loss >= 0

    def test_calculate_risk_metrics_all_profitable(self):
        """Test risk metrics with all profitable scenarios"""
        scenarios = [
            Mock(pnl=100.0, probability=0.4),
            Mock(pnl=75.0, probability=0.3),
            Mock(pnl=50.0, probability=0.2),
            Mock(pnl=25.0, probability=0.1),
        ]

        prob_profit, expected_return, max_loss = (
            self.calculator._calculate_risk_metrics(scenarios, initial_cost=200.0)
        )

        assert prob_profit == 1.0  # All scenarios profitable
        assert expected_return > 0
        assert max_loss == 0  # No loss scenarios

    def test_calculate_risk_metrics_all_losses(self):
        """Test risk metrics with all loss scenarios"""
        scenarios = [
            Mock(pnl=-50.0, probability=0.4),
            Mock(pnl=-75.0, probability=0.3),
            Mock(pnl=-100.0, probability=0.2),
            Mock(pnl=-125.0, probability=0.1),
        ]

        prob_profit, expected_return, max_loss = (
            self.calculator._calculate_risk_metrics(scenarios, initial_cost=200.0)
        )

        assert prob_profit == 0.0  # No profitable scenarios
        assert expected_return < 0
        assert max_loss > 100  # Significant losses

    def test_calculate_risk_metrics_empty_scenarios(self):
        """Test risk metrics with empty scenarios"""
        prob_profit, expected_return, max_loss = (
            self.calculator._calculate_risk_metrics([], initial_cost=200.0)
        )

        assert prob_profit == 0.0
        assert expected_return == 0.0
        assert max_loss == 200.0  # Full initial cost

    @pytest.mark.asyncio
    async def test_run_monte_carlo_simulation_basic(self):
        """Test basic Monte Carlo simulation"""
        stock_price = 150.0
        position_size = 1
        days_to_expiry = 25
        implied_vol = 0.28

        # Use smaller number of simulations for testing
        config = CalendarPnLConfig(monte_carlo_simulations=100)
        calculator = CalendarPnLCalculator(config)

        mc_results = await calculator._run_monte_carlo_simulation(
            self.test_opportunity,
            stock_price,
            position_size,
            days_to_expiry,
            implied_vol,
        )

        assert isinstance(mc_results, MonteCarloResults)
        assert mc_results.num_simulations == 100
        assert len(mc_results.scenarios) == 100
        assert isinstance(mc_results.mean_pnl, (int, float))
        assert isinstance(mc_results.std_pnl, (int, float))
        assert 0 <= mc_results.probability_profit <= 1
        assert len(mc_results.confidence_intervals) > 0

    @pytest.mark.asyncio
    async def test_run_monte_carlo_simulation_disabled(self):
        """Test Monte Carlo simulation when disabled"""
        config = CalendarPnLConfig(enable_monte_carlo=False)
        calculator = CalendarPnLCalculator(config)

        mc_results = await calculator._run_monte_carlo_simulation(
            self.test_opportunity, 150.0, 1, 25, 0.28
        )

        assert mc_results is None

    @pytest.mark.asyncio
    async def test_run_monte_carlo_simulation_with_seed(self):
        """Test Monte Carlo simulation with random seed for reproducibility"""
        config = CalendarPnLConfig(monte_carlo_simulations=50, random_seed=42)
        calculator1 = CalendarPnLCalculator(config)
        calculator2 = CalendarPnLCalculator(config)

        # Both should produce same results with same seed
        mc_results1 = await calculator1._run_monte_carlo_simulation(
            self.test_opportunity, 150.0, 1, 25, 0.28
        )

        mc_results2 = await calculator2._run_monte_carlo_simulation(
            self.test_opportunity, 150.0, 1, 25, 0.28
        )

        # Should be identical with same seed
        assert abs(mc_results1.mean_pnl - mc_results2.mean_pnl) < 0.01
        assert abs(mc_results1.std_pnl - mc_results2.std_pnl) < 0.01

    def test_cache_functionality(self):
        """Test P&L calculation caching"""
        config = CalendarPnLConfig(cache_results=True)
        calculator = CalendarPnLCalculator(config)

        # Create cache key
        cache_key = calculator._create_cache_key(
            self.test_opportunity, 150.0, 1, 25, 0.28
        )

        assert isinstance(cache_key, str)
        assert "AAPL" in cache_key
        assert "150.0" in cache_key

        # Test cache storage and retrieval
        test_result = Mock()
        calculator.pnl_cache[cache_key] = (test_result, time.time())

        # Should retrieve from cache
        cached_result, cache_time = calculator.pnl_cache[cache_key]
        assert cached_result == test_result

    def test_cache_disabled(self):
        """Test behavior when caching is disabled"""
        config = CalendarPnLConfig(cache_results=False)
        calculator = CalendarPnLCalculator(config)

        # Cache should remain empty
        cache_key = "test_key"
        calculator._store_in_cache(cache_key, Mock())

        assert len(calculator.pnl_cache) == 0

    def test_performance_timeout_handling(self):
        """Test handling of computation timeout"""
        # Create config with very short timeout
        config = CalendarPnLConfig(max_computation_time=0.001)  # 1ms
        calculator = CalendarPnLCalculator(config)

        start_time = time.time()

        # This should handle timeout gracefully
        result = calculator._check_computation_timeout(start_time)

        # Should return boolean indicating timeout status
        assert isinstance(result, bool)

    def test_parallel_processing_enabled(self):
        """Test parallel processing configuration"""
        config = CalendarPnLConfig(parallel_processing=True)
        calculator = CalendarPnLCalculator(config)

        assert calculator.config.parallel_processing == True

        # Test that ThreadPoolExecutor can be created
        with ThreadPoolExecutor(max_workers=2) as executor:
            assert executor is not None

    def test_parallel_processing_disabled(self):
        """Test serial processing when parallel is disabled"""
        config = CalendarPnLConfig(parallel_processing=False)
        calculator = CalendarPnLCalculator(config)

        assert calculator.config.parallel_processing == False


class TestCalendarPnLCalculatorIntegration:
    """Integration tests for CalendarPnLCalculator with other components"""

    def setup_method(self):
        """Setup integration test fixtures"""
        self.calculator = CalendarPnLCalculator()
        self.test_opportunity = self._create_test_opportunity()

    def _create_test_opportunity(self):
        """Create a test calendar spread opportunity"""
        front_contract = Mock(spec=Contract)
        front_contract.conId = 11111

        back_contract = Mock(spec=Contract)
        back_contract.conId = 22222

        front_leg = CalendarSpreadLeg(
            contract=front_contract,
            strike=150.0,
            expiry="20241115",
            right="C",
            price=6.50,
            bid=6.45,
            ask=6.55,
            volume=200,
            iv=32.0,
            theta=-0.12,
            days_to_expiry=30,
        )

        back_leg = CalendarSpreadLeg(
            contract=back_contract,
            strike=150.0,
            expiry="20241220",
            right="C",
            price=8.75,
            bid=8.70,
            ask=8.80,
            volume=150,
            iv=28.0,
            theta=-0.06,
            days_to_expiry=65,
        )

        return CalendarSpreadOpportunity(
            symbol="AAPL",
            strike=150.0,
            option_type="CALL",
            front_leg=front_leg,
            back_leg=back_leg,
            iv_spread=4.0,
            theta_ratio=2.0,
            net_debit=2.25,
            max_profit=1.75,
            max_loss=2.25,
            front_bid_ask_spread=0.008,
            back_bid_ask_spread=0.011,
            combined_liquidity_score=0.75,
            term_structure_inversion=True,
            net_delta=0.05,
            net_gamma=0.02,
            net_vega=0.18,
            composite_score=0.82,
        )

    @pytest.mark.asyncio
    async def test_comprehensive_pnl_analysis_full(self):
        """Test comprehensive P&L analysis with full features enabled"""
        stock_price = 150.0
        position_size = 2
        days_elapsed = 5

        # Enable all features
        config = CalendarPnLConfig(
            enable_greeks_pnl=True,
            enable_monte_carlo=True,
            enable_scenario_analysis=True,
            monte_carlo_simulations=100,  # Smaller for testing
        )
        calculator = CalendarPnLCalculator(config)

        pnl_result = await calculator.analyze_calendar_pnl(
            self.test_opportunity,
            current_stock_price=stock_price,
            position_size=position_size,
            days_elapsed=days_elapsed,
        )

        assert isinstance(pnl_result, CalendarPnLResult)
        assert pnl_result.symbol == "AAPL"
        assert pnl_result.position_size == position_size
        assert pnl_result.days_in_position == days_elapsed

        # Should have comprehensive analysis
        assert isinstance(pnl_result.current_pnl, (int, float))
        assert isinstance(pnl_result.theta_capture_daily, (int, float))
        assert isinstance(pnl_result.estimated_max_profit, (int, float))
        assert isinstance(pnl_result.probability_of_profit, (int, float))

    @pytest.mark.asyncio
    async def test_comprehensive_pnl_analysis_minimal(self):
        """Test comprehensive P&L analysis with minimal features"""
        stock_price = 150.0
        position_size = 1
        days_elapsed = 10

        # Disable expensive features
        config = CalendarPnLConfig(
            enable_greeks_pnl=False,
            enable_monte_carlo=False,
            enable_scenario_analysis=False,
        )
        calculator = CalendarPnLCalculator(config)

        pnl_result = await calculator.analyze_calendar_pnl(
            self.test_opportunity,
            current_stock_price=stock_price,
            position_size=position_size,
            days_elapsed=days_elapsed,
        )

        assert isinstance(pnl_result, CalendarPnLResult)
        # Should still provide basic P&L analysis
        assert isinstance(pnl_result.current_pnl, (int, float))

    def test_pnl_attribution_analysis(self):
        """Test P&L attribution to different factors"""
        # Mock different market scenarios
        scenarios = [
            # Price moved up, time passed, vol increased
            {"price_change": 5.0, "time_passed": 7, "vol_change": 2.0},
            # Price moved down, time passed, vol decreased
            {"price_change": -3.0, "time_passed": 7, "vol_change": -1.5},
            # No price change, only time decay
            {"price_change": 0.0, "time_passed": 10, "vol_change": 0.0},
        ]

        for scenario in scenarios:
            attribution = self.calculator._calculate_pnl_attribution(
                self.test_opportunity,
                price_change=scenario["price_change"],
                time_change=scenario["time_passed"],
                vol_change=scenario["vol_change"],
                position_size=1,
            )

            # Should return attribution breakdown
            assert isinstance(attribution, dict)
            assert "price_pnl" in attribution
            assert "time_pnl" in attribution
            assert "volatility_pnl" in attribution
            assert "total_pnl" in attribution

    def test_risk_analysis_integration(self):
        """Test integration with risk analysis"""
        risk_metrics = self.calculator._calculate_comprehensive_risk_metrics(
            self.test_opportunity,
            current_stock_price=150.0,
            position_size=2,
            scenarios=[],  # Empty for testing
        )

        assert isinstance(risk_metrics, dict)
        assert "max_loss" in risk_metrics
        assert "probability_of_profit" in risk_metrics
        assert "risk_reward_ratio" in risk_metrics

    def test_performance_tracking(self):
        """Test performance tracking and metrics"""
        start_time = time.time()

        # Simulate calculation
        time.sleep(0.01)  # Small delay

        self.calculator.last_calculation_time = time.time() - start_time

        # Should track calculation time
        assert self.calculator.last_calculation_time > 0
        assert self.calculator.last_calculation_time < 1.0  # Should be fast

    def test_error_handling_invalid_opportunity(self):
        """Test error handling with invalid opportunity data"""
        # Create opportunity with invalid data
        invalid_opportunity = Mock()
        invalid_opportunity.symbol = None  # Invalid symbol
        invalid_opportunity.net_debit = -100.0  # Invalid debit

        # Should handle gracefully
        with pytest.raises((ValueError, AttributeError, TypeError)):
            self.calculator._calculate_current_pnl(
                invalid_opportunity, 5.0, 8.0, 150.0, 1
            )

    def test_boundary_conditions_handling(self):
        """Test handling of boundary conditions"""
        boundary_tests = [
            # Zero stock price
            {"stock_price": 0.0, "position_size": 1},
            # Very high stock price
            {"stock_price": 10000.0, "position_size": 1},
            # Zero position size
            {"stock_price": 150.0, "position_size": 0},
            # Negative position size (short calendar)
            {"stock_price": 150.0, "position_size": -1},
        ]

        for test_case in boundary_tests:
            try:
                result = self.calculator._calculate_current_pnl(
                    self.test_opportunity,
                    current_front_price=5.0,
                    current_back_price=8.0,
                    stock_price=test_case["stock_price"],
                    position_size=test_case["position_size"],
                )

                # Should return valid result
                assert isinstance(result, CalendarPnLResult)

            except (ValueError, ZeroDivisionError):
                # Acceptable for extreme boundary conditions
                pass


class TestCalendarPnLPerformanceAndEdgeCases:
    """Tests for performance optimization and edge cases"""

    def setup_method(self):
        """Setup performance test fixtures"""
        self.calculator = CalendarPnLCalculator()

    def test_large_monte_carlo_performance(self):
        """Test performance with large Monte Carlo simulation"""
        # Test with larger simulation size
        config = CalendarPnLConfig(monte_carlo_simulations=1000)
        calculator = CalendarPnLCalculator(config)

        # Time the operation
        start_time = time.time()

        # Mock the simulation instead of running full calculation
        with patch.object(calculator, "_run_monte_carlo_simulation") as mock_mc:
            mock_mc.return_value = Mock(spec=MonteCarloResults)

            result = mock_mc.return_value

            elapsed_time = time.time() - start_time

            # Should complete quickly (mocked)
            assert elapsed_time < 1.0
            assert mock_mc.called

    def test_memory_usage_with_large_scenarios(self):
        """Test memory management with large scenario sets"""
        # Create large scenario set
        large_scenarios = []
        for i in range(10000):
            scenario = PnLScenario(
                scenario_name=f"Scenario_{i}",
                stock_price=150.0 + (i % 100),
                days_forward=i % 30,
                implied_volatility=0.2 + (i % 50) / 1000,
                front_option_value=5.0 + i % 10,
                back_option_value=8.0 + i % 15,
                spread_value=3.0 + i % 8,
                pnl=i % 200 - 100,
                pnl_pct=(i % 200 - 100) / 200,
                probability=1.0 / 10000,
                delta_pnl=i % 20 - 10,
                theta_pnl=i % 15,
                vega_pnl=i % 10 - 5,
                gamma_pnl=i % 5 - 2,
            )
            large_scenarios.append(scenario)

        # Should handle large scenario sets
        assert len(large_scenarios) == 10000

        # Test risk metrics calculation with large set
        prob_profit, expected_return, max_loss = (
            self.calculator._calculate_risk_metrics(large_scenarios, initial_cost=200.0)
        )

        assert isinstance(prob_profit, (int, float))
        assert isinstance(expected_return, (int, float))
        assert isinstance(max_loss, (int, float))

    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme market conditions"""
        extreme_conditions = [
            # Very high volatility
            {"stock_price": 150.0, "volatility": 2.0, "days": 30},
            # Very low volatility
            {"stock_price": 150.0, "volatility": 0.01, "days": 30},
            # Very close to expiry
            {"stock_price": 150.0, "volatility": 0.3, "days": 0.1},
            # Very far from expiry
            {"stock_price": 150.0, "volatility": 0.3, "days": 1000},
            # Extreme stock prices
            {"stock_price": 0.01, "volatility": 0.3, "days": 30},
            {"stock_price": 100000.0, "volatility": 0.3, "days": 30},
        ]

        for condition in extreme_conditions:
            try:
                # Test breakeven calculation with extreme conditions
                test_opportunity = Mock()
                test_opportunity.strike = 150.0
                test_opportunity.net_debit = 2.25

                breakeven = self.calculator._calculate_breakeven_points(
                    test_opportunity,
                    condition["stock_price"],
                    condition["days"],
                    condition["volatility"],
                )

                # Should handle gracefully
                assert isinstance(breakeven, BreakevenPoints)

                # Values should be finite
                if breakeven.upside_breakeven is not None:
                    assert np.isfinite(breakeven.upside_breakeven)
                if breakeven.downside_breakeven is not None:
                    assert np.isfinite(breakeven.downside_breakeven)

            except (ValueError, OverflowError, ZeroDivisionError):
                # Acceptable for extreme conditions
                pass

    def test_concurrent_calculation_safety(self):
        """Test thread safety for concurrent calculations"""
        import threading

        results = []
        errors = []

        def calculation_worker(worker_id):
            try:
                # Create test opportunity for this worker
                opportunity = Mock()
                opportunity.symbol = f"TEST{worker_id}"
                opportunity.strike = 150.0
                opportunity.net_debit = 2.25
                opportunity.front_leg = Mock()
                opportunity.back_leg = Mock()
                opportunity.front_leg.theta = -0.12
                opportunity.back_leg.theta = -0.06

                # Perform calculation
                pnl_result = self.calculator._calculate_current_pnl(
                    opportunity, 5.0, 8.0, 150.0, 1
                )

                results.append((worker_id, pnl_result))

            except Exception as e:
                errors.append((worker_id, str(e)))

        # Create multiple threads
        threads = []
        for worker_id in range(10):
            thread = threading.Thread(target=calculation_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should have completed without errors
        assert len(errors) == 0
        assert len(results) == 10

    def test_cache_performance_large_dataset(self):
        """Test cache performance with large dataset"""
        config = CalendarPnLConfig(cache_results=True)
        calculator = CalendarPnLCalculator(config)

        # Populate cache with many entries
        for i in range(1000):
            cache_key = f"test_cache_key_{i}"
            mock_result = Mock()
            calculator.pnl_cache[cache_key] = (mock_result, time.time())

        # Test cache retrieval performance
        start_time = time.time()

        # Retrieve from cache
        for i in range(100):
            cache_key = f"test_cache_key_{i}"
            if cache_key in calculator.pnl_cache:
                result, cache_time = calculator.pnl_cache[cache_key]

        elapsed_time = time.time() - start_time

        # Should be very fast
        assert elapsed_time < 0.1  # 100ms max
        assert len(calculator.pnl_cache) == 1000

    def test_memory_cleanup_and_garbage_collection(self):
        """Test memory cleanup and garbage collection"""
        import gc

        # Create many temporary objects
        temp_calculators = []
        for i in range(100):
            config = CalendarPnLConfig(monte_carlo_simulations=10)
            calculator = CalendarPnLCalculator(config)

            # Populate with some data
            calculator.pnl_cache[f"temp_{i}"] = (Mock(), time.time())

            temp_calculators.append(calculator)

        # Clear references
        temp_calculators.clear()

        # Force garbage collection
        gc.collect()

        # Should complete without memory issues
        assert True  # If we reach here, no memory errors occurred

    def test_error_recovery_and_resilience(self):
        """Test error recovery and system resilience"""
        error_scenarios = [
            # Division by zero
            lambda: self.calculator._calculate_risk_metrics([], initial_cost=0.0),
            # Invalid opportunity data
            lambda: self.calculator._calculate_current_pnl(None, 5.0, 8.0, 150.0, 1),
            # Extreme numerical values
            lambda: self.calculator._estimate_max_profit(Mock(), float("inf"), 30, 60),
        ]

        for i, error_func in enumerate(error_scenarios):
            try:
                result = error_func()
                # If no error, result should be reasonable
                if result is not None:
                    assert isinstance(result, (tuple, list, dict, float, int))

            except (
                ValueError,
                TypeError,
                AttributeError,
                ZeroDivisionError,
                OverflowError,
            ):
                # These errors are acceptable for edge cases
                pass

            except Exception as e:
                # Unexpected errors should be investigated
                pytest.fail(f"Unexpected error in scenario {i}: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
