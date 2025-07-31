"""
Comprehensive unit tests for CalendarGreeks module.

This test suite provides extensive coverage of the CalendarGreeks implementation,
including all classes, methods, edge cases, error conditions, and integration scenarios.

Test Coverage:
- GreeksRiskLevel enum functionality
- AdjustmentType enum functionality
- GreeksEvolution scenario modeling
- PositionAdjustment recommendations
- CalendarGreeks main calculations
- PortfolioGreeks aggregation
- CalendarGreeksCalculator comprehensive analysis
- Risk assessment and threshold monitoring
- Time-based Greeks evolution
- Performance optimization and numerical stability
- Edge cases and error handling
"""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from enum import Enum
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest
from ib_async import Contract, Ticker
from scipy import stats

# Import the modules under test
from modules.Arbitrage.CalendarGreeks import (
    AdjustmentType,
    CalendarGreeks,
    CalendarGreeksCalculator,
    GreeksEvolution,
    GreeksRiskLevel,
    PortfolioGreeks,
    PositionAdjustment,
)
from modules.Arbitrage.CalendarPnL import CalendarPnLResult
from modules.Arbitrage.CalendarSpread import (
    CalendarSpreadLeg,
    CalendarSpreadOpportunity,
)
from modules.Arbitrage.TermStructure import IVDataPoint, TermStructureAnalyzer


class TestGreeksRiskLevel:
    """Comprehensive tests for GreeksRiskLevel enum"""

    def test_risk_level_values(self):
        """Test that all risk levels have correct values"""
        assert GreeksRiskLevel.LOW.value == "LOW"
        assert GreeksRiskLevel.MEDIUM.value == "MEDIUM"
        assert GreeksRiskLevel.HIGH.value == "HIGH"
        assert GreeksRiskLevel.CRITICAL.value == "CRITICAL"

    def test_risk_level_ordering(self):
        """Test risk level ordering logic"""
        risk_levels = [
            GreeksRiskLevel.LOW,
            GreeksRiskLevel.MEDIUM,
            GreeksRiskLevel.HIGH,
            GreeksRiskLevel.CRITICAL,
        ]

        # Should be orderable by severity
        assert len(risk_levels) == 4
        assert GreeksRiskLevel.LOW in risk_levels
        assert GreeksRiskLevel.CRITICAL in risk_levels

    def test_risk_level_comparison(self):
        """Test risk level comparison functionality"""
        # Test enum membership
        assert GreeksRiskLevel.LOW != GreeksRiskLevel.HIGH
        assert GreeksRiskLevel.CRITICAL != GreeksRiskLevel.LOW

        # Test string representation
        assert str(GreeksRiskLevel.LOW) == "GreeksRiskLevel.LOW"
        assert repr(GreeksRiskLevel.HIGH) == "<GreeksRiskLevel.HIGH: 'HIGH'>"

    def test_risk_level_iteration(self):
        """Test iteration over risk levels"""
        all_levels = list(GreeksRiskLevel)
        assert len(all_levels) == 4
        assert GreeksRiskLevel.LOW in all_levels
        assert GreeksRiskLevel.CRITICAL in all_levels

    def test_risk_level_lookup(self):
        """Test risk level lookup by value"""
        assert GreeksRiskLevel("LOW") == GreeksRiskLevel.LOW
        assert GreeksRiskLevel("CRITICAL") == GreeksRiskLevel.CRITICAL

        # Test invalid lookup
        with pytest.raises(ValueError):
            GreeksRiskLevel("INVALID")


class TestAdjustmentType:
    """Comprehensive tests for AdjustmentType enum"""

    def test_adjustment_type_values(self):
        """Test that all adjustment types have correct values"""
        assert AdjustmentType.DELTA_HEDGE.value == "DELTA_HEDGE"
        assert AdjustmentType.GAMMA_HEDGE.value == "GAMMA_HEDGE"
        assert AdjustmentType.VEGA_HEDGE.value == "VEGA_HEDGE"
        assert AdjustmentType.CLOSE_POSITION.value == "CLOSE_POSITION"
        assert AdjustmentType.ROLL_POSITION.value == "ROLL_POSITION"
        assert AdjustmentType.SCALE_DOWN.value == "SCALE_DOWN"

    def test_adjustment_type_coverage(self):
        """Test that we have all expected adjustment types"""
        expected_types = {
            "DELTA_HEDGE",
            "GAMMA_HEDGE",
            "VEGA_HEDGE",
            "CLOSE_POSITION",
            "ROLL_POSITION",
            "SCALE_DOWN",
        }

        actual_types = {adj_type.value for adj_type in AdjustmentType}
        assert actual_types == expected_types

    def test_adjustment_type_string_representation(self):
        """Test string representation of adjustment types"""
        assert str(AdjustmentType.DELTA_HEDGE) == "AdjustmentType.DELTA_HEDGE"
        assert AdjustmentType.VEGA_HEDGE.name == "VEGA_HEDGE"

    def test_adjustment_type_lookup(self):
        """Test adjustment type lookup"""
        assert AdjustmentType("DELTA_HEDGE") == AdjustmentType.DELTA_HEDGE
        assert AdjustmentType("CLOSE_POSITION") == AdjustmentType.CLOSE_POSITION

        with pytest.raises(ValueError):
            AdjustmentType("INVALID_TYPE")


class TestGreeksEvolution:
    """Comprehensive tests for GreeksEvolution dataclass"""

    def test_greeks_evolution_creation_basic(self):
        """Test basic Greeks evolution creation"""
        price_scenarios = [140.0, 145.0, 150.0, 155.0, 160.0]
        time_horizon = 30

        delta_evolution = {
            140.0: [0.20, 0.18, 0.15, 0.12, 0.08],
            145.0: [0.35, 0.32, 0.28, 0.24, 0.18],
            150.0: [0.50, 0.45, 0.40, 0.35, 0.28],
            155.0: [0.65, 0.58, 0.52, 0.45, 0.38],
            160.0: [0.80, 0.72, 0.65, 0.58, 0.48],
        }

        gamma_evolution = {
            140.0: [0.01, 0.012, 0.014, 0.015, 0.014],
            145.0: [0.02, 0.022, 0.024, 0.025, 0.024],
            150.0: [0.03, 0.032, 0.034, 0.035, 0.034],
            155.0: [0.02, 0.022, 0.024, 0.025, 0.024],
            160.0: [0.01, 0.012, 0.014, 0.015, 0.014],
        }

        evolution = GreeksEvolution(
            time_horizon_days=time_horizon,
            underlying_price_scenarios=price_scenarios,
            delta_evolution=delta_evolution,
            gamma_evolution=gamma_evolution,
            vega_evolution={price: [0.15] * 5 for price in price_scenarios},
            theta_evolution={
                price: [-0.08, -0.09, -0.10, -0.12, -0.15] for price in price_scenarios
            },
            rho_evolution={price: [0.05] * 5 for price in price_scenarios},
            scenario_probabilities=[0.1, 0.2, 0.4, 0.2, 0.1],
            expected_values={
                "delta": [0.50, 0.45, 0.40, 0.35, 0.28],
                "gamma": [0.03, 0.032, 0.034, 0.035, 0.034],
            },
        )

        assert evolution.time_horizon_days == 30
        assert len(evolution.underlying_price_scenarios) == 5
        assert len(evolution.delta_evolution) == 5
        assert len(evolution.gamma_evolution) == 5
        assert len(evolution.scenario_probabilities) == 5
        assert sum(evolution.scenario_probabilities) == 1.0

        # Test data structure consistency
        for price in price_scenarios:
            assert price in evolution.delta_evolution
            assert price in evolution.gamma_evolution
            assert len(evolution.delta_evolution[price]) == 5  # 5 time steps

    def test_greeks_evolution_empty_scenarios(self):
        """Test Greeks evolution with empty scenarios"""
        evolution = GreeksEvolution(
            time_horizon_days=0,
            underlying_price_scenarios=[],
            delta_evolution={},
            gamma_evolution={},
            vega_evolution={},
            theta_evolution={},
            rho_evolution={},
        )

        assert evolution.time_horizon_days == 0
        assert len(evolution.underlying_price_scenarios) == 0
        assert len(evolution.delta_evolution) == 0
        assert len(evolution.scenario_probabilities) == 0

    def test_greeks_evolution_single_scenario(self):
        """Test Greeks evolution with single price scenario"""
        single_price = 150.0
        evolution = GreeksEvolution(
            time_horizon_days=7,
            underlying_price_scenarios=[single_price],
            delta_evolution={single_price: [0.50, 0.48, 0.45, 0.42, 0.38, 0.33, 0.28]},
            gamma_evolution={
                single_price: [0.03, 0.032, 0.034, 0.035, 0.034, 0.032, 0.028]
            },
            vega_evolution={single_price: [0.25, 0.23, 0.21, 0.19, 0.16, 0.13, 0.10]},
            theta_evolution={
                single_price: [-0.08, -0.09, -0.10, -0.12, -0.15, -0.18, -0.22]
            },
            rho_evolution={
                single_price: [0.05, 0.048, 0.045, 0.042, 0.038, 0.033, 0.028]
            },
            scenario_probabilities=[1.0],
            expected_values={"delta": [0.50, 0.48, 0.45, 0.42, 0.38, 0.33, 0.28]},
        )

        assert len(evolution.underlying_price_scenarios) == 1
        assert evolution.scenario_probabilities[0] == 1.0
        assert len(evolution.delta_evolution[single_price]) == 7

    def test_greeks_evolution_extreme_scenarios(self):
        """Test Greeks evolution with extreme market scenarios"""
        extreme_prices = [50.0, 100.0, 200.0, 500.0]  # Wide price range

        evolution = GreeksEvolution(
            time_horizon_days=1,
            underlying_price_scenarios=extreme_prices,
            delta_evolution={
                50.0: [0.01],  # Deep OTM
                100.0: [0.25],  # OTM
                200.0: [0.75],  # ITM
                500.0: [0.99],  # Deep ITM
            },
            gamma_evolution={price: [0.001] for price in extreme_prices},
            vega_evolution={price: [0.1] for price in extreme_prices},
            theta_evolution={price: [-0.05] for price in extreme_prices},
            rho_evolution={price: [0.02] for price in extreme_prices},
            scenario_probabilities=[0.25, 0.25, 0.25, 0.25],
        )

        assert len(evolution.underlying_price_scenarios) == 4
        assert all(0 <= delta[0] <= 1 for delta in evolution.delta_evolution.values())

    def test_greeks_evolution_data_validation(self):
        """Test data validation in Greeks evolution"""
        # Test mismatched scenario lengths
        evolution = GreeksEvolution(
            time_horizon_days=5,
            underlying_price_scenarios=[140.0, 150.0, 160.0],
            delta_evolution={
                140.0: [0.3, 0.25, 0.2],  # 3 time steps
                150.0: [0.5, 0.45, 0.4],  # 3 time steps
                160.0: [0.7, 0.65, 0.6],  # 3 time steps
            },
            gamma_evolution={
                price: [0.02, 0.025, 0.03] for price in [140.0, 150.0, 160.0]
            },
            vega_evolution={
                price: [0.15, 0.12, 0.1] for price in [140.0, 150.0, 160.0]
            },
            theta_evolution={
                price: [-0.08, -0.1, -0.12] for price in [140.0, 150.0, 160.0]
            },
            rho_evolution={
                price: [0.05, 0.04, 0.03] for price in [140.0, 150.0, 160.0]
            },
            scenario_probabilities=[0.3, 0.4, 0.3],  # Should sum to 1.0
        )

        assert len(evolution.scenario_probabilities) == 3
        assert abs(sum(evolution.scenario_probabilities) - 1.0) < 0.001


class TestPositionAdjustment:
    """Comprehensive tests for PositionAdjustment dataclass"""

    def test_position_adjustment_creation_basic(self):
        """Test basic position adjustment creation"""
        mock_contract = Mock(spec=Contract)
        mock_contract.symbol = "AAPL"
        mock_contract.strike = 150.0

        adjustment = PositionAdjustment(
            adjustment_type=AdjustmentType.DELTA_HEDGE,
            priority=1,
            reason="Delta exposure exceeds risk limits",
            recommended_action="Buy 100 shares of AAPL to hedge delta",
            hedge_quantity=100,
            hedge_contract=mock_contract,
            expected_cost=15000.0,
            risk_reduction=0.85,
            time_sensitivity="IMMEDIATE",
        )

        assert adjustment.adjustment_type == AdjustmentType.DELTA_HEDGE
        assert adjustment.priority == 1
        assert adjustment.reason == "Delta exposure exceeds risk limits"
        assert adjustment.recommended_action == "Buy 100 shares of AAPL to hedge delta"
        assert adjustment.hedge_quantity == 100
        assert adjustment.hedge_contract == mock_contract
        assert adjustment.expected_cost == 15000.0
        assert adjustment.risk_reduction == 0.85
        assert adjustment.time_sensitivity == "IMMEDIATE"

    def test_position_adjustment_gamma_hedge(self):
        """Test position adjustment for gamma hedging"""
        adjustment = PositionAdjustment(
            adjustment_type=AdjustmentType.GAMMA_HEDGE,
            priority=2,
            reason="Gamma exposure creating excessive convexity risk",
            recommended_action="Sell 5 ATM straddles to reduce gamma",
            hedge_quantity=5,
            hedge_contract=None,  # Complex strategy
            expected_cost=2500.0,
            risk_reduction=0.60,
            time_sensitivity="WITHIN_HOUR",
        )

        assert adjustment.adjustment_type == AdjustmentType.GAMMA_HEDGE
        assert adjustment.priority == 2
        assert adjustment.hedge_contract is None
        assert adjustment.time_sensitivity == "WITHIN_HOUR"

    def test_position_adjustment_vega_hedge(self):
        """Test position adjustment for vega hedging"""
        adjustment = PositionAdjustment(
            adjustment_type=AdjustmentType.VEGA_HEDGE,
            priority=2,
            reason="High vega exposure to volatility changes",
            recommended_action="Buy 3 calendar spreads in different strikes",
            hedge_quantity=3,
            expected_cost=675.0,
            risk_reduction=0.75,
            time_sensitivity="END_OF_DAY",
        )

        assert adjustment.adjustment_type == AdjustmentType.VEGA_HEDGE
        assert adjustment.time_sensitivity == "END_OF_DAY"
        assert adjustment.risk_reduction == 0.75

    def test_position_adjustment_close_position(self):
        """Test position adjustment for closing position"""
        adjustment = PositionAdjustment(
            adjustment_type=AdjustmentType.CLOSE_POSITION,
            priority=1,
            reason="Position has reached maximum loss threshold",
            recommended_action="Close entire calendar spread position immediately",
            hedge_quantity=None,
            hedge_contract=None,
            expected_cost=None,  # Market order
            risk_reduction=1.0,  # Complete risk elimination
            time_sensitivity="IMMEDIATE",
        )

        assert adjustment.adjustment_type == AdjustmentType.CLOSE_POSITION
        assert adjustment.hedge_quantity is None
        assert adjustment.expected_cost is None
        assert adjustment.risk_reduction == 1.0
        assert adjustment.priority == 1

    def test_position_adjustment_roll_position(self):
        """Test position adjustment for rolling position"""
        adjustment = PositionAdjustment(
            adjustment_type=AdjustmentType.ROLL_POSITION,
            priority=3,
            reason="Front month approaching expiry with profitable position",
            recommended_action="Roll front month to next expiry cycle",
            hedge_quantity=2,
            expected_cost=-150.0,  # Credit received
            risk_reduction=0.2,
            time_sensitivity="END_OF_DAY",
        )

        assert adjustment.adjustment_type == AdjustmentType.ROLL_POSITION
        assert adjustment.expected_cost < 0  # Credit transaction
        assert adjustment.priority == 3  # Optional

    def test_position_adjustment_scale_down(self):
        """Test position adjustment for scaling down"""
        adjustment = PositionAdjustment(
            adjustment_type=AdjustmentType.SCALE_DOWN,
            priority=2,
            reason="Position size too large for current market volatility",
            recommended_action="Close 50% of calendar spread position",
            hedge_quantity=5,  # Number of spreads to close
            expected_cost=1250.0,
            risk_reduction=0.50,
            time_sensitivity="WITHIN_HOUR",
        )

        assert adjustment.adjustment_type == AdjustmentType.SCALE_DOWN
        assert adjustment.risk_reduction == 0.50
        assert adjustment.hedge_quantity == 5

    def test_position_adjustment_default_values(self):
        """Test position adjustment with default values"""
        adjustment = PositionAdjustment(
            adjustment_type=AdjustmentType.DELTA_HEDGE,
            priority=2,
            reason="Test adjustment",
            recommended_action="Test action",
        )

        # Test default values
        assert adjustment.hedge_quantity is None
        assert adjustment.hedge_contract is None
        assert adjustment.expected_cost is None
        assert adjustment.risk_reduction is None
        assert adjustment.time_sensitivity == "IMMEDIATE"

    def test_position_adjustment_priority_validation(self):
        """Test position adjustment priority levels"""
        priorities = [1, 2, 3]  # 1=urgent, 2=important, 3=optional

        for priority in priorities:
            adjustment = PositionAdjustment(
                adjustment_type=AdjustmentType.DELTA_HEDGE,
                priority=priority,
                reason=f"Priority {priority} test",
                recommended_action="Test action",
            )

            assert adjustment.priority == priority
            assert adjustment.priority in [1, 2, 3]

    def test_position_adjustment_time_sensitivity_values(self):
        """Test valid time sensitivity values"""
        valid_sensitivities = ["IMMEDIATE", "WITHIN_HOUR", "END_OF_DAY"]

        for sensitivity in valid_sensitivities:
            adjustment = PositionAdjustment(
                adjustment_type=AdjustmentType.VEGA_HEDGE,
                priority=2,
                reason="Time sensitivity test",
                recommended_action="Test action",
                time_sensitivity=sensitivity,
            )

            assert adjustment.time_sensitivity == sensitivity


class TestCalendarGreeks:
    """Comprehensive tests for CalendarGreeks dataclass"""

    def setup_method(self):
        """Setup test fixtures"""
        self.mock_opportunity = self._create_mock_opportunity()

    def _create_mock_opportunity(self):
        """Create mock calendar spread opportunity"""
        mock_opportunity = Mock()
        mock_opportunity.symbol = "AAPL"
        mock_opportunity.strike = 150.0
        mock_opportunity.option_type = "CALL"
        mock_opportunity.front_leg = Mock()
        mock_opportunity.back_leg = Mock()
        mock_opportunity.front_leg.days_to_expiry = 30
        mock_opportunity.back_leg.days_to_expiry = 65
        return mock_opportunity

    def test_calendar_greeks_creation_complete(self):
        """Test creating complete CalendarGreeks object"""
        greeks = CalendarGreeks(
            symbol="AAPL",
            position_size=5,
            underlying_price=150.0,
            # Net Greeks
            net_delta=0.25,
            net_gamma=0.15,
            net_vega=0.85,
            net_theta=-0.45,
            net_rho=0.30,
            # Front leg Greeks
            front_delta=0.65,
            front_gamma=0.08,
            front_vega=0.40,
            front_theta=-0.25,
            front_rho=0.20,
            # Back leg Greeks
            back_delta=0.40,
            back_gamma=0.05,
            back_vega=0.45,
            back_theta=-0.20,
            back_rho=0.25,
            # Risk metrics
            delta_risk_score=0.65,
            gamma_risk_score=0.45,
            vega_risk_score=0.75,
            theta_risk_score=0.30,
            overall_risk_level=GreeksRiskLevel.MEDIUM,
            # Time metrics
            days_to_front_expiry=25,
            days_to_back_expiry=60,
            time_decay_rate=0.85,
            last_updated=time.time(),
        )

        # Test all fields
        assert greeks.symbol == "AAPL"
        assert greeks.position_size == 5
        assert greeks.underlying_price == 150.0
        assert greeks.net_delta == 0.25
        assert greeks.net_gamma == 0.15
        assert greeks.net_vega == 0.85
        assert greeks.net_theta == -0.45
        assert greeks.net_rho == 0.30
        assert greeks.front_delta == 0.65
        assert greeks.back_delta == 0.40
        assert greeks.delta_risk_score == 0.65
        assert greeks.overall_risk_level == GreeksRiskLevel.MEDIUM
        assert greeks.days_to_front_expiry == 25
        assert greeks.time_decay_rate == 0.85

    def test_calendar_greeks_put_position(self):
        """Test CalendarGreeks for put calendar spread"""
        greeks = CalendarGreeks(
            symbol="AAPL",
            position_size=3,
            underlying_price=150.0,
            # Put calendar spread Greeks (negative deltas)
            net_delta=-0.15,
            net_gamma=0.12,
            net_vega=0.70,
            net_theta=-0.35,
            net_rho=-0.25,
            front_delta=-0.45,  # Put delta negative
            front_gamma=0.07,
            front_vega=0.35,
            front_theta=-0.20,
            front_rho=-0.15,
            back_delta=-0.30,  # Put delta negative
            back_gamma=0.05,
            back_vega=0.35,
            back_theta=-0.15,
            back_rho=-0.10,
            delta_risk_score=0.40,
            gamma_risk_score=0.35,
            vega_risk_score=0.65,
            theta_risk_score=0.25,
            overall_risk_level=GreeksRiskLevel.LOW,
            days_to_front_expiry=28,
            days_to_back_expiry=63,
            time_decay_rate=0.90,
            last_updated=time.time(),
        )

        assert greeks.net_delta < 0  # Put spread has negative delta
        assert greeks.front_delta < 0  # Put option delta negative
        assert greeks.back_delta < 0  # Put option delta negative
        assert greeks.overall_risk_level == GreeksRiskLevel.LOW

    def test_calendar_greeks_high_risk_position(self):
        """Test CalendarGreeks with high risk levels"""
        greeks = CalendarGreeks(
            symbol="VOLATILE",
            position_size=10,  # Large position
            underlying_price=100.0,
            # High risk Greeks
            net_delta=1.50,  # High delta exposure
            net_gamma=0.85,  # High gamma exposure
            net_vega=2.50,  # Very high vega exposure
            net_theta=-1.25,  # High theta decay
            net_rho=0.75,
            front_delta=0.85,
            front_gamma=0.45,
            front_vega=1.25,
            front_theta=-0.75,
            front_rho=0.40,
            back_delta=0.65,
            back_gamma=0.40,
            back_vega=1.25,
            back_theta=-0.50,
            back_rho=0.35,
            # High risk scores
            delta_risk_score=0.95,
            gamma_risk_score=0.88,
            vega_risk_score=0.92,
            theta_risk_score=0.75,
            overall_risk_level=GreeksRiskLevel.CRITICAL,
            days_to_front_expiry=5,  # Close to expiry
            days_to_back_expiry=40,
            time_decay_rate=2.5,  # High time decay acceleration
            last_updated=time.time(),
        )

        assert greeks.overall_risk_level == GreeksRiskLevel.CRITICAL
        assert greeks.delta_risk_score > 0.90
        assert greeks.vega_risk_score > 0.90
        assert greeks.days_to_front_expiry < 10  # Close to expiry

    def test_calendar_greeks_neutral_position(self):
        """Test CalendarGreeks with well-hedged neutral position"""
        greeks = CalendarGreeks(
            symbol="STABLE",
            position_size=2,
            underlying_price=150.0,
            # Well-hedged Greeks (low exposures)
            net_delta=0.02,  # Nearly delta neutral
            net_gamma=0.05,  # Low gamma
            net_vega=0.15,  # Controlled vega
            net_theta=-0.12,  # Moderate theta
            net_rho=0.08,
            front_delta=0.52,
            front_gamma=0.08,
            front_vega=0.25,
            front_theta=-0.15,
            front_rho=0.12,
            back_delta=0.50,
            back_gamma=0.03,
            back_vega=0.10,
            back_theta=-0.03,
            back_rho=0.04,
            # Low risk scores
            delta_risk_score=0.15,
            gamma_risk_score=0.20,
            vega_risk_score=0.25,
            theta_risk_score=0.30,
            overall_risk_level=GreeksRiskLevel.LOW,
            days_to_front_expiry=35,
            days_to_back_expiry=70,
            time_decay_rate=0.65,
            last_updated=time.time(),
        )

        assert abs(greeks.net_delta) < 0.05  # Nearly delta neutral
        assert greeks.overall_risk_level == GreeksRiskLevel.LOW
        assert all(
            score < 0.35
            for score in [
                greeks.delta_risk_score,
                greeks.gamma_risk_score,
                greeks.vega_risk_score,
                greeks.theta_risk_score,
            ]
        )

    def test_calendar_greeks_edge_cases(self):
        """Test CalendarGreeks with edge case values"""
        # Zero Greeks position
        zero_greeks = CalendarGreeks(
            symbol="ZERO",
            position_size=0,
            underlying_price=100.0,
            net_delta=0.0,
            net_gamma=0.0,
            net_vega=0.0,
            net_theta=0.0,
            net_rho=0.0,
            front_delta=0.0,
            front_gamma=0.0,
            front_vega=0.0,
            front_theta=0.0,
            front_rho=0.0,
            back_delta=0.0,
            back_gamma=0.0,
            back_vega=0.0,
            back_theta=0.0,
            back_rho=0.0,
            delta_risk_score=0.0,
            gamma_risk_score=0.0,
            vega_risk_score=0.0,
            theta_risk_score=0.0,
            overall_risk_level=GreeksRiskLevel.LOW,
            days_to_front_expiry=0,
            days_to_back_expiry=0,
            time_decay_rate=0.0,
            last_updated=time.time(),
        )

        assert zero_greeks.position_size == 0
        assert zero_greeks.net_delta == 0.0
        assert zero_greeks.overall_risk_level == GreeksRiskLevel.LOW

        # Extreme Greeks position
        extreme_greeks = CalendarGreeks(
            symbol="EXTREME",
            position_size=1000,
            underlying_price=1.0,  # Very low price
            net_delta=100.0,  # Extreme delta
            net_gamma=50.0,  # Extreme gamma
            net_vega=200.0,  # Extreme vega
            net_theta=-100.0,  # Extreme theta
            net_rho=75.0,
            front_delta=75.0,
            front_gamma=30.0,
            front_vega=150.0,
            front_theta=-75.0,
            front_rho=50.0,
            back_delta=25.0,
            back_gamma=20.0,
            back_vega=50.0,
            back_theta=-25.0,
            back_rho=25.0,
            delta_risk_score=1.0,  # Maximum risk
            gamma_risk_score=1.0,
            vega_risk_score=1.0,
            theta_risk_score=1.0,
            overall_risk_level=GreeksRiskLevel.CRITICAL,
            days_to_front_expiry=1,
            days_to_back_expiry=2,
            time_decay_rate=10.0,
            last_updated=time.time(),
        )

        assert extreme_greeks.net_delta == 100.0
        assert extreme_greeks.overall_risk_level == GreeksRiskLevel.CRITICAL
        assert all(
            score == 1.0
            for score in [
                extreme_greeks.delta_risk_score,
                extreme_greeks.gamma_risk_score,
                extreme_greeks.vega_risk_score,
                extreme_greeks.theta_risk_score,
            ]
        )

    def test_calendar_greeks_time_freshness(self):
        """Test time freshness tracking in CalendarGreeks"""
        current_time = time.time()

        greeks = CalendarGreeks(
            symbol="TIME_TEST",
            position_size=1,
            underlying_price=150.0,
            net_delta=0.5,
            net_gamma=0.1,
            net_vega=0.2,
            net_theta=-0.1,
            net_rho=0.05,
            front_delta=0.6,
            front_gamma=0.08,
            front_vega=0.15,
            front_theta=-0.08,
            front_rho=0.04,
            back_delta=0.1,
            back_gamma=0.02,
            back_vega=0.05,
            back_theta=-0.02,
            back_rho=0.01,
            delta_risk_score=0.3,
            gamma_risk_score=0.2,
            vega_risk_score=0.4,
            theta_risk_score=0.1,
            overall_risk_level=GreeksRiskLevel.LOW,
            days_to_front_expiry=30,
            days_to_back_expiry=65,
            time_decay_rate=0.8,
            last_updated=current_time,
        )

        # Should be very close to current time
        assert abs(greeks.last_updated - current_time) < 1.0

        # Test age calculation
        time.sleep(0.01)  # Small delay
        age = time.time() - greeks.last_updated
        assert age > 0


class TestPortfolioGreeks:
    """Comprehensive tests for PortfolioGreeks dataclass"""

    def test_portfolio_greeks_creation_basic(self):
        """Test basic portfolio Greeks creation"""
        individual_positions = []

        # Create multiple positions
        for i in range(3):
            position = CalendarGreeks(
                symbol=f"STOCK{i}",
                position_size=2,
                underlying_price=150.0 + i * 10,
                net_delta=0.25 + i * 0.1,
                net_gamma=0.15,
                net_vega=0.80,
                net_theta=-0.40,
                net_rho=0.25,
                front_delta=0.60,
                front_gamma=0.08,
                front_vega=0.40,
                front_theta=-0.25,
                front_rho=0.15,
                back_delta=0.35,
                back_gamma=0.07,
                back_vega=0.40,
                back_theta=-0.15,
                back_rho=0.10,
                delta_risk_score=0.30,
                gamma_risk_score=0.25,
                vega_risk_score=0.35,
                theta_risk_score=0.20,
                overall_risk_level=GreeksRiskLevel.LOW,
                days_to_front_expiry=30,
                days_to_back_expiry=65,
                time_decay_rate=0.75,
                last_updated=time.time(),
            )
            individual_positions.append(position)

        portfolio = PortfolioGreeks(
            total_positions=3,
            individual_positions=individual_positions,
            # Aggregate Greeks
            total_net_delta=0.75,  # Sum of individual deltas
            total_net_gamma=0.45,  # Sum of individual gammas
            total_net_vega=2.40,  # Sum of individual vegas
            total_net_theta=-1.20,  # Sum of individual thetas
            total_net_rho=0.75,  # Sum of individual rhos
            # Portfolio-level risk metrics
            portfolio_delta_limit=2.0,
            portfolio_gamma_limit=1.0,
            portfolio_vega_limit=5.0,
            delta_utilization=0.375,  # 0.75 / 2.0
            gamma_utilization=0.45,  # 0.45 / 1.0
            vega_utilization=0.48,  # 2.40 / 5.0
            # Correlation adjustments
            correlation_adjusted_delta=0.68,
            correlation_adjusted_gamma=0.41,
            correlation_adjusted_vega=2.15,
            overall_portfolio_risk=GreeksRiskLevel.LOW,
            requires_rebalancing=False,
            last_updated=time.time(),
        )

        assert portfolio.total_positions == 3
        assert len(portfolio.individual_positions) == 3
        assert portfolio.total_net_delta == 0.75
        assert portfolio.delta_utilization == 0.375
        assert portfolio.overall_portfolio_risk == GreeksRiskLevel.LOW
        assert portfolio.requires_rebalancing == False

    def test_portfolio_greeks_high_utilization(self):
        """Test portfolio Greeks with high risk utilization"""
        # Create high-risk positions
        high_risk_positions = []
        for i in range(5):
            position = CalendarGreeks(
                symbol=f"HIGHRISK{i}",
                position_size=10,  # Large positions
                underlying_price=100.0,
                net_delta=0.8,  # High individual deltas
                net_gamma=0.4,
                net_vega=1.5,
                net_theta=-0.6,
                net_rho=0.3,
                front_delta=0.9,
                front_gamma=0.25,
                front_vega=0.8,
                front_theta=-0.4,
                front_rho=0.2,
                back_delta=0.1,
                back_gamma=0.15,
                back_vega=0.7,
                back_theta=-0.2,
                back_rho=0.1,
                delta_risk_score=0.85,
                gamma_risk_score=0.75,
                vega_risk_score=0.90,
                theta_risk_score=0.65,
                overall_risk_level=GreeksRiskLevel.HIGH,
                days_to_front_expiry=10,  # Close to expiry
                days_to_back_expiry=45,
                time_decay_rate=1.8,
                last_updated=time.time(),
            )
            high_risk_positions.append(position)

        portfolio = PortfolioGreeks(
            total_positions=5,
            individual_positions=high_risk_positions,
            # High aggregate exposure
            total_net_delta=4.0,  # 5 * 0.8
            total_net_gamma=2.0,  # 5 * 0.4
            total_net_vega=7.5,  # 5 * 1.5
            total_net_theta=-3.0,  # 5 * -0.6
            total_net_rho=1.5,  # 5 * 0.3
            # Risk limits
            portfolio_delta_limit=3.0,
            portfolio_gamma_limit=1.5,
            portfolio_vega_limit=6.0,
            # High utilization
            delta_utilization=1.33,  # Over limit! 4.0 / 3.0
            gamma_utilization=1.33,  # Over limit! 2.0 / 1.5
            vega_utilization=1.25,  # Over limit! 7.5 / 6.0
            # Correlation effects (might reduce aggregate risk)
            correlation_adjusted_delta=3.2,  # Some correlation benefit
            correlation_adjusted_gamma=1.7,
            correlation_adjusted_vega=6.8,
            overall_portfolio_risk=GreeksRiskLevel.CRITICAL,
            requires_rebalancing=True,
            last_updated=time.time(),
        )

        assert portfolio.delta_utilization > 1.0  # Over limit
        assert portfolio.gamma_utilization > 1.0  # Over limit
        assert portfolio.vega_utilization > 1.0  # Over limit
        assert portfolio.overall_portfolio_risk == GreeksRiskLevel.CRITICAL
        assert portfolio.requires_rebalancing == True

    def test_portfolio_greeks_empty_portfolio(self):
        """Test portfolio Greeks with empty portfolio"""
        empty_portfolio = PortfolioGreeks(
            total_positions=0,
            individual_positions=[],
            total_net_delta=0.0,
            total_net_gamma=0.0,
            total_net_vega=0.0,
            total_net_theta=0.0,
            total_net_rho=0.0,
            portfolio_delta_limit=2.0,
            portfolio_gamma_limit=1.0,
            portfolio_vega_limit=5.0,
            delta_utilization=0.0,
            gamma_utilization=0.0,
            vega_utilization=0.0,
            correlation_adjusted_delta=0.0,
            correlation_adjusted_gamma=0.0,
            correlation_adjusted_vega=0.0,
            overall_portfolio_risk=GreeksRiskLevel.LOW,
            requires_rebalancing=False,
            last_updated=time.time(),
        )

        assert empty_portfolio.total_positions == 0
        assert len(empty_portfolio.individual_positions) == 0
        assert empty_portfolio.total_net_delta == 0.0
        assert empty_portfolio.overall_portfolio_risk == GreeksRiskLevel.LOW

    def test_portfolio_greeks_mixed_risk_levels(self):
        """Test portfolio Greeks with mixed risk level positions"""
        mixed_positions = []

        # Low risk position
        low_risk = CalendarGreeks(
            symbol="LOW_RISK",
            position_size=2,
            underlying_price=150.0,
            net_delta=0.1,
            net_gamma=0.05,
            net_vega=0.3,
            net_theta=-0.2,
            net_rho=0.1,
            front_delta=0.4,
            front_gamma=0.03,
            front_vega=0.2,
            front_theta=-0.15,
            front_rho=0.08,
            back_delta=0.3,
            back_gamma=0.02,
            back_vega=0.1,
            back_theta=-0.05,
            back_rho=0.02,
            delta_risk_score=0.2,
            gamma_risk_score=0.15,
            vega_risk_score=0.25,
            theta_risk_score=0.1,
            overall_risk_level=GreeksRiskLevel.LOW,
            days_to_front_expiry=40,
            days_to_back_expiry=75,
            time_decay_rate=0.6,
            last_updated=time.time(),
        )

        # High risk position
        high_risk = CalendarGreeks(
            symbol="HIGH_RISK",
            position_size=5,
            underlying_price=150.0,
            net_delta=1.2,
            net_gamma=0.6,
            net_vega=2.0,
            net_theta=-0.8,
            net_rho=0.4,
            front_delta=0.8,
            front_gamma=0.35,
            front_vega=1.2,
            front_theta=-0.6,
            front_rho=0.3,
            back_delta=0.4,
            back_gamma=0.25,
            back_vega=0.8,
            back_theta=-0.2,
            back_rho=0.1,
            delta_risk_score=0.9,
            gamma_risk_score=0.8,
            vega_risk_score=0.95,
            theta_risk_score=0.7,
            overall_risk_level=GreeksRiskLevel.HIGH,
            days_to_front_expiry=8,
            days_to_back_expiry=43,
            time_decay_rate=2.2,
            last_updated=time.time(),
        )

        mixed_positions = [low_risk, high_risk]

        portfolio = PortfolioGreeks(
            total_positions=2,
            individual_positions=mixed_positions,
            total_net_delta=1.3,  # 0.1 + 1.2
            total_net_gamma=0.65,  # 0.05 + 0.6
            total_net_vega=2.3,  # 0.3 + 2.0
            total_net_theta=-1.0,  # -0.2 + -0.8
            total_net_rho=0.5,  # 0.1 + 0.4
            portfolio_delta_limit=2.0,
            portfolio_gamma_limit=1.0,
            portfolio_vega_limit=3.0,
            delta_utilization=0.65,  # 1.3 / 2.0
            gamma_utilization=0.65,  # 0.65 / 1.0
            vega_utilization=0.77,  # 2.3 / 3.0
            correlation_adjusted_delta=1.15,  # Some correlation benefit
            correlation_adjusted_gamma=0.58,
            correlation_adjusted_vega=2.1,
            overall_portfolio_risk=GreeksRiskLevel.MEDIUM,  # Average of mixed risks
            requires_rebalancing=False,  # Within limits
            last_updated=time.time(),
        )

        assert portfolio.total_positions == 2
        assert portfolio.overall_portfolio_risk == GreeksRiskLevel.MEDIUM
        assert portfolio.delta_utilization < 1.0  # Within limits
        assert portfolio.requires_rebalancing == False

    def test_portfolio_greeks_correlation_effects(self):
        """Test correlation adjustments in portfolio Greeks"""
        # Highly correlated positions (should have less diversification benefit)
        correlated_positions = []
        for i in range(3):
            position = CalendarGreeks(
                symbol=f"CORR_STOCK{i}",  # Highly correlated stocks
                position_size=3,
                underlying_price=150.0,
                net_delta=0.5,
                net_gamma=0.2,
                net_vega=0.8,
                net_theta=-0.3,
                net_rho=0.15,
                front_delta=0.6,
                front_gamma=0.12,
                front_vega=0.4,
                front_theta=-0.2,
                front_rho=0.1,
                back_delta=0.1,
                back_gamma=0.08,
                back_vega=0.4,
                back_theta=-0.1,
                back_rho=0.05,
                delta_risk_score=0.4,
                gamma_risk_score=0.3,
                vega_risk_score=0.5,
                theta_risk_score=0.2,
                overall_risk_level=GreeksRiskLevel.MEDIUM,
                days_to_front_expiry=25,
                days_to_back_expiry=60,
                time_decay_rate=0.9,
                last_updated=time.time(),
            )
            correlated_positions.append(position)

        portfolio = PortfolioGreeks(
            total_positions=3,
            individual_positions=correlated_positions,
            total_net_delta=1.5,  # 3 * 0.5
            total_net_gamma=0.6,  # 3 * 0.2
            total_net_vega=2.4,  # 3 * 0.8
            total_net_theta=-0.9,  # 3 * -0.3
            total_net_rho=0.45,  # 3 * 0.15
            portfolio_delta_limit=2.0,
            portfolio_gamma_limit=1.0,
            portfolio_vega_limit=3.0,
            delta_utilization=0.75,
            gamma_utilization=0.6,
            vega_utilization=0.8,
            # High correlation = less diversification benefit
            correlation_adjusted_delta=1.4,  # Only slight reduction
            correlation_adjusted_gamma=0.57,  # Only slight reduction
            correlation_adjusted_vega=2.25,  # Only slight reduction
            overall_portfolio_risk=GreeksRiskLevel.MEDIUM,
            requires_rebalancing=False,
            last_updated=time.time(),
        )

        # Correlation adjustment should be small for highly correlated positions
        assert portfolio.correlation_adjusted_delta < portfolio.total_net_delta
        assert portfolio.correlation_adjusted_gamma < portfolio.total_net_gamma
        assert portfolio.correlation_adjusted_vega < portfolio.total_net_vega

        # But not dramatically different due to high correlation
        assert portfolio.correlation_adjusted_delta > 0.9 * portfolio.total_net_delta
        assert portfolio.correlation_adjusted_gamma > 0.9 * portfolio.total_net_gamma
        assert portfolio.correlation_adjusted_vega > 0.9 * portfolio.total_net_vega


class TestCalendarGreeksCalculator:
    """Comprehensive tests for CalendarGreeksCalculator class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.calculator = CalendarGreeksCalculator()
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

    def test_calculator_initialization(self):
        """Test CalendarGreeksCalculator initialization"""
        calculator = CalendarGreeksCalculator()

        assert hasattr(calculator, "greeks_cache")
        assert hasattr(calculator, "risk_thresholds")
        assert hasattr(calculator, "last_calculation_time")
        assert len(calculator.greeks_cache) == 0

    def test_calculate_individual_greeks_basic(self):
        """Test basic individual Greeks calculation"""
        stock_price = 150.0
        position_size = 2

        # Mock Greeks calculation
        with patch.object(self.calculator, "_calculate_option_greeks") as mock_greeks:
            mock_greeks.side_effect = [
                # Front leg Greeks
                (0.65, 0.08, 0.40, -0.25, 0.20),
                # Back leg Greeks
                (0.40, 0.05, 0.45, -0.20, 0.25),
            ]

            greeks = self.calculator.calculate_calendar_greeks(
                self.test_opportunity, stock_price, position_size
            )

            assert isinstance(greeks, CalendarGreeks)
            assert greeks.symbol == "AAPL"
            assert greeks.position_size == position_size
            assert greeks.underlying_price == stock_price

            # Check that Greeks were calculated for both legs
            assert mock_greeks.call_count == 2

    def test_calculate_net_greeks_calendar_spread(self):
        """Test net Greeks calculation for calendar spread"""
        # Calendar spread: Buy back month, sell front month
        front_greeks = (0.65, 0.08, 0.40, -0.25, 0.20)  # delta, gamma, vega, theta, rho
        back_greeks = (0.40, 0.05, 0.45, -0.20, 0.25)
        position_size = 2

        net_greeks = self.calculator._calculate_net_greeks(
            front_greeks, back_greeks, position_size
        )

        # Net = (back - front) * position_size * 100
        expected_net_delta = (0.40 - 0.65) * position_size * 100  # -50
        expected_net_gamma = (0.05 - 0.08) * position_size * 100  # -6
        expected_net_vega = (0.45 - 0.40) * position_size * 100  # 10
        expected_net_theta = (-0.20 - (-0.25)) * position_size * 100  # 10
        expected_net_rho = (0.25 - 0.20) * position_size * 100  # 10

        net_delta, net_gamma, net_vega, net_theta, net_rho = net_greeks

        assert abs(net_delta - expected_net_delta) < 0.01
        assert abs(net_gamma - expected_net_gamma) < 0.01
        assert abs(net_vega - expected_net_vega) < 0.01
        assert abs(net_theta - expected_net_theta) < 0.01
        assert abs(net_rho - expected_net_rho) < 0.01

    def test_calculate_net_greeks_zero_position(self):
        """Test net Greeks calculation with zero position size"""
        front_greeks = (0.65, 0.08, 0.40, -0.25, 0.20)
        back_greeks = (0.40, 0.05, 0.45, -0.20, 0.25)
        position_size = 0

        net_greeks = self.calculator._calculate_net_greeks(
            front_greeks, back_greeks, position_size
        )

        # All should be zero with zero position size
        assert all(greek == 0.0 for greek in net_greeks)

    def test_calculate_risk_scores_low_risk(self):
        """Test risk score calculation for low-risk position"""
        # Low Greeks values
        net_delta = 5.0  # Low delta
        net_gamma = 2.0  # Low gamma
        net_vega = 10.0  # Low vega
        net_theta = -8.0  # Low theta

        risk_scores = self.calculator._calculate_risk_scores(
            net_delta, net_gamma, net_vega, net_theta
        )

        delta_score, gamma_score, vega_score, theta_score, overall_level = risk_scores

        assert 0.0 <= delta_score <= 1.0
        assert 0.0 <= gamma_score <= 1.0
        assert 0.0 <= vega_score <= 1.0
        assert 0.0 <= theta_score <= 1.0
        assert overall_level in [GreeksRiskLevel.LOW, GreeksRiskLevel.MEDIUM]

    def test_calculate_risk_scores_high_risk(self):
        """Test risk score calculation for high-risk position"""
        # High Greeks values
        net_delta = 150.0  # High delta
        net_gamma = 80.0  # High gamma
        net_vega = 250.0  # High vega
        net_theta = -120.0  # High theta

        risk_scores = self.calculator._calculate_risk_scores(
            net_delta, net_gamma, net_vega, net_theta
        )

        delta_score, gamma_score, vega_score, theta_score, overall_level = risk_scores

        # High values should result in high risk scores
        assert delta_score > 0.7
        assert gamma_score > 0.7
        assert vega_score > 0.7
        assert overall_level in [GreeksRiskLevel.HIGH, GreeksRiskLevel.CRITICAL]

    def test_calculate_risk_scores_extreme_values(self):
        """Test risk score calculation with extreme values"""
        # Extreme Greeks values
        net_delta = 1000.0  # Extreme delta
        net_gamma = 500.0  # Extreme gamma
        net_vega = 1500.0  # Extreme vega
        net_theta = -800.0  # Extreme theta

        risk_scores = self.calculator._calculate_risk_scores(
            net_delta, net_gamma, net_vega, net_theta
        )

        delta_score, gamma_score, vega_score, theta_score, overall_level = risk_scores

        # Should cap at maximum risk level
        assert delta_score >= 0.9
        assert gamma_score >= 0.9
        assert vega_score >= 0.9
        assert overall_level == GreeksRiskLevel.CRITICAL

    def test_generate_adjustment_recommendations_delta_hedge(self):
        """Test adjustment recommendations for delta hedging"""
        # High delta exposure requiring hedge
        greeks = CalendarGreeks(
            symbol="AAPL",
            position_size=10,
            underlying_price=150.0,
            net_delta=200.0,  # High delta requiring hedge
            net_gamma=50.0,
            net_vega=150.0,
            net_theta=-80.0,
            net_rho=40.0,
            front_delta=0.8,
            front_gamma=0.3,
            front_vega=0.8,
            front_theta=-0.5,
            front_rho=0.3,
            back_delta=0.3,
            back_gamma=0.2,
            back_vega=0.7,
            back_theta=-0.3,
            back_rho=0.1,
            delta_risk_score=0.95,
            gamma_risk_score=0.6,
            vega_risk_score=0.7,
            theta_risk_score=0.4,
            overall_risk_level=GreeksRiskLevel.HIGH,
            days_to_front_expiry=20,
            days_to_back_expiry=55,
            time_decay_rate=1.2,
            last_updated=time.time(),
        )

        adjustments = self.calculator.generate_adjustment_recommendations(greeks)

        assert len(adjustments) > 0

        # Should recommend delta hedging
        delta_adjustments = [
            adj
            for adj in adjustments
            if adj.adjustment_type == AdjustmentType.DELTA_HEDGE
        ]
        assert len(delta_adjustments) > 0

        delta_adj = delta_adjustments[0]
        assert delta_adj.priority <= 2  # High priority
        assert "delta" in delta_adj.reason.lower()
        assert delta_adj.time_sensitivity in ["IMMEDIATE", "WITHIN_HOUR"]

    def test_generate_adjustment_recommendations_close_position(self):
        """Test adjustment recommendations for closing position"""
        # Critical risk level requiring position closure
        greeks = CalendarGreeks(
            symbol="DANGER",
            position_size=20,
            underlying_price=100.0,
            net_delta=500.0,
            net_gamma=200.0,
            net_vega=800.0,
            net_theta=-300.0,
            net_rho=150.0,
            front_delta=0.95,
            front_gamma=0.8,
            front_vega=2.0,
            front_theta=-1.5,
            front_rho=0.8,
            back_delta=0.2,
            back_gamma=0.4,
            back_vega=1.8,
            back_theta=-0.8,
            back_rho=0.3,
            delta_risk_score=1.0,
            gamma_risk_score=1.0,
            vega_risk_score=1.0,
            theta_risk_score=1.0,
            overall_risk_level=GreeksRiskLevel.CRITICAL,
            days_to_front_expiry=2,
            days_to_back_expiry=37,
            time_decay_rate=5.0,
            last_updated=time.time(),
        )

        adjustments = self.calculator.generate_adjustment_recommendations(greeks)

        # Should recommend closing position
        close_adjustments = [
            adj
            for adj in adjustments
            if adj.adjustment_type == AdjustmentType.CLOSE_POSITION
        ]
        assert len(close_adjustments) > 0

        close_adj = close_adjustments[0]
        assert close_adj.priority == 1  # Urgent
        assert close_adj.time_sensitivity == "IMMEDIATE"
        assert close_adj.risk_reduction >= 0.9  # Should eliminate most risk

    def test_generate_adjustment_recommendations_roll_position(self):
        """Test adjustment recommendations for rolling position"""
        # Position close to expiry with profit potential
        greeks = CalendarGreeks(
            symbol="ROLL_ME",
            position_size=5,
            underlying_price=150.0,
            net_delta=25.0,
            net_gamma=15.0,
            net_vega=40.0,
            net_theta=-60.0,
            net_rho=20.0,
            front_delta=0.7,
            front_gamma=0.4,
            front_vega=0.3,
            front_theta=-0.8,
            front_rho=0.2,
            back_delta=0.5,
            back_gamma=0.25,
            back_vega=0.5,
            back_theta=-0.2,
            back_rho=0.15,
            delta_risk_score=0.3,
            gamma_risk_score=0.4,
            vega_risk_score=0.3,
            theta_risk_score=0.8,
            overall_risk_level=GreeksRiskLevel.MEDIUM,
            days_to_front_expiry=3,
            days_to_back_expiry=38,
            time_decay_rate=3.5,
            last_updated=time.time(),
        )

        adjustments = self.calculator.generate_adjustment_recommendations(greeks)

        # Should consider rolling position
        roll_adjustments = [
            adj
            for adj in adjustments
            if adj.adjustment_type == AdjustmentType.ROLL_POSITION
        ]

        if roll_adjustments:  # May or may not recommend rolling based on conditions
            roll_adj = roll_adjustments[0]
            assert roll_adj.priority >= 2  # Not urgent
            assert (
                "roll" in roll_adj.reason.lower() or "expiry" in roll_adj.reason.lower()
            )

    def test_generate_adjustment_recommendations_no_action(self):
        """Test adjustment recommendations for well-managed position"""
        # Low-risk position requiring no action
        greeks = CalendarGreeks(
            symbol="STABLE",
            position_size=2,
            underlying_price=150.0,
            net_delta=8.0,
            net_gamma=3.0,
            net_vega=15.0,
            net_theta=-12.0,
            net_rho=5.0,
            front_delta=0.55,
            front_gamma=0.08,
            front_vega=0.25,
            front_theta=-0.18,
            front_rho=0.12,
            back_delta=0.51,
            back_gamma=0.05,
            back_vega=0.1,
            back_theta=-0.06,
            back_rho=0.07,
            delta_risk_score=0.15,
            gamma_risk_score=0.12,
            vega_risk_score=0.18,
            theta_risk_score=0.20,
            overall_risk_level=GreeksRiskLevel.LOW,
            days_to_front_expiry=35,
            days_to_back_expiry=70,
            time_decay_rate=0.6,
            last_updated=time.time(),
        )

        adjustments = self.calculator.generate_adjustment_recommendations(greeks)

        # Should have few or no urgent adjustments
        urgent_adjustments = [adj for adj in adjustments if adj.priority == 1]
        assert len(urgent_adjustments) == 0  # No urgent actions needed

    def test_calculate_portfolio_greeks_multiple_positions(self):
        """Test portfolio Greeks calculation with multiple positions"""
        positions = []

        # Create 3 different positions
        for i in range(3):
            greeks = CalendarGreeks(
                symbol=f"STOCK{i}",
                position_size=2 + i,
                underlying_price=150.0 + i * 5,
                net_delta=20.0 + i * 10,
                net_gamma=8.0 + i * 2,
                net_vega=30.0 + i * 5,
                net_theta=-15.0 - i * 5,
                net_rho=10.0 + i * 2,
                front_delta=0.6,
                front_gamma=0.1,
                front_vega=0.3,
                front_theta=-0.2,
                front_rho=0.15,
                back_delta=0.4,
                back_gamma=0.06,
                back_vega=0.25,
                back_theta=-0.1,
                back_rho=0.08,
                delta_risk_score=0.3 + i * 0.1,
                gamma_risk_score=0.2 + i * 0.1,
                vega_risk_score=0.4 + i * 0.1,
                theta_risk_score=0.25 + i * 0.1,
                overall_risk_level=(
                    GreeksRiskLevel.LOW if i == 0 else GreeksRiskLevel.MEDIUM
                ),
                days_to_front_expiry=30 - i * 5,
                days_to_back_expiry=65 - i * 5,
                time_decay_rate=0.8 + i * 0.2,
                last_updated=time.time(),
            )
            positions.append(greeks)

        portfolio = self.calculator.calculate_portfolio_greeks(positions)

        assert isinstance(portfolio, PortfolioGreeks)
        assert portfolio.total_positions == 3
        assert len(portfolio.individual_positions) == 3

        # Check aggregated Greeks
        expected_total_delta = sum(pos.net_delta for pos in positions)
        expected_total_gamma = sum(pos.net_gamma for pos in positions)
        expected_total_vega = sum(pos.net_vega for pos in positions)

        assert abs(portfolio.total_net_delta - expected_total_delta) < 0.01
        assert abs(portfolio.total_net_gamma - expected_total_gamma) < 0.01
        assert abs(portfolio.total_net_vega - expected_total_vega) < 0.01

    def test_calculate_portfolio_greeks_empty_portfolio(self):
        """Test portfolio Greeks calculation with empty portfolio"""
        portfolio = self.calculator.calculate_portfolio_greeks([])

        assert isinstance(portfolio, PortfolioGreeks)
        assert portfolio.total_positions == 0
        assert len(portfolio.individual_positions) == 0
        assert portfolio.total_net_delta == 0.0
        assert portfolio.total_net_gamma == 0.0
        assert portfolio.total_net_vega == 0.0
        assert portfolio.overall_portfolio_risk == GreeksRiskLevel.LOW


class TestCalendarGreeksIntegration:
    """Integration tests for CalendarGreeks with other components"""

    def setup_method(self):
        """Setup integration test fixtures"""
        self.calculator = CalendarGreeksCalculator()

    def test_greeks_evolution_modeling(self):
        """Test Greeks evolution over time"""
        time_horizon = 14  # 2 weeks
        price_scenarios = [140.0, 145.0, 150.0, 155.0, 160.0]

        evolution = self.calculator.model_greeks_evolution(
            self._create_test_opportunity(),
            current_stock_price=150.0,
            position_size=3,
            time_horizon_days=time_horizon,
            price_scenarios=price_scenarios,
        )

        assert isinstance(evolution, GreeksEvolution)
        assert evolution.time_horizon_days == time_horizon
        assert len(evolution.underlying_price_scenarios) == len(price_scenarios)
        assert len(evolution.delta_evolution) == len(price_scenarios)

        # Each price scenario should have time_horizon + 1 data points (day 0 to day N)
        for price in price_scenarios:
            assert len(evolution.delta_evolution[price]) == time_horizon + 1
            assert len(evolution.gamma_evolution[price]) == time_horizon + 1
            assert len(evolution.vega_evolution[price]) == time_horizon + 1

    def _create_test_opportunity(self):
        """Helper to create test opportunity"""
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

    def test_risk_threshold_monitoring(self):
        """Test risk threshold monitoring and alerts"""
        # Create position that exceeds risk thresholds
        high_risk_greeks = CalendarGreeks(
            symbol="RISKY",
            position_size=15,
            underlying_price=100.0,
            net_delta=300.0,  # Exceeds threshold
            net_gamma=150.0,  # Exceeds threshold
            net_vega=500.0,  # Exceeds threshold
            net_theta=-200.0,
            net_rho=100.0,
            front_delta=0.9,
            front_gamma=0.5,
            front_vega=1.5,
            front_theta=-1.0,
            front_rho=0.6,
            back_delta=0.1,
            back_gamma=0.4,
            back_vega=1.0,
            back_theta=-0.3,
            back_rho=0.1,
            delta_risk_score=1.0,
            gamma_risk_score=1.0,
            vega_risk_score=1.0,
            theta_risk_score=0.8,
            overall_risk_level=GreeksRiskLevel.CRITICAL,
            days_to_front_expiry=5,
            days_to_back_expiry=40,
            time_decay_rate=4.0,
            last_updated=time.time(),
        )

        alerts = self.calculator.check_risk_thresholds(high_risk_greeks)

        assert len(alerts) > 0
        assert any("delta" in alert.lower() for alert in alerts)
        assert any("gamma" in alert.lower() for alert in alerts)
        assert any("vega" in alert.lower() for alert in alerts)

    def test_real_time_greeks_update(self):
        """Test real-time Greeks updates with market data changes"""
        opportunity = self._create_test_opportunity()

        # Initial calculation
        initial_greeks = self.calculator.calculate_calendar_greeks(
            opportunity, stock_price=150.0, position_size=5
        )

        # Simulate stock price movement
        updated_greeks = self.calculator.calculate_calendar_greeks(
            opportunity, stock_price=155.0, position_size=5  # Stock moved up
        )

        # Greeks should change with stock price movement
        assert updated_greeks.net_delta != initial_greeks.net_delta
        assert updated_greeks.underlying_price != initial_greeks.underlying_price
        assert updated_greeks.last_updated > initial_greeks.last_updated

    def test_performance_optimization_caching(self):
        """Test performance optimization through caching"""
        opportunity = self._create_test_opportunity()

        # First calculation - should cache results
        start_time = time.time()
        greeks1 = self.calculator.calculate_calendar_greeks(opportunity, 150.0, 3)
        first_calc_time = time.time() - start_time

        # Second calculation with same parameters - should use cache
        start_time = time.time()
        greeks2 = self.calculator.calculate_calendar_greeks(opportunity, 150.0, 3)
        second_calc_time = time.time() - start_time

        # Results should be identical
        assert greeks1.net_delta == greeks2.net_delta
        assert greeks1.net_gamma == greeks2.net_gamma
        assert greeks1.net_vega == greeks2.net_vega

        # Second calculation should be faster (using cache)
        assert second_calc_time <= first_calc_time

    def test_numerical_stability_extreme_conditions(self):
        """Test numerical stability under extreme market conditions"""
        extreme_conditions = [
            {"stock_price": 0.01, "volatility": 5.0},  # Very low price, high vol
            {"stock_price": 10000.0, "volatility": 0.01},  # Very high price, low vol
            {"stock_price": 150.0, "volatility": 0.001},  # Ultra low vol
            {"stock_price": 150.0, "volatility": 10.0},  # Ultra high vol
        ]

        for condition in extreme_conditions:
            try:
                # Mock extreme conditions
                opportunity = self._create_test_opportunity()

                greeks = self.calculator.calculate_calendar_greeks(
                    opportunity, condition["stock_price"], 1
                )

                # Should handle gracefully and return finite values
                assert np.isfinite(greeks.net_delta)
                assert np.isfinite(greeks.net_gamma)
                assert np.isfinite(greeks.net_vega)
                assert np.isfinite(greeks.net_theta)
                assert np.isfinite(greeks.net_rho)

            except (ValueError, OverflowError, ZeroDivisionError):
                # Acceptable for extreme conditions
                pass

    def test_error_handling_invalid_inputs(self):
        """Test error handling with invalid inputs"""
        invalid_scenarios = [
            # None opportunity
            (None, 150.0, 1),
            # Invalid stock price
            (self._create_test_opportunity(), -100.0, 1),
            # Invalid position size
            (self._create_test_opportunity(), 150.0, -5),
        ]

        for opportunity, stock_price, position_size in invalid_scenarios:
            try:
                greeks = self.calculator.calculate_calendar_greeks(
                    opportunity, stock_price, position_size
                )

                # If no error, should return valid result
                if greeks is not None:
                    assert isinstance(greeks, CalendarGreeks)

            except (ValueError, TypeError, AttributeError):
                # Expected for invalid inputs
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
