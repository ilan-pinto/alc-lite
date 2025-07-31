"""
Unit tests for Calendar Spread Greeks Enhancement.

Tests the enhanced Greeks calculations (theta, delta, gamma, vega) that use real IB API data
instead of placeholder calculations. Covers all Greeks methods with comprehensive fallback testing.
"""

import time
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from modules.Arbitrage.CalendarSpread import CalendarSpread


class TestCalendarSpreadGreeksEnhancement:
    """Test enhanced Greeks calculations in CalendarSpread module"""

    def setup_method(self):
        """Set up test fixtures"""
        self.calendar = CalendarSpread()

    def test_theta_calculation_with_model_greeks(self):
        """Test theta calculation using ticker.modelGreeks.theta (priority 1)"""
        # Mock ticker with model Greeks theta
        ticker = Mock()
        ticker.modelGreeks = Mock()
        ticker.modelGreeks.theta = -0.025  # -0.025 theta per day
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 12345
        contract.symbol = "SPY"
        contract.strike = 450.0

        # Calculate theta
        theta = self.calendar._calculate_theta(ticker, contract, 30)

        # Should return -0.025
        assert theta == -0.025

    def test_theta_calculation_with_bid_ask_average(self):
        """Test theta calculation using bid/ask Greeks average (priority 2)"""
        # Mock ticker with bid/ask Greeks theta
        ticker = Mock()
        ticker.modelGreeks = None
        ticker.bidGreeks = Mock()
        ticker.bidGreeks.theta = -0.03  # -0.03 theta
        ticker.askGreeks = Mock()
        ticker.askGreeks.theta = -0.02  # -0.02 theta
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 12346
        contract.symbol = "QQQ"
        contract.strike = 350.0

        # Calculate theta
        theta = self.calendar._calculate_theta(ticker, contract, 20)

        # Should return average: (-0.03 + -0.02) / 2 = -0.025
        assert theta == -0.025

    def test_theta_calculation_fallback_time_estimation(self):
        """Test theta calculation fallback to time estimation"""
        # Mock ticker with no IB Greeks data
        ticker = Mock()
        ticker.modelGreeks = None
        ticker.bidGreeks = None
        ticker.askGreeks = None
        ticker.midpoint = lambda: 5.0
        ticker.close = 5.0
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 12347
        contract.symbol = "AAPL"
        contract.strike = 180.0

        # Calculate theta
        theta = self.calendar._calculate_theta(ticker, contract, 30)

        # Should use time estimation formula
        time_factor = max(1.0, np.sqrt(30 / 30.0))  # 1.0
        expected_theta = -(5.0 / time_factor) * 0.05  # -0.25
        assert theta == expected_theta

    def test_theta_validation_range_limits(self):
        """Test theta validation enforces -10.0 to +0.1 range"""
        # Mock ticker with extreme theta value
        ticker = Mock()
        ticker.modelGreeks = Mock()
        ticker.modelGreeks.theta = -50.0  # Extreme negative theta
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 12348
        contract.symbol = "NVDA"
        contract.strike = 500.0

        # Calculate theta
        theta = self.calendar._calculate_theta(ticker, contract, 10)

        # Should be capped at -10.0
        assert theta == -10.0

    def test_theta_cache_functionality(self):
        """Test that theta calculation properly uses cache"""
        # Mock ticker
        ticker = Mock()
        ticker.modelGreeks = Mock()
        ticker.modelGreeks.theta = -0.05
        ticker.time = 123456789.0

        # Mock contract
        contract = Mock()
        contract.conId = 12349
        contract.symbol = "META"
        contract.strike = 300.0

        # First calculation
        theta1 = self.calendar._calculate_theta(ticker, contract, 25)
        assert theta1 == -0.05

        # Second calculation should hit cache (same time)
        theta2 = self.calendar._calculate_theta(ticker, contract, 25)
        assert theta2 == -0.05

        # Cache should have the entry
        cache_key = f"{contract.conId}_theta_{ticker.time}"
        assert cache_key in self.calendar.greeks_cache


class TestCalendarSpreadDeltaEnhancement:
    """Test enhanced delta calculation in CalendarSpread module"""

    def setup_method(self):
        """Set up test fixtures"""
        self.calendar = CalendarSpread()

    def test_delta_calculation_with_model_greeks(self):
        """Test delta calculation using ticker.modelGreeks.delta"""
        # Mock ticker with model Greeks delta
        ticker = Mock()
        ticker.modelGreeks = Mock()
        ticker.modelGreeks.delta = 0.65  # Call delta
        ticker.time = time.time()

        # Mock call contract
        contract = Mock()
        contract.conId = 22345
        contract.symbol = "SPY"
        contract.strike = 450.0
        contract.right = "C"

        # Calculate delta
        delta = self.calendar._calculate_delta(ticker, contract)

        # Should return 0.65
        assert delta == 0.65

    def test_delta_calculation_put_with_bid_ask_average(self):
        """Test delta calculation for put using bid/ask average"""
        # Mock ticker with bid/ask Greeks delta
        ticker = Mock()
        ticker.modelGreeks = None
        ticker.bidGreeks = Mock()
        ticker.bidGreeks.delta = -0.4  # Put delta
        ticker.askGreeks = Mock()
        ticker.askGreeks.delta = -0.3  # Put delta
        ticker.time = time.time()

        # Mock put contract
        contract = Mock()
        contract.conId = 22346
        contract.symbol = "QQQ"
        contract.strike = 350.0
        contract.right = "P"

        # Calculate delta
        delta = self.calendar._calculate_delta(ticker, contract)

        # Should return average: (-0.4 + -0.3) / 2 = -0.35
        assert delta == -0.35

    def test_delta_call_validation_range(self):
        """Test delta validation for calls (0.0 to 1.0)"""
        # Mock ticker with out-of-range delta
        ticker = Mock()
        ticker.modelGreeks = Mock()
        ticker.modelGreeks.delta = 1.5  # Out of range for call
        ticker.time = time.time()

        # Mock call contract
        contract = Mock()
        contract.conId = 22347
        contract.symbol = "AAPL"
        contract.strike = 180.0
        contract.right = "C"

        # Calculate delta
        delta = self.calendar._calculate_delta(ticker, contract)

        # Should be capped at 1.0 for calls
        assert delta == 1.0

    def test_delta_put_validation_range(self):
        """Test delta validation for puts (-1.0 to 0.0)"""
        # Mock ticker with out-of-range delta
        ticker = Mock()
        ticker.modelGreeks = Mock()
        ticker.modelGreeks.delta = -1.5  # Out of range for put
        ticker.time = time.time()

        # Mock put contract
        contract = Mock()
        contract.conId = 22348
        contract.symbol = "MSFT"
        contract.strike = 300.0
        contract.right = "P"

        # Calculate delta
        delta = self.calendar._calculate_delta(ticker, contract)

        # Should be capped at -1.0 for puts
        assert delta == -1.0

    def test_delta_fallback_estimation(self):
        """Test delta fallback to ATM estimation"""
        # Mock ticker with no IB Greeks data
        ticker = Mock()
        ticker.modelGreeks = None
        ticker.bidGreeks = None
        ticker.askGreeks = None
        ticker.time = time.time()

        # Mock call contract
        call_contract = Mock()
        call_contract.conId = 22349
        call_contract.symbol = "TSLA"
        call_contract.strike = 250.0
        call_contract.right = "C"

        # Calculate delta for call
        call_delta = self.calendar._calculate_delta(ticker, call_contract)
        assert call_delta == 0.5  # ATM call estimate

        # Mock put contract
        put_contract = Mock()
        put_contract.conId = 22350
        put_contract.symbol = "TSLA"
        put_contract.strike = 250.0
        put_contract.right = "P"

        # Calculate delta for put
        put_delta = self.calendar._calculate_delta(ticker, put_contract)
        assert put_delta == -0.5  # ATM put estimate


class TestCalendarSpreadGammaEnhancement:
    """Test enhanced gamma calculation in CalendarSpread module"""

    def setup_method(self):
        """Set up test fixtures"""
        self.calendar = CalendarSpread()

    def test_gamma_calculation_with_model_greeks(self):
        """Test gamma calculation using ticker.modelGreeks.gamma"""
        # Mock ticker with model Greeks gamma
        ticker = Mock()
        ticker.modelGreeks = Mock()
        ticker.modelGreeks.gamma = 0.12  # Gamma value
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 32345
        contract.symbol = "SPY"
        contract.strike = 450.0

        # Calculate gamma
        gamma = self.calendar._calculate_gamma(ticker, contract)

        # Should return 0.12
        assert gamma == 0.12

    def test_gamma_calculation_with_last_greeks(self):
        """Test gamma calculation using ticker.lastGreeks.gamma"""
        # Mock ticker with last Greeks gamma only
        ticker = Mock()
        ticker.modelGreeks = None
        ticker.bidGreeks = None
        ticker.askGreeks = None
        ticker.lastGreeks = Mock()
        ticker.lastGreeks.gamma = 0.08
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 32346
        contract.symbol = "QQQ"
        contract.strike = 350.0

        # Calculate gamma
        gamma = self.calendar._calculate_gamma(ticker, contract)

        # Should return 0.08
        assert gamma == 0.08

    def test_gamma_validation_positive_range(self):
        """Test gamma validation enforces positive range (0.0 to 1.0)"""
        # Mock ticker with negative gamma (invalid)
        ticker = Mock()
        ticker.modelGreeks = Mock()
        ticker.modelGreeks.gamma = -0.05  # Invalid negative gamma
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 32347
        contract.symbol = "AAPL"
        contract.strike = 180.0

        # Calculate gamma
        gamma = self.calendar._calculate_gamma(ticker, contract)

        # Should be capped at 0.0 (gamma should always be positive)
        assert gamma == 0.0

    def test_gamma_fallback_fixed_estimation(self):
        """Test gamma fallback to fixed estimation"""
        # Mock ticker with no IB Greeks data
        ticker = Mock()
        ticker.modelGreeks = None
        ticker.bidGreeks = None
        ticker.askGreeks = None
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 32348
        contract.symbol = "NVDA"
        contract.strike = 500.0

        # Calculate gamma
        gamma = self.calendar._calculate_gamma(ticker, contract)

        # Should return fixed estimate 0.05
        assert gamma == 0.05


class TestCalendarSpreadVegaEnhancement:
    """Test enhanced vega calculation in CalendarSpread module"""

    def setup_method(self):
        """Set up test fixtures"""
        self.calendar = CalendarSpread()

    def test_vega_calculation_with_model_greeks(self):
        """Test vega calculation using ticker.modelGreeks.vega"""
        # Mock ticker with model Greeks vega
        ticker = Mock()
        ticker.modelGreeks = Mock()
        ticker.modelGreeks.vega = 0.25  # Vega value
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 42345
        contract.symbol = "SPY"
        contract.strike = 450.0

        # Calculate vega
        vega = self.calendar._calculate_vega(ticker, contract)

        # Should return 0.25
        assert vega == 0.25

    def test_vega_calculation_with_bid_ask_single_source(self):
        """Test vega calculation using single bid Greeks source"""
        # Mock ticker with only bid Greeks vega
        ticker = Mock()
        ticker.modelGreeks = None
        ticker.bidGreeks = Mock()
        ticker.bidGreeks.vega = 0.18
        ticker.askGreeks = Mock()
        ticker.askGreeks.vega = None  # No ask vega
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 42346
        contract.symbol = "QQQ"
        contract.strike = 350.0

        # Calculate vega
        vega = self.calendar._calculate_vega(ticker, contract)

        # Should return bid vega 0.18
        assert vega == 0.18

    def test_vega_fallback_price_estimation(self):
        """Test vega fallback to price-based estimation"""
        # Mock ticker with no IB Greeks data but valid price
        ticker = Mock()
        ticker.modelGreeks = None
        ticker.bidGreeks = None
        ticker.askGreeks = None
        ticker.midpoint = lambda: 3.5
        ticker.close = 3.5
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 42347
        contract.symbol = "AAPL"
        contract.strike = 180.0

        # Calculate vega
        vega = self.calendar._calculate_vega(ticker, contract)

        # Should use price estimation: 3.5 * 0.1 = 0.35
        assert abs(vega - 0.35) < 0.001  # Account for floating point precision

    def test_vega_validation_positive_range(self):
        """Test vega validation enforces positive range (0.0 to 100.0)"""
        # Mock ticker with extreme vega value
        ticker = Mock()
        ticker.modelGreeks = Mock()
        ticker.modelGreeks.vega = 150.0  # Out of range
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 42348
        contract.symbol = "MSFT"
        contract.strike = 300.0

        # Calculate vega
        vega = self.calendar._calculate_vega(ticker, contract)

        # Should be capped at 100.0
        assert vega == 100.0

    def test_vega_default_fallback(self):
        """Test vega with complete data unavailability"""
        # Mock ticker with no usable data
        ticker = Mock()
        ticker.modelGreeks = None
        ticker.bidGreeks = None
        ticker.askGreeks = None
        ticker.midpoint = lambda: np.nan
        ticker.close = 0
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 42349
        contract.symbol = "AMZN"
        contract.strike = 3200.0

        # Calculate vega
        vega = self.calendar._calculate_vega(ticker, contract)

        # Should return default 0.1
        assert vega == 0.1


@pytest.mark.unit
class TestGreeksEnhancementIntegration:
    """Test integration between all enhanced Greeks calculations"""

    def test_greeks_consistency_with_same_ticker_data(self):
        """Test that all Greeks use consistent IB API data sources"""
        # Set up CalendarSpread
        calendar = CalendarSpread()

        # Mock ticker with complete model Greeks
        ticker = Mock()
        ticker.modelGreeks = Mock()
        ticker.modelGreeks.theta = -0.03
        ticker.modelGreeks.delta = 0.65
        ticker.modelGreeks.gamma = 0.08
        ticker.modelGreeks.vega = 0.22
        ticker.time = time.time()

        # Mock call contract
        contract = Mock()
        contract.conId = 99999
        contract.symbol = "GOOGL"
        contract.strike = 2800.0
        contract.right = "C"

        # Calculate all Greeks
        theta = calendar._calculate_theta(ticker, contract, 30)
        delta = calendar._calculate_delta(ticker, contract)
        gamma = calendar._calculate_gamma(ticker, contract)
        vega = calendar._calculate_vega(ticker, contract)

        # All should use model Greeks data
        assert theta == -0.03
        assert delta == 0.65
        assert gamma == 0.08
        assert vega == 0.22

    def test_greeks_cache_independence(self):
        """Test that different Greeks use independent cache keys"""
        # Set up CalendarSpread
        calendar = CalendarSpread()

        # Mock ticker
        ticker = Mock()
        ticker.modelGreeks = Mock()
        ticker.modelGreeks.theta = -0.04
        ticker.modelGreeks.delta = 0.75
        ticker.time = 123456789.0

        # Mock contract
        contract = Mock()
        contract.conId = 88888
        contract.symbol = "NFLX"
        contract.strike = 500.0
        contract.right = "C"

        # Calculate theta and delta
        theta = calendar._calculate_theta(ticker, contract, 25)
        delta = calendar._calculate_delta(ticker, contract)

        # Check that cache has separate entries
        theta_cache_key = f"{contract.conId}_theta_{ticker.time}"
        delta_cache_key = f"{contract.conId}_delta_{ticker.time}"

        assert theta_cache_key in calendar.greeks_cache
        assert delta_cache_key in calendar.greeks_cache
        assert theta_cache_key != delta_cache_key

    def test_greeks_error_handling_consistency(self):
        """Test that all Greeks handle errors consistently"""
        # Set up CalendarSpread
        calendar = CalendarSpread()

        # Mock ticker that will cause errors
        ticker = Mock()
        ticker.modelGreeks = None
        # Don't mock bidGreeks/askGreeks to trigger AttributeError
        del ticker.bidGreeks
        del ticker.askGreeks
        ticker.midpoint = lambda: 2.0
        ticker.close = 2.0
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 77777
        contract.symbol = "AMZN"
        contract.strike = 3200.0
        contract.right = "C"

        # All Greeks should handle errors gracefully
        theta = calendar._calculate_theta(ticker, contract, 20)
        delta = calendar._calculate_delta(ticker, contract)
        gamma = calendar._calculate_gamma(ticker, contract)
        vega = calendar._calculate_vega(ticker, contract)

        # All should return fallback values without raising exceptions
        assert isinstance(theta, float)
        assert isinstance(delta, float)
        assert isinstance(gamma, float)
        assert isinstance(vega, float)

        # Validate specific fallback values
        assert theta < 0  # Theta should be negative
        assert 0 <= delta <= 1  # Call delta should be in valid range
        assert gamma >= 0  # Gamma should be positive
        assert vega >= 0  # Vega should be positive
