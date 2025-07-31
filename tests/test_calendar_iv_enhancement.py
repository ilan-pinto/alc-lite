"""
Unit tests for Calendar Spread IV Enhancement.

Tests the enhanced implied volatility calculation that uses real IB API data
instead of placeholder calculations. Covers both CalendarSpread and TermStructure modules.
"""

import time
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from modules.Arbitrage.CalendarSpread import CalendarSpread, CalendarSpreadConfig
from modules.Arbitrage.TermStructure import TermStructureAnalyzer, TermStructureConfig


class TestCalendarSpreadIVEnhancement:
    """Test enhanced IV calculation in CalendarSpread module"""

    def setup_method(self):
        """Set up test fixtures"""
        self.calendar = CalendarSpread()

    def test_iv_calculation_with_direct_iv(self):
        """Test IV calculation using ticker.impliedVolatility (priority 1)"""
        # Mock ticker with direct IV
        ticker = Mock()
        ticker.impliedVolatility = 0.25  # 25% IV
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 12345
        contract.symbol = "SPY"
        contract.strike = 450.0

        # Calculate IV
        iv = self.calendar._calculate_implied_volatility(ticker, contract)

        # Should return 25% (converted from decimal)
        assert iv == 25.0

    def test_iv_calculation_with_model_greeks(self):
        """Test IV calculation using ticker.modelGreeks.impliedVol (priority 2)"""
        # Mock ticker with model Greeks IV
        ticker = Mock()
        ticker.impliedVolatility = None
        ticker.modelGreeks = Mock()
        ticker.modelGreeks.impliedVol = 0.30  # 30% IV
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 12346
        contract.symbol = "QQQ"
        contract.strike = 350.0

        # Calculate IV
        iv = self.calendar._calculate_implied_volatility(ticker, contract)

        # Should return 30% (converted from decimal)
        assert iv == 30.0

    def test_iv_calculation_with_bid_ask_average(self):
        """Test IV calculation using bid/ask Greeks average (priority 3)"""
        # Mock ticker with bid/ask Greeks
        ticker = Mock()
        ticker.impliedVolatility = None
        ticker.modelGreeks = None
        ticker.bidGreeks = Mock()
        ticker.bidGreeks.impliedVol = 0.22  # 22% IV
        ticker.askGreeks = Mock()
        ticker.askGreeks.impliedVol = 0.28  # 28% IV
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 12347
        contract.symbol = "AAPL"
        contract.strike = 180.0

        # Calculate IV
        iv = self.calendar._calculate_implied_volatility(ticker, contract)

        # Should return average: (22 + 28) / 2 = 25%
        assert iv == 25.0

    def test_iv_calculation_fallback_to_spread_estimation(self):
        """Test IV calculation fallback to spread estimation"""
        # Mock ticker with no IB IV data but valid bid/ask
        ticker = Mock()
        ticker.impliedVolatility = None
        ticker.modelGreeks = None
        ticker.bidGreeks = None
        ticker.askGreeks = None
        ticker.bid = 2.0
        ticker.ask = 2.2
        ticker.midpoint = lambda: 2.1
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 12348
        contract.symbol = "MSFT"
        contract.strike = 300.0

        # Calculate IV
        iv = self.calendar._calculate_implied_volatility(ticker, contract)

        # Should use spread estimation formula
        spread_ratio = (2.2 - 2.0) / 2.1  # 0.095
        expected_iv = min(100.0, max(10.0, spread_ratio * 200.0))  # ~19%
        assert abs(iv - expected_iv) < 0.1

    def test_iv_calculation_default_fallback(self):
        """Test IV calculation with complete fallback to default"""
        # Mock ticker with no usable data
        ticker = Mock()
        ticker.impliedVolatility = None
        ticker.modelGreeks = None
        ticker.bidGreeks = None
        ticker.askGreeks = None
        ticker.bid = 0
        ticker.ask = 0
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 12349
        contract.symbol = "TSLA"
        contract.strike = 250.0

        # Calculate IV
        iv = self.calendar._calculate_implied_volatility(ticker, contract)

        # Should return default 25% IV
        assert iv == 25.0

    def test_iv_validation_range_limits(self):
        """Test IV validation enforces 5%-200% range"""
        # Mock ticker with extreme IV values
        ticker = Mock()
        ticker.impliedVolatility = 5.0  # 500% - should be capped at 200%
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 12350
        contract.symbol = "NVDA"
        contract.strike = 500.0

        # Calculate IV
        iv = self.calendar._calculate_implied_volatility(ticker, contract)

        # Should be capped at 200%
        assert iv == 200.0

    def test_iv_cache_functionality(self):
        """Test that IV calculation properly uses cache"""
        # Mock ticker
        ticker = Mock()
        ticker.impliedVolatility = 0.20  # 20% IV
        ticker.time = 123456789.0

        # Mock contract
        contract = Mock()
        contract.conId = 12351
        contract.symbol = "META"
        contract.strike = 300.0

        # First calculation should hit the calculation
        iv1 = self.calendar._calculate_implied_volatility(ticker, contract)
        assert iv1 == 20.0

        # Second calculation should hit cache (same time)
        iv2 = self.calendar._calculate_implied_volatility(ticker, contract)
        assert iv2 == 20.0

        # Cache should have the entry
        cache_key = f"{contract.conId}_{ticker.time}"
        assert cache_key in self.calendar.iv_cache


class TestTermStructureIVEnhancement:
    """Test enhanced IV calculation in TermStructure module"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = TermStructureConfig()
        self.analyzer = TermStructureAnalyzer(config=self.config)

    def test_termstructure_iv_calculation_with_direct_iv(self):
        """Test TermStructure IV calculation using ticker.impliedVolatility"""
        # Mock ticker with direct IV
        ticker = Mock()
        ticker.impliedVolatility = 0.35  # 35% IV
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 22345
        contract.symbol = "SPY"
        contract.strike = 450.0

        # Calculate IV
        iv = self.analyzer._calculate_implied_volatility_cached(ticker, contract)

        # Should return 35% (converted from decimal)
        assert iv == 35.0

    def test_termstructure_iv_fallback_consistency(self):
        """Test that TermStructure fallback is consistent with CalendarSpread"""
        # Mock ticker with no IB IV data
        ticker = Mock()
        ticker.impliedVolatility = None
        ticker.modelGreeks = None
        ticker.bidGreeks = None
        ticker.askGreeks = None
        ticker.bid = 1.5
        ticker.ask = 1.65
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 22346
        contract.symbol = "QQQ"
        contract.strike = 350.0

        # Calculate IV
        iv = self.analyzer._calculate_implied_volatility_cached(ticker, contract)

        # Should use spread estimation formula (same as CalendarSpread)
        mid_price = (1.5 + 1.65) / 2.0  # 1.575
        spread_ratio = (1.65 - 1.5) / mid_price  # ~0.095
        expected_iv = min(100.0, max(5.0, spread_ratio * 150.0 + 15.0))
        assert abs(iv - expected_iv) < 0.1

    def test_termstructure_cache_with_ttl(self):
        """Test TermStructure cache respects TTL configuration"""
        # Mock ticker
        ticker = Mock()
        ticker.impliedVolatility = 0.25  # 25% IV
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 22347
        contract.symbol = "AAPL"
        contract.strike = 180.0

        # First calculation
        iv1 = self.analyzer._calculate_implied_volatility_cached(ticker, contract)
        assert iv1 == 25.0

        # Check cache has entry
        cache_key = f"{contract.conId}_{ticker.time}"
        assert cache_key in self.analyzer.iv_calculation_cache

        # Second calculation should use cache
        iv2 = self.analyzer._calculate_implied_volatility_cached(ticker, contract)
        assert iv2 == 25.0


@pytest.mark.unit
class TestIVEnhancementIntegration:
    """Test integration between CalendarSpread and TermStructure IV calculations"""

    def test_iv_consistency_between_modules(self):
        """Test that both modules produce consistent IV results with same data"""
        # Set up both modules
        calendar = CalendarSpread()

        term_config = TermStructureConfig()
        analyzer = TermStructureAnalyzer(config=term_config)

        # Mock identical ticker data
        ticker = Mock()
        ticker.impliedVolatility = 0.32  # 32% IV
        ticker.time = time.time()

        # Mock identical contract
        contract = Mock()
        contract.conId = 33333
        contract.symbol = "GOOGL"
        contract.strike = 2800.0

        # Calculate IV with both modules
        calendar_iv = calendar._calculate_implied_volatility(ticker, contract)
        term_iv = analyzer._calculate_implied_volatility_cached(ticker, contract)

        # Results should be identical
        assert calendar_iv == term_iv == 32.0

    def test_iv_error_handling_consistency(self):
        """Test that both modules handle errors consistently"""
        # Set up both modules
        calendar = CalendarSpread()

        term_config = TermStructureConfig()
        analyzer = TermStructureAnalyzer(config=term_config)

        # Mock ticker that will cause AttributeError
        ticker = Mock()
        ticker.impliedVolatility = None
        ticker.modelGreeks = None
        # Don't mock bidGreeks/askGreeks to trigger AttributeError
        del ticker.bidGreeks
        del ticker.askGreeks
        ticker.bid = 0
        ticker.ask = 0
        ticker.time = time.time()

        # Mock contract
        contract = Mock()
        contract.conId = 44444
        contract.symbol = "AMZN"
        contract.strike = 3200.0

        # Both should handle error gracefully and return default
        calendar_iv = calendar._calculate_implied_volatility(ticker, contract)
        term_iv = analyzer._calculate_implied_volatility_cached(ticker, contract)

        # Both should return defaults (25% for calendar, 20% for termstructure)
        assert calendar_iv == 25.0
        assert term_iv == 20.0  # TermStructure has slightly different default
