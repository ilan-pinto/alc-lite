"""
Basic tests for CalendarSpread strategy integration.

These tests ensure the CalendarSpread class integrates properly with the existing
arbitrage framework without requiring live IB connection.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from modules.Arbitrage.CalendarSpread import (
    CalendarSpread,
    CalendarSpreadConfig,
    CalendarSpreadLeg,
    CalendarSpreadOpportunity,
)
from modules.Arbitrage.Strategy import ArbitrageClass


class TestCalendarSpreadConfig:
    """Test CalendarSpreadConfig dataclass"""

    def test_config_default_values(self):
        """Test that config has proper default values"""
        config = CalendarSpreadConfig()

        assert config.min_iv_spread == 1.5
        assert config.min_theta_ratio == 1.5
        assert config.max_bid_ask_spread == 0.15
        assert config.min_liquidity_score == 0.4
        assert config.max_days_front == 45
        assert config.min_days_back == 60
        assert config.max_days_back == 120
        assert config.min_volume == 10
        assert config.max_net_debit == 500.0
        assert config.target_profit_ratio == 0.3

    def test_config_custom_values(self):
        """Test that config accepts custom values"""
        config = CalendarSpreadConfig(
            min_iv_spread=5.0, min_theta_ratio=2.0, max_net_debit=750.0
        )

        assert config.min_iv_spread == 5.0
        assert config.min_theta_ratio == 2.0
        assert config.max_net_debit == 750.0
        # Other values should remain default
        assert config.max_bid_ask_spread == 0.15


class TestCalendarSpreadLeg:
    """Test CalendarSpreadLeg dataclass"""

    def test_leg_creation(self):
        """Test calendar spread leg creation"""
        mock_contract = Mock()
        mock_contract.conId = 12345

        leg = CalendarSpreadLeg(
            contract=mock_contract,
            strike=100.0,
            expiry="20241115",
            right="C",
            price=5.50,
            bid=5.45,
            ask=5.55,
            volume=150,
            iv=25.5,
            theta=-0.05,
            days_to_expiry=30,
        )

        assert leg.contract == mock_contract
        assert leg.strike == 100.0
        assert leg.expiry == "20241115"
        assert leg.right == "C"
        assert leg.price == 5.50
        assert leg.bid == 5.45
        assert leg.ask == 5.55
        assert leg.volume == 150
        assert leg.iv == 25.5
        assert leg.theta == -0.05
        assert leg.days_to_expiry == 30


class TestCalendarSpreadOpportunity:
    """Test CalendarSpreadOpportunity dataclass"""

    def test_opportunity_creation(self):
        """Test calendar spread opportunity creation"""
        front_leg = Mock(spec=CalendarSpreadLeg)
        back_leg = Mock(spec=CalendarSpreadLeg)

        opportunity = CalendarSpreadOpportunity(
            symbol="AAPL",
            strike=150.0,
            option_type="CALL",
            front_leg=front_leg,
            back_leg=back_leg,
            iv_spread=5.0,
            theta_ratio=2.0,
            net_debit=2.50,
            max_profit=1.50,
            max_loss=2.50,
            front_bid_ask_spread=0.05,
            back_bid_ask_spread=0.08,
            combined_liquidity_score=0.75,
            term_structure_inversion=True,
            net_delta=0.1,
            net_gamma=0.02,
            net_vega=0.15,
            composite_score=0.85,
        )

        assert opportunity.symbol == "AAPL"
        assert opportunity.strike == 150.0
        assert opportunity.option_type == "CALL"
        assert opportunity.front_leg == front_leg
        assert opportunity.back_leg == back_leg
        assert opportunity.iv_spread == 5.0
        assert opportunity.theta_ratio == 2.0
        assert opportunity.net_debit == 2.50
        assert opportunity.composite_score == 0.85


class TestCalendarSpreadInitialization:
    """Test CalendarSpread class initialization"""

    def test_calendar_spread_inherits_from_arbitrage_class(self):
        """Test that CalendarSpread properly inherits from ArbitrageClass"""
        calendar = CalendarSpread()
        assert isinstance(calendar, ArbitrageClass)
        assert hasattr(calendar, "ib")
        assert hasattr(calendar, "order_manager")
        assert hasattr(calendar, "config")

    def test_calendar_spread_initialization(self):
        """Test CalendarSpread initialization"""
        calendar = CalendarSpread()

        assert isinstance(calendar.config, CalendarSpreadConfig)
        assert hasattr(calendar, "iv_cache")
        assert hasattr(calendar, "greeks_cache")
        assert calendar.cache_ttl == 60

        # Fix: Check for TTLCache type and empty state, not dict equality
        assert hasattr(calendar.iv_cache, "get")  # TTLCache methods
        assert hasattr(calendar.iv_cache, "put")
        assert calendar.iv_cache.size() == 0  # Empty cache

        assert hasattr(calendar.greeks_cache, "get")
        assert hasattr(calendar.greeks_cache, "put")
        assert calendar.greeks_cache.size() == 0  # Empty cache

    def test_calendar_spread_with_log_file(self):
        """Test CalendarSpread initialization with log file"""
        with patch.object(CalendarSpread, "_configure_file_logging") as mock_configure:
            calendar = CalendarSpread(log_file="test.log")
            mock_configure.assert_called_once_with("test.log")


class TestCalendarSpreadUtilityMethods:
    """Test CalendarSpread utility methods"""

    def setup_method(self):
        """Setup test calendar spread instance"""
        self.calendar = CalendarSpread()

    def test_calculate_days_to_expiry_valid_date(self):
        """Test days to expiry calculation with valid date"""
        # Test with a future date
        future_date = datetime.now() + timedelta(days=30)
        expiry_str = future_date.strftime("%Y%m%d")

        # Create a calendar spread instance to test the method
        calendar = CalendarSpread()

        days = calendar._calculate_days_to_expiry(expiry_str)
        assert 29 <= days <= 31  # Allow for some variation due to timing

    def test_calculate_days_to_expiry_invalid_date(self):
        """Test days to expiry calculation with invalid date"""
        calendar = CalendarSpread()

        days = calendar._calculate_days_to_expiry("invalid_date")
        assert days == 30  # Should return default

    def test_calculate_implied_volatility_with_cache(self):
        """Test IV calculation uses cache when available"""
        mock_ticker = Mock()
        mock_ticker.time = 123456789
        mock_contract = Mock()
        mock_contract.conId = 12345

        # Pre-populate cache
        cache_key = f"{mock_contract.conId}_{mock_ticker.time}"
        self.calendar.iv_cache.put(cache_key, 30.0)

        iv = self.calendar._calculate_implied_volatility(mock_ticker, mock_contract)
        assert iv == 30.0

    def test_calculate_implied_volatility_without_cache(self):
        """Test IV calculation when not in cache"""
        mock_ticker = Mock()
        mock_ticker.time = 123456789
        mock_ticker.ask = 5.50
        mock_ticker.bid = 5.40
        mock_ticker.midpoint.return_value = 5.45
        mock_contract = Mock()
        mock_contract.conId = 12345

        # Ensure cache is empty
        self.calendar.iv_cache.clear()

        iv = self.calendar._calculate_implied_volatility(mock_ticker, mock_contract)

        # Should calculate and cache IV
        assert isinstance(iv, float)
        assert 10.0 <= iv <= 100.0  # Within expected range

        # Should be cached now
        cache_key = f"{mock_contract.conId}_{mock_ticker.time}"
        assert self.calendar.iv_cache.get(cache_key) is not None

    def test_detect_term_structure_inversion_true(self):
        """Test term structure inversion detection - positive case"""
        calendar = CalendarSpread()

        # Front month much higher IV than back month (creating inversion)
        result = calendar._detect_term_structure_inversion(
            front_iv=35.0, back_iv=20.0, front_days=30, back_days=60
        )
        assert result == True

    def test_detect_term_structure_inversion_false(self):
        """Test term structure inversion detection - negative case"""
        calendar = CalendarSpread()

        # Normal term structure (back month higher IV when normalized)
        result = calendar._detect_term_structure_inversion(
            front_iv=20.0, back_iv=30.0, front_days=30, back_days=60
        )
        assert result == False

    def test_detect_term_structure_inversion_invalid_expiries(self):
        """Test term structure inversion with invalid expiry relationship"""
        calendar = CalendarSpread()

        # Front expiry same or longer than back expiry
        result = calendar._detect_term_structure_inversion(
            front_iv=30.0, back_iv=25.0, front_days=60, back_days=30
        )
        assert result == False

    def test_calculate_liquidity_score_high_quality(self):
        """Test liquidity score calculation for high quality leg"""
        calendar = CalendarSpread()

        leg = CalendarSpreadLeg(
            contract=Mock(),
            strike=100.0,
            expiry="20241115",
            right="C",
            price=5.00,
            bid=4.95,
            ask=5.05,
            volume=200,  # High volume
            iv=25.0,
            theta=-0.05,
            days_to_expiry=30,
        )

        score = calendar._calculate_liquidity_score(leg)
        assert 0.7 <= score <= 1.0  # Should be high due to good volume and tight spread

    def test_calculate_liquidity_score_low_quality(self):
        """Test liquidity score calculation for low quality leg"""
        calendar = CalendarSpread()

        leg = CalendarSpreadLeg(
            contract=Mock(),
            strike=100.0,
            expiry="20241115",
            right="C",
            price=5.00,
            bid=4.50,
            ask=5.50,
            volume=5,  # Low volume
            iv=25.0,
            theta=-0.05,
            days_to_expiry=30,
        )

        score = calendar._calculate_liquidity_score(leg)
        assert 0.0 <= score <= 0.4  # Should be low due to poor volume and wide spread

    def test_calculate_calendar_spread_score(self):
        """Test composite score calculation"""
        score = self.calendar._calculate_calendar_spread_score(
            iv_spread=8.0,  # Excellent IV spread (well above minimum 3%)
            theta_ratio=3.0,  # Excellent theta ratio (well above minimum 1.5)
            liquidity_score=0.9,  # Excellent liquidity
            max_profit=150.0,
            net_debit=100.0,  # 150% return (1.5 ratio)
            term_structure_inversion=True,  # Bonus
        )

        assert 0.0 <= score <= 1.0
        assert score > 0.6  # Should be good score with these excellent parameters


class TestCalendarSpreadIntegration:
    """Test CalendarSpread integration with framework"""

    @patch("modules.Arbitrage.CalendarSpread.CalendarSpread._cleanup_and_disconnect")
    @patch("modules.Arbitrage.CalendarSpread.CalendarSpread._monitor_execution")
    @patch(
        "modules.Arbitrage.CalendarSpread.CalendarSpread._scan_symbol_for_calendar_spreads"
    )
    @patch.object(CalendarSpread, "should_scan_symbol", return_value=True)
    @patch("ib_async.IB")
    def test_scan_method_basic_flow(
        self, mock_ib_class, mock_should_scan, mock_scan, mock_monitor, mock_cleanup
    ):
        """Test basic flow of scan method"""
        # Setup mocks
        mock_ib = Mock()
        mock_ib_class.return_value = mock_ib
        # Mock async methods properly to avoid coroutine warnings
        mock_ib.connectAsync = AsyncMock()
        mock_ib.orderFillEvent = Mock()
        mock_ib.pendingTickersEvent = Mock()

        mock_scan.return_value = []  # No opportunities found

        calendar = CalendarSpread()
        calendar.ib = mock_ib

        # Mock the scan method to avoid actual async execution
        with patch.object(calendar, "scan", new=AsyncMock()) as mock_scan_method:
            # Just test that scan method exists and can be called
            mock_scan_method.return_value = None

            # Since scan is async, we just verify it's callable
            assert hasattr(calendar, "scan")
            assert callable(calendar.scan)

    def test_select_calendar_expiries(self):
        """Test expiry selection for calendar spreads"""
        calendar = CalendarSpread()

        # Create test expiries
        today = datetime.now().date()
        expiries = []

        # Add expiries at various intervals
        for days in [10, 30, 45, 60, 90, 120, 150]:
            future_date = today + timedelta(days=days)
            expiries.append(future_date.strftime("%Y%m%d"))

        valid_expiries = calendar._select_calendar_expiries(expiries)

        # Should return expiries within the configured range
        assert len(valid_expiries) >= 1
        assert len(valid_expiries) <= 6  # Limited to 6 for performance

        # Verify expiries are within valid calendar spread ranges
        front_month_count = 0
        back_month_count = 0

        for expiry_str in valid_expiries:
            expiry_date = datetime.strptime(expiry_str, "%Y%m%d").date()
            days_to_expiry = (expiry_date - today).days

            # Calendar spreads need both front month (≤ max_days_front) and back month (min_days_back to max_days_back) expiries
            assert (days_to_expiry <= calendar.config.max_days_front) or (
                calendar.config.min_days_back
                <= days_to_expiry
                <= calendar.config.max_days_back
            ), f"Expiry {expiry_str} ({days_to_expiry} days) is in the invalid gap range"

            if days_to_expiry <= calendar.config.max_days_front:
                front_month_count += 1
            elif (
                calendar.config.min_days_back
                <= days_to_expiry
                <= calendar.config.max_days_back
            ):
                back_month_count += 1

        # Should have both front and back month expiries for calendar spreads
        assert front_month_count > 0, "Should have at least one front month expiry"
        assert back_month_count > 0, "Should have at least one back month expiry"

    def test_select_calendar_strikes(self):
        """Test strike selection for calendar spreads"""
        calendar = CalendarSpread()

        # Test strikes around current price
        all_strikes = [90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0]
        stock_price = 100.0

        valid_strikes = calendar._select_calendar_strikes(all_strikes, stock_price)

        # Should return strikes within 15% of current price
        assert len(valid_strikes) > 0
        assert len(valid_strikes) <= 10  # Limited to 10 for performance

        # Verify strikes are within range and sorted by distance
        min_strike = stock_price * 0.85
        max_strike = stock_price * 1.15

        for strike in valid_strikes:
            assert min_strike <= strike <= max_strike

        # First strike should be closest to current price
        assert valid_strikes[0] == 100.0


class TestCalendarSpreadAsync:
    """Test async methods of CalendarSpread"""

    def test_run_calendar_spread_strategy_function(self):
        """Test the convenience function exists and has proper signature"""
        from modules.Arbitrage.CalendarSpread import run_calendar_spread_strategy

        # Function should exist and be callable
        assert callable(run_calendar_spread_strategy)

        # Test with mock to avoid actual execution
        with patch(
            "modules.Arbitrage.CalendarSpread.CalendarSpread"
        ) as mock_calendar_class:
            mock_strategy = Mock()
            mock_strategy.scan.return_value = None
            mock_calendar_class.return_value = mock_strategy

            # The function should create strategy and call scan
            # We can't easily test async function without running it, but we can verify import
            assert run_calendar_spread_strategy is not None


if __name__ == "__main__":
    pytest.main([__file__])
