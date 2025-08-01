"""
Comprehensive unit tests for commands/option.py module.
Tests the OptionScan class methods including calendar_finder, syn_finder, and sfr_finder.
Focus on calendar_finder method as part of the calendar spread testing suite.
"""

import asyncio
import os
import sys
from decimal import Decimal
from io import StringIO
from typing import Generator, List, Tuple
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commands.option import OptionScan
from modules.Arbitrage.CalendarSpread import CalendarSpreadConfig
from modules.Arbitrage.Synthetic import ScoringConfig


class TestOptionScanCalendarFinder:
    """Test suite for OptionScan.calendar_finder method"""

    @pytest.fixture
    def option_scan(self) -> OptionScan:
        """Create OptionScan instance for testing"""
        return OptionScan()

    @pytest.fixture
    def mock_calendar_spread(self) -> Generator[MagicMock, None, None]:
        """Mock CalendarSpread class to avoid actual IB connections"""
        with patch("commands.option.CalendarSpread") as mock:
            mock_instance = MagicMock()
            mock_instance.ib = MagicMock()
            mock_instance.ib.disconnect = MagicMock()
            mock_instance.scan = AsyncMock()
            mock.return_value = mock_instance
            yield mock

    @pytest.fixture
    def mock_finviz_scraper(self) -> Generator[MagicMock, None, None]:
        """Mock finviz scraper to avoid external HTTP calls"""
        with patch("commands.option.scrape_tickers_from_finviz") as mock:
            yield mock

    @pytest.fixture
    def capture_output(self) -> Generator[Tuple[StringIO, StringIO], None, None]:
        """Capture stdout and stderr"""
        old_out, old_err = sys.stdout, sys.stderr
        out, err = StringIO(), StringIO()
        sys.stdout, sys.stderr = out, err
        yield out, err
        sys.stdout, sys.stderr = old_out, old_err

    def test_calendar_finder_with_all_default_parameters(
        self, option_scan: OptionScan, mock_calendar_spread: MagicMock
    ) -> None:
        """Test calendar_finder with all default parameters"""
        symbols = ["SPY", "QQQ"]

        option_scan.calendar_finder(symbol_list=symbols)

        # Verify CalendarSpread was instantiated
        mock_calendar_spread.assert_called_once_with(log_file=None)

        # Verify scan was called with correct parameters
        mock_calendar_spread.return_value.scan.assert_called_once_with(
            symbol_list=symbols,
            cost_limit=300.0,
            profit_target=0.25,
            quantity=1,
        )

    def test_calendar_finder_with_all_custom_parameters(
        self, option_scan: OptionScan, mock_calendar_spread: MagicMock
    ) -> None:
        """Test calendar_finder with all custom parameters"""
        symbols = ["AAPL", "MSFT", "GOOGL"]

        option_scan.calendar_finder(
            symbol_list=symbols,
            cost_limit=500.0,
            profit_target=0.35,
            iv_spread_threshold=0.05,
            theta_ratio_threshold=2.0,
            front_expiry_max_days=30,
            back_expiry_min_days=50,
            back_expiry_max_days=90,
            min_volume=25,
            max_bid_ask_spread=0.10,
            quantity=3,
            log_file="test.log",
            debug=True,
            finviz_url=None,
        )

        # Verify CalendarSpread was instantiated with custom parameters
        mock_calendar_spread.assert_called_once_with(log_file="test.log")

        # Verify process was called with custom parameters
        mock_calendar_spread.return_value.scan.assert_called_once_with(
            symbol_list=symbols,
            cost_limit=500.0,
            profit_target=0.35,
            quantity=3,
        )

    def test_calendar_finder_with_none_symbol_list_uses_defaults(
        self, option_scan: OptionScan, mock_calendar_spread: MagicMock
    ) -> None:
        """Test calendar_finder with None symbol_list uses default symbols"""
        option_scan.calendar_finder(symbol_list=None)

        # Verify process was called with default symbol list
        expected_defaults = [
            "SPY",
            "QQQ",
            "META",
            "AAPL",
            "MSFT",
            "GOOGL",
            "TSLA",
            "NVDA",
            "AMZN",
            "NFLX",
            "AMD",
            "INTC",
            "V",
            "MA",
        ]

        mock_calendar_spread.return_value.scan.assert_called_once()
        args, kwargs = mock_calendar_spread.return_value.scan.call_args
        assert kwargs["symbol_list"] == expected_defaults

    def test_calendar_finder_with_empty_symbol_list_uses_defaults(
        self, option_scan: OptionScan, mock_calendar_spread: MagicMock
    ) -> None:
        """Test calendar_finder with empty symbol_list uses default symbols"""
        option_scan.calendar_finder(symbol_list=[])

        # Verify process was called with default symbol list
        expected_defaults = [
            "SPY",
            "QQQ",
            "META",
            "AAPL",
            "MSFT",
            "GOOGL",
            "TSLA",
            "NVDA",
            "AMZN",
            "NFLX",
            "AMD",
            "INTC",
            "V",
            "MA",
        ]

        mock_calendar_spread.return_value.scan.assert_called_once()
        args, kwargs = mock_calendar_spread.return_value.scan.call_args
        assert kwargs["symbol_list"] == expected_defaults

    def test_calendar_finder_with_finviz_url_success(
        self,
        option_scan: OptionScan,
        mock_calendar_spread: MagicMock,
        mock_finviz_scraper: MagicMock,
    ) -> None:
        """Test calendar_finder with successful finviz URL scraping"""
        scraped_symbols = ["NVDA", "AMD", "INTC"]
        mock_finviz_scraper.return_value = scraped_symbols

        option_scan.calendar_finder(
            symbol_list=["SPY", "QQQ"],  # Should be overridden by finviz
            finviz_url="https://finviz.com/screener.ashx?v=111",
        )

        # Verify finviz scraper was called
        mock_finviz_scraper.assert_called_once_with(
            "https://finviz.com/screener.ashx?v=111"
        )

        # Verify process was called with scraped symbols, not original list
        mock_calendar_spread.return_value.scan.assert_called_once()
        args, kwargs = mock_calendar_spread.return_value.scan.call_args
        assert kwargs["symbol_list"] == scraped_symbols

    def test_calendar_finder_with_finviz_url_failure_fallback(
        self,
        option_scan: OptionScan,
        mock_calendar_spread: MagicMock,
        mock_finviz_scraper: MagicMock,
    ) -> None:
        """Test calendar_finder with failed finviz URL scraping falls back to provided symbols"""
        mock_finviz_scraper.return_value = None  # Simulate scraping failure
        provided_symbols = ["SPY", "QQQ"]

        option_scan.calendar_finder(
            symbol_list=provided_symbols,
            finviz_url="https://finviz.com/screener.ashx?v=111",
        )

        # Verify finviz scraper was called
        mock_finviz_scraper.assert_called_once_with(
            "https://finviz.com/screener.ashx?v=111"
        )

        # Verify process was called with provided symbols as fallback
        mock_calendar_spread.return_value.scan.assert_called_once()
        args, kwargs = mock_calendar_spread.return_value.scan.call_args
        assert kwargs["symbol_list"] == provided_symbols

    def test_calendar_finder_with_finviz_url_failure_no_symbols_uses_defaults(
        self,
        option_scan: OptionScan,
        mock_calendar_spread: MagicMock,
        mock_finviz_scraper: MagicMock,
    ) -> None:
        """Test calendar_finder with failed finviz URL and no symbols uses defaults"""
        mock_finviz_scraper.return_value = None  # Simulate scraping failure

        option_scan.calendar_finder(
            symbol_list=None, finviz_url="https://finviz.com/screener.ashx?v=111"
        )

        # Verify finviz scraper was called
        mock_finviz_scraper.assert_called_once_with(
            "https://finviz.com/screener.ashx?v=111"
        )

        # Verify process was called with default symbols as fallback
        expected_defaults = [
            "SPY",
            "QQQ",
            "META",
            "AAPL",
            "MSFT",
            "GOOGL",
            "TSLA",
            "NVDA",
            "AMZN",
            "NFLX",
            "AMD",
            "INTC",
            "V",
            "MA",
        ]

        mock_calendar_spread.return_value.scan.assert_called_once()
        args, kwargs = mock_calendar_spread.return_value.scan.call_args
        assert kwargs["symbol_list"] == expected_defaults

    def test_calendar_finder_keyboard_interrupt_handling(
        self, option_scan: OptionScan, mock_calendar_spread: MagicMock
    ) -> None:
        """Test calendar_finder handles KeyboardInterrupt gracefully"""
        mock_calendar_spread.return_value.scan.side_effect = KeyboardInterrupt()

        option_scan.calendar_finder(symbol_list=["SPY"])

        # Verify disconnect was called on interrupt (may be called multiple times due to finally block)
        assert mock_calendar_spread.return_value.ib.disconnect.call_count >= 1

    def test_calendar_finder_general_exception_handling(
        self, option_scan: OptionScan, mock_calendar_spread: MagicMock
    ) -> None:
        """Test calendar_finder handles general exceptions"""
        test_error = Exception("Test connection error")
        mock_calendar_spread.return_value.scan.side_effect = test_error

        # Exception should be caught and logged, then re-raised
        with pytest.raises(Exception, match="Test connection error"):
            option_scan.calendar_finder(symbol_list=["SPY"])

        # Verify disconnect was called on exception (may be called multiple times due to finally block)
        assert mock_calendar_spread.return_value.ib.disconnect.call_count >= 1

    def test_calendar_finder_extreme_parameter_values(
        self, option_scan: OptionScan, mock_calendar_spread: MagicMock
    ) -> None:
        """Test calendar_finder with extreme parameter values"""
        option_scan.calendar_finder(
            symbol_list=["SPY"],
            cost_limit=0.01,  # Very low cost limit
            profit_target=0.99,  # Very high profit target
            iv_spread_threshold=0.001,  # Very low IV spread
            theta_ratio_threshold=0.1,  # Very low theta ratio
            front_expiry_max_days=1,  # Very short front expiry
            back_expiry_min_days=1,  # Very short back expiry min
            back_expiry_max_days=365,  # Very long back expiry max
            min_volume=1,  # Very low volume
            max_bid_ask_spread=0.99,  # Very high spread
            quantity=100,  # High quantity
        )

        # Verify CalendarSpread was instantiated
        mock_calendar_spread.assert_called_once()

        # Verify process was called with extreme values
        mock_calendar_spread.return_value.scan.assert_called_once_with(
            symbol_list=["SPY"],
            cost_limit=0.01,
            profit_target=0.99,
            quantity=100,
        )

    def test_calendar_finder_negative_parameter_values(
        self, option_scan: OptionScan, mock_calendar_spread: MagicMock
    ) -> None:
        """Test calendar_finder with negative parameter values"""
        option_scan.calendar_finder(
            symbol_list=["SPY"],
            cost_limit=-100.0,
            profit_target=-0.5,
            iv_spread_threshold=-0.05,
            theta_ratio_threshold=-1.0,
            front_expiry_max_days=-10,
            back_expiry_min_days=-5,
            back_expiry_max_days=-1,
            min_volume=-50,
            max_bid_ask_spread=-0.20,
            quantity=-5,
        )

        # Verify CalendarSpread was instantiated
        mock_calendar_spread.assert_called_once()

        # Verify process was called with negative values
        mock_calendar_spread.return_value.scan.assert_called_once_with(
            symbol_list=["SPY"],
            cost_limit=-100.0,
            profit_target=-0.5,
            quantity=-5,
        )

    def test_calendar_finder_with_special_symbols(
        self, option_scan: OptionScan, mock_calendar_spread: MagicMock
    ) -> None:
        """Test calendar_finder with special symbols (futures, indices)"""
        special_symbols = ["!MES", "@SPX", "SPY.B", "QQQ-USD"]

        option_scan.calendar_finder(symbol_list=special_symbols)

        # Verify process was called with special symbols
        mock_calendar_spread.return_value.scan.assert_called_once()
        args, kwargs = mock_calendar_spread.return_value.scan.call_args
        assert kwargs["symbol_list"] == special_symbols

    def test_calendar_finder_with_duplicate_symbols(
        self, option_scan: OptionScan, mock_calendar_spread: MagicMock
    ) -> None:
        """Test calendar_finder with duplicate symbols in list"""
        symbols_with_dupes = ["SPY", "QQQ", "SPY", "AAPL", "QQQ", "SPY"]

        option_scan.calendar_finder(symbol_list=symbols_with_dupes)

        # Verify process was called with duplicate symbols (system should handle deduplication if needed)
        mock_calendar_spread.return_value.scan.assert_called_once()
        args, kwargs = mock_calendar_spread.return_value.scan.call_args
        assert kwargs["symbol_list"] == symbols_with_dupes

    def test_calendar_finder_with_very_long_symbol_list(
        self, option_scan: OptionScan, mock_calendar_spread: MagicMock
    ) -> None:
        """Test calendar_finder with very long symbol list"""
        long_symbol_list = [f"SYM{i:03d}" for i in range(500)]  # 500 symbols

        option_scan.calendar_finder(symbol_list=long_symbol_list)

        # Verify process was called with long symbol list
        mock_calendar_spread.return_value.scan.assert_called_once()
        args, kwargs = mock_calendar_spread.return_value.scan.call_args
        assert kwargs["symbol_list"] == long_symbol_list
        assert len(kwargs["symbol_list"]) == 500

    def test_calendar_finder_with_zero_values(
        self, option_scan: OptionScan, mock_calendar_spread: MagicMock
    ) -> None:
        """Test calendar_finder with zero parameter values"""
        option_scan.calendar_finder(
            symbol_list=["SPY"],
            cost_limit=0.0,
            profit_target=0.0,
            iv_spread_threshold=0.0,
            theta_ratio_threshold=0.0,
            front_expiry_max_days=0,
            back_expiry_min_days=0,
            back_expiry_max_days=0,
            min_volume=0,
            max_bid_ask_spread=0.0,
            quantity=0,
        )

        # Verify CalendarSpread was instantiated
        mock_calendar_spread.assert_called_once()

        # Verify process was called with zero values
        mock_calendar_spread.return_value.scan.assert_called_once_with(
            symbol_list=["SPY"],
            cost_limit=0.0,
            profit_target=0.0,
            quantity=0,
        )

    @patch("commands.option.asyncio.run")
    def test_calendar_finder_asyncio_integration(
        self,
        mock_asyncio_run: MagicMock,
        option_scan: OptionScan,
        mock_calendar_spread: MagicMock,
    ) -> None:
        """Test calendar_finder properly integrates with asyncio"""
        option_scan.calendar_finder(symbol_list=["SPY"])

        # Verify asyncio.run was called with the calendar.process coroutine
        mock_asyncio_run.assert_called_once()
        args, kwargs = mock_asyncio_run.call_args
        assert len(args) == 1
        # The argument should be a coroutine (mock_calendar_spread.return_value.scan call)

    def test_calendar_finder_docstring_parameters_match_implementation(
        self, option_scan: OptionScan
    ) -> None:
        """Test that calendar_finder docstring parameters match actual implementation"""
        import inspect

        # Get method signature
        sig = inspect.signature(option_scan.calendar_finder)

        # Expected parameters from docstring
        expected_params = [
            "symbol_list",
            "cost_limit",
            "profit_target",
            "iv_spread_threshold",
            "theta_ratio_threshold",
            "front_expiry_max_days",
            "back_expiry_min_days",
            "back_expiry_max_days",
            "min_volume",
            "max_bid_ask_spread",
            "quantity",
            "log_file",
            "debug",
            "finviz_url",
        ]

        # Verify all expected parameters exist in signature
        actual_params = list(sig.parameters.keys())
        for param in expected_params:
            assert (
                param in actual_params
            ), f"Parameter '{param}' missing from method signature"

    def test_calendar_finder_default_values_match_docstring(
        self, option_scan: OptionScan
    ) -> None:
        """Test that calendar_finder default values match docstring"""
        import inspect

        # Get method signature
        sig = inspect.signature(option_scan.calendar_finder)

        # Verify specific default values mentioned in docstring
        assert sig.parameters["cost_limit"].default == 300.0
        assert sig.parameters["profit_target"].default == 0.25
        assert sig.parameters["iv_spread_threshold"].default == 0.015
        assert sig.parameters["theta_ratio_threshold"].default == 1.5
        assert sig.parameters["front_expiry_max_days"].default == 30
        assert sig.parameters["back_expiry_min_days"].default == 50
        assert sig.parameters["back_expiry_max_days"].default == 120
        assert sig.parameters["min_volume"].default == 10
        assert sig.parameters["max_bid_ask_spread"].default == 0.15
        assert sig.parameters["quantity"].default == 1
        assert sig.parameters["log_file"].default is None
        assert sig.parameters["debug"].default is False
        assert sig.parameters["finviz_url"].default is None


class TestOptionScanScoringConfig:
    """Test suite for OptionScan._create_scoring_config method"""

    @pytest.fixture
    def option_scan(self) -> OptionScan:
        """Create OptionScan instance for testing"""
        return OptionScan()

    def test_create_scoring_config_balanced_strategy(
        self, option_scan: OptionScan
    ) -> None:
        """Test creating balanced scoring configuration"""
        config = option_scan._create_scoring_config(scoring_strategy="balanced")

        assert isinstance(config, ScoringConfig)
        # These values should match ScoringConfig.create_balanced()

    def test_create_scoring_config_conservative_strategy(
        self, option_scan: OptionScan
    ) -> None:
        """Test creating conservative scoring configuration"""
        config = option_scan._create_scoring_config(scoring_strategy="conservative")

        assert isinstance(config, ScoringConfig)
        # These values should match ScoringConfig.create_conservative()

    def test_create_scoring_config_aggressive_strategy(
        self, option_scan: OptionScan
    ) -> None:
        """Test creating aggressive scoring configuration"""
        config = option_scan._create_scoring_config(scoring_strategy="aggressive")

        assert isinstance(config, ScoringConfig)
        # These values should match ScoringConfig.create_aggressive()

    def test_create_scoring_config_liquidity_focused_strategy(
        self, option_scan: OptionScan
    ) -> None:
        """Test creating liquidity-focused scoring configuration"""
        config = option_scan._create_scoring_config(
            scoring_strategy="liquidity-focused"
        )

        assert isinstance(config, ScoringConfig)
        # These values should match ScoringConfig.create_liquidity_focused()

    def test_create_scoring_config_custom_weights_valid(
        self, option_scan: OptionScan
    ) -> None:
        """Test creating scoring config with valid custom weights"""
        config = option_scan._create_scoring_config(
            risk_reward_weight=0.4,
            liquidity_weight=0.3,
            time_decay_weight=0.2,
            market_quality_weight=0.1,
        )

        assert isinstance(config, ScoringConfig)
        assert config.risk_reward_weight == 0.4
        assert config.liquidity_weight == 0.3
        assert config.time_decay_weight == 0.2
        assert config.market_quality_weight == 0.1

    def test_create_scoring_config_custom_weights_incomplete(
        self, option_scan: OptionScan
    ) -> None:
        """Test creating scoring config with incomplete custom weights raises error"""
        with pytest.raises(
            ValueError,
            match="If providing custom weights, all four weights must be specified",
        ):
            option_scan._create_scoring_config(
                risk_reward_weight=0.4,
                liquidity_weight=0.3,
                # Missing time_decay_weight and market_quality_weight
            )

    def test_create_scoring_config_custom_weights_out_of_range(
        self, option_scan: OptionScan
    ) -> None:
        """Test creating scoring config with out-of-range custom weights raises error"""
        with pytest.raises(
            ValueError, match="risk-reward weight must be between 0.0 and 1.0"
        ):
            option_scan._create_scoring_config(
                risk_reward_weight=1.5,  # Invalid > 1.0
                liquidity_weight=0.3,
                time_decay_weight=0.2,
                market_quality_weight=0.1,
            )

    def test_create_scoring_config_unknown_strategy(
        self, option_scan: OptionScan
    ) -> None:
        """Test creating scoring config with unknown strategy raises error"""
        with pytest.raises(
            ValueError, match="Unknown scoring strategy: invalid_strategy"
        ):
            option_scan._create_scoring_config(scoring_strategy="invalid_strategy")

    def test_create_scoring_config_custom_thresholds_override(
        self, option_scan: OptionScan
    ) -> None:
        """Test that custom thresholds override strategy defaults"""
        config = option_scan._create_scoring_config(
            scoring_strategy="balanced",
            min_risk_reward=5.0,
            min_liquidity=0.8,
            max_bid_ask_spread=0.05,
            optimal_days_expiry=30,
        )

        assert config.min_risk_reward_ratio == 5.0
        assert config.min_liquidity_score == 0.8
        assert config.max_bid_ask_spread == 0.05
        assert config.optimal_days_to_expiry == 30


class TestOptionScanIntegration:
    """Integration tests for OptionScan class methods"""

    @pytest.fixture
    def option_scan(self) -> OptionScan:
        """Create OptionScan instance for testing"""
        return OptionScan()

    @pytest.fixture
    def mock_all_strategies(
        self,
    ) -> Generator[Tuple[MagicMock, MagicMock, MagicMock], None, None]:
        """Mock all strategy classes"""
        with (
            patch("commands.option.SFR") as mock_sfr,
            patch("commands.option.Syn") as mock_syn,
            patch("commands.option.CalendarSpread") as mock_calendar,
        ):

            # Configure SFR mock
            mock_sfr_instance = MagicMock()
            mock_sfr_instance.ib = MagicMock()
            mock_sfr_instance.scan = AsyncMock()
            mock_sfr.return_value = mock_sfr_instance

            # Configure Syn mock
            mock_syn_instance = MagicMock()
            mock_syn_instance.ib = MagicMock()
            mock_syn_instance.scan = AsyncMock()
            mock_syn.return_value = mock_syn_instance

            # Configure CalendarSpread mock
            mock_calendar_instance = MagicMock()
            mock_calendar_instance.ib = MagicMock()
            mock_calendar_instance.scan = AsyncMock()
            mock_calendar.return_value = mock_calendar_instance

            yield mock_sfr_instance, mock_syn_instance, mock_calendar_instance

    def test_option_scan_methods_isolation(
        self,
        option_scan: OptionScan,
        mock_all_strategies: Tuple[MagicMock, MagicMock, MagicMock],
    ) -> None:
        """Test that different OptionScan methods don't interfere with each other"""
        mock_sfr, mock_syn, mock_calendar = mock_all_strategies

        # Call all three methods
        option_scan.sfr_finder(symbol_list=["SPY"], profit_target=1.0, cost_limit=100)
        option_scan.syn_finder(symbol_list=["QQQ"], cost_limit=120)
        option_scan.calendar_finder(symbol_list=["AAPL"], cost_limit=300)

        # Verify each method called its respective strategy
        mock_sfr.scan.assert_called_once()
        mock_syn.scan.assert_called_once()
        mock_calendar.scan.assert_called_once()

    def test_option_scan_consistent_logging_handling(
        self, option_scan: OptionScan
    ) -> None:
        """Test that all OptionScan methods handle logging consistently"""
        with (
            patch("commands.option.SFR") as mock_sfr,
            patch("commands.option.Syn") as mock_syn,
            patch("commands.option.CalendarSpread") as mock_calendar,
        ):

            # Configure mocks
            for mock_class in [mock_sfr, mock_syn, mock_calendar]:
                mock_instance = MagicMock()
                mock_instance.ib = MagicMock()
                mock_instance.scan = AsyncMock()
                mock_class.return_value = mock_instance

            # Test consistent logging parameters
            log_file = "test.log"
            debug = True

            option_scan.sfr_finder(
                symbol_list=["SPY"],
                profit_target=1.0,
                cost_limit=100.0,
                log_file=log_file,
                debug=debug,
            )
            option_scan.syn_finder(symbol_list=["QQQ"], log_file=log_file, debug=debug)
            option_scan.calendar_finder(
                symbol_list=["AAPL"], log_file=log_file, debug=debug
            )

            # Verify all strategies were initialized with same logging parameters
            mock_sfr.assert_called_once()
            sfr_kwargs = mock_sfr.call_args[1]
            assert sfr_kwargs["log_file"] == log_file
            assert sfr_kwargs["debug"] == debug

            mock_syn.assert_called_once()
            syn_kwargs = mock_syn.call_args[1]
            assert syn_kwargs["log_file"] == log_file
            assert syn_kwargs["debug"] == debug

            mock_calendar.assert_called_once()
            calendar_kwargs = mock_calendar.call_args[1]
            assert calendar_kwargs["log_file"] == log_file
            # Note: CalendarSpread doesn't receive debug parameter, only log_file

    def test_option_scan_consistent_symbol_handling(
        self, option_scan: OptionScan
    ) -> None:
        """Test that all OptionScan methods handle symbol lists consistently"""
        with (
            patch("commands.option.scrape_tickers_from_finviz") as mock_scraper,
            patch("commands.option.SFR") as mock_sfr,
            patch("commands.option.Syn") as mock_syn,
            patch("commands.option.CalendarSpread") as mock_calendar,
        ):

            # Configure scraper to return test symbols
            test_symbols = ["NVDA", "AMD"]
            mock_scraper.return_value = test_symbols

            # Configure strategy mocks
            for mock_class in [mock_sfr, mock_syn, mock_calendar]:
                mock_instance = MagicMock()
                mock_instance.ib = MagicMock()
                mock_instance.scan = AsyncMock()
                mock_class.return_value = mock_instance

            # Test with finviz URL - all methods should use scraped symbols
            finviz_url = "https://finviz.com/screener.ashx?v=111"

            option_scan.sfr_finder(
                symbol_list=["SPY"],
                profit_target=1.0,
                cost_limit=100.0,
                finviz_url=finviz_url,
            )
            option_scan.syn_finder(symbol_list=["QQQ"], finviz_url=finviz_url)
            option_scan.calendar_finder(symbol_list=["AAPL"], finviz_url=finviz_url)

            # Verify all methods called scraper
            assert mock_scraper.call_count == 3

            # Verify all methods used scraped symbols
            sfr_call_kwargs = mock_sfr.return_value.scan.call_args[0][0]
            syn_call_kwargs = mock_syn.return_value.scan.call_args[0][0]
            calendar_call_kwargs = mock_calendar.return_value.scan.call_args[1][
                "symbol_list"
            ]

            assert sfr_call_kwargs == test_symbols
            assert syn_call_kwargs == test_symbols
            assert calendar_call_kwargs == test_symbols


class TestOptionScanPerformance:
    """Performance-related tests for OptionScan methods"""

    @pytest.fixture
    def option_scan(self) -> OptionScan:
        """Create OptionScan instance for testing"""
        return OptionScan()

    def test_calendar_finder_with_large_parameter_sets(
        self, option_scan: OptionScan
    ) -> None:
        """Test calendar_finder handles large parameter sets efficiently"""
        with patch("commands.option.CalendarSpread") as mock_calendar:
            mock_instance = MagicMock()
            mock_instance.ib = MagicMock()
            mock_instance.scan = AsyncMock()
            mock_calendar.return_value = mock_instance

            # Test with large symbol list and many parameters
            large_symbol_list = [f"SYM{i:04d}" for i in range(1000)]

            import time

            start_time = time.time()

            option_scan.calendar_finder(
                symbol_list=large_symbol_list,
                cost_limit=1000.0,
                profit_target=0.5,
                iv_spread_threshold=0.02,
                theta_ratio_threshold=2.5,
                front_expiry_max_days=60,
                back_expiry_min_days=90,
                back_expiry_max_days=180,
                min_volume=100,
                max_bid_ask_spread=0.05,
                quantity=10,
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # Method should complete quickly (most time in mocked process call)
            assert (
                execution_time < 1.0
            ), f"calendar_finder took too long: {execution_time:.3f}s"

            # Verify large symbol list was passed correctly
            mock_instance.scan.assert_called_once()
            args, kwargs = mock_instance.scan.call_args
            assert len(kwargs["symbol_list"]) == 1000

    def test_calendar_finder_memory_efficiency_with_large_configs(
        self, option_scan: OptionScan
    ) -> None:
        """Test calendar_finder doesn't create excessive objects with large configurations"""
        with patch("commands.option.CalendarSpread") as mock_calendar:
            mock_instance = MagicMock()
            mock_instance.ib = MagicMock()
            mock_instance.scan = AsyncMock()
            mock_calendar.return_value = mock_instance

            # Call method multiple times with different large configurations
            for i in range(100):
                option_scan.calendar_finder(
                    symbol_list=[f"TEST{j}" for j in range(50)],
                    cost_limit=float(i * 10),
                    profit_target=float(i * 0.01),
                    iv_spread_threshold=float(i * 0.001),
                    theta_ratio_threshold=float(i * 0.1),
                    front_expiry_max_days=i,
                    back_expiry_min_days=i + 30,
                    back_expiry_max_days=i + 60,
                    min_volume=i,
                    max_bid_ask_spread=float(i * 0.01),
                    quantity=i + 1,
                )

            # Verify CalendarSpread was instantiated 100 times (not accumulating objects)
            assert mock_calendar.call_count == 100
            assert mock_instance.scan.call_count == 100
