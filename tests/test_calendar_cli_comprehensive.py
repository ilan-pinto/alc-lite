"""
Comprehensive CLI tests for calendar subcommand in alchimest.py.
Tests all calendar command argument combinations, validation, and edge cases.
"""

import os
import sys
from io import StringIO
from typing import Generator, Tuple
from unittest.mock import MagicMock, patch

import pytest

# Add the project root to the path so we can import alchimest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alchimest


class TestCalendarCLIArguments:
    """Test class for calendar CLI argument validation"""

    @pytest.fixture
    def mock_option_scan(self) -> Generator[MagicMock, None, None]:
        """Mock OptionScan class to avoid actual IB connections"""
        with patch("alchimest.OptionScan") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def capture_output(self) -> Generator[Tuple[StringIO, StringIO], None, None]:
        """Capture stdout and stderr"""
        old_out, old_err = sys.stdout, sys.stderr
        out, err = StringIO(), StringIO()
        sys.stdout, sys.stderr = out, err
        yield out, err
        sys.stdout, sys.stderr = old_out, old_err

    @pytest.mark.integration
    def test_calendar_command_with_all_default_arguments(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with minimal arguments (using defaults)"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
            "QQQ",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.calendar_finder.assert_called_once_with(
            symbol_list=["SPY", "QQQ"],
            cost_limit=300.0,  # Default
            profit_target=0.25,  # Default
            iv_spread_threshold=0.015,  # Default
            theta_ratio_threshold=1.5,  # Default
            front_expiry_max_days=45,  # Default
            back_expiry_min_days=60,  # Default
            back_expiry_max_days=120,  # Default
            min_volume=10,  # Default
            max_bid_ask_spread=0.15,  # Default
            quantity=1,  # Default
            log_file=None,  # Default
            debug=False,  # Default
            finviz_url=None,  # Default
        )

    @pytest.mark.integration
    def test_calendar_command_with_all_custom_arguments(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with all custom arguments"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "AAPL",
            "MSFT",
            "GOOGL",
            "-l",
            "500.0",
            "-p",
            "0.35",
            "--iv-spread-threshold",
            "0.05",
            "--theta-ratio-threshold",
            "2.0",
            "--front-expiry-max-days",
            "30",
            "--back-expiry-min-days",
            "50",
            "--back-expiry-max-days",
            "90",
            "--min-volume",
            "25",
            "--max-bid-ask-spread",
            "0.10",
            "-q",
            "3",
            "--log",
            "calendar_test.log",
            "--debug",
            "-f",
            "https://finviz.com/screener.ashx?v=111",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.calendar_finder.assert_called_once_with(
            symbol_list=["AAPL", "MSFT", "GOOGL"],
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
            log_file="calendar_test.log",
            debug=True,
            finviz_url="https://finviz.com/screener.ashx?v=111",
        )

    @pytest.mark.integration
    def test_calendar_command_with_no_symbols_should_use_defaults(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with no symbols uses default symbol list"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-l",
            "400.0",
            "-p",
            "0.30",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.calendar_finder.assert_called_once_with(
            symbol_list=None,  # Should trigger default list in calendar_finder
            cost_limit=400.0,
            profit_target=0.30,
            iv_spread_threshold=0.015,
            theta_ratio_threshold=1.5,
            front_expiry_max_days=45,
            back_expiry_min_days=60,
            back_expiry_max_days=120,
            min_volume=10,
            max_bid_ask_spread=0.15,
            quantity=1,
            log_file=None,
            debug=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_calendar_command_with_single_symbol(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with single symbol"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
            "-l",
            "250.0",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.calendar_finder.assert_called_once_with(
            symbol_list=["SPY"],
            cost_limit=250.0,
            profit_target=0.25,
            iv_spread_threshold=0.015,
            theta_ratio_threshold=1.5,
            front_expiry_max_days=45,
            back_expiry_min_days=60,
            back_expiry_max_days=120,
            min_volume=10,
            max_bid_ask_spread=0.15,
            quantity=1,
            log_file=None,
            debug=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_calendar_command_with_special_symbols(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with special symbols (futures, indices)"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "!MES",
            "@SPX",
            "SPY.B",
            "-l",
            "600.0",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.calendar_finder.assert_called_once_with(
            symbol_list=["!MES", "@SPX", "SPY.B"],
            cost_limit=600.0,
            profit_target=0.25,
            iv_spread_threshold=0.015,
            theta_ratio_threshold=1.5,
            front_expiry_max_days=45,
            back_expiry_min_days=60,
            back_expiry_max_days=120,
            min_volume=10,
            max_bid_ask_spread=0.15,
            quantity=1,
            log_file=None,
            debug=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_calendar_command_with_extreme_values(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with extreme parameter values"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
            "-l",
            "0.01",  # Very low cost limit
            "-p",
            "0.99",  # Very high profit target
            "--iv-spread-threshold",
            "0.001",  # Very low IV spread
            "--theta-ratio-threshold",
            "0.1",  # Very low theta ratio
            "--front-expiry-max-days",
            "1",  # Very short front expiry
            "--back-expiry-min-days",
            "1",  # Very short back expiry min
            "--back-expiry-max-days",
            "365",  # Very long back expiry max
            "--min-volume",
            "1",  # Very low volume
            "--max-bid-ask-spread",
            "0.99",  # Very high spread
            "-q",
            "100",  # High quantity
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.calendar_finder.assert_called_once_with(
            symbol_list=["SPY"],
            cost_limit=0.01,
            profit_target=0.99,
            iv_spread_threshold=0.001,
            theta_ratio_threshold=0.1,
            front_expiry_max_days=1,
            back_expiry_min_days=1,
            back_expiry_max_days=365,
            min_volume=1,
            max_bid_ask_spread=0.99,
            quantity=100,
            log_file=None,
            debug=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_calendar_command_with_negative_values(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with negative parameter values"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
            "-l",
            "-100.0",
            "-p",
            "-0.5",
            "--iv-spread-threshold",
            "-0.05",
            "--theta-ratio-threshold",
            "-1.0",
            "--front-expiry-max-days",
            "-10",
            "--back-expiry-min-days",
            "-5",
            "--back-expiry-max-days",
            "-1",
            "--min-volume",
            "-50",
            "--max-bid-ask-spread",
            "-0.20",
            "-q",
            "-5",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.calendar_finder.assert_called_once_with(
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
            log_file=None,
            debug=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_calendar_command_with_zero_values(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with zero parameter values"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
            "-l",
            "0.0",
            "-p",
            "0.0",
            "--iv-spread-threshold",
            "0.0",
            "--theta-ratio-threshold",
            "0.0",
            "--front-expiry-max-days",
            "0",
            "--back-expiry-min-days",
            "0",
            "--back-expiry-max-days",
            "0",
            "--min-volume",
            "0",
            "--max-bid-ask-spread",
            "0.0",
            "-q",
            "0",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.calendar_finder.assert_called_once_with(
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
            log_file=None,
            debug=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_calendar_command_with_large_values(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with large parameter values"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
            "-l",
            "99999.99",
            "-p",
            "999.99",
            "--iv-spread-threshold",
            "10.0",
            "--theta-ratio-threshold",
            "100.0",
            "--front-expiry-max-days",
            "9999",
            "--back-expiry-min-days",
            "5000",
            "--back-expiry-max-days",
            "10000",
            "--min-volume",
            "1000000",
            "--max-bid-ask-spread",
            "100.0",
            "-q",
            "999",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.calendar_finder.assert_called_once_with(
            symbol_list=["SPY"],
            cost_limit=99999.99,
            profit_target=999.99,
            iv_spread_threshold=10.0,
            theta_ratio_threshold=100.0,
            front_expiry_max_days=9999,
            back_expiry_min_days=5000,
            back_expiry_max_days=10000,
            min_volume=1000000,
            max_bid_ask_spread=100.0,
            quantity=999,
            log_file=None,
            debug=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_calendar_command_with_warning_logging(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with warning logging enabled"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
            "--warning",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.calendar_finder.assert_called_once()
        args, kwargs = mock_option_scan.calendar_finder.call_args
        assert kwargs["debug"] is False  # debug flag should be False even with warning

    @pytest.mark.integration
    def test_calendar_command_with_debug_logging(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with debug logging enabled"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
            "--debug",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.calendar_finder.assert_called_once()
        args, kwargs = mock_option_scan.calendar_finder.call_args
        assert kwargs["debug"] is True

    @pytest.mark.integration
    def test_calendar_command_with_both_debug_and_warning(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with both debug and warning flags (debug should take precedence)"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
            "--debug",
            "--warning",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.calendar_finder.assert_called_once()
        args, kwargs = mock_option_scan.calendar_finder.call_args
        assert kwargs["debug"] is True  # debug should take precedence

    @pytest.mark.integration
    def test_calendar_command_with_log_file(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with log file specified"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
            "--log",
            "calendar_trading.log",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.calendar_finder.assert_called_once()
        args, kwargs = mock_option_scan.calendar_finder.call_args
        assert kwargs["log_file"] == "calendar_trading.log"

    @pytest.mark.integration
    def test_calendar_command_with_finviz_url(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with finviz URL specified"""
        finviz_url = "https://finviz.com/screener.ashx?v=111&f=cap_midover"
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
            "-f",
            finviz_url,
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.calendar_finder.assert_called_once()
        args, kwargs = mock_option_scan.calendar_finder.call_args
        assert kwargs["finviz_url"] == finviz_url

    @pytest.mark.integration
    def test_calendar_command_help(
        self, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command help display"""
        test_args = ["alchimest.py", "calendar", "--help"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                alchimest.main()
            # argparse exits with code 0 for help
            assert exc_info.value.code == 0

    @pytest.mark.integration
    def test_calendar_command_with_many_symbols(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with many symbols"""
        many_symbols = [
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
        ]
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
        ] + many_symbols

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.calendar_finder.assert_called_once()
        args, kwargs = mock_option_scan.calendar_finder.call_args
        assert kwargs["symbol_list"] == many_symbols

    @pytest.mark.integration
    def test_calendar_command_with_decimal_precision_values(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with high precision decimal values"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
            "-l",
            "123.456789",
            "-p",
            "0.123456",
            "--iv-spread-threshold",
            "0.012345",
            "--theta-ratio-threshold",
            "1.234567",
            "--max-bid-ask-spread",
            "0.098765",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.calendar_finder.assert_called_once_with(
            symbol_list=["SPY"],
            cost_limit=123.456789,
            profit_target=0.123456,
            iv_spread_threshold=0.012345,
            theta_ratio_threshold=1.234567,
            front_expiry_max_days=45,
            back_expiry_min_days=60,
            back_expiry_max_days=120,
            min_volume=10,
            max_bid_ask_spread=0.098765,
            quantity=1,
            log_file=None,
            debug=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_calendar_command_short_form_arguments(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with short form arguments"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
            "QQQ",
            "-l",
            "400.0",
            "-p",
            "0.30",
            "-q",
            "2",
            "-f",
            "https://finviz.com/screener.ashx?v=111",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.calendar_finder.assert_called_once_with(
            symbol_list=["SPY", "QQQ"],
            cost_limit=400.0,
            profit_target=0.30,
            iv_spread_threshold=0.015,
            theta_ratio_threshold=1.5,
            front_expiry_max_days=45,
            back_expiry_min_days=60,
            back_expiry_max_days=120,
            min_volume=10,
            max_bid_ask_spread=0.15,
            quantity=2,
            log_file=None,
            debug=False,
            finviz_url="https://finviz.com/screener.ashx?v=111",
        )

    @pytest.mark.integration
    def test_calendar_command_long_form_arguments(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with long form arguments"""
        test_args = [
            "alchimest.py",
            "calendar",
            "--symbols",
            "AAPL",
            "MSFT",
            "--cost-limit",
            "750.0",
            "--profit-target",
            "0.40",
            "--quantity",
            "5",
            "--fin",
            "https://finviz.com/screener.ashx?v=222",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.calendar_finder.assert_called_once_with(
            symbol_list=["AAPL", "MSFT"],
            cost_limit=750.0,
            profit_target=0.40,
            iv_spread_threshold=0.015,
            theta_ratio_threshold=1.5,
            front_expiry_max_days=45,
            back_expiry_min_days=60,
            back_expiry_max_days=120,
            min_volume=10,
            max_bid_ask_spread=0.15,
            quantity=5,
            log_file=None,
            debug=False,
            finviz_url="https://finviz.com/screener.ashx?v=222",
        )


class TestCalendarCLIEdgeCases:
    """Test edge cases and error conditions for calendar CLI"""

    @pytest.fixture
    def capture_output(self) -> Generator[Tuple[StringIO, StringIO], None, None]:
        """Capture stdout and stderr"""
        old_out, old_err = sys.stdout, sys.stderr
        out, err = StringIO(), StringIO()
        sys.stdout, sys.stderr = out, err
        yield out, err
        sys.stdout, sys.stderr = old_out, old_err

    @pytest.mark.integration
    def test_calendar_command_invalid_float_argument(
        self, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with invalid float argument"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
            "-l",
            "not_a_number",
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                alchimest.main()
            # argparse uses exit code 2 for CLI errors
            assert exc_info.value.code == 2

    @pytest.mark.integration
    def test_calendar_command_invalid_int_argument(
        self, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with invalid integer argument"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
            "--front-expiry-max-days",
            "not_an_integer",
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                alchimest.main()
            assert exc_info.value.code == 2

    @pytest.mark.integration
    def test_calendar_command_unknown_argument(
        self, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with unknown argument"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
            "--unknown-parameter",
            "value",
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                alchimest.main()
            assert exc_info.value.code == 2

    @pytest.mark.integration
    def test_calendar_command_missing_required_argument_value(
        self, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with missing required argument value"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",  # Missing symbol list
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                alchimest.main()
            assert exc_info.value.code == 2

    @pytest.mark.integration
    def test_calendar_command_empty_symbol_argument(
        self, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test calendar command with empty symbol argument"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "",  # Empty symbol
        ]

        with patch.object(sys, "argv", test_args):
            with patch("alchimest.OptionScan") as mock_option_scan:
                mock_instance = MagicMock()
                mock_option_scan.return_value = mock_instance

                alchimest.main()

                # Should still call calendar_finder with empty string in list
                mock_instance.calendar_finder.assert_called_once()
                args, kwargs = mock_instance.calendar_finder.call_args
                assert kwargs["symbol_list"] == [""]


class TestCalendarCLIIntegration:
    """Integration tests for calendar CLI command"""

    @pytest.fixture
    def mock_option_scan(self) -> Generator[MagicMock, None, None]:
        """Mock OptionScan class to avoid actual IB connections"""
        with patch("alchimest.OptionScan") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.mark.integration
    def test_calendar_command_logging_configuration(
        self, mock_option_scan: MagicMock
    ) -> None:
        """Test that calendar command properly configures logging"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
            "--debug",
            "--log",
            "test.log",
        ]

        with patch("alchimest.configure_logging") as mock_configure_logging:
            with patch.object(sys, "argv", test_args):
                alchimest.main()

            # Verify logging was configured with correct parameters
            mock_configure_logging.assert_called_once_with(
                debug=True, warning=False, log_file="test.log"
            )

    @pytest.mark.integration
    def test_calendar_command_version_and_welcome_integration(
        self, mock_option_scan: MagicMock
    ) -> None:
        """Test that calendar command integrates with version and welcome display"""
        test_args = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
        ]

        with patch("alchimest.print_welcome") as mock_print_welcome:
            with patch.object(sys, "argv", test_args):
                alchimest.main()

            # Verify welcome message was printed with correct version
            mock_print_welcome.assert_called_once()
            args, kwargs = mock_print_welcome.call_args
            # Should include version and default profit
            assert alchimest.__version__ in str(args) or alchimist.__version__ in str(
                kwargs
            )

    @pytest.mark.integration
    def test_calendar_command_argument_parsing_order_independence(
        self, mock_option_scan: MagicMock
    ) -> None:
        """Test that calendar command argument order doesn't matter"""
        # Test with different argument orders
        test_args_1 = [
            "alchimest.py",
            "calendar",
            "-s",
            "SPY",
            "-l",
            "300",
            "-p",
            "0.25",
        ]
        test_args_2 = [
            "alchimest.py",
            "calendar",
            "-p",
            "0.25",
            "-s",
            "SPY",
            "-l",
            "300",
        ]
        test_args_3 = [
            "alchimest.py",
            "calendar",
            "-l",
            "300",
            "-p",
            "0.25",
            "-s",
            "SPY",
        ]

        expected_call = {
            "symbol_list": ["SPY"],
            "cost_limit": 300.0,
            "profit_target": 0.25,
            "iv_spread_threshold": 0.03,
            "theta_ratio_threshold": 1.5,
            "front_expiry_max_days": 45,
            "back_expiry_min_days": 60,
            "back_expiry_max_days": 120,
            "min_volume": 10,
            "max_bid_ask_spread": 0.15,
            "quantity": 1,
            "log_file": None,
            "debug": False,
            "finviz_url": None,
        }

        for test_args in [test_args_1, test_args_2, test_args_3]:
            mock_option_scan.reset_mock()

            with patch.object(sys, "argv", test_args):
                alchimest.main()

            mock_option_scan.calendar_finder.assert_called_once_with(**expected_call)

    @pytest.mark.integration
    def test_calendar_command_multiple_executions(
        self, mock_option_scan: MagicMock
    ) -> None:
        """Test that calendar command can be executed multiple times"""
        test_configs = [
            (["SPY"], 300.0, 0.25),
            (["QQQ"], 400.0, 0.30),
            (["AAPL", "MSFT"], 500.0, 0.35),
        ]

        for symbols, cost_limit, profit_target in test_configs:
            mock_option_scan.reset_mock()

            test_args = (
                [
                    "alchimist.py",
                    "calendar",
                    "-s",
                ]
                + symbols
                + [
                    "-l",
                    str(cost_limit),
                    "-p",
                    str(profit_target),
                ]
            )

            with patch.object(sys, "argv", test_args):
                alchimest.main()

            mock_option_scan.calendar_finder.assert_called_once()
            args, kwargs = mock_option_scan.calendar_finder.call_args
            assert kwargs["symbol_list"] == symbols
            assert kwargs["cost_limit"] == cost_limit
            assert kwargs["profit_target"] == profit_target
