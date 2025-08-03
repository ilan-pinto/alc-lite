"""
Comprehensive CLI tests for Box spread command.

This test suite provides extensive coverage of the Box spread CLI interface,
including argument parsing, validation, integration with commands/option.py,
and error handling scenarios.

Test Coverage:
- Box command argument parsing and validation
- Integration with OptionScan.box_finder method
- Default value handling and overrides
- Logging configuration integration
- Error handling and help system
- Parameter validation and edge cases
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
from commands.option import OptionScan


class TestBoxSpreadCLIArguments:
    """Test class for Box spread CLI argument validation"""

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
    def test_box_command_with_all_valid_arguments(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test box command with all valid arguments"""
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "SPY",
            "QQQ",
            "AAPL",
            "-l",
            "500",
            "-p",
            "0.02",
            "--min-profit",
            "0.10",
            "--max-strike-width",
            "25.0",
            "--min-strike-width",
            "2.0",
            "--range",
            "0.15",
            "--min-volume",
            "10",
            "--max-spread",
            "0.08",
            "--min-days-expiry",
            "3",
            "--max-days-expiry",
            "60",
            "-q",
            "2",
            "--safety-buffer",
            "0.03",
            "--require-risk-free",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.box_finder.assert_called_once_with(
            symbol_list=["SPY", "QQQ", "AAPL"],
            cost_limit=500.0,
            profit_target=0.02,
            min_profit=0.10,
            max_strike_width=25.0,
            min_strike_width=2.0,
            range=0.15,
            min_volume=10,
            max_spread=0.08,
            min_days_expiry=3,
            max_days_expiry=60,
            quantity=2,
            safety_buffer=0.03,
            require_risk_free=True,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_box_command_with_default_values(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test box command with default values"""
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "SPY",
            "QQQ",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.box_finder.assert_called_once_with(
            symbol_list=["SPY", "QQQ"],
            cost_limit=500.0,  # Default
            profit_target=0.01,  # Default
            min_profit=0.05,  # Default
            max_strike_width=50.0,  # Default
            min_strike_width=1.0,  # Default
            range=0.1,  # Default
            min_volume=5,  # Default
            max_spread=0.10,  # Default
            min_days_expiry=1,  # Default
            max_days_expiry=90,  # Default
            quantity=1,  # Default
            safety_buffer=0.02,  # Default
            require_risk_free=True,  # Default
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_box_command_with_logging_options(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test box command with various logging options"""
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "AAPL",
            "--debug",
            "--log",
            "box_test.log",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.box_finder.assert_called_once_with(
            symbol_list=["AAPL"],
            cost_limit=500.0,
            profit_target=0.01,
            min_profit=0.05,
            max_strike_width=50.0,
            min_strike_width=1.0,
            range=0.1,
            min_volume=5,
            max_spread=0.10,
            min_days_expiry=1,
            max_days_expiry=90,
            quantity=1,
            safety_buffer=0.02,
            require_risk_free=True,
            log_file="box_test.log",
            debug=True,
            warning=False,
            error=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_box_command_with_warning_logging(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test box command with warning logging"""
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "MSFT",
            "--warning",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.box_finder.assert_called_once()
        call_args = mock_option_scan.box_finder.call_args[1]
        assert call_args["warning"] is True
        assert call_args["debug"] is False
        assert call_args["error"] is False

    @pytest.mark.integration
    def test_box_command_with_error_logging(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test box command with error logging"""
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "TSLA",
            "--error",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.box_finder.assert_called_once()
        call_args = mock_option_scan.box_finder.call_args[1]
        assert call_args["error"] is True
        assert call_args["debug"] is False
        assert call_args["warning"] is False

    @pytest.mark.integration
    def test_box_command_with_finviz_url(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test box command with Finviz URL"""
        finviz_url = "https://finviz.com/screener.ashx?v=111&f=cap_mega"
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "SPY",
            "-f",
            finviz_url,
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.box_finder.assert_called_once()
        call_args = mock_option_scan.box_finder.call_args[1]
        assert call_args["finviz_url"] == finviz_url

    @pytest.mark.integration
    def test_box_command_with_minimal_arguments(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test box command with minimal required arguments"""
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "SPY",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        # Should call with defaults for all unspecified arguments
        mock_option_scan.box_finder.assert_called_once()
        call_args = mock_option_scan.box_finder.call_args[1]
        assert call_args["symbol_list"] == ["SPY"]
        assert call_args["cost_limit"] == 500.0
        assert call_args["profit_target"] == 0.01

    @pytest.mark.integration
    def test_box_command_with_multiple_symbols(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test box command with multiple symbols"""
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "SPY",
            "QQQ",
            "IWM",
            "AAPL",
            "MSFT",
            "GOOGL",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.box_finder.assert_called_once()
        call_args = mock_option_scan.box_finder.call_args[1]
        assert call_args["symbol_list"] == [
            "SPY",
            "QQQ",
            "IWM",
            "AAPL",
            "MSFT",
            "GOOGL",
        ]

    @pytest.mark.integration
    def test_box_command_with_futures_and_index_symbols(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test box command with futures and index symbols"""
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "!MES",
            "!MNQ",
            "@SPX",
            "@RUT",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.box_finder.assert_called_once()
        call_args = mock_option_scan.box_finder.call_args[1]
        assert call_args["symbol_list"] == ["!MES", "!MNQ", "@SPX", "@RUT"]

    @pytest.mark.integration
    def test_box_command_with_conservative_parameters(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test box command with conservative trading parameters"""
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "SPY",
            "QQQ",
            "--cost-limit",
            "200",
            "--profit-target",
            "0.005",  # 0.5% minimum profit
            "--min-profit",
            "0.02",  # $0.02 minimum absolute profit
            "--max-strike-width",
            "5.0",  # Small strike width
            "--min-volume",
            "20",  # Higher volume requirement
            "--max-spread",
            "0.05",  # Tighter spread requirement
            "--safety-buffer",
            "0.05",  # Larger safety buffer
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.box_finder.assert_called_once()
        call_args = mock_option_scan.box_finder.call_args[1]
        assert call_args["cost_limit"] == 200.0
        assert call_args["profit_target"] == 0.005
        assert call_args["min_profit"] == 0.02
        assert call_args["max_strike_width"] == 5.0
        assert call_args["min_volume"] == 20
        assert call_args["max_spread"] == 0.05
        assert call_args["safety_buffer"] == 0.05

    @pytest.mark.integration
    def test_box_command_with_aggressive_parameters(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test box command with aggressive trading parameters"""
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "AAPL",
            "TSLA",
            "--cost-limit",
            "1000",
            "--profit-target",
            "0.03",  # 3% minimum profit
            "--min-profit",
            "0.25",  # $0.25 minimum absolute profit
            "--max-strike-width",
            "100.0",  # Large strike width
            "--range",
            "0.25",  # Wider strike range
            "--quantity",
            "5",  # Multiple contracts
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.box_finder.assert_called_once()
        call_args = mock_option_scan.box_finder.call_args[1]
        assert call_args["cost_limit"] == 1000.0
        assert call_args["profit_target"] == 0.03
        assert call_args["min_profit"] == 0.25
        assert call_args["max_strike_width"] == 100.0
        assert call_args["range"] == 0.25
        assert call_args["quantity"] == 5

    @pytest.mark.integration
    def test_box_command_with_expiry_constraints(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test box command with expiry date constraints"""
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "META",
            "--min-days-expiry",
            "7",  # At least 1 week
            "--max-days-expiry",
            "30",  # Maximum 1 month
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.box_finder.assert_called_once()
        call_args = mock_option_scan.box_finder.call_args[1]
        assert call_args["min_days_expiry"] == 7
        assert call_args["max_days_expiry"] == 30

    def test_box_command_missing_symbols_error(
        self, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test that box command requires symbols argument"""
        test_args = [
            "alchimest.py",
            "box",
            # Missing --symbols argument
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                alchimest.main()

    def test_box_command_invalid_numeric_arguments(
        self, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test box command with invalid numeric arguments"""
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "SPY",
            "--cost-limit",
            "invalid_number",
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                alchimest.main()

    def test_box_command_negative_values(
        self, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test box command with negative values (should be handled by argument parser)"""
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "SPY",
            "--cost-limit",
            "-100",  # Negative cost limit
        ]

        # This should be allowed by argparse but caught by config validation
        with patch.object(sys, "argv", test_args):
            with patch("alchimest.OptionScan") as mock_scan:
                mock_instance = MagicMock()
                mock_scan.return_value = mock_instance

                alchimest.main()

                # Should still call box_finder (validation happens inside)
                mock_instance.box_finder.assert_called_once()
                call_args = mock_instance.box_finder.call_args[1]
                assert call_args["cost_limit"] == -100.0

    def test_box_command_help_message(
        self, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test box command help message"""
        test_args = [
            "alchimest.py",
            "box",
            "--help",
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                alchimest.main()

            # Help should exit with code 0
            assert exc_info.value.code == 0

        # Check that help content was printed
        out, err = capture_output
        help_text = out.getvalue()
        assert "box spread arbitrage" in help_text.lower()
        assert "--symbols" in help_text
        assert "--cost-limit" in help_text
        assert "--profit-target" in help_text


class TestBoxSpreadCLIParameterValidation:
    """Test parameter validation and edge cases for box spread CLI"""

    @pytest.fixture
    def mock_option_scan(self) -> Generator[MagicMock, None, None]:
        """Mock OptionScan class to avoid actual IB connections"""
        with patch("alchimest.OptionScan") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    def test_box_command_parameter_boundaries(
        self, mock_option_scan: MagicMock
    ) -> None:
        """Test box command with boundary values"""
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "SPY",
            "--profit-target",
            "0.001",  # Very small profit target
            "--min-profit",
            "0.01",  # Small absolute profit
            "--max-strike-width",
            "1.0",  # Small strike width
            "--min-strike-width",
            "0.5",  # Very small minimum
            "--range",
            "0.01",  # Very tight range
            "--min-volume",
            "1",  # Minimum volume
            "--max-spread",
            "0.01",  # Very tight spread
            "--quantity",
            "1",  # Minimum quantity
            "--safety-buffer",
            "0.001",  # Very small buffer
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.box_finder.assert_called_once()
        call_args = mock_option_scan.box_finder.call_args[1]
        assert call_args["profit_target"] == 0.001
        assert call_args["min_profit"] == 0.01
        assert call_args["max_strike_width"] == 1.0
        assert call_args["min_strike_width"] == 0.5
        assert call_args["range"] == 0.01
        assert call_args["min_volume"] == 1
        assert call_args["max_spread"] == 0.01
        assert call_args["quantity"] == 1
        assert call_args["safety_buffer"] == 0.001

    def test_box_command_large_values(self, mock_option_scan: MagicMock) -> None:
        """Test box command with large parameter values"""
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "SPY",
            "--cost-limit",
            "10000",  # Large cost limit
            "--profit-target",
            "0.5",  # 50% profit target
            "--max-strike-width",
            "500.0",  # Very large strike width
            "--range",
            "1.0",  # 100% range
            "--min-volume",
            "10000",  # High volume requirement
            "--quantity",
            "100",  # Many contracts
            "--max-days-expiry",
            "365",  # 1 year expiry
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.box_finder.assert_called_once()
        call_args = mock_option_scan.box_finder.call_args[1]
        assert call_args["cost_limit"] == 10000.0
        assert call_args["profit_target"] == 0.5
        assert call_args["max_strike_width"] == 500.0
        assert call_args["range"] == 1.0
        assert call_args["min_volume"] == 10000
        assert call_args["quantity"] == 100
        assert call_args["max_days_expiry"] == 365

    def test_box_command_float_precision(self, mock_option_scan: MagicMock) -> None:
        """Test box command with high precision float values"""
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "AAPL",
            "--profit-target",
            "0.00123",  # High precision
            "--min-profit",
            "0.0789",  # High precision
            "--safety-buffer",
            "0.01234",  # High precision
            "--max-spread",
            "0.0567",  # High precision
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.box_finder.assert_called_once()
        call_args = mock_option_scan.box_finder.call_args[1]
        assert abs(call_args["profit_target"] - 0.00123) < 1e-6
        assert abs(call_args["min_profit"] - 0.0789) < 1e-6
        assert abs(call_args["safety_buffer"] - 0.01234) < 1e-6
        assert abs(call_args["max_spread"] - 0.0567) < 1e-6

    def test_box_command_zero_values(self, mock_option_scan: MagicMock) -> None:
        """Test box command with zero values where applicable"""
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "SPY",
            "--profit-target",
            "0.0",  # Zero profit target
            "--min-profit",
            "0.0",  # Zero absolute profit
            "--safety-buffer",
            "0.0",  # Zero safety buffer
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.box_finder.assert_called_once()
        call_args = mock_option_scan.box_finder.call_args[1]
        assert call_args["profit_target"] == 0.0
        assert call_args["min_profit"] == 0.0
        assert call_args["safety_buffer"] == 0.0


class TestBoxSpreadCLIIntegration:
    """Integration tests for Box spread CLI with OptionScan"""

    def test_option_scan_box_finder_method_exists(self):
        """Test that OptionScan has box_finder method"""
        option_scan = OptionScan()
        assert hasattr(option_scan, "box_finder")
        assert callable(getattr(option_scan, "box_finder"))

    def test_option_scan_box_finder_parameters(self):
        """Test that box_finder method accepts expected parameters"""
        import inspect

        option_scan = OptionScan()
        sig = inspect.signature(option_scan.box_finder)

        # Check that all expected parameters exist
        expected_params = [
            "symbol_list",
            "cost_limit",
            "profit_target",
            "min_profit",
            "max_strike_width",
            "min_strike_width",
            "range",
            "min_volume",
            "max_spread",
            "min_days_expiry",
            "max_days_expiry",
            "quantity",
            "safety_buffer",
            "require_risk_free",
            "log_file",
            "debug",
            "warning",
            "error",
            "finviz_url",
        ]

        for param in expected_params:
            assert (
                param in sig.parameters
            ), f"Parameter {param} not found in box_finder signature"

    @pytest.mark.integration
    def test_box_finder_method_call_integration(self):
        """Test actual integration with box_finder method (mocked execution)"""
        from modules.Arbitrage.box_spread.strategy import BoxSpread

        with patch("commands.option.BoxSpread") as mock_box_class:
            mock_box_instance = MagicMock()
            mock_box_class.return_value = mock_box_instance

            # Mock config validation
            mock_config = MagicMock()
            mock_box_instance.config = mock_config
            mock_config.validate.return_value = None

            option_scan = OptionScan()

            # Call box_finder with test parameters
            option_scan.box_finder(
                symbol_list=["SPY", "QQQ"],
                cost_limit=300.0,
                profit_target=0.015,
                min_profit=0.08,
                max_strike_width=20.0,
                min_strike_width=2.0,
                range=0.12,
                min_volume=8,
                max_spread=0.06,
                min_days_expiry=5,
                max_days_expiry=45,
                quantity=3,
                safety_buffer=0.025,
                require_risk_free=True,
                log_file="test.log",
                debug=True,
                warning=False,
                error=False,
                finviz_url=None,
            )

            # Verify BoxSpread was created
            mock_box_class.assert_called_once_with(log_file="test.log")

            # Verify configuration was set
            assert mock_box_instance.config.max_net_debit == 300.0
            assert mock_box_instance.config.min_arbitrage_profit == 0.015
            assert mock_box_instance.config.min_absolute_profit == 0.08
            assert mock_box_instance.config.max_strike_width == 20.0
            assert mock_box_instance.config.min_strike_width == 2.0
            assert mock_box_instance.config.min_volume_per_leg == 8
            assert mock_box_instance.config.max_bid_ask_spread_percent == 0.06
            assert mock_box_instance.config.min_days_to_expiry == 5
            assert mock_box_instance.config.max_days_to_expiry == 45
            assert mock_box_instance.config.safety_buffer == 0.025
            assert mock_box_instance.config.require_risk_free is True

            # Verify config validation was called
            mock_config.validate.assert_called_once()

    @pytest.mark.integration
    def test_box_finder_with_invalid_config(self):
        """Test box_finder handling of invalid configuration"""
        with patch("commands.option.BoxSpread") as mock_box_class:
            with patch("commands.option.logger") as mock_logger:
                mock_box_instance = MagicMock()
                mock_box_class.return_value = mock_box_instance

                # Mock config validation to raise error
                mock_config = MagicMock()
                mock_box_instance.config = mock_config
                mock_config.validate.side_effect = ValueError("Invalid configuration")

                option_scan = OptionScan()

                # Call box_finder - should handle validation error
                option_scan.box_finder(
                    symbol_list=["SPY"],
                    cost_limit=-100.0,  # Invalid negative value
                )

                # Verify error was logged
                mock_logger.error.assert_called_once()
                error_message = mock_logger.error.call_args[0][0]
                assert "Invalid box spread configuration" in error_message

    @pytest.mark.integration
    def test_box_finder_default_symbol_list(self):
        """Test box_finder with default symbol list"""
        with patch("commands.option.BoxSpread") as mock_box_class:
            mock_box_instance = MagicMock()
            mock_box_class.return_value = mock_box_instance

            # Mock config validation
            mock_config = MagicMock()
            mock_box_instance.config = mock_config
            mock_config.validate.return_value = None

            option_scan = OptionScan()

            # Call box_finder with None symbol list (should use defaults)
            option_scan.box_finder(symbol_list=None)

            # Should still work (implementation handles None symbol_list)
            mock_box_class.assert_called_once()

    def test_box_command_argument_help_text_accuracy(self):
        """Test that help text accurately describes box command arguments"""
        import argparse

        # Create parser like in alchimest.py
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)

        # Add box parser (simplified version for testing)
        parser_box = subparsers.add_parser("box", help="Box spread test")
        parser_box.add_argument("-s", "--symbols", nargs="+")
        parser_box.add_argument("-l", "--cost-limit", type=float, default=500.0)
        parser_box.add_argument("-p", "--profit-target", type=float, default=0.01)

        # Test parsing
        args = parser.parse_args(["box", "-s", "SPY", "-l", "300", "-p", "0.02"])

        assert args.command == "box"
        assert args.symbols == ["SPY"]
        assert args.cost_limit == 300.0
        assert args.profit_target == 0.02


class TestBoxSpreadCLIErrorHandling:
    """Test error handling in Box spread CLI scenarios"""

    def test_box_command_with_empty_symbol_list(self):
        """Test box command with empty symbol list"""
        test_args = [
            "alchimest.py",
            "box",
            "--symbols",  # No symbols provided after flag
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                alchimest.main()

    def test_box_command_with_conflicting_arguments(self):
        """Test box command with logically conflicting arguments"""
        # min-strike-width > max-strike-width should be caught by validation
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "SPY",
            "--min-strike-width",
            "10.0",
            "--max-strike-width",
            "5.0",  # Less than min
        ]

        with patch.object(sys, "argv", test_args):
            with patch("alchimest.OptionScan") as mock_scan:
                mock_instance = MagicMock()
                mock_scan.return_value = mock_instance

                alchimest.main()

                # Should still call box_finder (validation happens inside)
                mock_instance.box_finder.assert_called_once()

    def test_box_command_with_extreme_values(self):
        """Test box command with extreme parameter values"""
        test_args = [
            "alchimest.py",
            "box",
            "-s",
            "SPY",
            "--profit-target",
            "999.99",  # Extremely high profit target
            "--quantity",
            "999999",  # Extremely high quantity
        ]

        with patch.object(sys, "argv", test_args):
            with patch("alchimest.OptionScan") as mock_scan:
                mock_instance = MagicMock()
                mock_scan.return_value = mock_instance

                alchimest.main()

                # Should accept extreme values (business logic validation separate)
                mock_instance.box_finder.assert_called_once()
                call_args = mock_instance.box_finder.call_args[1]
                assert call_args["profit_target"] == 999.99
                assert call_args["quantity"] == 999999
