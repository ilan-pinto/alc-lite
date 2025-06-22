"""
Integration tests for CLI argument validation in alchimest.py
Tests all argument combinations and edge cases for sfr and syn commands
"""

import os
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

# Add the project root to the path so we can import alchimest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alchimest


class TestCLIArguments:
    """Test class for CLI argument validation"""

    @pytest.fixture
    def mock_option_scan(self):
        """Mock OptionScan class to avoid actual IB connections"""
        with patch("alchimest.OptionScan") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def capture_output(self):
        """Capture stdout and stderr"""
        old_out, old_err = sys.stdout, sys.stderr
        out, err = StringIO(), StringIO()
        sys.stdout, sys.stderr = out, err
        yield out, err
        sys.stdout, sys.stderr = old_out, old_err

    @pytest.mark.integration
    def test_sfr_command_with_all_valid_arguments(
        self, mock_option_scan, capture_output
    ):
        """Test sfr command with all valid arguments"""
        test_args = [
            "alchimest.py",
            "sfr",
            "-s",
            "SPY",
            "QQQ",
            "META",
            "-p",
            "1.5",
            "-l",
            "100",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=["SPY", "QQQ", "META"], profit_target=1.5, cost_limit=100
        )

    @pytest.mark.integration
    def test_sfr_command_with_default_profit(self, mock_option_scan, capture_output):
        """Test sfr command with default profit value"""
        test_args = [
            "alchimest.py",
            "sfr",
            "-s",
            "SPY",
            "QQQ",
            "-l",
            "150",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=["SPY", "QQQ"], profit_target=None, cost_limit=150
        )

    @pytest.mark.integration
    def test_sfr_command_with_default_cost_limit(
        self, mock_option_scan, capture_output
    ):
        """Test sfr command with default cost limit"""
        test_args = ["alchimest.py", "sfr", "-s", "SPY", "QQQ", "-p", "2.0"]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=["SPY", "QQQ"],
            profit_target=2.0,
            cost_limit=120,  # Default value
        )

    @pytest.mark.integration
    def test_sfr_command_with_no_symbols(self, mock_option_scan, capture_output):
        """Test sfr command with no symbols (should use default list)"""
        test_args = ["alchimest.py", "sfr", "-p", "1.0", "-l", "200"]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=None, profit_target=1.0, cost_limit=200
        )

    @pytest.mark.integration
    def test_syn_command_with_all_valid_arguments(
        self, mock_option_scan, capture_output
    ):
        """Test syn command with all valid arguments"""
        test_args = [
            "alchimest.py",
            "syn",
            "-s",
            "SPY",
            "QQQ",
            "META",
            "-l",
            "100",
            "-ml",
            "50",
            "-mp",
            "200",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["SPY", "QQQ", "META"],
            cost_limit=100,
            max_loss=50,
            max_profit=200,
        )

    @pytest.mark.integration
    def test_syn_command_with_default_cost_limit(
        self, mock_option_scan, capture_output
    ):
        """Test syn command with default cost limit"""
        test_args = [
            "alchimest.py",
            "syn",
            "-s",
            "SPY",
            "QQQ",
            "-ml",
            "30",
            "-mp",
            "150",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["SPY", "QQQ"],
            cost_limit=120,  # Default value
            max_loss=30,
            max_profit=150,
        )

    @pytest.mark.integration
    def test_syn_command_with_no_symbols(self, mock_option_scan, capture_output):
        """Test syn command with no symbols (should use default list)"""
        test_args = ["alchimest.py", "syn", "-l", "80", "-ml", "25"]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=None, cost_limit=80, max_loss=25, max_profit=None
        )

    @pytest.mark.integration
    def test_syn_command_with_only_max_profit(self, mock_option_scan, capture_output):
        """Test syn command with only max_profit specified"""
        test_args = ["alchimest.py", "syn", "-s", "SPY", "-mp", "300"]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["SPY"],
            cost_limit=120,  # Default value
            max_loss=None,
            max_profit=300,
        )

    @pytest.mark.integration
    def test_syn_command_with_only_max_loss(self, mock_option_scan, capture_output):
        """Test syn command with only max_loss specified"""
        test_args = ["alchimest.py", "syn", "-s", "SPY", "QQQ", "-ml", "40"]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["SPY", "QQQ"],
            cost_limit=120,  # Default value
            max_loss=40,
            max_profit=None,
        )

    @pytest.mark.integration
    def test_syn_command_with_no_optional_args(self, mock_option_scan, capture_output):
        """Test syn command with no optional arguments"""
        test_args = ["alchimest.py", "syn", "-s", "SPY"]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["SPY"],
            cost_limit=120,  # Default value
            max_loss=None,
            max_profit=None,
        )

    @pytest.mark.integration
    def test_invalid_command(self, capture_output):
        """Test invalid command handling"""
        test_args = ["alchimest.py", "invalid_command"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as excinfo:
                alchimest.main()
            assert excinfo.value.code == 2  # argparse uses exit code 2 for CLI errors

    @pytest.mark.integration
    def test_no_command_provided(self, capture_output):
        """Test when no command is provided"""
        test_args = ["alchimest.py"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                alchimest.main()

    @pytest.mark.integration
    def test_sfr_command_help(self, capture_output):
        """Test sfr command help"""
        test_args = ["alchimest.py", "sfr", "--help"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                alchimest.main()

    @pytest.mark.integration
    def test_syn_command_help(self, capture_output):
        """Test syn command help"""
        test_args = ["alchimest.py", "syn", "--help"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                alchimest.main()

    @pytest.mark.integration
    def test_main_help(self, capture_output):
        """Test main help"""
        test_args = ["alchimest.py", "--help"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                alchimest.main()

    @pytest.mark.integration
    def test_sfr_command_with_special_symbols(self, mock_option_scan, capture_output):
        """Test sfr command with special symbols like futures and indices"""
        test_args = [
            "alchimest.py",
            "sfr",
            "-s",
            "!MES",
            "@SPX",
            "SPY",
            "-p",
            "0.8",
            "-l",
            "90",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=["!MES", "@SPX", "SPY"], profit_target=0.8, cost_limit=90
        )

    @pytest.mark.integration
    def test_syn_command_with_special_symbols(self, mock_option_scan, capture_output):
        """Test syn command with special symbols like futures and indices"""
        test_args = [
            "alchimest.py",
            "syn",
            "-s",
            "!MES",
            "@SPX",
            "SPY",
            "-l",
            "75",
            "-ml",
            "20",
            "-mp",
            "180",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["!MES", "@SPX", "SPY"],
            cost_limit=75,
            max_loss=20,
            max_profit=180,
        )

    @pytest.mark.integration
    def test_sfr_command_with_negative_profit(self, mock_option_scan, capture_output):
        """Test sfr command with negative profit value"""
        test_args = ["alchimest.py", "sfr", "-s", "SPY", "-p", "-0.5", "-l", "100"]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=["SPY"], profit_target=-0.5, cost_limit=100
        )

    @pytest.mark.integration
    def test_syn_command_with_negative_values(self, mock_option_scan, capture_output):
        """Test syn command with negative max_loss and max_profit values"""
        test_args = [
            "alchimest.py",
            "syn",
            "-s",
            "SPY",
            "-l",
            "100",
            "-ml",
            "-10",
            "-mp",
            "-50",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["SPY"], cost_limit=100, max_loss=-10, max_profit=-50
        )

    @pytest.mark.integration
    def test_sfr_command_with_zero_values(self, mock_option_scan, capture_output):
        """Test sfr command with zero values"""
        test_args = ["alchimest.py", "sfr", "-s", "SPY", "-p", "0.0", "-l", "0"]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=["SPY"], profit_target=0.0, cost_limit=0
        )

    @pytest.mark.integration
    def test_syn_command_with_zero_values(self, mock_option_scan, capture_output):
        """Test syn command with zero values"""
        test_args = [
            "alchimest.py",
            "syn",
            "-s",
            "SPY",
            "-l",
            "0",
            "-ml",
            "0",
            "-mp",
            "0",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["SPY"], cost_limit=0, max_loss=0, max_profit=0
        )

    @pytest.mark.integration
    def test_sfr_command_with_large_values(self, mock_option_scan, capture_output):
        """Test sfr command with large values"""
        test_args = ["alchimest.py", "sfr", "-s", "SPY", "-p", "999.99", "-l", "9999"]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=["SPY"], profit_target=999.99, cost_limit=9999
        )

    @pytest.mark.integration
    def test_syn_command_with_large_values(self, mock_option_scan, capture_output):
        """Test syn command with large values"""
        test_args = [
            "alchimest.py",
            "syn",
            "-s",
            "SPY",
            "-l",
            "9999",
            "-ml",
            "5000",
            "-mp",
            "10000",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["SPY"], cost_limit=9999, max_loss=5000, max_profit=10000
        )
