"""
Integration tests for CLI argument validation in alchimest.py
Tests all argument combinations and edge cases for sfr and syn commands
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


class TestCLIArguments:
    """Test class for CLI argument validation"""

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
    def test_sfr_command_with_all_valid_arguments(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
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
            "-q",
            "5",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=["SPY", "QQQ", "META"],
            profit_target=1.5,
            cost_limit=100.0,
            quantity=5,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_sfr_command_with_default_profit(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
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
            symbol_list=["SPY", "QQQ"],
            profit_target=None,
            cost_limit=150.0,
            quantity=1,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_sfr_command_with_default_cost_limit(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test sfr command with default cost limit"""
        test_args = ["alchimest.py", "sfr", "-s", "SPY", "QQQ", "-p", "2.0"]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=["SPY", "QQQ"],
            profit_target=2.0,
            cost_limit=120.0,  # Default value
            quantity=1,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_sfr_command_with_no_symbols(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test sfr command with no symbols (should use default list)"""
        test_args = ["alchimest.py", "sfr", "-p", "1.0", "-l", "200"]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=None,
            profit_target=1.0,
            cost_limit=200.0,
            quantity=1,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_syn_command_with_all_valid_arguments(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
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
            "-pr",
            "2.5",
            "-q",
            "7",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["SPY", "QQQ", "META"],
            cost_limit=100.0,
            max_loss_threshold=50.0,
            max_profit_threshold=200.0,
            profit_ratio_threshold=2.5,
            quantity=7,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
            scoring_strategy="balanced",
            risk_reward_weight=None,
            liquidity_weight=None,
            time_decay_weight=None,
            market_quality_weight=None,
            min_risk_reward=None,
            min_liquidity=None,
            max_bid_ask_spread=None,
            optimal_days_expiry=None,
        )

    @pytest.mark.integration
    def test_syn_command_with_default_cost_limit(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
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
            "-pr",
            "3.0",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["SPY", "QQQ"],
            cost_limit=120.0,  # Default value
            max_loss_threshold=30.0,
            max_profit_threshold=150.0,
            profit_ratio_threshold=3.0,
            quantity=1,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
            scoring_strategy="balanced",
            risk_reward_weight=None,
            liquidity_weight=None,
            time_decay_weight=None,
            market_quality_weight=None,
            min_risk_reward=None,
            min_liquidity=None,
            max_bid_ask_spread=None,
            optimal_days_expiry=None,
        )

    @pytest.mark.integration
    def test_syn_command_with_no_symbols(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test syn command with no symbols (should use default list)"""
        test_args = ["alchimest.py", "syn", "-l", "80", "-ml", "25", "-pr", "1.5"]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=None,
            cost_limit=80.0,
            max_loss_threshold=25.0,
            max_profit_threshold=None,
            profit_ratio_threshold=1.5,
            quantity=1,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
            scoring_strategy="balanced",
            risk_reward_weight=None,
            liquidity_weight=None,
            time_decay_weight=None,
            market_quality_weight=None,
            min_risk_reward=None,
            min_liquidity=None,
            max_bid_ask_spread=None,
            optimal_days_expiry=None,
        )

    @pytest.mark.integration
    def test_syn_command_with_only_max_profit(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test syn command with only max_profit specified"""
        test_args = ["alchimest.py", "syn", "-s", "SPY", "-mp", "300", "-pr", "2.0"]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["SPY"],
            cost_limit=120.0,  # Default value
            max_loss_threshold=None,
            max_profit_threshold=300.0,
            profit_ratio_threshold=2.0,
            quantity=1,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
            scoring_strategy="balanced",
            risk_reward_weight=None,
            liquidity_weight=None,
            time_decay_weight=None,
            market_quality_weight=None,
            min_risk_reward=None,
            min_liquidity=None,
            max_bid_ask_spread=None,
            optimal_days_expiry=None,
        )

    @pytest.mark.integration
    def test_syn_command_with_only_max_loss(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test syn command with only max_loss specified"""
        test_args = [
            "alchimest.py",
            "syn",
            "-s",
            "SPY",
            "QQQ",
            "-ml",
            "40",
            "-pr",
            "1.8",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["SPY", "QQQ"],
            cost_limit=120.0,  # Default value
            max_loss_threshold=40.0,
            max_profit_threshold=None,
            profit_ratio_threshold=1.8,
            quantity=1,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
            scoring_strategy="balanced",
            risk_reward_weight=None,
            liquidity_weight=None,
            time_decay_weight=None,
            market_quality_weight=None,
            min_risk_reward=None,
            min_liquidity=None,
            max_bid_ask_spread=None,
            optimal_days_expiry=None,
        )

    @pytest.mark.integration
    def test_syn_command_with_no_optional_args(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test syn command with no optional arguments"""
        test_args = ["alchimest.py", "syn", "-s", "SPY"]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["SPY"],
            cost_limit=120.0,  # Default value
            max_loss_threshold=None,
            max_profit_threshold=None,
            profit_ratio_threshold=None,
            quantity=1,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
            scoring_strategy="balanced",
            risk_reward_weight=None,
            liquidity_weight=None,
            time_decay_weight=None,
            market_quality_weight=None,
            min_risk_reward=None,
            min_liquidity=None,
            max_bid_ask_spread=None,
            optimal_days_expiry=None,
        )

    @pytest.mark.integration
    def test_invalid_command(self, capture_output: Tuple[StringIO, StringIO]) -> None:
        """Test invalid command handling"""
        test_args = ["alchimest.py", "invalid_command"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as excinfo:
                alchimest.main()
            assert excinfo.value.code == 2  # argparse uses exit code 2 for CLI errors

    @pytest.mark.integration
    def test_no_command_provided(
        self, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test when no command is provided"""
        test_args = ["alchimest.py"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                alchimest.main()

    @pytest.mark.integration
    def test_sfr_command_help(self, capture_output: Tuple[StringIO, StringIO]) -> None:
        """Test sfr command help"""
        test_args = ["alchimest.py", "sfr", "--help"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                alchimest.main()

    @pytest.mark.integration
    def test_syn_command_help(self, capture_output: Tuple[StringIO, StringIO]) -> None:
        """Test syn command help"""
        test_args = ["alchimest.py", "syn", "--help"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                alchimest.main()

    @pytest.mark.integration
    def test_main_help(self, capture_output: Tuple[StringIO, StringIO]) -> None:
        """Test main help"""
        test_args = ["alchimest.py", "--help"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                alchimest.main()

    @pytest.mark.integration
    def test_sfr_command_with_special_symbols(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
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
            symbol_list=["!MES", "@SPX", "SPY"],
            profit_target=0.8,
            cost_limit=90.0,
            quantity=1,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_syn_command_with_special_symbols(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
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
            "-pr",
            "2.2",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["!MES", "@SPX", "SPY"],
            cost_limit=75.0,
            max_loss_threshold=20.0,
            max_profit_threshold=180.0,
            profit_ratio_threshold=2.2,
            quantity=1,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
            scoring_strategy="balanced",
            risk_reward_weight=None,
            liquidity_weight=None,
            time_decay_weight=None,
            market_quality_weight=None,
            min_risk_reward=None,
            min_liquidity=None,
            max_bid_ask_spread=None,
            optimal_days_expiry=None,
        )

    @pytest.mark.integration
    def test_sfr_command_with_negative_profit(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test sfr command with negative profit value"""
        test_args = ["alchimest.py", "sfr", "-s", "SPY", "-p", "-0.5", "-l", "100"]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=["SPY"],
            profit_target=-0.5,
            cost_limit=100.0,
            quantity=1,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_syn_command_with_negative_values(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
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
            "-pr",
            "0.5",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["SPY"],
            cost_limit=100.0,
            max_loss_threshold=-10.0,
            max_profit_threshold=-50.0,
            profit_ratio_threshold=0.5,
            quantity=1,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
            scoring_strategy="balanced",
            risk_reward_weight=None,
            liquidity_weight=None,
            time_decay_weight=None,
            market_quality_weight=None,
            min_risk_reward=None,
            min_liquidity=None,
            max_bid_ask_spread=None,
            optimal_days_expiry=None,
        )

    @pytest.mark.integration
    def test_sfr_command_with_zero_values(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test sfr command with zero values"""
        test_args = ["alchimest.py", "sfr", "-s", "SPY", "-p", "0.0", "-l", "0"]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=["SPY"],
            profit_target=0.0,
            cost_limit=0.0,
            quantity=1,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_syn_command_with_zero_values(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
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
            "-pr",
            "0",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["SPY"],
            cost_limit=0.0,
            max_loss_threshold=0.0,
            max_profit_threshold=0.0,
            profit_ratio_threshold=0.0,
            quantity=1,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
            scoring_strategy="balanced",
            risk_reward_weight=None,
            liquidity_weight=None,
            time_decay_weight=None,
            market_quality_weight=None,
            min_risk_reward=None,
            min_liquidity=None,
            max_bid_ask_spread=None,
            optimal_days_expiry=None,
        )

    @pytest.mark.integration
    def test_sfr_command_with_large_values(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test sfr command with large values"""
        test_args = ["alchimest.py", "sfr", "-s", "SPY", "-p", "999.99", "-l", "9999"]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=["SPY"],
            profit_target=999.99,
            cost_limit=9999.0,
            quantity=1,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_syn_command_with_large_values(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
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
            "-pr",
            "10.5",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["SPY"],
            cost_limit=9999.0,
            max_loss_threshold=5000.0,
            max_profit_threshold=10000.0,
            profit_ratio_threshold=10.5,
            quantity=1,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
            scoring_strategy="balanced",
            risk_reward_weight=None,
            liquidity_weight=None,
            time_decay_weight=None,
            market_quality_weight=None,
            min_risk_reward=None,
            min_liquidity=None,
            max_bid_ask_spread=None,
            optimal_days_expiry=None,
        )

    @pytest.mark.integration
    def test_sfr_command_with_default_quantity(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test sfr command with default quantity value (should be 1)"""
        test_args = [
            "alchimest.py",
            "sfr",
            "-s",
            "SPY",
            "QQQ",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=["SPY", "QQQ"],
            profit_target=None,
            cost_limit=120.0,
            quantity=1,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
        )

    @pytest.mark.integration
    def test_syn_command_with_default_quantity(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test syn command with default quantity value (should be 1)"""
        test_args = [
            "alchimest.py",
            "syn",
            "-s",
            "SPY",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=["SPY"],
            cost_limit=120.0,
            max_loss_threshold=None,
            max_profit_threshold=None,
            profit_ratio_threshold=None,
            quantity=1,
            log_file=None,
            debug=False,
            warning=False,
            error=False,
            finviz_url=None,
            scoring_strategy="balanced",
            risk_reward_weight=None,
            liquidity_weight=None,
            time_decay_weight=None,
            market_quality_weight=None,
            min_risk_reward=None,
            min_liquidity=None,
            max_bid_ask_spread=None,
            optimal_days_expiry=None,
        )
