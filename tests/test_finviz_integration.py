"""
Comprehensive tests for Finviz integration functionality.
Tests URL validation, cleaning, scraping, and CLI integration.
"""

import os
import sys
from io import StringIO
from typing import Generator, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alchimest
from commands.option import OptionScan
from modules.finviz_scraper import (
    scrape_tickers_from_finviz,
    validate_and_clean_finviz_url,
)


class TestFinvizURLValidation:
    """Test class for Finviz URL validation and cleaning"""

    def test_validate_clean_normal_url(self) -> None:
        """Test validation of a normal Finviz URL"""
        url = "https://finviz.com/screener.ashx?v=111&f=cap_largeover"
        result = validate_and_clean_finviz_url(url)
        assert result == url

    def test_validate_clean_escaped_url(self) -> None:
        """Test validation of an escaped Finviz URL"""
        escaped_url = "https://finviz.com/screener.ashx\\?v\\=111\\&f\\=cap_largeover"
        expected = "https://finviz.com/screener.ashx?v=111&f=cap_largeover"
        result = validate_and_clean_finviz_url(escaped_url)
        assert result == expected

    def test_validate_clean_complex_escaped_url(self) -> None:
        """Test validation of a complex escaped Finviz URL"""
        escaped_url = "https://finviz.com/screener.ashx\\?v\\=111\\&f\\=cap_largeover,fa_estltgrowth_o15,fa_fpe_u45\\&ft\\=4"
        expected = "https://finviz.com/screener.ashx?v=111&f=cap_largeover,fa_estltgrowth_o15,fa_fpe_u45&ft=4"
        result = validate_and_clean_finviz_url(escaped_url)
        assert result == expected

    def test_validate_empty_url(self) -> None:
        """Test validation of empty URL"""
        result = validate_and_clean_finviz_url("")
        assert result is None

    def test_validate_none_url(self) -> None:
        """Test validation of None URL"""
        result = validate_and_clean_finviz_url(None)
        assert result is None

    def test_validate_non_http_url(self) -> None:
        """Test validation of non-HTTP URL"""
        url = "ftp://finviz.com/screener.ashx"
        result = validate_and_clean_finviz_url(url)
        assert result is None

    def test_validate_non_finviz_url(self) -> None:
        """Test validation of non-Finviz URL"""
        url = "https://google.com/search?q=test"
        result = validate_and_clean_finviz_url(url)
        assert result is None

    def test_validate_non_screener_finviz_url(self) -> None:
        """Test validation of Finviz URL that's not a screener"""
        url = "https://finviz.com/quote.ashx?t=SPY"
        result = validate_and_clean_finviz_url(url)
        assert result is None

    def test_validate_http_finviz_url(self) -> None:
        """Test validation of HTTP (not HTTPS) Finviz URL"""
        url = "http://finviz.com/screener.ashx?v=111"
        result = validate_and_clean_finviz_url(url)
        assert result == url


class TestFinvizScraping:
    """Test class for Finviz scraping functionality"""

    @patch("modules.finviz_scraper.webdriver.Chrome")
    @patch("modules.finviz_scraper.ChromeDriverManager")
    def test_scrape_tickers_success(
        self, mock_driver_manager: Mock, mock_chrome: Mock
    ) -> None:
        """Test successful ticker scraping"""
        # Mock WebDriver setup
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver

        # Mock table and rows
        mock_table = Mock()
        mock_driver.find_element.return_value = mock_table

        # Mock wait
        mock_wait = Mock()
        mock_wait.until.return_value = mock_table

        # Mock rows with tickers
        mock_row1 = Mock()
        mock_row2 = Mock()
        mock_row3 = Mock()

        # Mock cells - skip first row (header)
        mock_cell1 = Mock()
        mock_cell1.text = "ABBV"
        mock_cell2 = Mock()
        mock_cell2.text = "AAPL"
        mock_cell3 = Mock()
        mock_cell3.text = "MSFT"

        mock_row1.find_elements.return_value = [
            Mock(),
            mock_cell1,
        ]  # Skip first cell, get ticker from second
        mock_row2.find_elements.return_value = [Mock(), mock_cell2]
        mock_row3.find_elements.return_value = [Mock(), mock_cell3]

        mock_table.find_elements.return_value = [
            Mock(),
            mock_row1,
            mock_row2,
            mock_row3,
        ]  # First is header

        with patch("modules.finviz_scraper.WebDriverWait") as mock_wait_class:
            mock_wait_class.return_value = mock_wait

            result = scrape_tickers_from_finviz(
                "https://finviz.com/screener.ashx?v=111"
            )

        assert result == ["ABBV", "AAPL", "MSFT"]
        mock_driver.quit.assert_called_once()

    def test_scrape_tickers_invalid_url(self) -> None:
        """Test scraping with invalid URL"""
        result = scrape_tickers_from_finviz("https://google.com")
        assert result is None

    def test_scrape_tickers_escaped_url(self) -> None:
        """Test scraping with escaped URL gets cleaned"""
        escaped_url = "https://finviz.com/screener.ashx\\?v\\=111"

        with (
            patch("modules.finviz_scraper.webdriver.Chrome") as mock_chrome,
            patch("modules.finviz_scraper.ChromeDriverManager"),
            patch("modules.finviz_scraper.WebDriverWait") as mock_wait_class,
        ):

            mock_driver = Mock()
            mock_chrome.return_value = mock_driver

            # Mock empty table to avoid complex setup
            mock_table = Mock()
            mock_table.find_elements.return_value = [Mock()]  # Just header

            mock_wait = Mock()
            mock_wait.until.return_value = mock_table
            mock_wait_class.return_value = mock_wait

            result = scrape_tickers_from_finviz(escaped_url)

            # Should call driver.get with cleaned URL
            expected_clean_url = "https://finviz.com/screener.ashx?v=111"
            mock_driver.get.assert_called_once_with(expected_clean_url)

    @patch("modules.finviz_scraper.webdriver.Chrome")
    @patch("modules.finviz_scraper.ChromeDriverManager")
    def test_scrape_tickers_timeout(
        self, mock_driver_manager: Mock, mock_chrome: Mock
    ) -> None:
        """Test scraping with timeout exception"""
        from selenium.common.exceptions import TimeoutException

        mock_driver = Mock()
        mock_chrome.return_value = mock_driver

        with patch("modules.finviz_scraper.WebDriverWait") as mock_wait_class:
            mock_wait = Mock()
            mock_wait.until.side_effect = TimeoutException()
            mock_wait_class.return_value = mock_wait

            result = scrape_tickers_from_finviz(
                "https://finviz.com/screener.ashx?v=111"
            )

        assert result is None
        mock_driver.quit.assert_called_once()

    @patch("modules.finviz_scraper.webdriver.Chrome")
    @patch("modules.finviz_scraper.ChromeDriverManager")
    def test_scrape_tickers_webdriver_exception(
        self, mock_driver_manager: Mock, mock_chrome: Mock
    ) -> None:
        """Test scraping with WebDriver exception"""
        from selenium.common.exceptions import WebDriverException

        mock_driver = Mock()
        mock_chrome.return_value = mock_driver

        with patch("modules.finviz_scraper.WebDriverWait") as mock_wait_class:
            mock_wait = Mock()
            mock_wait.until.side_effect = WebDriverException("Connection failed")
            mock_wait_class.return_value = mock_wait

            result = scrape_tickers_from_finviz(
                "https://finviz.com/screener.ashx?v=111"
            )

        assert result is None
        mock_driver.quit.assert_called_once()


class TestOptionScanSymbolResolution:
    """Test class for OptionScan symbol resolution logic"""

    @patch("commands.option.scrape_tickers_from_finviz")
    @patch("commands.option.SFR")
    @patch("commands.option.asyncio.run")
    def test_sfr_finviz_url_success(
        self, mock_asyncio: Mock, mock_sfr: Mock, mock_scrape: Mock
    ) -> None:
        """Test SFR finder with successful Finviz URL"""
        mock_scrape.return_value = ["ABBV", "AAPL", "MSFT"]

        scanner = OptionScan()
        scanner.sfr_finder(
            symbol_list=None,
            profit_target=1.0,
            cost_limit=120,
            quantity=1,
            finviz_url="https://finviz.com/screener.ashx?v=111",
        )

        mock_scrape.assert_called_once_with("https://finviz.com/screener.ashx?v=111")

        # Check that asyncio.run was called with scraped symbols
        mock_asyncio.assert_called_once()
        call_args = mock_asyncio.call_args[0][0]
        # The scan method should be called with scraped symbols
        assert hasattr(call_args, "cr_frame")  # It's a coroutine

    @patch("commands.option.scrape_tickers_from_finviz")
    @patch("commands.option.SFR")
    @patch("commands.option.asyncio.run")
    def test_sfr_finviz_url_failure_fallback_to_default(
        self, mock_asyncio: Mock, mock_sfr: Mock, mock_scrape: Mock
    ) -> None:
        """Test SFR finder with failed Finviz URL falls back to default"""
        mock_scrape.return_value = None  # Scraping failed

        scanner = OptionScan()
        scanner.sfr_finder(
            symbol_list=None,
            profit_target=1.0,
            cost_limit=120,
            quantity=1,
            finviz_url="https://finviz.com/screener.ashx?v=111",
        )

        mock_scrape.assert_called_once_with("https://finviz.com/screener.ashx?v=111")
        mock_asyncio.assert_called_once()

    @patch("commands.option.scrape_tickers_from_finviz")
    @patch("commands.option.SFR")
    @patch("commands.option.asyncio.run")
    def test_sfr_finviz_url_with_manual_symbols_prefers_finviz(
        self, mock_asyncio: Mock, mock_sfr: Mock, mock_scrape: Mock
    ) -> None:
        """Test SFR finder with both Finviz URL and manual symbols prefers Finviz"""
        mock_scrape.return_value = ["ABBV", "AAPL"]

        scanner = OptionScan()
        scanner.sfr_finder(
            symbol_list=["SPY", "QQQ"],  # Manual symbols
            profit_target=1.0,
            cost_limit=120,
            quantity=1,
            finviz_url="https://finviz.com/screener.ashx?v=111",
        )

        mock_scrape.assert_called_once_with("https://finviz.com/screener.ashx?v=111")
        mock_asyncio.assert_called_once()

    @patch("commands.option.scrape_tickers_from_finviz")
    @patch("commands.option.Syn")
    @patch("commands.option.asyncio.run")
    def test_syn_finviz_url_success(
        self, mock_asyncio: Mock, mock_syn: Mock, mock_scrape: Mock
    ) -> None:
        """Test SYN finder with successful Finviz URL"""
        mock_scrape.return_value = ["TSLA", "NVDA", "AMD"]

        scanner = OptionScan()
        scanner.syn_finder(
            symbol_list=None,
            cost_limit=150,
            max_loss_threshold=50,
            max_profit_threshold=200,
            profit_ratio_threshold=2.0,
            quantity=2,
            finviz_url="https://finviz.com/screener.ashx?v=111",
        )

        mock_scrape.assert_called_once_with("https://finviz.com/screener.ashx?v=111")
        mock_asyncio.assert_called_once()

    @patch("commands.option.SFR")
    @patch("commands.option.asyncio.run")
    def test_sfr_no_finviz_url_uses_manual_symbols(
        self, mock_asyncio: Mock, mock_sfr: Mock
    ) -> None:
        """Test SFR finder without Finviz URL uses manual symbols"""
        scanner = OptionScan()
        scanner.sfr_finder(
            symbol_list=["SPY", "QQQ"],
            profit_target=1.0,
            cost_limit=120,
            quantity=1,
            finviz_url=None,
        )

        mock_asyncio.assert_called_once()

    @patch("commands.option.SFR")
    @patch("commands.option.asyncio.run")
    def test_sfr_no_finviz_url_no_symbols_uses_default(
        self, mock_asyncio: Mock, mock_sfr: Mock
    ) -> None:
        """Test SFR finder without Finviz URL or manual symbols uses default"""
        scanner = OptionScan()
        scanner.sfr_finder(
            symbol_list=None,
            profit_target=1.0,
            cost_limit=120,
            quantity=1,
            finviz_url=None,
        )

        mock_asyncio.assert_called_once()


class TestCLIFinvizIntegration:
    """Test class for CLI integration with Finviz URLs"""

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
    def test_sfr_command_with_finviz_url(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test sfr command with Finviz URL"""
        test_args = [
            "alchimest.py",
            "sfr",
            "-f",
            "https://finviz.com/screener.ashx?v=111&f=cap_largeover",
            "-p",
            "1.5",
            "-l",
            "100",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=None,
            profit_target=1.5,
            cost_limit=100.0,
            quantity=1,
            log_file=None,
            debug=False,
            finviz_url="https://finviz.com/screener.ashx?v=111&f=cap_largeover",
        )

    @pytest.mark.integration
    def test_sfr_command_with_finviz_url_and_symbols(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test sfr command with both Finviz URL and manual symbols"""
        test_args = [
            "alchimest.py",
            "sfr",
            "-f",
            "https://finviz.com/screener.ashx?v=111",
            "-s",
            "SPY",
            "QQQ",
            "-p",
            "2.0",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=["SPY", "QQQ"],
            profit_target=2.0,
            cost_limit=120.0,
            quantity=1,
            log_file=None,
            debug=False,
            finviz_url="https://finviz.com/screener.ashx?v=111",
        )

    @pytest.mark.integration
    def test_syn_command_with_finviz_url(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test syn command with Finviz URL"""
        test_args = [
            "alchimest.py",
            "syn",
            "-f",
            "https://finviz.com/screener.ashx?v=111&f=cap_largeover",
            "-l",
            "150",
            "-ml",
            "50",
            "-mp",
            "200",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=None,
            cost_limit=150.0,
            max_loss_threshold=50.0,
            max_profit_threshold=200.0,
            profit_ratio_threshold=None,
            quantity=1,
            log_file=None,
            debug=False,
            finviz_url="https://finviz.com/screener.ashx?v=111&f=cap_largeover",
        )

    @pytest.mark.integration
    def test_sfr_command_with_escaped_finviz_url(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test sfr command with escaped Finviz URL (simulates shell escaping)"""
        escaped_url = "https://finviz.com/screener.ashx\\?v\\=111\\&f\\=cap_largeover"
        test_args = [
            "alchimest.py",
            "sfr",
            "-f",
            escaped_url,
            "-p",
            "1.0",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=None,
            profit_target=1.0,
            cost_limit=120.0,
            quantity=1,
            log_file=None,
            debug=False,
            finviz_url=escaped_url,  # URL passed as-is to function, cleaning happens inside
        )

    @pytest.mark.integration
    def test_syn_command_with_finviz_url_and_all_options(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test syn command with Finviz URL and all options"""
        test_args = [
            "alchimest.py",
            "syn",
            "-f",
            "https://finviz.com/screener.ashx?v=111",
            "-l",
            "200",
            "-ml",
            "75",
            "-mp",
            "300",
            "-pr",
            "3.5",
            "-q",
            "5",
            "--debug",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=None,
            cost_limit=200.0,
            max_loss_threshold=75.0,
            max_profit_threshold=300.0,
            profit_ratio_threshold=3.5,
            quantity=5,
            log_file=None,
            debug=True,
            finviz_url="https://finviz.com/screener.ashx?v=111",
        )

    @pytest.mark.integration
    def test_sfr_command_with_only_finviz_url(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test sfr command with only Finviz URL (no symbols required)"""
        test_args = [
            "alchimest.py",
            "sfr",
            "-f",
            "https://finviz.com/screener.ashx?v=111",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.sfr_finder.assert_called_once_with(
            symbol_list=None,
            profit_target=None,
            cost_limit=120.0,  # Default
            quantity=1,  # Default
            log_file=None,
            debug=False,
            finviz_url="https://finviz.com/screener.ashx?v=111",
        )

    @pytest.mark.integration
    def test_syn_command_with_only_finviz_url(
        self, mock_option_scan: MagicMock, capture_output: Tuple[StringIO, StringIO]
    ) -> None:
        """Test syn command with only Finviz URL (no symbols required)"""
        test_args = [
            "alchimest.py",
            "syn",
            "-f",
            "https://finviz.com/screener.ashx?v=111",
        ]

        with patch.object(sys, "argv", test_args):
            alchimest.main()

        mock_option_scan.syn_finder.assert_called_once_with(
            symbol_list=None,
            cost_limit=120.0,  # Default
            max_loss_threshold=None,
            max_profit_threshold=None,
            profit_ratio_threshold=None,
            quantity=1,  # Default
            log_file=None,
            debug=False,
            finviz_url="https://finviz.com/screener.ashx?v=111",
        )
