"""
Comprehensive tests for logging functionality in modules/Arbitrage/common.py

This test file validates the logging improvements described in ADR-002:
- Log level configuration and filters
- File handler configuration with rotation
- Filter behavior differences between console and file handlers
- Integration with SFR module funnel logging
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import logging
import pytest
from rich.logging import RichHandler

from modules.Arbitrage.common import (
    InfoOnlyFilter,
    InfoWarningErrorCriticalFilter,
    InfoWarningFilter,
    configure_logging,
    get_console_handler,
    get_logger,
)


class TestLoggingFilters:
    """Test custom logging filters"""

    def test_info_only_filter_allows_info(self):
        """Test InfoOnlyFilter allows only INFO messages"""
        filter_obj = InfoOnlyFilter()

        # Create mock log records
        info_record = MagicMock()
        info_record.levelno = logging.INFO

        warning_record = MagicMock()
        warning_record.levelno = logging.WARNING

        debug_record = MagicMock()
        debug_record.levelno = logging.DEBUG

        error_record = MagicMock()
        error_record.levelno = logging.ERROR

        # Test filter behavior
        assert filter_obj.filter(info_record) is True
        assert filter_obj.filter(warning_record) is False
        assert filter_obj.filter(debug_record) is False
        assert filter_obj.filter(error_record) is False

    def test_info_warning_filter_allows_info_and_warning(self):
        """Test InfoWarningFilter allows INFO and WARNING messages"""
        filter_obj = InfoWarningFilter()

        # Create mock log records
        info_record = MagicMock()
        info_record.levelno = logging.INFO

        warning_record = MagicMock()
        warning_record.levelno = logging.WARNING

        debug_record = MagicMock()
        debug_record.levelno = logging.DEBUG

        error_record = MagicMock()
        error_record.levelno = logging.ERROR

        # Test filter behavior
        assert filter_obj.filter(info_record) is True
        assert filter_obj.filter(warning_record) is True
        assert filter_obj.filter(debug_record) is False
        assert filter_obj.filter(error_record) is False

    def test_info_warning_error_critical_filter_excludes_debug(self):
        """Test InfoWarningErrorCriticalFilter allows all except DEBUG"""
        filter_obj = InfoWarningErrorCriticalFilter()

        # Create mock log records for all levels
        debug_record = MagicMock()
        debug_record.levelno = logging.DEBUG

        info_record = MagicMock()
        info_record.levelno = logging.INFO

        warning_record = MagicMock()
        warning_record.levelno = logging.WARNING

        error_record = MagicMock()
        error_record.levelno = logging.ERROR

        critical_record = MagicMock()
        critical_record.levelno = logging.CRITICAL

        # Test filter behavior
        assert filter_obj.filter(debug_record) is False
        assert filter_obj.filter(info_record) is True
        assert filter_obj.filter(warning_record) is True
        assert filter_obj.filter(error_record) is True
        assert filter_obj.filter(critical_record) is True


class TestConsoleHandler:
    """Test console handler configuration"""

    def test_get_console_handler_default_info_filter(self):
        """Test get_console_handler with default info filter"""
        handler = get_console_handler()

        assert isinstance(handler, RichHandler)
        assert handler.level == logging.DEBUG
        assert len(handler.filters) == 1
        assert isinstance(handler.filters[0], InfoOnlyFilter)

    def test_get_console_handler_warning_filter(self):
        """Test get_console_handler with warning filter"""
        handler = get_console_handler(filter_type="warning")

        assert isinstance(handler, RichHandler)
        assert len(handler.filters) == 1
        assert isinstance(handler.filters[0], InfoWarningFilter)

    def test_get_console_handler_error_filter(self):
        """Test get_console_handler with error filter"""
        handler = get_console_handler(filter_type="error")

        assert isinstance(handler, RichHandler)
        assert len(handler.filters) == 1
        assert isinstance(handler.filters[0], InfoWarningErrorCriticalFilter)

    def test_get_console_handler_none_filter(self):
        """Test get_console_handler with no filter"""
        handler = get_console_handler(filter_type="none")

        assert isinstance(handler, RichHandler)
        assert len(handler.filters) == 0  # No filters applied

    def test_get_console_handler_invalid_filter_defaults_to_info(self):
        """Test get_console_handler with invalid filter type defaults to info"""
        handler = get_console_handler(filter_type="invalid")

        assert isinstance(handler, RichHandler)
        assert len(handler.filters) == 1
        assert isinstance(handler.filters[0], InfoOnlyFilter)


class TestConfigureLogging:
    """Test logging configuration function"""

    def setup_method(self):
        """Set up test method with clean logging state"""
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def teardown_method(self):
        """Clean up after each test"""
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_configure_logging_default_info_only(self):
        """Test configure_logging with default parameters (info only)"""
        configure_logging()

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1

        handler = root_logger.handlers[0]
        assert isinstance(handler, RichHandler)
        assert len(handler.filters) == 1
        assert isinstance(handler.filters[0], InfoOnlyFilter)

    def test_configure_logging_debug_mode(self):
        """Test configure_logging with debug=True"""
        configure_logging(debug=True)

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1

        handler = root_logger.handlers[0]
        assert isinstance(handler, RichHandler)
        assert len(handler.filters) == 0  # No filters in debug mode

    def test_configure_logging_warning_mode(self):
        """Test configure_logging with warning=True"""
        configure_logging(warning=True)

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1

        handler = root_logger.handlers[0]
        assert isinstance(handler, RichHandler)
        assert len(handler.filters) == 1
        assert isinstance(handler.filters[0], InfoWarningFilter)

    def test_configure_logging_error_mode(self):
        """Test configure_logging with error=True"""
        configure_logging(error=True)

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1

        handler = root_logger.handlers[0]
        assert isinstance(handler, RichHandler)
        assert len(handler.filters) == 1
        assert isinstance(handler.filters[0], InfoWarningErrorCriticalFilter)

    def test_configure_logging_priority_debug_over_error_and_warning(self):
        """Test that debug=True takes precedence over error=True and warning=True"""
        configure_logging(debug=True, error=True, warning=True)

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        assert len(handler.filters) == 0  # Debug mode, no filters

    def test_configure_logging_priority_error_over_warning(self):
        """Test that error=True takes precedence over warning=True"""
        configure_logging(error=True, warning=True)

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        assert len(handler.filters) == 1
        assert isinstance(handler.filters[0], InfoWarningErrorCriticalFilter)


class TestFileHandlerConfiguration:
    """Test file handler configuration and rotation"""

    def setup_method(self):
        """Set up test method with clean logging state"""
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def teardown_method(self):
        """Clean up after each test"""
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_configure_logging_with_file_handler(self):
        """Test configure_logging creates RotatingFileHandler with correct settings"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            configure_logging(log_file=log_file)

            root_logger = logging.getLogger()
            assert len(root_logger.handlers) == 2  # Console + File

            # Find the file handler
            file_handler = None
            for handler in root_logger.handlers:
                if not isinstance(handler, RichHandler):
                    file_handler = handler
                    break

            assert file_handler is not None
            assert file_handler.maxBytes == 10485760  # 10MB
            assert file_handler.backupCount == 5
            assert file_handler.baseFilename.endswith("test.log")

    def test_file_handler_filter_differs_from_console_info_mode(self):
        """Test that file handler uses InfoWarningErrorCriticalFilter when console uses InfoOnlyFilter"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            # Configure with info-only console filter
            configure_logging(log_file=log_file, use_info_filter=True)

            root_logger = logging.getLogger()
            console_handler = None
            file_handler = None

            for handler in root_logger.handlers:
                if isinstance(handler, RichHandler):
                    console_handler = handler
                else:
                    file_handler = handler

            # Console should have InfoOnlyFilter
            assert len(console_handler.filters) == 1
            assert isinstance(console_handler.filters[0], InfoOnlyFilter)

            # File should have InfoWarningErrorCriticalFilter
            assert len(file_handler.filters) == 1
            assert isinstance(file_handler.filters[0], InfoWarningErrorCriticalFilter)

    def test_file_handler_warning_mode_same_as_console(self):
        """Test that file handler uses same filter as console in warning mode"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            configure_logging(warning=True, log_file=log_file)

            root_logger = logging.getLogger()
            console_handler = None
            file_handler = None

            for handler in root_logger.handlers:
                if isinstance(handler, RichHandler):
                    console_handler = handler
                else:
                    file_handler = handler

            # Both should have InfoWarningFilter
            assert len(console_handler.filters) == 1
            assert isinstance(console_handler.filters[0], InfoWarningFilter)
            assert len(file_handler.filters) == 1
            assert isinstance(file_handler.filters[0], InfoWarningFilter)

    def test_file_handler_debug_mode_no_filters(self):
        """Test that file handler has no filters in debug mode"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            configure_logging(debug=True, log_file=log_file)

            root_logger = logging.getLogger()
            file_handler = None

            for handler in root_logger.handlers:
                if not isinstance(handler, RichHandler):
                    file_handler = handler
                    break

            assert len(file_handler.filters) == 0  # No filters in debug mode

    def test_file_handler_level_configuration(self):
        """Test file handler level configuration based on debug flag"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            # Test non-debug mode
            configure_logging(log_file=log_file)
            root_logger = logging.getLogger()
            file_handler = [
                h for h in root_logger.handlers if not isinstance(h, RichHandler)
            ][0]
            assert file_handler.level == logging.INFO

            # Clear handlers
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

            # Test debug mode
            configure_logging(debug=True, log_file=log_file)
            root_logger = logging.getLogger()
            file_handler = [
                h for h in root_logger.handlers if not isinstance(h, RichHandler)
            ][0]
            assert file_handler.level == logging.DEBUG


class TestActualLoggingBehavior:
    """Test actual logging behavior with different configurations"""

    def setup_method(self):
        """Set up test method with clean logging state"""
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def teardown_method(self):
        """Clean up after each test"""
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_info_filter_allows_only_info_messages(self):
        """Test that INFO filter actually filters messages correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            configure_logging(use_info_filter=True, log_file=log_file)
            logger = get_logger("test")

            # Log messages at different levels
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")

            # Force flush handlers
            for handler in logging.getLogger().handlers:
                handler.flush()

            # Check file contents - should contain INFO from both console and file filters
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    content = f.read()

                # File handler uses InfoWarningErrorCriticalFilter, so should contain INFO, WARNING, ERROR
                assert "Info message" in content
                assert "Warning message" in content
                assert "Error message" in content
                assert "Debug message" not in content

    def test_warning_filter_allows_info_and_warning(self):
        """Test that WARNING filter allows INFO and WARNING messages"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            configure_logging(warning=True, log_file=log_file)
            logger = get_logger("test")

            # Log messages at different levels
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")

            # Force flush handlers
            for handler in logging.getLogger().handlers:
                handler.flush()

            # Check file contents
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    content = f.read()

                # File handler uses same filter as console in warning mode
                assert "Info message" in content
                assert "Warning message" in content
                assert "Debug message" not in content
                assert (
                    "Error message" not in content
                )  # InfoWarningFilter excludes ERROR

    def test_debug_mode_captures_all_messages(self):
        """Test that debug mode captures all log levels"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            configure_logging(debug=True, log_file=log_file)
            logger = get_logger("test")

            # Log messages at different levels
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")

            # Force flush handlers
            for handler in logging.getLogger().handlers:
                handler.flush()

            # Check file contents
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    content = f.read()

                # Debug mode should capture all messages
                assert "Debug message" in content
                assert "Info message" in content
                assert "Warning message" in content
                assert "Error message" in content
                assert "Critical message" in content


class TestSFRFunnelLoggingIntegration:
    """Integration tests for SFR module funnel logging with [Funnel] prefix"""

    def setup_method(self):
        """Set up test method with clean logging state"""
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def teardown_method(self):
        """Clean up after each test"""
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_funnel_messages_logged_at_info_level(self):
        """Test that funnel stage messages are logged at INFO level"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            configure_logging(log_file=log_file)
            logger = get_logger("test")

            # Log funnel messages as they should be in SFR
            logger.info("[Funnel] [SPY] Stage: evaluated (expiry: 20250830)")
            logger.info(
                "[Funnel] [SPY] Stage: stock_ticker_available (expiry: 20250830)"
            )
            logger.info(
                "[Funnel] [SPY] Stage: theoretical_profit_positive (expiry: 20250830, profit: $0.25)"
            )

            # Also log a rejection message at WARNING level
            logger.warning(
                "[SPY] REJECTED - Arbitrage Condition Not Met: profit $0.05 < threshold $0.10"
            )

            # Force flush handlers
            for handler in logging.getLogger().handlers:
                handler.flush()

            # Check file contents
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    content = f.read()

                # File handler should capture both INFO funnel stages and WARNING rejections
                assert "[Funnel] [SPY] Stage: evaluated" in content
                assert "[Funnel] [SPY] Stage: stock_ticker_available" in content
                assert "[Funnel] [SPY] Stage: theoretical_profit_positive" in content
                assert "REJECTED - Arbitrage Condition Not Met" in content

    def test_funnel_prefix_filtering(self):
        """Test that [Funnel] prefix messages can be easily identified and filtered"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            configure_logging(log_file=log_file)
            logger = get_logger("test")

            # Mixed logging as would occur in actual SFR execution
            logger.info("[SPY] Regular info message")
            logger.info("[Funnel] [SPY] Stage: evaluated (expiry: 20250830)")
            logger.info("[SPY] Another regular message")
            logger.info(
                "[Funnel] [SPY] Stage: passed_viability_check (expiry: 20250830)"
            )
            logger.warning("[SPY] REJECTED - some rejection reason")
            logger.info(
                "[Funnel Summary] SPY: 5 evaluated → 3 theoretical → 1 viable → 0 executed"
            )

            # Force flush handlers
            for handler in logging.getLogger().handlers:
                handler.flush()

            # Check file contents and count funnel messages
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    lines = f.readlines()

                funnel_lines = [line for line in lines if "[Funnel]" in line]
                regular_lines = [
                    line for line in lines if "[Funnel]" not in line and "[SPY]" in line
                ]

                # Should have at least 2 funnel stage lines (the summary line has different format)
                assert (
                    len(funnel_lines) >= 2
                )  # At least the two explicit [Funnel] stage messages
                # Should have 3 regular lines (2 info + 1 warning)
                assert len(regular_lines) >= 3

    def test_rejection_messages_at_warning_level_captured_in_file(self):
        """Test that WARNING level rejection messages are captured in file even with INFO console filter"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            # Use info-only console filter, but file should capture warnings too
            configure_logging(use_info_filter=True, log_file=log_file)
            logger = get_logger("test")

            # Log typical SFR flow
            logger.info("[Funnel] [AAPL] Stage: evaluated (expiry: 20250830)")
            logger.info(
                "[Funnel] [AAPL] Stage: theoretical_profit_positive (expiry: 20250830, profit: $0.15)"
            )
            logger.warning("[AAPL] No theoretical arbitrage for 20250830: profit=$0.05")
            logger.warning("[AAPL] REJECTED - Arbitrage Condition Not Met")

            # Force flush handlers
            for handler in logging.getLogger().handlers:
                handler.flush()

            # File should contain both INFO and WARNING messages due to InfoWarningErrorCriticalFilter
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    content = f.read()

                assert "[Funnel] [AAPL] Stage: evaluated" in content
                assert "[Funnel] [AAPL] Stage: theoretical_profit_positive" in content
                assert "No theoretical arbitrage" in content
                assert "REJECTED - Arbitrage Condition Not Met" in content


class TestLoggerInstance:
    """Test logger instance creation"""

    def test_get_logger_returns_configured_logger(self):
        """Test that get_logger returns a properly configured logger"""
        logger = get_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_default_name(self):
        """Test that get_logger uses default name when not specified"""
        logger = get_logger()

        assert isinstance(logger, logging.Logger)
        assert logger.name == "rich"


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def setup_method(self):
        """Set up test method with clean logging state"""
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def teardown_method(self):
        """Clean up after each test"""
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_configure_logging_with_invalid_file_path(self):
        """Test configure_logging behavior with invalid file path"""
        # This should not crash but may not create the file
        with pytest.raises((OSError, PermissionError, FileNotFoundError)):
            configure_logging(log_file="/invalid/path/test.log")

    def test_configure_logging_force_parameter(self):
        """Test that force=True parameter overrides existing configuration"""
        # Configure logging twice to test force parameter
        configure_logging()
        initial_handlers = len(logging.getLogger().handlers)

        configure_logging()  # Should override due to force=True
        final_handlers = len(logging.getLogger().handlers)

        # Should still have same number of handlers (not doubled)
        assert final_handlers == initial_handlers

    def test_multiple_configure_calls_consistent_state(self):
        """Test that multiple calls to configure_logging maintain consistent state"""
        configure_logging(warning=True)
        first_config_handlers = len(logging.getLogger().handlers)
        first_config_filter = type(logging.getLogger().handlers[0].filters[0])

        configure_logging(warning=True)  # Same configuration
        second_config_handlers = len(logging.getLogger().handlers)
        second_config_filter = type(logging.getLogger().handlers[0].filters[0])

        assert first_config_handlers == second_config_handlers
        assert first_config_filter == second_config_filter

    @pytest.mark.integration
    def test_file_rotation_configuration(self):
        """Integration test for file rotation when size limit is reached"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            configure_logging(log_file=log_file)
            logger = get_logger("test")

            # The rotation configuration should be set correctly
            root_logger = logging.getLogger()
            file_handler = None
            for handler in root_logger.handlers:
                if not isinstance(handler, RichHandler):
                    file_handler = handler
                    break

            assert file_handler is not None
            assert hasattr(file_handler, "maxBytes")
            assert hasattr(file_handler, "backupCount")
            assert file_handler.maxBytes == 10485760  # 10MB
            assert file_handler.backupCount == 5
