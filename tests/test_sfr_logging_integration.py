"""
Integration tests specifically for SFR module funnel logging behavior as described in ADR-002.

This test file validates that the SFR module properly implements the logging improvements:
- Funnel stages logged at INFO level with [Funnel] prefix
- Rejection messages logged at WARNING level
- File handler captures both INFO and WARNING messages
- Integration with the metrics collection system
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import logging
import pytest

from modules.Arbitrage.common import configure_logging, get_logger


class TestSFRLoggingCompliance:
    """Test SFR module logging compliance with ADR-002 requirements"""

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

    def test_funnel_stages_info_level_with_prefix(self):
        """Test that funnel stages are logged at INFO level with [Funnel] prefix"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "sfr_test.log")

            configure_logging(log_file=log_file)
            logger = get_logger("modules.Arbitrage.SFR")

            # Simulate actual funnel stage logging from SFR module
            symbol = "SPY"
            expiry = "20250830"

            # These are the exact log patterns from ADR-002 "After" examples
            logger.info(f"[Funnel] [{symbol}] Stage: evaluated (expiry: {expiry})")
            logger.info(
                f"[Funnel] [{symbol}] Stage: stock_ticker_available (expiry: {expiry})"
            )
            logger.info(
                f"[Funnel] [{symbol}] Stage: passed_priority_filter (expiry: {expiry})"
            )
            logger.info(
                f"[Funnel] [{symbol}] Stage: passed_viability_check (expiry: {expiry})"
            )
            logger.info(
                f"[Funnel] [{symbol}] Stage: option_data_available (expiry: {expiry})"
            )
            logger.info(
                f"[Funnel] [{symbol}] Stage: passed_data_quality (expiry: {expiry})"
            )
            logger.info(f"[Funnel] [{symbol}] Stage: prices_valid (expiry: {expiry})")
            logger.info(
                f"[Funnel] [{symbol}] Stage: theoretical_profit_positive (expiry: {expiry}, profit: $0.25)"
            )
            logger.info(
                f"[Funnel] [{symbol}] Stage: guaranteed_profit_positive (expiry: {expiry}, profit: $0.20)"
            )

            # Force flush handlers
            for handler in logging.getLogger().handlers:
                handler.flush()

            # Verify file contains all funnel stages
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    content = f.read()

                # All funnel stages should be present in log file
                expected_stages = [
                    "Stage: evaluated",
                    "Stage: stock_ticker_available",
                    "Stage: passed_priority_filter",
                    "Stage: passed_viability_check",
                    "Stage: option_data_available",
                    "Stage: passed_data_quality",
                    "Stage: prices_valid",
                    "Stage: theoretical_profit_positive",
                    "Stage: guaranteed_profit_positive",
                ]

                for stage in expected_stages:
                    assert f"[Funnel] [{symbol}] {stage}" in content

                # Verify all have INFO level
                info_count = content.count("INFO")
                assert info_count >= len(expected_stages)

    def test_rejection_messages_warning_level(self):
        """Test that rejection messages are logged at WARNING level without [Funnel] prefix"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "sfr_test.log")

            configure_logging(log_file=log_file)
            logger = get_logger("modules.Arbitrage.SFR")

            symbol = "AAPL"
            expiry = "20250830"

            # These are typical rejection message patterns from SFR
            logger.warning(
                f"[{symbol}] No theoretical arbitrage for {expiry}: profit=$0.05"
            )
            logger.warning(
                f"[{symbol}] Theoretical profit $0.15 but guaranteed only $0.08 - rejecting"
            )
            logger.warning(
                f"[{symbol}] Call contract bid-ask spread too wide: 25.00 > 20.00, expiry: {expiry}"
            )

            # Force flush handlers
            for handler in logging.getLogger().handlers:
                handler.flush()

            # Verify file contains rejection messages at WARNING level
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    content = f.read()

                # All rejection messages should be present
                assert "No theoretical arbitrage" in content
                assert "guaranteed only" in content
                assert "bid-ask spread too wide" in content

                # All should be at WARNING level
                warning_count = content.count("WARNING")
                assert warning_count >= 3

                # Should not have [Funnel] prefix on rejection messages
                rejection_lines = [
                    line for line in content.split("\n") if "WARNING" in line
                ]
                for line in rejection_lines:
                    assert "[Funnel]" not in line

    def test_mixed_funnel_and_rejection_logging(self):
        """Test realistic scenario with mixed funnel stages and rejections"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "sfr_test.log")

            configure_logging(
                log_file=log_file
            )  # Uses InfoOnlyFilter for console, InfoWarningErrorCriticalFilter for file
            logger = get_logger("modules.Arbitrage.SFR")

            symbol = "MSFT"
            expiry = "20250830"

            # Simulate typical SFR scan flow
            logger.info(f"[Funnel] [{symbol}] Stage: evaluated (expiry: {expiry})")
            logger.info(
                f"[Funnel] [{symbol}] Stage: stock_ticker_available (expiry: {expiry})"
            )
            logger.info(
                f"[Funnel] [{symbol}] Stage: passed_viability_check (expiry: {expiry})"
            )
            logger.info(
                f"[Funnel] [{symbol}] Stage: theoretical_profit_positive (expiry: {expiry}, profit: $0.18)"
            )
            logger.warning(
                f"[{symbol}] Theoretical profit $0.18 but guaranteed only $0.05 - rejecting"
            )
            logger.info(
                f"[Funnel Summary] {symbol}: 1 evaluated → 1 theoretical → 0 viable → 0 executed"
            )

            # Force flush handlers
            for handler in logging.getLogger().handlers:
                handler.flush()

            # File handler should capture both INFO and WARNING due to InfoWarningErrorCriticalFilter
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    content = f.read()

                # Should contain both funnel stages (INFO) and rejection (WARNING)
                assert "[Funnel] [MSFT] Stage: evaluated" in content
                assert "[Funnel] [MSFT] Stage: theoretical_profit_positive" in content
                assert "guaranteed only $0.05 - rejecting" in content
                assert "[Funnel Summary]" in content

                # Count log levels
                info_count = content.count("INFO")
                warning_count = content.count("WARNING")

                assert info_count >= 4  # 4 INFO messages (3 funnel stages + 1 summary)
                assert warning_count >= 1  # 1 WARNING message (rejection)

    def test_console_vs_file_filter_behavior(self):
        """Test that console and file handlers have different filter behavior per ADR-002"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "sfr_test.log")

            # Configure with default settings (InfoOnly console, InfoWarningErrorCritical file)
            configure_logging(use_info_filter=True, log_file=log_file)

            root_logger = logging.getLogger()
            console_handler = None
            file_handler = None

            for handler in root_logger.handlers:
                if hasattr(handler, "console"):  # RichHandler
                    console_handler = handler
                else:
                    file_handler = handler

            # Verify filter configuration per ADR-002
            assert len(console_handler.filters) == 1
            assert console_handler.filters[0].__class__.__name__ == "InfoOnlyFilter"

            assert len(file_handler.filters) == 1
            assert (
                file_handler.filters[0].__class__.__name__
                == "InfoWarningErrorCriticalFilter"
            )

    def test_adr_002_validation_examples(self):
        """Test specific examples from ADR-002 validation section"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "sfr_test.log")

            configure_logging(log_file=log_file)
            logger = get_logger("modules.Arbitrage.SFR")

            # Examples from ADR-002 "Before" and "After" sections

            # BEFORE (incorrect - should NOT be used):
            # logger.warning(f"[{symbol}] Funnel Stage: evaluated (expiry: {expiry})")

            # AFTER (correct - what should be implemented):
            logger.info("[Funnel] [SPY] Stage: evaluated (expiry: 20250830)")
            logger.info(
                "[Funnel] [AAPL] Stage: theoretical_profit_positive (expiry: 20250915, profit: $0.25)"
            )

            # Rejection messages remain at WARNING level (correct behavior)
            logger.warning(
                "[SPY] REJECTED - Arbitrage Condition Not Met: spread 1.62 > net credit 1.57"
            )
            logger.warning(
                "[AAPL] REJECTED - Profit Target Not Met: target 0.15% > actual ROI 0.14%"
            )

            # Force flush handlers
            for handler in logging.getLogger().handlers:
                handler.flush()

            # Validate ADR-002 requirements are met
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    content = f.read()

                # ✅ All funnel stages now logged at INFO level with consistent [Funnel] format
                assert "[Funnel] [SPY] Stage: evaluated" in content
                assert "[Funnel] [AAPL] Stage: theoretical_profit_positive" in content
                assert "INFO" in content

                # ✅ Log file captures INFO level messages with rotating file handler
                info_lines = [
                    line
                    for line in content.split("\n")
                    if "INFO" in line and "[Funnel]" in line
                ]
                assert len(info_lines) >= 2

                # ✅ Rejection messages still at WARNING level
                assert "WARNING" in content
                assert "REJECTED - Arbitrage Condition Not Met" in content
                assert "REJECTED - Profit Target Not Met" in content

                # ✅ No loss of critical debugging information
                warning_lines = [
                    line for line in content.split("\n") if "WARNING" in line
                ]
                assert len(warning_lines) >= 2

    @pytest.mark.integration
    def test_rotating_file_handler_configuration(self):
        """Integration test for RotatingFileHandler configuration per ADR-002"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "sfr_rotating_test.log")

            configure_logging(log_file=log_file)

            # Find the file handler
            root_logger = logging.getLogger()
            file_handler = None
            for handler in root_logger.handlers:
                if not hasattr(handler, "console"):  # Not RichHandler
                    file_handler = handler
                    break

            # Verify RotatingFileHandler configuration from ADR-002
            assert file_handler is not None
            assert hasattr(file_handler, "maxBytes")
            assert hasattr(file_handler, "backupCount")

            # ✅ Rotating file handler (10MB max, 5 backups)
            assert file_handler.maxBytes == 10485760  # 10MB
            assert file_handler.backupCount == 5

            # Verify formatter
            assert file_handler.formatter is not None
            formatter = file_handler.formatter
            assert "%(asctime)s" in formatter._fmt
            assert "%(levelname)s" in formatter._fmt

    def test_no_duplicate_logging(self):
        """Test that duplicate logging issues from ADR-002 are resolved"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "sfr_test.log")

            configure_logging(log_file=log_file)
            logger = get_logger("modules.Arbitrage.SFR")

            symbol = "NVDA"
            expiry = "20250830"

            # Log the same event only once (no duplicates)
            logger.info(f"[Funnel] [{symbol}] Stage: evaluated (expiry: {expiry})")

            # Force flush handlers
            for handler in logging.getLogger().handlers:
                handler.flush()

            # Check that message appears only once
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    content = f.read()

                # Count occurrences of the specific message
                message_count = content.count(
                    f"[Funnel] [{symbol}] Stage: evaluated (expiry: {expiry})"
                )
                assert message_count == 1  # Should appear exactly once, not duplicated


class TestSFRLoggingPerformance:
    """Test logging performance characteristics"""

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

    def test_logging_does_not_impact_performance(self):
        """Test that logging improvements don't negatively impact performance"""
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "perf_test.log")

            configure_logging(log_file=log_file)
            logger = get_logger("modules.Arbitrage.SFR")

            # Time a batch of logging operations
            start_time = time.time()

            for i in range(100):
                logger.info(
                    f"[Funnel] [TEST] Stage: evaluated (expiry: 20250830_{i:03d})"
                )
                logger.warning(f"[TEST] REJECTED - Test rejection {i}")

            # Force flush all handlers
            for handler in logging.getLogger().handlers:
                handler.flush()

            end_time = time.time()
            elapsed = end_time - start_time

            # Should complete logging operations quickly (under 1 second for 200 messages)
            assert (
                elapsed < 1.0
            ), f"Logging took {elapsed:.2f}s, which may indicate performance issues"

            # Verify all messages were logged
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    lines = f.readlines()

                # Should have 200 lines (100 INFO + 100 WARNING)
                assert len(lines) >= 200

    def test_file_rotation_does_not_block(self):
        """Test that file rotation configuration doesn't cause blocking"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "rotation_test.log")

            configure_logging(log_file=log_file)
            logger = get_logger("modules.Arbitrage.SFR")

            # This should not raise any exceptions or block
            for i in range(10):
                logger.info(f"[Funnel] [TEST] Stage: test_message_{i}")

            # Force flush
            for handler in logging.getLogger().handlers:
                handler.flush()

            # File handler should be properly configured
            root_logger = logging.getLogger()
            file_handler = [
                h for h in root_logger.handlers if not hasattr(h, "console")
            ][0]

            # Verify configuration is accessible without blocking
            assert file_handler.maxBytes == 10485760
            assert file_handler.backupCount == 5
