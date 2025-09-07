"""
Test pause/resume functionality in Strategy.py per ADR-003
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from modules.Arbitrage.Strategy import ArbitrageClass


class TestStrategyPauseResume:
    """Test the new pause/resume functionality in Strategy.py"""

    @pytest.fixture
    def strategy(self):
        """Create a Strategy instance for testing"""
        strategy = ArbitrageClass()
        return strategy

    @pytest.mark.asyncio
    async def test_pause_all_other_executors(self, strategy):
        """Test pausing all other executors"""
        executing_symbol = "SPY"

        # Initially not paused
        assert strategy._executor_paused is False
        assert strategy.active_parallel_symbol is None

        # Pause all other executors
        await strategy.pause_all_other_executors(executing_symbol)

        # Verify pause state
        assert strategy._executor_paused is True
        assert strategy.active_parallel_symbol == executing_symbol

        # Test is_paused method
        assert strategy.is_paused("QQQ") is True  # Other symbol should be paused
        assert (
            strategy.is_paused("SPY") is False
        )  # Executing symbol should NOT be paused
        assert strategy.is_paused("AAPL") is True  # Other symbol should be paused

    @pytest.mark.asyncio
    async def test_resume_all_executors(self, strategy):
        """Test resuming all executors"""
        # First pause
        await strategy.pause_all_other_executors("SPY")
        assert strategy._executor_paused is True
        assert strategy.active_parallel_symbol == "SPY"

        # Then resume
        await strategy.resume_all_executors()

        # Verify resume state
        assert strategy._executor_paused is False
        assert strategy.active_parallel_symbol is None

        # All symbols should now be unpaused
        assert strategy.is_paused("QQQ") is False
        assert strategy.is_paused("SPY") is False
        assert strategy.is_paused("AAPL") is False

    @pytest.mark.asyncio
    async def test_stop_all_executors(self, strategy):
        """Test stopping all executors"""
        # Initially not stopped
        assert strategy._executor_paused is False
        assert strategy.order_filled is False

        # Stop all executors
        await strategy.stop_all_executors()

        # Verify stop state
        assert strategy._executor_paused is True
        assert strategy.order_filled is True  # Should trigger scan loop exit

        # All symbols should be paused
        assert strategy.is_paused("SPY") is True
        assert strategy.is_paused("QQQ") is True

    def test_is_paused_logic(self, strategy):
        """Test the is_paused logic comprehensively"""
        # Case 1: Not paused globally
        strategy._executor_paused = False
        strategy.active_parallel_symbol = None
        assert strategy.is_paused("SPY") is False
        assert strategy.is_paused("QQQ") is False

        # Case 2: Paused globally, with active symbol
        strategy._executor_paused = True
        strategy.active_parallel_symbol = "SPY"
        assert strategy.is_paused("SPY") is False  # Active symbol not paused
        assert strategy.is_paused("QQQ") is True  # Other symbols paused
        assert strategy.is_paused("AAPL") is True  # Other symbols paused

        # Case 3: Paused globally, no active symbol (stop state)
        strategy._executor_paused = True
        strategy.active_parallel_symbol = None
        assert strategy.is_paused("SPY") is True  # All paused
        assert strategy.is_paused("QQQ") is True  # All paused

    @pytest.mark.asyncio
    async def test_pause_resume_cycle(self, strategy):
        """Test complete pause/resume cycle"""
        symbols = ["SPY", "QQQ", "AAPL", "TSLA"]

        # Initially all unpaused
        for symbol in symbols:
            assert strategy.is_paused(symbol) is False

        # Pause for SPY execution
        await strategy.pause_all_other_executors("SPY")
        assert strategy.is_paused("SPY") is False
        for symbol in ["QQQ", "AAPL", "TSLA"]:
            assert strategy.is_paused(symbol) is True

        # Resume all
        await strategy.resume_all_executors()
        for symbol in symbols:
            assert strategy.is_paused(symbol) is False

        # Stop all
        await strategy.stop_all_executors()
        for symbol in symbols:
            assert strategy.is_paused(symbol) is True


class TestPauseResumeIntegration:
    """Test pause/resume integration with scan loops"""

    @pytest.mark.asyncio
    async def test_scan_loop_pause_behavior(self):
        """Test that scan loop respects pause state"""
        strategy = ArbitrageClass()

        scan_attempts = []

        # Mock scan method to track attempts
        async def mock_scan_with_throttle(symbol, *args, **kwargs):
            scan_attempts.append(symbol)
            await asyncio.sleep(0.01)  # Brief delay

        strategy.scan_with_throttle = mock_scan_with_throttle
        strategy.scan_sfr = MagicMock()

        # Start with paused state for QQQ
        await strategy.pause_all_other_executors("SPY")

        # Simulate scan loop checking for multiple symbols
        symbols = ["SPY", "QQQ", "AAPL"]

        for symbol in symbols:
            if not strategy.is_paused(symbol):
                await strategy.scan_with_throttle(symbol, None, 1, 0.5, 100)

        # Only SPY should have been scanned (not paused)
        assert "SPY" in scan_attempts
        assert "QQQ" not in scan_attempts
        assert "AAPL" not in scan_attempts

    @pytest.mark.asyncio
    async def test_pause_logging(self, caplog):
        """Test that pause/resume operations are properly logged"""
        strategy = ArbitrageClass()

        # Test pause logging
        await strategy.pause_all_other_executors("SPY")
        assert "Pausing all other executors per ADR-003" in caplog.text
        assert "SPY" in caplog.text

        caplog.clear()

        # Test resume logging
        await strategy.resume_all_executors()
        assert "Resuming all executors" in caplog.text

        caplog.clear()

        # Test stop logging
        await strategy.stop_all_executors()
        assert "Stopping all executors" in caplog.text
