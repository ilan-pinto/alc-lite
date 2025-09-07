"""
Unit tests for Rollback Manager system.

Tests rollback attempt tracking, limits enforcement, and
sophisticated partial position unwinding logic.
"""

import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.Arbitrage.sfr.rollback_manager import (
    RollbackAttempt,
    RollbackManager,
    RollbackPlan,
    RollbackPosition,
    RollbackReason,
    RollbackStrategy,
)


# Test helper classes
class MockLeg:
    """Mock leg for testing rollback scenarios"""

    def __init__(self, leg_type, price, quantity=100):
        self.leg_type = leg_type
        self.price = price
        self.quantity = quantity
        self.filled = quantity > 0

    def __str__(self):
        return f"{self.leg_type}@{self.price}"


class TestRollbackManager:
    """Test rollback attempt tracking and limits"""

    @pytest.fixture
    def rollback_manager(self):
        """Create fresh rollback manager"""
        manager = RollbackManager()
        # Reset all counters for clean test state
        manager.rollback_attempts = []
        manager.symbol_rollback_counts = {}
        manager.symbol_pause_until = {}
        manager.global_rollback_count = 0
        return manager

    def test_initialization(self, rollback_manager):
        """Test rollback manager initialization"""
        assert rollback_manager.rollback_attempts == []
        assert rollback_manager.symbol_rollback_counts == {}
        assert rollback_manager.global_rollback_count == 0
        assert hasattr(rollback_manager, "symbol_pause_until")

    def test_global_rollback_limit_enforcement(self, rollback_manager):
        """Test global rollback limit (3 attempts max)"""

        # Patch constants for testing
        with patch("modules.Arbitrage.sfr.rollback_manager.MAX_ROLLBACK_ATTEMPTS", 3):

            # Should allow first 3 rollbacks
            for i in range(3):
                assert rollback_manager.should_stop_rollbacks("SPY") is False
                rollback_id = rollback_manager.start_rollback(
                    symbol="SPY", filled_legs=[MockLeg("stock", 100.0)], reason="test"
                )
                asyncio.run(rollback_manager.complete_rollback(rollback_id, 0.0, True))

            # 4th rollback should be blocked
            assert rollback_manager.should_stop_rollbacks("QQQ") is True

    def test_per_symbol_rollback_limit_enforcement(self, rollback_manager):
        """Test per-symbol rollback limit (2 attempts max)"""

        with patch("modules.Arbitrage.sfr.rollback_manager.MAX_ROLLBACK_ATTEMPTS", 10):
            with patch(
                "modules.Arbitrage.sfr.rollback_manager.MAX_ROLLBACK_ATTEMPTS_PER_SYMBOL",
                2,
            ):

                # SPY: 2 rollbacks allowed
                for i in range(2):
                    assert rollback_manager.should_stop_rollbacks("SPY") is False
                    rollback_id = rollback_manager.start_rollback(
                        symbol="SPY",
                        filled_legs=[MockLeg("stock", 100.0)],
                        reason="test",
                    )
                    asyncio.run(
                        rollback_manager.complete_rollback(rollback_id, 0.0, True)
                    )

                # SPY: 3rd rollback blocked (per-symbol limit)
                assert rollback_manager.should_stop_rollbacks("SPY") is True

                # QQQ: Should still be allowed (different symbol)
                assert rollback_manager.should_stop_rollbacks("QQQ") is False

    def test_rollback_attempt_tracking(self, rollback_manager):
        """Test detailed rollback attempt tracking"""

        filled_legs = [MockLeg("stock", 100.0), MockLeg("call", 8.5)]

        rollback_id = rollback_manager.start_rollback(
            symbol="SPY", filled_legs=filled_legs, reason="partial_fill_timeout"
        )

        # Verify attempt tracked
        attempt = next(
            (
                a
                for a in rollback_manager.rollback_attempts
                if a.rollback_id == rollback_id
            ),
            None,
        )
        assert attempt is not None
        assert attempt.symbol == "SPY"
        assert attempt.reason == "partial_fill_timeout"
        assert attempt.filled_legs == ["stock", "call"]
        assert attempt.success is False  # Not completed yet
        assert isinstance(attempt.timestamp, datetime)

        # Verify counters updated
        assert rollback_manager.global_rollback_count == 1
        assert rollback_manager.symbol_rollback_counts["SPY"] == 1

    @pytest.mark.asyncio
    async def test_rollback_completion_tracking(self, rollback_manager):
        """Test rollback completion with success/failure tracking"""

        rollback_id = rollback_manager.start_rollback(
            symbol="SPY", filled_legs=[MockLeg("stock", 100.0)], reason="test"
        )

        # Complete successfully
        await rollback_manager.complete_rollback(
            rollback_id=rollback_id, cost=2.50, success=True
        )

        attempt = next(
            a
            for a in rollback_manager.rollback_attempts
            if a.rollback_id == rollback_id
        )

        assert attempt.success is True
        assert attempt.rollback_cost == 2.50
        assert attempt.rollback_time > 0
        assert attempt.error_message is None

    @pytest.mark.asyncio
    async def test_rollback_failure_tracking(self, rollback_manager):
        """Test rollback failure tracking"""

        rollback_id = rollback_manager.start_rollback(
            symbol="SPY", filled_legs=[MockLeg("stock", 100.0)], reason="test"
        )

        # Complete with failure
        await rollback_manager.complete_rollback(
            rollback_id=rollback_id,
            cost=0.0,
            success=False,
            error="Market order failed",
        )

        attempt = next(
            a
            for a in rollback_manager.rollback_attempts
            if a.rollback_id == rollback_id
        )

        assert attempt.success is False
        assert attempt.rollback_cost == 0.0
        assert attempt.error_message == "Market order failed"

    def test_symbol_pause_recommendation(self, rollback_manager):
        """Test symbol pause logic after repeated rollbacks"""

        with patch(
            "modules.Arbitrage.sfr.rollback_manager.MAX_ROLLBACK_ATTEMPTS_PER_SYMBOL", 2
        ):

            # One rollback for SPY
            rollback_id = rollback_manager.start_rollback("SPY", [], "test")
            asyncio.run(rollback_manager.complete_rollback(rollback_id, 0.0, True))

            # Should recommend pause after 1st rollback (approaching limit)
            assert rollback_manager.should_pause_symbol("SPY") is True

            # Fresh symbol should not need pause
            assert rollback_manager.should_pause_symbol("QQQ") is False

    def test_rollback_statistics_accuracy(self, rollback_manager):
        """Test rollback statistics calculation"""

        # Create mix of successful and failed rollbacks
        costs = [2.0, 3.0, 0.0, 1.5]  # 0.0 for failed rollback
        successes = [True, True, False, True]

        for i, (success, cost) in enumerate(zip(successes, costs)):
            rollback_id = rollback_manager.start_rollback(f"SYM{i}", [], "test")
            asyncio.run(rollback_manager.complete_rollback(rollback_id, cost, success))

        stats = rollback_manager.get_rollback_statistics()

        assert stats["total_attempts"] == 4
        assert stats["successful"] == 3
        assert stats["failed"] == 1
        assert stats["success_rate"] == 0.75
        assert stats["total_cost"] == 6.5  # 2.0 + 3.0 + 0.0 + 1.5
        assert stats["symbols_affected"] == 4

    def test_rollback_summary_logging(self, rollback_manager, caplog):
        """Test rollback summary log generation"""

        rollback_id = rollback_manager.start_rollback(
            symbol="SPY",
            filled_legs=[MockLeg("stock", 100.0)],
            reason="partial_fill_timeout",
        )

        # Should generate start log
        assert "ROLLBACK SUMMARY" in caplog.text
        assert "STARTED" in caplog.text
        assert "SPY" in caplog.text

        caplog.clear()

        # Complete rollback
        asyncio.run(rollback_manager.complete_rollback(rollback_id, 5.0, True))

        # Should generate completion log
        assert "COMPLETED" in caplog.text
        assert "$5.00" in caplog.text

    def test_rollback_id_uniqueness(self, rollback_manager):
        """Test that rollback IDs are unique"""

        rollback_ids = []
        for i in range(5):
            rollback_id = rollback_manager.start_rollback(
                symbol=f"SYM{i}", filled_legs=[MockLeg("stock", 100.0)], reason="test"
            )
            rollback_ids.append(rollback_id)
            time.sleep(0.001)  # Ensure time difference

        # All IDs should be unique
        assert len(set(rollback_ids)) == len(rollback_ids)

        # All should contain symbol and timestamp info
        for rollback_id in rollback_ids:
            assert "_" in rollback_id  # Should have delimiter
            parts = rollback_id.split("_")
            assert len(parts) >= 3  # symbol, timestamp, sequence

    @pytest.mark.asyncio
    async def test_rollback_not_found_handling(self, rollback_manager):
        """Test handling of invalid rollback ID in completion"""

        # Try to complete non-existent rollback
        await rollback_manager.complete_rollback(
            rollback_id="invalid_id", cost=0.0, success=False, error="Should not crash"
        )

        # Should handle gracefully without crashing
        assert len(rollback_manager.rollback_attempts) == 0

    def test_rollback_attempt_dataclass_validation(self):
        """Test RollbackAttempt dataclass structure"""

        from dataclasses import fields

        # Verify all expected fields exist
        field_names = {f.name for f in fields(RollbackAttempt)}
        expected_fields = {
            "rollback_id",
            "symbol",
            "timestamp",
            "reason",
            "filled_legs",
            "rollback_cost",
            "rollback_time",
            "success",
            "error_message",
            "legs_rolled_back",
        }

        assert expected_fields.issubset(field_names)

    @pytest.mark.asyncio
    async def test_concurrent_rollback_tracking(self, rollback_manager):
        """Test concurrent rollback attempts are tracked correctly"""

        async def start_rollback(symbol, delay=0):
            if delay:
                await asyncio.sleep(delay)
            return rollback_manager.start_rollback(
                symbol=symbol,
                filled_legs=[MockLeg("stock", 100.0)],
                reason="concurrent_test",
            )

        # Start multiple rollbacks concurrently
        tasks = [
            start_rollback("SPY", 0.01),
            start_rollback("QQQ", 0.02),
            start_rollback("TSLA", 0.01),
        ]

        rollback_ids = await asyncio.gather(*tasks)

        # All should be tracked
        assert len(rollback_manager.rollback_attempts) == 3
        assert rollback_manager.global_rollback_count == 3

        # Each symbol should have count of 1
        for symbol in ["SPY", "QQQ", "TSLA"]:
            assert rollback_manager.symbol_rollback_counts[symbol] == 1

    @pytest.mark.asyncio
    async def test_rollback_timing_accuracy(self, rollback_manager):
        """Test rollback timing measurement accuracy"""

        rollback_id = rollback_manager.start_rollback(
            symbol="SPY", filled_legs=[MockLeg("stock", 100.0)], reason="timing_test"
        )

        # Wait a specific amount of time
        await asyncio.sleep(0.1)

        await rollback_manager.complete_rollback(rollback_id, 1.0, True)

        attempt = next(
            a
            for a in rollback_manager.rollback_attempts
            if a.rollback_id == rollback_id
        )

        # Should measure approximately the sleep time
        assert 0.05 <= attempt.rollback_time <= 0.2  # Allow some variance

    def test_rollback_cost_tracking(self, rollback_manager):
        """Test rollback cost accumulation"""

        costs = [1.25, 2.50, 0.75, 3.00]

        for i, cost in enumerate(costs):
            rollback_id = rollback_manager.start_rollback(f"SYM{i}", [], "cost_test")
            asyncio.run(rollback_manager.complete_rollback(rollback_id, cost, True))

        stats = rollback_manager.get_rollback_statistics()
        expected_total = sum(costs)
        expected_average = expected_total / len(costs)

        assert abs(stats["total_cost"] - expected_total) < 0.01
        assert (
            abs(stats["average_cost"] - expected_average) < 0.01
            if "average_cost" in stats
            else True
        )

    def test_warning_thresholds(self, rollback_manager, caplog):
        """Test warning logs when approaching limits"""

        with patch("modules.Arbitrage.sfr.rollback_manager.MAX_ROLLBACK_ATTEMPTS", 3):
            with patch(
                "modules.Arbitrage.sfr.rollback_manager.MAX_ROLLBACK_ATTEMPTS_PER_SYMBOL",
                2,
            ):

                # Second rollback for a symbol should warn
                rollback_id1 = rollback_manager.start_rollback(
                    "SPY", [], "warning_test"
                )
                asyncio.run(rollback_manager.complete_rollback(rollback_id1, 1.0, True))

                rollback_id2 = rollback_manager.start_rollback(
                    "SPY", [], "warning_test"
                )

                # Should contain warning about approaching limits
                log_text = caplog.text
                assert "WARNING" in log_text or "approaching" in log_text.lower()

    @pytest.mark.asyncio
    async def test_save_rollback_report(self, rollback_manager):
        """Test rollback report saving (if implemented)"""

        rollback_id = rollback_manager.start_rollback(
            symbol="SPY",
            filled_legs=[MockLeg("stock", 100.0), MockLeg("call", 8.5)],
            reason="save_test",
        )

        await rollback_manager.complete_rollback(rollback_id, 2.0, True)

        # If save method exists, test it
        if hasattr(rollback_manager, "save_rollback_report"):
            attempt = rollback_manager.rollback_attempts[-1]
            result = await rollback_manager.save_rollback_report(attempt)
            assert result is not None


class TestRollbackManagerEdgeCases:
    """Test edge cases and error scenarios for rollback manager"""

    @pytest.fixture
    def rollback_manager(self):
        """Create fresh rollback manager"""
        return RollbackManager()

    def test_empty_filled_legs(self, rollback_manager):
        """Test rollback with no filled legs"""

        rollback_id = rollback_manager.start_rollback(
            symbol="SPY", filled_legs=[], reason="empty_legs_test"  # Empty list
        )

        attempt = next(
            a
            for a in rollback_manager.rollback_attempts
            if a.rollback_id == rollback_id
        )

        assert attempt.filled_legs == []

    def test_very_long_symbol_name(self, rollback_manager):
        """Test with unusually long symbol names"""

        long_symbol = "A" * 100  # 100 character symbol

        rollback_id = rollback_manager.start_rollback(
            symbol=long_symbol,
            filled_legs=[MockLeg("stock", 100.0)],
            reason="long_symbol_test",
        )

        assert long_symbol in rollback_id
        assert rollback_manager.symbol_rollback_counts[long_symbol] == 1

    def test_negative_rollback_cost(self, rollback_manager):
        """Test handling of negative rollback cost (profit from rollback)"""

        rollback_id = rollback_manager.start_rollback("SPY", [], "negative_cost_test")

        # Negative cost means rollback actually made money
        asyncio.run(rollback_manager.complete_rollback(rollback_id, -1.50, True))

        attempt = rollback_manager.rollback_attempts[-1]
        assert attempt.rollback_cost == -1.50
        assert attempt.success is True

    def test_zero_rollback_time(self, rollback_manager):
        """Test instant rollback completion"""

        rollback_id = rollback_manager.start_rollback("SPY", [], "instant_test")

        # Complete immediately
        asyncio.run(rollback_manager.complete_rollback(rollback_id, 0.0, True))

        attempt = rollback_manager.rollback_attempts[-1]
        assert attempt.rollback_time >= 0  # Should be non-negative

    @pytest.mark.asyncio
    async def test_rollback_exception_during_completion(self, rollback_manager):
        """Test handling of exceptions during rollback completion"""

        rollback_id = rollback_manager.start_rollback("SPY", [], "exception_test")

        # Mock save method to raise exception
        with patch.object(
            rollback_manager,
            "save_rollback_report",
            side_effect=Exception("Save failed"),
        ):
            # Should not crash even if save fails
            await rollback_manager.complete_rollback(rollback_id, 1.0, True)

            # Rollback should still be marked complete
            attempt = rollback_manager.rollback_attempts[-1]
            assert attempt.success is True

    def test_multiple_legs_same_type(self, rollback_manager):
        """Test rollback with multiple legs of same type"""

        # Multiple stock legs (unusual but possible)
        filled_legs = [
            MockLeg("stock", 100.0),
            MockLeg("stock", 101.0),
            MockLeg("call", 8.5),
        ]

        rollback_id = rollback_manager.start_rollback(
            symbol="SPY", filled_legs=filled_legs, reason="multi_leg_test"
        )

        attempt = rollback_manager.rollback_attempts[-1]
        assert attempt.filled_legs == ["stock", "stock", "call"]


@pytest.mark.performance
class TestRollbackManagerPerformance:
    """Performance tests for rollback manager"""

    @pytest.fixture
    def rollback_manager(self):
        return RollbackManager()

    def test_large_number_of_rollbacks_performance(self, rollback_manager):
        """Test performance with many rollback attempts"""

        start_time = time.time()

        # Create 1000 rollback attempts
        for i in range(1000):
            rollback_id = rollback_manager.start_rollback(
                symbol=f"SYM{i % 10}",  # Reuse symbols to test per-symbol tracking
                filled_legs=[MockLeg("stock", 100.0)],
                reason="performance_test",
            )
            asyncio.run(rollback_manager.complete_rollback(rollback_id, 1.0, True))

        total_time = time.time() - start_time

        # Should handle 1000 rollbacks in reasonable time
        assert total_time < 10.0  # Less than 10 seconds

        # Verify all tracked correctly
        assert len(rollback_manager.rollback_attempts) == 1000
        assert rollback_manager.global_rollback_count == 1000

    def test_statistics_calculation_performance(self, rollback_manager):
        """Test performance of statistics calculation"""

        # Generate many rollback attempts
        for i in range(500):
            rollback_id = rollback_manager.start_rollback(f"SYM{i}", [], "stats_test")
            success = i % 3 != 0  # 2/3 success rate
            cost = i * 0.1 if success else 0.0
            asyncio.run(rollback_manager.complete_rollback(rollback_id, cost, success))

        # Statistics calculation should be fast
        start_time = time.time()
        stats = rollback_manager.get_rollback_statistics()
        calc_time = time.time() - start_time

        assert calc_time < 1.0  # Should calculate stats in under 1 second
        assert stats["total_attempts"] == 500

    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self, rollback_manager):
        """Test performance of concurrent rollback operations"""

        async def rollback_workflow(symbol_id):
            rollback_id = rollback_manager.start_rollback(
                symbol=f"SYM{symbol_id}",
                filled_legs=[MockLeg("stock", 100.0)],
                reason="concurrent_test",
            )
            await asyncio.sleep(0.01)  # Simulate work
            await rollback_manager.complete_rollback(rollback_id, 1.0, True)

        start_time = time.time()

        # Run 100 concurrent rollback workflows
        tasks = [rollback_workflow(i) for i in range(100)]
        await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Should handle concurrent operations efficiently
        assert total_time < 5.0  # Should complete in under 5 seconds
        assert len(rollback_manager.rollback_attempts) == 100
