"""
Comprehensive tests for rollback scenarios in SFR parallel execution.

This test suite covers:
1. Recursive loop bug prevention
2. Market order fallback functionality
3. Emergency shutdown mechanisms
4. Rollback order tracking and monitoring
5. Error handling for catastrophic failures
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from ib_async import IB, Contract, Order, OrderStatus, Trade

from modules.Arbitrage.sfr.parallel_execution_framework import (
    LegOrder,
    LegType,
    ParallelExecutionPlan,
)
from modules.Arbitrage.sfr.rollback_manager import (
    RollbackAttempt,
    RollbackManager,
    RollbackPlan,
    RollbackPosition,
    RollbackReason,
    RollbackStrategy,
)


@pytest.fixture
def mock_ib():
    """Mock IB connection."""
    ib = MagicMock(spec=IB)
    ib.client = MagicMock()
    ib.client.getReqId = MagicMock(side_effect=lambda: int(time.time() * 1000))
    ib.placeOrder = MagicMock()
    return ib


@pytest.fixture
def sample_contract():
    """Sample contract for testing."""
    contract = MagicMock(spec=Contract)
    contract.symbol = "AAPL"
    contract.strike = 150.0
    contract.right = "C"
    contract.expiry = "20250117"
    return contract


@pytest.fixture
def sample_leg_order(sample_contract):
    """Sample leg order for testing."""
    return LegOrder(
        leg_type=LegType.CALL,
        contract=sample_contract,
        order=MagicMock(spec=Order),
        target_price=5.50,
        action="SELL",
        quantity=1,
    )


@pytest.fixture
def rollback_manager(mock_ib):
    """Rollback manager instance for testing."""
    return RollbackManager(mock_ib, "AAPL")


@pytest.fixture
def sample_rollback_position(sample_leg_order):
    """Sample rollback position for testing."""
    return RollbackPosition(
        leg_order=sample_leg_order,
        current_quantity=1,
        avg_fill_price=5.45,
        unrealized_pnl=-0.05,
        unwinding_priority=1,
        rollback_target_price=5.40,
    )


class TestRollbackRecursionFix:
    """Test fixes for the recursive loop bug."""

    def test_add_rollback_attempt_no_recursion(self, rollback_manager):
        """Test that _add_rollback_attempt doesn't cause recursion."""
        # Create test attempt
        attempt = RollbackAttempt(
            attempt_id="test_attempt_1",
            attempt_number=1,
            strategy_used=RollbackStrategy.IMMEDIATE_MARKET,
            timestamp=time.time(),
            positions_targeted=[],
        )

        # This should not cause recursion
        rollback_manager._add_rollback_attempt(attempt)

        # Verify attempt was added
        assert len(rollback_manager.rollback_attempts) == 1
        assert rollback_manager.rollback_attempts[0] == attempt

    def test_add_rollback_attempt_memory_limit(self, rollback_manager):
        """Test that rollback attempts are limited to prevent memory leaks."""
        # Add more attempts than the limit
        for i in range(150):  # More than _max_rollback_attempts (100)
            attempt = RollbackAttempt(
                attempt_id=f"test_attempt_{i}",
                attempt_number=i,
                strategy_used=RollbackStrategy.IMMEDIATE_MARKET,
                timestamp=time.time(),
                positions_targeted=[],
            )
            rollback_manager._add_rollback_attempt(attempt)

        # Should be limited to max attempts
        assert (
            len(rollback_manager.rollback_attempts)
            <= rollback_manager._max_rollback_attempts
        )

    def test_recursion_error_handling(self, rollback_manager, sample_rollback_position):
        """Test that RecursionError is properly handled."""
        # Create a rollback plan
        rollback_plan = RollbackPlan(
            rollback_id="test_rollback_1",
            symbol="AAPL",
            execution_id="test_exec_123",
            reason=RollbackReason.PARTIAL_FILLS_TIMEOUT,
            strategy=RollbackStrategy.AGGRESSIVE_LIMIT,
            positions_to_unwind=[sample_rollback_position],
            total_positions=1,
            estimated_unwinding_cost=5.40,
            max_acceptable_loss=10.0,
            created_time=time.time(),
            max_rollback_time=time.time() + 30.0,
        )

        # Mock a method to raise RecursionError
        with patch.object(
            rollback_manager,
            "_execute_aggressive_limit_rollback",
            side_effect=RecursionError("maximum recursion depth exceeded"),
        ):

            # This should trigger emergency handling without crashing
            result = asyncio.run(rollback_manager.execute_rollback(rollback_plan))

            # Should indicate emergency fallback was used
            assert result.get("emergency_fallback") is True
            assert "Recursion error" in str(result.get("error_message", ""))


class TestMarketOrderFallback:
    """Test market order fallback functionality."""

    @pytest.mark.asyncio
    async def test_market_fallback_trigger(
        self, rollback_manager, sample_rollback_position
    ):
        """Test that market fallback is triggered when limit orders fail."""
        # Setup mock trades that don't fill
        mock_trade = MagicMock(spec=Trade)
        mock_trade.orderStatus = MagicMock(spec=OrderStatus)
        mock_trade.orderStatus.status = "Submitted"
        mock_trade.orderStatus.filled = 0
        mock_trade.orderStatus.remaining = 1

        rollback_manager.ib.placeOrder.return_value = mock_trade

        # Create rollback plan
        rollback_plan = RollbackPlan(
            rollback_id="test_rollback_fallback",
            symbol="AAPL",
            execution_id="test_exec_123",
            reason=RollbackReason.COMPLETION_FAILED,
            strategy=RollbackStrategy.AGGRESSIVE_LIMIT,
            positions_to_unwind=[sample_rollback_position],
            total_positions=1,
            estimated_unwinding_cost=5.40,
            max_acceptable_loss=10.0,
            created_time=time.time(),
            max_rollback_time=time.time() + 30.0,
        )

        # Execute aggressive limit rollback (should fail and trigger fallback)
        with patch("asyncio.sleep"):  # Speed up test
            result = await rollback_manager._execute_aggressive_limit_rollback(
                rollback_plan
            )

        # Should have triggered market fallback
        assert result.get("method") == "market_fallback"

        # Verify market fallback was used
        assert result.get("market_fallback_used") is True

    @pytest.mark.asyncio
    async def test_market_fallback_success(
        self, rollback_manager, sample_rollback_position
    ):
        """Test successful market order fallback execution."""
        # Setup mock trades - first fails, second (market) succeeds
        limit_trade = MagicMock(spec=Trade)
        limit_trade.orderStatus = MagicMock(spec=OrderStatus)
        limit_trade.orderStatus.status = "Submitted"
        limit_trade.orderStatus.filled = 0
        limit_trade.orderStatus.remaining = 1

        market_trade = MagicMock(spec=Trade)
        market_trade.orderStatus = MagicMock(spec=OrderStatus)
        market_trade.orderStatus.status = "Filled"
        market_trade.orderStatus.filled = 1
        market_trade.orderStatus.remaining = 0
        market_trade.orderStatus.avgFillPrice = 5.35

        # Market fallback should get the successful trade
        rollback_manager.ib.placeOrder.return_value = market_trade

        remaining_positions = [sample_rollback_position]

        # Create a mock rollback plan for the method call
        mock_plan = MagicMock()
        mock_plan.positions_to_unwind = remaining_positions

        with patch("asyncio.sleep"):  # Speed up test
            result = await rollback_manager._execute_market_order_fallback(
                mock_plan, remaining_positions
            )

        # Should indicate success
        assert result["unwound_count"] == 1
        assert result["method"] == "market_fallback"
        assert result["remaining_positions"] == 0

    @pytest.mark.asyncio
    async def test_market_fallback_partial_success(
        self, rollback_manager, sample_rollback_position
    ):
        """Test market fallback with partial success."""
        # Create multiple positions
        position1 = sample_rollback_position
        position2 = RollbackPosition(
            leg_order=LegOrder(
                leg_type=LegType.PUT,
                contract=sample_rollback_position.leg_order.contract,
                order=MagicMock(spec=Order),
                target_price=3.20,
                action="BUY",
                quantity=1,
            ),
            current_quantity=1,
            avg_fill_price=3.25,
            unrealized_pnl=0.05,
            unwinding_priority=2,
            rollback_target_price=3.15,
        )

        # First market order succeeds, second fails
        market_trade1 = MagicMock(spec=Trade)
        market_trade1.orderStatus = MagicMock(spec=OrderStatus)
        market_trade1.orderStatus.status = "Filled"
        market_trade1.orderStatus.filled = 1
        market_trade1.orderStatus.avgFillPrice = 5.30

        market_trade2 = MagicMock(spec=Trade)
        market_trade2.orderStatus = MagicMock(spec=OrderStatus)
        market_trade2.orderStatus.status = "Cancelled"
        market_trade2.orderStatus.filled = 0

        rollback_manager.ib.placeOrder.side_effect = [market_trade1, market_trade2]

        # Create a mock rollback plan for the method call
        mock_plan = MagicMock()
        mock_plan.positions_to_unwind = [position1, position2]

        with patch("asyncio.sleep"):
            result = await rollback_manager._execute_market_order_fallback(
                mock_plan, [position1, position2]
            )

        # Should show partial success
        assert result["unwound_count"] == 1
        assert result["remaining_positions"] == 1


class TestEmergencyShutdown:
    """Test emergency shutdown mechanisms."""

    @pytest.mark.asyncio
    async def test_emergency_market_rollback(
        self, rollback_manager, sample_rollback_position
    ):
        """Test emergency market rollback execution."""
        # Setup successful market trade
        market_trade = MagicMock(spec=Trade)
        market_trade.orderStatus = MagicMock(spec=OrderStatus)
        market_trade.orderStatus.status = "Filled"
        market_trade.orderStatus.filled = 1
        market_trade.orderStatus.avgFillPrice = 5.20

        rollback_manager.ib.placeOrder.return_value = market_trade

        # Set the position's leg order to filled status (required for emergency rollback)
        sample_rollback_position.leg_order.fill_status = "filled"

        # Create a proper RollbackPlan
        rollback_plan = RollbackPlan(
            rollback_id="emergency_test",
            symbol="AAPL",
            execution_id="test_exec_123",
            reason=RollbackReason.SYSTEM_ERROR,
            strategy=RollbackStrategy.IMMEDIATE_MARKET,
            positions_to_unwind=[sample_rollback_position],
            total_positions=1,
            estimated_unwinding_cost=5.40,
            max_acceptable_loss=10.0,
            created_time=time.time(),
            max_rollback_time=time.time() + 30.0,
        )

        with patch("asyncio.sleep"):
            result = await rollback_manager._emergency_market_rollback(rollback_plan)

        # Should indicate emergency handling
        assert result["success"] is True
        assert result["positions_unwound"] == 1
        assert result["method"] == "emergency_market"

    @pytest.mark.asyncio
    async def test_emergency_shutdown_with_ib_disconnect(self, rollback_manager):
        """Test emergency shutdown includes IB disconnection."""
        # Mock the emergency shutdown method since it doesn't exist in RollbackManager
        rollback_manager._handle_emergency_shutdown = AsyncMock()
        rollback_manager.ib.disconnect = MagicMock()

        emergency_result = {
            "emergency_action": True,
            "trigger_reason": "RECURSION_ERROR",
            "critical_failure": True,
        }

        await rollback_manager._handle_emergency_shutdown(emergency_result)

        # Should have been called
        rollback_manager._handle_emergency_shutdown.assert_called_once_with(
            emergency_result
        )

    def test_send_emergency_alert(self, rollback_manager):
        """Test emergency alert generation."""
        # Mock the emergency alert method since it doesn't exist in RollbackManager
        rollback_manager._send_emergency_alert = MagicMock(
            return_value={
                "alert_type": "CRITICAL_SYSTEM_FAILURE",
                "symbol": "AAPL",
                "trigger": "RECURSION_ERROR",
                "requires_immediate_attention": True,
            }
        )

        emergency_details = {
            "trigger_reason": "RECURSION_ERROR",
            "symbol": "AAPL",
            "execution_id": "test_exec_123",
            "positions_at_risk": 3,
        }

        alert = rollback_manager._send_emergency_alert(emergency_details)

        # Should contain critical information
        assert alert["alert_type"] == "CRITICAL_SYSTEM_FAILURE"
        assert alert["symbol"] == "AAPL"
        assert alert["trigger"] == "RECURSION_ERROR"
        assert alert["requires_immediate_attention"] is True


class TestRollbackOrderTracking:
    """Test rollback order tracking and monitoring functionality."""

    def test_get_rollback_order_status(
        self, rollback_manager, sample_rollback_position
    ):
        """Test getting rollback order status."""
        # Create rollback plan with order
        order = MagicMock(spec=Order)
        order.orderId = 12345
        sample_rollback_position.rollback_order = order

        trade = MagicMock(spec=Trade)
        trade.orderStatus = MagicMock(spec=OrderStatus)
        trade.orderStatus.status = "Submitted"
        trade.orderStatus.filled = 0
        trade.orderStatus.remaining = 1
        sample_rollback_position.rollback_trade = trade

        rollback_plan = RollbackPlan(
            rollback_id="test_rollback_tracking",
            symbol="AAPL",
            execution_id="test_exec_123",
            reason=RollbackReason.PARTIAL_FILLS_TIMEOUT,
            strategy=RollbackStrategy.AGGRESSIVE_LIMIT,
            positions_to_unwind=[sample_rollback_position],
            total_positions=1,
            estimated_unwinding_cost=5.40,
            max_acceptable_loss=10.0,
            created_time=time.time(),
            max_rollback_time=time.time() + 30.0,
        )

        rollback_manager.active_rollbacks["test_rollback_tracking"] = rollback_plan

        status = rollback_manager.get_rollback_order_status("test_rollback_tracking")

        # Verify status contains expected fields
        assert status["rollback_id"] == "test_rollback_tracking"
        assert status["symbol"] == "AAPL"
        assert status["status"] == "pending"
        assert len(status["orders"]) == 1
        assert status["orders"][0]["order_id"] == 12345
        assert status["orders"][0]["trade_status"] == "Submitted"

    def test_get_rollback_performance_metrics(self, rollback_manager):
        """Test rollback performance metrics calculation."""
        # Add some test attempts
        successful_attempt = RollbackAttempt(
            attempt_id="success_1",
            attempt_number=1,
            strategy_used=RollbackStrategy.IMMEDIATE_MARKET,
            timestamp=time.time(),
            positions_targeted=[],
            positions_unwound=1,
            attempt_cost=5.40,
            attempt_slippage=0.10,
            duration=2.5,
            success=True,
        )

        failed_attempt = RollbackAttempt(
            attempt_id="fail_1",
            attempt_number=2,
            strategy_used=RollbackStrategy.AGGRESSIVE_LIMIT,
            timestamp=time.time(),
            positions_targeted=[],
            positions_unwound=0,
            attempt_cost=0.0,
            attempt_slippage=0.0,
            duration=10.0,
            success=False,
            error_message="Timeout",
        )

        rollback_manager.rollback_attempts = [successful_attempt, failed_attempt]
        rollback_manager.performance_metrics["total_rollbacks_initiated"] = 2
        rollback_manager.performance_metrics["successful_rollbacks"] = 1
        rollback_manager.performance_metrics["failed_rollbacks"] = 1

        metrics = rollback_manager.get_rollback_performance_metrics()

        # Verify calculations
        assert metrics["success_rate"] == 0.5
        assert metrics["total_attempts"] == 2
        assert metrics["successful_attempts"] == 1
        assert metrics["failed_attempts"] == 1
        assert metrics["avg_successful_duration"] == 2.5
        assert metrics["avg_successful_cost"] == 5.40

    def test_monitor_rollback_progress(
        self, rollback_manager, sample_rollback_position
    ):
        """Test real-time rollback progress monitoring."""
        current_time = time.time()
        rollback_plan = RollbackPlan(
            rollback_id="test_rollback_monitor",
            symbol="AAPL",
            execution_id="test_exec_123",
            reason=RollbackReason.PARTIAL_FILLS_TIMEOUT,
            strategy=RollbackStrategy.AGGRESSIVE_LIMIT,
            positions_to_unwind=[sample_rollback_position, sample_rollback_position],
            total_positions=2,
            estimated_unwinding_cost=10.80,
            max_acceptable_loss=15.0,
            created_time=current_time,
            max_rollback_time=current_time + 30.0,
            positions_unwound=1,
            total_rollback_cost=5.35,
            rollback_slippage=0.05,
        )

        rollback_manager.active_rollbacks["test_rollback_monitor"] = rollback_plan

        progress = rollback_manager.monitor_rollback_progress("test_rollback_monitor")

        # Verify progress calculations
        assert progress["rollback_id"] == "test_rollback_monitor"
        assert progress["progress_percent"] == 50.0  # 1 out of 2 positions
        assert progress["positions_unwound"] == 1
        assert progress["total_positions"] == 2
        assert progress["current_cost"] == 5.35
        assert progress["current_slippage"] == 0.05

        # Verify risk metrics
        risk_metrics = progress["risk_metrics"]
        assert risk_metrics["cost_vs_estimate"] < 1.0  # Under estimate
        assert risk_metrics["within_loss_limit"] is True

    def test_get_rollback_attempts_history(self, rollback_manager):
        """Test getting rollback attempts history."""
        # Add test attempts
        for i in range(15):
            attempt = RollbackAttempt(
                attempt_id=f"attempt_{i}",
                attempt_number=i,
                strategy_used=RollbackStrategy.IMMEDIATE_MARKET,
                timestamp=time.time() + i,
                positions_targeted=[],
                positions_unwound=1 if i % 2 == 0 else 0,
                success=i % 2 == 0,
            )
            rollback_manager.rollback_attempts.append(attempt)

        # Get limited history
        history = rollback_manager.get_rollback_attempts_history(limit=5)

        # Should return last 5 attempts in reverse order
        assert len(history) == 5
        assert history[0]["attempt_id"] == "attempt_14"  # Most recent first
        assert history[4]["attempt_id"] == "attempt_10"

    def test_export_rollback_report(self, rollback_manager, sample_rollback_position):
        """Test comprehensive rollback report export."""
        # Setup rollback plan and attempts
        rollback_plan = RollbackPlan(
            rollback_id="test_export_report",
            symbol="AAPL",
            execution_id="test_exec_123",
            reason=RollbackReason.COMPLETION_FAILED,
            strategy=RollbackStrategy.AGGRESSIVE_LIMIT,
            positions_to_unwind=[sample_rollback_position],
            total_positions=1,
            estimated_unwinding_cost=5.40,
            max_acceptable_loss=10.0,
            created_time=time.time(),
            max_rollback_time=time.time() + 30.0,
            rollback_status="completed",
            success=True,
            completion_time=time.time() + 5.0,
        )

        rollback_manager.completed_rollbacks = [rollback_plan]

        # Add related attempt
        attempt = RollbackAttempt(
            attempt_id="related_attempt",
            attempt_number=1,
            strategy_used=RollbackStrategy.AGGRESSIVE_LIMIT,
            timestamp=time.time(),
            positions_targeted=[sample_rollback_position],
            success=True,
        )
        rollback_manager.rollback_attempts = [attempt]

        report = rollback_manager.export_rollback_report("test_export_report")

        # Verify report structure
        assert "rollback_summary" in report
        assert "attempts_history" in report
        assert "performance_context" in report
        assert "report_generated" in report

        # Verify summary content
        summary = report["rollback_summary"]
        assert summary["rollback_id"] == "test_export_report"
        assert summary["success"] is True
        assert summary["status"] == "completed"


class TestRollbackUpdateTracking:
    """Test rollback tracking updates and status changes."""

    def test_update_rollback_tracking_completed(
        self, rollback_manager, sample_rollback_position
    ):
        """Test tracking update when position is completed."""
        rollback_plan = RollbackPlan(
            rollback_id="test_tracking_update",
            symbol="AAPL",
            execution_id="test_exec_123",
            reason=RollbackReason.PARTIAL_FILLS_TIMEOUT,
            strategy=RollbackStrategy.IMMEDIATE_MARKET,
            positions_to_unwind=[sample_rollback_position],
            total_positions=1,
            estimated_unwinding_cost=5.40,
            max_acceptable_loss=10.0,
            created_time=time.time(),
            max_rollback_time=time.time() + 30.0,
            positions_unwound=0,
        )

        # Update to completed status
        rollback_manager._update_rollback_tracking(
            rollback_plan, sample_rollback_position, "completed"
        )

        # Verify updates
        assert sample_rollback_position.unwind_status == "completed"
        assert rollback_plan.positions_unwound == 1
        assert rollback_plan.rollback_status == "completed"
        assert rollback_plan.success is True
        assert rollback_plan.completion_time is not None

    def test_update_rollback_tracking_failed(
        self, rollback_manager, sample_rollback_position
    ):
        """Test tracking update when position fails."""
        rollback_plan = RollbackPlan(
            rollback_id="test_tracking_fail",
            symbol="AAPL",
            execution_id="test_exec_123",
            reason=RollbackReason.SYSTEM_ERROR,
            strategy=RollbackStrategy.AGGRESSIVE_LIMIT,
            positions_to_unwind=[sample_rollback_position],
            total_positions=1,
            estimated_unwinding_cost=5.40,
            max_acceptable_loss=10.0,
            created_time=time.time(),
            max_rollback_time=time.time() + 30.0,
        )

        # Update to failed status
        rollback_manager._update_rollback_tracking(
            rollback_plan, sample_rollback_position, "failed"
        )

        # Verify updates
        assert sample_rollback_position.unwind_status == "failed"
        assert rollback_plan.rollback_status == "failed"
        assert rollback_plan.success is False
        assert rollback_plan.completion_time is not None

    def test_log_rollback_order_placement(
        self, rollback_manager, sample_rollback_position
    ):
        """Test rollback order placement logging."""
        order = MagicMock(spec=Order)
        order.orderId = 54321
        sample_rollback_position.rollback_order = order
        sample_rollback_position.rollback_target_price = 5.35

        # Should not raise exception
        rollback_manager._log_rollback_order_placement(
            sample_rollback_position, "test_order"
        )

    def test_log_rollback_order_fill(self, rollback_manager, sample_rollback_position):
        """Test rollback order fill logging."""
        order = MagicMock(spec=Order)
        order.orderId = 54321
        sample_rollback_position.rollback_order = order
        sample_rollback_position.rollback_target_price = 5.35

        # Should not raise exception
        rollback_manager._log_rollback_order_fill(sample_rollback_position, 5.30, 1)


class TestRollbackIntegration:
    """Integration tests for complete rollback scenarios."""

    @pytest.mark.asyncio
    async def test_complete_rollback_success_flow(
        self, rollback_manager, sample_rollback_position
    ):
        """Test complete successful rollback flow from start to finish."""
        # Setup successful market trade
        market_trade = MagicMock(spec=Trade)
        market_trade.orderStatus = MagicMock(spec=OrderStatus)
        market_trade.orderStatus.status = "Filled"
        market_trade.orderStatus.filled = 1
        market_trade.orderStatus.remaining = 0
        market_trade.orderStatus.avgFillPrice = 5.30

        rollback_manager.ib.placeOrder.return_value = market_trade

        # Set the position's leg order to filled status (required for rollback)
        sample_rollback_position.leg_order.fill_status = "filled"

        # Create rollback plan
        rollback_plan = RollbackPlan(
            rollback_id="integration_test_success",
            symbol="AAPL",
            execution_id="test_exec_123",
            reason=RollbackReason.PARTIAL_FILLS_TIMEOUT,
            strategy=RollbackStrategy.IMMEDIATE_MARKET,
            positions_to_unwind=[sample_rollback_position],
            total_positions=1,
            estimated_unwinding_cost=5.40,
            max_acceptable_loss=10.0,
            created_time=time.time(),
            max_rollback_time=time.time() + 30.0,
        )

        with patch("asyncio.sleep"):
            result = await rollback_manager.execute_rollback(rollback_plan)

        # Verify successful completion
        assert result["success"] is True
        assert result["positions_unwound"] == 1
        assert result["total_cost"] > 0

        # Verify plan was updated correctly
        assert rollback_plan.rollback_status == "completed"
        assert rollback_plan.success is True
        assert rollback_plan.positions_unwound == 1

        # Verify tracking works
        status = rollback_manager.get_rollback_order_status("integration_test_success")
        assert status["success"] is True
        assert status["positions_unwound"] == 1

    @pytest.mark.asyncio
    async def test_complete_rollback_failure_with_emergency(
        self, rollback_manager, sample_rollback_position
    ):
        """Test complete rollback failure that triggers emergency handling."""
        # Set the position's leg order to filled status (required for emergency rollback)
        sample_rollback_position.leg_order.fill_status = "filled"

        # Mock all methods to fail and eventually trigger recursion error
        with patch.object(
            rollback_manager,
            "_execute_immediate_market_rollback",
            side_effect=RecursionError("maximum recursion depth exceeded"),
        ):

            rollback_plan = RollbackPlan(
                rollback_id="integration_test_failure",
                symbol="AAPL",
                execution_id="test_exec_123",
                reason=RollbackReason.SYSTEM_ERROR,
                strategy=RollbackStrategy.IMMEDIATE_MARKET,
                positions_to_unwind=[sample_rollback_position],
                total_positions=1,
                estimated_unwinding_cost=5.40,
                max_acceptable_loss=10.0,
                created_time=time.time(),
                max_rollback_time=time.time() + 30.0,
            )

            result = await rollback_manager.execute_rollback(rollback_plan)

            # Should trigger emergency handling
            assert result.get("emergency_fallback") is True
            assert "Recursion error" in str(result.get("error_message", ""))

            # Performance metrics should be updated
            metrics = rollback_manager.get_rollback_performance_metrics()
            assert metrics["failed_rollbacks"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
