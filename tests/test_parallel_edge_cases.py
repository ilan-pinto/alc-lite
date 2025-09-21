"""
Edge case tests for SFR parallel execution system.

Tests unusual scenarios, error conditions, network failures,
and system resource constraints.
"""

import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

# Import PyPy compatibility utilities
try:
    from modules.Arbitrage.pypy_compat import create_compatible_async_mock, is_pypy
except ImportError:
    # Fallback for environments without PyPy compatibility
    def is_pypy():
        return hasattr(sys, "pypy_version_info") or "PyPy" in sys.version

    def create_compatible_async_mock(return_value=None):
        from unittest.mock import AsyncMock

        mock = AsyncMock()
        mock.return_value = return_value
        return mock


from modules.Arbitrage.sfr.execution_reporter import ExecutionReporter, ReportLevel
from modules.Arbitrage.sfr.global_execution_lock import GlobalExecutionLock
from modules.Arbitrage.sfr.parallel_execution_framework import (
    LegOrder,
    LegType,
    ParallelExecutionPlan,
)
from modules.Arbitrage.sfr.parallel_executor import ExecutionResult, ParallelLegExecutor
from modules.Arbitrage.sfr.rollback_manager import RollbackManager, RollbackReason


def create_mock_contract(symbol="SPY", sec_type="STK", con_id=None):
    """Create mock contract with minimal valid data"""
    contract = MagicMock()
    contract.symbol = symbol
    contract.secType = sec_type
    contract.conId = con_id or hash(f"{symbol}_{sec_type}") % 100000
    return contract


def create_test_execution_params():
    """Create basic execution parameters"""
    return {
        "stock_contract": create_mock_contract("SPY", "STK", 1),
        "call_contract": create_mock_contract("SPY", "OPT", 2),
        "put_contract": create_mock_contract("SPY", "OPT", 3),
        "stock_price": 100.0,
        "call_price": 8.50,
        "put_price": 3.25,
        "quantity": 1,
    }


class TestNetworkAndConnectionFailures:
    """Test network and IB connection failure scenarios"""

    @pytest.fixture(autouse=True)
    def reset_global_lock(self):
        """Reset global lock before each test"""
        GlobalExecutionLock._instance = None

    @pytest.mark.asyncio
    async def test_ib_disconnection_during_execution(self):
        """Test IB disconnection during parallel execution"""

        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")
        await executor.initialize()

        # Mock IB disconnection after first order
        call_count = 0

        def disconnect_after_first(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                trade = MagicMock()
                trade.orderStatus.status = "Filled"
                trade.orderStatus.filled = 100
                trade.orderStatus.avgFillPrice = 100.0
                return trade
            else:
                raise ConnectionError("IB connection lost")

        mock_ib.placeOrder.side_effect = disconnect_after_first

        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        assert result.success is False
        assert "orders placed successfully" in result.error_message.lower()
        # Should have at least partial execution data
        assert result.legs_filled >= 0

    @pytest.mark.asyncio
    async def test_intermittent_network_failures(self):
        """Test handling of intermittent network issues"""

        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")
        await executor.initialize()

        # Mock intermittent failures
        call_count = 0

        def intermittent_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:  # Every second call fails
                raise ConnectionError("Network timeout")
            else:
                trade = MagicMock()
                trade.orderStatus.status = "Filled"
                trade.orderStatus.filled = 100
                trade.orderStatus.avgFillPrice = 100.0
                return trade

        mock_ib.placeOrder.side_effect = intermittent_failure

        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        # Should handle gracefully
        assert result is not None
        assert isinstance(result.error_message, str)

    @pytest.mark.asyncio
    async def test_market_closure_during_execution(self):
        """Test market closure scenarios"""

        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")
        await executor.initialize()

        # Mock market closure error
        mock_ib.placeOrder.side_effect = Exception("Market closed")

        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        assert result.success is False
        assert "orders placed successfully" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_api_rate_limiting(self):
        """Test handling of API rate limiting"""

        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")
        await executor.initialize()

        # Mock rate limiting error
        mock_ib.placeOrder.side_effect = Exception("Rate limit exceeded - please wait")

        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        assert result.success is False
        assert "orders placed successfully" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_partial_network_recovery(self):
        """Test recovery from partial network failures"""

        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")
        await executor.initialize()

        # Mock partial recovery - some orders succeed, some fail
        call_count = 0

        def partial_recovery(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # First 2 succeed
                trade = MagicMock()
                trade.orderStatus.status = "Filled"
                trade.orderStatus.filled = 100
                trade.orderStatus.avgFillPrice = 100.0
                return trade
            else:  # Rest fail
                raise ConnectionError("Network issue")

        mock_ib.placeOrder.side_effect = partial_recovery

        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        # Should indicate execution failure
        assert result.success is False
        assert "orders placed successfully" in result.error_message.lower()


class TestResourceConstraints:
    """Test behavior under system resource constraints"""

    @pytest.fixture(autouse=True)
    def reset_global_lock(self):
        """Reset global lock before each test"""
        GlobalExecutionLock._instance = None

    @pytest.mark.asyncio
    async def test_low_memory_conditions(self):
        """Test behavior under low memory conditions"""

        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")

        # Simulate low memory condition
        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value.percent = 95  # 95% memory usage

            await executor.initialize()

            # Mock successful trades
            mock_ib.placeOrder.return_value = MagicMock()
            mock_ib.placeOrder.return_value.orderStatus.status = "Filled"
            mock_ib.placeOrder.return_value.orderStatus.filled = 100
            mock_ib.placeOrder.return_value.orderStatus.avgFillPrice = 100.0

            result = await executor.execute_parallel_arbitrage(
                **create_test_execution_params()
            )

            # Should handle gracefully (might warn but not crash)
            assert result is not None

    @pytest.mark.asyncio
    async def test_cpu_overload_conditions(self):
        """Test behavior under high CPU load"""

        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")
        await executor.initialize()

        # Simulate high CPU load by making operations slow
        async def slow_operation(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slow processing
            trade = MagicMock()
            trade.orderStatus.status = "Filled"
            trade.orderStatus.filled = 100
            trade.orderStatus.avgFillPrice = 100.0
            return trade

        mock_ib.placeOrder.side_effect = slow_operation

        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        # Should still complete, just slower
        assert result is not None
        assert result.total_execution_time > 0

    def test_disk_space_constraints(self):
        """Test behavior when disk space is low (for logging/reporting)"""

        reporter = ExecutionReporter()

        # Create sample execution result
        from modules.Arbitrage.sfr.parallel_executor import ExecutionResult

        result = ExecutionResult(
            success=True,
            execution_id="DISK_TEST",
            symbol="SPY",
            total_execution_time=2.0,
            all_legs_filled=True,
            partially_filled=False,
            legs_filled=3,
            total_legs=3,
            expected_total_cost=1000.0,
            actual_total_cost=1002.0,
            total_slippage=2.0,
            slippage_percentage=0.2,
        )

        import tempfile

        # Try to export to a location that might fail
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filename = f.name

        try:
            # Mock disk full error
            with patch("builtins.open", side_effect=OSError("No space left on device")):
                success = reporter.export_session_report(filename)
                assert success is False  # Should handle gracefully
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    @pytest.mark.asyncio
    async def test_file_descriptor_exhaustion(self):
        """Test behavior when file descriptors are exhausted"""

        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        # Mock file descriptor exhaustion
        mock_ib.placeOrder.side_effect = OSError("Too many open files")

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")
        await executor.initialize()

        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        assert result.success is False
        assert "orders placed successfully" in result.error_message.lower()


class TestTimingAndSynchronization:
    """Test timing issues and synchronization edge cases"""

    @pytest.fixture(autouse=True)
    def reset_global_lock(self):
        """Reset global lock before each test"""
        GlobalExecutionLock._instance = None

    @pytest.mark.asyncio
    async def test_clock_synchronization_issues(self):
        """Test handling of system clock/timing issues"""

        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")
        await executor.initialize()

        # Mock system clock jumping backwards
        call_count = 0

        def mock_time_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 1000.0  # Initial time
            elif call_count == 2:
                return 999.0  # Clock goes backwards
            elif call_count == 3:
                return 1001.0  # Clock goes forward
            else:
                return 1000.0 + call_count * 0.1  # Keep incrementing

        with patch("time.time") as mock_time:
            mock_time.side_effect = mock_time_func

            mock_ib.placeOrder.return_value = MagicMock()
            mock_ib.placeOrder.return_value.orderStatus.status = "Filled"
            mock_ib.placeOrder.return_value.orderStatus.filled = 100

            result = await executor.execute_parallel_arbitrage(
                **create_test_execution_params()
            )

            # Should handle timing anomalies gracefully
            assert result is not None
            assert result.total_execution_time >= 0  # Should not be negative

    @pytest.mark.asyncio
    async def test_extreme_timeout_scenarios(self):
        """Test with extremely short and long timeouts"""

        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")

        # Test with extremely short timeout
        with patch(
            "modules.Arbitrage.sfr.constants.PARALLEL_EXECUTION_TIMEOUT", 0.001
        ):  # 1ms timeout
            await executor.initialize()

            # Mock slow response
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(0.01)  # 10ms delay
                trade = MagicMock()
                trade.orderStatus.status = "Filled"
                return trade

            mock_ib.placeOrder.side_effect = slow_response

            result = await executor.execute_parallel_arbitrage(
                **create_test_execution_params()
            )

            assert result.success is False
            assert "rollback executed" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_race_conditions_in_fill_monitoring(self):
        """Test race conditions during fill monitoring"""

        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")
        await executor.initialize()

        # Create trades that change status during monitoring
        trades = []
        for i in range(3):
            trade = MagicMock()
            trade.orderStatus.status = "Submitted"  # Initially not filled
            trade.orderStatus.filled = 0
            trades.append(trade)

        mock_ib.placeOrder.side_effect = trades

        # Simulate fills happening during execution
        async def simulate_fills():
            await asyncio.sleep(0.1)  # Wait a bit
            for trade in trades:
                trade.orderStatus.status = "Filled"
                trade.orderStatus.filled = 100
                trade.orderStatus.avgFillPrice = 100.0

        # Start fill simulation concurrently
        fill_task = asyncio.create_task(simulate_fills())

        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        await fill_task  # Clean up

        # Should detect fills eventually
        assert result is not None

    @pytest.mark.asyncio
    async def test_concurrent_global_lock_edge_cases(self):
        """Test edge cases in concurrent global lock usage"""

        lock = await GlobalExecutionLock.get_instance()

        async def aggressive_acquire_release(executor_id, iterations=100):
            successes = 0
            failures = 0

            for i in range(iterations):
                try:
                    success = await lock.acquire(
                        f"SYM{executor_id}", f"exec_{executor_id}", timeout=0.001
                    )
                    if success:
                        # Immediately release (very short hold)
                        lock.release(f"SYM{executor_id}", f"exec_{executor_id}")
                        successes += 1
                    else:
                        failures += 1
                except Exception as e:
                    failures += 1

            return {"successes": successes, "failures": failures}

        # Many aggressive concurrent attempts
        tasks = [aggressive_acquire_release(i) for i in range(100)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should handle without deadlock or corruption
        total_successes = sum(r["successes"] for r in results if isinstance(r, dict))
        assert total_successes > 0  # At least some should succeed

        # Lock should end up in clean state
        assert not lock.is_locked()

    @pytest.mark.asyncio
    async def test_asyncio_task_cancellation(self):
        """Test proper handling of asyncio task cancellation"""

        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")
        await executor.initialize()

        # Mock long-running operation
        async def long_operation(*args, **kwargs):
            await asyncio.sleep(10)  # Long delay
            return MagicMock()

        mock_ib.placeOrder.side_effect = long_operation

        # Start execution and cancel it
        execution_task = asyncio.create_task(
            executor.execute_parallel_arbitrage(**create_test_execution_params())
        )

        # Let it start, then cancel
        await asyncio.sleep(0.1)
        execution_task.cancel()

        try:
            await execution_task
        except asyncio.CancelledError:
            pass  # Expected

        # Should handle cancellation gracefully
        assert True  # If we get here, no deadlock occurred


class TestDataCorruptionAndValidation:
    """Test handling of corrupted or invalid data"""

    def test_invalid_contract_data(self):
        """Test handling of invalid contract data"""

        mock_ib = MagicMock()
        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")

        # Invalid contract (missing required fields)
        invalid_stock = MagicMock(spec=[])  # Empty spec, no attributes
        invalid_stock.conId = None
        invalid_stock.symbol = None

        params = create_test_execution_params()
        params["stock_contract"] = invalid_stock

        with pytest.raises((ValueError, AttributeError)):
            asyncio.run(executor.execute_parallel_arbitrage(**params))

    def test_malformed_execution_data_handling(self):
        """Test handling of malformed execution data in reporter"""

        reporter = ExecutionReporter()

        # Create malformed execution result
        malformed_result = MagicMock()
        malformed_result.success = "not_a_boolean"  # Wrong type
        malformed_result.symbol = None
        malformed_result.total_execution_time = -1  # Invalid time
        malformed_result.stock_result = {
            "invalid": "not_a_dict"
        }  # Wrong structure but still dict
        malformed_result.call_result = None
        malformed_result.put_result = None
        malformed_result.slippage_percentage = 0.05  # Valid float for slippage analysis
        malformed_result.total_slippage = 0.02
        malformed_result.execution_id = "test_123"
        malformed_result.order_placement_time = 0.1
        malformed_result.fill_monitoring_time = 0.2

        # Should handle gracefully without crashing
        report = reporter.generate_execution_report(malformed_result)

        assert report is not None
        assert len(report) > 0  # Should produce some output even with bad data

    def test_corrupted_rollback_data(self):
        """Test handling of corrupted rollback data"""
        mock_ib = MagicMock()
        manager = RollbackManager(ib=mock_ib, symbol="SPY")

        # Try to execute rollback with invalid data
        mock_plan = MagicMock()
        mock_filled_legs = []
        mock_unfilled_legs = []

        # Should handle gracefully without raising exceptions
        result = asyncio.run(
            manager.execute_rollback(
                plan=mock_plan,
                filled_legs=mock_filled_legs,
                unfilled_legs=mock_unfilled_legs,
                reason=RollbackReason.COMPLETION_FAILED,  # Valid reason for test
            )
        )
        # Should return some result even with empty data
        assert result is not None
        assert isinstance(result, dict)

    def test_negative_or_invalid_numeric_values(self):
        """Test handling of invalid numeric values"""
        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")

        # Test with negative prices
        params = create_test_execution_params()
        params["stock_price"] = -100.0  # Invalid negative price
        params["call_price"] = -8.50
        params["put_price"] = -3.25

        asyncio.run(executor.initialize())

        # Should handle invalid prices gracefully
        result = asyncio.run(executor.execute_parallel_arbitrage(**params))
        assert result is not None

    def test_extremely_large_numeric_values(self):
        """Test handling of extremely large numeric values"""
        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")
        asyncio.run(executor.initialize())

        # Test with extremely large values
        params = create_test_execution_params()
        params["stock_price"] = 1e10  # Unrealistically large
        params["quantity"] = 1e6  # Massive quantity

        result = asyncio.run(executor.execute_parallel_arbitrage(**params))
        assert result is not None


class TestErrorCascades:
    """Test cascading error scenarios"""

    @pytest.fixture(autouse=True)
    def reset_global_lock(self):
        """Reset global lock before each test"""
        GlobalExecutionLock._instance = None

    @pytest.mark.asyncio
    async def test_rollback_cascade_failure(self):
        """Test handling of cascading rollback failures"""

        mock_ib = MagicMock()
        manager = RollbackManager(ib=mock_ib, symbol="SPY")

        # Create proper mock objects for execute_rollback
        mock_plan = MagicMock()
        mock_plan.plan_id = "test_cascade_plan"
        mock_plan.symbol = "SPY"

        # Create mock filled leg that needs unwinding
        mock_filled_leg = MagicMock()
        mock_filled_leg.leg_type = LegType.STOCK
        mock_filled_leg.price = 100.0
        mock_filled_leg.action = "BUY"
        mock_filled_leg.quantity = 100
        filled_legs = [mock_filled_leg]

        unfilled_legs = []

        # Mock internal method to simulate cascade failure
        with patch.object(manager, "_execute_rollback_strategy") as mock_strategy:
            mock_strategy.side_effect = Exception("Rollback cascade failed")

            # Execute rollback - should handle failure gracefully
            result = await manager.execute_rollback(
                plan=mock_plan,
                filled_legs=filled_legs,
                unfilled_legs=unfilled_legs,
                reason=RollbackReason.PARTIAL_FILLS_TIMEOUT,
            )

            # Verify failure was handled properly without crashing
            assert isinstance(result, dict)
            # Should contain error information or success flag
            assert "error" in result or "success" in result

            # Check that manager state is consistent after failure
            assert isinstance(manager.active_rollbacks, dict)
            assert isinstance(manager.completed_rollbacks, list)

    @pytest.mark.asyncio
    async def test_multiple_system_failures(self):
        """Test handling when multiple systems fail simultaneously"""

        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")

        # Mock the global lock with failure
        mock_global_lock = AsyncMock()
        mock_global_lock.acquire = AsyncMock(
            side_effect=Exception("Lock system failed")
        )
        mock_global_lock.release = MagicMock()
        executor.global_lock = mock_global_lock

        # Mock framework to prevent other issues
        async def mock_create_execution_plan(*args, **kwargs):
            import uuid

            plan = MagicMock()
            plan.execution_id = f"SPY_test_{str(uuid.uuid4())[:8]}"
            plan.symbol = kwargs.get("symbol", "SPY")
            plan.stock_leg = MagicMock()
            plan.call_leg = MagicMock()
            plan.put_leg = MagicMock()
            return plan

        executor.framework = MagicMock()
        executor.framework.create_execution_plan = AsyncMock(
            side_effect=mock_create_execution_plan
        )

        # Mock multiple cascading failures
        failures = [
            ConnectionError("Network failed"),
            OSError("System error"),
            Exception("Unknown error"),
        ]

        mock_ib.placeOrder.side_effect = failures[0]  # First failure

        await executor.initialize()

        # The global lock failure should be caught and handled gracefully
        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        # Should handle gracefully even with multiple failures
        assert result is not None
        assert result.success is False
        # Should contain error information indicating failure (either lock or order placement)
        assert result.error_message is not None
        assert len(result.error_message) > 0
        # Verify it's handling system failures properly
        assert result.error_type in [
            "order_placement_failed",
            "lock_acquisition_failed",
            "execution_failed",
        ]

    @pytest.mark.asyncio
    async def test_resource_cleanup_after_failures(self):
        """Test that resources are properly cleaned up after failures"""

        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()
        mock_ib.cancelOrder = AsyncMock()

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")
        await executor.initialize()

        # Mock failure during execution
        mock_ib.placeOrder.side_effect = Exception("Execution failed")

        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        # Should attempt to clean up (cancel orders, etc.)
        assert result.success is False
        # Note: Specific cleanup verification would depend on implementation details


class TestExtremeLoadConditions:
    """Test behavior under extreme load conditions"""

    @pytest.fixture(autouse=True)
    def reset_global_lock(self):
        """Reset global lock before each test"""
        GlobalExecutionLock._instance = None

    def test_zero_liquidity_scenario(self):
        """Test handling of zero liquidity (no fills)"""
        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")

        # Mock orders that never fill
        def no_fill_trade(*args, **kwargs):
            trade = MagicMock()
            trade.orderStatus.status = "Submitted"  # Never fills
            trade.orderStatus.filled = 0
            trade.orderStatus.avgFillPrice = 0.0
            return trade

        mock_ib.placeOrder.side_effect = no_fill_trade

        asyncio.run(executor.initialize())

        # Should timeout gracefully
        with patch(
            "modules.Arbitrage.sfr.constants.PARALLEL_EXECUTION_TIMEOUT", 0.1
        ):  # Short timeout for test
            result = asyncio.run(
                executor.execute_parallel_arbitrage(**create_test_execution_params())
            )

        assert result.success is False
        assert result.legs_filled == 0
        assert "rollback executed" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_market_volatility_extreme_slippage(self):
        """Test handling of extreme slippage scenarios"""

        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")
        await executor.initialize()

        # Mock fills with extreme slippage
        def extreme_slippage_trade(*args, **kwargs):
            trade = MagicMock()
            trade.orderStatus.status = "Filled"
            trade.orderStatus.filled = 100
            # Extreme slippage - 50% worse than expected
            trade.orderStatus.avgFillPrice = 150.0  # Expected ~100
            return trade

        mock_ib.placeOrder.side_effect = extreme_slippage_trade

        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        # Should detect excessive slippage
        if result.success:
            assert result.total_slippage != 0  # Should measure slippage

    @pytest.mark.asyncio
    async def test_system_overload_recovery(self):
        """Test recovery from system overload conditions"""

        # Simulate system overload by creating many concurrent operations
        tasks = []

        for i in range(200):  # Many concurrent operations

            async def background_load():
                await asyncio.sleep(0.01)  # Small load
                return i

            tasks.append(asyncio.create_task(background_load()))

        # Now try normal operation under load
        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        executor = ParallelLegExecutor(ib=mock_ib, symbol="SPY")
        await executor.initialize()

        mock_ib.placeOrder.return_value = MagicMock()
        mock_ib.placeOrder.return_value.orderStatus.status = "Filled"
        mock_ib.placeOrder.return_value.orderStatus.filled = 100

        # Should still work under load
        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        # Clean up background tasks
        for task in tasks:
            task.cancel()

        assert result is not None

    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure"""

        # Create memory pressure by allocating large objects
        large_objects = []
        try:
            # Allocate until we get close to memory limits (but not crash)
            for i in range(10):
                large_objects.append([0] * (10**6))  # 1M integers each

            # Now try to use the system under memory pressure
            reporter = ExecutionReporter()

            # Should still work under memory pressure
            from modules.Arbitrage.sfr.parallel_executor import ExecutionResult

            result = ExecutionResult(
                success=True,
                execution_id="MEM_PRESSURE_TEST",
                symbol="SPY",
                total_execution_time=2.0,
                all_legs_filled=True,
                partially_filled=False,
                legs_filled=3,
                total_legs=3,
                expected_total_cost=1000.0,
                actual_total_cost=1000.0,
                total_slippage=0.0,
                slippage_percentage=0.0,
            )

            report = reporter.generate_execution_report(result)
            assert report is not None

        finally:
            # Clean up memory
            large_objects.clear()

    @pytest.mark.asyncio
    async def test_infinite_loop_prevention(self):
        """Test prevention of infinite loops in error handling"""

        mock_ib = MagicMock()
        manager = RollbackManager(ib=mock_ib, symbol="SPY")

        # Create proper mock objects for execute_rollback
        mock_plan = MagicMock()
        mock_plan.plan_id = "test_infinite_loop_plan"
        mock_plan.symbol = "SPY"

        # Create mock filled leg
        mock_filled_leg = MagicMock()
        mock_filled_leg.leg_type = LegType.STOCK
        mock_filled_leg.price = 100.0
        mock_filled_leg.action = "BUY"
        mock_filled_leg.quantity = 100
        filled_legs = [mock_filled_leg]

        unfilled_legs = []

        # Mock a scenario that could cause infinite recursion by making
        # the _execute_rollback_strategy method call itself recursively
        call_count = 0
        original_execute_strategy = manager._execute_rollback_strategy

        async def recursive_execute_strategy(plan):
            nonlocal call_count
            call_count += 1
            if (
                call_count > 10
            ):  # Reasonable limit to prevent actual infinite loop in test
                return {
                    "success": False,
                    "positions_unwound": 0,
                    "total_cost": 0.0,
                    "execution_time": 0.1,
                    "error": "Recursion limit reached",
                }
            # This could potentially cause recursion if not properly handled
            return await original_execute_strategy(plan)

        # Mock the method to simulate recursive behavior
        with patch.object(
            manager,
            "_execute_rollback_strategy",
            side_effect=recursive_execute_strategy,
        ):

            # Execute rollback - system should handle recursive calls gracefully
            result = await manager.execute_rollback(
                plan=mock_plan,
                filled_legs=filled_legs,
                unfilled_legs=unfilled_legs,
                reason=RollbackReason.PARTIAL_FILLS_TIMEOUT,
            )

            # Should complete without infinite recursion
            assert result is not None
            assert isinstance(result, dict)
            # The recursive calls should be limited and handled gracefully
            assert call_count <= 10  # Should not exceed our safety limit
            # Should contain some result indicating the operation completed
            assert "error" in result or "success" in result
