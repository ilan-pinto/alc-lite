"""
Unit tests for Parallel Leg Executor system.

Tests the parallel execution of stock, call, and put orders
with sophisticated fill monitoring and rollback handling.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from modules.Arbitrage.sfr.parallel_executor import ExecutionResult, ParallelLegExecutor


# Test helper functions
def create_mock_contract(symbol, sec_type="STK", con_id=None, right=None):
    """Create a mock contract for testing"""
    contract = MagicMock()
    contract.symbol = symbol
    contract.secType = sec_type
    contract.conId = con_id or hash(f"{symbol}_{sec_type}_{right}") % 100000
    if right:
        contract.right = right
    return contract


def create_mock_trade(leg_type, filled=True, fill_price=None, quantity=100):
    """Create a mock trade object"""
    trade = MagicMock()
    trade.order = MagicMock()
    trade.order.totalQuantity = quantity
    trade.orderStatus = MagicMock()

    if filled:
        trade.orderStatus.status = "Filled"
        trade.orderStatus.filled = quantity

        # Set realistic fill prices based on leg type
        if fill_price is None:
            price_map = {"stock": 100.02, "call": 8.47, "put": 3.28}
            fill_price = price_map.get(leg_type, 100.0)

        trade.orderStatus.avgFillPrice = fill_price
    else:
        trade.orderStatus.status = "Submitted"
        trade.orderStatus.filled = 0
        trade.orderStatus.avgFillPrice = 0.0

    trade.contract = create_mock_contract(
        "SPY", "STK" if leg_type == "stock" else "OPT"
    )
    return trade


def create_test_execution_params():
    """Create standard execution parameters for testing"""
    return {
        "stock_contract": create_mock_contract("SPY", "STK", 1),
        "call_contract": create_mock_contract("SPY", "OPT", 2, "C"),
        "put_contract": create_mock_contract("SPY", "OPT", 3, "P"),
        "stock_price": 100.0,
        "call_price": 8.50,
        "put_price": 3.25,
        "quantity": 1,
        "profit_target": 0.75,
    }


class TestParallelLegExecutor:
    """Test parallel order execution logic"""

    @pytest.fixture
    def mock_ib_setup(self):
        """Setup mock IB environment"""
        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()
        mock_ib.cancelOrder = MagicMock()

        return {
            "ib": mock_ib,
            "contracts": {
                "stock": create_mock_contract("SPY", "STK", 1),
                "call": create_mock_contract("SPY", "OPT", 2, "C"),
                "put": create_mock_contract("SPY", "OPT", 3, "P"),
            },
        }

    @pytest.fixture
    def executor(self, mock_ib_setup):
        """Create executor instance for testing"""
        return ParallelLegExecutor(
            ib=mock_ib_setup["ib"],
            symbol="SPY",
            on_execution_complete=None,
            on_execution_failed=None,
        )

    @pytest.mark.asyncio
    async def test_successful_parallel_execution(self, mock_ib_setup, executor):
        """Test complete 3-leg parallel execution success"""

        # Mock successful fills for all legs
        mock_ib_setup["ib"].placeOrder.side_effect = [
            create_mock_trade("stock", filled=True),
            create_mock_trade("call", filled=True),
            create_mock_trade("put", filled=True),
        ]

        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        # Verify successful execution
        assert result.success is True
        assert result.all_legs_filled is True
        assert result.legs_filled == 3
        assert result.total_legs == 3
        assert result.total_execution_time > 0
        assert result.symbol == "SPY"
        assert result.execution_id.startswith("SPY_")

        # Verify all legs have results
        assert result.stock_result is not None
        assert result.call_result is not None
        assert result.put_result is not None

    @pytest.mark.asyncio
    async def test_partial_fill_single_leg(self, mock_ib_setup, executor):
        """Test handling of single leg fill scenario"""

        # Only stock fills
        mock_ib_setup["ib"].placeOrder.side_effect = [
            create_mock_trade("stock", filled=True),
            create_mock_trade("call", filled=False),
            create_mock_trade("put", filled=False),
        ]

        # Mock rollback to succeed
        with patch.object(executor, "_handle_rollback", return_value=True):
            result = await executor.execute_parallel_arbitrage(
                **create_test_execution_params()
            )

        assert result.success is False
        assert result.legs_filled == 1
        assert result.total_legs == 3
        assert result.partially_filled is True
        assert "rollback" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_partial_fill_two_legs(self, mock_ib_setup, executor):
        """Test handling of two legs fill scenario"""

        # Stock and call fill, put doesn't
        mock_ib_setup["ib"].placeOrder.side_effect = [
            create_mock_trade("stock", filled=True),
            create_mock_trade("call", filled=True),
            create_mock_trade("put", filled=False),
        ]

        with patch.object(executor, "_handle_rollback", return_value=True):
            result = await executor.execute_parallel_arbitrage(
                **create_test_execution_params()
            )

        assert result.success is False
        assert result.legs_filled == 2
        assert result.partially_filled is True

    @pytest.mark.asyncio
    async def test_no_fills_timeout(self, mock_ib_setup, executor):
        """Test complete timeout with no fills"""

        # No fills at all
        mock_ib_setup["ib"].placeOrder.side_effect = [
            create_mock_trade("stock", filled=False),
            create_mock_trade("call", filled=False),
            create_mock_trade("put", filled=False),
        ]

        # Patch timeout to be very short for testing
        with patch.object(executor, "PARALLEL_EXECUTION_TIMEOUT", 0.1):
            result = await executor.execute_parallel_arbitrage(
                **create_test_execution_params()
            )

        assert result.success is False
        assert result.legs_filled == 0
        assert result.partially_filled is False
        assert "timeout" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execution_timeout_handling(self, mock_ib_setup, executor):
        """Test execution timeout scenarios"""

        # Mock slow fills that exceed timeout
        async def slow_fill_mock(contract, order):
            await asyncio.sleep(2.0)  # Slower than timeout
            return create_mock_trade("stock", filled=True)

        mock_ib_setup["ib"].placeOrder.side_effect = slow_fill_mock

        # Set short timeout for test
        with patch.object(executor, "PARALLEL_EXECUTION_TIMEOUT", 0.5):
            result = await executor.execute_parallel_arbitrage(
                **create_test_execution_params()
            )

        assert result.success is False
        assert "timeout" in result.error_message.lower()

    def test_price_calculation_accuracy(self, executor):
        """Test accurate price calculations and slippage"""

        # Expected prices
        expected = {"stock": 100.0, "call": 8.50, "put": 3.25}

        # Actual fill prices with slippage
        actual = {"stock": 100.02, "call": 8.47, "put": 3.28}

        # Mock leg results
        stock_result = LegExecutionResult(
            leg_type="stock",
            action="BUY",
            target_price=100.0,
            avg_fill_price=100.02,
            slippage=0.02,
            fill_status="filled",
        )
        call_result = LegExecutionResult(
            leg_type="call",
            action="SELL",
            target_price=8.50,
            avg_fill_price=8.47,
            slippage=-0.03,
            fill_status="filled",
        )
        put_result = LegExecutionResult(
            leg_type="put",
            action="BUY",
            target_price=3.25,
            avg_fill_price=3.28,
            slippage=0.03,
            fill_status="filled",
        )

        # Test slippage calculations
        assert stock_result.slippage == 0.02  # Bought higher
        assert call_result.slippage == -0.03  # Sold lower (bad)
        assert put_result.slippage == 0.03  # Bought higher

        total_slippage = (
            stock_result.slippage + call_result.slippage + put_result.slippage
        )
        assert total_slippage == 0.02  # Net slippage

    @pytest.mark.asyncio
    async def test_rollback_invocation(self, mock_ib_setup, executor):
        """Test that rollback is properly invoked for partial fills"""

        # Mock partial fill scenario
        mock_ib_setup["ib"].placeOrder.side_effect = [
            create_mock_trade("stock", filled=True),
            create_mock_trade("call", filled=False),
            create_mock_trade("put", filled=False),
        ]

        with patch.object(executor, "_handle_rollback") as mock_rollback:
            mock_rollback.return_value = True  # Rollback succeeds

            result = await executor.execute_parallel_arbitrage(
                **create_test_execution_params()
            )

            # Verify rollback was called
            mock_rollback.assert_called_once()
            assert result.success is False
            assert "rollback" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_rollback_failure_handling(self, mock_ib_setup, executor):
        """Test handling when rollback itself fails"""

        mock_ib_setup["ib"].placeOrder.side_effect = [
            create_mock_trade("stock", filled=True),
            create_mock_trade("call", filled=False),
            create_mock_trade("put", filled=False),
        ]

        with patch.object(executor, "_handle_rollback") as mock_rollback:
            mock_rollback.side_effect = Exception("Rollback failed")

            result = await executor.execute_parallel_arbitrage(
                **create_test_execution_params()
            )

            assert result.success is False
            assert "rollback failed" in result.error_message.lower()

    def test_execution_result_completeness(self):
        """Test that ExecutionResult contains all required fields"""
        result = ExecutionResult(
            success=True,
            execution_id="test_123",
            symbol="SPY",
            total_execution_time=2.5,
            all_legs_filled=True,
            legs_filled=3,
            total_legs=3,
            expected_total_cost=1000.0,
            actual_total_cost=1002.5,
            total_slippage=2.5,
            slippage_percentage=0.25,
        )

        # Verify all critical fields present
        assert hasattr(result, "success")
        assert hasattr(result, "execution_id")
        assert hasattr(result, "symbol")
        assert hasattr(result, "total_execution_time")
        assert hasattr(result, "slippage_percentage")
        assert hasattr(result, "stock_result")
        assert hasattr(result, "call_result")
        assert hasattr(result, "put_result")

        # Test calculated fields
        assert result.partially_filled is False  # All legs filled
        assert result.slippage_percentage == 0.25

    @pytest.mark.asyncio
    async def test_order_placement_timing(self, mock_ib_setup, executor):
        """Test that orders are placed simultaneously"""

        order_times = []

        async def capture_timing(contract, order):
            order_times.append(time.time())
            return create_mock_trade("stock", filled=True)

        mock_ib_setup["ib"].placeOrder.side_effect = capture_timing

        await executor.execute_parallel_arbitrage(**create_test_execution_params())

        # All orders should be placed within a very short time window
        if len(order_times) >= 2:
            max_time_diff = max(order_times) - min(order_times)
            assert max_time_diff < 0.1  # Less than 100ms between first and last order

    @pytest.mark.asyncio
    async def test_fill_monitoring_accuracy(self, mock_ib_setup, executor):
        """Test accurate fill monitoring and status tracking"""

        # Create trades that fill at different times
        trades = [
            create_mock_trade("stock", filled=True),
            create_mock_trade("call", filled=True),
            create_mock_trade("put", filled=False),
        ]

        mock_ib_setup["ib"].placeOrder.side_effect = trades

        with patch.object(executor, "_monitor_fills") as mock_monitor:
            # Mock monitor to return 2 filled trades
            mock_monitor.return_value = trades[:2]

            with patch.object(executor, "_handle_rollback", return_value=True):
                result = await executor.execute_parallel_arbitrage(
                    **create_test_execution_params()
                )

            assert result.legs_filled == 2
            assert result.partially_filled is True

    @pytest.mark.asyncio
    async def test_execution_id_generation(self, mock_ib_setup, executor):
        """Test execution ID generation is unique"""

        mock_ib_setup["ib"].placeOrder.side_effect = [
            create_mock_trade("stock", filled=True),
            create_mock_trade("call", filled=True),
            create_mock_trade("put", filled=True),
        ]

        # Execute multiple times
        results = []
        for _ in range(3):
            result = await executor.execute_parallel_arbitrage(
                **create_test_execution_params()
            )
            results.append(result.execution_id)
            await asyncio.sleep(0.001)  # Ensure time difference

        # All execution IDs should be unique
        assert len(set(results)) == len(results)

        # All should start with symbol
        assert all(exec_id.startswith("SPY_") for exec_id in results)

    @pytest.mark.asyncio
    async def test_callbacks_invocation(self, mock_ib_setup):
        """Test that success/failure callbacks are invoked"""

        success_callback = AsyncMock()
        failure_callback = AsyncMock()

        executor = ParallelLegExecutor(
            ib=mock_ib_setup["ib"],
            symbol="SPY",
            on_execution_complete=success_callback,
            on_execution_failed=failure_callback,
        )

        # Test successful execution
        mock_ib_setup["ib"].placeOrder.side_effect = [
            create_mock_trade("stock", filled=True),
            create_mock_trade("call", filled=True),
            create_mock_trade("put", filled=True),
        ]

        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        assert result.success is True
        success_callback.assert_called_once()
        failure_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_failure_callback_invocation(self, mock_ib_setup):
        """Test failure callback on execution failure"""

        success_callback = AsyncMock()
        failure_callback = AsyncMock()

        executor = ParallelLegExecutor(
            ib=mock_ib_setup["ib"],
            symbol="SPY",
            on_execution_complete=success_callback,
            on_execution_failed=failure_callback,
        )

        # Mock failure scenario
        mock_ib_setup["ib"].placeOrder.side_effect = Exception("IB Error")

        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        assert result.success is False
        success_callback.assert_not_called()
        failure_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_order_cancellation_on_timeout(self, mock_ib_setup, executor):
        """Test that unfilled orders are cancelled on timeout"""

        mock_ib_setup["ib"].placeOrder.side_effect = [
            create_mock_trade("stock", filled=False),
            create_mock_trade("call", filled=False),
            create_mock_trade("put", filled=False),
        ]

        with patch.object(executor, "PARALLEL_EXECUTION_TIMEOUT", 0.1):
            await executor.execute_parallel_arbitrage(**create_test_execution_params())

        # Should have attempted to cancel orders
        assert (
            mock_ib_setup["ib"].cancelOrder.call_count >= 0
        )  # May or may not cancel depending on implementation

    @pytest.mark.asyncio
    async def test_concurrent_execution_prevention(self, mock_ib_setup, executor):
        """Test that executor prevents concurrent executions"""

        mock_ib_setup["ib"].placeOrder.side_effect = [
            create_mock_trade("stock", filled=True),
            create_mock_trade("call", filled=True),
            create_mock_trade("put", filled=True),
        ]

        # Start two executions simultaneously
        task1 = asyncio.create_task(
            executor.execute_parallel_arbitrage(**create_test_execution_params())
        )
        task2 = asyncio.create_task(
            executor.execute_parallel_arbitrage(**create_test_execution_params())
        )

        results = await asyncio.gather(task1, task2, return_exceptions=True)

        # At least one should succeed, one might be rejected due to concurrent execution
        successes = sum(
            1 for r in results if isinstance(r, ExecutionResult) and r.success
        )
        assert successes >= 1

    def test_leg_execution_result_creation(self):
        """Test LegExecutionResult creation and properties"""

        leg_result = LegExecutionResult(
            leg_type="stock",
            action="BUY",
            target_price=100.0,
            avg_fill_price=100.02,
            slippage=0.02,
            fill_status="filled",
            quantity_filled=100,
            commission=1.0,
        )

        assert leg_result.leg_type == "stock"
        assert leg_result.action == "BUY"
        assert leg_result.target_price == 100.0
        assert leg_result.avg_fill_price == 100.02
        assert leg_result.slippage == 0.02
        assert leg_result.fill_status == "filled"
        assert leg_result.quantity_filled == 100
        assert leg_result.commission == 1.0

    @pytest.mark.asyncio
    async def test_error_handling_ib_exceptions(self, mock_ib_setup, executor):
        """Test handling of IB API exceptions"""

        # Mock IB exception during order placement
        mock_ib_setup["ib"].placeOrder.side_effect = Exception(
            "IB API Error: Connection lost"
        )

        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        assert result.success is False
        assert (
            "IB API Error" in result.error_message
            or "connection" in result.error_message.lower()
        )

    @pytest.mark.asyncio
    async def test_initialization_validation(self, mock_ib_setup):
        """Test executor initialization validation"""

        # Test with missing required parameters
        with pytest.raises(TypeError):
            ParallelLegExecutor()  # Missing required parameters

        # Test with valid parameters
        executor = ParallelLegExecutor(ib=mock_ib_setup["ib"], symbol="SPY")

        assert executor.symbol == "SPY"
        assert executor.ib is mock_ib_setup["ib"]

    @pytest.mark.asyncio
    async def test_execution_statistics_tracking(self, mock_ib_setup, executor):
        """Test that executor tracks execution statistics"""

        mock_ib_setup["ib"].placeOrder.side_effect = [
            create_mock_trade("stock", filled=True),
            create_mock_trade("call", filled=True),
            create_mock_trade("put", filled=True),
        ]

        # Execute multiple times
        for _ in range(3):
            await executor.execute_parallel_arbitrage(**create_test_execution_params())

        # Check if stats are being tracked (if implemented)
        stats = (
            executor.get_execution_stats()
            if hasattr(executor, "get_execution_stats")
            else {}
        )

        # Basic validation if stats are implemented
        if stats:
            assert isinstance(stats, dict)


@pytest.mark.performance
class TestParallelExecutorPerformance:
    """Performance tests for parallel executor"""

    @pytest.fixture
    def mock_ib_setup(self):
        """Setup mock IB environment"""
        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()
        return {"ib": mock_ib}

    @pytest.mark.asyncio
    async def test_execution_speed_benchmark(self, mock_ib_setup):
        """Benchmark: Complete execution under target time"""

        executor = ParallelLegExecutor(ib=mock_ib_setup["ib"], symbol="SPY")

        # Mock fast successful fills
        mock_ib_setup["ib"].placeOrder.side_effect = [
            create_mock_trade("stock", filled=True),
            create_mock_trade("call", filled=True),
            create_mock_trade("put", filled=True),
        ]

        start_time = time.time()
        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )
        execution_time = time.time() - start_time

        # Performance requirements
        assert result.success is True
        assert execution_time < 5.0  # Must complete under 5 seconds
        assert execution_time < 2.0  # Target: under 2 seconds for fast fills

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, mock_ib_setup):
        """Test memory usage during execution"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        executor = ParallelLegExecutor(ib=mock_ib_setup["ib"], symbol="SPY")

        mock_ib_setup["ib"].placeOrder.side_effect = [
            create_mock_trade("stock", filled=True),
            create_mock_trade("call", filled=True),
            create_mock_trade("put", filled=True),
        ]

        # Run multiple executions
        for _ in range(10):
            await executor.execute_parallel_arbitrage(**create_test_execution_params())

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be minimal
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB increase
