"""
End-to-end test for pause/resume functionality per ADR-003
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.Arbitrage.sfr.global_execution_lock import GlobalExecutionLock
from modules.Arbitrage.sfr.parallel_executor import ParallelLegExecutor
from modules.Arbitrage.Strategy import ArbitrageClass


class TestPauseResumeE2E:
    """End-to-end tests for pause/resume behavior"""

    @pytest.fixture
    async def setup_e2e_environment(self):
        """Setup complete environment for e2e testing"""
        # Create strategy
        strategy = ArbitrageClass()

        # Create mock IB
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True

        # Create parallel executor
        executor = ParallelLegExecutor(
            ib=mock_ib, symbol="SPY", strategy=strategy  # Pass strategy reference
        )

        # Initialize the executor to set up framework
        await executor.initialize()

        # Reset global lock state
        lock = await GlobalExecutionLock.get_instance()
        if lock.is_locked():
            await lock.force_release("test_setup")
        lock.reset_stats()

        return {
            "strategy": strategy,
            "executor": executor,
            "mock_ib": mock_ib,
            "lock": lock,
        }

    @pytest.mark.asyncio
    async def test_complete_pause_resume_flow(self, setup_e2e_environment):
        """Test the complete pause/resume flow during parallel execution"""
        env = setup_e2e_environment
        strategy = env["strategy"]
        executor = env["executor"]

        # Create mock contracts
        stock_contract = MagicMock(conId=1, symbol="SPY")
        call_contract = MagicMock(conId=2, symbol="SPY", right="C")
        put_contract = MagicMock(conId=3, symbol="SPY", right="P")

        # Track the pause/resume calls
        pause_calls = []
        resume_calls = []
        stop_calls = []

        original_pause = strategy.pause_all_other_executors
        original_resume = strategy.resume_all_executors
        original_stop = strategy.stop_all_executors

        async def track_pause(symbol):
            pause_calls.append(("pause", symbol, time.time()))
            await original_pause(symbol)

        async def track_resume():
            resume_calls.append(("resume", time.time()))
            await original_resume()

        async def track_stop():
            stop_calls.append(("stop", time.time()))
            await original_stop()

        strategy.pause_all_other_executors = track_pause
        strategy.resume_all_executors = track_resume
        strategy.stop_all_executors = track_stop

        # Mock successful execution
        with patch.object(executor.framework, "create_execution_plan") as mock_plan:
            with patch.object(
                executor.framework, "place_orders_parallel", return_value=True
            ):
                with patch.object(
                    executor, "_monitor_fills_with_timeout"
                ) as mock_monitor:

                    # Mock successful fill result
                    mock_monitor.return_value = {
                        "all_filled": True,
                        "filled_count": 3,
                        "fill_times": {"stock": 0.1, "call": 0.15, "put": 0.2},
                    }

                    # Mock execution plan
                    mock_execution_plan = MagicMock()
                    mock_execution_plan.execution_id = "test_123"
                    mock_execution_plan.stock_leg = MagicMock(avg_fill_price=100.0)
                    mock_execution_plan.call_leg = MagicMock(avg_fill_price=8.5)
                    mock_execution_plan.put_leg = MagicMock(avg_fill_price=3.25)
                    mock_plan.return_value = mock_execution_plan

                    # Execute parallel arbitrage
                    result = await executor.execute_parallel_arbitrage(
                        stock_contract=stock_contract,
                        call_contract=call_contract,
                        put_contract=put_contract,
                        stock_price=100.0,
                        call_price=8.5,
                        put_price=3.25,
                        quantity=1,
                    )

        # Verify the flow
        assert result.success is True

        # Should have called pause once
        assert len(pause_calls) == 1
        assert pause_calls[0][1] == "SPY"  # Symbol

        # Should have called stop once (successful execution)
        assert len(stop_calls) == 1

        # Should NOT have called resume (because execution was successful)
        assert len(resume_calls) == 0

        # Verify final state
        assert strategy._executor_paused is True  # Stopped state
        assert strategy.order_filled is True  # Should exit scan loops

    @pytest.mark.asyncio
    async def test_pause_resume_on_failure(self, setup_e2e_environment):
        """Test pause/resume flow when execution fails"""
        env = setup_e2e_environment
        strategy = env["strategy"]
        executor = env["executor"]

        # Create mock contracts
        stock_contract = MagicMock(conId=1, symbol="SPY")
        call_contract = MagicMock(conId=2, symbol="SPY", right="C")
        put_contract = MagicMock(conId=3, symbol="SPY", right="P")

        # Track calls
        calls_sequence = []

        async def track_pause(symbol):
            calls_sequence.append(("pause", symbol))
            await strategy.__class__.pause_all_other_executors(strategy, symbol)

        async def track_resume():
            calls_sequence.append(("resume",))
            await strategy.__class__.resume_all_executors(strategy)

        async def track_stop():
            calls_sequence.append(("stop",))
            await strategy.__class__.stop_all_executors(strategy)

        strategy.pause_all_other_executors = track_pause
        strategy.resume_all_executors = track_resume
        strategy.stop_all_executors = track_stop

        # Mock partial fill (only 1 leg fills)
        with patch.object(executor.framework, "create_execution_plan") as mock_plan:
            with patch.object(
                executor.framework, "place_orders_parallel", return_value=True
            ):
                with patch.object(
                    executor, "_monitor_fills_with_timeout"
                ) as mock_monitor:
                    with patch.object(
                        executor.rollback_manager,
                        "execute_rollback",
                        return_value={"success": True, "rolled_back_count": 1},
                    ):

                        # Mock partial fill result
                        mock_monitor.return_value = {
                            "all_filled": False,
                            "filled_count": 1,
                            "fill_times": {"stock": 0.1},
                        }

                        # Mock execution plan
                        mock_execution_plan = MagicMock()
                        mock_execution_plan.execution_id = "test_456"
                        mock_execution_plan.stock_leg = MagicMock(avg_fill_price=100.0)
                        mock_execution_plan.call_leg = MagicMock(avg_fill_price=None)
                        mock_execution_plan.put_leg = MagicMock(avg_fill_price=None)
                        mock_plan.return_value = mock_execution_plan

                        # Execute parallel arbitrage
                        result = await executor.execute_parallel_arbitrage(
                            stock_contract=stock_contract,
                            call_contract=call_contract,
                            put_contract=put_contract,
                            stock_price=100.0,
                            call_price=8.5,
                            put_price=3.25,
                            quantity=1,
                        )

        # Verify the failure flow
        assert result.success is False
        assert result.partially_filled is True

        # Should have called pause, then resume (not stop)
        assert ("pause", "SPY") in calls_sequence
        assert ("resume",) in calls_sequence
        assert ("stop",) not in calls_sequence

        # Verify final state - should be resumed
        assert strategy._executor_paused is False  # Resumed state
        assert strategy.active_parallel_symbol is None

    @pytest.mark.asyncio
    async def test_concurrent_execution_prevention_with_pause(self):
        """Test that global lock + pause prevents concurrent executions"""
        strategy = ArbitrageClass()

        # Create two executors for different symbols
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True

        executor1 = ParallelLegExecutor(ib=mock_ib, symbol="SPY", strategy=strategy)
        executor2 = ParallelLegExecutor(ib=mock_ib, symbol="QQQ", strategy=strategy)

        # Initialize both executors to set up framework
        await executor1.initialize()
        await executor2.initialize()

        # Track which executor gets to pause
        pause_calls = []

        async def track_pause(symbol):
            pause_calls.append(symbol)
            # Call the original method directly
            await strategy.__class__.pause_all_other_executors(strategy, symbol)

        strategy.pause_all_other_executors = track_pause

        # Mock the global lock to simulate timeout for the second executor
        lock_call_count = 0
        original_acquire = executor1.global_lock.acquire

        async def mock_lock_acquire(symbol, executor_id, operation, timeout=None):
            nonlocal lock_call_count
            lock_call_count += 1
            if lock_call_count == 1:
                # First call succeeds (executor1)
                return await original_acquire(symbol, executor_id, operation, timeout)
            else:
                # Second call fails with timeout (executor2)
                return False

        # Mock execution for both
        with patch.object(
            executor1.global_lock, "acquire", side_effect=mock_lock_acquire
        ):
            with patch.object(
                executor2.global_lock, "acquire", side_effect=mock_lock_acquire
            ):
                with patch.object(executor1.framework, "create_execution_plan"):
                    with patch.object(
                        executor1.framework, "place_orders_parallel", return_value=True
                    ):
                        with patch.object(
                            executor1, "_monitor_fills_with_timeout"
                        ) as mock_monitor1:
                            with patch.object(
                                executor2.framework, "create_execution_plan"
                            ):
                                with patch.object(
                                    executor2.framework,
                                    "place_orders_parallel",
                                    return_value=True,
                                ):
                                    with patch.object(
                                        executor2, "_monitor_fills_with_timeout"
                                    ) as mock_monitor2:

                                        # First executor succeeds
                                        mock_monitor1.return_value = {
                                            "all_filled": True,
                                            "filled_count": 3,
                                            "fill_times": {
                                                "stock": 0.1,
                                                "call": 0.15,
                                                "put": 0.2,
                                            },
                                        }

                                        # Second executor (won't be called due to lock timeout)
                                        mock_monitor2.return_value = {
                                            "all_filled": True,
                                            "filled_count": 3,
                                            "fill_times": {
                                                "stock": 0.1,
                                                "call": 0.15,
                                                "put": 0.2,
                                            },
                                        }

                                # Create mock contracts
                                contracts = {
                                    "stock": MagicMock(conId=1),
                                    "call": MagicMock(conId=2, right="C"),
                                    "put": MagicMock(conId=3, right="P"),
                                }

                                # Execute both simultaneously
                                task1 = asyncio.create_task(
                                    executor1.execute_parallel_arbitrage(
                                        stock_contract=contracts["stock"],
                                        call_contract=contracts["call"],
                                        put_contract=contracts["put"],
                                        stock_price=100.0,
                                        call_price=8.5,
                                        put_price=3.25,
                                        quantity=1,
                                    )
                                )

                                # Small delay to let first one acquire lock
                                await asyncio.sleep(0.01)

                                task2 = asyncio.create_task(
                                    executor2.execute_parallel_arbitrage(
                                        stock_contract=contracts["stock"],
                                        call_contract=contracts["call"],
                                        put_contract=contracts["put"],
                                        stock_price=100.0,
                                        call_price=8.5,
                                        put_price=3.25,
                                        quantity=1,
                                    )
                                )

                                results = await asyncio.gather(
                                    task1, task2, return_exceptions=True
                                )

        # One should succeed, one should fail due to lock timeout
        successes = [r for r in results if hasattr(r, "success") and r.success]
        failures = [r for r in results if hasattr(r, "success") and not r.success]

        assert len(successes) == 1, f"Expected 1 success, got {len(successes)}"
        assert len(failures) == 1, f"Expected 1 failure, got {len(failures)}"

        # Only one should have been able to pause
        assert len(pause_calls) == 1
        assert pause_calls[0] in ["SPY", "QQQ"]


if __name__ == "__main__":
    # Quick manual test
    async def manual_test():
        test_instance = TestPauseResumeE2E()
        env = await test_instance.setup_e2e_environment()
        await test_instance.test_complete_pause_resume_flow(env)
        print("✅ Manual test passed!")

    asyncio.run(manual_test())
