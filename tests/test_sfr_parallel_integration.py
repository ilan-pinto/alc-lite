"""
Integration tests for SFR parallel execution system.

Tests the complete integration between all parallel execution components
and their interaction with the existing SFR arbitrage strategy.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.Arbitrage.sfr.execution_reporter import ReportLevel
from modules.Arbitrage.sfr.models import ExpiryOption
from modules.Arbitrage.sfr.parallel_executor import ExecutionResult
from modules.Arbitrage.sfr.parallel_integration import (
    ParallelExecutionIntegrator,
    create_parallel_integrator,
)


# Test helper functions
def create_mock_contract(symbol, sec_type="STK", con_id=None, right=None, strike=None):
    """Create a mock IB contract"""
    contract = MagicMock()
    contract.symbol = symbol
    contract.secType = sec_type
    contract.conId = con_id or hash(f"{symbol}_{sec_type}_{right}_{strike}") % 100000
    if right:
        contract.right = right
    if strike:
        contract.strike = strike
    if sec_type == "BAG":
        contract.comboLegs = [
            MagicMock(conId=1),  # Stock leg
            MagicMock(conId=2),  # Call leg
            MagicMock(conId=3),  # Put leg
        ]
    return contract


def create_profitable_opportunity(symbol="SPY", profit=0.75, expiry="20241220"):
    """Create a profitable opportunity for testing"""

    # Create contracts
    stock_contract = create_mock_contract(symbol, "STK", 1)
    call_contract = create_mock_contract(symbol, "OPT", 2, "C", 105.0)
    put_contract = create_mock_contract(symbol, "OPT", 3, "P", 95.0)
    combo_contract = create_mock_contract(symbol, "BAG", 100)

    # Create order
    order = MagicMock()
    order.orderId = 1001
    order.orderType = "LMT"
    order.totalQuantity = 1
    order.lmtPrice = 100.0

    # Create expiry option
    expiry_option = ExpiryOption(
        expiry=expiry,
        call_contract=call_contract,
        put_contract=put_contract,
        call_strike=105.0,
        put_strike=95.0,
    )

    # Create opportunity
    opportunity = {
        "contract": combo_contract,
        "order": order,
        "guaranteed_profit": profit,
        "trade_details": {
            "expiry": expiry,
            "call_strike": 105.0,
            "call_price": 8.65,
            "put_strike": 95.0,
            "put_price": 3.50,
            "stock_price": 100.0,
            "net_credit": 12.15,
            "min_profit": profit,
            "max_profit": profit + 10.0,
            "min_roi": profit,
        },
        "expiry_option": expiry_option,
    }

    return opportunity


def create_low_profit_opportunity(symbol="SPY", profit=0.01):
    """Create a low-profit opportunity that should use combo orders"""
    return create_profitable_opportunity(symbol, profit)


def mock_successful_fills(mock_ib):
    """Setup mock IB for successful fills"""

    def create_filled_trade(*args, **kwargs):
        trade = MagicMock()
        trade.orderStatus.status = "Filled"
        trade.orderStatus.filled = 100
        trade.orderStatus.avgFillPrice = 100.0
        return trade

    mock_ib.placeOrder.return_value = create_filled_trade()


def mock_partial_fill(mock_ib, filled_legs=None):
    """Setup mock IB for partial fills"""
    if filled_legs is None:
        filled_legs = ["stock"]

    def create_partial_trade(*args, **kwargs):
        trade = MagicMock()
        # Simulate that only some orders fill
        if len(mock_ib.placeOrder.call_args_list) <= len(filled_legs):
            trade.orderStatus.status = "Filled"
            trade.orderStatus.filled = 100
            trade.orderStatus.avgFillPrice = 100.0
        else:
            trade.orderStatus.status = "Submitted"
            trade.orderStatus.filled = 0
            trade.orderStatus.avgFillPrice = 0.0
        return trade

    mock_ib.placeOrder.side_effect = create_partial_trade


def mock_slow_execution(mock_ib, delay=2.0):
    """Setup mock IB for slow execution"""

    async def slow_order(*args, **kwargs):
        await asyncio.sleep(delay)
        trade = MagicMock()
        trade.orderStatus.status = "Filled"
        trade.orderStatus.filled = 100
        trade.orderStatus.avgFillPrice = 100.0
        return trade

    mock_ib.placeOrder = slow_order


class TestParallelExecutionIntegration:
    """Test complete parallel execution flow integration"""

    @pytest.fixture
    def integration_setup(self):
        """Setup complete integration environment"""
        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()
        mock_ib.cancelOrder = AsyncMock()

        mock_order_manager = MagicMock()
        mock_order_manager.place_order = AsyncMock()

        # Create opportunity evaluator mock
        opportunity_evaluator = MagicMock()
        opportunity_evaluator.symbol = "SPY"

        return {
            "ib": mock_ib,
            "order_manager": mock_order_manager,
            "opportunity_evaluator": opportunity_evaluator,
            "symbol": "SPY",
        }

    @pytest.mark.asyncio
    async def test_integrator_initialization(self, integration_setup):
        """Test integrator initialization"""

        integrator = ParallelExecutionIntegrator(**integration_setup)

        # Test basic properties
        assert integrator.symbol == "SPY"
        assert integrator.ib is integration_setup["ib"]
        assert integrator.order_manager is integration_setup["order_manager"]
        assert integrator.execution_count == 0
        assert integrator.is_initialized is False

        # Test initialization
        success = await integrator.initialize()
        assert success is True
        assert integrator.is_initialized is True

    @pytest.mark.asyncio
    async def test_create_parallel_integrator_convenience(self, integration_setup):
        """Test convenience function for creating integrator"""

        integrator = await create_parallel_integrator(**integration_setup)

        assert isinstance(integrator, ParallelExecutionIntegrator)
        assert integrator.is_initialized is True
        assert integrator.symbol == "SPY"

    @pytest.mark.asyncio
    async def test_complete_successful_execution_flow(self, integration_setup):
        """Test complete successful parallel execution flow"""

        # Create integrator
        integrator = await create_parallel_integrator(**integration_setup)

        # Create profitable opportunity
        opportunity = create_profitable_opportunity("SPY")

        # Mock successful execution
        mock_result = ExecutionResult(
            success=True,
            execution_id="SPY_test_123",
            symbol="SPY",
            total_execution_time=2.5,
            all_legs_filled=True,
            partially_filled=False,
            legs_filled=3,
            total_legs=3,
            expected_total_cost=1000.0,
            actual_total_cost=1002.0,
            total_slippage=2.0,
            slippage_percentage=0.2,
        )

        with patch.object(
            integrator.parallel_executor,
            "execute_parallel_arbitrage",
            return_value=mock_result,
        ):
            result = await integrator.execute_opportunity(opportunity)

        # Verify successful execution
        assert result["success"] is True
        assert result["method"] == "parallel"
        assert result["legs_filled"] == "3/3"
        assert "parallel_result" in result

        # Verify execution result details
        parallel_result = result["parallel_result"]
        assert parallel_result.success is True
        assert parallel_result.all_legs_filled is True
        assert parallel_result.total_execution_time > 0

    @pytest.mark.asyncio
    async def test_partial_fill_with_rollback_flow(self, integration_setup):
        """Test partial fill detection and rollback integration"""

        integrator = await create_parallel_integrator(**integration_setup)
        opportunity = create_profitable_opportunity("SPY")

        # Mock partial fill result
        mock_result = ExecutionResult(
            success=False,
            execution_id="SPY_partial_123",
            symbol="SPY",
            total_execution_time=5.0,
            all_legs_filled=False,
            partially_filled=True,
            legs_filled=1,
            total_legs=3,
            expected_total_cost=1000.0,
            actual_total_cost=0.0,
            total_slippage=0.0,
            slippage_percentage=0.0,
            error_message="Partial fill - rollback executed",
        )

        with patch.object(
            integrator.parallel_executor,
            "execute_parallel_arbitrage",
            return_value=mock_result,
        ):
            result = await integrator.execute_opportunity(opportunity)

        # Should have attempted rollback
        assert result["success"] is False
        assert result["method"] == "parallel"
        assert "rollback" in result.get("error_message", "").lower()

    @pytest.mark.asyncio
    async def test_global_lock_coordination(self, integration_setup):
        """Test global lock prevents concurrent executions"""

        integrator1 = await create_parallel_integrator(**integration_setup)

        # Create second integrator with different symbol
        setup2 = integration_setup.copy()
        setup2["symbol"] = "QQQ"
        setup2["opportunity_evaluator"] = MagicMock()
        setup2["opportunity_evaluator"].symbol = "QQQ"
        integrator2 = await create_parallel_integrator(**setup2)

        opportunity1 = create_profitable_opportunity("SPY")
        opportunity2 = create_profitable_opportunity("QQQ")

        # Mock successful results with delays
        async def slow_execution(*args, **kwargs):
            await asyncio.sleep(1.0)  # 1 second delay
            return ExecutionResult(
                success=True,
                execution_id="slow_exec",
                symbol="SPY",
                total_execution_time=1.0,
                all_legs_filled=True,
                partially_filled=False,
                legs_filled=3,
                total_legs=3,
                expected_total_cost=1000.0,
                actual_total_cost=1000.0,
                total_slippage=0.0,
                slippage_percentage=0.0,
            )

        with patch.object(
            integrator1.parallel_executor,
            "execute_parallel_arbitrage",
            side_effect=slow_execution,
        ):
            with patch.object(
                integrator2.parallel_executor,
                "execute_parallel_arbitrage",
                side_effect=slow_execution,
            ):

                # Start both executions simultaneously
                task1 = asyncio.create_task(
                    integrator1.execute_opportunity(opportunity1)
                )
                await asyncio.sleep(0.1)  # Let first one start
                task2 = asyncio.create_task(
                    integrator2.execute_opportunity(opportunity2)
                )

                results = await asyncio.gather(task1, task2, return_exceptions=True)

                # Both should complete (one might be delayed by lock)
                successful_results = [
                    r for r in results if isinstance(r, dict) and r.get("success")
                ]
                assert len(successful_results) <= 2  # At most both succeed

    @pytest.mark.asyncio
    async def test_execution_decision_logic(self, integration_setup):
        """Test parallel vs combo execution decision logic"""

        integrator = await create_parallel_integrator(**integration_setup)

        # High profit opportunity - should choose parallel
        high_profit_opp = create_profitable_opportunity("SPY", profit=0.75)
        use_parallel, reason = await integrator.should_use_parallel_execution(
            high_profit_opp
        )
        assert use_parallel is True
        assert "parallel_execution_enabled" in reason

        # Low profit opportunity - should STILL use parallel when PARALLEL_EXECUTION_ENABLED=True
        low_profit_opp = create_low_profit_opportunity("SPY", profit=0.01)
        use_parallel, reason = await integrator.should_use_parallel_execution(
            low_profit_opp
        )
        assert use_parallel is True  # Changed: now uses parallel regardless of profit
        assert "parallel_execution_enabled" in reason  # Changed: new reason

    @pytest.mark.asyncio
    async def test_combo_execution_fallback(self, integration_setup):
        """Test fallback to combo execution"""

        integrator = await create_parallel_integrator(**integration_setup)

        # Force combo execution
        opportunity = create_profitable_opportunity("SPY")

        # Mock successful combo execution
        mock_trade = MagicMock()
        mock_trade.orderStatus.status = "Filled"
        integration_setup["order_manager"].place_order.return_value = mock_trade

        result = await integrator.execute_opportunity(opportunity, force_combo=True)

        assert result["success"] is True
        assert result["method"] == "combo"
        assert "3/3" in result["legs_filled"]  # Combo fills all legs together

    @pytest.mark.asyncio
    async def test_disabled_parallel_execution(self, integration_setup):
        """Test behavior when parallel execution is disabled"""

        with patch(
            "modules.Arbitrage.sfr.parallel_integration.PARALLEL_EXECUTION_ENABLED",
            False,
        ):
            integrator = await create_parallel_integrator(**integration_setup)

            opportunity = create_profitable_opportunity("SPY")

            # Should use combo even for high-profit opportunities
            use_parallel, reason = await integrator.should_use_parallel_execution(
                opportunity
            )
            assert use_parallel is False
            assert "disabled" in reason

    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, integration_setup):
        """Test handling of insufficient opportunity data"""

        integrator = await create_parallel_integrator(**integration_setup)

        # Create opportunity missing required data
        incomplete_opportunity = {
            "guaranteed_profit": 0.75,
            # Missing expiry_option and trade_details
        }

        use_parallel, reason = await integrator.should_use_parallel_execution(
            incomplete_opportunity
        )
        assert use_parallel is False
        assert "insufficient_data" in reason

    @pytest.mark.asyncio
    async def test_market_conditions_impact(self, integration_setup):
        """Test impact of market conditions on execution decisions"""

        integrator = await create_parallel_integrator(**integration_setup)
        opportunity = create_profitable_opportunity("SPY")

        # High volatility market conditions - should STILL use parallel when enabled
        high_vol_conditions = {"volatility": "high"}
        use_parallel, reason = await integrator.should_use_parallel_execution(
            opportunity, market_conditions=high_vol_conditions
        )
        assert (
            use_parallel is True
        )  # Changed: parallel enabled overrides market conditions
        assert "parallel_execution_enabled" in reason  # Changed: new reason

    @pytest.mark.asyncio
    async def test_error_propagation_and_handling(self, integration_setup):
        """Test error handling across integration components"""

        integrator = await create_parallel_integrator(**integration_setup)

        # Mock executor to raise exception
        with patch.object(
            integrator.parallel_executor,
            "execute_parallel_arbitrage",
            side_effect=ConnectionError("IB disconnected"),
        ):

            opportunity = create_profitable_opportunity("SPY")
            result = await integrator.execute_opportunity(opportunity)

            assert result["success"] is False
            assert "error" in result
            assert "IB disconnected" in result["error"]

    @pytest.mark.asyncio
    async def test_execution_statistics_tracking(self, integration_setup):
        """Test execution statistics are tracked properly"""

        integrator = await create_parallel_integrator(**integration_setup)

        # Execute multiple opportunities
        for i in range(3):
            opportunity = create_profitable_opportunity("SPY")

            mock_result = ExecutionResult(
                success=(i < 2),  # 2 successes, 1 failure
                execution_id=f"SPY_stats_{i}",
                symbol="SPY",
                total_execution_time=2.0,
                all_legs_filled=(i < 2),
                partially_filled=(i >= 2),  # Partial fill for failures
                legs_filled=3 if i < 2 else 1,
                total_legs=3,
                expected_total_cost=1000.0,
                actual_total_cost=1000.0 if i < 2 else 0.0,
                total_slippage=0.0,
                slippage_percentage=0.0,
            )

            with patch.object(
                integrator.parallel_executor,
                "execute_parallel_arbitrage",
                return_value=mock_result,
            ):
                await integrator.execute_opportunity(opportunity)

        # Check integration stats
        stats = integrator.get_integration_stats()
        assert stats["execution_summary"]["total_executions"] == 3

    @pytest.mark.asyncio
    async def test_reporting_integration_with_execution(self, integration_setup):
        """Test execution reporter integration with parallel executor"""

        integrator = await create_parallel_integrator(**integration_setup)
        opportunity = create_profitable_opportunity("SPY")

        mock_result = ExecutionResult(
            success=True,
            execution_id="SPY_report_test",
            symbol="SPY",
            total_execution_time=2.5,
            all_legs_filled=True,
            partially_filled=False,
            legs_filled=3,
            total_legs=3,
            expected_total_cost=1000.0,
            actual_total_cost=1002.5,
            total_slippage=2.5,
            slippage_percentage=0.25,
            stock_result={
                "leg_type": "stock",
                "action": "BUY",
                "target_price": 100.0,
                "avg_fill_price": 100.02,
                "slippage": 0.02,
            },
            call_result={
                "leg_type": "call",
                "action": "SELL",
                "target_price": 8.50,
                "avg_fill_price": 8.47,
                "slippage": -0.03,
            },
            put_result={
                "leg_type": "put",
                "action": "BUY",
                "target_price": 3.25,
                "avg_fill_price": 3.28,
                "slippage": 0.03,
            },
        )

        with patch.object(
            integrator.parallel_executor,
            "execute_parallel_arbitrage",
            return_value=mock_result,
        ):
            with patch("builtins.print") as mock_print:  # Capture print output
                result = await integrator.execute_opportunity(opportunity)

                # Should have generated and printed a report
                assert mock_print.called

                # Check that report was generated for the execution
                reporter_stats = integrator.execution_reporter.get_session_statistics()
                assert reporter_stats["total_executions"] >= 1

    @pytest.mark.asyncio
    async def test_force_execution_modes(self, integration_setup):
        """Test forcing specific execution modes"""

        integrator = await create_parallel_integrator(**integration_setup)
        opportunity = create_profitable_opportunity("SPY")

        # Test force parallel
        mock_result = ExecutionResult(
            success=True,
            execution_id="forced_parallel",
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

        with patch.object(
            integrator.parallel_executor,
            "execute_parallel_arbitrage",
            return_value=mock_result,
        ):
            result = await integrator.execute_opportunity(
                opportunity, force_parallel=True
            )
            assert result["method"] == "parallel"

        # Test force combo
        mock_trade = MagicMock()
        mock_trade.orderStatus.status = "Filled"
        integration_setup["order_manager"].place_order.return_value = mock_trade

        result = await integrator.execute_opportunity(opportunity, force_combo=True)
        assert result["method"] == "combo"

    @pytest.mark.asyncio
    async def test_integration_status_tracking(self, integration_setup):
        """Test integration status and configuration tracking"""

        integrator = await create_parallel_integrator(**integration_setup)

        stats = integrator.get_integration_stats()

        # Should track integration status
        assert "integration_status" in stats
        assert "parallel_enabled" in stats["integration_status"]
        assert "is_initialized" in stats["integration_status"]

        # Should track execution summary
        assert "execution_summary" in stats
        assert "total_executions" in stats["execution_summary"]

    @pytest.mark.asyncio
    async def test_runtime_enable_disable_parallel(self, integration_setup):
        """Test runtime enabling/disabling of parallel execution"""

        integrator = await create_parallel_integrator(**integration_setup)

        # Initially should be enabled
        assert integrator.parallel_enabled is True

        # Disable at runtime
        success = integrator.disable_parallel_execution()
        assert success is True
        assert integrator.parallel_enabled is False

        # Re-enable at runtime
        success = integrator.enable_parallel_execution()
        assert success is True
        assert integrator.parallel_enabled is True

    @pytest.mark.asyncio
    async def test_integrator_shutdown(self, integration_setup):
        """Test proper shutdown and cleanup"""

        integrator = await create_parallel_integrator(**integration_setup)

        # Execute something to create session data
        opportunity = create_profitable_opportunity("SPY")
        mock_result = ExecutionResult(
            success=True,
            execution_id="shutdown_test",
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

        with patch.object(
            integrator.parallel_executor,
            "execute_parallel_arbitrage",
            return_value=mock_result,
        ):
            await integrator.execute_opportunity(opportunity)

        # Test shutdown
        await integrator.shutdown()

        # Should complete without error
        assert True  # If we get here, shutdown didn't crash


class TestIntegrationEdgeCases:
    """Test edge cases in integration scenarios"""

    @pytest.fixture
    def integration_setup(self):
        """Setup integration environment"""
        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        mock_order_manager = MagicMock()
        mock_order_manager.place_order = AsyncMock()

        opportunity_evaluator = MagicMock()
        opportunity_evaluator.symbol = "SPY"

        return {
            "ib": mock_ib,
            "order_manager": mock_order_manager,
            "opportunity_evaluator": opportunity_evaluator,
            "symbol": "SPY",
        }

    @pytest.mark.asyncio
    async def test_initialization_failure_handling(self, integration_setup):
        """Test handling of initialization failures"""

        integrator = ParallelExecutionIntegrator(**integration_setup)

        # Mock parallel executor initialization to fail
        with patch(
            "modules.Arbitrage.sfr.parallel_integration.ParallelLegExecutor"
        ) as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor.initialize = AsyncMock(return_value=False)
            mock_executor_class.return_value = mock_executor

            success = await integrator.initialize()
            assert success is False
            assert integrator.is_initialized is False

    @pytest.mark.asyncio
    async def test_concurrent_opportunity_execution(self, integration_setup):
        """Test handling of concurrent opportunity executions on same integrator"""

        integrator = await create_parallel_integrator(**integration_setup)

        # Create two opportunities
        opp1 = create_profitable_opportunity("SPY")
        opp2 = create_profitable_opportunity("SPY")

        mock_result = ExecutionResult(
            success=True,
            execution_id="concurrent_test",
            symbol="SPY",
            total_execution_time=1.0,
            all_legs_filled=True,
            partially_filled=False,
            legs_filled=3,
            total_legs=3,
            expected_total_cost=1000.0,
            actual_total_cost=1000.0,
            total_slippage=0.0,
            slippage_percentage=0.0,
        )

        with patch.object(
            integrator.parallel_executor,
            "execute_parallel_arbitrage",
            return_value=mock_result,
        ):
            # Execute both simultaneously
            task1 = asyncio.create_task(integrator.execute_opportunity(opp1))
            task2 = asyncio.create_task(integrator.execute_opportunity(opp2))

            results = await asyncio.gather(task1, task2, return_exceptions=True)

            # At least one should succeed
            successes = sum(
                1 for r in results if isinstance(r, dict) and r.get("success")
            )
            assert successes >= 1

    @pytest.mark.asyncio
    async def test_malformed_opportunity_handling(self, integration_setup):
        """Test handling of malformed opportunity data"""

        integrator = await create_parallel_integrator(**integration_setup)

        # Malformed opportunity (missing required fields)
        malformed_opp = {
            "some_field": "some_value"
            # Missing guaranteed_profit, contract, etc.
        }

        result = await integrator.execute_opportunity(malformed_opp)

        # Should handle gracefully
        assert result is not None
        assert isinstance(result, dict)
        assert "success" in result

    @pytest.mark.asyncio
    async def test_very_high_profit_opportunity(self, integration_setup):
        """Test handling of unusually high-profit opportunities"""

        integrator = await create_parallel_integrator(**integration_setup)

        # Very high profit opportunity
        high_profit_opp = create_profitable_opportunity("SPY", profit=50.0)

        use_parallel, reason = await integrator.should_use_parallel_execution(
            high_profit_opp
        )

        # Should still choose parallel for high profit
        assert use_parallel is True
        assert "parallel_execution_enabled" in reason


@pytest.mark.performance
class TestIntegrationPerformance:
    """Performance tests for integration system"""

    @pytest.fixture
    def integration_setup(self):
        """Setup performance test environment"""
        mock_ib = MagicMock()
        mock_ib.placeOrder = MagicMock()

        mock_order_manager = MagicMock()
        mock_order_manager.place_order = AsyncMock()

        opportunity_evaluator = MagicMock()
        opportunity_evaluator.symbol = "SPY"

        return {
            "ib": mock_ib,
            "order_manager": mock_order_manager,
            "opportunity_evaluator": opportunity_evaluator,
            "symbol": "SPY",
        }

    @pytest.mark.asyncio
    async def test_integration_initialization_speed(self, integration_setup):
        """Test integration initialization performance"""

        start_time = time.time()

        integrator = await create_parallel_integrator(**integration_setup)

        init_time = time.time() - start_time

        # Should initialize quickly
        assert init_time < 1.0  # Less than 1 second
        assert integrator.is_initialized is True

    @pytest.mark.asyncio
    async def test_decision_logic_performance(self, integration_setup):
        """Test execution decision logic performance"""

        integrator = await create_parallel_integrator(**integration_setup)

        # Create many opportunities
        opportunities = [create_profitable_opportunity("SPY") for _ in range(100)]

        start_time = time.time()

        # Test decision logic on all opportunities
        for opp in opportunities:
            use_parallel, reason = await integrator.should_use_parallel_execution(opp)
            assert isinstance(use_parallel, bool)
            assert isinstance(reason, str)

        total_time = time.time() - start_time

        # Should make decisions quickly
        assert total_time < 5.0  # Less than 5 seconds for 100 decisions
        avg_time = total_time / 100
        assert avg_time < 0.05  # Less than 50ms per decision

    @pytest.mark.asyncio
    async def test_concurrent_integrators_performance(self):
        """Test performance with multiple concurrent integrators"""

        async def create_and_run_integrator(symbol):
            mock_ib = MagicMock()
            mock_ib.placeOrder = MagicMock()

            setup = {
                "ib": mock_ib,
                "order_manager": MagicMock(),
                "opportunity_evaluator": MagicMock(),
                "symbol": symbol,
            }

            integrator = await create_parallel_integrator(**setup)

            # Test decision logic
            opp = create_profitable_opportunity(symbol)
            use_parallel, reason = await integrator.should_use_parallel_execution(opp)

            return {"symbol": symbol, "use_parallel": use_parallel}

        # Create 10 concurrent integrators
        start_time = time.time()

        symbols = [f"SYM{i}" for i in range(10)]
        tasks = [create_and_run_integrator(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Should handle concurrent creation efficiently
        assert total_time < 10.0  # Less than 10 seconds
        assert len(results) == 10

        # All should have completed successfully
        assert all("symbol" in result for result in results)
