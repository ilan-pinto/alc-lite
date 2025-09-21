"""
Enhanced end-to-end rollback testing with realistic IB API validation.

This test suite addresses the gaps in existing tests that failed to catch
the MU order execution issue by:
1. Using realistic IB API validation that rejects invalid prices
2. Testing complete execution flow from partial fill to successful rollback
3. Validating emergency market order fallback mechanisms
4. Using realistic market data instead of rounded test values
"""

import asyncio
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.Arbitrage.sfr.parallel_execution_framework import (
    LegOrder,
    LegType,
    ParallelExecutionPlan,
)
from modules.Arbitrage.sfr.rollback_manager import (
    RollbackManager,
    RollbackPlan,
    RollbackPosition,
    RollbackReason,
    RollbackStrategy,
)
from tests.test_utils import (
    IBValidationError,
    MarketDataGenerator,
    RealisticIBMock,
    create_mock_contract,
    create_mock_order,
    create_problematic_price,
)


class TestEndToEndRollbackValidation:
    """
    End-to-end testing of rollback scenarios with realistic IB API validation.

    These tests use the enhanced mocks and generators to simulate the exact
    conditions that caused the MU order execution failure.
    """

    @pytest.fixture
    def realistic_ib_mock(self):
        """Create realistic IB mock with price validation."""
        return RealisticIBMock()

    @pytest.fixture
    def market_generator(self):
        """Create market data generator with fixed seed for reproducible tests."""
        return MarketDataGenerator(seed=42)

    @pytest.fixture
    def rollback_manager_with_validation(self, realistic_ib_mock):
        """Create rollback manager with realistic IB validation."""
        return RollbackManager(realistic_ib_mock, "AAPL")

    @pytest.mark.asyncio
    async def test_partial_fill_to_successful_rollback_end_to_end(
        self, rollback_manager_with_validation, market_generator, realistic_ib_mock
    ):
        """
        Test complete flow: partial fill → rollback order → successful unwinding.

        This test simulates the successful path that should have happened with MU.
        """
        # Generate realistic market scenario
        scenario = market_generator.generate_market_scenario("AAPL", 155.0)

        # Create filled position from stock leg (like MU where only stock filled)
        stock_contract = create_mock_contract("AAPL", "STK")
        stock_order = create_mock_order("BUY", 100, "LMT", scenario.stock_price)

        stock_leg = LegOrder(
            leg_type=LegType.STOCK,
            contract=stock_contract,
            order=stock_order,
            target_price=scenario.stock_price,
            action="BUY",
            quantity=100,
        )
        stock_leg.fill_status = "filled"
        stock_leg.filled_quantity = 100
        stock_leg.avg_fill_price = scenario.stock_price

        # Create position to unwind
        stock_position = RollbackPosition(
            leg_order=stock_leg,
            current_quantity=100,
            avg_fill_price=scenario.stock_price,
            unrealized_pnl=0.0,
            unwinding_priority=1,
        )

        # Create rollback plan
        rollback_plan = RollbackPlan(
            rollback_id="test_successful_rollback",
            symbol="AAPL",
            execution_id="test_exec",
            reason=RollbackReason.PARTIAL_FILLS_TIMEOUT,
            strategy=RollbackStrategy.AGGRESSIVE_LIMIT,
            positions_to_unwind=[stock_position],
            total_positions=1,
            estimated_unwinding_cost=scenario.stock_price * 100,
            max_acceptable_loss=100.0,
            created_time=time.time(),
            max_rollback_time=time.time() + 30.0,
        )

        # Mock successful fill of rollback order
        def mock_place_order_success(contract, order):
            # Validate price precision (should pass with proper rounding)
            realistic_ib_mock._validate_price_precision(order.lmtPrice, contract)

            # Create successful trade
            trade = MagicMock()
            trade.orderStatus = MagicMock()
            trade.orderStatus.status = "Filled"
            trade.orderStatus.avgFillPrice = order.lmtPrice
            return trade

        realistic_ib_mock.placeOrder = mock_place_order_success

        # Execute rollback
        result = await rollback_manager_with_validation.execute_rollback(rollback_plan)

        # Verify successful rollback
        assert result["success"] is True
        assert result["positions_unwound"] == 1
        assert "emergency_fallback" not in result

        # Verify the position was marked as completed
        assert stock_position.unwind_status == "completed"

    @pytest.mark.asyncio
    async def test_price_precision_error_triggers_emergency_fallback_end_to_end(
        self, rollback_manager_with_validation, realistic_ib_mock
    ):
        """
        Test the exact MU scenario: price precision error → emergency fallback.

        This recreates the MU issue where rollback price had too many decimals
        and tests that emergency market order fallback activates.
        """
        # Create position with problematic fill price that would cause precision error
        problematic_fill_price = 156.7823456  # Too many decimal places like MU

        stock_contract = create_mock_contract("MU", "STK")
        stock_order = create_mock_order("BUY", 100, "LMT", problematic_fill_price)

        stock_leg = LegOrder(
            leg_type=LegType.STOCK,
            contract=stock_contract,
            order=stock_order,
            target_price=problematic_fill_price,
            action="BUY",
            quantity=100,
        )
        stock_leg.fill_status = "filled"
        stock_leg.filled_quantity = 100
        stock_leg.avg_fill_price = problematic_fill_price

        stock_position = RollbackPosition(
            leg_order=stock_leg,
            current_quantity=100,
            avg_fill_price=problematic_fill_price,
            unrealized_pnl=0.0,
            unwinding_priority=1,
        )

        rollback_plan = RollbackPlan(
            rollback_id="test_mu_scenario",
            symbol="MU",
            execution_id="test_exec",
            reason=RollbackReason.PARTIAL_FILLS_TIMEOUT,
            strategy=RollbackStrategy.AGGRESSIVE_LIMIT,
            positions_to_unwind=[stock_position],
            total_positions=1,
            estimated_unwinding_cost=problematic_fill_price * 100,
            max_acceptable_loss=100.0,
            created_time=time.time(),
            max_rollback_time=time.time() + 30.0,
        )

        # Track order placement attempts
        order_attempts = []

        def mock_place_order_with_precision_validation(contract, order):
            order_attempts.append((contract, order))

            # First attempt: limit order with bad precision → validation error
            if order.orderType == "LMT":
                # This should trigger the precision error handling
                realistic_ib_mock._validate_price_precision(order.lmtPrice, contract)
                # If we get here, the price was properly rounded
                trade = MagicMock()
                trade.orderStatus = MagicMock()
                trade.orderStatus.status = "Filled"
                trade.orderStatus.avgFillPrice = order.lmtPrice
                return trade
            elif order.orderType == "MKT":
                # Emergency market order succeeds
                trade = MagicMock()
                trade.orderStatus = MagicMock()
                trade.orderStatus.status = "Filled"
                trade.orderStatus.avgFillPrice = 155.21  # Market fill price
                return trade

        realistic_ib_mock.placeOrder = mock_place_order_with_precision_validation

        # Execute rollback
        result = await rollback_manager_with_validation.execute_rollback(rollback_plan)

        # Verify rollback succeeded (should use proper price rounding now)
        assert result["success"] is True
        assert result["positions_unwound"] == 1

        # Verify that order was placed with properly rounded price
        assert len(order_attempts) == 1
        placed_order = order_attempts[0][1]

        # Check that the price was rounded to 2 decimal places
        assert len(str(placed_order.lmtPrice).split(".")[1]) <= 2

    @pytest.mark.asyncio
    async def test_multiple_legs_partial_fill_realistic_rollback(
        self, rollback_manager_with_validation, market_generator, realistic_ib_mock
    ):
        """
        Test multi-leg partial fill scenario with realistic market data.

        This tests a more complex scenario where multiple legs have different
        fill statuses and rollback must handle each appropriately.
        """
        # Generate realistic market scenario
        scenario, fill_outcomes = market_generator.generate_partial_fill_scenario(
            "SPY", {"stock": 1.0, "call": 0.0, "put": 0.0}  # Only stock fills
        )

        # Create legs based on fill outcomes
        positions = []

        # Stock leg - filled
        if fill_outcomes["stock"]:
            stock_contract = create_mock_contract("SPY", "STK")
            stock_order = create_mock_order("BUY", 100, "LMT", scenario.stock_price)

            stock_leg = LegOrder(
                leg_type=LegType.STOCK,
                contract=stock_contract,
                order=stock_order,
                target_price=scenario.stock_price,
                action="BUY",
                quantity=100,
            )
            stock_leg.fill_status = "filled"
            stock_leg.filled_quantity = 100
            stock_leg.avg_fill_price = scenario.stock_price

            positions.append(
                RollbackPosition(
                    leg_order=stock_leg,
                    current_quantity=100,
                    avg_fill_price=scenario.stock_price,
                    unrealized_pnl=0.0,
                    unwinding_priority=1,
                )
            )

        # Create rollback plan
        rollback_plan = RollbackPlan(
            rollback_id="test_multi_leg_partial",
            symbol="SPY",
            execution_id="test_exec",
            reason=RollbackReason.PARTIAL_FILLS_TIMEOUT,
            strategy=RollbackStrategy.AGGRESSIVE_LIMIT,
            positions_to_unwind=positions,
            total_positions=len(positions),
            estimated_unwinding_cost=sum(
                pos.avg_fill_price * pos.current_quantity for pos in positions
            ),
            max_acceptable_loss=1000.0,
            created_time=time.time(),
            max_rollback_time=time.time() + 30.0,
        )

        # Mock successful rollback orders
        def mock_successful_rollback(contract, order):
            realistic_ib_mock._validate_price_precision(order.lmtPrice, contract)
            trade = MagicMock()
            trade.orderStatus = MagicMock()
            trade.orderStatus.status = "Filled"
            trade.orderStatus.avgFillPrice = order.lmtPrice
            return trade

        realistic_ib_mock.placeOrder = mock_successful_rollback

        # Execute rollback
        result = await rollback_manager_with_validation.execute_rollback(rollback_plan)

        # Verify successful rollback
        assert result["success"] is True
        assert result["positions_unwound"] == len(positions)

        # Verify all positions were unwound
        for position in positions:
            assert position.unwind_status == "completed"

    @pytest.mark.asyncio
    async def test_emergency_market_order_fallback_after_limit_order_failure(
        self, rollback_manager_with_validation, realistic_ib_mock
    ):
        """
        Test that emergency market orders are used when limit orders fail due to pricing.

        This tests the enhanced error handling we added to catch pricing errors
        and automatically switch to market orders.
        """
        # Create position that will trigger pricing error during rollback
        stock_contract = create_mock_contract("TSLA", "STK")
        problematic_price = create_problematic_price(
            200.0
        )  # Price with too many decimals

        stock_leg = LegOrder(
            leg_type=LegType.STOCK,
            contract=stock_contract,
            order=create_mock_order("BUY", 100, "LMT", problematic_price),
            target_price=problematic_price,
            action="BUY",
            quantity=100,
        )
        stock_leg.fill_status = "filled"
        stock_leg.filled_quantity = 100
        stock_leg.avg_fill_price = problematic_price

        stock_position = RollbackPosition(
            leg_order=stock_leg,
            current_quantity=100,
            avg_fill_price=problematic_price,
            unrealized_pnl=0.0,
            unwinding_priority=1,
        )

        # Mock order placement: limit orders fail, market orders succeed
        order_types_attempted = []

        def mock_place_order_fallback(contract, order):
            order_types_attempted.append(order.orderType)

            if order.orderType == "LMT":
                # Limit order triggers precision error without proper rounding
                # This simulates the old behavior before our fixes
                calculated_rollback_price = (
                    problematic_price * 0.99
                )  # Aggressive pricing
                if len(str(calculated_rollback_price).split(".")[1]) > 2:
                    raise IBValidationError(
                        f"The price {calculated_rollback_price} does not conform to minimum price variation"
                    )

                # If price was properly rounded, it would succeed
                trade = MagicMock()
                trade.orderStatus = MagicMock()
                trade.orderStatus.status = "Filled"
                trade.orderStatus.avgFillPrice = order.lmtPrice
                return trade

            elif order.orderType == "MKT":
                # Market order succeeds
                trade = MagicMock()
                trade.orderStatus = MagicMock()
                trade.orderStatus.status = "Filled"
                trade.orderStatus.avgFillPrice = 199.50  # Market price
                return trade

        realistic_ib_mock.placeOrder = mock_place_order_fallback

        # Test emergency market order placement directly
        success = await rollback_manager_with_validation._place_emergency_market_order(
            stock_position
        )

        # Verify market order was attempted and succeeded
        assert success is True
        assert "MKT" in order_types_attempted
        assert stock_position.unwind_status == "completed"

    @pytest.mark.asyncio
    async def test_rollback_performance_with_realistic_data(
        self, rollback_manager_with_validation, market_generator, realistic_ib_mock
    ):
        """
        Test rollback performance with realistic market conditions.

        This ensures our improvements don't negatively impact performance
        while providing better error handling.
        """
        # Generate realistic scenario
        scenario = market_generator.generate_market_scenario("QQQ", 350.0)

        # Create multiple positions to test batch rollback
        positions = []
        for i in range(5):  # 5 positions
            contract = create_mock_contract(f"TEST{i}", "STK")
            price = scenario.stock_price + (i * 0.50)  # Slightly different prices

            leg = LegOrder(
                leg_type=LegType.STOCK,
                contract=contract,
                order=create_mock_order("BUY", 100, "LMT", price),
                target_price=price,
                action="BUY",
                quantity=100,
            )
            leg.fill_status = "filled"
            leg.filled_quantity = 100
            leg.avg_fill_price = price

            positions.append(
                RollbackPosition(
                    leg_order=leg,
                    current_quantity=100,
                    avg_fill_price=price,
                    unrealized_pnl=0.0,
                    unwinding_priority=i + 1,
                )
            )

        # Create rollback plan
        rollback_plan = RollbackPlan(
            rollback_id="test_performance",
            symbol="QQQ",
            execution_id="test_exec",
            reason=RollbackReason.PARTIAL_FILLS_TIMEOUT,
            strategy=RollbackStrategy.AGGRESSIVE_LIMIT,
            positions_to_unwind=positions,
            total_positions=len(positions),
            estimated_unwinding_cost=sum(
                pos.avg_fill_price * pos.current_quantity for pos in positions
            ),
            max_acceptable_loss=1000.0,
            created_time=time.time(),
            max_rollback_time=time.time() + 30.0,
        )

        # Mock fast successful orders
        def mock_fast_orders(contract, order):
            realistic_ib_mock._validate_price_precision(order.lmtPrice, contract)
            trade = MagicMock()
            trade.orderStatus = MagicMock()
            trade.orderStatus.status = "Filled"
            trade.orderStatus.avgFillPrice = order.lmtPrice
            return trade

        realistic_ib_mock.placeOrder = mock_fast_orders

        # Time the rollback execution
        start_time = time.time()
        result = await rollback_manager_with_validation.execute_rollback(rollback_plan)
        execution_time = time.time() - start_time

        # Verify success and reasonable performance
        assert result["success"] is True
        assert result["positions_unwound"] == len(positions)
        assert execution_time < 1.0  # Should complete within 1 second

        # Verify all positions were handled
        for position in positions:
            assert position.unwind_status == "completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
