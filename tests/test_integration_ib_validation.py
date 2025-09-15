"""
Integration test framework for IB paper trading validation.

This module provides integration tests that can optionally run against
the real IB paper trading API to validate behavior. These tests are:
1. Marked as integration tests that don't run by default
2. Configurable to run against mock or real IB API
3. Designed to catch discrepancies between our mocks and real IB behavior

Usage:
    # Run with mocks (default, fast)
    pytest tests/test_integration_ib_validation.py

    # Run against real IB paper trading (slow, requires IB Gateway)
    pytest tests/test_integration_ib_validation.py --ib-integration

Prerequisites for real IB testing:
- IB Gateway or TWS running on localhost:7497
- Paper trading account configured
- Valid market data subscriptions
"""

import asyncio
import os
import time
from typing import Optional
from unittest.mock import MagicMock

import pytest
from ib_async import IB, Contract, Option, Order, Stock

from modules.Arbitrage.sfr.parallel_execution_framework import (
    ParallelExecutionFramework,
)
from modules.Arbitrage.sfr.rollback_manager import RollbackManager
from tests.test_utils import (
    IBValidationError,
    MarketDataGenerator,
    RealisticIBMock,
    create_mock_contract,
    create_mock_order,
)

# Configuration for integration tests
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7497"))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "999"))
IB_TIMEOUT = int(os.getenv("IB_TIMEOUT", "10"))


def pytest_addoption(parser):
    """Add command line option for IB integration testing."""
    parser.addoption(
        "--ib-integration",
        action="store_true",
        default=False,
        help="Run integration tests against real IB API (requires IB Gateway)",
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "ib_integration: mark test as IB integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Skip IB integration tests unless explicitly requested."""
    if config.getoption("--ib-integration"):
        return

    skip_integration = pytest.mark.skip(reason="need --ib-integration option to run")
    for item in items:
        if "ib_integration" in item.keywords:
            item.add_marker(skip_integration)


class IBConnectionManager:
    """Manage IB connections for integration testing."""

    def __init__(self, use_real_ib: bool = False):
        self.use_real_ib = use_real_ib
        self.ib: Optional[IB] = None
        self.connected = False

    async def connect(self) -> IB:
        """Connect to IB (real or mock)."""
        if self.use_real_ib:
            self.ib = IB()
            try:
                await asyncio.wait_for(
                    self.ib.connectAsync(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID),
                    timeout=IB_TIMEOUT,
                )
                self.connected = True
                print(f"✅ Connected to real IB at {IB_HOST}:{IB_PORT}")
            except Exception as e:
                pytest.skip(f"Cannot connect to IB: {e}")
        else:
            self.ib = RealisticIBMock()
            self.connected = True
            print("🤖 Using realistic IB mock")

        return self.ib

    async def disconnect(self):
        """Disconnect from IB."""
        if self.connected and self.use_real_ib and self.ib:
            self.ib.disconnect()
            self.connected = False


@pytest.fixture
def ib_connection_manager(request):
    """Fixture to provide IB connection manager."""
    use_real_ib = request.config.getoption("--ib-integration")
    return IBConnectionManager(use_real_ib)


@pytest.fixture
async def ib_connection(ib_connection_manager):
    """Fixture to provide connected IB instance."""
    ib = await ib_connection_manager.connect()
    yield ib
    await ib_connection_manager.disconnect()


class TestIBPricePrecisionValidation:
    """Test price precision validation against real/mock IB API."""

    @pytest.mark.ib_integration
    @pytest.mark.asyncio
    async def test_stock_price_precision_real_vs_mock(self, ib_connection):
        """
        Compare stock price precision validation between real IB and mock.

        This test verifies that our mock accurately simulates IB's price
        precision validation rules.
        """
        # Test cases with different decimal precision
        test_prices = [
            100.12,  # Valid: 2 decimal places
            100.123,  # Invalid: 3 decimal places
            100.1234,  # Invalid: 4 decimal places
            100.12345,  # Invalid: 5 decimal places
            50.5,  # Valid: 1 decimal place
            50.0,  # Valid: 0 decimal places
        ]

        # Create test contract
        if hasattr(ib_connection, "placeOrder"):  # Real IB
            contract = Stock("AAPL", "SMART", "USD")
        else:  # Mock IB
            contract = create_mock_contract("AAPL", "STK")

        # Test each price
        for price in test_prices:
            # Create test order
            if hasattr(ib_connection, "placeOrder"):  # Real IB
                order = Order(
                    orderId=int(time.time() * 1000) % 1000000,
                    orderType="LMT",
                    action="BUY",
                    totalQuantity=1,
                    lmtPrice=price,
                    tif="DAY",
                )
            else:  # Mock IB
                order = create_mock_order("BUY", 1, "LMT", price)

            # Test order placement
            try:
                result = ib_connection.placeOrder(contract, order)
                order_accepted = True
                print(f"✅ Price {price}: Order accepted")
            except Exception as e:
                order_accepted = False
                print(f"❌ Price {price}: {str(e)}")

            # Validate expectations
            expected_valid = (
                len(str(price).split(".")[1]) <= 2 if "." in str(price) else True
            )

            if expected_valid:
                assert (
                    order_accepted
                ), f"Price {price} should be accepted but was rejected"
            else:
                assert (
                    not order_accepted
                ), f"Price {price} should be rejected but was accepted"

    @pytest.mark.ib_integration
    @pytest.mark.asyncio
    async def test_option_price_precision_validation(self, ib_connection):
        """Test option price precision validation."""
        test_prices = [
            5.25,  # Valid
            5.255,  # Invalid: too many decimal places
            0.05,  # Valid: minimum option price
            0.001,  # Invalid: below minimum
        ]

        # Create option contract
        if hasattr(ib_connection, "placeOrder"):  # Real IB
            contract = Option("AAPL", "20241220", 150, "C", "SMART")
        else:  # Mock IB
            contract = create_mock_contract("AAPL", "OPT")
            contract.right = "C"
            contract.strike = 150

        for price in test_prices:
            # Create test order
            if hasattr(ib_connection, "placeOrder"):  # Real IB
                order = Order(
                    orderId=int(time.time() * 1000) % 1000000,
                    orderType="LMT",
                    action="BUY",
                    totalQuantity=1,
                    lmtPrice=price,
                    tif="DAY",
                )
            else:  # Mock IB
                order = create_mock_order("BUY", 1, "LMT", price)

            try:
                result = ib_connection.placeOrder(contract, order)
                order_accepted = True
            except Exception:
                order_accepted = False

            # Options should follow similar precision rules as stocks
            expected_valid = (
                len(str(price).split(".")[1]) <= 2 if "." in str(price) else True
            ) and price >= 0.01

            if expected_valid:
                assert order_accepted, f"Option price {price} should be accepted"
            else:
                assert not order_accepted, f"Option price {price} should be rejected"

    @pytest.mark.ib_integration
    @pytest.mark.asyncio
    async def test_rollback_price_calculation_integration(self, ib_connection):
        """
        Test that rollback price calculations work with real IB validation.

        This integration test ensures our rollback price rounding fixes
        work correctly with the actual IB API.
        """
        # Create rollback manager
        rollback_manager = RollbackManager(ib_connection, "AAPL")

        # Test problematic prices that would have caused the MU issue
        problematic_fill_prices = [
            156.7823456,  # MU-style problematic price
            99.9999999,  # Edge case near round number
            0.00123456,  # Very small price with many decimals
            1234.56789,  # Large price with many decimals
        ]

        for fill_price in problematic_fill_prices:
            print(f"\n🧪 Testing rollback with fill price: {fill_price}")

            # Calculate what rollback price would be (aggressive limit)
            pricing_factor = 0.02  # 2% aggressive pricing
            rollback_price = fill_price * (1.0 - pricing_factor)  # SELL action

            print(f"📊 Raw rollback price: {rollback_price}")

            # Apply our price rounding fix
            from modules.Arbitrage.sfr.utils import round_price_to_tick_size

            rounded_rollback_price = round_price_to_tick_size(rollback_price, "stock")

            print(f"✨ Rounded rollback price: {rounded_rollback_price}")

            # Test that rounded price is accepted by IB
            if hasattr(ib_connection, "placeOrder"):  # Real IB
                contract = Stock("AAPL", "SMART", "USD")
                order = Order(
                    orderId=int(time.time() * 1000) % 1000000,
                    orderType="LMT",
                    action="SELL",
                    totalQuantity=1,
                    lmtPrice=rounded_rollback_price,
                    tif="DAY",
                )
            else:  # Mock IB
                contract = create_mock_contract("AAPL", "STK")
                order = create_mock_order("SELL", 1, "LMT", rounded_rollback_price)

            # This should not raise an exception
            try:
                result = ib_connection.placeOrder(contract, order)
                print(f"✅ Rollback order accepted with rounded price")

                # Cancel the order if using real IB to avoid accumulating orders
                if hasattr(ib_connection, "cancelOrder"):
                    ib_connection.cancelOrder(order)

            except Exception as e:
                pytest.fail(
                    f"Rollback order should be accepted after price rounding: {e}"
                )

    @pytest.mark.ib_integration
    @pytest.mark.asyncio
    async def test_parallel_execution_pricing_integration(self, ib_connection):
        """
        Test parallel execution framework with real IB price validation.

        This verifies that our improved aggressive pricing strategy
        works correctly with the actual IB API.
        """
        # Create parallel execution framework
        framework = ParallelExecutionFramework(ib_connection)

        # Create realistic contracts
        if hasattr(ib_connection, "placeOrder"):  # Real IB
            stock_contract = Stock("AAPL", "SMART", "USD")
            call_contract = Option("AAPL", "20241220", 150, "C", "SMART")
            put_contract = Option("AAPL", "20241220", 150, "P", "SMART")
        else:  # Mock IB
            stock_contract = create_mock_contract("AAPL", "STK")
            call_contract = create_mock_contract("AAPL", "OPT")
            call_contract.right = "C"
            put_contract = create_mock_contract("AAPL", "OPT")
            put_contract.right = "P"

        # Generate realistic market data
        generator = MarketDataGenerator(seed=123)
        scenario = generator.generate_market_scenario("AAPL", 150.0)

        # Create execution plan with realistic prices
        try:
            execution_plan = await framework.create_execution_plan(
                symbol="AAPL",
                expiry="20241220",
                stock_contract=stock_contract,
                call_contract=call_contract,
                put_contract=put_contract,
                stock_price=scenario.stock_price,
                call_price=scenario.call_price,
                put_price=scenario.put_price,
                quantity=1,
            )

            print(f"✅ Execution plan created with prices:")
            print(f"   Stock: {execution_plan.stock_leg.target_price}")
            print(f"   Call: {execution_plan.call_leg.target_price}")
            print(f"   Put: {execution_plan.put_leg.target_price}")

            # Verify all prices have proper precision
            for leg in [
                execution_plan.stock_leg,
                execution_plan.call_leg,
                execution_plan.put_leg,
            ]:
                price_str = str(leg.target_price)
                decimal_places = len(price_str.split(".")[1]) if "." in price_str else 0
                assert (
                    decimal_places <= 2
                ), f"Price {leg.target_price} has too many decimal places"

        except Exception as e:
            pytest.fail(
                f"Execution plan creation should succeed with proper price rounding: {e}"
            )

    @pytest.mark.ib_integration
    @pytest.mark.asyncio
    async def test_mock_vs_real_ib_consistency(self, ib_connection_manager):
        """
        Test that our mock behaves consistently with real IB API.

        This test runs the same operations against both mock and real IB
        to verify consistency in behavior.
        """
        test_operations = [
            # (price, expected_valid)
            (100.12, True),
            (100.123, False),
            (0.01, True),
            (0.001, False),
        ]

        results_mock = []
        results_real = []

        # Test with mock
        mock_ib = RealisticIBMock()
        for price, expected in test_operations:
            contract = create_mock_contract("TEST", "STK")
            order = create_mock_order("BUY", 1, "LMT", price)

            try:
                mock_ib.placeOrder(contract, order)
                results_mock.append(True)
            except Exception:
                results_mock.append(False)

        # Test with real IB if available
        if ib_connection_manager.use_real_ib:
            real_ib = await ib_connection_manager.connect()

            for price, expected in test_operations:
                contract = Stock("AAPL", "SMART", "USD")
                order = Order(
                    orderId=int(time.time() * 1000) % 1000000,
                    orderType="LMT",
                    action="BUY",
                    totalQuantity=1,
                    lmtPrice=price,
                    tif="DAY",
                )

                try:
                    real_ib.placeOrder(contract, order)
                    results_real.append(True)
                    # Cancel order immediately
                    real_ib.cancelOrder(order)
                except Exception:
                    results_real.append(False)

            # Compare results
            assert (
                results_mock == results_real
            ), f"Mock and real IB results differ: mock={results_mock}, real={results_real}"

            await ib_connection_manager.disconnect()
        else:
            # Just verify mock results match expectations
            expected_results = [expected for price, expected in test_operations]
            assert (
                results_mock == expected_results
            ), f"Mock results don't match expectations: {results_mock} vs {expected_results}"


class TestIBIntegrationPerformance:
    """Test performance characteristics of IB integration."""

    @pytest.mark.ib_integration
    @pytest.mark.asyncio
    async def test_order_placement_performance(self, ib_connection):
        """Test that order placement performance is acceptable."""
        num_orders = 10
        start_time = time.time()

        for i in range(num_orders):
            if hasattr(ib_connection, "placeOrder"):  # Real IB
                contract = Stock("AAPL", "SMART", "USD")
                order = Order(
                    orderId=int(time.time() * 1000 + i) % 1000000,
                    orderType="LMT",
                    action="BUY",
                    totalQuantity=1,
                    lmtPrice=100.00,
                    tif="DAY",
                )
                result = ib_connection.placeOrder(contract, order)
                # Cancel immediately to avoid accumulating orders
                ib_connection.cancelOrder(order)
            else:  # Mock IB
                contract = create_mock_contract("AAPL", "STK")
                order = create_mock_order("BUY", 1, "LMT", 100.00)
                result = ib_connection.placeOrder(contract, order)

        elapsed_time = time.time() - start_time
        avg_time_per_order = elapsed_time / num_orders

        print(f"⏱️ Placed {num_orders} orders in {elapsed_time:.3f}s")
        print(f"📊 Average time per order: {avg_time_per_order:.3f}s")

        # Performance assertion
        assert (
            avg_time_per_order < 0.1
        ), f"Order placement too slow: {avg_time_per_order:.3f}s per order"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--ib-integration"])
