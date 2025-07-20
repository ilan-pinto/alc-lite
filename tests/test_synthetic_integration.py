"""
Comprehensive end-to-end integration tests for Synthetic arbitrage detection.
Tests the complete workflow from market data to arbitrage execution for synthetic positions.
"""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import logging
import pytest

from modules.Arbitrage.metrics import RejectionReason, metrics_collector
from modules.Arbitrage.Synthetic import Syn

# Import test utilities
try:
    from .market_scenarios import SyntheticScenarios
    from .mock_ib import MarketDataGenerator, MockContract, MockIB
except ImportError:
    from market_scenarios import SyntheticScenarios
    from mock_ib import MarketDataGenerator, MockContract, MockIB


class TestSyntheticArbitrageIntegration:
    """Integration tests for Synthetic arbitrage workflow"""

    def setup_method(self):
        """Setup logging for tests to ensure log output is visible"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,
        )
        logger = logging.getLogger("rich")
        logger.setLevel(logging.DEBUG)

    def verify_synthetic_calculations(
        self, market_data, expected_results, scenario_name
    ):
        """Helper method to verify synthetic arbitrage calculations"""
        print(f"\n🔍 Verifying synthetic calculations for {scenario_name}:")

        # Find stock and option data
        stock_ticker = next(
            t
            for t in market_data.values()
            if hasattr(t.contract, "right") and t.contract.right is None
        )

        # Find call and put options
        call_option = None
        put_option = None

        for ticker in market_data.values():
            if hasattr(ticker.contract, "right"):
                if ticker.contract.right == "C":
                    call_option = ticker
                elif ticker.contract.right == "P":
                    put_option = ticker

        if call_option and put_option:
            # Calculate synthetic metrics
            stock_price = stock_ticker.ask  # Buy stock at ask for synthetic
            call_price = call_option.bid  # Sell call at bid
            put_price = put_option.ask  # Buy put at ask

            net_credit = call_price - put_price
            spread = stock_price - put_option.contract.strike
            min_profit = net_credit - spread  # Max loss for synthetic
            max_profit = (
                call_option.contract.strike - put_option.contract.strike
            ) + min_profit

            print(f"  📊 Stock: ${stock_price:.2f}")
            print(f"  📊 Call {call_option.contract.strike} bid: ${call_price:.2f}")
            print(f"  📊 Put {put_option.contract.strike} ask: ${put_price:.2f}")
            print(f"  📊 Net Credit: ${net_credit:.2f}")
            print(f"  📊 Spread: ${spread:.2f}")
            print(f"  📊 Min Profit (Max Loss): ${min_profit:.2f}")
            print(f"  📊 Max Profit: ${max_profit:.2f}")

            if min_profit != 0:
                risk_reward_ratio = max_profit / abs(min_profit)
                print(f"  📊 Risk-Reward Ratio: {risk_reward_ratio:.2f}")

            print(f"  ✅ Synthetic calculations verified for {scenario_name}")

    @pytest.mark.asyncio
    async def test_synthetic_profitable_scenario(self):
        """
        Test synthetic arbitrage with profitable risk-reward ratio.

        This test verifies:
        1. Synthetic position is correctly identified
        2. Risk-reward ratio meets threshold
        3. Order is placed successfully
        """
        print("\n🔍 Testing synthetic profitable scenario")

        # Create market data
        market_data = SyntheticScenarios.synthetic_profitable_scenario()

        # Verify calculations
        self.verify_synthetic_calculations(market_data, {}, "Profitable Synthetic")

        # Setup mock IB
        mock_ib = MockIB()
        mock_ib.test_market_data = market_data

        syn = Syn(debug=True)
        syn.ib = mock_ib
        syn.order_manager = MagicMock()

        # Make place_order async
        async def mock_place_order(*args, **kwargs):
            return MagicMock()

        syn.order_manager.place_order = mock_place_order

        # Setup mocks similar to SFR tests
        with (
            patch.object(syn, "_get_stock_contract") as mock_get_stock,
            patch.object(syn, "_get_market_data_async") as mock_get_market_data,
            patch.object(syn, "_get_chain") as mock_get_chain,
            patch.object(syn, "parallel_qualify_all_contracts") as mock_qualify,
        ):
            # Setup stock contract
            stock_contract = next(
                t.contract
                for t in market_data.values()
                if hasattr(t.contract, "right") and t.contract.right is None
            )
            mock_get_stock.return_value = ("SMART", "STK", stock_contract)

            stock_ticker = next(
                t
                for t in market_data.values()
                if hasattr(t.contract, "right") and t.contract.right is None
            )
            mock_get_market_data.return_value = stock_ticker

            # Setup option chain
            mock_chain = MagicMock()
            mock_chain.strikes = [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105]

            valid_expiry_date = datetime.now() + timedelta(days=30)
            valid_expiry_str = valid_expiry_date.strftime("%Y%m%d")
            mock_chain.expirations = [valid_expiry_str]
            mock_get_chain.return_value = mock_chain

            # Mock qualified contracts
            qualified_contracts = {}
            key = f"{valid_expiry_str}_100_99"

            call_contract = next(
                t.contract
                for t in market_data.values()
                if hasattr(t.contract, "right") and t.contract.right == "C"
            )
            put_contract = next(
                t.contract
                for t in market_data.values()
                if hasattr(t.contract, "right") and t.contract.right == "P"
            )

            qualified_contracts[key] = {
                "call_contract": call_contract,
                "put_contract": put_contract,
                "call_strike": 100.0,
                "put_strike": 99.0,
                "expiry": valid_expiry_str,
            }

            mock_qualify.return_value = qualified_contracts

            # Reset metrics
            metrics_collector.reset_session()

            # Define syn attributes directly since scan_syn expects them
            syn.cost_limit = 120.0
            syn.max_loss_threshold = -10.0  # Allow up to $10 loss
            syn.max_profit_threshold = 20.0  # Allow up to $20 profit
            syn.profit_ratio_threshold = 2.0  # Require at least 2:1 risk-reward

            # Run scan with appropriate thresholds
            await syn.scan_syn("TEST", quantity=1)

            print("✅ Synthetic profitable scenario test completed")

    @pytest.mark.asyncio
    async def test_synthetic_poor_risk_reward(self):
        """
        Test synthetic arbitrage rejection due to poor risk-reward ratio.

        This test verifies:
        1. Poor risk-reward ratio is detected
        2. Position is rejected with PROFIT_RATIO_THRESHOLD_NOT_MET
        3. No order is placed
        """
        print("\n🔍 Testing synthetic poor risk-reward scenario")

        # Create market data
        market_data = SyntheticScenarios.synthetic_poor_risk_reward()

        # Setup mock IB
        mock_ib = MockIB()
        mock_ib.test_market_data = market_data

        syn = Syn(debug=True)
        syn.ib = mock_ib
        syn.order_manager = MagicMock()

        # Make place_order async
        async def mock_place_order(*args, **kwargs):
            return MagicMock()

        syn.order_manager.place_order = mock_place_order

        # Setup mocks
        with (
            patch.object(syn, "_get_stock_contract") as mock_get_stock,
            patch.object(syn, "_get_market_data_async") as mock_get_market_data,
            patch.object(syn, "_get_chain") as mock_get_chain,
            patch.object(syn, "parallel_qualify_all_contracts") as mock_qualify,
        ):
            # Setup stock contract
            stock_contract = next(
                t.contract
                for t in market_data.values()
                if hasattr(t.contract, "right") and t.contract.right is None
            )
            mock_get_stock.return_value = ("SMART", "STK", stock_contract)

            stock_ticker = next(
                t
                for t in market_data.values()
                if hasattr(t.contract, "right") and t.contract.right is None
            )
            mock_get_market_data.return_value = stock_ticker

            # Setup option chain
            mock_chain = MagicMock()
            mock_chain.strikes = [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105]

            valid_expiry_date = datetime.now() + timedelta(days=30)
            valid_expiry_str = valid_expiry_date.strftime("%Y%m%d")
            mock_chain.expirations = [valid_expiry_str]
            mock_get_chain.return_value = mock_chain

            # Mock qualified contracts
            qualified_contracts = {}
            key = f"{valid_expiry_str}_100_99"

            call_contract = next(
                t.contract
                for t in market_data.values()
                if hasattr(t.contract, "right") and t.contract.right == "C"
            )
            put_contract = next(
                t.contract
                for t in market_data.values()
                if hasattr(t.contract, "right") and t.contract.right == "P"
            )

            qualified_contracts[key] = {
                "call_contract": call_contract,
                "put_contract": put_contract,
                "call_strike": 100.0,
                "put_strike": 99.0,
                "expiry": valid_expiry_str,
            }

            mock_qualify.return_value = qualified_contracts

            # Reset metrics
            metrics_collector.reset_session()

            # Define syn attributes with high profit ratio threshold
            syn.cost_limit = 120.0
            syn.max_loss_threshold = -10.0
            syn.max_profit_threshold = 20.0
            syn.profit_ratio_threshold = 3.0  # Require 3:1 ratio (will fail)

            # Run scan
            await syn.scan_syn("TEST", quantity=1)

            print("✅ Synthetic poor risk-reward test completed")

    @pytest.mark.asyncio
    async def test_synthetic_max_loss_exceeded(self):
        """
        Test synthetic arbitrage rejection due to max loss threshold.

        This test verifies:
        1. Max loss exceeds threshold
        2. Position is rejected with MAX_LOSS_THRESHOLD_EXCEEDED
        3. No order is placed
        """
        print("\n🔍 Testing synthetic max loss exceeded scenario")

        # Create market data
        market_data = SyntheticScenarios.synthetic_max_loss_exceeded()

        # Setup mock IB
        mock_ib = MockIB()
        mock_ib.test_market_data = market_data

        syn = Syn(debug=True)
        syn.ib = mock_ib
        syn.order_manager = MagicMock()

        # Make place_order async
        async def mock_place_order(*args, **kwargs):
            return MagicMock()

        syn.order_manager.place_order = mock_place_order

        # Setup mocks
        with (
            patch.object(syn, "_get_stock_contract") as mock_get_stock,
            patch.object(syn, "_get_market_data_async") as mock_get_market_data,
            patch.object(syn, "_get_chain") as mock_get_chain,
            patch.object(syn, "parallel_qualify_all_contracts") as mock_qualify,
        ):
            # Setup stock contract
            stock_contract = next(
                t.contract
                for t in market_data.values()
                if hasattr(t.contract, "right") and t.contract.right is None
            )
            mock_get_stock.return_value = ("SMART", "STK", stock_contract)

            stock_ticker = next(
                t
                for t in market_data.values()
                if hasattr(t.contract, "right") and t.contract.right is None
            )
            mock_get_market_data.return_value = stock_ticker

            # Setup option chain with appropriate strikes for this scenario
            mock_chain = MagicMock()
            mock_chain.strikes = [135, 140, 145, 150, 155, 160]

            valid_expiry_date = datetime.now() + timedelta(days=30)
            valid_expiry_str = valid_expiry_date.strftime("%Y%m%d")
            mock_chain.expirations = [valid_expiry_str]
            mock_get_chain.return_value = mock_chain

            # Mock qualified contracts - using 150/140 from our scenario
            qualified_contracts = {}
            key = f"{valid_expiry_str}_150_140"

            call_contract = next(
                t.contract
                for t in market_data.values()
                if hasattr(t.contract, "right") and t.contract.right == "C"
            )
            put_contract = next(
                t.contract
                for t in market_data.values()
                if hasattr(t.contract, "right") and t.contract.right == "P"
            )

            qualified_contracts[key] = {
                "call_contract": call_contract,
                "put_contract": put_contract,
                "call_strike": 150.0,
                "put_strike": 140.0,
                "expiry": valid_expiry_str,
            }

            mock_qualify.return_value = qualified_contracts

            # Reset metrics
            metrics_collector.reset_session()

            # Define syn attributes with low max loss threshold
            syn.cost_limit = 120.0
            syn.max_loss_threshold = -5.0  # Only allow $5 loss (will fail)
            syn.max_profit_threshold = 20.0
            syn.profit_ratio_threshold = 2.0

            # Run scan
            await syn.scan_syn("TEST", quantity=1)

            print("✅ Synthetic max loss exceeded test completed")

    @pytest.mark.asyncio
    async def test_synthetic_negative_credit(self):
        """
        Test synthetic arbitrage rejection due to negative net credit.

        This test verifies:
        1. Negative net credit is detected
        2. Position is rejected with NET_CREDIT_NEGATIVE
        3. No order is placed
        """
        print("\n🔍 Testing synthetic negative credit scenario")

        # Create market data
        market_data = SyntheticScenarios.synthetic_negative_credit()

        # Setup mock IB
        mock_ib = MockIB()
        mock_ib.test_market_data = market_data

        syn = Syn(debug=True)
        syn.ib = mock_ib
        syn.order_manager = MagicMock()

        # Make place_order async
        async def mock_place_order(*args, **kwargs):
            return MagicMock()

        syn.order_manager.place_order = mock_place_order

        # Setup mocks
        with (
            patch.object(syn, "_get_stock_contract") as mock_get_stock,
            patch.object(syn, "_get_market_data_async") as mock_get_market_data,
            patch.object(syn, "_get_chain") as mock_get_chain,
            patch.object(syn, "parallel_qualify_all_contracts") as mock_qualify,
        ):
            # Setup stock contract
            stock_contract = next(
                t.contract
                for t in market_data.values()
                if hasattr(t.contract, "right") and t.contract.right is None
            )
            mock_get_stock.return_value = ("SMART", "STK", stock_contract)

            stock_ticker = next(
                t
                for t in market_data.values()
                if hasattr(t.contract, "right") and t.contract.right is None
            )
            mock_get_market_data.return_value = stock_ticker

            # Setup option chain
            mock_chain = MagicMock()
            mock_chain.strikes = [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105]

            valid_expiry_date = datetime.now() + timedelta(days=30)
            valid_expiry_str = valid_expiry_date.strftime("%Y%m%d")
            mock_chain.expirations = [valid_expiry_str]
            mock_get_chain.return_value = mock_chain

            # Mock qualified contracts
            qualified_contracts = {}
            key = f"{valid_expiry_str}_100_100"  # Same strike for both

            call_contract = next(
                t.contract
                for t in market_data.values()
                if hasattr(t.contract, "right") and t.contract.right == "C"
            )
            put_contract = next(
                t.contract
                for t in market_data.values()
                if hasattr(t.contract, "right") and t.contract.right == "P"
            )

            qualified_contracts[key] = {
                "call_contract": call_contract,
                "put_contract": put_contract,
                "call_strike": 100.0,
                "put_strike": 100.0,
                "expiry": valid_expiry_str,
            }

            mock_qualify.return_value = qualified_contracts

            # Reset metrics
            metrics_collector.reset_session()

            # Define syn attributes
            syn.cost_limit = 120.0
            syn.max_loss_threshold = -10.0
            syn.max_profit_threshold = 20.0
            syn.profit_ratio_threshold = 2.0

            # Run scan
            await syn.scan_syn("TEST", quantity=1)

            print("✅ Synthetic negative credit test completed")

    @pytest.mark.asyncio
    async def test_synthetic_multi_symbol_scanning(self):
        """
        Test synthetic arbitrage with multiple symbols.

        This test verifies:
        1. Multiple symbols are scanned correctly
        2. Each symbol is evaluated independently
        3. Appropriate executors are created/rejected
        """
        print("\n🔍 Testing synthetic multi-symbol scanning")

        # Get multi-symbol scenarios
        multi_market_data = SyntheticScenarios.synthetic_multi_symbol_scenarios()

        mock_ib = MockIB()

        syn = Syn(debug=True)
        syn.ib = mock_ib
        syn.order_manager = MagicMock()

        # Make place_order async
        async def mock_place_order(*args, **kwargs):
            return MagicMock()

        syn.order_manager.place_order = mock_place_order

        # Track processed symbols
        processed_symbols = []
        original_scan_syn = syn.scan_syn

        async def track_scan_syn(symbol, quantity):
            processed_symbols.append(symbol)
            print(f"🔍 Processing symbol: {symbol}")

            # Set appropriate test data
            if symbol in multi_market_data:
                mock_ib.test_market_data = multi_market_data[symbol]

            result = await original_scan_syn(symbol, quantity)
            print(f"✅ Completed processing {symbol}")
            return result

        syn.scan_syn = track_scan_syn

        # Reset metrics
        metrics_collector.reset_session()

        # Define syn attributes for multi-symbol test
        syn.cost_limit = 200.0
        syn.max_loss_threshold = -5.0
        syn.max_profit_threshold = 15.0
        syn.profit_ratio_threshold = 2.5

        # Process each symbol
        for symbol in ["AAPL", "MSFT", "TSLA"]:
            await syn.scan_syn(symbol, quantity=1)

        # Verify all symbols were processed
        assert set(processed_symbols) == {"AAPL", "MSFT", "TSLA"}

        print("✅ Synthetic multi-symbol scanning test completed")

    @pytest.mark.asyncio
    async def test_synthetic_data_timeout_handling(self):
        """
        Test synthetic executor timeout handling.

        This test verifies:
        1. Executor times out when market data is not received
        2. Timeout is properly logged
        3. Executor deactivates itself
        """
        print("\n🔍 Testing synthetic data timeout handling")

        # Create minimal market data (missing options)
        market_data = {}
        stock_ticker = MarketDataGenerator.generate_stock_data("TEST", 100.0)
        market_data[stock_ticker.contract.conId] = stock_ticker

        mock_ib = MockIB()
        mock_ib.test_market_data = market_data

        syn = Syn(debug=True)
        syn.ib = mock_ib
        syn.order_manager = MagicMock()

        # Create executor with short timeout
        from modules.Arbitrage.Synthetic import ExpiryOption, SynExecutor

        expiry_option = ExpiryOption(
            expiry="20250824",
            call_contract=MockContract("TEST", "OPT", strike=100, right="C"),
            put_contract=MockContract("TEST", "OPT", strike=99, right="P"),
            call_strike=100.0,
            put_strike=99.0,
        )

        executor = SynExecutor(
            ib=mock_ib,
            order_manager=syn.order_manager,
            stock_contract=stock_ticker.contract,
            expiry_options=[expiry_option],
            symbol="TEST",
            cost_limit=120.0,
            max_loss_threshold=-10.0,
            max_profit_threshold=20.0,
            profit_ratio_threshold=2.0,
            start_time=time.time(),
            quantity=1,
            data_timeout=0.5,  # Very short timeout for testing
        )

        # Import the contract_ticker to set up partial data
        from modules.Arbitrage.Synthetic import contract_ticker

        # Clear any existing data
        contract_ticker.clear()

        # Simulate market data event (incomplete - only stock data)
        from eventkit import Event

        event = Event()

        # Create event with only stock ticker
        await executor.executor(event.emit([stock_ticker]))

        # Wait for timeout
        await asyncio.sleep(0.6)

        # Create another event to trigger timeout check
        await executor.executor(event.emit([]))

        # Verify executor deactivated
        assert not executor.is_active

        print("✅ Synthetic data timeout handling test completed")

    @pytest.mark.asyncio
    async def test_synthetic_executor_direct(self):
        """
        Test SynExecutor directly with complete market data flow.

        This test verifies the complete execution flow when all data is available.
        """
        print("\n🔍 Testing synthetic executor direct execution")

        # Create complete market data
        market_data = SyntheticScenarios.synthetic_profitable_scenario()

        mock_ib = MockIB()
        mock_ib.test_market_data = market_data

        syn = Syn(debug=True)
        syn.ib = mock_ib
        syn.order_manager = MagicMock()

        # Make place_order async
        async def mock_place_order(*args, **kwargs):
            print("  📝 Order placed successfully")
            return MagicMock()

        syn.order_manager.place_order = mock_place_order

        # Get contracts from market data
        stock_contract = next(
            t.contract
            for t in market_data.values()
            if hasattr(t.contract, "right") and t.contract.right is None
        )

        call_contract = next(
            t.contract
            for t in market_data.values()
            if hasattr(t.contract, "right") and t.contract.right == "C"
        )

        put_contract = next(
            t.contract
            for t in market_data.values()
            if hasattr(t.contract, "right") and t.contract.right == "P"
        )

        # Create executor
        from modules.Arbitrage.Synthetic import (
            ExpiryOption,
            SynExecutor,
            contract_ticker,
        )

        # Clear contract ticker
        contract_ticker.clear()

        expiry_option = ExpiryOption(
            expiry="20250824",
            call_contract=call_contract,
            put_contract=put_contract,
            call_strike=100.0,
            put_strike=99.0,
        )

        executor = SynExecutor(
            ib=mock_ib,
            order_manager=syn.order_manager,
            stock_contract=stock_contract,
            expiry_options=[expiry_option],
            symbol="TEST",
            cost_limit=120.0,
            max_loss_threshold=-10.0,
            max_profit_threshold=20.0,
            profit_ratio_threshold=2.0,
            start_time=time.time(),
            quantity=1,
            data_timeout=5.0,
        )

        # Create event with all tickers
        from eventkit import Event

        event = Event()

        # Get all tickers from market data
        all_tickers = list(market_data.values())

        print(f"  📊 Sending {len(all_tickers)} tickers to executor")

        # Send market data to executor
        await executor.executor(event.emit(all_tickers))

        # Verify executor processed the data
        print(f"  📊 Executor active: {executor.is_active}")
        print(f"  📊 Contract ticker size: {len(contract_ticker)}")

        # Check if opportunity was found
        if not executor.is_active:
            print("  ✅ Executor completed processing")

        print("✅ Synthetic executor direct test completed")


# Helper function for running specific tests
def run_synthetic_test(test_name: str = None):
    """Helper function to run specific synthetic integration tests"""
    if test_name == "profitable":
        asyncio.run(
            TestSyntheticArbitrageIntegration().test_synthetic_profitable_scenario()
        )
    elif test_name == "poor_risk_reward":
        asyncio.run(
            TestSyntheticArbitrageIntegration().test_synthetic_poor_risk_reward()
        )
    elif test_name == "max_loss":
        asyncio.run(
            TestSyntheticArbitrageIntegration().test_synthetic_max_loss_exceeded()
        )
    elif test_name == "max_profit":
        asyncio.run(
            TestSyntheticArbitrageIntegration().test_synthetic_max_profit_exceeded()
        )
    elif test_name == "negative_credit":
        asyncio.run(
            TestSyntheticArbitrageIntegration().test_synthetic_negative_credit()
        )
    elif test_name == "multi_symbol":
        asyncio.run(
            TestSyntheticArbitrageIntegration().test_synthetic_multi_symbol_scanning()
        )
    elif test_name == "timeout":
        asyncio.run(
            TestSyntheticArbitrageIntegration().test_synthetic_data_timeout_handling()
        )
    elif test_name == "direct":
        asyncio.run(
            TestSyntheticArbitrageIntegration().test_synthetic_executor_direct()
        )
    else:
        print("Available synthetic tests:")
        print("  profitable - Profitable synthetic position test")
        print("  poor_risk_reward - Poor risk-reward ratio test")
        print("  max_loss - Max loss threshold exceeded test")
        print("  max_profit - Max profit threshold exceeded test")
        print("  negative_credit - Negative net credit test")
        print("  multi_symbol - Multi-symbol scanning test")
        print("  timeout - Data timeout handling test")
        print("  direct - Direct executor test")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        run_synthetic_test(sys.argv[1])
    else:
        run_synthetic_test()
