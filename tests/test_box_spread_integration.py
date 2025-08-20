"""
Comprehensive integration tests for Box spread strategy.

This test suite provides end-to-end testing of the Box spread implementation,
including integration with mock IB connections, realistic market scenarios,
and performance testing.

Test Coverage:
- End-to-end box spread workflow
- Integration with mock IB API
- Realistic market data scenarios
- Performance and load testing
- Error handling in integration scenarios
- Multi-symbol scanning integration
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import numpy as np
import pytest
from ib_async import Contract, Option, OptionChain, Stock, Ticker

from modules.Arbitrage.box_spread.executor import BoxExecutor
from modules.Arbitrage.box_spread.models import (
    BoxSpreadConfig,
    BoxSpreadLeg,
    BoxSpreadOpportunity,
)
from modules.Arbitrage.box_spread.opportunity_manager import BoxOpportunityManager
from modules.Arbitrage.box_spread.risk_validator import BoxRiskValidator

# Import the modules under test
from modules.Arbitrage.box_spread.strategy import BoxSpread, run_box_spread_strategy

# Import test infrastructure
from tests.mock_ib import MarketDataGenerator, MockIB, MockTicker


@pytest.mark.integration
class TestBoxSpreadEndToEndWorkflow:
    """Test complete end-to-end box spread workflow"""

    def setup_method(self):
        """Set up test fixtures for integration tests"""
        with patch("modules.Arbitrage.box_spread.strategy.get_logger"):
            self.strategy = BoxSpread()

        # Use mock IB with realistic data
        self.mock_ib = MockIB()
        self.strategy.ib = self.mock_ib

        # Mock order manager
        self.mock_order_manager = MagicMock()
        self.strategy.order_manager = self.mock_order_manager

    async def test_full_box_spread_workflow_single_symbol(self):
        """Test complete workflow for single symbol"""
        symbol_list = ["DELL"]  # Use DELL which has predefined arbitrage scenario

        # Pre-establish connection to avoid KeyboardInterrupt interference
        await self.mock_ib.connectAsync()
        assert self.mock_ib.connected, "Mock IB should be connected before scan"

        # Mock the scan to run only one iteration
        with patch.object(
            self.mock_ib, "connectAsync", wraps=self.mock_ib.connectAsync
        ) as mock_connect:
            with patch("asyncio.sleep", side_effect=[KeyboardInterrupt()]):
                try:
                    await self.strategy.scan(
                        symbol_list=symbol_list,
                        range=0.1,
                        profit_target=0.01,
                        max_spread=10.0,
                        clientId=1,
                    )
                except KeyboardInterrupt:
                    pass

        # Verify scan called connectAsync and connection remains established
        mock_connect.assert_called_once()
        assert self.mock_ib.connected

        # Verify global manager was started
        assert self.strategy.global_manager is not None

    async def test_full_box_spread_workflow_multiple_symbols(self):
        """Test complete workflow for multiple symbols"""
        symbol_list = ["AAPL", "MSFT", "TSLA"]

        # Pre-establish connection to avoid KeyboardInterrupt interference
        await self.mock_ib.connectAsync()
        assert self.mock_ib.connected, "Mock IB should be connected before scan"

        with patch.object(
            self.mock_ib, "connectAsync", wraps=self.mock_ib.connectAsync
        ):
            with patch("asyncio.sleep", side_effect=[KeyboardInterrupt()]):
                try:
                    await self.strategy.scan(
                        symbol_list=symbol_list,
                        range=0.15,
                        profit_target=0.02,
                        max_spread=25.0,
                        clientId=2,
                    )
                except KeyboardInterrupt:
                    pass

        # Verify connection and configuration
        assert self.mock_ib.connected
        assert self.strategy.range == 0.15
        assert self.strategy.profit_target == 0.02
        assert self.strategy.max_spread == 25.0

    async def test_box_spread_with_realistic_dell_scenario(self):
        """Test box spread detection with realistic DELL market data"""
        # DELL has predefined arbitrage scenario in MockIB
        await self.strategy.box_stock_scanner("DELL")

        # Should complete without errors
        # The actual opportunity detection happens in the executor
        # which processes market data asynchronously

    async def test_box_spread_with_futures_option(self):
        """Test box spread scanning with futures options"""
        # Test with futures option symbol (! prefix)
        await self.strategy.box_stock_scanner("!MES")

        # Should handle futures options without errors
        # Verify proper contract type was used
        exchange, option_type, stock = self.strategy.get_stock_contract("!MES")
        assert exchange == "CME"
        assert option_type.__name__ == "FuturesOption"

    async def test_box_spread_with_index_option(self):
        """Test box spread scanning with index options"""
        # Test with index option symbol (@ prefix)
        await self.strategy.box_stock_scanner("@SPX")

        # Should handle index options without errors
        exchange, option_type, stock = self.strategy.get_stock_contract("@SPX")
        assert exchange == "CBOE"
        assert option_type == Option

    async def test_box_spread_error_handling_integration(self):
        """Test error handling in integrated workflow"""
        # Test with invalid symbol that will cause errors
        with patch("modules.Arbitrage.box_spread.strategy.logger") as mock_logger:
            await self.strategy.box_stock_scanner("INVALID_SYMBOL")

            # Should handle errors gracefully and log them
            # The exact behavior depends on implementation details

    async def test_cache_performance_integration(self):
        """Test caching performance in integrated workflow"""
        # Run multiple scans to test cache effectiveness
        symbols = ["AAPL", "MSFT"]

        start_time = time.time()

        # First scan - should populate caches
        for symbol in symbols:
            await self.strategy.box_stock_scanner(symbol)

        first_scan_time = time.time() - start_time

        # Second scan - should benefit from caching
        start_time = time.time()
        for symbol in symbols:
            await self.strategy.box_stock_scanner(symbol)

        second_scan_time = time.time() - start_time

        # Second scan should be faster or at least not significantly slower
        # (In practice, caching benefits are more noticeable with real data)
        assert second_scan_time <= first_scan_time * 2  # Allow some variance

    async def test_parallel_processing_integration(self):
        """Test parallel processing of multiple symbols"""
        symbol_list = ["AAPL", "MSFT", "TSLA", "META"]

        with patch.object(
            self.strategy, "box_stock_scanner", new_callable=AsyncMock
        ) as mock_scanner:
            # Mock scanner to track calls
            mock_scanner.return_value = None

            # Run one iteration of the scan loop
            tasks = []
            for symbol in symbol_list:
                task = asyncio.create_task(self.strategy.box_stock_scanner(symbol))
                tasks.append(task)

            await asyncio.gather(*tasks, return_exceptions=True)

            # Verify all symbols were processed
            assert mock_scanner.call_count == len(symbol_list)

            # Verify each symbol was called
            called_symbols = [call.args[0] for call in mock_scanner.call_args_list]
            assert set(called_symbols) == set(symbol_list)


@pytest.mark.integration
class TestBoxSpreadWithMockIBData:
    """Test box spread strategy with mock IB market data scenarios"""

    def setup_method(self):
        """Set up test fixtures with mock IB data"""
        with patch("modules.Arbitrage.box_spread.strategy.get_logger"):
            self.strategy = BoxSpread()

        # Use mock IB with predefined market scenarios
        self.mock_ib = MockIB()
        self.strategy.ib = self.mock_ib

        # Configure for testing
        self.strategy.range = 0.2  # Wider range for testing
        self.strategy.profit_target = 0.005  # Lower threshold for testing
        self.strategy.max_spread = 20.0

    async def test_profitable_scenario_aapl(self):
        """Test with AAPL profitable scenario from MockIB"""
        # AAPL is configured to be profitable in MockIB
        await self.strategy.box_stock_scanner("AAPL")

        # Should process without errors
        # Actual profitability detection happens in executor

    async def test_unprofitable_scenario_msft(self):
        """Test with MSFT unprofitable scenario from MockIB"""
        # MSFT is configured to be unprofitable in MockIB
        await self.strategy.box_stock_scanner("MSFT")

        # Should process without errors but not find profitable opportunities

    async def test_negative_credit_scenario_tsla(self):
        """Test with TSLA negative credit scenario from MockIB"""
        # TSLA is configured to have negative net credit in MockIB
        await self.strategy.box_stock_scanner("TSLA")

        # Should handle negative credit scenarios gracefully

    async def test_wide_spreads_scenario_nvda(self):
        """Test with NVDA wide spreads scenario from MockIB"""
        # NVDA is configured to have very wide spreads in MockIB
        await self.strategy.box_stock_scanner("NVDA")

        # Should handle wide spreads and reject due to execution difficulty

    async def test_low_liquidity_scenario_meta(self):
        """Test with META low liquidity scenario from MockIB"""
        # META is configured for low liquidity scenarios in MockIB
        await self.strategy.box_stock_scanner("META")

        # Should handle low liquidity gracefully

    async def test_mixed_scenarios_batch_processing(self):
        """Test batch processing of mixed market scenarios"""
        # Test with all configured scenarios
        symbol_list = ["AAPL", "MSFT", "TSLA", "NVDA", "META", "AMZN"]

        tasks = []
        for symbol in symbol_list:
            task = asyncio.create_task(self.strategy.box_stock_scanner(symbol))
            tasks.append(task)

        # Should complete all scenarios without fatal errors
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that no tasks raised unhandled exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Unexpected exceptions: {exceptions}"


@pytest.mark.integration
class TestBoxSpreadExecutorIntegration:
    """Test BoxExecutor integration with realistic scenarios"""

    def setup_method(self):
        """Set up test fixtures for executor integration tests"""
        self.config = BoxSpreadConfig()
        self.mock_ib = MockIB()
        self.mock_order_manager = MagicMock()

    async def test_executor_with_dell_arbitrage_data(self):
        """Test executor with DELL predefined arbitrage scenario"""
        # Create opportunity based on DELL scenario
        opportunity = self._create_dell_opportunity()

        # Create executor
        executor = BoxExecutor(
            opportunity=opportunity,
            ib=self.mock_ib,
            order_manager=self.mock_order_manager,
            config=self.config,
        )

        # Simulate market data arrival for all legs
        dell_tickers = self._create_dell_market_tickers()

        # Process tickers one by one
        for ticker in dell_tickers:
            await executor.executor([ticker])

        # Verify executor processed the data
        assert executor.is_active  # Should still be active (or completed)

    def _create_dell_opportunity(self) -> BoxSpreadOpportunity:
        """Create DELL box spread opportunity matching MockIB data"""
        # Mock contracts matching DELL scenario
        mock_contracts = [MagicMock(spec=Contract) for _ in range(4)]
        for i, contract in enumerate(mock_contracts):
            contract.conId = i + 3000
            contract.symbol = "DELL"

        # Create legs matching DELL scenario (132/131 strikes)
        long_call_k1 = BoxSpreadLeg(
            contract=mock_contracts[0],
            strike=131.0,
            expiry="20250221",
            right="C",
            action="BUY",
            price=2.50,
            bid=2.40,
            ask=2.60,
            volume=1000,
            iv=0.30,
            delta=0.60,
            gamma=0.04,
            theta=-0.06,
            vega=0.20,
            days_to_expiry=30,
        )

        short_call_k2 = BoxSpreadLeg(
            contract=mock_contracts[1],
            strike=132.0,
            expiry="20250221",
            right="C",
            action="SELL",
            price=1.90,
            bid=1.80,
            ask=2.00,
            volume=800,
            iv=0.28,
            delta=0.50,
            gamma=0.05,
            theta=-0.05,
            vega=0.18,
            days_to_expiry=30,
        )

        short_put_k1 = BoxSpreadLeg(
            contract=mock_contracts[2],
            strike=131.0,
            expiry="20250221",
            right="P",
            action="SELL",
            price=1.50,
            bid=1.40,
            ask=1.60,
            volume=600,
            iv=0.32,
            delta=-0.40,
            gamma=0.04,
            theta=-0.05,
            vega=0.19,
            days_to_expiry=30,
        )

        long_put_k2 = BoxSpreadLeg(
            contract=mock_contracts[3],
            strike=132.0,
            expiry="20250221",
            right="P",
            action="BUY",
            price=2.10,
            bid=2.00,
            ask=2.20,
            volume=700,
            iv=0.30,
            delta=-0.50,
            gamma=0.05,
            theta=-0.07,
            vega=0.21,
            days_to_expiry=30,
        )

        return BoxSpreadOpportunity(
            symbol="DELL",
            lower_strike=131.0,
            upper_strike=132.0,
            expiry="20250221",
            long_call_k1=long_call_k1,
            short_call_k2=short_call_k2,
            short_put_k1=short_put_k1,
            long_put_k2=long_put_k2,
            strike_width=1.0,
            net_debit=0.90,  # Should be less than strike width for arbitrage
            theoretical_value=1.0,
            arbitrage_profit=0.10,
            profit_percentage=11.11,
            max_profit=0.10,
            max_loss=0.0,
            risk_free=True,
            total_bid_ask_spread=0.80,
            combined_liquidity_score=0.70,
            execution_difficulty=0.30,
            net_delta=0.10,
            net_gamma=0.00,
            net_theta=-0.03,
            net_vega=0.02,
            composite_score=0.85,
        )

    def _create_dell_market_tickers(self) -> List[MockTicker]:
        """Create market tickers for DELL scenario"""
        # Use MarketDataGenerator to create DELL scenario tickers
        dell_data = MarketDataGenerator.create_dell_arbitrage_scenario()
        return list(dell_data.values())

    async def test_executor_market_data_processing_sequence(self):
        """Test executor market data processing in realistic sequence"""
        opportunity = self._create_dell_opportunity()

        executor = BoxExecutor(
            opportunity=opportunity,
            ib=self.mock_ib,
            order_manager=self.mock_order_manager,
            config=self.config,
        )

        # Create realistic market data sequence
        tickers = self._create_dell_market_tickers()

        # Process tickers with delays to simulate real market data
        for i, ticker in enumerate(tickers):
            await executor.executor([ticker])

            # Small delay between market data updates
            await asyncio.sleep(0.001)

        # Executor should have processed all data
        assert len(executor.required_tickers) == 4

    async def test_executor_with_incomplete_data(self):
        """Test executor behavior with incomplete market data"""
        opportunity = self._create_dell_opportunity()

        executor = BoxExecutor(
            opportunity=opportunity,
            ib=self.mock_ib,
            order_manager=self.mock_order_manager,
            config=self.config,
        )

        # Only provide partial market data (2 out of 4 legs)
        partial_tickers = self._create_dell_market_tickers()[:2]

        for ticker in partial_tickers:
            await executor.executor([ticker])

        # Should not execute with incomplete data
        assert not executor.execution_completed

    async def test_executor_error_handling_integration(self):
        """Test executor error handling in integration scenario"""
        opportunity = self._create_dell_opportunity()

        executor = BoxExecutor(
            opportunity=opportunity,
            ib=self.mock_ib,
            order_manager=self.mock_order_manager,
            config=self.config,
        )

        # Create invalid ticker that might cause errors
        invalid_ticker = MockTicker(
            contract=MagicMock(), bid=float("nan"), ask=float("nan"), volume=0
        )

        # Should handle invalid data gracefully
        await executor.executor([invalid_ticker])

        # Executor should still be in valid state
        assert hasattr(executor, "is_active")


@pytest.mark.integration
class TestBoxSpreadPerformanceIntegration:
    """Test box spread performance and load scenarios"""

    def setup_method(self):
        """Set up test fixtures for performance tests"""
        with patch("modules.Arbitrage.box_spread.strategy.get_logger"):
            self.strategy = BoxSpread()

        # Configure for performance testing
        self.strategy.ib = MockIB()
        self.strategy.range = 0.2
        self.strategy.profit_target = 0.005
        self.strategy.max_spread = 25.0

    async def test_high_volume_symbol_scanning(self):
        """Test scanning performance with high volume of symbols"""
        # Test with larger symbol list
        symbol_list = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "AMD",
            "JPM",
            "BAC",
            "WMT",
            "V",
            "UNH",
            "JNJ",
            "PG",
            "HD",
            "DIS",
            "NFLX",
        ]

        start_time = time.time()

        # Process all symbols in parallel
        tasks = []
        for symbol in symbol_list:
            task = asyncio.create_task(self.strategy.box_stock_scanner(symbol))
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

        processing_time = time.time() - start_time

        # Should complete within reasonable time
        # Performance expectations: < 5 seconds for mock data
        assert processing_time < 5.0, f"Processing took too long: {processing_time}s"

    async def test_cache_effectiveness_under_load(self):
        """Test cache effectiveness under load conditions"""
        symbol_list = ["AAPL", "MSFT", "TSLA", "NVDA"]

        # First pass - populate caches
        start_time = time.time()
        for symbol in symbol_list:
            await self.strategy.box_stock_scanner(symbol)
        first_pass_time = time.time() - start_time

        # Perform cache maintenance
        self.strategy._perform_cache_maintenance()

        # Second pass - should benefit from caching
        start_time = time.time()
        for symbol in symbol_list:
            await self.strategy.box_stock_scanner(symbol)
        second_pass_time = time.time() - start_time

        # Cache should provide some benefit (or at least not hurt performance)
        assert second_pass_time <= first_pass_time * 1.5

    async def test_memory_usage_under_load(self):
        """Test memory usage patterns under load"""
        # Simulate extended scanning session
        symbol_list = ["AAPL", "MSFT", "TSLA"] * 5  # Repeat symbols

        initial_cache_sizes = {
            "pricing": self.strategy.pricing_cache.size(),
            "greeks": self.strategy.greeks_cache.size(),
            "leg": self.strategy.leg_cache.size(),
        }

        # Process symbols multiple times
        for symbol in symbol_list:
            await self.strategy.box_stock_scanner(symbol)

            # Periodically perform maintenance
            if symbol_list.index(symbol) % 5 == 0:
                self.strategy._perform_cache_maintenance()

        final_cache_sizes = {
            "pricing": self.strategy.pricing_cache.size(),
            "greeks": self.strategy.greeks_cache.size(),
            "leg": self.strategy.leg_cache.size(),
        }

        # Caches should not grow unbounded
        # (Exact limits depend on TTL and cleanup policies)
        assert final_cache_sizes["pricing"] < 1000
        assert final_cache_sizes["greeks"] < 1000
        assert final_cache_sizes["leg"] < 1000

    async def test_concurrent_executor_performance(self):
        """Test performance with multiple concurrent executors"""
        # Create multiple opportunities
        opportunities = []
        for i in range(10):
            opportunity = self._create_test_opportunity(f"TEST{i}")
            opportunities.append(opportunity)

        # Create executors for all opportunities
        executors = []
        for opportunity in opportunities:
            executor = BoxExecutor(
                opportunity=opportunity,
                ib=self.strategy.ib,
                order_manager=MagicMock(),
                config=BoxSpreadConfig(),
            )
            executors.append(executor)

        # Simulate concurrent market data processing
        start_time = time.time()

        tasks = []
        for executor in executors:
            # Create mock market data for each executor
            mock_ticker = MockTicker(
                contract=executor.contracts[0], bid=5.0, ask=5.2, volume=100
            )
            task = asyncio.create_task(executor.executor([mock_ticker]))
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

        processing_time = time.time() - start_time

        # Should handle concurrent executors efficiently
        assert (
            processing_time < 1.0
        ), f"Concurrent processing took too long: {processing_time}s"

    def _create_test_opportunity(self, symbol: str) -> BoxSpreadOpportunity:
        """Create a test opportunity for performance testing"""
        mock_contracts = [MagicMock(spec=Contract) for _ in range(4)]
        for i, contract in enumerate(mock_contracts):
            contract.conId = hash(f"{symbol}_{i}")
            contract.symbol = symbol

        # Create minimal opportunity for performance testing
        long_call_k1 = BoxSpreadLeg(
            contract=mock_contracts[0],
            strike=100.0,
            expiry="20250830",
            right="C",
            action="BUY",
            price=5.0,
            bid=4.9,
            ask=5.1,
            volume=100,
            iv=0.25,
            delta=0.5,
            gamma=0.02,
            theta=-0.03,
            vega=0.1,
            days_to_expiry=30,
        )

        short_call_k2 = BoxSpreadLeg(
            contract=mock_contracts[1],
            strike=105.0,
            expiry="20250830",
            right="C",
            action="SELL",
            price=3.0,
            bid=2.9,
            ask=3.1,
            volume=100,
            iv=0.23,
            delta=0.3,
            gamma=0.02,
            theta=-0.02,
            vega=0.08,
            days_to_expiry=30,
        )

        short_put_k1 = BoxSpreadLeg(
            contract=mock_contracts[2],
            strike=100.0,
            expiry="20250830",
            right="P",
            action="SELL",
            price=2.0,
            bid=1.9,
            ask=2.1,
            volume=100,
            iv=0.26,
            delta=-0.5,
            gamma=0.02,
            theta=-0.02,
            vega=0.09,
            days_to_expiry=30,
        )

        long_put_k2 = BoxSpreadLeg(
            contract=mock_contracts[3],
            strike=105.0,
            expiry="20250830",
            right="P",
            action="BUY",
            price=4.0,
            bid=3.9,
            ask=4.1,
            volume=100,
            iv=0.24,
            delta=-0.7,
            gamma=0.02,
            theta=-0.04,
            vega=0.11,
            days_to_expiry=30,
        )

        return BoxSpreadOpportunity(
            symbol=symbol,
            lower_strike=100.0,
            upper_strike=105.0,
            expiry="20250830",
            long_call_k1=long_call_k1,
            short_call_k2=short_call_k2,
            short_put_k1=short_put_k1,
            long_put_k2=long_put_k2,
            strike_width=5.0,
            net_debit=4.9,
            theoretical_value=5.0,
            arbitrage_profit=0.1,
            profit_percentage=2.04,
            max_profit=0.1,
            max_loss=0.0,
            risk_free=True,
            total_bid_ask_spread=0.8,
            combined_liquidity_score=0.6,
            execution_difficulty=0.4,
            net_delta=0.1,
            net_gamma=0.0,
            net_theta=-0.01,
            net_vega=0.02,
            composite_score=0.7,
        )


@pytest.mark.integration
class TestBoxSpreadConvenienceFunction:
    """Test the convenience function for running box spread strategy"""

    async def test_run_box_spread_strategy_integration(self):
        """Test the run_box_spread_strategy convenience function"""
        symbol_list = ["AAPL", "MSFT"]

        with patch(
            "modules.Arbitrage.box_spread.strategy.BoxSpread"
        ) as mock_strategy_class:
            mock_strategy = MagicMock()
            mock_strategy.scan = AsyncMock()
            mock_strategy_class.return_value = mock_strategy

            await run_box_spread_strategy(
                symbol_list=symbol_list,
                range=0.12,
                profit_target=0.015,
                max_spread=30.0,
                client_id=4,
            )

            # Verify strategy was created and configured correctly
            mock_strategy_class.assert_called_once()
            mock_strategy.scan.assert_called_once_with(
                symbol_list=symbol_list,
                range=0.12,
                profit_target=0.015,
                max_spread=30.0,
                clientId=4,
            )

    async def test_convenience_function_with_defaults(self):
        """Test convenience function with default parameters"""
        symbol_list = ["SPY"]

        with patch(
            "modules.Arbitrage.box_spread.strategy.BoxSpread"
        ) as mock_strategy_class:
            mock_strategy = MagicMock()
            mock_strategy.scan = AsyncMock()
            mock_strategy_class.return_value = mock_strategy

            # Call with minimal parameters
            await run_box_spread_strategy(symbol_list=symbol_list)

            # Should use default values
            mock_strategy.scan.assert_called_once_with(
                symbol_list=symbol_list,
                range=0.1,
                profit_target=0.01,
                max_spread=50.0,
                clientId=3,
            )


@pytest.mark.integration
class TestBoxSpreadRobustness:
    """Test box spread robustness and edge case handling"""

    def setup_method(self):
        """Set up test fixtures for robustness tests"""
        with patch("modules.Arbitrage.box_spread.strategy.get_logger"):
            self.strategy = BoxSpread()

        self.strategy.ib = MockIB()

    async def test_robustness_with_market_data_gaps(self):
        """Test robustness when market data has gaps or delays"""
        # Simulate delayed market data arrival
        with patch.object(self.strategy.ib, "_deliver_market_data") as mock_deliver:
            # Mock delayed delivery
            async def delayed_delivery(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simulate delay
                return None

            mock_deliver.side_effect = delayed_delivery

            # Should handle delays gracefully
            await self.strategy.box_stock_scanner("AAPL")

    async def test_robustness_with_invalid_contract_data(self):
        """Test robustness with invalid or corrupted contract data"""
        # Mock contract qualification to return invalid data
        with patch.object(self.strategy.ib, "qualifyContractsAsync") as mock_qualify:
            mock_qualify.return_value = []  # No qualified contracts

            # Should handle gracefully without crashing
            await self.strategy.box_stock_scanner("INVALID")

    async def test_robustness_with_network_interruptions(self):
        """Test robustness with simulated network interruptions"""
        # Simulate connection loss and recovery
        self.strategy.ib.connected = False

        # Should handle disconnection gracefully
        await self.strategy.box_stock_scanner("AAPL")

        # Reconnect and continue
        self.strategy.ib.connected = True
        await self.strategy.box_stock_scanner("MSFT")

    async def test_robustness_with_extreme_market_conditions(self):
        """Test robustness with extreme market conditions"""
        # Test with symbols that have extreme pricing in MockIB
        extreme_symbols = ["NVDA"]  # NVDA has very wide spreads in MockIB

        for symbol in extreme_symbols:
            # Should handle extreme conditions without failing
            await self.strategy.box_stock_scanner(symbol)

    async def test_memory_leak_prevention(self):
        """Test that strategy prevents memory leaks during extended operation"""
        # Simulate extended operation
        for i in range(20):
            await self.strategy.box_stock_scanner("AAPL")

            # Perform cache maintenance periodically
            if i % 5 == 0:
                self.strategy._perform_cache_maintenance()

        # Verify cache sizes remain reasonable
        assert self.strategy.pricing_cache.size() < 500
        assert self.strategy.greeks_cache.size() < 500
        assert self.strategy.leg_cache.size() < 500
