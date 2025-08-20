"""
Comprehensive unit tests for BoxSpread strategy class.

This test suite provides extensive coverage of the BoxSpread strategy implementation,
including market data processing, opportunity detection, risk validation, and
integration with the existing arbitrage framework.

Test Coverage:
- BoxSpread strategy initialization and configuration
- Market data processing and option chain handling
- Opportunity detection and evaluation logic
- Strike selection and filtering algorithms
- Error handling and edge cases
- Integration with IB API mocking
- Performance and caching mechanisms
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

# Import the modules under test
from modules.Arbitrage.box_spread.strategy import BoxSpread
from modules.Arbitrage.Strategy import ArbitrageClass

# Import test infrastructure
from tests.mock_ib import MarketDataGenerator, MockIB, MockTicker


class TestBoxSpreadInitialization:
    """Test BoxSpread strategy initialization and configuration"""

    def test_box_spread_initialization_with_defaults(self):
        """Test BoxSpread initialization with default configuration"""
        with patch("modules.Arbitrage.box_spread.strategy.get_logger"):
            strategy = BoxSpread()

        # Verify basic initialization
        assert strategy is not None
        assert isinstance(strategy.config, BoxSpreadConfig)
        assert strategy.range == 0.1
        assert strategy.profit_target == 0.01
        assert strategy.max_spread == 50.0
        assert strategy.profit_target_multiplier == 1.10

        # Verify cache initialization
        assert strategy.pricing_cache is not None
        assert strategy.greeks_cache is not None
        assert strategy.leg_cache is not None
        assert strategy.cache_manager is not None
        assert strategy.profiler is not None

        # Verify global manager initialization
        assert strategy.global_manager is not None

    def test_box_spread_initialization_with_log_file(self):
        """Test BoxSpread initialization with custom log file"""
        with patch("modules.Arbitrage.box_spread.strategy.get_logger"):
            strategy = BoxSpread(log_file="test_box.log")

        assert strategy is not None
        assert isinstance(strategy.config, BoxSpreadConfig)

    def test_box_spread_inherits_from_arbitrage_class(self):
        """Test that BoxSpread properly inherits from ArbitrageClass"""
        with patch("modules.Arbitrage.box_spread.strategy.get_logger"):
            strategy = BoxSpread()

        assert isinstance(strategy, ArbitrageClass)
        # Verify it has key methods from the base class
        assert hasattr(strategy, "ib")
        assert hasattr(strategy, "order_manager")
        assert hasattr(strategy, "semaphore")


class TestBoxSpreadMarketDataProcessing:
    """Test BoxSpread market data processing and chain handling"""

    def setup_method(self):
        """Set up test fixtures for each test method"""
        with patch("modules.Arbitrage.box_spread.strategy.get_logger"):
            self.strategy = BoxSpread()

        # Mock IB connection
        self.mock_ib = MockIB()
        self.strategy.ib = self.mock_ib

    async def test_get_stock_contract_regular_stock(self):
        """Test getting stock contract for regular symbols"""
        exchange, option_type, stock = self.strategy.get_stock_contract("AAPL")

        assert exchange == "SMART"
        assert option_type == Option
        assert stock.symbol == "AAPL"
        assert stock.secType == "STK"
        assert stock.exchange == "SMART"
        assert stock.currency == "USD"

    async def test_get_stock_contract_futures_option(self):
        """Test getting stock contract for futures options (! prefix)"""
        exchange, option_type, stock = self.strategy.get_stock_contract("!MES")

        assert exchange == "CME"
        assert option_type.__name__ == "FuturesOption"
        assert stock.symbol == "MES"
        assert stock.secType == "IND"
        assert stock.exchange == "CME"

    async def test_get_stock_contract_index_option(self):
        """Test getting stock contract for index options (@ prefix)"""
        exchange, option_type, stock = self.strategy.get_stock_contract("@SPX")

        assert exchange == "CBOE"
        assert option_type == Option
        assert stock.symbol == "SPX"
        assert stock.secType == "IND"
        assert stock.exchange == "CBOE"

    def test_select_anchor_strikes_normal_case(self):
        """Test strike selection around current stock price"""
        available_strikes = [170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0]
        stock_price = 185.0

        selected_strikes = self.strategy._select_anchor_strikes(
            available_strikes, stock_price
        )

        # Should select strikes within range (10% by default)
        # Range: 185 * 0.9 = 166.5 to 185 * 1.1 = 203.5
        expected_strikes = [170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0]
        assert selected_strikes == expected_strikes

    def test_select_anchor_strikes_limited_count(self):
        """Test strike selection with more than max allowed strikes"""
        # Generate many strikes
        available_strikes = [float(i) for i in range(150, 220, 2)]  # 35 strikes
        stock_price = 185.0

        selected_strikes = self.strategy._select_anchor_strikes(
            available_strikes, stock_price
        )

        # Should limit to max_strikes (10 by default)
        assert len(selected_strikes) <= 10

        # Should be centered around stock price
        assert min(selected_strikes) <= stock_price <= max(selected_strikes)

    def test_select_anchor_strikes_edge_case_no_strikes_in_range(self):
        """Test strike selection when no strikes are in range"""
        available_strikes = [100.0, 105.0, 110.0, 250.0, 255.0, 260.0]
        stock_price = 185.0  # No strikes near this price

        selected_strikes = self.strategy._select_anchor_strikes(
            available_strikes, stock_price
        )

        # Should return empty list or limited strikes outside range
        assert len(selected_strikes) == 0

    async def test_create_box_spread_contracts_valid_inputs(self):
        """Test creating valid box spread contracts"""
        symbol = "AAPL"
        expiry = "20250830"
        k1_strike = 180.0
        k2_strike = 185.0
        option_type = Option

        # Mock option chain
        mock_chain = MagicMock()
        mock_chain.exchange = "SMART"
        mock_chain.tradingClass = "AAPL"

        contracts = await self.strategy._create_box_spread_contracts(
            symbol, expiry, k1_strike, k2_strike, option_type, mock_chain
        )

        assert contracts is not None
        assert len(contracts) == 4

        long_call_k1, short_call_k2, short_put_k1, long_put_k2 = contracts

        # Verify long call K1
        assert long_call_k1.symbol == symbol
        assert long_call_k1.strike == k1_strike
        assert long_call_k1.right == "C"
        assert long_call_k1.lastTradeDateOrContractMonth == expiry

        # Verify short call K2
        assert short_call_k2.symbol == symbol
        assert short_call_k2.strike == k2_strike
        assert short_call_k2.right == "C"

        # Verify short put K1
        assert short_put_k1.symbol == symbol
        assert short_put_k1.strike == k1_strike
        assert short_put_k1.right == "P"

        # Verify long put K2
        assert long_put_k2.symbol == symbol
        assert long_put_k2.strike == k2_strike
        assert long_put_k2.right == "P"

    async def test_create_box_spread_contracts_error_handling(self):
        """Test error handling in contract creation"""
        with patch("modules.Arbitrage.box_spread.strategy.logger") as mock_logger:
            # Test with invalid inputs that cause exception
            with patch.object(
                Option, "__init__", side_effect=Exception("Contract error")
            ):
                contracts = await self.strategy._create_box_spread_contracts(
                    "INVALID", "20250830", 180.0, 185.0, Option, MagicMock()
                )

        assert contracts is None


class TestBoxSpreadOpportunityDetection:
    """Test BoxSpread opportunity detection and evaluation"""

    def setup_method(self):
        """Set up test fixtures for each test method"""
        with patch("modules.Arbitrage.box_spread.strategy.get_logger"):
            self.strategy = BoxSpread()

        # Mock IB connection with test data
        self.mock_ib = MockIB()
        self.strategy.ib = self.mock_ib

        # Mock order manager
        self.mock_order_manager = MagicMock()
        self.strategy.order_manager = self.mock_order_manager

    async def test_box_stock_scanner_regular_stock(self):
        """Test scanning for box spreads in regular stock"""
        with patch.object(self.strategy, "_get_market_data_async") as mock_market_data:
            with patch.object(self.strategy, "_get_chain") as mock_get_chain:
                with patch.object(self.strategy, "search_box_in_chain") as mock_search:

                    # Setup mocks
                    mock_ticker = MockTicker(
                        contract=MagicMock(),
                        bid=184.0,
                        ask=186.0,
                        close=185.0,
                        last=185.5,
                    )
                    mock_market_data.return_value = mock_ticker

                    mock_chain = MagicMock()
                    mock_get_chain.return_value = mock_chain

                    # Execute
                    await self.strategy.box_stock_scanner("AAPL")

                    # Verify calls
                    mock_market_data.assert_called_once()
                    mock_get_chain.assert_called_once()
                    # The method should pass the stock contract created by get_stock_contract
                    expected_stock = Stock("AAPL", "SMART", "USD")
                    mock_search.assert_called_once_with(
                        mock_chain,
                        Option,
                        expected_stock,
                        185.5,
                    )

    async def test_box_stock_scanner_futures_option(self):
        """Test scanning for box spreads in futures options"""
        with patch.object(self.strategy, "_get_market_data_async") as mock_market_data:
            with patch.object(self.strategy, "_get_chains") as mock_get_chains:
                with patch.object(self.strategy, "search_box_in_chain") as mock_search:

                    # Setup mocks
                    mock_ticker = MockTicker(
                        contract=MagicMock(),
                        bid=4400.0,
                        ask=4402.0,
                        close=4401.0,
                        last=4401.5,
                    )
                    mock_market_data.return_value = mock_ticker

                    mock_chains = [MagicMock(), MagicMock()]
                    mock_get_chains.return_value = mock_chains

                    # Execute
                    await self.strategy.box_stock_scanner("!MES")

                    # Verify calls
                    mock_market_data.assert_called_once()
                    mock_get_chains.assert_called_once()
                    assert mock_search.call_count == len(mock_chains)

    async def test_box_stock_scanner_error_handling(self):
        """Test error handling in stock scanner"""
        with patch.object(
            self.strategy,
            "_get_market_data_async",
            side_effect=Exception("Market data error"),
        ):
            with patch("modules.Arbitrage.box_spread.strategy.logger") as mock_logger:
                with patch(
                    "modules.Arbitrage.box_spread.strategy.metrics_collector"
                ) as mock_metrics:

                    await self.strategy.box_stock_scanner("ERROR")

                    # Verify error was logged
                    mock_logger.error.assert_called_once()
                    mock_metrics.add_rejection_reason.assert_called_once()

    async def test_search_box_in_chain_normal_processing(self):
        """Test normal processing of option chain for box spreads"""
        # Mock option chain
        mock_chain = MagicMock()
        mock_chain.strikes = [175.0, 180.0, 185.0, 190.0, 195.0]
        mock_chain.expirations = ["20250830", "20250927", "20251025"]

        # Mock stock contract
        mock_stock = MagicMock()
        mock_stock.symbol = "AAPL"

        with patch.object(
            self.strategy, "_process_expiry_for_box_spreads"
        ) as mock_process:
            await self.strategy.search_box_in_chain(
                mock_chain, Option, mock_stock, 185.0
            )

            # Should process limited number of expiries (max 3)
            expected_calls = min(3, len(mock_chain.expirations))
            assert mock_process.call_count == expected_calls

    async def test_search_box_in_chain_with_semaphore(self):
        """Test that search_box_in_chain respects semaphore limits"""
        # Mock semaphore
        mock_semaphore = AsyncMock()
        self.strategy.semaphore = mock_semaphore

        mock_chain = MagicMock()
        mock_chain.strikes = [180.0, 185.0]
        mock_chain.expirations = ["20250830"]

        mock_stock = MagicMock()
        mock_stock.symbol = "TEST"

        with patch.object(self.strategy, "_process_expiry_for_box_spreads"):
            await self.strategy.search_box_in_chain(
                mock_chain, Option, mock_stock, 185.0
            )

            # Verify semaphore was used
            mock_semaphore.__aenter__.assert_called_once()
            mock_semaphore.__aexit__.assert_called_once()

    async def test_process_expiry_for_box_spreads_strike_pairs(self):
        """Test processing expiry generates correct strike pairs"""
        mock_chain = MagicMock()
        expiry = "20250830"
        anchor_strikes = [180.0, 185.0, 190.0]

        with patch.object(
            self.strategy, "_evaluate_box_spread_opportunity"
        ) as mock_evaluate:
            await self.strategy._process_expiry_for_box_spreads(
                mock_chain, expiry, anchor_strikes, Option, MagicMock()
            )

            # Should generate pairs where k1 < k2
            # Expected pairs: (180,185), (180,190), (185,190)
            assert mock_evaluate.call_count == 3

    async def test_process_expiry_respects_max_spread_limit(self):
        """Test that processing respects maximum spread width"""
        # Set small max_spread for testing
        self.strategy.max_spread = 2.0

        mock_chain = MagicMock()
        expiry = "20250830"
        anchor_strikes = [180.0, 185.0, 190.0]  # Max difference is 10, but limit is 2

        with patch.object(
            self.strategy, "_evaluate_box_spread_opportunity"
        ) as mock_evaluate:
            await self.strategy._process_expiry_for_box_spreads(
                mock_chain, expiry, anchor_strikes, Option, MagicMock()
            )

            # Should only generate pairs within max_spread limit
            # Only (180,185) has difference <= 2.0, but that's 5.0
            # So no pairs should be generated
            assert mock_evaluate.call_count == 0

    async def test_evaluate_box_spread_opportunity_success(self):
        """Test successful evaluation of box spread opportunity"""
        mock_chain = MagicMock()
        mock_chain.exchange = "SMART"
        mock_chain.tradingClass = "AAPL"

        mock_stock = MagicMock()
        mock_stock.symbol = "AAPL"

        with patch.object(self.strategy, "_create_box_spread_contracts") as mock_create:
            with patch.object(
                self.strategy, "_create_and_start_executor"
            ) as mock_executor:
                with patch.object(
                    self.strategy.ib, "qualifyContractsAsync"
                ) as mock_qualify:
                    with patch.object(self.strategy.ib, "reqMktData") as mock_req_data:

                        # Mock contract creation
                        mock_contracts = (
                            MagicMock(),
                            MagicMock(),
                            MagicMock(),
                            MagicMock(),
                        )
                        mock_create.return_value = mock_contracts

                        await self.strategy._evaluate_box_spread_opportunity(
                            mock_chain, "20250830", 180.0, 185.0, Option, mock_stock
                        )

                        # Verify all steps were called
                        mock_create.assert_called_once()
                        mock_qualify.assert_called_once_with(*mock_contracts)
                        assert mock_req_data.call_count == 4
                        mock_executor.assert_called_once()

    async def test_evaluate_box_spread_opportunity_no_contracts(self):
        """Test handling when contract creation fails"""
        mock_chain = MagicMock()
        mock_stock = MagicMock()

        with patch.object(self.strategy, "_create_box_spread_contracts") as mock_create:
            with patch.object(
                self.strategy, "_create_and_start_executor"
            ) as mock_executor:

                # Mock contract creation failure
                mock_create.return_value = None

                await self.strategy._evaluate_box_spread_opportunity(
                    mock_chain, "20250830", 180.0, 185.0, Option, mock_stock
                )

                # Should not proceed to executor creation
                mock_executor.assert_not_called()

    async def test_evaluate_box_spread_opportunity_error_handling(self):
        """Test error handling in opportunity evaluation"""
        with patch.object(
            self.strategy,
            "_create_box_spread_contracts",
            side_effect=Exception("Contract error"),
        ):
            with patch("modules.Arbitrage.box_spread.strategy.logger") as mock_logger:

                await self.strategy._evaluate_box_spread_opportunity(
                    MagicMock(), "20250830", 180.0, 185.0, Option, MagicMock()
                )

                # Should log the error
                mock_logger.debug.assert_called_once()


class TestBoxSpreadCacheManagement:
    """Test BoxSpread caching mechanisms and performance optimization"""

    def setup_method(self):
        """Set up test fixtures for each test method"""
        with patch("modules.Arbitrage.box_spread.strategy.get_logger"):
            self.strategy = BoxSpread()

    def test_cache_initialization(self):
        """Test that all caches are properly initialized"""
        assert self.strategy.pricing_cache is not None
        assert self.strategy.greeks_cache is not None
        assert self.strategy.leg_cache is not None
        assert self.strategy.cache_manager is not None

    def test_perform_cache_maintenance(self):
        """Test cache maintenance functionality"""
        # Add some test data to caches
        self.strategy.pricing_cache.put("test_key1", "test_value1")
        self.strategy.greeks_cache.put("test_key2", "test_value2")
        self.strategy.leg_cache.put("test_key3", "test_value3")

        with patch("modules.Arbitrage.box_spread.strategy.logger") as mock_logger:
            # Perform maintenance
            self.strategy._perform_cache_maintenance()

            # Should not log anything if no cleanup was needed
            # (since we just added fresh data)

    def test_cache_maintenance_with_expired_entries(self):
        """Test cache maintenance with expired entries"""
        # Mock cache cleanup to return expired count
        with patch.object(
            self.strategy.pricing_cache, "cleanup_expired", return_value=5
        ):
            with patch.object(
                self.strategy.greeks_cache, "cleanup_expired", return_value=3
            ):
                with patch.object(
                    self.strategy.leg_cache, "cleanup_expired", return_value=2
                ):
                    with patch.object(
                        self.strategy.cache_manager, "cleanup_if_needed", return_value=0
                    ):
                        with patch(
                            "modules.Arbitrage.box_spread.strategy.logger"
                        ) as mock_logger:

                            self.strategy._perform_cache_maintenance()

                            # Should log maintenance activity
                            mock_logger.info.assert_called_once()


class TestBoxSpreadScanMethod:
    """Test BoxSpread main scan method and orchestration"""

    def setup_method(self):
        """Set up test fixtures for each test method"""
        with patch("modules.Arbitrage.box_spread.strategy.get_logger"):
            self.strategy = BoxSpread()

        # Mock IB connection with proper async methods
        self.mock_ib = AsyncMock()
        self.mock_ib.connectAsync.return_value = (
            True  # Make connectAsync return immediately
        )
        self.strategy.ib = self.mock_ib

        # Mock global manager
        self.mock_global_manager = MagicMock()
        self.strategy.global_manager = self.mock_global_manager

    async def test_scan_method_parameter_setting(self):
        """Test that scan method sets parameters correctly"""
        symbol_list = ["AAPL", "MSFT"]
        range_val = 0.15
        profit_target = 0.02
        max_spread = 25.0
        profit_multiplier = 1.15
        client_id = 5

        with patch.object(self.strategy, "box_stock_scanner", new_callable=AsyncMock):
            with patch(
                "asyncio.sleep", side_effect=[KeyboardInterrupt()]
            ):  # Stop at first sleep (main loop)
                with patch.object(self.strategy, "_perform_cache_maintenance"):
                    with patch.object(self.strategy.global_manager, "start_scan"):
                        try:
                            await self.strategy.scan(
                                symbol_list=symbol_list,
                                range=range_val,
                                profit_target=profit_target,
                                max_spread=max_spread,
                                profit_target_multiplier=profit_multiplier,
                                clientId=client_id,
                            )
                        except KeyboardInterrupt:
                            pass  # Expected - this means we got past parameter setting

        # Verify parameters were set
        assert self.strategy.range == range_val
        assert self.strategy.profit_target == profit_target
        assert self.strategy.max_spread == max_spread
        assert self.strategy.profit_target_multiplier == profit_multiplier
        assert self.strategy.config.min_arbitrage_profit == profit_target
        assert self.strategy.config.max_strike_width == max_spread

    async def test_scan_method_ib_connection(self):
        """Test that scan method connects to IB"""
        symbol_list = ["AAPL"]

        with patch.object(self.strategy, "box_stock_scanner", new_callable=AsyncMock):
            with patch(
                "asyncio.sleep", side_effect=[KeyboardInterrupt()]
            ):  # Stop at first sleep (main loop)
                with patch.object(self.strategy, "_perform_cache_maintenance"):
                    with patch.object(self.strategy.global_manager, "start_scan"):
                        try:
                            await self.strategy.scan(
                                symbol_list=symbol_list, clientId=7
                            )
                        except KeyboardInterrupt:
                            pass  # Expected - this means we got past connectAsync

        # Verify IB connection was attempted
        self.mock_ib.connectAsync.assert_called_once_with("127.0.0.1", 7497, clientId=7)

    async def test_scan_method_calls_stock_scanner(self):
        """Test that scan method calls stock scanner for each symbol"""
        symbol_list = ["AAPL", "MSFT", "TSLA"]

        with patch.object(
            self.strategy, "box_stock_scanner", new_callable=AsyncMock
        ) as mock_scanner:
            # Interrupt after the symbols are scanned (after the gather call in main loop)
            # The asyncio.sleep(30) happens after the symbol scanning in the main loop
            with patch(
                "asyncio.sleep", side_effect=[KeyboardInterrupt()]
            ):  # Stop at main loop sleep
                with patch.object(self.strategy, "_perform_cache_maintenance"):
                    with patch.object(self.strategy.global_manager, "start_scan"):
                        try:
                            await self.strategy.scan(symbol_list=symbol_list)
                        except KeyboardInterrupt:
                            pass  # Expected - this means we completed one full scan iteration

        # Should have called scanner for each symbol at least once
        assert mock_scanner.call_count >= len(symbol_list)

    async def test_scan_method_handles_keyboard_interrupt(self):
        """Test that scan method handles KeyboardInterrupt gracefully"""
        symbol_list = ["AAPL"]

        with patch.object(self.strategy, "box_stock_scanner", new_callable=AsyncMock):
            with patch("asyncio.sleep", side_effect=KeyboardInterrupt):
                with patch(
                    "modules.Arbitrage.box_spread.strategy.logger"
                ) as mock_logger:

                    await self.strategy.scan(symbol_list=symbol_list)

                    # Should log interruption (check that it was called, not necessarily the last call)
                    mock_logger.info.assert_any_call(
                        "Box spread scan interrupted by user"
                    )

    async def test_scan_method_handles_general_exception(self):
        """Test that scan method handles general exceptions"""
        symbol_list = ["AAPL"]

        # Mock the _get_market_data_async method to raise an exception
        with patch.object(
            self.strategy, "_get_market_data_async", side_effect=Exception("Test error")
        ):
            with patch(
                "asyncio.sleep", side_effect=KeyboardInterrupt
            ):  # Stop after first iteration
                with patch(
                    "modules.Arbitrage.box_spread.strategy.logger"
                ) as mock_logger:

                    await self.strategy.scan(symbol_list=symbol_list)

                    # Should log error from box_stock_scanner exception handling
                    mock_logger.error.assert_called()

    async def test_scan_method_profit_target_multiplier(self):
        """Test that profit target increases with multiplier"""
        symbol_list = ["AAPL"]
        initial_profit = 0.01
        multiplier = 2.0

        with patch.object(self.strategy, "box_stock_scanner", new_callable=AsyncMock):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                with patch.object(self.strategy, "_perform_cache_maintenance"):

                    # Mock sleep to control iteration count
                    iteration_count = 0

                    def mock_sleep_side_effect(duration):
                        nonlocal iteration_count
                        iteration_count += 1
                        if iteration_count >= 2:  # Stop after 2 iterations
                            raise KeyboardInterrupt()
                        return AsyncMock()

                    mock_sleep.side_effect = mock_sleep_side_effect

                    try:
                        await self.strategy.scan(
                            symbol_list=symbol_list,
                            profit_target=initial_profit,
                            profit_target_multiplier=multiplier,
                        )
                    except KeyboardInterrupt:
                        pass

        # Profit target should have increased
        expected_profit = initial_profit * multiplier
        assert abs(self.strategy.profit_target - expected_profit) < 0.001


class TestBoxSpreadUtilityFunctions:
    """Test BoxSpread utility and helper functions"""

    def setup_method(self):
        """Set up test fixtures for each test method"""
        with patch("modules.Arbitrage.box_spread.strategy.get_logger"):
            self.strategy = BoxSpread()

    async def test_create_and_start_executor_creation(self):
        """Test executor creation and initialization"""
        # Mock contracts
        mock_contracts = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

        # Mock IB and order manager
        mock_ib = MagicMock()
        mock_order_manager = MagicMock()
        self.strategy.ib = mock_ib
        self.strategy.order_manager = mock_order_manager

        with patch(
            "modules.Arbitrage.box_spread.strategy.BoxExecutor"
        ) as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor_class.return_value = mock_executor

            await self.strategy._create_and_start_executor(
                "AAPL", 180.0, 185.0, "20250830", mock_contracts
            )

            # Verify executor was created
            mock_executor_class.assert_called_once()

            # Verify executor was added to active executors
            assert "AAPL" in self.strategy.active_executors
            assert self.strategy.active_executors["AAPL"] == mock_executor

    async def test_create_and_start_executor_error_handling(self):
        """Test error handling in executor creation"""
        mock_contracts = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

        with patch(
            "modules.Arbitrage.box_spread.strategy.BoxExecutor",
            side_effect=Exception("Executor error"),
        ):
            with patch("modules.Arbitrage.box_spread.strategy.logger") as mock_logger:

                await self.strategy._create_and_start_executor(
                    "ERROR", 180.0, 185.0, "20250830", mock_contracts
                )

                # Should log error
                mock_logger.error.assert_called_once()

    def test_get_market_data_async_integration(self):
        """Test integration with base class market data method"""
        # Test that the method exists and is callable
        assert hasattr(self.strategy, "_get_market_data_async")
        assert callable(getattr(self.strategy, "_get_market_data_async"))

    def test_get_chain_integration(self):
        """Test integration with base class chain retrieval"""
        # Test that the method exists and is callable
        assert hasattr(self.strategy, "_get_chain")
        assert callable(getattr(self.strategy, "_get_chain"))

    def test_get_chains_integration(self):
        """Test integration with base class chains retrieval"""
        # Test that the method exists and is callable
        assert hasattr(self.strategy, "_get_chains")
        assert callable(getattr(self.strategy, "_get_chains"))


@pytest.mark.asyncio
async def test_run_box_spread_strategy_convenience_function():
    """Test the convenience function for running box spread strategy"""
    from modules.Arbitrage.box_spread.strategy import run_box_spread_strategy

    symbol_list = ["AAPL", "MSFT"]

    with patch(
        "modules.Arbitrage.box_spread.strategy.BoxSpread"
    ) as mock_strategy_class:
        mock_strategy = AsyncMock()
        mock_strategy_class.return_value = mock_strategy

        await run_box_spread_strategy(
            symbol_list=symbol_list,
            range=0.12,
            profit_target=0.015,
            max_spread=30.0,
            client_id=4,
        )

        # Verify strategy was created and scan was called
        mock_strategy_class.assert_called_once()
        mock_strategy.scan.assert_called_once_with(
            symbol_list=symbol_list,
            range=0.12,
            profit_target=0.015,
            max_spread=30.0,
            clientId=4,
        )


@pytest.mark.integration
class TestBoxSpreadIntegrationWithMockIB:
    """Integration tests with mock IB data"""

    def setup_method(self):
        """Set up test fixtures for each test method"""
        with patch("modules.Arbitrage.box_spread.strategy.get_logger"):
            self.strategy = BoxSpread()

        # Use mock IB with realistic data
        self.mock_ib = MockIB()
        self.strategy.ib = self.mock_ib

    async def test_integration_full_scan_cycle(self):
        """Test full scan cycle with mock data"""
        symbol_list = ["DELL"]  # Use DELL which has predefined arbitrage scenario

        # First, manually establish the connection to ensure it's set up properly
        # This prevents the KeyboardInterrupt from interfering with the connection process
        await self.mock_ib.connectAsync()
        assert self.mock_ib.connected, "Mock IB should be connected after connectAsync"

        # Patch connectAsync to verify it's called during scan, and patch sleep to interrupt after connection
        with patch.object(
            self.mock_ib, "connectAsync", wraps=self.mock_ib.connectAsync
        ) as mock_connect:
            with patch("asyncio.sleep", side_effect=[KeyboardInterrupt()]):
                with patch.object(self.strategy, "_perform_cache_maintenance"):

                    try:
                        await self.strategy.scan(symbol_list=symbol_list)
                    except KeyboardInterrupt:
                        pass

        # Verify scan called connectAsync (indicating proper integration flow)
        mock_connect.assert_called_once()
        # Verify IB connection remains established throughout the test
        assert self.mock_ib.connected

    async def test_integration_with_realistic_market_data(self):
        """Test integration with realistic market data scenarios"""
        # This test verifies that the strategy can handle the predefined
        # market data scenarios in MockIB

        await self.strategy.box_stock_scanner("DELL")

        # Should complete without errors
        # The actual arbitrage detection happens in the executor
        # which is tested separately
