#!/usr/bin/env python3
"""
Simple performance test for SynExecutor and GlobalOpportunityManager.

Tests the core functionality with mock data to verify performance characteristics.
"""

import asyncio

# Add project to path
import os
import sys
import time
from datetime import datetime, timedelta

import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import MagicMock

from modules.Arbitrage.Synthetic import (
    ExpiryOption,
    GlobalOpportunityManager,
    ScoringConfig,
    SynExecutor,
    contract_ticker,
)
from tests.mock_ib import MockContract, MockIB, MockTicker

# Enable logging to see rejection reasons
logging.basicConfig(level=logging.INFO)


def create_test_scenario(symbol: str = "TEST", num_expiries: int = 3):
    """Create test data for a symbol with multiple expiries"""

    # Stock contract and ticker
    stock_contract = MockContract(symbol, "STK")
    stock_contract.conId = 1000
    stock_ticker = MockTicker(stock_contract, bid=100.0, ask=100.0, volume=1000000)

    # Store in global ticker dict
    contract_ticker[stock_contract.conId] = stock_ticker

    # Create expiry options
    expiry_options = []
    base_date = datetime.now()

    for i in range(num_expiries):
        expiry_date = base_date + timedelta(days=30 + i * 30)
        expiry_str = expiry_date.strftime("%Y%m%d")

        # Create multiple strike combinations for each expiry
        for j in range(3):  # 3 different strike combinations
            call_strike = 101 + j  # Call strike higher
            put_strike = 100 + j  # Put strike at or above stock price for synthetic

            # Create contracts
            call_contract = MockContract(
                symbol,
                "OPT",
                strike=call_strike,
                right="C",
                lastTradeDateOrContractMonth=expiry_str,
            )
            call_contract.conId = 2000 + i * 10 + j * 2

            put_contract = MockContract(
                symbol,
                "OPT",
                strike=put_strike,
                right="P",
                lastTradeDateOrContractMonth=expiry_str,
            )
            put_contract.conId = 2001 + i * 10 + j * 2

            # Create tickers with opportunity
            # For synthetic arbitrage: Sell Call + Buy Put
            # Net Credit = Call Bid - Put Ask (must be positive)
            # For a synthetic position to work:
            # - Call strike > Put strike (we have 101 > 100)
            # - Net credit should create arbitrage opportunity
            call_ticker = MockTicker(
                call_contract,
                bid=3.5 - j * 0.1,  # Lower call prices for OTM calls
                ask=3.55 - j * 0.1,
                volume=1000 - j * 100,
            )

            put_ticker = MockTicker(
                put_contract,
                bid=2.45 + j * 0.1,  # Higher put prices for ATM/ITM puts
                ask=2.50 + j * 0.1,  # This creates net credit of ~1.0
                volume=800 - j * 50,
            )

            # Store in global ticker dict
            contract_ticker[call_contract.conId] = call_ticker
            contract_ticker[put_contract.conId] = put_ticker

        # Create expiry option (just one per expiry for simplicity)
        expiry_option = ExpiryOption(
            expiry=expiry_str,
            call_contract=call_contract,
            put_contract=put_contract,
            call_strike=call_strike,
            put_strike=put_strike,
        )
        expiry_options.append(expiry_option)

    return stock_contract, expiry_options


async def test_syn_executor_performance():
    """Test SynExecutor performance with multiple expiries"""

    print("\n🚀 Testing SynExecutor Performance")
    print("=" * 60)

    # Create mock components
    mock_ib = MockIB()
    order_manager = MagicMock()
    global_manager = GlobalOpportunityManager()

    # Test configuration
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    total_opportunities = 0
    total_time = 0

    for symbol in symbols:
        print(f"\n📊 Processing {symbol}...")

        # Create test scenario
        stock_contract, expiry_options = create_test_scenario(symbol, num_expiries=3)

        # Create executor
        executor = SynExecutor(
            ib=mock_ib,
            order_manager=order_manager,
            stock_contract=stock_contract,
            expiry_options=expiry_options,
            symbol=symbol,
            cost_limit=120,
            max_loss_threshold=50,
            max_profit_threshold=100,
            profit_ratio_threshold=2.0,
            start_time=time.time(),
            global_manager=global_manager,
            quantity=1,
        )

        # Measure processing time
        start_time = time.time()

        # Simulate executor processing
        # The executor would normally be triggered by market data events
        # Here we directly call the opportunity calculation
        opportunities_found = 0

        for expiry_option in expiry_options:
            opportunity = executor.calc_price_and_build_order_for_expiry(expiry_option)
            if opportunity:
                opportunities_found += 1

                # Report to global manager
                conversion_contract, order, _, trade_details = opportunity
                call_ticker = contract_ticker.get(expiry_option.call_contract.conId)
                put_ticker = contract_ticker.get(expiry_option.put_contract.conId)

                print(f"    Found opportunity: {trade_details}")

                if call_ticker and put_ticker:
                    global_manager.add_opportunity(
                        symbol=symbol,
                        conversion_contract=conversion_contract,
                        order=order,
                        trade_details=trade_details,
                        call_ticker=call_ticker,
                        put_ticker=put_ticker,
                    )

        elapsed = time.time() - start_time
        total_time += elapsed
        total_opportunities += opportunities_found

        print(f"  ✅ Found {opportunities_found} opportunities in {elapsed*1000:.2f}ms")

    # Get best opportunity
    selection_start = time.time()
    best_opportunity = global_manager.get_best_opportunity()
    selection_time = time.time() - selection_start

    # Summary
    print(f"\n📊 Performance Summary:")
    print(f"  Total symbols processed: {len(symbols)}")
    print(f"  Total opportunities found: {total_opportunities}")
    print(f"  Total processing time: {total_time*1000:.2f}ms")
    print(f"  Average time per symbol: {(total_time/len(symbols))*1000:.2f}ms")
    print(f"  Global selection time: {selection_time*1000:.3f}ms")

    if best_opportunity:
        print(
            f"  Best opportunity: {best_opportunity.symbol} - Score: {best_opportunity.score.composite_score:.3f}"
        )

    # Test different scoring strategies
    print("\n📊 Testing Scoring Strategies:")
    strategies = [
        ("Conservative", ScoringConfig.create_conservative()),
        ("Aggressive", ScoringConfig.create_aggressive()),
        ("Balanced", ScoringConfig.create_balanced()),
        ("Liquidity-Focused", ScoringConfig.create_liquidity_focused()),
    ]

    for name, config in strategies:
        manager = GlobalOpportunityManager(config)

        # Add some test opportunities
        for i in range(10):
            symbol = f"TEST{i}"
            stock_contract, expiry_options = create_test_scenario(
                symbol, num_expiries=1
            )

            executor = SynExecutor(
                ib=mock_ib,
                order_manager=order_manager,
                stock_contract=stock_contract,
                expiry_options=expiry_options,
                symbol=symbol,
                cost_limit=120,
                max_loss_threshold=50,
                max_profit_threshold=100,
                profit_ratio_threshold=2.0,
                start_time=time.time(),
                global_manager=manager,
                quantity=1,
            )

            # Process first expiry
            opportunity = executor.calc_price_and_build_order_for_expiry(
                expiry_options[0]
            )
            if opportunity:
                conversion_contract, order, _, trade_details = opportunity
                call_ticker = contract_ticker.get(expiry_options[0].call_contract.conId)
                put_ticker = contract_ticker.get(expiry_options[0].put_contract.conId)

                if call_ticker and put_ticker:
                    manager.add_opportunity(
                        symbol=symbol,
                        conversion_contract=conversion_contract,
                        order=order,
                        trade_details=trade_details,
                        call_ticker=call_ticker,
                        put_ticker=put_ticker,
                    )

        # Time selection
        start_time = time.time()
        best = manager.get_best_opportunity()
        elapsed = time.time() - start_time

        print(
            f"  {name}: {elapsed*1000:.3f}ms - Selected: {best.symbol if best else 'None'}"
        )


async def test_global_manager_stress():
    """Stress test GlobalOpportunityManager with many opportunities"""

    print("\n💪 Stress Testing GlobalOpportunityManager")
    print("=" * 60)

    manager = GlobalOpportunityManager()
    mock_ib = MockIB()
    order_manager = MagicMock()

    # Add many opportunities
    num_opportunities = 1000
    print(f"  Adding {num_opportunities} opportunities...")

    start_time = time.time()

    for i in range(num_opportunities):
        symbol = f"SYM{i:04d}"
        stock_contract, expiry_options = create_test_scenario(symbol, num_expiries=1)

        # Create a simple opportunity
        expiry_option = expiry_options[0]

        # Create mock trade details
        trade_details = {
            "max_profit": 100 + (i % 50),
            "min_profit": -50 + (i % 30),
            "net_credit": 2.0 + (i % 10) * 0.1,
            "stock_price": 100,
            "expiry": expiry_option.expiry,
            "call_strike": expiry_option.call_strike,
            "put_strike": expiry_option.put_strike,
            "min_roi": 5.0 + (i % 20) * 0.5,
        }

        # Create mock tickers
        call_ticker = MockTicker(
            expiry_option.call_contract, bid=5.0, ask=5.05, volume=1000 + i % 500
        )
        put_ticker = MockTicker(
            expiry_option.put_contract, bid=3.0, ask=3.04, volume=800 + i % 400
        )

        # Add opportunity
        manager.add_opportunity(
            symbol=symbol,
            conversion_contract=MagicMock(),
            order=MagicMock(),
            trade_details=trade_details,
            call_ticker=call_ticker,
            put_ticker=put_ticker,
        )

    add_time = time.time() - start_time
    print(f"  ✅ Added {num_opportunities} opportunities in {add_time:.3f}s")
    print(f"  📊 Rate: {num_opportunities/add_time:.1f} opportunities/second")

    # Test selection performance
    print("\n  Testing selection performance...")
    selection_times = []

    for _ in range(100):
        start = time.time()
        best = manager.get_best_opportunity()
        elapsed = time.time() - start
        selection_times.append(elapsed)

    avg_selection = sum(selection_times) / len(selection_times)
    max_selection = max(selection_times)

    print(f"  ✅ Selection performance (100 iterations):")
    print(f"     Average: {avg_selection*1000:.3f}ms")
    print(f"     Max: {max_selection*1000:.3f}ms")

    if best:
        print(
            f"  📊 Best opportunity: {best.symbol} - Score: {best.score.composite_score:.3f}"
        )


async def main():
    """Run all performance tests"""

    print("🚀 SynExecutor and GlobalOpportunityManager Performance Tests")
    print("=" * 70)

    # Test 1: SynExecutor performance
    await test_syn_executor_performance()

    # Test 2: Global manager stress test
    await test_global_manager_stress()

    print("\n✨ Performance testing complete!")


if __name__ == "__main__":
    asyncio.run(main())
