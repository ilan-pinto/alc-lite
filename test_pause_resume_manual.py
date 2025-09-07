#!/usr/bin/env python3
"""
Manual test script for pause/resume functionality
Run this to manually verify the pause/resume logic works correctly
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.Arbitrage.Strategy import ArbitrageClass


async def test_pause_resume_logic():
    """Manual test of pause/resume functionality"""

    print("🧪 Testing Pause/Resume Logic per ADR-003")
    print("=" * 50)

    # Create strategy instance
    strategy = ArbitrageClass()
    print(f"✅ Strategy created")

    # Test 1: Initial state
    print(f"\n📋 Test 1: Initial State")
    print(f"   _executor_paused: {strategy._executor_paused}")
    print(f"   active_parallel_symbol: {strategy.active_parallel_symbol}")
    print(f"   order_filled: {strategy.order_filled}")

    symbols = ["SPY", "QQQ", "AAPL", "TSLA"]
    pause_states = {symbol: strategy.is_paused(symbol) for symbol in symbols}
    print(f"   Pause states: {pause_states}")
    assert all(
        not paused for paused in pause_states.values()
    ), "All should be unpaused initially"
    print("   ✅ All symbols unpaused initially")

    # Test 2: Pause all other executors
    print(f"\n📋 Test 2: Pause All Other Executors")
    await strategy.pause_all_other_executors("SPY")
    print(f"   _executor_paused: {strategy._executor_paused}")
    print(f"   active_parallel_symbol: {strategy.active_parallel_symbol}")

    pause_states = {symbol: strategy.is_paused(symbol) for symbol in symbols}
    print(f"   Pause states: {pause_states}")
    assert not strategy.is_paused("SPY"), "SPY should not be paused (executing)"
    assert strategy.is_paused("QQQ"), "QQQ should be paused"
    assert strategy.is_paused("AAPL"), "AAPL should be paused"
    assert strategy.is_paused("TSLA"), "TSLA should be paused"
    print("   ✅ Only SPY continues, others paused")

    # Test 3: Resume all executors
    print(f"\n📋 Test 3: Resume All Executors")
    await strategy.resume_all_executors()
    print(f"   _executor_paused: {strategy._executor_paused}")
    print(f"   active_parallel_symbol: {strategy.active_parallel_symbol}")

    pause_states = {symbol: strategy.is_paused(symbol) for symbol in symbols}
    print(f"   Pause states: {pause_states}")
    assert all(
        not paused for paused in pause_states.values()
    ), "All should be unpaused after resume"
    print("   ✅ All symbols resumed successfully")

    # Test 4: Stop all executors
    print(f"\n📋 Test 4: Stop All Executors")
    await strategy.stop_all_executors()
    print(f"   _executor_paused: {strategy._executor_paused}")
    print(f"   active_parallel_symbol: {strategy.active_parallel_symbol}")
    print(f"   order_filled: {strategy.order_filled}")

    pause_states = {symbol: strategy.is_paused(symbol) for symbol in symbols}
    print(f"   Pause states: {pause_states}")
    assert all(
        paused for paused in pause_states.values()
    ), "All should be paused after stop"
    assert strategy.order_filled, "order_filled should be True to exit scan loops"
    print("   ✅ All symbols stopped, order_filled=True")

    # Test 5: Pause/Resume cycle
    print(f"\n📋 Test 5: Multiple Pause/Resume Cycles")

    # Reset state
    strategy._executor_paused = False
    strategy.active_parallel_symbol = None
    strategy.order_filled = False

    test_symbols = ["QQQ", "AAPL", "TSLA"]
    for executing_symbol in test_symbols:
        print(f"   🔄 Testing execution by {executing_symbol}")

        # Pause for this symbol
        await strategy.pause_all_other_executors(executing_symbol)

        # Check that only this symbol is not paused
        for symbol in symbols:
            is_paused = strategy.is_paused(symbol)
            expected_paused = symbol != executing_symbol
            assert is_paused == expected_paused, f"{symbol} pause state incorrect"

        print(f"      ✅ {executing_symbol} executing, others paused")

        # Resume
        await strategy.resume_all_executors()

        # Check all resumed
        for symbol in symbols:
            assert not strategy.is_paused(symbol), f"{symbol} should be resumed"

        print(f"      ✅ All symbols resumed")

    print(f"\n🎉 All tests passed! Pause/resume logic working correctly.")

    # Test 6: Scan loop integration simulation
    print(f"\n📋 Test 6: Scan Loop Integration Simulation")

    # Simulate what the scan loop would do
    symbols_to_scan = ["SPY", "QQQ", "AAPL", "TSLA"]

    # Normal state - all should scan
    print("   Normal state (all should scan):")
    scan_count = 0
    for symbol in symbols_to_scan:
        if not strategy.is_paused(symbol):
            scan_count += 1
            print(f"     📊 Scanning {symbol}")
        else:
            print(f"     ⏸️  Skipping {symbol} (paused)")
    assert scan_count == 4, "All 4 symbols should scan normally"

    # Paused state - only SPY should scan
    await strategy.pause_all_other_executors("SPY")
    print("   Paused state (only SPY should scan):")
    scan_count = 0
    for symbol in symbols_to_scan:
        if not strategy.is_paused(symbol):
            scan_count += 1
            print(f"     📊 Scanning {symbol}")
        else:
            print(f"     ⏸️  Skipping {symbol} (paused)")
    assert scan_count == 1, "Only 1 symbol (SPY) should scan when paused"

    # Stopped state - none should scan
    await strategy.stop_all_executors()
    print("   Stopped state (none should scan):")
    print(
        f"     Debug: _executor_paused={strategy._executor_paused}, active_parallel_symbol={strategy.active_parallel_symbol}"
    )
    scan_count = 0
    for symbol in symbols_to_scan:
        is_paused = strategy.is_paused(symbol)
        print(f"     Debug: {symbol} is_paused={is_paused}")
        if not is_paused:
            scan_count += 1
            print(f"     📊 Scanning {symbol}")
        else:
            print(f"     ⏸️  Skipping {symbol} (paused)")
    print(f"     Debug: scan_count={scan_count}")
    assert (
        scan_count == 0
    ), f"No symbols should scan when stopped, but {scan_count} are scanning"

    print("   ✅ Scan loop integration working correctly")

    print(f"\n🚀 All pause/resume tests completed successfully!")
    print("   The implementation correctly follows ADR-003 specification.")


if __name__ == "__main__":
    asyncio.run(test_pause_resume_logic())
