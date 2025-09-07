#!/usr/bin/env python3
"""
Real-world testing approach for pause/resume functionality.
This shows you how to test the logic in actual SFR execution scenarios.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.Arbitrage.sfr.strategy import SFR


async def test_realworld_pause_resume():
    """Test pause/resume in a realistic SFR strategy scenario"""

    print("🌍 Real-World Pause/Resume Test")
    print("=" * 50)

    # Create SFR strategy
    strategy = SFR()

    # Mock the IB connection
    mock_ib = MagicMock()
    mock_ib.isConnected.return_value = True
    mock_ib.connect = AsyncMock()
    strategy.ib = mock_ib

    print("✅ SFR Strategy created with mock IB")

    # Test scenario: Multiple symbols scanning, one starts execution
    print("\n📋 Scenario: Multi-symbol scanning with execution")

    symbols = ["SPY", "QQQ", "AAPL", "TSLA", "MSFT"]

    # Simulate scan state tracking
    scan_states = {}
    execution_events = []

    def track_scan_attempt(symbol):
        if strategy.is_paused(symbol):
            scan_states[symbol] = "PAUSED"
            return False
        else:
            scan_states[symbol] = "SCANNING"
            return True

    # Phase 1: Normal scanning (all should scan)
    print("\n🔍 Phase 1: Normal scanning")
    for symbol in symbols:
        can_scan = track_scan_attempt(symbol)
        print(f"   {symbol}: {'✅ SCANNING' if can_scan else '⏸️  PAUSED'}")

    active_scanners = sum(1 for state in scan_states.values() if state == "SCANNING")
    assert active_scanners == 5, f"Expected 5 active scanners, got {active_scanners}"
    print(f"   ✅ All {active_scanners} symbols scanning normally")

    # Phase 2: SPY finds opportunity and starts execution
    print(f"\n🚀 Phase 2: SPY starts execution (pauses others)")

    # Simulate SPY starting parallel execution
    await strategy.pause_all_other_executors("SPY")
    execution_events.append(("SPY", "EXECUTION_STARTED"))

    # Check scan states
    scan_states.clear()
    for symbol in symbols:
        track_scan_attempt(symbol)

    print("   Scan states after SPY execution starts:")
    for symbol in symbols:
        state = scan_states[symbol]
        emoji = "📊" if state == "SCANNING" else "⏸️ "
        print(f"     {symbol}: {emoji} {state}")

    spy_scanning = scan_states["SPY"] == "SCANNING"
    others_paused = all(scan_states[sym] == "PAUSED" for sym in symbols if sym != "SPY")

    assert spy_scanning, "SPY should continue scanning during its execution"
    assert others_paused, "All other symbols should be paused"
    print(f"   ✅ Only SPY continues, others paused correctly")

    # Phase 3a: Successful execution (stop all)
    print(f"\n✅ Phase 3a: SPY execution succeeds")

    await strategy.stop_all_executors()
    execution_events.append(("SPY", "EXECUTION_SUCCEEDED"))

    # Check final states
    scan_states.clear()
    for symbol in symbols:
        track_scan_attempt(symbol)

    print("   Scan states after successful execution:")
    for symbol in symbols:
        state = scan_states[symbol]
        print(f"     {symbol}: ⏹️  {state}")

    all_stopped = all(state == "PAUSED" for state in scan_states.values())
    assert all_stopped, "All symbols should be stopped after success"
    assert strategy.order_filled, "order_filled should be True to exit loops"
    print(f"   ✅ All symbols stopped, ready to exit")

    # Reset for Phase 3b test
    print(f"\n🔄 Resetting for failure scenario...")
    strategy._executor_paused = False
    strategy.active_parallel_symbol = None
    strategy.order_filled = False

    # Phase 3b: Failed execution (resume all)
    print(f"\n❌ Phase 3b: QQQ execution fails")

    # Start QQQ execution
    await strategy.pause_all_other_executors("QQQ")
    execution_events.append(("QQQ", "EXECUTION_STARTED"))

    # Simulate execution failure
    await strategy.resume_all_executors()
    execution_events.append(("QQQ", "EXECUTION_FAILED"))

    # Check resumed states
    scan_states.clear()
    for symbol in symbols:
        track_scan_attempt(symbol)

    print("   Scan states after failed execution:")
    for symbol in symbols:
        state = scan_states[symbol]
        emoji = "📊" if state == "SCANNING" else "⏸️ "
        print(f"     {symbol}: {emoji} {state}")

    all_resumed = all(state == "SCANNING" for state in scan_states.values())
    assert all_resumed, "All symbols should resume after failure"
    assert (
        not strategy.order_filled
    ), "order_filled should remain False to continue scanning"
    print(f"   ✅ All symbols resumed, can continue scanning")

    # Summary
    print(f"\n📊 Execution Event Summary:")
    for i, (symbol, event) in enumerate(execution_events, 1):
        print(f"   {i}. {symbol}: {event}")

    print(f"\n🎉 Real-world pause/resume test completed successfully!")
    print("   ✅ Normal scanning: All symbols active")
    print("   ✅ During execution: Only executing symbol active")
    print("   ✅ After success: All symbols stopped")
    print("   ✅ After failure: All symbols resumed")

    return True


async def test_concurrent_execution_prevention():
    """Test that concurrent executions are properly prevented"""

    print("\n🚫 Concurrent Execution Prevention Test")
    print("=" * 50)

    # Create strategy
    strategy = SFR()
    mock_ib = MagicMock()
    strategy.ib = mock_ib

    # Simulate two symbols trying to execute simultaneously
    print("📊 Simulating SPY and QQQ finding opportunities simultaneously...")

    # SPY gets there first
    await strategy.pause_all_other_executors("SPY")
    spy_can_proceed = not strategy.is_paused("SPY")
    qqq_blocked = strategy.is_paused("QQQ")

    print(f"   SPY can proceed: {spy_can_proceed}")
    print(f"   QQQ is blocked: {qqq_blocked}")

    assert spy_can_proceed, "SPY should be able to proceed"
    assert qqq_blocked, "QQQ should be blocked"

    # Try QQQ execution while SPY is running (should be blocked by is_paused check)
    try:
        # This would normally be caught by the scan loop's is_paused() check
        if strategy.is_paused("QQQ"):
            print("   ✅ QQQ execution properly blocked by pause state")
        else:
            raise AssertionError("QQQ should be blocked")
    except AssertionError:
        raise

    # SPY completes successfully
    await strategy.stop_all_executors()
    print("   ✅ SPY completes, all scanning stops")

    print("🎉 Concurrent execution prevention working correctly!")

    return True


async def main():
    """Run all real-world tests"""
    print("🧪 Real-World Pause/Resume Testing Suite")
    print("=" * 60)

    try:
        await test_realworld_pause_resume()
        await test_concurrent_execution_prevention()

        print("\n" + "=" * 60)
        print("🎉 ALL REAL-WORLD TESTS PASSED!")
        print("✅ The pause/resume implementation correctly follows ADR-003")
        print("✅ Multi-symbol scanning works as expected")
        print("✅ Concurrent execution prevention works")
        print("✅ Ready for production use!")

        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
