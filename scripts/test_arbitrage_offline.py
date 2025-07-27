#!/usr/bin/env python3
"""
Standalone arbitrage testing script for market-closed scenarios.
This script allows comprehensive testing of the arbitrage detection system
when markets are closed using realistic mock data.

Usage:
    python test_arbitrage_offline.py
    python test_arbitrage_offline.py --scenario dell_profitable
    python test_arbitrage_offline.py --debug --scenario spy_multiple_expiries
"""

import asyncio
import sys
import time
from pathlib import Path

import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from tests.market_scenarios import ArbitrageTestCases, MarketScenarios
    from tests.mock_ib import MockIB
    from tests.test_arbitrage_integration import TestStandaloneArbitrage
except ImportError:
    # When running as standalone script
    import sys

    sys.path.append("tests")
    from market_scenarios import ArbitrageTestCases, MarketScenarios
    from mock_ib import MockIB
    from test_arbitrage_integration import TestStandaloneArbitrage


class OfflineArbitrageTester:
    """Main class for offline arbitrage testing"""

    def __init__(self, debug: bool = False):
        self.debug = debug

    async def test_scenario(self, scenario_name: str):
        """Test a specific market scenario"""
        print(f"🔍 Testing scenario: {scenario_name}")
        print("=" * 60)

        try:
            # Get scenario data
            if scenario_name in [
                "dell_profitable",
                "dell_no_arbitrage",
                "dell_wide_spreads",
            ]:
                market_data = MarketScenarios.get_scenario(scenario_name)
            else:
                market_data = MarketScenarios.get_scenario(scenario_name)

            # Analyze the scenario
            await self._analyze_scenario(scenario_name, market_data)

        except ValueError as e:
            print(f"❌ Error: {e}")
            self._list_available_scenarios()

    async def _analyze_scenario(self, scenario_name: str, market_data: dict):
        """Analyze a market scenario for arbitrage opportunities"""

        # Find stock data (right attribute is None for stocks)
        stock_ticker = next(
            (
                t
                for t in market_data.values()
                if hasattr(t.contract, "right") and t.contract.right is None
            ),
            None,
        )
        if not stock_ticker:
            print("❌ No stock data found in scenario")
            return

        print(f"📈 Stock: {stock_ticker.contract.symbol} @ ${stock_ticker.last:.2f}")
        print(
            f"   Bid: ${stock_ticker.bid:.2f}, Ask: ${stock_ticker.ask:.2f}, Volume: {stock_ticker.volume:,}"
        )
        print()

        # Find option data (right attribute is not None for options)
        option_tickers = [
            t
            for t in market_data.values()
            if hasattr(t.contract, "right") and t.contract.right is not None
        ]

        if not option_tickers:
            print("❌ No option data found in scenario")
            return

        # Group by expiry
        expiries = {}
        for ticker in option_tickers:
            expiry = ticker.contract.lastTradeDateOrContractMonth
            if expiry not in expiries:
                expiries[expiry] = {"calls": [], "puts": []}

            if ticker.contract.right == "C":
                expiries[expiry]["calls"].append(ticker)
            else:
                expiries[expiry]["puts"].append(ticker)

        # Analyze each expiry
        total_opportunities = 0
        for expiry, options in expiries.items():
            opportunities = await self._analyze_expiry(stock_ticker, expiry, options)
            total_opportunities += opportunities

        print(f"\n🎯 Total arbitrage opportunities found: {total_opportunities}")

        if total_opportunities == 0:
            print("💡 Tips for finding arbitrage:")
            print("   - Look for net credit > spread")
            print("   - Try 1-strike difference combinations")
            print("   - Check bid-ask spreads aren't too wide")

    async def _analyze_expiry(self, stock_ticker, expiry: str, options: dict) -> int:
        """Analyze a specific expiry for arbitrage opportunities"""
        print(f"📅 Expiry: {expiry}")

        calls = sorted(options["calls"], key=lambda x: x.contract.strike)
        puts = sorted(options["puts"], key=lambda x: x.contract.strike)

        if not calls or not puts:
            print("   ❌ Missing calls or puts for this expiry")
            return 0

        # Test strike combinations using the new adaptive logic
        stock_price = stock_ticker.last
        opportunities_found = 0

        # Simulate the adaptive strike position logic
        all_strikes = sorted(list(set([c.contract.strike for c in calls + puts])))
        stock_position = self._find_stock_position(stock_price, all_strikes)

        # Call candidates: stock position ± 1
        call_start = max(0, stock_position - 1)
        call_end = min(len(all_strikes), stock_position + 2)
        call_strikes = all_strikes[call_start:call_end]

        # Put candidates: at/below stock position
        put_strikes = all_strikes[: stock_position + 2]

        print(
            f"   📊 Testing {len(call_strikes)} call strikes × {len(put_strikes)} put strikes"
        )

        # Priority combinations (1-strike difference)
        priority_combinations = []
        secondary_combinations = []

        for call_strike in call_strikes:
            for put_strike in put_strikes:
                if call_strike > put_strike:
                    call_idx = all_strikes.index(call_strike)
                    put_idx = all_strikes.index(put_strike)
                    strike_diff = call_idx - put_idx

                    combination = (call_strike, put_strike, strike_diff)

                    if strike_diff == 1:
                        priority_combinations.append(combination)
                    elif strike_diff <= 3:
                        secondary_combinations.append(combination)

        # Test combinations
        all_combinations = priority_combinations + secondary_combinations

        for call_strike, put_strike, strike_diff in all_combinations[
            :4
        ]:  # Limit to top 4
            opportunity = await self._test_combination(
                stock_ticker, calls, puts, call_strike, put_strike, strike_diff
            )
            if opportunity:
                opportunities_found += 1

        return opportunities_found

    def _find_stock_position(self, stock_price: float, strikes: list) -> int:
        """Find stock position within strikes (same logic as SFR.py)"""
        for i, strike in enumerate(strikes):
            if strike >= stock_price:
                return max(0, i - 1) if strike > stock_price and i > 0 else i
        return len(strikes) - 1

    async def _test_combination(
        self, stock_ticker, calls, puts, call_strike, put_strike, strike_diff
    ):
        """Test a specific call/put combination for arbitrage"""

        # Find matching tickers
        call_ticker = next((c for c in calls if c.contract.strike == call_strike), None)
        put_ticker = next((p for p in puts if p.contract.strike == put_strike), None)

        if not call_ticker or not put_ticker:
            return False

        # Check bid-ask spreads
        call_spread = (
            call_ticker.ask - call_ticker.bid
            if call_ticker.bid > 0 and call_ticker.ask > 0
            else float("inf")
        )
        put_spread = (
            put_ticker.ask - put_ticker.bid
            if put_ticker.bid > 0 and put_ticker.ask > 0
            else float("inf")
        )

        if call_spread > 20 or put_spread > 20:
            print(
                f"   🚫 C{call_strike}/P{put_strike}: Spread too wide (C:{call_spread:.2f}, P:{put_spread:.2f})"
            )
            return False

        # Calculate arbitrage metrics
        stock_price = stock_ticker.last
        call_price = call_ticker.bid  # Selling call
        put_price = put_ticker.ask  # Buying put

        net_credit = call_price - put_price
        spread = stock_price - put_strike
        min_profit = net_credit - spread
        max_profit = (call_strike - put_strike) + net_credit

        # Check arbitrage condition
        is_arbitrage = spread < net_credit and min_profit > 0 and net_credit > 0

        priority_indicator = "🔥" if strike_diff == 1 else "💡"
        status = "✅ ARBITRAGE" if is_arbitrage else "❌ No arbitrage"

        print(
            f"   {priority_indicator} C{call_strike}/P{put_strike} (diff:{strike_diff}): {status}"
        )

        if self.debug or is_arbitrage:
            print(f"      Call: ${call_price:.2f} (bid), Put: ${put_price:.2f} (ask)")
            print(f"      Net Credit: ${net_credit:.2f}, Spread: ${spread:.2f}")
            print(f"      Min Profit: ${min_profit:.2f}, Max Profit: ${max_profit:.2f}")
            if is_arbitrage:
                roi = (min_profit / (stock_price + net_credit)) * 100
                print(f"      Min ROI: {roi:.2f}%")

        return is_arbitrage

    def _list_available_scenarios(self):
        """List all available test scenarios"""
        print("\n📋 Available scenarios:")
        scenarios = [
            ("dell_profitable", "DELL profitable conversion arbitrage"),
            ("dell_no_arbitrage", "DELL normal market conditions"),
            ("dell_wide_spreads", "DELL with wide bid-ask spreads"),
            ("dell_low_volume", "DELL with low volume options"),
            ("spy_multiple_expiries", "SPY with multiple expiries"),
        ]

        for name, description in scenarios:
            print(f"   • {name}: {description}")

    async def run_all_scenarios(self):
        """Run all available test scenarios"""
        scenarios = [
            "dell_profitable",
            "dell_no_arbitrage",
            "dell_wide_spreads",
            "dell_low_volume",
        ]

        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{'='*60}")
            print(f"Test {i}/{len(scenarios)}: {scenario}")
            print("=" * 60)
            await self.test_scenario(scenario)

            if i < len(scenarios):
                print("\nPress Enter to continue to next test...")
                input()

    def run_manual_analysis(self):
        """Run the manual step-by-step analysis"""
        print("🔬 Running manual DELL arbitrage analysis...")
        print("=" * 60)

        test_standalone = TestStandaloneArbitrage()
        test_standalone.test_manual_dell_arbitrage_analysis()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Offline arbitrage testing tool")
    parser.add_argument("--scenario", type=str, help="Specific scenario to test")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--manual", action="store_true", help="Run manual analysis")

    args = parser.parse_args()

    tester = OfflineArbitrageTester(debug=args.debug)

    print("🚀 Offline Arbitrage Tester")
    print("Testing arbitrage detection when markets are closed")
    print("=" * 60)

    try:
        if args.manual:
            tester.run_manual_analysis()
        elif args.all:
            await tester.run_all_scenarios()
        elif args.scenario:
            await tester.test_scenario(args.scenario)
        else:
            # Interactive mode
            print("Select testing mode:")
            print("1. Test specific scenario")
            print("2. Run all scenarios")
            print("3. Manual analysis")
            print("4. List scenarios")

            choice = input("\nEnter choice (1-4): ").strip()

            if choice == "1":
                tester._list_available_scenarios()
                scenario = input("\nEnter scenario name: ").strip()
                await tester.test_scenario(scenario)
            elif choice == "2":
                await tester.run_all_scenarios()
            elif choice == "3":
                tester.run_manual_analysis()
            elif choice == "4":
                tester._list_available_scenarios()
            else:
                print("Invalid choice")

    except KeyboardInterrupt:
        print("\n\n👋 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
