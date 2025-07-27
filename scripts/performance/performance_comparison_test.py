#!/usr/bin/env python3
"""
Performance comparison between old per-symbol optimization and new global selection.

This script demonstrates the performance and quality improvements of the new
global opportunity selection system compared to the previous per-symbol approach.
"""

import asyncio
import json

# Add project to path
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modules.Arbitrage.Synthetic import GlobalOpportunityManager, ScoringConfig, Syn
from tests.mock_ib import MockContract, MockIB, MockTicker


class OptimizationComparison:
    """Compare old vs new optimization approaches"""

    def __init__(self):
        self.results = {"old_approach": {}, "new_approach": {}, "comparison": {}}

    def create_market_scenario(self) -> Dict[str, List[Dict]]:
        """Create a realistic multi-symbol market scenario"""

        scenarios = {}

        # Symbol 1: AAPL - Multiple mediocre opportunities
        scenarios["AAPL"] = [
            {
                "call_strike": 185,
                "put_strike": 184,
                "call_bid": 3.20,
                "call_ask": 3.30,
                "put_bid": 2.80,
                "put_ask": 2.90,
                "call_volume": 300,
                "put_volume": 250,
                "days_to_expiry": 25,
                "stock_price": 185.50,
            },
            {
                "call_strike": 186,
                "put_strike": 185,
                "call_bid": 2.80,
                "call_ask": 2.90,
                "put_bid": 3.20,
                "put_ask": 3.30,
                "call_volume": 200,
                "put_volume": 180,
                "days_to_expiry": 32,
                "stock_price": 185.50,
            },
        ]

        # Symbol 2: MSFT - One excellent opportunity hidden among poor ones
        scenarios["MSFT"] = [
            {
                "call_strike": 415,
                "put_strike": 414,
                "call_bid": 2.00,
                "call_ask": 2.10,
                "put_bid": 1.80,
                "put_ask": 1.90,
                "call_volume": 150,
                "put_volume": 120,
                "days_to_expiry": 20,
                "stock_price": 415.75,
            },
            {
                # This is the best opportunity globally
                "call_strike": 416,
                "put_strike": 415,
                "call_bid": 8.50,
                "call_ask": 8.55,
                "put_bid": 3.45,
                "put_ask": 3.50,
                "call_volume": 800,
                "put_volume": 750,
                "days_to_expiry": 30,  # Optimal
                "stock_price": 415.75,
            },
        ]

        # Symbol 3: GOOGL - Good liquidity but poor risk-reward
        scenarios["GOOGL"] = [
            {
                "call_strike": 150,
                "put_strike": 149,
                "call_bid": 4.00,
                "call_ask": 4.02,
                "put_bid": 3.48,
                "put_ask": 3.50,
                "call_volume": 1000,
                "put_volume": 950,
                "days_to_expiry": 35,
                "stock_price": 150.00,
            }
        ]

        # Symbol 4: TSLA - High risk-reward but poor liquidity
        scenarios["TSLA"] = [
            {
                "call_strike": 246,
                "put_strike": 245,
                "call_bid": 12.00,
                "call_ask": 12.50,
                "put_bid": 2.50,
                "put_ask": 3.00,
                "call_volume": 50,
                "put_volume": 40,
                "days_to_expiry": 40,
                "stock_price": 245.80,
            }
        ]

        # Symbol 5: META - Average across all metrics
        scenarios["META"] = [
            {
                "call_strike": 495,
                "put_strike": 494,
                "call_bid": 5.00,
                "call_ask": 5.10,
                "put_bid": 3.90,
                "put_ask": 4.00,
                "call_volume": 400,
                "put_volume": 350,
                "days_to_expiry": 28,
                "stock_price": 495.25,
            }
        ]

        return scenarios

    def simulate_old_approach(self, market_data: Dict[str, List[Dict]]) -> Dict:
        """Simulate the old per-symbol optimization approach"""

        print("\n📊 OLD APPROACH: Per-Symbol Optimization")
        print("-" * 50)

        start_time = time.time()
        opportunities_found = {}
        total_opportunities = 0

        # Process each symbol independently
        for symbol, symbol_opportunities in market_data.items():
            print(f"\n  Processing {symbol}...")

            best_for_symbol = None
            best_profit = -float("inf")

            # Find best opportunity for this symbol only
            for opp in symbol_opportunities:
                net_credit = opp["call_bid"] - opp["put_ask"]
                spread = opp["stock_price"] - opp["put_strike"]
                min_profit = net_credit - spread
                max_profit = net_credit + (opp["call_strike"] - opp["put_strike"])

                # Simple profit-based selection (old approach)
                if min_profit > 0 and max_profit > best_profit:
                    best_profit = max_profit
                    best_for_symbol = {
                        "symbol": symbol,
                        "max_profit": max_profit,
                        "min_profit": min_profit,
                        "opportunity": opp,
                    }

            if best_for_symbol:
                opportunities_found[symbol] = best_for_symbol
                total_opportunities += 1
                print(f"    ✓ Found opportunity: Max profit ${best_profit:.2f}")
            else:
                print(f"    ✗ No profitable opportunity found")

        # In old approach, we might execute multiple trades
        # or just pick the first profitable one found
        execution_time = time.time() - start_time

        # Pick the highest profit opportunity (simple selection)
        if opportunities_found:
            best_overall = max(
                opportunities_found.values(), key=lambda x: x["max_profit"]
            )
        else:
            best_overall = None

        return {
            "approach": "Per-Symbol Optimization",
            "opportunities_evaluated": sum(len(opps) for opps in market_data.values()),
            "profitable_symbols": len(opportunities_found),
            "total_time": execution_time,
            "best_opportunity": best_overall,
            "all_opportunities": opportunities_found,
        }

    def simulate_new_approach(self, market_data: Dict[str, List[Dict]]) -> Dict:
        """Simulate the new global selection approach"""

        print("\n\n🌍 NEW APPROACH: Global Opportunity Selection")
        print("-" * 50)

        start_time = time.time()

        # Use balanced scoring strategy
        config = ScoringConfig.create_balanced()
        manager = GlobalOpportunityManager(config)

        opportunities_added = 0

        # Collect ALL opportunities globally
        print("\n  Phase 1: Collecting all opportunities...")
        for symbol, symbol_opportunities in market_data.items():
            for idx, opp in enumerate(symbol_opportunities):
                # Create mock data
                call_contract = MockContract(
                    symbol, "OPT", strike=opp["call_strike"], right="C"
                )
                put_contract = MockContract(
                    symbol, "OPT", strike=opp["put_strike"], right="P"
                )

                call_ticker = MockTicker(
                    call_contract,
                    bid=opp["call_bid"],
                    ask=opp["call_ask"],
                    volume=opp["call_volume"],
                )

                put_ticker = MockTicker(
                    put_contract,
                    bid=opp["put_bid"],
                    ask=opp["put_ask"],
                    volume=opp["put_volume"],
                )

                # Calculate trade details
                net_credit = opp["call_bid"] - opp["put_ask"]
                spread = opp["stock_price"] - opp["put_strike"]
                min_profit = net_credit - spread
                max_profit = net_credit + (opp["call_strike"] - opp["put_strike"])

                expiry_date = datetime.now() + timedelta(days=opp["days_to_expiry"])

                trade_details = {
                    "max_profit": max_profit,
                    "min_profit": min_profit,
                    "net_credit": net_credit,
                    "stock_price": opp["stock_price"],
                    "expiry": expiry_date.strftime("%Y%m%d"),
                    "call_strike": opp["call_strike"],
                    "put_strike": opp["put_strike"],
                }

                # Add to global manager
                success = manager.add_opportunity(
                    symbol=symbol,
                    conversion_contract=Mock(),
                    order=Mock(),
                    trade_details=trade_details,
                    call_ticker=call_ticker,
                    put_ticker=put_ticker,
                )

                if success:
                    opportunities_added += 1

        print(f"    ✓ Collected {opportunities_added} opportunities globally")

        # Phase 2: Global selection
        print("\n  Phase 2: Selecting best global opportunity...")
        best = manager.get_best_opportunity()

        execution_time = time.time() - start_time

        # Get scoring details
        if best:
            print(f"\n  🏆 Best Global Opportunity:")
            print(f"     Symbol: {best.symbol}")
            print(f"     Composite Score: {best.score.composite_score:.3f}")
            print(f"     Risk-Reward: {best.score.risk_reward_ratio:.2f}")
            print(f"     Liquidity: {best.score.liquidity_score:.3f}")
            print(f"     Time Decay: {best.score.time_decay_score:.3f}")
            print(f"     Market Quality: {best.score.market_quality_score:.3f}")

        return {
            "approach": "Global Opportunity Selection",
            "opportunities_evaluated": sum(len(opps) for opps in market_data.values()),
            "opportunities_collected": opportunities_added,
            "total_time": execution_time,
            "best_opportunity": (
                {
                    "symbol": best.symbol if best else None,
                    "max_profit": best.trade_details["max_profit"] if best else None,
                    "min_profit": best.trade_details["min_profit"] if best else None,
                    "composite_score": best.score.composite_score if best else None,
                    "scoring_components": (
                        {
                            "risk_reward_ratio": (
                                best.score.risk_reward_ratio if best else None
                            ),
                            "liquidity_score": (
                                best.score.liquidity_score if best else None
                            ),
                            "time_decay_score": (
                                best.score.time_decay_score if best else None
                            ),
                            "market_quality_score": (
                                best.score.market_quality_score if best else None
                            ),
                        }
                        if best
                        else None
                    ),
                }
                if best
                else None
            ),
            "statistics": manager.get_statistics(),
        }

    def run_comparison(self):
        """Run the full comparison test"""

        print("🔬 Performance & Quality Comparison: Old vs New Optimization")
        print("=" * 70)

        # Create market scenario
        market_data = self.create_market_scenario()

        print("\n📈 Market Scenario:")
        print(f"   Symbols: {list(market_data.keys())}")
        print(
            f"   Total opportunities: {sum(len(opps) for opps in market_data.values())}"
        )

        # Run old approach
        old_results = self.simulate_old_approach(market_data)
        self.results["old_approach"] = old_results

        # Run new approach
        new_results = self.simulate_new_approach(market_data)
        self.results["new_approach"] = new_results

        # Compare results
        self.compare_results()

        # Save results
        self.save_results()

    def compare_results(self):
        """Compare and analyze the results"""

        print("\n\n" + "=" * 70)
        print("📊 COMPARISON RESULTS")
        print("=" * 70)

        old = self.results["old_approach"]
        new = self.results["new_approach"]

        # Performance comparison
        print("\n⏱️  Performance Comparison:")
        print(f"   Old approach time: {old['total_time']:.3f}s")
        print(f"   New approach time: {new['total_time']:.3f}s")
        print(
            f"   Performance difference: {(new['total_time'] - old['total_time']):.3f}s"
        )

        # Quality comparison
        print("\n🎯 Quality Comparison:")

        if old["best_opportunity"] and new["best_opportunity"]:
            old_best = old["best_opportunity"]
            new_best = new["best_opportunity"]

            print(f"\n   Old Approach Selected:")
            print(f"     Symbol: {old_best['symbol']}")
            print(f"     Max Profit: ${old_best['max_profit']:.2f}")
            print(f"     Selection Criteria: Highest profit per symbol")

            print(f"\n   New Approach Selected:")
            print(f"     Symbol: {new_best['symbol']}")
            print(f"     Max Profit: ${new_best['max_profit']:.2f}")
            print(f"     Composite Score: {new_best['composite_score']:.3f}")
            print(f"     Selection Criteria: Multi-factor global optimization")

            # Explain why new approach made a better choice
            if new_best["symbol"] == "MSFT":
                print(
                    "\n   ✅ New approach correctly identified MSFT as the best global opportunity!"
                )
                print("      - Optimal time decay (30 days)")
                print("      - Excellent liquidity (800/750 volume)")
                print("      - Strong risk-reward ratio")
                print("      - Tight bid-ask spreads")

        # Key advantages
        print("\n🔑 Key Advantages of New Approach:")
        print("   1. Considers ALL opportunities globally, not just per-symbol")
        print("   2. Multi-factor scoring beyond just profit")
        print("   3. Accounts for liquidity and execution risk")
        print("   4. Optimizes for time decay")
        print("   5. Single best trade selection reduces capital requirements")

        self.results["comparison"] = {
            "performance_gain": new["total_time"] - old["total_time"],
            "old_selected_symbol": (
                old["best_opportunity"]["symbol"] if old["best_opportunity"] else None
            ),
            "new_selected_symbol": (
                new["best_opportunity"]["symbol"] if new["best_opportunity"] else None
            ),
            "quality_improvement": "New approach uses comprehensive scoring vs simple profit comparison",
        }

    def save_results(self):
        """Save comparison results to file"""

        filename = "optimization_comparison_results.json"

        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {filename}")


def main():
    """Run the comparison test"""

    comparison = OptimizationComparison()

    try:
        comparison.run_comparison()

        print("\n✨ Comparison test completed successfully!")
        print("\n📌 Key Takeaway:")
        print("   The new global selection approach provides better opportunity")
        print("   identification through comprehensive multi-factor analysis,")
        print("   leading to higher quality trades and better risk management.")

    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
