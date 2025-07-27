#!/usr/bin/env python3
"""
Comprehensive performance testing for Global Opportunity Selection system.

This script tests the performance characteristics of the new global opportunity
selection logic under various realistic scenarios and loads.
"""

import asyncio
import gc
import json

# Add project to path
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Dict, List, Tuple

import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modules.Arbitrage.Synthetic import (
    GlobalOpportunityManager,
    OpportunityScore,
    ScoringConfig,
)
from tests.mock_ib import MockContract, MockTicker


@dataclass
class PerformanceResult:
    """Container for performance test results"""

    test_name: str
    opportunities_count: int
    symbols_count: int
    collection_time: float
    selection_time: float
    memory_used_mb: float
    scoring_strategy: str
    avg_score_time: float
    statistics_time: float

    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "opportunities_count": self.opportunities_count,
            "symbols_count": self.symbols_count,
            "collection_time_ms": self.collection_time * 1000,
            "selection_time_ms": self.selection_time * 1000,
            "memory_used_mb": self.memory_used_mb,
            "scoring_strategy": self.scoring_strategy,
            "avg_score_time_ms": self.avg_score_time * 1000,
            "statistics_time_ms": self.statistics_time * 1000,
            "ops_per_second": (
                self.opportunities_count / self.collection_time
                if self.collection_time > 0
                else 0
            ),
        }


class GlobalSelectionPerformanceTester:
    """Performance testing suite for global opportunity selection"""

    def __init__(self):
        self.results: List[PerformanceResult] = []

    def create_realistic_opportunity(
        self, symbol: str, opportunity_idx: int, base_stock_price: float = 100.0
    ) -> Dict:
        """Create realistic opportunity data for testing"""

        # Vary parameters realistically
        strike_offset = (opportunity_idx % 5) - 2  # -2 to +2 from ATM
        days_to_expiry = 20 + (opportunity_idx % 40)  # 20-60 days

        call_strike = base_stock_price + strike_offset
        put_strike = base_stock_price + strike_offset - 1

        # Realistic volume patterns
        volume_base = 500 - (abs(strike_offset) * 100)  # Higher volume near ATM
        call_volume = max(50, volume_base + (opportunity_idx % 200))
        put_volume = max(40, int(call_volume * 0.8))

        # Realistic spreads
        spread_multiplier = 1 + (
            abs(strike_offset) * 0.05
        )  # Wider spreads away from ATM
        call_spread = 0.05 * spread_multiplier
        put_spread = 0.04 * spread_multiplier

        # Create contracts and tickers
        call_contract = MockContract(symbol, "OPT", strike=call_strike, right="C")
        put_contract = MockContract(symbol, "OPT", strike=put_strike, right="P")

        call_ticker = MockTicker(
            call_contract,
            bid=5.0 - (strike_offset * 0.5),
            ask=5.0 - (strike_offset * 0.5) + call_spread,
            volume=call_volume,
        )

        put_ticker = MockTicker(
            put_contract,
            bid=3.0 + (strike_offset * 0.3),
            ask=3.0 + (strike_offset * 0.3) + put_spread,
            volume=put_volume,
        )

        # Calculate realistic P&L
        net_credit = call_ticker.bid - put_ticker.ask
        spread = base_stock_price - put_strike
        min_profit = net_credit - spread
        max_profit = net_credit + (call_strike - put_strike)

        expiry_date = datetime.now() + timedelta(days=days_to_expiry)

        return {
            "symbol": symbol,
            "conversion_contract": MockContract(symbol, "BAG"),
            "order": {"totalQuantity": 1},
            "trade_details": {
                "max_profit": max_profit,
                "min_profit": min_profit,
                "net_credit": net_credit,
                "stock_price": base_stock_price,
                "expiry": expiry_date.strftime("%Y%m%d"),
                "call_strike": call_strike,
                "put_strike": put_strike,
            },
            "call_ticker": call_ticker,
            "put_ticker": put_ticker,
        }

    def measure_memory_usage(self) -> float:
        """Measure current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def test_large_scale_performance(
        self,
        num_symbols: int = 50,
        opportunities_per_symbol: int = 10,
        scoring_strategy: str = "balanced",
    ) -> PerformanceResult:
        """Test performance with large number of opportunities"""

        print(
            f"\n🔍 Testing {num_symbols} symbols × {opportunities_per_symbol} opportunities "
            f"= {num_symbols * opportunities_per_symbol} total opportunities"
        )

        # Configure scoring strategy
        if scoring_strategy == "conservative":
            config = ScoringConfig.create_conservative()
        elif scoring_strategy == "aggressive":
            config = ScoringConfig.create_aggressive()
        elif scoring_strategy == "liquidity-focused":
            config = ScoringConfig.create_liquidity_focused()
        else:
            config = ScoringConfig.create_balanced()

        manager = GlobalOpportunityManager(config)

        # Start memory tracking
        gc.collect()
        memory_before = self.measure_memory_usage()

        # Phase 1: Collection
        print("  📊 Phase 1: Collecting opportunities...")
        collection_start = time.time()
        opportunities_added = 0

        # Popular tech stocks for realistic simulation
        base_symbols = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "TSLA",
            "NVDA",
            "AMD",
            "INTC",
            "NFLX",
            "ORCL",
            "CRM",
            "ADBE",
            "PYPL",
            "SQ",
        ]

        for symbol_idx in range(num_symbols):
            # Use real symbols when available, then synthetic ones
            if symbol_idx < len(base_symbols):
                symbol = base_symbols[symbol_idx]
                base_price = 100 + (symbol_idx * 10)  # Vary stock prices
            else:
                symbol = f"SYM{symbol_idx:03d}"
                base_price = 50 + (symbol_idx * 5)

            for opp_idx in range(opportunities_per_symbol):
                opportunity_data = self.create_realistic_opportunity(
                    symbol, opp_idx, base_price
                )

                if manager.add_opportunity(**opportunity_data):
                    opportunities_added += 1

        collection_time = time.time() - collection_start

        # Phase 2: Selection
        print("  🎯 Phase 2: Selecting best opportunity...")
        selection_times = []

        # Run selection multiple times for average
        for _ in range(10):
            selection_start = time.time()
            best = manager.get_best_opportunity()
            selection_time = time.time() - selection_start
            selection_times.append(selection_time)

        avg_selection_time = mean(selection_times)

        # Phase 3: Score calculation timing
        print("  📈 Phase 3: Measuring scoring performance...")
        score_times = []

        # Sample some opportunities for scoring time
        with manager.lock:
            sample_opportunities = manager.opportunities[
                : min(50, len(manager.opportunities))
            ]

        for opp in sample_opportunities:
            score_start = time.time()
            # Re-calculate score
            _ = manager.calculate_opportunity_score(
                opp.trade_details,
                opp.call_volume,
                opp.put_volume,
                opp.call_bid_ask_spread,
                opp.put_bid_ask_spread,
                opp.days_to_expiry,
            )
            score_time = time.time() - score_start
            score_times.append(score_time)

        avg_score_time = mean(score_times) if score_times else 0

        # Phase 4: Statistics
        print("  📊 Phase 4: Generating statistics...")
        stats_start = time.time()
        stats = manager.get_statistics()
        stats_time = time.time() - stats_start

        # Memory usage
        memory_after = self.measure_memory_usage()
        memory_used = memory_after - memory_before

        # Results
        result = PerformanceResult(
            test_name=f"large_scale_{num_symbols}x{opportunities_per_symbol}",
            opportunities_count=opportunities_added,
            symbols_count=stats.get("unique_symbols", 0),
            collection_time=collection_time,
            selection_time=avg_selection_time,
            memory_used_mb=memory_used,
            scoring_strategy=scoring_strategy,
            avg_score_time=avg_score_time,
            statistics_time=stats_time,
        )

        # Print summary
        print(f"\n  ✅ Performance Summary:")
        print(f"     Opportunities collected: {opportunities_added}")
        print(
            f"     Collection time: {collection_time:.3f}s ({opportunities_added/collection_time:.1f} ops/sec)"
        )
        print(f"     Selection time: {avg_selection_time*1000:.2f}ms (avg of 10 runs)")
        print(f"     Score calculation: {avg_score_time*1000:.3f}ms per opportunity")
        print(f"     Statistics generation: {stats_time*1000:.2f}ms")
        print(f"     Memory used: {memory_used:.2f} MB")

        if best:
            print(
                f"     Best opportunity: {best.symbol} (score: {best.score.composite_score:.3f})"
            )

        return result

    def test_concurrent_access_performance(
        self, num_threads: int = 10
    ) -> PerformanceResult:
        """Test performance under concurrent access"""
        import threading

        print(f"\n🔄 Testing concurrent access with {num_threads} threads...")

        manager = GlobalOpportunityManager()
        results = []
        errors = []

        def add_opportunities_worker(thread_id: int):
            """Worker thread to add opportunities"""
            try:
                for i in range(20):  # Each thread adds 20 opportunities
                    symbol = f"T{thread_id}S{i}"
                    data = self.create_realistic_opportunity(symbol, i)
                    manager.add_opportunity(**data)
                    # Intersperse with selections
                    if i % 5 == 0:
                        best = manager.get_best_opportunity()
                        if best:
                            results.append((thread_id, best.symbol))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Start timing
        start_time = time.time()

        # Launch threads
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=add_opportunities_worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        total_time = time.time() - start_time

        print(f"  ✅ Concurrent test completed:")
        print(f"     Total time: {total_time:.3f}s")
        print(f"     Opportunities: {manager.get_opportunity_count()}")
        print(f"     Selections made: {len(results)}")
        print(f"     Errors: {len(errors)}")

        return PerformanceResult(
            test_name=f"concurrent_access_{num_threads}_threads",
            opportunities_count=manager.get_opportunity_count(),
            symbols_count=manager.get_statistics().get("unique_symbols", 0),
            collection_time=total_time,
            selection_time=0,  # Not applicable for this test
            memory_used_mb=0,
            scoring_strategy="balanced",
            avg_score_time=0,
            statistics_time=0,
        )

    def test_scoring_strategy_performance(self) -> List[PerformanceResult]:
        """Compare performance across different scoring strategies"""

        print("\n📊 Comparing scoring strategy performance...")

        strategies = ["conservative", "aggressive", "balanced", "liquidity-focused"]
        strategy_results = []

        for strategy in strategies:
            print(f"\n  Testing {strategy} strategy...")
            result = self.test_large_scale_performance(
                num_symbols=30, opportunities_per_symbol=5, scoring_strategy=strategy
            )
            strategy_results.append(result)

        # Compare results
        print("\n📈 Strategy Performance Comparison:")
        print(
            f"{'Strategy':<20} {'Collection (ms)':<15} {'Selection (ms)':<15} {'Memory (MB)':<12}"
        )
        print("-" * 65)

        for result in strategy_results:
            print(
                f"{result.scoring_strategy:<20} "
                f"{result.collection_time*1000:<15.1f} "
                f"{result.selection_time*1000:<15.2f} "
                f"{result.memory_used_mb:<12.2f}"
            )

        return strategy_results

    def run_comprehensive_tests(self):
        """Run all performance tests"""

        print("🚀 Starting Comprehensive Global Selection Performance Tests")
        print("=" * 70)

        all_results = []

        # Test 1: Varying scale
        print("\n1️⃣ SCALABILITY TESTS")
        scales = [
            (10, 5),  # 50 opportunities
            (20, 10),  # 200 opportunities
            (50, 10),  # 500 opportunities
            (100, 10),  # 1000 opportunities
        ]

        for num_symbols, opps_per_symbol in scales:
            result = self.test_large_scale_performance(num_symbols, opps_per_symbol)
            all_results.append(result)
            self.results.append(result)

        # Test 2: Concurrent access
        print("\n2️⃣ CONCURRENCY TESTS")
        concurrent_result = self.test_concurrent_access_performance(num_threads=20)
        all_results.append(concurrent_result)
        self.results.append(concurrent_result)

        # Test 3: Scoring strategies
        print("\n3️⃣ SCORING STRATEGY COMPARISON")
        strategy_results = self.test_scoring_strategy_performance()
        all_results.extend(strategy_results)
        self.results.extend(strategy_results)

        # Memory profiling for largest test
        print("\n4️⃣ MEMORY PROFILING (1000 opportunities)")
        tracemalloc.start()

        large_result = self.test_large_scale_performance(100, 10)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\n  💾 Memory Profile:")
        print(f"     Current memory: {current / 1024 / 1024:.2f} MB")
        print(f"     Peak memory: {peak / 1024 / 1024:.2f} MB")

        # Generate report
        self.generate_report()

    def generate_report(self):
        """Generate performance report"""

        print("\n" + "=" * 70)
        print("📊 PERFORMANCE TEST REPORT")
        print("=" * 70)

        # Save results to JSON
        report_data = {
            "test_date": datetime.now().isoformat(),
            "results": [r.to_dict() for r in self.results],
            "summary": self.generate_summary(),
        }

        with open("global_selection_performance_report.json", "w") as f:
            json.dump(report_data, f, indent=2)

        print("\n✅ Report saved to: global_selection_performance_report.json")

        # Print summary
        print("\n📈 PERFORMANCE SUMMARY:")
        for key, value in report_data["summary"].items():
            print(f"   {key}: {value}")

    def generate_summary(self) -> Dict:
        """Generate performance summary statistics"""

        if not self.results:
            return {}

        collection_times = [
            r.collection_time for r in self.results if r.collection_time > 0
        ]
        selection_times = [
            r.selection_time for r in self.results if r.selection_time > 0
        ]
        memory_usage = [r.memory_used_mb for r in self.results if r.memory_used_mb > 0]

        return {
            "total_tests": len(self.results),
            "avg_collection_time_ms": (
                mean(collection_times) * 1000 if collection_times else 0
            ),
            "avg_selection_time_ms": (
                mean(selection_times) * 1000 if selection_times else 0
            ),
            "avg_memory_usage_mb": mean(memory_usage) if memory_usage else 0,
            "max_opportunities_tested": max(
                [r.opportunities_count for r in self.results]
            ),
            "performance_grade": self.calculate_performance_grade(selection_times),
        }

    def calculate_performance_grade(self, selection_times: List[float]) -> str:
        """Calculate overall performance grade"""

        if not selection_times:
            return "N/A"

        avg_selection_ms = mean(selection_times) * 1000

        if avg_selection_ms < 10:
            return "A+ (Excellent)"
        elif avg_selection_ms < 25:
            return "A (Very Good)"
        elif avg_selection_ms < 50:
            return "B (Good)"
        elif avg_selection_ms < 100:
            return "C (Acceptable)"
        else:
            return "D (Needs Optimization)"


if __name__ == "__main__":
    tester = GlobalSelectionPerformanceTester()

    try:
        tester.run_comprehensive_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()

    print("\n✨ Performance testing complete!")
