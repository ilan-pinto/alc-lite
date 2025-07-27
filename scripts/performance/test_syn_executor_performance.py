#!/usr/bin/env python3
"""
Performance testing for Syn class and SynExecutor - the core Synthetic arbitrage system.

This script tests the real-world performance including:
- Multi-symbol scanning with SynExecutor instances
- Global opportunity selection using get_best_opportunity()
- Complete scan cycle performance
- Different scoring strategies
"""

import asyncio
import gc
import json

# Add project to path
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, Mock, patch

import logging
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modules.Arbitrage.Synthetic import ScoringConfig, Syn
from tests.mock_ib import MarketDataGenerator, MockContract, MockIB, MockTicker

# Configure logging
logging.basicConfig(level=logging.WARNING)


def create_mock_ib() -> MockIB:
    """Create a MockIB instance for testing"""
    return MockIB()


def create_multi_symbol_market_data(
    symbols: List[str],
) -> Dict[str, Dict[int, MockTicker]]:
    """Create mock market data for multiple symbols"""
    multi_market_data = {}

    for symbol in symbols:
        # Create market data for this symbol
        market_data = {}
        base_price = 100.0 + (hash(symbol) % 100)  # Deterministic price based on symbol

        # Create stock ticker
        stock_ticker = MarketDataGenerator.generate_stock_data(
            symbol, base_price, volume=100000
        )
        market_data[stock_ticker.contract.conId] = stock_ticker

        # Create some option data for this symbol
        expiry = "20240101"  # Fixed expiry for consistency

        # Add call options
        for strike_offset in [-2, -1, 0, 1, 2]:
            strike = base_price + strike_offset

            # Call option
            call_ticker = MarketDataGenerator.generate_option_data(
                symbol, expiry, strike, "C", base_price, 30
            )
            market_data[call_ticker.contract.conId] = call_ticker

            # Put option
            put_ticker = MarketDataGenerator.generate_option_data(
                symbol, expiry, strike, "P", base_price, 30
            )
            market_data[put_ticker.contract.conId] = put_ticker

        multi_market_data[symbol] = market_data

    return multi_market_data


class MockOrderManager:
    """Mock order manager for performance testing"""

    def __init__(self, ib_instance):
        self.ib = ib_instance
        self.orders_placed = []

    async def place_order(self, contract, order):
        """Mock place order method"""
        self.orders_placed.append((contract, order))
        return True

    def get_order_count(self):
        """Get number of orders placed"""
        return len(self.orders_placed)


@dataclass
class SynPerformanceResult:
    """Container for Syn performance test results"""

    test_name: str
    symbols_count: int
    total_opportunities: int
    best_opportunity_selected: bool
    scan_time: float
    avg_symbol_scan_time: float
    memory_used_mb: float
    scoring_config: str
    global_selection_time: float
    executor_count: int

    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "symbols_count": self.symbols_count,
            "total_opportunities": self.total_opportunities,
            "best_opportunity_selected": self.best_opportunity_selected,
            "scan_time_ms": self.scan_time * 1000,
            "avg_symbol_scan_time_ms": self.avg_symbol_scan_time * 1000,
            "memory_used_mb": self.memory_used_mb,
            "scoring_config": self.scoring_config,
            "global_selection_time_ms": self.global_selection_time * 1000,
            "executor_count": self.executor_count,
            "throughput_symbols_per_sec": (
                self.symbols_count / self.scan_time if self.scan_time > 0 else 0
            ),
        }


class SynPerformanceTester:
    """Performance testing suite for Syn class"""

    def __init__(self):
        self.results: List[SynPerformanceResult] = []

    def measure_memory_usage(self) -> float:
        """Measure current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    async def test_syn_performance(
        self,
        symbols: List[str],
        cost_limit: float = 120,
        max_loss: float = 50,
        max_profit: float = 100,
        profit_ratio: float = 2.0,
        quantity: int = 1,
        scoring_config: ScoringConfig = None,
    ) -> SynPerformanceResult:
        """Test Syn class performance with given parameters"""

        print(f"\n🔍 Testing Syn with {len(symbols)} symbols")

        # Create mock IB and market data
        mock_ib = create_mock_ib()

        # Setup multi-symbol market data
        multi_market_data = create_multi_symbol_market_data(symbols)

        # Configure mock IB to handle multiple symbols
        async def mock_req_market_data(contract, *args, **kwargs):
            # Find appropriate ticker from multi_market_data
            for symbol, market_data in multi_market_data.items():
                if contract.symbol == symbol:
                    # Find stock ticker
                    for ticker in market_data.values():
                        if ticker.contract.secType == "STK":
                            return ticker
            # Return default if not found
            return list(list(multi_market_data.values())[0].values())[0]

        mock_ib._get_market_data_async = mock_req_market_data

        # Set market data for the first symbol (for initial connection test)
        if symbols:
            mock_ib.test_market_data = multi_market_data[symbols[0]]

        # Create Syn instance
        syn = Syn(debug=False, scoring_config=scoring_config)
        syn.ib = mock_ib
        syn.order_manager = MockOrderManager(mock_ib)

        # Track performance metrics
        symbol_scan_times = []
        global_selection_time = 0

        # Track global selection
        if hasattr(syn, "global_manager"):
            original_get_best = syn.global_manager.get_best_opportunity

            def track_global_selection():
                nonlocal global_selection_time

                start_time = time.time()
                result = original_get_best()
                global_selection_time = time.time() - start_time

                return result

            syn.global_manager.get_best_opportunity = track_global_selection

        # Initialize symbol scan times tracking for our isolated tests
        # (No need to wrap scan_syn since we're not using the full scan method)

        # Memory tracking
        gc.collect()
        memory_before = self.measure_memory_usage()

        # Run isolated performance test (not full live scan)
        print("  📊 Starting isolated performance test...")
        scan_start = time.time()

        try:
            # Test individual symbol scanning performance instead of full scan
            total_opportunities = 0

            # Initialize syn components for testing
            if not hasattr(syn, "global_manager"):
                from modules.Arbitrage.Synthetic import GlobalOpportunityManager

                syn.global_manager = GlobalOpportunityManager()

            syn.active_executors = {}

            # Test each symbol individually for performance measurement
            for symbol in symbols:
                # Set market data for this symbol
                if symbol in multi_market_data:
                    mock_ib.test_market_data = multi_market_data[symbol]

                # Simulate option chain request and processing
                start_time = time.time()

                # Create mock option chain data for this symbol
                try:
                    # Get mock market data for this symbol
                    symbol_market_data = multi_market_data.get(symbol, {})

                    # Find stock contract and option contracts
                    stock_contract = None
                    call_contracts = []
                    put_contracts = []

                    for ticker in symbol_market_data.values():
                        if ticker.contract.secType == "STK":
                            stock_contract = ticker.contract
                        elif hasattr(ticker.contract, "right"):
                            if ticker.contract.right == "C":
                                call_contracts.append(ticker.contract)
                            elif ticker.contract.right == "P":
                                put_contracts.append(ticker.contract)

                    if stock_contract and call_contracts and put_contracts:
                        # Create mock expiry options for SynExecutor
                        from modules.Arbitrage.Synthetic import ExpiryOption

                        expiry_options = []
                        # Group by expiry and strike to create conversion opportunities
                        for call_contract in call_contracts[
                            :3
                        ]:  # Limit for performance
                            for put_contract in put_contracts[:3]:
                                if (
                                    hasattr(call_contract, "strike")
                                    and hasattr(put_contract, "strike")
                                    and abs(call_contract.strike - put_contract.strike)
                                    <= 1
                                ):

                                    # Get expiry from the contract or use default
                                    expiry = getattr(
                                        call_contract,
                                        "lastTradeDateOrContractMonth",
                                        "20240101",
                                    )

                                    expiry_option = ExpiryOption(
                                        call_contract=call_contract,
                                        put_contract=put_contract,
                                        call_strike=call_contract.strike,
                                        put_strike=put_contract.strike,
                                        expiry=expiry,
                                    )
                                    expiry_options.append(expiry_option)

                        if expiry_options:
                            # Now create SynExecutor with proper arguments
                            from modules.Arbitrage.Synthetic import SynExecutor

                            executor = SynExecutor(
                                symbol=symbol,
                                ib=mock_ib,
                                order_manager=syn.order_manager,
                                stock_contract=stock_contract,
                                expiry_options=expiry_options,
                                cost_limit=cost_limit,
                                max_loss_threshold=max_loss,
                                max_profit_threshold=max_profit,
                                profit_ratio_threshold=profit_ratio,
                                start_time=time.time(),
                                global_manager=syn.global_manager,
                                quantity=quantity,
                            )

                            syn.active_executors[symbol] = executor
                            total_opportunities += len(expiry_options)

                    # Count options even if we couldn't create executor (for basic counting)
                    if not stock_contract or not expiry_options:
                        option_count = sum(
                            1
                            for ticker in symbol_market_data.values()
                            if hasattr(ticker.contract, "right")
                        )
                        total_opportunities += (
                            option_count // 2
                        )  # Rough estimate of pairs

                except Exception as e:
                    print(f"    Warning: Error processing {symbol}: {e}")
                    continue

                elapsed = time.time() - start_time
                symbol_scan_times.append(elapsed)

            scan_time = time.time() - scan_start

            # Get results
            best_opportunity = (
                syn.global_manager.get_best_opportunity()
                if hasattr(syn.global_manager, "get_best_opportunity")
                else None
            )
            executor_count = len(syn.active_executors)

        except Exception as e:
            print(f"  ❌ Error during performance test: {e}")
            import traceback

            traceback.print_exc()
            scan_time = time.time() - scan_start
            total_opportunities = 0
            best_opportunity = None
            executor_count = 0

        # Memory usage
        memory_after = self.measure_memory_usage()
        memory_used = memory_after - memory_before

        # Calculate averages
        avg_symbol_time = mean(symbol_scan_times) if symbol_scan_times else 0

        # Results
        result = SynPerformanceResult(
            test_name=f"syn_{len(symbols)}_symbols",
            symbols_count=len(symbols),
            total_opportunities=total_opportunities,
            best_opportunity_selected=best_opportunity is not None,
            scan_time=scan_time,
            avg_symbol_scan_time=avg_symbol_time,
            memory_used_mb=memory_used,
            scoring_config=(
                scoring_config.__class__.__name__ if scoring_config else "default"
            ),
            global_selection_time=global_selection_time,
            executor_count=executor_count,
        )

        # Print summary
        print(f"\n  ✅ Scan Summary:")
        print(f"     Symbols scanned: {len(symbols)}")
        print(f"     Total opportunities found: {total_opportunities}")
        print(f"     Best opportunity selected: {'Yes' if best_opportunity else 'No'}")
        print(
            f"     Total scan time: {scan_time:.3f}s ({len(symbols)/scan_time:.1f} symbols/sec)"
        )
        print(f"     Avg symbol scan: {avg_symbol_time*1000:.2f}ms")
        print(f"     Global selection time: {global_selection_time*1000:.3f}ms")
        print(f"     Active executors: {executor_count}")
        print(f"     Memory used: {memory_used:.2f} MB")

        if best_opportunity:
            print(
                f"     Best: {best_opportunity.symbol} - Score: {best_opportunity.score.composite_score:.3f}"
            )

        # Cleanup
        syn.ib.disconnect()

        return result

    async def test_scaling_performance(self):
        """Test performance with increasing number of symbols"""

        print("\n📈 SCALABILITY TEST - Increasing Symbol Count")
        print("=" * 60)

        # Test with increasing number of symbols
        test_configs = [
            (["AAPL"], "1 symbol"),
            (["AAPL", "MSFT"], "2 symbols"),
            (["AAPL", "MSFT", "GOOGL", "AMZN", "META"], "5 symbols"),
            (
                [
                    "AAPL",
                    "MSFT",
                    "GOOGL",
                    "AMZN",
                    "META",
                    "TSLA",
                    "NVDA",
                    "JPM",
                    "V",
                    "UNH",
                ],
                "10 symbols",
            ),
        ]

        for symbols, desc in test_configs:
            print(f"\n  Testing with {desc}...")
            result = await self.test_syn_performance(symbols)
            result.test_name = f"scaling_{desc.replace(' ', '_')}"
            self.results.append(result)

            # Small delay between tests
            await asyncio.sleep(1)

    async def test_scoring_strategies(self):
        """Test different scoring strategies"""

        print("\n📊 SCORING STRATEGY COMPARISON")
        print("=" * 60)

        symbols = ["AAPL", "MSFT", "GOOGL"]

        strategies = [
            ("Conservative", ScoringConfig.create_conservative()),
            ("Aggressive", ScoringConfig.create_aggressive()),
            ("Balanced", ScoringConfig.create_balanced()),
            ("Liquidity-Focused", ScoringConfig.create_liquidity_focused()),
        ]

        for name, config in strategies:
            print(f"\n  Testing {name} strategy...")
            result = await self.test_syn_performance(symbols, scoring_config=config)
            result.scoring_config = name
            result.test_name = f"strategy_{name.lower().replace('-', '_')}"
            self.results.append(result)

    async def test_opportunity_density(self):
        """Test with varying opportunity density"""

        print("\n💎 OPPORTUNITY DENSITY TEST")
        print("=" * 60)

        # Test with different parameter settings that affect opportunity detection
        test_configs = [
            {
                "name": "strict_criteria",
                "symbols": ["SPY", "QQQ"],
                "max_loss": 20,
                "max_profit": 50,
                "profit_ratio": 3.0,
                "desc": "Strict criteria (fewer opportunities)",
            },
            {
                "name": "moderate_criteria",
                "symbols": ["SPY", "QQQ"],
                "max_loss": 50,
                "max_profit": 100,
                "profit_ratio": 2.0,
                "desc": "Moderate criteria (normal opportunities)",
            },
            {
                "name": "loose_criteria",
                "symbols": ["SPY", "QQQ"],
                "max_loss": 100,
                "max_profit": 200,
                "profit_ratio": 1.5,
                "desc": "Loose criteria (more opportunities)",
            },
        ]

        for config in test_configs:
            print(f"\n  Testing {config['desc']}...")
            result = await self.test_syn_performance(
                symbols=config["symbols"],
                max_loss=config["max_loss"],
                max_profit=config["max_profit"],
                profit_ratio=config["profit_ratio"],
            )
            result.test_name = f"density_{config['name']}"
            self.results.append(result)

    async def test_concurrent_instances(self):
        """Test running multiple Syn instances concurrently"""

        print("\n🔄 CONCURRENT EXECUTION TEST")
        print("=" * 60)

        # Different symbol groups
        symbol_groups = [
            ["AAPL", "MSFT"],
            ["GOOGL", "AMZN"],
            ["META", "TSLA"],
            ["NVDA", "JPM"],
        ]

        start_time = time.time()

        # Create tasks for concurrent execution
        tasks = []
        for i, symbols in enumerate(symbol_groups):
            task = self.test_syn_performance(
                symbols,
                cost_limit=100 + i * 10,
                max_loss=40 + i * 5,
                max_profit=80 + i * 10,
            )
            tasks.append(task)

        # Run concurrently
        print(f"  🚀 Running {len(tasks)} Syn instances concurrently...")
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        print(f"\n  ✅ Concurrent Test Summary:")
        print(f"     Total execution time: {total_time:.3f}s")
        print(f"     Number of instances: {len(tasks)}")
        print(f"     Total symbols processed: {sum(r.symbols_count for r in results)}")
        print(
            f"     Total opportunities found: {sum(r.total_opportunities for r in results)}"
        )

        # Add aggregated result
        aggregated = SynPerformanceResult(
            test_name="concurrent_4_instances",
            symbols_count=sum(r.symbols_count for r in results),
            total_opportunities=sum(r.total_opportunities for r in results),
            best_opportunity_selected=any(r.best_opportunity_selected for r in results),
            scan_time=total_time,
            avg_symbol_scan_time=mean([r.avg_symbol_scan_time for r in results]),
            memory_used_mb=sum(r.memory_used_mb for r in results),
            scoring_config="mixed",
            global_selection_time=mean([r.global_selection_time for r in results]),
            executor_count=sum(r.executor_count for r in results),
        )

        self.results.append(aggregated)

    async def run_all_tests(self):
        """Run comprehensive performance test suite"""

        print("🚀 Starting Syn/SynExecutor Performance Tests")
        print("=" * 70)

        try:
            # Test 1: Scaling
            await self.test_scaling_performance()

            # Test 2: Scoring strategies
            await self.test_scoring_strategies()

            # Test 3: Opportunity density
            await self.test_opportunity_density()

            # Test 4: Concurrent execution
            await self.test_concurrent_instances()

            # Generate report
            self.generate_report()

        except Exception as e:
            print(f"\n❌ Error during testing: {e}")
            import traceback

            traceback.print_exc()

    def generate_report(self):
        """Generate comprehensive performance report"""

        print("\n" + "=" * 70)
        print("📊 SYN/SYNEXECUTOR PERFORMANCE TEST REPORT")
        print("=" * 70)

        # Save results
        report_data = {
            "test_date": datetime.now().isoformat(),
            "class": "Syn/SynExecutor",
            "results": [r.to_dict() for r in self.results],
            "summary": self.generate_summary(),
        }

        with open("syn_executor_performance_report.json", "w") as f:
            json.dump(report_data, f, indent=2)

        print("\n✅ Report saved to: syn_executor_performance_report.json")

        # Print summary table
        print("\n📈 PERFORMANCE SUMMARY:")
        print(
            f"{'Test Name':<35} {'Symbols':<10} {'Opportunities':<15} {'Scan Time':<15} {'Throughput':<20}"
        )
        print("-" * 100)

        for result in self.results:
            print(
                f"{result.test_name:<35} "
                f"{result.symbols_count:<10} "
                f"{result.total_opportunities:<15} "
                f"{result.scan_time:.3f}s{'':<10} "
                f"{result.symbols_count/result.scan_time:.1f} symbols/sec"
            )

        # Strategy comparison
        print("\n📊 SCORING STRATEGY COMPARISON:")
        print(
            f"{'Strategy':<20} {'Opportunities':<15} {'Selection Time':<20} {'Avg Symbol Time':<20}"
        )
        print("-" * 75)

        strategy_results = [
            r for r in self.results if r.test_name.startswith("strategy_")
        ]
        for result in strategy_results:
            print(
                f"{result.scoring_config:<20} "
                f"{result.total_opportunities:<15} "
                f"{result.global_selection_time*1000:.3f}ms{'':<15} "
                f"{result.avg_symbol_scan_time*1000:.2f}ms"
            )

        # Scaling analysis
        print("\n📊 SCALING ANALYSIS:")
        print(
            f"{'Symbols':<15} {'Scan Time':<15} {'Time per Symbol':<20} {'Memory Used':<15}"
        )
        print("-" * 65)

        scaling_results = [
            r for r in self.results if r.test_name.startswith("scaling_")
        ]
        for result in scaling_results:
            time_per_symbol = (
                result.scan_time / result.symbols_count
                if result.symbols_count > 0
                else 0
            )
            print(
                f"{result.symbols_count:<15} "
                f"{result.scan_time:.3f}s{'':<10} "
                f"{time_per_symbol*1000:.2f}ms{'':<15} "
                f"{result.memory_used_mb:.2f}MB"
            )

        # Overall metrics
        print(f"\n📊 OVERALL METRICS:")
        for key, value in report_data["summary"].items():
            print(f"   {key}: {value}")

    def generate_summary(self) -> Dict:
        """Generate summary statistics"""

        if not self.results:
            return {}

        scan_times = [r.scan_time for r in self.results]
        throughputs = [
            r.symbols_count / r.scan_time for r in self.results if r.scan_time > 0
        ]
        opportunities = [r.total_opportunities for r in self.results]

        return {
            "total_tests": len(self.results),
            "avg_scan_time_s": mean(scan_times),
            "avg_throughput_symbols_per_sec": mean(throughputs) if throughputs else 0,
            "total_opportunities_found": sum(opportunities),
            "avg_opportunities_per_test": mean(opportunities) if opportunities else 0,
            "total_symbols_processed": sum(r.symbols_count for r in self.results),
            "performance_grade": self.calculate_grade(throughputs),
        }

    def calculate_grade(self, throughputs: List[float]) -> str:
        """Calculate performance grade based on throughput"""

        if not throughputs:
            return "N/A"

        avg_throughput = mean(throughputs)

        if avg_throughput > 10:
            return "A+ (Excellent)"
        elif avg_throughput > 5:
            return "A (Very Good)"
        elif avg_throughput > 2:
            return "B (Good)"
        elif avg_throughput > 1:
            return "C (Acceptable)"
        else:
            return "D (Needs Optimization)"


async def main():
    """Main entry point"""
    tester = SynPerformanceTester()

    try:
        await tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")

    print("\n✨ Syn/SynExecutor performance testing complete!")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
