#!/usr/bin/env python3
"""
PyPy vs CPython Performance Benchmarks for alc-lite

This module provides comprehensive benchmarks comparing PyPy and CPython performance
for key alc-lite operations including options chain processing, arbitrage detection,
and parallel execution simulation.
"""

import asyncio
import json

# Add project root to path for imports
import os
import random
import statistics
import sys
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from modules.Arbitrage.pypy_config import (
        get_optimization_hints,
        get_performance_config,
        is_pypy,
    )
    from modules.Arbitrage.sfr.constants import (
        BATCH_PROCESSING_SIZE,
        DEFAULT_CACHE_SIZE,
        NUMPY_VECTORIZATION_THRESHOLD,
    )
except ImportError as e:
    print(f"Warning: Could not import alc-lite modules: {e}")

    # Fallback values for standalone benchmarking
    def get_performance_config():
        return {"batch_size": 50, "cache_ttl": 300}

    def get_optimization_hints():
        return {"runtime": "Unknown"}

    def is_pypy():
        return hasattr(sys, "pypy_version_info")

    NUMPY_VECTORIZATION_THRESHOLD = 10
    BATCH_PROCESSING_SIZE = 50
    DEFAULT_CACHE_SIZE = 200


class PerformanceBenchmark:
    """Main benchmark class for PyPy vs CPython performance testing"""

    def __init__(self):
        self.results = {}
        self.runtime_info = self._get_runtime_info()
        self.config = get_performance_config()

    def _get_runtime_info(self) -> Dict[str, Any]:
        """Get information about the current Python runtime"""
        info = {
            "python_version": sys.version,
            "is_pypy": is_pypy(),
            "platform": sys.platform,
            "timestamp": datetime.now().isoformat(),
        }

        if is_pypy():
            info.update(
                {
                    "runtime": "PyPy",
                    "pypy_version": f"{sys.pypy_version_info.major}.{sys.pypy_version_info.minor}.{sys.pypy_version_info.micro}",
                    "jit_enabled": True,
                }
            )
        else:
            info.update(
                {
                    "runtime": "CPython",
                    "implementation": sys.implementation.name,
                }
            )

        return info

    def time_function(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Time a function call and return result and execution time"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time

    async def time_async_function(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Time an async function call and return result and execution time"""
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time

    def run_multiple_times(
        self, func, iterations: int = 10, *args, **kwargs
    ) -> Dict[str, float]:
        """Run a function multiple times and return timing statistics"""
        times = []

        for i in range(iterations):
            _, duration = self.time_function(func, *args, **kwargs)
            times.append(duration)

            # Progress indicator for long benchmarks
            if iterations > 50 and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{iterations} iterations completed")

        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
            "iterations": iterations,
            "total_time": sum(times),
        }

    def benchmark_options_chain_processing(self) -> Dict[str, Any]:
        """Benchmark options chain processing performance"""
        print("🔄 Benchmarking options chain processing...")

        def generate_mock_options_chain(size: int) -> List[Dict]:
            """Generate mock options data for benchmarking"""
            options = []
            base_price = 100.0

            for i in range(size):
                strike = base_price + (i - size // 2) * 5
                options.append(
                    {
                        "strike": strike,
                        "expiry": "20250315",
                        "bid": max(0.01, strike - base_price + random.uniform(-5, 5)),
                        "ask": max(0.02, strike - base_price + random.uniform(-4, 6)),
                        "volume": random.randint(0, 1000),
                        "option_type": "C" if i % 2 else "P",
                    }
                )

            return options

        def process_options_chain(options: List[Dict]) -> Dict:
            """Simulate options chain processing"""
            result = {
                "total_options": len(options),
                "calls": [],
                "puts": [],
                "atm_options": [],
                "high_volume_options": [],
            }

            base_price = 100.0

            for option in options:
                # Simulate complex processing
                mid_price = (option["bid"] + option["ask"]) / 2
                moneyness = option["strike"] / base_price

                if option["option_type"] == "C":
                    result["calls"].append(
                        {
                            "strike": option["strike"],
                            "mid_price": mid_price,
                            "moneyness": moneyness,
                        }
                    )
                else:
                    result["puts"].append(
                        {
                            "strike": option["strike"],
                            "mid_price": mid_price,
                            "moneyness": moneyness,
                        }
                    )

                # Simulate ATM detection
                if 0.95 <= moneyness <= 1.05:
                    result["atm_options"].append(option)

                # Simulate volume filtering
                if option["volume"] > 100:
                    result["high_volume_options"].append(option)

            return result

        # Test different chain sizes
        sizes = [50, 100, 500, 1000, 2000]
        results = {}

        for size in sizes:
            print(f"  Testing chain size: {size}")
            options = generate_mock_options_chain(size)

            # Use PyPy-aware batch size
            batch_size = self.config.get("batch_size", 50)
            iterations = max(5, 50 // (size // 100)) if size > 100 else 20

            timing_stats = self.run_multiple_times(
                process_options_chain, iterations, options
            )

            results[f"chain_size_{size}"] = {
                "timing": timing_stats,
                "throughput_ops_per_sec": size / timing_stats["mean"],
                "batch_size_used": batch_size,
            }

        return results

    def benchmark_arbitrage_detection(self) -> Dict[str, Any]:
        """Benchmark arbitrage detection algorithms"""
        print("🔄 Benchmarking arbitrage detection...")

        def generate_mock_arbitrage_data(num_strikes: int) -> List[Dict]:
            """Generate mock arbitrage opportunity data"""
            base_price = 100.0
            opportunities = []

            for i in range(num_strikes):
                call_strike = base_price + i * 5
                put_strike = call_strike

                # Create synthetic arbitrage scenario
                call_bid = max(0.01, random.uniform(1, 10))
                call_ask = call_bid + random.uniform(0.05, 0.50)
                put_bid = max(0.01, random.uniform(1, 10))
                put_ask = put_bid + random.uniform(0.05, 0.50)

                opportunities.append(
                    {
                        "call_strike": call_strike,
                        "put_strike": put_strike,
                        "call_bid": call_bid,
                        "call_ask": call_ask,
                        "put_bid": put_bid,
                        "put_ask": put_ask,
                        "stock_price": base_price,
                        "expiry_days": random.randint(1, 60),
                    }
                )

            return opportunities

        def detect_arbitrage_opportunities(data: List[Dict]) -> List[Dict]:
            """Simulate arbitrage detection algorithm"""
            opportunities = []

            for item in data:
                # Simulate complex arbitrage calculations
                synthetic_stock_cost = item["call_ask"] - item["put_bid"]
                actual_stock_price = item["stock_price"]

                # Calculate theoretical profit
                profit = abs(
                    actual_stock_price - (item["call_strike"] + synthetic_stock_cost)
                )

                # Apply quality scoring (simulate complex calculations)
                volume_score = random.uniform(0.5, 1.0)
                spread_score = 1.0 / (1.0 + (item["call_ask"] - item["call_bid"]))
                time_score = 1.0 / (1.0 + item["expiry_days"] / 30.0)

                quality_score = (
                    volume_score * 0.4 + spread_score * 0.3 + time_score * 0.3
                )

                if profit > 0.05 and quality_score > 0.6:
                    opportunities.append(
                        {
                            "profit": profit,
                            "quality_score": quality_score,
                            "call_strike": item["call_strike"],
                            "put_strike": item["put_strike"],
                            "cost": synthetic_stock_cost,
                        }
                    )

            # Sort by profit (simulate ranking)
            return sorted(opportunities, key=lambda x: x["profit"], reverse=True)

        # Test different data sizes
        sizes = [10, 50, 100, 500, 1000]
        results = {}

        for size in sizes:
            print(f"  Testing arbitrage detection with {size} strikes")
            data = generate_mock_arbitrage_data(size)

            iterations = max(5, 100 // (size // 50)) if size > 50 else 50

            timing_stats = self.run_multiple_times(
                detect_arbitrage_opportunities, iterations, data
            )

            results[f"strikes_{size}"] = {
                "timing": timing_stats,
                "throughput_items_per_sec": size / timing_stats["mean"],
            }

        return results

    def benchmark_parallel_execution_simulation(self) -> Dict[str, Any]:
        """Benchmark parallel execution monitoring simulation"""
        print("🔄 Benchmarking parallel execution simulation...")

        async def simulate_order_monitoring(
            num_orders: int, monitoring_duration: float = 1.0
        ) -> Dict:
            """Simulate the order monitoring loop from parallel executor"""
            orders = [
                {
                    "id": f"order_{i}",
                    "status": "pending",
                    "fill_time": None,
                    "fill_probability": random.uniform(0.1, 0.9),
                }
                for i in range(num_orders)
            ]

            start_time = time.time()
            filled_orders = []
            monitoring_cycles = 0

            # Simulate the monitoring loop
            while (time.time() - start_time) < monitoring_duration and len(
                filled_orders
            ) < num_orders:
                monitoring_cycles += 1

                # Check each pending order (simulate the hot loop)
                for order in orders[:]:  # Copy to avoid modification during iteration
                    if order["status"] == "pending":
                        # Simulate fill check with some randomness
                        if (
                            random.random() < order["fill_probability"] * 0.1
                        ):  # 10% chance per cycle
                            order["status"] = "filled"
                            order["fill_time"] = time.time()
                            filled_orders.append(order)

                # Remove filled orders from monitoring (simulate PyPy-optimized batch removal)
                orders = [order for order in orders if order["status"] == "pending"]

                # Simulate brief pause
                await asyncio.sleep(0.01)  # 10ms like the real executor

            return {
                "filled_orders": len(filled_orders),
                "total_orders": num_orders,
                "monitoring_cycles": monitoring_cycles,
                "fill_rate": len(filled_orders) / num_orders if num_orders > 0 else 0,
            }

        async def run_monitoring_benchmark():
            order_counts = [1, 3, 5, 10]  # Typical number of legs in parallel execution
            results = {}

            for count in order_counts:
                print(f"  Testing monitoring with {count} orders")

                # Run multiple iterations
                iterations = 10
                times = []
                fill_rates = []

                for _ in range(iterations):
                    start_time = time.perf_counter()
                    result = await simulate_order_monitoring(
                        count, monitoring_duration=0.5
                    )
                    end_time = time.perf_counter()

                    times.append(end_time - start_time)
                    fill_rates.append(result["fill_rate"])

                results[f"orders_{count}"] = {
                    "timing": {
                        "mean": statistics.mean(times),
                        "median": statistics.median(times),
                        "min": min(times),
                        "max": max(times),
                        "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
                    },
                    "fill_rate": {
                        "mean": statistics.mean(fill_rates),
                        "min": min(fill_rates),
                        "max": max(fill_rates),
                    },
                }

            return results

        return asyncio.run(run_monitoring_benchmark())

    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns"""
        print("🔄 Benchmarking memory usage...")

        try:
            import psutil

            process = psutil.Process()

            def get_memory_usage():
                return process.memory_info().rss / 1024 / 1024  # MB

        except ImportError:
            print("  Warning: psutil not available, skipping memory benchmarks")
            return {"error": "psutil not available"}

        def memory_intensive_operation(size: int) -> List:
            """Simulate memory-intensive operation"""
            # Create large data structures typical of options processing
            data = []
            for i in range(size):
                option_data = {
                    "id": f"option_{i}",
                    "strike": 100 + i,
                    "prices": [random.uniform(1, 100) for _ in range(100)],
                    "greeks": {
                        "delta": random.uniform(-1, 1),
                        "gamma": random.uniform(0, 0.1),
                        "theta": random.uniform(-0.1, 0),
                        "vega": random.uniform(0, 1),
                        "rho": random.uniform(-1, 1),
                    },
                    "history": [random.uniform(0.5, 2.0) for _ in range(1000)],
                }
                data.append(option_data)

            # Simulate processing that creates temporary objects
            processed = []
            for item in data:
                processed_item = {
                    "id": item["id"],
                    "avg_price": sum(item["prices"]) / len(item["prices"]),
                    "price_variance": statistics.variance(item["prices"]),
                    "history_trend": sum(item["history"]) / len(item["history"]),
                }
                processed.append(processed_item)

            return processed

        sizes = [100, 500, 1000, 2000]
        results = {}

        initial_memory = get_memory_usage()

        for size in sizes:
            print(f"  Testing memory usage with {size} options")

            before_memory = get_memory_usage()

            # Run the memory-intensive operation
            start_time = time.perf_counter()
            data = memory_intensive_operation(size)
            end_time = time.perf_counter()

            after_memory = get_memory_usage()

            # Force garbage collection
            import gc

            gc.collect()

            after_gc_memory = get_memory_usage()

            results[f"size_{size}"] = {
                "execution_time": end_time - start_time,
                "memory_before_mb": before_memory,
                "memory_after_mb": after_memory,
                "memory_after_gc_mb": after_gc_memory,
                "memory_used_mb": after_memory - before_memory,
                "memory_retained_mb": after_gc_memory - before_memory,
                "items_processed": len(data),
            }

        results["baseline_memory_mb"] = initial_memory
        return results

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark suites"""
        print(f"🚀 Running benchmarks on {self.runtime_info['runtime']}")
        print(f"📊 Python version: {self.runtime_info['python_version'].split()[0]}")

        if is_pypy():
            print(f"⚡ PyPy version: {self.runtime_info['pypy_version']}")
            print(
                "💡 Expecting significant performance improvements for pure Python code"
            )
        else:
            print("🐍 Running on CPython - baseline performance")

        print()

        all_results = {
            "runtime_info": self.runtime_info,
            "config_used": self.config,
            "benchmarks": {},
        }

        # Run each benchmark suite
        try:
            all_results["benchmarks"][
                "options_chain_processing"
            ] = self.benchmark_options_chain_processing()
        except Exception as e:
            print(f"❌ Options chain processing benchmark failed: {e}")
            all_results["benchmarks"]["options_chain_processing"] = {"error": str(e)}

        try:
            all_results["benchmarks"][
                "arbitrage_detection"
            ] = self.benchmark_arbitrage_detection()
        except Exception as e:
            print(f"❌ Arbitrage detection benchmark failed: {e}")
            all_results["benchmarks"]["arbitrage_detection"] = {"error": str(e)}

        try:
            all_results["benchmarks"][
                "parallel_execution_simulation"
            ] = self.benchmark_parallel_execution_simulation()
        except Exception as e:
            print(f"❌ Parallel execution simulation benchmark failed: {e}")
            all_results["benchmarks"]["parallel_execution_simulation"] = {
                "error": str(e)
            }

        try:
            all_results["benchmarks"]["memory_usage"] = self.benchmark_memory_usage()
        except Exception as e:
            print(f"❌ Memory usage benchmark failed: {e}")
            all_results["benchmarks"]["memory_usage"] = {"error": str(e)}

        return all_results

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save benchmark results to JSON file"""
        if filename is None:
            runtime = "pypy" if is_pypy() else "cpython"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{runtime}_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"📄 Results saved to: {filename}")
        return filename


def print_performance_summary(results: Dict[str, Any]):
    """Print a summary of benchmark results"""
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)

    runtime_info = results["runtime_info"]
    print(f"Runtime: {runtime_info['runtime']}")
    print(
        f"Version: {runtime_info.get('pypy_version', runtime_info['python_version'].split()[0])}"
    )
    print(f"Platform: {runtime_info['platform']}")
    print()

    benchmarks = results["benchmarks"]

    # Options chain processing summary
    if "options_chain_processing" in benchmarks:
        chain_results = benchmarks["options_chain_processing"]
        print("📈 Options Chain Processing:")
        for size_key, data in chain_results.items():
            if "error" not in data:
                size = size_key.split("_")[-1]
                mean_time = data["timing"]["mean"]
                throughput = data["throughput_ops_per_sec"]
                print(
                    f"  • {size} options: {mean_time:.4f}s avg, {throughput:.0f} ops/sec"
                )
        print()

    # Arbitrage detection summary
    if "arbitrage_detection" in benchmarks:
        arb_results = benchmarks["arbitrage_detection"]
        print("🎯 Arbitrage Detection:")
        for size_key, data in arb_results.items():
            if "error" not in data:
                size = size_key.split("_")[-1]
                mean_time = data["timing"]["mean"]
                throughput = data["throughput_items_per_sec"]
                print(
                    f"  • {size} strikes: {mean_time:.4f}s avg, {throughput:.0f} items/sec"
                )
        print()

    # Parallel execution summary
    if "parallel_execution_simulation" in benchmarks:
        parallel_results = benchmarks["parallel_execution_simulation"]
        print("⚡ Parallel Execution Monitoring:")
        for orders_key, data in parallel_results.items():
            if "error" not in data:
                orders = orders_key.split("_")[-1]
                mean_time = data["timing"]["mean"]
                fill_rate = data["fill_rate"]["mean"]
                print(
                    f"  • {orders} orders: {mean_time:.4f}s avg, {fill_rate:.1%} fill rate"
                )
        print()

    # Memory usage summary
    if "memory_usage" in benchmarks:
        mem_results = benchmarks["memory_usage"]
        print("💾 Memory Usage:")
        if "error" not in mem_results:
            for size_key, data in mem_results.items():
                if size_key.startswith("size_"):
                    size = size_key.split("_")[-1]
                    memory_used = data["memory_used_mb"]
                    memory_retained = data["memory_retained_mb"]
                    exec_time = data["execution_time"]
                    print(
                        f"  • {size} options: {memory_used:.1f}MB used, {memory_retained:.1f}MB retained, {exec_time:.3f}s"
                    )
        else:
            print(f"  ❌ {mem_results['error']}")
        print()

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="PyPy vs CPython Performance Benchmarks for alc-lite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pypy_performance.py                    # Run all benchmarks
  python pypy_performance.py --output results.json  # Save to specific file
  pypy3 pypy_performance.py                    # Run with PyPy for comparison
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output file for benchmark results (default: auto-generated)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks with fewer iterations",
    )

    args = parser.parse_args()

    # Create benchmark instance
    benchmark = PerformanceBenchmark()

    print("🏎️ alc-lite Performance Benchmarks")
    print("=" * 60)

    # Run benchmarks
    results = benchmark.run_all_benchmarks()

    # Print summary
    print_performance_summary(results)

    # Save results
    output_file = benchmark.save_results(results, args.output)

    print(f"\n✅ Benchmarks completed! Results saved to {output_file}")

    if is_pypy():
        print("\n💡 For comparison, run the same benchmarks with CPython:")
        print("   python benchmarks/pypy_performance.py")
    else:
        print("\n💡 For performance comparison, run with PyPy:")
        print("   conda activate alc-pypy")
        print("   pypy3 benchmarks/pypy_performance.py")


if __name__ == "__main__":
    main()
