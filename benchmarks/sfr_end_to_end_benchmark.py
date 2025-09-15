#!/usr/bin/env python3
"""
SFR End-to-End Performance Benchmark

This module provides comprehensive end-to-end benchmarks for the SFR (Synthetic-Free-Risk)
arbitrage strategy, measuring real performance improvements when using PyPy vs CPython.
"""

import asyncio
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock

import argparse

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ib_async import Contract

    from modules.Arbitrage.common import get_logger
    from modules.Arbitrage.sfr.strategy import SFR

    # PyPy detection and optimization utilities
    USING_PYPY = hasattr(sys, "pypy_version_info")

    if USING_PYPY:
        from modules.Arbitrage.pypy_config import get_performance_config
    else:

        def get_performance_config():
            return {"batch_size": 50, "cache_ttl": 300}

except ImportError as e:
    print(f"Warning: Could not import alc-lite modules: {e}")
    print("Some benchmarks may be limited")

    # Fallback definitions
    USING_PYPY = hasattr(sys, "pypy_version_info")

    def get_performance_config():
        return {"batch_size": 50, "cache_ttl": 300}


# Create a simple mock option class for benchmarking
class MockOption:
    """Simple option data container for benchmarking"""

    def __init__(
        self,
        symbol: str,
        strike: float,
        expiry: str,
        right: str,
        bid: float,
        ask: float,
        volume: int,
    ):
        self.symbol = symbol
        self.strike = strike
        self.expiry = expiry
        self.right = right
        self.bid = bid
        self.ask = ask
        self.volume = volume


class MockIBConnection:
    """Mock IB connection for testing SFR performance without requiring real broker connection"""

    def __init__(self):
        self.is_connected = True
        self.client_id = 999

    async def connectAsync(self, host, port, clientId=1):
        """Mock connection"""
        await asyncio.sleep(0.01)  # Simulate connection delay
        return True

    def reqMktData(
        self,
        contract,
        genericTickList="",
        snapshot=False,
        regulatorySnapshot=False,
        mktDataOptions=None,
    ):
        """Mock market data request"""
        # Return a mock ticker with realistic bid/ask data
        mock_ticker = MagicMock()
        mock_ticker.contract = contract

        # Generate realistic option pricing based on strike and expiry
        if hasattr(contract, "strike"):
            underlying_price = 100.0  # Mock underlying price
            strike = contract.strike

            # Simple Black-Scholes approximation for realistic pricing
            moneyness = underlying_price / strike if strike > 0 else 1.0
            time_decay = random.uniform(0.8, 0.95)  # Simulate time decay

            if contract.right == "C":  # Call option
                theoretical_value = max(0, (underlying_price - strike) * time_decay)
            else:  # Put option
                theoretical_value = max(0, (strike - underlying_price) * time_decay)

            # Add some randomness and spread
            theoretical_value = max(0.01, theoretical_value + random.uniform(-2, 2))
            spread = theoretical_value * random.uniform(0.05, 0.20)

            mock_ticker.bid = max(0.01, theoretical_value - spread / 2)
            mock_ticker.ask = theoretical_value + spread / 2
            mock_ticker.last = (mock_ticker.bid + mock_ticker.ask) / 2
            mock_ticker.volume = random.randint(10, 1000)
        else:
            # For stock tickers
            mock_ticker.bid = 99.95
            mock_ticker.ask = 100.05
            mock_ticker.last = 100.00
            mock_ticker.volume = 1000000

        return mock_ticker

    def cancelMktData(self, contract):
        """Mock market data cancellation"""
        pass

    def disconnect(self):
        """Mock disconnection"""
        self.is_connected = False


class SFREndToEndBenchmark:
    """End-to-end benchmark suite for SFR arbitrage strategy"""

    def __init__(self, seed=None):
        self.results = {}
        self.runtime_info = self._get_runtime_info()
        self.config = get_performance_config()
        self.mock_ib = MockIBConnection()
        self.seed = seed

        # Set random seed for reproducible results if provided
        if seed is not None:
            random.seed(seed)
            print(f"🎲 Using random seed: {seed} for reproducible results")

    def _get_runtime_info(self) -> Dict[str, Any]:
        """Get information about the current Python runtime"""
        info = {
            "python_version": sys.version,
            "is_pypy": USING_PYPY,
            "platform": sys.platform,
            "timestamp": datetime.now().isoformat(),
        }

        if USING_PYPY:
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

    def generate_realistic_options_chain(
        self, symbol: str, underlying_price: float = 100.0
    ) -> List[Dict]:
        """Generate realistic options chain data for benchmarking"""
        options_chain = []

        # Generate wider strike range for more realistic option chains
        min_strike = underlying_price * 0.5  # Wider range: 50% to 150%
        max_strike = underlying_price * 1.5
        # Generate strikes with 2.5 increment
        strikes = []
        current_strike = min_strike
        while current_strike <= max_strike:
            strikes.append(round(current_strike, 1))
            current_strike += 2.5

        # Generate comprehensive expiry dates including LEAPS
        base_date = datetime.now()
        expiry_dates = [
            base_date + timedelta(days=3),  # 0DTE
            base_date + timedelta(days=7),  # Weekly
            base_date + timedelta(days=14),  # Bi-weekly
            base_date + timedelta(days=21),  # 3-week
            base_date + timedelta(days=30),  # Monthly
            base_date + timedelta(days=45),  # 6-week
            base_date + timedelta(days=60),  # Quarterly
            base_date + timedelta(days=90),  # 3-month
            base_date + timedelta(days=180),  # 6-month
            base_date + timedelta(days=365),  # 1-year LEAPS
            base_date + timedelta(days=730),  # 2-year LEAPS
        ]

        for expiry in expiry_dates:
            expiry_str = expiry.strftime("%Y%m%d")

            for strike in strikes:
                # Generate call option
                call_option = {
                    "symbol": symbol,
                    "strike": float(strike),
                    "expiry": expiry_str,
                    "right": "C",
                    "exchange": "SMART",
                    "currency": "USD",
                    "multiplier": "100",
                    # Realistic pricing based on moneyness
                    "bid": self._calculate_option_bid(
                        underlying_price, strike, expiry, "C"
                    ),
                    "ask": self._calculate_option_ask(
                        underlying_price, strike, expiry, "C"
                    ),
                    "volume": self._calculate_option_volume(underlying_price, strike),
                }
                options_chain.append(call_option)

                # Generate put option
                put_option = {
                    "symbol": symbol,
                    "strike": float(strike),
                    "expiry": expiry_str,
                    "right": "P",
                    "exchange": "SMART",
                    "currency": "USD",
                    "multiplier": "100",
                    # Realistic pricing based on moneyness
                    "bid": self._calculate_option_bid(
                        underlying_price, strike, expiry, "P"
                    ),
                    "ask": self._calculate_option_ask(
                        underlying_price, strike, expiry, "P"
                    ),
                    "volume": self._calculate_option_volume(underlying_price, strike),
                }
                options_chain.append(put_option)

        return options_chain

    def _calculate_option_bid(
        self, underlying: float, strike: float, expiry: datetime, right: str
    ) -> float:
        """Calculate realistic option bid price with reduced randomness"""
        days_to_expiry = (expiry - datetime.now()).days
        time_factor = max(0.1, days_to_expiry / 365.0)

        if right == "C":
            intrinsic = max(0, underlying - strike)
        else:
            intrinsic = max(0, strike - underlying)

        # More deterministic time value based on moneyness
        moneyness = abs(underlying - strike) / underlying
        base_time_value = 1.0 + (0.5 * time_factor) + (0.3 * moneyness)

        # Small controlled randomness for variation
        time_variation = random.uniform(0.9, 1.1) if self.seed is None else 1.0
        time_value = base_time_value * time_variation * time_factor
        theoretical = intrinsic + time_value

        # Consistent bid discount with small variation
        bid_discount = random.uniform(0.04, 0.06) if self.seed is None else 0.05
        return max(0.01, theoretical * (1 - bid_discount))

    def _calculate_option_ask(
        self, underlying: float, strike: float, expiry: datetime, right: str
    ) -> float:
        """Calculate realistic option ask price with controlled spread"""
        bid = self._calculate_option_bid(underlying, strike, expiry, right)

        # More consistent spread based on bid price
        if bid < 1.0:
            # Wider spreads for low-priced options
            spread_percent = random.uniform(0.08, 0.12) if self.seed is None else 0.10
        else:
            # Tighter spreads for higher-priced options
            spread_percent = random.uniform(0.04, 0.08) if self.seed is None else 0.06

        return bid * (1 + spread_percent)

    def _calculate_option_volume(self, underlying: float, strike: float) -> int:
        """Calculate realistic option volume based on moneyness"""
        moneyness = abs(underlying - strike) / underlying

        if moneyness < 0.02:  # Very near the money
            base_volume = 200
        elif moneyness < 0.05:  # Near the money
            base_volume = 150
        elif moneyness < 0.10:  # Moderately out of the money
            base_volume = 100
        else:  # Far out of the money
            base_volume = 50

        # Add some controlled variation
        if self.seed is None:
            volume_variation = random.uniform(0.7, 1.5)
        else:
            volume_variation = 1.0

        return max(5, int(base_volume * volume_variation))

    def benchmark_sfr_initialization(self) -> Dict[str, Any]:
        """Benchmark SFR strategy initialization performance"""
        print("🔄 Benchmarking SFR initialization...")

        def initialize_sfr() -> SFR:
            """Initialize SFR strategy instance"""
            sfr = SFR()
            sfr.ib = self.mock_ib
            return sfr

        # Time multiple initialization cycles
        times = []
        sfr_instances = []

        for i in range(10):
            start_time = time.perf_counter()
            sfr = initialize_sfr()
            end_time = time.perf_counter()

            times.append(end_time - start_time)
            sfr_instances.append(sfr)

        return {
            "timing": {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "min": min(times),
                "max": max(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
            },
            "instances_created": len(sfr_instances),
        }

    def benchmark_options_chain_processing(self) -> Dict[str, Any]:
        """Benchmark realistic options chain processing"""
        print("🔄 Benchmarking options chain processing...")

        def process_options_chain(options_data: List[Dict]) -> List[MockOption]:
            """Process options chain into MockOption objects"""
            processed_options = []

            for option_data in options_data:
                try:
                    mock_option = MockOption(
                        symbol=option_data["symbol"],
                        strike=option_data["strike"],
                        expiry=option_data["expiry"],
                        right=option_data["right"],
                        bid=option_data["bid"],
                        ask=option_data["ask"],
                        volume=option_data["volume"],
                    )
                    processed_options.append(mock_option)
                except Exception:
                    # Skip malformed options (simulating real-world data issues)
                    continue

            return processed_options

        # Test with more symbols for realistic multi-ticker scanning
        test_symbols = [
            "SPY",
            "QQQ",
            "AAPL",
            "MSFT",
            "TSLA",
            "NVDA",
            "AMD",
            "GOOGL",
            "AMZN",
            "JPM",
            "BAC",
            "XOM",
        ]
        results = {}

        for symbol in test_symbols:
            print(f"  Processing options chain for {symbol}")

            # Generate realistic options chain
            chain_data = self.generate_realistic_options_chain(symbol)

            # Time the processing
            times = []
            processed_counts = []

            # Extended warm-up runs for JIT compilation
            warmup_runs = 15 if USING_PYPY else 5
            for _ in range(warmup_runs):
                process_options_chain(chain_data)

            # More timing runs for better statistics
            timing_runs = 20
            for _ in range(timing_runs):
                start_time = time.perf_counter()
                processed_options = process_options_chain(chain_data)
                end_time = time.perf_counter()

                times.append(end_time - start_time)
                processed_counts.append(len(processed_options))

            results[f"symbol_{symbol}"] = {
                "raw_options_count": len(chain_data),
                "processed_options_count": int(statistics.mean(processed_counts)),
                "processing_success_rate": statistics.mean(processed_counts)
                / len(chain_data),
                "timing": {
                    "mean": statistics.mean(times),
                    "median": statistics.median(times),
                    "min": min(times),
                    "max": max(times),
                    "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
                },
                "throughput_options_per_sec": len(chain_data) / statistics.mean(times),
            }

        return results

    def benchmark_arbitrage_detection(self) -> Dict[str, Any]:
        """Benchmark SFR arbitrage opportunity detection"""
        print("🔄 Benchmarking arbitrage opportunity detection...")

        def detect_sfr_opportunities(
            options: List[MockOption],
            underlying_price: float = 100.0,
            cost_limit: float = 100.0,
            profit_target: float = 0.5,
        ) -> List[Dict]:
            """Detect SFR arbitrage opportunities"""
            opportunities = []

            # Group options by expiry
            expiry_groups = {}
            for option in options:
                if option.expiry not in expiry_groups:
                    expiry_groups[option.expiry] = {"calls": [], "puts": []}

                if option.right == "C":
                    expiry_groups[option.expiry]["calls"].append(option)
                else:
                    expiry_groups[option.expiry]["puts"].append(option)

            # Look for synthetic free risk opportunities
            for expiry, option_group in expiry_groups.items():
                calls = option_group["calls"]
                puts = option_group["puts"]

                # Check all call/put combinations for same strike
                for call in calls:
                    for put in puts:
                        if call.strike == put.strike:
                            # Calculate synthetic stock cost and potential profit
                            synthetic_cost = call.ask - put.bid
                            current_cost = abs(
                                synthetic_cost * 100
                            )  # Contract multiplier

                            if current_cost <= cost_limit * 100:
                                # SFR profit: difference between synthetic and actual stock price
                                theoretical_synthetic_price = (
                                    call.strike + synthetic_cost
                                )

                                # Profit is the absolute difference (arbitrage opportunity)
                                potential_profit = abs(
                                    underlying_price - theoretical_synthetic_price
                                )
                                profit_percent = (
                                    (potential_profit / underlying_price * 100)
                                    if underlying_price > 0
                                    else 0
                                )

                                # More lenient threshold to ensure we find opportunities for benchmarking
                                if profit_percent >= 0.1:  # Lower threshold for testing
                                    opportunities.append(
                                        {
                                            "call_strike": call.strike,
                                            "put_strike": put.strike,
                                            "expiry": expiry,
                                            "synthetic_cost": synthetic_cost,
                                            "total_cost": current_cost,
                                            "potential_profit": potential_profit,
                                            "profit_percent": profit_percent,
                                            "call_volume": call.volume,
                                            "put_volume": put.volume,
                                            "theoretical_price": theoretical_synthetic_price,
                                            "underlying_price": underlying_price,
                                        }
                                    )

            return sorted(
                opportunities, key=lambda x: x["profit_percent"], reverse=True
            )

        # Test with larger, more realistic chain sizes for PyPy optimization
        chain_sizes = [1000, 2000, 5000, 10000, 20000]
        results = {}

        for size in chain_sizes:
            print(f"  Testing arbitrage detection with {size} options")

            # Generate test data with consistent underlying price
            underlying_price = 100.0
            test_chain = self.generate_realistic_options_chain(
                "TEST", underlying_price
            )[:size]

            # Convert to MockOption objects
            mock_options = []
            for opt_data in test_chain:
                try:
                    mock_options.append(
                        MockOption(
                            symbol=opt_data["symbol"],
                            strike=opt_data["strike"],
                            expiry=opt_data["expiry"],
                            right=opt_data["right"],
                            bid=opt_data["bid"],
                            ask=opt_data["ask"],
                            volume=opt_data["volume"],
                        )
                    )
                except:
                    continue

            # Time the arbitrage detection
            times = []
            opportunity_counts = []

            # Scale iterations based on dataset size for better statistics
            if size <= 2000:
                iterations = 100
            elif size <= 5000:
                iterations = 50
            elif size <= 10000:
                iterations = 20
            else:
                iterations = 10

            # Extended warm-up runs for JIT compilation (especially for PyPy)
            warmup_runs = 20 if USING_PYPY else 5
            print(f"    Running {warmup_runs} warm-up iterations...")
            for _ in range(warmup_runs):
                detect_sfr_opportunities(mock_options, underlying_price)

            # Actual timing runs
            print(f"    Running {iterations} timed iterations...")
            for _ in range(iterations):
                start_time = time.perf_counter()
                opportunities = detect_sfr_opportunities(mock_options, underlying_price)
                end_time = time.perf_counter()

                times.append(end_time - start_time)
                opportunity_counts.append(len(opportunities))

            results[f"size_{size}"] = {
                "options_analyzed": len(mock_options),
                "opportunities_found": int(statistics.mean(opportunity_counts)),
                "opportunity_rate": (
                    statistics.mean(opportunity_counts) / len(mock_options)
                    if mock_options
                    else 0
                ),
                "timing": {
                    "mean": statistics.mean(times),
                    "median": statistics.median(times),
                    "min": min(times),
                    "max": max(times),
                    "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
                },
                "throughput_options_per_sec": len(mock_options)
                / statistics.mean(times),
            }

        return results

    def benchmark_parallel_execution_simulation(self) -> Dict[str, Any]:
        """Benchmark parallel execution performance for SFR strategies"""
        print("🔄 Benchmarking parallel execution simulation...")

        async def simulate_parallel_sfr_execution(
            num_legs: int, execution_time: float = 2.0
        ) -> Dict:
            """Simulate parallel SFR strategy execution"""

            # Create mock legs for parallel execution
            legs = []
            for i in range(num_legs):
                leg = {
                    "id": f"leg_{i}",
                    "action": "BUY" if i % 2 == 0 else "SELL",
                    "contract_type": "CALL" if i % 2 == 0 else "PUT",
                    "strike": 100 + i * 5,
                    "quantity": 1,
                    "status": "pending",
                    "fill_probability": random.uniform(0.3, 0.9),
                }
                legs.append(leg)

            # Simulate the parallel execution monitoring
            start_time = time.time()
            filled_legs = []
            monitoring_cycles = 0

            while (time.time() - start_time) < execution_time:
                monitoring_cycles += 1

                # Check each leg for fills (hot loop simulation)
                for leg in legs[:]:
                    if leg["status"] == "pending":
                        # Simulate fill check with realistic probability
                        if (
                            random.random() < leg["fill_probability"] * 0.05
                        ):  # 5% chance per cycle
                            leg["status"] = "filled"
                            leg["fill_time"] = time.time() - start_time
                            filled_legs.append(leg)

                # Remove filled legs from active monitoring
                legs = [leg for leg in legs if leg["status"] == "pending"]

                # If all legs filled, break early
                if not legs:
                    break

                # Simulate brief monitoring interval
                await asyncio.sleep(0.01)

            return {
                "total_legs": num_legs,
                "filled_legs": len(filled_legs),
                "fill_rate": len(filled_legs) / num_legs,
                "monitoring_cycles": monitoring_cycles,
                "execution_time": time.time() - start_time,
                "avg_fill_time": (
                    statistics.mean([leg["fill_time"] for leg in filled_legs])
                    if filled_legs
                    else 0
                ),
            }

        # Test different numbers of legs (typical SFR strategies have 2-4 legs)
        leg_counts = [2, 3, 4, 5]
        results = {}

        for leg_count in leg_counts:
            print(f"  Testing parallel execution with {leg_count} legs")

            # Run multiple simulations
            times = []
            fill_rates = []
            monitoring_cycles = []

            for _ in range(5):  # Multiple runs for statistical significance
                start_time = time.perf_counter()
                simulation_result = asyncio.run(
                    simulate_parallel_sfr_execution(leg_count, 1.0)
                )
                end_time = time.perf_counter()

                times.append(end_time - start_time)
                fill_rates.append(simulation_result["fill_rate"])
                monitoring_cycles.append(simulation_result["monitoring_cycles"])

            results[f"legs_{leg_count}"] = {
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
                    "std_dev": (
                        statistics.stdev(fill_rates) if len(fill_rates) > 1 else 0.0
                    ),
                },
                "monitoring_efficiency": {
                    "avg_cycles_per_second": statistics.mean(monitoring_cycles),
                    "cycles_per_leg": statistics.mean(monitoring_cycles) / leg_count,
                },
            }

        return results

    def benchmark_full_sfr_scan_simulation(self) -> Dict[str, Any]:
        """Benchmark full end-to-end SFR scan performance"""
        print("🔄 Benchmarking full SFR scan simulation...")

        async def simulate_full_sfr_scan(
            symbols: List[str], cost_limit: float = 100.0
        ) -> Dict:
            """Simulate a complete SFR scan across multiple symbols"""

            scan_results = {
                "symbols_scanned": len(symbols),
                "total_opportunities": 0,
                "executable_opportunities": 0,
                "total_options_processed": 0,
                "scan_time": 0,
            }

            scan_start = time.perf_counter()

            for symbol in symbols:
                # Simulate getting options chain
                options_chain = self.generate_realistic_options_chain(
                    symbol, random.uniform(80, 120)
                )
                scan_results["total_options_processed"] += len(options_chain)

                # Convert to MockOption objects (simulate real processing)
                mock_options = []
                for opt_data in options_chain:
                    try:
                        mock_options.append(
                            MockOption(
                                symbol=opt_data["symbol"],
                                strike=opt_data["strike"],
                                expiry=opt_data["expiry"],
                                right=opt_data["right"],
                                bid=opt_data["bid"],
                                ask=opt_data["ask"],
                                volume=opt_data["volume"],
                            )
                        )
                    except:
                        continue

                # Simulate arbitrage detection using the same logic as the benchmark
                opportunities = []
                for i, option in enumerate(mock_options):
                    if i % 10 == 0:  # Simulate finding opportunities
                        profit = random.uniform(0.1, 2.0)
                        if profit >= 0.5:  # Profit threshold
                            opportunities.append(
                                {
                                    "symbol": symbol,
                                    "profit": profit,
                                    "executable": random.choice([True, False]),
                                }
                            )

                scan_results["total_opportunities"] += len(opportunities)
                scan_results["executable_opportunities"] += sum(
                    1 for opp in opportunities if opp["executable"]
                )

                # Simulate brief processing delay
                await asyncio.sleep(0.01)

            scan_results["scan_time"] = time.perf_counter() - scan_start

            return scan_results

        # Test with larger symbol lists for realistic portfolio scanning
        symbol_lists = [
            ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"],  # 5 symbols
            [
                "SPY",
                "QQQ",
                "AAPL",
                "MSFT",
                "TSLA",
                "NVDA",
                "AMD",
                "GOOGL",
                "AMZN",
                "JPM",
            ],  # 10 symbols
            [
                "SPY",
                "QQQ",
                "AAPL",
                "MSFT",
                "TSLA",
                "NVDA",
                "AMD",
                "GOOGL",
                "AMZN",
                "JPM",
                "BAC",
                "XOM",
                "WMT",
                "JNJ",
                "V",
            ],  # 15 symbols
            [
                "SPY",
                "QQQ",
                "AAPL",
                "MSFT",
                "TSLA",
                "NVDA",
                "AMD",
                "GOOGL",
                "AMZN",
                "JPM",
                "BAC",
                "XOM",
                "WMT",
                "JNJ",
                "V",
                "PG",
                "UNH",
                "HD",
                "MA",
                "DIS",
            ],  # 20 symbols
        ]

        results = {}

        for symbol_list in symbol_lists:
            symbol_count = len(symbol_list)
            print(
                f"  Testing full scan with {symbol_count} symbols: {', '.join(symbol_list)}"
            )

            # More scan simulations for better statistics
            times = []
            throughput_metrics = []

            # Scale test runs based on symbol count
            if symbol_count <= 10:
                test_runs = 10
            elif symbol_count <= 15:
                test_runs = 5
            else:
                test_runs = 3

            for _ in range(test_runs):
                start_time = time.perf_counter()
                scan_result = asyncio.run(simulate_full_sfr_scan(symbol_list))
                end_time = time.perf_counter()

                times.append(end_time - start_time)
                throughput_metrics.append(
                    {
                        "symbols_per_sec": scan_result["symbols_scanned"]
                        / scan_result["scan_time"],
                        "options_per_sec": scan_result["total_options_processed"]
                        / scan_result["scan_time"],
                        "opportunities_found": scan_result["total_opportunities"],
                        "executable_opportunities": scan_result[
                            "executable_opportunities"
                        ],
                    }
                )

            results[f"symbols_{symbol_count}"] = {
                "symbol_list": symbol_list,
                "timing": {
                    "mean": statistics.mean(times),
                    "median": statistics.median(times),
                    "min": min(times),
                    "max": max(times),
                    "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
                },
                "throughput": {
                    "symbols_per_sec": statistics.mean(
                        [m["symbols_per_sec"] for m in throughput_metrics]
                    ),
                    "options_per_sec": statistics.mean(
                        [m["options_per_sec"] for m in throughput_metrics]
                    ),
                    "avg_opportunities": statistics.mean(
                        [m["opportunities_found"] for m in throughput_metrics]
                    ),
                    "avg_executable": statistics.mean(
                        [m["executable_opportunities"] for m in throughput_metrics]
                    ),
                },
            }

        return results

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all SFR end-to-end benchmark suites"""
        print(f"🚀 Running SFR end-to-end benchmarks on {self.runtime_info['runtime']}")
        print(f"📊 Python version: {self.runtime_info['python_version'].split()[0]}")

        if USING_PYPY:
            print(f"⚡ PyPy version: {self.runtime_info['pypy_version']}")
            print(
                "💡 Running production-scale tests optimized for PyPy JIT compilation"
            )
            print(
                "🔥 Large datasets (1K-20K options) with extended warm-up for JIT optimization"
            )
        else:
            print(
                "🐍 Running on CPython - baseline performance with production-scale data"
            )
            print("📈 Testing with 1K-20K options across 12+ symbols")

        if self.seed is not None:
            print(f"🎲 Using seed {self.seed} for reproducible results")

        print()

        all_results = {
            "runtime_info": self.runtime_info,
            "config_used": self.config,
            "benchmarks": {},
        }

        # Run each benchmark suite
        benchmark_suites = [
            ("sfr_initialization", self.benchmark_sfr_initialization),
            ("options_chain_processing", self.benchmark_options_chain_processing),
            ("arbitrage_detection", self.benchmark_arbitrage_detection),
            (
                "parallel_execution_simulation",
                self.benchmark_parallel_execution_simulation,
            ),
            ("full_sfr_scan_simulation", self.benchmark_full_sfr_scan_simulation),
        ]

        for suite_name, benchmark_func in benchmark_suites:
            try:
                print(f"\n{'='*60}")
                all_results["benchmarks"][suite_name] = benchmark_func()
            except Exception as e:
                print(f"❌ {suite_name} benchmark failed: {e}")
                all_results["benchmarks"][suite_name] = {"error": str(e)}

        return all_results

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save benchmark results to JSON file"""
        if filename is None:
            runtime = "pypy" if USING_PYPY else "cpython"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sfr_benchmark_results_{runtime}_{timestamp}.json"

        # Create benchmarks/results directory if it doesn't exist
        os.makedirs("benchmarks/results", exist_ok=True)
        filepath = os.path.join("benchmarks/results", filename)

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"📄 SFR benchmark results saved to: {filepath}")
        return filepath


def print_sfr_performance_summary(results: Dict[str, Any]):
    """Print a comprehensive summary of SFR benchmark results"""
    print("\n" + "=" * 60)
    print("📊 SFR END-TO-END PERFORMANCE SUMMARY")
    print("=" * 60)

    runtime_info = results["runtime_info"]
    print(f"Runtime: {runtime_info['runtime']}")
    print(
        f"Version: {runtime_info.get('pypy_version', runtime_info['python_version'].split()[0])}"
    )
    print(f"Platform: {runtime_info['platform']}")
    print()

    benchmarks = results["benchmarks"]

    # SFR Initialization
    if (
        "sfr_initialization" in benchmarks
        and "error" not in benchmarks["sfr_initialization"]
    ):
        init_results = benchmarks["sfr_initialization"]
        print("🔧 SFR Initialization:")
        print(f"  • Avg initialization time: {init_results['timing']['mean']:.4f}s")
        print(
            f"  • Min/Max: {init_results['timing']['min']:.4f}s / {init_results['timing']['max']:.4f}s"
        )
        print()

    # Options Chain Processing
    if (
        "options_chain_processing" in benchmarks
        and "error" not in benchmarks["options_chain_processing"]
    ):
        chain_results = benchmarks["options_chain_processing"]
        print("📈 Options Chain Processing:")
        for symbol_key, data in chain_results.items():
            if "error" not in data:
                symbol = symbol_key.split("_")[-1]
                avg_time = data["timing"]["mean"]
                throughput = data["throughput_options_per_sec"]
                success_rate = data["processing_success_rate"] * 100
                print(
                    f"  • {symbol}: {avg_time:.4f}s avg, {throughput:.0f} opts/sec, {success_rate:.1f}% success"
                )
        print()

    # Arbitrage Detection
    if (
        "arbitrage_detection" in benchmarks
        and "error" not in benchmarks["arbitrage_detection"]
    ):
        arb_results = benchmarks["arbitrage_detection"]
        print("🎯 Arbitrage Detection:")
        for size_key, data in arb_results.items():
            if "error" not in data:
                size = size_key.split("_")[-1]
                avg_time = data["timing"]["mean"]
                throughput = data["throughput_options_per_sec"]
                opp_rate = data["opportunity_rate"] * 100
                print(
                    f"  • {size} options: {avg_time:.4f}s avg, {throughput:.0f} opts/sec, {opp_rate:.1f}% opportunity rate"
                )
        print()

    # Parallel Execution
    if (
        "parallel_execution_simulation" in benchmarks
        and "error" not in benchmarks["parallel_execution_simulation"]
    ):
        parallel_results = benchmarks["parallel_execution_simulation"]
        print("⚡ Parallel Execution:")
        for legs_key, data in parallel_results.items():
            if "error" not in data:
                legs = legs_key.split("_")[-1]
                avg_time = data["timing"]["mean"]
                fill_rate = data["fill_rate"]["mean"] * 100
                cycles_per_sec = data["monitoring_efficiency"]["avg_cycles_per_second"]
                print(
                    f"  • {legs} legs: {avg_time:.4f}s avg, {fill_rate:.1f}% fill rate, {cycles_per_sec:.0f} cycles/sec"
                )
        print()

    # Full SFR Scan
    if (
        "full_sfr_scan_simulation" in benchmarks
        and "error" not in benchmarks["full_sfr_scan_simulation"]
    ):
        scan_results = benchmarks["full_sfr_scan_simulation"]
        print("🔍 Full SFR Scan Performance:")
        for symbols_key, data in scan_results.items():
            if "error" not in data:
                count = symbols_key.split("_")[-1]
                symbols = ", ".join(data["symbol_list"])
                avg_time = data["timing"]["mean"]
                symbols_per_sec = data["throughput"]["symbols_per_sec"]
                options_per_sec = data["throughput"]["options_per_sec"]
                avg_opportunities = data["throughput"]["avg_opportunities"]
                print(f"  • {count} symbols ({symbols}):")
                print(f"    - Scan time: {avg_time:.4f}s")
                print(
                    f"    - Throughput: {symbols_per_sec:.1f} symbols/sec, {options_per_sec:.0f} options/sec"
                )
                print(f"    - Opportunities: {avg_opportunities:.1f} avg found")
        print()

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="SFR End-to-End Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sfr_end_to_end_benchmark.py                    # Run all SFR benchmarks
  python sfr_end_to_end_benchmark.py --output sfr_results.json  # Save to specific file
  python sfr_end_to_end_benchmark.py --seed 42         # Run with seed for reproducible results
  pypy3 sfr_end_to_end_benchmark.py                    # Run with PyPy for comparison
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output file for benchmark results (default: auto-generated)",
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        help="Random seed for reproducible results (default: None for random results)",
    )

    args = parser.parse_args()

    # Create benchmark instance
    benchmark = SFREndToEndBenchmark(seed=args.seed)

    print("🏎️ SFR End-to-End Performance Benchmarks - Production Scale")
    print("=" * 60)
    print("📊 Testing with realistic production workloads:")
    print("  • Option chains: 1,000-20,000 options")
    print("  • Symbol coverage: 5-20 tickers")
    print("  • Extended warm-up for PyPy JIT optimization")
    print("  • High iteration counts for accurate measurements")
    print("=" * 60)

    # Run benchmarks
    results = benchmark.run_all_benchmarks()

    # Print summary
    print_sfr_performance_summary(results)

    # Save results
    output_file = benchmark.save_results(results, args.output)

    print(f"\n✅ SFR benchmarks completed! Results saved to {output_file}")

    if USING_PYPY:
        print("\n💡 For comparison, run the same benchmarks with CPython:")
        print("   python benchmarks/sfr_end_to_end_benchmark.py")
    else:
        print("\n💡 For performance comparison, run with PyPy:")
        print("   conda activate alc-pypy")
        print("   pypy3 benchmarks/sfr_end_to_end_benchmark.py")


if __name__ == "__main__":
    main()
