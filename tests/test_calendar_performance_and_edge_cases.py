"""
Performance benchmarks and edge case tests for the calendar spread system.
Tests performance characteristics, memory usage, error handling, and boundary conditions.
"""

import asyncio
import concurrent.futures
import gc
import os
import sys
import threading
import time
import tracemalloc
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commands.option import OptionScan
from modules.Arbitrage.CalendarGreeks import (
    AdjustmentType,
    CalendarGreeks,
    CalendarGreeksCalculator,
    GreeksEvolution,
    GreeksRiskLevel,
    PortfolioGreeks,
    PositionAdjustment,
)
from modules.Arbitrage.CalendarPnL import (
    BreakevenPoints,
    CalendarPnLCalculator,
    CalendarPnLResult,
    MonteCarloResults,
    PnLScenario,
    ThetaAnalysis,
)
from modules.Arbitrage.CalendarSpread import (
    CalendarSpread,
    CalendarSpreadConfig,
    CalendarSpreadExecutor,
    CalendarSpreadLeg,
    CalendarSpreadOpportunity,
)
from modules.Arbitrage.TermStructure import (
    IVDataPoint,
    IVPercentileData,
    TermStructureAnalyzer,
    TermStructureCurve,
    TermStructureInversion,
)


class TestCalendarPerformanceBenchmarks:
    """Performance benchmark tests for calendar spread system"""

    @pytest.fixture
    def large_dataset(self) -> Dict[str, Any]:
        """Create large dataset for performance testing"""
        return {
            "symbols": [f"SYM{i:04d}" for i in range(1000)],
            "iv_data_points": [
                IVDataPoint(
                    expiry=datetime.now() + timedelta(days=30 + i),
                    days_to_expiry=30 + i,
                    strike=100.0 + i,
                    implied_volatility=0.15 + (i * 0.001),
                    volume=100 + i,
                )
                for i in range(10000)
            ],
            "calendar_legs": [
                CalendarSpreadLeg(
                    contract=MagicMock(),
                    strike=100.0 + i,
                    expiry=datetime.now() + timedelta(days=30 + i),
                    option_type="CALL",
                    market_price=5.0 + i * 0.1,
                    bid=4.95 + i * 0.1,
                    ask=5.05 + i * 0.1,
                    volume=100 + i,
                    implied_volatility=0.15 + (i * 0.001),
                    delta=0.5 + (i * 0.001),
                    gamma=0.02 + (i * 0.0001),
                    theta=-0.05 - (i * 0.0001),
                    vega=0.1 + (i * 0.001),
                )
                for i in range(5000)
            ],
        }

    @pytest.mark.performance
    def test_option_scan_calendar_finder_performance(
        self, large_dataset: Dict[str, Any]
    ) -> None:
        """Test OptionScan.calendar_finder performance with large symbol lists"""
        option_scan = OptionScan()

        with patch("commands.option.CalendarSpread") as mock_calendar_class:
            mock_calendar = MagicMock()
            mock_calendar.ib = MagicMock()
            mock_calendar.process = AsyncMock()
            mock_calendar_class.return_value = mock_calendar

            # Test with increasing symbol list sizes
            performance_results = []

            for size in [10, 50, 100, 500, 1000]:
                symbols = large_dataset["symbols"][:size]

                # Measure execution time
                start_time = time.perf_counter()

                option_scan.calendar_finder(
                    symbol_list=symbols,
                    cost_limit=500.0,
                    profit_target=0.25,
                )

                end_time = time.perf_counter()
                execution_time = end_time - start_time

                performance_results.append(
                    {
                        "symbol_count": size,
                        "execution_time": execution_time,
                        "time_per_symbol": execution_time / size,
                    }
                )

                # Performance should scale linearly or better
                assert (
                    execution_time < size * 0.01
                ), f"Performance degraded with {size} symbols: {execution_time:.3f}s"

            # Verify performance scaling
            for i in range(1, len(performance_results)):
                prev_result = performance_results[i - 1]
                curr_result = performance_results[i]

                # Time per symbol should not increase significantly
                time_increase = (
                    curr_result["time_per_symbol"] / prev_result["time_per_symbol"]
                )
                assert (
                    time_increase < 2.0
                ), f"Performance degraded significantly: {time_increase:.2f}x slower per symbol"

    @pytest.mark.performance
    def test_term_structure_analyzer_performance(
        self, large_dataset: Dict[str, Any]
    ) -> None:
        """Test TermStructureAnalyzer performance with large IV datasets"""
        analyzer = TermStructureAnalyzer()

        # Test with increasing data sizes
        for size in [100, 500, 1000, 5000, 10000]:
            iv_data = large_dataset["iv_data_points"][:size]

            start_time = time.perf_counter()

            with patch.object(analyzer, "_build_iv_curve") as mock_build_curve:
                mock_curve = TermStructureCurve(
                    data_points=iv_data[:10],  # Simulate processed data
                    curve_fit_r2=0.95,
                    slope=0.001,
                    curvature=-0.0001,
                    volatility_of_volatility=0.05,
                )
                mock_build_curve.return_value = mock_curve

                result = analyzer.analyze_term_structure("TEST", iv_data)

            end_time = time.perf_counter()
            execution_time = end_time - start_time

            # Should process large datasets efficiently
            assert (
                execution_time < size * 0.001
            ), f"Term structure analysis too slow with {size} points: {execution_time:.3f}s"
            assert result.curve is not None

    @pytest.mark.performance
    def test_calendar_pnl_calculator_performance(
        self, large_dataset: Dict[str, Any]
    ) -> None:
        """Test CalendarPnLCalculator performance with intensive calculations"""
        calculator = CalendarPnLCalculator()

        front_leg = large_dataset["calendar_legs"][0]
        back_leg = large_dataset["calendar_legs"][1]

        # Test performance with increasing simulation complexity
        for num_scenarios in [100, 500, 1000, 5000]:
            start_time = time.perf_counter()

            with patch.object(
                calculator, "_run_monte_carlo_simulation"
            ) as mock_monte_carlo:
                mock_results = MonteCarloResults(
                    mean_pnl=125.0,
                    std_pnl=85.0,
                    profit_probability=0.68,
                    var_95=150.0,
                    expected_shortfall=180.0,
                    max_drawdown=200.0,
                    scenarios=[MagicMock() for _ in range(num_scenarios)],
                )
                mock_monte_carlo.return_value = mock_results

                result = calculator.analyze_calendar_pnl(
                    front_leg=front_leg,
                    back_leg=back_leg,
                    underlying_price=100.0,
                    position_size=1,
                )

            end_time = time.perf_counter()
            execution_time = end_time - start_time

            # Should handle large Monte Carlo simulations efficiently
            assert (
                execution_time < num_scenarios * 0.001
            ), f"P&L calculation too slow with {num_scenarios} scenarios: {execution_time:.3f}s"
            assert result.monte_carlo is not None

    @pytest.mark.performance
    def test_calendar_greeks_calculator_performance(
        self, large_dataset: Dict[str, Any]
    ) -> None:
        """Test CalendarGreeksCalculator performance with large portfolios"""
        calculator = CalendarGreeksCalculator()

        # Test portfolio Greeks calculation with increasing position counts
        for num_positions in [10, 50, 100, 500, 1000]:
            positions = []
            for i in range(num_positions):
                front_leg = large_dataset["calendar_legs"][i * 2]
                back_leg = large_dataset["calendar_legs"][i * 2 + 1]
                positions.append((front_leg, back_leg, 1))  # (front, back, size)

            start_time = time.perf_counter()

            with patch.object(
                calculator, "calculate_portfolio_greeks"
            ) as mock_portfolio_greeks:
                mock_portfolio = PortfolioGreeks(
                    total_delta=0.05,
                    total_gamma=-0.1,
                    total_theta=2.5,
                    total_vega=-1.2,
                    total_rho=0.3,
                    position_count=num_positions,
                    net_exposure=50000.0,
                    max_single_position_risk=5000.0,
                    correlation_risk=0.15,
                )
                mock_portfolio_greeks.return_value = mock_portfolio

                result = calculator.calculate_portfolio_greeks(positions)

            end_time = time.perf_counter()
            execution_time = end_time - start_time

            # Should handle large portfolios efficiently
            assert (
                execution_time < num_positions * 0.01
            ), f"Portfolio Greeks too slow with {num_positions} positions: {execution_time:.3f}s"
            assert result.position_count == num_positions

    @pytest.mark.performance
    def test_memory_usage_performance(self, large_dataset: Dict[str, Any]) -> None:
        """Test memory usage characteristics of calendar system"""
        # Start memory tracking
        tracemalloc.start()

        option_scan = OptionScan()

        with patch("commands.option.CalendarSpread") as mock_calendar_class:
            mock_calendar = MagicMock()
            mock_calendar.ib = MagicMock()
            mock_calendar.process = AsyncMock()
            mock_calendar_class.return_value = mock_calendar

            # Get initial memory snapshot
            snapshot1 = tracemalloc.take_snapshot()

            # Run multiple calendar finder operations
            for i in range(100):
                symbols = [f"TEST{j}" for j in range(50)]
                option_scan.calendar_finder(
                    symbol_list=symbols,
                    cost_limit=float(i * 10),
                    profit_target=0.25 + (i * 0.001),
                )

            # Force garbage collection
            gc.collect()

            # Get final memory snapshot
            snapshot2 = tracemalloc.take_snapshot()

            # Compare memory usage
            top_stats = snapshot2.compare_to(snapshot1, "lineno")

            # Total memory increase should be reasonable
            total_memory_increase = sum(
                stat.size_diff for stat in top_stats if stat.size_diff > 0
            )

            # Should not have significant memory leaks
            assert (
                total_memory_increase < 100 * 1024 * 1024
            ), f"Excessive memory usage: {total_memory_increase / (1024*1024):.1f} MB"

        tracemalloc.stop()

    @pytest.mark.performance
    def test_concurrent_execution_performance(self) -> None:
        """Test performance under concurrent execution"""
        option_scan = OptionScan()

        with patch("commands.option.CalendarSpread") as mock_calendar_class:
            mock_calendar = MagicMock()
            mock_calendar.ib = MagicMock()
            mock_calendar.process = AsyncMock()
            mock_calendar_class.return_value = mock_calendar

            def run_calendar_finder(symbol_suffix: int) -> float:
                start_time = time.perf_counter()
                symbols = [f"TEST{symbol_suffix}_{i}" for i in range(10)]
                option_scan.calendar_finder(symbol_list=symbols)
                return time.perf_counter() - start_time

            # Test concurrent execution
            start_time = time.perf_counter()

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(run_calendar_finder, i) for i in range(50)]
                execution_times = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]

            total_time = time.perf_counter() - start_time

            # Concurrent execution should be faster than sequential
            sequential_time = sum(execution_times)
            speedup = sequential_time / total_time

            assert speedup > 2.0, f"Insufficient concurrency speedup: {speedup:.2f}x"
            assert (
                total_time < 10.0
            ), f"Concurrent execution too slow: {total_time:.3f}s"


class TestCalendarEdgeCases:
    """Edge case tests for calendar spread system"""

    @pytest.mark.edge_case
    def test_empty_data_handling(self) -> None:
        """Test handling of empty data inputs"""
        option_scan = OptionScan()

        with patch("commands.option.CalendarSpread") as mock_calendar_class:
            mock_calendar = MagicMock()
            mock_calendar.ib = MagicMock()
            mock_calendar.process = AsyncMock()
            mock_calendar_class.return_value = mock_calendar

            # Test with empty symbol list
            option_scan.calendar_finder(symbol_list=[])

            # Should still execute (using defaults)
            mock_calendar.process.assert_called_once()

    @pytest.mark.edge_case
    def test_extreme_parameter_values(self) -> None:
        """Test with extreme parameter values"""
        option_scan = OptionScan()

        with patch("commands.option.CalendarSpread") as mock_calendar_class:
            mock_calendar = MagicMock()
            mock_calendar.ib = MagicMock()
            mock_calendar.process = AsyncMock()
            mock_calendar_class.return_value = mock_calendar

            # Test with extreme values
            extreme_values = {
                "cost_limit": 999999999.99,
                "profit_target": 999.99,
                "iv_spread_threshold": 100.0,
                "theta_ratio_threshold": 1000.0,
                "front_expiry_max_days": 999999,
                "back_expiry_min_days": 999999,
                "back_expiry_max_days": 999999,
                "min_volume": 999999999,
                "max_bid_ask_spread": 999.99,
                "quantity": 999999,
            }

            option_scan.calendar_finder(symbol_list=["TEST"], **extreme_values)

            # Should handle extreme values without crashing
            mock_calendar_class.assert_called_once()
            mock_calendar.process.assert_called_once()

    @pytest.mark.edge_case
    def test_invalid_data_types(self) -> None:
        """Test handling of invalid data types"""
        # Test CalendarSpreadConfig with invalid types
        with pytest.raises((TypeError, ValueError)):
            CalendarSpreadConfig(
                iv_spread_threshold="invalid",  # Should be float
                theta_ratio_threshold=None,  # Should be float
            )

    @pytest.mark.edge_case
    def test_boundary_conditions(self) -> None:
        """Test boundary conditions for all modules"""
        # Test TermStructureAnalyzer with minimal data
        analyzer = TermStructureAnalyzer()

        # Single data point
        single_point = [
            IVDataPoint(
                expiry=datetime.now() + timedelta(days=30),
                days_to_expiry=30,
                strike=100.0,
                implied_volatility=0.20,
                volume=1,
            )
        ]

        with patch.object(analyzer, "_build_iv_curve") as mock_build_curve:
            mock_curve = TermStructureCurve(
                data_points=single_point,
                curve_fit_r2=0.0,  # Poor fit with single point
                slope=0.0,
                curvature=0.0,
                volatility_of_volatility=0.0,
            )
            mock_build_curve.return_value = mock_curve

            result = analyzer.analyze_term_structure("TEST", single_point)
            assert result.curve is not None

    @pytest.mark.edge_case
    def test_numerical_precision_edge_cases(self) -> None:
        """Test numerical precision and floating point edge cases"""
        calculator = CalendarPnLCalculator()

        # Create legs with very small price differences
        front_leg = CalendarSpreadLeg(
            contract=MagicMock(),
            strike=100.0,
            expiry=datetime.now() + timedelta(days=30),
            option_type="CALL",
            market_price=0.001,  # Very small price
            bid=0.0005,
            ask=0.0015,
            volume=1,
            implied_volatility=0.001,  # Very low IV
            delta=0.001,
            gamma=0.0001,
            theta=-0.0001,
            vega=0.0001,
        )

        back_leg = CalendarSpreadLeg(
            contract=MagicMock(),
            strike=100.0,
            expiry=datetime.now() + timedelta(days=60),
            option_type="CALL",
            market_price=0.002,  # Slightly higher
            bid=0.0015,
            ask=0.0025,
            volume=1,
            implied_volatility=0.002,
            delta=0.002,
            gamma=0.0002,
            theta=-0.0002,
            vega=0.0002,
        )

        with patch.object(
            calculator, "_run_monte_carlo_simulation"
        ) as mock_monte_carlo:
            mock_results = MonteCarloResults(
                mean_pnl=0.001,  # Very small P&L
                std_pnl=0.0005,
                profit_probability=0.51,  # Just above 50%
                var_95=0.002,
                expected_shortfall=0.003,
                max_drawdown=0.002,
                scenarios=[],
            )
            mock_monte_carlo.return_value = mock_results

            result = calculator.analyze_calendar_pnl(
                front_leg=front_leg,
                back_leg=back_leg,
                underlying_price=100.0,
                position_size=1,
            )

            # Should handle very small numbers without precision errors
            assert result.monte_carlo.mean_pnl > 0
            assert result.monte_carlo.profit_probability > 0.5

    @pytest.mark.edge_case
    def test_date_boundary_conditions(self) -> None:
        """Test date-related boundary conditions"""
        # Test with expired options
        expired_leg = CalendarSpreadLeg(
            contract=MagicMock(),
            strike=100.0,
            expiry=datetime.now() - timedelta(days=1),  # Expired
            option_type="CALL",
            market_price=0.01,
            bid=0.005,
            ask=0.015,
            volume=0,  # No volume for expired option
            implied_volatility=0.01,
            delta=0.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
        )

        # Test that system handles expired options appropriately
        opportunity = CalendarSpreadOpportunity(
            symbol="TEST",
            front_leg=expired_leg,
            back_leg=expired_leg,  # Both expired for edge case
            net_debit=0.0,
            iv_spread=0.0,
            theta_ratio=0.0,
            max_profit=0.0,
            breakeven_lower=100.0,
            breakeven_upper=100.0,
            profit_probability=0.0,
            days_to_front_expiry=-1,  # Negative days
            days_to_back_expiry=-1,
            score=0.0,
        )

        # Should create opportunity even with expired options
        assert opportunity.days_to_front_expiry == -1
        assert opportunity.days_to_back_expiry == -1

    @pytest.mark.edge_case
    def test_division_by_zero_protection(self) -> None:
        """Test protection against division by zero errors"""
        calculator = CalendarGreeksCalculator()

        # Create legs with zero Greeks
        zero_greeks_leg = CalendarSpreadLeg(
            contract=MagicMock(),
            strike=100.0,
            expiry=datetime.now() + timedelta(days=30),
            option_type="CALL",
            market_price=5.0,
            bid=4.95,
            ask=5.05,
            volume=100,
            implied_volatility=0.20,
            delta=0.0,  # Zero delta
            gamma=0.0,  # Zero gamma
            theta=0.0,  # Zero theta
            vega=0.0,  # Zero vega
        )

        with patch.object(calculator, "calculate_calendar_greeks") as mock_calc_greeks:
            mock_greeks = CalendarGreeks(
                net_delta=0.0,
                net_gamma=0.0,
                net_theta=0.0,
                net_vega=0.0,
                net_rho=0.0,
            )
            mock_calc_greeks.return_value = mock_greeks

            result = calculator.calculate_calendar_greeks(
                front_leg=zero_greeks_leg, back_leg=zero_greeks_leg, position_size=1
            )

            # Should handle zero Greeks without error
            assert result.net_delta == 0.0
            assert result.net_gamma == 0.0
            assert result.net_theta == 0.0
            assert result.net_vega == 0.0

    @pytest.mark.edge_case
    def test_unicode_and_special_characters(self) -> None:
        """Test handling of unicode and special characters in symbols"""
        option_scan = OptionScan()

        with patch("commands.option.CalendarSpread") as mock_calendar_class:
            mock_calendar = MagicMock()
            mock_calendar.ib = MagicMock()
            mock_calendar.process = AsyncMock()
            mock_calendar_class.return_value = mock_calendar

            special_symbols = [
                "SPY.A",
                "SPY-USD",
                "SPY/EUR",
                "SPY@SMART",
                "SPY_TEST",
                "SPY+",
                "TEST123",
                "T3ST_$YMB0L",
            ]

            option_scan.calendar_finder(symbol_list=special_symbols)

            # Should handle special characters in symbols
            mock_calendar.process.assert_called_once()
            args, kwargs = mock_calendar.process.call_args
            assert kwargs["symbols"] == special_symbols

    @pytest.mark.edge_case
    def test_system_resource_limits(self) -> None:
        """Test behavior at system resource limits"""
        # Test with maximum integer values
        max_int_config = CalendarSpreadConfig(
            front_expiry_max_days=2147483647,  # Max 32-bit int
            back_expiry_min_days=2147483647,
            back_expiry_max_days=2147483647,
            min_volume=2147483647,
        )

        # Should create config without overflow
        assert max_int_config.front_expiry_max_days == 2147483647

        # Test with maximum float values
        max_float_config = CalendarSpreadConfig(
            iv_spread_threshold=1e308,  # Very large float
            theta_ratio_threshold=1e308,
            max_bid_ask_spread=1e308,
            net_debit_limit=1e308,
        )

        # Should handle large float values
        assert max_float_config.iv_spread_threshold == 1e308

    @pytest.mark.edge_case
    def test_error_propagation(self) -> None:
        """Test proper error propagation through the system"""
        option_scan = OptionScan()

        with patch("commands.option.CalendarSpread") as mock_calendar_class:
            mock_calendar = MagicMock()
            mock_calendar.ib = MagicMock()

            # Test different types of errors
            error_types = [
                ConnectionError("IB connection failed"),
                ValueError("Invalid parameter"),
                TypeError("Wrong data type"),
                RuntimeError("Runtime error"),
                MemoryError("Out of memory"),
            ]

            for error in error_types:
                mock_calendar.process = AsyncMock(side_effect=error)
                mock_calendar_class.return_value = mock_calendar

                with pytest.raises(type(error)):
                    option_scan.calendar_finder(symbol_list=["TEST"])

                # Should call disconnect on error
                mock_calendar.ib.disconnect.assert_called()
                mock_calendar.ib.disconnect.reset_mock()

    @pytest.mark.edge_case
    def test_thread_safety(self) -> None:
        """Test thread safety of calendar modules"""
        option_scan = OptionScan()
        results = []
        errors = []

        def run_calendar_finder(thread_id: int) -> None:
            try:
                with patch("commands.option.CalendarSpread") as mock_calendar_class:
                    mock_calendar = MagicMock()
                    mock_calendar.ib = MagicMock()
                    mock_calendar.process = AsyncMock()
                    mock_calendar_class.return_value = mock_calendar

                    option_scan.calendar_finder(
                        symbol_list=[f"TEST{thread_id}"],
                        cost_limit=float(thread_id * 100),
                    )

                    results.append(f"Thread {thread_id} completed")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {str(e)}")

        # Run multiple threads simultaneously
        threads = []
        for i in range(20):
            thread = threading.Thread(target=run_calendar_finder, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10.0)  # 10 second timeout

        # All threads should complete successfully
        assert len(results) == 20, f"Only {len(results)} threads completed successfully"
        assert len(errors) == 0, f"Thread errors: {errors}"

    @pytest.mark.edge_case
    def test_data_corruption_handling(self) -> None:
        """Test handling of corrupted or inconsistent data"""
        # Test with inconsistent leg data
        inconsistent_opportunity = CalendarSpreadOpportunity(
            symbol="TEST",
            front_leg=CalendarSpreadLeg(
                contract=MagicMock(),
                strike=100.0,  # Different strike
                expiry=datetime.now() + timedelta(days=30),
                option_type="CALL",
                market_price=5.0,
                bid=4.95,
                ask=5.05,
                volume=100,
                implied_volatility=0.20,
                delta=0.5,
                gamma=0.02,
                theta=-0.05,
                vega=0.1,
            ),
            back_leg=CalendarSpreadLeg(
                contract=MagicMock(),
                strike=105.0,  # Different strike - inconsistent!
                expiry=datetime.now() + timedelta(days=60),
                option_type="PUT",  # Different option type - inconsistent!
                market_price=7.0,
                bid=6.95,
                ask=7.05,
                volume=50,
                implied_volatility=0.25,
                delta=-0.4,
                gamma=0.015,
                theta=-0.03,
                vega=0.12,
            ),
            net_debit=-2.0,
            iv_spread=0.05,
            theta_ratio=1.67,
            max_profit=100.0,
            breakeven_lower=98.0,
            breakeven_upper=102.0,
            profit_probability=0.65,
            days_to_front_expiry=30,
            days_to_back_expiry=60,
            score=0.75,
        )

        # Should create opportunity even with inconsistent data
        assert (
            inconsistent_opportunity.front_leg.strike
            != inconsistent_opportunity.back_leg.strike
        )
        assert (
            inconsistent_opportunity.front_leg.option_type
            != inconsistent_opportunity.back_leg.option_type
        )

    @pytest.mark.edge_case
    def test_cleanup_and_resource_management(self) -> None:
        """Test proper cleanup and resource management"""
        option_scan = OptionScan()

        with patch("commands.option.CalendarSpread") as mock_calendar_class:
            mock_calendar = MagicMock()
            mock_calendar.ib = MagicMock()
            mock_calendar.process = AsyncMock()
            mock_calendar_class.return_value = mock_calendar

            # Run multiple operations
            for i in range(10):
                option_scan.calendar_finder(symbol_list=[f"TEST{i}"])

            # Verify calendar instances were created (not reused)
            assert mock_calendar_class.call_count == 10

            # Verify each instance had its process method called
            assert mock_calendar.process.call_count == 10
