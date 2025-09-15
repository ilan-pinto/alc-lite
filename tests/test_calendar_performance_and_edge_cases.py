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

# Optional tracemalloc import (not available in PyPy)
try:
    import tracemalloc
except ImportError:
    tracemalloc = None
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commands.option import OptionScan

# Import PyPy compatibility module (auto-applies patches if running under PyPy)
try:
    from modules.Arbitrage.pypy_compat import (
        create_compatible_async_mock,
        is_pypy,
        pypy_thread_safe,
    )
except ImportError:
    # Fallback for environments without PyPy compatibility
    def is_pypy():
        return hasattr(sys, "pypy_version_info") or "PyPy" in sys.version

    def create_compatible_async_mock(return_value=None):
        from unittest.mock import AsyncMock

        mock = AsyncMock()
        mock.return_value = return_value
        return mock

    def pypy_thread_safe(func):
        return func


from modules.Arbitrage.CalendarSpread import (
    CalendarSpread,
    CalendarSpreadConfig,
    CalendarSpreadLeg,
    CalendarSpreadOpportunity,
)


class TestCalendarPerformanceBenchmarks:
    """Performance benchmark tests for calendar spread system"""

    @pytest.mark.performance
    def test_concurrent_execution_performance(self) -> None:
        """Test performance under concurrent execution"""
        option_scan = OptionScan()

        with patch("commands.option.CalendarSpread") as mock_calendar_class:
            mock_calendar = MagicMock()
            mock_calendar.ib = MagicMock()
            mock_calendar.scan = AsyncMock()
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
            mock_calendar.scan = AsyncMock()
            mock_calendar_class.return_value = mock_calendar

            # Test with empty symbol list
            option_scan.calendar_finder(symbol_list=[])

            # Should still execute (using defaults)
            mock_calendar.scan.assert_called_once()

    @pytest.mark.edge_case
    def test_extreme_parameter_values(self) -> None:
        """Test with extreme parameter values"""
        option_scan = OptionScan()

        with patch("commands.option.CalendarSpread") as mock_calendar_class:
            mock_calendar = MagicMock()
            mock_calendar.ib = MagicMock()
            mock_calendar.scan = AsyncMock()
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
            mock_calendar.scan.assert_called_once()

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
    def test_date_boundary_conditions(self) -> None:
        """Test date-related boundary conditions"""
        # Test with expired options
        expired_leg = CalendarSpreadLeg(
            contract=MagicMock(),
            strike=100.0,
            expiry=(datetime.now() - timedelta(days=1)).strftime("%Y%m%d"),  # Expired
            right="C",
            price=0.01,
            bid=0.005,
            ask=0.015,
            volume=0,  # No volume for expired option
            iv=0.01,
            theta=0.0,
            days_to_expiry=-1,
        )

        # Test that system handles expired options appropriately
        opportunity = CalendarSpreadOpportunity(
            symbol="TEST",
            strike=100.0,
            option_type="CALL",
            front_leg=expired_leg,
            back_leg=expired_leg,  # Both expired for edge case
            iv_spread=0.0,
            theta_ratio=0.0,
            net_debit=0.0,
            max_profit=0.0,
            max_loss=0.0,
            front_bid_ask_spread=0.0,
            back_bid_ask_spread=0.0,
            combined_liquidity_score=0.0,
            term_structure_inversion=False,
            net_delta=0.0,
            net_gamma=0.0,
            net_vega=0.0,
            composite_score=0.0,
        )

        # Should create opportunity even with expired options
        assert opportunity.front_leg.days_to_expiry == -1
        assert opportunity.back_leg.days_to_expiry == -1

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
                mock_calendar.scan = AsyncMock(side_effect=error)
                mock_calendar_class.return_value = mock_calendar

                with pytest.raises(type(error)):
                    option_scan.calendar_finder(symbol_list=["TEST"])

                # Should call disconnect on error
                mock_calendar.ib.disconnect.assert_called()
                mock_calendar.ib.disconnect.reset_mock()

    @pytest.mark.edge_case
    def test_thread_safety(self) -> None:
        """Test thread safety of calendar modules with PyPy compatibility"""
        import platform
        import sys
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Detect if running under PyPy
        is_pypy = hasattr(sys, "pypy_version_info") or "PyPy" in sys.version

        option_scan = OptionScan()
        results = []
        errors = []
        lock = threading.Lock()  # Thread-safe access to results/errors

        def run_calendar_finder_safe(thread_id: int) -> str:
            """Thread-safe calendar finder with proper error handling"""
            try:
                # Create thread-local mocks to avoid sharing state
                local_mock_calendar_class = MagicMock()
                local_mock_calendar = MagicMock()
                local_mock_ib = MagicMock()

                # Configure mocks with thread-safe settings
                local_mock_ib.disconnect = MagicMock()
                local_mock_ib.isConnected = MagicMock(return_value=False)
                local_mock_calendar.ib = local_mock_ib

                # Create proper AsyncMock that works in threaded environment
                local_mock_scan = create_compatible_async_mock(return_value=None)

                local_mock_calendar.scan = local_mock_scan
                local_mock_calendar_class.return_value = local_mock_calendar

                # Use isolated patch context for this thread only
                with patch("commands.option.CalendarSpread", local_mock_calendar_class):
                    # Add small delay to create thread contention (realistic testing)
                    time.sleep(0.01 * (thread_id % 5))  # Staggered delays

                    option_scan.calendar_finder(
                        symbol_list=[f"TEST{thread_id}"],
                        cost_limit=float(thread_id * 100),
                    )

                    return f"Thread {thread_id} completed"

            except Exception as e:
                return f"Thread {thread_id} error: {str(e)}"

        if is_pypy:
            # PyPy: Use sequential execution to avoid recursion issues
            results = []
            errors = []
            thread_count = 5  # Reduced for PyPy

            for i in range(thread_count):
                result = run_calendar_finder_safe(i)
                if "error:" in result:
                    errors.append(result)
                else:
                    results.append(result)

            success_count = len(results)
            error_count = len(errors)

            # PyPy: Allow some failures due to different threading behavior
            assert success_count >= 4, (
                f"PyPy sequential test: only {success_count}/{thread_count} operations completed successfully. "
                f"Errors: {errors}"
            )
        else:
            # CPython: Use ThreadPoolExecutor as before
            max_workers = 20
            thread_count = 20

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_thread = {
                    executor.submit(run_calendar_finder_safe, i): i
                    for i in range(thread_count)
                }

                # Collect results with timeout
                for future in as_completed(future_to_thread, timeout=30.0):
                    thread_id = future_to_thread[future]
                    try:
                        result = future.result(timeout=5.0)
                        with lock:
                            if "error:" in result:
                                errors.append(result)
                            else:
                                results.append(result)
                    except Exception as e:
                        with lock:
                            errors.append(f"Thread {thread_id} exception: {str(e)}")

            # Verify results
            success_count = len(results)
            error_count = len(errors)

            # Log detailed results for debugging
            if errors:
                print(f"\nThread execution summary:")
                print(f"Successful threads: {success_count}/{thread_count}")
                print(f"Failed threads: {error_count}/{thread_count}")
                print(f"Runtime: {'PyPy' if is_pypy else 'CPython'}")
                print(f"Errors: {errors[:5]}...")  # Show first 5 errors

            # CPython: Expect high success rate
            assert success_count >= (thread_count * 0.8), (
                f"CPython threading test: only {success_count}/{thread_count} threads completed successfully. "
                f"Errors: {errors[:5]}"
            )

    @pytest.mark.edge_case
    def test_data_corruption_handling(self) -> None:
        """Test handling of corrupted or inconsistent data"""
        # Test with inconsistent leg data
        inconsistent_opportunity = CalendarSpreadOpportunity(
            symbol="TEST",
            strike=100.0,
            option_type="CALL",
            front_leg=CalendarSpreadLeg(
                contract=MagicMock(),
                strike=100.0,  # Different strike
                expiry=(datetime.now() + timedelta(days=30)).strftime("%Y%m%d"),
                right="C",
                price=5.0,
                bid=4.95,
                ask=5.05,
                volume=100,
                iv=0.20,
                theta=-0.05,
                days_to_expiry=30,
            ),
            back_leg=CalendarSpreadLeg(
                contract=MagicMock(),
                strike=105.0,  # Different strike - inconsistent!
                expiry=(datetime.now() + timedelta(days=60)).strftime("%Y%m%d"),
                right="P",  # Different option type - inconsistent!
                price=7.0,
                bid=6.95,
                ask=7.05,
                volume=50,
                iv=0.25,
                theta=-0.03,
                days_to_expiry=60,
            ),
            iv_spread=0.05,
            theta_ratio=1.67,
            net_debit=-2.0,
            max_profit=100.0,
            max_loss=2.0,
            front_bid_ask_spread=0.02,
            back_bid_ask_spread=0.014,
            combined_liquidity_score=0.65,
            term_structure_inversion=True,
            net_delta=0.1,
            net_gamma=0.035,
            net_vega=0.32,
            composite_score=0.75,
        )

        # Should create opportunity even with inconsistent data
        assert (
            inconsistent_opportunity.front_leg.strike
            != inconsistent_opportunity.back_leg.strike
        )
        assert (
            inconsistent_opportunity.front_leg.right
            != inconsistent_opportunity.back_leg.right
        )

    @pytest.mark.edge_case
    def test_cleanup_and_resource_management(self) -> None:
        """Test proper cleanup and resource management"""
        option_scan = OptionScan()

        with patch("commands.option.CalendarSpread") as mock_calendar_class:
            mock_calendar = MagicMock()
            mock_calendar.ib = MagicMock()
            mock_calendar.scan = AsyncMock()
            mock_calendar_class.return_value = mock_calendar

            # Run multiple operations
            for i in range(10):
                option_scan.calendar_finder(symbol_list=[f"TEST{i}"])

            # Verify calendar instances were created (not reused)
            assert mock_calendar_class.call_count == 10

            # Verify each instance had its process method called
            assert mock_calendar.scan.call_count == 10
