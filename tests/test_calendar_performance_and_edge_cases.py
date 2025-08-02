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
        """Test thread safety of calendar modules"""
        option_scan = OptionScan()
        results = []
        errors = []

        def run_calendar_finder(thread_id: int) -> None:
            try:
                with patch("commands.option.CalendarSpread") as mock_calendar_class:
                    mock_calendar = MagicMock()
                    mock_calendar.ib = MagicMock()
                    mock_calendar.scan = AsyncMock()
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
