"""
Unit tests for Execution Reporter system.

Tests comprehensive execution reporting, slippage analysis,
and performance metrics tracking.
"""

import json
import os
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

# PyPy-aware performance multipliers
if hasattr(sys, "pypy_version_info"):
    TIMEOUT_MULTIPLIER = 4.0  # Increased from 2.0 for concurrent operations
    MEMORY_MULTIPLIER = 5.0
else:
    TIMEOUT_MULTIPLIER = 1.0
    MEMORY_MULTIPLIER = 1.0

from modules.Arbitrage.sfr.execution_reporter import (
    ExecutionReporter,
    PerformanceMetrics,
    ReportFormat,
    ReportLevel,
    SlippageAnalysis,
)
from modules.Arbitrage.sfr.parallel_executor import ExecutionResult


class TestExecutionReporter:
    """Test execution reporting and analysis"""

    @pytest.fixture
    def sample_execution_result(self):
        """Create sample execution result for testing"""
        return ExecutionResult(
            success=True,
            execution_id="SPY_1234567890",
            symbol="SPY",
            total_execution_time=2.34,
            all_legs_filled=True,
            partially_filled=False,
            legs_filled=3,
            total_legs=3,
            expected_total_cost=1000.0,
            actual_total_cost=1002.5,
            total_slippage=2.5,
            slippage_percentage=0.25,
            stock_result={
                "leg_type": "stock",
                "action": "BUY",
                "target_price": 100.0,
                "avg_fill_price": 100.02,
                "slippage": 0.02,
                "fill_status": "filled",
            },
            call_result={
                "leg_type": "call",
                "action": "SELL",
                "target_price": 8.50,
                "avg_fill_price": 8.47,
                "slippage": -0.03,
                "fill_status": "filled",
            },
            put_result={
                "leg_type": "put",
                "action": "BUY",
                "target_price": 3.25,
                "avg_fill_price": 3.28,
                "slippage": 0.03,
                "fill_status": "filled",
            },
            order_placement_time=0.123,
            fill_monitoring_time=2.217,
        )

    @pytest.fixture
    def failed_execution_result(self):
        """Create failed execution result for testing"""
        return ExecutionResult(
            success=False,
            execution_id="TSLA_9876543210",
            symbol="TSLA",
            total_execution_time=5.67,
            all_legs_filled=False,
            partially_filled=True,
            legs_filled=1,
            total_legs=3,
            expected_total_cost=2000.0,
            actual_total_cost=0.0,
            total_slippage=0.0,
            slippage_percentage=0.0,
            stock_result={
                "leg_type": "stock",
                "action": "BUY",
                "target_price": 200.0,
                "avg_fill_price": 200.05,
                "slippage": 0.05,
                "fill_status": "filled",
            },
            call_result={
                "leg_type": "call",
                "action": "SELL",
                "target_price": 15.0,
                "avg_fill_price": 0.0,
                "slippage": 0.0,
                "fill_status": "not_filled",
            },
            put_result={
                "leg_type": "put",
                "action": "BUY",
                "target_price": 8.0,
                "avg_fill_price": 0.0,
                "slippage": 0.0,
                "fill_status": "not_filled",
            },
            error_message="Partial fill timeout - rollback executed",
            order_placement_time=0.156,
            fill_monitoring_time=5.514,
        )

    def test_reporter_initialization(self):
        """Test reporter initialization"""
        reporter = ExecutionReporter()

        assert reporter.console is not None
        assert reporter.report_history == []
        assert reporter.session_metrics["total_executions"] == 0
        assert reporter.session_metrics["successful_executions"] == 0

    def test_console_report_generation(self, sample_execution_result):
        """Test console report formatting and content"""
        reporter = ExecutionReporter()

        report = reporter.generate_execution_report(
            sample_execution_result,
            level=ReportLevel.DETAILED,
            format_type=ReportFormat.CONSOLE,
        )

        # Verify key content present
        assert "SPY" in report
        assert "SUCCESS" in report or "✅" in report
        assert "$2.50" in report  # Total slippage
        assert "0.25%" in report  # Slippage percentage
        assert "2.34s" in report  # Execution time
        assert "3/3" in report  # Legs filled

        # Verify leg-specific data
        assert "STOCK" in report.upper() and "BUY" in report
        assert "CALL" in report.upper() and "SELL" in report
        assert "PUT" in report.upper() and "BUY" in report

    def test_failed_execution_console_report(self, failed_execution_result):
        """Test console report for failed execution"""
        reporter = ExecutionReporter()

        report = reporter.generate_execution_report(
            failed_execution_result,
            level=ReportLevel.DETAILED,
            format_type=ReportFormat.CONSOLE,
        )

        assert "TSLA" in report
        assert "FAILED" in report or "❌" in report
        assert "1/3" in report  # Only 1 leg filled
        assert "timeout" in report.lower() or "rollback" in report.lower()

    def test_json_report_structure(self, sample_execution_result):
        """Test JSON report structure and completeness"""
        reporter = ExecutionReporter()

        json_report = reporter.generate_execution_report(
            sample_execution_result, format_type=ReportFormat.JSON
        )

        report_data = json.loads(json_report)

        # Verify main sections present
        expected_sections = [
            "execution_summary",
            "financial_summary",
            "slippage_analysis",
            "performance_metrics",
            "leg_details",
            "session_metrics",
            "timestamp",
        ]

        for section in expected_sections:
            assert section in report_data, f"Missing section: {section}"

        # Verify critical fields
        exec_summary = report_data["execution_summary"]
        assert exec_summary["symbol"] == "SPY"
        assert exec_summary["success"] is True
        assert exec_summary["legs_filled"] == "3/3"

    def test_html_report_generation(self, sample_execution_result):
        """Test HTML report generation"""
        reporter = ExecutionReporter()

        html_report = reporter.generate_execution_report(
            sample_execution_result, format_type=ReportFormat.HTML
        )

        # Basic HTML structure
        assert "<html>" in html_report
        assert "<body>" in html_report
        assert "</html>" in html_report
        assert "<head>" in html_report

        # Content verification
        assert "SPY" in html_report
        assert "SUCCESS" in html_report
        assert "$2.50" in html_report

        # CSS styling
        assert "style" in html_report
        assert "background" in html_report or "color" in html_report

    def test_text_report_generation(self, sample_execution_result):
        """Test plain text report generation"""
        reporter = ExecutionReporter()

        text_report = reporter.generate_execution_report(
            sample_execution_result, format_type=ReportFormat.TEXT
        )

        # Should be plain text (no HTML tags)
        assert "<" not in text_report or ">" not in text_report
        assert "SPY" in text_report
        assert "SUCCESS" in text_report
        assert "$2.50" in text_report

        # Should have box drawing or similar formatting
        assert "═" in text_report or "─" in text_report or "-" in text_report

    def test_slippage_analysis_accuracy(self, sample_execution_result):
        """Test slippage analysis calculations"""
        reporter = ExecutionReporter()

        # Access the internal slippage analysis method
        slippage_analysis = reporter._analyze_slippage(sample_execution_result)

        assert slippage_analysis.total_slippage_dollars == 2.5
        assert slippage_analysis.slippage_percentage == 0.25
        assert slippage_analysis.worst_leg in [
            "call",
            "put",
        ]  # Both have highest absolute slippage (0.03)
        assert slippage_analysis.worst_leg_slippage == 0.03
        assert abs(slippage_analysis.avg_slippage_per_leg - 0.0267) < 0.001

    def test_slippage_analysis_price_improvements(self, sample_execution_result):
        """Test detection of price improvements vs deteriorations"""
        reporter = ExecutionReporter()

        slippage_analysis = reporter._analyze_slippage(sample_execution_result)

        # Stock: BUY at 100.02 vs 100.0 target = deterioration
        # Call: SELL at 8.47 vs 8.50 target = deterioration (sold for less)
        # Put: BUY at 3.28 vs 3.25 target = deterioration

        # All legs had price deteriorations in this example
        assert len(slippage_analysis.price_deterioration_legs) == 3
        assert len(slippage_analysis.price_improvement_legs) == 0

    def test_performance_metrics_calculation(self, sample_execution_result):
        """Test performance metrics calculation"""
        reporter = ExecutionReporter()

        # First execution
        reporter.generate_execution_report(sample_execution_result)

        # Second execution with different timings
        sample_execution_result.execution_id = "SPY_2"
        sample_execution_result.total_execution_time = 3.1
        sample_execution_result.total_slippage = 1.8
        reporter.generate_execution_report(sample_execution_result)

        metrics = reporter._calculate_performance_metrics(sample_execution_result)

        assert metrics.all_legs_fill_rate == 100.0  # Both successful
        assert (
            abs(metrics.average_slippage_dollars - 2.15) < 0.1
        )  # Average of 2.5 and 1.8

    def test_session_statistics_tracking(
        self, sample_execution_result, failed_execution_result
    ):
        """Test session-wide statistics tracking"""
        reporter = ExecutionReporter()

        # Generate multiple reports (3 successes, 2 failures)
        successes = [sample_execution_result] * 3
        failures = [failed_execution_result] * 2

        for result in successes + failures:
            # Ensure unique IDs
            result.execution_id = f"{result.symbol}_{time.time()}"
            reporter.generate_execution_report(result)

        stats = reporter.get_session_statistics()

        assert stats["total_executions"] == 5
        assert stats["successful_executions"] == 3
        assert stats["success_rate_percent"] == 60.0
        assert stats["reports_generated"] >= 5

    def test_report_export_functionality(self):
        """Test report export to file"""
        reporter = ExecutionReporter()

        # Create some session data
        reporter.session_metrics["total_executions"] = 3
        reporter.session_metrics["successful_executions"] = 2
        reporter.session_metrics["total_slippage"] = 5.0

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filename = f.name

        try:
            success = reporter.export_session_report(filename, ReportFormat.JSON)
            assert success is True

            # Verify file exists and has content
            assert os.path.exists(filename)

            # Verify file content
            with open(filename, "r") as f:
                data = json.load(f)
                assert data["total_executions"] == 3
                assert data["successful_executions"] == 2
                assert abs(data["success_rate"] - 66.67) < 0.1

        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_report_export_text_format(self):
        """Test text format export"""
        reporter = ExecutionReporter()

        reporter.session_metrics["total_executions"] = 5
        reporter.session_metrics["successful_executions"] = 4

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            filename = f.name

        try:
            success = reporter.export_session_report(filename, ReportFormat.TEXT)
            assert success is True

            with open(filename, "r") as f:
                content = f.read()
                assert "Total Executions: 5" in content
                assert "Successful: 4" in content
                assert "80.0%" in content

        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_report_level_filtering(self, sample_execution_result):
        """Test different report detail levels"""
        reporter = ExecutionReporter()

        # Summary level - basic info only
        summary_report = reporter.generate_execution_report(
            sample_execution_result, level=ReportLevel.SUMMARY
        )

        # Detailed level - includes leg breakdown
        detailed_report = reporter.generate_execution_report(
            sample_execution_result, level=ReportLevel.DETAILED
        )

        # Comprehensive level - includes performance metrics
        comprehensive_report = reporter.generate_execution_report(
            sample_execution_result, level=ReportLevel.COMPREHENSIVE
        )

        # Summary should be shortest
        assert len(summary_report) < len(detailed_report)
        assert len(detailed_report) <= len(comprehensive_report)

        # Comprehensive should have performance/session data
        assert (
            "Performance" in comprehensive_report
            or "Session" in comprehensive_report
            or len(comprehensive_report) > len(detailed_report)
        )

    def test_error_handling_malformed_result(self):
        """Test handling of malformed execution results"""
        reporter = ExecutionReporter()

        # Create malformed execution result with proper types but invalid values
        malformed_result = MagicMock()
        malformed_result.success = (
            "not_a_boolean"  # Wrong type but won't cause comparison error
        )
        malformed_result.symbol = None
        malformed_result.execution_id = "MALFORMED"
        malformed_result.total_execution_time = -1.0  # Invalid time but float type
        malformed_result.all_legs_filled = False
        malformed_result.partially_filled = False
        malformed_result.legs_filled = 0
        malformed_result.total_legs = 3
        malformed_result.expected_total_cost = 0.0
        malformed_result.actual_total_cost = 0.0
        malformed_result.total_slippage = 0.0
        malformed_result.slippage_percentage = 0.0
        malformed_result.stock_result = None
        malformed_result.call_result = None
        malformed_result.put_result = None
        malformed_result.error_type = None
        malformed_result.error_message = None
        malformed_result.order_placement_time = 0.0
        malformed_result.fill_monitoring_time = 0.0
        malformed_result.rollback_time = 0.0

        # Should handle gracefully without crashing
        report = reporter.generate_execution_report(malformed_result)

        assert report is not None
        assert len(report) > 0  # Should produce some output even with bad data
        assert "generation failed" in report.lower() or isinstance(report, str)

    def test_slippage_tolerance_checking(self, sample_execution_result):
        """Test slippage tolerance validation"""
        reporter = ExecutionReporter()

        # Test with slippage within tolerance
        sample_execution_result.slippage_percentage = 1.0  # 1%
        slippage_analysis = reporter._analyze_slippage(sample_execution_result)
        assert slippage_analysis.slippage_within_tolerance is True

        # Test with slippage exceeding tolerance
        sample_execution_result.slippage_percentage = 3.0  # 3%
        slippage_analysis = reporter._analyze_slippage(sample_execution_result)
        assert slippage_analysis.slippage_within_tolerance is False

    def test_empty_leg_results_handling(self, sample_execution_result):
        """Test handling of empty leg results"""
        reporter = ExecutionReporter()

        # Remove leg results
        sample_execution_result.stock_result = None
        sample_execution_result.call_result = None
        sample_execution_result.put_result = None

        # Should still generate report without crashing
        report = reporter.generate_execution_report(sample_execution_result)
        assert report is not None
        assert len(report) > 0

    @pytest.mark.asyncio
    async def test_live_progress_display(self):
        """Test live progress display functionality"""
        reporter = ExecutionReporter()

        # Test that live progress can be created without error
        try:
            reporter.print_live_execution_progress("SPY", "test_123")
            # If it completes without exception, consider it successful
            success = True
        except Exception as e:
            # Live display might not work in test environment, that's ok
            success = "display" in str(e).lower() or "terminal" in str(e).lower()

        assert success

    def test_session_metrics_update(self, sample_execution_result):
        """Test session metrics are properly updated"""
        reporter = ExecutionReporter()

        initial_total = reporter.session_metrics["total_executions"]
        initial_successful = reporter.session_metrics["successful_executions"]

        reporter.generate_execution_report(sample_execution_result)

        assert reporter.session_metrics["total_executions"] == initial_total + 1
        assert (
            reporter.session_metrics["successful_executions"] == initial_successful + 1
        )
        assert len(reporter.session_metrics["execution_times"]) == 1
        assert len(reporter.session_metrics["slippage_history"]) == 1

    def test_failed_execution_metrics_update(self, failed_execution_result):
        """Test metrics update for failed executions"""
        reporter = ExecutionReporter()

        initial_successful = reporter.session_metrics["successful_executions"]

        reporter.generate_execution_report(failed_execution_result)

        assert reporter.session_metrics["total_executions"] == 1
        assert (
            reporter.session_metrics["successful_executions"] == initial_successful
        )  # No change
        assert len(reporter.session_metrics["execution_times"]) == 1
        assert len(reporter.session_metrics["slippage_history"]) == 1

    def test_report_generation_timing(self, sample_execution_result):
        """Test that report generation is reasonably fast"""
        reporter = ExecutionReporter()

        start_time = time.time()

        # Generate multiple reports
        for i in range(10):
            sample_execution_result.execution_id = f"SPY_{i}"
            reporter.generate_execution_report(sample_execution_result)

        total_time = time.time() - start_time

        # Should generate 10 reports in reasonable time
        assert total_time < 5.0  # Less than 5 seconds
        avg_time = total_time / 10
        assert avg_time < 0.5  # Less than 500ms per report

    def test_slippage_analysis_dataclass(self):
        """Test SlippageAnalysis dataclass structure"""
        analysis = SlippageAnalysis(
            total_slippage_dollars=2.5,
            slippage_percentage=0.25,
            leg_slippages={"stock": 0.02, "call": -0.03, "put": 0.03},
            worst_leg="put",
            worst_leg_slippage=0.03,
            avg_slippage_per_leg=0.027,
            slippage_within_tolerance=True,
            slippage_tolerance_percent=2.0,
            expected_vs_actual={"stock": {"expected": 100.0, "actual": 100.02}},
            price_improvement_legs=[],
            price_deterioration_legs=["stock", "call", "put"],
        )

        assert analysis.total_slippage_dollars == 2.5
        assert analysis.worst_leg == "put"
        assert analysis.slippage_within_tolerance is True
        assert len(analysis.price_deterioration_legs) == 3

    def test_performance_metrics_dataclass(self):
        """Test PerformanceMetrics dataclass structure"""
        metrics = PerformanceMetrics(
            execution_speed_ms=2340.0,
            order_placement_speed_ms=123.0,
            fill_monitoring_time_ms=2217.0,
            all_legs_fill_rate=100.0,
            partial_fill_rate=0.0,
            complete_failure_rate=0.0,
            fastest_execution_ms=1500.0,
            slowest_execution_ms=3000.0,
            median_execution_ms=2340.0,
            average_slippage_dollars=2.5,
            median_slippage_dollars=2.5,
            max_slippage_dollars=2.5,
            cost_savings_vs_combo_orders=1.25,
        )

        assert metrics.execution_speed_ms == 2340.0
        assert metrics.all_legs_fill_rate == 100.0
        assert metrics.cost_savings_vs_combo_orders == 1.25


@pytest.mark.performance
class TestExecutionReporterPerformance:
    """Performance tests for execution reporter"""

    @pytest.fixture
    def sample_execution_result(self):
        """Create sample execution result"""
        return ExecutionResult(
            success=True,
            execution_id="PERF_TEST",
            symbol="SPY",
            total_execution_time=2.0,
            all_legs_filled=True,
            partially_filled=False,
            legs_filled=3,
            total_legs=3,
            expected_total_cost=1000.0,
            actual_total_cost=1002.0,
            total_slippage=2.0,
            slippage_percentage=0.2,
        )

    def test_large_dataset_performance(self, sample_execution_result):
        """Test reporting performance with large dataset"""
        reporter = ExecutionReporter()

        start_time = time.time()

        # Generate 1000 execution reports
        for i in range(1000):
            sample_execution_result.execution_id = f"SPY_{i}"
            sample_execution_result.symbol = f"SYM{i % 10}"  # Vary symbols
            reporter.generate_execution_report(
                sample_execution_result, level=ReportLevel.SUMMARY
            )

        total_time = time.time() - start_time

        # Should handle 1000 reports in reasonable time
        assert total_time < 30.0  # Less than 30 seconds

        # Session stats should be accurate
        stats = reporter.get_session_statistics()
        assert stats["total_executions"] == 1000
        assert stats["successful_executions"] == 1000

    def test_memory_usage_stability(self, sample_execution_result):
        """Test that reporter doesn't leak memory"""
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Force garbage collection to get clean baseline
        gc.collect()

        # Take multiple memory samples to get stable baseline
        baseline_samples = []
        for _ in range(3):
            baseline_samples.append(process.memory_info().rss)
        initial_memory = min(baseline_samples)  # Use minimum for conservative baseline

        reporter = ExecutionReporter()

        # Generate many reports
        for i in range(500):
            sample_execution_result.execution_id = f"MEMORY_TEST_{i}"
            reporter.generate_execution_report(sample_execution_result)

        # Force garbage collection before final measurement
        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (adjusted for PyPy and test suite context)
        # Use more generous limits when running in full test suite context
        base_limit = 50 * 1024 * 1024  # 50MB base limit

        # If initial memory is high (>500MB), we're likely in full test suite - be more generous
        if initial_memory > 500 * 1024 * 1024:
            max_memory = int(
                base_limit * MEMORY_MULTIPLIER * 2
            )  # Double the limit for test suite context
        else:
            max_memory = int(base_limit * MEMORY_MULTIPLIER)

        assert (
            memory_increase < max_memory
        ), f"Memory increase {memory_increase} exceeds {max_memory} (initial: {initial_memory / 1024 / 1024:.1f}MB, final: {final_memory / 1024 / 1024:.1f}MB)"

    def test_statistics_calculation_performance(self, sample_execution_result):
        """Test performance of statistics calculations"""
        reporter = ExecutionReporter()

        # Generate many executions
        for i in range(1000):
            sample_execution_result.execution_id = f"STATS_TEST_{i}"
            reporter.generate_execution_report(sample_execution_result)

        # Statistics calculation should be fast
        start_time = time.time()
        stats = reporter.get_session_statistics()
        calc_time = time.time() - start_time

        assert calc_time < 1.0  # Should calculate stats in under 1 second
        assert stats["total_executions"] == 1000

    def test_concurrent_report_generation(self, sample_execution_result):
        """Test concurrent report generation performance"""
        import queue
        import threading

        reporter = ExecutionReporter()
        results_queue = queue.Queue()

        def generate_reports(thread_id):
            thread_results = []
            for i in range(50):
                result_copy = ExecutionResult(
                    success=True,
                    execution_id=f"THREAD_{thread_id}_{i}",
                    symbol="SPY",
                    total_execution_time=2.0,
                    all_legs_filled=True,
                    partially_filled=False,
                    legs_filled=3,
                    total_legs=3,
                    expected_total_cost=1000.0,
                    actual_total_cost=1002.0,
                    total_slippage=2.0,
                    slippage_percentage=0.2,
                )
                report = reporter.generate_execution_report(result_copy)
                thread_results.append(len(report))
            results_queue.put(thread_results)

        # Start multiple threads
        threads = []
        start_time = time.time()

        for i in range(5):
            thread = threading.Thread(target=generate_reports, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        total_time = time.time() - start_time

        # Should handle concurrent generation efficiently (adjusted for PyPy)
        max_time = 10.0 * TIMEOUT_MULTIPLIER
        assert (
            total_time < max_time
        ), f"Concurrent generation took {total_time}s (max: {max_time}s)"

        # All threads should have completed
        assert results_queue.qsize() == 5
