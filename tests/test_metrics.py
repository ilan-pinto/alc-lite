"""
Comprehensive tests for the metrics collection system.

Tests cover:
- RejectionReason enum functionality
- TimingMetric and CounterMetric classes
- ScanMetrics and CycleMetrics data classes
- MetricsCollector core functionality
- Performance comparison and analysis
- Rejection reason tracking and analysis
- JSON export capabilities
- Session summary generation
"""

import json
import tempfile
import time
from datetime import datetime
from unittest.mock import mock_open, patch

import pytest

from modules.Arbitrage.metrics import (
    CounterMetric,
    CycleMetrics,
    MetricsCollector,
    RejectionReason,
    ScanMetrics,
    TimingMetric,
)


@pytest.mark.unit
class TestRejectionReason:
    """Test RejectionReason enum functionality"""

    def test_rejection_reason_values(self):
        """Test that all rejection reasons have expected string values"""
        assert RejectionReason.SPREAD_TOO_WIDE.value == "spread_too_wide"
        assert (
            RejectionReason.BID_ASK_SPREAD_TOO_WIDE.value == "bid_ask_spread_too_wide"
        )
        assert RejectionReason.PROFIT_TARGET_NOT_MET.value == "profit_target_not_met"
        assert RejectionReason.INVALID_CONTRACT_DATA.value == "invalid_contract_data"
        assert RejectionReason.ORDER_NOT_FILLED.value == "order_not_filled"

    def test_rejection_reason_categories(self):
        """Test that rejection reasons are properly categorized"""
        price_reasons = [
            RejectionReason.SPREAD_TOO_WIDE,
            RejectionReason.BID_ASK_SPREAD_TOO_WIDE,
            RejectionReason.PRICE_LIMIT_EXCEEDED,
            RejectionReason.NET_CREDIT_NEGATIVE,
        ]

        profitability_reasons = [
            RejectionReason.PROFIT_TARGET_NOT_MET,
            RejectionReason.MIN_ROI_NOT_MET,
            RejectionReason.MAX_LOSS_THRESHOLD_EXCEEDED,
        ]

        for reason in price_reasons:
            assert isinstance(reason.value, str)
            assert (
                "spread" in reason.value
                or "price" in reason.value
                or "credit" in reason.value
            )

        for reason in profitability_reasons:
            assert isinstance(reason.value, str)
            assert (
                "profit" in reason.value
                or "roi" in reason.value
                or "loss" in reason.value
            )

    def test_all_rejection_reasons_unique(self):
        """Test that all rejection reason values are unique"""
        values = [reason.value for reason in RejectionReason]
        assert len(values) == len(set(values))


@pytest.mark.unit
class TestTimingMetric:
    """Test TimingMetric class functionality"""

    def test_timing_metric_creation(self):
        """Test TimingMetric creation and initial state"""
        start_time = time.time()
        timing = TimingMetric("test_operation", start_time)

        assert timing.name == "test_operation"
        assert timing.start_time == start_time
        assert timing.end_time is None
        assert timing.duration is None

    def test_timing_metric_finish(self):
        """Test finishing a timing measurement"""
        start_time = time.time()
        timing = TimingMetric("test_operation", start_time)

        # Small delay to ensure measurable duration
        time.sleep(0.01)
        duration = timing.finish()

        assert timing.end_time is not None
        assert timing.duration is not None
        assert duration == timing.duration
        assert timing.duration > 0
        assert timing.end_time > timing.start_time

    def test_timing_metric_multiple_finish_calls(self):
        """Test that multiple finish calls don't change the duration"""
        start_time = time.time()
        timing = TimingMetric("test_operation", start_time)

        duration1 = timing.finish()
        time.sleep(0.01)
        duration2 = timing.finish()

        assert duration1 == duration2
        assert timing.duration == duration1


@pytest.mark.unit
class TestCounterMetric:
    """Test CounterMetric class functionality"""

    def test_counter_metric_creation(self):
        """Test CounterMetric creation and initial state"""
        counter = CounterMetric("test_counter")

        assert counter.name == "test_counter"
        assert counter.count == 0

    def test_counter_increment_default(self):
        """Test default increment behavior"""
        counter = CounterMetric("test_counter")
        counter.increment()

        assert counter.count == 1

    def test_counter_increment_custom_value(self):
        """Test increment with custom value"""
        counter = CounterMetric("test_counter")
        counter.increment(5)

        assert counter.count == 5

    def test_counter_multiple_increments(self):
        """Test multiple increments accumulate correctly"""
        counter = CounterMetric("test_counter")
        counter.increment(3)
        counter.increment(2)
        counter.increment()

        assert counter.count == 6

    def test_counter_reset(self):
        """Test counter reset functionality"""
        counter = CounterMetric("test_counter")
        counter.increment(10)
        counter.reset()

        assert counter.count == 0


@pytest.mark.unit
class TestScanMetrics:
    """Test ScanMetrics data class functionality"""

    def test_scan_metrics_creation(self):
        """Test ScanMetrics creation with required fields"""
        start_time = time.time()
        scan = ScanMetrics("AAPL", "SFR", start_time)

        assert scan.symbol == "AAPL"
        assert scan.strategy == "SFR"
        assert scan.scan_start_time == start_time
        assert scan.scan_end_time is None
        assert scan.rejection_reasons == []
        assert scan.rejection_details == []

    def test_scan_metrics_add_rejection(self):
        """Test adding rejection reasons"""
        scan = ScanMetrics("AAPL", "SFR", time.time())
        details = {"spread": 0.05, "threshold": 0.03}

        scan.add_rejection(RejectionReason.SPREAD_TOO_WIDE, details)

        assert len(scan.rejection_reasons) == 1
        assert scan.rejection_reasons[0] == RejectionReason.SPREAD_TOO_WIDE
        assert len(scan.rejection_details) == 1
        assert scan.rejection_details[0] == details

    def test_scan_metrics_add_rejection_without_details(self):
        """Test adding rejection reason without details"""
        scan = ScanMetrics("AAPL", "SFR", time.time())

        scan.add_rejection(RejectionReason.PROFIT_TARGET_NOT_MET)

        assert len(scan.rejection_reasons) == 1
        assert len(scan.rejection_details) == 0

    def test_scan_metrics_finish(self):
        """Test finishing a scan"""
        start_time = time.time()
        scan = ScanMetrics("AAPL", "SFR", start_time)

        time.sleep(0.01)
        scan.finish(success=True)

        assert scan.scan_end_time is not None
        assert scan.success is True
        assert scan.error_message is None
        assert scan.total_duration > 0

    def test_scan_metrics_finish_with_error(self):
        """Test finishing a scan with error"""
        scan = ScanMetrics("AAPL", "SFR", time.time())
        error_msg = "Connection timeout"

        scan.finish(success=False, error_message=error_msg)

        assert scan.success is False
        assert scan.error_message == error_msg

    def test_scan_metrics_total_duration_before_finish(self):
        """Test total_duration property before finish"""
        scan = ScanMetrics("AAPL", "SFR", time.time())

        assert scan.total_duration is None


@pytest.mark.unit
class TestCycleMetrics:
    """Test CycleMetrics data class functionality"""

    def test_cycle_metrics_creation(self):
        """Test CycleMetrics creation"""
        start_time = time.time()
        cycle = CycleMetrics(start_time)

        assert cycle.cycle_start_time == start_time
        assert cycle.cycle_end_time is None
        assert cycle.rejection_summary == {}

    def test_cycle_metrics_finish(self):
        """Test finishing a cycle"""
        start_time = time.time()
        cycle = CycleMetrics(start_time)

        time.sleep(0.01)
        cycle.finish()

        assert cycle.cycle_end_time is not None
        assert cycle.cycle_duration > 0

    def test_cycle_metrics_success_rate(self):
        """Test success rate calculation"""
        cycle = CycleMetrics(time.time())
        cycle.total_symbols = 10
        cycle.successful_scans = 8

        assert cycle.success_rate == 80.0

    def test_cycle_metrics_success_rate_zero_symbols(self):
        """Test success rate with zero symbols"""
        cycle = CycleMetrics(time.time())
        cycle.total_symbols = 0

        assert cycle.success_rate == 0.0

    def test_cycle_metrics_fill_rate(self):
        """Test fill rate calculation"""
        cycle = CycleMetrics(time.time())
        cycle.total_orders_placed = 5
        cycle.total_orders_filled = 3

        assert cycle.fill_rate == 60.0

    def test_cycle_metrics_fill_rate_zero_orders(self):
        """Test fill rate with zero orders"""
        cycle = CycleMetrics(time.time())
        cycle.total_orders_placed = 0

        assert cycle.fill_rate == 0.0


@pytest.mark.unit
class TestMetricsCollector:
    """Test MetricsCollector core functionality"""

    def test_metrics_collector_initialization(self):
        """Test MetricsCollector initial state"""
        collector = MetricsCollector()

        assert collector.session_start_time > 0
        assert len(collector.active_timings) == 0
        assert len(collector.scan_metrics) == 0
        assert len(collector.cycle_metrics) == 0
        assert collector.current_scan is None
        assert collector.current_cycle is None

        # Check that standard counters are initialized
        assert "total_symbols_scanned" in collector.counters
        assert "total_orders_placed" in collector.counters
        assert collector.get_counter("total_symbols_scanned") == 0

    def test_timing_context_manager(self):
        """Test timing context manager functionality"""
        collector = MetricsCollector()

        with collector.timing("test_operation") as timing:
            time.sleep(0.01)
            assert isinstance(timing, TimingMetric)
            assert timing.name == "test_operation"

        assert timing.duration > 0
        assert "test_operation" in collector.active_timings

    def test_start_and_finish_cycle(self):
        """Test cycle tracking"""
        collector = MetricsCollector()

        cycle = collector.start_cycle(5)
        assert collector.current_cycle == cycle
        assert cycle.total_symbols == 5

        time.sleep(0.01)
        collector.finish_cycle()

        assert collector.current_cycle is None
        assert len(collector.cycle_metrics) == 1
        assert collector.cycle_metrics[0].cycle_duration > 0

    def test_start_and_finish_scan(self):
        """Test scan tracking"""
        collector = MetricsCollector()

        scan = collector.start_scan("AAPL", "SFR")
        assert collector.current_scan == scan
        assert scan.symbol == "AAPL"
        assert scan.strategy == "SFR"

        time.sleep(0.01)
        collector.finish_scan(success=True)

        assert collector.current_scan is None
        assert len(collector.scan_metrics) == 1
        assert collector.scan_metrics[0].success is True

    def test_scan_failure_increments_error_counter(self):
        """Test that failed scans increment error counter"""
        collector = MetricsCollector()

        collector.start_scan("AAPL", "SFR")
        collector.finish_scan(success=False, error_message="Test error")

        assert collector.get_counter("total_errors") == 1

    def test_add_rejection_reason(self):
        """Test adding rejection reasons to current scan"""
        collector = MetricsCollector()

        collector.start_scan("AAPL", "SFR")
        collector.add_rejection_reason(
            RejectionReason.SPREAD_TOO_WIDE, {"spread": 0.05}
        )
        collector.finish_scan()

        scan = collector.scan_metrics[0]
        assert len(scan.rejection_reasons) == 1
        assert scan.rejection_reasons[0] == RejectionReason.SPREAD_TOO_WIDE

    def test_add_rejection_reason_no_current_scan(self):
        """Test adding rejection reason when no current scan"""
        collector = MetricsCollector()

        # Should not raise error
        collector.add_rejection_reason(RejectionReason.SPREAD_TOO_WIDE)

    def test_increment_counter(self):
        """Test counter increment functionality"""
        collector = MetricsCollector()

        collector.increment_counter("test_counter", 3)
        assert collector.get_counter("test_counter") == 3

        collector.increment_counter("test_counter")
        assert collector.get_counter("test_counter") == 4

    def test_record_methods(self):
        """Test various record methods"""
        collector = MetricsCollector()

        collector.start_scan("AAPL", "SFR")
        collector.record_data_collection_time(1.5)
        collector.record_execution_time(0.8)
        collector.record_contracts_count(100)
        collector.record_order_placed()
        collector.record_order_filled()
        collector.record_opportunity_found()
        collector.record_expiries_scanned(3)
        collector.finish_scan()

        scan = collector.scan_metrics[0]
        assert scan.data_collection_time == 1.5
        assert scan.execution_time == 0.8
        assert scan.total_contracts == 100
        assert scan.orders_placed == 1
        assert scan.orders_filled == 1
        assert scan.opportunities_found == 1
        assert scan.expiries_scanned == 3

        # Check global counters
        assert collector.get_counter("total_contracts_fetched") == 100
        assert collector.get_counter("total_orders_placed") == 1
        assert collector.get_counter("total_orders_filled") == 1
        assert collector.get_counter("total_opportunities_found") == 1

    def test_reset_session(self):
        """Test session reset functionality"""
        collector = MetricsCollector()

        # Add some data
        collector.start_scan("AAPL", "SFR")
        collector.increment_counter("test_counter", 5)
        collector.finish_scan()

        initial_start_time = collector.session_start_time
        time.sleep(0.01)

        collector.reset_session()

        assert collector.session_start_time > initial_start_time
        assert len(collector.scan_metrics) == 0
        assert len(collector.cycle_metrics) == 0
        assert collector.get_counter("test_counter") == 0


@pytest.mark.unit
class TestPerformanceComparison:
    """Test performance comparison functionality"""

    def test_performance_comparison_insufficient_data(self):
        """Test performance comparison with insufficient cycle data"""
        collector = MetricsCollector()

        comparison = collector.get_performance_comparison()

        assert comparison["comparison_available"] is False
        assert "Not enough cycle data" in comparison["reason"]

    def test_performance_comparison_with_data(self):
        """Test performance comparison with sufficient data"""
        collector = MetricsCollector()

        # First cycle
        collector.start_cycle(5)
        # Add scans to populate the cycle metrics properly
        collector.start_scan("AAPL", "SFR")
        collector.record_opportunity_found()
        collector.record_order_placed()
        collector.finish_scan()
        collector.finish_cycle()

        # Second cycle
        collector.start_cycle(5)
        collector.start_scan("MSFT", "SFR")
        collector.record_opportunity_found()
        collector.record_opportunity_found()  # 2 opportunities
        collector.record_order_placed()
        collector.record_order_filled()
        collector.finish_scan()
        collector.finish_cycle()

        comparison = collector.get_performance_comparison()

        assert comparison["comparison_available"] is True
        assert "current_cycle" in comparison
        assert "previous_cycle" in comparison
        assert "percentage_changes" in comparison

        # Check percentage changes
        changes = comparison["percentage_changes"]
        assert changes["opportunities"] == 100.0  # 2 vs 1 = 100% increase
        assert changes["orders_placed"] == 0.0  # 1 vs 1 = 0% change
        assert changes["orders_filled"] == 100.0  # 1 vs 0 = 100% increase

    def test_percentage_change_calculation_zero_previous(self):
        """Test percentage change calculation when previous value is zero"""
        collector = MetricsCollector()

        # First cycle with zero opportunities
        collector.start_cycle(5)
        # Add a scan but no opportunities
        collector.start_scan("AAPL", "SFR")
        collector.finish_scan()
        collector.finish_cycle()

        # Second cycle with some opportunities
        collector.start_cycle(5)
        collector.start_scan("MSFT", "SFR")
        collector.record_opportunity_found()
        collector.finish_scan()
        collector.finish_cycle()

        comparison = collector.get_performance_comparison()
        changes = comparison["percentage_changes"]

        # Should return 100% when going from 0 to positive value
        assert changes["opportunities"] == 100.0


@pytest.mark.unit
class TestRejectionAnalysis:
    """Test rejection analysis functionality"""

    def test_rejection_analysis_no_data(self):
        """Test rejection analysis with no scan data"""
        collector = MetricsCollector()

        analysis = collector.get_rejection_analysis()

        assert analysis["total_rejections"] == 0
        assert analysis["rejection_breakdown"] == {}

    def test_rejection_analysis_with_data(self):
        """Test rejection analysis with rejection data"""
        collector = MetricsCollector()

        # Add scans with various rejections
        collector.start_scan("AAPL", "SFR")
        collector.add_rejection_reason(RejectionReason.SPREAD_TOO_WIDE)
        collector.add_rejection_reason(RejectionReason.PROFIT_TARGET_NOT_MET)
        collector.finish_scan()

        collector.start_scan("MSFT", "SFR")
        collector.add_rejection_reason(RejectionReason.SPREAD_TOO_WIDE)
        collector.finish_scan()

        analysis = collector.get_rejection_analysis()

        assert analysis["total_rejections"] == 3

        breakdown = analysis["rejection_breakdown"]
        assert breakdown["spread_too_wide"]["count"] == 2
        assert (
            abs(breakdown["spread_too_wide"]["percentage"] - 66.67) < 0.01
        )  # 2/3 * 100
        assert breakdown["profit_target_not_met"]["count"] == 1
        assert (
            abs(breakdown["profit_target_not_met"]["percentage"] - 33.33) < 0.01
        )  # 1/3 * 100

        # Check most common rejections
        most_common = analysis["most_common_rejections"]
        assert len(most_common) == 2
        assert most_common[0][0] == "spread_too_wide"  # Most common
        assert most_common[0][1]["count"] == 2


@pytest.mark.unit
class TestSessionSummary:
    """Test session summary functionality"""

    def test_session_summary_empty(self):
        """Test session summary with no data"""
        collector = MetricsCollector()

        summary = collector.get_session_summary()

        assert "session_info" in summary
        assert "performance_metrics" in summary
        assert "counters" in summary
        assert summary["success_rate"] == 0

        session_info = summary["session_info"]
        assert session_info["total_scans"] == 0
        assert session_info["successful_scans"] == 0
        assert session_info["failed_scans"] == 0

    def test_session_summary_with_data(self):
        """Test session summary with comprehensive data"""
        collector = MetricsCollector()

        # Add successful scan
        collector.start_scan("AAPL", "SFR")
        collector.record_data_collection_time(1.0)
        collector.record_execution_time(0.5)
        collector.record_contracts_count(50)
        collector.finish_scan(success=True)

        # Add failed scan
        collector.start_scan("MSFT", "SFR")
        collector.finish_scan(success=False, error_message="Timeout")

        summary = collector.get_session_summary()

        session_info = summary["session_info"]
        assert session_info["total_scans"] == 2
        assert session_info["successful_scans"] == 1
        assert session_info["failed_scans"] == 1

        assert summary["success_rate"] == 0.5  # 1/2

        perf_metrics = summary["performance_metrics"]
        assert perf_metrics["avg_contracts_per_scan"] == 50.0
        assert perf_metrics["avg_data_collection_time"] == 1.0
        assert perf_metrics["avg_execution_time"] == 0.5


@pytest.mark.unit
class TestJSONExport:
    """Test JSON export functionality"""

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_export_to_json_default_filename(self, mock_json_dump, mock_file):
        """Test JSON export with default filename"""
        collector = MetricsCollector()

        # Add some data
        collector.start_scan("AAPL", "SFR")
        collector.finish_scan()

        with patch("modules.Arbitrage.metrics.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20231215_143022"
            mock_datetime.now.return_value.isoformat.return_value = (
                "2023-12-15T14:30:22"
            )

            filename = collector.export_to_json()

        assert filename == "arbitrage_metrics_20231215_143022.json"
        mock_file.assert_called_once_with(filename, "w")
        mock_json_dump.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_export_to_json_custom_filename(self, mock_json_dump, mock_file):
        """Test JSON export with custom filename"""
        collector = MetricsCollector()
        custom_filename = "custom_metrics.json"

        filename = collector.export_to_json(custom_filename)

        assert filename == custom_filename
        mock_file.assert_called_once_with(custom_filename, "w")
        mock_json_dump.assert_called_once()

    def test_export_json_data_structure(self):
        """Test the structure of exported JSON data"""
        collector = MetricsCollector()

        # Add sample data
        collector.start_scan("AAPL", "SFR")
        collector.add_rejection_reason(RejectionReason.SPREAD_TOO_WIDE)
        collector.finish_scan()

        # Mock the file operations and capture the data
        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("json.dump") as mock_json_dump,
        ):

            collector.export_to_json("test.json")

            # Verify file operations were called
            mock_file.assert_called_once_with("test.json", "w")

            # Get the data passed to json.dump
            call_args = mock_json_dump.call_args
            data = call_args[0][0]  # First argument to json.dump

            assert "summary" in data
            assert "scan_details" in data
            assert "cycle_details" in data
            assert "export_timestamp" in data

            # Check that scan_details contains our scan
            assert len(data["scan_details"]) == 1
            scan_data = data["scan_details"][0]
            assert scan_data["symbol"] == "AAPL"
            assert scan_data["strategy"] == "SFR"


@pytest.mark.unit
class TestPrintSummary:
    """Test print summary functionality"""

    @patch("rich.console.Console")
    def test_print_summary_console_creation(self, mock_console_class):
        """Test that print_summary creates a console and prints panels"""
        mock_console = mock_console_class.return_value
        collector = MetricsCollector()

        collector.print_summary()

        mock_console_class.assert_called_once()
        assert (
            mock_console.print.call_count >= 1
        )  # Should print at least session summary

    @patch("rich.console.Console")
    def test_print_summary_with_rejections(self, mock_console_class):
        """Test print summary with rejection data"""
        mock_console = mock_console_class.return_value
        collector = MetricsCollector()

        # Add data with rejections
        collector.start_scan("AAPL", "SFR")
        collector.add_rejection_reason(RejectionReason.SPREAD_TOO_WIDE)
        collector.finish_scan()

        collector.print_summary()

        # Should print multiple panels/tables including rejection analysis
        assert mock_console.print.call_count >= 3

    @patch("rich.console.Console")
    def test_print_summary_with_performance_comparison(self, mock_console_class):
        """Test print summary with performance comparison data"""
        mock_console = mock_console_class.return_value
        collector = MetricsCollector()

        # Create two cycles for comparison
        collector.start_cycle(5)
        collector.start_scan("AAPL", "SFR")
        collector.finish_scan()
        collector.finish_cycle()
        collector.start_cycle(5)
        collector.start_scan("MSFT", "SFR")
        collector.finish_scan()
        collector.finish_cycle()

        collector.print_summary()

        # Should include performance comparison panel
        assert mock_console.print.call_count >= 4


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios"""

    def test_complete_trading_session_scenario(self):
        """Test a complete trading session scenario"""
        collector = MetricsCollector()

        # Start a cycle
        collector.start_cycle(3)

        # Scan 1: Successful with opportunity
        collector.start_scan("AAPL", "SFR")
        collector.record_contracts_count(100)
        collector.record_data_collection_time(2.0)
        collector.record_execution_time(1.0)
        collector.record_opportunity_found()
        collector.record_order_placed()
        collector.record_order_filled()
        collector.finish_scan(success=True)

        # Scan 2: Failed due to rejection
        collector.start_scan("MSFT", "SFR")
        collector.record_contracts_count(80)
        collector.add_rejection_reason(
            RejectionReason.SPREAD_TOO_WIDE, {"spread": 0.08}
        )
        collector.finish_scan(success=True)  # Rejection doesn't mean failure

        # Scan 3: Failed with error
        collector.start_scan("GOOGL", "SFR")
        collector.finish_scan(success=False, error_message="Connection timeout")

        # Finish cycle
        collector.finish_cycle()

        # Verify session state
        assert len(collector.scan_metrics) == 3
        assert len(collector.cycle_metrics) == 1
        assert collector.get_counter("total_symbols_scanned") == 3
        assert collector.get_counter("total_orders_placed") == 1
        assert collector.get_counter("total_orders_filled") == 1
        assert collector.get_counter("total_errors") == 1

        # Check cycle metrics
        cycle_metrics = collector.cycle_metrics[0]
        assert cycle_metrics.successful_scans == 2
        assert cycle_metrics.failed_scans == 1
        assert cycle_metrics.total_opportunities == 1
        assert cycle_metrics.total_orders_placed == 1
        assert cycle_metrics.total_orders_filled == 1
        assert cycle_metrics.total_contracts_processed == 180
        assert "spread_too_wide" in cycle_metrics.rejection_summary

        # Verify summary generation
        summary = collector.get_session_summary()
        assert summary["success_rate"] == 2 / 3  # 2 successful out of 3

        # Verify rejection analysis
        rejection_analysis = collector.get_rejection_analysis()
        assert rejection_analysis["total_rejections"] == 1
        assert "spread_too_wide" in rejection_analysis["rejection_breakdown"]

    def test_timing_context_manager_with_exception(self):
        """Test timing context manager handles exceptions properly"""
        collector = MetricsCollector()

        try:
            with collector.timing("failing_operation"):
                time.sleep(0.01)
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Timing should still be recorded despite exception
        assert "failing_operation" in collector.active_timings
        timing = collector.active_timings["failing_operation"]
        assert timing.duration is not None
        assert timing.duration > 0

    def test_cycle_rejection_summary_calculation(self):
        """Test that cycle rejection summary is calculated correctly"""
        collector = MetricsCollector()

        collector.start_cycle(2)

        # First scan with multiple rejections
        collector.start_scan("AAPL", "SFR")
        collector.add_rejection_reason(RejectionReason.SPREAD_TOO_WIDE)
        collector.add_rejection_reason(RejectionReason.PROFIT_TARGET_NOT_MET)
        collector.finish_scan()

        # Second scan with same rejection as first
        collector.start_scan("MSFT", "SFR")
        collector.add_rejection_reason(RejectionReason.SPREAD_TOO_WIDE)
        collector.finish_scan()

        collector.finish_cycle()

        cycle = collector.cycle_metrics[0]
        rejection_summary = cycle.rejection_summary

        assert rejection_summary["spread_too_wide"] == 2
        assert rejection_summary["profit_target_not_met"] == 1

    def test_multiple_cycles_performance_tracking(self):
        """Test performance tracking across multiple cycles"""
        collector = MetricsCollector()

        # First cycle - baseline performance
        collector.start_cycle(2)
        for symbol in ["AAPL", "MSFT"]:
            collector.start_scan(symbol, "SFR")
            collector.record_opportunity_found()
            collector.record_order_placed()
            collector.finish_scan()
        collector.finish_cycle()

        # Second cycle - improved performance
        collector.start_cycle(2)
        for symbol in ["GOOGL", "AMZN"]:
            collector.start_scan(symbol, "SFR")
            collector.record_opportunity_found()
            collector.record_opportunity_found()  # More opportunities
            collector.record_order_placed()
            collector.record_order_filled()  # Better fill rate
            collector.finish_scan()
        collector.finish_cycle()

        # Verify performance comparison
        comparison = collector.get_performance_comparison()
        assert comparison["comparison_available"] is True

        changes = comparison["percentage_changes"]
        assert changes["opportunities"] == 100.0  # 4 vs 2 opportunities
        assert changes["orders_filled"] > 0  # Improvement in fill rate
