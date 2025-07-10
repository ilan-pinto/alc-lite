"""
Metrics collection system for arbitrage strategies.

This module provides comprehensive metrics tracking for:
- Execution timing
- Order counts and success rates
- Performance analytics
- Data collection metrics
"""

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


@dataclass
class TimingMetric:
    """Individual timing measurement"""

    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None

    def finish(self) -> float:
        """Mark timing as complete and return duration"""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time
        return self.duration


@dataclass
class CounterMetric:
    """Counter for tracking quantities"""

    name: str
    count: int = 0

    def increment(self, value: int = 1) -> None:
        """Increment counter by value"""
        self.count += value

    def reset(self) -> None:
        """Reset counter to zero"""
        self.count = 0


@dataclass
class ScanMetrics:
    """Metrics for a single scan operation"""

    symbol: str
    strategy: str
    scan_start_time: float
    scan_end_time: Optional[float] = None
    total_contracts: int = 0
    data_collection_time: Optional[float] = None
    execution_time: Optional[float] = None
    orders_placed: int = 0
    opportunities_found: int = 0
    expiries_scanned: int = 0
    success: bool = False
    error_message: Optional[str] = None

    def finish(self, success: bool = True, error_message: Optional[str] = None) -> None:
        """Mark scan as complete"""
        self.scan_end_time = time.time()
        self.success = success
        self.error_message = error_message

    @property
    def total_duration(self) -> Optional[float]:
        """Total scan duration"""
        if self.scan_end_time:
            return self.scan_end_time - self.scan_start_time
        return None


class MetricsCollector:
    """
    Central metrics collection system for arbitrage strategies.

    Features:
    - Timing measurements with context managers
    - Counter tracking for orders, contracts, etc.
    - Per-scan and aggregate metrics
    - Export to JSON/CSV formats
    - Performance analytics
    """

    def __init__(self):
        self.session_start_time = time.time()
        self.active_timings: Dict[str, TimingMetric] = {}
        self.counters: Dict[str, CounterMetric] = {}
        self.scan_metrics: List[ScanMetrics] = []
        self.current_scan: Optional[ScanMetrics] = None

        # Initialize common counters
        self._init_counters()

    def _init_counters(self) -> None:
        """Initialize standard counters"""
        standard_counters = [
            "total_symbols_scanned",
            "total_contracts_fetched",
            "total_orders_placed",
            "total_orders_filled",
            "total_opportunities_found",
            "total_errors",
            "data_requests_made",
            "data_requests_successful",
        ]

        for counter_name in standard_counters:
            self.counters[counter_name] = CounterMetric(counter_name)

    @contextmanager
    def timing(self, name: str):
        """Context manager for timing operations"""
        timing_metric = TimingMetric(name, time.time())
        self.active_timings[name] = timing_metric

        try:
            yield timing_metric
        finally:
            timing_metric.finish()
            # Log timing if it's significant
            if timing_metric.duration and timing_metric.duration > 0.1:
                logger.debug(f"Timing [{name}]: {timing_metric.duration:.3f}s")

    def start_scan(self, symbol: str, strategy: str) -> ScanMetrics:
        """Start tracking a new scan operation"""
        self.current_scan = ScanMetrics(
            symbol=symbol, strategy=strategy, scan_start_time=time.time()
        )
        self.increment_counter("total_symbols_scanned")
        return self.current_scan

    def finish_scan(
        self, success: bool = True, error_message: Optional[str] = None
    ) -> None:
        """Complete the current scan tracking"""
        if self.current_scan:
            self.current_scan.finish(success, error_message)
            self.scan_metrics.append(self.current_scan)

            if not success:
                self.increment_counter("total_errors")

            self.current_scan = None

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter by value"""
        if name not in self.counters:
            self.counters[name] = CounterMetric(name)
        self.counters[name].increment(value)

    def get_counter(self, name: str) -> int:
        """Get current counter value"""
        return self.counters.get(name, CounterMetric(name)).count

    def record_data_collection_time(self, duration: float) -> None:
        """Record time spent collecting market data"""
        if self.current_scan:
            self.current_scan.data_collection_time = duration

    def record_execution_time(self, duration: float) -> None:
        """Record time spent in strategy execution"""
        if self.current_scan:
            self.current_scan.execution_time = duration

    def record_contracts_count(self, count: int) -> None:
        """Record number of contracts processed"""
        if self.current_scan:
            self.current_scan.total_contracts = count
        self.increment_counter("total_contracts_fetched", count)

    def record_order_placed(self) -> None:
        """Record that an order was placed"""
        if self.current_scan:
            self.current_scan.orders_placed += 1
        self.increment_counter("total_orders_placed")

    def record_order_filled(self) -> None:
        """Record that an order was filled"""
        self.increment_counter("total_orders_filled")

    def record_opportunity_found(self) -> None:
        """Record that an opportunity was found"""
        if self.current_scan:
            self.current_scan.opportunities_found += 1
        self.increment_counter("total_opportunities_found")

    def record_expiries_scanned(self, count: int) -> None:
        """Record number of expiries scanned"""
        if self.current_scan:
            self.current_scan.expiries_scanned = count

    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        session_duration = time.time() - self.session_start_time
        completed_scans = [scan for scan in self.scan_metrics if scan.success]
        failed_scans = [scan for scan in self.scan_metrics if not scan.success]

        # Calculate averages
        avg_scan_time = 0
        avg_contracts_per_scan = 0
        avg_data_collection_time = 0
        avg_execution_time = 0

        if completed_scans:
            total_scan_time = sum(scan.total_duration or 0 for scan in completed_scans)
            avg_scan_time = total_scan_time / len(completed_scans)

            total_contracts = sum(scan.total_contracts for scan in completed_scans)
            avg_contracts_per_scan = total_contracts / len(completed_scans)

            data_collection_times = [
                scan.data_collection_time
                for scan in completed_scans
                if scan.data_collection_time
            ]
            if data_collection_times:
                avg_data_collection_time = sum(data_collection_times) / len(
                    data_collection_times
                )

            execution_times = [
                scan.execution_time for scan in completed_scans if scan.execution_time
            ]
            if execution_times:
                avg_execution_time = sum(execution_times) / len(execution_times)

        return {
            "session_info": {
                "start_time": datetime.fromtimestamp(
                    self.session_start_time
                ).isoformat(),
                "session_duration": session_duration,
                "total_scans": len(self.scan_metrics),
                "successful_scans": len(completed_scans),
                "failed_scans": len(failed_scans),
            },
            "performance_metrics": {
                "avg_scan_time": avg_scan_time,
                "avg_contracts_per_scan": avg_contracts_per_scan,
                "avg_data_collection_time": avg_data_collection_time,
                "avg_execution_time": avg_execution_time,
            },
            "counters": {
                name: counter.count for name, counter in self.counters.items()
            },
            "success_rate": (
                len(completed_scans) / len(self.scan_metrics)
                if self.scan_metrics
                else 0
            ),
        }

    def export_to_json(self, filename: Optional[str] = None) -> str:
        """Export metrics to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"arbitrage_metrics_{timestamp}.json"

        data = {
            "summary": self.get_session_summary(),
            "scan_details": [asdict(scan) for scan in self.scan_metrics],
            "export_timestamp": datetime.now().isoformat(),
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        return filename

    def print_summary(self) -> None:
        """Print formatted summary to console"""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        summary = self.get_session_summary()

        # Session info panel
        session_info = summary["session_info"]
        session_text = f"""
Session Duration: {session_info['session_duration']:.1f}s
Total Scans: {session_info['total_scans']}
Successful: {session_info['successful_scans']}
Failed: {session_info['failed_scans']}
Success Rate: {summary['success_rate']:.1%}
        """
        console.print(Panel(session_text.strip(), title="Session Summary"))

        # Performance metrics table
        perf_table = Table(title="Performance Metrics")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="magenta")

        perf_metrics = summary["performance_metrics"]
        perf_table.add_row("Avg Scan Time", f"{perf_metrics['avg_scan_time']:.2f}s")
        perf_table.add_row(
            "Avg Contracts/Scan", f"{perf_metrics['avg_contracts_per_scan']:.1f}"
        )
        perf_table.add_row(
            "Avg Data Collection", f"{perf_metrics['avg_data_collection_time']:.2f}s"
        )
        perf_table.add_row(
            "Avg Execution Time", f"{perf_metrics['avg_execution_time']:.2f}s"
        )

        console.print(perf_table)

        # Counters table
        counter_table = Table(title="Operation Counters")
        counter_table.add_column("Counter", style="cyan")
        counter_table.add_column("Count", style="magenta")

        for name, count in summary["counters"].items():
            if count > 0:  # Only show non-zero counters
                counter_table.add_row(name.replace("_", " ").title(), str(count))

        console.print(counter_table)

    def reset_session(self) -> None:
        """Reset all metrics for a new session"""
        self.session_start_time = time.time()
        self.active_timings.clear()
        self.scan_metrics.clear()
        self.current_scan = None

        # Reset all counters
        for counter in self.counters.values():
            counter.reset()


# Global metrics collector instance
metrics_collector = MetricsCollector()
