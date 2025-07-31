"""
Metrics collection system for arbitrage strategies.

This module provides comprehensive metrics tracking for:
- Execution timing
- Order counts and success rates
- Performance analytics
- Data collection metrics
- Order rejection reasons
- Historical performance comparisons
"""

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


class RejectionReason(Enum):
    """Enumeration of reasons why an order was not placed"""

    # Price/Spread conditions
    SPREAD_TOO_WIDE = "spread_too_wide"
    BID_ASK_SPREAD_TOO_WIDE = "bid_ask_spread_too_wide"
    PRICE_LIMIT_EXCEEDED = "price_limit_exceeded"
    NET_CREDIT_NEGATIVE = "net_credit_negative"

    # Profitability conditions
    PROFIT_TARGET_NOT_MET = "profit_target_not_met"
    MIN_ROI_NOT_MET = "min_roi_not_met"
    MAX_LOSS_THRESHOLD_EXCEEDED = "max_loss_threshold_exceeded"
    MAX_PROFIT_THRESHOLD_NOT_MET = "max_profit_threshold_not_met"
    PROFIT_RATIO_THRESHOLD_NOT_MET = "profit_ratio_threshold_not_met"

    # Arbitrage conditions
    ARBITRAGE_CONDITION_NOT_MET = "arbitrage_condition_not_met"

    # Contract/Data issues
    INVALID_CONTRACT_DATA = "invalid_contract_data"
    MISSING_MARKET_DATA = "missing_market_data"
    DATA_COLLECTION_TIMEOUT = "data_collection_timeout"

    # Order execution issues
    ORDER_NOT_FILLED = "order_not_filled"
    ORDER_REJECTED = "order_rejected"

    # Strike/Option issues
    INVALID_STRIKE_COMBINATION = "invalid_strike_combination"
    INSUFFICIENT_VALID_STRIKES = "insufficient_valid_strikes"
    NO_VALID_EXPIRIES = "no_valid_expiries"

    # Volume/Liquidity issues
    VOLUME_TOO_LOW = "volume_too_low"
    LIQUIDITY_INSUFFICIENT = "liquidity_insufficient"
    INSUFFICIENT_LIQUIDITY = "insufficient_liquidity"

    # Calendar spread specific issues
    INSUFFICIENT_IV_SPREAD = "insufficient_iv_spread"
    INSUFFICIENT_THETA_RATIO = "insufficient_theta_ratio"
    COST_LIMIT_EXCEEDED = "cost_limit_exceeded"
    WIDE_BID_ASK_SPREAD = "wide_bid_ask_spread"
    DATA_TIMEOUT = "data_timeout"
    NO_OPTIONS_CHAIN = "no_options_chain"
    INSUFFICIENT_EXPIRY_OPTIONS = "insufficient_expiry_options"
    INSUFFICIENT_VOLUME = "insufficient_volume"


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
    orders_filled: int = 0
    opportunities_found: int = 0
    expiries_scanned: int = 0
    success: bool = False
    error_message: Optional[str] = None
    rejection_reasons: List[RejectionReason] = None
    rejection_details: List[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize lists if None"""
        if self.rejection_reasons is None:
            self.rejection_reasons = []
        if self.rejection_details is None:
            self.rejection_details = []

    def add_rejection(
        self, reason: RejectionReason, details: Dict[str, Any] = None
    ) -> None:
        """Add a rejection reason with optional details"""
        self.rejection_reasons.append(reason)
        if details:
            self.rejection_details.append(details)

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


@dataclass
class CycleMetrics:
    """Metrics for a complete cycle (all symbols scanned)"""

    cycle_start_time: float
    cycle_end_time: Optional[float] = None
    total_symbols: int = 0
    successful_scans: int = 0
    failed_scans: int = 0
    total_opportunities: int = 0
    total_orders_placed: int = 0
    total_orders_filled: int = 0
    total_contracts_processed: int = 0
    rejection_summary: Dict[str, int] = None

    def __post_init__(self):
        """Initialize rejection summary if None"""
        if self.rejection_summary is None:
            self.rejection_summary = {}

    def finish(self) -> None:
        """Mark cycle as complete"""
        self.cycle_end_time = time.time()

    @property
    def cycle_duration(self) -> Optional[float]:
        """Total cycle duration"""
        if self.cycle_end_time:
            return self.cycle_end_time - self.cycle_start_time
        return None

    @property
    def success_rate(self) -> float:
        """Success rate as percentage"""
        if self.total_symbols > 0:
            return (self.successful_scans / self.total_symbols) * 100
        return 0.0

    @property
    def fill_rate(self) -> float:
        """Order fill rate as percentage"""
        if self.total_orders_placed > 0:
            return (self.total_orders_filled / self.total_orders_placed) * 100
        return 0.0


class MetricsCollector:
    """
    Central metrics collection system for arbitrage strategies.

    Features:
    - Timing measurements with context managers
    - Counter tracking for orders, contracts, etc.
    - Per-scan and aggregate metrics
    - Rejection reason tracking
    - Historical performance comparison
    - Export to JSON/CSV formats
    - Performance analytics
    """

    def __init__(self):
        self.session_start_time = time.time()
        self.active_timings: Dict[str, TimingMetric] = {}
        self.counters: Dict[str, CounterMetric] = {}
        self.scan_metrics: List[ScanMetrics] = []
        self.cycle_metrics: List[CycleMetrics] = []
        self.current_scan: Optional[ScanMetrics] = None
        self.current_cycle: Optional[CycleMetrics] = None

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

    def start_cycle(self, total_symbols: int) -> CycleMetrics:
        """Start tracking a new cycle"""
        self.current_cycle = CycleMetrics(
            cycle_start_time=time.time(), total_symbols=total_symbols
        )
        return self.current_cycle

    def finish_cycle(self) -> None:
        """Complete the current cycle tracking"""
        if self.current_cycle:
            self.current_cycle.finish()
            # Calculate rejection summary from scan metrics in this cycle
            cycle_start_time = self.current_cycle.cycle_start_time
            cycle_scans = [
                scan
                for scan in self.scan_metrics
                if scan.scan_start_time >= cycle_start_time
            ]

            # Count rejection reasons
            rejection_summary = {}
            for scan in cycle_scans:
                for reason in scan.rejection_reasons:
                    reason_key = reason.value
                    rejection_summary[reason_key] = (
                        rejection_summary.get(reason_key, 0) + 1
                    )

            self.current_cycle.rejection_summary = rejection_summary

            # Update cycle totals
            self.current_cycle.successful_scans = sum(
                1 for scan in cycle_scans if scan.success
            )
            self.current_cycle.failed_scans = sum(
                1 for scan in cycle_scans if not scan.success
            )
            self.current_cycle.total_opportunities = sum(
                scan.opportunities_found for scan in cycle_scans
            )
            self.current_cycle.total_orders_placed = sum(
                scan.orders_placed for scan in cycle_scans
            )
            self.current_cycle.total_orders_filled = sum(
                scan.orders_filled for scan in cycle_scans
            )
            self.current_cycle.total_contracts_processed = sum(
                scan.total_contracts for scan in cycle_scans
            )

            self.cycle_metrics.append(self.current_cycle)
            self.current_cycle = None

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

    def add_rejection_reason(
        self, reason: RejectionReason, details: Dict[str, Any] = None
    ) -> None:
        """Add a rejection reason to the current scan"""
        if self.current_scan:
            self.current_scan.add_rejection(reason, details or {})

            # Log rejection reason immediately for real-time visibility
            symbol = details.get("symbol", "Unknown") if details else "Unknown"
            reason_text = reason.value.replace("_", " ").title()

            # Create a more detailed log message based on the rejection reason
            if reason == RejectionReason.BID_ASK_SPREAD_TOO_WIDE:
                contract_type = details.get("contract_type", "") if details else ""
                spread = details.get("bid_ask_spread", 0) if details else 0
                threshold = details.get("threshold", 0) if details else 0
                logger.info(
                    f"[{symbol}] REJECTED - {reason_text}: {contract_type} spread {spread:.2f} > {threshold}"
                )
            elif reason == RejectionReason.PRICE_LIMIT_EXCEEDED:
                limit_price = details.get("combo_limit_price", 0) if details else 0
                cost_limit = details.get("cost_limit", 0) if details else 0
                logger.info(
                    f"[{symbol}] REJECTED - {reason_text}: limit price {limit_price:.2f} > cost limit {cost_limit:.2f}"
                )
            elif reason == RejectionReason.PROFIT_TARGET_NOT_MET:
                profit_target = details.get("profit_target", 0) if details else 0
                min_roi = details.get("min_roi", 0) if details else 0
                logger.info(
                    f"[{symbol}] REJECTED - {reason_text}: target {profit_target:.2f}% > actual ROI {min_roi:.2f}%"
                )
            elif reason == RejectionReason.MAX_LOSS_THRESHOLD_EXCEEDED:
                max_loss_threshold = (
                    details.get("max_loss_threshold", 0) if details else 0
                )
                min_profit = details.get("min_profit", 0) if details else 0
                profit_ratio = details.get("profit_ratio", 0) if details else 0
                logger.info(
                    f"[{symbol}] REJECTED - {reason_text}: max loss {max_loss_threshold:.2f} >= calculated loss {min_profit:.2f} (profit ratio: {profit_ratio:.2f})"
                )
            elif reason == RejectionReason.MAX_PROFIT_THRESHOLD_NOT_MET:
                max_profit_threshold = (
                    details.get("max_profit_threshold", 0) if details else 0
                )
                max_profit = details.get("max_profit", 0) if details else 0
                profit_ratio = details.get("profit_ratio", 0) if details else 0
                logger.info(
                    f"[{symbol}] REJECTED - {reason_text}: threshold {max_profit_threshold:.2f} < max profit {max_profit:.2f} (profit ratio: {profit_ratio:.2f})"
                )
            elif reason == RejectionReason.PROFIT_RATIO_THRESHOLD_NOT_MET:
                profit_ratio_threshold = (
                    details.get("profit_ratio_threshold", 0) if details else 0
                )
                max_profit = details.get("max_profit", 0) if details else 0
                min_profit = details.get("min_profit", 0) if details else 0
                actual_ratio = max_profit / abs(min_profit) if min_profit != 0 else 0
                logger.info(
                    f"[{symbol}] REJECTED - {reason_text}: threshold {profit_ratio_threshold:.2f} > actual ratio {actual_ratio:.2f}"
                )
            elif reason == RejectionReason.INSUFFICIENT_VALID_STRIKES:
                count = details.get("valid_strikes_count", 0) if details else 0
                required = details.get("required_strikes", 0) if details else 0
                logger.info(
                    f"[{symbol}] REJECTED - {reason_text}: found {count} strikes, need {required}"
                )
            elif reason == RejectionReason.INVALID_STRIKE_COMBINATION:
                call_strike = details.get("call_strike", 0) if details else 0
                put_strike = details.get("put_strike", 0) if details else 0
                logger.info(
                    f"[{symbol}] REJECTED - {reason_text}: call {call_strike} vs put {put_strike}"
                )
            elif reason == RejectionReason.MISSING_MARKET_DATA:
                contract_type = details.get("contract_type", "") if details else ""
                logger.info(
                    f"[{symbol}] REJECTED - {reason_text}: missing {contract_type} data"
                )
            elif reason == RejectionReason.ARBITRAGE_CONDITION_NOT_MET:
                spread = details.get("spread", 0) if details else 0
                net_credit = details.get("net_credit", 0) if details else 0
                logger.info(
                    f"[{symbol}] REJECTED - {reason_text}: spread {spread:.2f} > net credit {net_credit:.2f}"
                )
            elif reason == RejectionReason.NET_CREDIT_NEGATIVE:
                net_credit = details.get("net_credit", 0) if details else 0
                stock_price = details.get("stock_price", 0) if details else 0
                spread = details.get("spread", 0) if details else 0
                logger.info(
                    f"[{symbol}] REJECTED - {reason_text}: net credit {net_credit:.2f} < 0 (stock: {stock_price:.2f}, spread: {spread:.2f})"
                )
            elif reason == RejectionReason.ORDER_NOT_FILLED:
                order_id = details.get("order_id", "") if details else ""
                timeout = details.get("timeout_seconds", 0) if details else 0
                filled_qty = details.get("filled_quantity", 0) if details else 0
                total_qty = details.get("total_quantity", 0) if details else 0
                logger.info(
                    f"[{symbol}] REJECTED - {reason_text}: order {order_id} not filled within {timeout}s (filled: {filled_qty}/{total_qty})"
                )
            elif reason == RejectionReason.ORDER_REJECTED:
                order_id = details.get("order_id", "") if details else ""
                reject_reason = details.get("reject_reason", "") if details else ""
                logger.info(
                    f"[{symbol}] REJECTED - {reason_text}: order {order_id} rejected - {reject_reason}"
                )
            else:
                # Generic rejection message
                logger.info(f"[{symbol}] REJECTED - {reason_text}")

            # Add expiry information if available
            if details and "expiry" in details:
                expiry = details["expiry"]
                logger.debug(
                    f"[{symbol}] Rejection details - Expiry: {expiry}, Reason: {reason_text}"
                )

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
        if self.current_scan:
            self.current_scan.orders_filled += 1
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

    def get_performance_comparison(self) -> Dict[str, Any]:
        """Get performance comparison with previous cycle"""
        if len(self.cycle_metrics) < 2:
            return {"comparison_available": False, "reason": "Not enough cycle data"}

        current_cycle = self.cycle_metrics[-1]
        previous_cycle = self.cycle_metrics[-2]

        def calculate_percentage_change(current: float, previous: float) -> float:
            """Calculate percentage change between two values"""
            if previous == 0:
                return 100.0 if current > 0 else 0.0
            return ((current - previous) / previous) * 100

        comparison = {
            "comparison_available": True,
            "current_cycle": {
                "duration": current_cycle.cycle_duration,
                "success_rate": current_cycle.success_rate,
                "fill_rate": current_cycle.fill_rate,
                "opportunities": current_cycle.total_opportunities,
                "orders_placed": current_cycle.total_orders_placed,
                "orders_filled": current_cycle.total_orders_filled,
                "contracts_processed": current_cycle.total_contracts_processed,
            },
            "previous_cycle": {
                "duration": previous_cycle.cycle_duration,
                "success_rate": previous_cycle.success_rate,
                "fill_rate": previous_cycle.fill_rate,
                "opportunities": previous_cycle.total_opportunities,
                "orders_placed": previous_cycle.total_orders_placed,
                "orders_filled": previous_cycle.total_orders_filled,
                "contracts_processed": previous_cycle.total_contracts_processed,
            },
            "percentage_changes": {
                "duration": calculate_percentage_change(
                    current_cycle.cycle_duration or 0,
                    previous_cycle.cycle_duration or 0,
                ),
                "success_rate": calculate_percentage_change(
                    current_cycle.success_rate, previous_cycle.success_rate
                ),
                "fill_rate": calculate_percentage_change(
                    current_cycle.fill_rate, previous_cycle.fill_rate
                ),
                "opportunities": calculate_percentage_change(
                    current_cycle.total_opportunities,
                    previous_cycle.total_opportunities,
                ),
                "orders_placed": calculate_percentage_change(
                    current_cycle.total_orders_placed,
                    previous_cycle.total_orders_placed,
                ),
                "orders_filled": calculate_percentage_change(
                    current_cycle.total_orders_filled,
                    previous_cycle.total_orders_filled,
                ),
                "contracts_processed": calculate_percentage_change(
                    current_cycle.total_contracts_processed,
                    previous_cycle.total_contracts_processed,
                ),
            },
        }

        return comparison

    def get_rejection_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of rejection reasons"""
        if not self.scan_metrics:
            return {"total_rejections": 0, "rejection_breakdown": {}}

        rejection_counts = {}
        total_rejections = 0

        for scan in self.scan_metrics:
            for reason in scan.rejection_reasons:
                reason_key = reason.value
                rejection_counts[reason_key] = rejection_counts.get(reason_key, 0) + 1
                total_rejections += 1

        # Calculate percentages
        rejection_breakdown = {}
        for reason, count in rejection_counts.items():
            percentage = (count / total_rejections) * 100 if total_rejections > 0 else 0
            rejection_breakdown[reason] = {"count": count, "percentage": percentage}

        return {
            "total_rejections": total_rejections,
            "rejection_breakdown": rejection_breakdown,
            "most_common_rejections": sorted(
                rejection_breakdown.items(), key=lambda x: x[1]["count"], reverse=True
            )[:5],
        }

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
                "total_cycles": len(self.cycle_metrics),
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
            "rejection_analysis": self.get_rejection_analysis(),
            "performance_comparison": self.get_performance_comparison(),
        }

    def export_to_json(self, filename: Optional[str] = None) -> str:
        """Export metrics to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"arbitrage_metrics_{timestamp}.json"

        data = {
            "summary": self.get_session_summary(),
            "scan_details": [asdict(scan) for scan in self.scan_metrics],
            "cycle_details": [asdict(cycle) for cycle in self.cycle_metrics],
            "export_timestamp": datetime.now().isoformat(),
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return filename

    def print_summary(self) -> None:
        """Print formatted summary to console"""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

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
Total Cycles: {session_info['total_cycles']}
        """
        console.print(Panel(session_text.strip(), title="Session Summary"))

        # Performance comparison panel
        perf_comparison = summary["performance_comparison"]
        if perf_comparison["comparison_available"]:
            comparison_text = ""
            for metric, change in perf_comparison["percentage_changes"].items():
                direction = "↑" if change > 0 else "↓" if change < 0 else "→"
                color = "green" if change > 0 else "red" if change < 0 else "yellow"
                comparison_text += (
                    f"{metric.replace('_', ' ').title()}: {direction} {change:+.1f}%\n"
                )

            console.print(
                Panel(comparison_text.strip(), title="Performance vs Previous Cycle")
            )

        # Rejection analysis table
        rejection_analysis = summary["rejection_analysis"]
        if rejection_analysis["total_rejections"] > 0:
            rejection_table = Table(title="Rejection Analysis")
            rejection_table.add_column("Reason", style="cyan")
            rejection_table.add_column("Count", style="magenta")
            rejection_table.add_column("Percentage", style="green")

            for reason, data in rejection_analysis["rejection_breakdown"].items():
                rejection_table.add_row(
                    reason.replace("_", " ").title(),
                    str(data["count"]),
                    f"{data['percentage']:.1f}%",
                )

            console.print(rejection_table)

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
        self.cycle_metrics.clear()
        self.current_scan = None
        self.current_cycle = None

        # Reset all counters
        for counter in self.counters.values():
            counter.reset()


# Global metrics collector instance
metrics_collector = MetricsCollector()
