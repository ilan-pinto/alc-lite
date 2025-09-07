"""
Beautiful execution reporter for SFR parallel execution system.

This module provides comprehensive, formatted reporting of parallel execution results
including detailed slippage analysis, performance metrics, and operational insights.
"""

import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from rich.text import Text

from ..common import get_logger

logger = get_logger()


class ReportLevel(Enum):
    """Report detail levels."""

    SUMMARY = "summary"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    DEBUG = "debug"


class ReportFormat(Enum):
    """Report output formats."""

    CONSOLE = "console"
    HTML = "html"
    JSON = "json"
    TEXT = "text"


@dataclass
class SlippageAnalysis:
    """Detailed slippage analysis for execution results."""

    total_slippage_dollars: float
    slippage_percentage: float
    leg_slippages: Dict[str, float]

    # Analysis
    worst_leg: str
    worst_leg_slippage: float
    avg_slippage_per_leg: float
    slippage_within_tolerance: bool
    slippage_tolerance_percent: float

    # Breakdown
    expected_vs_actual: Dict[str, Dict[str, float]]
    price_improvement_legs: List[str]
    price_deterioration_legs: List[str]


@dataclass
class PerformanceMetrics:
    """Performance metrics for execution analysis."""

    execution_speed_ms: float
    order_placement_speed_ms: float
    fill_monitoring_time_ms: float

    # Success rates
    all_legs_fill_rate: float
    partial_fill_rate: float
    complete_failure_rate: float

    # Timing analysis
    fastest_execution_ms: float
    slowest_execution_ms: float
    median_execution_ms: float

    # Cost analysis
    average_slippage_dollars: float
    median_slippage_dollars: float
    max_slippage_dollars: float
    cost_savings_vs_combo_orders: float


class ExecutionReporter:
    """
    Beautiful, comprehensive reporter for SFR parallel execution results.

    Provides:
    1. Formatted execution summaries with Rich console output
    2. Detailed slippage analysis and breakdown
    3. Performance trend analysis
    4. Beautiful tables and charts
    5. Export capabilities (HTML, JSON, text)
    """

    def __init__(self):
        self.console = Console(width=120, force_terminal=True)
        self.report_history: List[Dict] = []
        self._max_history_size = 100  # Limit history to prevent memory leaks
        self.session_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "total_slippage": 0.0,
            "execution_times": [],
            "slippage_history": [],
        }
        self._max_metrics_history = 1000  # Limit metrics history

        logger.debug("ExecutionReporter initialized")

    def generate_execution_report(
        self,
        result: Any,  # ExecutionResult object
        level: ReportLevel = ReportLevel.DETAILED,
        format_type: ReportFormat = ReportFormat.CONSOLE,
    ) -> str:
        """
        Generate a comprehensive execution report.

        Args:
            result: ExecutionResult object with execution details
            level: Detail level for the report
            format_type: Output format

        Returns:
            Formatted report string
        """
        report_start_time = time.time()

        # Update session metrics
        self._update_session_metrics(result)

        # Generate slippage analysis
        slippage_analysis = self._analyze_slippage(result)

        # Generate performance metrics
        performance_metrics = self._calculate_performance_metrics(result)

        # Track report in history
        report_data = {
            "execution_id": result.execution_id,
            "symbol": result.symbol,
            "success": result.success,
            "timestamp": datetime.now().isoformat(),
            "format_type": format_type.value,
            "level": level.value,
        }
        self.report_history.append(report_data)
        # Limit history size to prevent memory leaks
        if len(self.report_history) > self._max_history_size:
            self.report_history.pop(0)

        try:
            if format_type == ReportFormat.CONSOLE:
                return self._generate_console_report(
                    result, slippage_analysis, performance_metrics, level
                )
            elif format_type == ReportFormat.HTML:
                return self._generate_html_report(
                    result, slippage_analysis, performance_metrics, level
                )
            elif format_type == ReportFormat.JSON:
                return self._generate_json_report(
                    result, slippage_analysis, performance_metrics
                )
            else:  # TEXT
                return self._generate_text_report(
                    result, slippage_analysis, performance_metrics, level
                )

        except Exception as e:
            logger.error(f"Error generating execution report: {e}")
            return f"Report generation failed: {str(e)}"

        finally:
            report_time = time.time() - report_start_time
            logger.debug(f"Report generated in {report_time:.3f}s")

    def _update_session_metrics(self, result: Any) -> None:
        """Update session-wide metrics."""
        self.session_metrics["total_executions"] += 1

        if result.success:
            self.session_metrics["successful_executions"] += 1

        self.session_metrics["total_slippage"] += abs(result.total_slippage)
        self.session_metrics["execution_times"].append(result.total_execution_time)
        self.session_metrics["slippage_history"].append(result.total_slippage)

        # Limit metrics history size to prevent memory leaks
        if len(self.session_metrics["execution_times"]) > self._max_metrics_history:
            self.session_metrics["execution_times"].pop(0)
        if len(self.session_metrics["slippage_history"]) > self._max_metrics_history:
            self.session_metrics["slippage_history"].pop(0)

    def _analyze_slippage(self, result: Any) -> SlippageAnalysis:
        """Perform detailed slippage analysis."""

        leg_slippages = {}
        expected_vs_actual = {}
        price_improvement_legs = []
        price_deterioration_legs = []

        # Analyze each leg
        for leg_name, leg_result in [
            ("stock", result.stock_result),
            ("call", result.call_result),
            ("put", result.put_result),
        ]:
            if not leg_result:
                continue

            slippage = leg_result.get("slippage", 0.0)
            target_price = leg_result.get("target_price", 0.0)
            fill_price = leg_result.get("avg_fill_price", 0.0)

            leg_slippages[leg_name] = slippage
            expected_vs_actual[leg_name] = {
                "expected": target_price,
                "actual": fill_price,
                "slippage": slippage,
                "slippage_percent": (
                    (abs(slippage) / target_price * 100) if target_price > 0 else 0.0
                ),
            }

            # Determine if price improved or deteriorated
            if leg_result.get("action") == "BUY":
                # For buying, lower price is better
                if fill_price < target_price:
                    price_improvement_legs.append(leg_name)
                elif fill_price > target_price:
                    price_deterioration_legs.append(leg_name)
            else:  # SELL
                # For selling, higher price is better
                if fill_price > target_price:
                    price_improvement_legs.append(leg_name)
                elif fill_price < target_price:
                    price_deterioration_legs.append(leg_name)

        # Find worst leg
        worst_leg = ""
        worst_slippage = 0.0
        if leg_slippages:
            worst_leg = max(leg_slippages.keys(), key=lambda k: abs(leg_slippages[k]))
            worst_slippage = abs(leg_slippages[worst_leg])

        # Calculate averages
        avg_slippage = (
            statistics.mean([abs(s) for s in leg_slippages.values()])
            if leg_slippages
            else 0.0
        )

        # Check tolerance (assume 2% tolerance)
        tolerance_percent = 2.0
        within_tolerance = abs(result.slippage_percentage) <= tolerance_percent

        return SlippageAnalysis(
            total_slippage_dollars=result.total_slippage,
            slippage_percentage=result.slippage_percentage,
            leg_slippages=leg_slippages,
            worst_leg=worst_leg,
            worst_leg_slippage=worst_slippage,
            avg_slippage_per_leg=avg_slippage,
            slippage_within_tolerance=within_tolerance,
            slippage_tolerance_percent=tolerance_percent,
            expected_vs_actual=expected_vs_actual,
            price_improvement_legs=price_improvement_legs,
            price_deterioration_legs=price_deterioration_legs,
        )

    def _calculate_performance_metrics(self, result: Any) -> PerformanceMetrics:
        """Calculate performance metrics."""

        execution_times = self.session_metrics["execution_times"]
        slippage_history = self.session_metrics["slippage_history"]

        # Speed metrics (convert to milliseconds)
        execution_speed_ms = result.total_execution_time * 1000
        placement_speed_ms = result.order_placement_time * 1000
        monitoring_time_ms = result.fill_monitoring_time * 1000

        # Success rates
        total = self.session_metrics["total_executions"]
        successful = self.session_metrics["successful_executions"]

        all_legs_fill_rate = (successful / total * 100) if total > 0 else 0.0
        partial_fill_rate = ((total - successful) / total * 100) if total > 0 else 0.0
        complete_failure_rate = 0.0  # Would need more granular data

        # Timing analysis
        if execution_times:
            fastest_ms = min(execution_times) * 1000
            slowest_ms = max(execution_times) * 1000
            median_ms = statistics.median(execution_times) * 1000
        else:
            fastest_ms = slowest_ms = median_ms = execution_speed_ms

        # Cost analysis
        if slippage_history:
            avg_slippage = statistics.mean([abs(s) for s in slippage_history])
            median_slippage = statistics.median([abs(s) for s in slippage_history])
            max_slippage = max([abs(s) for s in slippage_history])
        else:
            avg_slippage = median_slippage = max_slippage = abs(result.total_slippage)

        # Estimate combo order savings (rough estimate)
        combo_order_slippage_estimate = (
            avg_slippage * 1.5
        )  # Assume combo orders have 50% more slippage
        cost_savings = combo_order_slippage_estimate - avg_slippage

        return PerformanceMetrics(
            execution_speed_ms=execution_speed_ms,
            order_placement_speed_ms=placement_speed_ms,
            fill_monitoring_time_ms=monitoring_time_ms,
            all_legs_fill_rate=all_legs_fill_rate,
            partial_fill_rate=partial_fill_rate,
            complete_failure_rate=complete_failure_rate,
            fastest_execution_ms=fastest_ms,
            slowest_execution_ms=slowest_ms,
            median_execution_ms=median_ms,
            average_slippage_dollars=avg_slippage,
            median_slippage_dollars=median_slippage,
            max_slippage_dollars=max_slippage,
            cost_savings_vs_combo_orders=cost_savings,
        )

    def _generate_console_report(
        self,
        result: Any,
        slippage_analysis: SlippageAnalysis,
        performance_metrics: PerformanceMetrics,
        level: ReportLevel,
    ) -> str:
        """Generate beautiful console report using Rich."""

        # Create layout
        layout = Layout()

        # Header panel
        status_emoji = "✅" if result.success else "❌"
        title = f"{status_emoji} SFR Parallel Execution Report - {result.symbol}"

        header_text = Text()
        header_text.append(f"Execution ID: ", style="bold blue")
        header_text.append(f"{result.execution_id}\n", style="white")
        header_text.append(f"Symbol: ", style="bold blue")
        header_text.append(f"{result.symbol}\n", style="white")
        header_text.append(f"Status: ", style="bold blue")

        if result.success:
            header_text.append("SUCCESS", style="bold green")
        else:
            header_text.append("FAILED", style="bold red")
            if result.error_message:
                header_text.append(f" - {result.error_message}", style="red")

        header_panel = Panel(header_text, title=title, box=box.DOUBLE)

        # Execution Summary Table
        summary_table = Table(title="📊 Execution Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="bold cyan", min_width=20)
        summary_table.add_column("Value", style="white", min_width=15)
        summary_table.add_column("Analysis", style="yellow", min_width=30)

        # Timing metrics
        timing_style = (
            "green"
            if result.total_execution_time < 5.0
            else "yellow" if result.total_execution_time < 10.0 else "red"
        )
        timing_analysis = (
            "Excellent"
            if result.total_execution_time < 2.0
            else "Good" if result.total_execution_time < 5.0 else "Needs Improvement"
        )

        summary_table.add_row(
            "⏱️  Total Time",
            f"{result.total_execution_time:.2f}s",
            f"[{timing_style}]{timing_analysis}[/{timing_style}]",
        )

        summary_table.add_row(
            "🎯 Legs Filled",
            f"{result.legs_filled}/{result.total_legs}",
            (
                "Perfect"
                if result.all_legs_filled
                else f"Partial ({result.legs_filled}/{result.total_legs})"
            ),
        )

        # Cost analysis
        cost_color = (
            "green"
            if abs(result.total_slippage) < 1.0
            else "yellow" if abs(result.total_slippage) < 5.0 else "red"
        )
        summary_table.add_row(
            "💰 Total Slippage",
            f"[{cost_color}]${result.total_slippage:.2f}[/{cost_color}]",
            f"[{cost_color}]{result.slippage_percentage:.2f}%[/{cost_color}]",
        )

        if result.success:
            profit_color = "green" if result.slippage_percentage < 1.0 else "yellow"
            summary_table.add_row(
                "📈 Net Result",
                f"Expected: ${result.expected_total_cost:.2f}",
                f"[{profit_color}]Actual: ${result.actual_total_cost:.2f}[/{profit_color}]",
            )

        # Detailed leg analysis (if detailed level)
        if level in [ReportLevel.DETAILED, ReportLevel.COMPREHENSIVE]:
            legs_table = Table(title="🦵 Individual Leg Analysis", box=box.SIMPLE)
            legs_table.add_column("Leg", style="bold magenta", min_width=8)
            legs_table.add_column("Action", style="cyan", min_width=6)
            legs_table.add_column("Target Price", style="white", min_width=12)
            legs_table.add_column("Fill Price", style="white", min_width=12)
            legs_table.add_column("Slippage", style="yellow", min_width=10)
            legs_table.add_column("Status", style="green", min_width=8)

            for leg_name, leg_result in [
                ("Stock", result.stock_result),
                ("Call", result.call_result),
                ("Put", result.put_result),
            ]:
                if leg_result:
                    slippage = leg_result.get("slippage", 0.0)
                    slippage_color = (
                        "green"
                        if abs(slippage) < 0.10
                        else "yellow" if abs(slippage) < 0.50 else "red"
                    )

                    status_text = leg_result.get("fill_status", "unknown")
                    status_color = "green" if status_text == "filled" else "red"

                    legs_table.add_row(
                        f"{leg_result.get('leg_type', leg_name).upper()}",
                        leg_result.get("action", "N/A"),
                        f"${leg_result.get('target_price', 0):.2f}",
                        f"${leg_result.get('avg_fill_price', 0):.2f}",
                        f"[{slippage_color}]${slippage:.3f}[/{slippage_color}]",
                        f"[{status_color}]{status_text.upper()}[/{status_color}]",
                    )

        # Performance metrics (if comprehensive)
        if level == ReportLevel.COMPREHENSIVE:
            perf_table = Table(title="⚡ Performance Metrics", box=box.SIMPLE)
            perf_table.add_column("Category", style="bold blue", min_width=20)
            perf_table.add_column("Current", style="white", min_width=15)
            perf_table.add_column("Session Avg", style="yellow", min_width=15)

            perf_table.add_row(
                "Order Placement",
                f"{performance_metrics.order_placement_speed_ms:.0f}ms",
                (
                    f"{statistics.mean([t*1000 for t in self.session_metrics['execution_times']]):.0f}ms"
                    if self.session_metrics["execution_times"]
                    else "N/A"
                ),
            )

            perf_table.add_row(
                "Fill Monitoring",
                f"{performance_metrics.fill_monitoring_time_ms:.0f}ms",
                f"{performance_metrics.median_execution_ms:.0f}ms",
            )

            perf_table.add_row(
                "Success Rate",
                "100%" if result.success else "0%",
                f"{performance_metrics.all_legs_fill_rate:.1f}%",
            )

            perf_table.add_row(
                "Slippage Control",
                f"${abs(result.total_slippage):.2f}",
                f"${performance_metrics.average_slippage_dollars:.2f}",
            )

        # Generate the complete report
        console_output = []

        with self.console.capture() as capture:
            self.console.print(header_panel)
            self.console.print("\n")
            self.console.print(summary_table)

            if level in [ReportLevel.DETAILED, ReportLevel.COMPREHENSIVE]:
                self.console.print("\n")
                self.console.print(legs_table)

            if level == ReportLevel.COMPREHENSIVE:
                self.console.print("\n")
                self.console.print(perf_table)

            # Slippage analysis panel
            if level != ReportLevel.SUMMARY:
                slippage_text = Text()

                if slippage_analysis.slippage_within_tolerance:
                    slippage_text.append("✅ Slippage within tolerance ", style="green")
                else:
                    slippage_text.append("⚠️  Slippage exceeds tolerance ", style="red")

                slippage_text.append(
                    f"({slippage_analysis.slippage_tolerance_percent}%)\n"
                )

                if slippage_analysis.worst_leg:
                    slippage_text.append(
                        f"Worst leg: {slippage_analysis.worst_leg.upper()} ",
                        style="white",
                    )
                    slippage_text.append(
                        f"(${slippage_analysis.worst_leg_slippage:.3f})\n", style="red"
                    )

                if slippage_analysis.price_improvement_legs:
                    slippage_text.append("✨ Price improvements: ", style="green")
                    slippage_text.append(
                        ", ".join(slippage_analysis.price_improvement_legs).upper(),
                        style="green",
                    )
                    slippage_text.append("\n")

                if slippage_analysis.price_deterioration_legs:
                    slippage_text.append("📉 Price deteriorations: ", style="red")
                    slippage_text.append(
                        ", ".join(slippage_analysis.price_deterioration_legs).upper(),
                        style="red",
                    )

                slippage_panel = Panel(
                    slippage_text, title="📊 Slippage Analysis", box=box.SIMPLE
                )
                self.console.print("\n")
                self.console.print(slippage_panel)

            # Footer with session stats
            if level == ReportLevel.COMPREHENSIVE:
                footer_text = Text()
                footer_text.append(
                    f"Session: {self.session_metrics['total_executions']} executions, ",
                    style="dim white",
                )
                footer_text.append(
                    f"{self.session_metrics['successful_executions']} successful ",
                    style="green",
                )
                footer_text.append(
                    f"({performance_metrics.all_legs_fill_rate:.1f}%)",
                    style="dim white",
                )

                footer_panel = Panel(
                    footer_text, title="📈 Session Statistics", box=box.SIMPLE
                )
                self.console.print("\n")
                self.console.print(footer_panel)

        return capture.get()

    def _generate_html_report(
        self,
        result: Any,
        slippage_analysis: SlippageAnalysis,
        performance_metrics: PerformanceMetrics,
        level: ReportLevel,
    ) -> str:
        """Generate HTML report."""
        # This would generate a beautiful HTML report
        # For now, return basic HTML structure

        status_class = "success" if result.success else "failure"

        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SFR Parallel Execution Report - {result.symbol}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
        .{status_class} {{ color: {"#28a745" if result.success else "#dc3545"}; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; margin: 10px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .legs-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .legs-table th, .legs-table td {{ border: 1px solid #dee2e6; padding: 12px; text-align: left; }}
        .legs-table th {{ background-color: #e9ecef; }}
        .slippage-positive {{ color: #28a745; }}
        .slippage-negative {{ color: #dc3545; }}
        .footer {{ margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 SFR Parallel Execution Report</h1>
        <p><strong>Symbol:</strong> {result.symbol} | <strong>Execution ID:</strong> {result.execution_id}</p>
        <p class="{status_class}"><strong>Status:</strong> {"SUCCESS" if result.success else "FAILED"}</p>
    </div>

    <div class="metric-card">
        <h3>⚡ Execution Summary</h3>
        <p><strong>Total Time:</strong> {result.total_execution_time:.2f}s</p>
        <p><strong>Legs Filled:</strong> {result.legs_filled}/{result.total_legs}</p>
        <p><strong>Total Slippage:</strong> ${result.total_slippage:.2f} ({result.slippage_percentage:.2f}%)</p>
    </div>

    <div class="metric-card">
        <h3>📊 Slippage Analysis</h3>
        <p><strong>Within Tolerance:</strong> {"Yes" if slippage_analysis.slippage_within_tolerance else "No"}</p>
        <p><strong>Worst Leg:</strong> {slippage_analysis.worst_leg.upper() if slippage_analysis.worst_leg else "N/A"} (${slippage_analysis.worst_leg_slippage:.3f})</p>
    </div>

    <div class="footer">
        <p>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Session: {self.session_metrics['total_executions']} executions</p>
    </div>
</body>
</html>
"""
        return html_template

    def _generate_json_report(
        self,
        result: Any,
        slippage_analysis: SlippageAnalysis,
        performance_metrics: PerformanceMetrics,
    ) -> str:
        """Generate JSON report."""
        import json

        report_data = {
            "execution_summary": {
                "execution_id": result.execution_id,
                "symbol": result.symbol,
                "success": result.success,
                "total_execution_time": result.total_execution_time,
                "legs_filled": f"{result.legs_filled}/{result.total_legs}",
                "all_legs_filled": result.all_legs_filled,
                "partially_filled": result.partially_filled,
            },
            "financial_summary": {
                "expected_total_cost": result.expected_total_cost,
                "actual_total_cost": result.actual_total_cost,
                "total_slippage": result.total_slippage,
                "slippage_percentage": result.slippage_percentage,
            },
            "slippage_analysis": {
                "total_slippage_dollars": slippage_analysis.total_slippage_dollars,
                "slippage_percentage": slippage_analysis.slippage_percentage,
                "leg_slippages": slippage_analysis.leg_slippages,
                "worst_leg": slippage_analysis.worst_leg,
                "worst_leg_slippage": slippage_analysis.worst_leg_slippage,
                "slippage_within_tolerance": slippage_analysis.slippage_within_tolerance,
                "price_improvement_legs": slippage_analysis.price_improvement_legs,
                "price_deterioration_legs": slippage_analysis.price_deterioration_legs,
            },
            "performance_metrics": {
                "execution_speed_ms": performance_metrics.execution_speed_ms,
                "order_placement_speed_ms": performance_metrics.order_placement_speed_ms,
                "fill_monitoring_time_ms": performance_metrics.fill_monitoring_time_ms,
                "all_legs_fill_rate": performance_metrics.all_legs_fill_rate,
                "average_slippage_dollars": performance_metrics.average_slippage_dollars,
            },
            "leg_details": {
                "stock": result.stock_result,
                "call": result.call_result,
                "put": result.put_result,
            },
            "session_metrics": self.session_metrics,
            "timestamp": datetime.now().isoformat(),
        }

        return json.dumps(report_data, indent=2, default=str)

    def _generate_text_report(
        self,
        result: Any,
        slippage_analysis: SlippageAnalysis,
        performance_metrics: PerformanceMetrics,
        level: ReportLevel,
    ) -> str:
        """Generate plain text report."""

        status_symbol = "✓" if result.success else "✗"

        report_lines = [
            f"╔═══════════════════════════════════════════════════════════════════════════╗",
            f"║                    SFR PARALLEL EXECUTION REPORT                         ║",
            f"╚═══════════════════════════════════════════════════════════════════════════╝",
            f"",
            f"Symbol: {result.symbol}",
            f"Execution ID: {result.execution_id}",
            f"Status: {status_symbol} {'SUCCESS' if result.success else 'FAILED'}",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"EXECUTION SUMMARY:",
            f"─────────────────",
            f"Total Time:       {result.total_execution_time:.2f} seconds",
            f"Order Placement:  {result.order_placement_time:.3f} seconds",
            f"Fill Monitoring:  {result.fill_monitoring_time:.3f} seconds",
            f"Legs Filled:      {result.legs_filled}/{result.total_legs}",
            f"All Legs Filled:  {'Yes' if result.all_legs_filled else 'No'}",
            f"",
            f"FINANCIAL ANALYSIS:",
            f"──────────────────",
            f"Expected Cost:    ${result.expected_total_cost:.2f}",
            f"Actual Cost:      ${result.actual_total_cost:.2f}",
            f"Total Slippage:   ${result.total_slippage:.2f} ({result.slippage_percentage:.2f}%)",
            f"",
            f"SLIPPAGE BREAKDOWN:",
            f"──────────────────",
        ]

        # Add leg-specific slippage
        for leg_name, slippage in slippage_analysis.leg_slippages.items():
            report_lines.append(f"{leg_name.upper():8}: ${slippage:+.3f}")

        report_lines.extend(
            [
                f"",
                f"Worst Leg:        {slippage_analysis.worst_leg.upper() if slippage_analysis.worst_leg else 'N/A'}",
                f"Worst Slippage:   ${slippage_analysis.worst_leg_slippage:.3f}",
                f"Within Tolerance: {'Yes' if slippage_analysis.slippage_within_tolerance else 'No'} ({slippage_analysis.slippage_tolerance_percent}%)",
            ]
        )

        if slippage_analysis.price_improvement_legs:
            report_lines.append(
                f"Price Improvements: {', '.join(slippage_analysis.price_improvement_legs).upper()}"
            )

        if slippage_analysis.price_deterioration_legs:
            report_lines.append(
                f"Price Deteriorations: {', '.join(slippage_analysis.price_deterioration_legs).upper()}"
            )

        # Add session statistics
        report_lines.extend(
            [
                f"",
                f"SESSION STATISTICS:",
                f"──────────────────",
                f"Total Executions: {self.session_metrics['total_executions']}",
                f"Successful:       {self.session_metrics['successful_executions']} ({performance_metrics.all_legs_fill_rate:.1f}%)",
                f"Average Slippage: ${performance_metrics.average_slippage_dollars:.2f}",
                f"",
                f"═══════════════════════════════════════════════════════════════════════════",
            ]
        )

        return "\n".join(report_lines)

    def print_live_execution_progress(self, symbol: str, execution_id: str) -> None:
        """Print live execution progress (for real-time monitoring)."""

        with Live(
            self._create_progress_layout(symbol, execution_id), refresh_per_second=4
        ) as live:
            # This would be called during execution for live updates
            # For now, just show a static progress display
            time.sleep(2)  # Simulate some execution time

    def _create_progress_layout(self, symbol: str, execution_id: str) -> Layout:
        """Create a live progress layout."""

        layout = Layout()

        # Progress bars for each phase
        progress = Progress()

        task1 = progress.add_task("[cyan]Order Placement", total=100)
        task2 = progress.add_task("[yellow]Fill Monitoring", total=100)
        task3 = progress.add_task("[green]Result Analysis", total=100)

        # Update progress (would be dynamic in real implementation)
        progress.update(task1, completed=100)
        progress.update(task2, completed=65)
        progress.update(task3, completed=0)

        header = Panel(
            f"🚀 Live Execution Progress\nSymbol: {symbol} | Execution: {execution_id}",
            box=box.DOUBLE,
        )

        layout.split_column(Layout(header, size=3), Layout(progress))

        return layout

    def export_session_report(
        self, filename: str, format_type: ReportFormat = ReportFormat.HTML
    ) -> bool:
        """Export complete session report to file."""
        try:
            session_summary = {
                "total_executions": self.session_metrics["total_executions"],
                "successful_executions": self.session_metrics["successful_executions"],
                "success_rate": (
                    self.session_metrics["successful_executions"]
                    / self.session_metrics["total_executions"]
                    * 100
                    if self.session_metrics["total_executions"] > 0
                    else 0.0
                ),
                "total_slippage": self.session_metrics["total_slippage"],
                "average_execution_time": (
                    statistics.mean(self.session_metrics["execution_times"])
                    if self.session_metrics["execution_times"]
                    else 0.0
                ),
                "report_history_count": len(self.report_history),
            }

            if format_type == ReportFormat.JSON:
                import json

                with open(filename, "w") as f:
                    json.dump(session_summary, f, indent=2, default=str)
            else:
                # Default to text format for session summary
                with open(filename, "w") as f:
                    f.write("SFR PARALLEL EXECUTION SESSION REPORT\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(
                        f"Total Executions: {session_summary['total_executions']}\n"
                    )
                    f.write(
                        f"Successful: {session_summary['successful_executions']} ({session_summary['success_rate']:.1f}%)\n"
                    )
                    f.write(
                        f"Total Slippage: ${session_summary['total_slippage']:.2f}\n"
                    )
                    f.write(
                        f"Average Time: {session_summary['average_execution_time']:.2f}s\n"
                    )
                    f.write(
                        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )

            logger.info(f"Session report exported to {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to export session report: {e}")
            return False

    def get_session_statistics(self) -> Dict:
        """Get current session statistics."""
        total = self.session_metrics["total_executions"]

        return {
            "total_executions": total,
            "successful_executions": self.session_metrics["successful_executions"],
            "success_rate_percent": (
                self.session_metrics["successful_executions"] / total * 100
                if total > 0
                else 0.0
            ),
            "total_slippage_dollars": self.session_metrics["total_slippage"],
            "average_slippage_dollars": (
                self.session_metrics["total_slippage"] / total if total > 0 else 0.0
            ),
            "average_execution_time_seconds": (
                statistics.mean(self.session_metrics["execution_times"])
                if self.session_metrics["execution_times"]
                else 0.0
            ),
            "fastest_execution_seconds": (
                min(self.session_metrics["execution_times"])
                if self.session_metrics["execution_times"]
                else 0.0
            ),
            "slowest_execution_seconds": (
                max(self.session_metrics["execution_times"])
                if self.session_metrics["execution_times"]
                else 0.0
            ),
            "reports_generated": len(self.report_history),
        }
