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
from dataclasses import asdict, dataclass, field
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
class OpportunityFunnelMetrics:
    """Track opportunities through the evaluation funnel"""

    symbol: str
    expiry: str

    # Funnel stages (cumulative counts)
    evaluated: int = 0
    stock_ticker_available: int = 0
    passed_priority_filter: int = 0
    passed_viability_check: int = 0
    option_data_available: int = 0
    passed_data_quality: int = 0
    prices_valid: int = 0
    theoretical_profit_positive: int = 0
    guaranteed_profit_positive: int = 0
    executed: int = 0

    # Profit tracking
    theoretical_profits: List[float] = field(default_factory=list)
    guaranteed_profits: List[float] = field(default_factory=list)


@dataclass
class StrikeEffectivenessMetrics:
    """Track strike selection effectiveness and success rates"""

    symbol: str

    # Strike availability
    total_strikes_available: int = 0
    valid_strikes_found: int = 0

    # Combination generation
    combinations_generated: int = 0
    combinations_tested: int = 0

    # Success by strike difference
    success_by_strike_diff: Dict[int, int] = field(
        default_factory=dict
    )  # {1: 2, 2: 1, 3: 0, ...}
    rejection_by_strike_diff: Dict[int, Dict[str, int]] = field(
        default_factory=dict
    )  # strike_diff -> reason -> count
    profit_by_strike_diff: Dict[int, List[float]] = field(
        default_factory=dict
    )  # strike_diff -> list of profits


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

        # New enhanced metrics
        self.funnel_metrics: Dict[str, OpportunityFunnelMetrics] = (
            {}
        )  # key: symbol_expiry
        self.strike_metrics: Dict[str, StrikeEffectivenessMetrics] = {}  # key: symbol

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
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: {contract_type} spread {spread:.2f} > {threshold}"
                )
            elif reason == RejectionReason.PRICE_LIMIT_EXCEEDED:
                limit_price = details.get("combo_limit_price", 0) if details else 0
                cost_limit = details.get("cost_limit", 0) if details else 0
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: limit price {limit_price:.2f} > cost limit {cost_limit:.2f}"
                )
            elif reason == RejectionReason.PROFIT_TARGET_NOT_MET:
                profit_target = details.get("profit_target", 0) if details else 0
                min_roi = details.get("min_roi", 0) if details else 0
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: target {profit_target:.2f}% > actual ROI {min_roi:.2f}%"
                )
            elif reason == RejectionReason.MAX_LOSS_THRESHOLD_EXCEEDED:
                max_loss_threshold = (
                    details.get("max_loss_threshold", 0) if details else 0
                )
                min_profit = details.get("min_profit", 0) if details else 0
                profit_ratio = details.get("profit_ratio", 0) if details else 0
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: max loss {max_loss_threshold:.2f} >= calculated loss {min_profit:.2f} (profit ratio: {profit_ratio:.2f})"
                )
            elif reason == RejectionReason.MAX_PROFIT_THRESHOLD_NOT_MET:
                max_profit_threshold = (
                    details.get("max_profit_threshold", 0) if details else 0
                )
                max_profit = details.get("max_profit", 0) if details else 0
                profit_ratio = details.get("profit_ratio", 0) if details else 0
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: threshold {max_profit_threshold:.2f} < max profit {max_profit:.2f} (profit ratio: {profit_ratio:.2f})"
                )
            elif reason == RejectionReason.PROFIT_RATIO_THRESHOLD_NOT_MET:
                profit_ratio_threshold = (
                    details.get("profit_ratio_threshold", 0) if details else 0
                )
                max_profit = details.get("max_profit", 0) if details else 0
                min_profit = details.get("min_profit", 0) if details else 0
                actual_ratio = max_profit / abs(min_profit) if min_profit != 0 else 0
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: threshold {profit_ratio_threshold:.2f} > actual ratio {actual_ratio:.2f}"
                )
            elif reason == RejectionReason.INSUFFICIENT_VALID_STRIKES:
                count = details.get("valid_strikes_count", 0) if details else 0
                required = details.get("required_strikes", 0) if details else 0
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: found {count} strikes, need {required}"
                )
            elif reason == RejectionReason.INVALID_STRIKE_COMBINATION:
                call_strike = details.get("call_strike", 0) if details else 0
                put_strike = details.get("put_strike", 0) if details else 0
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: call {call_strike} vs put {put_strike}"
                )
            elif reason == RejectionReason.MISSING_MARKET_DATA:
                contract_type = details.get("contract_type", "") if details else ""
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: missing {contract_type} data"
                )
            elif reason == RejectionReason.ARBITRAGE_CONDITION_NOT_MET:
                spread = (
                    details.get("theoretical_spread", details.get("spread", 0))
                    if details
                    else 0
                )
                net_credit = (
                    details.get("theoretical_net_credit", details.get("net_credit", 0))
                    if details
                    else 0
                )
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: spread {spread:.2f} > net credit {net_credit:.2f}"
                )
            elif reason == RejectionReason.NET_CREDIT_NEGATIVE:
                net_credit = details.get("net_credit", 0) if details else 0
                stock_price = details.get("stock_price", 0) if details else 0
                spread = details.get("spread", 0) if details else 0
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: net credit {net_credit:.2f} < 0 (stock: {stock_price:.2f}, spread: {spread:.2f})"
                )
            elif reason == RejectionReason.ORDER_NOT_FILLED:
                order_id = details.get("order_id", "") if details else ""
                timeout = details.get("timeout_seconds", 0) if details else 0
                filled_qty = details.get("filled_quantity", 0) if details else 0
                total_qty = details.get("total_quantity", 0) if details else 0
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: order {order_id} not filled within {timeout}s (filled: {filled_qty}/{total_qty})"
                )
            elif reason == RejectionReason.ORDER_REJECTED:
                order_id = details.get("order_id", "") if details else ""
                reject_reason = details.get("reject_reason", "") if details else ""
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: order {order_id} rejected - {reject_reason}"
                )
            elif reason == RejectionReason.NO_VALID_EXPIRIES:
                available = details.get("available_expiries", 0) if details else 0
                days_range = details.get("days_range", "N/A") if details else "N/A"
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: found {available} expiries, need {days_range} days"
                )
            elif reason == RejectionReason.INVALID_CONTRACT_DATA:
                contract_type = (
                    details.get("contract_type", "contract") if details else "contract"
                )
                expiry = details.get("expiry", "N/A") if details else "N/A"
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: invalid {contract_type} data (expiry: {expiry})"
                )
            elif reason == RejectionReason.DATA_COLLECTION_TIMEOUT:
                timeout = details.get("timeout_seconds", 0) if details else 0
                received = details.get("contracts_received", 0) if details else 0
                expected = details.get("contracts_expected", 0) if details else 0
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: timeout after {timeout}s ({received}/{expected} contracts)"
                )
            elif reason == RejectionReason.VOLUME_TOO_LOW:
                volume = details.get("volume", 0) if details else 0
                min_volume = details.get("min_volume", 0) if details else 0
                contract_type = (
                    details.get("contract_type", "contract") if details else "contract"
                )
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: {contract_type} volume {volume} < minimum {min_volume}"
                )
            elif (
                reason == RejectionReason.LIQUIDITY_INSUFFICIENT
                or reason == RejectionReason.INSUFFICIENT_LIQUIDITY
            ):
                contract_type = (
                    details.get("contract_type", "contract") if details else "contract"
                )
                expiry = details.get("expiry", "N/A") if details else "N/A"
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: insufficient {contract_type} liquidity (expiry: {expiry})"
                )
            elif reason == RejectionReason.SPREAD_TOO_WIDE:
                spread = details.get("spread", 0) if details else 0
                threshold = details.get("threshold", 0) if details else 0
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: spread {spread:.2f} > threshold {threshold:.2f}"
                )
            elif reason == RejectionReason.MIN_ROI_NOT_MET:
                min_roi = details.get("min_roi", 0) if details else 0
                actual_roi = details.get("actual_roi", 0) if details else 0
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: minimum ROI {min_roi:.2f}% > actual ROI {actual_roi:.2f}%"
                )
            elif reason == RejectionReason.INSUFFICIENT_IV_SPREAD:
                iv_spread = details.get("iv_spread", 0) if details else 0
                min_iv_spread = details.get("min_iv_spread", 0) if details else 0
                logger.warning(
                    f"[{symbol}] REJECTED - {reason_text}: IV spread {iv_spread:.2f} < minimum {min_iv_spread:.2f}"
                )
            else:
                # Generic rejection message with additional details if available
                detail_parts = []
                if details:
                    if "expiry" in details:
                        detail_parts.append(f"expiry: {details['expiry']}")
                    if "strike" in details:
                        detail_parts.append(f"strike: {details['strike']}")
                    if "contract_type" in details:
                        detail_parts.append(f"type: {details['contract_type']}")

                detail_str = f" ({', '.join(detail_parts)})" if detail_parts else ""
                logger.warning(f"[{symbol}] REJECTED - {reason_text}{detail_str}")

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

    def record_opportunity_found(self, symbol: Optional[str] = None) -> None:
        """Record that an opportunity was found"""
        # If symbol is provided, find the specific scan for that symbol
        if symbol:
            found_scan = False

            # First, search in completed scans (in reverse order for most recent)
            for scan in reversed(self.scan_metrics):
                if scan.symbol == symbol:
                    scan.opportunities_found += 1
                    found_scan = True
                    break

            # If not found in completed scans, check current scan
            if (
                not found_scan
                and self.current_scan
                and self.current_scan.symbol == symbol
            ):
                self.current_scan.opportunities_found += 1
                found_scan = True

            # If still not found, create a scan entry to record the opportunity
            if not found_scan:
                # As a last resort, create a scan entry if none exists
                temp_scan = ScanMetrics(
                    symbol=symbol, strategy="SFR", scan_start_time=time.time()
                )
                temp_scan.opportunities_found = 1
                temp_scan.finish(success=True)
                self.scan_metrics.append(temp_scan)
                found_scan = True
        else:
            # Fallback to current scan behavior
            if self.current_scan:
                self.current_scan.opportunities_found += 1

        self.increment_counter("total_opportunities_found")

    def record_expiries_scanned(self, count: int) -> None:
        """Record number of expiries scanned"""
        if self.current_scan:
            self.current_scan.expiries_scanned = count

    def get_or_create_funnel_metrics(
        self, symbol: str, expiry: str
    ) -> OpportunityFunnelMetrics:
        """Get or create funnel metrics for a symbol-expiry combination"""
        key = f"{symbol}_{expiry}"
        if key not in self.funnel_metrics:
            self.funnel_metrics[key] = OpportunityFunnelMetrics(
                symbol=symbol, expiry=expiry
            )
        return self.funnel_metrics[key]

    def get_or_create_strike_metrics(self, symbol: str) -> StrikeEffectivenessMetrics:
        """Get or create strike effectiveness metrics for a symbol"""
        if symbol not in self.strike_metrics:
            self.strike_metrics[symbol] = StrikeEffectivenessMetrics(symbol=symbol)
        return self.strike_metrics[symbol]

    def record_funnel_stage(
        self, symbol: str, expiry: str, stage: str, value: int = 1
    ) -> None:
        """Record progression through funnel stages"""
        metrics = self.get_or_create_funnel_metrics(symbol, expiry)
        if hasattr(metrics, stage):
            setattr(metrics, stage, getattr(metrics, stage) + value)

    def record_profit_calculation(
        self,
        symbol: str,
        expiry: str,
        theoretical_profit: float,
        guaranteed_profit: float = None,
    ) -> None:
        """Record all profit calculations for distribution analysis"""
        metrics = self.get_or_create_funnel_metrics(symbol, expiry)
        metrics.theoretical_profits.append(theoretical_profit)
        if guaranteed_profit is not None:
            metrics.guaranteed_profits.append(guaranteed_profit)

    def record_strike_effectiveness(
        self,
        symbol: str,
        total_strikes: int,
        valid_strikes: int,
        combinations_generated: int,
        combinations_tested: int,
    ) -> None:
        """Record strike selection effectiveness"""
        metrics = self.get_or_create_strike_metrics(symbol)
        metrics.total_strikes_available = total_strikes
        metrics.valid_strikes_found = valid_strikes
        metrics.combinations_generated = combinations_generated
        metrics.combinations_tested = combinations_tested

    def record_strike_success(
        self, symbol: str, strike_diff: int, profitable: bool, profit: float = None
    ) -> None:
        """Record success/failure for specific strike differences"""
        metrics = self.get_or_create_strike_metrics(symbol)

        if strike_diff not in metrics.success_by_strike_diff:
            metrics.success_by_strike_diff[strike_diff] = 0
        if strike_diff not in metrics.profit_by_strike_diff:
            metrics.profit_by_strike_diff[strike_diff] = []

        if profitable:
            metrics.success_by_strike_diff[strike_diff] += 1
            if profit is not None:
                metrics.profit_by_strike_diff[strike_diff].append(profit)

    def record_strike_rejection(
        self, symbol: str, strike_diff: int, reason: str
    ) -> None:
        """Record rejection for specific strike differences"""
        metrics = self.get_or_create_strike_metrics(symbol)

        if strike_diff not in metrics.rejection_by_strike_diff:
            metrics.rejection_by_strike_diff[strike_diff] = {}
        if reason not in metrics.rejection_by_strike_diff[strike_diff]:
            metrics.rejection_by_strike_diff[strike_diff][reason] = 0

        metrics.rejection_by_strike_diff[strike_diff][reason] += 1

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

    def get_funnel_analysis(self) -> Dict[str, Any]:
        """Get detailed funnel analysis showing dropout rates at each stage"""
        if not self.funnel_metrics:
            return {
                "total_opportunities": 0,
                "funnel_stages": {},
                "conversion_rates": {},
            }

        # Aggregate funnel metrics across all symbol-expiry combinations
        stage_totals = {
            "evaluated": 0,
            "stock_ticker_available": 0,
            "passed_priority_filter": 0,
            "passed_viability_check": 0,
            "option_data_available": 0,
            "passed_data_quality": 0,
            "prices_valid": 0,
            "theoretical_profit_positive": 0,
            "guaranteed_profit_positive": 0,
            "executed": 0,
        }

        total_theoretical_profits = []
        total_guaranteed_profits = []

        for funnel_metric in self.funnel_metrics.values():
            for stage in stage_totals.keys():
                stage_totals[stage] += getattr(funnel_metric, stage, 0)
            total_theoretical_profits.extend(funnel_metric.theoretical_profits)
            total_guaranteed_profits.extend(funnel_metric.guaranteed_profits)

        # Calculate conversion rates between stages
        conversion_rates = {}
        stage_names = list(stage_totals.keys())
        for i in range(len(stage_names) - 1):
            current_stage = stage_names[i]
            next_stage = stage_names[i + 1]
            current_count = stage_totals[current_stage]
            next_count = stage_totals[next_stage]

            if current_count > 0:
                conversion_rates[f"{current_stage}_to_{next_stage}"] = (
                    next_count / current_count
                ) * 100
            else:
                conversion_rates[f"{current_stage}_to_{next_stage}"] = 0.0

        # Calculate profit distribution statistics
        profit_stats = {}
        if total_theoretical_profits:
            profit_stats["theoretical"] = {
                "count": len(total_theoretical_profits),
                "positive_count": len([p for p in total_theoretical_profits if p > 0]),
                "negative_count": len([p for p in total_theoretical_profits if p <= 0]),
                "avg": sum(total_theoretical_profits) / len(total_theoretical_profits),
                "max": max(total_theoretical_profits),
                "min": min(total_theoretical_profits),
            }

        if total_guaranteed_profits:
            profit_stats["guaranteed"] = {
                "count": len(total_guaranteed_profits),
                "positive_count": len([p for p in total_guaranteed_profits if p > 0]),
                "negative_count": len([p for p in total_guaranteed_profits if p <= 0]),
                "avg": sum(total_guaranteed_profits) / len(total_guaranteed_profits),
                "max": max(total_guaranteed_profits),
                "min": min(total_guaranteed_profits),
            }

        return {
            "total_opportunities": stage_totals["evaluated"],
            "funnel_stages": stage_totals,
            "conversion_rates": conversion_rates,
            "profit_distribution": profit_stats,
        }

    def get_strike_effectiveness_analysis(self) -> Dict[str, Any]:
        """Get detailed strike effectiveness analysis"""
        if not self.strike_metrics:
            return {
                "total_symbols": 0,
                "strike_performance": {},
                "effectiveness_summary": {},
            }

        # Aggregate strike metrics across all symbols
        total_strikes_available = 0
        total_valid_strikes = 0
        total_combinations_generated = 0
        total_combinations_tested = 0

        aggregated_success_by_diff = {}
        aggregated_profit_by_diff = {}
        aggregated_rejection_by_diff = {}

        for strike_metric in self.strike_metrics.values():
            total_strikes_available += strike_metric.total_strikes_available
            total_valid_strikes += strike_metric.valid_strikes_found
            total_combinations_generated += strike_metric.combinations_generated
            total_combinations_tested += strike_metric.combinations_tested

            # Aggregate success by strike difference
            for diff, count in strike_metric.success_by_strike_diff.items():
                aggregated_success_by_diff[diff] = (
                    aggregated_success_by_diff.get(diff, 0) + count
                )

            # Aggregate profits by strike difference
            for diff, profits in strike_metric.profit_by_strike_diff.items():
                if diff not in aggregated_profit_by_diff:
                    aggregated_profit_by_diff[diff] = []
                aggregated_profit_by_diff[diff].extend(profits)

            # Aggregate rejections by strike difference
            for diff, rejections in strike_metric.rejection_by_strike_diff.items():
                if diff not in aggregated_rejection_by_diff:
                    aggregated_rejection_by_diff[diff] = {}
                for reason, count in rejections.items():
                    aggregated_rejection_by_diff[diff][reason] = (
                        aggregated_rejection_by_diff[diff].get(reason, 0) + count
                    )

        # Calculate effectiveness metrics by strike difference
        strike_performance = {}
        for diff in sorted(
            set(
                list(aggregated_success_by_diff.keys())
                + list(aggregated_profit_by_diff.keys())
                + list(aggregated_rejection_by_diff.keys())
            )
        ):

            success_count = aggregated_success_by_diff.get(diff, 0)
            profits = aggregated_profit_by_diff.get(diff, [])
            rejections = aggregated_rejection_by_diff.get(diff, {})
            total_attempts = success_count + sum(rejections.values())

            performance = {
                "success_count": success_count,
                "total_attempts": total_attempts,
                "success_rate": (
                    (success_count / total_attempts * 100) if total_attempts > 0 else 0
                ),
                "avg_profit": sum(profits) / len(profits) if profits else 0,
                "max_profit": max(profits) if profits else 0,
                "rejection_breakdown": rejections,
            }
            strike_performance[f"{diff}_strike_difference"] = performance

        effectiveness_summary = {
            "total_strikes_in_chains": total_strikes_available,
            "valid_strikes_selected": total_valid_strikes,
            "strike_selection_rate": (
                (total_valid_strikes / total_strikes_available * 100)
                if total_strikes_available > 0
                else 0
            ),
            "combinations_generated": total_combinations_generated,
            "combinations_tested": total_combinations_tested,
            "combination_test_rate": (
                (total_combinations_tested / total_combinations_generated * 100)
                if total_combinations_generated > 0
                else 0
            ),
        }

        return {
            "total_symbols": len(self.strike_metrics),
            "strike_performance": strike_performance,
            "effectiveness_summary": effectiveness_summary,
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
            "funnel_analysis": self.get_funnel_analysis(),
            "strike_effectiveness": self.get_strike_effectiveness_analysis(),
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

        # Reset new enhanced metrics
        self.funnel_metrics.clear()
        self.strike_metrics.clear()

        # Reset all counters
        for counter in self.counters.values():
            counter.reset()


# Global metrics collector instance
metrics_collector = MetricsCollector()
