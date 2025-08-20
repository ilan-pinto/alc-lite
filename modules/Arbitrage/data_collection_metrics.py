"""
Data Collection Metrics and Utilities for Progressive Timeout Strategy.

This module provides comprehensive tracking and configuration for the progressive
data collection system used in SFR arbitrage scanning.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


class ContractPriority(Enum):
    """Priority levels for contracts in data collection"""

    CRITICAL = "critical"  # ATM ± 5%, most likely to have arbitrage
    IMPORTANT = "important"  # ATM ± 10%, still viable for arbitrage
    OPTIONAL = "optional"  # Far OTM/ITM, unlikely but possible


class CollectionPhase(Enum):
    """Phases of progressive data collection"""

    INITIALIZING = "initializing"
    PHASE_1_CRITICAL = "phase_1_critical"  # 0-0.5s
    PHASE_2_IMPORTANT = "phase_2_important"  # 0.5-1.5s
    PHASE_3_FINAL = "phase_3_final"  # 1.5-3.0s
    COMPLETED = "completed"
    TIMEOUT = "timeout"


@dataclass
class DataCollectionMetrics:
    """Comprehensive metrics for data collection performance"""

    symbol: str
    start_time: float

    # Phase completion tracking
    phase_1_complete_time: Optional[float] = None
    phase_2_complete_time: Optional[float] = None
    phase_3_complete_time: Optional[float] = None

    # Contract counting by priority
    contracts_expected: Dict[str, int] = field(
        default_factory=lambda: {"critical": 0, "important": 0, "optional": 0}
    )
    contracts_received: Dict[str, int] = field(
        default_factory=lambda: {"critical": 0, "important": 0, "optional": 0}
    )

    # Timing metrics
    time_to_first_data: Optional[float] = None
    time_to_decision: Optional[float] = None
    final_phase: Optional[CollectionPhase] = None

    # Quality metrics
    stale_data_count: int = 0
    invalid_data_count: int = 0
    wide_spread_count: int = 0

    # Decision metrics
    decision_confidence: Optional[float] = None  # % of expected data available
    opportunity_found: bool = False
    execution_triggered: bool = False

    def get_total_expected(self) -> int:
        """Get total number of expected contracts"""
        return sum(self.contracts_expected.values())

    def get_total_received(self) -> int:
        """Get total number of received contracts"""
        return sum(self.contracts_received.values())

    def get_completion_percentage(self) -> float:
        """Get overall data completion percentage"""
        total_expected = self.get_total_expected()
        if total_expected == 0:
            return 0.0
        return (self.get_total_received() / total_expected) * 100

    def get_critical_completion_percentage(self) -> float:
        """Get critical contracts completion percentage"""
        expected = self.contracts_expected["critical"]
        if expected == 0:
            return 0.0
        return (self.contracts_received["critical"] / expected) * 100


@dataclass
class ProgressiveTimeoutConfig:
    """Configuration for progressive timeout strategy"""

    is_market_hours: bool
    total_contracts: int

    # Phase timeouts (seconds)
    phase_1_timeout: float = 0.5  # Critical data
    phase_2_timeout: float = 1.5  # Important data
    phase_3_timeout: float = 3.0  # Final timeout

    # Minimum data thresholds (percentage)
    critical_threshold: float = 0.80  # Need 80% of critical
    important_threshold: float = 0.60  # Need 60% of important

    # Profit thresholds for early execution
    phase_1_profit_threshold: float = 0.50  # 50¢ for immediate execution
    phase_2_profit_threshold: float = 0.20  # 20¢ for phase 2 execution
    phase_3_profit_threshold: float = 0.10  # 10¢ for final phase

    def __post_init__(self):
        """Adjust timeouts based on market conditions"""
        if not self.is_market_hours:
            # Longer timeouts for pre/post market
            self.phase_1_timeout *= 2.0
            self.phase_2_timeout *= 2.0
            self.phase_3_timeout *= 1.5

            # More lenient thresholds after hours
            self.critical_threshold = 0.70
            self.important_threshold = 0.50

    @classmethod
    def create_for_market_conditions(
        cls, is_market_hours: bool, total_contracts: int
    ) -> "ProgressiveTimeoutConfig":
        """Factory method to create config based on market conditions"""
        return cls(is_market_hours=is_market_hours, total_contracts=total_contracts)


class ContractPrioritizer:
    """Utility class for categorizing contracts by priority"""

    @staticmethod
    def categorize_by_moneyness(
        expiry_options, stock_price: float
    ) -> Dict[ContractPriority, List]:
        """Categorize option contracts based on moneyness (distance from ATM)"""
        priority_map = {
            ContractPriority.CRITICAL: [],
            ContractPriority.IMPORTANT: [],
            ContractPriority.OPTIONAL: [],
        }

        # Handle invalid stock price
        if stock_price <= 0:
            # If stock price is invalid, treat all options as optional priority
            for expiry_option in expiry_options:
                priority_map[ContractPriority.OPTIONAL].append(expiry_option)
            return priority_map

        for expiry_option in expiry_options:
            # Calculate moneyness for both call and put strikes
            call_distance = abs(expiry_option.call_strike - stock_price) / stock_price
            put_distance = abs(expiry_option.put_strike - stock_price) / stock_price

            # Use the closer strike to determine priority
            min_distance = min(call_distance, put_distance)

            if min_distance <= 0.05:  # Within 5% of ATM
                priority_map[ContractPriority.CRITICAL].append(expiry_option)
            elif min_distance <= 0.10:  # Within 10% of ATM
                priority_map[ContractPriority.IMPORTANT].append(expiry_option)
            else:  # Far from ATM
                priority_map[ContractPriority.OPTIONAL].append(expiry_option)

        return priority_map

    @staticmethod
    def get_contract_priority(
        contract, expiry_options, stock_price: float
    ) -> ContractPriority:
        """Get priority for a specific contract"""
        priority_map = ContractPrioritizer.categorize_by_moneyness(
            expiry_options, stock_price
        )

        # Find which priority tier this contract belongs to
        for priority, options in priority_map.items():
            for option in options:
                if (
                    contract.conId == option.call_contract.conId
                    or contract.conId == option.put_contract.conId
                ):
                    return priority

        # Default to optional if not found
        return ContractPriority.OPTIONAL


class DataVelocityTracker:
    """Track the velocity of incoming data"""

    def __init__(self, window_size: int = 10):
        self.data_points = []  # (timestamp, contracts_received)
        self.window_size = window_size

    def add_data_point(self, contracts_received: int):
        """Add a new data point"""
        timestamp = time.time()
        self.data_points.append((timestamp, contracts_received))

        # Keep only the most recent points
        if len(self.data_points) > self.window_size:
            self.data_points.pop(0)

    def get_current_velocity(self) -> float:
        """Get current data velocity (contracts/second)"""
        if len(self.data_points) < 2:
            return 0.0

        # Calculate velocity from first to last point
        first_time, first_count = self.data_points[0]
        last_time, last_count = self.data_points[-1]

        time_diff = last_time - first_time
        count_diff = last_count - first_count

        if time_diff <= 0.0001:  # Use small epsilon to handle floating-point precision
            return 0.0

        return count_diff / time_diff

    def estimate_time_to_completion(self, target_contracts: int) -> float:
        """Estimate time to reach target number of contracts"""
        velocity = self.get_current_velocity()
        if velocity <= 0:
            return float("inf")

        current_contracts = self.data_points[-1][1] if self.data_points else 0
        remaining = target_contracts - current_contracts

        if remaining <= 0:
            return 0.0

        return remaining / velocity


def log_phase_transition(
    symbol: str,
    from_phase: CollectionPhase,
    to_phase: CollectionPhase,
    metrics: DataCollectionMetrics,
):
    """Log phase transition with structured data"""
    elapsed = time.time() - metrics.start_time
    completion = metrics.get_completion_percentage()

    logger.info(
        f"[{symbol}] PHASE_TRANSITION: {from_phase.value} -> {to_phase.value} "
        f"| Elapsed: {elapsed:.3f}s | Data: {completion:.1f}% "
        f"({metrics.get_total_received()}/{metrics.get_total_expected()})"
    )


def should_continue_waiting(
    metrics: DataCollectionMetrics,
    config: ProgressiveTimeoutConfig,
    velocity_tracker: DataVelocityTracker,
):
    """
    Determine if we should continue waiting for more data.

    Returns:
        tuple: (should_continue: bool, stop_reason: str)
        - should_continue: True if we should keep waiting, False if we should stop
        - stop_reason: Description of why we should stop (only meaningful when should_continue=False)
    """
    elapsed = time.time() - metrics.start_time
    completion_pct = metrics.get_completion_percentage()

    # Hard timeout check
    if elapsed > config.phase_3_timeout:
        logger.debug(
            f"Hard timeout reached: {elapsed:.1f}s > {config.phase_3_timeout}s"
        )
        return False, "hard_timeout"

    # AGGRESSIVE: Stop immediately if we have massive data overflow
    if completion_pct >= 500:  # 5x expected data is definitely enough
        logger.info(
            f"Data overflow detected: {completion_pct:.0f}% completion - stopping immediately"
        )
        return False, "data_overflow"

    # Check if we have minimum viable data
    critical_pct = metrics.get_critical_completion_percentage()
    if critical_pct >= config.critical_threshold * 100:
        logger.debug(
            f"Critical threshold met: {critical_pct:.1f}% >= {config.critical_threshold * 100}%"
        )
        return False, "critical_threshold_met"

    # BURST HANDLING: Skip velocity checks entirely when we have overflow
    if completion_pct > 200:  # Have 2x+ expected data
        logger.debug(
            f"Data burst detected ({completion_pct:.0f}%) - skipping velocity checks"
        )
        # Only continue if we haven't waited too long AND don't have critical data
        if elapsed > 2.0 or critical_pct >= 50:  # Be more aggressive with time limits
            logger.debug(f"Sufficient data with burst pattern - stopping collection")
            return False, "data_burst_sufficient"
        return True, ""  # Continue only briefly to see if critical data arrives

    # LENIENT VELOCITY CHECKS: Only apply when we don't have overflow
    velocity = velocity_tracker.get_current_velocity()

    # Give the system time to establish velocity (avoid premature velocity failures)
    if elapsed < 3.0:
        return True, ""  # Always wait at least 3 seconds for velocity to stabilize

    # Very lenient velocity thresholds after initial period
    min_velocity_threshold = 0.5  # Only 0.5 contracts per second minimum

    if completion_pct > 100:
        # Already have enough data, use very lenient threshold
        if velocity < 0.2 and elapsed > 4.0:  # Almost no velocity after 4 seconds
            logger.debug(
                f"Very poor velocity with sufficient data: {velocity:.2f} contracts/s"
            )
            return False, "poor_velocity_sufficient_data"
    elif completion_pct > 50:
        # Have decent data, still be lenient
        if velocity < min_velocity_threshold and elapsed > 3.5:
            logger.debug(f"Poor velocity with decent data: {velocity:.2f} contracts/s")
            return False, "poor_velocity_decent_data"
    else:
        # Limited data, wait longer with lenient threshold
        if velocity < min_velocity_threshold and elapsed > 4.5:
            logger.debug(
                f"Poor velocity with limited data: {velocity:.2f} contracts/s after {elapsed:.1f}s"
            )
            return False, "poor_velocity_limited_data"

    # Estimate if we'll get more data soon (but be more optimistic)
    total_expected = metrics.get_total_expected()
    time_to_completion = velocity_tracker.estimate_time_to_completion(total_expected)

    # More generous time thresholds based on current completion
    if completion_pct < 25:
        max_wait_time = 15.0  # Wait longer if we have very little data
    elif completion_pct < 50:
        max_wait_time = 10.0
    else:
        max_wait_time = 6.0

    if time_to_completion > max_wait_time:
        logger.debug(
            f"Estimated completion time too long: {time_to_completion:.1f}s > {max_wait_time}s"
        )
        return False, "estimated_completion_too_long"

    return True, ""
