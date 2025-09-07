"""
Sophisticated partial fill handler for SFR parallel execution.

This module handles the complex scenarios where some legs of the SFR arbitrage
get filled while others don't, implementing intelligent strategies to:

1. Detect partial fill scenarios quickly
2. Assess risk and opportunity cost of partial positions
3. Attempt completion of unfilled legs with aggressive pricing
4. Coordinate with rollback manager for position unwinding
5. Track partial fill patterns for strategy optimization
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from ib_async import IB, Contract, Order, OrderStatus, Trade

from ..common import get_logger
from .constants import (
    PARTIAL_FILL_AGGRESSIVE_TIMEOUT,
    PARTIAL_FILL_COMPLETION_ATTEMPTS,
    PARTIAL_FILL_MAX_SLIPPAGE,
)
from .parallel_execution_framework import LegOrder, LegType, ParallelExecutionPlan

logger = get_logger()


class PartialFillStrategy(Enum):
    """Strategies for handling partial fills."""

    IMMEDIATE_ROLLBACK = "immediate_rollback"
    AGGRESSIVE_COMPLETION = "aggressive_completion"
    WAIT_AND_RETRY = "wait_and_retry"
    MARKET_ORDER_COMPLETION = "market_order_completion"


@dataclass
class PartialFillAssessment:
    """Assessment of partial fill situation."""

    filled_legs: List[LegOrder]
    unfilled_legs: List[LegOrder]
    filled_count: int
    unfilled_count: int

    # Risk assessment
    current_exposure: float  # Dollar exposure from filled legs
    max_potential_loss: float  # Worst case scenario loss
    completion_probability: float  # Estimated probability of completing unfilled legs

    # Recommended strategy
    recommended_strategy: PartialFillStrategy
    strategy_reasoning: str

    # Timing factors
    time_since_placement: float
    market_conditions_favorable: bool
    liquidity_assessment: str  # "high", "medium", "low"


@dataclass
class CompletionAttempt:
    """Record of an attempt to complete unfilled legs."""

    attempt_id: str
    timestamp: float
    unfilled_legs: List[LegOrder]
    strategy_used: PartialFillStrategy

    # Pricing adjustments
    original_prices: Dict[str, float]
    adjusted_prices: Dict[str, float]
    max_slippage_allowed: float

    # Results
    success: bool = False
    legs_completed: int = 0
    total_slippage: float = 0.0
    completion_time: float = 0.0
    error_message: Optional[str] = None


class PartialFillHandler:
    """
    Sophisticated handler for partial fill scenarios in SFR arbitrage.

    This handler provides:
    1. Rapid assessment of partial fill situations
    2. Multiple strategies for handling unfilled legs
    3. Risk-based decision making for completion vs. rollback
    4. Performance tracking and optimization
    5. Integration with rollback manager
    """

    def __init__(self, ib: IB, symbol: str):
        self.ib = ib
        self.symbol = symbol
        self.handler_id = f"partial_fill_{symbol}_{int(time.time())}"

        # Tracking
        self.active_assessments: Dict[str, PartialFillAssessment] = {}
        self.completion_attempts: List[CompletionAttempt] = []
        self.performance_metrics: Dict[str, float] = {
            "total_partial_fills": 0,
            "successful_completions": 0,
            "immediate_rollbacks": 0,
            "aggressive_completions": 0,
            "average_completion_time": 0.0,
            "average_slippage_on_completion": 0.0,
        }

        logger.debug(f"[{symbol}] PartialFillHandler initialized: {self.handler_id}")

    async def initialize(self) -> bool:
        """Initialize the partial fill handler."""
        try:
            logger.info(f"[{self.symbol}] Partial fill handler initialized")
            return True
        except Exception as e:
            logger.error(
                f"[{self.symbol}] Failed to initialize partial fill handler: {e}"
            )
            return False

    async def assess_partial_fill_situation(
        self,
        plan: ParallelExecutionPlan,
        filled_legs: List[LegOrder],
        unfilled_legs: List[LegOrder],
        execution_start_time: float,
    ) -> PartialFillAssessment:
        """
        Assess a partial fill situation and recommend strategy.

        Args:
            plan: The execution plan
            filled_legs: List of legs that have been filled
            unfilled_legs: List of legs that haven't been filled
            execution_start_time: When execution started

        Returns:
            PartialFillAssessment with analysis and recommendations
        """
        self.performance_metrics["total_partial_fills"] += 1

        assessment_start = time.time()
        time_since_placement = assessment_start - execution_start_time

        logger.info(
            f"[{self.symbol}] Assessing partial fill: "
            f"{len(filled_legs)}/{len(filled_legs) + len(unfilled_legs)} legs filled "
            f"after {time_since_placement:.1f}s"
        )

        # Calculate current exposure from filled legs
        current_exposure = self._calculate_current_exposure(filled_legs)

        # Assess market conditions for unfilled legs
        liquidity_assessment = await self._assess_liquidity(unfilled_legs)
        market_favorable = await self._assess_market_conditions(unfilled_legs)

        # Calculate completion probability based on multiple factors
        completion_probability = self._estimate_completion_probability(
            unfilled_legs, time_since_placement, liquidity_assessment, market_favorable
        )

        # Calculate maximum potential loss
        max_potential_loss = self._calculate_max_potential_loss(
            filled_legs, unfilled_legs
        )

        # Determine recommended strategy
        strategy, reasoning = self._determine_optimal_strategy(
            filled_legs,
            unfilled_legs,
            current_exposure,
            max_potential_loss,
            completion_probability,
            time_since_placement,
            market_favorable,
        )

        assessment = PartialFillAssessment(
            filled_legs=filled_legs,
            unfilled_legs=unfilled_legs,
            filled_count=len(filled_legs),
            unfilled_count=len(unfilled_legs),
            current_exposure=current_exposure,
            max_potential_loss=max_potential_loss,
            completion_probability=completion_probability,
            recommended_strategy=strategy,
            strategy_reasoning=reasoning,
            time_since_placement=time_since_placement,
            market_conditions_favorable=market_favorable,
            liquidity_assessment=liquidity_assessment,
        )

        # Store assessment for tracking
        self.active_assessments[plan.execution_id] = assessment

        logger.info(
            f"[{self.symbol}] Assessment complete: Strategy={strategy.value}, "
            f"Exposure=${current_exposure:.2f}, MaxLoss=${max_potential_loss:.2f}, "
            f"CompletionProb={completion_probability:.1%}"
        )

        return assessment

    async def execute_completion_strategy(
        self, assessment: PartialFillAssessment, plan: ParallelExecutionPlan
    ) -> CompletionAttempt:
        """
        Execute the recommended completion strategy.

        Args:
            assessment: The partial fill assessment
            plan: The execution plan

        Returns:
            CompletionAttempt with results
        """
        attempt_id = f"completion_{int(time.time())}"
        attempt_start = time.time()

        logger.info(
            f"[{self.symbol}] Executing completion strategy: {assessment.recommended_strategy.value}"
        )

        # Create attempt record
        attempt = CompletionAttempt(
            attempt_id=attempt_id,
            timestamp=attempt_start,
            unfilled_legs=assessment.unfilled_legs.copy(),
            strategy_used=assessment.recommended_strategy,
            original_prices={
                leg.leg_type.value: leg.target_price for leg in assessment.unfilled_legs
            },
            adjusted_prices={},
            max_slippage_allowed=PARTIAL_FILL_MAX_SLIPPAGE,
        )

        try:
            if (
                assessment.recommended_strategy
                == PartialFillStrategy.AGGRESSIVE_COMPLETION
            ):
                result = await self._execute_aggressive_completion(attempt, assessment)
            elif assessment.recommended_strategy == PartialFillStrategy.WAIT_AND_RETRY:
                result = await self._execute_wait_and_retry(attempt, assessment)
            elif (
                assessment.recommended_strategy
                == PartialFillStrategy.MARKET_ORDER_COMPLETION
            ):
                result = await self._execute_market_completion(attempt, assessment)
            else:  # IMMEDIATE_ROLLBACK
                result = await self._execute_immediate_rollback(attempt, assessment)

            attempt.success = result.get("success", False)
            attempt.legs_completed = result.get("legs_completed", 0)
            attempt.total_slippage = result.get("total_slippage", 0.0)
            attempt.completion_time = time.time() - attempt_start
            attempt.error_message = result.get("error_message")

            # Update performance metrics
            if attempt.success:
                self.performance_metrics["successful_completions"] += 1
                if (
                    assessment.recommended_strategy
                    == PartialFillStrategy.AGGRESSIVE_COMPLETION
                ):
                    self.performance_metrics["aggressive_completions"] += 1
            else:
                if (
                    assessment.recommended_strategy
                    == PartialFillStrategy.IMMEDIATE_ROLLBACK
                ):
                    self.performance_metrics["immediate_rollbacks"] += 1

            # Update averages
            self._update_performance_averages(attempt)

        except Exception as e:
            attempt.success = False
            attempt.error_message = str(e)
            logger.error(f"[{self.symbol}] Error in completion strategy: {e}")

        # Store attempt
        self.completion_attempts.append(attempt)

        logger.info(
            f"[{self.symbol}] Completion attempt finished: "
            f"Success={attempt.success}, Completed={attempt.legs_completed}, "
            f"Time={attempt.completion_time:.2f}s"
        )

        return attempt

    def _calculate_current_exposure(self, filled_legs: List[LegOrder]) -> float:
        """Calculate dollar exposure from filled legs."""
        exposure = 0.0

        for leg in filled_legs:
            if leg.fill_status == "filled":
                if leg.leg_type == LegType.STOCK:
                    # Stock: positive exposure (we bought)
                    exposure += leg.avg_fill_price * leg.filled_quantity
                elif leg.leg_type == LegType.CALL:
                    # Call: negative exposure (we sold, received premium)
                    exposure -= leg.avg_fill_price * leg.filled_quantity
                elif leg.leg_type == LegType.PUT:
                    # Put: positive exposure (we bought)
                    exposure += leg.avg_fill_price * leg.filled_quantity

        return exposure

    def _calculate_max_potential_loss(
        self, filled_legs: List[LegOrder], unfilled_legs: List[LegOrder]
    ) -> float:
        """Calculate maximum potential loss if unfilled legs can't be completed."""
        # This is a simplified calculation - in production would be more sophisticated
        current_exposure = self._calculate_current_exposure(filled_legs)

        # For SFR, the max loss would typically be the premium paid for protective puts
        # plus any adverse stock price movements
        max_loss = abs(current_exposure) * 0.1  # Rough estimate: 10% of exposure

        return max_loss

    async def _assess_liquidity(self, unfilled_legs: List[LegOrder]) -> str:
        """Assess liquidity conditions for unfilled legs."""
        # In production, this would check bid-ask spreads, volume, etc.
        # For now, return simplified assessment
        return "medium"  # "high", "medium", "low"

    async def _assess_market_conditions(self, unfilled_legs: List[LegOrder]) -> bool:
        """Assess if market conditions are favorable for completion."""
        # In production, this would check market volatility, time of day, etc.
        # For now, return simplified assessment
        current_hour = datetime.now().hour
        # More favorable during main trading hours
        return 10 <= current_hour <= 15

    def _estimate_completion_probability(
        self,
        unfilled_legs: List[LegOrder],
        time_elapsed: float,
        liquidity: str,
        market_favorable: bool,
    ) -> float:
        """Estimate probability of completing unfilled legs."""
        base_probability = 0.7  # Base 70% chance

        # Adjust based on time elapsed
        if time_elapsed > 10:  # After 10 seconds, probability drops
            base_probability *= 0.8
        elif time_elapsed > 30:  # After 30 seconds, significant drop
            base_probability *= 0.5

        # Adjust based on liquidity
        liquidity_multipliers = {"high": 1.2, "medium": 1.0, "low": 0.7}
        base_probability *= liquidity_multipliers.get(liquidity, 1.0)

        # Adjust based on market conditions
        if market_favorable:
            base_probability *= 1.1
        else:
            base_probability *= 0.9

        # Adjust based on number of unfilled legs
        if len(unfilled_legs) == 1:
            base_probability *= 1.2  # Easier to fill one leg
        elif len(unfilled_legs) >= 2:
            base_probability *= 0.8  # Harder to fill multiple legs

        return min(base_probability, 0.95)  # Cap at 95%

    def _determine_optimal_strategy(
        self,
        filled_legs: List[LegOrder],
        unfilled_legs: List[LegOrder],
        exposure: float,
        max_loss: float,
        completion_prob: float,
        time_elapsed: float,
        market_favorable: bool,
    ) -> Tuple[PartialFillStrategy, str]:
        """Determine the optimal strategy for handling partial fills."""

        # Decision logic based on multiple factors

        # If very high completion probability and low elapsed time, try aggressive completion
        if completion_prob > 0.8 and time_elapsed < 5.0 and len(unfilled_legs) <= 2:
            return (
                PartialFillStrategy.AGGRESSIVE_COMPLETION,
                f"High completion probability ({completion_prob:.1%}) with recent placement",
            )

        # If moderate exposure and decent completion probability, wait and retry
        if abs(exposure) < 1000 and completion_prob > 0.6 and time_elapsed < 15.0:
            return (
                PartialFillStrategy.WAIT_AND_RETRY,
                f"Moderate exposure (${exposure:.2f}) with decent completion chance",
            )

        # If only one unfilled leg and market favorable, try market order
        if len(unfilled_legs) == 1 and market_favorable and completion_prob > 0.5:
            return (
                PartialFillStrategy.MARKET_ORDER_COMPLETION,
                "Single unfilled leg with favorable conditions",
            )

        # Default to immediate rollback for risk management
        return (
            PartialFillStrategy.IMMEDIATE_ROLLBACK,
            f"Risk management: exposure=${exposure:.2f}, maxLoss=${max_loss:.2f}, "
            f"completionProb={completion_prob:.1%}",
        )

    async def _execute_aggressive_completion(
        self, attempt: CompletionAttempt, assessment: PartialFillAssessment
    ) -> Dict:
        """Execute aggressive completion strategy with adjusted pricing."""
        logger.info(f"[{self.symbol}] Executing aggressive completion")

        # Adjust prices more aggressively for quicker fills
        for leg in attempt.unfilled_legs:
            if leg.leg_type == LegType.STOCK:
                # For stock, increase bid price slightly
                adjusted_price = leg.target_price * 1.002  # 0.2% higher
            elif leg.leg_type == LegType.CALL and leg.action == "SELL":
                # For selling calls, lower ask price slightly
                adjusted_price = leg.target_price * 0.998  # 0.2% lower
            elif leg.leg_type == LegType.PUT and leg.action == "BUY":
                # For buying puts, increase bid price slightly
                adjusted_price = leg.target_price * 1.002  # 0.2% higher
            else:
                adjusted_price = leg.target_price

            attempt.adjusted_prices[leg.leg_type.value] = adjusted_price

            # Cancel existing order and place new one with adjusted price
            try:
                if leg.trade:
                    self.ib.cancelOrder(leg.order)
                    await asyncio.sleep(0.1)  # Brief pause

                # Create new order with adjusted price
                new_order = Order(
                    orderId=self.ib.client.getReqId(),
                    orderType="LMT",
                    action=leg.action,
                    totalQuantity=leg.quantity,
                    lmtPrice=adjusted_price,
                    tif="DAY",
                )

                # Place new order
                new_trade = self.ib.placeOrder(leg.contract, new_order)
                leg.trade = new_trade
                leg.order = new_order

                logger.debug(
                    f"[{self.symbol}] Adjusted {leg.leg_type.value} price: "
                    f"{leg.target_price:.2f} -> {adjusted_price:.2f}"
                )

            except Exception as e:
                logger.error(
                    f"[{self.symbol}] Error adjusting {leg.leg_type.value} order: {e}"
                )

        # Wait for completion with timeout
        completion_start = time.time()
        legs_completed = 0
        total_slippage = 0.0

        while time.time() - completion_start < PARTIAL_FILL_AGGRESSIVE_TIMEOUT:
            filled_this_cycle = []

            for leg in attempt.unfilled_legs:
                if leg.trade and leg.trade.orderStatus.status == "Filled":
                    filled_this_cycle.append(leg)
                    legs_completed += 1

                    # Calculate slippage
                    slippage = abs(
                        leg.trade.orderStatus.avgFillPrice - leg.target_price
                    )
                    total_slippage += slippage

                    logger.info(
                        f"[{self.symbol}] ✓ {leg.leg_type.value} completed: "
                        f"{leg.trade.orderStatus.avgFillPrice:.2f} (slippage: {slippage:.3f})"
                    )

            # Remove completed legs
            for leg in filled_this_cycle:
                attempt.unfilled_legs.remove(leg)

            # Check if all completed
            if not attempt.unfilled_legs:
                return {
                    "success": True,
                    "legs_completed": legs_completed,
                    "total_slippage": total_slippage,
                    "completion_method": "aggressive",
                }

            await asyncio.sleep(0.2)  # Check every 200ms

        # Timeout occurred
        return {
            "success": legs_completed > 0,
            "legs_completed": legs_completed,
            "total_slippage": total_slippage,
            "error_message": f"Timeout: completed {legs_completed}/{len(assessment.unfilled_legs)} legs",
        }

    async def _execute_wait_and_retry(
        self, attempt: CompletionAttempt, assessment: PartialFillAssessment
    ) -> Dict:
        """Execute wait and retry strategy."""
        logger.info(f"[{self.symbol}] Executing wait and retry")

        # Wait a bit longer for existing orders to fill
        wait_start = time.time()
        legs_completed = 0

        while time.time() - wait_start < PARTIAL_FILL_AGGRESSIVE_TIMEOUT * 1.5:
            for leg in attempt.unfilled_legs[
                :
            ]:  # Copy list to avoid modification during iteration
                if leg.trade and leg.trade.orderStatus.status == "Filled":
                    legs_completed += 1
                    attempt.unfilled_legs.remove(leg)

                    logger.info(
                        f"[{self.symbol}] ✓ {leg.leg_type.value} filled during wait"
                    )

            if not attempt.unfilled_legs:
                return {
                    "success": True,
                    "legs_completed": legs_completed,
                    "total_slippage": 0.0,
                    "completion_method": "wait_and_retry",
                }

            await asyncio.sleep(0.5)  # Longer wait intervals

        return {
            "success": legs_completed > 0,
            "legs_completed": legs_completed,
            "total_slippage": 0.0,
            "error_message": f"Wait timeout: completed {legs_completed} legs",
        }

    async def _execute_market_completion(
        self, attempt: CompletionAttempt, assessment: PartialFillAssessment
    ) -> Dict:
        """Execute market order completion (high risk, high speed)."""
        logger.warning(
            f"[{self.symbol}] Executing MARKET ORDER completion - high slippage risk!"
        )

        legs_completed = 0
        total_slippage = 0.0

        for leg in attempt.unfilled_legs:
            try:
                # Cancel existing limit order
                if leg.trade:
                    self.ib.cancelOrder(leg.order)
                    await asyncio.sleep(0.1)

                # Place market order
                market_order = Order(
                    orderId=self.ib.client.getReqId(),
                    orderType="MKT",
                    action=leg.action,
                    totalQuantity=leg.quantity,
                    tif="DAY",
                )

                new_trade = self.ib.placeOrder(leg.contract, market_order)

                # Wait for fill (market orders should fill quickly)
                fill_timeout = 5.0
                fill_start = time.time()

                while time.time() - fill_start < fill_timeout:
                    if new_trade.orderStatus.status == "Filled":
                        legs_completed += 1
                        fill_price = new_trade.orderStatus.avgFillPrice
                        slippage = abs(fill_price - leg.target_price)
                        total_slippage += slippage

                        logger.info(
                            f"[{self.symbol}] ✓ {leg.leg_type.value} market fill: "
                            f"{fill_price:.2f} (slippage: {slippage:.3f})"
                        )
                        break

                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(
                    f"[{self.symbol}] Error with market order for {leg.leg_type.value}: {e}"
                )

        return {
            "success": legs_completed > 0,
            "legs_completed": legs_completed,
            "total_slippage": total_slippage,
            "completion_method": "market_order",
            "warning": "High slippage risk method used",
        }

    async def _execute_immediate_rollback(
        self, attempt: CompletionAttempt, assessment: PartialFillAssessment
    ) -> Dict:
        """Execute immediate rollback - cancel unfilled and prepare rollback."""
        logger.info(f"[{self.symbol}] Executing immediate rollback")

        # Cancel all unfilled orders
        cancelled_count = 0
        for leg in attempt.unfilled_legs:
            try:
                if leg.trade:
                    self.ib.cancelOrder(leg.order)
                    cancelled_count += 1
                    logger.debug(
                        f"[{self.symbol}] Cancelled {leg.leg_type.value} order"
                    )
            except Exception as e:
                logger.error(
                    f"[{self.symbol}] Error cancelling {leg.leg_type.value}: {e}"
                )

        return {
            "success": False,  # Not a completion success
            "legs_completed": 0,
            "total_slippage": 0.0,
            "completion_method": "immediate_rollback",
            "cancelled_orders": cancelled_count,
            "message": "Proceeding to rollback filled positions",
        }

    def _update_performance_averages(self, attempt: CompletionAttempt) -> None:
        """Update running performance averages."""
        if attempt.success:
            # Update average completion time
            current_avg = self.performance_metrics["average_completion_time"]
            successful = self.performance_metrics["successful_completions"]
            new_avg = (
                (current_avg * (successful - 1)) + attempt.completion_time
            ) / successful
            self.performance_metrics["average_completion_time"] = new_avg

            # Update average slippage
            current_slippage_avg = self.performance_metrics[
                "average_slippage_on_completion"
            ]
            new_slippage_avg = (
                (current_slippage_avg * (successful - 1)) + attempt.total_slippage
            ) / successful
            self.performance_metrics["average_slippage_on_completion"] = (
                new_slippage_avg
            )

    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics."""
        total = self.performance_metrics["total_partial_fills"]
        success_rate = (
            self.performance_metrics["successful_completions"] / total
            if total > 0
            else 0.0
        )

        return {
            "handler_id": self.handler_id,
            "symbol": self.symbol,
            "total_partial_fills_handled": total,
            "successful_completions": self.performance_metrics[
                "successful_completions"
            ],
            "immediate_rollbacks": self.performance_metrics["immediate_rollbacks"],
            "aggressive_completions": self.performance_metrics[
                "aggressive_completions"
            ],
            "success_rate_percent": success_rate * 100,
            "average_completion_time_seconds": self.performance_metrics[
                "average_completion_time"
            ],
            "average_slippage_dollars": self.performance_metrics[
                "average_slippage_on_completion"
            ],
            "active_assessments": len(self.active_assessments),
            "completion_attempts_total": len(self.completion_attempts),
        }
