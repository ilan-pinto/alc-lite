"""
Sophisticated rollback manager for SFR parallel execution.

This module handles the complex process of unwinding partially filled SFR positions
when complete execution is not possible. It provides:

1. Intelligent rollback strategies based on position analysis
2. Risk-limited rollback attempts with safety mechanisms
3. Position unwinding with optimal timing and pricing
4. Comprehensive logging and reporting of rollback operations
5. Integration with broker APIs for reliable order management
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from ib_async import IB, Contract, Order, OrderStatus, Trade

from ..common import get_logger
from .constants import (
    ROLLBACK_AGGRESSIVE_PRICING_FACTOR,
    ROLLBACK_MAX_ATTEMPTS,
    ROLLBACK_MAX_SLIPPAGE_PERCENT,
    ROLLBACK_TIMEOUT_PER_ATTEMPT,
)
from .parallel_execution_framework import LegOrder, LegType, ParallelExecutionPlan

logger = get_logger()


class RollbackStrategy(Enum):
    """Different rollback strategies based on situation."""

    IMMEDIATE_MARKET = "immediate_market"  # Market orders for fast unwinding
    AGGRESSIVE_LIMIT = "aggressive_limit"  # Aggressive limit prices
    GRADUAL_LIMIT = "gradual_limit"  # Conservative limit prices with retries
    STOP_LOSS = "stop_loss"  # Use stop-loss orders for protection


class RollbackReason(Enum):
    """Reasons for initiating rollback."""

    PARTIAL_FILLS_TIMEOUT = "partial_fills_timeout"
    COMPLETION_FAILED = "completion_failed"
    RISK_LIMIT_EXCEEDED = "risk_limit_exceeded"
    MARKET_CONDITIONS_ADVERSE = "market_conditions_adverse"
    MANUAL_TRIGGER = "manual_trigger"
    SYSTEM_ERROR = "system_error"


@dataclass
class RollbackPosition:
    """Information about a position that needs to be unwound."""

    leg_order: LegOrder
    current_quantity: int
    avg_fill_price: float
    unrealized_pnl: float
    unwinding_priority: int  # 1=highest, 3=lowest
    rollback_target_price: Optional[float] = None
    rollback_order: Optional[Order] = None
    rollback_trade: Optional[Trade] = None
    unwind_status: str = "pending"  # pending, unwinding, completed, failed


@dataclass
class RollbackPlan:
    """Complete plan for rolling back positions."""

    rollback_id: str
    symbol: str
    execution_id: str
    reason: RollbackReason
    strategy: RollbackStrategy

    # Positions to unwind
    positions_to_unwind: List[RollbackPosition]
    total_positions: int
    estimated_unwinding_cost: float
    max_acceptable_loss: float

    # Timing
    created_time: float
    max_rollback_time: float

    # Status tracking
    rollback_status: str = "pending"  # pending, executing, completed, failed, cancelled
    positions_unwound: int = 0
    total_rollback_cost: float = 0.0
    rollback_slippage: float = 0.0

    # Results
    success: bool = False
    completion_time: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class RollbackAttempt:
    """Individual attempt to unwind positions."""

    attempt_id: str
    attempt_number: int
    strategy_used: RollbackStrategy
    timestamp: float
    positions_targeted: List[RollbackPosition]

    # Results
    positions_unwound: int = 0
    attempt_cost: float = 0.0
    attempt_slippage: float = 0.0
    duration: float = 0.0
    success: bool = False
    error_message: Optional[str] = None


class RollbackManager:
    """
    Sophisticated manager for rolling back partial SFR positions.

    This manager provides:
    1. Risk-assessed rollback planning
    2. Multiple rollback strategies for different situations
    3. Intelligent order management for position unwinding
    4. Comprehensive tracking and reporting
    5. Safety limits and circuit breakers
    """

    def __init__(self, ib: IB, symbol: str):
        self.ib = ib
        self.symbol = symbol
        self.manager_id = f"rollback_{symbol}_{int(time.time())}"

        # Active rollbacks tracking
        self.active_rollbacks: Dict[str, RollbackPlan] = {}
        self.completed_rollbacks: List[RollbackPlan] = []
        self.rollback_attempts: List[RollbackAttempt] = []
        self._max_completed_rollbacks = 20  # Limit history to prevent memory leaks
        self._max_rollback_attempts = 100  # Limit attempts history

        # Performance metrics
        self.performance_metrics = {
            "total_rollbacks_initiated": 0,
            "successful_rollbacks": 0,
            "failed_rollbacks": 0,
            "average_rollback_time": 0.0,
            "average_rollback_cost": 0.0,
            "total_slippage": 0.0,
        }

        # Safety limits
        self.daily_rollback_limit = 10  # Max rollbacks per day
        self.daily_rollback_count = 0
        self.max_single_rollback_cost = 1000.0  # Max cost per rollback

        logger.debug(f"[{symbol}] RollbackManager initialized: {self.manager_id}")

    def _add_rollback_attempt(self, attempt: RollbackAttempt) -> None:
        """Add rollback attempt with memory limit enforcement."""
        self.rollback_attempts.append(attempt)
        # Limit attempts history to prevent memory leaks
        if len(self.rollback_attempts) > self._max_rollback_attempts:
            self.rollback_attempts.pop(0)

    async def initialize(self) -> bool:
        """Initialize the rollback manager."""
        try:
            logger.info(f"[{self.symbol}] Rollback manager initialized")
            return True
        except Exception as e:
            logger.error(f"[{self.symbol}] Failed to initialize rollback manager: {e}")
            return False

    async def execute_rollback(
        self,
        plan: ParallelExecutionPlan,
        filled_legs: List[LegOrder],
        unfilled_legs: List[LegOrder],
        reason: RollbackReason = RollbackReason.PARTIAL_FILLS_TIMEOUT,
        max_acceptable_loss: Optional[float] = None,
    ) -> Dict:
        """
        Execute rollback of filled positions.

        Args:
            plan: The original execution plan
            filled_legs: List of legs that were filled and need unwinding
            unfilled_legs: List of legs that weren't filled (will be cancelled)
            reason: Reason for the rollback
            max_acceptable_loss: Maximum loss acceptable for rollback

        Returns:
            Dictionary with rollback results
        """
        # Check daily limits
        if self.daily_rollback_count >= self.daily_rollback_limit:
            logger.error(
                f"[{self.symbol}] Daily rollback limit ({self.daily_rollback_limit}) exceeded"
            )
            return {
                "success": False,
                "error": "daily_rollback_limit_exceeded",
                "positions_unwound": False,
            }

        rollback_start_time = time.time()
        self.performance_metrics["total_rollbacks_initiated"] += 1
        self.daily_rollback_count += 1

        logger.info(
            f"[{self.symbol}] Initiating rollback for execution {plan.execution_id}: "
            f"reason={reason.value}, filled_legs={len(filled_legs)}"
        )

        try:
            # Step 1: Create rollback plan
            rollback_plan = await self._create_rollback_plan(
                plan, filled_legs, unfilled_legs, reason, max_acceptable_loss
            )

            if not rollback_plan:
                return {
                    "success": False,
                    "error": "rollback_plan_creation_failed",
                    "positions_unwound": False,
                }

            self.active_rollbacks[rollback_plan.rollback_id] = rollback_plan

            # Step 2: Cancel unfilled orders first
            await self._cancel_unfilled_orders(unfilled_legs)

            # Step 3: Execute rollback strategy
            rollback_result = await self._execute_rollback_strategy(rollback_plan)

            # Step 4: Finalize rollback
            rollback_plan.completion_time = time.time() - rollback_start_time
            rollback_plan.success = rollback_result.get("success", False)

            if rollback_plan.success:
                self.performance_metrics["successful_rollbacks"] += 1
                logger.info(
                    f"[{self.symbol}] ✓ Rollback SUCCESSFUL: {rollback_plan.rollback_id} "
                    f"in {rollback_plan.completion_time:.2f}s"
                )
            else:
                self.performance_metrics["failed_rollbacks"] += 1
                rollback_plan.error_message = rollback_result.get(
                    "error_message", "Unknown error"
                )
                logger.error(
                    f"[{self.symbol}] ✗ Rollback FAILED: {rollback_plan.error_message}"
                )

            # Move to completed rollbacks
            self.completed_rollbacks.append(rollback_plan)
            # Limit completed rollbacks history to prevent memory leaks
            if len(self.completed_rollbacks) > self._max_completed_rollbacks:
                self.completed_rollbacks.pop(0)
            del self.active_rollbacks[rollback_plan.rollback_id]

            # Update performance metrics
            self._update_performance_metrics(rollback_plan)

            return {
                "success": rollback_plan.success,
                "rollback_id": rollback_plan.rollback_id,
                "positions_unwound": rollback_plan.positions_unwound
                == rollback_plan.total_positions,
                "unwound_count": rollback_plan.positions_unwound,
                "total_positions": rollback_plan.total_positions,
                "rollback_cost": rollback_plan.total_rollback_cost,
                "rollback_time": rollback_plan.completion_time,
                "error_message": rollback_plan.error_message,
            }

        except Exception as e:
            logger.error(f"[{self.symbol}] Critical error in rollback execution: {e}")
            return {
                "success": False,
                "error": f"rollback_exception: {str(e)}",
                "positions_unwound": False,
            }

    async def _create_rollback_plan(
        self,
        plan: ParallelExecutionPlan,
        filled_legs: List[LegOrder],
        unfilled_legs: List[LegOrder],
        reason: RollbackReason,
        max_acceptable_loss: Optional[float],
    ) -> Optional[RollbackPlan]:
        """Create a comprehensive rollback plan."""

        rollback_id = f"rollback_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        logger.info(f"[{self.symbol}] Creating rollback plan {rollback_id}")

        # Analyze positions to unwind
        positions_to_unwind = []
        total_unrealized_pnl = 0.0

        for leg in filled_legs:
            if leg.fill_status != "filled":
                continue

            # Calculate current unrealized P&L (simplified)
            # In production, would get current market prices
            current_market_price = leg.avg_fill_price  # Simplified
            unrealized_pnl = 0.0

            if leg.leg_type == LegType.STOCK and leg.action == "BUY":
                unrealized_pnl = (
                    current_market_price - leg.avg_fill_price
                ) * leg.filled_quantity
            elif leg.leg_type == LegType.CALL and leg.action == "SELL":
                unrealized_pnl = (
                    leg.avg_fill_price - current_market_price
                ) * leg.filled_quantity
            elif leg.leg_type == LegType.PUT and leg.action == "BUY":
                unrealized_pnl = (
                    current_market_price - leg.avg_fill_price
                ) * leg.filled_quantity

            # Determine unwinding priority (stock first, then options)
            priority = 1 if leg.leg_type == LegType.STOCK else 2

            position = RollbackPosition(
                leg_order=leg,
                current_quantity=leg.filled_quantity,
                avg_fill_price=leg.avg_fill_price,
                unrealized_pnl=unrealized_pnl,
                unwinding_priority=priority,
            )

            positions_to_unwind.append(position)
            total_unrealized_pnl += unrealized_pnl

        if not positions_to_unwind:
            logger.warning(f"[{self.symbol}] No positions to unwind found")
            return None

        # Estimate unwinding cost (spread costs + slippage)
        estimated_cost = self._estimate_unwinding_cost(positions_to_unwind)

        # Determine rollback strategy
        strategy = self._determine_rollback_strategy(
            positions_to_unwind, total_unrealized_pnl, estimated_cost, reason
        )

        # Set maximum acceptable loss
        if max_acceptable_loss is None:
            max_acceptable_loss = min(
                self.max_single_rollback_cost, estimated_cost * 2.0
            )

        rollback_plan = RollbackPlan(
            rollback_id=rollback_id,
            symbol=self.symbol,
            execution_id=plan.execution_id,
            reason=reason,
            strategy=strategy,
            positions_to_unwind=positions_to_unwind,
            total_positions=len(positions_to_unwind),
            estimated_unwinding_cost=estimated_cost,
            max_acceptable_loss=max_acceptable_loss,
            created_time=time.time(),
            max_rollback_time=ROLLBACK_TIMEOUT_PER_ATTEMPT * ROLLBACK_MAX_ATTEMPTS,
        )

        logger.info(
            f"[{self.symbol}] Rollback plan created: strategy={strategy.value}, "
            f"positions={len(positions_to_unwind)}, estimated_cost=${estimated_cost:.2f}"
        )

        return rollback_plan

    async def _cancel_unfilled_orders(self, unfilled_legs: List[LegOrder]) -> None:
        """Cancel all unfilled orders before starting rollback."""
        logger.info(f"[{self.symbol}] Cancelling {len(unfilled_legs)} unfilled orders")

        for leg in unfilled_legs:
            try:
                if leg.trade and leg.trade.orderStatus.status in [
                    "Submitted",
                    "PreSubmitted",
                    "PendingSubmit",
                ]:
                    self.ib.cancelOrder(leg.order)
                    logger.debug(
                        f"[{self.symbol}] Cancelled unfilled {leg.leg_type.value} order"
                    )
            except Exception as e:
                logger.warning(
                    f"[{self.symbol}] Error cancelling {leg.leg_type.value} order: {e}"
                )

    def _estimate_unwinding_cost(self, positions: List[RollbackPosition]) -> float:
        """Estimate the cost of unwinding positions."""
        total_cost = 0.0

        for position in positions:
            # Estimate bid-ask spread cost
            spread_cost = position.avg_fill_price * 0.01  # 1% spread estimate

            # Estimate market impact cost
            quantity_factor = min(
                position.current_quantity / 1000.0, 0.05
            )  # Up to 5% impact
            impact_cost = position.avg_fill_price * quantity_factor

            # Total cost for this position
            position_cost = (spread_cost + impact_cost) * position.current_quantity
            total_cost += position_cost

        return total_cost

    def _determine_rollback_strategy(
        self,
        positions: List[RollbackPosition],
        unrealized_pnl: float,
        estimated_cost: float,
        reason: RollbackReason,
    ) -> RollbackStrategy:
        """Determine the optimal rollback strategy based on situation."""

        # If system error or risk limit exceeded, use immediate market orders
        if reason in [RollbackReason.SYSTEM_ERROR, RollbackReason.RISK_LIMIT_EXCEEDED]:
            return RollbackStrategy.IMMEDIATE_MARKET

        # If unrealized loss is significant, use aggressive limit orders
        if unrealized_pnl < -100.0:  # More than $100 unrealized loss
            return RollbackStrategy.AGGRESSIVE_LIMIT

        # If estimated cost is low, use gradual approach
        if estimated_cost < 50.0:
            return RollbackStrategy.GRADUAL_LIMIT

        # Default to aggressive limit for most cases
        return RollbackStrategy.AGGRESSIVE_LIMIT

    async def _execute_rollback_strategy(self, plan: RollbackPlan) -> Dict:
        """Execute the specific rollback strategy."""

        plan.rollback_status = "executing"
        strategy_start_time = time.time()

        logger.info(
            f"[{self.symbol}] Executing rollback strategy: {plan.strategy.value} "
            f"for {plan.total_positions} positions"
        )

        try:
            if plan.strategy == RollbackStrategy.IMMEDIATE_MARKET:
                result = await self._execute_immediate_market_rollback(plan)
            elif plan.strategy == RollbackStrategy.AGGRESSIVE_LIMIT:
                result = await self._execute_aggressive_limit_rollback(plan)
            elif plan.strategy == RollbackStrategy.GRADUAL_LIMIT:
                result = await self._execute_gradual_limit_rollback(plan)
            else:  # Default to aggressive limit
                result = await self._execute_aggressive_limit_rollback(plan)

            plan.rollback_status = "completed" if result["success"] else "failed"
            return result

        except RecursionError as re:
            plan.rollback_status = "failed"
            error_msg = f"Recursion error in rollback: {str(re)}"
            logger.error(f"[{self.symbol}] {error_msg}")
            # Emergency market order fallback for recursion errors
            emergency_result = await self._emergency_market_rollback(plan)
            return {
                "success": False,
                "error_message": error_msg,
                "positions_unwound": emergency_result.get("positions_unwound", 0),
                "emergency_fallback": True,
            }
        except Exception as e:
            plan.rollback_status = "failed"
            logger.error(f"[{self.symbol}] Error in rollback strategy execution: {e}")
            return {"success": False, "error_message": str(e), "positions_unwound": 0}

    async def _execute_immediate_market_rollback(self, plan: RollbackPlan) -> Dict:
        """Execute immediate market order rollback (highest speed, highest slippage risk)."""

        logger.warning(
            f"[{self.symbol}] Executing IMMEDIATE MARKET rollback - high slippage risk!"
        )

        attempt = RollbackAttempt(
            attempt_id=f"market_{int(time.time())}",
            attempt_number=1,
            strategy_used=RollbackStrategy.IMMEDIATE_MARKET,
            timestamp=time.time(),
            positions_targeted=plan.positions_to_unwind.copy(),
        )

        unwound_count = 0
        total_cost = 0.0
        total_slippage = 0.0

        # Sort by priority (stock first)
        sorted_positions = sorted(
            plan.positions_to_unwind, key=lambda p: p.unwinding_priority
        )

        for position in sorted_positions:
            try:
                # Determine opposite action for unwinding
                unwind_action = "SELL" if position.leg_order.action == "BUY" else "BUY"

                # Create market order
                market_order = Order(
                    orderId=self.ib.client.getReqId(),
                    orderType="MKT",
                    action=unwind_action,
                    totalQuantity=position.current_quantity,
                    tif="DAY",
                )

                # Place market order
                unwind_trade = self.ib.placeOrder(
                    position.leg_order.contract, market_order
                )
                position.rollback_trade = unwind_trade
                position.rollback_order = market_order

                # Update tracking
                self._update_rollback_tracking(rollback_plan, position, "unwinding")
                self._log_rollback_order_placement(position, "market")

                # Wait for fill (market orders should fill quickly)
                fill_start = time.time()
                fill_timeout = 10.0  # 10 seconds max for market order

                while time.time() - fill_start < fill_timeout:
                    if unwind_trade.orderStatus.status == "Filled":
                        fill_price = unwind_trade.orderStatus.avgFillPrice
                        slippage = abs(fill_price - position.avg_fill_price)
                        cost = fill_price * position.current_quantity

                        # Update tracking
                        self._update_rollback_tracking(
                            rollback_plan, position, "completed"
                        )
                        self._log_rollback_order_fill(
                            position, fill_price, position.current_quantity
                        )

                        unwound_count += 1
                        total_cost += cost
                        total_slippage += slippage

                        logger.info(
                            f"[{self.symbol}] ✓ Market unwind: {position.leg_order.leg_type.value} "
                            f"@ {fill_price:.2f} (slippage: ${slippage:.2f})"
                        )
                        break

                    await asyncio.sleep(0.1)

                if position.unwind_status != "completed":
                    position.unwind_status = "failed"
                    logger.error(
                        f"[{self.symbol}] Market order timeout for {position.leg_order.leg_type.value}"
                    )

            except Exception as e:
                position.unwind_status = "failed"
                logger.error(
                    f"[{self.symbol}] Error unwinding {position.leg_order.leg_type.value}: {e}"
                )

        # Update attempt results
        attempt.positions_unwound = unwound_count
        attempt.attempt_cost = total_cost
        attempt.attempt_slippage = total_slippage
        attempt.duration = time.time() - attempt.timestamp
        attempt.success = unwound_count == len(plan.positions_to_unwind)

        # Update plan results
        plan.positions_unwound = unwound_count
        plan.total_rollback_cost = total_cost
        plan.rollback_slippage = total_slippage

        self._add_rollback_attempt(attempt)

        return {
            "success": attempt.success,
            "positions_unwound": unwound_count,
            "total_cost": total_cost,
            "total_slippage": total_slippage,
            "duration": attempt.duration,
            "method": "immediate_market",
        }

    async def _execute_aggressive_limit_rollback(self, plan: RollbackPlan) -> Dict:
        """Execute aggressive limit order rollback with multiple attempts."""

        logger.info(f"[{self.symbol}] Executing aggressive limit rollback")

        total_unwound = 0
        total_cost = 0.0
        total_slippage = 0.0

        for attempt_num in range(1, ROLLBACK_MAX_ATTEMPTS + 1):
            if total_unwound == plan.total_positions:
                break  # All positions unwound

            attempt = RollbackAttempt(
                attempt_id=f"aggressive_{attempt_num}_{int(time.time())}",
                attempt_number=attempt_num,
                strategy_used=RollbackStrategy.AGGRESSIVE_LIMIT,
                timestamp=time.time(),
                positions_targeted=[
                    p for p in plan.positions_to_unwind if p.unwind_status == "pending"
                ],
            )

            logger.info(
                f"[{self.symbol}] Rollback attempt {attempt_num}/{ROLLBACK_MAX_ATTEMPTS}: "
                f"{len(attempt.positions_targeted)} positions remaining"
            )

            attempt_result = await self._execute_single_aggressive_attempt(attempt)

            # Update totals
            total_unwound += attempt_result["unwound_count"]
            total_cost += attempt_result["cost"]
            total_slippage += attempt_result["slippage"]

            self._add_rollback_attempt(attempt)

            # Check if we're done
            if total_unwound == plan.total_positions:
                break

            # Brief pause between attempts
            await asyncio.sleep(0.5)

        # Update plan results
        plan.positions_unwound = total_unwound
        plan.total_rollback_cost = total_cost
        plan.rollback_slippage = total_slippage

        success = total_unwound == plan.total_positions

        # Market order fallback if aggressive limit fails
        if not success and total_unwound < plan.total_positions:
            logger.warning(
                f"[{self.symbol}] Aggressive limit rollback incomplete ({total_unwound}/{plan.total_positions}). "
                f"Triggering market order fallback."
            )

            # Get remaining positions
            remaining_positions = [
                p for p in plan.positions_to_unwind if p.unwind_status == "pending"
            ]

            if remaining_positions:
                # Execute market order fallback
                market_fallback_result = await self._execute_market_order_fallback(
                    plan, remaining_positions
                )

                # Update totals with fallback results
                total_unwound += market_fallback_result["unwound_count"]
                total_cost += market_fallback_result["cost"]
                total_slippage += market_fallback_result["slippage"]

                # Update final plan results
                plan.positions_unwound = total_unwound
                plan.total_rollback_cost = total_cost
                plan.rollback_slippage = total_slippage

                success = total_unwound == plan.total_positions

        return {
            "success": success,
            "positions_unwound": total_unwound,
            "total_cost": total_cost,
            "total_slippage": total_slippage,
            "attempts_used": min(attempt_num, ROLLBACK_MAX_ATTEMPTS),
            "method": (
                "aggressive_limit_with_market_fallback"
                if total_unwound > 0
                else "aggressive_limit"
            ),
            "market_fallback_used": total_unwound > 0 and not success,
        }

    async def _execute_single_aggressive_attempt(
        self, attempt: RollbackAttempt
    ) -> Dict:
        """Execute a single aggressive limit attempt."""

        attempt_start = time.time()
        unwound_count = 0
        total_cost = 0.0
        total_slippage = 0.0

        # Calculate aggressive pricing factor based on attempt number
        pricing_factor = ROLLBACK_AGGRESSIVE_PRICING_FACTOR * attempt.attempt_number

        for position in attempt.positions_targeted:
            if position.unwind_status != "pending":
                continue

            try:
                # Determine opposite action and aggressive price
                unwind_action = "SELL" if position.leg_order.action == "BUY" else "BUY"

                if unwind_action == "SELL":
                    # Selling: reduce price to encourage fills
                    aggressive_price = position.avg_fill_price * (1.0 - pricing_factor)
                else:
                    # Buying: increase price to encourage fills
                    aggressive_price = position.avg_fill_price * (1.0 + pricing_factor)

                position.rollback_target_price = aggressive_price

                # Create aggressive limit order
                aggressive_order = Order(
                    orderId=self.ib.client.getReqId(),
                    orderType="LMT",
                    action=unwind_action,
                    totalQuantity=position.current_quantity,
                    lmtPrice=aggressive_price,
                    tif="DAY",
                )

                # Place order
                unwind_trade = self.ib.placeOrder(
                    position.leg_order.contract, aggressive_order
                )
                position.rollback_trade = unwind_trade
                position.rollback_order = aggressive_order

                # Update tracking
                self._update_rollback_tracking(rollback_plan, position, "unwinding")
                self._log_rollback_order_placement(position, "aggressive_limit")

                logger.debug(
                    f"[{self.symbol}] Placed aggressive {unwind_action} order: "
                    f"{position.leg_order.leg_type.value} @ {aggressive_price:.2f}"
                )

            except Exception as e:
                logger.error(
                    f"[{self.symbol}] Error placing aggressive order for "
                    f"{position.leg_order.leg_type.value}: {e}"
                )
                position.unwind_status = "failed"

        # Monitor fills for this attempt
        monitor_start = time.time()

        while time.time() - monitor_start < ROLLBACK_TIMEOUT_PER_ATTEMPT:
            filled_this_cycle = []

            for position in attempt.positions_targeted:
                if (
                    position.unwind_status == "unwinding"
                    and position.rollback_trade
                    and position.rollback_trade.orderStatus.status == "Filled"
                ):

                    fill_price = position.rollback_trade.orderStatus.avgFillPrice
                    slippage = abs(fill_price - position.avg_fill_price)
                    cost = fill_price * position.current_quantity

                    position.unwind_status = "completed"
                    unwound_count += 1
                    total_cost += cost
                    total_slippage += slippage
                    filled_this_cycle.append(position)

                    logger.info(
                        f"[{self.symbol}] ✓ Aggressive unwind: {position.leg_order.leg_type.value} "
                        f"@ {fill_price:.2f} (slippage: ${slippage:.2f})"
                    )

            if not filled_this_cycle:
                await asyncio.sleep(0.2)  # Brief pause if no fills

            # Check if all targeted positions are done
            remaining = [
                p for p in attempt.positions_targeted if p.unwind_status == "unwinding"
            ]
            if not remaining:
                break

        # Cancel any remaining unfilled orders
        for position in attempt.positions_targeted:
            if (
                position.unwind_status == "unwinding"
                and position.rollback_trade
                and position.rollback_trade.orderStatus.status
                not in ["Filled", "Cancelled"]
            ):
                try:
                    self.ib.cancelOrder(position.rollback_order)
                    position.unwind_status = (
                        "pending"  # Will be retried in next attempt
                    )
                except Exception as e:
                    logger.warning(
                        f"[{self.symbol}] Error cancelling rollback order: {e}"
                    )

        # Update attempt results
        attempt.positions_unwound = unwound_count
        attempt.attempt_cost = total_cost
        attempt.attempt_slippage = total_slippage
        attempt.duration = time.time() - attempt_start
        attempt.success = unwound_count > 0

        return {
            "unwound_count": unwound_count,
            "cost": total_cost,
            "slippage": total_slippage,
            "duration": attempt.duration,
        }

    async def _execute_gradual_limit_rollback(self, plan: RollbackPlan) -> Dict:
        """Execute gradual limit order rollback (lowest slippage, but slower)."""
        logger.info(f"[{self.symbol}] Executing gradual limit rollback")

        # For now, implement as a conservative version of aggressive limit
        # In production, would use more conservative pricing and longer timeouts

        total_unwound = 0
        total_cost = 0.0
        total_slippage = 0.0

        # Use more conservative pricing and longer timeouts
        for attempt_num in range(1, ROLLBACK_MAX_ATTEMPTS + 2):  # More attempts
            if total_unwound == plan.total_positions:
                break

            attempt = RollbackAttempt(
                attempt_id=f"gradual_{attempt_num}_{int(time.time())}",
                attempt_number=attempt_num,
                strategy_used=RollbackStrategy.GRADUAL_LIMIT,
                timestamp=time.time(),
                positions_targeted=[
                    p for p in plan.positions_to_unwind if p.unwind_status == "pending"
                ],
            )

            # Use more conservative pricing (half the aggressive factor)
            conservative_factor = (ROLLBACK_AGGRESSIVE_PRICING_FACTOR / 2) * attempt_num

            attempt_result = await self._execute_conservative_attempt(
                attempt, conservative_factor
            )

            total_unwound += attempt_result["unwound_count"]
            total_cost += attempt_result["cost"]
            total_slippage += attempt_result["slippage"]

            self._add_rollback_attempt(attempt)

            # Longer pause between gradual attempts
            await asyncio.sleep(1.0)

        plan.positions_unwound = total_unwound
        plan.total_rollback_cost = total_cost
        plan.rollback_slippage = total_slippage

        return {
            "success": total_unwound == plan.total_positions,
            "positions_unwound": total_unwound,
            "total_cost": total_cost,
            "total_slippage": total_slippage,
            "method": "gradual_limit",
        }

    async def _execute_conservative_attempt(
        self, attempt: RollbackAttempt, pricing_factor: float
    ) -> Dict:
        """Execute a single conservative attempt with longer timeout."""

        # Similar to aggressive attempt but with more conservative pricing and longer timeout
        attempt_start = time.time()
        unwound_count = 0
        total_cost = 0.0
        total_slippage = 0.0

        # Place orders with conservative pricing
        for position in attempt.positions_targeted:
            if position.unwind_status != "pending":
                continue

            try:
                unwind_action = "SELL" if position.leg_order.action == "BUY" else "BUY"

                if unwind_action == "SELL":
                    conservative_price = position.avg_fill_price * (
                        1.0 - pricing_factor
                    )
                else:
                    conservative_price = position.avg_fill_price * (
                        1.0 + pricing_factor
                    )

                position.rollback_target_price = conservative_price

                conservative_order = Order(
                    orderId=self.ib.client.getReqId(),
                    orderType="LMT",
                    action=unwind_action,
                    totalQuantity=position.current_quantity,
                    lmtPrice=conservative_price,
                    tif="DAY",
                )

                unwind_trade = self.ib.placeOrder(
                    position.leg_order.contract, conservative_order
                )
                position.rollback_trade = unwind_trade
                position.rollback_order = conservative_order

                # Update tracking
                self._update_rollback_tracking(rollback_plan, position, "unwinding")
                self._log_rollback_order_placement(position, "conservative_limit")

            except Exception as e:
                logger.error(f"[{self.symbol}] Error placing conservative order: {e}")
                position.unwind_status = "failed"

        # Monitor with longer timeout for conservative approach
        monitor_timeout = ROLLBACK_TIMEOUT_PER_ATTEMPT * 2  # Double the timeout
        monitor_start = time.time()

        while time.time() - monitor_start < monitor_timeout:
            for position in attempt.positions_targeted:
                if (
                    position.unwind_status == "unwinding"
                    and position.rollback_trade
                    and position.rollback_trade.orderStatus.status == "Filled"
                ):

                    fill_price = position.rollback_trade.orderStatus.avgFillPrice
                    slippage = abs(fill_price - position.avg_fill_price)
                    cost = fill_price * position.current_quantity

                    position.unwind_status = "completed"
                    unwound_count += 1
                    total_cost += cost
                    total_slippage += slippage

                    logger.info(
                        f"[{self.symbol}] ✓ Conservative unwind: {position.leg_order.leg_type.value} "
                        f"@ {fill_price:.2f} (slippage: ${slippage:.2f})"
                    )

            await asyncio.sleep(0.5)  # Longer pause between checks

        # Cancel unfilled orders
        for position in attempt.positions_targeted:
            if (
                position.unwind_status == "unwinding"
                and position.rollback_trade
                and position.rollback_trade.orderStatus.status
                not in ["Filled", "Cancelled"]
            ):
                try:
                    self.ib.cancelOrder(position.rollback_order)
                    position.unwind_status = "pending"
                except Exception:
                    pass

        attempt.positions_unwound = unwound_count
        attempt.attempt_cost = total_cost
        attempt.attempt_slippage = total_slippage
        attempt.duration = time.time() - attempt_start
        attempt.success = unwound_count > 0

        return {
            "unwound_count": unwound_count,
            "cost": total_cost,
            "slippage": total_slippage,
            "duration": attempt.duration,
        }

    def _update_performance_metrics(self, plan: RollbackPlan) -> None:
        """Update performance metrics after rollback completion."""
        if plan.completion_time:
            total_rollbacks = self.performance_metrics["total_rollbacks_initiated"]
            current_avg_time = self.performance_metrics["average_rollback_time"]
            new_avg_time = (
                (current_avg_time * (total_rollbacks - 1)) + plan.completion_time
            ) / total_rollbacks
            self.performance_metrics["average_rollback_time"] = new_avg_time

        if plan.success:
            successful = self.performance_metrics["successful_rollbacks"]
            current_avg_cost = self.performance_metrics["average_rollback_cost"]
            new_avg_cost = (
                (current_avg_cost * (successful - 1)) + plan.total_rollback_cost
            ) / successful
            self.performance_metrics["average_rollback_cost"] = new_avg_cost

            self.performance_metrics["total_slippage"] += plan.rollback_slippage

    def get_performance_stats(self) -> Dict:
        """Get comprehensive rollback performance statistics."""
        total = self.performance_metrics["total_rollbacks_initiated"]
        success_rate = (
            self.performance_metrics["successful_rollbacks"] / total
            if total > 0
            else 0.0
        )

        return {
            "manager_id": self.manager_id,
            "symbol": self.symbol,
            "total_rollbacks_initiated": total,
            "successful_rollbacks": self.performance_metrics["successful_rollbacks"],
            "failed_rollbacks": self.performance_metrics["failed_rollbacks"],
            "success_rate_percent": success_rate * 100,
            "average_rollback_time_seconds": self.performance_metrics[
                "average_rollback_time"
            ],
            "average_rollback_cost_dollars": self.performance_metrics[
                "average_rollback_cost"
            ],
            "total_slippage_dollars": self.performance_metrics["total_slippage"],
            "daily_rollback_count": self.daily_rollback_count,
            "daily_rollback_limit": self.daily_rollback_limit,
            "active_rollbacks": len(self.active_rollbacks),
            "completed_rollbacks": len(self.completed_rollbacks),
            "rollback_attempts_total": len(self.rollback_attempts),
        }

    async def force_cancel_active_rollbacks(
        self, reason: str = "manual_override"
    ) -> int:
        """Force cancel all active rollbacks (emergency use only)."""
        cancelled_count = 0

        for rollback_id, plan in list(self.active_rollbacks.items()):
            try:
                # Cancel all active rollback orders
                for position in plan.positions_to_unwind:
                    if (
                        position.unwind_status == "unwinding"
                        and position.rollback_trade
                    ):
                        self.ib.cancelOrder(position.rollback_order)

                plan.rollback_status = "cancelled"
                plan.error_message = f"Force cancelled: {reason}"
                plan.completion_time = time.time() - plan.created_time

                # Move to completed
                self.completed_rollbacks.append(plan)
                # Limit completed rollbacks history to prevent memory leaks
                if len(self.completed_rollbacks) > self._max_completed_rollbacks:
                    self.completed_rollbacks.pop(0)
                del self.active_rollbacks[rollback_id]
                cancelled_count += 1

            except Exception as e:
                logger.error(
                    f"[{self.symbol}] Error force cancelling rollback {rollback_id}: {e}"
                )

        if cancelled_count > 0:
            logger.warning(
                f"[{self.symbol}] Force cancelled {cancelled_count} active rollbacks: {reason}"
            )

        return cancelled_count

    async def _emergency_market_rollback(self, plan: RollbackPlan) -> Dict:
        """
        Emergency market order rollback for catastrophic failures.

        This method bypasses all normal rollback logic and immediately
        places market orders to unwind positions. Used as last resort
        when normal rollback fails due to recursion or other critical errors.
        """
        logger.critical(
            f"[{self.symbol}] EMERGENCY ROLLBACK: Attempting immediate market liquidation"
        )

        positions_unwound = 0
        total_cost = 0.0

        # Get all filled positions that need unwinding
        positions_to_unwind = [
            p for p in plan.positions_to_unwind if p.leg_order.fill_status == "filled"
        ]

        for position in positions_to_unwind:
            try:
                # Cancel any existing rollback orders first
                if position.rollback_trade:
                    try:
                        self.ib.cancelOrder(position.rollback_order)
                        await asyncio.sleep(0.1)  # Brief pause
                    except Exception:
                        pass  # Ignore cancellation errors in emergency

                # Determine opposite action for unwinding
                unwind_action = "SELL" if position.leg_order.action == "BUY" else "BUY"

                # Create emergency market order - no limit price
                emergency_order = Order(
                    orderId=self.ib.client.getReqId(),
                    orderType="MKT",
                    action=unwind_action,
                    totalQuantity=position.current_quantity,
                    tif="DAY",
                )

                # Place emergency market order
                emergency_trade = self.ib.placeOrder(
                    position.leg_order.contract, emergency_order
                )

                # Wait for emergency fill with very short timeout
                emergency_start = time.time()
                emergency_timeout = 5.0  # Only 5 seconds for emergency

                while time.time() - emergency_start < emergency_timeout:
                    if emergency_trade.orderStatus.status == "Filled":
                        fill_price = emergency_trade.orderStatus.avgFillPrice
                        cost = fill_price * position.current_quantity
                        total_cost += cost
                        positions_unwound += 1

                        logger.warning(
                            f"[{self.symbol}] EMERGENCY UNWIND: {position.leg_order.leg_type.value} "
                            f"@ {fill_price:.2f} (MARKET ORDER)"
                        )
                        break

                    await asyncio.sleep(0.1)

                if emergency_trade.orderStatus.status != "Filled":
                    logger.error(
                        f"[{self.symbol}] EMERGENCY ROLLBACK FAILED: "
                        f"{position.leg_order.leg_type.value} market order didn't fill"
                    )

            except Exception as e:
                logger.error(
                    f"[{self.symbol}] Error in emergency rollback for "
                    f"{position.leg_order.leg_type.value}: {e}"
                )

        # Update plan with emergency results
        plan.positions_unwound = positions_unwound
        plan.total_rollback_cost = total_cost

        result = {
            "positions_unwound": positions_unwound,
            "total_positions": len(positions_to_unwind),
            "emergency_cost": total_cost,
            "method": "emergency_market",
            "success": positions_unwound > 0,
        }

        if positions_unwound == len(positions_to_unwind):
            logger.warning(
                f"[{self.symbol}] EMERGENCY ROLLBACK SUCCESSFUL: All positions liquidated"
            )
        else:
            logger.critical(
                f"[{self.symbol}] EMERGENCY ROLLBACK INCOMPLETE: {positions_unwound}/{len(positions_to_unwind)} positions liquidated"
            )

        return result

    async def _execute_market_order_fallback(
        self, plan: RollbackPlan, remaining_positions: List[RollbackPosition]
    ) -> Dict:
        """
        Execute market order fallback for positions that couldn't be unwound with limit orders.

        This is the last resort when aggressive limit orders fail to unwind positions.
        Uses market orders to ensure positions are liquidated, accepting higher slippage.
        """
        logger.warning(
            f"[{self.symbol}] Executing MARKET ORDER FALLBACK for {len(remaining_positions)} positions"
        )

        fallback_attempt = RollbackAttempt(
            attempt_id=f"market_fallback_{int(time.time())}",
            attempt_number=ROLLBACK_MAX_ATTEMPTS + 1,  # Beyond normal attempts
            strategy_used=RollbackStrategy.IMMEDIATE_MARKET,
            timestamp=time.time(),
            positions_targeted=remaining_positions.copy(),
        )

        unwound_count = 0
        total_cost = 0.0
        total_slippage = 0.0
        fallback_start = time.time()

        # Sort by priority (stock first, then options)
        sorted_positions = sorted(
            remaining_positions, key=lambda p: p.unwinding_priority
        )

        for position in sorted_positions:
            if position.unwind_status != "pending":
                continue

            try:
                # Cancel any existing limit orders first
                if position.rollback_trade:
                    try:
                        self.ib.cancelOrder(position.rollback_order)
                        await asyncio.sleep(0.1)  # Brief pause
                    except Exception:
                        pass  # Ignore cancellation errors in fallback

                # Determine opposite action for unwinding
                unwind_action = "SELL" if position.leg_order.action == "BUY" else "BUY"

                # Create market order for immediate execution
                market_order = Order(
                    orderId=self.ib.client.getReqId(),
                    orderType="MKT",
                    action=unwind_action,
                    totalQuantity=position.current_quantity,
                    tif="DAY",
                )

                # Place market order
                market_trade = self.ib.placeOrder(
                    position.leg_order.contract, market_order
                )
                position.rollback_trade = market_trade
                position.rollback_order = market_order

                # Update tracking
                self._update_rollback_tracking(rollback_plan, position, "unwinding")
                self._log_rollback_order_placement(position, "market_fallback")

                logger.warning(
                    f"[{self.symbol}] MARKET FALLBACK: Placed market {unwind_action} order for "
                    f"{position.leg_order.leg_type.value} (quantity: {position.current_quantity})"
                )

                # Wait for market fill with short timeout
                fill_start = time.time()
                fill_timeout = 8.0  # 8 seconds for market order (should be fast)

                while time.time() - fill_start < fill_timeout:
                    if market_trade.orderStatus.status == "Filled":
                        fill_price = market_trade.orderStatus.avgFillPrice
                        slippage = abs(fill_price - position.avg_fill_price)
                        cost = fill_price * position.current_quantity

                        position.unwind_status = "completed"
                        unwound_count += 1
                        total_cost += cost
                        total_slippage += slippage

                        logger.warning(
                            f"[{self.symbol}] ✓ MARKET FALLBACK FILL: {position.leg_order.leg_type.value} "
                            f"@ {fill_price:.2f} (slippage: ${slippage:.2f}, HIGH RISK)"
                        )
                        break

                    await asyncio.sleep(0.1)

                if position.unwind_status != "completed":
                    position.unwind_status = "failed"
                    logger.error(
                        f"[{self.symbol}] MARKET FALLBACK TIMEOUT: {position.leg_order.leg_type.value} "
                        f"market order failed to fill within {fill_timeout}s"
                    )

            except Exception as e:
                position.unwind_status = "failed"
                logger.error(
                    f"[{self.symbol}] Error in market fallback for "
                    f"{position.leg_order.leg_type.value}: {e}"
                )

        # Update attempt results
        fallback_attempt.positions_unwound = unwound_count
        fallback_attempt.attempt_cost = total_cost
        fallback_attempt.attempt_slippage = total_slippage
        fallback_attempt.duration = time.time() - fallback_start
        fallback_attempt.success = unwound_count > 0

        self._add_rollback_attempt(fallback_attempt)

        if unwound_count == len(remaining_positions):
            logger.warning(
                f"[{self.symbol}] ✓ MARKET FALLBACK SUCCESS: All {unwound_count} positions unwound "
                f"(cost: ${total_cost:.2f}, slippage: ${total_slippage:.2f})"
            )
        elif unwound_count > 0:
            logger.error(
                f"[{self.symbol}] ⚠️ MARKET FALLBACK PARTIAL: {unwound_count}/{len(remaining_positions)} "
                f"positions unwound - MANUAL INTERVENTION REQUIRED"
            )
        else:
            logger.critical(
                f"[{self.symbol}] ❌ MARKET FALLBACK FAILED: No positions unwound - "
                f"CRITICAL MANUAL INTERVENTION REQUIRED"
            )

        return {
            "unwound_count": unwound_count,
            "cost": total_cost,
            "slippage": total_slippage,
            "duration": fallback_attempt.duration,
            "method": "market_fallback",
            "remaining_positions": len(remaining_positions) - unwound_count,
        }

    def get_rollback_order_status(self, rollback_id: str) -> Dict:
        """Get comprehensive status of all orders in a rollback."""
        if rollback_id not in self.active_rollbacks:
            completed = next(
                (
                    rb
                    for rb in self.completed_rollbacks
                    if rb.rollback_id == rollback_id
                ),
                None,
            )
            if completed:
                return self._get_rollback_status_dict(completed)
            return {"error": f"Rollback {rollback_id} not found"}

        rollback_plan = self.active_rollbacks[rollback_id]
        return self._get_rollback_status_dict(rollback_plan)

    def _get_rollback_status_dict(self, rollback_plan: RollbackPlan) -> Dict:
        """Create comprehensive status dictionary for a rollback."""
        order_statuses = []

        for position in rollback_plan.positions_to_unwind:
            if position.rollback_order:
                order_status = {
                    "leg_type": position.leg_order.leg_type.value,
                    "contract": f"{position.leg_order.contract.symbol} {position.leg_order.contract.strike}",
                    "order_id": position.rollback_order.orderId,
                    "quantity": position.current_quantity,
                    "avg_fill_price": position.avg_fill_price,
                    "target_price": position.rollback_target_price,
                    "unwind_status": position.unwind_status,
                    "unrealized_pnl": position.unrealized_pnl,
                    "priority": position.unwinding_priority,
                }

                # Add trade info if available
                if position.rollback_trade:
                    trade = position.rollback_trade
                    order_status.update(
                        {
                            "trade_status": (
                                trade.orderStatus.status
                                if trade.orderStatus
                                else "Unknown"
                            ),
                            "filled_quantity": (
                                trade.orderStatus.filled if trade.orderStatus else 0
                            ),
                            "remaining_quantity": (
                                trade.orderStatus.remaining
                                if trade.orderStatus
                                else position.current_quantity
                            ),
                            "avg_fill_price_actual": (
                                trade.orderStatus.avgFillPrice
                                if trade.orderStatus
                                else None
                            ),
                        }
                    )

                order_statuses.append(order_status)

        return {
            "rollback_id": rollback_plan.rollback_id,
            "symbol": rollback_plan.symbol,
            "execution_id": rollback_plan.execution_id,
            "status": rollback_plan.rollback_status,
            "reason": rollback_plan.reason.value,
            "strategy": rollback_plan.strategy.value,
            "positions_to_unwind": rollback_plan.total_positions,
            "positions_unwound": rollback_plan.positions_unwound,
            "total_cost": rollback_plan.total_rollback_cost,
            "slippage": rollback_plan.rollback_slippage,
            "max_acceptable_loss": rollback_plan.max_acceptable_loss,
            "estimated_cost": rollback_plan.estimated_unwinding_cost,
            "created_time": rollback_plan.created_time,
            "completion_time": rollback_plan.completion_time,
            "duration": (
                (rollback_plan.completion_time - rollback_plan.created_time)
                if rollback_plan.completion_time
                else None
            ),
            "success": rollback_plan.success,
            "error_message": rollback_plan.error_message,
            "orders": order_statuses,
        }

    def get_all_active_rollbacks_status(self) -> Dict[str, Dict]:
        """Get status of all active rollbacks."""
        return {
            rollback_id: self._get_rollback_status_dict(rollback_plan)
            for rollback_id, rollback_plan in self.active_rollbacks.items()
        }

    def get_rollback_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics for rollback operations."""
        metrics = dict(self.performance_metrics)

        # Calculate success rate
        total_rollbacks = metrics["total_rollbacks_initiated"]
        if total_rollbacks > 0:
            metrics["success_rate"] = metrics["successful_rollbacks"] / total_rollbacks
        else:
            metrics["success_rate"] = 0.0

        # Calculate attempt statistics
        if self.rollback_attempts:
            successful_attempts = [a for a in self.rollback_attempts if a.success]
            failed_attempts = [a for a in self.rollback_attempts if not a.success]

            metrics["total_attempts"] = len(self.rollback_attempts)
            metrics["successful_attempts"] = len(successful_attempts)
            metrics["failed_attempts"] = len(failed_attempts)

            if successful_attempts:
                metrics["avg_successful_duration"] = sum(
                    a.duration for a in successful_attempts
                ) / len(successful_attempts)
                metrics["avg_successful_cost"] = sum(
                    a.attempt_cost for a in successful_attempts
                ) / len(successful_attempts)
                metrics["avg_successful_slippage"] = sum(
                    a.attempt_slippage for a in successful_attempts
                ) / len(successful_attempts)

            # Strategy breakdown
            strategy_counts = {}
            for attempt in self.rollback_attempts:
                strategy = attempt.strategy_used.value
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            metrics["strategy_usage"] = strategy_counts

        return metrics

    def monitor_rollback_progress(self, rollback_id: str) -> Dict:
        """Monitor real-time progress of an active rollback."""
        if rollback_id not in self.active_rollbacks:
            return {"error": f"Active rollback {rollback_id} not found"}

        rollback_plan = self.active_rollbacks[rollback_id]
        current_time = time.time()
        elapsed_time = current_time - rollback_plan.created_time
        time_remaining = rollback_plan.max_rollback_time - elapsed_time

        # Calculate progress percentage
        progress = 0.0
        if rollback_plan.total_positions > 0:
            progress = rollback_plan.positions_unwound / rollback_plan.total_positions

        # Check for stalled orders
        stalled_orders = []
        for position in rollback_plan.positions_to_unwind:
            if position.rollback_trade and position.rollback_trade.orderStatus:
                order_status = position.rollback_trade.orderStatus.status
                if (
                    order_status in ["Submitted", "PreSubmitted"]
                    and position.unwind_status == "unwinding"
                ):
                    # Check if order has been pending too long
                    if (
                        hasattr(position.rollback_trade, "log")
                        and position.rollback_trade.log
                    ):
                        last_log = position.rollback_trade.log[-1]
                        if hasattr(last_log, "time"):
                            time_since_last_update = (
                                current_time - last_log.time.timestamp()
                            )
                            if time_since_last_update > 30:  # 30 seconds without update
                                stalled_orders.append(
                                    {
                                        "order_id": position.rollback_order.orderId,
                                        "leg_type": position.leg_order.leg_type.value,
                                        "time_stalled": time_since_last_update,
                                    }
                                )

        return {
            "rollback_id": rollback_id,
            "status": rollback_plan.rollback_status,
            "progress_percent": progress * 100,
            "positions_unwound": rollback_plan.positions_unwound,
            "total_positions": rollback_plan.total_positions,
            "elapsed_time": elapsed_time,
            "time_remaining": max(0, time_remaining),
            "current_cost": rollback_plan.total_rollback_cost,
            "estimated_cost": rollback_plan.estimated_unwinding_cost,
            "current_slippage": rollback_plan.rollback_slippage,
            "stalled_orders": stalled_orders,
            "risk_metrics": {
                "cost_vs_estimate": rollback_plan.total_rollback_cost
                / max(rollback_plan.estimated_unwinding_cost, 0.01),
                "within_loss_limit": rollback_plan.total_rollback_cost
                <= rollback_plan.max_acceptable_loss,
                "time_pressure": 1
                - (
                    time_remaining
                    / (rollback_plan.max_rollback_time - rollback_plan.created_time)
                ),
            },
        }

    def get_rollback_attempts_history(self, limit: int = 10) -> List[Dict]:
        """Get history of rollback attempts with details."""
        recent_attempts = (
            self.rollback_attempts[-limit:]
            if len(self.rollback_attempts) > limit
            else self.rollback_attempts
        )

        return [
            {
                "attempt_id": attempt.attempt_id,
                "attempt_number": attempt.attempt_number,
                "strategy": attempt.strategy_used.value,
                "timestamp": datetime.fromtimestamp(attempt.timestamp).isoformat(),
                "positions_targeted": len(attempt.positions_targeted),
                "positions_unwound": attempt.positions_unwound,
                "cost": attempt.attempt_cost,
                "slippage": attempt.attempt_slippage,
                "duration": attempt.duration,
                "success": attempt.success,
                "error_message": attempt.error_message,
            }
            for attempt in reversed(recent_attempts)
        ]

    def export_rollback_report(self, rollback_id: str, format: str = "dict") -> Dict:
        """Export comprehensive rollback report for analysis."""
        status = self.get_rollback_order_status(rollback_id)
        if "error" in status:
            return status

        # Add related attempts
        related_attempts = [
            attempt
            for attempt in self.rollback_attempts
            if any(
                pos.leg_order.execution_id == status["execution_id"]
                for pos in attempt.positions_targeted
            )
        ]

        report = {
            "rollback_summary": status,
            "attempts_history": [
                {
                    "attempt_id": attempt.attempt_id,
                    "attempt_number": attempt.attempt_number,
                    "strategy": attempt.strategy_used.value,
                    "timestamp": datetime.fromtimestamp(attempt.timestamp).isoformat(),
                    "duration": attempt.duration,
                    "success": attempt.success,
                    "positions_unwound": attempt.positions_unwound,
                    "cost": attempt.attempt_cost,
                    "slippage": attempt.attempt_slippage,
                    "error_message": attempt.error_message,
                }
                for attempt in related_attempts
            ],
            "performance_context": self.get_rollback_performance_metrics(),
            "report_generated": datetime.now().isoformat(),
        }

        return report

    def _update_rollback_tracking(
        self, rollback_plan: RollbackPlan, position: RollbackPosition, status: str
    ):
        """Update rollback tracking when order status changes."""
        position.unwind_status = status

        # Log the status change
        logger.info(
            f"[{self.symbol}] Rollback tracking update - "
            f"Order {position.rollback_order.orderId if position.rollback_order else 'N/A'} "
            f"for {position.leg_order.leg_type.value} -> {status}"
        )

        # Update rollback plan status based on individual position statuses
        if status == "completed":
            rollback_plan.positions_unwound += 1

            # Check if all positions are completed
            completed_positions = sum(
                1
                for pos in rollback_plan.positions_to_unwind
                if pos.unwind_status == "completed"
            )
            if completed_positions == rollback_plan.total_positions:
                rollback_plan.rollback_status = "completed"
                rollback_plan.success = True
                rollback_plan.completion_time = time.time()
                logger.info(
                    f"[{self.symbol}] Rollback {rollback_plan.rollback_id} completed successfully"
                )

        elif status == "failed":
            # Check if any position has failed
            failed_positions = [
                pos
                for pos in rollback_plan.positions_to_unwind
                if pos.unwind_status == "failed"
            ]
            if failed_positions:
                rollback_plan.rollback_status = "failed"
                rollback_plan.success = False
                rollback_plan.completion_time = time.time()
                logger.error(
                    f"[{self.symbol}] Rollback {rollback_plan.rollback_id} failed"
                )

        elif status == "unwinding":
            if rollback_plan.rollback_status == "pending":
                rollback_plan.rollback_status = "executing"

    def _log_rollback_order_placement(
        self, position: RollbackPosition, order_type: str
    ):
        """Log when a rollback order is placed."""
        if position.rollback_order:
            logger.info(
                f"[{self.symbol}] Rollback {order_type} order placed - "
                f"ID: {position.rollback_order.orderId}, "
                f"Leg: {position.leg_order.leg_type.value}, "
                f"Qty: {position.current_quantity}, "
                f"Target: ${position.rollback_target_price:.2f if position.rollback_target_price else 0}"
            )

    def _log_rollback_order_fill(
        self, position: RollbackPosition, fill_price: float, fill_qty: int
    ):
        """Log when a rollback order is filled."""
        logger.info(
            f"[{self.symbol}] Rollback order filled - "
            f"ID: {position.rollback_order.orderId if position.rollback_order else 'N/A'}, "
            f"Leg: {position.leg_order.leg_type.value}, "
            f"Qty: {fill_qty}, "
            f"Price: ${fill_price:.2f}, "
            f"Slippage: ${abs(fill_price - (position.rollback_target_price or 0)):.2f}"
        )
