"""
Complete parallel leg executor for SFR arbitrage strategy.

This module implements the full parallel execution logic that replaces combo orders
with simultaneous individual leg orders to eliminate slippage issues. It provides:

1. Complete execution monitoring with real-time fill detection
2. Sophisticated partial fill handling and rollback mechanisms
3. Beautiful execution reporting with slippage analysis
4. Integration with existing SFR executor framework
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple

from ib_async import IB, Contract, Order, OrderStatus, Trade

from ..common import get_logger
from .constants import (
    PARALLEL_EXECUTION_TIMEOUT,
    PARALLEL_FILL_TIMEOUT_PER_LEG,
    PARALLEL_MAX_GLOBAL_ATTEMPTS,
    PARALLEL_MAX_SLIPPAGE_PERCENT,
    PARALLEL_MAX_SYMBOL_ATTEMPTS,
)
from .execution_reporter import ExecutionReporter
from .global_execution_lock import GlobalExecutionLock, acquire_global_lock
from .parallel_execution_framework import (
    ExecutionState,
    LegOrder,
    LegType,
    ParallelExecutionFramework,
    ParallelExecutionPlan,
)
from .partial_fill_handler import PartialFillHandler
from .rollback_manager import RollbackManager

logger = get_logger()


@dataclass
class ExecutionResult:
    """Result of parallel execution attempt."""

    success: bool
    execution_id: str
    symbol: str
    total_execution_time: float

    # Fill information
    all_legs_filled: bool
    partially_filled: bool
    legs_filled: int
    total_legs: int

    # Price analysis
    expected_total_cost: float
    actual_total_cost: float
    total_slippage: float
    slippage_percentage: float

    # Individual leg results
    stock_result: Optional[Dict] = None
    call_result: Optional[Dict] = None
    put_result: Optional[Dict] = None

    # Error information
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    requires_rollback: bool = False

    # Execution metrics
    order_placement_time: float = 0.0
    fill_monitoring_time: float = 0.0
    rollback_time: float = 0.0


class ParallelLegExecutor:
    """
    Complete parallel leg executor for SFR arbitrage.

    This executor orchestrates the entire parallel execution process:
    1. Acquires global execution lock to prevent interference
    2. Places all 3 orders simultaneously with sub-second timing
    3. Monitors fills in real-time with sophisticated timeout handling
    4. Handles partial fills with intelligent rollback strategies
    5. Provides beautiful execution reporting and analysis
    6. Manages executor pause/resume for system coordination
    """

    def __init__(
        self,
        ib: IB,
        symbol: str,
        on_execution_complete: Optional[Callable] = None,
        on_execution_failed: Optional[Callable] = None,
    ):
        self.ib = ib
        self.symbol = symbol
        self.executor_id = f"parallel_{symbol}_{int(time.time())}"

        # Callback functions
        self.on_execution_complete = on_execution_complete
        self.on_execution_failed = on_execution_failed

        # Framework components
        self.framework: Optional[ParallelExecutionFramework] = None
        self.global_lock: Optional[GlobalExecutionLock] = None
        self.reporter = ExecutionReporter()
        self.rollback_manager = RollbackManager(ib, symbol)
        self.fill_handler = PartialFillHandler(ib, symbol)

        # Execution tracking
        self.current_execution: Optional[ParallelExecutionPlan] = None
        self.execution_history: List[ExecutionResult] = []
        self._max_execution_history = 50  # Limit history to prevent memory leaks
        self.active_trades: Set[Trade] = set()

        # Performance metrics
        self._total_attempts = 0
        self._successful_executions = 0
        self._failed_executions = 0
        self._total_execution_time = 0.0

        # Global attempt tracking for limits
        self._global_attempt_count = 0
        self._symbol_attempt_count = {symbol: 0}

        logger.debug(f"[{symbol}] ParallelLegExecutor initialized: {self.executor_id}")

    async def initialize(self) -> bool:
        """Initialize the parallel leg executor."""
        try:
            # Get framework components
            self.framework = ParallelExecutionFramework(self.ib)
            await self.framework.initialize()

            self.global_lock = await GlobalExecutionLock.get_instance()

            # Initialize sub-components
            await self.rollback_manager.initialize()
            await self.fill_handler.initialize()

            logger.info(
                f"[{self.symbol}] Parallel leg executor initialized successfully"
            )
            return True

        except Exception as e:
            logger.error(f"[{self.symbol}] Failed to initialize parallel executor: {e}")
            return False

    async def execute_parallel_arbitrage(
        self,
        stock_contract: Contract,
        call_contract: Contract,
        put_contract: Contract,
        stock_price: float,
        call_price: float,
        put_price: float,
        quantity: int = 1,
        profit_target: float = 0.5,
        execution_params: Optional[Dict] = None,
    ) -> ExecutionResult:
        """
        Execute SFR arbitrage with parallel leg orders.

        Args:
            stock_contract: Stock contract for purchase
            call_contract: Call option contract to sell
            put_contract: Put option contract to buy
            stock_price: Target stock execution price
            call_price: Target call execution price (what we want to receive)
            put_price: Target put execution price (what we want to pay)
            quantity: Number of contracts to trade
            profit_target: Target profit threshold
            execution_params: Optional execution parameters

        Returns:
            ExecutionResult with complete execution details
        """
        self._total_attempts += 1
        self._global_attempt_count += 1
        self._symbol_attempt_count[self.symbol] += 1

        # Check attempt limits
        if self._global_attempt_count > PARALLEL_MAX_GLOBAL_ATTEMPTS:
            return self._create_error_result(
                "global_attempt_limit_exceeded",
                f"Global attempt limit of {PARALLEL_MAX_GLOBAL_ATTEMPTS} exceeded",
            )

        if self._symbol_attempt_count[self.symbol] > PARALLEL_MAX_SYMBOL_ATTEMPTS:
            return self._create_error_result(
                "symbol_attempt_limit_exceeded",
                f"Symbol attempt limit of {PARALLEL_MAX_SYMBOL_ATTEMPTS} exceeded for {self.symbol}",
            )

        execution_start_time = time.time()

        # Step 1: Acquire global execution lock
        logger.info(
            f"[{self.symbol}] Starting parallel execution attempt #{self._symbol_attempt_count[self.symbol]}"
        )

        lock_acquired = await self.global_lock.acquire(
            self.symbol, self.executor_id, "parallel_execution", timeout=10.0
        )

        if not lock_acquired:
            return self._create_error_result(
                "lock_timeout", "Failed to acquire global execution lock within timeout"
            )

        try:
            # Step 2: Create execution plan
            expiry = getattr(call_contract, "lastTradeDateOrContractMonth", "unknown")

            execution_plan = await self.framework.create_execution_plan(
                symbol=self.symbol,
                expiry=expiry,
                stock_contract=stock_contract,
                call_contract=call_contract,
                put_contract=put_contract,
                stock_price=stock_price,
                call_price=call_price,
                put_price=put_price,
                quantity=quantity,
                execution_params=execution_params,
            )

            self.current_execution = execution_plan

            # Calculate expected costs for analysis
            expected_total_cost = (
                stock_price * quantity * 100
                - call_price * quantity
                + put_price * quantity
            )

            logger.info(
                f"[{self.symbol}] Execution plan {execution_plan.execution_id}: "
                f"Expected cost: ${expected_total_cost:.2f}, Target profit: ${profit_target:.2f}"
            )

            # Step 3: Execute the parallel strategy
            result = await self._execute_parallel_strategy(
                execution_plan, expected_total_cost, profit_target, execution_start_time
            )

            # Step 4: Handle result and update metrics
            if result.success:
                self._successful_executions += 1
                logger.info(
                    f"[{self.symbol}] ✓ Parallel execution SUCCESSFUL: {result.execution_id}"
                )

                if self.on_execution_complete:
                    try:
                        await self.on_execution_complete(result)
                    except Exception as e:
                        logger.warning(
                            f"[{self.symbol}] Error in completion callback: {e}"
                        )
            else:
                self._failed_executions += 1
                logger.warning(
                    f"[{self.symbol}] ✗ Parallel execution FAILED: {result.error_message}"
                )

                if self.on_execution_failed:
                    try:
                        await self.on_execution_failed(result)
                    except Exception as e:
                        logger.warning(
                            f"[{self.symbol}] Error in failure callback: {e}"
                        )

            # Add to execution history
            self.execution_history.append(result)
            # Limit history size to prevent memory leaks
            if len(self.execution_history) > self._max_execution_history:
                self.execution_history.pop(0)
            self._total_execution_time += result.total_execution_time

            return result

        finally:
            # Always release the global lock
            self.global_lock.release(self.symbol, self.executor_id)
            self.current_execution = None

    async def _execute_parallel_strategy(
        self,
        plan: ParallelExecutionPlan,
        expected_cost: float,
        profit_target: float,
        start_time: float,
    ) -> ExecutionResult:
        """Execute the core parallel strategy logic."""

        try:
            # Phase 1: Place orders in parallel (sub-second timing critical)
            logger.info(f"[{self.symbol}] Phase 1: Placing orders in parallel")
            placement_start = time.time()

            placement_success = await self.framework.place_orders_parallel(plan)
            placement_time = time.time() - placement_start

            if not placement_success:
                return self._create_error_result(
                    "order_placement_failed",
                    plan.execution_error or "Failed to place orders",
                    placement_time=placement_time,
                )

            logger.info(f"[{self.symbol}] ✓ All orders placed in {placement_time:.3f}s")

            # Phase 2: Monitor fills with sophisticated timeout handling
            logger.info(f"[{self.symbol}] Phase 2: Monitoring fills")
            monitoring_start = time.time()

            fill_result = await self._monitor_fills_with_timeout(plan)
            monitoring_time = time.time() - monitoring_start

            # Phase 3: Analyze results and handle partial fills
            analysis_start = time.time()

            if fill_result["all_filled"]:
                # Success case - calculate actual costs and slippage
                actual_cost = self._calculate_actual_cost(plan)
                slippage = actual_cost - expected_cost
                slippage_pct = (
                    (slippage / abs(expected_cost)) * 100 if expected_cost != 0 else 0
                )

                logger.info(
                    f"[{self.symbol}] ✓ All legs filled! "
                    f"Expected: ${expected_cost:.2f}, Actual: ${actual_cost:.2f}, "
                    f"Slippage: ${slippage:.2f} ({slippage_pct:.2f}%)"
                )

                # Check if slippage is within acceptable limits
                if abs(slippage_pct) > PARALLEL_MAX_SLIPPAGE_PERCENT:
                    logger.warning(
                        f"[{self.symbol}] Slippage {slippage_pct:.2f}% exceeds limit of {PARALLEL_MAX_SLIPPAGE_PERCENT}%"
                    )

                total_time = time.time() - start_time

                return ExecutionResult(
                    success=True,
                    execution_id=plan.execution_id,
                    symbol=self.symbol,
                    total_execution_time=total_time,
                    all_legs_filled=True,
                    partially_filled=False,
                    legs_filled=3,
                    total_legs=3,
                    expected_total_cost=expected_cost,
                    actual_total_cost=actual_cost,
                    total_slippage=slippage,
                    slippage_percentage=slippage_pct,
                    stock_result=self._get_leg_result(plan.stock_leg),
                    call_result=self._get_leg_result(plan.call_leg),
                    put_result=self._get_leg_result(plan.put_leg),
                    order_placement_time=placement_time,
                    fill_monitoring_time=monitoring_time,
                )

            else:
                # Partial fill case - initiate rollback
                logger.warning(
                    f"[{self.symbol}] Partial fills detected: "
                    f"{fill_result['filled_count']}/3 legs filled"
                )

                # Phase 4: Execute rollback strategy
                rollback_start = time.time()
                rollback_result = await self._execute_rollback_strategy(
                    plan, fill_result
                )
                rollback_time = time.time() - rollback_start

                total_time = time.time() - start_time

                return ExecutionResult(
                    success=False,
                    execution_id=plan.execution_id,
                    symbol=self.symbol,
                    total_execution_time=total_time,
                    all_legs_filled=False,
                    partially_filled=True,
                    legs_filled=fill_result["filled_count"],
                    total_legs=3,
                    expected_total_cost=expected_cost,
                    actual_total_cost=self._calculate_actual_cost(plan),
                    total_slippage=0.0,  # Will be calculated if rollback succeeds
                    slippage_percentage=0.0,
                    stock_result=self._get_leg_result(plan.stock_leg),
                    call_result=self._get_leg_result(plan.call_leg),
                    put_result=self._get_leg_result(plan.put_leg),
                    error_type="partial_fills",
                    error_message=f"Only {fill_result['filled_count']}/3 legs filled, rollback executed",
                    requires_rollback=True,
                    order_placement_time=placement_time,
                    fill_monitoring_time=monitoring_time,
                    rollback_time=rollback_time,
                )

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"[{self.symbol}] Critical error in parallel execution: {e}")

            return ExecutionResult(
                success=False,
                execution_id=plan.execution_id,
                symbol=self.symbol,
                total_execution_time=total_time,
                all_legs_filled=False,
                partially_filled=False,
                legs_filled=0,
                total_legs=3,
                expected_total_cost=expected_cost,
                actual_total_cost=0.0,
                total_slippage=0.0,
                slippage_percentage=0.0,
                error_type="execution_exception",
                error_message=str(e),
            )

    async def _monitor_fills_with_timeout(self, plan: ParallelExecutionPlan) -> Dict:
        """Monitor order fills with sophisticated timeout handling."""

        legs = self.framework._get_all_legs(plan)
        monitoring_start = time.time()

        # Track fill status
        fill_status = {
            "all_filled": False,
            "filled_count": 0,
            "pending_legs": legs.copy(),
            "filled_legs": [],
            "timeout_occurred": False,
        }

        logger.info(f"[{self.symbol}] Monitoring {len(legs)} legs for fills")

        try:
            # Monitor with timeout
            while time.time() - monitoring_start < PARALLEL_EXECUTION_TIMEOUT:
                filled_this_cycle = []

                # Check each pending leg
                for leg in fill_status["pending_legs"]:
                    if await self._check_leg_filled(leg):
                        filled_this_cycle.append(leg)
                        fill_status["filled_legs"].append(leg)
                        logger.info(
                            f"[{self.symbol}] ✓ {leg.leg_type.value} leg filled: "
                            f"{leg.filled_quantity}@${leg.avg_fill_price:.2f}"
                        )

                # Remove filled legs from pending
                for leg in filled_this_cycle:
                    fill_status["pending_legs"].remove(leg)

                fill_status["filled_count"] = len(fill_status["filled_legs"])

                # Check if all legs are filled
                if fill_status["filled_count"] == len(legs):
                    fill_status["all_filled"] = True
                    logger.info(f"[{self.symbol}] ✓ ALL LEGS FILLED successfully!")
                    break

                # Check individual leg timeouts
                current_time = time.time()
                for leg in fill_status["pending_legs"]:
                    leg_elapsed = current_time - (leg.fill_time or monitoring_start)
                    if leg_elapsed > PARALLEL_FILL_TIMEOUT_PER_LEG:
                        logger.warning(
                            f"[{self.symbol}] {leg.leg_type.value} leg timeout after {leg_elapsed:.1f}s"
                        )

                # Brief pause before next check
                await asyncio.sleep(0.1)

            # Check for overall timeout
            if not fill_status["all_filled"]:
                total_elapsed = time.time() - monitoring_start
                if total_elapsed >= PARALLEL_EXECUTION_TIMEOUT:
                    fill_status["timeout_occurred"] = True
                    logger.warning(
                        f"[{self.symbol}] Fill monitoring timeout after {total_elapsed:.1f}s: "
                        f"{fill_status['filled_count']}/{len(legs)} legs filled"
                    )

            return fill_status

        except Exception as e:
            logger.error(f"[{self.symbol}] Error in fill monitoring: {e}")
            return fill_status

    async def _check_leg_filled(self, leg: LegOrder) -> bool:
        """Check if a leg order is filled."""
        if not leg.trade:
            return False

        # Update trade status
        try:
            # Get latest order status
            order_status = leg.trade.orderStatus

            if order_status.status in ["Filled"]:
                leg.fill_status = "filled"
                leg.filled_quantity = order_status.filled
                leg.avg_fill_price = order_status.avgFillPrice or leg.target_price
                leg.fill_time = time.time()
                return True
            elif order_status.status in ["PartiallyFilled"]:
                leg.fill_status = "partial"
                leg.filled_quantity = order_status.filled
                leg.avg_fill_price = order_status.avgFillPrice or leg.target_price
                return False
            elif order_status.status in ["Cancelled", "Inactive"]:
                leg.fill_status = "cancelled"
                return False
            else:
                # Still pending
                return False

        except Exception as e:
            logger.debug(
                f"[{self.symbol}] Error checking {leg.leg_type.value} leg status: {e}"
            )
            return False

    async def _execute_rollback_strategy(
        self, plan: ParallelExecutionPlan, fill_result: Dict
    ) -> Dict:
        """Execute sophisticated rollback strategy for partial fills."""
        logger.info(f"[{self.symbol}] Executing rollback strategy")

        try:
            # Use RollbackManager for sophisticated rollback
            rollback_result = await self.rollback_manager.execute_rollback(
                plan, fill_result["filled_legs"], fill_result["pending_legs"]
            )

            return rollback_result

        except Exception as e:
            logger.error(f"[{self.symbol}] Error in rollback execution: {e}")
            return {"success": False, "error": str(e), "positions_unwound": False}

    def _calculate_actual_cost(self, plan: ParallelExecutionPlan) -> float:
        """Calculate actual total cost based on fill prices."""
        total_cost = 0.0

        # Stock cost (positive - we're buying)
        if plan.stock_leg and plan.stock_leg.fill_status == "filled":
            total_cost += plan.stock_leg.avg_fill_price * plan.stock_leg.filled_quantity

        # Call credit (negative - we're selling)
        if plan.call_leg and plan.call_leg.fill_status == "filled":
            total_cost -= plan.call_leg.avg_fill_price * plan.call_leg.filled_quantity

        # Put cost (positive - we're buying)
        if plan.put_leg and plan.put_leg.fill_status == "filled":
            total_cost += plan.put_leg.avg_fill_price * plan.put_leg.filled_quantity

        return total_cost

    def _get_leg_result(self, leg: Optional[LegOrder]) -> Optional[Dict]:
        """Get detailed result information for a leg."""
        if not leg:
            return None

        return {
            "leg_type": leg.leg_type.value,
            "action": leg.action,
            "quantity": leg.quantity,
            "target_price": leg.target_price,
            "fill_status": leg.fill_status,
            "filled_quantity": leg.filled_quantity,
            "avg_fill_price": leg.avg_fill_price,
            "fill_time": leg.fill_time,
            "slippage": (
                (leg.avg_fill_price - leg.target_price) if leg.avg_fill_price else 0.0
            ),
            "order_id": leg.order_id,
        }

    def _create_error_result(
        self, error_type: str, error_message: str, placement_time: float = 0.0
    ) -> ExecutionResult:
        """Create an error result for failed executions."""
        return ExecutionResult(
            success=False,
            execution_id=f"error_{int(time.time())}",
            symbol=self.symbol,
            total_execution_time=0.0,
            all_legs_filled=False,
            partially_filled=False,
            legs_filled=0,
            total_legs=3,
            expected_total_cost=0.0,
            actual_total_cost=0.0,
            total_slippage=0.0,
            slippage_percentage=0.0,
            error_type=error_type,
            error_message=error_message,
            order_placement_time=placement_time,
        )

    def get_execution_stats(self) -> Dict:
        """Get comprehensive execution statistics."""
        success_rate = (
            self._successful_executions / self._total_attempts
            if self._total_attempts > 0
            else 0.0
        )

        avg_execution_time = (
            self._total_execution_time / self._successful_executions
            if self._successful_executions > 0
            else 0.0
        )

        return {
            "executor_id": self.executor_id,
            "symbol": self.symbol,
            "total_attempts": self._total_attempts,
            "successful_executions": self._successful_executions,
            "failed_executions": self._failed_executions,
            "success_rate_percent": success_rate * 100,
            "average_execution_time_seconds": avg_execution_time,
            "global_attempt_count": self._global_attempt_count,
            "symbol_attempt_count": self._symbol_attempt_count.get(self.symbol, 0),
            "execution_history_size": len(self.execution_history),
            "currently_executing": self.current_execution is not None,
        }

    def get_recent_executions(self, count: int = 5) -> List[Dict]:
        """Get recent execution results summary."""
        recent = (
            self.execution_history[-count:] if count > 0 else self.execution_history
        )

        return [
            {
                "execution_id": result.execution_id,
                "success": result.success,
                "execution_time": result.total_execution_time,
                "legs_filled": f"{result.legs_filled}/{result.total_legs}",
                "slippage_percent": result.slippage_percentage,
                "error_type": result.error_type,
                "timestamp": datetime.now().isoformat(),  # Would use actual timestamp in production
            }
            for result in recent
        ]

    async def cancel_current_execution(self, reason: str = "manual_cancel") -> bool:
        """Cancel currently active execution."""
        if not self.current_execution:
            logger.info(f"[{self.symbol}] No active execution to cancel")
            return False

        logger.info(f"[{self.symbol}] Cancelling current execution: {reason}")

        # Use framework to cancel
        success = self.framework.cancel_execution(
            self.current_execution.execution_id, reason
        )

        if success:
            # Release global lock if held
            if self.global_lock and self.global_lock.is_locked():
                self.global_lock.release(self.symbol, self.executor_id)

        return success

    def reset_attempt_counters(self) -> None:
        """Reset attempt counters (use carefully)."""
        logger.info(f"[{self.symbol}] Resetting attempt counters")
        self._global_attempt_count = 0
        self._symbol_attempt_count[self.symbol] = 0
