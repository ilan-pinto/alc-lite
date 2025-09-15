"""
Basic parallel execution framework for SFR arbitrage.

This module provides the foundational framework for executing individual legs
of SFR arbitrage strategies in parallel, replacing the combo order approach
to eliminate slippage issues.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from ib_async import IB, Contract, Order, Trade

from ..common import get_logger
from .global_execution_lock import GlobalExecutionLock
from .utils import calculate_aggressive_execution_price, round_price_to_tick_size

logger = get_logger()


class ExecutionState(Enum):
    """States for parallel execution process."""

    INITIALIZING = "initializing"
    PLACING_ORDERS = "placing_orders"
    MONITORING_FILLS = "monitoring_fills"
    ROLLING_BACK = "rolling_back"
    COMPLETED_SUCCESS = "completed_success"
    COMPLETED_FAILURE = "completed_failure"
    CANCELLED = "cancelled"


class LegType(Enum):
    """Types of legs in SFR arbitrage."""

    STOCK = "stock"
    CALL = "call"
    PUT = "put"


@dataclass
class LegOrder:
    """Individual leg order information."""

    leg_type: LegType
    contract: Contract
    order: Order
    target_price: float
    action: str  # BUY or SELL
    quantity: int
    trade: Optional[Trade] = None
    fill_status: str = "pending"  # pending, filled, partial, failed, cancelled
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    fill_time: Optional[float] = None
    order_id: Optional[int] = None


@dataclass
class ParallelExecutionPlan:
    """Plan for parallel execution of SFR arbitrage legs."""

    execution_id: str
    symbol: str
    expiry: str
    strategy_type: str = "SFR"

    # Leg orders
    stock_leg: Optional[LegOrder] = None
    call_leg: Optional[LegOrder] = None
    put_leg: Optional[LegOrder] = None

    # Execution parameters
    max_execution_time: float = 5.0  # Maximum time to wait for all fills
    max_fill_time_per_leg: float = 2.0  # Maximum time to wait for individual leg
    slippage_tolerance: float = 0.02  # 2% slippage tolerance per leg

    # State tracking
    execution_state: ExecutionState = ExecutionState.INITIALIZING
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Results
    all_legs_filled: bool = False
    partially_filled: bool = False
    total_slippage: float = 0.0
    execution_error: Optional[str] = None


class ParallelExecutionFramework:
    """
    Framework for executing SFR arbitrage legs in parallel.

    This framework provides the foundation for:
    1. Simultaneous order placement for all legs
    2. Real-time fill monitoring
    3. Partial fill detection and handling
    4. Execution state management
    """

    def __init__(self, ib: IB):
        self.ib = ib
        self._active_executions: Dict[str, ParallelExecutionPlan] = {}
        self._global_lock: Optional[GlobalExecutionLock] = None

        # Performance tracking
        self._total_executions = 0
        self._successful_executions = 0
        self._failed_executions = 0
        self._average_execution_time = 0.0

        logger.debug("ParallelExecutionFramework initialized")

    async def initialize(self) -> bool:
        """Initialize the parallel execution framework."""
        try:
            self._global_lock = await GlobalExecutionLock.get_instance()
            logger.info("Parallel execution framework initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize parallel execution framework: {e}")
            return False

    async def create_execution_plan(
        self,
        symbol: str,
        expiry: str,
        stock_contract: Contract,
        call_contract: Contract,
        put_contract: Contract,
        stock_price: float,
        call_price: float,
        put_price: float,
        quantity: int = 1,
        execution_params: Optional[Dict] = None,
    ) -> ParallelExecutionPlan:
        """
        Create a parallel execution plan for SFR arbitrage (legacy method).

        For better fill rates, use create_execution_plan_with_tickers() instead.
        """
        return await self._create_execution_plan_internal(
            symbol,
            expiry,
            stock_contract,
            call_contract,
            put_contract,
            stock_price,
            call_price,
            put_price,
            quantity,
            execution_params,
            stock_ticker=None,
            call_ticker=None,
            put_ticker=None,
        )

    async def create_execution_plan_with_tickers(
        self,
        symbol: str,
        expiry: str,
        stock_contract: Contract,
        call_contract: Contract,
        put_contract: Contract,
        stock_price: float,
        call_price: float,
        put_price: float,
        stock_ticker=None,
        call_ticker=None,
        put_ticker=None,
        quantity: int = 1,
        execution_params: Optional[Dict] = None,
    ) -> ParallelExecutionPlan:
        """
        Create a parallel execution plan with ticker data for improved pricing.

        This version uses bid/ask data to create more aggressive prices that are
        more likely to fill quickly, reducing the chance of partial fills.
        """
        return await self._create_execution_plan_internal(
            symbol,
            expiry,
            stock_contract,
            call_contract,
            put_contract,
            stock_price,
            call_price,
            put_price,
            quantity,
            execution_params,
            stock_ticker,
            call_ticker,
            put_ticker,
        )

    async def _create_execution_plan_internal(
        self,
        symbol: str,
        expiry: str,
        stock_contract: Contract,
        call_contract: Contract,
        put_contract: Contract,
        stock_price: float,
        call_price: float,
        put_price: float,
        quantity: int = 1,
        execution_params: Optional[Dict] = None,
        stock_ticker=None,
        call_ticker=None,
        put_ticker=None,
    ) -> ParallelExecutionPlan:
        """
        Create a parallel execution plan for SFR arbitrage.

        Args:
            symbol: Trading symbol
            expiry: Option expiry date
            stock_contract: Stock contract
            call_contract: Call option contract
            put_contract: Put option contract
            stock_price: Target stock price
            call_price: Target call option price (what we want to receive)
            put_price: Target put option price (what we want to pay)
            quantity: Number of contracts
            execution_params: Optional execution parameters
            stock_ticker: Stock ticker for aggressive pricing (optional)
            call_ticker: Call option ticker for aggressive pricing (optional)
            put_ticker: Put option ticker for aggressive pricing (optional)

        Returns:
            ParallelExecutionPlan ready for execution
        """
        execution_id = str(uuid.uuid4())[:8]

        # Apply execution parameters
        params = execution_params or {}
        max_execution_time = params.get("max_execution_time", 5.0)
        max_fill_time_per_leg = params.get("max_fill_time_per_leg", 2.0)
        slippage_tolerance = params.get("slippage_tolerance", 0.02)

        # Enhanced pricing parameters
        pricing_aggressiveness = params.get(
            "pricing_aggressiveness", 0.02
        )  # 2% more aggressive

        # Calculate aggressive execution prices using bid/ask data when available
        if stock_ticker:
            aggressive_stock_price = calculate_aggressive_execution_price(
                stock_ticker, "BUY", stock_price, pricing_aggressiveness
            )
            logger.debug(
                f"[{symbol}] Stock aggressive pricing: {stock_price:.2f} -> {aggressive_stock_price:.2f}"
            )
        else:
            aggressive_stock_price = round_price_to_tick_size(stock_price, "stock")

        if call_ticker:
            aggressive_call_price = calculate_aggressive_execution_price(
                call_ticker, "SELL", call_price, pricing_aggressiveness
            )
            logger.debug(
                f"[{symbol}] Call aggressive pricing: {call_price:.2f} -> {aggressive_call_price:.2f}"
            )
        else:
            aggressive_call_price = round_price_to_tick_size(call_price, "option")

        if put_ticker:
            aggressive_put_price = calculate_aggressive_execution_price(
                put_ticker, "BUY", put_price, pricing_aggressiveness
            )
            logger.debug(
                f"[{symbol}] Put aggressive pricing: {put_price:.2f} -> {aggressive_put_price:.2f}"
            )
        else:
            aggressive_put_price = round_price_to_tick_size(put_price, "option")

        # Create leg orders with aggressive pricing
        stock_order = Order(
            orderId=self.ib.client.getReqId(),
            orderType="LMT",
            action="BUY",
            totalQuantity=quantity * 100,  # 100 shares per option contract
            lmtPrice=aggressive_stock_price,
            tif="DAY",
        )

        call_order = Order(
            orderId=self.ib.client.getReqId(),
            orderType="LMT",
            action="SELL",
            totalQuantity=quantity,
            lmtPrice=aggressive_call_price,
            tif="DAY",
        )

        put_order = Order(
            orderId=self.ib.client.getReqId(),
            orderType="LMT",
            action="BUY",
            totalQuantity=quantity,
            lmtPrice=aggressive_put_price,
            tif="DAY",
        )

        # Create leg order objects with aggressive pricing
        stock_leg = LegOrder(
            leg_type=LegType.STOCK,
            contract=stock_contract,
            order=stock_order,
            target_price=aggressive_stock_price,
            action="BUY",
            quantity=quantity * 100,
            order_id=stock_order.orderId,
        )

        call_leg = LegOrder(
            leg_type=LegType.CALL,
            contract=call_contract,
            order=call_order,
            target_price=aggressive_call_price,
            action="SELL",
            quantity=quantity,
            order_id=call_order.orderId,
        )

        put_leg = LegOrder(
            leg_type=LegType.PUT,
            contract=put_contract,
            order=put_order,
            target_price=aggressive_put_price,
            action="BUY",
            quantity=quantity,
            order_id=put_order.orderId,
        )

        # Create execution plan
        plan = ParallelExecutionPlan(
            execution_id=execution_id,
            symbol=symbol,
            expiry=expiry,
            stock_leg=stock_leg,
            call_leg=call_leg,
            put_leg=put_leg,
            max_execution_time=max_execution_time,
            max_fill_time_per_leg=max_fill_time_per_leg,
            slippage_tolerance=slippage_tolerance,
        )

        # Log both original and aggressive prices for comparison
        if stock_ticker or call_ticker or put_ticker:
            logger.info(
                f"[{symbol}] Created parallel execution plan {execution_id} with aggressive pricing: "
                f"stock@{stock_price:.2f}->{aggressive_stock_price:.2f}, "
                f"call@{call_price:.2f}->{aggressive_call_price:.2f}, "
                f"put@{put_price:.2f}->{aggressive_put_price:.2f}"
            )
        else:
            logger.info(
                f"[{symbol}] Created parallel execution plan {execution_id}: "
                f"stock@{aggressive_stock_price:.2f}, call@{aggressive_call_price:.2f}, put@{aggressive_put_price:.2f}"
            )

        return plan

    def _get_all_legs(self, plan: ParallelExecutionPlan) -> List[LegOrder]:
        """Get all leg orders from an execution plan."""
        legs = []
        if plan.stock_leg:
            legs.append(plan.stock_leg)
        if plan.call_leg:
            legs.append(plan.call_leg)
        if plan.put_leg:
            legs.append(plan.put_leg)
        return legs

    async def place_orders_parallel(self, plan: ParallelExecutionPlan) -> bool:
        """
        Place all orders for the execution plan in parallel.

        Args:
            plan: Execution plan with prepared orders

        Returns:
            True if all orders placed successfully, False otherwise
        """
        plan.execution_state = ExecutionState.PLACING_ORDERS
        plan.start_time = time.time()

        legs = self._get_all_legs(plan)

        logger.info(
            f"[{plan.symbol}] Placing {len(legs)} orders in parallel "
            f"(execution: {plan.execution_id})"
        )

        try:
            # Place all orders simultaneously using asyncio.gather
            placement_tasks = []
            for leg in legs:
                task = asyncio.create_task(self._place_single_order(leg))
                placement_tasks.append(task)

            # Wait for all orders to be placed (with timeout)
            start_time = time.time()
            results = await asyncio.wait_for(
                asyncio.gather(*placement_tasks, return_exceptions=True),
                timeout=1.0,  # 1 second timeout for order placement
            )

            placement_time = time.time() - start_time

            # Check results
            successful_placements = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"[{plan.symbol}] Failed to place {legs[i].leg_type.value} order: {result}"
                    )
                    legs[i].fill_status = "failed"
                elif result:
                    successful_placements += 1
                    legs[i].fill_status = "pending"
                else:
                    logger.warning(
                        f"[{plan.symbol}] Order placement returned False for {legs[i].leg_type.value}"
                    )
                    legs[i].fill_status = "failed"

            logger.info(
                f"[{plan.symbol}] Order placement complete: "
                f"{successful_placements}/{len(legs)} successful "
                f"in {placement_time:.3f}s"
            )

            # Update execution plan
            if successful_placements == len(legs):
                plan.execution_state = ExecutionState.MONITORING_FILLS
                return True
            else:
                plan.execution_state = ExecutionState.COMPLETED_FAILURE
                plan.execution_error = f"Only {successful_placements}/{len(legs)} orders placed successfully"
                return False

        except asyncio.TimeoutError:
            logger.error(f"[{plan.symbol}] Order placement timed out after 1.0s")
            plan.execution_state = ExecutionState.COMPLETED_FAILURE
            plan.execution_error = "Order placement timeout"
            return False
        except Exception as e:
            logger.error(f"[{plan.symbol}] Error in parallel order placement: {e}")
            plan.execution_state = ExecutionState.COMPLETED_FAILURE
            plan.execution_error = f"Order placement error: {str(e)}"
            return False

    async def _place_single_order(self, leg: LegOrder) -> bool:
        """
        Place a single leg order.

        Args:
            leg: Leg order to place

        Returns:
            True if order placed successfully, False otherwise
        """
        try:
            trade = self.ib.placeOrder(leg.contract, leg.order)
            leg.trade = trade
            logger.debug(
                f"Placed {leg.leg_type.value} order {leg.order_id}: "
                f"{leg.action} {leg.quantity} @ {leg.target_price:.2f}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to place {leg.leg_type.value} order: {e}")
            return False

    def get_active_executions(self) -> Dict[str, ParallelExecutionPlan]:
        """Get all currently active executions."""
        return self._active_executions.copy()

    def get_execution_stats(self) -> Dict:
        """Get execution statistics."""
        success_rate = (
            self._successful_executions / self._total_executions
            if self._total_executions > 0
            else 0.0
        )

        return {
            "total_executions": self._total_executions,
            "successful_executions": self._successful_executions,
            "failed_executions": self._failed_executions,
            "success_rate_percent": success_rate * 100,
            "average_execution_time_seconds": self._average_execution_time,
            "active_executions": len(self._active_executions),
        }

    def cancel_execution(
        self, execution_id: str, reason: str = "manual_cancel"
    ) -> bool:
        """
        Cancel an active execution.

        Args:
            execution_id: ID of execution to cancel
            reason: Reason for cancellation

        Returns:
            True if cancelled successfully, False if not found or already completed
        """
        if execution_id not in self._active_executions:
            logger.warning(f"Cannot cancel execution {execution_id}: not found")
            return False

        plan = self._active_executions[execution_id]

        if plan.execution_state in [
            ExecutionState.COMPLETED_SUCCESS,
            ExecutionState.COMPLETED_FAILURE,
        ]:
            logger.warning(f"Cannot cancel execution {execution_id}: already completed")
            return False

        logger.info(f"[{plan.symbol}] Cancelling execution {execution_id}: {reason}")

        # Cancel all orders
        legs = self._get_all_legs(plan)
        for leg in legs:
            if leg.trade and leg.fill_status in ["pending", "partial"]:
                try:
                    self.ib.cancelOrder(leg.order)
                    leg.fill_status = "cancelled"
                    logger.debug(f"Cancelled {leg.leg_type.value} order {leg.order_id}")
                except Exception as e:
                    logger.error(f"Error cancelling {leg.leg_type.value} order: {e}")

        # Update plan state
        plan.execution_state = ExecutionState.CANCELLED
        plan.execution_error = f"Cancelled: {reason}"
        plan.end_time = time.time()

        # Clean up
        del self._active_executions[execution_id]

        return True


# Convenience functions for framework access
_framework_instance: Optional[ParallelExecutionFramework] = None


async def get_parallel_framework(ib: IB) -> ParallelExecutionFramework:
    """Get or create parallel execution framework instance."""
    global _framework_instance
    if _framework_instance is None:
        _framework_instance = ParallelExecutionFramework(ib)
        await _framework_instance.initialize()
    return _framework_instance


async def create_sfr_execution_plan(
    ib: IB,
    symbol: str,
    expiry: str,
    stock_contract: Contract,
    call_contract: Contract,
    put_contract: Contract,
    stock_price: float,
    call_price: float,
    put_price: float,
    quantity: int = 1,
    execution_params: Optional[Dict] = None,
) -> ParallelExecutionPlan:
    """Convenience function to create SFR execution plan."""
    framework = await get_parallel_framework(ib)
    return await framework.create_execution_plan(
        symbol,
        expiry,
        stock_contract,
        call_contract,
        put_contract,
        stock_price,
        call_price,
        put_price,
        quantity,
        execution_params,
    )
