"""
Integration layer for SFR parallel execution system.

This module provides seamless integration between the existing SFR executor
and the new parallel execution framework. It handles:

1. Configuration-based switching between combo and parallel execution
2. Backward compatibility with existing SFR logic
3. Enhanced opportunity evaluation for parallel execution
4. Integration with existing logging and metrics systems
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from ib_async import IB, Contract, Order

from ..common import get_logger
from ..metrics import metrics_collector
from .constants import (
    EXECUTION_REPORT_AUTO_EXPORT,
    EXECUTION_REPORT_DEFAULT_LEVEL,
    PARALLEL_EXECUTION_DEBUG_MODE,
    PARALLEL_EXECUTION_DRY_RUN,
    PARALLEL_EXECUTION_ENABLED,
    SLIPPAGE_ERROR_THRESHOLD,
    SLIPPAGE_WARNING_THRESHOLD,
)
from .execution_reporter import ExecutionReporter, ReportFormat, ReportLevel
from .models import ExpiryOption
from .opportunity_evaluator import OpportunityEvaluator
from .parallel_executor import ExecutionResult, ParallelLegExecutor

logger = get_logger()


class ParallelExecutionIntegrator:
    """
    Integration layer between SFR executor and parallel execution system.

    This class provides:
    1. Seamless switching between combo and parallel execution
    2. Enhanced opportunity evaluation for parallel execution
    3. Beautiful reporting and monitoring
    4. Backward compatibility with existing SFR logic
    """

    def __init__(
        self,
        ib: IB,
        order_manager: Any,
        symbol: str,
        opportunity_evaluator: OpportunityEvaluator,
    ):
        self.ib = ib
        self.order_manager = order_manager
        self.symbol = symbol
        self.opportunity_evaluator = opportunity_evaluator

        # Parallel execution components
        self.parallel_executor: Optional[ParallelLegExecutor] = None
        self.execution_reporter = ExecutionReporter()

        # Configuration
        self.parallel_enabled = (
            PARALLEL_EXECUTION_ENABLED and not PARALLEL_EXECUTION_DRY_RUN
        )
        self.debug_mode = PARALLEL_EXECUTION_DEBUG_MODE
        self.auto_export_reports = EXECUTION_REPORT_AUTO_EXPORT

        # State tracking
        self.is_initialized = False
        self.execution_count = 0
        self.last_execution_result: Optional[ExecutionResult] = None

        logger.debug(
            f"[{symbol}] ParallelExecutionIntegrator created (enabled: {self.parallel_enabled})"
        )

    async def initialize(self) -> bool:
        """Initialize the parallel execution integrator."""
        try:
            if self.parallel_enabled:
                # Initialize parallel executor
                self.parallel_executor = ParallelLegExecutor(
                    ib=self.ib,
                    symbol=self.symbol,
                    on_execution_complete=self._on_execution_complete,
                    on_execution_failed=self._on_execution_failed,
                )

                success = await self.parallel_executor.initialize()
                if not success:
                    logger.error(
                        f"[{self.symbol}] Failed to initialize parallel executor"
                    )
                    return False

                logger.info(
                    f"[{self.symbol}] Parallel execution integrator initialized"
                )
            else:
                logger.info(
                    f"[{self.symbol}] Parallel execution disabled, using combo orders"
                )

            self.is_initialized = True
            return True

        except Exception as e:
            logger.error(
                f"[{self.symbol}] Error initializing parallel execution integrator: {e}"
            )
            return False

    async def should_use_parallel_execution(
        self, opportunity: Dict, market_conditions: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Determine whether to use parallel execution for this opportunity.

        Args:
            opportunity: Opportunity details from evaluator
            market_conditions: Optional market conditions assessment

        Returns:
            Tuple of (should_use_parallel, reason)
        """
        if not self.parallel_enabled:
            return False, "parallel_execution_disabled"

        if not self.is_initialized:
            return False, "integrator_not_initialized"

        if PARALLEL_EXECUTION_DRY_RUN:
            return False, "dry_run_mode_enabled"

        # Check opportunity characteristics
        profit = opportunity.get("guaranteed_profit", 0.0)
        if profit < 0.20:  # Less than 20 cents profit
            return False, f"profit_too_low_for_parallel_{profit:.2f}"

        # Check if we have all required data
        if not self._has_sufficient_data_for_parallel(opportunity):
            return False, "insufficient_data_for_parallel"

        # Check market conditions (simplified)
        if market_conditions:
            volatility = market_conditions.get("volatility", "normal")
            if volatility == "high":
                return False, "high_volatility_detected"

        # Check recent performance
        recent_success_rate = self._get_recent_parallel_success_rate()
        if recent_success_rate < 0.7:  # Less than 70% success rate recently
            logger.warning(
                f"[{self.symbol}] Recent parallel success rate low: {recent_success_rate:.1%}"
            )
            # Still proceed but log warning

        return True, f"parallel_execution_favorable_profit_{profit:.2f}"

    async def execute_opportunity(
        self, opportunity: Dict, force_parallel: bool = False, force_combo: bool = False
    ) -> Dict:
        """
        Execute opportunity using optimal method (parallel vs combo).

        Args:
            opportunity: Opportunity details from evaluator
            force_parallel: Force parallel execution even if not optimal
            force_combo: Force combo execution even if parallel is available

        Returns:
            Dictionary with execution results
        """
        self.execution_count += 1
        execution_start_time = time.time()

        logger.info(
            f"[{self.symbol}] Starting opportunity execution #{self.execution_count}"
        )

        try:
            # Determine execution method
            if force_combo:
                use_parallel = False
                reason = "force_combo_requested"
            elif force_parallel:
                use_parallel = True
                reason = "force_parallel_requested"
            else:
                use_parallel, reason = await self.should_use_parallel_execution(
                    opportunity
                )

            logger.info(
                f"[{self.symbol}] Execution method: {'PARALLEL' if use_parallel else 'COMBO'} ({reason})"
            )

            # Execute using chosen method
            if use_parallel:
                result = await self._execute_parallel_opportunity(opportunity)
            else:
                result = await self._execute_combo_opportunity(opportunity)

            # Post-execution processing
            await self._process_execution_result(
                result, use_parallel, execution_start_time
            )

            return result

        except Exception as e:
            logger.error(
                f"[{self.symbol}] Critical error in opportunity execution: {e}"
            )

            # Return error result
            total_time = time.time() - execution_start_time
            return {
                "success": False,
                "method": "error",
                "error": str(e),
                "execution_time": total_time,
                "symbol": self.symbol,
            }

    async def _execute_parallel_opportunity(self, opportunity: Dict) -> Dict:
        """Execute opportunity using parallel leg execution."""

        logger.info(f"[{self.symbol}] Executing with PARALLEL leg orders")

        try:
            # Extract contracts and pricing from opportunity
            trade_details = opportunity["trade_details"]
            expiry_option = opportunity["expiry_option"]

            # Get contracts
            stock_contract = self._get_stock_contract_from_opportunity(opportunity)
            call_contract = expiry_option.call_contract
            put_contract = expiry_option.put_contract

            # Get target prices
            stock_price = trade_details["stock_price"]
            call_price = trade_details["call_price"]
            put_price = trade_details["put_price"]

            # Execute with parallel executor
            result = await self.parallel_executor.execute_parallel_arbitrage(
                stock_contract=stock_contract,
                call_contract=call_contract,
                put_contract=put_contract,
                stock_price=stock_price,
                call_price=call_price,
                put_price=put_price,
                quantity=1,  # Could be extracted from opportunity
                profit_target=opportunity["guaranteed_profit"],
            )

            self.last_execution_result = result

            # Convert to standard format
            return {
                "success": result.success,
                "method": "parallel",
                "execution_time": result.total_execution_time,
                "legs_filled": f"{result.legs_filled}/{result.total_legs}",
                "slippage_dollars": result.total_slippage,
                "slippage_percent": result.slippage_percentage,
                "expected_cost": result.expected_total_cost,
                "actual_cost": result.actual_total_cost,
                "error_message": result.error_message,
                "execution_id": result.execution_id,
                "symbol": result.symbol,
                "parallel_result": result,  # Full result for detailed analysis
            }

        except Exception as e:
            logger.error(f"[{self.symbol}] Error in parallel execution: {e}")
            return {
                "success": False,
                "method": "parallel",
                "error": str(e),
                "symbol": self.symbol,
            }

    async def _execute_combo_opportunity(self, opportunity: Dict) -> Dict:
        """Execute opportunity using traditional combo orders."""

        logger.info(f"[{self.symbol}] Executing with COMBO orders (traditional)")

        combo_start_time = time.time()

        try:
            # Use existing order manager for combo execution
            contract = opportunity["contract"]
            order = opportunity["order"]

            # Place combo order
            trade = await self.order_manager.place_order(contract, order)

            combo_execution_time = time.time() - combo_start_time

            if trade:
                logger.info(f"[{self.symbol}] Combo order placed successfully")

                return {
                    "success": True,
                    "method": "combo",
                    "execution_time": combo_execution_time,
                    "legs_filled": "3/3",  # Combo order fills all legs together
                    "slippage_dollars": 0.0,  # Cannot measure individual leg slippage with combo
                    "slippage_percent": 0.0,
                    "trade": trade,
                    "symbol": self.symbol,
                }
            else:
                logger.warning(f"[{self.symbol}] Combo order placement failed")

                return {
                    "success": False,
                    "method": "combo",
                    "execution_time": combo_execution_time,
                    "error": "combo_order_placement_failed",
                    "symbol": self.symbol,
                }

        except Exception as e:
            logger.error(f"[{self.symbol}] Error in combo execution: {e}")
            return {
                "success": False,
                "method": "combo",
                "error": str(e),
                "execution_time": time.time() - combo_start_time,
                "symbol": self.symbol,
            }

    async def _process_execution_result(
        self, result: Dict, used_parallel: bool, execution_start_time: float
    ) -> None:
        """Process execution result with reporting and metrics."""

        success = result.get("success", False)
        total_time = time.time() - execution_start_time
        slippage = result.get("slippage_dollars", 0.0)

        # Update metrics
        if success:
            metrics_collector.record_opportunity_found(self.symbol)
            if used_parallel:
                metrics_collector.increment_counter("parallel_executions_successful", 1)
        else:
            if used_parallel:
                metrics_collector.increment_counter("parallel_executions_failed", 1)

        # Track execution method by incrementing appropriate counter
        if used_parallel:
            metrics_collector.increment_counter("parallel_method_used", 1)
        else:
            metrics_collector.increment_counter("combo_method_used", 1)

        # Generate execution report for parallel executions
        if used_parallel and "parallel_result" in result:
            try:
                report_level = (
                    ReportLevel.DETAILED if self.debug_mode else ReportLevel.SUMMARY
                )

                report = self.execution_reporter.generate_execution_report(
                    result["parallel_result"],
                    level=report_level,
                    format_type=ReportFormat.CONSOLE,
                )

                # Print report
                print(report)

                # Export if configured
                if self.auto_export_reports:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"sfr_execution_{self.symbol}_{timestamp}.html"

                    html_report = self.execution_reporter.generate_execution_report(
                        result["parallel_result"],
                        level=ReportLevel.COMPREHENSIVE,
                        format_type=ReportFormat.HTML,
                    )

                    try:
                        with open(filename, "w") as f:
                            f.write(html_report)
                        logger.info(
                            f"[{self.symbol}] Execution report exported to {filename}"
                        )
                    except Exception as e:
                        logger.warning(f"[{self.symbol}] Failed to export report: {e}")

            except Exception as e:
                logger.warning(
                    f"[{self.symbol}] Error generating execution report: {e}"
                )

        # Log performance warnings/errors
        if total_time > 30.0:
            logger.error(f"[{self.symbol}] SLOW EXECUTION: {total_time:.2f}s")
        elif total_time > 10.0:
            logger.warning(f"[{self.symbol}] Slow execution: {total_time:.2f}s")

        if abs(slippage) > SLIPPAGE_ERROR_THRESHOLD:
            logger.error(f"[{self.symbol}] HIGH SLIPPAGE: ${slippage:.2f}")
        elif abs(slippage) > SLIPPAGE_WARNING_THRESHOLD:
            logger.warning(f"[{self.symbol}] Elevated slippage: ${slippage:.2f}")

        # Final execution summary
        method_str = "PARALLEL" if used_parallel else "COMBO"
        status_str = "SUCCESS" if success else "FAILED"

        logger.info(
            f"[{self.symbol}] Execution #{self.execution_count} complete: "
            f"{method_str} {status_str} in {total_time:.2f}s"
        )

    def _has_sufficient_data_for_parallel(self, opportunity: Dict) -> bool:
        """Check if we have sufficient data quality for parallel execution."""
        try:
            # Check if we have all required contracts
            if "expiry_option" not in opportunity:
                return False

            expiry_option = opportunity["expiry_option"]
            if not all([expiry_option.call_contract, expiry_option.put_contract]):
                return False

            # Check trade details
            trade_details = opportunity.get("trade_details", {})
            required_fields = ["stock_price", "call_price", "put_price", "net_credit"]

            for field in required_fields:
                if field not in trade_details or trade_details[field] is None:
                    return False

            return True

        except Exception as e:
            logger.debug(f"[{self.symbol}] Error checking data sufficiency: {e}")
            return False

    def _get_stock_contract_from_opportunity(self, opportunity: Dict) -> Contract:
        """Extract stock contract from opportunity data."""
        # This would extract the stock contract from the combo contract
        # For now, return a simplified version
        try:
            contract = opportunity["contract"]
            if hasattr(contract, "comboLegs") and contract.comboLegs:
                # First leg should be stock
                stock_leg = contract.comboLegs[0]
                # Would need to reconstruct stock contract from leg
                # For now, return a placeholder
                from ib_async import Stock

                return Stock(self.symbol, "SMART", "USD")
            else:
                # Fallback
                from ib_async import Stock

                return Stock(self.symbol, "SMART", "USD")
        except:
            from ib_async import Stock

            return Stock(self.symbol, "SMART", "USD")

    def _get_recent_parallel_success_rate(self, lookback_executions: int = 10) -> float:
        """Get recent parallel execution success rate."""
        if not self.parallel_executor:
            return 1.0  # No data, assume good

        try:
            stats = self.parallel_executor.get_execution_stats()
            success_rate = stats.get("success_rate_percent", 100.0) / 100.0
            return success_rate
        except:
            return 1.0  # Assume good if error

    async def _on_execution_complete(self, result: ExecutionResult) -> None:
        """Callback for successful parallel executions."""
        logger.info(
            f"[{self.symbol}] Parallel execution completed successfully: "
            f"${result.total_slippage:.2f} slippage in {result.total_execution_time:.2f}s"
        )

        # Could add additional success handling here
        # e.g., update strategy parameters, send notifications, etc.

    async def _on_execution_failed(self, result: ExecutionResult) -> None:
        """Callback for failed parallel executions."""
        logger.warning(
            f"[{self.symbol}] Parallel execution failed: {result.error_message} "
            f"after {result.total_execution_time:.2f}s"
        )

        # Could add failure handling logic here
        # e.g., adjust parameters, switch to combo mode temporarily, etc.

    def get_integration_stats(self) -> Dict:
        """Get comprehensive integration statistics."""
        stats = {
            "integration_status": {
                "parallel_enabled": self.parallel_enabled,
                "is_initialized": self.is_initialized,
                "debug_mode": self.debug_mode,
                "auto_export_reports": self.auto_export_reports,
            },
            "execution_summary": {
                "total_executions": self.execution_count,
                "last_execution_successful": (
                    self.last_execution_result.success
                    if self.last_execution_result
                    else None
                ),
            },
        }

        # Add parallel executor stats if available
        if self.parallel_executor:
            stats["parallel_executor"] = self.parallel_executor.get_execution_stats()

        # Add reporter stats
        stats["execution_reporter"] = self.execution_reporter.get_session_statistics()

        return stats

    def enable_parallel_execution(self) -> bool:
        """Enable parallel execution (runtime toggle)."""
        if not self.is_initialized:
            logger.error(
                f"[{self.symbol}] Cannot enable parallel execution - not initialized"
            )
            return False

        self.parallel_enabled = True
        logger.info(f"[{self.symbol}] Parallel execution ENABLED")
        return True

    def disable_parallel_execution(self) -> bool:
        """Disable parallel execution (runtime toggle)."""
        self.parallel_enabled = False
        logger.info(f"[{self.symbol}] Parallel execution DISABLED")
        return True

    async def shutdown(self) -> None:
        """Shutdown the integrator and cleanup resources."""
        try:
            if self.parallel_executor:
                # Cancel any active executions
                await self.parallel_executor.cancel_current_execution("system_shutdown")

            # Export final session report if configured
            if self.auto_export_reports and self.execution_count > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"sfr_session_{self.symbol}_{timestamp}.json"

                success = self.execution_reporter.export_session_report(
                    filename, ReportFormat.JSON
                )

                if success:
                    logger.info(
                        f"[{self.symbol}] Final session report exported to {filename}"
                    )

            logger.info(
                f"[{self.symbol}] Parallel execution integrator shutdown complete"
            )

        except Exception as e:
            logger.error(f"[{self.symbol}] Error during integrator shutdown: {e}")


# Convenience function for easy integration
async def create_parallel_integrator(
    ib: IB, order_manager: Any, symbol: str, opportunity_evaluator: OpportunityEvaluator
) -> ParallelExecutionIntegrator:
    """
    Create and initialize a parallel execution integrator.

    Args:
        ib: IB connection instance
        order_manager: Order manager instance
        symbol: Trading symbol
        opportunity_evaluator: Opportunity evaluator instance

    Returns:
        Initialized ParallelExecutionIntegrator
    """
    integrator = ParallelExecutionIntegrator(
        ib, order_manager, symbol, opportunity_evaluator
    )

    success = await integrator.initialize()
    if not success:
        logger.error(f"[{symbol}] Failed to initialize parallel execution integrator")
        raise RuntimeError(
            f"Parallel execution integrator initialization failed for {symbol}"
        )

    return integrator
