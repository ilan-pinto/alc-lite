"""
Core SFR executor logic.

This module contains the simplified SFRExecutor class that coordinates all the
modular components for SFR arbitrage strategy execution.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple

from eventkit import Event
from ib_async import IB, Contract, Order

from ..common import get_logger
from ..data_collection_metrics import (
    CollectionPhase,
    ContractPriority,
    should_continue_waiting,
)
from ..metrics import metrics_collector
from ..Strategy import BaseExecutor, OrderManagerClass
from .constants import DEFAULT_DATA_TIMEOUT
from .data_collector import DataCollectionCoordinator, DataCollectionManager
from .models import ExpiryOption, OpportunityTuple
from .opportunity_evaluator import OpportunityEvaluator
from .parallel_integration import create_parallel_integrator
from .utils import flush_all_handlers, get_stock_midpoint, log_funnel_summary
from .validation import MarketValidator

logger = get_logger()


class SFRExecutor(BaseExecutor):
    """
    Simplified SFR Executor class that orchestrates modular components.

    This class is responsible for:
    1. Coordinating data collection through DataCollectionManager
    2. Evaluating opportunities through OpportunityEvaluator
    3. Managing progressive timeout strategies
    4. Executing trades when conditions are met
    """

    def __init__(
        self,
        ib: IB,
        order_manager: OrderManagerClass,
        stock_contract: Contract,
        expiry_options: List[ExpiryOption],
        symbol: str,
        profit_target: float,
        cost_limit: float,
        quantity: int = 1,
        start_time: float = None,
        data_timeout: float = DEFAULT_DATA_TIMEOUT,
        strategy=None,
    ) -> None:
        """Initialize the SFR Executor with modular components."""
        # Create list of all option contracts
        option_contracts = []
        for expiry_option in expiry_options:
            option_contracts.extend(
                [expiry_option.call_contract, expiry_option.put_contract]
            )

        super().__init__(
            ib,
            order_manager,
            stock_contract,
            option_contracts,
            symbol,
            cost_limit,
            expiry_options[0].expiry if expiry_options else "",
            start_time,
        )

        self.profit_target = profit_target
        self.cost_limit = cost_limit
        self.quantity = quantity
        self.expiry_options = expiry_options
        self.all_contracts = [stock_contract] + option_contracts
        self.is_active = True
        self.data_timeout = data_timeout
        self.start_time = start_time or time.time()
        self.strategy = strategy

        # Initialize modular components
        self.data_manager = DataCollectionManager(symbol)
        self.data_coordinator = self.data_manager.initialize_collection(
            all_contracts=self.all_contracts,
            expiry_options=expiry_options,
            data_timeout=data_timeout,
        )

        # Initialize opportunity evaluator with ticker getter function
        self.opportunity_evaluator = OpportunityEvaluator(
            symbol=symbol,
            expiry_options=expiry_options,
            ticker_getter_func=self.data_coordinator.ticker_manager.get_ticker,
            stock_contract=stock_contract,
        )

        # Initialize last stock price for priority calculation
        self.last_stock_price = None

        # Initialize parallel execution integrator
        self.parallel_integrator = None
        self._parallel_integration_initialized = False

    def set_contract_ticker_reference(self, contract_ticker: dict):
        """Set reference to global contract_ticker dictionary"""
        self.data_manager.set_contract_ticker_reference(contract_ticker)

    async def _initialize_parallel_integration(self) -> None:
        """Initialize parallel execution integration if not already done."""
        if self._parallel_integration_initialized:
            return

        try:

            self.parallel_integrator = await create_parallel_integrator(
                ib=self.ib,
                order_manager=self.order_manager,
                symbol=self.symbol,
                opportunity_evaluator=self.opportunity_evaluator,
                strategy=self.strategy,
            )

            self._parallel_integration_initialized = True
            logger.info(f"[{self.symbol}] Parallel execution integration initialized")

        except Exception as e:
            logger.warning(
                f"[{self.symbol}] Failed to initialize parallel integration: {e}"
            )
            logger.info(f"[{self.symbol}] Falling back to traditional combo orders")
            self._parallel_integration_initialized = False

    async def executor(self, event: Event) -> None:
        """
        Progressive data collection executor with 3-phase timeout strategy.
        This method makes decisions as soon as sufficient data is available.
        """
        if not self.is_active:
            return

        try:
            # Update contract_ticker with new data and track metrics
            logger.debug(
                f"[{self.symbol}] Processing ticker event with {len(event)} contracts"
            )

            # Process ticker events through data coordinator
            valid_processed, skipped_contracts = (
                self.data_coordinator.process_ticker_event(event)
            )

            # Update velocity tracker
            self.data_coordinator.update_velocity_tracking()

            # Progressive phase checking
            elapsed_time = self.data_coordinator.get_elapsed_time()

            # Initialize contract priorities on first stock data
            if (
                not self.data_coordinator.priority_tiers
                and self.data_coordinator.has_stock_data()
            ):
                stock_ticker = self.data_coordinator.get_stock_ticker()
                if stock_ticker:
                    stock_price = get_stock_midpoint(stock_ticker)
                    if stock_price is not None and stock_price > 0:
                        self.last_stock_price = stock_price
                        self.data_coordinator.initialize_contract_priorities(
                            stock_price
                        )

            # Phase 1: Check critical contracts
            if self.data_coordinator.should_check_phase1():
                self.data_coordinator.transition_to_phase1()

                if self.data_coordinator.has_sufficient_critical_data():
                    opportunity = await self.evaluate_with_available_data(
                        ContractPriority.CRITICAL
                    )
                    if (
                        opportunity
                        and opportunity["guaranteed_profit"]
                        >= self.data_coordinator.timeout_config.phase_1_profit_threshold
                    ):
                        logger.info(
                            f"[{self.symbol}] Phase 1 execution: profit={opportunity['guaranteed_profit']:.2f}"
                        )
                        await self.execute_opportunity(opportunity)
                        return

            # Phase 2: Check important contracts
            if self.data_coordinator.should_check_phase2():
                self.data_coordinator.transition_to_phase2()

                if self.data_coordinator.has_sufficient_important_data():
                    opportunity = await self.evaluate_with_available_data(
                        ContractPriority.IMPORTANT
                    )
                    if (
                        opportunity
                        and opportunity["guaranteed_profit"]
                        >= self.data_coordinator.timeout_config.phase_2_profit_threshold
                    ):
                        logger.info(
                            f"[{self.symbol}] Phase 2 execution: profit={opportunity['guaranteed_profit']:.2f}"
                        )
                        await self.execute_opportunity(opportunity)
                        return

            # Phase 3: Final check with all available data
            if self.data_coordinator.should_check_phase3():
                self.data_coordinator.transition_to_phase3()

                if self.data_coordinator.has_minimum_viable_data():
                    # Use vectorized evaluation for better performance and spread analysis
                    opportunity = await self.evaluate_with_available_data_vectorized(
                        ContractPriority.OPTIONAL
                    )
                    if (
                        opportunity
                        and opportunity["guaranteed_profit"]
                        >= self.data_coordinator.timeout_config.phase_3_profit_threshold
                    ):
                        logger.info(
                            f"[{self.symbol}] Phase 3 execution: profit={opportunity['guaranteed_profit']:.2f}"
                        )
                        await self.execute_opportunity(opportunity)
                        return

                # No profitable opportunity found
                logger.info(
                    f"[{self.symbol}] No profitable opportunity after {elapsed_time:.1f}s"
                )
                self.finish_collection_without_execution("no_profitable_opportunity")
                return

            # Check if we should stop waiting based on data collection conditions
            if elapsed_time > 1.0:
                should_continue, stop_reason = should_continue_waiting(
                    self.data_coordinator.collection_metrics,
                    self.data_coordinator.timeout_config,
                    self.data_coordinator.velocity_tracker,
                )
                if not should_continue:
                    await self._handle_early_completion(stop_reason, elapsed_time)
                    return

        except Exception as e:
            logger.error(f"Error in progressive executor: {str(e)}")
            self.finish_collection_without_execution(f"error: {str(e)}")

    async def _handle_early_completion(self, stop_reason: str, elapsed_time: float):
        """Handle early completion of data collection"""
        logger.info(
            f"[{self.symbol}] Data collection complete early ({stop_reason}), evaluating opportunities..."
        )

        # Always evaluate if we have data overflow OR minimum viable data
        should_evaluate = (
            self.data_coordinator.has_minimum_viable_data()
            or stop_reason == "data_overflow"
        )

        if should_evaluate:
            # Use vectorized evaluation for faster processing and better spread analysis
            opportunity = await self.evaluate_with_available_data_vectorized(
                ContractPriority.OPTIONAL  # Use all available data since collection is complete
            )
            if (
                opportunity
                and opportunity["guaranteed_profit"]
                >= self.data_coordinator.timeout_config.phase_3_profit_threshold
            ):
                logger.info(
                    f"[{self.symbol}] Early completion execution: profit={opportunity['guaranteed_profit']:.2f}"
                )
                await self.execute_opportunity(opportunity)
                return

            # Log evaluation result even if no opportunity found
            if not opportunity:
                logger.info(
                    f"[{self.symbol}] Evaluation complete - no profitable opportunities found "
                    f"(completion: {self.data_coordinator.collection_metrics.get_completion_percentage():.1f}%)"
                )
        else:
            logger.warning(
                f"[{self.symbol}] Skipping evaluation - insufficient viable data "
                f"({self.data_coordinator.collection_metrics.get_completion_percentage():.1f}% collected)"
            )

        # Map stop reasons to user-friendly messages
        reason_messages = {
            "data_overflow": "sufficient data collected (overflow detected)",
            "data_burst_sufficient": "sufficient data collected (burst pattern)",
            "critical_threshold_met": "critical data threshold achieved",
            "hard_timeout": "collection timeout reached",
            "poor_velocity_sufficient_data": "poor data velocity with sufficient data",
            "poor_velocity_decent_data": "poor data velocity with decent data",
            "poor_velocity_limited_data": "poor data velocity with limited data",
            "estimated_completion_too_long": "estimated completion time too long",
        }

        user_message = reason_messages.get(
            stop_reason, f"collection stopped ({stop_reason})"
        )
        logger.info(
            f"[{self.symbol}] No opportunities found - stopping early due to: {user_message}"
        )
        self.finish_collection_without_execution(stop_reason)

    async def evaluate_with_available_data(
        self, max_priority: ContractPriority
    ) -> Optional[Dict]:
        """Evaluate opportunities with currently available data up to specified priority"""
        logger.info(
            f"[{self.symbol}] Starting evaluation with {len(self.expiry_options)} expiry options, "
            f"max_priority={max_priority.value}"
        )

        best_opportunity = None
        best_profit = 0

        # Determine which expiry options to consider based on priority
        eligible_options = []
        if self.data_coordinator.priority_tiers:
            for priority in [
                ContractPriority.CRITICAL,
                ContractPriority.IMPORTANT,
                ContractPriority.OPTIONAL,
            ]:
                if priority in self.data_coordinator.priority_tiers:
                    eligible_options.extend(
                        self.data_coordinator.priority_tiers[priority]
                    )
                if priority == max_priority:
                    break
        else:
            # Fallback to all options if priorities not initialized
            eligible_options = self.expiry_options

        logger.info(
            f"[{self.symbol}] Eligible options for evaluation: {len(eligible_options)}"
        )

        for expiry_option in eligible_options:
            # Skip if we don't have data for this option pair
            if not self.data_coordinator.has_data_for_option_pair(expiry_option):
                logger.debug(
                    f"[{self.symbol}] Skipping {expiry_option.expiry} - missing data for option pair"
                )
                continue

            try:
                # Use opportunity evaluator for calculation
                opportunity_result = (
                    self.opportunity_evaluator.calc_price_and_build_order_for_expiry(
                        expiry_option=expiry_option,
                        stock_contract=self.stock_contract,
                        profit_target=self.profit_target,
                        cost_limit=self.cost_limit,
                        quantity=self.quantity,
                        build_order_func=self.build_order,
                        priority_filter=max_priority,
                    )
                )

                if (
                    opportunity_result and opportunity_result[2] > best_profit
                ):  # opportunity_result[2] is min_profit
                    best_opportunity = {
                        "contract": opportunity_result[0],
                        "order": opportunity_result[1],
                        "guaranteed_profit": opportunity_result[2],
                        "trade_details": opportunity_result[3],
                        "expiry_option": expiry_option,
                    }
                    best_profit = opportunity_result[2]

            except Exception as e:
                logger.debug(f"Error evaluating {expiry_option.expiry}: {str(e)}")
                continue

        # Log summary if no opportunity found
        if not best_opportunity:
            logger.info(
                f"[{self.symbol}] Evaluation complete: examined {len(eligible_options)} options, "
                f"no profitable opportunities found"
            )

        return best_opportunity

    async def evaluate_with_available_data_vectorized(
        self, max_priority: ContractPriority
    ) -> Optional[Dict]:
        """Use vectorized evaluation for faster processing"""
        result = (
            await self.opportunity_evaluator.evaluate_with_available_data_vectorized(
                max_priority
            )
        )

        if not result:
            return None

        # Convert vectorized result to standard format
        best_idx = result["best_idx"]
        # Use the ExpiryOption object directly from vectorized evaluation to prevent index mismatch
        best_expiry = result[
            "best_expiry_option"
        ]  # Use the actual ExpiryOption from result

        # Add contract verification logging
        logger.info(f"[{self.symbol}] EXECUTOR CONTRACT VERIFICATION:")
        logger.info(f"  Using ExpiryOption from vectorized result (not index lookup)")
        logger.info(
            f"  Expiry: {best_expiry.expiry}, Call Strike: {best_expiry.call_strike}, Put Strike: {best_expiry.put_strike}"
        )

        # Use calc_price_and_build_order_for_expiry for proper limit price calculation
        # Pass cached pricing data to avoid re-fetching and re-evaluating
        pricing_data = result.get("pricing_data")
        if pricing_data:
            logger.info(
                f"[{self.symbol}] Passing cached pricing data to order building - profit: ${pricing_data.get('guaranteed_profit', 0):.2f}"
            )
        else:
            logger.warning(
                f"[{self.symbol}] No cached pricing data available - will re-fetch market data"
            )

        opportunity_result = (
            self.opportunity_evaluator.calc_price_and_build_order_for_expiry(
                expiry_option=best_expiry,
                stock_contract=self.stock_contract,
                profit_target=self.profit_target,
                cost_limit=self.cost_limit,
                quantity=self.quantity,
                build_order_func=self.build_order,
                priority_filter=max_priority,
                pricing_data=pricing_data,
            )
        )

        if not opportunity_result:
            return None

        return {
            "contract": opportunity_result[0],
            "order": opportunity_result[1],
            "guaranteed_profit": opportunity_result[2],
            "trade_details": opportunity_result[3],
            "expiry_option": best_expiry,
            "statistics": result["statistics"],
        }

    async def execute_opportunity(self, opportunity: Dict):
        """Execute a trading opportunity"""
        try:
            self.data_coordinator.collection_metrics.decision_confidence = (
                self.data_coordinator.collection_metrics.get_completion_percentage()
            )
            self.data_coordinator.collection_metrics.opportunity_found = True
            self.data_coordinator.collection_metrics.time_to_decision = (
                time.time() - self.data_coordinator.data_collection_start
            )
            self.data_coordinator.collection_metrics.final_phase = (
                self.data_coordinator.current_phase
            )

            # Log trade details
            trade_details = opportunity["trade_details"]
            logger.info(
                f"[{self.symbol}] Executing opportunity for expiry: {trade_details['expiry']}"
            )

            self._log_trade_details(
                trade_details["call_strike"],
                trade_details["call_price"],
                trade_details["put_strike"],
                trade_details["put_price"],
                trade_details["stock_price"],
                trade_details["net_credit"],
                trade_details["min_profit"],
                trade_details["max_profit"],
                trade_details["min_roi"],
            )

            # CRITICAL SAFETY CHECK: Verify arbitrage condition before execution
            net_credit = trade_details["net_credit"]
            spread = trade_details["stock_price"] - trade_details["put_strike"]
            final_profit = net_credit - spread

            logger.info(f"[{self.symbol}] FINAL SAFETY CHECK BEFORE EXECUTION:")
            logger.info(f"  Net Credit: ${net_credit:.2f}, Spread: ${spread:.2f}")
            logger.info(
                f"  Final Profit: ${final_profit:.2f} (should match min_profit: ${trade_details['min_profit']:.2f})"
            )

            # Safety check: ensure we still have positive profit
            if final_profit <= 0.0:
                logger.error(
                    f"[{self.symbol}] CRITICAL ERROR: Final profit check failed! "
                    f"Net credit ${net_credit:.2f} <= spread ${spread:.2f}. "
                    f"Calculated profit: ${final_profit:.2f}. BLOCKING EXECUTION!"
                )
                self.finish_collection_without_execution(
                    "negative_profit_detected_at_execution"
                )
                return

            # Verify profit matches expected
            profit_diff = abs(final_profit - trade_details["min_profit"])
            if profit_diff > 0.01:  # Allow 1 cent difference for rounding
                logger.warning(
                    f"[{self.symbol}] PROFIT MISMATCH WARNING: "
                    f"Final profit ${final_profit:.2f} differs from expected ${trade_details['min_profit']:.2f} "
                    f"by ${profit_diff:.2f}"
                )

            # CRITICAL CONTRACT VERIFICATION: Extract and verify contract details
            if hasattr(opportunity["contract"], "comboLegs"):
                combo_legs = opportunity["contract"].comboLegs
                logger.info(f"[{self.symbol}] Order ComboLeg Contract IDs:")

                # Extract the call and put leg contract IDs (skip stock leg at index 0)
                actual_call_conid = combo_legs[1].conId if len(combo_legs) > 1 else None
                actual_put_conid = combo_legs[2].conId if len(combo_legs) > 2 else None

                for i, leg in enumerate(combo_legs):
                    action = (
                        "BUY STOCK"
                        if i == 0
                        else ("SELL CALL" if i == 1 else "BUY PUT")
                    )
                    logger.info(f"  Leg {i}: {action}, ConId={leg.conId}")

                # Get expected contract IDs from the expiry_option
                if "expiry_option" in opportunity:
                    expected_call_conid = opportunity[
                        "expiry_option"
                    ].call_contract.conId
                    expected_put_conid = opportunity["expiry_option"].put_contract.conId

                    logger.info(
                        f"[{self.symbol}] Expected Contract IDs - Call: {expected_call_conid}, Put: {expected_put_conid}"
                    )
                    logger.info(
                        f"[{self.symbol}] Actual Contract IDs - Call: {actual_call_conid}, Put: {actual_put_conid}"
                    )

                    # CRITICAL: Block execution if contract IDs don't match
                    if (
                        actual_call_conid != expected_call_conid
                        or actual_put_conid != expected_put_conid
                    ):
                        logger.error(
                            f"[{self.symbol}] CRITICAL CONTRACT MISMATCH DETECTED! BLOCKING EXECUTION!"
                        )
                        logger.error(
                            f"  Expected Call ID: {expected_call_conid}, Got: {actual_call_conid}"
                        )
                        logger.error(
                            f"  Expected Put ID: {expected_put_conid}, Got: {actual_put_conid}"
                        )
                        logger.error(
                            f"  Expected strikes - Call: {trade_details['call_strike']}, Put: {trade_details['put_strike']}"
                        )
                        self.finish_collection_without_execution(
                            "contract_id_mismatch_at_execution"
                        )
                        return  # BLOCK EXECUTION

                # Log expected strikes for reference
                expected_call_strike = trade_details["call_strike"]
                expected_put_strike = trade_details["put_strike"]
                logger.info(
                    f"[{self.symbol}] Expected strikes - Call: {expected_call_strike}, Put: {expected_put_strike}"
                )

                if "expiry_option" in opportunity:
                    if (
                        actual_call_conid == expected_call_conid
                        and actual_put_conid == expected_put_conid
                    ):
                        logger.info(
                            f"[{self.symbol}] ✓ Contract IDs verified! About to place order with correct contracts"
                        )
                    else:
                        logger.error(
                            f"[{self.symbol}] Should not reach here - contract verification failed but not blocked"
                        )
                else:
                    logger.warning(
                        f"[{self.symbol}] ⚠ Cannot verify contract IDs - expiry_option not in opportunity dict"
                    )

            # Initialize parallel integration if not already done
            await self._initialize_parallel_integration()

            # Place the order using optimal execution method
            self.is_active = False  # Prevent multiple executions

            if self.parallel_integrator and self._parallel_integration_initialized:
                # Use parallel execution integrator
                logger.info(f"[{self.symbol}] Executing with parallel execution system")
                execution_result = await self.parallel_integrator.execute_opportunity(
                    opportunity
                )

                if execution_result["success"]:
                    self.data_coordinator.collection_metrics.execution_triggered = True
                    logger.info(
                        f"[{self.symbol}] {execution_result['method'].upper()} execution successful: "
                        f"{execution_result.get('legs_filled', 'N/A')} legs, "
                        f"slippage: ${execution_result.get('slippage_dollars', 0.0):.2f}"
                    )
                    # Set order_filled flag to exit scan loop
                    if self.strategy:
                        self.strategy.order_filled = True
                        logger.info(
                            f"[{self.symbol}] Setting order_filled flag to exit scan loop"
                        )
                    # Note: Opportunity recording is handled by parallel_integration.py to avoid double-counting
                    metrics_collector.finish_scan(success=True)
                else:
                    logger.warning(
                        f"[{self.symbol}] {execution_result['method'].upper()} execution failed: "
                        f"{execution_result.get('error_message', 'Unknown error')}"
                    )
                    metrics_collector.finish_scan(
                        success=False,
                        error_message=execution_result.get(
                            "error_message", "Execution failed"
                        ),
                    )
            else:
                # Fallback to traditional combo order execution
                logger.info(f"[{self.symbol}] Executing with traditional combo orders")
                result = await self.order_manager.place_order(
                    opportunity["contract"], opportunity["order"]
                )

                if result:
                    self.data_coordinator.collection_metrics.execution_triggered = True
                    logger.info(f"[{self.symbol}] Combo order placed successfully")
                    # Note: Don't record opportunity here - wait for execution confirmation
                    # The opportunity will be recorded in the parallel execution path after fills are confirmed
                    metrics_collector.finish_scan(success=True)
                else:
                    logger.warning(f"[{self.symbol}] Combo order placement failed")
                    metrics_collector.finish_scan(
                        success=False, error_message="Order placement failed"
                    )

            self.deactivate()

        except Exception as e:
            logger.error(f"Error executing opportunity: {str(e)}")
            self.finish_collection_without_execution(f"execution_error: {str(e)}")

    def finish_collection_without_execution(self, reason: str):
        """Finish collection without executing any trades"""
        self.data_coordinator.collection_metrics.time_to_decision = (
            time.time() - self.data_coordinator.data_collection_start
        )
        self.data_coordinator.collection_metrics.final_phase = (
            self.data_coordinator.current_phase
        )

        logger.info(f"[{self.symbol}] Collection finished without execution: {reason}")
        logger.info(
            f"[{self.symbol}] Final data: {self.data_coordinator.collection_metrics.get_completion_percentage():.1f}% "
            f"({self.data_coordinator.collection_metrics.get_total_received()}/{self.data_coordinator.collection_metrics.get_total_expected()})"
        )

        self.deactivate()
        metrics_collector.finish_scan(
            success=True
        )  # No opportunity is still a successful scan

    def deactivate(self):
        """Deactivate the executor and properly clean up market data subscriptions"""
        if self.is_active:
            logger.info(
                f"[{self.symbol}] Deactivating executor and cleaning up market data subscriptions"
            )

            # Cancel market data for all contracts
            for contract in self.all_contracts:
                try:
                    self.ib.cancelMktData(contract)
                except Exception as e:
                    logger.debug(
                        f"Error cancelling market data for contract {contract.conId}: {str(e)}"
                    )

            # Clean up all symbol-specific tickers from global dictionary
            self.data_manager.cleanup()

            # Log funnel summary before deactivating
            self.log_funnel_summary()

            # Ensure all log messages are written to file
            flush_all_handlers()

            # Call parent deactivate method
            super().deactivate()
            logger.debug(
                f"[{self.symbol}] Executor deactivated and cleaned up {len(self.all_contracts)} contracts"
            )

    def log_funnel_summary(self):
        """Log concise funnel analysis summary"""
        funnel_analysis = metrics_collector.get_funnel_analysis()
        log_funnel_summary(self.symbol, funnel_analysis)

    # === Delegation Methods for Backward Compatibility ===
    # These methods delegate to specialized validators and evaluators
    # They are kept for backward compatibility with tests and legacy code

    def check_conditions(
        self,
        symbol: str,
        profit_target: float,
        cost_limit: float,
        put_strike: float,
        lmt_price: float,
        net_credit: float,
        min_roi: float,
        stock_price: float,
        min_profit: float,
    ):
        """Delegate to conditions validator for backward compatibility with tests and legacy code"""
        from .validation import ConditionsValidator

        validator = ConditionsValidator()
        return validator.check_conditions(
            symbol,
            profit_target,
            cost_limit,
            put_strike,
            lmt_price,
            net_credit,
            min_roi,
            stock_price,
            min_profit,
        )

    def calc_price_and_build_order_for_expiry(self, expiry_option):
        """Delegate to opportunity evaluator for backward compatibility with tests and legacy code"""
        # Force recreation for proper ticker function setup (for test compatibility)
        if True:  # Always recreate to ensure proper ticker setup
            from .opportunity_evaluator import OpportunityEvaluator

            # Create a ticker getter that works with the test setup (compatible with original SFR)
            def get_ticker_func(conId):
                # Import the global contract_ticker from the sfr module for test compatibility
                from . import contract_ticker

                return contract_ticker.get((self.symbol, conId))

            self.opportunity_evaluator = OpportunityEvaluator(
                symbol=self.symbol,
                expiry_options=self.expiry_options,
                ticker_getter_func=get_ticker_func,
                stock_contract=self.stock_contract,
                check_conditions_func=self.check_conditions,
            )

        # Use the stock_contract passed in constructor (preserved from original)
        stock_contract = self.stock_contract

        # Use the inherited build_order method for compatibility
        def build_order_func(*args, **kwargs):
            return self.build_order(*args, **kwargs)

        return self.opportunity_evaluator.calc_price_and_build_order_for_expiry(
            expiry_option=expiry_option,
            stock_contract=stock_contract,
            profit_target=self.profit_target,
            cost_limit=self.cost_limit,
            quantity=self.quantity,
            build_order_func=build_order_func,
        )

    def find_stock_position_in_strikes(self, stock_price: float, valid_strikes):
        """Delegate to StrikeValidator for backward compatibility with tests and legacy code"""
        from .validation import StrikeValidator

        return StrikeValidator.find_stock_position_in_strikes(
            stock_price, valid_strikes
        )
