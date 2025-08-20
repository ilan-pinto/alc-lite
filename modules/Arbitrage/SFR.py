import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import logging
import numpy as np
from eventkit import Event
from ib_async import IB, Contract, Order, Ticker

from modules.Arbitrage.Strategy import ArbitrageClass, BaseExecutor, OrderManagerClass

from .common import get_logger
from .data_collection_metrics import (
    CollectionPhase,
    ContractPrioritizer,
    ContractPriority,
    DataCollectionMetrics,
    DataVelocityTracker,
    ProgressiveTimeoutConfig,
    log_phase_transition,
    should_continue_waiting,
)
from .metrics import RejectionReason, metrics_collector

# Configure logging will be done in main
logger = get_logger()

# Global contract_ticker for use in SFRExecutor and patching in tests
contract_ticker = {}


@dataclass
class ExpiryOption:
    """Data class to hold option contract information for a specific expiry"""

    expiry: str
    call_contract: Contract
    put_contract: Contract
    call_strike: float
    put_strike: float


class SFRExecutor(BaseExecutor):
    """
    Synthetic-Free-Risk (SFR) Executor class that handles the execution of SFR arbitrage strategies.

    This class is responsible for:
    1. Monitoring market data for stock and options across multiple expiries
    2. Calculating potential arbitrage opportunities
    3. Executing trades when conditions are met
    4. Logging trade details and results

    Attributes:
        ib (IB): Interactive Brokers connection instance
        order_manager (OrderManagerClass): Manager for handling order execution
        stock_contract (Contract): The underlying stock contract
        expiry_options (List[ExpiryOption]): List of option data for different expiries
        symbol (str): Trading symbol
        profit_target (float): Minimum profit target for execution
        cost_limit (float): Maximum price limit for execution
        start_time (float): Start time of the execution
        quantity (int): Quantity of contracts to execute
        all_contracts (List[Contract]): All contracts (stock + all options)
        is_active (bool): Whether the executor is currently active
        data_timeout (float): Maximum time to wait for all contract data (seconds)
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
        start_time: float,
        quantity: int = 1,
        data_timeout: float = 30.0,  # 30 seconds timeout for data collection
    ) -> None:
        """
        Initialize the SFR Executor.

        Args:
            ib: Interactive Brokers connection instance
            order_manager: Manager for handling order execution
            stock_contract: The underlying stock contract
            expiry_options: List of option data for different expiries
            symbol: Trading symbol
            profit_target: Minimum profit target for execution
            cost_limit: Maximum price limit for execution
            start_time: Start time of the execution
            quantity: Quantity of contracts to execute
            data_timeout: Maximum time to wait for all contract data (seconds)
        """
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
        self.quantity = quantity
        self.expiry_options = expiry_options
        self.all_contracts = [stock_contract] + option_contracts
        self.is_active = True
        self.data_timeout = data_timeout
        self.data_collection_start = time.time()

        # Progressive collection components
        self.phase1_checked = False
        self.phase2_checked = False
        self.current_phase = CollectionPhase.INITIALIZING
        self.priority_tiers = {}
        self.velocity_tracker = DataVelocityTracker()
        self.collection_metrics = DataCollectionMetrics(
            symbol=symbol, start_time=self.data_collection_start
        )

        # Determine market hours for timeout configuration
        self.timeout_config = ProgressiveTimeoutConfig.create_for_market_conditions(
            is_market_hours=self._is_market_hours(),
            total_contracts=len(self.all_contracts),
        )

    def find_stock_position_in_strikes(
        self, stock_price: float, valid_strikes: List[float]
    ) -> int:
        """
        Find the position of stock price within valid strikes array.
        Returns the index of the strike closest to or just below the stock price.
        """
        if not valid_strikes:
            return 0

        # Sort strikes to ensure proper positioning
        sorted_strikes = sorted(valid_strikes)

        # Find position - prefer strike at or just below stock price
        for i, strike in enumerate(sorted_strikes):
            if strike >= stock_price:
                # If exact match or first strike above stock price
                return max(0, i - 1) if strike > stock_price and i > 0 else i

        # Stock price is above all strikes
        return len(sorted_strikes) - 1

    def quick_viability_check(
        self, expiry_option: ExpiryOption, stock_price: float
    ) -> Tuple[bool, Optional[str]]:
        """Fast pre-filtering to eliminate non-viable opportunities early"""
        # Quick strike spread check
        strike_spread = expiry_option.call_strike - expiry_option.put_strike
        if strike_spread < 1.0 or strike_spread > 50.0:
            return False, "invalid_strike_spread"

        # Quick time to expiry check
        from datetime import datetime

        try:
            expiry_date = datetime.strptime(expiry_option.expiry, "%Y%m%d")
            days_to_expiry = (expiry_date - datetime.now()).days

            # DEBUG: Log expiry calculations for AAPL
            if expiry_option.call_strike == 184.5 and expiry_option.put_strike == 183.5:
                logger.warning(f"[AAPL DEBUG] Expiry check:")
                logger.warning(
                    f"  expiry_date={expiry_date}, current_date={datetime.now()}"
                )
                logger.warning(f"  days_to_expiry={days_to_expiry}")
                logger.warning(
                    f"  expiry_check: {days_to_expiry} < 15 or {days_to_expiry} > 50 = {days_to_expiry < 15 or days_to_expiry > 50}"
                )

            if days_to_expiry < 15 or days_to_expiry > 50:
                return False, "expiry_out_of_range"
        except ValueError:
            return False, "invalid_expiry_format"

        # Quick moneyness check - re-enabled with optimized thresholds
        if stock_price <= 0:
            return False, "invalid_stock_price"
        call_moneyness = expiry_option.call_strike / stock_price
        put_moneyness = expiry_option.put_strike / stock_price

        # More flexible thresholds to capture more opportunities
        if (
            call_moneyness < 0.90  # Allow deeper ITM calls
            or call_moneyness > 1.15  # Allow more OTM calls
            or put_moneyness < 0.80  # Allow deeper ITM puts
            or put_moneyness > 1.10  # Allow more OTM puts
        ):
            return False, "poor_moneyness"

        return True, None

    def _is_market_hours(self) -> bool:
        """Check if current time is during market hours (9:30 AM - 4:00 PM ET)"""
        from datetime import datetime, timedelta, timezone

        # Get current time in ET
        et_tz = timezone(timedelta(hours=-5))  # EST (adjust for DST if needed)
        current_et = datetime.now(et_tz)

        # Check if it's a weekday
        if current_et.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Check if time is between 9:30 AM and 4:00 PM ET
        market_open = current_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_et.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= current_et <= market_close

    def _is_valid_price(self, value) -> bool:
        """Check if a price value is valid (not None, not empty array, not NaN)"""
        if value is None:
            return False
        if hasattr(value, "__len__") and len(value) == 0:
            return False
        try:
            return not np.isnan(value)
        except (TypeError, ValueError):
            return False

    def _calculate_data_quality_score(
        self, stock_ticker, call_ticker, put_ticker
    ) -> float:
        """Calculate a quality score (0-1) based on data completeness and freshness"""
        score = 0.0

        # Stock data quality (30% weight)
        if stock_ticker:
            stock_score = 0.0
            if hasattr(stock_ticker, "bid") and self._is_valid_price(stock_ticker.bid):
                stock_score += 0.1
            if hasattr(stock_ticker, "ask") and self._is_valid_price(stock_ticker.ask):
                stock_score += 0.1
            if hasattr(stock_ticker, "last") and self._is_valid_price(
                stock_ticker.last
            ):
                stock_score += 0.05
            if hasattr(stock_ticker, "volume") and stock_ticker.volume > 0:
                stock_score += 0.05
            score += stock_score

        # Call option data quality (35% weight)
        if call_ticker:
            call_score = 0.0
            if hasattr(call_ticker, "bid") and self._is_valid_price(call_ticker.bid):
                call_score += 0.1
            if hasattr(call_ticker, "ask") and self._is_valid_price(call_ticker.ask):
                call_score += 0.1
            if hasattr(call_ticker, "last") and self._is_valid_price(call_ticker.last):
                call_score += 0.05
            if hasattr(call_ticker, "volume") and call_ticker.volume > 0:
                call_score += 0.05
            # Bid-ask spread quality
            if (
                hasattr(call_ticker, "bid")
                and hasattr(call_ticker, "ask")
                and self._is_valid_price(call_ticker.bid)
                and self._is_valid_price(call_ticker.ask)
            ):
                spread = abs(call_ticker.ask - call_ticker.bid)
                if spread < 5.0:  # Reasonable spread
                    call_score += 0.05
            score += call_score

        # Put option data quality (35% weight)
        if put_ticker:
            put_score = 0.0
            if hasattr(put_ticker, "bid") and self._is_valid_price(put_ticker.bid):
                put_score += 0.1
            if hasattr(put_ticker, "ask") and self._is_valid_price(put_ticker.ask):
                put_score += 0.1
            if hasattr(put_ticker, "last") and self._is_valid_price(put_ticker.last):
                put_score += 0.05
            if hasattr(put_ticker, "volume") and put_ticker.volume > 0:
                put_score += 0.05
            # Bid-ask spread quality
            if (
                hasattr(put_ticker, "bid")
                and hasattr(put_ticker, "ask")
                and self._is_valid_price(put_ticker.bid)
                and self._is_valid_price(put_ticker.ask)
            ):
                spread = abs(put_ticker.ask - put_ticker.bid)
                if spread < 5.0:  # Reasonable spread
                    put_score += 0.05
            score += put_score

        return min(score, 1.0)  # Cap at 1.0

    def initialize_contract_priorities(self, stock_price: float):
        """Initialize contract priority tiers based on moneyness"""
        self.priority_tiers = ContractPrioritizer.categorize_by_moneyness(
            self.expiry_options, stock_price
        )

        # Update metrics with expected contract counts
        for priority in [
            ContractPriority.CRITICAL,
            ContractPriority.IMPORTANT,
            ContractPriority.OPTIONAL,
        ]:
            count = (
                len(self.priority_tiers[priority]) * 2
            )  # 2 contracts per expiry (call + put)
            self.collection_metrics.contracts_expected[priority.value] = count

        logger.info(
            f"[{self.symbol}] Contract priorities initialized: "
            f"Critical={self.collection_metrics.contracts_expected['critical']}, "
            f"Important={self.collection_metrics.contracts_expected['important']}, "
            f"Optional={self.collection_metrics.contracts_expected['optional']}"
        )

    def get_contract_priority(self, contract) -> ContractPriority:
        """Get the priority of a specific contract"""
        return ContractPrioritizer.get_contract_priority(
            contract,
            self.expiry_options,
            self.last_stock_price if hasattr(self, "last_stock_price") else 0,
        )

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
    ) -> Tuple[bool, Optional[RejectionReason]]:

        spread = stock_price - put_strike

        # For conversion arbitrage: min_profit = net_credit - spread
        # We want min_profit > 0, so net_credit > spread
        if (
            net_credit <= spread
        ):  # Reject if net_credit <= spread (no arbitrage opportunity)
            logger.info(
                f"[{symbol}] net_credit[{net_credit}] <= spread[{spread}] - no arbitrage opportunity"
            )
            return False, RejectionReason.ARBITRAGE_CONDITION_NOT_MET

        elif min_profit < 0.05:  # Ensure minimum profit threshold (5 cents)
            logger.info(
                f"[{symbol}] min_profit[{min_profit}] < 0.05 - below minimum threshold"
            )
            return False, RejectionReason.ARBITRAGE_CONDITION_NOT_MET

        elif net_credit < 0:
            logger.info(
                f"[{symbol}] net_credit[{net_credit}] > 0 - doesn't meet conditions"
            )
            return False, RejectionReason.NET_CREDIT_NEGATIVE
        elif profit_target is not None and profit_target > min_roi:
            logger.info(
                f"[{symbol}]  profit_target({profit_target}) >  min_roi({min_roi} - doesn't meet conditions) "
            )
            return False, RejectionReason.PROFIT_TARGET_NOT_MET
        elif np.isnan(lmt_price) or lmt_price > cost_limit:
            logger.info(
                f"[{symbol}] np.isnan(lmt_price) or lmt_price > limit - doesn't meet conditions"
            )
            return False, RejectionReason.PRICE_LIMIT_EXCEEDED

        else:
            logger.info(f"[{symbol}] meets conditions - initiating order")
            return True, None

    async def executor(self, event: Event) -> None:
        """
        Progressive data collection executor with 3-phase timeout strategy.
        This method makes decisions as soon as sufficient data is available.
        """
        if not self.is_active:
            return

        try:
            # Update contract_ticker with new data and track metrics
            for tick in event:
                ticker: Ticker = tick
                contract = ticker.contract

                # Update ticker data with volume-based filtering and priority tracking
                if ticker.volume >= 1 and (
                    ticker.bid > 0 or ticker.ask > 0 or ticker.close > 0
                ):
                    contract_ticker[contract.conId] = ticker

                    # Track data arrival by priority
                    priority = self.get_contract_priority(contract)
                    self.collection_metrics.contracts_received[priority.value] += 1

                    # Record first data arrival time
                    if self.collection_metrics.time_to_first_data is None:
                        elapsed = time.time() - self.data_collection_start
                        self.collection_metrics.time_to_first_data = elapsed

                    if ticker.volume < 5:
                        logger.debug(
                            f"[{self.symbol}] Low volume ({ticker.volume}) for {contract.conId}"
                        )
                else:
                    logger.debug(
                        f"Skipping contract {contract.conId}: no valid data or volume"
                    )

            # Update velocity tracker
            total_received = self.collection_metrics.get_total_received()
            self.velocity_tracker.add_data_point(total_received)

            # Progressive phase checking
            elapsed_time = time.time() - self.data_collection_start

            # Initialize contract priorities on first stock data
            if not self.priority_tiers and self.has_stock_data():
                stock_ticker = contract_ticker.get(self.stock_contract.conId)
                if stock_ticker:
                    stock_price = self.get_stock_midpoint(stock_ticker)
                    if stock_price is not None and stock_price > 0:
                        self.last_stock_price = stock_price
                        self.initialize_contract_priorities(stock_price)

            # Phase 1: Check critical contracts (after 0.5s)
            if (
                elapsed_time >= self.timeout_config.phase_1_timeout
                and not self.phase1_checked
                and self.priority_tiers
            ):

                self.phase1_checked = True
                self.current_phase = CollectionPhase.PHASE_1_CRITICAL
                log_phase_transition(
                    self.symbol,
                    CollectionPhase.INITIALIZING,
                    self.current_phase,
                    self.collection_metrics,
                )

                if self.has_sufficient_critical_data():
                    opportunity = await self.evaluate_with_available_data(
                        ContractPriority.CRITICAL
                    )
                    if (
                        opportunity
                        and opportunity["guaranteed_profit"]
                        >= self.timeout_config.phase_1_profit_threshold
                    ):
                        logger.info(
                            f"[{self.symbol}] Phase 1 execution: profit={opportunity['guaranteed_profit']:.2f}"
                        )
                        await self.execute_opportunity(opportunity)
                        return

            # Phase 2: Check important contracts (after 1.5s)
            if (
                elapsed_time >= self.timeout_config.phase_2_timeout
                and not self.phase2_checked
                and self.priority_tiers
            ):

                self.phase2_checked = True
                self.current_phase = CollectionPhase.PHASE_2_IMPORTANT
                log_phase_transition(
                    self.symbol,
                    CollectionPhase.PHASE_1_CRITICAL,
                    self.current_phase,
                    self.collection_metrics,
                )

                if self.has_sufficient_important_data():
                    opportunity = await self.evaluate_with_available_data(
                        ContractPriority.IMPORTANT
                    )
                    if (
                        opportunity
                        and opportunity["guaranteed_profit"]
                        >= self.timeout_config.phase_2_profit_threshold
                    ):
                        logger.info(
                            f"[{self.symbol}] Phase 2 execution: profit={opportunity['guaranteed_profit']:.2f}"
                        )
                        await self.execute_opportunity(opportunity)
                        return

            # Phase 3: Final check with all available data (after 3.0s)
            if elapsed_time >= self.timeout_config.phase_3_timeout:
                self.current_phase = CollectionPhase.PHASE_3_FINAL
                log_phase_transition(
                    self.symbol,
                    CollectionPhase.PHASE_2_IMPORTANT,
                    self.current_phase,
                    self.collection_metrics,
                )

                if self.has_minimum_viable_data():
                    opportunity = await self.evaluate_with_available_data(
                        ContractPriority.OPTIONAL
                    )
                    if (
                        opportunity
                        and opportunity["guaranteed_profit"]
                        >= self.timeout_config.phase_3_profit_threshold
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
                    self.collection_metrics, self.timeout_config, self.velocity_tracker
                )
                if not should_continue:
                    # Before finishing without execution, try to evaluate opportunities with available data
                    logger.info(
                        f"[{self.symbol}] Data collection complete early ({stop_reason}), evaluating opportunities..."
                    )

                    if self.has_minimum_viable_data():
                        opportunity = await self.evaluate_with_available_data(
                            ContractPriority.OPTIONAL  # Use all available data since collection is complete
                        )
                        if (
                            opportunity
                            and opportunity["guaranteed_profit"]
                            >= 0.10  # Use minimum profit threshold
                        ):
                            logger.info(
                                f"[{self.symbol}] Early completion execution: profit={opportunity['guaranteed_profit']:.2f}"
                            )
                            await self.execute_opportunity(opportunity)
                            return

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
                    return

        except Exception as e:
            logger.error(f"Error in progressive executor: {str(e)}")
            self.finish_collection_without_execution(f"error: {str(e)}")

    def calc_price_and_build_order_for_expiry(
        self,
        expiry_option: ExpiryOption,
        priority_filter: Optional[ContractPriority] = None,
    ) -> Optional[Tuple[Contract, Order, float, Dict]]:
        """
        Calculate price and build order for a specific expiry option.
        Priority filter allows evaluation with partial data - only evaluates contracts
        of specified priority or higher.
        Returns tuple of (contract, order, min_profit, trade_details) or None if no opportunity.
        """
        try:
            # DEBUG: Log AAPL evaluation entry
            if (
                self.symbol == "AAPL"
                and expiry_option.call_strike == 184.5
                and expiry_option.put_strike == 183.5
            ):
                logger.warning(
                    f"[AAPL DEBUG] calc_price_and_build_order_for_expiry called for C{expiry_option.call_strike}/P{expiry_option.put_strike}"
                )

            # Track entry into evaluation funnel
            metrics_collector.record_funnel_stage(
                self.symbol, expiry_option.expiry, "evaluated"
            )

            # Fast pre-filtering to eliminate non-viable opportunities early
            stock_ticker = contract_ticker.get(self.stock_contract.conId)
            if not stock_ticker:
                return None

            # Track stock ticker availability
            metrics_collector.record_funnel_stage(
                self.symbol, expiry_option.expiry, "stock_ticker_available"
            )

            # Apply priority filtering if specified (for partial data collection)
            if priority_filter:
                stock_price = self.get_stock_midpoint(
                    stock_ticker
                )  # Fallback internally handled
                if stock_price is None:
                    return None

                contract_priority = ContractPrioritizer.get_contract_priority(
                    expiry_option.call_contract, self.expiry_options, stock_price
                )

                # Skip if contract doesn't meet priority threshold
                priority_order = {
                    ContractPriority.CRITICAL: 0,
                    ContractPriority.IMPORTANT: 1,
                    ContractPriority.OPTIONAL: 2,
                }

                if priority_order[contract_priority] > priority_order[priority_filter]:
                    logger.debug(
                        f"[{self.symbol}] Skipping {expiry_option.expiry} - priority {contract_priority.value} "
                        f"not meeting filter {priority_filter.value}"
                    )
                    return None

            # Track passing priority filter
            metrics_collector.record_funnel_stage(
                self.symbol, expiry_option.expiry, "passed_priority_filter"
            )

            # STAGE 1: Use midpoint for opportunity detection
            stock_fair = stock_ticker.midpoint()

            # Fallback if midpoint not available
            if (
                stock_fair is None
                or (hasattr(stock_fair, "__len__") and len(stock_fair) == 0)
                or np.isnan(stock_fair)
            ):
                stock_fair = (
                    stock_ticker.last
                    if not np.isnan(stock_ticker.last)
                    else stock_ticker.close
                )

            viable, reason = self.quick_viability_check(expiry_option, stock_fair)
            if not viable:
                logger.debug(
                    f"[{self.symbol}] Quick rejection for {expiry_option.expiry}: {reason}"
                )
                return None

            # Track passing viability check
            metrics_collector.record_funnel_stage(
                self.symbol, expiry_option.expiry, "passed_viability_check"
            )

            # Get option data
            call_ticker = contract_ticker.get(expiry_option.call_contract.conId)
            put_ticker = contract_ticker.get(expiry_option.put_contract.conId)

            if not call_ticker or not put_ticker:
                # In progressive mode, missing data might be acceptable for lower priority contracts
                if priority_filter and priority_filter in [
                    ContractPriority.IMPORTANT,
                    ContractPriority.OPTIONAL,
                ]:
                    logger.debug(
                        f"[{self.symbol}] Missing option data for {expiry_option.expiry} - skipping due to partial collection"
                    )
                    return None

                metrics_collector.add_rejection_reason(
                    RejectionReason.MISSING_MARKET_DATA,
                    {
                        "symbol": self.symbol,
                        "contract_type": "options",
                        "expiry": expiry_option.expiry,
                        "call_strike": expiry_option.call_strike,
                        "put_strike": expiry_option.put_strike,
                        "missing_call_data": not call_ticker,
                        "missing_put_data": not put_ticker,
                    },
                )
                return None

            # Track option data availability
            metrics_collector.record_funnel_stage(
                self.symbol, expiry_option.expiry, "option_data_available"
            )

            # Calculate data quality score for better decision making in partial mode
            data_quality_score = self._calculate_data_quality_score(
                stock_ticker, call_ticker, put_ticker
            )

            # In partial data mode, require higher quality for execution
            min_quality_threshold = 0.8 if priority_filter else 0.6
            if data_quality_score < min_quality_threshold:
                logger.debug(
                    f"[{self.symbol}] Data quality {data_quality_score:.2f} below threshold "
                    f"{min_quality_threshold} for {expiry_option.expiry}"
                )
                return None

            # Track passing data quality check
            metrics_collector.record_funnel_stage(
                self.symbol, expiry_option.expiry, "passed_data_quality"
            )

            # Get midpoint prices for theoretical calculation
            call_fair = call_ticker.midpoint()
            put_fair = put_ticker.midpoint()

            # Fallback if midpoints not available
            if (
                call_fair is None
                or (hasattr(call_fair, "__len__") and len(call_fair) == 0)
                or np.isnan(call_fair)
            ):
                call_fair = (
                    call_ticker.last
                    if not np.isnan(call_ticker.last)
                    else call_ticker.close
                )
            if (
                put_fair is None
                or (hasattr(put_fair, "__len__") and len(put_fair) == 0)
                or np.isnan(put_fair)
            ):
                put_fair = (
                    put_ticker.last
                    if not np.isnan(put_ticker.last)
                    else put_ticker.close
                )

            if np.isnan(call_fair) or np.isnan(put_fair):
                metrics_collector.add_rejection_reason(
                    RejectionReason.INVALID_CONTRACT_DATA,
                    {
                        "symbol": self.symbol,
                        "contract_type": "options",
                        "expiry": expiry_option.expiry,
                        "call_strike": expiry_option.call_strike,
                        "put_strike": expiry_option.put_strike,
                        "call_price_invalid": np.isnan(call_fair),
                        "put_price_invalid": np.isnan(put_fair),
                    },
                )
                return None

            # Track valid prices
            metrics_collector.record_funnel_stage(
                self.symbol, expiry_option.expiry, "prices_valid"
            )

            # Calculate theoretical arbitrage with fair values
            theoretical_net_credit = call_fair - put_fair
            theoretical_spread = stock_fair - expiry_option.put_strike
            theoretical_profit = theoretical_net_credit - theoretical_spread

            # DEBUG: Log exact values for AAPL test scenario
            if (
                self.symbol == "AAPL"
                and expiry_option.call_strike == 184.5
                and expiry_option.put_strike == 183.5
            ):
                logger.warning(f"[AAPL DEBUG] Theoretical calculation:")
                logger.warning(
                    f"  call_fair={call_fair:.4f}, put_fair={put_fair:.4f}, stock_fair={stock_fair:.4f}"
                )
                logger.warning(
                    f"  call_strike={expiry_option.call_strike}, put_strike={expiry_option.put_strike}"
                )
                logger.warning(f"  theoretical_net_credit={theoretical_net_credit:.4f}")
                logger.warning(f"  theoretical_spread={theoretical_spread:.4f}")
                logger.warning(f"  theoretical_profit={theoretical_profit:.4f}")

            # Track ALL theoretical profit calculations (positive and negative)
            metrics_collector.record_profit_calculation(
                self.symbol, expiry_option.expiry, theoretical_profit
            )

            # Quick reject if no theoretical opportunity
            if theoretical_profit < 0.20:  # 20 cents minimum theoretical
                logger.warning(
                    f"[{self.symbol}] No theoretical arbitrage for {expiry_option.expiry}: "
                    f"profit=${theoretical_profit:.2f}"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.ARBITRAGE_CONDITION_NOT_MET,
                    {
                        "symbol": self.symbol,
                        "theoretical_profit": theoretical_profit,
                        "theoretical_net_credit": theoretical_net_credit,
                        "theoretical_spread": theoretical_spread,
                        "stage": "theoretical_evaluation",
                    },
                )
                return None

            # Track positive theoretical profit
            metrics_collector.record_funnel_stage(
                self.symbol, expiry_option.expiry, "theoretical_profit_positive"
            )

            # STAGE 2: Execution validation using actual prices
            # These are the prices we'll actually pay/receive
            call_exec = (
                call_ticker.bid if not np.isnan(call_ticker.bid) else call_ticker.close
            )
            put_exec = (
                put_ticker.ask if not np.isnan(put_ticker.ask) else put_ticker.close
            )
            stock_exec = (
                stock_ticker.ask
                if not np.isnan(stock_ticker.ask)
                else stock_ticker.close
            )

            if np.isnan(call_exec) or np.isnan(put_exec) or np.isnan(stock_exec):
                metrics_collector.add_rejection_reason(
                    RejectionReason.INVALID_CONTRACT_DATA,
                    {
                        "symbol": self.symbol,
                        "contract_type": "execution_prices",
                        "expiry": expiry_option.expiry,
                        "call_strike": expiry_option.call_strike,
                        "put_strike": expiry_option.put_strike,
                        "call_exec_invalid": np.isnan(call_exec),
                        "put_exec_invalid": np.isnan(put_exec),
                        "stock_exec_invalid": np.isnan(stock_exec),
                    },
                )
                return None

            # Calculate guaranteed profit with execution prices
            guaranteed_net_credit = call_exec - put_exec
            guaranteed_spread = stock_exec - expiry_option.put_strike
            guaranteed_profit = guaranteed_net_credit - guaranteed_spread

            # DEBUG: Log exact values for AAPL test scenario
            if (
                self.symbol == "AAPL"
                and expiry_option.call_strike == 184.5
                and expiry_option.put_strike == 183.5
            ):
                logger.warning(f"[AAPL DEBUG] Guaranteed calculation:")
                logger.warning(
                    f"  call_exec={call_exec:.4f}, put_exec={put_exec:.4f}, stock_exec={stock_exec:.4f}"
                )
                logger.warning(f"  guaranteed_net_credit={guaranteed_net_credit:.4f}")
                logger.warning(f"  guaranteed_spread={guaranteed_spread:.4f}")
                logger.warning(f"  guaranteed_profit={guaranteed_profit:.4f}")

            # Track guaranteed profit calculation
            metrics_collector.record_profit_calculation(
                self.symbol, expiry_option.expiry, theoretical_profit, guaranteed_profit
            )

            # Must have guaranteed profit after execution
            if guaranteed_profit < 0.10:  # 10 cents minimum guaranteed
                logger.info(
                    f"[{self.symbol}] Theoretical profit ${theoretical_profit:.2f} "
                    f"but guaranteed only ${guaranteed_profit:.2f} - rejecting"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.ARBITRAGE_CONDITION_NOT_MET,
                    {
                        "symbol": self.symbol,
                        "theoretical_profit": theoretical_profit,
                        "guaranteed_profit": guaranteed_profit,
                        "stock_fair": stock_fair,
                        "stock_exec": stock_exec,
                        "stage": "execution_validation",
                    },
                )
                return None

            # Track positive guaranteed profit
            metrics_collector.record_funnel_stage(
                self.symbol, expiry_option.expiry, "guaranteed_profit_positive"
            )

            # Check bid-ask spread for both call and put contracts to prevent crazy price ranges
            call_bid_ask_spread = (
                abs(call_ticker.ask - call_ticker.bid)
                if (not np.isnan(call_ticker.ask) and not np.isnan(call_ticker.bid))
                else float("inf")
            )
            put_bid_ask_spread = (
                abs(put_ticker.ask - put_ticker.bid)
                if (not np.isnan(put_ticker.ask) and not np.isnan(put_ticker.bid))
                else float("inf")
            )

            if call_bid_ask_spread > 20:
                logger.info(
                    f"[{self.symbol}] Call contract bid-ask spread too wide: {call_bid_ask_spread:.2f} > 15.00, "
                    f"expiry: {expiry_option.expiry}, strike: {expiry_option.call_strike}"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.BID_ASK_SPREAD_TOO_WIDE,
                    {
                        "symbol": self.symbol,
                        "contract_type": "call",
                        "expiry": expiry_option.expiry,
                        "strike": expiry_option.call_strike,
                        "bid_ask_spread": call_bid_ask_spread,
                        "threshold": 20,
                    },
                )
                return None

            if put_bid_ask_spread > 20:
                logger.info(
                    f"[{self.symbol}] Put contract bid-ask spread too wide: {put_bid_ask_spread:.2f} > 15.00, "
                    f"expiry: {expiry_option.expiry}, strike: {expiry_option.put_strike}"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.BID_ASK_SPREAD_TOO_WIDE,
                    {
                        "symbol": self.symbol,
                        "contract_type": "put",
                        "expiry": expiry_option.expiry,
                        "strike": expiry_option.put_strike,
                        "bid_ask_spread": put_bid_ask_spread,
                        "threshold": 20,
                    },
                )
                return None

            # Use guaranteed execution prices for final calculations
            net_credit = guaranteed_net_credit
            stock_price = stock_exec
            call_price = call_exec
            put_price = put_exec

            # Round values for display
            net_credit = round(net_credit, 2)
            stock_price = round(stock_price, 2)
            call_price = round(call_price, 2)
            put_price = round(put_price, 2)

            # temp condition
            if expiry_option.call_strike < expiry_option.put_strike:
                logger.info(
                    f"call_strike:{expiry_option.call_strike} < put_strike:{expiry_option.put_strike}"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.INVALID_STRIKE_COMBINATION,
                    {
                        "symbol": self.symbol,
                        "expiry": expiry_option.expiry,
                        "call_strike": expiry_option.call_strike,
                        "put_strike": expiry_option.put_strike,
                        "reason": "call_strike < put_strike",
                    },
                )
                return None

            # Use guaranteed profit that we already calculated
            spread = stock_price - expiry_option.put_strike
            min_profit = guaranteed_profit  # Already calculated and validated above
            max_profit = (
                expiry_option.call_strike - expiry_option.put_strike
            ) + net_credit
            min_roi = (
                (min_profit / (stock_price + net_credit)) * 100
                if (stock_price + net_credit) > 0
                else 0
            )

            # Calculate precise combo limit price based on target leg prices
            combo_limit_price = self.calculate_combo_limit_price(
                stock_price=stock_price,
                call_price=call_price,
                put_price=put_price,
                buffer_percent=0.01,  # 1% buffer for realistic execution
            )

            logger.info(
                f"[{self.symbol}] Expiry: {expiry_option.expiry} theoretical_profit:{theoretical_profit:.2f}, "
                f"guaranteed_profit:{min_profit:.2f}, max_profit:{max_profit:.2f}, min_roi:{min_roi:.2f}%"
            )

            conditions_met, rejection_reason = self.check_conditions(
                self.symbol,
                self.profit_target,
                self.cost_limit,
                expiry_option.put_strike,
                combo_limit_price,  # Use calculated precise limit price
                net_credit,
                min_roi,
                stock_price,
                min_profit,
            )

            if conditions_met:
                # Build order with precise limit price and target leg prices
                conversion_contract, order = self.build_order(
                    self.symbol,
                    self.stock_contract,
                    expiry_option.call_contract,
                    expiry_option.put_contract,
                    combo_limit_price,  # Use calculated precise limit price
                    self.quantity,
                    call_price=call_price,  # Target call leg price
                    put_price=put_price,  # Target put leg price
                )

                # Prepare trade details for logging (don't log yet)
                trade_details = {
                    "call_strike": expiry_option.call_strike,
                    "call_price": call_price,
                    "put_strike": expiry_option.put_strike,
                    "put_price": put_price,
                    "stock_price": stock_price,
                    "net_credit": net_credit,
                    "min_profit": min_profit,
                    "max_profit": max_profit,
                    "min_roi": min_roi,
                    "expiry": expiry_option.expiry,
                }

                return conversion_contract, order, min_profit, trade_details
            else:
                # Record rejection reason
                if rejection_reason:
                    # Calculate profit_ratio for context
                    profit_ratio = (
                        max_profit / abs(min_profit) if min_profit != 0 else 0
                    )

                    metrics_collector.add_rejection_reason(
                        rejection_reason,
                        {
                            "symbol": self.symbol,
                            "expiry": expiry_option.expiry,
                            "call_strike": expiry_option.call_strike,
                            "put_strike": expiry_option.put_strike,
                            "stock_price": stock_price,
                            "net_credit": net_credit,
                            "min_profit": min_profit,
                            "max_profit": max_profit,
                            "min_roi": min_roi,
                            "combo_limit_price": combo_limit_price,
                            "cost_limit": self.cost_limit,
                            "profit_target": self.profit_target,
                            "spread": spread,
                            "profit_ratio": profit_ratio,
                        },
                    )

            return None
        except Exception as e:
            logger.error(f"Error in calc_price_and_build_order_for_expiry: {str(e)}")
            return None

    def has_stock_data(self) -> bool:
        """Check if we have stock data available"""
        return contract_ticker.get(self.stock_contract.conId) is not None

    def get_stock_midpoint(self, stock_ticker) -> Optional[float]:
        """Get stock midpoint price, with fallbacks"""
        try:
            midpoint = stock_ticker.midpoint()
            if midpoint is not None and not np.isnan(midpoint) and midpoint > 0:
                return midpoint
        except (ZeroDivisionError, TypeError, AttributeError):
            pass

        # Fallback to last or close
        if hasattr(stock_ticker, "last") and not np.isnan(stock_ticker.last):
            return stock_ticker.last
        if hasattr(stock_ticker, "close") and not np.isnan(stock_ticker.close):
            return stock_ticker.close

        return None

    def has_sufficient_critical_data(self) -> bool:
        """Check if we have sufficient critical contract data"""
        if not self.priority_tiers:
            return False

        critical_received = self.collection_metrics.contracts_received["critical"]
        critical_expected = self.collection_metrics.contracts_expected["critical"]

        if critical_expected == 0 or critical_expected <= 0:
            return False

        percentage = critical_received / critical_expected
        return percentage >= self.timeout_config.critical_threshold

    def has_sufficient_important_data(self) -> bool:
        """Check if we have sufficient critical + important contract data"""
        if not self.priority_tiers:
            return False

        critical_received = self.collection_metrics.contracts_received["critical"]
        critical_expected = self.collection_metrics.contracts_expected["critical"]
        important_received = self.collection_metrics.contracts_received["important"]
        important_expected = self.collection_metrics.contracts_expected["important"]

        total_received = critical_received + important_received
        total_expected = critical_expected + important_expected

        if total_expected == 0 or total_expected <= 0:
            return False

        percentage = total_received / total_expected
        return percentage >= self.timeout_config.important_threshold

    def has_minimum_viable_data(self) -> bool:
        """Check if we have minimum viable data to make any decision"""
        # Need stock data
        if not self.has_stock_data():
            return False

        # Need at least 1 option pair with recent data (relaxed for testing scenarios)
        viable_pairs = 0

        for expiry_option in self.expiry_options:
            call_ticker = contract_ticker.get(expiry_option.call_contract.conId)
            put_ticker = contract_ticker.get(expiry_option.put_contract.conId)

            if call_ticker and put_ticker:
                # Check data freshness (assuming we have timestamp or volume > 0 indicates fresh data)
                if (
                    hasattr(call_ticker, "volume")
                    and call_ticker.volume > 0
                    and hasattr(put_ticker, "volume")
                    and put_ticker.volume > 0
                ):
                    viable_pairs += 1

        return viable_pairs >= 1  # Allow evaluation with single expiry for testing

    async def evaluate_with_available_data(
        self, max_priority: ContractPriority
    ) -> Optional[Dict]:
        """Evaluate opportunities with currently available data up to specified priority"""
        best_opportunity = None
        best_profit = 0

        # Determine which expiry options to consider based on priority
        eligible_options = []
        for priority in [
            ContractPriority.CRITICAL,
            ContractPriority.IMPORTANT,
            ContractPriority.OPTIONAL,
        ]:
            if priority in self.priority_tiers:
                eligible_options.extend(self.priority_tiers[priority])
            if priority == max_priority:
                break

        for expiry_option in eligible_options:
            # DEBUG: Check AAPL data availability
            if (
                self.symbol == "AAPL"
                and expiry_option.call_strike == 184.5
                and expiry_option.put_strike == 183.5
            ):
                has_data = self.has_data_for_option_pair(expiry_option)
                logger.warning(
                    f"[AAPL DEBUG] evaluate_with_available_data: has_data_for_option_pair={has_data}"
                )

            # Skip if we don't have data for this option pair
            if not self.has_data_for_option_pair(expiry_option):
                continue

            try:
                # Use enhanced calculation method with priority filtering
                opportunity_result = self.calc_price_and_build_order_for_expiry(
                    expiry_option, max_priority
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

        return best_opportunity

    def has_data_for_option_pair(self, expiry_option: ExpiryOption) -> bool:
        """Check if we have data for both call and put contracts"""
        call_data = contract_ticker.get(expiry_option.call_contract.conId)
        put_data = contract_ticker.get(expiry_option.put_contract.conId)
        return call_data is not None and put_data is not None

    async def execute_opportunity(self, opportunity: Dict):
        """Execute a trading opportunity"""
        try:
            self.collection_metrics.decision_confidence = (
                self.collection_metrics.get_completion_percentage()
            )
            self.collection_metrics.opportunity_found = True
            self.collection_metrics.time_to_decision = (
                time.time() - self.data_collection_start
            )
            self.collection_metrics.final_phase = self.current_phase

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

            # Place the order
            self.is_active = False  # Prevent multiple executions
            result = await self.order_manager.place_order(
                opportunity["contract"], opportunity["order"]
            )

            if result:
                self.collection_metrics.execution_triggered = True
                logger.info(f"[{self.symbol}] Order placed successfully")
                metrics_collector.record_opportunity_found(self.symbol)
                metrics_collector.finish_scan(success=True)
            else:
                logger.warning(f"[{self.symbol}] Order placement failed")
                metrics_collector.finish_scan(
                    success=False, error_message="Order placement failed"
                )

            self.deactivate()

        except Exception as e:
            logger.error(f"Error executing opportunity: {str(e)}")
            self.finish_collection_without_execution(f"execution_error: {str(e)}")

    def finish_collection_without_execution(self, reason: str):
        """Finish collection without executing any trades"""
        self.collection_metrics.time_to_decision = (
            time.time() - self.data_collection_start
        )
        self.collection_metrics.final_phase = self.current_phase

        logger.info(f"[{self.symbol}] Collection finished without execution: {reason}")
        logger.info(
            f"[{self.symbol}] Final data: {self.collection_metrics.get_completion_percentage():.1f}% "
            f"({self.collection_metrics.get_total_received()}/{self.collection_metrics.get_total_expected()})"
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
                    # Clean up from global contract_ticker
                    if contract.conId in contract_ticker:
                        del contract_ticker[contract.conId]
                except Exception as e:
                    logger.debug(
                        f"Error cancelling market data for contract {contract.conId}: {str(e)}"
                    )

            # Call parent deactivate method
            super().deactivate()
            logger.debug(
                f"[{self.symbol}] Executor deactivated and cleaned up {len(self.all_contracts)} contracts"
            )


class SFR(ArbitrageClass):
    """
    Synthetic-Free-Risk (SFR) arbitrage strategy class.
    This class uses a more efficient approach by creating one executor per symbol
    that handles all expiries, eliminating the need to constantly add/remove event handlers.
    """

    def __init__(self, log_file: str = None):
        super().__init__(log_file=log_file)
        # Default strike selection parameters for backward compatibility with tests
        self.max_combinations = 10
        self.max_strike_difference = 5

    def cleanup_inactive_executors(self):
        """Enhanced cleanup that properly cancels market data subscriptions"""
        inactive_symbols = [
            symbol
            for symbol, executor in self.active_executors.items()
            if not executor.is_active
        ]

        # Ensure inactive executors have been properly deactivated
        for symbol in inactive_symbols:
            executor = self.active_executors[symbol]
            if hasattr(executor, "deactivate") and executor.is_active:
                # Call deactivate to clean up market data subscriptions
                executor.deactivate()

        # Call parent cleanup method
        super().cleanup_inactive_executors()

        if inactive_symbols:
            logger.info(
                f"[SFR] Enhanced cleanup completed for {len(inactive_symbols)} inactive executors"
            )

    async def scan(
        self,
        symbol_list,
        cost_limit,
        profit_target=0.50,
        volume_limit=100,
        quantity=1,
        max_combinations=10,
        max_strike_difference=5,
    ):
        """
        scan for SFR and execute order

        symbol list - list of valid symbols
        cost_limit - min price for the contract. e.g limit=50 means willing to pay up to 5000$
        profit_target - min acceptable min roi
        volume_limit - min option contract volume
        quantity - number of contracts to trade
        """
        # Global
        global contract_ticker
        contract_ticker = {}

        # set configuration
        self.profit_target = profit_target
        self.volume_limit = volume_limit
        self.cost_limit = cost_limit
        self.quantity = quantity
        self.max_combinations = max_combinations
        self.max_strike_difference = max_strike_difference

        await self.ib.connectAsync("127.0.0.1", 7497, clientId=2)
        self.ib.orderStatusEvent += self.onFill

        # Set up single event handler for all symbols
        self.ib.pendingTickersEvent += self.master_executor

        try:
            while not self.order_filled:
                # Start cycle tracking
                _ = metrics_collector.start_cycle(len(symbol_list))

                tasks = []
                for symbol in symbol_list:
                    # Check if order was filled during symbol processing
                    if self.order_filled:
                        break

                    # Use throttled scanning instead of fixed delays
                    task = asyncio.create_task(
                        self.scan_with_throttle(
                            symbol,
                            self.scan_sfr,
                            self.quantity,
                            self.profit_target,
                            self.cost_limit,
                        )
                    )
                    tasks.append(task)
                    # Minimal delay for API rate limiting
                    await asyncio.sleep(0.1)

                # Wait for all tasks to complete
                _ = await asyncio.gather(*tasks, return_exceptions=True)

                # Clean up inactive executors
                self.cleanup_inactive_executors()

                # Finish cycle tracking
                metrics_collector.finish_cycle()

                # Print metrics summary periodically
                if len(metrics_collector.scan_metrics) > 0:
                    metrics_collector.print_summary()

                # Check if order was filled before continuing
                if self.order_filled:
                    logger.info("Order filled - exiting scan loop")
                    break

                # Reset for next iteration
                contract_ticker = {}
                await asyncio.sleep(5)  # Reduced wait time for faster cycles
        except Exception as e:
            logger.error(f"Error in scan loop: {str(e)}")
        finally:
            # Always print final metrics summary before exiting
            logger.info("Scanning complete - printing final metrics summary")
            if len(metrics_collector.scan_metrics) > 0:
                metrics_collector.print_summary()

            # Deactivate all executors and disconnect from IB
            logger.info("Deactivating all executors and disconnecting from IB")
            self.deactivate_all_executors()
            self.ib.disconnect()

    def find_stock_position_in_strikes(
        self, stock_price: float, valid_strikes: List[float]
    ) -> int:
        """
        Find the position of stock price within valid strikes array.
        Returns the index of the strike closest to or just below the stock price.
        """
        if not valid_strikes:
            return 0

        # Sort strikes to ensure proper positioning
        sorted_strikes = sorted(valid_strikes)

        # Find position - prefer strike at or just below stock price
        for i, strike in enumerate(sorted_strikes):
            if strike >= stock_price:
                # If exact match or first strike above stock price
                return max(0, i - 1) if strike > stock_price and i > 0 else i

        # Stock price is above all strikes
        return len(sorted_strikes) - 1

    async def scan_sfr(self, symbol, quantity=1, profit_target=0.50, cost_limit=120.0):
        """
        Scan for SFR opportunities for a specific symbol.
        Creates a single executor per symbol that handles all expiries.

        Args:
            symbol: Trading symbol to scan
            quantity: Number of contracts to trade
            profit_target: Minimum profit target (default: 0.50%)
            cost_limit: Maximum cost limit (default: $120.0)
        """
        # Start metrics collection for this scan
        _ = metrics_collector.start_scan(symbol, "SFR")

        # Set configuration for this scan
        self.profit_target = profit_target
        self.cost_limit = cost_limit

        # Ensure strike selection parameters have defaults (for backward compatibility)
        if not hasattr(self, "max_combinations"):
            self.max_combinations = 10
        if not hasattr(self, "max_strike_difference"):
            self.max_strike_difference = 5

        try:
            _, _, stock = self._get_stock_contract(symbol)

            # Request market data for the stock
            market_data = await self._get_market_data_async(stock)

            stock_price = (
                market_data.last
                if not np.isnan(market_data.last)
                else market_data.close
            )

            logger.info(f"price for [{symbol}: {stock_price} ]")

            # Request options chain
            chain = await self._get_chain(stock, exchange="SMART")

            # Define parameters for the options (expiry and strike price)
            valid_strikes = [
                s for s in chain.strikes if s <= stock_price and s > stock_price - 25
            ]  # Example strike price

            if len(valid_strikes) < 2:
                logger.info(
                    f"Not enough valid strikes found for {symbol} (found: {len(valid_strikes)})"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.INSUFFICIENT_VALID_STRIKES,
                    {
                        "symbol": symbol,
                        "valid_strikes_count": len(valid_strikes),
                        "required_strikes": 2,
                        "stock_price": stock_price,
                    },
                )
                return

            # Prepare for parallel contract qualification
            valid_expiries = self.filter_expirations_within_range(
                chain.expirations, 19, 45
            )

            if len(valid_expiries) == 0:
                logger.warning(
                    f"No valid expiries found for {symbol} in range 19-45 days, skipping scan"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.NO_VALID_EXPIRIES,
                    {
                        "available_expiries": len(chain.expirations),
                        "days_range": "19-45",
                    },
                )
                return

            if len(valid_strikes) < 2:
                logger.info(
                    f"Not enough valid strikes for {symbol}, skipping parallel qualification"
                )
                return

            # Adaptive strike position logic for conversion arbitrage
            valid_strike_pairs = []

            # Sort strikes for position-based selection
            sorted_strikes = sorted(valid_strikes)

            # Find stock price position within valid strikes
            stock_position = self.find_stock_position_in_strikes(
                stock_price, sorted_strikes
            )

            # Adaptive strike ranges based on position (not dollar amounts)
            # Call candidates: stock position ± 3 (expanded for more opportunities)
            call_start = max(0, stock_position - 3)
            call_end = min(len(sorted_strikes), stock_position + 4)
            call_candidates = sorted_strikes[call_start:call_end]

            # Put candidates: stock position -2 to +2 (expanded range for flexibility)
            put_start = max(0, stock_position - 2)
            put_end = min(len(sorted_strikes), stock_position + 3)
            put_candidates = sorted_strikes[put_start:put_end]

            # Generate combinations with expanded strike differences
            priority_combinations = []  # Strike difference = 1-2 (highest priority)
            secondary_combinations = []  # Strike difference = 3-5 (lower priority)

            for call_strike in call_candidates:
                for put_strike in put_candidates:
                    # Enforce call_strike > put_strike for proper conversion arbitrage
                    if call_strike > put_strike:
                        # Calculate strike difference in position terms
                        call_idx = sorted_strikes.index(call_strike)
                        put_idx = sorted_strikes.index(put_strike)
                        strike_diff = call_idx - put_idx

                        combination = (call_strike, put_strike, strike_diff)

                        if strike_diff in [1, 2]:
                            # 1-2 strike difference: highest probability of trade
                            priority_combinations.append(combination)
                        elif strike_diff <= self.max_strike_difference:
                            # 3+ strike difference: configurable secondary options
                            secondary_combinations.append(combination)

            # Sort by strike difference (lower differences have higher priority) for optimal trade probability
            priority_combinations.sort(key=lambda x: x[2])
            secondary_combinations.sort(key=lambda x: x[2])

            # Combine and limit to configurable number of best combinations
            all_combinations = priority_combinations + secondary_combinations
            valid_strike_pairs = [
                (call, put)
                for call, put, _ in all_combinations[: self.max_combinations]
            ]

            # Track strike selection effectiveness
            total_strikes = len(chain.strikes)
            combinations_generated = len(all_combinations)
            combinations_tested = len(valid_strike_pairs)
            metrics_collector.record_strike_effectiveness(
                symbol,
                total_strikes,
                len(valid_strikes),
                combinations_generated,
                combinations_tested,
            )

            logger.info(
                f"[{symbol}] Testing {len(valid_strike_pairs)} conversion-optimized strike combinations "
                f"(stock position: {stock_position}, price: ${stock_price:.2f})"
            )
            if priority_combinations:
                logger.info(
                    f"[{symbol}] Found {len(priority_combinations)} high-probability 1-2 strike difference combinations"
                )
            if secondary_combinations:
                logger.info(
                    f"[{symbol}] Found {len(secondary_combinations)} secondary 3-{self.max_strike_difference} strike difference combinations"
                )

            # Parallel qualification of all contracts
            qualified_contracts_map = await self.parallel_qualify_all_contracts(
                symbol, valid_expiries, valid_strike_pairs
            )

            # Build expiry options from qualified contracts
            expiry_options = []
            all_contracts = [stock]

            # Limit expiry options to avoid too many low volume contracts
            max_expiry_options = (
                12  # Expanded limit to accommodate more strike combinations
            )

            for expiry in valid_expiries:
                if len(expiry_options) >= max_expiry_options:
                    logger.debug(
                        f"[{symbol}] Reached maximum expiry options limit ({max_expiry_options})"
                    )
                    break

                # Try all strike combinations for this expiry (prioritized by volume)
                found_valid_combination = False
                for call_strike, put_strike in valid_strike_pairs:
                    key = f"{expiry}_{call_strike}_{put_strike}"
                    if key in qualified_contracts_map:
                        contract_info = qualified_contracts_map[key]
                        expiry_option = ExpiryOption(
                            expiry=contract_info["expiry"],
                            call_contract=contract_info["call_contract"],
                            put_contract=contract_info["put_contract"],
                            call_strike=contract_info["call_strike"],
                            put_strike=contract_info["put_strike"],
                        )
                        expiry_options.append(expiry_option)
                        all_contracts.extend(
                            [
                                contract_info["call_contract"],
                                contract_info["put_contract"],
                            ]
                        )
                        found_valid_combination = True
                        logger.debug(
                            f"Using strike combination C{call_strike}/P{put_strike} for {symbol} expiry {expiry}"
                        )
                        break  # Found valid combination, move to next expiry

                if not found_valid_combination:
                    logger.debug(
                        f"No valid contract pair found for {symbol} expiry {expiry}"
                    )

            if not expiry_options:
                logger.info(f"No valid expiry options found for {symbol}")
                metrics_collector.add_rejection_reason(
                    RejectionReason.INVALID_CONTRACT_DATA,
                    {
                        "symbol": symbol,
                        "reason": "no valid expiry options found",
                        "total_expiries_checked": len(
                            self.filter_expirations_within_range(
                                chain.expirations, 19, 45
                            )
                        ),
                    },
                )
                return

            # Log the total number of combinations being scanned
            logger.info(
                f"[{symbol}] Scanning {len(expiry_options)} strike combinations across "
                f"{len(set(opt.expiry for opt in expiry_options))} expiries"
            )

            # Clean up any existing executor for this symbol before creating new one
            if symbol in self.active_executors:
                old_executor = self.active_executors[symbol]
                logger.info(
                    f"[{symbol}] Cleaning up existing executor before creating new one"
                )
                old_executor.deactivate()
                del self.active_executors[symbol]

            # Create single executor for this symbol
            srf_executor = SFRExecutor(
                ib=self.ib,
                order_manager=self.order_manager,
                stock_contract=stock,
                expiry_options=expiry_options,
                symbol=symbol,
                profit_target=self.profit_target,
                cost_limit=self.cost_limit,
                start_time=time.time(),
                quantity=quantity,
                data_timeout=45.0,  # Give more time for data collection
            )

            # Store executor and request market data for all contracts
            self.active_executors[symbol] = srf_executor

            # Clean up any stale data in contract_ticker for this symbol's contracts
            for contract in all_contracts:
                if (
                    hasattr(contract, "conId")
                    and contract.conId
                    and contract.conId in contract_ticker
                ):
                    del contract_ticker[contract.conId]
                    logger.debug(f"Cleaned up stale data for contract {contract.conId}")

            # Request market data for all contracts with detailed logging
            logger.info(
                f"[{symbol}] Requesting market data for {len(all_contracts)} contracts:"
            )

            # Log stock contract
            logger.info(f"  Stock: {stock.symbol} (conId: {stock.conId})")

            # Log option contracts
            for expiry_option in expiry_options:
                logger.info(
                    f"  Call: {expiry_option.call_contract.symbol} {expiry_option.call_strike} "
                    f"{expiry_option.expiry} (conId: {expiry_option.call_contract.conId})"
                )
                logger.info(
                    f"  Put: {expiry_option.put_contract.symbol} {expiry_option.put_strike} "
                    f"{expiry_option.expiry} (conId: {expiry_option.put_contract.conId})"
                )

            # Request market data for all contracts in parallel
            data_collection_start = time.time()
            await self.request_market_data_batch(all_contracts)
            data_collection_time = time.time() - data_collection_start

            # Record metrics
            metrics_collector.record_contracts_count(len(all_contracts))
            metrics_collector.record_data_collection_time(data_collection_time)
            metrics_collector.record_expiries_scanned(len(expiry_options))

            logger.info(
                f"Created executor for {symbol} with {len(expiry_options)} expiry options "
                f"({len(all_contracts)} total contracts)"
            )

            # Don't finish scan here - let the executor finish it when done processing
            # The executor will call metrics_collector.finish_scan() when it's inactive

        except Exception as e:
            logger.error(f"Error in scan_sfr for {symbol}: {str(e)}")
            metrics_collector.finish_scan(success=False, error_message=str(e))
