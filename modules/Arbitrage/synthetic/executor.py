"""
Execution logic for Synthetic arbitrage strategy.

This module contains the SynExecutor class responsible for:
- Market data handling and processing
- Opportunity calculation and evaluation
- Vectorized calculations for performance
- Global opportunity reporting
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from ib_async import IB, Contract, Event, Order, Ticker

from ..common import get_logger
from ..metrics import RejectionReason, metrics_collector
from ..pypy_config import optimizer as pypy_optimizer
from ..Strategy import BaseExecutor, OrderManagerClass
from .constants import (
    ADAPTIVE_TIMEOUT_MULTIPLIER,
    DEFAULT_DATA_TIMEOUT,
    MAX_BID_ASK_SPREAD,
)
from .data_collector import DataCollector, contract_ticker
from .global_opportunity_manager import GlobalOpportunityManager
from .models import ExpiryOption
from .validation import ValidationEngine

logger = get_logger()


class SynExecutor(BaseExecutor):
    """
    Synthetic not free risk (Syn) Executor class that handles the execution of Syn Synthetic option strategies.

    This class is responsible for:
    1. Monitoring market data for stock and options across multiple expiries
    2. Calculating potential arbitrage opportunities
    3. Reporting opportunities to the global manager (no longer executes directly)
    4. Logging trade details and results

    Attributes:
        ib (IB): Interactive Brokers connection instance
        order_manager (OrderManagerClass): Manager for handling order execution
        stock_contract (Contract): The underlying stock contract
        expiry_options (List[ExpiryOption]): List of option data for different expiries
        symbol (str): Trading symbol
        cost_limit (float): Maximum price limit for execution
        max_loss_threshold (float): Maximum loss for execution
        max_profit_threshold (float): Maximum profit for execution
        profit_ratio_threshold (float): Maximum profit to loss ratio for execution
        start_time (float): Start time of the execution
        quantity (int): Quantity of contracts to execute
        all_contracts (List[Contract]): All contracts (stock + all options)
        is_active (bool): Whether the executor is currently active
        data_timeout (float): Maximum time to wait for all contract data (seconds)
        global_manager (GlobalOpportunityManager): Manager for global opportunity collection
    """

    def __init__(
        self,
        ib: IB,
        order_manager: OrderManagerClass,
        stock_contract: Contract,
        expiry_options: List[ExpiryOption],
        symbol: str,
        cost_limit: float,
        max_loss_threshold: float,
        max_profit_threshold: float,
        profit_ratio_threshold: float,
        start_time: float,
        global_manager: GlobalOpportunityManager,
        quantity: int = 1,
        data_timeout: float = DEFAULT_DATA_TIMEOUT,
    ) -> None:
        """
        Initialize the Syn Executor.

        Args:
            ib: Interactive Brokers connection instance
            order_manager: Manager for handling order execution
            stock_contract: The underlying stock contract
            expiry_options: List of option data for different expiries
            symbol: Trading symbol
            cost_limit: Maximum price limit for execution
            max_loss_threshold: Maximum loss for execution
            max_profit_threshold: Maximum profit for execution
            profit_ratio_threshold: Maximum profit to loss ratio for execution
            start_time: Start time of the execution
            global_manager: Manager for global opportunity collection
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
        self.max_loss_threshold = max_loss_threshold
        self.max_profit_threshold = max_profit_threshold
        self.profit_ratio_threshold = profit_ratio_threshold
        self.quantity = quantity
        self.expiry_options = expiry_options
        self.all_contracts = [stock_contract] + option_contracts
        self.is_active = True
        self.data_timeout = data_timeout
        self.global_manager = global_manager

        # Initialize data collector and validation engine
        self.data_collector = DataCollector(symbol, data_timeout)
        self.validator = ValidationEngine(ib)

    def _get_ticker(self, conId):
        """Get ticker for this symbol's contract using composite key"""
        return self.data_collector.get_ticker(conId)

    def _set_ticker(self, conId, ticker):
        """Set ticker for this symbol's contract using composite key"""
        self.data_collector.set_ticker(conId, ticker)

    def _clear_symbol_tickers(self):
        """Clear all tickers for this symbol from global dictionary"""
        return self.data_collector.clear_symbol_tickers()

    def quick_viability_check(
        self, expiry_option: ExpiryOption, stock_price: float
    ) -> Tuple[bool, Optional[str]]:
        """Fast pre-filtering to eliminate non-viable opportunities early"""
        return self.validator.quick_viability_check(expiry_option, stock_price)

    def check_conditions(
        self,
        symbol: str,
        cost_limit: float,
        lmt_price: float,
        net_credit: float,
        min_roi: float,
        min_profit: float,
        max_profit: float,
    ) -> Tuple[bool, Optional[RejectionReason]]:
        """Check if opportunity meets all trading conditions"""
        return self.validator.check_conditions(
            symbol,
            cost_limit,
            lmt_price,
            net_credit,
            min_roi,
            min_profit,
            max_profit,
            self.max_loss_threshold,
            self.max_profit_threshold,
            self.profit_ratio_threshold,
        )

    async def executor(self, event: Event) -> None:
        """
        Main executor method that processes market data events for all contracts.
        This method is called once per symbol and handles all expiries for that symbol.

        PyPy Optimization: Cache frequently accessed attributes as local variables
        to reduce attribute lookup overhead in hot loops (~40% speedup with JIT).
        """
        if not self.is_active:
            return

        # PyPy optimization: Cache attributes as local variables for hot loop
        # This reduces attribute lookups significantly when JIT-compiled
        symbol = self.symbol
        data_collector = self.data_collector
        all_contracts = self.all_contracts
        is_active = self.is_active

        try:
            # Update contract_ticker with new data
            for tick in event:
                ticker: Ticker = tick
                contract = ticker.contract

                # Validate and set ticker data
                if data_collector.validate_ticker_data(ticker):
                    self._set_ticker(contract.conId, ticker)

            # Check for timeout
            timed_out, timeout_message = data_collector.check_data_timeout(
                all_contracts
            )
            if timed_out:
                logger.warning(timeout_message)
                self.is_active = False
                metrics_collector.finish_scan(
                    success=False, error_message="Data collection timeout"
                )
                return

            # Check if we have data for all contracts
            if data_collector.has_all_data(all_contracts):
                # Check if still active before proceeding
                if not self.is_active:
                    return

                logger.info(
                    f"[{self.symbol}] Fetched ticker for {len(self.all_contracts)} contracts"
                )
                execution_time = time.time() - self.start_time
                logger.info(f"time to execution: {execution_time} sec")
                metrics_collector.record_execution_time(execution_time)

                # Process all expiries and report opportunities to global manager
                opportunities_found = 0

                for expiry_option in self.expiry_options:
                    opportunity = self.calc_price_and_build_order_for_expiry(
                        expiry_option
                    )
                    if opportunity:
                        conversion_contract, order, _, trade_details = opportunity

                        # Get ticker data for scoring
                        call_ticker = self._get_ticker(
                            expiry_option.call_contract.conId
                        )
                        put_ticker = self._get_ticker(expiry_option.put_contract.conId)

                        if call_ticker and put_ticker:
                            # Report opportunity to global manager instead of executing
                            success = self.global_manager.add_opportunity(
                                symbol=self.symbol,
                                conversion_contract=conversion_contract,
                                order=order,
                                trade_details=trade_details,
                                call_ticker=call_ticker,
                                put_ticker=put_ticker,
                            )

                            if success:
                                opportunities_found += 1
                                max_profit = trade_details["max_profit"]
                                min_profit = trade_details["min_profit"]
                                risk_reward_ratio = (
                                    max_profit / abs(min_profit)
                                    if min_profit != 0
                                    else 0
                                )

                                logger.info(
                                    f"[{self.symbol}] Reported opportunity for expiry: {trade_details['expiry']} - "
                                    f"Risk-Reward Ratio: {risk_reward_ratio:.3f} "
                                    f"(max_profit: {max_profit:.2f}, min_profit: {min_profit:.2f})"
                                )

                # Log summary for this symbol
                if opportunities_found > 0:
                    logger.info(
                        f"[{self.symbol}] Reported {opportunities_found} opportunities to global manager"
                    )
                    metrics_collector.record_opportunity_found()
                else:
                    logger.info(f"[{self.symbol}] No suitable opportunities found")

                # Always finish scan successfully - global manager will handle execution
                self.is_active = False
                metrics_collector.finish_scan(success=True)

            else:
                # Still waiting for data from some contracts
                missing_contracts = self.data_collector.get_missing_contracts(
                    self.all_contracts
                )
                elapsed_time = time.time() - self.data_collector.data_collection_start
                # Only log debug message every 5 seconds to reduce noise
                if int(elapsed_time) % 5 == 0:
                    logger.debug(
                        f"[{self.symbol}] Still waiting for data from {len(missing_contracts)} contracts "
                        f"(elapsed: {elapsed_time:.1f}s)"
                    )

        except Exception as e:
            logger.error(f"Error in executor: {str(e)}")
            self.is_active = False
            # Finish scan with error
            metrics_collector.finish_scan(success=False, error_message=str(e))

    def calc_price_and_build_order_for_expiry(
        self, expiry_option: ExpiryOption
    ) -> Optional[Tuple[Contract, Order, float, Dict]]:
        """
        Calculate price and build order for a specific expiry option.
        Returns tuple of (contract, order, min_profit, trade_details) or None if no opportunity.
        """
        try:
            # Fast pre-filtering to eliminate non-viable opportunities early
            stock_ticker = self._get_ticker(self.stock_contract.conId)
            if not stock_ticker:
                return None

            stock_price = (
                stock_ticker.ask
                if not np.isnan(stock_ticker.ask)
                else stock_ticker.close
            )

            viable, reason = self.quick_viability_check(expiry_option, stock_price)
            if not viable:
                logger.debug(
                    f"[{self.symbol}] Quick rejection for {expiry_option.expiry}: {reason}"
                )
                return None

            # Get option data
            call_ticker = self._get_ticker(expiry_option.call_contract.conId)
            put_ticker = self._get_ticker(expiry_option.put_contract.conId)

            if not call_ticker or not put_ticker:
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

            # Get validated prices for individual legs
            call_price = (
                call_ticker.bid if not np.isnan(call_ticker.bid) else call_ticker.close
            )
            put_price = (
                put_ticker.ask if not np.isnan(put_ticker.ask) else put_ticker.close
            )

            if np.isnan(call_price) or np.isnan(put_price):
                metrics_collector.add_rejection_reason(
                    RejectionReason.INVALID_CONTRACT_DATA,
                    {
                        "symbol": self.symbol,
                        "contract_type": "options",
                        "expiry": expiry_option.expiry,
                        "call_strike": expiry_option.call_strike,
                        "put_strike": expiry_option.put_strike,
                        "call_price_invalid": np.isnan(call_price),
                        "put_price_invalid": np.isnan(put_price),
                    },
                )
                return None

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

            if call_bid_ask_spread > MAX_BID_ASK_SPREAD:
                logger.info(
                    f"[{self.symbol}] Call contract bid-ask spread too wide: {call_bid_ask_spread:.2f} > {MAX_BID_ASK_SPREAD:.2f}, "
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
                        "threshold": MAX_BID_ASK_SPREAD,
                    },
                )
                return None

            if put_bid_ask_spread > MAX_BID_ASK_SPREAD:
                logger.info(
                    f"[{self.symbol}] Put contract bid-ask spread too wide: {put_bid_ask_spread:.2f} > {MAX_BID_ASK_SPREAD:.2f}, "
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
                        "threshold": MAX_BID_ASK_SPREAD,
                    },
                )
                return None

            # Calculate net credit
            net_credit = call_price - put_price
            stock_price = round(stock_price, 2)
            net_credit = round(net_credit, 2)
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

            spread = stock_price - expiry_option.put_strike
            min_profit = net_credit - spread  # max loss
            max_profit = (
                expiry_option.call_strike - expiry_option.put_strike
            ) + min_profit
            min_roi = (min_profit / (stock_price + net_credit)) * 100

            # Calculate precise combo limit price based on target leg prices
            combo_limit_price = self.calculate_combo_limit_price(
                stock_price=stock_price,
                call_price=call_price,
                put_price=put_price,
                buffer_percent=0.00,  # No buffer for combo limit price
            )

            logger.info(
                f"[{self.symbol}] Expiry: {expiry_option.expiry} min_profit:{min_profit:.2f}, max_profit:{max_profit:.2f}, min_roi:{min_roi:.2f}%"
            )

            conditions_met, rejection_reason = self.check_conditions(
                self.symbol,
                self.cost_limit,
                combo_limit_price,  # Use calculated precise limit price
                net_credit,
                min_roi,
                min_profit,
                max_profit,
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
                            "max_loss_threshold": self.max_loss_threshold,
                            "max_profit_threshold": self.max_profit_threshold,
                            "profit_ratio_threshold": self.profit_ratio_threshold,
                            "spread": spread,
                            "profit_ratio": profit_ratio,
                        },
                    )

            return None
        except Exception as e:
            logger.error(f"Error in calc_price_and_build_order_for_expiry: {str(e)}")
            return None

    def calculate_all_opportunities_vectorized(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Calculate all synthetic arbitrage opportunities in parallel using NumPy.
        Returns arrays of profits and metadata for all expiry options.
        """
        # Gather all data into NumPy arrays
        num_options = len(self.expiry_options)

        # Pre-allocate arrays for all data
        call_bids = np.zeros(num_options)
        call_asks = np.zeros(num_options)
        put_bids = np.zeros(num_options)
        put_asks = np.zeros(num_options)
        call_strikes = np.zeros(num_options)
        put_strikes = np.zeros(num_options)
        stock_bids = np.zeros(num_options)
        stock_asks = np.zeros(num_options)

        # Populate arrays (this is the only loop needed)
        valid_mask = np.zeros(num_options, dtype=bool)

        for i, expiry_option in enumerate(self.expiry_options):
            call_ticker = self._get_ticker(expiry_option.call_contract.conId)
            put_ticker = self._get_ticker(expiry_option.put_contract.conId)
            stock_ticker = self._get_ticker(self.stock_contract.conId)

            if call_ticker and put_ticker and stock_ticker:
                # Check data validity
                if (
                    hasattr(call_ticker, "bid")
                    and call_ticker.bid > 0
                    and hasattr(call_ticker, "ask")
                    and call_ticker.ask > 0
                    and hasattr(put_ticker, "bid")
                    and put_ticker.bid > 0
                    and hasattr(put_ticker, "ask")
                    and put_ticker.ask > 0
                ):

                    call_bids[i] = call_ticker.bid
                    call_asks[i] = call_ticker.ask
                    put_bids[i] = put_ticker.bid
                    put_asks[i] = put_ticker.ask
                    call_strikes[i] = expiry_option.call_strike
                    put_strikes[i] = expiry_option.put_strike
                    stock_bids[i] = (
                        stock_ticker.bid if stock_ticker.bid > 0 else stock_ticker.last
                    )
                    stock_asks[i] = (
                        stock_ticker.ask if stock_ticker.ask > 0 else stock_ticker.last
                    )
                    valid_mask[i] = True

        # VECTORIZED CALCULATIONS - All opportunities calculated at once!
        # For synthetic: net_credit = call_price - put_price
        # min_profit = net_credit - (stock_price - put_strike) [this is max loss]
        # max_profit = (call_strike - put_strike) + min_profit

        # Execution prices (what we actually get/pay)
        exec_net_credits = (
            call_bids - put_asks
        )  # We sell call (get bid), buy put (pay ask)
        exec_stock_prices = stock_asks  # We buy stock (pay ask)
        exec_spreads = exec_stock_prices - put_strikes  # Stock price - put strike

        # Calculate min profits (actually max losses, but kept as min_profit for consistency)
        min_profits = exec_net_credits - exec_spreads

        # Calculate max profits
        strike_spreads = call_strikes - put_strikes
        max_profits = strike_spreads + min_profits

        # Apply validity mask
        min_profits[~valid_mask] = -np.inf
        max_profits[~valid_mask] = -np.inf

        return (
            min_profits,  # Actually max losses
            max_profits,
            {
                "call_bids": call_bids,
                "call_asks": call_asks,
                "put_bids": put_bids,
                "put_asks": put_asks,
                "call_strikes": call_strikes,
                "put_strikes": put_strikes,
                "stock_bids": stock_bids,
                "stock_asks": stock_asks,
                "valid_mask": valid_mask,
                "net_credits": exec_net_credits,
                "spreads": exec_spreads,
            },
        )

    async def evaluate_with_available_data_vectorized(self) -> Optional[Dict]:
        """
        Vectorized evaluation of all synthetic opportunities at once.
        Similar to SFR but adapted for synthetic strategy parameters.
        """
        logger.info(
            f"[{self.symbol}] Starting vectorized synthetic evaluation with {len(self.expiry_options)} options"
        )

        # Step 1: Calculate all opportunities in parallel
        min_profits, max_profits, market_data = (
            self.calculate_all_opportunities_vectorized()
        )

        # Step 2: Apply filters based on synthetic strategy thresholds
        profit_ratios = np.where(min_profits != 0, max_profits / np.abs(min_profits), 0)

        # Apply synthetic-specific filters
        viable_mask = (
            market_data["valid_mask"]
            & (
                min_profits >= -abs(self.max_loss_threshold)
                if self.max_loss_threshold is not None
                else True
            )
            & (
                max_profits <= self.max_profit_threshold
                if self.max_profit_threshold is not None
                else True
            )
            & (
                profit_ratios >= self.profit_ratio_threshold
                if self.profit_ratio_threshold is not None
                else True
            )
            & (market_data["net_credits"] >= 0)  # Positive net credit
        )

        # Step 3: Find best opportunity
        if not np.any(viable_mask):
            logger.info(
                f"[{self.symbol}] No viable synthetic opportunities found after vectorized evaluation"
            )
            return None

        # Create composite score (higher is better for synthetic)
        # Focus on profit ratio and credit quality
        composite_scores = np.zeros(len(self.expiry_options))
        composite_scores[viable_mask] = (
            profit_ratios[viable_mask] * 0.6  # Profit ratio weight
            + (
                market_data["net_credits"][viable_mask]
                / np.mean(market_data["net_credits"][viable_mask])
            )
            * 0.4  # Credit quality
        )

        best_idx = np.argmax(composite_scores)
        best_opportunity = self.expiry_options[best_idx]

        logger.info(
            f"[{self.symbol}] Best synthetic opportunity found: "
            f"Expiry {best_opportunity.expiry}, "
            f"Min profit: ${min_profits[best_idx]:.2f}, "
            f"Max profit: ${max_profits[best_idx]:.2f}, "
            f"Profit ratio: {profit_ratios[best_idx]:.3f}"
        )

        # Build the order for the best opportunity
        combo_limit_price = self.calculate_combo_limit_price(
            stock_price=market_data["stock_asks"][best_idx],
            call_price=market_data["call_bids"][best_idx],
            put_price=market_data["put_asks"][best_idx],
            buffer_percent=0.01,
        )

        conversion_contract, order = self.build_order(
            self.symbol,
            self.stock_contract,
            best_opportunity.call_contract,
            best_opportunity.put_contract,
            combo_limit_price,
            self.quantity,
            call_price=market_data["call_bids"][best_idx],
            put_price=market_data["put_asks"][best_idx],
        )

        return {
            "contract": conversion_contract,
            "order": order,
            "min_profit": min_profits[best_idx],
            "trade_details": {
                "expiry": best_opportunity.expiry,
                "call_strike": best_opportunity.call_strike,
                "put_strike": best_opportunity.put_strike,
                "call_price": market_data["call_bids"][best_idx],
                "put_price": market_data["put_asks"][best_idx],
                "stock_price": market_data["stock_asks"][best_idx],
                "net_credit": market_data["net_credits"][best_idx],
                "min_profit": min_profits[best_idx],
                "max_profit": max_profits[best_idx],
                "profit_ratio": profit_ratios[best_idx],
            },
            "expiry_option": best_opportunity,
        }

    def deactivate(self):
        """Deactivate the executor"""
        self.is_active = False
