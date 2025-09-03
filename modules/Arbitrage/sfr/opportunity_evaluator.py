"""
Opportunity evaluation and analysis for SFR arbitrage strategy.

This module contains the complex logic for evaluating arbitrage opportunities,
including vectorized calculations, spread analysis, and profit optimization.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from ib_async import Contract, Order

from ..common import get_logger
from ..data_collection_metrics import ContractPriority
from ..metrics import RejectionReason, metrics_collector
from .constants import (
    MAX_ACCEPTABLE_SPREAD_PCT,
    MAX_BID_ASK_SPREAD,
    MAX_TOTAL_SPREAD_COST,
    MIN_GUARANTEED_PROFIT,
    MIN_THEORETICAL_PROFIT,
    OUTLIER_PENALTY,
    OUTLIER_Z_SCORE_THRESHOLD,
)
from .models import (
    ExpiryOption,
    OpportunityTuple,
    PricingData,
    SpreadAnalysis,
    VectorizedOpportunityData,
)
from .utils import calculate_combo_limit_price, calculate_z_scores, get_stock_midpoint
from .validation import (
    ConditionsValidator,
    DataQualityValidator,
    OpportunityValidator,
    PriceValidator,
)

logger = get_logger()


class OpportunityCalculator:
    """Handles individual opportunity calculations"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.opportunity_validator = OpportunityValidator()
        self.data_quality_validator = DataQualityValidator()
        self.conditions_validator = ConditionsValidator()
        self.price_validator = PriceValidator()

    def calculate_pricing_data(
        self,
        stock_ticker,
        call_ticker,
        put_ticker,
        expiry_option: ExpiryOption,
    ) -> Optional[PricingData]:
        """Calculate comprehensive pricing data for opportunity evaluation"""

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
                put_ticker.last if not np.isnan(put_ticker.last) else put_ticker.close
            )

        if np.isnan(call_fair) or np.isnan(put_fair):
            return None

        # STAGE 2: Execution validation using actual prices
        # These are the prices we'll actually pay/receive
        call_exec = (
            call_ticker.bid if not np.isnan(call_ticker.bid) else call_ticker.close
        )
        put_exec = put_ticker.ask if not np.isnan(put_ticker.ask) else put_ticker.close
        stock_exec = (
            stock_ticker.ask if not np.isnan(stock_ticker.ask) else stock_ticker.close
        )

        if np.isnan(call_exec) or np.isnan(put_exec) or np.isnan(stock_exec):
            return None

        # Calculate theoretical arbitrage with fair values
        theoretical_net_credit = call_fair - put_fair
        theoretical_spread = stock_fair - expiry_option.put_strike
        theoretical_profit = theoretical_net_credit - theoretical_spread

        # Calculate guaranteed profit with execution prices
        guaranteed_net_credit = call_exec - put_exec
        guaranteed_spread = stock_exec - expiry_option.put_strike
        guaranteed_profit = guaranteed_net_credit - guaranteed_spread

        return PricingData(
            stock_fair=stock_fair,
            stock_exec=stock_exec,
            call_fair=call_fair,
            call_exec=call_exec,
            put_fair=put_fair,
            put_exec=put_exec,
            theoretical_net_credit=theoretical_net_credit,
            theoretical_spread=theoretical_spread,
            theoretical_profit=theoretical_profit,
            guaranteed_net_credit=guaranteed_net_credit,
            guaranteed_spread=guaranteed_spread,
            guaranteed_profit=guaranteed_profit,
        )

    def validate_bid_ask_spreads(
        self, call_ticker, put_ticker, expiry_option: ExpiryOption
    ) -> bool:
        """Validate bid-ask spreads are within acceptable ranges"""
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
                f"[{self.symbol}] Call contract bid-ask spread too wide: {call_bid_ask_spread:.2f} > {MAX_BID_ASK_SPREAD}, "
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
            return False

        if put_bid_ask_spread > MAX_BID_ASK_SPREAD:
            logger.info(
                f"[{self.symbol}] Put contract bid-ask spread too wide: {put_bid_ask_spread:.2f} > {MAX_BID_ASK_SPREAD}, "
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
            return False

        return True


class VectorizedOpportunityEvaluator:
    """Handles vectorized evaluation of multiple opportunities simultaneously"""

    def __init__(
        self, symbol: str, expiry_options: List[ExpiryOption], ticker_getter_func
    ):
        self.symbol = symbol
        self.expiry_options = expiry_options
        self.get_ticker = ticker_getter_func

    def calculate_all_opportunities_vectorized(self) -> VectorizedOpportunityData:
        """
        Calculate all arbitrage opportunities in parallel using NumPy.
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

        # Get stock ticker once (should be same for all options)
        stock_contract = next(
            (
                c
                for c in [opt.call_contract for opt in self.expiry_options]
                if hasattr(c, "symbol")
            ),
            None,
        )
        if stock_contract:
            stock_ticker = self.get_ticker(stock_contract.conId)
        else:
            stock_ticker = None

        for i, expiry_option in enumerate(self.expiry_options):
            call_ticker = self.get_ticker(expiry_option.call_contract.conId)
            put_ticker = self.get_ticker(expiry_option.put_contract.conId)

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
        # Theoretical profits (using midpoints)
        call_mids = (call_bids + call_asks) / 2
        put_mids = (put_bids + put_asks) / 2
        stock_mids = (stock_bids + stock_asks) / 2

        theoretical_net_credits = call_mids - put_mids
        theoretical_spreads = stock_mids - put_strikes
        theoretical_profits = theoretical_net_credits - theoretical_spreads

        # Guaranteed profits (using execution prices)
        guaranteed_net_credits = call_bids - put_asks  # What we actually get
        guaranteed_spreads = stock_asks - put_strikes  # What we actually pay
        guaranteed_profits = guaranteed_net_credits - guaranteed_spreads

        # Apply validity mask
        theoretical_profits[~valid_mask] = -np.inf
        guaranteed_profits[~valid_mask] = -np.inf

        market_data = {
            "call_bids": call_bids,
            "call_asks": call_asks,
            "put_bids": put_bids,
            "put_asks": put_asks,
            "call_strikes": call_strikes,
            "put_strikes": put_strikes,
            "stock_bids": stock_bids,
            "stock_asks": stock_asks,
            "valid_mask": valid_mask,
        }

        return VectorizedOpportunityData(
            theoretical_profits=theoretical_profits,
            guaranteed_profits=guaranteed_profits,
            market_data=market_data,
            viable_mask=valid_mask,
            spread_stats={},
        )

    def analyze_spreads_vectorized(self, market_data: dict) -> Tuple[np.ndarray, dict]:
        """
        Perform statistical analysis on bid-ask spreads to filter opportunities.
        Returns a mask of viable opportunities and spread statistics.
        """
        # Calculate all spreads at once
        call_spreads = market_data["call_asks"] - market_data["call_bids"]
        put_spreads = market_data["put_asks"] - market_data["put_bids"]
        stock_spreads = market_data["stock_asks"] - market_data["stock_bids"]

        # Calculate spread as percentage of midpoint
        call_mids = (market_data["call_asks"] + market_data["call_bids"]) / 2
        put_mids = (market_data["put_asks"] + market_data["put_bids"]) / 2
        stock_mids = (market_data["stock_asks"] + market_data["stock_bids"]) / 2

        # Avoid division by zero
        call_mids = np.where(call_mids > 0, call_mids, 1)
        put_mids = np.where(put_mids > 0, put_mids, 1)
        stock_mids = np.where(stock_mids > 0, stock_mids, 1)

        call_spread_pct = call_spreads / call_mids
        put_spread_pct = put_spreads / put_mids
        stock_spread_pct = stock_spreads / stock_mids

        # Statistical analysis
        # Calculate z-scores for outlier detection
        call_z_scores = calculate_z_scores(call_spread_pct[market_data["valid_mask"]])
        put_z_scores = calculate_z_scores(put_spread_pct[market_data["valid_mask"]])

        # Create quality scores based on spreads
        # Lower spreads = higher quality
        max_acceptable_spread_pct = MAX_ACCEPTABLE_SPREAD_PCT

        spread_quality_scores = np.ones(len(call_spreads))

        # Penalize wide spreads
        spread_quality_scores -= (
            np.clip(call_spread_pct / max_acceptable_spread_pct, 0, 1) * 0.3
        )
        spread_quality_scores -= (
            np.clip(put_spread_pct / max_acceptable_spread_pct, 0, 1) * 0.3
        )
        spread_quality_scores -= (
            np.clip(stock_spread_pct / max_acceptable_spread_pct, 0, 1) * 0.2
        )

        # Penalize outliers (z-score > 2)
        outlier_penalty = OUTLIER_PENALTY
        full_call_z = np.zeros(len(call_spreads))
        full_put_z = np.zeros(len(put_spreads))

        valid_indices = np.where(market_data["valid_mask"])[0]
        for idx, z_idx in enumerate(valid_indices):
            if idx < len(call_z_scores):
                full_call_z[z_idx] = call_z_scores[idx]
            if idx < len(put_z_scores):
                full_put_z[z_idx] = put_z_scores[idx]

        spread_quality_scores[
            np.abs(full_call_z) > OUTLIER_Z_SCORE_THRESHOLD
        ] -= outlier_penalty
        spread_quality_scores[
            np.abs(full_put_z) > OUTLIER_Z_SCORE_THRESHOLD
        ] -= outlier_penalty

        # Calculate execution cost impact
        # This estimates how much profit we lose to spreads
        total_spread_cost = call_spreads + put_spreads + stock_spreads

        # Create viability mask
        viable_mask = (
            market_data["valid_mask"]  # Has data
            & (spread_quality_scores > 0.5)  # Decent spread quality
            & (call_spread_pct < max_acceptable_spread_pct)  # Call spread acceptable
            & (put_spread_pct < max_acceptable_spread_pct)  # Put spread acceptable
            & (total_spread_cost < MAX_TOTAL_SPREAD_COST)  # Total spread cost < $5
        )

        spread_stats = {
            "mean_call_spread": (
                np.mean(call_spreads[market_data["valid_mask"]])
                if np.any(market_data["valid_mask"])
                else 0
            ),
            "mean_put_spread": (
                np.mean(put_spreads[market_data["valid_mask"]])
                if np.any(market_data["valid_mask"])
                else 0
            ),
            "median_call_spread": (
                np.median(call_spreads[market_data["valid_mask"]])
                if np.any(market_data["valid_mask"])
                else 0
            ),
            "median_put_spread": (
                np.median(put_spreads[market_data["valid_mask"]])
                if np.any(market_data["valid_mask"])
                else 0
            ),
            "spread_quality_scores": spread_quality_scores,
            "total_spread_costs": total_spread_cost,
            "viable_count": np.sum(viable_mask),
            "rejected_by_spread": np.sum(market_data["valid_mask"] & ~viable_mask),
        }

        logger.info(
            f"[{self.symbol}] Spread analysis: {spread_stats['viable_count']} viable out of "
            f"{np.sum(market_data['valid_mask'])} with data "
            f"({spread_stats['rejected_by_spread']} rejected by spreads)"
        )

        return viable_mask, spread_stats


class OpportunityEvaluator:
    """Main opportunity evaluator that coordinates all evaluation components"""

    def __init__(
        self,
        symbol: str,
        expiry_options: List[ExpiryOption],
        ticker_getter_func,
        check_conditions_func=None,
    ):
        self.symbol = symbol
        self.expiry_options = expiry_options
        self.get_ticker = ticker_getter_func
        self.check_conditions_func = check_conditions_func

        self.calculator = OpportunityCalculator(symbol)
        self.vectorized_evaluator = VectorizedOpportunityEvaluator(
            symbol, expiry_options, ticker_getter_func
        )
        if check_conditions_func is None:
            self.conditions_validator = ConditionsValidator()
        else:
            self.conditions_validator = None  # Will use the provided function

    def calc_price_and_build_order_for_expiry(
        self,
        expiry_option: ExpiryOption,
        stock_contract: Contract,
        profit_target: float,
        cost_limit: float,
        quantity: int,
        build_order_func,
        priority_filter: Optional[ContractPriority] = None,
    ) -> Optional[OpportunityTuple]:
        """
        Calculate price and build order for a specific expiry option.
        Priority filter allows evaluation with partial data - only evaluates contracts
        of specified priority or higher.
        Returns tuple of (contract, order, min_profit, trade_details) or None if no opportunity.
        """
        try:
            # Track entry into evaluation funnel
            logger.info(
                f"[Funnel] [{self.symbol}] Stage: evaluated (expiry: {expiry_option.expiry})"
            )
            metrics_collector.record_funnel_stage(
                self.symbol, expiry_option.expiry, "evaluated"
            )

            # Fast pre-filtering to eliminate non-viable opportunities early
            stock_ticker = self.get_ticker(stock_contract.conId)
            if not stock_ticker:
                metrics_collector.add_rejection_reason(
                    RejectionReason.MISSING_MARKET_DATA,
                    {
                        "symbol": self.symbol,
                        "contract_type": "stock",
                        "expiry": expiry_option.expiry,
                        "stage": "stock_ticker_check",
                    },
                )
                return None

            # Track stock ticker availability
            logger.info(
                f"[Funnel] [{self.symbol}] Stage: stock_ticker_available (expiry: {expiry_option.expiry})"
            )
            metrics_collector.record_funnel_stage(
                self.symbol, expiry_option.expiry, "stock_ticker_available"
            )

            # Get stock price for viability check
            stock_fair = stock_ticker.midpoint()
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

            viable, reason = (
                self.calculator.opportunity_validator.quick_viability_check(
                    expiry_option, stock_fair
                )
            )
            if not viable:
                logger.debug(
                    f"[{self.symbol}] Quick rejection for {expiry_option.expiry}: {reason}"
                )
                # Map viability reasons to appropriate rejection reasons
                rejection_reason_map = {
                    "invalid_strike_spread": RejectionReason.INVALID_STRIKE_COMBINATION,
                    "expiry_out_of_range": RejectionReason.NO_VALID_EXPIRIES,
                    "invalid_expiry_format": RejectionReason.INVALID_CONTRACT_DATA,
                    "invalid_stock_price": RejectionReason.INVALID_CONTRACT_DATA,
                    "poor_moneyness": RejectionReason.INVALID_STRIKE_COMBINATION,
                }
                rejection_reason = rejection_reason_map.get(
                    reason, RejectionReason.INVALID_CONTRACT_DATA
                )
                metrics_collector.add_rejection_reason(
                    rejection_reason,
                    {
                        "symbol": self.symbol,
                        "expiry": expiry_option.expiry,
                        "viability_reason": reason,
                        "stage": "quick_viability_check",
                    },
                )
                return None

            # Track passing viability check
            logger.info(
                f"[Funnel] [{self.symbol}] Stage: passed_viability_check (expiry: {expiry_option.expiry})"
            )
            metrics_collector.record_funnel_stage(
                self.symbol, expiry_option.expiry, "passed_viability_check"
            )

            # Get option data
            call_ticker = self.get_ticker(expiry_option.call_contract.conId)
            put_ticker = self.get_ticker(expiry_option.put_contract.conId)

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
            logger.info(
                f"[Funnel] [{self.symbol}] Stage: option_data_available (expiry: {expiry_option.expiry})"
            )
            metrics_collector.record_funnel_stage(
                self.symbol, expiry_option.expiry, "option_data_available"
            )

            # Calculate data quality score for better decision making in partial mode
            data_quality_score = (
                self.calculator.data_quality_validator.calculate_data_quality_score(
                    stock_ticker, call_ticker, put_ticker
                )
            )

            # In partial data mode, require higher quality for execution
            min_quality_threshold = 0.8 if priority_filter else 0.6
            if data_quality_score < min_quality_threshold:
                logger.info(
                    f"[{self.symbol}] Data quality {data_quality_score:.2f} below threshold "
                    f"{min_quality_threshold} for {expiry_option.expiry} "
                    f"(call_strike={expiry_option.call_strike}, put_strike={expiry_option.put_strike})"
                )
                return None

            # Track passing data quality check
            logger.info(
                f"[Funnel] [{self.symbol}] Stage: passed_data_quality (expiry: {expiry_option.expiry})"
            )
            metrics_collector.record_funnel_stage(
                self.symbol, expiry_option.expiry, "passed_data_quality"
            )

            # Validate bid-ask spreads
            if not self.calculator.validate_bid_ask_spreads(
                call_ticker, put_ticker, expiry_option
            ):
                return None

            # Calculate comprehensive pricing data
            pricing_data = self.calculator.calculate_pricing_data(
                stock_ticker, call_ticker, put_ticker, expiry_option
            )

            if not pricing_data:
                metrics_collector.add_rejection_reason(
                    RejectionReason.INVALID_CONTRACT_DATA,
                    {
                        "symbol": self.symbol,
                        "contract_type": "pricing",
                        "expiry": expiry_option.expiry,
                        "call_strike": expiry_option.call_strike,
                        "put_strike": expiry_option.put_strike,
                    },
                )
                return None

            # Track ALL theoretical profit calculations (positive and negative)
            metrics_collector.record_profit_calculation(
                self.symbol, expiry_option.expiry, pricing_data.theoretical_profit
            )

            # Quick reject if no theoretical opportunity
            if pricing_data.theoretical_profit < MIN_THEORETICAL_PROFIT:
                logger.warning(
                    f"[{self.symbol}] No theoretical arbitrage for {expiry_option.expiry}: "
                    f"profit=${pricing_data.theoretical_profit:.2f}"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.ARBITRAGE_CONDITION_NOT_MET,
                    {
                        "symbol": self.symbol,
                        "theoretical_profit": pricing_data.theoretical_profit,
                        "theoretical_net_credit": pricing_data.theoretical_net_credit,
                        "theoretical_spread": pricing_data.theoretical_spread,
                        "stage": "theoretical_evaluation",
                    },
                )
                return None

            # Track positive theoretical profit
            logger.info(
                f"[Funnel] [{self.symbol}] Stage: theoretical_profit_positive (expiry: {expiry_option.expiry}, profit: ${pricing_data.theoretical_profit:.2f})"
            )
            metrics_collector.record_funnel_stage(
                self.symbol, expiry_option.expiry, "theoretical_profit_positive"
            )

            # Must have guaranteed profit after execution
            if pricing_data.guaranteed_profit < MIN_GUARANTEED_PROFIT:
                logger.info(
                    f"[{self.symbol}] Theoretical profit ${pricing_data.theoretical_profit:.2f} "
                    f"but guaranteed only ${pricing_data.guaranteed_profit:.2f} - rejecting"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.ARBITRAGE_CONDITION_NOT_MET,
                    {
                        "symbol": self.symbol,
                        "theoretical_profit": pricing_data.theoretical_profit,
                        "guaranteed_profit": pricing_data.guaranteed_profit,
                        "stock_fair": pricing_data.stock_fair,
                        "stock_exec": pricing_data.stock_exec,
                        "stage": "execution_validation",
                    },
                )
                return None

            # Track positive guaranteed profit
            logger.info(
                f"[Funnel] [{self.symbol}] Stage: guaranteed_profit_positive (expiry: {expiry_option.expiry}, profit: ${pricing_data.guaranteed_profit:.2f})"
            )
            metrics_collector.record_funnel_stage(
                self.symbol, expiry_option.expiry, "guaranteed_profit_positive"
            )

            # Validate strike ordering
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

            # Calculate final trade parameters
            min_profit = (
                pricing_data.guaranteed_profit
            )  # Already calculated and validated
            max_profit = (
                expiry_option.call_strike - expiry_option.put_strike
            ) + pricing_data.guaranteed_net_credit
            min_roi = (
                (
                    min_profit
                    / (pricing_data.stock_exec + pricing_data.guaranteed_net_credit)
                )
                * 100
                if (pricing_data.stock_exec + pricing_data.guaranteed_net_credit) > 0
                else 0
            )

            # Calculate precise combo limit price based on target leg prices
            combo_limit_price = calculate_combo_limit_price(
                stock_price=pricing_data.stock_exec,
                call_price=pricing_data.call_exec,
                put_price=pricing_data.put_exec,
                buffer_percent=0.01,  # 1% buffer for realistic execution
            )

            # Comprehensive logging for debugging contract and profit issues
            logger.info(f"[{self.symbol}] COMPREHENSIVE PRICING ANALYSIS:")
            logger.info(f"  Expiry: {expiry_option.expiry}")
            logger.info(
                f"  Contract Details - Call Strike: {expiry_option.call_strike}, Put Strike: {expiry_option.put_strike}"
            )
            logger.info(
                f"  Contract IDs - Call: {expiry_option.call_contract.conId}, Put: {expiry_option.put_contract.conId}"
            )
            logger.info(
                f"  Execution Prices - Stock: ${pricing_data.stock_exec:.2f}, Call: ${pricing_data.call_exec:.2f}, Put: ${pricing_data.put_exec:.2f}"
            )
            logger.info(
                f"  Fair Values - Stock: ${pricing_data.stock_fair:.2f}, Call: ${pricing_data.call_fair:.2f}, Put: ${pricing_data.put_fair:.2f}"
            )
            logger.info(
                f"  Theoretical - Net Credit: ${pricing_data.theoretical_net_credit:.2f}, Spread: ${pricing_data.theoretical_spread:.2f}, Profit: ${pricing_data.theoretical_profit:.2f}"
            )
            logger.info(
                f"  Guaranteed - Net Credit: ${pricing_data.guaranteed_net_credit:.2f}, Spread: ${pricing_data.guaranteed_spread:.2f}, Profit: ${pricing_data.guaranteed_profit:.2f}"
            )
            logger.info(
                f"  Final Metrics - Min Profit: ${min_profit:.2f}, Max Profit: ${max_profit:.2f}, Min ROI: {min_roi:.2f}%"
            )
            logger.info(f"  Combo Limit Price: ${combo_limit_price:.2f}")

            # Use provided check_conditions function or default validator
            if self.check_conditions_func:
                conditions_met, rejection_reason = self.check_conditions_func(
                    self.symbol,
                    profit_target,
                    cost_limit,
                    expiry_option.put_strike,
                    combo_limit_price,  # Use calculated precise limit price
                    pricing_data.guaranteed_net_credit,
                    min_roi,
                    pricing_data.stock_exec,
                    min_profit,
                )
            else:
                conditions_met, rejection_reason = (
                    self.conditions_validator.check_conditions(
                        self.symbol,
                        profit_target,
                        cost_limit,
                        expiry_option.put_strike,
                        combo_limit_price,  # Use calculated precise limit price
                        pricing_data.guaranteed_net_credit,
                        min_roi,
                        pricing_data.stock_exec,
                        min_profit,
                    )
                )

            if conditions_met:
                # Build order with precise limit price and target leg prices
                conversion_contract, order = build_order_func(
                    self.symbol,
                    stock_contract,
                    expiry_option.call_contract,
                    expiry_option.put_contract,
                    combo_limit_price,  # Use calculated precise limit price
                    quantity,
                    call_price=pricing_data.call_exec,  # Target call leg price
                    put_price=pricing_data.put_exec,  # Target put leg price
                )

                # Prepare trade details for logging (don't log yet)
                trade_details = {
                    "call_strike": expiry_option.call_strike,
                    "call_price": pricing_data.call_exec,
                    "put_strike": expiry_option.put_strike,
                    "put_price": pricing_data.put_exec,
                    "stock_price": pricing_data.stock_exec,
                    "net_credit": pricing_data.guaranteed_net_credit,
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
                            "stock_price": pricing_data.stock_exec,
                            "net_credit": pricing_data.guaranteed_net_credit,
                            "min_profit": min_profit,
                            "max_profit": max_profit,
                            "min_roi": min_roi,
                            "combo_limit_price": combo_limit_price,
                            "cost_limit": cost_limit,
                            "profit_target": profit_target,
                            "spread": pricing_data.guaranteed_spread,
                            "profit_ratio": profit_ratio,
                        },
                    )

            return None
        except Exception as e:
            logger.error(f"Error in calc_price_and_build_order_for_expiry: {str(e)}")
            return None

    async def evaluate_with_available_data_vectorized(
        self, max_priority: ContractPriority
    ) -> Optional[Dict]:
        """
        Vectorized evaluation of all opportunities at once.
        10-100x faster than sequential evaluation.
        """
        logger.info(
            f"[{self.symbol}] Starting vectorized evaluation with {len(self.expiry_options)} options, max_priority={max_priority.value}"
        )

        # Step 1: Calculate all opportunities in parallel
        vectorized_data = (
            self.vectorized_evaluator.calculate_all_opportunities_vectorized()
        )

        # Step 2: Apply spread analysis and filtering
        viable_mask, spread_stats = (
            self.vectorized_evaluator.analyze_spreads_vectorized(
                vectorized_data.market_data
            )
        )

        # Update vectorized data with results
        vectorized_data.viable_mask = viable_mask
        vectorized_data.spread_stats = spread_stats

        # Step 3: Apply additional filters
        # Combine all filters
        profitable_mask = (
            viable_mask
            & (vectorized_data.theoretical_profits >= MIN_THEORETICAL_PROFIT)
            & (vectorized_data.guaranteed_profits >= MIN_GUARANTEED_PROFIT)
        )

        # Step 4: Rank opportunities by profit potential
        # Create composite score considering both profit and spread quality
        profit_scores = vectorized_data.guaranteed_profits.copy()
        profit_scores[~profitable_mask] = -np.inf

        # Adjust scores by spread quality
        if "spread_quality_scores" in spread_stats:
            profit_scores = profit_scores * spread_stats["spread_quality_scores"]

        # Step 5: Find best opportunity
        if np.all(profit_scores == -np.inf):
            logger.info(
                f"[{self.symbol}] No profitable opportunities found after vectorized evaluation"
            )
            return None

        best_idx = np.argmax(profit_scores)
        best_profit = vectorized_data.guaranteed_profits[best_idx]

        logger.info(
            f"[{self.symbol}] Best opportunity found: "
            f"Expiry {self.expiry_options[best_idx].expiry}, "
            f"Guaranteed profit: ${best_profit:.2f}, "
            f"Theoretical profit: ${vectorized_data.theoretical_profits[best_idx]:.2f}"
        )

        # Log rejection statistics
        total_evaluated = len(self.expiry_options)
        with_data = np.sum(vectorized_data.market_data["valid_mask"])
        theoretically_profitable = np.sum(
            vectorized_data.theoretical_profits >= MIN_THEORETICAL_PROFIT
        )
        guaranteed_profitable = np.sum(
            vectorized_data.guaranteed_profits >= MIN_GUARANTEED_PROFIT
        )
        after_spread_filter = np.sum(viable_mask)

        logger.info(
            f"[{self.symbol}] Funnel: {total_evaluated} evaluated → "
            f"{with_data} with data → "
            f"{theoretically_profitable} theoretical → "
            f"{after_spread_filter} good spreads → "
            f"{guaranteed_profitable} guaranteed → "
            f"1 selected"
        )

        # Get the actual ExpiryOption for the best opportunity
        best_expiry_option = self.expiry_options[best_idx]

        # Log detailed information about the selected opportunity for debugging
        logger.info(f"[{self.symbol}] SELECTED OPPORTUNITY DETAILS:")
        logger.info(f"  Index: {best_idx}, Expiry: {best_expiry_option.expiry}")
        logger.info(
            f"  Call Strike: {best_expiry_option.call_strike}, Put Strike: {best_expiry_option.put_strike}"
        )
        logger.info(
            f"  Contract IDs - Call: {best_expiry_option.call_contract.conId}, Put: {best_expiry_option.put_contract.conId}"
        )

        return {
            "best_idx": best_idx,
            "best_expiry_option": best_expiry_option,  # Return actual ExpiryOption object
            "best_profit": best_profit,
            "vectorized_data": vectorized_data,
            "statistics": {
                "total_evaluated": total_evaluated,
                "rejected_by_spreads": spread_stats["rejected_by_spread"],
                "mean_call_spread": spread_stats["mean_call_spread"],
                "mean_put_spread": spread_stats["mean_put_spread"],
            },
        }
