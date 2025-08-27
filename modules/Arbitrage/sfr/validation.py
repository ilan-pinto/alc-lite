"""
Validation logic for SFR arbitrage strategy.

This module contains all the validation functions for contracts, opportunities,
and market conditions used by the SFR strategy.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np
from ib_async import Ticker

from ..common import get_logger
from .constants import (
    CALL_MONEYNESS_MAX,
    CALL_MONEYNESS_MIN,
    MAX_DAYS_TO_EXPIRY,
    MAX_STRIKE_SPREAD,
    MIN_DAYS_TO_EXPIRY,
    MIN_STRIKE_SPREAD,
    PUT_MONEYNESS_MAX,
    PUT_MONEYNESS_MIN,
)
from .models import ExpiryOption, ViabilityCheck

logger = get_logger()


class MarketValidator:
    """Handles market-related validations for SFR strategy"""

    @staticmethod
    def is_market_hours() -> bool:
        """Check if current time is during market hours (9:30 AM - 4:00 PM ET)"""
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


class PriceValidator:
    """Handles price-related validations"""

    @staticmethod
    def is_valid_price(value) -> bool:
        """Check if a price value is valid (not None, not empty array, not NaN)"""
        if value is None:
            return False
        if hasattr(value, "__len__") and len(value) == 0:
            return False
        try:
            return not np.isnan(value)
        except (TypeError, ValueError):
            return False


class StrikeValidator:
    """Handles strike-related validations"""

    @staticmethod
    def find_stock_position_in_strikes(
        stock_price: float, valid_strikes: List[float]
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


class OpportunityValidator:
    """Handles opportunity-specific validations"""

    def __init__(self):
        self.price_validator = PriceValidator()

    def quick_viability_check(
        self, expiry_option: ExpiryOption, stock_price: float
    ) -> ViabilityCheck:
        """Fast pre-filtering to eliminate non-viable opportunities early"""
        # Quick strike spread check
        strike_spread = expiry_option.call_strike - expiry_option.put_strike
        if strike_spread < MIN_STRIKE_SPREAD or strike_spread > MAX_STRIKE_SPREAD:
            return False, "invalid_strike_spread"

        # Quick time to expiry check
        try:
            expiry_date = datetime.strptime(expiry_option.expiry, "%Y%m%d")
            days_to_expiry = (expiry_date - datetime.now()).days

            if (
                days_to_expiry < MIN_DAYS_TO_EXPIRY
                or days_to_expiry > MAX_DAYS_TO_EXPIRY
            ):
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
            call_moneyness < CALL_MONEYNESS_MIN  # Allow deeper ITM calls
            or call_moneyness > CALL_MONEYNESS_MAX  # Allow more OTM calls
            or put_moneyness < PUT_MONEYNESS_MIN  # Allow deeper ITM puts
            or put_moneyness > PUT_MONEYNESS_MAX  # Allow more OTM puts
        ):
            return False, "poor_moneyness"

        return True, None


class DataQualityValidator:
    """Handles data quality validations"""

    def __init__(self):
        self.price_validator = PriceValidator()

    def calculate_data_quality_score(
        self, stock_ticker, call_ticker, put_ticker
    ) -> float:
        """Calculate a quality score (0-1) based on data completeness and freshness"""
        score = 0.0

        # Stock data quality (30% weight)
        if stock_ticker:
            stock_score = 0.0
            if hasattr(stock_ticker, "bid") and self.price_validator.is_valid_price(
                stock_ticker.bid
            ):
                stock_score += 0.1
            if hasattr(stock_ticker, "ask") and self.price_validator.is_valid_price(
                stock_ticker.ask
            ):
                stock_score += 0.1
            if hasattr(stock_ticker, "last") and self.price_validator.is_valid_price(
                stock_ticker.last
            ):
                stock_score += 0.05
            if hasattr(stock_ticker, "volume") and stock_ticker.volume > 0:
                stock_score += 0.05
            score += stock_score

        # Call option data quality (35% weight)
        if call_ticker:
            call_score = 0.0
            if hasattr(call_ticker, "bid") and self.price_validator.is_valid_price(
                call_ticker.bid
            ):
                call_score += 0.1
            if hasattr(call_ticker, "ask") and self.price_validator.is_valid_price(
                call_ticker.ask
            ):
                call_score += 0.1
            if hasattr(call_ticker, "last") and self.price_validator.is_valid_price(
                call_ticker.last
            ):
                call_score += 0.05
            if hasattr(call_ticker, "volume") and call_ticker.volume > 0:
                call_score += 0.05
            # Bid-ask spread quality
            if (
                hasattr(call_ticker, "bid")
                and hasattr(call_ticker, "ask")
                and self.price_validator.is_valid_price(call_ticker.bid)
                and self.price_validator.is_valid_price(call_ticker.ask)
            ):
                spread = abs(call_ticker.ask - call_ticker.bid)
                if spread < 5.0:  # Reasonable spread
                    call_score += 0.05
            score += call_score

        # Put option data quality (35% weight)
        if put_ticker:
            put_score = 0.0
            if hasattr(put_ticker, "bid") and self.price_validator.is_valid_price(
                put_ticker.bid
            ):
                put_score += 0.1
            if hasattr(put_ticker, "ask") and self.price_validator.is_valid_price(
                put_ticker.ask
            ):
                put_score += 0.1
            if hasattr(put_ticker, "last") and self.price_validator.is_valid_price(
                put_ticker.last
            ):
                put_score += 0.05
            if hasattr(put_ticker, "volume") and put_ticker.volume > 0:
                put_score += 0.05
            # Bid-ask spread quality
            if (
                hasattr(put_ticker, "bid")
                and hasattr(put_ticker, "ask")
                and self.price_validator.is_valid_price(put_ticker.bid)
                and self.price_validator.is_valid_price(put_ticker.ask)
            ):
                spread = abs(put_ticker.ask - put_ticker.bid)
                if spread < 5.0:  # Reasonable spread
                    put_score += 0.05
            score += put_score

        # Log details for low quality scores
        final_score = min(score, 1.0)
        if (
            final_score < 0.6
        ):  # Log breakdown for scores that might cause evaluation failures
            self._log_quality_breakdown(
                stock_ticker, call_ticker, put_ticker, final_score
            )

        return final_score

    def _log_quality_breakdown(
        self, stock_ticker, call_ticker, put_ticker, final_score
    ):
        """Log detailed breakdown of quality scores"""
        stock_score = 0.0
        call_score = 0.0
        put_score = 0.0

        # Recalculate component scores for logging
        if stock_ticker:
            if hasattr(stock_ticker, "bid") and self.price_validator.is_valid_price(
                stock_ticker.bid
            ):
                stock_score += 0.1
            if hasattr(stock_ticker, "ask") and self.price_validator.is_valid_price(
                stock_ticker.ask
            ):
                stock_score += 0.1
            if hasattr(stock_ticker, "last") and self.price_validator.is_valid_price(
                stock_ticker.last
            ):
                stock_score += 0.05
            if hasattr(stock_ticker, "volume") and stock_ticker.volume > 0:
                stock_score += 0.05

        if call_ticker:
            if hasattr(call_ticker, "bid") and self.price_validator.is_valid_price(
                call_ticker.bid
            ):
                call_score += 0.1
            if hasattr(call_ticker, "ask") and self.price_validator.is_valid_price(
                call_ticker.ask
            ):
                call_score += 0.1
            if hasattr(call_ticker, "last") and self.price_validator.is_valid_price(
                call_ticker.last
            ):
                call_score += 0.05
            if hasattr(call_ticker, "volume") and call_ticker.volume > 0:
                call_score += 0.05
            if (
                hasattr(call_ticker, "bid")
                and hasattr(call_ticker, "ask")
                and self.price_validator.is_valid_price(call_ticker.bid)
                and self.price_validator.is_valid_price(call_ticker.ask)
            ):
                spread = abs(call_ticker.ask - call_ticker.bid)
                if spread < 5.0:
                    call_score += 0.05

        if put_ticker:
            if hasattr(put_ticker, "bid") and self.price_validator.is_valid_price(
                put_ticker.bid
            ):
                put_score += 0.1
            if hasattr(put_ticker, "ask") and self.price_validator.is_valid_price(
                put_ticker.ask
            ):
                put_score += 0.1
            if hasattr(put_ticker, "last") and self.price_validator.is_valid_price(
                put_ticker.last
            ):
                put_score += 0.05
            if hasattr(put_ticker, "volume") and put_ticker.volume > 0:
                put_score += 0.05
            if (
                hasattr(put_ticker, "bid")
                and hasattr(put_ticker, "ask")
                and self.price_validator.is_valid_price(put_ticker.bid)
                and self.price_validator.is_valid_price(put_ticker.ask)
            ):
                spread = abs(put_ticker.ask - put_ticker.bid)
                if spread < 5.0:
                    put_score += 0.05

        logger.debug(
            f"Low data quality score: {final_score:.2f} "
            f"(stock={stock_score:.2f}, call={call_score:.2f}, put={put_score:.2f})"
        )


class ConditionsValidator:
    """Handles conditions validation for trade execution"""

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
    ) -> Tuple[bool, Optional["RejectionReason"]]:
        """Check if trading conditions are met for SFR arbitrage"""
        from ..metrics import RejectionReason

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

        elif min_profit < 0.03:  # Lowered minimum profit threshold to 3 cents
            logger.info(
                f"[{symbol}] min_profit[{min_profit:.2f}] < 0.03 - below minimum threshold"
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
