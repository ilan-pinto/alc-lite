"""
Validation logic for Synthetic arbitrage strategy.

This module contains:
- Quick viability checks for opportunities
- Condition validation for trades
- Strike validation with caching
- Global caching system for performance
"""

import time
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from ib_async import Option

from ..common import get_logger
from ..metrics import RejectionReason
from .constants import (
    CACHE_TTL,
    CALL_MONEYNESS_MAX,
    CALL_MONEYNESS_MIN,
    MAX_DAYS_TO_EXPIRY,
    MAX_STRIKE_SPREAD,
    MIN_DAYS_TO_EXPIRY,
    MIN_STRIKE_SPREAD,
    PUT_MONEYNESS_MAX,
    PUT_MONEYNESS_MIN,
)
from .models import ExpiryOption

logger = get_logger()

# Global strike cache for validated strikes per expiry
strike_cache = {}


class ValidationEngine:
    """
    Validation engine for synthetic arbitrage opportunities.
    Handles quick viability checks, condition validation, and strike validation.
    """

    def __init__(self, ib_connection=None):
        """
        Initialize validation engine.

        Args:
            ib_connection: Interactive Brokers connection for strike validation
        """
        self.ib = ib_connection
        self.logger = logger

    def quick_viability_check(
        self, expiry_option: ExpiryOption, stock_price: float
    ) -> Tuple[bool, Optional[str]]:
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

        # Quick moneyness check for synthetics (more lenient than SFR)
        call_moneyness = expiry_option.call_strike / stock_price
        put_moneyness = expiry_option.put_strike / stock_price
        if (
            call_moneyness < CALL_MONEYNESS_MIN
            or call_moneyness > CALL_MONEYNESS_MAX
            or put_moneyness < PUT_MONEYNESS_MIN
            or put_moneyness > PUT_MONEYNESS_MAX
        ):
            return False, "poor_moneyness"

        return True, None

    def check_conditions(
        self,
        symbol: str,
        cost_limit: float,
        lmt_price: float,
        net_credit: float,
        min_roi: float,
        min_profit: float,
        max_profit: float,
        max_loss_threshold: Optional[float] = None,
        max_profit_threshold: Optional[float] = None,
        profit_ratio_threshold: Optional[float] = None,
    ) -> Tuple[bool, Optional[RejectionReason]]:
        """
        Check if opportunity meets all trading conditions.

        Args:
            symbol: Trading symbol
            cost_limit: Maximum allowed cost
            lmt_price: Limit price for the trade
            net_credit: Net credit received
            min_roi: Minimum ROI required
            min_profit: Minimum profit (negative = max loss)
            max_profit: Maximum profit potential
            max_loss_threshold: Optional maximum loss threshold
            max_profit_threshold: Optional maximum profit threshold
            profit_ratio_threshold: Optional profit ratio threshold

        Returns:
            Tuple of (meets_conditions, rejection_reason)
        """
        profit_ratio = max_profit / abs(min_profit) if min_profit != 0 else float("inf")

        if (
            max_loss_threshold is not None and max_loss_threshold >= min_profit
        ):  # no arbitrage condition
            self.logger.info(
                f"max_loss limit [{max_loss_threshold}] >  calculated max_loss [{min_profit}] - <doesn't meet conditions>"
            )
            return False, RejectionReason.MAX_LOSS_THRESHOLD_EXCEEDED

        elif net_credit < 0:
            self.logger.info(
                f"[{symbol}] net_credit[{net_credit}] < 0 - doesn't meet conditions"
            )
            return False, RejectionReason.NET_CREDIT_NEGATIVE

        elif max_profit_threshold is not None and max_profit_threshold < max_profit:
            self.logger.info(
                f"[{symbol}] max_profit threshold [{max_profit_threshold }] < max_profit [{max_profit}] - doesn't meet conditions"
            )
            return False, RejectionReason.MAX_PROFIT_THRESHOLD_NOT_MET

        elif (
            profit_ratio_threshold is not None and profit_ratio_threshold > profit_ratio
        ):
            self.logger.info(
                f"[{symbol}] profit_ratio_threshold  [{profit_ratio_threshold }] > profit_ratio [{profit_ratio}] - doesn't meet conditions"
            )
            return False, RejectionReason.PROFIT_RATIO_THRESHOLD_NOT_MET

        elif np.isnan(lmt_price) or lmt_price > cost_limit:
            self.logger.info(
                f"[{symbol}] np.isnan(lmt_price) or lmt_price > limit - doesn't meet conditions"
            )
            return False, RejectionReason.PRICE_LIMIT_EXCEEDED

        else:
            self.logger.info(
                f"[{symbol}] meets conditions - initiating order. [profit_ratio: {profit_ratio}]"
            )
            return True, None

    async def validate_strikes_for_expiry(
        self, symbol: str, expiry: str, potential_strikes: List[float]
    ) -> List[float]:
        """
        Validate which strikes actually exist for a specific expiry.
        Uses global caching to improve performance.

        Args:
            symbol: Trading symbol
            expiry: Specific expiry date (e.g., '20250912')
            potential_strikes: List of strikes to validate

        Returns:
            List of strikes that actually exist for this expiry
        """
        # Check cache first (5 minute TTL for strike validation)
        cache_key = f"{symbol}_{expiry}"
        current_time = time.time()

        if cache_key in strike_cache:
            cached_data = strike_cache[cache_key]
            if current_time - cached_data["timestamp"] < CACHE_TTL:
                self.logger.debug(
                    f"[{symbol}] Using cached strikes for {expiry}: {len(cached_data['strikes'])} strikes"
                )
                return list(cached_data["strikes"])

        # Limit strikes to reasonable range to avoid overwhelming API
        nearby_strikes = potential_strikes[:20]  # Limit to first 20 for API efficiency

        if not nearby_strikes:
            return []

        self.logger.debug(
            f"[{symbol}] Validating {len(nearby_strikes)} strikes for expiry {expiry}"
        )

        # Create test contracts for batch validation
        test_contracts = []
        for strike in nearby_strikes:
            # Test with call options (calls usually have same availability as puts)
            test_contract = Option(symbol, expiry, strike, "C", "SMART")
            test_contracts.append(test_contract)

        valid_strikes = []

        try:
            if self.ib:
                # Use qualifyContractsAsync for batch validation
                qualified_contracts = await self.ib.qualifyContractsAsync(
                    *test_contracts
                )

                for i, qualified in enumerate(qualified_contracts):
                    if qualified and hasattr(qualified, "conId") and qualified.conId:
                        valid_strikes.append(nearby_strikes[i])
            else:
                # Fallback when no IB connection available (testing)
                valid_strikes = nearby_strikes

        except Exception as e:
            self.logger.warning(
                f"[{symbol}] Strike validation failed for {expiry}: {e}"
            )
            # Fallback: assume all strikes are valid (better than blocking)
            valid_strikes = nearby_strikes

        # Cache the result
        strike_cache[cache_key] = {
            "strikes": set(valid_strikes),
            "timestamp": current_time,
        }

        self.logger.info(
            f"[{symbol}] Expiry {expiry}: {len(valid_strikes)}/{len(nearby_strikes)} strikes are valid"
        )
        if len(valid_strikes) < len(nearby_strikes):
            invalid_strikes = [s for s in nearby_strikes if s not in valid_strikes]
            self.logger.debug(
                f"[{symbol}] Invalid strikes for {expiry}: {invalid_strikes}"
            )

        return valid_strikes

    def clear_strike_cache(self):
        """Clear the global strike cache"""
        strike_cache.clear()
        self.logger.debug("Strike cache cleared")

    def get_cache_stats(self) -> dict:
        """Get statistics about the strike cache"""
        current_time = time.time()

        active_entries = 0
        expired_entries = 0

        for cache_key, cached_data in strike_cache.items():
            if current_time - cached_data["timestamp"] < CACHE_TTL:
                active_entries += 1
            else:
                expired_entries += 1

        return {
            "total_entries": len(strike_cache),
            "active_entries": active_entries,
            "expired_entries": expired_entries,
            "cache_ttl": CACHE_TTL,
        }


# Helper functions for backward compatibility
def quick_viability_check(
    expiry_option: ExpiryOption, stock_price: float, ib_connection=None
) -> Tuple[bool, Optional[str]]:
    """Standalone quick viability check function"""
    validator = ValidationEngine(ib_connection)
    return validator.quick_viability_check(expiry_option, stock_price)


def check_conditions(
    symbol: str,
    cost_limit: float,
    lmt_price: float,
    net_credit: float,
    min_roi: float,
    min_profit: float,
    max_profit: float,
    max_loss_threshold: Optional[float] = None,
    max_profit_threshold: Optional[float] = None,
    profit_ratio_threshold: Optional[float] = None,
) -> Tuple[bool, Optional[RejectionReason]]:
    """Standalone condition check function"""
    validator = ValidationEngine()
    return validator.check_conditions(
        symbol,
        cost_limit,
        lmt_price,
        net_credit,
        min_roi,
        min_profit,
        max_profit,
        max_loss_threshold,
        max_profit_threshold,
        profit_ratio_threshold,
    )


async def validate_strikes_for_expiry(
    symbol: str, expiry: str, potential_strikes: List[float], ib_connection=None
) -> List[float]:
    """Standalone strike validation function"""
    validator = ValidationEngine(ib_connection)
    return await validator.validate_strikes_for_expiry(
        symbol, expiry, potential_strikes
    )
