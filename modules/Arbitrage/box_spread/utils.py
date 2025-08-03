"""
Utility functions and classes for box spread strategy.

This module reuses utilities from calendar_spread where applicable and adds
box-specific functionality.
"""

from typing import List, Tuple

# Box-specific utility functions
import numpy as np

# Import and re-export calendar_spread utilities
from ..calendar_spread.utils import (
    SCIPY_AVAILABLE,
    AdaptiveCacheManager,
    PerformanceProfiler,
    TTLCache,
    VectorizedGreeksCalculator,
    _safe_isnan,
)
from ..common import get_logger

logger = get_logger()


def calculate_box_arbitrage_profit(
    long_call_k1_price: float,
    short_call_k2_price: float,
    short_put_k1_price: float,
    long_put_k2_price: float,
    strike_width: float,
    safety_buffer: float = 0.0,
) -> Tuple[float, float, bool]:
    """
    Calculate arbitrage profit for a box spread.

    Args:
        long_call_k1_price: Price to pay for long call at K1
        short_call_k2_price: Price to receive for short call at K2
        short_put_k1_price: Price to receive for short put at K1
        long_put_k2_price: Price to pay for long put at K2
        strike_width: K2 - K1 (guaranteed payoff)
        safety_buffer: Safety margin to subtract from profit

    Returns:
        Tuple of (net_debit, arbitrage_profit, is_risk_free)
    """
    # Net debit = what we pay - what we receive
    net_debit = (long_call_k1_price + long_put_k2_price) - (
        short_call_k2_price + short_put_k1_price
    )

    # Arbitrage profit = guaranteed payoff - net cost
    arbitrage_profit = strike_width - net_debit - safety_buffer

    # Risk-free if we profit regardless of outcome
    is_risk_free = arbitrage_profit > 0

    return net_debit, arbitrage_profit, is_risk_free


def validate_box_spread_legs(
    long_call_k1_strike: float,
    short_call_k2_strike: float,
    short_put_k1_strike: float,
    long_put_k2_strike: float,
    expiry1: str,
    expiry2: str,
    expiry3: str,
    expiry4: str,
) -> bool:
    """
    Validate that the 4 legs form a proper box spread.

    Args:
        Strikes and expiries for all 4 legs

    Returns:
        True if valid box spread configuration
    """
    # All legs must have same expiry
    if not (expiry1 == expiry2 == expiry3 == expiry4):
        logger.warning("Box spread legs must have same expiry")
        return False

    # K1 must be less than K2
    if long_call_k1_strike >= short_call_k2_strike:
        logger.warning(
            f"Lower strike {long_call_k1_strike} must be less than upper strike {short_call_k2_strike}"
        )
        return False

    # Put strikes must match call strikes
    if long_call_k1_strike != short_put_k1_strike:
        logger.warning(
            f"Long call K1 strike {long_call_k1_strike} must match short put K1 strike {short_put_k1_strike}"
        )
        return False

    if short_call_k2_strike != long_put_k2_strike:
        logger.warning(
            f"Short call K2 strike {short_call_k2_strike} must match long put K2 strike {long_put_k2_strike}"
        )
        return False

    return True


def calculate_box_greeks(
    call_k1_greeks: dict, call_k2_greeks: dict, put_k1_greeks: dict, put_k2_greeks: dict
) -> dict:
    """
    Calculate net Greeks for the box spread position.

    Box position:
    +1 Call K1, -1 Call K2, -1 Put K1, +1 Put K2

    Args:
        Greeks dictionaries for each leg with keys: delta, gamma, theta, vega

    Returns:
        Dictionary with net Greeks
    """
    net_greeks = {}

    # Calculate net Greeks (long positions add, short positions subtract)
    for greek in ["delta", "gamma", "theta", "vega"]:
        net_greek = (
            call_k1_greeks.get(greek, 0)  # +1 Call K1
            - call_k2_greeks.get(greek, 0)  # -1 Call K2
            - put_k1_greeks.get(greek, 0)  # -1 Put K1
            + put_k2_greeks.get(greek, 0)  # +1 Put K2
        )
        net_greeks[greek] = net_greek

    return net_greeks


def calculate_liquidity_score(
    call_k1_volume: int,
    call_k2_volume: int,
    put_k1_volume: int,
    put_k2_volume: int,
    call_k1_bid_ask_spread: float,
    call_k2_bid_ask_spread: float,
    put_k1_bid_ask_spread: float,
    put_k2_bid_ask_spread: float,
) -> float:
    """
    Calculate combined liquidity score for box spread.

    Score is based on:
    - Minimum volume across all legs (bottleneck)
    - Average bid-ask spread (execution cost)

    Returns:
        Float between 0 and 1 (higher = better liquidity)
    """
    # Volume component (use minimum as bottleneck)
    min_volume = min(call_k1_volume, call_k2_volume, put_k1_volume, put_k2_volume)
    volume_score = min(1.0, min_volume / 100.0)  # Normalize to 100 volume = 1.0

    # Spread component (use average)
    avg_spread = np.mean(
        [
            call_k1_bid_ask_spread,
            call_k2_bid_ask_spread,
            put_k1_bid_ask_spread,
            put_k2_bid_ask_spread,
        ]
    )

    # Convert spread to score (lower spread = higher score)
    spread_score = max(0.0, 1.0 - (avg_spread / 0.1))  # 0.1 spread = 0 score

    # Combine scores (equal weight)
    combined_score = (volume_score + spread_score) / 2.0

    return max(0.0, min(1.0, combined_score))


def calculate_execution_difficulty(
    total_legs: int,
    total_bid_ask_spread: float,
    liquidity_score: float,
    market_volatility: float = 0.2,
) -> float:
    """
    Calculate execution difficulty score for box spread.

    Higher score = more difficult to execute

    Args:
        total_legs: Number of option legs (4 for box)
        total_bid_ask_spread: Sum of all bid-ask spreads
        liquidity_score: Combined liquidity score (0-1)
        market_volatility: Current market volatility estimate

    Returns:
        Difficulty score (0 = easy, 1 = very difficult)
    """
    # Base difficulty from number of legs
    leg_difficulty = (total_legs - 1) / 10.0  # 4 legs = 0.3 base difficulty

    # Spread difficulty
    spread_difficulty = min(
        1.0, total_bid_ask_spread / 0.5
    )  # 0.5 total spread = max difficulty

    # Liquidity difficulty (inverse of liquidity score)
    liquidity_difficulty = 1.0 - liquidity_score

    # Volatility adjustment
    volatility_adjustment = min(0.3, market_volatility)  # Cap at 30% adjustment

    # Combine factors
    total_difficulty = (
        leg_difficulty * 0.2
        + spread_difficulty * 0.4
        + liquidity_difficulty * 0.3
        + volatility_adjustment * 0.1
    )

    return max(0.0, min(1.0, total_difficulty))


def format_box_spread_summary(opportunity) -> str:
    """
    Format a box spread opportunity for display.

    Args:
        opportunity: BoxSpreadOpportunity object

    Returns:
        Formatted string summary
    """
    return (
        f"Box Spread {opportunity.symbol}: "
        f"K1={opportunity.lower_strike:.2f}, K2={opportunity.upper_strike:.2f}, "
        f"Width={opportunity.strike_width:.2f}, "
        f"Debit=${opportunity.net_debit:.2f}, "
        f"Profit=${opportunity.arbitrage_profit:.2f} ({opportunity.profit_percentage:.1f}%), "
        f"Risk-Free={opportunity.risk_free}, "
        f"Score={opportunity.composite_score:.3f}"
    )
