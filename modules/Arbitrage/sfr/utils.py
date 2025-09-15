"""
Utility functions for SFR arbitrage strategy.

This module contains helper functions and utilities used throughout the SFR strategy.
"""

from typing import Optional

import logging
import numpy as np
from ib_async import Ticker

from ..common import get_logger

logger = get_logger()


def get_symbol_contract_count(symbol: str, contract_ticker: dict) -> int:
    """Get count of contracts for a specific symbol"""
    return sum(1 for k in contract_ticker.keys() if k[0] == symbol)


def debug_contract_ticker_state(contract_ticker: dict) -> dict:
    """Debug helper to show contract_ticker state by symbol"""
    by_symbol = {}
    for (symbol, conId), _ in contract_ticker.items():
        if symbol not in by_symbol:
            by_symbol[symbol] = 0
        by_symbol[symbol] += 1
    logger.debug(f"Contract ticker state: {by_symbol}")
    return by_symbol


def get_stock_midpoint(stock_ticker: Ticker) -> Optional[float]:
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


def calculate_combo_limit_price(
    stock_price: float,
    call_price: float,
    put_price: float,
    buffer_percent: float = 0.02,  # 2% buffer for slippage (typically called with 0.01)
) -> float:
    """
    Calculate precise combo limit price based on individual leg target prices.

    Args:
        stock_price: Current stock price
        call_price: Target call option price (what we want to receive)
        put_price: Target put option price (what we want to pay)
        buffer_percent: Buffer percentage to account for slippage

    Returns:
        Calculated limit price for the combo order
    """
    # For conversion: Buy stock, Sell call, Buy put
    # Net cost = Stock price - Call premium + Put premium
    theoretical_cost = stock_price - call_price + put_price

    # Add buffer for market movement and slippage
    buffer_amount = theoretical_cost * buffer_percent
    limit_price = theoretical_cost + buffer_amount

    logger.info(
        f"Calculated combo limit: stock={stock_price:.2f}, call={call_price:.2f}, put={put_price:.2f}"
    )
    logger.info(
        f"Theoretical cost: {theoretical_cost:.2f}, with buffer: {limit_price:.2f}"
    )

    return round(limit_price, 2)


def round_price_to_tick_size(price: float, contract_type: str = "stock") -> float:
    """
    Round price to the appropriate tick size for the contract type.

    Args:
        price: The price to round
        contract_type: Type of contract ("stock", "option", "index")

    Returns:
        Price rounded to appropriate decimal places, with minimum 0.01 for positive prices
    """
    rounded_price = round(price, 2)

    # Ensure positive prices don't round to zero (minimum tick size)
    if price > 0 and rounded_price <= 0:
        return 0.01

    return rounded_price


def calculate_aggressive_execution_price(
    ticker, action: str, base_price: float, aggressiveness: float = 0.01
) -> float:
    """
    Calculate an aggressive execution price based on bid/ask spreads to improve fill rates.

    Args:
        ticker: IB ticker object with bid/ask data
        action: "BUY" or "SELL"
        base_price: Fallback price if bid/ask not available
        aggressiveness: Factor to penetrate into the spread (0.01 = 1% more aggressive)

    Returns:
        Rounded aggressive price for better fill probability
    """
    try:
        if action.upper() == "BUY":
            # For buying: start with ask price and make it slightly more aggressive
            if hasattr(ticker, "ask") and not np.isnan(ticker.ask) and ticker.ask > 0:
                # Use ask price + small premium to ensure fill
                aggressive_price = ticker.ask * (1.0 + aggressiveness)
            elif hasattr(ticker, "bid") and not np.isnan(ticker.bid) and ticker.bid > 0:
                # Fallback: bid + spread
                spread = (
                    abs(ticker.ask - ticker.bid)
                    if hasattr(ticker, "ask") and not np.isnan(ticker.ask)
                    else ticker.bid * 0.02
                )
                aggressive_price = ticker.bid + spread + (spread * aggressiveness)
            else:
                # Ultimate fallback: base price + small premium
                aggressive_price = base_price * (1.0 + aggressiveness)

        else:  # SELL
            # For selling: start with bid price and make it slightly more aggressive
            if hasattr(ticker, "bid") and not np.isnan(ticker.bid) and ticker.bid > 0:
                # Use bid price - small discount to ensure fill
                aggressive_price = ticker.bid * (1.0 - aggressiveness)
            elif hasattr(ticker, "ask") and not np.isnan(ticker.ask) and ticker.ask > 0:
                # Fallback: ask - spread
                spread = (
                    abs(ticker.ask - ticker.bid)
                    if hasattr(ticker, "bid") and not np.isnan(ticker.bid)
                    else ticker.ask * 0.02
                )
                aggressive_price = ticker.ask - spread - (spread * aggressiveness)
            else:
                # Ultimate fallback: base price - small discount
                aggressive_price = base_price * (1.0 - aggressiveness)

        # Ensure price is positive
        aggressive_price = max(aggressive_price, 0.01)

        # Determine contract type for rounding
        contract_type = "option"  # Most common case
        return round_price_to_tick_size(aggressive_price, contract_type)

    except Exception as e:
        logger.warning(f"Error calculating aggressive price: {e}, using base price")
        return round_price_to_tick_size(base_price, "option")


def flush_all_handlers():
    """Flush all logging handlers to ensure messages are written to files"""
    for handler in logging.getLogger().handlers:
        if hasattr(handler, "flush"):
            handler.flush()


def log_funnel_summary(symbol: str, funnel_analysis: dict):
    """Log concise funnel analysis summary"""
    if funnel_analysis["total_opportunities"] == 0:
        logger.info(f"[Funnel Summary] {symbol}: No opportunities evaluated")
        return

    funnel_stages = funnel_analysis["funnel_stages"]
    total = funnel_stages.get("evaluated", 0)
    theoretical_positive = funnel_stages.get("theoretical_profit_positive", 0)
    guaranteed_positive = funnel_stages.get("guaranteed_profit_positive", 0)
    executed = funnel_stages.get("executed", 0)

    # Single line summary with key metrics
    logger.info(
        f"[Funnel Summary] {symbol}: {total} evaluated → "
        f"{theoretical_positive} theoretical → {guaranteed_positive} viable → {executed} executed"
    )


def calculate_z_scores(data: np.ndarray) -> np.ndarray:
    """Calculate z-scores for outlier detection"""
    if len(data) == 0:
        return np.array([])

    mean = np.mean(data)
    std = np.std(data)

    if std == 0:
        return np.zeros(len(data))

    return (data - mean) / std


def get_contract_key(contract) -> str:
    """Generate a deterministic key for a contract based on its content"""
    try:
        symbol = getattr(contract, "symbol", "").upper()
        expiry = getattr(contract, "lastTradeDateOrContractMonth", "")
        strike = getattr(contract, "strike", 0)
        right = getattr(contract, "right", "")
        return f"{symbol}_{expiry}_{strike}_{right}"
    except Exception:
        # Fallback to id() if contract doesn't have expected attributes
        return str(id(contract))


def calculate_days_to_expiry(expiry_str: str) -> int:
    """Calculate days to expiry from expiry string"""
    from datetime import datetime

    try:
        expiry_date = datetime.strptime(expiry_str, "%Y%m%d")
        return (expiry_date - datetime.now()).days
    except (ValueError, TypeError):
        return 0


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max bounds"""
    return max(min_val, min(value, max_val))
