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
    buffer_percent: float = 0.00,  # 2% buffer for slippage
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


def validate_strike_ordering(call_strike: float, put_strike: float) -> bool:
    """Validate that call strike is greater than put strike"""
    return call_strike > put_strike


def format_trade_summary(
    symbol: str,
    expiry: str,
    call_strike: float,
    put_strike: float,
    call_price: float,
    put_price: float,
    stock_price: float,
    net_credit: float,
    min_profit: float,
    max_profit: float,
    min_roi: float,
) -> str:
    """Format a comprehensive trade summary string"""
    return (
        f"[{symbol}] Trade Summary:\n"
        f"  Expiry: {expiry}\n"
        f"  Call: {call_strike} @ ${call_price:.2f}\n"
        f"  Put: {put_strike} @ ${put_price:.2f}\n"
        f"  Stock: ${stock_price:.2f}\n"
        f"  Net Credit: ${net_credit:.2f}\n"
        f"  Profit Range: ${min_profit:.2f} - ${max_profit:.2f}\n"
        f"  Min ROI: {min_roi:.2f}%"
    )


def round_prices(prices: dict) -> dict:
    """Round all price values to 2 decimal places"""
    rounded = {}
    for key, value in prices.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            rounded[key] = round(value, 2)
        else:
            rounded[key] = value
    return rounded


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    try:
        if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default


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


def format_percentage(value: float, precision: int = 2) -> str:
    """Format a decimal value as a percentage string"""
    return f"{value * 100:.{precision}f}%"


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max bounds"""
    return max(min_val, min(value, max_val))


def is_weekend() -> bool:
    """Check if current day is weekend (Saturday or Sunday)"""
    from datetime import datetime

    return datetime.now().weekday() >= 5
