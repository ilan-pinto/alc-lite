"""
Data models for calendar spread strategy.

This module contains all the data classes and configuration objects used
in the calendar spread strategy implementation.
"""

from dataclasses import dataclass
from typing import Optional

from ib_async import Contract


@dataclass
class CalendarSpreadLeg:
    """Data class to hold calendar spread leg information"""

    contract: Contract
    strike: float
    expiry: str
    right: str  # 'C' for call, 'P' for put
    price: float
    bid: float
    ask: float
    volume: int
    iv: float  # Implied volatility
    theta: float  # Time decay
    days_to_expiry: int


@dataclass
class CalendarSpreadOpportunity:
    """Data class to hold complete calendar spread opportunity"""

    symbol: str
    strike: float
    option_type: str  # 'CALL' or 'PUT'
    front_leg: CalendarSpreadLeg
    back_leg: CalendarSpreadLeg

    # Spread metrics
    iv_spread: float  # Back IV - Front IV (%)
    theta_ratio: float  # Front theta / Back theta
    net_debit: float  # Cost to enter position
    max_profit: float  # Maximum theoretical profit
    max_loss: float  # Maximum loss (net debit)

    # Quality metrics
    front_bid_ask_spread: float
    back_bid_ask_spread: float
    combined_liquidity_score: float
    term_structure_inversion: bool

    # Greeks analysis
    net_delta: float
    net_gamma: float
    net_vega: float

    # Scoring
    composite_score: float

    # Profitability boundaries (optional fields must be last)
    lower_breakeven: Optional[float] = None  # Lower breakeven stock price
    upper_breakeven: Optional[float] = None  # Upper breakeven stock price
    profitability_range: Optional[float] = None  # Distance between breakevens


@dataclass
class CalendarSpreadConfig:
    """Configuration for calendar spread detection"""

    min_iv_spread: float = 1.5  # Minimum IV spread (%)
    min_theta_ratio: float = 1.5  # Minimum theta ratio (front/back)
    max_bid_ask_spread: float = 0.15  # Maximum bid-ask spread as % of mid
    min_liquidity_score: float = 0.4  # Minimum liquidity threshold
    max_days_front: int = 45  # Maximum days to front expiry
    min_days_back: int = 60  # Minimum days to back expiry
    max_days_back: int = 120  # Maximum days to back expiry
    min_volume: int = 10  # Minimum daily volume per leg
    max_net_debit: float = 500.0  # Maximum cost to enter position
    target_profit_ratio: float = 0.3  # Target profit as % of max profit

    # Pricing optimization parameters
    base_edge_factor: float = 0.3  # Base edge to give up (30% of spread)
    max_edge_factor: float = 0.65  # Maximum edge in wide spreads
    wide_spread_threshold: float = 0.15  # Spread % to trigger higher edge
    time_adjustment_enabled: bool = True  # Enable time-of-day adjustments
