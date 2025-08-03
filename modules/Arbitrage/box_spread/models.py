"""
Data models for box spread strategy.

This module contains all the data classes and configuration objects used
in the box spread strategy implementation.

Box spread consists of 4 legs:
- Long call at K1 (lower strike)
- Short call at K2 (higher strike)
- Short put at K1 (lower strike)
- Long put at K2 (higher strike)

Risk-free arbitrage when: net_debit < (K2 - K1)
"""

from dataclasses import dataclass
from typing import Optional

from ib_async import Contract


@dataclass
class BoxSpreadLeg:
    """Data class to hold box spread leg information"""

    contract: Contract
    strike: float
    expiry: str
    right: str  # 'C' for call, 'P' for put
    action: str  # 'BUY' or 'SELL'
    price: float
    bid: float
    ask: float
    volume: int
    iv: float  # Implied volatility
    delta: float
    gamma: float
    theta: float  # Time decay
    vega: float
    days_to_expiry: int


@dataclass
class BoxSpreadOpportunity:
    """Data class to hold complete box spread opportunity"""

    symbol: str
    lower_strike: float  # K1
    upper_strike: float  # K2
    expiry: str

    # The 4 legs of the box spread
    long_call_k1: BoxSpreadLeg  # Buy call at K1
    short_call_k2: BoxSpreadLeg  # Sell call at K2
    short_put_k1: BoxSpreadLeg  # Sell put at K1
    long_put_k2: BoxSpreadLeg  # Buy put at K2

    # Spread metrics
    strike_width: float  # K2 - K1
    net_debit: float  # Total cost to enter position
    theoretical_value: float  # Strike width (guaranteed payoff)
    arbitrage_profit: float  # Theoretical value - net debit
    profit_percentage: float  # (arbitrage_profit / net_debit) * 100

    # Risk metrics
    max_profit: float  # Same as arbitrage_profit for true arbitrage
    max_loss: float  # Should be 0 for true arbitrage, or net_debit for failed arbitrage
    risk_free: bool  # True if net_debit < strike_width

    # Quality metrics
    total_bid_ask_spread: float  # Sum of all leg bid-ask spreads
    combined_liquidity_score: float
    execution_difficulty: float  # Higher = harder to execute

    # Greeks analysis
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float

    # Scoring
    composite_score: float

    # Execution timing metrics (optional fields must be last)
    time_to_expiry_days: Optional[int] = None
    interest_rate_sensitivity: Optional[float] = (
        None  # How much profit depends on rates
    )
    early_exercise_risk: Optional[float] = None  # Risk of early assignment


@dataclass
class BoxSpreadConfig:
    """Configuration for box spread detection and execution"""

    # Arbitrage detection parameters
    min_arbitrage_profit: float = 0.01  # Minimum profit per dollar invested (1%)
    min_absolute_profit: float = 0.05  # Minimum absolute profit per spread ($0.05)
    max_net_debit: float = 1000.0  # Maximum cost to enter position

    # Strike and expiry filters
    min_strike_width: float = 1.0  # Minimum difference between K1 and K2
    max_strike_width: float = 50.0  # Maximum difference between K1 and K2
    min_days_to_expiry: int = 1  # Minimum days to expiration
    max_days_to_expiry: int = 90  # Maximum days to expiration

    # Liquidity and quality filters
    min_volume_per_leg: int = 5  # Minimum daily volume per option leg
    max_bid_ask_spread_percent: float = 0.10  # Maximum bid-ask spread as % of mid (10%)
    min_liquidity_score: float = 0.3  # Minimum combined liquidity threshold

    # Execution parameters
    safety_buffer: float = 0.02  # Safety margin for pricing (2% of net debit)
    max_execution_legs: int = (
        4  # Always 4 for box spreads, but configurable for testing
    )
    order_timeout_seconds: int = 30  # Timeout for individual leg orders

    # Risk management
    max_greek_exposure: float = 0.1  # Maximum net greek exposure tolerance
    require_risk_free: bool = True  # Only execute if truly risk-free
    early_exercise_protection: bool = True  # Avoid deep ITM options near expiry

    # Performance optimization
    enable_caching: bool = True  # Enable contract and pricing caches
    cache_ttl_seconds: int = 60  # Cache time-to-live
    enable_parallel_processing: bool = True  # Process multiple symbols in parallel
    max_concurrent_scans: int = 5  # Maximum concurrent symbol scans

    # Pricing precision
    price_precision_decimals: int = 2  # Decimal places for price calculations
    profit_calculation_buffer: float = 0.001  # Small buffer for floating point errors

    def validate(self) -> None:
        """Validate configuration parameters"""
        if self.min_arbitrage_profit <= 0:
            raise ValueError("min_arbitrage_profit must be positive")

        if self.max_net_debit <= 0:
            raise ValueError("max_net_debit must be positive")

        if self.min_strike_width <= 0:
            raise ValueError("min_strike_width must be positive")

        if self.max_strike_width <= self.min_strike_width:
            raise ValueError("max_strike_width must be greater than min_strike_width")

        if self.min_days_to_expiry < 1:
            raise ValueError("min_days_to_expiry must be at least 1")

        if self.max_days_to_expiry <= self.min_days_to_expiry:
            raise ValueError(
                "max_days_to_expiry must be greater than min_days_to_expiry"
            )

        if not (0 <= self.max_bid_ask_spread_percent <= 1):
            raise ValueError("max_bid_ask_spread_percent must be between 0 and 1")

        if not (0 <= self.min_liquidity_score <= 1):
            raise ValueError("min_liquidity_score must be between 0 and 1")

        if self.safety_buffer < 0:
            raise ValueError("safety_buffer must be non-negative")
