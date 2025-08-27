"""
Data models and structures for SFR arbitrage strategy.

This module contains all the dataclasses and type definitions used by the SFR strategy.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ib_async import Contract


@dataclass
class ExpiryOption:
    """Data class to hold option contract information for a specific expiry"""

    expiry: str
    call_contract: Contract
    put_contract: Contract
    call_strike: float
    put_strike: float


@dataclass
class OpportunityResult:
    """Data class to hold the result of opportunity evaluation"""

    contract: Contract
    order: object  # Order type
    min_profit: float
    trade_details: Dict
    expiry_option: ExpiryOption


@dataclass
class MarketDataQuality:
    """Data class to hold market data quality metrics"""

    stock_score: float
    call_score: float
    put_score: float
    composite_score: float
    has_volume: bool
    spread_quality: float


@dataclass
class PricingData:
    """Data class to hold pricing information for opportunity calculation"""

    stock_fair: float
    stock_exec: float
    call_fair: float
    call_exec: float
    put_fair: float
    put_exec: float

    # Calculated values
    theoretical_net_credit: float
    theoretical_spread: float
    theoretical_profit: float
    guaranteed_net_credit: float
    guaranteed_spread: float
    guaranteed_profit: float


@dataclass
class SpreadAnalysis:
    """Data class to hold spread analysis results"""

    call_spread: float
    put_spread: float
    call_spread_pct: float
    put_spread_pct: float
    viable: bool
    quality_score: float


@dataclass
class VectorizedOpportunityData:
    """Data class to hold vectorized opportunity calculation results"""

    theoretical_profits: "np.ndarray"
    guaranteed_profits: "np.ndarray"
    market_data: Dict
    viable_mask: "np.ndarray"
    spread_stats: Dict


# Type aliases for complex return types
OpportunityTuple = Tuple[
    Contract, object, float, Dict
]  # (contract, order, min_profit, trade_details)
ViabilityCheck = Tuple[bool, Optional[str]]  # (is_viable, rejection_reason)
ConditionsCheck = Tuple[
    bool, Optional["RejectionReason"]
]  # (conditions_met, rejection_reason)
