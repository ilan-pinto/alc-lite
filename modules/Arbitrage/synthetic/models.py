"""
Data models and structures for Synthetic arbitrage strategy.

This module contains all the dataclasses and type definitions used by the Synthetic strategy.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from ib_async import Contract, Order

from ..common import get_logger

logger = get_logger()


@dataclass
class ExpiryOption:
    """Data class to hold option contract information for a specific expiry"""

    expiry: str
    call_contract: Contract
    put_contract: Contract
    call_strike: float
    put_strike: float


@dataclass
class OpportunityScore:
    """Data class to hold scoring components for an opportunity"""

    risk_reward_ratio: float
    liquidity_score: float
    time_decay_score: float
    market_quality_score: float
    composite_score: float


@dataclass
class GlobalOpportunity:
    """Data class to hold a complete arbitrage opportunity with scoring"""

    symbol: str
    conversion_contract: Contract
    order: Order
    trade_details: Dict
    score: OpportunityScore
    timestamp: float

    # Additional metadata for decision making
    call_volume: float
    put_volume: float
    call_bid_ask_spread: float
    put_bid_ask_spread: float
    days_to_expiry: int
