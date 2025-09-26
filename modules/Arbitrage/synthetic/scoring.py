"""
Scoring configurations and algorithms for Synthetic arbitrage strategy.

This module contains:
- ScoringConfig class with factory methods
- Scoring algorithms for opportunities
- Weight normalization and validation
- Preset configurations (conservative, aggressive, balanced, liquidity-focused)
"""

from dataclasses import dataclass

from .constants import (
    DEFAULT_LIQUIDITY_WEIGHT,
    DEFAULT_MARKET_QUALITY_WEIGHT,
    DEFAULT_RISK_REWARD_WEIGHT,
    DEFAULT_TIME_DECAY_WEIGHT,
    MAX_BID_ASK_SPREAD,
    MIN_LIQUIDITY_SCORE,
    MIN_RISK_REWARD_RATIO,
    OPTIMAL_DAYS_TO_EXPIRY,
)


@dataclass
class ScoringConfig:
    """Configuration for opportunity scoring weights"""

    risk_reward_weight: float = DEFAULT_RISK_REWARD_WEIGHT
    liquidity_weight: float = DEFAULT_LIQUIDITY_WEIGHT
    time_decay_weight: float = DEFAULT_TIME_DECAY_WEIGHT
    market_quality_weight: float = DEFAULT_MARKET_QUALITY_WEIGHT

    # Thresholds
    min_liquidity_score: float = MIN_LIQUIDITY_SCORE
    min_risk_reward_ratio: float = MIN_RISK_REWARD_RATIO
    max_bid_ask_spread: float = MAX_BID_ASK_SPREAD
    optimal_days_to_expiry: int = OPTIMAL_DAYS_TO_EXPIRY

    @classmethod
    def create_conservative(cls) -> "ScoringConfig":
        """Create conservative scoring configuration - prioritizes safety"""
        return cls(
            risk_reward_weight=0.30,
            liquidity_weight=0.35,
            time_decay_weight=0.25,
            market_quality_weight=0.10,
            min_liquidity_score=0.5,
            min_risk_reward_ratio=2.0,
            max_bid_ask_spread=15.0,
            optimal_days_to_expiry=25,
        )

    @classmethod
    def create_aggressive(cls) -> "ScoringConfig":
        """Create aggressive scoring configuration - prioritizes returns"""
        return cls(
            risk_reward_weight=0.50,
            liquidity_weight=0.15,
            time_decay_weight=0.20,
            market_quality_weight=0.15,
            min_liquidity_score=0.2,
            min_risk_reward_ratio=1.2,
            max_bid_ask_spread=25.0,
            optimal_days_to_expiry=35,
        )

    @classmethod
    def create_balanced(cls) -> "ScoringConfig":
        """Create balanced scoring configuration - default settings"""
        return cls()

    @classmethod
    def create_liquidity_focused(cls) -> "ScoringConfig":
        """Create liquidity-focused configuration - prioritizes execution certainty"""
        return cls(
            risk_reward_weight=0.25,
            liquidity_weight=0.40,
            time_decay_weight=0.15,
            market_quality_weight=0.20,
            min_liquidity_score=0.6,
            min_risk_reward_ratio=1.3,
            max_bid_ask_spread=12.0,
            optimal_days_to_expiry=28,
        )

    def validate(self) -> bool:
        """Validate that weights sum to approximately 1.0"""
        total_weight = (
            self.risk_reward_weight
            + self.liquidity_weight
            + self.time_decay_weight
            + self.market_quality_weight
        )
        return abs(total_weight - 1.0) < 0.01

    def normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total = (
            self.risk_reward_weight
            + self.liquidity_weight
            + self.time_decay_weight
            + self.market_quality_weight
        )
        if total > 0:
            self.risk_reward_weight /= total
            self.liquidity_weight /= total
            self.time_decay_weight /= total
            self.market_quality_weight /= total
