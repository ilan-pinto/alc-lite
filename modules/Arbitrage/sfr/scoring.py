"""
Scoring configurations and algorithms for SFR (Synthetic Free Risk) arbitrage strategy.

This module contains:
- SFRScoringConfig class with factory methods for different trading styles
- Scoring algorithms optimized for risk-free arbitrage opportunities
- Comprehensive logging of scoring decisions
- Weight normalization and validation
- Preset configurations (profit-focused, liquidity-focused, balanced, conservative)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..common import get_logger

logger = get_logger()


@dataclass
class SFRScoringConfig:
    """Configuration for SFR opportunity scoring weights.

    Since SFR is risk-free arbitrage, profit is the dominant factor,
    but execution quality (liquidity, spreads) and timing still matter.
    """

    # Scoring weights (must sum to 1.0)
    profit_weight: float = 0.50  # Guaranteed profit is most important for SFR
    liquidity_weight: float = 0.25  # Execution certainty
    spread_quality_weight: float = 0.15  # Bid-ask spread quality
    time_decay_weight: float = 0.10  # Time to expiry optimization

    # Thresholds
    min_liquidity_volume: int = 5  # Minimum volume for liquidity scoring
    max_bid_ask_spread: float = 20.0  # Maximum acceptable bid-ask spread
    optimal_days_to_expiry: int = 30  # Optimal time to expiration
    min_days_to_expiry: int = 15  # Minimum days to expiry
    max_days_to_expiry: int = 50  # Maximum days to expiry

    # Logging configuration
    enable_detailed_logging: bool = True  # Enable comprehensive scoring logs
    log_score_components: bool = True  # Log individual score components
    log_threshold: float = 0.0  # Only log opportunities above this score

    @classmethod
    def create_profit_focused(cls) -> "SFRScoringConfig":
        """Create profit-focused configuration - maximizes guaranteed returns.

        Best for: Traders prioritizing maximum profit per trade
        Trade-off: May miss opportunities with better execution certainty
        """
        config = cls(
            profit_weight=0.70,
            liquidity_weight=0.15,
            spread_quality_weight=0.10,
            time_decay_weight=0.05,
            min_liquidity_volume=3,
            max_bid_ask_spread=25.0,
            optimal_days_to_expiry=35,
        )
        logger.info("Created PROFIT-FOCUSED SFR scoring configuration")
        logger.info(
            f"  Weights: Profit={config.profit_weight:.0%}, Liquidity={config.liquidity_weight:.0%}, "
            f"Spreads={config.spread_quality_weight:.0%}, Time={config.time_decay_weight:.0%}"
        )
        return config

    @classmethod
    def create_liquidity_focused(cls) -> "SFRScoringConfig":
        """Create liquidity-focused configuration - prioritizes execution certainty.

        Best for: High-volume trading or when execution certainty is critical
        Trade-off: May accept lower profits for better fills
        """
        config = cls(
            profit_weight=0.35,
            liquidity_weight=0.35,
            spread_quality_weight=0.20,
            time_decay_weight=0.10,
            min_liquidity_volume=10,
            max_bid_ask_spread=15.0,
            optimal_days_to_expiry=28,
        )
        logger.info("Created LIQUIDITY-FOCUSED SFR scoring configuration")
        logger.info(
            f"  Weights: Profit={config.profit_weight:.0%}, Liquidity={config.liquidity_weight:.0%}, "
            f"Spreads={config.spread_quality_weight:.0%}, Time={config.time_decay_weight:.0%}"
        )
        return config

    @classmethod
    def create_balanced(cls) -> "SFRScoringConfig":
        """Create balanced scoring configuration - default settings.

        Best for: General purpose trading with balanced considerations
        Trade-off: No specific optimization, good all-around performance
        """
        config = cls()
        logger.info("Created BALANCED SFR scoring configuration (default)")
        logger.info(
            f"  Weights: Profit={config.profit_weight:.0%}, Liquidity={config.liquidity_weight:.0%}, "
            f"Spreads={config.spread_quality_weight:.0%}, Time={config.time_decay_weight:.0%}"
        )
        return config

    @classmethod
    def create_conservative(cls) -> "SFRScoringConfig":
        """Create conservative configuration - emphasizes quality over profit.

        Best for: Risk-averse traders or volatile market conditions
        Trade-off: May miss profitable but lower-quality opportunities
        """
        config = cls(
            profit_weight=0.40,
            liquidity_weight=0.30,
            spread_quality_weight=0.20,
            time_decay_weight=0.10,
            min_liquidity_volume=20,
            max_bid_ask_spread=10.0,
            optimal_days_to_expiry=25,
            min_days_to_expiry=20,
            max_days_to_expiry=40,
        )
        logger.info("Created CONSERVATIVE SFR scoring configuration")
        logger.info(
            f"  Weights: Profit={config.profit_weight:.0%}, Liquidity={config.liquidity_weight:.0%}, "
            f"Spreads={config.spread_quality_weight:.0%}, Time={config.time_decay_weight:.0%}"
        )
        return config

    def validate(self) -> bool:
        """Validate that weights sum to approximately 1.0"""
        total_weight = (
            self.profit_weight
            + self.liquidity_weight
            + self.spread_quality_weight
            + self.time_decay_weight
        )
        is_valid = abs(total_weight - 1.0) < 0.01

        if not is_valid:
            logger.warning(
                f"SFR scoring weights sum to {total_weight:.3f} instead of 1.0"
            )

        return is_valid

    def normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total = (
            self.profit_weight
            + self.liquidity_weight
            + self.spread_quality_weight
            + self.time_decay_weight
        )
        if total > 0:
            self.profit_weight /= total
            self.liquidity_weight /= total
            self.spread_quality_weight /= total
            self.time_decay_weight /= total

            logger.info("Normalized SFR scoring weights to sum to 1.0")
            logger.info(
                f"  New weights: Profit={self.profit_weight:.3f}, Liquidity={self.liquidity_weight:.3f}, "
                f"Spreads={self.spread_quality_weight:.3f}, Time={self.time_decay_weight:.3f}"
            )


class SFRScoringEngine:
    """Scoring engine for SFR opportunities with comprehensive logging."""

    def __init__(self, config: Optional[SFRScoringConfig] = None):
        self.config = config or SFRScoringConfig()
        self.logger = logger

        # Validate and normalize configuration
        if not self.config.validate():
            self.logger.warning(
                "SFR scoring configuration invalid, normalizing weights..."
            )
            self.config.normalize_weights()

        self.logger.info(f"Initialized SFR Scoring Engine")
        if self.config.enable_detailed_logging:
            self.logger.info(f"  Detailed logging: ENABLED")
            self.logger.info(f"  Score threshold: {self.config.log_threshold:.3f}")

    def calculate_profit_score(
        self, guaranteed_profit: float, max_observed_profit: float
    ) -> Tuple[float, str]:
        """Calculate normalized profit score (0-1 scale).

        Args:
            guaranteed_profit: The guaranteed profit for this opportunity
            max_observed_profit: The maximum profit observed across all opportunities

        Returns:
            Tuple of (score, explanation)
        """
        if max_observed_profit <= 0:
            return 0.0, "No positive profits observed"

        # Normalize profit to 0-1 scale
        score = min(1.0, guaranteed_profit / max_observed_profit)

        # Apply non-linear scaling to reward higher profits more
        score = np.power(score, 0.8)  # Slight curve to favor higher profits

        explanation = f"Profit ${guaranteed_profit:.2f} / max ${max_observed_profit:.2f} = {score:.3f}"

        return score, explanation

    def calculate_liquidity_score(
        self,
        call_volume: float,
        put_volume: float,
        stock_volume: float,
        call_open_interest: float = 0,
        put_open_interest: float = 0,
    ) -> Tuple[float, str]:
        """Calculate liquidity score based on volume and open interest.

        Returns:
            Tuple of (score, explanation)
        """
        # Minimum volume check
        min_vol = min(call_volume, put_volume)
        if min_vol < self.config.min_liquidity_volume:
            score = min_vol / self.config.min_liquidity_volume * 0.5  # Harsh penalty
            explanation = (
                f"Below minimum volume ({min_vol} < {self.config.min_liquidity_volume})"
            )
            return score, explanation

        # Volume score (log scale for better distribution)
        volume_score = min(1.0, np.log10(min_vol + 1) / 3)  # log10(1000) = 3

        # Open interest bonus (if available)
        oi_bonus = 0.0
        if call_open_interest > 0 and put_open_interest > 0:
            min_oi = min(call_open_interest, put_open_interest)
            oi_bonus = min(0.2, np.log10(min_oi + 1) / 4)  # Up to 20% bonus

        score = min(1.0, volume_score + oi_bonus)

        explanation = (
            f"Vol(C:{call_volume:.0f},P:{put_volume:.0f}), score={volume_score:.3f}"
        )
        if oi_bonus > 0:
            explanation += f", OI bonus={oi_bonus:.3f}"

        return score, explanation

    def calculate_spread_quality_score(
        self,
        call_bid_ask_spread: float,
        put_bid_ask_spread: float,
        stock_bid_ask_spread: float,
        call_price: float,
        put_price: float,
        stock_price: float,
    ) -> Tuple[float, str]:
        """Calculate spread quality score.

        Lower spreads = higher quality = better execution

        Returns:
            Tuple of (score, explanation)
        """
        # Calculate spread as percentage of price
        call_spread_pct = (
            call_bid_ask_spread / max(call_price, 0.01) if call_price > 0 else 1.0
        )
        put_spread_pct = (
            put_bid_ask_spread / max(put_price, 0.01) if put_price > 0 else 1.0
        )
        stock_spread_pct = (
            stock_bid_ask_spread / max(stock_price, 0.01) if stock_price > 0 else 0.01
        )

        # Weight options spreads more heavily (they matter more for execution)
        weighted_spread_pct = (
            call_spread_pct * 0.4 + put_spread_pct * 0.4 + stock_spread_pct * 0.2
        )

        # Convert to score (lower spread = higher score)
        # Using exponential decay for smooth scoring
        score = np.exp(-weighted_spread_pct * 10)  # Fast decay for wide spreads

        explanation = f"Spreads: C:{call_spread_pct:.1%}, P:{put_spread_pct:.1%}, S:{stock_spread_pct:.1%}"

        # Check against maximum threshold
        max_spread = max(call_bid_ask_spread, put_bid_ask_spread)
        if max_spread > self.config.max_bid_ask_spread:
            score *= 0.5  # Heavy penalty
            explanation += f" [WIDE: ${max_spread:.2f}]"

        return score, explanation

    def calculate_time_decay_score(self, days_to_expiry: int) -> Tuple[float, str]:
        """Calculate time decay score - favor optimal time to expiration.

        Returns:
            Tuple of (score, explanation)
        """
        optimal = self.config.optimal_days_to_expiry

        if days_to_expiry < self.config.min_days_to_expiry:
            score = days_to_expiry / self.config.min_days_to_expiry * 0.5
            explanation = (
                f"Too short: {days_to_expiry}d < {self.config.min_days_to_expiry}d min"
            )
        elif days_to_expiry > self.config.max_days_to_expiry:
            excess = days_to_expiry - self.config.max_days_to_expiry
            score = max(0.3, 1.0 - excess / 30)  # Gradual decay after max
            explanation = (
                f"Too long: {days_to_expiry}d > {self.config.max_days_to_expiry}d max"
            )
        else:
            # Bell curve centered at optimal
            distance_from_optimal = abs(days_to_expiry - optimal)
            score = np.exp(
                -(distance_from_optimal**2) / (2 * 10**2)
            )  # Gaussian with σ=10
            explanation = f"{days_to_expiry}d (optimal={optimal}d)"

        return score, explanation

    def calculate_composite_score(
        self,
        symbol: str,
        expiry: str,
        guaranteed_profit: float,
        max_observed_profit: float,
        call_volume: float,
        put_volume: float,
        stock_volume: float,
        call_bid: float,
        call_ask: float,
        put_bid: float,
        put_ask: float,
        stock_bid: float,
        stock_ask: float,
        days_to_expiry: int,
        call_strike: float,
        put_strike: float,
        call_open_interest: float = 0,
        put_open_interest: float = 0,
    ) -> Dict:
        """Calculate comprehensive composite score with detailed logging.

        Returns:
            Dictionary with score, components, and detailed explanations
        """
        # Calculate individual components
        profit_score, profit_exp = self.calculate_profit_score(
            guaranteed_profit, max_observed_profit
        )

        liquidity_score, liquidity_exp = self.calculate_liquidity_score(
            call_volume, put_volume, stock_volume, call_open_interest, put_open_interest
        )

        # Calculate spreads
        call_spread = call_ask - call_bid
        put_spread = put_ask - put_bid
        stock_spread = stock_ask - stock_bid
        call_mid = (call_ask + call_bid) / 2
        put_mid = (put_ask + put_bid) / 2
        stock_mid = (stock_ask + stock_bid) / 2

        spread_score, spread_exp = self.calculate_spread_quality_score(
            call_spread, put_spread, stock_spread, call_mid, put_mid, stock_mid
        )

        time_score, time_exp = self.calculate_time_decay_score(days_to_expiry)

        # Calculate weighted composite score
        composite_score = (
            profit_score * self.config.profit_weight
            + liquidity_score * self.config.liquidity_weight
            + spread_score * self.config.spread_quality_weight
            + time_score * self.config.time_decay_weight
        )

        # Prepare result dictionary
        result = {
            "symbol": symbol,
            "expiry": expiry,
            "composite_score": composite_score,
            "components": {
                "profit": {
                    "score": profit_score,
                    "weight": self.config.profit_weight,
                    "weighted": profit_score * self.config.profit_weight,
                },
                "liquidity": {
                    "score": liquidity_score,
                    "weight": self.config.liquidity_weight,
                    "weighted": liquidity_score * self.config.liquidity_weight,
                },
                "spread_quality": {
                    "score": spread_score,
                    "weight": self.config.spread_quality_weight,
                    "weighted": spread_score * self.config.spread_quality_weight,
                },
                "time_decay": {
                    "score": time_score,
                    "weight": self.config.time_decay_weight,
                    "weighted": time_score * self.config.time_decay_weight,
                },
            },
            "explanations": {
                "profit": profit_exp,
                "liquidity": liquidity_exp,
                "spread_quality": spread_exp,
                "time_decay": time_exp,
            },
            "details": {
                "guaranteed_profit": guaranteed_profit,
                "call_strike": call_strike,
                "put_strike": put_strike,
                "days_to_expiry": days_to_expiry,
                "call_volume": call_volume,
                "put_volume": put_volume,
            },
        }

        # Comprehensive logging if enabled
        if (
            self.config.enable_detailed_logging
            and composite_score >= self.config.log_threshold
        ):
            self._log_scoring_decision(result)

        return result

    def _log_scoring_decision(self, score_result: Dict):
        """Log comprehensive scoring decision details."""
        symbol = score_result["symbol"]
        expiry = score_result["expiry"]
        composite = score_result["composite_score"]
        components = score_result["components"]
        explanations = score_result["explanations"]
        details = score_result["details"]

        self.logger.info(
            f"[SFR SCORING] {symbol} {expiry} | Score: {composite:.3f} | Profit: ${details['guaranteed_profit']:.2f}"
        )

        if self.config.log_score_components:
            self.logger.info(f"  Component Scores:")
            self.logger.info(
                f"    Profit:    {components['profit']['score']:.3f} × {components['profit']['weight']:.2f} = {components['profit']['weighted']:.3f} | {explanations['profit']}"
            )
            self.logger.info(
                f"    Liquidity: {components['liquidity']['score']:.3f} × {components['liquidity']['weight']:.2f} = {components['liquidity']['weighted']:.3f} | {explanations['liquidity']}"
            )
            self.logger.info(
                f"    Spreads:   {components['spread_quality']['score']:.3f} × {components['spread_quality']['weight']:.2f} = {components['spread_quality']['weighted']:.3f} | {explanations['spread_quality']}"
            )
            self.logger.info(
                f"    Time:      {components['time_decay']['score']:.3f} × {components['time_decay']['weight']:.2f} = {components['time_decay']['weighted']:.3f} | {explanations['time_decay']}"
            )
            self.logger.info(f"    ─────────────────────────────────────────")
            self.logger.info(f"    TOTAL:     {composite:.3f}")

        # Log strike details
        self.logger.info(
            f"  Strikes: Call=${details['call_strike']:.2f}, Put=${details['put_strike']:.2f}"
        )

    def rank_opportunities(self, opportunities: list) -> list:
        """Rank a list of opportunity score results.

        Args:
            opportunities: List of score result dictionaries

        Returns:
            Sorted list with ranking information added
        """
        # Sort by composite score (descending)
        sorted_opps = sorted(
            opportunities, key=lambda x: x["composite_score"], reverse=True
        )

        # Add ranking information
        for rank, opp in enumerate(sorted_opps, 1):
            opp["rank"] = rank

        # Log ranking summary
        if self.config.enable_detailed_logging and sorted_opps:
            self.logger.info(f"[SFR SCORING] Ranked {len(sorted_opps)} opportunities:")
            for opp in sorted_opps[:5]:  # Show top 5
                self.logger.info(
                    f"  #{opp['rank']}: {opp['symbol']} {opp['expiry']} | "
                    f"Score: {opp['composite_score']:.3f} | "
                    f"Profit: ${opp['details']['guaranteed_profit']:.2f}"
                )

        return sorted_opps
