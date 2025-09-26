"""
Global opportunity management for Synthetic arbitrage strategy.

This module contains the GlobalOpportunityManager class responsible for:
- Collecting opportunities from all symbols
- Scoring and ranking opportunities
- Selecting the globally best opportunity
- Cycle management and reporting
"""

import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from ib_async import Contract, Order, Ticker

from ..common import get_logger
from .constants import (
    CREDIT_QUALITY_MULTIPLIER,
    QUALITY_SPREAD_COMPONENT_WEIGHT,
    QUALITY_VOLUME_COMPONENT_WEIGHT,
    SPREAD_QUALITY_MULTIPLIER,
    VOLUME_NORMALIZATION_THRESHOLD,
)
from .models import GlobalOpportunity, OpportunityScore
from .scoring import ScoringConfig


class GlobalOpportunityManager:
    """
    Manages collection, scoring, and selection of arbitrage opportunities across all symbols.
    Thread-safe implementation to handle concurrent opportunity submissions.
    """

    def __init__(self, scoring_config: ScoringConfig = None):
        self.scoring_config = scoring_config or ScoringConfig()
        self.opportunities: List[GlobalOpportunity] = []
        self.lock = threading.Lock()
        self.logger = get_logger()

        # Validate and normalize configuration
        if not self.scoring_config.validate():
            self.logger.warning(
                "Scoring configuration weights don't sum to 1.0, normalizing..."
            )
            self.scoring_config.normalize_weights()

        # Log the configuration being used
        self.logger.info(f"Initialized GlobalOpportunityManager with scoring config:")
        self.logger.info(f"  Risk-Reward: {self.scoring_config.risk_reward_weight:.2f}")
        self.logger.info(f"  Liquidity: {self.scoring_config.liquidity_weight:.2f}")
        self.logger.info(f"  Time Decay: {self.scoring_config.time_decay_weight:.2f}")
        self.logger.info(
            f"  Market Quality: {self.scoring_config.market_quality_weight:.2f}"
        )
        self.logger.info(
            f"  Min Risk-Reward Ratio: {self.scoring_config.min_risk_reward_ratio:.2f}"
        )
        self.logger.info(
            f"  Min Liquidity Score: {self.scoring_config.min_liquidity_score:.2f}"
        )

    def clear_opportunities(self):
        """Clear all collected opportunities for new cycle"""
        with self.lock:
            self.opportunities.clear()
            self.logger.debug("Cleared all opportunities for new cycle")

    def calculate_liquidity_score(
        self,
        call_volume: float,
        put_volume: float,
        call_spread: float,
        put_spread: float,
    ) -> float:
        """Calculate liquidity score based on volume and bid-ask spreads"""
        # Volume component (normalized to 0-1 scale)
        volume_score = min(
            1.0, (call_volume + put_volume) / VOLUME_NORMALIZATION_THRESHOLD
        )

        # Spread component (inverted - tighter spreads are better)
        avg_spread = (call_spread + put_spread) / 2.0
        spread_score = max(
            0.0, 1.0 - (avg_spread / self.scoring_config.max_bid_ask_spread)
        )

        # Combined liquidity score
        return (volume_score * QUALITY_VOLUME_COMPONENT_WEIGHT) + (
            spread_score * QUALITY_SPREAD_COMPONENT_WEIGHT
        )

    def calculate_time_decay_score(self, days_to_expiry: int) -> float:
        """Calculate time decay score - favor optimal time to expiration"""
        optimal_days = self.scoring_config.optimal_days_to_expiry

        if days_to_expiry <= 0:
            return 0.0

        # Score peaks at optimal days, decreases as we move away
        if days_to_expiry <= optimal_days:
            return days_to_expiry / optimal_days
        else:
            # Penalty for being too far out
            excess_days = days_to_expiry - optimal_days
            return max(0.1, 1.0 - (excess_days / (optimal_days * 2)))

    def calculate_market_quality_score(
        self, trade_details: Dict, call_spread: float, put_spread: float
    ) -> float:
        """Calculate market quality score based on spreads and pricing"""
        net_credit = trade_details.get("net_credit", 0)
        stock_price = trade_details.get("stock_price", 1)

        # Spread quality (tighter is better)
        avg_spread = (call_spread + put_spread) / 2.0
        spread_quality = max(
            0.0, 1.0 - (avg_spread / self.scoring_config.max_bid_ask_spread)
        )

        # Credit quality (positive credit is better)
        credit_quality = min(1.0, max(0.0, net_credit / (stock_price * 0.1)))

        return (spread_quality * SPREAD_QUALITY_MULTIPLIER) + (
            credit_quality * CREDIT_QUALITY_MULTIPLIER
        )

    def calculate_opportunity_score(
        self,
        trade_details: Dict,
        call_volume: float,
        put_volume: float,
        call_spread: float,
        put_spread: float,
        days_to_expiry: int,
    ) -> OpportunityScore:
        """Calculate comprehensive opportunity score"""

        # Risk-reward ratio
        max_profit = trade_details.get("max_profit", 0)
        min_profit = trade_details.get("min_profit", -1)
        risk_reward_ratio = max_profit / abs(min_profit) if min_profit != 0 else 0

        # Component scores
        liquidity_score = self.calculate_liquidity_score(
            call_volume, put_volume, call_spread, put_spread
        )
        time_decay_score = self.calculate_time_decay_score(days_to_expiry)
        market_quality_score = self.calculate_market_quality_score(
            trade_details, call_spread, put_spread
        )

        # Weighted composite score
        composite_score = (
            risk_reward_ratio * self.scoring_config.risk_reward_weight
            + liquidity_score * self.scoring_config.liquidity_weight
            + time_decay_score * self.scoring_config.time_decay_weight
            + market_quality_score * self.scoring_config.market_quality_weight
        )

        return OpportunityScore(
            risk_reward_ratio=risk_reward_ratio,
            liquidity_score=liquidity_score,
            time_decay_score=time_decay_score,
            market_quality_score=market_quality_score,
            composite_score=composite_score,
        )

    def add_opportunity(
        self,
        symbol: str,
        conversion_contract: Contract,
        order: Order,
        trade_details: Dict,
        call_ticker: Ticker,
        put_ticker: Ticker,
    ) -> bool:
        """Add an opportunity to the global collection"""

        # Calculate additional metadata
        call_volume = getattr(call_ticker, "volume", 0)
        put_volume = getattr(put_ticker, "volume", 0)
        call_spread = (
            abs(call_ticker.ask - call_ticker.bid)
            if (not np.isnan(call_ticker.ask) and not np.isnan(call_ticker.bid))
            else float("inf")
        )
        put_spread = (
            abs(put_ticker.ask - put_ticker.bid)
            if (not np.isnan(put_ticker.ask) and not np.isnan(put_ticker.bid))
            else float("inf")
        )

        # Calculate days to expiry
        try:
            expiry_str = trade_details.get("expiry", "")
            expiry_date = datetime.strptime(expiry_str, "%Y%m%d")
            days_to_expiry = (expiry_date - datetime.now()).days
        except (ValueError, TypeError):
            days_to_expiry = 0

        # Calculate opportunity score
        score = self.calculate_opportunity_score(
            trade_details,
            call_volume,
            put_volume,
            call_spread,
            put_spread,
            days_to_expiry,
        )

        # Apply minimum thresholds
        if (
            score.liquidity_score < self.scoring_config.min_liquidity_score
            or score.risk_reward_ratio < self.scoring_config.min_risk_reward_ratio
        ):
            self.logger.debug(
                f"[{symbol}] Opportunity rejected due to minimum thresholds: "
                f"liquidity={score.liquidity_score:.3f}, risk_reward={score.risk_reward_ratio:.3f}"
            )
            return False

        # Create global opportunity
        global_opportunity = GlobalOpportunity(
            symbol=symbol,
            conversion_contract=conversion_contract,
            order=order,
            trade_details=trade_details,
            score=score,
            timestamp=time.time(),
            call_volume=call_volume,
            put_volume=put_volume,
            call_bid_ask_spread=call_spread,
            put_bid_ask_spread=put_spread,
            days_to_expiry=days_to_expiry,
        )

        # Thread-safe addition
        with self.lock:
            self.opportunities.append(global_opportunity)
            self.logger.info(
                f"[{symbol}] Added opportunity with composite score: {score.composite_score:.3f} "
                f"(risk_reward: {score.risk_reward_ratio:.3f}, liquidity: {score.liquidity_score:.3f}, "
                f"time: {score.time_decay_score:.3f}, quality: {score.market_quality_score:.3f})"
            )

        return True

    def get_best_opportunity(self) -> Optional[GlobalOpportunity]:
        """Get the best opportunity based on composite score"""
        with self.lock:
            if not self.opportunities:
                return None

            # Sort by composite score (highest first)
            sorted_opportunities = sorted(
                self.opportunities,
                key=lambda opp: opp.score.composite_score,
                reverse=True,
            )

            best = sorted_opportunities[0]

            # Log comparison details
            self.logger.info(
                f"Global opportunity selection from {len(self.opportunities)} opportunities:"
            )
            for i, opp in enumerate(sorted_opportunities[:5]):  # Show top 5
                self.logger.info(
                    f"  #{i+1}: [{opp.symbol}] Score: {opp.score.composite_score:.3f} "
                    f"Expiry: {opp.trade_details.get('expiry', 'N/A')} "
                    f"Profit: ${opp.trade_details.get('max_profit', 0):.2f}"
                )

            return best

    def get_opportunity_count(self) -> int:
        """Get current number of collected opportunities"""
        with self.lock:
            return len(self.opportunities)

    def log_cycle_summary(self):
        """Log detailed summary of all opportunities in current cycle"""
        with self.lock:
            if not self.opportunities:
                self.logger.info("No opportunities collected in this cycle")
                return

            # Group opportunities by symbol
            by_symbol = defaultdict(list)
            for opp in self.opportunities:
                by_symbol[opp.symbol].append(opp)

            self.logger.info(
                f"=== CYCLE SUMMARY: {len(self.opportunities)} opportunities across {len(by_symbol)} symbols ==="
            )

            # Summary statistics
            scores = [opp.score.composite_score for opp in self.opportunities]
            risk_rewards = [opp.score.risk_reward_ratio for opp in self.opportunities]

            self.logger.info(
                f"Score Range: {min(scores):.3f} - {max(scores):.3f} (avg: {sum(scores)/len(scores):.3f})"
            )
            self.logger.info(
                f"Risk-Reward Range: {min(risk_rewards):.3f} - {max(risk_rewards):.3f} (avg: {sum(risk_rewards)/len(risk_rewards):.3f})"
            )

            # Per-symbol breakdown
            for symbol, symbol_opps in by_symbol.items():
                best_symbol_opp = max(
                    symbol_opps, key=lambda x: x.score.composite_score
                )
                self.logger.info(
                    f"  [{symbol}]: {len(symbol_opps)} opportunities, "
                    f"best score: {best_symbol_opp.score.composite_score:.3f} "
                    f"(expiry: {best_symbol_opp.trade_details.get('expiry', 'N/A')})"
                )

    def get_statistics(self) -> Dict:
        """Get statistical summary of current opportunities"""
        with self.lock:
            if not self.opportunities:
                return {}

            scores = [opp.score.composite_score for opp in self.opportunities]
            risk_rewards = [opp.score.risk_reward_ratio for opp in self.opportunities]
            liquidity_scores = [opp.score.liquidity_score for opp in self.opportunities]

            by_symbol = defaultdict(int)
            for opp in self.opportunities:
                by_symbol[opp.symbol] += 1

            return {
                "total_opportunities": len(self.opportunities),
                "unique_symbols": len(by_symbol),
                "score_stats": {
                    "min": min(scores),
                    "max": max(scores),
                    "avg": sum(scores) / len(scores),
                },
                "risk_reward_stats": {
                    "min": min(risk_rewards),
                    "max": max(risk_rewards),
                    "avg": sum(risk_rewards) / len(risk_rewards),
                },
                "liquidity_stats": {
                    "min": min(liquidity_scores),
                    "max": max(liquidity_scores),
                    "avg": sum(liquidity_scores) / len(liquidity_scores),
                },
                "opportunities_per_symbol": dict(by_symbol),
            }
