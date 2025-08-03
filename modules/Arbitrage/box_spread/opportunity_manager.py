"""
Box spread opportunity detection and management.

This module provides the BoxOpportunityManager class which handles:
- Detection of box spread opportunities
- Ranking and scoring
- Global opportunity selection
- Performance tracking
"""

import asyncio
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..common import get_logger
from ..metrics import RejectionReason, metrics_collector
from .models import BoxSpreadConfig, BoxSpreadLeg, BoxSpreadOpportunity
from .risk_validator import BoxRiskValidator
from .utils import (
    TTLCache,
    _safe_isnan,
    calculate_box_arbitrage_profit,
    calculate_box_greeks,
    calculate_execution_difficulty,
    calculate_liquidity_score,
    format_box_spread_summary,
)

logger = get_logger()


class BoxOpportunityManager:
    """
    Manages box spread opportunity detection and ranking.

    Provides global opportunity selection to find the best box spread
    arbitrage across multiple symbols and strike combinations.
    """

    def __init__(self, config: BoxSpreadConfig = None):
        self.config = config or BoxSpreadConfig()
        self.validator = BoxRiskValidator(self.config)

        # Opportunity tracking
        self.opportunities: List[BoxSpreadOpportunity] = []
        self.rejected_opportunities: Dict[str, List[str]] = {}

        # Performance tracking
        self.scan_start_time = None
        self.total_combinations_evaluated = 0
        self.total_opportunities_found = 0

        # Caching for performance
        if self.config.enable_caching:
            self.leg_cache = TTLCache(
                max_size=1000, ttl_seconds=self.config.cache_ttl_seconds
            )
            self.greeks_cache = TTLCache(
                max_size=500, ttl_seconds=self.config.cache_ttl_seconds
            )
        else:
            self.leg_cache = None
            self.greeks_cache = None

    def start_scan(self) -> None:
        """Initialize a new opportunity scan"""
        self.scan_start_time = time.time()
        self.opportunities.clear()
        self.rejected_opportunities.clear()
        self.total_combinations_evaluated = 0
        self.total_opportunities_found = 0

        logger.info("Starting box spread opportunity scan")

    def evaluate_box_spread(
        self,
        symbol: str,
        k1_strike: float,
        k2_strike: float,
        expiry: str,
        call_k1_data: dict,
        call_k2_data: dict,
        put_k1_data: dict,
        put_k2_data: dict,
    ) -> Optional[BoxSpreadOpportunity]:
        """
        Evaluate a potential box spread opportunity.

        Args:
            symbol: Trading symbol
            k1_strike: Lower strike price
            k2_strike: Upper strike price
            expiry: Expiration date
            call/put_data: Market data dictionaries with keys: bid, ask, volume, iv, greeks

        Returns:
            BoxSpreadOpportunity if valid, None if rejected
        """
        self.total_combinations_evaluated += 1

        try:
            # Create leg objects
            legs = self._create_box_spread_legs(
                symbol,
                k1_strike,
                k2_strike,
                expiry,
                call_k1_data,
                call_k2_data,
                put_k1_data,
                put_k2_data,
            )

            if not legs:
                self._record_rejection(
                    symbol, f"K1={k1_strike}, K2={k2_strike}", ["Invalid leg data"]
                )
                return None

            long_call_k1, short_call_k2, short_put_k1, long_put_k2 = legs

            # Calculate arbitrage metrics
            arbitrage_metrics = self._calculate_arbitrage_metrics(legs)
            if not arbitrage_metrics:
                self._record_rejection(
                    symbol,
                    f"K1={k1_strike}, K2={k2_strike}",
                    ["Failed arbitrage calculation"],
                )
                return None

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(legs)

            # Calculate Greeks
            greeks = self._calculate_combined_greeks(legs)

            # Create opportunity object
            opportunity = BoxSpreadOpportunity(
                symbol=symbol,
                lower_strike=k1_strike,
                upper_strike=k2_strike,
                expiry=expiry,
                long_call_k1=long_call_k1,
                short_call_k2=short_call_k2,
                short_put_k1=short_put_k1,
                long_put_k2=long_put_k2,
                strike_width=k2_strike - k1_strike,
                **arbitrage_metrics,
                **quality_metrics,
                **greeks,
                composite_score=0.0,  # Will be calculated later
            )

            # Validate opportunity
            is_valid, rejection_reasons = self.validator.validate_opportunity(
                opportunity
            )

            if not is_valid:
                self._record_rejection(
                    symbol, f"K1={k1_strike}, K2={k2_strike}", rejection_reasons
                )
                return None

            # Calculate composite score
            opportunity.composite_score = self._calculate_composite_score(opportunity)

            # Track metrics
            metrics_collector.record_opportunity_found()

            # Log successful opportunity found
            logger.info(
                f"[{symbol}] OPPORTUNITY FOUND - Box spread K1={k1_strike}, K2={k2_strike}: "
                f"profit=${opportunity.arbitrage_profit:.4f}, score={opportunity.composite_score:.3f}"
            )

            self.total_opportunities_found += 1
            logger.debug(
                f"Found valid box spread: {format_box_spread_summary(opportunity)}"
            )

            return opportunity

        except Exception as e:
            logger.error(
                f"Error evaluating box spread {symbol} K1={k1_strike} K2={k2_strike}: {e}"
            )
            self._record_rejection(
                symbol,
                f"K1={k1_strike}, K2={k2_strike}",
                [f"Evaluation error: {str(e)}"],
            )
            return None

    def _create_box_spread_legs(
        self,
        symbol: str,
        k1_strike: float,
        k2_strike: float,
        expiry: str,
        call_k1_data: dict,
        call_k2_data: dict,
        put_k1_data: dict,
        put_k2_data: dict,
    ) -> Optional[Tuple[BoxSpreadLeg, BoxSpreadLeg, BoxSpreadLeg, BoxSpreadLeg]]:
        """Create the 4 legs of the box spread"""

        try:
            # Long call at K1
            long_call_k1 = BoxSpreadLeg(
                contract=call_k1_data.get("contract"),
                strike=k1_strike,
                expiry=expiry,
                right="C",
                action="BUY",
                price=(call_k1_data["bid"] + call_k1_data["ask"]) / 2,
                bid=call_k1_data["bid"],
                ask=call_k1_data["ask"],
                volume=call_k1_data.get("volume", 0),
                iv=call_k1_data.get("iv", 0),
                delta=call_k1_data.get("delta", 0),
                gamma=call_k1_data.get("gamma", 0),
                theta=call_k1_data.get("theta", 0),
                vega=call_k1_data.get("vega", 0),
                days_to_expiry=call_k1_data.get("days_to_expiry", 30),
            )

            # Short call at K2
            short_call_k2 = BoxSpreadLeg(
                contract=call_k2_data.get("contract"),
                strike=k2_strike,
                expiry=expiry,
                right="C",
                action="SELL",
                price=(call_k2_data["bid"] + call_k2_data["ask"]) / 2,
                bid=call_k2_data["bid"],
                ask=call_k2_data["ask"],
                volume=call_k2_data.get("volume", 0),
                iv=call_k2_data.get("iv", 0),
                delta=call_k2_data.get("delta", 0),
                gamma=call_k2_data.get("gamma", 0),
                theta=call_k2_data.get("theta", 0),
                vega=call_k2_data.get("vega", 0),
                days_to_expiry=call_k2_data.get("days_to_expiry", 30),
            )

            # Short put at K1
            short_put_k1 = BoxSpreadLeg(
                contract=put_k1_data.get("contract"),
                strike=k1_strike,
                expiry=expiry,
                right="P",
                action="SELL",
                price=(put_k1_data["bid"] + put_k1_data["ask"]) / 2,
                bid=put_k1_data["bid"],
                ask=put_k1_data["ask"],
                volume=put_k1_data.get("volume", 0),
                iv=put_k1_data.get("iv", 0),
                delta=put_k1_data.get("delta", 0),
                gamma=put_k1_data.get("gamma", 0),
                theta=put_k1_data.get("theta", 0),
                vega=put_k1_data.get("vega", 0),
                days_to_expiry=put_k1_data.get("days_to_expiry", 30),
            )

            # Long put at K2
            long_put_k2 = BoxSpreadLeg(
                contract=put_k2_data.get("contract"),
                strike=k2_strike,
                expiry=expiry,
                right="P",
                action="BUY",
                price=(put_k2_data["bid"] + put_k2_data["ask"]) / 2,
                bid=put_k2_data["bid"],
                ask=put_k2_data["ask"],
                volume=put_k2_data.get("volume", 0),
                iv=put_k2_data.get("iv", 0),
                delta=put_k2_data.get("delta", 0),
                gamma=put_k2_data.get("gamma", 0),
                theta=put_k2_data.get("theta", 0),
                vega=put_k2_data.get("vega", 0),
                days_to_expiry=put_k2_data.get("days_to_expiry", 30),
            )

            return long_call_k1, short_call_k2, short_put_k1, long_put_k2

        except (KeyError, TypeError, ValueError) as e:
            logger.debug(f"Failed to create box spread legs for {symbol}: {e}")
            return None

    def _calculate_arbitrage_metrics(
        self, legs: Tuple[BoxSpreadLeg, ...]
    ) -> Optional[dict]:
        """Calculate arbitrage profit and related metrics"""

        long_call_k1, short_call_k2, short_put_k1, long_put_k2 = legs

        try:
            # Calculate arbitrage profit using our execution prices
            # For box spread: pay ask for long positions, receive bid for short positions
            long_call_price = long_call_k1.ask  # We buy at ask
            short_call_price = short_call_k2.bid  # We sell at bid
            short_put_price = short_put_k1.bid  # We sell at bid
            long_put_price = long_put_k2.ask  # We buy at ask

            strike_width = long_put_k2.strike - long_call_k1.strike

            net_debit, arbitrage_profit, is_risk_free = calculate_box_arbitrage_profit(
                long_call_price,
                short_call_price,
                short_put_price,
                long_put_price,
                strike_width,
                self.config.safety_buffer,
            )

            profit_percentage = (arbitrage_profit / max(net_debit, 0.01)) * 100

            return {
                "net_debit": round(net_debit, self.config.price_precision_decimals),
                "theoretical_value": round(
                    strike_width, self.config.price_precision_decimals
                ),
                "arbitrage_profit": round(
                    arbitrage_profit, self.config.price_precision_decimals
                ),
                "profit_percentage": round(profit_percentage, 2),
                "max_profit": round(
                    arbitrage_profit, self.config.price_precision_decimals
                ),
                "max_loss": round(
                    max(0, -arbitrage_profit), self.config.price_precision_decimals
                ),
                "risk_free": is_risk_free,
            }

        except Exception as e:
            logger.debug(f"Failed to calculate arbitrage metrics: {e}")
            return None

    def _calculate_quality_metrics(self, legs: Tuple[BoxSpreadLeg, ...]) -> dict:
        """Calculate liquidity and execution quality metrics"""

        long_call_k1, short_call_k2, short_put_k1, long_put_k2 = legs

        # Calculate total bid-ask spread
        total_spread = sum(
            [
                long_call_k1.ask - long_call_k1.bid,
                short_call_k2.ask - short_call_k2.bid,
                short_put_k1.ask - short_put_k1.bid,
                long_put_k2.ask - long_put_k2.bid,
            ]
        )

        # Calculate liquidity score
        liquidity_score = calculate_liquidity_score(
            long_call_k1.volume,
            short_call_k2.volume,
            short_put_k1.volume,
            long_put_k2.volume,
            (long_call_k1.ask - long_call_k1.bid) / max(long_call_k1.price, 0.01),
            (short_call_k2.ask - short_call_k2.bid) / max(short_call_k2.price, 0.01),
            (short_put_k1.ask - short_put_k1.bid) / max(short_put_k1.price, 0.01),
            (long_put_k2.ask - long_put_k2.bid) / max(long_put_k2.price, 0.01),
        )

        # Calculate execution difficulty
        execution_difficulty = calculate_execution_difficulty(
            4, total_spread, liquidity_score
        )

        return {
            "total_bid_ask_spread": round(
                total_spread, self.config.price_precision_decimals
            ),
            "combined_liquidity_score": round(liquidity_score, 3),
            "execution_difficulty": round(execution_difficulty, 3),
        }

    def _calculate_combined_greeks(self, legs: Tuple[BoxSpreadLeg, ...]) -> dict:
        """Calculate net Greeks for the box spread position"""

        long_call_k1, short_call_k2, short_put_k1, long_put_k2 = legs

        # Create Greeks dictionaries for each leg
        call_k1_greeks = {
            "delta": long_call_k1.delta,
            "gamma": long_call_k1.gamma,
            "theta": long_call_k1.theta,
            "vega": long_call_k1.vega,
        }

        call_k2_greeks = {
            "delta": short_call_k2.delta,
            "gamma": short_call_k2.gamma,
            "theta": short_call_k2.theta,
            "vega": short_call_k2.vega,
        }

        put_k1_greeks = {
            "delta": short_put_k1.delta,
            "gamma": short_put_k1.gamma,
            "theta": short_put_k1.theta,
            "vega": short_put_k1.vega,
        }

        put_k2_greeks = {
            "delta": long_put_k2.delta,
            "gamma": long_put_k2.gamma,
            "theta": long_put_k2.theta,
            "vega": long_put_k2.vega,
        }

        net_greeks = calculate_box_greeks(
            call_k1_greeks, call_k2_greeks, put_k1_greeks, put_k2_greeks
        )

        return {
            "net_delta": round(net_greeks.get("delta", 0), 4),
            "net_gamma": round(net_greeks.get("gamma", 0), 4),
            "net_theta": round(net_greeks.get("theta", 0), 4),
            "net_vega": round(net_greeks.get("vega", 0), 4),
        }

    def _calculate_composite_score(self, opportunity: BoxSpreadOpportunity) -> float:
        """
        Calculate composite score for ranking opportunities.

        Score components:
        - Arbitrage profit percentage (50%)
        - Liquidity score (25%)
        - Low execution difficulty (15%)
        - Low Greek exposure (10%)
        """

        try:
            # Normalize profit percentage (0-100% -> 0-1)
            profit_score = min(1.0, opportunity.profit_percentage / 100.0)

            # Liquidity score (already 0-1)
            liquidity_score = opportunity.combined_liquidity_score

            # Execution difficulty (invert so lower difficulty = higher score)
            execution_score = 1.0 - opportunity.execution_difficulty

            # Greek exposure (lower exposure = higher score)
            max_greek_exposure = max(
                abs(opportunity.net_delta),
                abs(opportunity.net_gamma),
                abs(opportunity.net_theta) / 10,  # Scale theta
                abs(opportunity.net_vega) / 10,  # Scale vega
            )
            greek_score = max(
                0.0, 1.0 - (max_greek_exposure / self.config.max_greek_exposure)
            )

            # Weighted composite score
            composite_score = (
                profit_score * 0.50
                + liquidity_score * 0.25
                + execution_score * 0.15
                + greek_score * 0.10
            )

            return max(0.0, min(1.0, composite_score))

        except Exception as e:
            logger.debug(f"Failed to calculate composite score: {e}")
            return 0.0

    def add_opportunity(self, opportunity: BoxSpreadOpportunity) -> None:
        """Add a validated opportunity to the manager"""
        self.opportunities.append(opportunity)

    def get_best_opportunities(self, max_count: int = 10) -> List[BoxSpreadOpportunity]:
        """Get the best opportunities sorted by composite score"""
        sorted_opportunities = sorted(
            self.opportunities, key=lambda x: x.composite_score, reverse=True
        )
        return sorted_opportunities[:max_count]

    def get_best_opportunity(self) -> Optional[BoxSpreadOpportunity]:
        """Get the single best opportunity"""
        best_opportunities = self.get_best_opportunities(1)
        return best_opportunities[0] if best_opportunities else None

    def _record_rejection(
        self, symbol: str, combination: str, reasons: List[str]
    ) -> None:
        """Record a rejected opportunity for analysis"""
        key = f"{symbol}_{combination}"
        self.rejected_opportunities[key] = reasons

        # Log rejection with details
        reason_text = ", ".join(reasons)
        logger.warning(f"[{symbol}] REJECTED - Box spread {combination}: {reason_text}")

        # Record metrics - use appropriate rejection reason
        for reason in reasons:
            # Map rejection reasons to appropriate enum values
            if "profit" in reason.lower() or "target" in reason.lower():
                rejection_reason = RejectionReason.PROFIT_TARGET_NOT_MET
            elif "strike" in reason.lower() or "combination" in reason.lower():
                rejection_reason = RejectionReason.INVALID_STRIKE_COMBINATION
            elif "liquidity" in reason.lower() or "volume" in reason.lower():
                rejection_reason = RejectionReason.VOLUME_TOO_LOW
            elif "spread" in reason.lower():
                rejection_reason = RejectionReason.BID_ASK_SPREAD_TOO_WIDE
            else:
                rejection_reason = RejectionReason.INVALID_CONTRACT_DATA

            metrics_collector.add_rejection_reason(
                rejection_reason,
                {"symbol": symbol, "combination": combination, "reason": reason},
            )

    def get_scan_summary(self) -> dict:
        """Get summary statistics for the scan"""
        scan_duration = (
            time.time() - self.scan_start_time if self.scan_start_time else 0
        )

        return {
            "scan_duration_seconds": round(scan_duration, 2),
            "combinations_evaluated": self.total_combinations_evaluated,
            "opportunities_found": self.total_opportunities_found,
            "opportunities_validated": len(self.opportunities),
            "rejection_rate": (
                (self.total_combinations_evaluated - self.total_opportunities_found)
                / max(self.total_combinations_evaluated, 1)
            )
            * 100,
            "avg_evaluation_time_ms": (
                (scan_duration * 1000) / max(self.total_combinations_evaluated, 1)
            ),
            "best_score": (
                max(opp.composite_score for opp in self.opportunities)
                if self.opportunities
                else 0.0
            ),
        }
