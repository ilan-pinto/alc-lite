"""
Risk validation for box spread opportunities.

This module provides comprehensive validation of box spread opportunities
to ensure they meet risk management criteria and are truly risk-free arbitrage.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..common import get_logger
from .models import BoxSpreadConfig, BoxSpreadLeg, BoxSpreadOpportunity
from .utils import (
    _safe_isnan,
    calculate_box_arbitrage_profit,
    calculate_box_greeks,
    validate_box_spread_legs,
)

logger = get_logger()


class BoxRiskValidator:
    """
    Comprehensive risk validator for box spread opportunities.

    Validates:
    - True arbitrage conditions
    - Greek exposure limits
    - Liquidity requirements
    - Early exercise risks
    - Execution feasibility
    """

    def __init__(self, config: BoxSpreadConfig):
        self.config = config

    def validate_opportunity(
        self, opportunity: BoxSpreadOpportunity
    ) -> Tuple[bool, List[str]]:
        """
        Perform comprehensive validation of a box spread opportunity.

        Args:
            opportunity: BoxSpreadOpportunity to validate

        Returns:
            Tuple of (is_valid, list_of_rejection_reasons)
        """
        rejection_reasons = []

        # 1. Validate arbitrage conditions
        if not self._validate_arbitrage_conditions(opportunity):
            rejection_reasons.append("Does not meet arbitrage profit requirements")

        # 2. Validate structure integrity
        if not self._validate_structure_integrity(opportunity):
            rejection_reasons.append("Invalid box spread structure")

        # 3. Validate Greeks exposure
        greek_issues = self._validate_greeks_exposure(opportunity)
        rejection_reasons.extend(greek_issues)

        # 4. Validate liquidity
        if not self._validate_liquidity_requirements(opportunity):
            rejection_reasons.append("Insufficient liquidity")

        # 5. Validate early exercise risk
        if not self._validate_early_exercise_risk(opportunity):
            rejection_reasons.append("High early exercise risk")

        # 6. Validate execution feasibility
        if not self._validate_execution_feasibility(opportunity):
            rejection_reasons.append("Execution not feasible")

        # 7. Validate expiry timing
        if not self._validate_expiry_timing(opportunity):
            rejection_reasons.append("Expiry timing outside acceptable range")

        is_valid = len(rejection_reasons) == 0

        if not is_valid:
            logger.debug(
                f"Box spread validation failed for {opportunity.symbol}: {', '.join(rejection_reasons)}"
            )

        return is_valid, rejection_reasons

    def _validate_arbitrage_conditions(self, opportunity: BoxSpreadOpportunity) -> bool:
        """Validate that the opportunity meets true arbitrage criteria"""

        # Must be truly risk-free if required
        if self.config.require_risk_free and not opportunity.risk_free:
            return False

        # Must meet minimum profit requirements
        if opportunity.arbitrage_profit < self.config.min_absolute_profit:
            return False

        if opportunity.profit_percentage < self.config.min_arbitrage_profit * 100:
            return False

        # Net debit must be within limits
        if opportunity.net_debit > self.config.max_net_debit:
            return False

        # Strike width must be within bounds
        if not (
            self.config.min_strike_width
            <= opportunity.strike_width
            <= self.config.max_strike_width
        ):
            return False

        return True

    def _validate_structure_integrity(self, opportunity: BoxSpreadOpportunity) -> bool:
        """Validate the structural integrity of the box spread"""

        # Verify strikes
        if opportunity.lower_strike >= opportunity.upper_strike:
            return False

        # Verify strike width matches calculation
        expected_width = opportunity.upper_strike - opportunity.lower_strike
        if abs(opportunity.strike_width - expected_width) > 0.001:
            return False

        # Validate individual legs exist and have valid data
        legs = [
            opportunity.long_call_k1,
            opportunity.short_call_k2,
            opportunity.short_put_k1,
            opportunity.long_put_k2,
        ]

        for leg in legs:
            if not self._validate_leg_data(leg):
                return False

        # Validate strikes match between legs
        if not validate_box_spread_legs(
            opportunity.long_call_k1.strike,
            opportunity.short_call_k2.strike,
            opportunity.short_put_k1.strike,
            opportunity.long_put_k2.strike,
            opportunity.long_call_k1.expiry,
            opportunity.short_call_k2.expiry,
            opportunity.short_put_k1.expiry,
            opportunity.long_put_k2.expiry,
        ):
            return False

        return True

    def _validate_leg_data(self, leg: BoxSpreadLeg) -> bool:
        """Validate individual leg data quality"""

        # Check for NaN values in critical fields
        critical_values = [leg.price, leg.bid, leg.ask, leg.strike]
        if any(_safe_isnan(val) for val in critical_values):
            return False

        # Validate price relationships
        if leg.bid > leg.ask or leg.bid <= 0 or leg.ask <= 0:
            return False

        # Price should be between bid and ask
        if not (leg.bid <= leg.price <= leg.ask):
            return False

        # Volume should be reasonable
        if leg.volume < self.config.min_volume_per_leg:
            return False

        return True

    def _validate_greeks_exposure(self, opportunity: BoxSpreadOpportunity) -> List[str]:
        """Validate that Greek exposures are within acceptable limits"""

        issues = []

        # Check each Greek against limits
        greek_limits = {
            "delta": self.config.max_greek_exposure,
            "gamma": self.config.max_greek_exposure,
            "theta": self.config.max_greek_exposure * 10,  # Theta is usually smaller
            "vega": self.config.max_greek_exposure * 10,  # Vega is usually smaller
        }

        greeks = {
            "delta": opportunity.net_delta,
            "gamma": opportunity.net_gamma,
            "theta": opportunity.net_theta,
            "vega": opportunity.net_vega,
        }

        for greek_name, value in greeks.items():
            if not _safe_isnan(value) and abs(value) > greek_limits[greek_name]:
                issues.append(f"Excessive {greek_name} exposure: {value:.4f}")

        return issues

    def _validate_liquidity_requirements(
        self, opportunity: BoxSpreadOpportunity
    ) -> bool:
        """Validate liquidity requirements are met"""

        # Check combined liquidity score
        if opportunity.combined_liquidity_score < self.config.min_liquidity_score:
            return False

        # Check total bid-ask spread
        max_total_spread = (
            opportunity.net_debit * self.config.max_bid_ask_spread_percent
        )
        if opportunity.total_bid_ask_spread > max_total_spread:
            return False

        # Check individual leg volumes
        legs = [
            opportunity.long_call_k1,
            opportunity.short_call_k2,
            opportunity.short_put_k1,
            opportunity.long_put_k2,
        ]

        for leg in legs:
            if leg.volume < self.config.min_volume_per_leg:
                return False

        return True

    def _validate_early_exercise_risk(self, opportunity: BoxSpreadOpportunity) -> bool:
        """Validate early exercise risk is acceptable"""

        if not self.config.early_exercise_protection:
            return True  # Skip validation if protection disabled

        # Check if any short options are deep ITM near expiry
        days_to_expiry = opportunity.time_to_expiry_days or 30  # Default if not set

        if days_to_expiry <= 7:  # Within a week of expiry
            # Get current stock price estimate (use strike midpoint as proxy)
            estimated_stock_price = (
                opportunity.lower_strike + opportunity.upper_strike
            ) / 2

            # Check short call (we're short at K2)
            if estimated_stock_price > opportunity.upper_strike:
                moneyness = (
                    estimated_stock_price - opportunity.upper_strike
                ) / opportunity.upper_strike
                if moneyness > 0.05:  # More than 5% ITM
                    return False

            # Check short put (we're short at K1)
            if estimated_stock_price < opportunity.lower_strike:
                moneyness = (
                    opportunity.lower_strike - estimated_stock_price
                ) / opportunity.lower_strike
                if moneyness > 0.05:  # More than 5% ITM
                    return False

        return True

    def _validate_execution_feasibility(
        self, opportunity: BoxSpreadOpportunity
    ) -> bool:
        """Validate that the opportunity can be reasonably executed"""

        # Check execution difficulty
        if opportunity.execution_difficulty > 0.8:  # Very difficult to execute
            return False

        # Verify we have 4 legs as expected
        if self.config.max_execution_legs < 4:
            return False

        # Check that safety buffer is reasonable
        safety_amount = opportunity.net_debit * self.config.safety_buffer
        if (
            safety_amount > opportunity.arbitrage_profit * 0.5
        ):  # Safety buffer too large
            return False

        return True

    def _validate_expiry_timing(self, opportunity: BoxSpreadOpportunity) -> bool:
        """Validate expiry timing is within acceptable range"""

        days_to_expiry = opportunity.time_to_expiry_days
        if days_to_expiry is None:
            # Try to calculate from expiry string
            try:
                expiry_date = datetime.strptime(opportunity.expiry, "%Y%m%d")
                days_to_expiry = (expiry_date - datetime.now()).days
            except:
                logger.warning(f"Could not parse expiry date: {opportunity.expiry}")
                return True  # Skip validation if we can't parse

        if days_to_expiry is not None:
            if not (
                self.config.min_days_to_expiry
                <= days_to_expiry
                <= self.config.max_days_to_expiry
            ):
                return False

        return True

    def get_risk_assessment_summary(self, opportunity: BoxSpreadOpportunity) -> Dict:
        """
        Get a detailed risk assessment summary for the opportunity.

        Returns:
            Dictionary with risk metrics and assessments
        """
        is_valid, rejection_reasons = self.validate_opportunity(opportunity)

        return {
            "is_valid": is_valid,
            "rejection_reasons": rejection_reasons,
            "risk_free": opportunity.risk_free,
            "arbitrage_profit": opportunity.arbitrage_profit,
            "profit_percentage": opportunity.profit_percentage,
            "net_debit": opportunity.net_debit,
            "liquidity_score": opportunity.combined_liquidity_score,
            "execution_difficulty": opportunity.execution_difficulty,
            "total_bid_ask_spread": opportunity.total_bid_ask_spread,
            "days_to_expiry": opportunity.time_to_expiry_days,
            "greek_exposure": {
                "delta": opportunity.net_delta,
                "gamma": opportunity.net_gamma,
                "theta": opportunity.net_theta,
                "vega": opportunity.net_vega,
            },
        }
