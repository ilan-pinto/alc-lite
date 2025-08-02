"""
Opportunity manager for calendar spread strategy.

This module contains the CalendarSpreadOpportunityManager class that handles
collection, scoring, and selection of calendar spread opportunities across symbols.
"""

import threading
from collections import defaultdict
from typing import List, Optional

from ..common import get_logger
from .models import CalendarSpreadOpportunity


class CalendarSpreadOpportunityManager:
    """
    Manages collection, scoring, and selection of calendar spread opportunities across all symbols.
    Thread-safe implementation to handle concurrent opportunity submissions.
    """

    def __init__(self):
        self.opportunities: List[CalendarSpreadOpportunity] = []
        self.lock = threading.Lock()
        self.logger = get_logger()

    def clear_opportunities(self):
        """Clear all collected opportunities for new cycle"""
        with self.lock:
            self.opportunities.clear()
            self.logger.debug("Cleared all calendar spread opportunities for new cycle")

    def add_opportunity(
        self, symbol: str, opportunity: CalendarSpreadOpportunity
    ) -> bool:
        """
        Add a calendar spread opportunity to the collection.

        Args:
            symbol: Trading symbol
            opportunity: Calendar spread opportunity to add

        Returns:
            bool: True if opportunity was added successfully
        """
        try:
            with self.lock:
                self.opportunities.append(opportunity)
                self.logger.debug(
                    f"Added calendar spread opportunity for {symbol}: "
                    f"IV spread {opportunity.iv_spread:.1f}%, score {opportunity.composite_score:.3f}"
                )
                return True
        except Exception as e:
            self.logger.error(f"Error adding calendar spread opportunity: {str(e)}")
            return False

    def get_opportunity_count(self) -> int:
        """Get the total number of collected opportunities"""
        with self.lock:
            return len(self.opportunities)

    def get_best_opportunity(self) -> Optional[CalendarSpreadOpportunity]:
        """
        Get the best calendar spread opportunity based on composite score.

        Returns:
            CalendarSpreadOpportunity: Best opportunity or None if no opportunities
        """
        with self.lock:
            if not self.opportunities:
                return None

            # Sort by composite score (highest first)
            best_opportunity = max(self.opportunities, key=lambda x: x.composite_score)

            self.logger.info(
                f"Selected best calendar spread opportunity: {best_opportunity.symbol} "
                f"{best_opportunity.option_type} {best_opportunity.strike} "
                f"(score: {best_opportunity.composite_score:.3f})"
            )

            return best_opportunity

    def log_cycle_summary(self):
        """Log summary of current cycle's opportunities"""
        with self.lock:
            if not self.opportunities:
                self.logger.info(
                    "No calendar spread opportunities collected this cycle"
                )
                return

            self.logger.info(f"=== Calendar Spread Cycle Summary ===")
            self.logger.info(f"Total opportunities: {len(self.opportunities)}")

            # Group by symbol
            symbol_counts = defaultdict(int)
            for opp in self.opportunities:
                symbol_counts[opp.symbol] += 1

            self.logger.info(f"Symbols with opportunities: {len(symbol_counts)}")
            for symbol, count in symbol_counts.items():
                self.logger.info(f"  {symbol}: {count} opportunities")

            # Show top 3 opportunities
            top_opportunities = sorted(
                self.opportunities, key=lambda x: x.composite_score, reverse=True
            )[:3]

            self.logger.info("Top calendar spread opportunities:")
            for i, opp in enumerate(top_opportunities, 1):
                self.logger.info(
                    f"  #{i}: {opp.symbol} {opp.option_type} {opp.strike} "
                    f"IV: {opp.iv_spread:.1f}% Score: {opp.composite_score:.3f}"
                )
