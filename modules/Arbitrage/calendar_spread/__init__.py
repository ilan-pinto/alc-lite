"""
Calendar Spread Strategy Package

This package contains the refactored calendar spread strategy implementation,
broken down into focused modules for better maintainability and testing.

Main exports for backward compatibility with existing code.
"""

# Import data models
from .models import CalendarSpreadConfig, CalendarSpreadLeg, CalendarSpreadOpportunity

# Import opportunity manager
from .opportunity_manager import CalendarSpreadOpportunityManager

# For backward compatibility, export the global contract_ticker
# Convenience function
# Import main strategy class
from .strategy import CalendarSpread, contract_ticker, run_calendar_spread_strategy

# Import utility classes
from .utils import (
    AdaptiveCacheManager,
    PerformanceProfiler,
    TTLCache,
    VectorizedGreeksCalculator,
    _safe_isnan,
)

__all__ = [
    # Main strategy
    "CalendarSpread",
    "run_calendar_spread_strategy",
    # Data models
    "CalendarSpreadConfig",
    "CalendarSpreadLeg",
    "CalendarSpreadOpportunity",
    # Utility classes
    "TTLCache",
    "AdaptiveCacheManager",
    "PerformanceProfiler",
    "VectorizedGreeksCalculator",
    "_safe_isnan",
    # Opportunity management
    "CalendarSpreadOpportunityManager",
    # Global variable for backward compatibility
    "contract_ticker",
]
