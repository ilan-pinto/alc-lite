"""
Calendar Spread Strategy Module

This module provides backward compatibility by importing from the calendar_spread package.
The implementation has been refactored into a proper package structure for better maintainability.
"""

# Import everything from the calendar_spread package for backward compatibility
from .calendar_spread import (
    AdaptiveCacheManager,
    CalendarSpread,
    CalendarSpreadConfig,
    CalendarSpreadLeg,
    CalendarSpreadOpportunity,
    CalendarSpreadOpportunityManager,
    PerformanceProfiler,
    TTLCache,
    VectorizedGreeksCalculator,
    _safe_isnan,
    contract_ticker,
    run_calendar_spread_strategy,
)

# Export everything for backward compatibility
__all__ = [
    "CalendarSpread",
    "CalendarSpreadConfig",
    "CalendarSpreadLeg",
    "CalendarSpreadOpportunity",
    "CalendarSpreadOpportunityManager",
    "TTLCache",
    "AdaptiveCacheManager",
    "PerformanceProfiler",
    "VectorizedGreeksCalculator",
    "_safe_isnan",
    "run_calendar_spread_strategy",
    "contract_ticker",
]
