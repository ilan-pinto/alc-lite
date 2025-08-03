"""
Box Spread Strategy Package

This package contains the refactored box spread strategy implementation,
broken down into focused modules for better maintainability and testing.

Box spread arbitrage: A risk-free 4-leg options strategy consisting of:
- Long call at lower strike (K1)
- Short call at higher strike (K2)
- Short put at lower strike (K1)
- Long put at higher strike (K2)

When net_debit < strike_difference, this creates a risk-free arbitrage opportunity.

Main exports for backward compatibility with existing code.
"""

# Import executor
from .executor import BoxExecutor

# Import data models
from .models import BoxSpreadConfig, BoxSpreadLeg, BoxSpreadOpportunity

# Import opportunity manager
from .opportunity_manager import BoxOpportunityManager

# Import risk validator
from .risk_validator import BoxRiskValidator

# Import main strategy class
from .strategy import BoxSpread, contract_ticker, run_box_spread_strategy

# Import utility classes (reuse from calendar_spread where applicable)
from .utils import (
    AdaptiveCacheManager,
    PerformanceProfiler,
    TTLCache,
    VectorizedGreeksCalculator,
    _safe_isnan,
)

__all__ = [
    # Main strategy
    "BoxSpread",
    "run_box_spread_strategy",
    # Data models
    "BoxSpreadConfig",
    "BoxSpreadLeg",
    "BoxSpreadOpportunity",
    # Utility classes (reused from calendar_spread)
    "TTLCache",
    "AdaptiveCacheManager",
    "PerformanceProfiler",
    "VectorizedGreeksCalculator",
    "_safe_isnan",
    # Opportunity management
    "BoxOpportunityManager",
    # Executor
    "BoxExecutor",
    # Risk validation
    "BoxRiskValidator",
    # Global variable for backward compatibility
    "contract_ticker",
]
