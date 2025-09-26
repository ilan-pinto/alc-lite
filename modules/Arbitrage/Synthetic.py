"""
Backwards compatibility wrapper for Synthetic arbitrage strategy.

This module provides backwards compatibility for existing imports while using
the new modular implementation under the hood.

DEPRECATED: This module is provided for backwards compatibility only.
Please update your imports to use the new modular structure:

OLD:
    from modules.Arbitrage.Synthetic import Syn, ScoringConfig, GlobalOpportunityManager

NEW:
    from modules.Arbitrage.synthetic import Syn, ScoringConfig, GlobalOpportunityManager

The new modular structure provides:
- Better maintainability and testability
- Improved PyPy JIT compilation performance
- Cleaner separation of concerns
- Enhanced configurability
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "modules.Arbitrage.Synthetic is deprecated. Please use modules.Arbitrage.synthetic instead. "
    "The old import will continue to work but may be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Create a custom module class that can handle setattr properly
import sys
from types import ModuleType

# Store references to the modules for dynamic patching
import modules.Arbitrage.synthetic.data_collector as _dc_module
import modules.Arbitrage.synthetic.validation as _val_module

# Explicit re-exports for IDE support and backwards compatibility
# Import everything from the new modular implementation
from .synthetic import *  # noqa: F403, F401
from .synthetic import (  # Main strategy class; Core execution components; Configuration and scoring; Data models; Validation and data collection; Utilities; Constants that may be referenced in tests
    CACHE_TTL,
    CALL_MONEYNESS_MAX,
    CALL_MONEYNESS_MIN,
    DEFAULT_DATA_TIMEOUT,
    MAX_BID_ASK_SPREAD,
    MAX_DAYS_TO_EXPIRY,
    MIN_DAYS_TO_EXPIRY,
    MIN_LIQUIDITY_SCORE,
    MIN_RISK_REWARD_RATIO,
    OPTIMAL_DAYS_TO_EXPIRY,
    PUT_MONEYNESS_MAX,
    PUT_MONEYNESS_MIN,
    USING_PYPY,
    DataCollector,
    ExpiryOption,
    GlobalOpportunity,
    GlobalOpportunityManager,
    OpportunityScore,
    ScoringConfig,
    Syn,
    SynExecutor,
    ValidationEngine,
    create_syn_with_config,
    debug_contract_ticker_state,
    get_symbol_contract_count,
    test_global_opportunity_scoring,
)

# Import the actual global dictionaries from the new modules
from .synthetic.data_collector import contract_ticker
from .synthetic.validation import strike_cache

# Import global variables that tests might expect
# We need to be careful here - tests may monkeypatch these variables
# so we need to ensure the patching affects the actual global dictionaries

# The challenge: monkeypatch.setattr replaces the module attribute, but we need
# to ensure that the actual global dictionaries used by the new modular code get updated.


class CompatibilityModule(ModuleType):
    """Custom module class that handles backwards compatibility for global variables"""

    def __setattr__(self, name, value):
        if name == "contract_ticker":
            # When someone patches contract_ticker, update the actual global dictionary
            _dc_module.contract_ticker.clear()
            _dc_module.contract_ticker.update(value)
            # Also update our local reference
            super().__setattr__(name, _dc_module.contract_ticker)
        elif name == "strike_cache":
            # When someone patches strike_cache, update the actual global dictionary
            _val_module.strike_cache.clear()
            _val_module.strike_cache.update(value)
            # Also update our local reference
            super().__setattr__(name, _val_module.strike_cache)
        else:
            # For other attributes, use normal setattr behavior
            super().__setattr__(name, value)


# Replace this module with our custom compatibility module
current_module = sys.modules[__name__]
new_module = CompatibilityModule(__name__)

# Copy all existing attributes to the new module
for attr_name in dir(current_module):
    if not attr_name.startswith("_"):
        setattr(new_module, attr_name, getattr(current_module, attr_name))

# Set the global dictionaries
new_module.contract_ticker = contract_ticker
new_module.strike_cache = strike_cache

# Replace the module in sys.modules
sys.modules[__name__] = new_module

# Legacy aliases for any code that might depend on these
CACHE_TTL = CACHE_TTL  # Already imported above, but being explicit
STRIKE_CACHE_TTL = CACHE_TTL  # Alternative name that might be used

# Re-export version info from the new module
__version__ = "2.0.0-compat"
__description__ = "Backwards compatibility wrapper for modular Synthetic strategy"

# Preserve the same public API as the original module
__all__ = [
    # Main classes that were commonly imported
    "Syn",
    "SynExecutor",
    "GlobalOpportunityManager",
    "ScoringConfig",
    # Data models
    "ExpiryOption",
    "OpportunityScore",
    "GlobalOpportunity",
    # Validation and utilities
    "ValidationEngine",
    "DataCollector",
    # Global variables that tests depend on
    "contract_ticker",
    "strike_cache",
    # Utility functions
    "get_symbol_contract_count",
    "debug_contract_ticker_state",
    "test_global_opportunity_scoring",
    "create_syn_with_config",
    # Constants
    "CACHE_TTL",
    "DEFAULT_DATA_TIMEOUT",
    "MIN_DAYS_TO_EXPIRY",
    "MAX_DAYS_TO_EXPIRY",
    "CALL_MONEYNESS_MIN",
    "CALL_MONEYNESS_MAX",
    "PUT_MONEYNESS_MIN",
    "PUT_MONEYNESS_MAX",
    "MAX_BID_ASK_SPREAD",
    "MIN_LIQUIDITY_SCORE",
    "MIN_RISK_REWARD_RATIO",
    "OPTIMAL_DAYS_TO_EXPIRY",
    "USING_PYPY",
]
