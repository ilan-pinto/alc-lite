"""
Synthetic arbitrage strategy module.

This module provides a modular implementation of the Synthetic arbitrage strategy
with improved maintainability and reusability.

Main Components:
- Syn: Main strategy class
- SynExecutor: Execution engine
- GlobalOpportunityManager: Multi-symbol opportunity management
- ScoringConfig: Configurable scoring strategies
- ValidationEngine: Input validation and checks
- DataCollector: Market data collection and management

The module is structured for optimal PyPy JIT compilation and performance.
"""

# Constants (for advanced configuration)
from .constants import (
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
)
from .data_collector import DataCollector

# Core execution components
from .executor import SynExecutor
from .global_opportunity_manager import GlobalOpportunityManager

# Data models
from .models import ExpiryOption, GlobalOpportunity, OpportunityScore

# Configuration and scoring
from .scoring import ScoringConfig

# Main strategy class
from .strategy import Syn

# Utilities and helpers
from .utils import (
    create_syn_with_config,
    debug_contract_ticker_state,
    get_symbol_contract_count,
    test_global_opportunity_scoring,
)

# Validation and data collection
from .validation import ValidationEngine

# Version and metadata
__version__ = "2.0.0"
__author__ = "Alchemist Project"
__description__ = "Modular Synthetic Arbitrage Strategy Implementation"

# Public API
__all__ = [
    # Main strategy
    "Syn",
    # Core components
    "SynExecutor",
    "GlobalOpportunityManager",
    "ScoringConfig",
    # Data models
    "ExpiryOption",
    "OpportunityScore",
    "GlobalOpportunity",
    # Validation and data
    "ValidationEngine",
    "DataCollector",
    # Utilities
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
    # Metadata
    "__version__",
    "__author__",
    "__description__",
]

# Performance optimizations for PyPy
if USING_PYPY:
    # Pre-import numpy for JIT warmup
    import numpy as np

    # Warm up common operations
    _warmup_array = np.array([1.0, 2.0, 3.0])
    _warmup_result = np.sum(_warmup_array)
    del _warmup_array, _warmup_result
