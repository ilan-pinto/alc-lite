"""
SFR (Synthetic-Free-Risk) arbitrage strategy module.

This module provides a modular implementation of the SFR arbitrage strategy
with improved maintainability and reusability.
"""

from .executor import SFRExecutor
from .models import ExpiryOption

# Import main components for external use
from .strategy import SFR, contract_ticker
from .validation import StrikeValidator

# Export main classes and models for external use
__all__ = [
    "SFR",
    "SFRExecutor",
    "ExpiryOption",
    "StrikeValidator",
    "contract_ticker",  # Export for compatibility
]

# Import all components for internal use within the package
from .constants import *
from .data_collector import *
from .executor import *
from .models import *
from .opportunity_evaluator import *
from .utils import *
from .validation import *
