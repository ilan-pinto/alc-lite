"""
Constants and configuration values for Synthetic arbitrage strategy.

This module contains all the configuration parameters, thresholds, and constants
used throughout the Synthetic strategy implementation.
"""

import os
import sys

# Global cache settings
CACHE_TTL = 300  # 5 minutes

# Data collection timeouts
DEFAULT_DATA_TIMEOUT = 30.0  # 30 seconds timeout for data collection
ADAPTIVE_TIMEOUT_MULTIPLIER = 0.1  # Additional timeout per contract

# Strike and moneyness validation
MIN_STRIKE_SPREAD = 1.0
MAX_STRIKE_SPREAD = 50.0
MIN_DAYS_TO_EXPIRY = 15
MAX_DAYS_TO_EXPIRY = 50

# Moneyness thresholds (more lenient for Synthetic than SFR)
CALL_MONEYNESS_MIN = 0.90  # Allow deeper ITM calls
CALL_MONEYNESS_MAX = 1.2  # Allow more OTM calls
PUT_MONEYNESS_MIN = 0.80  # Allow deeper ITM puts
PUT_MONEYNESS_MAX = 1.1  # Allow more OTM puts

# Price validation thresholds
MAX_BID_ASK_SPREAD = 20.0  # Maximum allowed bid-ask spread for options
MIN_VOLUME_FOR_ACCEPTANCE = 1  # Minimum volume to accept contract
MIN_VOLUME_FOR_QUALITY = 10  # Minimum volume for quality scoring

# Buffer percentages
DEFAULT_BUFFER_PERCENT = 0.00  # 1.5% buffer for better execution
COMBO_PRICE_BUFFER = 0.00  # No buffer for combo limit price

# Scoring configuration defaults
DEFAULT_RISK_REWARD_WEIGHT = 0.40
DEFAULT_LIQUIDITY_WEIGHT = 0.25
DEFAULT_TIME_DECAY_WEIGHT = 0.20
DEFAULT_MARKET_QUALITY_WEIGHT = 0.15
MIN_LIQUIDITY_SCORE = 0.3
MIN_RISK_REWARD_RATIO = 1.5
OPTIMAL_DAYS_TO_EXPIRY = 30

# Performance optimization (PyPy-aware)
USING_PYPY = hasattr(sys, "pypy_version_info")

# Adjust thresholds based on runtime
if USING_PYPY:
    # PyPy-optimized values
    NUMPY_VECTORIZATION_THRESHOLD = 20  # PyPy handles larger thresholds better
    BATCH_PROCESSING_SIZE = 100  # PyPy can handle larger batches efficiently
    DEFAULT_CACHE_SIZE = 500  # Larger cache since PyPy manages memory better
    TIMEOUT_MULTIPLIER = 1.5  # PyPy needs more time during JIT warmup
else:
    # CPython-optimized values
    NUMPY_VECTORIZATION_THRESHOLD = 10  # CPython+numpy is efficient for smaller arrays
    BATCH_PROCESSING_SIZE = 50  # Conservative batch size for CPython
    DEFAULT_CACHE_SIZE = 200  # Smaller cache for CPython
    TIMEOUT_MULTIPLIER = 1.0

# Logging and debugging
DEBUG_LOG_INTERVAL = 5  # Log debug messages every N seconds
MAX_LOG_MISSING_CONTRACTS = 5  # Maximum number of missing contracts to log

# Quality score thresholds
VOLUME_NORMALIZATION_THRESHOLD = 1000.0  # Volume normalization threshold
QUALITY_VOLUME_COMPONENT_WEIGHT = 0.6
QUALITY_SPREAD_COMPONENT_WEIGHT = 0.4

# Spread analysis
MAX_TOTAL_SPREAD_COST = 5.0  # Maximum total spread cost in dollars
SPREAD_QUALITY_MULTIPLIER = 0.7
CREDIT_QUALITY_MULTIPLIER = 0.3
