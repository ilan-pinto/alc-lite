"""
Constants and configuration values for SFR arbitrage strategy.

This module contains all the configuration parameters, thresholds, and constants
used throughout the SFR strategy implementation.
"""

import os
import sys

# Data collection timeouts
DEFAULT_DATA_TIMEOUT = 30.0  # 30 seconds timeout for data collection

# Quality score thresholds
MIN_DATA_QUALITY_THRESHOLD_PARTIAL = 0.8  # Minimum quality for partial data mode
MIN_DATA_QUALITY_THRESHOLD_FULL = 0.6  # Minimum quality for full data mode

# Price validation thresholds
MAX_BID_ASK_SPREAD = 20.0  # Maximum allowed bid-ask spread for options

# Profit thresholds (lowered to capture more opportunities)
MIN_THEORETICAL_PROFIT = 0.01  # Minimum theoretical profit (1 cent)
MIN_GUARANTEED_PROFIT = 0.01  # Minimum guaranteed profit (1 cent)
MIN_PROFIT_THRESHOLD = 0.01  # Minimum profit threshold for conditions check (1 cent)

# Strike and moneyness validation
MIN_STRIKE_SPREAD = 1.0
MAX_STRIKE_SPREAD = 50.0
MIN_DAYS_TO_EXPIRY = 15
MAX_DAYS_TO_EXPIRY = 50

# Moneyness thresholds (more flexible for SFR)
CALL_MONEYNESS_MIN = 0.90  # Allow deeper ITM calls
CALL_MONEYNESS_MAX = 1.15  # Allow more OTM calls
PUT_MONEYNESS_MIN = 0.80  # Allow deeper ITM puts
PUT_MONEYNESS_MAX = 1.10  # Allow more OTM puts

# Volume thresholds
MIN_VOLUME_FOR_ACCEPTANCE = 1  # Minimum volume to accept contract
MIN_VOLUME_FOR_QUALITY = 5  # Minimum volume for quality scoring
QUALITY_VOLUME_THRESHOLD = 1000.0  # Volume normalization threshold

# Data quality score weights
STOCK_DATA_WEIGHT = 0.30
CALL_DATA_WEIGHT = 0.35
PUT_DATA_WEIGHT = 0.35

# Individual data quality component weights
PRICE_BID_WEIGHT = 0.1
PRICE_ASK_WEIGHT = 0.1
PRICE_LAST_WEIGHT = 0.05
VOLUME_WEIGHT = 0.05
SPREAD_QUALITY_WEIGHT = 0.05

# Buffer percentages
DEFAULT_BUFFER_PERCENT = 0.01  # 1% buffer for realistic execution
COMBO_PRICE_BUFFER = 0.00  # No buffer for combo limit price

# Market hours validation (ET timezone)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Statistical analysis thresholds
MAX_ACCEPTABLE_SPREAD_PCT = 0.05  # 5% max spread percentage
OUTLIER_Z_SCORE_THRESHOLD = 2.0  # Z-score threshold for outlier detection
OUTLIER_PENALTY = 0.5  # Penalty for outliers in scoring

# Vectorized evaluation thresholds
MAX_TOTAL_SPREAD_COST = 5.0  # Maximum total spread cost in dollars

# Timeout configuration defaults
DEFAULT_PHASE_1_TIMEOUT = 0.5  # 500ms for phase 1 (critical)
DEFAULT_PHASE_2_TIMEOUT = 1.5  # 1.5s for phase 2 (important)
DEFAULT_PHASE_3_TIMEOUT = 3.0  # 3.0s for phase 3 (final)

# Priority thresholds
DEFAULT_CRITICAL_THRESHOLD = 0.8
DEFAULT_IMPORTANT_THRESHOLD = 0.7

# Phase profit thresholds
PHASE_1_PROFIT_THRESHOLD = 0.20  # Higher threshold for quick execution
PHASE_2_PROFIT_THRESHOLD = 0.15  # Medium threshold
PHASE_3_PROFIT_THRESHOLD = 0.10  # Lower threshold for final evaluation

# Logging and debugging
DEBUG_LOG_INTERVAL = 5  # Log debug messages every N seconds
MAX_LOG_MISSING_CONTRACTS = 5  # Maximum number of missing contracts to log

# Performance optimization (PyPy-aware)
# PyPy detection
USING_PYPY = hasattr(sys, "pypy_version_info")

# Adjust thresholds based on runtime
if USING_PYPY:
    # PyPy-optimized values
    NUMPY_VECTORIZATION_THRESHOLD = 20  # PyPy handles larger thresholds better
    BATCH_PROCESSING_SIZE = 100  # PyPy can handle larger batches efficiently
    DEFAULT_CACHE_SIZE = 500  # Larger cache since PyPy manages memory better
    JIT_WARMUP_ITERATIONS = 50  # Allow JIT to warm up
    # PyPy timeout multiplier - needs more time during JIT warmup
    TIMEOUT_MULTIPLIER = 1.5
else:
    # CPython-optimized values
    NUMPY_VECTORIZATION_THRESHOLD = 10  # CPython+numpy is efficient for smaller arrays
    BATCH_PROCESSING_SIZE = 50  # Conservative batch size for CPython
    TIMEOUT_MULTIPLIER = 1.0
    DEFAULT_CACHE_SIZE = 200  # Smaller cache for CPython
    JIT_WARMUP_ITERATIONS = 0  # No warmup needed

# Error handling
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 0.1

# Cache settings
CONTRACT_CACHE_TTL = 300  # 5 minutes TTL for contract cache

# Parallel execution configuration
PARALLEL_EXECUTION_ENABLED = True  # Enable/disable parallel execution

# Base timeout values (increased from 30s->60s, 10s->20s for more patience)
_BASE_EXECUTION_TIMEOUT = 60.0
_BASE_LEG_TIMEOUT = 20.0
_BASE_PARTIAL_FILL_TIMEOUT = 10.0
_BASE_ROLLBACK_TIMEOUT = 25.0

# Runtime-adjusted timeout values (PyPy gets additional time for JIT warmup)
# Allow environment variable overrides for different market conditions
PARALLEL_EXECUTION_TIMEOUT = float(
    os.getenv(
        "PARALLEL_EXECUTION_TIMEOUT", _BASE_EXECUTION_TIMEOUT * TIMEOUT_MULTIPLIER
    )
)
PARALLEL_FILL_TIMEOUT_PER_LEG = float(
    os.getenv("PARALLEL_FILL_TIMEOUT_PER_LEG", _BASE_LEG_TIMEOUT * TIMEOUT_MULTIPLIER)
)
PARTIAL_FILL_AGGRESSIVE_TIMEOUT = float(
    os.getenv(
        "PARTIAL_FILL_AGGRESSIVE_TIMEOUT",
        _BASE_PARTIAL_FILL_TIMEOUT * TIMEOUT_MULTIPLIER,
    )
)

PARALLEL_MAX_SLIPPAGE_PERCENT = 2.0  # Maximum acceptable slippage percentage
PARALLEL_MAX_GLOBAL_ATTEMPTS = 5  # Maximum global execution attempts per session
PARALLEL_MAX_SYMBOL_ATTEMPTS = 3  # Maximum execution attempts per symbol

# Partial fill handling
PARTIAL_FILL_MAX_SLIPPAGE = (
    0.05  # Maximum slippage for partial fill completion (5 cents)
)
PARTIAL_FILL_COMPLETION_ATTEMPTS = 3  # Maximum attempts to complete partial fills

# Rollback configuration
ROLLBACK_MAX_ATTEMPTS = 3  # Maximum rollback attempts
ROLLBACK_TIMEOUT_PER_ATTEMPT = float(
    os.getenv(
        "ROLLBACK_TIMEOUT_PER_ATTEMPT", _BASE_ROLLBACK_TIMEOUT * TIMEOUT_MULTIPLIER
    )
)
ROLLBACK_AGGRESSIVE_PRICING_FACTOR = 0.01  # 1% price adjustment for aggressive rollback
ROLLBACK_MAX_SLIPPAGE_PERCENT = 3.0  # Maximum acceptable rollback slippage

# Execution reporting
EXECUTION_REPORT_DEFAULT_LEVEL = "detailed"  # summary, detailed, comprehensive, debug
EXECUTION_REPORT_AUTO_EXPORT = False  # Auto-export reports to files
EXECUTION_REPORT_EXPORT_FORMAT = "html"  # html, json, text

# Safety limits
MAX_CONCURRENT_PARALLEL_EXECUTIONS = 1  # Only one parallel execution at a time globally
EXECUTION_LOCK_TIMEOUT = 30.0  # Global execution lock timeout (seconds)
DAILY_PARALLEL_EXECUTION_LIMIT = 20  # Maximum parallel executions per day

# Performance thresholds
EXECUTION_TIME_WARNING_THRESHOLD = (
    10.0  # Warn if execution takes longer than this (seconds)
)
EXECUTION_TIME_ERROR_THRESHOLD = (
    30.0  # Error if execution takes longer than this (seconds)
)
SLIPPAGE_WARNING_THRESHOLD = 1.0  # Warn if slippage exceeds this (dollars)
SLIPPAGE_ERROR_THRESHOLD = 5.0  # Error if slippage exceeds this (dollars)

# Monitoring and alerts
ENABLE_EXECUTION_MONITORING = True  # Enable real-time execution monitoring
ENABLE_SLIPPAGE_ALERTS = True  # Enable slippage alerts
ENABLE_PERFORMANCE_LOGGING = True  # Enable detailed performance logging

# Development and testing
PARALLEL_EXECUTION_DRY_RUN = False  # Enable dry-run mode for testing
PARALLEL_EXECUTION_DEBUG_MODE = False  # Enable debug mode with extra logging
SIMULATE_PARTIAL_FILLS = False  # Simulate partial fills for testing (development only)

# Scoring system constants
# Default scoring weights for SFR opportunities (must sum to 1.0)
DEFAULT_PROFIT_WEIGHT = (
    0.50  # Guaranteed profit is most important for risk-free arbitrage
)
DEFAULT_LIQUIDITY_WEIGHT = 0.25  # Execution certainty through volume/spreads
DEFAULT_SPREAD_QUALITY_WEIGHT = 0.15  # Bid-ask spread quality
DEFAULT_TIME_DECAY_WEIGHT = 0.10  # Time to expiry optimization

# Liquidity scoring thresholds
MIN_LIQUIDITY_VOLUME_FOR_SCORING = 5  # Minimum volume for full liquidity scoring
LIQUIDITY_VOLUME_NORMALIZATION_THRESHOLD = 1000.0  # Volume normalization point
LIQUIDITY_OPEN_INTEREST_BONUS_MAX = 0.20  # Maximum bonus from open interest

# Spread quality scoring
SPREAD_QUALITY_EXPONENTIAL_DECAY = 10.0  # Spread quality decay rate
SPREAD_QUALITY_WIDE_PENALTY = 0.5  # Penalty multiplier for wide spreads

# Time decay scoring
OPTIMAL_DAYS_TO_EXPIRY_DEFAULT = 30  # Optimal days to expiry for scoring
TIME_DECAY_GAUSSIAN_SIGMA = 10.0  # Standard deviation for time decay gaussian

# Scoring logging configuration
ENABLE_DETAILED_SCORING_LOGS = True  # Enable comprehensive scoring logs
LOG_SCORE_COMPONENTS_DETAIL = True  # Log individual component scores
SCORING_LOG_THRESHOLD = 0.0  # Only log opportunities above this score
MAX_RANKING_DISPLAY = 5  # Maximum opportunities to display in ranking logs

# Scoring performance optimizations
SCORING_BATCH_SIZE = 50  # Process opportunities in batches
ENABLE_SCORING_CACHE = True  # Cache scoring calculations
SCORING_CACHE_TTL = 300  # 5 minutes TTL for scoring cache
