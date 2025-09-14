"""
Constants and configuration values for SFR arbitrage strategy.

This module contains all the configuration parameters, thresholds, and constants
used throughout the SFR strategy implementation.
"""

# Data collection timeouts
DEFAULT_DATA_TIMEOUT = 30.0  # 30 seconds timeout for data collection

# Quality score thresholds
MIN_DATA_QUALITY_THRESHOLD_PARTIAL = 0.8  # Minimum quality for partial data mode
MIN_DATA_QUALITY_THRESHOLD_FULL = 0.6  # Minimum quality for full data mode

# Price validation thresholds
MAX_BID_ASK_SPREAD = 20.0  # Maximum allowed bid-ask spread for options

# Profit thresholds
MIN_THEORETICAL_PROFIT = 0.10  # Minimum theoretical profit (10 cents)
MIN_GUARANTEED_PROFIT = 0.05  # Minimum guaranteed profit (5 cents)
MIN_PROFIT_THRESHOLD = 0.03  # Minimum profit threshold for conditions check

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
import sys

# PyPy detection
USING_PYPY = hasattr(sys, "pypy_version_info")

# Adjust thresholds based on runtime
if USING_PYPY:
    # PyPy-optimized values
    NUMPY_VECTORIZATION_THRESHOLD = 20  # PyPy handles larger thresholds better
    BATCH_PROCESSING_SIZE = 100  # PyPy can handle larger batches efficiently
    DEFAULT_CACHE_SIZE = 500  # Larger cache since PyPy manages memory better
    JIT_WARMUP_ITERATIONS = 50  # Allow JIT to warm up
else:
    # CPython-optimized values
    NUMPY_VECTORIZATION_THRESHOLD = 10  # CPython+numpy is efficient for smaller arrays
    BATCH_PROCESSING_SIZE = 50  # Conservative batch size for CPython
    DEFAULT_CACHE_SIZE = 200  # Smaller cache for CPython
    JIT_WARMUP_ITERATIONS = 0  # No warmup needed

# Error handling
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 0.1

# Cache settings
CONTRACT_CACHE_TTL = 300  # 5 minutes TTL for contract cache

# Parallel execution configuration
PARALLEL_EXECUTION_ENABLED = True  # Enable/disable parallel execution
PARALLEL_EXECUTION_TIMEOUT = 30.0  # Maximum time to wait for all legs to fill (seconds)
PARALLEL_FILL_TIMEOUT_PER_LEG = (
    10.0  # Maximum time to wait for individual leg (seconds)
)
PARALLEL_MAX_SLIPPAGE_PERCENT = 2.0  # Maximum acceptable slippage percentage
PARALLEL_MAX_GLOBAL_ATTEMPTS = 5  # Maximum global execution attempts per session
PARALLEL_MAX_SYMBOL_ATTEMPTS = 3  # Maximum execution attempts per symbol

# Partial fill handling
PARTIAL_FILL_AGGRESSIVE_TIMEOUT = 5.0  # Timeout for aggressive completion attempts
PARTIAL_FILL_MAX_SLIPPAGE = (
    0.05  # Maximum slippage for partial fill completion (5 cents)
)
PARTIAL_FILL_COMPLETION_ATTEMPTS = 3  # Maximum attempts to complete partial fills

# Rollback configuration
ROLLBACK_MAX_ATTEMPTS = 3  # Maximum rollback attempts
ROLLBACK_TIMEOUT_PER_ATTEMPT = 15.0  # Timeout per rollback attempt (seconds)
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
