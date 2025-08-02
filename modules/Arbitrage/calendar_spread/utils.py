"""
Utility classes for calendar spread strategy.

This module contains performance-oriented utility classes including:
- TTLCache: Time-to-live cache with LRU eviction
- AdaptiveCacheManager: Memory-aware cache management
- PerformanceProfiler: Performance monitoring and reporting
- VectorizedGreeksCalculator: High-performance Greeks calculations
- _safe_isnan: Safe NaN checking utility function
"""

import threading
import time
from collections import OrderedDict, defaultdict
from typing import Dict, List

import numpy as np

# Try to import scipy for advanced Greeks calculations
try:
    from scipy.stats import norm

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..common import get_logger

logger = get_logger()

# Export list for proper module interface
__all__ = [
    "SCIPY_AVAILABLE",
    "TTLCache",
    "AdaptiveCacheManager",
    "VectorizedGreeksCalculator",
    "PerformanceProfiler",
    "_safe_isnan",
]


def _safe_isnan(value) -> bool:
    """
    Safe NaN check that handles different data types including None values.

    Args:
        value: Value to check for NaN or None

    Returns:
        bool: True if value is NaN, None, or cannot be converted to float
    """
    if value is None:
        return True

    try:
        # Handle numpy arrays, scalars, and other numeric types
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            # For array-like objects, check if any element is NaN
            return bool(np.any(np.isnan(np.asarray(value, dtype=float))))
        else:
            # For scalar values, convert to float and check
            float_val = float(value)
            return np.isnan(float_val)
    except (TypeError, ValueError, OverflowError):
        # If conversion fails, treat as NaN
        return True


class TTLCache:
    """Time-to-live cache with size limits and LRU eviction for performance optimization"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self._lock = threading.Lock()

    def get(self, key):
        """Get value from cache, returns None if expired or not found"""
        with self._lock:
            if key not in self.cache:
                return None

            # Check TTL
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                self._remove(key)
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        """Store value in cache with LRU eviction if needed"""
        with self._lock:
            current_time = time.time()

            if key in self.cache:
                # Update existing key
                self.cache.move_to_end(key)
            else:
                # Add new key, check size limits
                if len(self.cache) >= self.max_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.cache))
                    self._remove(oldest_key)

            self.cache[key] = value
            self.timestamps[key] = current_time

    def _remove(self, key):
        """Internal method to remove key from cache"""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed items"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key
                for key, timestamp in self.timestamps.items()
                if current_time - timestamp > self.ttl_seconds
            ]

            for key in expired_keys:
                self._remove(key)

            return len(expired_keys)

    def size(self) -> int:
        """Return current cache size"""
        with self._lock:
            return len(self.cache)

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()


class AdaptiveCacheManager:
    """Cache manager that adapts to system memory pressure for optimal performance"""

    def __init__(self):
        self.memory_threshold = 0.85  # 85% memory usage threshold
        self.cleanup_percentage = 0.3  # Remove 30% when over threshold
        self._psutil_available = False

        # Try to import psutil for memory monitoring
        try:
            import psutil

            self._psutil = psutil
            self._psutil_available = True
            logger.debug("psutil available for memory monitoring")
        except ImportError:
            logger.warning("psutil not available - memory pressure detection disabled")

    def should_cleanup(self) -> bool:
        """Check if cache cleanup is needed based on memory pressure"""
        if not self._psutil_available:
            return False

        try:
            memory_percent = self._psutil.virtual_memory().percent
            return memory_percent > self.memory_threshold * 100
        except Exception as e:
            logger.warning(f"Memory monitoring failed: {e}")
            return False

    def cleanup_if_needed(self, caches: List[TTLCache]) -> int:
        """Cleanup caches if memory pressure detected"""
        if not self.should_cleanup():
            return 0

        total_cleaned = 0
        for cache in caches:
            # First try expiry cleanup
            expired = cache.cleanup_expired()

            # If still over threshold, remove oldest entries
            if self.should_cleanup():
                target_size = int(cache.max_size * (1 - self.cleanup_percentage))
                removed_count = 0

                with cache._lock:
                    while len(cache.cache) > target_size and len(cache.cache) > 0:
                        oldest_key = next(iter(cache.cache))
                        cache._remove(oldest_key)
                        removed_count += 1

                logger.debug(
                    f"Memory pressure cleanup: removed {removed_count} entries from cache"
                )

            total_cleaned += expired

        if total_cleaned > 0:
            logger.info(
                f"Memory pressure cleanup: removed {total_cleaned} cache entries"
            )
        return total_cleaned

    def get_memory_stats(self) -> Dict:
        """Get current memory statistics"""
        if not self._psutil_available:
            return {"memory_percent": "unavailable", "available_gb": "unavailable"}

        try:
            memory = self._psutil.virtual_memory()
            return {
                "memory_percent": f"{memory.percent:.1f}%",
                "available_gb": f"{memory.available / (1024**3):.1f}GB",
                "used_gb": f"{memory.used / (1024**3):.1f}GB",
            }
        except Exception as e:
            return {"error": str(e)}


class VectorizedGreeksCalculator:
    """High-performance Greeks calculations using vectorized operations"""

    @staticmethod
    def black_scholes_greeks_batch(
        S: np.ndarray,  # Stock prices
        K: np.ndarray,  # Strike prices
        T: np.ndarray,  # Time to expiry (years)
        r: float,  # Risk-free rate
        sigma: np.ndarray,  # Volatilities (as decimals, e.g., 0.20 for 20%)
        option_type: np.ndarray,  # 1 for call, -1 for put
    ) -> Dict[str, np.ndarray]:
        """Vectorized Black-Scholes Greeks calculation"""

        if not SCIPY_AVAILABLE:
            # Use numpy approximations if scipy not available
            return VectorizedGreeksCalculator._approximate_greeks_batch(
                S, K, T, r, sigma, option_type
            )

        # Prevent division by zero and invalid calculations
        T = np.maximum(T, 1e-6)  # Minimum 1 day
        sigma = np.maximum(sigma, 0.001)  # Minimum 0.1% volatility

        # Black-Scholes calculations
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Standard normal CDF and PDF
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)

        # Vectorized Greeks
        delta = option_type * N_d1
        gamma = n_d1 / (S * sigma * np.sqrt(T))
        theta = (
            -S * n_d1 * sigma / (2 * np.sqrt(T))
            - option_type * r * K * np.exp(-r * T) * N_d2
        )
        vega = S * n_d1 * np.sqrt(T)

        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta / 365,  # Daily theta
            "vega": vega / 100,  # Vega per 1% vol change
        }

    @staticmethod
    def black_scholes_price_batch(
        S: np.ndarray,  # Stock prices
        K: np.ndarray,  # Strike prices
        T: np.ndarray,  # Time to expiry (years)
        r: float,  # Risk-free rate
        sigma: np.ndarray,  # Volatilities (as decimals, e.g., 0.20 for 20%)
        option_type: np.ndarray,  # 1 for call, -1 for put
    ) -> np.ndarray:
        """Vectorized Black-Scholes option pricing calculation"""

        if not SCIPY_AVAILABLE:
            # Use numpy approximations if scipy not available
            return VectorizedGreeksCalculator._approximate_prices_batch(
                S, K, T, r, sigma, option_type
            )

        # Prevent division by zero and invalid calculations
        T = np.maximum(T, 1e-6)  # Minimum 1 day
        sigma = np.maximum(sigma, 0.001)  # Minimum 0.1% volatility

        # Black-Scholes calculations
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Standard normal CDF
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)

        # Calculate option prices
        call_price = S * N_d1 - K * np.exp(-r * T) * N_d2
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        # Return call or put price based on option_type
        return np.where(option_type == 1, call_price, put_price)

    @staticmethod
    def _approximate_prices_batch(
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: float,
        sigma: np.ndarray,
        option_type: np.ndarray,
    ) -> np.ndarray:
        """Approximate option pricing using numpy when scipy unavailable"""

        # Simple intrinsic value with time value approximation
        moneyness = S / K
        time_sqrt = np.sqrt(T)

        # Intrinsic value
        call_intrinsic = np.maximum(S - K, 0)
        put_intrinsic = np.maximum(K - S, 0)

        # Simple time value approximation
        time_value = S * sigma * time_sqrt * 0.4 * np.exp(-((moneyness - 1) ** 2) * 2)

        call_price = call_intrinsic + time_value
        put_price = put_intrinsic + time_value

        return np.where(option_type == 1, call_price, put_price)

    @staticmethod
    def _approximate_greeks_batch(
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: float,
        sigma: np.ndarray,
        option_type: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Approximate Greeks calculations using numpy when scipy unavailable"""

        # Simple approximations for Greeks
        moneyness = S / K
        time_sqrt = np.sqrt(T)

        # Rough delta approximation based on moneyness
        delta = np.where(
            option_type == 1,  # Calls
            np.clip(0.5 + (moneyness - 1) * 2, 0, 1),
            np.clip(0.5 - (1 - moneyness) * 2, -1, 0),
        )

        # Gamma approximation (highest near ATM)
        gamma = np.exp(-((moneyness - 1) ** 2) * 10) / (S * sigma * time_sqrt)

        # Theta approximation (time decay)
        theta = -S * sigma / (2 * time_sqrt) * 0.4 / 365  # Daily theta

        # Vega approximation
        vega = S * time_sqrt * np.exp(-((moneyness - 1) ** 2) * 5) / 100

        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


class PerformanceProfiler:
    """Performance profiling for calendar spread operations"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
        self._lock = threading.Lock()

    def start_timer(self, operation: str):
        """Start timing an operation"""
        with self._lock:
            self.start_times[operation] = time.time()

    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration"""
        with self._lock:
            if operation in self.start_times:
                duration = time.time() - self.start_times[operation]
                self.metrics[operation].append(duration)
                del self.start_times[operation]
                return duration
            return 0

    def get_performance_summary(self) -> Dict:
        """Get performance statistics summary"""
        with self._lock:
            summary = {}
            for operation, times in self.metrics.items():
                if times:
                    summary[operation] = {
                        "count": len(times),
                        "total_time": sum(times),
                        "avg_time": np.mean(times),
                        "max_time": max(times),
                        "min_time": min(times),
                        "std_time": np.std(times) if len(times) > 1 else 0,
                    }
            return summary

    def log_performance_report(self):
        """Log detailed performance report"""
        summary = self.get_performance_summary()
        if not summary:
            return

        logger.info("=== PERFORMANCE REPORT ===")
        for operation, stats in summary.items():
            logger.info(
                f"{operation}: {stats['count']} calls, "
                f"avg={stats['avg_time']:.3f}s, "
                f"total={stats['total_time']:.2f}s, "
                f"max={stats['max_time']:.3f}s"
            )
        logger.info("========================")

    def clear_metrics(self):
        """Clear all collected metrics"""
        with self._lock:
            self.metrics.clear()
            self.start_times.clear()
