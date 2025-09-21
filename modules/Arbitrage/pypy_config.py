"""
PyPy-specific configuration and optimizations for alc-lite

This module provides PyPy-aware configuration that optimizes performance
based on the runtime environment (CPython vs PyPy).
"""

import sys
from typing import Any, Dict


def is_pypy() -> bool:
    """Check if we're running on PyPy"""
    return hasattr(sys, "pypy_version_info")


def get_performance_config() -> Dict[str, Any]:
    """
    Get performance configuration optimized for the current Python runtime.

    Returns:
        Dictionary with performance-related configuration values
    """
    if is_pypy():
        return {
            # PyPy-optimized values
            "batch_size": 100,  # PyPy handles larger batches well
            "cache_ttl": 600,  # Longer cache TTL with PyPy's better memory handling
            "vectorization_threshold": 20,  # Different threshold for PyPy numpy operations
            "gc_threshold": (700, 10, 10),  # Optimized GC settings for PyPy
            "parallel_workers": 8,  # PyPy can handle more concurrent work
            "memory_limit_mb": 2048,  # PyPy uses more memory initially but more efficiently
            "jit_warmup_iterations": 50,  # Allow JIT to warm up
            # Hot loop optimizations
            "use_list_comprehensions": True,  # PyPy optimizes list comprehensions heavily
            "minimize_attribute_access": True,  # Cache attribute lookups
            "use_local_variables": True,  # Local variable access is faster on PyPy
            "precompile_regex": True,  # Pre-compile regex patterns
            # Data structure preferences
            "prefer_dicts_over_objects": True,  # Dicts are optimized on PyPy
            "use_builtin_functions": True,  # Built-ins are JIT-compiled
            "avoid_dynamic_attributes": True,  # Dynamic attributes hurt PyPy performance
        }
    else:
        return {
            # CPython-optimized values
            "batch_size": 50,
            "cache_ttl": 300,
            "vectorization_threshold": 10,  # CPython+numpy is efficient for smaller arrays
            "gc_threshold": None,  # Use default CPython GC settings
            "parallel_workers": 4,
            "memory_limit_mb": 1024,
            "jit_warmup_iterations": 0,  # No JIT warmup needed
            # CPython optimizations
            "use_list_comprehensions": True,  # Still good on CPython
            "minimize_attribute_access": False,  # Less critical on CPython
            "use_local_variables": False,  # Less critical on CPython
            "precompile_regex": False,  # Less critical on CPython
            # Data structure preferences
            "prefer_dicts_over_objects": False,
            "use_builtin_functions": True,
            "avoid_dynamic_attributes": False,
        }


def get_optimization_hints() -> Dict[str, str]:
    """
    Get optimization hints for the current runtime.

    Returns:
        Dictionary with optimization hints and explanations
    """
    if is_pypy():
        return {
            "runtime": "PyPy",
            "primary_benefits": "JIT compilation, optimized loops, better memory management",
            "best_workloads": "Long-running processes, numerical computations, tight loops",
            "considerations": "Higher initial memory usage, JIT warmup time",
            "numpy_performance": "May be slower than CPython for small arrays, faster for pure Python",
            "startup_time": "Slower initial startup, but faster execution after warmup",
        }
    else:
        return {
            "runtime": "CPython",
            "primary_benefits": "Fast startup, excellent numpy integration, mature ecosystem",
            "best_workloads": "Short-running scripts, numpy-heavy computations, I/O bound tasks",
            "considerations": "GIL limitations, slower pure Python loops",
            "numpy_performance": "Excellent for all array sizes",
            "startup_time": "Fast startup, consistent performance",
        }


class PyPyOptimizer:
    """
    Context manager and utility class for PyPy-specific optimizations
    """

    def __init__(self):
        self.config = get_performance_config()
        self.is_pypy = is_pypy()
        self._compiled_patterns = {}

    def get_compiled_pattern(self, pattern: str):
        """
        Get a compiled regex pattern with caching for PyPy optimization.

        Args:
            pattern: Regex pattern string

        Returns:
            Compiled regex pattern
        """
        if self.config["precompile_regex"]:
            if pattern not in self._compiled_patterns:
                import re

                self._compiled_patterns[pattern] = re.compile(pattern)
            return self._compiled_patterns[pattern]
        else:
            import re

            return re.compile(pattern)

    def optimize_loop_variables(self, obj: Any, attributes: list) -> dict:
        """
        Cache object attributes as local variables for loop optimization.

        Args:
            obj: Object to cache attributes from
            attributes: List of attribute names to cache

        Returns:
            Dictionary of cached attribute values
        """
        if self.config["minimize_attribute_access"]:
            return {attr: getattr(obj, attr) for attr in attributes}
        return {}

    def get_batch_size(self, default_size: int) -> int:
        """
        Get optimized batch size based on runtime.

        Args:
            default_size: Default batch size

        Returns:
            Optimized batch size
        """
        if self.is_pypy:
            return max(self.config["batch_size"], default_size)
        return default_size

    def should_use_vectorization(self, data_size: int) -> bool:
        """
        Determine if vectorization should be used based on data size and runtime.

        Args:
            data_size: Size of data to process

        Returns:
            True if vectorization is recommended
        """
        return data_size >= self.config["vectorization_threshold"]


# Global optimizer instance
optimizer = PyPyOptimizer()


# Convenience functions
def get_batch_size(default: int = 50) -> int:
    """Get optimized batch size for current runtime"""
    return optimizer.get_batch_size(default)


def should_vectorize(data_size: int) -> bool:
    """Check if vectorization is recommended for given data size"""
    return optimizer.should_use_vectorization(data_size)


def get_compiled_regex(pattern: str):
    """Get compiled regex pattern with runtime-specific caching"""
    return optimizer.get_compiled_pattern(pattern)
