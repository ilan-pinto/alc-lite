"""
PyPy Compatibility Module

This module provides compatibility utilities for running under PyPy,
addressing specific threading and asyncio issues that differ from CPython.
"""

import asyncio
import os
import sys
import threading
from functools import wraps
from typing import Any, Callable, Optional

import logging

logger = logging.getLogger(__name__)

# Global PyPy detection
IS_PYPY = hasattr(sys, "pypy_version_info") or "PyPy" in sys.version

# Test environment detection
IS_TEST_ENVIRONMENT = (
    "pytest" in sys.modules
    or os.environ.get("PYTEST_CURRENT_TEST") is not None
    or any("pytest" in arg for arg in sys.argv)
)

# Thread-local storage for event loops and recursion detection
_thread_local = threading.local()

# Module-level storage for original methods to prevent double-wrapping
_original_methods = {}


def is_pypy() -> bool:
    """Check if running under PyPy"""
    return IS_PYPY


def get_thread_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create thread-local event loop for PyPy compatibility"""
    if not hasattr(_thread_local, "loop") or _thread_local.loop is None:
        try:
            # Try to get existing loop
            loop = asyncio.get_event_loop()
            if loop and not loop.is_closed():
                _thread_local.loop = loop
            else:
                raise RuntimeError("Existing loop is closed")
        except RuntimeError:
            # Create new loop if none exists or existing is closed
            _thread_local.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_thread_local.loop)
            logger.debug("Created new event loop for PyPy thread")

    # Ensure we always return a valid loop
    if _thread_local.loop is None:
        _thread_local.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_thread_local.loop)

    return _thread_local.loop


def safe_asyncio_run(coro, timeout: float = 30.0) -> Any:
    """
    Run asyncio coroutine with PyPy threading compatibility.

    This function handles the differences in event loop management between
    PyPy and CPython when called from multiple threads.
    """
    if not IS_PYPY:
        # CPython: Use standard asyncio.run()
        return asyncio.run(coro)

    # PyPy: Use a fresh event loop each time to avoid recursion
    try:
        # Create a fresh loop for this operation
        loop = asyncio.new_event_loop()
        try:
            # Run the coroutine in the fresh loop
            return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
        finally:
            # Always close the loop when done
            loop.close()
    except asyncio.TimeoutError as e:
        logger.warning(f"PyPy asyncio operation timed out after {timeout} seconds: {e}")
        raise e  # Re-raise the exact exception
    except Exception as e:
        logger.debug(
            f"PyPy asyncio operation failed, re-raising: {type(e).__name__}: {e}"
        )
        raise e  # Re-raise the exact exception without modification


def pypy_thread_safe(func: Callable) -> Callable:
    """
    Decorator to make functions thread-safe under PyPy.

    This decorator ensures proper cleanup and error handling
    for functions that use asyncio in threaded environments.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not IS_PYPY:
            # CPython: Run normally
            return func(*args, **kwargs)

        # PyPy: Add thread safety measures
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            # Enhanced error handling for PyPy
            logger.error(f"PyPy thread-safe execution failed in {func.__name__}: {e}")
            raise
        finally:
            # Cleanup thread-local resources for PyPy
            cleanup_thread_resources()

    return wrapper


def cleanup_thread_resources():
    """Clean up thread-local resources for PyPy compatibility"""
    if IS_PYPY and hasattr(_thread_local, "loop"):
        try:
            loop = _thread_local.loop
            if loop and not loop.is_closed():
                # Cancel pending tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()

                # Don't close the loop here as it might be needed again
                # Just clear our reference
                _thread_local.loop = None
        except Exception as e:
            logger.debug(f"PyPy cleanup warning: {e}")


class PyPyAsyncMockCompat:
    """
    PyPy-compatible AsyncMock replacement.

    AsyncMock has known issues with PyPy's threading model.
    This class provides a compatible alternative.
    """

    def __init__(self, return_value=None):
        self.return_value = return_value
        self.call_count = 0
        self.called = False
        self.call_args_list = []

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        self.called = True
        self.call_args_list.append((args, kwargs))

        # Always return a coroutine for both PyPy and CPython
        # This ensures compatibility with asyncio.run()
        async def async_return():
            return self.return_value

        return async_return()

    def assert_called(self):
        assert self.called, "Expected call to have been made"

    def assert_called_once(self):
        assert self.call_count == 1, f"Expected 1 call, got {self.call_count}"


def create_compatible_async_mock(return_value=None):
    """Create AsyncMock compatible with both PyPy and CPython"""
    if IS_PYPY:
        return PyPyAsyncMockCompat(return_value=return_value)
    else:
        # Use standard AsyncMock for CPython
        from unittest.mock import AsyncMock

        mock = AsyncMock()
        mock.return_value = return_value
        return mock


def pypy_safe_asyncio_wrapper(original_method):
    """
    PyPy-safe wrapper that avoids recursion by using fresh event loops.
    """
    if not IS_PYPY:
        return original_method

    @wraps(original_method)
    def wrapper(self, *args, **kwargs):
        # For PyPy, temporarily replace asyncio.run with safe version
        import asyncio

        original_run = asyncio.run

        # Non-recursive safe run implementation
        def pypy_safe_run(coro):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        try:
            # Replace asyncio.run temporarily
            asyncio.run = pypy_safe_run
            # Call original method
            return original_method(self, *args, **kwargs)
        finally:
            # Always restore original asyncio.run
            asyncio.run = original_run

    wrapper._is_pypy_wrapped = True
    return wrapper


def patch_calendar_finder_for_pypy():
    """
    Simple monkey patch for calendar_finder method for PyPy compatibility.
    """
    if not IS_PYPY:
        return False  # No patching needed for CPython

    try:
        from commands.option import OptionScan

        # Check if already patched
        if hasattr(OptionScan.calendar_finder, "_is_pypy_wrapped"):
            logger.debug("PyPy calendar_finder already patched, skipping")
            return True

        # Store original method
        method_key = "OptionScan.calendar_finder"
        if method_key not in _original_methods:
            _original_methods[method_key] = OptionScan.calendar_finder
            logger.debug(
                "Stored original calendar_finder method for PyPy compatibility"
            )

        # Apply PyPy-safe wrapper
        original_method = _original_methods[method_key]
        OptionScan.calendar_finder = pypy_safe_asyncio_wrapper(original_method)

        logger.info("PyPy compatibility wrapper applied to OptionScan.calendar_finder")
        return True

    except Exception as e:
        logger.warning(f"Failed to apply PyPy compatibility patch: {e}")
        return False


def patch_asyncio_globally():
    """
    Globally replace asyncio.run with PyPy-safe version.
    This ensures ALL uses of asyncio.run use fresh event loops.
    """
    if not IS_PYPY:
        return False

    import asyncio

    # Check if already patched
    if hasattr(asyncio.run, "_pypy_patched"):
        logger.debug("asyncio.run already patched for PyPy")
        return True

    # Store original for potential restoration
    original_run = asyncio.run

    def pypy_safe_run(coro, *, debug=None):
        """
        PyPy-safe replacement for asyncio.run.
        Creates a fresh event loop for each call to prevent recursion.
        """
        # Always create a fresh event loop
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            if debug is not None:
                loop.set_debug(debug)
            return loop.run_until_complete(coro)
        finally:
            try:
                # Cleanup pending tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                # Give tasks a chance to clean up
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception as e:
                logger.debug(f"PyPy asyncio cleanup warning: {e}")
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    # Mark as patched and store original
    pypy_safe_run._pypy_patched = True
    pypy_safe_run._original = original_run

    # Replace globally
    asyncio.run = pypy_safe_run

    logger.info("Global asyncio.run replacement applied for PyPy")
    return True


# Simple setup without complex lazy loading
_setup_attempted = False


def setup_pypy_compatibility():
    """
    Set up PyPy compatibility patches.

    This applies global asyncio.run patching and method-specific patches.
    """
    global _setup_attempted

    if not IS_PYPY or _setup_attempted:
        return

    _setup_attempted = True

    # Apply global asyncio.run patch FIRST - this is the critical fix
    patch_asyncio_globally()

    # Then apply method-specific patches
    try:
        patch_calendar_finder_for_pypy()
        logger.info("PyPy compatibility patches applied")
    except ImportError:
        # commands.option not available yet, skip for now
        logger.debug("commands.option not available for patching yet")
    except Exception as e:
        logger.debug(f"PyPy compatibility setup failed: {e}")


# Auto-apply PyPy patches when module is imported
if IS_PYPY:
    if IS_TEST_ENVIRONMENT:
        logger.info(
            "PyPy runtime detected in test environment - setting up compatibility patches"
        )
    else:
        logger.info("PyPy runtime detected - setting up compatibility patches")
    setup_pypy_compatibility()
