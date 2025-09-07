"""
Global execution lock for SFR parallel execution system.

This module provides a singleton lock to ensure only one SFR arbitrage execution
happens at a time across all symbols and executors. This prevents concurrent
executions from interfering with each other and maintains system stability.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Set

from ..common import get_logger

logger = get_logger()


@dataclass
class ExecutionLockInfo:
    """Information about the current lock holder."""

    symbol: str
    executor_id: str
    lock_time: float
    operation: str  # e.g., "parallel_execution", "rollback", "cleanup"


class GlobalExecutionLock:
    """
    Singleton global execution lock for SFR parallel execution.

    This lock ensures that only one SFR execution can occur at a time across
    all symbols. It provides:
    - Atomic execution guarantee
    - Lock holder tracking
    - Timeout protection
    - Performance metrics
    """

    _instance: Optional["GlobalExecutionLock"] = None
    _lock = asyncio.Lock()  # Class-level lock for singleton creation

    def __new__(cls) -> "GlobalExecutionLock":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._execution_lock = asyncio.Lock()
        self._current_holder: Optional[ExecutionLockInfo] = None
        self._lock_history: list[ExecutionLockInfo] = []
        self._max_history_size = 100

        # Performance tracking
        self._total_locks_acquired = 0
        self._total_lock_time = 0.0
        self._longest_lock_duration = 0.0
        self._lock_contentions = 0  # Number of times executors waited

        # Active waiters tracking
        self._waiting_executors: Set[str] = set()

        logger.debug("GlobalExecutionLock singleton initialized")

    @classmethod
    async def get_instance(cls) -> "GlobalExecutionLock":
        """Get the singleton instance (async-safe)."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    async def acquire(
        self,
        symbol: str,
        executor_id: str,
        operation: str = "execution",
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Acquire the global execution lock.

        Args:
            symbol: Symbol being executed
            executor_id: Unique identifier for the executor
            operation: Type of operation (execution, rollback, cleanup)
            timeout: Optional timeout in seconds

        Returns:
            True if lock acquired, False if timeout
        """
        start_time = time.time()

        # Track waiting executors for contention metrics
        waiter_key = f"{symbol}_{executor_id}"
        self._waiting_executors.add(waiter_key)

        try:
            if timeout:
                # Use wait_for with timeout
                await asyncio.wait_for(self._execution_lock.acquire(), timeout=timeout)
            else:
                # Acquire without timeout
                await self._execution_lock.acquire()

            # Lock acquired successfully
            lock_time = time.time()
            wait_duration = lock_time - start_time

            # Track contention if we had to wait
            if wait_duration > 0.001:  # More than 1ms wait
                self._lock_contentions += 1
                logger.debug(
                    f"[{symbol}] Lock contention: waited {wait_duration:.3f}s for lock"
                )

            # Update lock holder info
            self._current_holder = ExecutionLockInfo(
                symbol=symbol,
                executor_id=executor_id,
                lock_time=lock_time,
                operation=operation,
            )

            self._total_locks_acquired += 1

            logger.info(
                f"[{symbol}] Global execution lock ACQUIRED by {executor_id} "
                f"for {operation} (wait: {wait_duration:.3f}s)"
            )

            return True

        except asyncio.TimeoutError:
            logger.warning(
                f"[{symbol}] Global execution lock TIMEOUT after {timeout}s "
                f"for {executor_id} {operation}"
            )
            return False

        except Exception as e:
            logger.error(f"[{symbol}] Error acquiring lock: {e}")
            return False

        finally:
            # Remove from waiting executors
            self._waiting_executors.discard(waiter_key)

    def release(self, symbol: str, executor_id: str) -> None:
        """
        Release the global execution lock.

        Args:
            symbol: Symbol that was being executed
            executor_id: Executor that held the lock
        """
        if not self._execution_lock.locked():
            logger.warning(
                f"[{symbol}] Attempted to release unlocked global execution lock"
            )
            return

        # Verify we're releasing from the correct holder
        if self._current_holder and (
            self._current_holder.symbol != symbol
            or self._current_holder.executor_id != executor_id
        ):
            logger.warning(
                f"[{symbol}] Lock release mismatch: held by "
                f"{self._current_holder.symbol}:{self._current_holder.executor_id}, "
                f"released by {symbol}:{executor_id}"
            )

        # Calculate lock duration for metrics
        if self._current_holder:
            lock_duration = time.time() - self._current_holder.lock_time
            self._total_lock_time += lock_duration

            if lock_duration > self._longest_lock_duration:
                self._longest_lock_duration = lock_duration

            # Add to history
            self._lock_history.append(self._current_holder)
            if len(self._lock_history) > self._max_history_size:
                self._lock_history.pop(0)

            logger.info(
                f"[{symbol}] Global execution lock RELEASED by {executor_id} "
                f"(held: {lock_duration:.3f}s, operation: {self._current_holder.operation})"
            )

        # Clear current holder and release lock
        self._current_holder = None
        self._execution_lock.release()

    def is_locked(self) -> bool:
        """Check if the global execution lock is currently held."""
        return self._execution_lock.locked()

    def get_current_holder(self) -> Optional[ExecutionLockInfo]:
        """Get information about the current lock holder."""
        return self._current_holder

    def get_lock_stats(self) -> Dict:
        """Get comprehensive lock statistics."""
        current_time = time.time()

        stats = {
            "total_locks_acquired": self._total_locks_acquired,
            "lock_contentions": self._lock_contentions,
            "currently_locked": self.is_locked(),
            "waiting_executors_count": len(self._waiting_executors),
            "waiting_executors": list(self._waiting_executors),
            "total_lock_time_seconds": self._total_lock_time,
            "longest_lock_duration_seconds": self._longest_lock_duration,
            "average_lock_duration_seconds": (
                self._total_lock_time / self._total_locks_acquired
                if self._total_locks_acquired > 0
                else 0
            ),
            "lock_history_size": len(self._lock_history),
        }

        # Add current holder info if locked
        if self._current_holder:
            current_duration = current_time - self._current_holder.lock_time
            stats.update(
                {
                    "current_holder_symbol": self._current_holder.symbol,
                    "current_holder_executor": self._current_holder.executor_id,
                    "current_holder_operation": self._current_holder.operation,
                    "current_lock_duration_seconds": current_duration,
                }
            )

        return stats

    def get_recent_history(self, count: int = 10) -> list[Dict]:
        """Get recent lock history."""
        recent = self._lock_history[-count:] if count > 0 else self._lock_history
        return [
            {
                "symbol": entry.symbol,
                "executor_id": entry.executor_id,
                "lock_time": datetime.fromtimestamp(entry.lock_time).isoformat(),
                "operation": entry.operation,
            }
            for entry in recent
        ]

    async def force_release(self, reason: str = "manual_override") -> bool:
        """
        Force release the lock (emergency use only).

        Args:
            reason: Reason for force release

        Returns:
            True if lock was released, False if not locked
        """
        if not self.is_locked():
            logger.info("Force release called but lock not held")
            return False

        current_holder = self._current_holder
        logger.warning(
            f"FORCE RELEASING global execution lock. Reason: {reason}. "
            f"Previous holder: {current_holder.symbol if current_holder else 'unknown'}"
        )

        # Clear holder and force release
        self._current_holder = None
        if self._execution_lock.locked():
            self._execution_lock.release()

        return True

    def reset_stats(self) -> None:
        """Reset all performance statistics."""
        self._total_locks_acquired = 0
        self._total_lock_time = 0.0
        self._longest_lock_duration = 0.0
        self._lock_contentions = 0
        self._waiting_executors.clear()
        logger.info("Global execution lock statistics reset")


# Convenience functions for easy access
async def acquire_global_lock(
    symbol: str,
    executor_id: str,
    operation: str = "execution",
    timeout: Optional[float] = None,
) -> bool:
    """Convenience function to acquire global execution lock."""
    lock = await GlobalExecutionLock.get_instance()
    return await lock.acquire(symbol, executor_id, operation, timeout)


def release_global_lock(symbol: str, executor_id: str) -> None:
    """Convenience function to release global execution lock."""
    # Note: This is synchronous since lock.release() is synchronous
    # We need the instance, but can't await in sync function
    # So we'll need to pass the instance or make this async
    pass  # Will be implemented when integrating with executor


async def get_global_lock_stats() -> Dict:
    """Convenience function to get global lock statistics."""
    lock = await GlobalExecutionLock.get_instance()
    return lock.get_lock_stats()


async def is_global_lock_held() -> bool:
    """Convenience function to check if global lock is held."""
    lock = await GlobalExecutionLock.get_instance()
    return lock.is_locked()
