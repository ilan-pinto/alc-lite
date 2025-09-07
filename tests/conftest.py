"""
Pytest configuration and common fixtures for alchimest tests
"""

import asyncio
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ib_async import IB, Contract, Fill, Option, Order, Stock, Trade

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def mock_logging():
    """Mock logging to avoid output during tests"""
    with patch("alchimest.configure_logging"):
        yield


@pytest.fixture(autouse=True)
def mock_rich_console():
    """Mock rich console to avoid output during tests"""
    with patch("alchimest.console"):
        yield


# Parallel execution test fixtures
@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_ib():
    """Mock IB connection for testing."""
    ib = AsyncMock(spec=IB)
    ib.isConnected.return_value = True
    ib.client = MagicMock()
    ib.client.getReqId.return_value = 1001
    return ib


@pytest.fixture
def sample_stock_contract():
    """Sample stock contract for testing."""
    return Stock("AAPL", "SMART", "USD")


@pytest.fixture
def sample_call_contract():
    """Sample call option contract for testing."""
    return Option("AAPL", "20240315", 150.0, "C", "SMART", tradingClass="AAPL")


@pytest.fixture
def sample_put_contract():
    """Sample put option contract for testing."""
    return Option("AAPL", "20240315", 150.0, "P", "SMART", tradingClass="AAPL")


@pytest.fixture
def sample_contracts(sample_stock_contract, sample_call_contract, sample_put_contract):
    """Complete set of SFR arbitrage contracts."""
    return {
        "stock": sample_stock_contract,
        "call": sample_call_contract,
        "put": sample_put_contract,
    }


@pytest.fixture
def sample_prices():
    """Sample market prices for testing."""
    return {
        "stock_price": Decimal("150.25"),
        "call_bid": Decimal("8.50"),
        "call_ask": Decimal("8.65"),
        "put_bid": Decimal("3.40"),
        "put_ask": Decimal("3.55"),
    }


@pytest.fixture
def temp_log_dir():
    """Temporary directory for log files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def performance_benchmarks():
    """Performance benchmark thresholds for testing."""
    return {
        "max_execution_time": 5.0,  # seconds
        "max_lock_acquisition_time": 0.1,  # seconds
        "max_memory_usage_mb": 100,  # MB
        "max_cpu_usage_percent": 50,  # %
        "target_success_rate": 0.95,  # 95%
        "max_rollback_rate": 0.1,  # 10%
    }


class ResourceMonitor:
    """Monitor system resources during tests."""

    def __init__(self):
        self.start_memory = 0
        self.peak_memory = 0
        self.start_cpu = 0
        self.peak_cpu = 0

    def start_monitoring(self):
        """Start resource monitoring."""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.start_cpu = process.cpu_percent()
            self.peak_memory = self.start_memory
            self.peak_cpu = self.start_cpu
        except ImportError:
            # psutil not available, use dummy values
            self.start_memory = 0
            self.peak_memory = 0
            self.start_cpu = 0
            self.peak_cpu = 0

    def get_usage(self):
        """Get current resource usage."""
        return {
            "memory_mb": self.peak_memory - self.start_memory,
            "cpu_percent": self.peak_cpu,
        }


@pytest.fixture
def resource_monitor():
    """Resource monitor for performance tests."""
    monitor = ResourceMonitor()
    monitor.start_monitoring()
    yield monitor


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line("markers", "stress: marks tests as stress tests")
