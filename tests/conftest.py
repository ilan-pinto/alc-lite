"""
Pytest configuration and common fixtures for alchimest tests
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

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
