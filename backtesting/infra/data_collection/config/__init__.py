"""
Configuration management for data collection pipeline.

This package provides configuration classes for:
- Database connection settings
- Historical data loading parameters
- Real-time collection settings
- Validation rules configuration
"""

from .config import CollectionConfig, DatabaseConfig, HistoricalConfig

__all__ = [
    "DatabaseConfig",
    "HistoricalConfig",
    "CollectionConfig",
]
