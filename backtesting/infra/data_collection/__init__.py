"""
Historical Data Collection Pipeline

A comprehensive system for collecting historical options and stock data
from Interactive Brokers for backtesting purposes.

This module provides functionality for:
- Real-time options data collection from Interactive Brokers
- Historical data fetching and storage
- VIX volatility data collection for correlation analysis
- Data validation and integrity checks
- Database integration with PostgreSQL/TimescaleDB
- Command-line pipeline for automated data loading

Main Components:
- HistoricalDataLoader: Load historical data from IB
- OptionsDataCollector: Real-time data collection
- VIXDataCollector: VIX volatility data collection
- DataValidator: Data quality validation
- Pipeline CLI: Command-line interface for data loading

Usage:
    from backtesting.infra.data_collection import HistoricalDataLoader

    # Or use the CLI
    python load_historical_pipeline.py --symbol SPY --days 30
"""

__version__ = "1.0.0"
__author__ = "AlcLite Trading System"

from .config.config import CollectionConfig, DatabaseConfig, HistoricalConfig
from .core.collector import OptionsDataCollector
from .core.historical_loader import HistoricalDataLoader
from .core.validators import DataValidator, MarketDataSnapshot
from .core.vix_collector import VIXDataCollector

# Import pipeline components with fallback for import errors
try:
    from .pipelines.load_historical_pipeline import (
        HistoricalDataPipeline,
        PipelineConfig,
    )
except ImportError:
    # Pipeline might have dependency issues, make optional
    HistoricalDataPipeline = None
    PipelineConfig = None

__all__ = [
    "OptionsDataCollector",
    "HistoricalDataLoader",
    "VIXDataCollector",
    "DataValidator",
    "MarketDataSnapshot",
    "DatabaseConfig",
    "CollectionConfig",
    "HistoricalConfig",
    "HistoricalDataPipeline",
    "PipelineConfig",
]
