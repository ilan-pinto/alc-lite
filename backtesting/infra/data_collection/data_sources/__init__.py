"""
Data source adapters for historical data collection.

This package provides adapters for various data sources including:
- Interactive Brokers API
- CSV files
- External data APIs (Alpha Vantage, Yahoo Finance, etc.)
- Cached/preprocessed data
"""

from .csv_data_source import CSVDataSource
from .data_source_adapter import DataSourceAdapter
from .external_api_source import ExternalAPISource
from .ib_data_source import IBDataSource

__all__ = [
    "DataSourceAdapter",
    "IBDataSource",
    "CSVDataSource",
    "ExternalAPISource",
]
