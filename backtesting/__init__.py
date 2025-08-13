"""
Backtesting Package

This package provides comprehensive backtesting functionality for options arbitrage strategies,
particularly SFR (Synthetic Free Risk) arbitrage analysis.

Key Components:
- SFRBacktestEngine: Main backtesting engine for SFR strategies
- SFRBacktestConfig: Configuration management for backtesting parameters
- SlippageModel: Different slippage calculation models

Usage:
    from backtesting.sfr_backtest_engine import SFRBacktestEngine, SFRBacktestConfig
    from backtesting.sfr_backtest_engine import SlippageModel
"""

# Import main classes for easier access
from .sfr_backtest_engine import SFRBacktestConfig, SFRBacktestEngine, SlippageModel

__version__ = "1.0.0"
__author__ = "Claude Code"

__all__ = ["SFRBacktestEngine", "SFRBacktestConfig", "SlippageModel"]
