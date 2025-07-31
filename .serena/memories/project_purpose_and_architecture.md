# Alchimist-Lite Project Overview

## Purpose
Alchimist-Lite is a command-line options arbitrage scanner that identifies and executes trading opportunities using Interactive Brokers. It focuses on algorithmic trading strategies including:
- Synthetic-Free-Risk (SFR) arbitrage
- Synthetic (non-risk-free) conversions
- Calendar spread arbitrage
- Box spreads and conversion strategies

## Architecture
- **Entry Point**: `alchimest.py` - CLI with subcommands for different strategies
- **Commands Layer**: `commands/` - CLI argument parsing and orchestration
- **Arbitrage Strategies**: `modules/Arbitrage/` - Core strategy implementations
- **Base Classes**: `Strategy.py` contains `ArbitrageClass` base and `BaseExecutor`
- **Integration**: Uses `ib_async` for Interactive Brokers connectivity

## Key Components
- **ArbitrageClass**: Base class with IB connection handling, caching, order management
- **BaseExecutor**: Common functionality for trade execution, logging, pricing
- **OrderManagerClass**: Order placement, monitoring, risk controls
- **ContractCache**: TTL caching system for contract qualification
- **Metrics Collection**: Performance tracking and reporting system
