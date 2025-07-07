# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Setup and Dependencies
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Main entry point
python alchimest.py --help

# SFR (Synthetic Free Risk) arbitrage scanning
python alchimest.py sfr --symbols MSFT AAPL --cost-limit 100 --profit 0.75 --quantity 2

# Synthetic (non-risk-free) arbitrage scanning
python alchimest.py syn --symbols TSLA NVDA --cost-limit 120 --max-loss 50 --max-profit 100 --quantity 3
```

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run tests with coverage
python -m pytest tests/ --cov=commands --cov=modules --cov-report=html

# Run specific test file
python -m pytest tests/test_cli_arguments.py -v

# Run integration tests only
python -m pytest tests/ -m integration -v
```

### Code Quality
```bash
# Format code with black
black .

# Sort imports
isort .

# Type checking
mypy .

# Linting
flake8 .
```

## Architecture Overview

This is a Python-based options trading arbitrage scanner that connects to Interactive Brokers (IB) for real-time market data. The architecture follows a modular design with clear separation of concerns:

### Core Components

1. **Entry Point (`alchimest.py`)**: Command-line interface with subcommands for different arbitrage strategies
2. **Commands Layer (`commands/`)**: CLI argument parsing and command orchestration
3. **Arbitrage Strategies (`modules/Arbitrage/`)**: Core arbitrage logic and scanning algorithms
4. **Strategy Base Class (`Strategy.py`)**: Abstract base class with common functionality like order management and IB connection handling

### Key Arbitrage Strategies

- **SFR (Synthetic Free Risk)**: Risk-free arbitrage opportunities using synthetic positions
- **Synthetic (Syn)**: Non-risk-free synthetic conversion opportunities with configurable risk parameters

### Data Flow

1. CLI commands are parsed in `alchimest.py`
2. `OptionScan` class in `commands/option.py` instantiates appropriate strategy classes
3. Strategy classes (`SFR`, `Syn`) inherit from `ArbitrageClass` in `Strategy.py`
4. Strategies use `ib_async` library to connect to Interactive Brokers for market data
5. Results are displayed using `rich` library for formatted console output

### Interactive Brokers Integration

The application uses `ib_async` for TWS/IB Gateway connectivity. The `OrderManagerClass` in `Strategy.py` handles:
- Order placement and management
- Position tracking
- Trade execution monitoring
- Filled order logging to `filled_orders.txt`

### Configuration and Defaults

- Default profit target: 0.5%
- Default cost limit: $120
- Default symbols list includes: SPY, QQQ, META, PLTR, etc.
- Logging configured with Rich console output and INFO-level filtering

### Testing Strategy

The test suite focuses on CLI argument validation and integration testing. All tests mock the IB connection to avoid requiring actual broker connectivity during testing.

## Development Notes

- The codebase uses `ib_async` for asynchronous IB operations
- `rich` library provides formatted console output and logging
- Code style enforced with black (88 character line length)
- Type hints required (mypy configuration in pyproject.toml)
- Test coverage targets `commands` and `modules` packages
- CI/CD handles automated versioning and releases
