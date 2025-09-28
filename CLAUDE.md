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

# Logging options
python alchimest.py sfr --warning --symbols SPY  # Shows INFO + WARNING messages
python alchimest.py syn --debug --symbols QQQ     # Shows all log levels (DEBUG, INFO, WARNING, ERROR)
python alchimest.py sfr --log trading.log --symbols META  # Log to file
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

### Test Failure Analysis (verify-test command)

The `verify-test` command provides automated analysis of pytest test failures, identifying root causes and providing fix recommendations:

```bash
# Analyze a specific test file
python alchimest.py verify-test tests/test_global_execution_lock.py

# Analyze all tests with detailed output
python alchimest.py verify-test tests/ --verbose

# Find similar failures across test files
python alchimest.py verify-test tests/test_*.py --find-similar

# Get agent consultation recommendations
python alchimest.py verify-test tests/ --consult-agents

# Quick analysis without similarity search
python alchimest.py verify-test tests/test_parallel_executor.py --no-similar

# Analyze multiple specific test files
python alchimest.py verify-test tests/test_sfr.py tests/test_arbitrage.py --verbose --consult-agents
```

#### verify-test Features

- **Automated Root Cause Analysis**: Identifies whether failures are due to code issues, test issues, environment problems, or dependency issues
- **Similarity Detection**: Finds similar failure patterns in other test files to identify recurring problems
- **Agent Consultation**: Recommends which specialized agents (pytest-test-engineer, algotrading-python-expert, etc.) to consult for deeper analysis
- **Fix Confidence Scoring**: Provides confidence levels for recommended fixes
- **Comprehensive Reporting**: Rich console output with color-coded results, failure categorization, and actionable recommendations

#### Common Failure Types Identified

1. **Test Issues**: Incorrect test expectations, assertion logic errors, test setup problems
2. **Code Issues**: Implementation bugs, performance problems, API changes, type mismatches
3. **Environment Issues**: Network connectivity, external service dependencies, system configuration
4. **Dependency Issues**: Missing packages, import errors, version conflicts

#### Example Output

```
Test Verification Summary
┏━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric      ┃ Count ┃ Percentage ┃
┡━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ Total Tests │    22 │      100.0%│
│ Passed      │    21 │       95.5%│
│ Failed      │     1 │        4.5%│
│ Skipped     │     0 │        0.0%│
└─────────────┴───────┴────────────┘

Root Cause: Test expectation mismatch - concurrent execution timing issue
Recommendation: Fix test timing by creating proper contention scenarios
Agent Consultation: pytest-test-engineer (HIGH priority)
Fix Confidence Score: 85%
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

## Logging Options

The application supports four logging levels to control output verbosity:

### Default Logging (INFO only)
```bash
python alchimest.py sfr --symbols SPY  # Shows only INFO messages
```

### Warning Logging (INFO + WARNING)
```bash
python alchimest.py sfr --warning --symbols SPY  # Shows INFO and WARNING messages
```

### Error Logging (INFO + WARNING + ERROR + CRITICAL)
```bash
python alchimest.py sfr --error --symbols SPY  # Shows INFO, WARNING, ERROR and CRITICAL messages
```

### Debug Logging (All levels including DEBUG)
```bash
python alchimest.py sfr --debug --symbols SPY  # Shows DEBUG, INFO, WARNING, ERROR, CRITICAL messages
```

### File Logging
```bash
python alchimest.py sfr --log trading.log --symbols SPY  # Log to file (same filter as console)
python alchimest.py sfr --debug --log debug.log --symbols SPY  # Debug mode logs all levels to file
python alchimest.py sfr --error --log error.log --symbols SPY  # Error mode logs INFO, WARNING, ERROR, CRITICAL to file
```

**Note**: Debug mode takes precedence over error and warning modes when multiple flags are used.

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
