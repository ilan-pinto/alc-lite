# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Phase 1: Daily Options Data Collection (Israel Timezone)

### Quick Start
```bash
# Manual collection run (Python wrapper - prevents timeout issues)
python scheduler/run_israel_collection_python.py

# Manual collection run (legacy shell script)
./scheduler/run_collection_now_israel.sh

# Check collection status
python scheduler/check_collection_status.py

# Force collection (even if already done today)
python scheduler/run_israel_collection_python.py --force

# Force collection with data truncation (deletes today's data first)
python scheduler/run_israel_collection_python.py --force --truncate

# Truncate today's data manually (with confirmation prompt)
./scheduler/truncate_today_collection.sh

# Collect specific symbol only
python scheduler/run_israel_collection_python.py --symbols=SPY

# Watch status (auto-refresh every 30s)
python scheduler/check_collection_status.py --watch
```

### Scheduler Management
```bash
# Install LaunchAgent for automatic daily collection
launchctl load ~/Library/LaunchAgents/com.alclite.daily-collector-israel.plist
launchctl load ~/Library/LaunchAgents/com.alclite.friday-expiry-israel.plist

# Check if scheduler is running
launchctl list | grep alclite

# Manually trigger scheduled job
launchctl start com.alclite.daily-collector-israel

# Disable automatic collection
launchctl unload ~/Library/LaunchAgents/com.alclite.daily-collector-israel.plist

# View logs
tail -f logs/daily_collector.log
tail -f logs/daily_collector_error.log
```

### Collection Schedule (Israel Time)
- **Daily EOD Collection**: 11:45 PM IST / 12:45 AM IDT
- **Friday Expiry Check**: 10:00 PM IST / 11:00 PM IDT (Fridays only)
- **Morning Health Check**: 8:00 AM Israel time

### Monitoring Queries
```sql
-- Check today's collection status
SELECT symbol, collection_type, status, records_collected
FROM daily_collection_status
WHERE collection_date = CURRENT_DATE;

-- Check collection health
SELECT * FROM check_collection_health(NULL);

-- Find data gaps
SELECT symbol, gap_date, gap_type
FROM data_gaps
WHERE backfilled = false
ORDER BY gap_date DESC;
```

### Handling Collection Errors

#### Duplicate Key Violation
If you encounter this error when forcing collection:
```
asyncpg.exceptions.UniqueViolationError: duplicate key value violates unique constraint
DETAIL: Key (collection_date, symbol, collection_type)=(2025-08-11, ALL, end_of_day) already exists.
```

**Solutions:**
1. Use the truncate flag: `./scheduler/run_collection_now_israel.sh --force --truncate`
2. Manually truncate: `./scheduler/truncate_today_collection.sh`
3. Direct SQL: `DELETE FROM daily_collection_status WHERE collection_date = CURRENT_DATE;`

### Prerequisites for Phase 1
1. **TimescaleDB** running on port 5433
2. **IB Gateway/TWS** running on port 7497 (paper trading)
3. **Python venv** activated with dependencies installed

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

## Data Collection Systems

### Two Collection Systems

This project has **two distinct data collection systems**:

1. **Historical Bars Collection** (`scheduler/run_collection_python.py`)
   - Collects historical bar data (OHLCV) for backtesting
   - Uses `historical_bars_collector.py` internally
   - Collection types: morning, midday, afternoon, eod, late_night, gap_fill, manual
   - Purpose: Populate historical database for strategy backtesting
   - Schedule: Multiple times per day via LaunchAgents

2. **Daily Options Collection** (`scheduler/run_israel_collection_python.py`)
   - Collects live options chain data for arbitrage scanning
   - Uses `daily_collector.py` internally
   - Collection types: end_of_day, friday_expiry_check, morning_check
   - Purpose: Real-time options data for arbitrage detection
   - Schedule: Daily at 11:45 PM Israel time, Friday expiry checks

### Python Wrappers vs Shell Scripts

Both systems now use **Python wrappers** instead of shell scripts to prevent asyncio.TimeoutError issues:
- Shell scripts caused subprocess isolation that interfered with ib_async connections
- Python wrappers maintain proper environment context and prevent timeouts
- Both wrappers include HTML stats page generation with auto-opening

## Running with PyPy for Enhanced Performance 🏎️

PyPy provides **2-10x performance improvements** for calculation-intensive operations like options chain processing and arbitrage detection.

### Quick Setup
```bash
# Install PyPy environment
./scripts/setup_pypy_conda.sh

# Run with PyPy (Method 1: Direct)
conda activate alc-pypy
pypy3 alchimest.py sfr --symbols SPY QQQ --debug

# Run with PyPy (Method 2: Convenience script)
./scripts/run_with_pypy.sh sfr --symbols SPY QQQ --debug
```

### Performance Comparison
```bash
# Benchmark PyPy vs CPython performance
./benchmarks/compare_runtimes.sh
```

### When to Use PyPy
- ✅ **Use PyPy for**: Long-running scans (5+ symbols), data collection pipelines, production trading
- ❌ **Use CPython for**: Quick development, single-symbol scans, interactive testing

### Expected Performance Gains
- **Options chain processing**: 3-5x faster
- **Arbitrage detection**: 2-4x faster
- **Data collection**: 2-5x faster
- **Parallel execution monitoring**: 2-3x faster

### PyPy Environment Management
```bash
# Create environment
./scripts/setup_pypy_conda.sh

# Activate/deactivate
conda activate alc-pypy
conda deactivate

# Test installation
pypy3 -c "import sys; print(f'PyPy {sys.pypy_version_info} ready!')"
```

For detailed PyPy documentation, see: `docs/PYPY_PERFORMANCE.md`

## Parallel Test Execution (Python + PyPy3)

Run tests with both CPython and PyPy3 interpreters simultaneously to ensure compatibility and compare performance:

### Quick Start
```bash
# Run all tests in parallel (with PyPy JIT warmup)
python scripts/parallel_test_runner.py tests

# Show real-time test progress with colors 🎬
python scripts/parallel_test_runner.py tests --live

# Show periodic progress updates
python scripts/parallel_test_runner.py tests --progress

# Run specific test file with warmup
python scripts/parallel_test_runner.py tests/test_cli_arguments.py

# Run with keyword filter
python scripts/parallel_test_runner.py tests -k "performance"

# Skip PyPy warmup for faster startup
python scripts/parallel_test_runner.py tests --skip-warmup

# Only run PyPy warmup (no tests) - useful for JIT preparation
python scripts/parallel_test_runner.py tests --warmup-only

# Run with pytest options
python scripts/parallel_test_runner.py tests --tb=short -v
```

### PyPy JIT Warmup
The parallel test runner includes **automatic PyPy JIT warmup** to ensure optimal performance:

- **🔥 Warmup Exercises**: Runs Fibonacci recursion, prime sieve, and matrix multiplication
- **🚀 JIT Optimization**: Pre-compiles common computational patterns
- **⏱️ Duration**: ~0.5-2 seconds warmup time for significant performance gains
- **💡 Smart**: Only warms up PyPy, CPython starts immediately

**Warmup Output Example:**
```
🔥 Warming up PyPy JIT compiler...
   🏃‍♂️ PyPy JIT Warmup - Running computational exercises...
      📈 Fibonacci(25): 75025 in 0.013s
      🔢 Primes to 10k: 1229 found in 0.003s
      🔢 Matrix multiply: 20x20 in 0.006s
   ✅ PyPy JIT warmup completed!
🚀 PyPy JIT is now warmed up and ready for optimal performance!
```

### Real-Time Progress Modes 🎬

**Live Mode (`--live`)**: Watch each test pass/fail in real-time with colors:
```
🎬 Live test progress mode enabled!
═══════════════════════════════════════════════════════════════
[CPython] ✓ test_sfr_command_with_all_valid_arguments [1]
[CPython] ✓ test_sfr_command_with_default_profit [2]
[PyPy3] ✓ test_sfr_command_with_all_valid_arguments [1]
[PyPy3] ✓ test_sfr_command_with_default_profit [2]
```

**Progress Mode (`--progress`)**: Periodic counter updates:
```
📊 Progress: CPython 5 tests | PyPy3 3 tests
📊 Progress: CPython 10 tests | PyPy3 8 tests
📊 Progress: CPython 15 tests | PyPy3 13 tests
```

**Benefits:**
- **Real-time feedback**: See tests as they pass/fail
- **Better debugging**: Identify slow or hanging tests immediately
- **Visual comparison**: Watch both interpreters progress side-by-side
- **Color-coded**: Blue for CPython, Green for PyPy3
- **Test counters**: Track progress with [1], [2], [3]...

### VS Code/Cursor IDE Integration

The parallel test setup includes full IDE integration:

**Keyboard Shortcuts:**
- `Cmd+Shift+T` → Run all tests in parallel
- `Cmd+Shift+F` → Run current file tests in parallel (when editing .py files)
- `Cmd+Shift+K` → Run tests with keyword filter prompt

**Launch Configurations:**
- "Test: Python + PyPy3 (Parallel)" - Smart parallel runner with aggregated results
- "Test: Python Only" - Standard pytest with CPython
- "Test: PyPy3 Only" - Standard pytest with PyPy3
- "Test: Both Interpreters" - Compound configuration (separate terminals)

**Command Palette Tasks:**
- "Test: Parallel (Python + PyPy3)" - Default parallel execution with warmup
- "Test: Parallel (Live Progress)" - Real-time test progress with colors
- "Test: Parallel (Progress Updates)" - Periodic progress updates
- "Test: Parallel (No Warmup)" - Skip PyPy warmup for faster startup
- "Test: PyPy Warmup Only" - Pre-warm PyPy JIT without running tests
- "Test: Python" - CPython only
- "Test: PyPy3" - PyPy3 only

### Understanding Performance Results

```
📊 PARALLEL TEST RESULTS COMPARISON
================================================================================

✅ CPython Results:
   Duration: 4.14s
   Passed:   25
   Failed:   0
   Total:    25

✅ PyPy3 Results:
   Duration: 9.94s
   Passed:   25
   Failed:   0
   Total:    25

🐢 PyPy Performance: 2.40x slower than CPython (JIT warmup?)

🎉 All tests pass on both interpreters!
```

**Performance Notes:**
- ✅ **PyPy faster**: Long-running tests with heavy computation
- 🐢 **PyPy slower**: Short tests with lots of setup/teardown (JIT warmup overhead)
- ⚖️ **Similar**: Medium-duration tests where JIT balances out

### Environment Requirements
- **CPython**: Uses default Python environment with all dependencies
- **PyPy3**: Uses `/Users/ilpinto/micromamba/envs/alc-pypy/bin/pypy3` environment

## Development Notes

- The codebase uses `ib_async` for asynchronous IB operations
- `rich` library provides formatted console output and logging
- Code style enforced with black (88 character line length)
- Type hints required (mypy configuration in pyproject.toml)
- Test coverage targets `commands` and `modules` packages
- CI/CD handles automated versioning and releases
- **PyPy support**: Automatic runtime detection with optimized code paths
