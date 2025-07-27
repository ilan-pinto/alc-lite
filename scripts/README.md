# Scripts Directory

This directory contains standalone scripts for testing, performance analysis, and utilities that are not part of the main pytest test suite.

## Directory Structure

### `performance/`
Performance testing and comparison scripts for the arbitrage system:

- `test_global_selection_performance.py` - Comprehensive performance testing for Global Opportunity Selection system
- `test_syn_executor_performance.py` - Performance testing for Syn class and SynExecutor
- `test_syn_simple_performance.py` - Simple performance test for SynExecutor and GlobalOpportunityManager
- `performance_test.py` - Performance comparison test for market data collection optimization
- `performance_comparison_test.py` - Performance comparison between old per-symbol and new global selection
- `performance_dashboard.py` - Performance visualization and reporting dashboard

### `profiling/`
Profiling utilities for detailed performance analysis:

- `profile_global_selection.py` - CPU and memory profiling for Global Opportunity Selection

### `utils/`
Utility scripts for displaying results and data:

- `show_performance_results.py` - Display performance test results in clean format

### Main Scripts
- `test_arbitrage_offline.py` - Standalone arbitrage testing script for market-closed scenarios

## Usage

These scripts are designed to be run independently and are not integrated with pytest. They provide detailed performance analysis, profiling capabilities, and specialized testing scenarios.

### Running Performance Tests
```bash
# Run performance tests
python scripts/performance/test_global_selection_performance.py
python scripts/performance/test_syn_executor_performance.py

# Generate performance dashboard
python scripts/performance/performance_dashboard.py

# Show results
python scripts/utils/show_performance_results.py
```

### Running Profiling
```bash
# Profile global selection performance
python scripts/profiling/profile_global_selection.py
```

### Running Offline Tests
```bash
# Test arbitrage scenarios when markets are closed
python scripts/test_arbitrage_offline.py
python scripts/test_arbitrage_offline.py --scenario dell_profitable
```

## Notes

- These scripts are separate from the main `tests/` directory which contains pytest-compatible integration tests
- Performance scripts generate JSON reports that can be visualized with the dashboard
- All scripts include proper error handling and detailed output formatting
