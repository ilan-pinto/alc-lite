# ADR-002: Fix Metric Collection and Logging System in SFR Module

## Status
Completed ✅

## Context
The SFR.py module contains a complex metric collection and logging system with several critical issues that affect observability, debugging, and performance analysis:

1. **Inappropriate log levels**: Funnel stage transitions are logged as WARNING instead of INFO
2. **Inconsistent logging behavior**: Mixed use of WARNING, INFO, and DEBUG levels without clear guidelines
3. **Incomplete rejection tracking**: Not all rejection paths are properly recorded in metrics
4. **Duplicate logging**: Same information logged multiple times in different formats
5. **Log file output issues**: Log file only captures WARNING messages, missing important INFO level data

### Current Problems
- The `sfr_log.txt` file only shows rejection messages, missing funnel progression data
- Funnel stages are logged at WARNING level (lines 612, 623, 655, 683, 718, 738, 785, 836, 915)
- Multiple metric collection systems running in parallel without coordination
- Early exit paths don't record metrics consistently
- Debug messages for specific symbols (AAPL) pollute the logs

### Evidence
```
2025-08-20 17:28:09,680 - modules.Arbitrage.metrics - WARNING - [avgo] REJECTED - Insufficient Valid Strikes: found 0 strikes, need 2
2025-08-20 17:28:11,644 - modules.Arbitrage.metrics - WARNING - [pltr] REJECTED - Arbitrage Condition Not Met: spread 1.62 > net credit 1.57
2025-08-20 18:06:34,571 - modules.Arbitrage.metrics - WARNING - [spy] REJECTED - Profit Target Not Met: target 0.15% > actual ROI 0.14%
```

The log file shows only rejection messages because funnel stages are incorrectly logged as WARNING, but the logging configuration filters them out or they're not reaching the file handler properly.

## Decision
Implement a comprehensive fix to the metric collection and logging system with the following changes:

### 1. Fix Log Levels (Priority: HIGH)
- Change all funnel stage logging from `logger.warning()` to `logger.info()`
- Add `[Funnel]` prefix to all funnel stage messages for better identification
- Remove duplicate debug messages for AAPL symbol

### 2. Consolidate Metric Recording (Priority: HIGH)
- Ensure ALL rejection paths record metrics properly
- Add missing metric recordings for early exits (stock ticker unavailable, invalid stock price, viability check failures)

### 3. Clean Up Duplicate Logging (Priority: MEDIUM)
- Remove redundant funnel stage logging or consolidate to key stages only
- Simplify the funnel summary output to single-line summaries
- Reduce verbosity in contract request logging

### 4. Fix Log File Output (Priority: LOW)
- Configure logging to capture INFO level messages in log files
- Consider separate log files for operations vs rejections
- Implement log rotation for large files

## Implementation Details

### Files Modified
- `/modules/Arbitrage/SFR.py` - Main fixes for logging levels and duplicate cleanup
- `/modules/Arbitrage/common.py` - Logging configuration updates

### Example Changes

#### Before:
```python
logger.warning(f"[{self.symbol}] Funnel Stage: evaluated (expiry: {expiry_option.expiry})")
```

#### After:
```python
logger.info(f"[Funnel] [{self.symbol}] Stage: evaluated (expiry: {expiry_option.expiry})")
```

### Lines Modified in SFR.py
- Lines 177-185, 601-609, 796-810, 876-888, 1195-1203: Removed AAPL debug messages
- Lines 612, 623, 655, 683, 718, 738, 785, 836, 915: Changed WARNING to INFO with [Funnel] prefix
- Lines 620, 634, 680: Added missing metric recording for early exits
- Lines 1343-1408: Simplified funnel summary output
- Lines 1846-1857: Reduced contract request logging verbosity

## Consequences

### Positive
- **Better observability**: Clear separation between warnings and informational messages
- **Improved debugging**: `[Funnel]` prefix makes it easy to filter funnel-related logs
- **Complete metrics**: All rejection paths will be tracked consistently
- **Cleaner logs**: Removal of duplicate and debug messages reduces noise
- **Accurate analytics**: Unified metric collection provides reliable data for performance analysis

### Negative
- **Breaking change**: Log parsing tools may need updates to handle new `[Funnel]` format
- **Initial effort**: Requires careful testing to ensure no critical metrics are lost
- **Log volume**: INFO level logging may initially increase log file size

## Validation
- [x] All funnel stages now logged at INFO level with consistent `[Funnel]` format
- [x] AAPL-specific debug messages removed
- [x] Early exit paths record metrics properly (stock ticker, stock price, viability checks)
- [x] Log file captures INFO level messages with rotating file handler
- [x] ~~Separate rejection log file created for WARNING+ messages~~ (Removed per user request - single file preferred)
- [x] Funnel summary simplified to single-line format
- [x] Contract logging reduced from detailed per-contract to summary format
- [x] No loss of critical debugging information
- [x] Performance impact testing (verified - no significant impact)
- [x] Empty log file issue resolved with `delay=False` and explicit flush calls
- [x] `_flush_all_handlers()` method implemented for proper cleanup

## Rollback Plan
If issues arise:
1. Revert log levels back to WARNING for funnel stages
2. Restore AAPL debug messages if needed for troubleshooting
3. Restore original logging configuration
4. Git revert commit if necessary

## References
- Original analysis: SFR.py metric collection system review
- Related modules: `metrics.py`, `data_collection_metrics.py`
- Issue: Inconsistent logging levels and incomplete rejection tracking
- Log sample: `sfr_log.txt` showing incomplete data capture
