# /verify-test - Automated Test Failure Analysis

## Purpose
Analyzes pytest test failures to identify root causes and provide fix recommendations.

## Usage
```bash
/verify-test tests/test_file.py
/verify-test tests/ --verbose
/verify-test tests/test_*.py --find-similar --consult-agents
```

## Arguments
- `test_path` (required): Path to test file(s) or directory
- `options` (optional):
  - `--verbose`: Detailed output with full failure analysis
  - `--find-similar`: Search for similar failures in other test files
  - `--consult-agents`: Get recommendations for specialized agent consultation
  - `--no-similar`: Skip similarity search for faster analysis

## Features

### Root Cause Analysis
- Classifies failures as test issues, code issues, environment issues, or dependency issues
- Identifies specific failure patterns (assertions, timeouts, imports, etc.)
- Provides confidence scoring for each diagnosis

### Pattern Recognition
- Detects common failure patterns across the test suite
- Groups failures by type for better understanding
- Tracks failure frequency and impact

### Similarity Detection
- Finds similar failures across different test files
- Helps identify recurring problems
- Uses pattern matching to find related issues

### Agent Recommendations
- Suggests which specialized agents to consult based on failure type:
  - `pytest-test-engineer`: For test-related issues
  - `algotrading-python-expert`: For trading/IB related issues
  - `options-backtesting-expert`: For backtesting issues
  - `code-reviewer`: For general code issues

### Confidence Scoring
- Provides confidence levels (0-100%) for fix recommendations
- Higher scores indicate more certain diagnoses
- Helps prioritize which fixes to attempt first

## Output Includes

1. **Test Execution Summary**
   - Total tests run
   - Pass/fail/skip counts
   - Success rate percentage
   - Execution time

2. **Detailed Failure Analysis**
   - File path and test name
   - Exception type and message
   - Failure classification
   - Severity level (CRITICAL/HIGH/MEDIUM/LOW)
   - Root cause explanation
   - Specific fix recommendations

3. **Similar Failure Detection**
   - Lists similar failures found in other tests
   - Similarity score (0-1.0)
   - Helps identify systemic issues

4. **Agent Consultation Recommendations**
   - Grouped by agent type
   - Priority levels (HIGH/MEDIUM/LOW)
   - Specific consultation guidance

5. **Quick Fix Summary**
   - Recommendations grouped by failure type
   - Actionable steps for resolution
   - Overall fix confidence score

6. **fix tests**
   -  for cases the overall fix is only tests related and fix fix the issue

## Example Output

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

Failure 1: test_concurrent_execution_creates_contention
File: tests/test_parallel_executor.py
Exception: AssertionError
Type: Test Issue
Severity: MEDIUM
Confidence: 85%

Root Cause: Test expectation mismatch - concurrent execution timing issue

Recommendation: Fix test timing by creating proper contention scenarios

Similar Failures Found:
  • test_execution_timing (score: 0.78)
  • test_parallel_execution (score: 0.65)

Agent Consultation Recommendations:
pytest-test-engineer:
  • HIGH - test_concurrent_execution_creates_contention
    Consult pytest-test-engineer agent for detailed analysis of Test Issue

Fix Confidence Score: 85%
```

## Implementation Details

The command uses the `TestFailureAnalyzer` class to:
1. Execute pytest on specified test files
2. Parse failure output using regex patterns
3. Classify failures based on error messages and stack traces
4. Calculate similarity scores between failures
5. Generate actionable recommendations

## When to Use

Use `/verify-test` when:
- Tests are failing and you need to understand why
- You want to find patterns in test failures
- You need recommendations for fixing test issues
- You want to identify which specialized agent to consult
- You're troubleshooting CI/CD test failures

## Tips

- Use `--verbose` for detailed analysis when debugging complex failures
- Use `--find-similar` to identify systemic issues affecting multiple tests
- Use `--consult-agents` when you need specialized help for specific failure types
- Run on entire test directories to get a comprehensive health check
- Check the fix confidence score to prioritize which issues to address first
