# Custom Claude Code Slash Commands

This directory contains custom slash commands for the alc-lite project.

## Available Commands

### `/verify-test` - Automated Test Failure Analysis

**Purpose**: Analyzes pytest test failures to identify root causes and provide fix recommendations.

**Usage**:
```
/verify-test tests/test_file.py
/verify-test tests/ --verbose
/verify-test tests/test_*.py --find-similar --consult-agents
```

**Arguments**:
- `test_path` (required): Path to test file(s) or directory
- `options` (optional):
  - `--verbose`: Detailed output with full failure analysis
  - `--find-similar`: Search for similar failures in other test files
  - `--consult-agents`: Get recommendations for specialized agent consultation

**Features**:
- **Root Cause Analysis**: Classifies failures as test issues, code issues, environment issues, or dependency issues
- **Pattern Recognition**: Identifies common failure patterns (assertions, timeouts, imports, etc.)
- **Similarity Detection**: Finds similar failures across the test suite
- **Agent Recommendations**: Suggests which specialized agents to consult based on failure type
- **Confidence Scoring**: Provides confidence levels for fix recommendations

**Output Includes**:
1. Test execution summary with pass/fail counts
2. Detailed failure analysis with root causes
3. Fix recommendations with confidence scores
4. Similar failure detection results
5. Agent consultation recommendations
6. Quick fix summary by failure type

**Example Output**:
```
Test Summary: 3 tests run, 1 passed, 2 failed
Failures:
  - test_assertion_failure: TEST ISSUE (confidence: 95%)
    Root Cause: Test expectation mismatch
    Recommendation: Review test expectations and verify expected values
    Agent: pytest-test-engineer

  - test_import_error: DEPENDENCY ISSUE (confidence: 90%)
    Root Cause: Missing module dependency
    Recommendation: Install missing package or check import path
    Agent: code-reviewer
```

## How to Use Slash Commands

1. Type `/verify-test` in Claude Code
2. Provide the test path as the first argument
3. Add optional flags for enhanced analysis
4. Claude will automatically run the analysis and provide detailed results

## Command Development

To create new slash commands:
1. Create a new `.md` file in `.claude/commands/`
2. Add frontmatter with argument definitions
3. Include `thinking: true` for analysis commands
4. Write the command logic in markdown format

See `verify-test.md` for a complete example.
