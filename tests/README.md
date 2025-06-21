# Alchimest Test Suite

This directory contains comprehensive integration tests for the alchimest CLI tool, specifically validating all arguments for the `sfr` and `syn` commands.

## Test Structure

- `test_cli_arguments.py` - Main integration tests for CLI argument validation
- `conftest.py` - Pytest configuration and common fixtures
- `__init__.py` - Makes tests a Python package

## Test Coverage

The test suite covers:

### SFR Command Tests
- ✅ All valid arguments (`-s`, `-p`, `-l`)
- ✅ Default profit value (None)
- ✅ Default cost limit (120)
- ✅ No symbols provided (uses default list)
- ✅ Special symbols (!MES, @SPX, SPY)
- ✅ Negative profit values
- ✅ Zero values
- ✅ Large values
- ✅ Help command

### SYN Command Tests
- ✅ All valid arguments (`-s`, `-l`, `-ml`, `-mp`)
- ✅ Default cost limit (120)
- ✅ No symbols provided (uses default list)
- ✅ Only max_profit specified
- ✅ Only max_loss specified
- ✅ No optional arguments
- ✅ Special symbols (!MES, @SPX, SPY)
- ✅ Negative values
- ✅ Zero values
- ✅ Large values
- ✅ Help command

### General CLI Tests
- ✅ Invalid command handling
- ✅ No command provided
- ✅ Main help command

## Running Tests

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run All Tests
```bash
pytest tests/ -v
```

### Run Only Integration Tests
```bash
pytest tests/ -m integration -v
```

### Run with Coverage
```bash
pytest tests/ --cov=commands --cov=modules --cov-report=html
```

### Run Specific Test File
```bash
pytest tests/test_cli_arguments.py -v
```

### Run Specific Test
```bash
pytest tests/test_cli_arguments.py::TestCLIArguments::test_sfr_command_with_all_valid_arguments -v
```

## Test Features

- **Mocking**: Uses `unittest.mock` to avoid actual IB connections
- **Argument Validation**: Tests all argument combinations and edge cases
- **Default Values**: Validates that default values are correctly applied
- **Error Handling**: Tests invalid commands and missing arguments
- **Special Cases**: Tests negative values, zero values, and large values
- **Symbol Types**: Tests regular symbols, futures (!MES), and indices (@SPX)

## Test Markers

- `@pytest.mark.integration` - Marks tests as integration tests
- `@pytest.mark.unit` - Marks tests as unit tests (not used yet)
- `@pytest.mark.slow` - Marks tests as slow running (not used yet)

## Configuration

Pytest configuration is in `pyproject.toml`:
- Test discovery in `tests/` directory
- Coverage reporting for `commands` and `modules` packages
- HTML coverage report generation
- Verbose output and short tracebacks

## Adding New Tests

To add new tests:

1. Create a new test file in the `tests/` directory
2. Use the `@pytest.mark.integration` decorator for integration tests
3. Use the provided fixtures for mocking and setup
4. Follow the existing naming convention: `test_*`
5. Add comprehensive docstrings explaining what each test validates

## Example Test Structure

```python
@pytest.mark.integration
def test_new_feature(self, mock_option_scan, capture_output):
    """Test description of what this test validates"""
    test_args = ['alchimest.py', 'command', 'arg1', 'arg2']
    
    with patch.object(sys, 'argv', test_args):
        alchimest.main()
    
    # Assert expected behavior
    mock_option_scan.method.assert_called_once_with(expected_args)
``` 