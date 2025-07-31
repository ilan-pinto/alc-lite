# Coding Style and Conventions

## Code Formatting
- **black** formatting with 88 character line length
- **isort** for import sorting with black profile
- Type hints required for all function definitions
- Docstrings in Google/NumPy style

## Naming Conventions
- **Classes**: PascalCase (e.g., `ArbitrageClass`, `SFRExecutor`)
- **Functions/Methods**: snake_case (e.g., `calculate_combo_limit_price`)
- **Variables**: snake_case (e.g., `contract_ticker`, `profit_target`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_CONTRACTS`)

## Architecture Patterns
- **Strategy Pattern**: All strategies inherit from `ArbitrageClass`
- **Executor Pattern**: Each strategy has corresponding executor class
- **Dataclasses**: Used for structured data (e.g., `ExpiryOption`, `OpportunityScore`)
- **Async/Await**: Extensive use for IB API interactions
- **Caching**: TTL-based caching for contract qualification

## Error Handling
- Comprehensive logging at DEBUG, INFO, WARNING, ERROR levels
- Try-catch blocks for IB API calls with specific error handling
- Metrics collection for rejection reasons and performance tracking
