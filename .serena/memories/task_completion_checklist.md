# Task Completion Checklist

## Before Committing Code
1. **Format and Lint**
   ```bash
   black .
   isort .
   flake8 .
   mypy .
   ```

2. **Run Tests**
   ```bash
   python -m pytest tests/ -v
   python -m pytest tests/ --cov=commands --cov=modules
   ```

3. **Integration Testing**
   ```bash
   python -m pytest tests/ -m integration -v
   ```

## Code Quality Standards
- All functions must have type hints
- Docstrings required for classes and public methods
- Error handling with appropriate logging levels
- Performance considerations for IB API calls
- Caching implementation where beneficial

## Testing Requirements
- Unit tests for core logic
- Integration tests for IB connectivity (mocked)
- Coverage targets: commands/ and modules/ packages
- All tests must pass before merging
