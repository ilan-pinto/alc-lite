# Essential Development Commands

## Setup
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## Running Strategies
```bash
# SFR arbitrage scanning
python alchimest.py sfr --symbols MSFT AAPL --cost-limit 100 --profit 0.75

# Synthetic arbitrage scanning
python alchimest.py syn --symbols TSLA NVDA --cost-limit 120 --max-loss 50

# Debug mode with logging
python alchimest.py sfr --debug --log debug.log --symbols SPY
```

## Testing
```bash
python -m pytest tests/ -v                    # Run all tests
python -m pytest tests/ --cov=commands --cov=modules --cov-report=html  # With coverage
python -m pytest tests/ -m integration -v     # Integration tests only
```

## Code Quality
```bash
black .           # Format code
isort .           # Sort imports
mypy .            # Type checking
flake8 .          # Linting
```

## Performance Testing
```bash
python scripts/performance/test_global_selection_performance.py
python scripts/performance/performance_dashboard.py
```
