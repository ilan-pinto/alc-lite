# Alchimist-Lite (alc-lite)

**Alchimist-Lite** is a powerful command-line tool designed for traders and financial analysts to scan for and analyze arbitrage opportunities in the options market. It currently supports Synthetic-Free-Risk (SFR) and Synthetic (non-risk-free) strategies.

The project is built with a focus on modularity and extensibility, allowing for the easy addition of new arbitrage strategies. It leverages `ib-async` for interacting with Interactive Brokers and `rich` for beautiful and informative console output.

## 🚀 Features

- **Arbitrage Strategy Scanning**: Scan for SFR and Synthetic arbitrage opportunities.
- **Interactive Brokers Integration**: Connects to IBKR for real-time market data.
- **Extensible Architecture**: Easily add new strategies by inheriting from the `ArbitrageClass`.
- **Command-Line Interface**: Simple and intuitive CLI for running scans.
- **CI/CD Automation**: Automated versioning, testing, and releases powered by GitHub Actions.

## ⚙️ Installation

To get started with Alchimist-Lite, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ilpinto/alc-lite.git
    cd alc-lite
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main entry point for the tool is `alchimest.py`. You can run different strategies using the available sub-commands.

### General Help
To see all available commands and options, run:
```bash
python alchimest.py --help
```

### Scanning for Synthetic-Free-Risk (SFR)
To scan for SFR opportunities for a list of symbols:
```bash
python alchimest.py sfr --symbols MSFT AAPL GOOG --cost-limit 100 --profit 0.75 --quantity 2
```

**Parameters:**
- `--symbols`: A list of stock symbols to scan.
- `--cost-limit`: The maximum price you are willing to pay for the BAG contract.
- `--profit`: The minimum required ROI for a trade to be considered.
- `--quantity`: The maximum number of contracts to purchase (default: 1).

**Advanced SFR Examples:**
```bash
# Basic SFR scan with default settings
python alchimest.py sfr --symbols SPY QQQ

# High-profit threshold scan with logging
python alchimest.py sfr --symbols MSFT AAPL --profit 1.0 --cost-limit 150 --log sfr_scan.log

# Debug mode with multiple symbols
python alchimest.py sfr --debug --symbols META GOOGL AMZN --cost-limit 200

# Using Finviz screener integration
python alchimest.py sfr --fin "https://finviz.com/screener.ashx?v=111&f=cap_largeover" --cost-limit 100

# Index and futures options (prefix with ! for futures, @ for indices)
python alchimest.py sfr --symbols !MES @SPX --cost-limit 80
```

### Scanning for Synthetic (Syn) Opportunities
To scan for Synthetic (non-risk-free) opportunities:
```bash
python alchimest.py syn --symbols TSLA NVDA --cost-limit 120 --max-loss 50 --max-profit 100 --quantity 3
```

**Parameters:**
- `--symbols`: A list of stock symbols to scan.
- `--cost-limit`: The maximum price for the contract.
- `--max-loss`: The maximum acceptable loss for the trade.
- `--max-profit`: The maximum target profit for the trade.
- `--quantity`: The maximum number of contracts to purchase (default: 1).

**Advanced Synthetic Examples:**
```bash
# Conservative synthetic scan with tight risk controls
python alchimest.py syn --symbols AAPL MSFT --max-loss 25 --max-profit 75 --cost-limit 100

# Aggressive profit ratio targeting
python alchimest.py syn --symbols TSLA NVDA --profit-ratio 2.5 --cost-limit 200

# Multi-contract trading with logging
python alchimest.py syn --symbols SPY QQQ IWM --quantity 5 --log synthetic_trades.log --debug

# Using Finviz for high-volume stocks
python alchimest.py syn --fin "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o2000" --max-loss 40
```

### Performance Metrics and Monitoring
The application includes comprehensive metrics collection to track scanning performance:

```bash
# Generate performance metrics report (JSON format)
python alchimest.py metrics --format json --output metrics_report.json

# Display metrics summary to console
python alchimest.py metrics --format console

# Reset metrics after generating report
python alchimest.py metrics --format json --reset
```

**Available Metrics:**
- Scan cycle performance and timing
- Contract data collection efficiency
- Opportunity detection rates
- Rejection reason analysis
- Order placement success rates

### Logging and Debug Options
Control the verbosity of output with different logging levels:

```bash
# Default logging (INFO level only)
python alchimest.py sfr --symbols SPY

# Warning level (INFO + WARNING messages)
python alchimest.py sfr --warning --symbols SPY

# Debug level (all message types: DEBUG, INFO, WARNING, ERROR)
python alchimest.py sfr --debug --symbols SPY

# File logging with debug information
python alchimest.py sfr --debug --log debug_output.log --symbols SPY
```

**Note:** Debug mode takes precedence over warning mode when both flags are used.

### Special Symbol Formats
The application supports different asset types with special prefixes:

- **Stocks**: No prefix required (e.g., `AAPL`, `MSFT`, `GOOGL`)
- **Index Options**: Use `@` prefix (e.g., `@SPX`, `@NDX`)
- **Futures Options**: Use `!` prefix (e.g., `!MES`, `!NQ`)

Example with mixed asset types:
```bash
python alchimest.py sfr --symbols AAPL @SPX !MES --cost-limit 120
```

### Integration with Finviz Screeners
Automatically extract symbols from Finviz screener URLs:

```bash
# Large cap stocks
python alchimest.py sfr --fin "https://finviz.com/screener.ashx?v=111&f=cap_largeover"

# High volume, liquid stocks
python alchimest.py syn --fin "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o2000"

# Tech sector momentum stocks
python alchimest.py sfr --fin "https://finviz.com/screener.ashx?v=111&f=sec_technology,ta_perf_1wup"
```

## 🧪 Testing

To run the test suite, use `pytest`:
```bash
python -m pytest tests/ -v
```

For more detailed testing information, see [TESTING_GUIDE.md](TESTING_GUIDE.md).

## 🔧 Troubleshooting

### Common Issues and Solutions

**Connection Issues with Interactive Brokers:**
```bash
# Ensure IB Gateway or TWS is running on the correct port (default: 7497)
# Check that API connections are enabled in TWS/IB Gateway settings

# Test connection with a simple scan
python alchimest.py sfr --symbols SPY --debug
```

**No Opportunities Found:**
```bash
# Try relaxing the profit requirements
python alchimest.py sfr --symbols SPY --profit 0.25 --cost-limit 200

# Use debug mode to see rejection reasons
python alchimest.py sfr --symbols SPY --debug
```

**High Memory Usage:**
```bash
# Use fewer symbols per scan
python alchimest.py sfr --symbols SPY QQQ  # Instead of many symbols

# Monitor memory usage
python -m memory_profiler alchimest.py sfr --symbols SPY
```

**Slow Performance:**
```bash
# Enable metrics to analyze bottlenecks
python alchimest.py sfr --symbols SPY
python alchimest.py metrics --format console

# Use warning mode instead of debug for better performance
python alchimest.py sfr --warning --symbols SPY QQQ
```

**Invalid Symbol Errors:**
```bash
# Verify symbol format for different asset types:
python alchimest.py sfr --symbols AAPL      # Stocks: no prefix
python alchimest.py sfr --symbols @SPX      # Index options: @ prefix
python alchimest.py sfr --symbols !MES      # Futures options: ! prefix
```

### Configuration Tips

**Optimal Settings for Different Trading Styles:**

*Conservative (Risk-Averse):*
```bash
python alchimest.py sfr --symbols SPY QQQ --profit 0.75 --cost-limit 100 --quantity 1
```

*Moderate (Balanced):*
```bash
python alchimest.py syn --symbols AAPL MSFT --max-loss 50 --max-profit 150 --cost-limit 150
```

*Aggressive (High-Risk/High-Reward):*
```bash
python alchimest.py syn --symbols TSLA NVDA --profit-ratio 3.0 --cost-limit 300 --quantity 3
```

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is provided for educational and research purposes only. Trading financial instruments involves significant risk. The authors and contributors are not responsible for any financial losses. Use at your own risk.
