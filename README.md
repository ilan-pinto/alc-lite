# Alchimist-Lite (alc-lite)

**Alchimist-Lite** is a powerful command-line tool designed for traders and financial analysts to scan for and analyze arbitrage opportunities in the options market. It currently supports Synthetic-Free-Risk (SFR) and Synthetic (non-risk-free) strategies.

The project is built with a focus on modularity and extensibility, allowing for the easy addition of new arbitrage strategies. It leverages `ib-async` for interacting with Interactive Brokers and `rich` for beautiful and informative console output.

## 🚀 Features

- **Arbitrage Strategy Scanning**: Scan for SFR and Synthetic arbitrage opportunities.
- **Global Opportunity Selection**: Intelligently ranks and selects the best opportunities across all symbols and expirations using advanced scoring algorithms.
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

### 🎯 Global Opportunity Selection (New Feature)

The synthetic scanner now features an advanced **Global Opportunity Selection** system that intelligently ranks and selects the best opportunities across all symbols and expirations. Instead of per-symbol optimization, it performs portfolio-level optimization.

#### Pre-defined Scoring Strategies

Choose from four pre-configured strategies optimized for different trading styles:

```bash
# Conservative Strategy - Prioritizes safety and liquidity
python alchimest.py syn --symbols AAPL MSFT GOOGL --scoring-strategy conservative

# Aggressive Strategy - Maximizes risk-reward ratio
python alchimest.py syn --symbols TSLA NVDA AMD --scoring-strategy aggressive

# Balanced Strategy (Default) - Well-rounded approach
python alchimest.py syn --symbols SPY QQQ IWM --scoring-strategy balanced

# Liquidity-Focused Strategy - Emphasizes execution certainty
python alchimest.py syn --symbols AAPL META AMZN --scoring-strategy liquidity-focused
```

**Strategy Characteristics:**
- **Conservative**: 30% risk-reward, 35% liquidity, 20% time decay, 15% market quality
- **Aggressive**: 50% risk-reward, 15% liquidity, 20% time decay, 15% market quality
- **Balanced**: 40% risk-reward, 25% liquidity, 20% time decay, 15% market quality
- **Liquidity-Focused**: 25% risk-reward, 40% liquidity, 20% time decay, 15% market quality

#### Custom Scoring Configuration (Advanced)

For experienced traders who want fine-grained control over the scoring algorithm:

```bash
# Custom weights (must sum to 1.0)
python alchimest.py syn --symbols SPY QQQ \
  --risk-reward-weight 0.35 \
  --liquidity-weight 0.30 \
  --time-decay-weight 0.20 \
  --market-quality-weight 0.15

# Custom thresholds for opportunity filtering
python alchimest.py syn --symbols AAPL MSFT \
  --min-risk-reward 2.5 \
  --min-liquidity 0.6 \
  --max-bid-ask-spread 15.0 \
  --optimal-days-expiry 30
```

#### How Global Selection Works

1. **Collection Phase**: Scans all symbols and collects potential opportunities
2. **Scoring Phase**: Each opportunity is scored on four dimensions:
   - **Risk-Reward Ratio**: Max profit vs max loss potential
   - **Liquidity Score**: Based on volume and bid-ask spreads
   - **Time Decay Score**: Optimal around 30 days to expiration
   - **Market Quality**: Spread tightness and credit quality
3. **Selection Phase**: Ranks all opportunities by composite score
4. **Execution Phase**: Executes only the highest-scoring opportunity

#### Examples with Global Selection

```bash
# Scan multiple symbols with conservative approach
python alchimest.py syn --symbols SPY QQQ IWM TLT GLD \
  --scoring-strategy conservative \
  --cost-limit 150 \
  --quantity 2

# Aggressive strategy with custom thresholds
python alchimest.py syn --symbols TSLA NVDA AMD MRVL \
  --scoring-strategy aggressive \
  --min-risk-reward 3.0 \
  --cost-limit 200

# Custom balanced approach for liquid options
python alchimest.py syn --symbols AAPL MSFT GOOGL META AMZN \
  --risk-reward-weight 0.30 \
  --liquidity-weight 0.40 \
  --time-decay-weight 0.20 \
  --market-quality-weight 0.10 \
  --min-liquidity 0.7

# Debug mode to see scoring details
python alchimest.py syn --debug --symbols SPY QQQ \
  --scoring-strategy balanced \
  --log scoring_details.log
```

#### Understanding the Scoring Output

When running with `--debug`, you'll see detailed scoring information:

```
[INFO] Global Opportunity Added: AAPL
  Composite Score: 0.7845
  Risk-Reward: 2.50 (weight: 40%)
  Liquidity: 0.8532 (weight: 25%)
  Time Decay: 0.9667 (weight: 20%)
  Market Quality: 0.9125 (weight: 15%)

[INFO] Best Global Opportunity Selected: MSFT
  Symbol: MSFT, Expiry: 20240315
  Max Profit: $85.00, Max Loss: $-25.00
  Composite Score: 0.8234
```

### Scanning for Calendar Spreads
Calendar spreads are market-neutral options strategies that profit from the differential time decay between front and back month options. This strategy is most effective when the front month option has higher implied volatility and decays faster than the back month option.

```bash
python alchimest.py calendar --symbols SPY QQQ AAPL --cost-limit 300 --profit-target 0.25
```

**Core Parameters:**
- `--symbols`: A list of stock symbols to scan
- `--cost-limit`: The maximum net debit to pay for the calendar spread (default: $300)
- `--profit-target`: Target profit as percentage of maximum profit (default: 0.25 = 25%)
- `--quantity`: Maximum number of calendar spreads to execute (default: 1)

**Advanced Calendar Spread Parameters:**
- `--iv-spread-threshold`: Minimum IV spread between back and front months (default: 0.015 = 1.5%)
- `--theta-ratio-threshold`: Minimum theta ratio (front/back) required (default: 1.5)
- `--front-expiry-max-days`: Maximum days to front expiration (default: 45)
- `--back-expiry-min-days`: Minimum days to back expiration (default: 60)
- `--back-expiry-max-days`: Maximum days to back expiration (default: 120)
- `--min-volume`: Minimum daily volume per option leg (default: 10)
- `--max-bid-ask-spread`: Maximum bid-ask spread as percentage of mid (default: 0.15)

**Calendar Spread Examples:**
```bash
# Basic calendar spread scan with default settings
python alchimest.py calendar --symbols SPY QQQ AAPL

# High IV environment with stricter requirements
python alchimest.py calendar --symbols AAPL TSLA --iv-spread-threshold 0.04 --theta-ratio-threshold 2.0

# Conservative approach with tight expiry windows
python alchimest.py calendar --symbols SPY IWM --front-expiry-max-days 30 --back-expiry-min-days 90

# High-volume liquid options only
python alchimest.py calendar --symbols MSFT GOOGL --min-volume 50 --max-bid-ask-spread 0.10

# Multiple contracts with higher cost tolerance
python alchimest.py calendar --symbols SPY QQQ --cost-limit 500 --quantity 3 --profit-target 0.30

# Using Finviz screener for large-cap stocks
python alchimest.py calendar --fin "https://finviz.com/screener.ashx?v=111&f=cap_largeover" --cost-limit 400

# Debug mode with detailed logging
python alchimest.py calendar --debug --symbols AAPL --log calendar_debug.log
```

#### How Calendar Spreads Work

Calendar spreads profit when:
1. **IV Spread Advantage**: Back month IV is higher than front month IV (term structure inversion)
2. **Theta Decay Differential**: Front month option decays faster than back month (higher theta ratio)
3. **Price Stability**: Underlying stays near the strike price at front expiration
4. **Volatility Contraction**: Front month IV decreases relative to back month

#### Strategy Logic

The calendar spread scanner evaluates opportunities using:

**Quality Filters:**
- Minimum implied volatility spread between expiration months
- Theta ratio requirements ensuring favorable time decay
- Liquidity filters (volume, bid-ask spreads)
- Expiration window constraints

**Scoring System:**
- **IV Spread Score**: Higher scores for greater back-front IV differential
- **Theta Ratio Score**: Rewards higher front/back theta ratios
- **Profit Potential**: Based on maximum profit vs net debit ratio
- **Market Quality**: Considers liquidity and execution quality

**Global Selection Process:**
1. Scans all symbols for calendar spread opportunities
2. Applies quality filters and calculates composite scores
3. Ranks opportunities across all symbols and expirations
4. Executes the highest-scoring opportunity that meets all criteria

#### Risk Management Features

- **Cost Limits**: Maximum net debit controls per-trade risk
- **Quality Thresholds**: Ensures adequate liquidity and tight spreads
- **Expiration Management**: Controlled front/back month relationships
- **Circuit Breakers**: API failure protection and recovery
- **Performance Monitoring**: Comprehensive metrics and profiling

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

### Main Test Suite (pytest-compatible)

Run the comprehensive integration and unit tests:
```bash
python -m pytest tests/ -v
```

### Performance Testing & Analysis

Run standalone performance scripts (located in `scripts/`):
```bash
# Comprehensive performance testing
python scripts/performance/test_global_selection_performance.py
python scripts/performance/test_syn_executor_performance.py

# Performance comparison between strategies
python scripts/performance/performance_comparison_test.py

# Performance dashboard generation
python scripts/performance/performance_dashboard.py

# CPU and memory profiling
python scripts/profiling/profile_global_selection.py

# Offline testing (when markets are closed)
python scripts/test_arbitrage_offline.py
```

For more detailed testing information, see [TESTING_GUIDE.md](TESTING_GUIDE.md) and [scripts/README.md](scripts/README.md).

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

**IMPORTANT: READ THIS DISCLAIMER CAREFULLY BEFORE USING THIS SOFTWARE**

This software is provided for educational and research purposes only.

**NO LIABILITY**: The authors, contributors, and maintainers of this software accept NO responsibility whatsoever for any damages, financial losses, or negative consequences that may result from using this code. By using this software, you acknowledge and agree that:

1. **USE AT YOUR OWN RISK**: All usage of this code for trading or any other purpose is entirely at your own risk and responsibility.

2. **NO WARRANTIES**: This software is provided "AS IS" without warranty of any kind, either express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, or non-infringement.

3. **NOT FINANCIAL ADVICE**: Nothing in this software constitutes financial, investment, legal, tax or other professional advice. You should consult with qualified professionals before making any trading or investment decisions.

4. **TRADING RISKS**: Trading financial instruments, including options, involves substantial risk of loss and is not suitable for all investors. You may lose some or all of your invested capital. Past performance is not indicative of future results.

5. **YOUR RESPONSIBILITY**: You are solely responsible for:
   - Testing and validating the software before any live trading
   - Understanding the strategies and their risks
   - Ensuring compliance with all applicable laws and regulations
   - Any trading decisions made using this software
   - Any modifications or customizations you make to the code

6. **NO GUARANTEE OF PROFITS**: This software does not guarantee any profits or successful trades. Market conditions can change rapidly and unpredictably.

7. **INDEMNIFICATION**: You agree to indemnify and hold harmless the authors and contributors from any claims, damages, losses, or expenses arising from your use of this software.

**By using this software, you acknowledge that you have read, understood, and agree to be bound by this disclaimer.**

If you do not agree with these terms, do not use this software.
