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
python alchimest.py sfr --symbols MSFT AAPL GOOG --cost-limit 100 --profit 0.75
```
- `--symbols`: A list of stock symbols to scan.
- `--cost-limit`: The maximum price you are willing to pay for the BAG contract.
- `--profit`: The minimum required ROI for a trade to be considered.

### Scanning for Synthetic (Syn) Opportunities
To scan for Synthetic (non-risk-free) opportunities:
```bash
python alchimest.py syn --symbols TSLA NVDA --cost-limit 120 --max-loss 50 --max-profit 100
```
- `--symbols`: A list of stock symbols to scan.
- `--cost-limit`: The maximum price for the contract.
- `--max-loss`: The maximum acceptable loss for the trade.
- `--max-profit`: The maximum target profit for the trade.

## 🧪 Testing

To run the test suite, use `pytest`:
```bash
python -m pytest tests/ -v
```

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is provided for educational and research purposes only. Trading financial instruments involves significant risk. The authors and contributors are not responsible for any financial losses. Use at your own risk.
