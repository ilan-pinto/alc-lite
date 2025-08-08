# Analysis Tools

This directory contains analysis tools and notebooks for exploring and analyzing trading data.

## Files

### option_chain_analysis.ipynb
Comprehensive Jupyter notebook for analyzing option chain data by joining underlying securities, option contracts, and market data.

**Features:**
- Complete option chain data with latest prices and Greeks
- Side-by-side option chain summaries (calls/puts by strike)
- Arbitrage opportunity scanning using put-call parity
- Interactive data visualization and filtering
- Export functionality for analysis results

**Key Queries:**
1. **Complete Option Chain**: All option data with latest prices, Greeks, and moneyness calculations
2. **Option Summary**: Traditional option chain format with calls and puts side-by-side
3. **Arbitrage Scanner**: Identifies potential arbitrage opportunities by comparing actual vs synthetic stock prices

**Usage:**
```bash
cd /Users/ilpinto/dev/AlchimistProject/alc-lite/backtesting/infra/analysis
jupyter notebook option_chain_analysis.ipynb
```

**Prerequisites:**
- Jupyter notebook environment
- Database connection to the options_arbitrage database
- Required Python packages: pandas, numpy, psycopg2, sqlalchemy, matplotlib, seaborn

### vix_correlation.py
Python module for analyzing VIX volatility correlations with underlying securities.

## Database Integration

All analysis tools connect to the TimescaleDB instance with the following schema:
- `underlying_securities` - Stock information
- `option_chains` - Option contract details
- `market_data_ticks` - Real-time option prices and Greeks
- `stock_data_ticks` - Stock price history
- `vix_data_*` - VIX volatility data tables

## Getting Started

1. Ensure the database is running and populated with data:
   ```bash
   cd ../database/podman
   docker-compose up -d
   ```

2. Load historical data using the data collection pipeline:
   ```bash
   cd ../data_collection
   python load_historical_pipeline.py --symbol SPY --days 30 --include-vix
   ```

3. Launch Jupyter notebook:
   ```bash
   cd ../analysis
   jupyter notebook
   ```

## Analysis Workflows

### Option Chain Analysis
1. Open `option_chain_analysis.ipynb`
2. Run setup cells to establish database connection
3. Execute query cells to load option chain data
4. Use visualization cells to analyze patterns
5. Export results using the provided export functions

### Arbitrage Scanning
The notebook includes a comprehensive arbitrage scanner that:
- Compares actual stock prices with synthetic stock prices from put-call parity
- Identifies pricing discrepancies that may indicate arbitrage opportunities
- Calculates potential profits and risk metrics
- Visualizes opportunities by symbol, expiration, and time

### Custom Analysis
Use the custom analysis cells in the notebook to:
- Run custom SQL queries
- Analyze specific symbols or time periods
- Create custom visualizations
- Export filtered datasets

## Performance Notes

- Queries use `DISTINCT ON` for efficient latest price retrieval
- Results are limited to active securities and unexpired options
- Consider adding date filters for very large datasets
- Use the export functions to save analysis results for offline processing

## Contributing

When adding new analysis tools:
1. Document the purpose and usage in this README
2. Include example outputs and visualizations
3. Ensure database connections are properly managed
4. Add error handling for missing data scenarios
