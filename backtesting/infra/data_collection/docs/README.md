# Historical Data Loading Pipeline

A comprehensive, production-ready pipeline for loading historical options and stock data from Interactive Brokers into your backtesting database. This pipeline provides robust data collection with progress tracking, validation, error handling, and flexible configuration options.

## Features

- 🚀 **Command-line interface** with intuitive parameters
- 📊 **Multi-symbol batch processing** with progress tracking
- 🔄 **Smart backfill** to identify and fill missing data gaps
- ✅ **Data validation** with quality checks and metrics
- 🎛️ **Flexible configuration** via YAML files and CLI overrides
- 📈 **Rich progress visualization** with real-time statistics
- 🛡️ **Robust error handling** with retry logic and recovery
- 📝 **Comprehensive logging** with multiple output formats
- ⚡ **Rate limiting** to respect IB API constraints

## Quick Start

### Prerequisites

1. **Database Setup**: Ensure TimescaleDB container is running:
   ```bash
   cd backtesting/infra/database/podman
   docker-compose up -d
   ```

2. **Interactive Brokers**: Have TWS or IB Gateway running on port 7497

3. **Python Dependencies**: Install required packages:
   ```bash
   pip install asyncpg ib-async rich pyyaml
   ```

### Basic Usage

```bash
# Complete data load (stock + options + VIX) - NEW DEFAULT BEHAVIOR
python load_historical_pipeline.py --symbol SPY --days 30 --include-vix

# Load multiple symbols with custom option filtering
python load_historical_pipeline.py --symbols SPY,QQQ,AAPL --days 30 --option-expiry-days 45 --option-strike-range 0.15

# Stock data only (skip option chain discovery)
python load_historical_pipeline.py --symbol TSLA --days 60 --skip-options

# Backfill missing data for last 90 days
python load_historical_pipeline.py --backfill --symbols SPY --days 90

# Complete data with validation and custom logging
python load_historical_pipeline.py --symbol TSLA --days 60 --include-vix --validate --log pipeline.log
```

### New Unified Pipeline Features

The pipeline now automatically loads **stock data AND option chains** by default. Key new features:

- **🔗 Option Chain Discovery**: Automatically fetches and stores available option contracts from IB
- **📊 Multi-Phase Progress**: Real-time progress tracking across stock, options, and VIX data loading
- **📈 VIX Integration**: Optional VIX volatility data collection for correlation analysis
- **⚙️ Flexible Configuration**: Customize option expiry ranges and strike price filters
- **🔄 Backward Compatible**: Use `--skip-options` for legacy stock-only behavior

## Installation

1. **Clone and navigate** to the data collection directory:
   ```bash
   cd backtesting/infra/data_collection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Make script executable** (optional):
   ```bash
   chmod +x load_historical_pipeline.py
   ```

## Command Line Interface

### Symbol Selection

Choose **one** of the following options:

- `--symbol SYMBOL`: Load data for a single symbol
- `--symbols SYMBOL1,SYMBOL2,SYMBOL3`: Load data for multiple symbols (comma-separated)

### Date Range

Choose **one** of the following options:

- `--days N`: Load last N days from today
- `--start YYYY-MM-DD`: Specify start date (optionally with `--end`)
- If no date option is provided, defaults to last 30 days

### Operation Modes

- `--backfill`: Identify and fill missing data gaps instead of full load
- `--complete`: Explicitly enable complete data loading (default behavior)
- `--validate`: Enable comprehensive data validation during loading

### New Data Collection Options

- `--skip-options`: Skip option chain discovery and loading (stock data only)
- `--include-vix`: Include VIX volatility data collection
- `--option-expiry-days N`: Maximum days ahead for option expiries (default: 60)
- `--option-strike-range N`: Strike price range as percent of stock price (default: 0.20)

### Configuration & Output

- `--config FILE`: Use custom configuration file (YAML or JSON)
- `--log FILE`: Write logs to specified file
- `--verbose, -v`: Enable debug logging
- `--quiet, -q`: Show only errors

### Examples

```bash
# Basic examples
python load_historical_pipeline.py --symbol SPY --days 30
python load_historical_pipeline.py --symbols AAPL,MSFT,GOOGL --days 7

# Date range examples
python load_historical_pipeline.py --symbol QQQ --start 2024-01-01 --end 2024-01-31
python load_historical_pipeline.py --symbols SPY,QQQ --start 2024-06-01

# Advanced examples
python load_historical_pipeline.py --symbols SPY,QQQ,IWM --days 60 --validate --verbose
python load_historical_pipeline.py --backfill --symbols TSLA,NVDA --days 90 --log backfill.log
python load_historical_pipeline.py --config custom_config.yaml --symbol META --days 45

# Bulk loading with custom configuration
python load_historical_pipeline.py --config production.yaml --symbols $(cat symbols.txt | tr '\n' ',') --days 30
```

## Configuration

### Configuration File

The pipeline uses `config.yaml` for default settings. Create custom configurations for different environments:

```yaml
# Custom configuration example
database:
  host: prod-db.example.com
  database: trading_production
  user: prod_user
  password: ${DB_PASSWORD}  # Environment variable

loading:
  batch_size: 20
  request_delay: 0.2
  bar_size: "5 mins"

validation:
  enabled: true
  max_spread_percent: 0.05

logging:
  level: DEBUG
  file: "/var/log/trading/pipeline.log"
```

### Environment Variables

Override configuration with environment variables:

```bash
export DB_HOST=localhost
export DB_PASSWORD=secure_password
export IB_HOST=127.0.0.1
export IB_PORT=7497
```

### Key Configuration Sections

#### Database Settings
- Connection parameters for PostgreSQL/TimescaleDB
- Pool size configuration for concurrent operations

#### IB Connection
- Interactive Brokers API connection settings
- Timeout and client ID configuration

#### Loading Parameters
- Request rate limiting and retry settings
- Historical data bar size and data type
- Batch processing configuration

#### Validation Rules
- Data quality thresholds and checks
- Greeks validation and bounds checking
- Price sanity and consistency rules

## Data Validation

The pipeline includes comprehensive data validation:

### Price Validation
- Bid ≤ Ask price relationships
- Positive price values
- Reasonable bid-ask spreads
- Volume and size consistency

### Greeks Validation
- Delta bounds (-1.0 to 1.0)
- Gamma positivity for long options
- Implied volatility ranges
- Theta reasonableness checks

### Time Consistency
- Timestamp validation
- Market hours checking
- Weekend data detection

### Quality Metrics
- Data coverage percentages
- Missing data identification
- Stale data detection
- Gap analysis and reporting

## Progress Tracking

The pipeline provides rich progress visualization:

```
Loading symbols... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3/3 • 0:02:30 • 0:00:00
Processing SPY... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 • 0:01:45
✓ SPY: 4320 stock bars, 12450 option bars
✓ QQQ: 4320 stock bars, 8930 option bars
✓ AAPL: 4320 stock bars, 15230 option bars
```

## Error Handling

### Retry Logic
- Automatic retries for failed API requests
- Exponential backoff for rate limiting
- Graceful handling of temporary network issues

### Error Recovery
- Continue processing other symbols on individual failures
- Detailed error logging and reporting
- Option to stop on critical errors

### Data Integrity
- Database transaction rollback on failures
- Duplicate detection and handling
- Corruption prevention measures

## Integration Examples

### With Existing ArbitrageClass

```python
from modules.Arbitrage.Strategy import ArbitrageClass
from backtesting.infra.data_collection.load_historical_pipeline import HistoricalDataPipeline

class EnhancedArbitrageStrategy(ArbitrageClass):
    async def load_backtesting_data(self, symbols, days=30):
        """Load historical data for backtesting."""
        config = PipelineConfig("config.yaml")
        pipeline = HistoricalDataPipeline(config)

        await pipeline.initialize()

        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        stats = await pipeline.load_historical_data(
            symbols, start_date, end_date, validate=True
        )

        await pipeline.cleanup()
        return stats
```

### Scheduled Data Collection

```python
import schedule
import time

def daily_data_collection():
    """Run daily data collection for key symbols."""
    subprocess.run([
        "python", "load_historical_pipeline.py",
        "--symbols", "SPY,QQQ,IWM,DIA",
        "--days", "3",  # Overlap to ensure no gaps
        "--validate",
        "--log", f"daily_load_{date.today()}.log"
    ])

# Schedule daily at 6 PM ET (after market close)
schedule.every().day.at("18:00").do(daily_data_collection)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Custom Validation Rules

```python
from backtesting.infra.data_collection.validators import ValidationRule, ValidationResult

class CustomPriceRule(ValidationRule):
    def __init__(self):
        super().__init__("custom_price", "Custom price validation", critical=True)

    async def validate(self, snapshot, context=None):
        # Your custom validation logic
        if snapshot.last_price > 1000:  # Example: reject very high prices
            return ValidationResult.INVALID, "Price too high"
        return ValidationResult.VALID, "Price acceptable"

# Add to pipeline configuration
config.defaults["validation"]["custom_rules"] = [CustomPriceRule()]
```

## Performance Tuning

### Database Optimization
- Adjust connection pool sizes based on system resources
- Configure appropriate batch sizes for bulk inserts
- Monitor TimescaleDB compression and partitioning

### IB API Optimization
- Respect rate limiting (60 requests/minute for historical data)
- Use appropriate bar sizes for your needs
- Batch contract requests efficiently

### Memory Management
- Process symbols in batches to control memory usage
- Configure maximum memory limits in config
- Monitor system resources during large loads

### Network Optimization
- Adjust retry delays based on network latency
- Configure appropriate timeouts
- Use persistent connections when possible

## Monitoring and Troubleshooting

### Log Analysis

```bash
# Monitor real-time logs
tail -f pipeline.log

# Search for errors
grep -i error pipeline.log

# Check validation issues
grep -i warning pipeline.log | grep validation
```

### Common Issues

#### Connection Problems
```
Error: Could not connect to Interactive Brokers
```
**Solution**: Ensure TWS/IB Gateway is running and accepting API connections

#### Database Connection Issues
```
Error: Database connection failed
```
**Solution**: Check database container status and connection parameters

#### Rate Limiting
```
Warning: Rate limiting: waiting 45.2s
```
**Solution**: This is normal - the pipeline respects IB API limits

#### Data Quality Issues
```
Warning: SPY: Low data coverage (65.2%)
```
**Solution**: Check market hours and data availability for the requested period

### Performance Monitoring

```python
# Get pipeline statistics
stats = pipeline.get_stats()
print(f"Processing rate: {stats['symbols_processed'] / duration:.2f} symbols/min")

# Monitor database performance
async with db_pool.acquire() as conn:
    query_stats = await conn.fetch("""
        SELECT query, calls, total_time, mean_time
        FROM pg_stat_statements
        ORDER BY total_time DESC
        LIMIT 10
    """)
```

## Best Practices

### Data Loading Strategy
1. **Start Small**: Test with single symbols and short date ranges
2. **Validate First**: Always run with `--validate` in production
3. **Monitor Resources**: Watch CPU, memory, and network usage
4. **Incremental Loading**: Load recent data first, backfill historical data separately
5. **Regular Maintenance**: Run backfill operations periodically

### Configuration Management
1. **Environment-Specific Configs**: Use different configs for dev/prod
2. **Sensitive Data**: Use environment variables for passwords
3. **Version Control**: Track configuration changes
4. **Documentation**: Document custom settings and their purposes

### Error Handling
1. **Log Everything**: Enable comprehensive logging
2. **Monitor Alerts**: Set up alerts for critical errors
3. **Graceful Degradation**: Design for partial failures
4. **Recovery Procedures**: Document manual recovery steps

## Troubleshooting Guide

### Issue: Pipeline hangs during execution
**Symptoms**: Progress bars stop updating, no log output
**Causes**: Network issues, IB API timeouts, database locks
**Solutions**:
- Check IB connection status
- Verify database connectivity
- Review timeout settings in config
- Check for database locks: `SELECT * FROM pg_stat_activity WHERE state = 'active'`

### Issue: High memory usage
**Symptoms**: System becomes slow, out of memory errors
**Causes**: Large batch sizes, too many concurrent operations
**Solutions**:
- Reduce `batch_size` in configuration
- Process fewer symbols at once
- Increase system memory or use smaller date ranges

### Issue: Data validation failures
**Symptoms**: High percentage of invalid data warnings
**Causes**: Market data quality issues, incorrect validation rules
**Solutions**:
- Check market conditions during data period
- Review validation thresholds in config
- Examine specific validation error messages

### Issue: Slow performance
**Symptoms**: Very slow progress, long execution times
**Causes**: Network latency, database performance, IB API throttling
**Solutions**:
- Check network connectivity to IB
- Optimize database indexes and configuration
- Reduce request frequency in config
- Use larger bar sizes for less granular data

## Support and Contributing

### Getting Help
- Review logs with `--verbose` flag for detailed information
- Check database and IB connection status
- Consult Interactive Brokers API documentation
- Review TimescaleDB performance guides

### Reporting Issues
When reporting issues, please include:
- Command line arguments used
- Configuration file (sanitized)
- Error messages and logs
- System specifications
- IB API version and setup

### Contributing
Contributions are welcome! Please ensure:
- Code follows existing style patterns
- Add tests for new functionality
- Update documentation
- Include configuration examples

## File Structure

```
backtesting/infra/data_collection/
├── load_historical_pipeline.py    # Main pipeline script
├── config.yaml                    # Default configuration
├── README.md                      # This documentation
├── historical_loader.py           # Core loading logic
├── collector.py                   # Real-time data collector
├── validators.py                  # Data validation rules
├── vix_collector.py              # VIX data collection
├── config.py                     # Configuration classes
└── examples/                     # Usage examples
    ├── basic_load.py
    ├── batch_load.py
    ├── backfill_example.py
    └── custom_validation.py
```

## License

This pipeline is part of the AlcLite trading system. See the main project license for terms and conditions.

---

**Note**: This pipeline is designed for educational and research purposes. Always test thoroughly in a development environment before using with live trading data. Past performance does not guarantee future results.
