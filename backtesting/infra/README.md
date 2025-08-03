# Backtesting Infrastructure

This directory contains the infrastructure components for the options arbitrage backtesting system, including database setup and data collection utilities.

## Architecture Overview

```
backtesting/infra/
├── database/           # Database container and schema
│   ├── podman/        # Container configuration
│   └── schema/        # SQL schema files
└── data_collection/   # Python data collection modules
```

## Database Setup

### Prerequisites

- Podman or Docker installed
- At least 16GB RAM and 500GB SSD storage
- Network access for container setup

### Quick Start

1. **Start the database container:**
   ```bash
   cd backtesting/infra/database/podman
   podman-compose up -d
   ```

2. **Verify the setup:**
   ```bash
   podman logs alc_timescaledb
   ```

3. **Connect to the database:**
   ```bash
   psql -h localhost -U trading_user -d options_arbitrage
   ```

### Database Schema

The database uses PostgreSQL 15 with TimescaleDB extension for optimal time-series performance:

- **underlying_securities**: Stock symbols and metadata
- **option_chains**: Option contracts with IB contract IDs
- **market_data_ticks**: High-frequency options data (hypertable)
- **stock_data_ticks**: Stock price data (hypertable)
- **arbitrage_opportunities**: Discovered arbitrage opportunities
- **corporate_actions**: Dividends, splits, etc.

### Key Features

- **Microsecond precision** timestamps
- **Automatic partitioning** by time
- **Data compression** for older records
- **Continuous aggregates** for performance
- **Data validation** and quality metrics

## Data Collection

### Real-time Collection

```python
from backtesting.infra.data_collection import OptionsDataCollector
import asyncpg
from ib_async import IB

# Setup database connection
db_pool = await asyncpg.create_pool(
    host="localhost",
    database="options_arbitrage",
    user="trading_user",
    password="secure_trading_password"
)

# Setup IB connection
ib = IB()
await ib.connectAsync('127.0.0.1', 7497, clientId=1)

# Start data collection
collector = OptionsDataCollector(db_pool, ib)
await collector.start_collection(['SPY', 'QQQ', 'AAPL'])
```

### Historical Data Loading

```python
from backtesting.infra.data_collection import HistoricalDataLoader
from datetime import date

loader = HistoricalDataLoader(db_pool, ib)

# Load 30 days of history for SPY
stats = await loader.load_symbol_history(
    'SPY',
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31)
)
```

### Data Validation

```python
from backtesting.infra.data_collection import DataValidator

validator = DataValidator(db_pool)

# Run quality check
report = await validator.run_quality_check(hours_back=24)
print(f"Data coverage: {report['price_data_coverage']:.1f}%")
```

## Configuration

### Environment Variables

```bash
# Database settings
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=options_arbitrage
export DB_USER=trading_user
export DB_PASSWORD=secure_trading_password

# Collection settings
export COLLECTION_INTERVAL_MS=100
export MAX_SPREAD_PERCENT=0.10
export EXPIRY_RANGE_DAYS=60
```

### Collection Parameters

- **Tick interval**: 100ms (configurable)
- **Batch size**: 1000 records
- **Strike range**: ±20% around ATM
- **Expiry range**: 60 days ahead
- **Greeks collection**: Enabled by default

## Integration with Existing Code

The data collection system integrates seamlessly with your existing `ArbitrageClass`:

```python
from modules.Arbitrage.Strategy import ArbitrageClass
from backtesting.infra.data_collection import OptionsDataCollector

class EnhancedArbitrageClass(ArbitrageClass):
    def __init__(self, collect_data=True):
        super().__init__()

        if collect_data:
            # Initialize data collection
            self.data_collector = OptionsDataCollector(
                self.db_pool, self.ib
            )

    async def log_arbitrage_opportunity(self, strategy_type, data):
        # Log opportunities to database
        await self._persist_opportunity(strategy_type, data)
```

## Monitoring

### Database Health

```sql
-- Check hypertable statistics
SELECT * FROM hypertable_stats;

-- Monitor data quality
SELECT * FROM data_quality_metrics
WHERE check_timestamp > NOW() - INTERVAL '1 hour';

-- View partition information
SELECT * FROM partition_info ORDER BY tablename;
```

### Performance Metrics

```python
# Get collector statistics
stats = collector.get_stats()
print(f"Ticks collected: {stats['ticks_collected']}")
print(f"Active contracts: {stats['active_contracts']}")
```

## Maintenance

### Automated Tasks

The system includes automated maintenance:

- **Partition creation**: New partitions created monthly
- **Data compression**: Chunks older than 7 days compressed
- **Quality monitoring**: Continuous validation

### Manual Operations

```bash
# Create additional partitions
SELECT create_partition_range('market_data_ticks', '2024-01-01', '2024-12-31');

# Compress old data
SELECT add_compression_policy('market_data_ticks', INTERVAL '3 days');

# Clean up old partitions (careful!)
SELECT drop_old_partitions('market_data_ticks', 24);  -- Keep 24 months
```

## Troubleshooting

### Common Issues

1. **Container won't start**
   - Check available disk space
   - Verify port 5432 is not in use
   - Check container logs: `podman logs alc_timescaledb`

2. **Data collection errors**
   - Verify IB connection is active
   - Check database connectivity
   - Review collector logs for specific errors

3. **Performance issues**
   - Monitor database CPU/memory usage
   - Check for slow queries in pg_stat_statements
   - Verify compression policies are working

### Logs and Debugging

```bash
# Container logs
podman logs -f alc_timescaledb

# PostgreSQL query logs
tail -f /path/to/postgres/log/postgresql-*.log

# Python logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Development

### Adding New Validation Rules

```python
from backtesting.infra.data_collection.validators import ValidationRule

class CustomRule(ValidationRule):
    def __init__(self):
        super().__init__("custom_rule", "Custom validation", critical=False)

    async def validate(self, snapshot, context=None):
        # Your validation logic here
        return ValidationResult.VALID, "OK"

# Add to validator
validator.rules.append(CustomRule())
```

### Schema Changes

1. Create new migration file in `database/schema/`
2. Apply manually or add to init script
3. Update Python models if needed
4. Test with sample data

## Performance Expectations

- **Ingestion rate**: 10,000+ ticks/second
- **Query performance**: <100ms for typical backtesting queries
- **Storage efficiency**: ~70% compression ratio for older data
- **Uptime**: 99.9% with proper monitoring

## Links

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [TimescaleDB Guides](https://docs.timescale.com/)
- [Interactive Brokers API](https://interactivebrokers.github.io/tws-api/)
- [GitHub Issue #2](https://github.com/ilan-pinto/alc-lite/issues/2)
