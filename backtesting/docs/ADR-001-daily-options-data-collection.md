# ADR-001: Daily Options Data Collection System

**Status**: Proposed
**Date**: August 10, 2025
**Authors**: AlcLite Trading System Team
**Decision**: Implement automated daily options data collection pipeline

## Table of Contents
1. [Context](#context)
2. [Problem Statement](#problem-statement)
3. [Decision](#decision)
4. [Architecture Overview](#architecture-overview)
5. [Data Collection Schedule](#data-collection-schedule)
6. [Data Specifications](#data-specifications)
7. [Storage Requirements](#storage-requirements)
8. [Infrastructure Proposal](#infrastructure-proposal)
9. [Implementation Plan](#implementation-plan)
10. [Phase 1: Local Laptop Implementation](#phase-1-local-laptop-implementation)
11. [Consequences](#consequences)
12. [Success Metrics](#success-metrics)

## Context

The AlcLite trading system requires comprehensive historical options data for backtesting arbitrage strategies (SFR, Synthetic, Box Spreads). Our analysis revealed critical limitations with Interactive Brokers API:

- **Option data availability**: Only current and future expiries are accessible
- **Data retention**: Expired options disappear from IB API 30-60 days post-expiration
- **Current infrastructure**: Manual pipeline exists but lacks automation for daily collection
- **Database**: TimescaleDB deployed with comprehensive schema but missing lifecycle tracking

## Problem Statement

### Core Challenges
1. **Data Ephemerality**: Options data becomes permanently inaccessible 30-60 days after expiration
2. **Manual Process**: Current pipeline requires manual execution, risking data loss
3. **No Lifecycle Tracking**: System doesn't track when new options are listed or expire
4. **Incomplete Coverage**: Risk of missing critical data points for backtesting accuracy
5. **Scalability**: Manual process doesn't scale to multiple symbols and expiries

### Requirements
- Automated daily collection with market-aware scheduling
- Comprehensive option chain discovery and lifecycle tracking
- Production-grade resilience and error recovery
- Storage optimization for multi-year historical data
- Integration with existing backtesting infrastructure

## Decision

Implement a production-ready automated daily options data collection system with the following components:

1. **DailyDataCollectionService**: Core orchestration service with market-aware scheduling
2. **OptionChainDiscovery**: Real-time discovery of new option contracts
3. **MarketScheduler**: Intelligent scheduling based on market calendar
4. **DataRetentionManager**: Lifecycle management and archival strategies
5. **Enhanced Database Schema**: Additional tables for lifecycle and quality tracking

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Orchestration Layer                      │
├───────────────────┬─────────────────┬───────────────────────┤
│  Market Scheduler │  Alert Manager  │  Health Monitor       │
└───────────────────┴─────────────────┴───────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                 Daily Collection Service                     │
├─────────────┬───────────────┬───────────────┬──────────────┤
│   Circuit   │   Option      │   Data        │   Quality    │
│   Breaker   │   Discovery   │   Loader      │   Validator  │
└─────────────┴───────────────┴───────────────┴──────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Data Sources                            │
├──────────────────────┬────────────────────────────────────┬─┤
│   IB Gateway API     │    Market Calendar API    │  VIX    │
└──────────────────────┴────────────────────────────────────┴─┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   TimescaleDB Storage                        │
├────────────────┬──────────────────┬─────────────────────────┤
│  Hypertables   │  Continuous Aggs │  Compression Policies   │
└────────────────┴──────────────────┴─────────────────────────┘
```

### Key Design Patterns

- **Circuit Breaker Pattern**: Prevent cascading failures with IB API
- **Dead Letter Queue**: Handle and retry failed data collection attempts
- **Bulkhead Pattern**: Isolate failures between different data types
- **Observer Pattern**: Real-time monitoring and alerting
- **Repository Pattern**: Abstract database operations

## Data Collection Schedule

### Daily Collections

| Time (ET) | Type | Purpose | Priority |
|-----------|------|---------|----------|
| 6:30 AM | Pre-Market | Overnight changes, new listings, corporate actions | Medium |
| 4:45 PM | Primary EOD | Final prices, volume, OI, Greeks, IV | Critical |

### Weekly Collections

| Day | Time (ET) | Type | Purpose |
|-----|-----------|------|---------|
| Thursday | 5:00 PM | Discovery | New weekly options for following week |
| Friday | 3:00 PM | Pre-Expiry | Emergency collection before expiry |
| Friday | 3:59 PM | Settlement | Final settlement prices |

### Monthly Collections

| Event | Timing | Purpose |
|-------|--------|---------|
| First Trading Day | 6:30 AM | Monthly contract discovery, strike range updates |
| Third Friday | Standard + Extended | Monthly expiry processing |
| Post-Expiry | Next Trading Day | Verification and archival |

### Market Holiday Adaptations

- **Early Close Days**: Adjust primary collection to 15 minutes post-close
- **Market Holidays**: Skip collection, run gap analysis next trading day
- **Weekend Processing**: Saturday 8:00 AM review, Sunday 6:00 PM preparation

## Data Specifications

### Option Contract Data

```yaml
contract_specifications:
  - contract_symbol: String (21 chars)
  - underlying_symbol: String (10 chars)
  - strike_price: Decimal(10,2)
  - option_type: Char(1) [C|P]
  - expiration_date: Date
  - ib_con_id: BigInt
  - multiplier: Integer (default: 100)
  - exchange: String (default: SMART)
  - trading_class: String

market_data_per_tick:
  - timestamp: TimestampTZ
  - bid_price: Decimal(10,4)
  - ask_price: Decimal(10,4)
  - bid_size: Integer
  - ask_size: Integer
  - last_price: Decimal(10,4)
  - volume: Integer
  - open_interest: Integer

greeks:
  - delta: Decimal(6,4)
  - gamma: Decimal(8,6)
  - theta: Decimal(8,4)
  - vega: Decimal(8,4)
  - rho: Decimal(8,4)
  - implied_volatility: Decimal(6,4)
```

### Collection Volume Estimates

| Symbol Type | Daily Contracts | Data Points/Contract | Total Daily Fields |
|-------------|-----------------|---------------------|-------------------|
| Major ETF (SPY) | 2,000-4,000 | 20 | 40,000-80,000 |
| Large Cap Stock | 500-1,000 | 20 | 10,000-20,000 |
| Small Cap Stock | 100-300 | 20 | 2,000-6,000 |

**10 Major Symbols**: ~400,000-800,000 fields/day
**Annual Volume**: ~150-300 million data points

### Dynamic Strike Selection

```yaml
strike_range_rules:
  low_iv: # IV < 20%
    range: 15%
    min_strikes: 11
  normal_iv: # 20% <= IV <= 40%
    range: 25%
    min_strikes: 21
  high_iv: # IV > 40%
    range: 35%
    min_strikes: 31

volume_filters:
  etf_minimum_oi: 100
  stock_minimum_oi: 50
  minimum_daily_volume: 10
  unusual_activity_multiplier: 5x
```

## Storage Requirements

### Data Volume Projections

| Timeframe | Raw Data | Compressed | With Indexes | Total |
|-----------|----------|------------|--------------|-------|
| Daily | 100 MB | 20 MB | 5 MB | 25 MB |
| Monthly | 3 GB | 600 MB | 150 MB | 750 MB |
| Yearly | 36 GB | 7.2 GB | 1.8 GB | 9 GB |
| 5 Years | 180 GB | 36 GB | 9 GB | 45 GB |

### TimescaleDB Optimization

```sql
-- Hypertable configuration
SELECT create_hypertable('market_data_ticks', 'time',
  chunk_time_interval => INTERVAL '1 day');

-- Compression policy (data > 7 days)
ALTER TABLE market_data_ticks SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'contract_id',
  timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('market_data_ticks',
  INTERVAL '7 days');

-- Continuous aggregates for common queries
CREATE MATERIALIZED VIEW hourly_option_summary
WITH (timescaledb.continuous) AS
SELECT
  contract_id,
  time_bucket('1 hour', time) as hour,
  AVG(bid_price) as avg_bid,
  AVG(ask_price) as avg_ask,
  SUM(volume) as total_volume,
  LAST(open_interest, time) as closing_oi
FROM market_data_ticks
GROUP BY contract_id, hour;

-- Retention policy (archive after 2 years)
SELECT add_retention_policy('market_data_ticks',
  INTERVAL '2 years');
```

### Storage Tiers

| Tier | Age | Storage Type | Compression | Access Pattern |
|------|-----|--------------|-------------|----------------|
| Hot | < 7 days | SSD, Uncompressed | None | Real-time queries |
| Warm | 7-90 days | SSD, Compressed | 5:1 | Backtesting |
| Cold | 90-365 days | HDD, Compressed | 10:1 | Historical analysis |
| Archive | > 1 year | Object Storage | 20:1 | Compliance/Research |

## Infrastructure Proposal

### Production Environment

```yaml
compute_infrastructure:
  application_servers:
    type: AWS EC2 / GCP Compute Engine
    instance_type: c5.2xlarge (8 vCPU, 16 GB RAM)
    count: 2 (Active/Standby)
    auto_scaling: Yes (2-4 instances)

  database_servers:
    type: AWS RDS for PostgreSQL with TimescaleDB
    instance_type: db.m5.2xlarge (8 vCPU, 32 GB RAM)
    storage: 1TB SSD (gp3)
    iops: 16,000
    multi_az: Yes
    read_replicas: 1

  cache_layer:
    type: AWS ElastiCache / GCP Memorystore
    instance_type: cache.m5.large
    memory: 8 GB

networking:
  vpc:
    cidr: 10.0.0.0/16
    availability_zones: 2

  subnets:
    public: 10.0.1.0/24, 10.0.2.0/24
    private: 10.0.10.0/24, 10.0.20.0/24
    database: 10.0.100.0/24, 10.0.200.0/24

  security_groups:
    app_sg:
      - port: 443 (HTTPS)
      - port: 8080 (Health checks)
    db_sg:
      - port: 5432 (PostgreSQL)
      - source: app_sg only
    ib_gateway_sg:
      - port: 7497 (IB API)
      - port: 4001 (IB Gateway)

container_orchestration:
  platform: Kubernetes (EKS/GKE)
  nodes: 3 (across AZs)

  deployments:
    daily_collector:
      replicas: 2
      cpu_request: 2
      memory_request: 4Gi
      cpu_limit: 4
      memory_limit: 8Gi

    market_scheduler:
      replicas: 1
      cpu_request: 0.5
      memory_request: 1Gi

    health_monitor:
      replicas: 1
      cpu_request: 0.5
      memory_request: 1Gi

storage:
  primary_database:
    type: SSD
    size: 1TB
    iops: 16,000
    backup: Daily snapshots, 30-day retention

  archive_storage:
    type: S3 / GCS
    lifecycle_policy:
      - Standard: 0-30 days
      - Infrequent Access: 30-365 days
      - Glacier: > 365 days

monitoring_stack:
  metrics: Prometheus + Grafana
  logging: ELK Stack (Elasticsearch, Logstash, Kibana)
  tracing: Jaeger
  alerting: PagerDuty / Opsgenie

third_party_services:
  ib_gateway:
    deployment: Docker container
    high_availability: 2 instances (primary/backup)
    connection_pool: 5-10 connections

  market_calendar_api:
    provider: TradingCalendar API
    cache_ttl: 24 hours
```

### Development Environment

```yaml
development:
  local_development:
    docker_compose: Yes
    services:
      - TimescaleDB (local)
      - IB Gateway (paper trading)
      - Redis (cache)

  staging_environment:
    type: Scaled-down production
    compute: t3.large (2 vCPU, 8 GB RAM)
    database: db.t3.medium
    data_subset: Last 90 days only
```

### Disaster Recovery

```yaml
backup_strategy:
  database:
    full_backup: Daily at 2:00 AM ET
    incremental: Every 6 hours
    retention: 30 days
    offsite_replication: Cross-region

  application_state:
    config_backup: Git repository
    secrets_backup: AWS Secrets Manager

recovery_objectives:
  rto: 4 hours (Recovery Time Objective)
  rpo: 1 hour (Recovery Point Objective)

failover_procedures:
  automatic: Database failover to read replica
  manual: Application switchover to standby region
```

### Cost Estimates (Monthly)

| Component | Phase 1 Local | Development | Staging | Production |
|-----------|--------------|-------------|---------|------------|
| Compute | $0 | $50 | $150 | $600 |
| Database | $0 | $100 | $200 | $800 |
| Storage | $0 | $20 | $50 | $200 |
| Network | $0 | $10 | $30 | $100 |
| Monitoring | $0 | $0 | $50 | $200 |
| **Total** | **$0** | **$180** | **$480** | **$1,900** |

## Implementation Plan

### Phase 1: Core Infrastructure (Weeks 1-2)
- Set up development environment with Docker Compose
- Deploy TimescaleDB with enhanced schema
- Implement DailyDataCollectionService wrapper
- Create structured logging framework
- Basic health check endpoints

### Phase 2: Smart Scheduling (Weeks 2-3)
- Integrate market calendar API
- Implement MarketScheduler with timezone awareness
- Add option expiry lifecycle management
- Create priority-based collection queues
- Test holiday and early close handling

### Phase 3: Production Resilience (Weeks 3-4)
- Implement circuit breaker pattern
- Add dead letter queue with retry logic
- Create comprehensive error categorization
- Optimize async/await patterns
- Add connection pooling with health checks

### Phase 4: Monitoring & Quality (Weeks 4-5)
- Deploy Prometheus and Grafana
- Implement structured logging with correlation IDs
- Create data quality validation framework
- Add alerting for critical failures
- Build operational dashboard

### Phase 5: Deployment (Weeks 5-6)
- Create Docker containers
- Deploy to staging environment
- Performance testing and optimization
- Production deployment with gradual rollout
- Documentation and runbooks

## Phase 1: Local Laptop Implementation

### Overview

Before scaling to production infrastructure, Phase 1 implements a proof-of-concept running entirely on a local laptop using existing resources. This approach validates the system with minimal cost and complexity.

**Scope**:
- **Symbols**: SPY, PLTR, TSLA (3 symbols only)
- **Infrastructure**: Local laptop with Podman containers and macOS scheduling
- **Location**: Israel (UTC+2/+3)
- **Cost**: $0 (uses existing local resources)
- **Timeline**: 1-2 weeks implementation

### Israel Timezone Considerations

#### US Market Hours in Israel Time

| Market Event | ET Time | Israel Winter (UTC+2) | Israel Summer (UTC+3) |
|--------------|---------|----------------------|----------------------|
| Pre-Market | 6:30 AM | 1:30 PM | 2:30 PM |
| Market Open | 9:30 AM | 4:30 PM | 5:30 PM |
| Market Close | 4:00 PM | 11:00 PM | 12:00 AM (midnight) |
| Primary Collection | 4:45 PM | 11:45 PM | 12:45 AM |
| Friday Expiry Check | 3:00 PM | 10:00 PM | 11:00 PM |

#### Adjusted Collection Schedule

```yaml
# scheduler/daily_schedule_israel.yaml
timezone: Asia/Jerusalem  # Handles DST automatically

symbols:
  - SPY
  - PLTR
  - TSLA

schedule:
  pre_market:
    time_et: "06:30"
    time_local: "13:30"  # 1:30 PM IST (winter)
    time_local_dst: "14:30"  # 2:30 PM IDT (summer)
    enabled: false  # Start with EOD only

  end_of_day:
    time_et: "16:45"
    time_local: "23:45"  # 11:45 PM IST (winter)
    time_local_dst: "00:45"  # 12:45 AM IDT (summer)
    enabled: true

  friday_expiry_check:
    time_et: "15:00"
    time_local: "22:00"  # 10:00 PM IST (winter)
    time_local_dst: "23:00"  # 11:00 PM IDT (summer)
    enabled: true
    day_of_week: 5  # Friday

  morning_check:
    time_local: "08:00"  # 8:00 AM Israel time
    enabled: true
    purpose: "Check overnight US market data gaps"
```

### Local Infrastructure Setup

#### Existing Resources (Already Running)
- **TimescaleDB**: Podman container on port 5433
- **IB Gateway/TWS**: Paper trading on port 7497
- **Python Environment**: Existing venv with dependencies

#### New Components
- **Scheduler**: macOS launchd (built-in, no installation needed)
- **Logging**: Local file system at `logs/`
- **Monitoring**: Simple Python script + manual checks

#### Storage Requirements (Local)
- **SPY**: ~50 MB/day (2000-4000 contracts)
- **PLTR**: ~10 MB/day (300-500 contracts)
- **TSLA**: ~15 MB/day (500-800 contracts)
- **Total**: ~75 MB/day, ~2.25 GB/month
- **Available Space**: Need minimum 10 GB free

### Implementation Components

#### 1. Database Schema Updates

```sql
-- backtesting/infra/database/schema/08_daily_collection_tables.sql

-- Collection status tracking with timezone support
CREATE TABLE daily_collection_status (
    id SERIAL PRIMARY KEY,
    collection_date DATE NOT NULL,
    collection_time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    collection_type VARCHAR(20),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    status VARCHAR(20),
    records_collected INTEGER,
    error_message TEXT,
    timezone VARCHAR(50) DEFAULT 'Asia/Jerusalem',
    UNIQUE(collection_date, symbol, collection_type)
);

-- Option lifecycle tracking
CREATE TABLE option_contract_lifecycle (
    contract_id INTEGER REFERENCES option_chains(id),
    first_seen DATE NOT NULL,
    last_seen DATE,
    expiry_collected BOOLEAN DEFAULT FALSE,
    status VARCHAR(20)
);

CREATE INDEX idx_collection_status_date ON daily_collection_status(collection_date);
CREATE INDEX idx_collection_status_symbol ON daily_collection_status(symbol);
CREATE INDEX idx_lifecycle_status ON option_contract_lifecycle(status);
```

#### 2. Main Collection Script

```python
# backtesting/infra/data_collection/daily_collector.py
"""
Daily options data collector for Phase 1 local implementation.
Simplified version focused on 3 symbols with Israel timezone support.
"""

import asyncio
import pytz
from datetime import datetime, date
from pathlib import Path
import yaml
import asyncpg
from ib_async import IB

class DailyCollector:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.israel_tz = pytz.timezone('Asia/Jerusalem')
        self.et_tz = pytz.timezone('US/Eastern')

    async def run(self):
        # Connect to TimescaleDB
        db_pool = await asyncpg.create_pool(
            host='localhost',
            port=5433,
            database='options_arbitrage',
            user='trading_user',
            password='secure_trading_password'
        )

        # Connect to IB
        ib = IB()
        await ib.connectAsync('127.0.0.1', 7497, clientId=999)

        # Collect data for each symbol
        for symbol in self.config['symbols']:
            await self.collect_symbol_data(symbol, db_pool, ib)

        # Cleanup
        ib.disconnect()
        await db_pool.close()
```

#### 3. macOS LaunchAgent Configuration

```xml
<!-- ~/Library/LaunchAgents/com.alclite.daily-collector-israel.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.alclite.daily-collector-israel</string>

    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/Users/ilpinto/dev/AlchimistProject/alc-lite/backtesting/infra/data_collection/daily_collector.py</string>
        <string>--config</string>
        <string>scheduler/daily_schedule_israel.yaml</string>
    </array>

    <key>StartCalendarInterval</key>
    <array>
        <!-- 11:45 PM IST daily -->
        <dict>
            <key>Hour</key>
            <integer>23</integer>
            <key>Minute</key>
            <integer>45</integer>
        </dict>
        <!-- 10:00 PM IST Friday for expiry -->
        <dict>
            <key>Weekday</key>
            <integer>5</integer>
            <key>Hour</key>
            <integer>22</integer>
            <key>Minute</key>
            <integer>0</integer>
        </dict>
    </array>

    <key>StandardOutPath</key>
    <string>/Users/ilpinto/dev/AlchimistProject/alc-lite/logs/daily_collector.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/ilpinto/dev/AlchimistProject/alc-lite/logs/daily_collector_error.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>TZ</key>
        <string>Asia/Jerusalem</string>
    </dict>
</dict>
</plist>
```

#### 4. Manual Run Script

```bash
#!/bin/bash
# run_collection_now_israel.sh
export TZ="Asia/Jerusalem"
cd /Users/ilpinto/dev/AlchimistProject/alc-lite
source venv/bin/activate

echo "Current Israel Time: $(date)"
echo "Current ET Time: $(TZ='America/New_York' date)"

python backtesting/infra/data_collection/daily_collector.py \
    --symbols SPY,PLTR,TSLA \
    --timezone Asia/Jerusalem \
    --force \
    --verbose
```

### Setup Instructions

```bash
# 1. Install timezone support
pip install pytz

# 2. Create logs directory
mkdir -p logs

# 3. Run database migrations
PGPASSWORD=secure_trading_password psql -h localhost -p 5433 \
    -U trading_user -d options_arbitrage \
    -f backtesting/infra/database/schema/08_daily_collection_tables.sql

# 4. Install the LaunchAgent
launchctl load ~/Library/LaunchAgents/com.alclite.daily-collector-israel.plist

# 5. Test manual collection
chmod +x run_collection_now_israel.sh
./run_collection_now_israel.sh

# 6. Verify scheduling
launchctl list | grep alclite
```

### Monitoring and Operations

#### Daily Operations
```bash
# Check collection status
python check_collection_status.py

# View logs with timestamps
tail -f logs/daily_collector.log

# Manual trigger
launchctl start com.alclite.daily-collector-israel

# Stop scheduled collection
launchctl unload ~/Library/LaunchAgents/com.alclite.daily-collector-israel.plist
```

#### Collection Status Query
```sql
-- Check today's collection status
SELECT
    symbol,
    collection_type,
    status,
    records_collected,
    completed_at AT TIME ZONE 'Asia/Jerusalem' as local_time
FROM daily_collection_status
WHERE collection_date = CURRENT_DATE
ORDER BY completed_at DESC;

-- Check for missing data
SELECT
    s.symbol,
    COUNT(DISTINCT oc.expiration_date) as expiry_count,
    COUNT(oc.id) as contract_count
FROM (VALUES ('SPY'), ('PLTR'), ('TSLA')) s(symbol)
LEFT JOIN underlying_securities us ON s.symbol = us.symbol
LEFT JOIN option_chains oc ON us.id = oc.underlying_id
    AND oc.expiration_date >= CURRENT_DATE
GROUP BY s.symbol;
```

### Phase 1 Success Criteria

- ✅ Daily collection runs automatically at 11:45 PM IST / 12:45 AM IDT
- ✅ Friday expiry check runs at 10:00 PM IST / 11:00 PM IDT
- ✅ SPY, PLTR, TSLA option chains collected successfully
- ✅ Data stored in TimescaleDB with proper timestamps
- ✅ Collection status tracked in database
- ✅ Manual monitoring shows data completeness >95%
- ✅ System runs for 2 weeks without manual intervention
- ✅ Handles Israel DST transition correctly

### Advantages of Israel Timezone

1. **Late Evening Collection**: 11:45 PM is convenient for monitoring
2. **Friday Timing**: 10:00 PM Friday works well before weekend
3. **Sunday Workday**: Can address any weekend issues on Sunday
4. **Morning Gap Check**: 8:00 AM catches overnight issues before work

### Next Steps After Phase 1

1. **Expand Symbols** (Week 3-4)
   - Add 5-7 more symbols
   - Test scalability limits

2. **Add Monitoring** (Week 4-5)
   - Email/Slack alerts for failures
   - Data quality dashboard
   - Automated gap detection

3. **Move to Cloud** (Week 6+)
   - Migrate to AWS/GCP
   - Implement high availability
   - Add production monitoring

## Consequences

### Positive

1. **Data Preservation**: Capture option data before 30-60 day expiration window
2. **Automation**: Eliminate manual intervention and human error
3. **Scalability**: Handle multiple symbols and growing data volumes
4. **Reliability**: Production-grade resilience with automatic recovery
5. **Backtesting Quality**: Comprehensive historical data for strategy validation
6. **Cost Efficiency**: Optimized storage with compression and tiering

### Negative

1. **Infrastructure Cost**: ~$2,000/month for production environment
2. **Complexity**: More components to maintain and monitor
3. **Learning Curve**: Team needs to understand new architecture
4. **IB API Limits**: Still constrained by rate limits and data availability
5. **Initial Development**: 6-week implementation timeline

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| IB API Changes | High | Abstract API layer, version pinning |
| Data Loss | Critical | Multiple collection attempts, backups |
| Cost Overrun | Medium | Auto-scaling limits, storage policies |
| Performance Issues | Medium | Caching, query optimization |
| Security Breach | High | Encryption, access controls, audit logs |

## Success Metrics

### Technical Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Data Coverage | >95% | Daily contracts collected / total available |
| Collection Success Rate | >95% | Successful runs / total scheduled |
| Recovery Time | <1 hour | Time to recover from failure |
| Query Performance | <100ms | P95 latency for common queries |
| Storage Efficiency | 10:1 | Compression ratio for cold data |

### Business Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Backtest Data Quality | >99% | Valid data points / total |
| Strategy Coverage | 100% | Strategies with sufficient data |
| Cost per Data Point | <$0.001 | Monthly cost / data points |
| Time to Insight | <5 min | Data available for analysis |

### Operational Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| System Uptime | 99.9% | Available time / total time |
| Alert Response Time | <15 min | Time to acknowledge critical alerts |
| Deployment Frequency | Weekly | Successful deployments / week |
| MTTR | <2 hours | Mean time to recovery |

## Approval

| Role | Name | Date | Decision |
|------|------|------|----------|
| Technical Lead | | | |
| Product Owner | | | |
| DevOps Lead | | | |
| Finance | | | |

## References

- [TimescaleDB Documentation](https://docs.timescale.com/)
- [Interactive Brokers API Guide](https://interactivebrokers.github.io/tws-api/)
- [AWS Architecture Best Practices](https://aws.amazon.com/architecture/well-architected/)
- [Kubernetes Production Patterns](https://kubernetes.io/docs/concepts/workloads/)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)

---

*This ADR is a living document and will be updated as the implementation progresses and new insights are gained.*
