# PostgreSQL MCP Integration Guide

## Overview

This guide explains how to use the PostgreSQL MCP (Model Context Protocol) integration with Claude Code for natural language querying of your options arbitrage backtesting database.

## Setup Status

✅ **Completed Setup:**
- PostgreSQL MCP server (crystaldba/postgres-mcp) installed via Podman
- Claude Code configuration updated with database connection
- Connection established to options_arbitrage database on localhost:5433
- Read-only access mode enabled for data security

## MCP Server Configuration

The PostgreSQL MCP server is configured with:
```bash
Server Name: postgresql-trading
Command: podman run --rm -i --network host crystaldba/postgres-mcp --access-mode restricted
Database: postgresql://trading_user:secure_trading_password@localhost:5433/options_arbitrage
Access Mode: Restricted (read-only)
```

## Available Database Tables

Your options arbitrage database contains these key tables:

### Core Trading Tables
- `underlying_securities` - Stock symbols and metadata
- `option_chains` - Option contract definitions with strikes and expiries
- `stock_data_ticks` - Historical stock price data
- `market_data_ticks` - Option pricing and Greeks data

### Backtesting Tables
- `sfr_opportunities` - SFR (Synthetic Free Risk) arbitrage opportunities
- `sfr_backtest_runs` - Backtest execution records
- `sfr_simulated_trades` - Individual trade simulations
- `sfr_performance_analytics` - Performance metrics and statistics

### VIX Correlation Tables
- `vix_data_ticks` - VIX price and volatility data
- `vix_arbitrage_correlation` - VIX-arbitrage correlations
- `vix_sfr_correlation_analysis` - SFR strategy VIX correlations

## Natural Language Query Examples

With the PostgreSQL MCP integration, you can now ask Claude Code:

### Basic Data Exploration
- "Show me the schema of the option_chains table"
- "How many SPY options are in the database?"
- "What date range does our stock data cover?"

### SFR Analysis Queries
- "Find all SFR opportunities for SPY with profits above 0.5%"
- "Show me the most profitable SFR trades from the last backtest run"
- "What's the average profit margin for SFR opportunities by expiry date?"

### Option Chain Analysis
- "Show me all SPY option strikes between 575-580 expiring in August 2025"
- "Which option expiries have the most available contracts?"
- "Find puts and calls with matching strikes for conversion opportunities"

### Performance Analysis
- "What's the success rate of SFR strategies by underlying symbol?"
- "Show me the risk metrics for the latest backtest runs"
- "Compare SFR performance across different volatility periods"

### VIX Correlation Analysis
- "How does SFR opportunity count correlate with VIX levels?"
- "Show periods of high VIX when SFR profits were elevated"
- "Find the strongest VIX-arbitrage correlations in our data"

## Usage Tips

### Query Structure
1. **Be specific about tables**: Reference exact table names from the schema above
2. **Use date ranges**: Many queries benefit from time constraints
3. **Reference IDs**: Use underlying_id=1 for SPY, etc.

### Example Query Session
```
You: "What option strikes do we have for SPY?"

Claude: [Queries option_chains table where underlying_id = 1]
"I found SPY options with strikes ranging from 575.00 to 580.00,
across expiration dates from 2025-08-11 to 2025-08-13."

You: "Show me potential SFR opportunities from those strikes"

Claude: [Joins option_chains with sfr_opportunities table]
"I found 3 SFR opportunities matching call strikes > put strikes..."
```

## Security Features

- **Read-only access**: MCP server runs in restricted mode, preventing data modification
- **Local connection**: Database access is limited to localhost network
- **Credential isolation**: Database credentials are contained within MCP configuration

## Troubleshooting

### Connection Issues
```bash
# Check MCP server status
claude mcp list

# Should show: postgresql-trading - ✓ Connected
```

### Database Access Issues
```bash
# Test database connection directly
podman run --rm --network host -it crystaldba/postgres-mcp --help
```

### Query Issues
- Verify table names match the schema exactly
- Check that underlying_id references exist (1=SPY, etc.)
- Use proper date formats: '2025-08-11'::date

## Advanced Usage

### Complex Multi-Table Queries
Ask Claude Code to join multiple tables for comprehensive analysis:
- "Compare SFR profitability across different VIX volatility quartiles"
- "Find option chains with the highest volume that generated successful SFR trades"

### Data Quality Checks
- "Check for missing data gaps in our stock price history"
- "Validate that all option contracts have corresponding market data"

### Performance Monitoring
- "Show database table sizes and row counts"
- "Find the most frequently queried option expiries"

This PostgreSQL MCP integration transforms Claude Code into a powerful database analysis tool, enabling natural language exploration of your complex options trading and backtesting data.
