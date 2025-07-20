# Comprehensive Arbitrage Testing Guide

This guide shows how to test the complete arbitrage detection system when markets are closed using realistic mock data and scenarios.

## 🚀 Quick Start

### Run Complete Offline Testing
```bash
# Interactive mode with menu
python test_arbitrage_offline.py

# Test specific scenario with debug output
python test_arbitrage_offline.py --scenario dell_profitable --debug

# Run all scenarios
python test_arbitrage_offline.py --all

# Manual step-by-step analysis
python test_arbitrage_offline.py --manual
```

## 📋 Available Test Scenarios

### 1. **dell_profitable** - Guaranteed Arbitrage Opportunity
- **Setup**: DELL at $131.24 with C132/P131 combination
- **Expected**: Profitable conversion arbitrage detected
- **Net Credit**: $0.80, Min Profit: $0.56, ROI: 0.42%

### 2. **dell_no_arbitrage** - Normal Market Conditions
- **Setup**: DELL with realistic pricing, no arbitrage opportunities
- **Expected**: No profitable opportunities found
- **Result**: All combinations rejected due to negative profit

### 3. **dell_wide_spreads** - Wide Bid-Ask Spreads
- **Setup**: DELL with excessively wide option spreads (>$20)
- **Expected**: Rejected due to BID_ASK_SPREAD_TOO_WIDE
- **Purpose**: Test spread filtering logic

### 4. **dell_low_volume** - Low Volume Options
- **Setup**: DELL with very low volume options (<5 contracts)
- **Expected**: Accepted but logged as debug warnings
- **Purpose**: Test volume filtering improvements

### 5. **spy_multiple_expiries** - Multiple Expiry Testing
- **Setup**: SPY with 3 different expiries and various opportunities
- **Expected**: Proper expiry prioritization and selection
- **Purpose**: Test expiry handling logic

## 🧪 Testing Framework Components

### 1. MockIB Class (`tests/mock_ib.py`)
- **Full IB API simulation** including `qualifyContractsAsync`, `reqMktData`, `pendingTickersEvent`
- **Realistic market data generation** with proper bid/ask spreads and volumes
- **Event-driven architecture** that mimics real IB behavior
- **Contract qualification** and market data delivery simulation

### 2. Market Data Generator (`tests/market_scenarios.py`)
- **Realistic option pricing** based on Black-Scholes approximation
- **Volume modeling** based on moneyness (ATM options have higher volume)
- **Bid-ask spread calculation** that reflects real market conditions
- **Known arbitrage opportunities** with precise profit calculations

### 3. Integration Tests (`tests/test_arbitrage_integration.py`)
- **End-to-end workflow testing** from market data to execution
- **Strike position logic verification** with multiple stock scenarios
- **Metrics collection validation** for rejection reasons and opportunities
- **Mock-based testing** that doesn't require live market connection

## 📊 Test Results Examples

### Profitable Arbitrage (DELL C132/P131)
```
📈 Stock: DELL @ $131.24
   Bid: $131.17, Ask: $131.31, Volume: 500,000

📅 Expiry: 20250221
   🔥 C132.0/P131.0 (diff:1): ✅ ARBITRAGE
      Call: $2.20 (bid), Put: $1.40 (ask)
      Net Credit: $0.80, Spread: $0.24
      Min Profit: $0.56, Max Profit: $1.80
      Min ROI: 0.42%

🎯 Total arbitrage opportunities found: 1
```

### No Arbitrage Scenario
```
📈 Stock: DELL @ $131.24

📅 Expiry: 20250221
   🔥 C130/P129 (diff:1): ❌ No arbitrage
   🔥 C131/P130 (diff:1): ❌ No arbitrage
   🔥 C132.0/P131 (diff:1): ❌ No arbitrage

🎯 Total arbitrage opportunities found: 0
💡 Tips for finding arbitrage:
   - Look for net credit > spread
   - Try 1-strike difference combinations
   - Check bid-ask spreads aren't too wide
```

## 🔧 Advanced Testing

### Custom Scenarios
```python
# Create custom market scenario
from tests.market_scenarios import MarketDataGenerator

market_data = {}
stock_ticker = MarketDataGenerator.generate_stock_data("AAPL", 150.0)
market_data[stock_ticker.contract.conId] = stock_ticker

# Add custom option data
call_155 = MarketDataGenerator.generate_option_data("AAPL", "20250221", 155.0, "C", 150.0, 30)
call_155.bid = 2.50  # Adjust for testing
market_data[call_155.contract.conId] = call_155
```

### Integration with Real SFR
```python
from modules.Arbitrage.SFR import SFR
from tests.mock_ib import MockIB
from tests.market_scenarios import MarketScenarios

# Create SFR with mock data
mock_ib = MockIB()
mock_ib.test_market_data = MarketScenarios.dell_profitable_conversion()

sfr = SFR(debug=True)
sfr.ib = mock_ib

# Run actual arbitrage scan with mock data
await sfr.scan_sfr("DELL", quantity=1)
```

## 🎯 Key Improvements Tested

### 1. Adaptive Strike Position Logic
- **Position-based selection** instead of fixed dollar amounts
- **Works with any strike spacing** ($1, $2.50, $5, $10)
- **1-strike difference prioritization** for higher trade probability

### 2. Volume Optimization
- **Tiered volume filtering** (High >50, Medium >10, Low ≥1)
- **Debug-level logging** for low volume contracts
- **Strike prioritization** based on proximity to stock price

### 3. Proper Contract Cleanup
- **Market data cancellation** when executors are deactivated
- **Contract lifecycle management** prevents stale data interference
- **Enhanced cleanup methods** for memory efficiency

### 4. Conversion Arbitrage Focus
- **Enforces call_strike > put_strike** for proper conversion structure
- **Net credit optimization** for maximum profit generation
- **Realistic profit calculations** including transaction costs

## 📈 Performance Testing

The testing framework can handle:
- **Large option chains** (100+ contracts per symbol)
- **Multiple expiries** (3-6 expiries simultaneously)
- **High-frequency testing** (rapid scenario switching)
- **Memory efficiency** (proper cleanup between tests)

## 🔍 Debugging Tools

### Enable Debug Output
```bash
python test_arbitrage_offline.py --scenario dell_profitable --debug
```

### Manual Analysis Mode
```bash
python test_arbitrage_offline.py --manual
```

### Integration Test Debugging
```python
# Run specific integration test
python -c "
from tests.test_arbitrage_integration import run_integration_test
run_integration_test('manual_analysis')
"
```

This comprehensive testing framework enables complete validation of the arbitrage detection system during market-closed periods, ensuring all components work correctly before live trading.
