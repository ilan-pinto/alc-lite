-- 01_create_tables.sql
-- Core tables for options arbitrage backtesting database

-- Underlying securities table
CREATE TABLE IF NOT EXISTS underlying_securities (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL UNIQUE,
    name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Option chains table with IB contract mapping
CREATE TABLE IF NOT EXISTS option_chains (
    id SERIAL PRIMARY KEY,
    underlying_id INTEGER NOT NULL REFERENCES underlying_securities(id),
    expiration_date DATE NOT NULL,
    strike_price DECIMAL(10,2) NOT NULL,
    option_type CHAR(1) NOT NULL CHECK (option_type IN ('C', 'P')),
    contract_symbol VARCHAR(21) NOT NULL UNIQUE, -- OCC format
    ib_con_id BIGINT UNIQUE, -- Interactive Brokers contract ID
    multiplier INTEGER DEFAULT 100,
    exchange VARCHAR(10) DEFAULT 'SMART',
    currency VARCHAR(3) DEFAULT 'USD',
    trading_class VARCHAR(10),
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(underlying_id, expiration_date, strike_price, option_type)
);

-- REMOVED: market_data_ticks table - unused (replaced by option_bars_5min)

-- Underlying stock data ticks
CREATE TABLE IF NOT EXISTS stock_data_ticks (
    time TIMESTAMP(6) WITH TIME ZONE NOT NULL,
    underlying_id INTEGER NOT NULL REFERENCES underlying_securities(id),
    price DECIMAL(10,4) NOT NULL,
    bid_price DECIMAL(10,4),
    ask_price DECIMAL(10,4),
    bid_size INTEGER,
    ask_size INTEGER,
    volume BIGINT,
    vwap DECIMAL(10,4), -- Volume weighted average price
    open_price DECIMAL(10,4),
    high_price DECIMAL(10,4),
    low_price DECIMAL(10,4),
    close_price DECIMAL(10,4),
    tick_type VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- REMOVED: arbitrage_opportunities table - unused (replaced by sfr_opportunities)

-- REMOVED: corporate_actions table - unused

-- REMOVED: market_events table - unused

-- REMOVED: data_quality_metrics table - unused (empty, 0 rows)

-- Backtesting runs metadata
CREATE TABLE IF NOT EXISTS backtest_runs (
    id SERIAL PRIMARY KEY,
    run_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    strategy_type VARCHAR(20) NOT NULL,
    start_date TIMESTAMP WITH TIME ZONE NOT NULL,
    end_date TIMESTAMP WITH TIME ZONE NOT NULL,
    symbols TEXT[], -- Array of symbols tested
    parameters JSONB NOT NULL, -- Strategy parameters
    -- Performance metrics
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_profit DECIMAL(12,4),
    max_drawdown DECIMAL(12,4),
    sharpe_ratio DECIMAL(8,4),
    win_rate DECIMAL(5,4),
    -- Execution details
    status VARCHAR(20) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')),
    error_message TEXT,
    execution_time_seconds INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Update triggers for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_underlying_securities_updated_at BEFORE UPDATE ON underlying_securities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_option_chains_updated_at BEFORE UPDATE ON option_chains
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
