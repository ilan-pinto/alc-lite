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

-- High-frequency market data ticks table (will be converted to hypertable)
CREATE TABLE IF NOT EXISTS market_data_ticks (
    time TIMESTAMP(6) WITH TIME ZONE NOT NULL, -- Microsecond precision
    contract_id INTEGER NOT NULL REFERENCES option_chains(id),
    bid_price DECIMAL(10,4),
    ask_price DECIMAL(10,4),
    last_price DECIMAL(10,4),
    bid_size INTEGER,
    ask_size INTEGER,
    last_size INTEGER,
    volume BIGINT,
    open_interest INTEGER,
    -- Greeks
    delta DECIMAL(8,6),
    gamma DECIMAL(8,6),
    theta DECIMAL(8,6),
    vega DECIMAL(8,6),
    rho DECIMAL(8,6),
    implied_volatility DECIMAL(8,6),
    -- Calculated fields
    bid_ask_spread DECIMAL(8,4) GENERATED ALWAYS AS (ask_price - bid_price) STORED,
    mid_price DECIMAL(10,4) GENERATED ALWAYS AS ((bid_price + ask_price) / 2) STORED,
    -- Metadata
    tick_type VARCHAR(20), -- 'DELAYED', 'REALTIME', 'SNAPSHOT'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

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

-- Arbitrage opportunities log
CREATE TABLE IF NOT EXISTS arbitrage_opportunities (
    id BIGSERIAL PRIMARY KEY,
    strategy_type VARCHAR(20) NOT NULL CHECK (strategy_type IN ('SFR', 'SYNTHETIC', 'BOX', 'CALENDAR')),
    underlying_id INTEGER NOT NULL REFERENCES underlying_securities(id),
    timestamp TIMESTAMP(6) WITH TIME ZONE NOT NULL,
    expiration_date DATE NOT NULL,
    -- Option details
    call_strike DECIMAL(10,2),
    put_strike DECIMAL(10,2),
    call_contract_id INTEGER REFERENCES option_chains(id),
    put_contract_id INTEGER REFERENCES option_chains(id),
    -- For calendar/box spreads
    near_expiry DATE,
    far_expiry DATE,
    second_call_contract_id INTEGER REFERENCES option_chains(id),
    second_put_contract_id INTEGER REFERENCES option_chains(id),
    -- Prices at discovery
    stock_price DECIMAL(10,4) NOT NULL,
    call_bid DECIMAL(10,4),
    call_ask DECIMAL(10,4),
    put_bid DECIMAL(10,4),
    put_ask DECIMAL(10,4),
    -- Profitability metrics
    theoretical_profit DECIMAL(10,4),
    max_profit DECIMAL(10,4),
    min_profit DECIMAL(10,4),
    roi_percent DECIMAL(8,4),
    net_credit DECIMAL(10,4),
    cost_basis DECIMAL(10,4),
    -- Risk metrics
    confidence_score DECIMAL(4,3) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    execution_window_ms INTEGER, -- How long opportunity lasted
    slippage_risk VARCHAR(10) CHECK (slippage_risk IN ('LOW', 'MEDIUM', 'HIGH')),
    -- Status
    discovered_by VARCHAR(50), -- 'SCANNER', 'BACKTEST', 'MANUAL'
    execution_attempted BOOLEAN DEFAULT false,
    execution_successful BOOLEAN,
    actual_profit DECIMAL(10,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Corporate actions tracking
CREATE TABLE IF NOT EXISTS corporate_actions (
    id SERIAL PRIMARY KEY,
    underlying_id INTEGER NOT NULL REFERENCES underlying_securities(id),
    action_type VARCHAR(20) NOT NULL CHECK (action_type IN ('DIVIDEND', 'SPLIT', 'MERGER', 'SPINOFF', 'SPECIAL_DIV')),
    ex_date DATE NOT NULL,
    record_date DATE,
    payable_date DATE,
    amount DECIMAL(10,4), -- Dividend amount or split ratio
    adjustment_factor DECIMAL(8,6), -- For contract adjustments
    description TEXT,
    processed BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Market holidays and trading halts
CREATE TABLE IF NOT EXISTS market_events (
    id SERIAL PRIMARY KEY,
    event_date DATE NOT NULL,
    event_type VARCHAR(20) NOT NULL CHECK (event_type IN ('HOLIDAY', 'EARLY_CLOSE', 'HALT', 'CIRCUIT_BREAKER')),
    underlying_id INTEGER REFERENCES underlying_securities(id), -- NULL for market-wide events
    start_time TIME,
    end_time TIME,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Data quality metrics
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    id SERIAL PRIMARY KEY,
    check_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    underlying_id INTEGER REFERENCES underlying_securities(id),
    metric_type VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10,4),
    threshold_value DECIMAL(10,4),
    passed BOOLEAN NOT NULL,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

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
