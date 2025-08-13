-- 09_intraday_bars_tables.sql
-- Tables for 5-minute historical bars collection system
-- Created: 2025-08-12
-- Purpose: Store high-frequency intraday options data for robust backtesting

-- ============================================================================
-- 5-MINUTE OPTION BARS TABLE
-- ============================================================================
-- Stores 5-minute OHLCV bars with Greeks for each option contract
CREATE TABLE IF NOT EXISTS option_bars_5min (
    time TIMESTAMPTZ NOT NULL,
    contract_id INTEGER NOT NULL REFERENCES option_chains(id) ON DELETE CASCADE,

    -- OHLCV Data
    open DECIMAL(10,4),
    high DECIMAL(10,4),
    low DECIMAL(10,4),
    close DECIMAL(10,4),
    volume BIGINT DEFAULT 0,
    bar_count INTEGER, -- Number of trades in this bar
    vwap DECIMAL(10,4), -- Volume-weighted average price

    -- Bid/Ask at close
    bid_close DECIMAL(10,4),
    ask_close DECIMAL(10,4),
    spread_close DECIMAL(10,4) GENERATED ALWAYS AS (ask_close - bid_close) STORED,
    mid_close DECIMAL(10,4) GENERATED ALWAYS AS ((bid_close + ask_close) / 2) STORED,

    -- Greeks at bar close
    implied_volatility DECIMAL(8,6),
    delta DECIMAL(8,6),
    gamma DECIMAL(8,6),
    theta DECIMAL(8,6),
    vega DECIMAL(8,6),
    rho DECIMAL(8,6),

    -- Open Interest
    open_interest INTEGER,

    -- Metadata
    collection_run_id INTEGER,
    data_source VARCHAR(20) DEFAULT 'TRADES', -- 'TRADES', 'BID_ASK', 'MIDPOINT', 'OPTION_IMPLIED_VOLATILITY'
    has_gaps BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    -- Prevent duplicate bars for same contract and time
    CONSTRAINT option_bars_5min_unique UNIQUE(contract_id, time)
);

-- Convert to TimescaleDB hypertable for efficient time-series operations
SELECT create_hypertable('option_bars_5min', 'time', if_not_exists => true);

-- Enable compression for data older than 7 days
ALTER TABLE option_bars_5min SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'contract_id',
    timescaledb.compress_orderby = 'time DESC'
);

-- Add compression policy
SELECT add_compression_policy('option_bars_5min', INTERVAL '7 days', if_not_exists => true);

-- ============================================================================
-- INTRADAY COLLECTION RUNS TABLE
-- ============================================================================
-- Tracks each collection run for monitoring and debugging
CREATE TABLE IF NOT EXISTS intraday_collection_runs (
    id SERIAL PRIMARY KEY,
    run_date DATE NOT NULL,
    run_type VARCHAR(30) NOT NULL, -- 'morning', 'midday', 'afternoon', 'eod', 'late_night', 'gap_fill'
    scheduled_time TIMESTAMPTZ,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'running', -- 'running', 'success', 'partial', 'failed'

    -- Collection scope
    symbols_requested TEXT[],
    duration_requested VARCHAR(20), -- e.g., '1 D', '2 hours'
    bar_size VARCHAR(10) DEFAULT '5 mins',

    -- Results
    contracts_requested INTEGER DEFAULT 0,
    contracts_successful INTEGER DEFAULT 0,
    bars_collected INTEGER DEFAULT 0,
    bars_updated INTEGER DEFAULT 0, -- Bars that already existed and were updated
    bars_skipped INTEGER DEFAULT 0, -- Bars that were identical to existing

    -- Error tracking
    errors INTEGER DEFAULT 0,
    rate_limit_hits INTEGER DEFAULT 0,
    error_details JSONB,

    -- Performance metrics
    duration_seconds INTEGER,
    avg_request_time_ms INTEGER,

    -- Configuration snapshot
    collection_config JSONB, -- Store config used for this run
    ib_client_ids INTEGER[], -- Which client IDs were used

    -- Ensure only one run of each type per day
    CONSTRAINT intraday_runs_unique UNIQUE(run_date, run_type)
);

-- REMOVED: historical_data_gaps table - unused (superseded by data_gaps)

-- REMOVED: collection_statistics table - unused

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Primary query patterns
CREATE INDEX idx_bars_5min_contract_time ON option_bars_5min(contract_id, time DESC);
CREATE INDEX idx_bars_5min_time ON option_bars_5min(time DESC);
CREATE INDEX idx_bars_5min_run ON option_bars_5min(collection_run_id) WHERE collection_run_id IS NOT NULL;

-- Collection monitoring
CREATE INDEX idx_collection_runs_date ON intraday_collection_runs(run_date DESC, run_type);
CREATE INDEX idx_collection_runs_status ON intraday_collection_runs(status) WHERE status != 'success';

-- REMOVED: Indexes for unused tables

-- ============================================================================
-- USEFUL VIEWS
-- ============================================================================

-- View for current day's collection status
CREATE OR REPLACE VIEW v_today_collection_status AS
SELECT
    run_type,
    scheduled_time,
    started_at,
    completed_at,
    status,
    contracts_successful || '/' || contracts_requested as contracts,
    bars_collected,
    errors,
    duration_seconds,
    CASE
        WHEN status = 'success' THEN '✅'
        WHEN status = 'partial' THEN '⚠️'
        WHEN status = 'failed' THEN '❌'
        ELSE '🔄'
    END as status_icon
FROM intraday_collection_runs
WHERE run_date = CURRENT_DATE
ORDER BY scheduled_time;

-- View for data coverage by symbol and date
CREATE OR REPLACE VIEW v_data_coverage AS
SELECT
    DATE(b.time) as trade_date,
    us.symbol,
    COUNT(DISTINCT b.contract_id) as unique_contracts,
    COUNT(*) as total_bars,
    COUNT(*) FILTER (WHERE b.volume > 0) as bars_with_volume,
    COUNT(*) FILTER (WHERE b.implied_volatility IS NOT NULL) as bars_with_iv,
    ROUND(AVG(b.spread_close), 4) as avg_spread,
    MIN(b.time)::TIME as first_bar_time,
    MAX(b.time)::TIME as last_bar_time
FROM option_bars_5min b
JOIN option_chains oc ON b.contract_id = oc.id
JOIN underlying_securities us ON oc.underlying_id = us.id
WHERE b.time > NOW() - INTERVAL '7 days'
GROUP BY DATE(b.time), us.symbol
ORDER BY trade_date DESC, us.symbol;

-- REMOVED: View for unused historical_data_gaps table

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to calculate collection completeness
CREATE OR REPLACE FUNCTION calculate_collection_completeness(
    p_symbol VARCHAR,
    p_date DATE DEFAULT CURRENT_DATE
)
RETURNS TABLE (
    symbol VARCHAR,
    collection_date DATE,
    expected_bars INTEGER,
    actual_bars INTEGER,
    completeness_percent DECIMAL(5,2),
    gaps_detected INTEGER
) AS $$
DECLARE
    v_market_open TIME := '09:30:00';
    v_market_close TIME := '16:00:00';
    v_expected_bars INTEGER;
BEGIN
    -- Calculate expected bars (78 per contract for a full trading day)
    v_expected_bars := 78; -- 6.5 hours * 12 bars per hour

    RETURN QUERY
    SELECT
        us.symbol,
        p_date,
        COUNT(DISTINCT oc.id) * v_expected_bars as expected_bars,
        COUNT(DISTINCT b.contract_id || '-' || b.time) as actual_bars,
        ROUND(
            CASE
                WHEN COUNT(DISTINCT oc.id) = 0 THEN 0
                ELSE (COUNT(DISTINCT b.contract_id || '-' || b.time)::DECIMAL /
                     (COUNT(DISTINCT oc.id) * v_expected_bars)) * 100
            END, 2
        ) as completeness_percent,
        0::INTEGER as gaps_detected -- Placeholder for removed gaps functionality
    FROM underlying_securities us
    LEFT JOIN option_chains oc ON us.id = oc.underlying_id
    LEFT JOIN option_bars_5min b ON oc.id = b.contract_id
        AND DATE(b.time) = p_date
    -- REMOVED: Reference to unused historical_data_gaps table
    WHERE us.symbol = p_symbol
    GROUP BY us.symbol;
END;
$$ LANGUAGE plpgsql;

-- REMOVED: detect_intraday_gaps function - used unused historical_data_gaps table

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_bars_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_option_bars_5min_updated_at
    BEFORE UPDATE ON option_bars_5min
    FOR EACH ROW
    EXECUTE FUNCTION update_bars_updated_at();

-- ============================================================================
-- PERMISSIONS
-- ============================================================================

GRANT ALL ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO trading_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO trading_user;

-- ============================================================================
-- DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE option_bars_5min IS 'High-frequency 5-minute OHLCV bars for option contracts with Greeks';
COMMENT ON TABLE intraday_collection_runs IS 'Tracks each historical data collection run for monitoring';
-- REMOVED: Comments for unused tables

COMMENT ON COLUMN option_bars_5min.vwap IS 'Volume-weighted average price for the 5-minute period';
COMMENT ON COLUMN option_bars_5min.bar_count IS 'Number of individual trades that occurred in this 5-minute period';
COMMENT ON COLUMN option_bars_5min.data_source IS 'IB data type used: TRADES for trade data, BID_ASK for quotes, OPTION_IMPLIED_VOLATILITY for Greeks';

COMMENT ON FUNCTION calculate_collection_completeness IS 'Calculates data collection completeness percentage for a symbol on a given date';
-- REMOVED: Comment for unused function
