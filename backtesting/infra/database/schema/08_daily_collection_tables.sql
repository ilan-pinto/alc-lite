-- 08_daily_collection_tables.sql
-- Tables for Phase 1 Daily Options Data Collection System
-- Created: 2025-08-10

-- Collection status tracking with timezone support
CREATE TABLE IF NOT EXISTS daily_collection_status (
    id SERIAL PRIMARY KEY,
    collection_date DATE NOT NULL,
    collection_time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    collection_type VARCHAR(20) NOT NULL, -- 'pre_market', 'eod', 'expiry', 'morning_check'
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    status VARCHAR(20) NOT NULL DEFAULT 'pending', -- 'pending', 'running', 'success', 'failed', 'partial'
    records_collected INTEGER DEFAULT 0,
    contracts_discovered INTEGER DEFAULT 0,
    contracts_updated INTEGER DEFAULT 0,
    error_message TEXT,
    error_count INTEGER DEFAULT 0,
    timezone VARCHAR(50) DEFAULT 'Asia/Jerusalem',
    execution_time_ms INTEGER, -- Time taken in milliseconds
    data_quality_score DECIMAL(3,2), -- 0.00 to 1.00
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(collection_date, symbol, collection_type)
);

-- REMOVED: option_contract_lifecycle table - unused

-- REMOVED: daily_collection_metrics table - unused

-- REMOVED: collection_queue table - unused

-- REMOVED: data_gaps table - unused (empty, 0 rows)

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_collection_status_date ON daily_collection_status(collection_date);
CREATE INDEX IF NOT EXISTS idx_collection_status_symbol ON daily_collection_status(symbol);
CREATE INDEX IF NOT EXISTS idx_collection_status_type ON daily_collection_status(collection_type);
CREATE INDEX IF NOT EXISTS idx_collection_status_composite ON daily_collection_status(collection_date, symbol, status);

-- REMOVED: Indexes for unused tables

-- REMOVED: Indexes for unused data_gaps table

-- REMOVED: Triggers for unused tables

-- Useful views for monitoring
CREATE OR REPLACE VIEW v_daily_collection_summary AS
SELECT
    collection_date,
    symbol,
    COUNT(*) as collection_runs,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_runs,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_runs,
    SUM(records_collected) as total_records,
    SUM(contracts_discovered) as total_contracts_discovered,
    AVG(execution_time_ms) as avg_execution_ms,
    AVG(data_quality_score) as avg_quality_score
FROM daily_collection_status
GROUP BY collection_date, symbol
ORDER BY collection_date DESC, symbol;

-- REMOVED: View for unused option_contract_lifecycle table

-- Function to check collection health
CREATE OR REPLACE FUNCTION check_collection_health(p_symbol VARCHAR DEFAULT NULL)
RETURNS TABLE (
    symbol VARCHAR,
    last_successful_collection TIMESTAMPTZ,
    hours_since_collection NUMERIC,
    today_success_rate NUMERIC,
    week_success_rate NUMERIC,
    active_contracts INTEGER,
    data_gaps_count INTEGER DEFAULT 0, -- Placeholder for removed functionality
    health_status VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COALESCE(p_symbol, us.symbol) as symbol,
        MAX(CASE WHEN dcs.status = 'success' THEN dcs.completed_at END) as last_successful_collection,
        EXTRACT(EPOCH FROM (NOW() - MAX(CASE WHEN dcs.status = 'success' THEN dcs.completed_at END)))/3600 as hours_since_collection,
        AVG(CASE WHEN dcs.collection_date = CURRENT_DATE AND dcs.status = 'success' THEN 1.0 ELSE 0.0 END) * 100 as today_success_rate,
        AVG(CASE WHEN dcs.collection_date >= CURRENT_DATE - INTERVAL '7 days' AND dcs.status = 'success' THEN 1.0 ELSE 0.0 END) * 100 as week_success_rate,
        COUNT(DISTINCT oc.id) FILTER (WHERE oc.active = true) as active_contracts,
        0 as data_gaps_count, -- Placeholder: data_gaps table removed
        CASE
            WHEN MAX(CASE WHEN dcs.status = 'success' THEN dcs.completed_at END) IS NULL THEN 'NO_DATA'
            WHEN EXTRACT(EPOCH FROM (NOW() - MAX(CASE WHEN dcs.status = 'success' THEN dcs.completed_at END)))/3600 > 48 THEN 'CRITICAL'
            WHEN EXTRACT(EPOCH FROM (NOW() - MAX(CASE WHEN dcs.status = 'success' THEN dcs.completed_at END)))/3600 > 24 THEN 'WARNING'
            WHEN AVG(CASE WHEN dcs.collection_date >= CURRENT_DATE - INTERVAL '7 days' AND dcs.status = 'success' THEN 1.0 ELSE 0.0 END) < 0.8 THEN 'DEGRADED'
            ELSE 'HEALTHY'
        END as health_status
    FROM underlying_securities us
    LEFT JOIN daily_collection_status dcs ON us.symbol = dcs.symbol
    LEFT JOIN option_chains oc ON us.id = oc.underlying_id
    -- REMOVED: JOIN to unused data_gaps table
    WHERE (p_symbol IS NULL OR us.symbol = p_symbol)
      AND us.symbol IN ('SPY', 'PLTR', 'TSLA')  -- Phase 1 symbols
    GROUP BY us.symbol;
END;
$$ LANGUAGE plpgsql;

-- Grant appropriate permissions
GRANT ALL ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO trading_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO trading_user;

-- Add comments for documentation
COMMENT ON TABLE daily_collection_status IS 'Tracks each daily collection run with status and metrics';
-- REMOVED: Comment for unused data_gaps table
COMMENT ON FUNCTION check_collection_health IS 'Returns health status of daily collection system for monitoring';
