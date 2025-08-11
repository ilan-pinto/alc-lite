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

-- Option contract lifecycle tracking
CREATE TABLE IF NOT EXISTS option_contract_lifecycle (
    id SERIAL PRIMARY KEY,
    contract_id INTEGER NOT NULL REFERENCES option_chains(id) ON DELETE CASCADE,
    first_seen DATE NOT NULL,
    last_seen DATE,
    last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    expiry_collected BOOLEAN DEFAULT FALSE,
    expiry_collected_at TIMESTAMPTZ,
    days_tracked INTEGER GENERATED ALWAYS AS (
        CASE
            WHEN last_seen IS NOT NULL THEN (last_seen - first_seen)
            ELSE (CURRENT_DATE - first_seen)
        END
    ) STORED,
    status VARCHAR(20) NOT NULL DEFAULT 'active', -- 'active', 'expired', 'delisted', 'archived'
    collection_count INTEGER DEFAULT 1, -- Number of times data was collected
    last_volume INTEGER,
    last_open_interest INTEGER,
    peak_volume INTEGER,
    peak_open_interest INTEGER,
    average_spread DECIMAL(10,4),
    data_gaps INTEGER DEFAULT 0, -- Number of missing collection days
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(contract_id)
);

-- Daily collection metrics for monitoring
CREATE TABLE IF NOT EXISTS daily_collection_metrics (
    id SERIAL PRIMARY KEY,
    collection_date DATE NOT NULL,
    metric_type VARCHAR(50) NOT NULL, -- 'coverage', 'quality', 'performance', 'errors'
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20,4),
    metric_unit VARCHAR(20), -- 'percent', 'count', 'ms', 'bytes'
    symbol VARCHAR(10),
    details JSONB, -- Flexible field for additional metrics
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(collection_date, metric_type, metric_name, symbol)
);

-- Collection queue for retry logic
CREATE TABLE IF NOT EXISTS collection_queue (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    collection_type VARCHAR(20) NOT NULL,
    scheduled_time TIMESTAMPTZ NOT NULL,
    priority INTEGER DEFAULT 5, -- 1 (highest) to 10 (lowest)
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed', 'cancelled'
    last_attempt_at TIMESTAMPTZ,
    next_retry_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Data gaps tracking for backfill
CREATE TABLE IF NOT EXISTS data_gaps (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    gap_date DATE NOT NULL,
    gap_type VARCHAR(20) NOT NULL, -- 'missing', 'incomplete', 'quality_issue'
    data_type VARCHAR(20), -- 'stock', 'options', 'both'
    detected_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    backfilled BOOLEAN DEFAULT FALSE,
    backfilled_at TIMESTAMPTZ,
    backfill_attempts INTEGER DEFAULT 0,
    notes TEXT,
    UNIQUE(symbol, gap_date, gap_type)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_collection_status_date ON daily_collection_status(collection_date);
CREATE INDEX IF NOT EXISTS idx_collection_status_symbol ON daily_collection_status(symbol);
CREATE INDEX IF NOT EXISTS idx_collection_status_type ON daily_collection_status(collection_type);
CREATE INDEX IF NOT EXISTS idx_collection_status_composite ON daily_collection_status(collection_date, symbol, status);

CREATE INDEX IF NOT EXISTS idx_lifecycle_status ON option_contract_lifecycle(status);
CREATE INDEX IF NOT EXISTS idx_lifecycle_contract ON option_contract_lifecycle(contract_id);
CREATE INDEX IF NOT EXISTS idx_lifecycle_dates ON option_contract_lifecycle(first_seen, last_seen);

CREATE INDEX IF NOT EXISTS idx_metrics_date ON daily_collection_metrics(collection_date);
CREATE INDEX IF NOT EXISTS idx_metrics_type ON daily_collection_metrics(metric_type);

CREATE INDEX IF NOT EXISTS idx_queue_status ON collection_queue(status, scheduled_time);
CREATE INDEX IF NOT EXISTS idx_queue_symbol ON collection_queue(symbol);

CREATE INDEX IF NOT EXISTS idx_gaps_symbol_date ON data_gaps(symbol, gap_date);
CREATE INDEX IF NOT EXISTS idx_gaps_backfilled ON data_gaps(backfilled);

-- Create update trigger for lifecycle table
CREATE OR REPLACE FUNCTION update_lifecycle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_option_contract_lifecycle_updated_at
    BEFORE UPDATE ON option_contract_lifecycle
    FOR EACH ROW
    EXECUTE FUNCTION update_lifecycle_updated_at();

-- Create trigger for collection queue
CREATE TRIGGER update_collection_queue_updated_at
    BEFORE UPDATE ON collection_queue
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

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

CREATE OR REPLACE VIEW v_active_option_contracts AS
SELECT
    oc.contract_symbol,
    oc.strike_price,
    oc.option_type,
    oc.expiration_date,
    us.symbol,
    ocl.first_seen,
    ocl.last_seen,
    ocl.days_tracked,
    ocl.collection_count,
    ocl.status,
    ocl.peak_volume,
    ocl.peak_open_interest
FROM option_contract_lifecycle ocl
JOIN option_chains oc ON ocl.contract_id = oc.id
JOIN underlying_securities us ON oc.underlying_id = us.id
WHERE ocl.status = 'active'
ORDER BY us.symbol, oc.expiration_date, oc.strike_price;

-- Function to check collection health
CREATE OR REPLACE FUNCTION check_collection_health(p_symbol VARCHAR DEFAULT NULL)
RETURNS TABLE (
    symbol VARCHAR,
    last_successful_collection TIMESTAMPTZ,
    hours_since_collection NUMERIC,
    today_success_rate NUMERIC,
    week_success_rate NUMERIC,
    active_contracts INTEGER,
    data_gaps_count INTEGER,
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
        COUNT(DISTINCT ocl.contract_id) FILTER (WHERE ocl.status = 'active') as active_contracts,
        COUNT(DISTINCT dg.id) FILTER (WHERE dg.backfilled = false) as data_gaps_count,
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
    LEFT JOIN option_contract_lifecycle ocl ON oc.id = ocl.contract_id
    LEFT JOIN data_gaps dg ON us.symbol = dg.symbol
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
COMMENT ON TABLE option_contract_lifecycle IS 'Tracks the lifecycle of each option contract from discovery to expiry';
COMMENT ON TABLE daily_collection_metrics IS 'Stores aggregated metrics for monitoring collection health';
COMMENT ON TABLE collection_queue IS 'Queue for managing collection tasks and retries';
COMMENT ON TABLE data_gaps IS 'Tracks missing data for backfill operations';
COMMENT ON FUNCTION check_collection_health IS 'Returns health status of daily collection system for monitoring';
