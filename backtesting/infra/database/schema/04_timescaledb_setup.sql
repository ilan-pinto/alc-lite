-- 04_timescaledb_setup.sql
-- TimescaleDB-specific optimizations for time-series data

-- Convert tables to hypertables
SELECT create_hypertable('market_data_ticks', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

SELECT create_hypertable('stock_data_ticks', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Convert VIX data table to hypertable
SELECT create_hypertable('vix_data_ticks', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Add additional dimensions for better data distribution
SELECT add_dimension('market_data_ticks', 'contract_id',
    number_partitions => 4,
    if_not_exists => TRUE
);

SELECT add_dimension('stock_data_ticks', 'underlying_id',
    number_partitions => 4,
    if_not_exists => TRUE
);

-- Add dimension for VIX data
SELECT add_dimension('vix_data_ticks', 'instrument_id',
    number_partitions => 2, -- Fewer partitions since we only have 5 VIX instruments
    if_not_exists => TRUE
);

-- Create continuous aggregates for common queries

-- 1-minute aggregates for market data
CREATE MATERIALIZED VIEW market_data_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    contract_id,
    FIRST(bid_price, time) AS open_bid,
    MAX(bid_price) AS high_bid,
    MIN(bid_price) AS low_bid,
    LAST(bid_price, time) AS close_bid,
    FIRST(ask_price, time) AS open_ask,
    MAX(ask_price) AS high_ask,
    MIN(ask_price) AS low_ask,
    LAST(ask_price, time) AS close_ask,
    AVG(bid_ask_spread) AS avg_spread,
    SUM(volume) AS total_volume,
    LAST(open_interest, time) AS open_interest,
    AVG(implied_volatility) AS avg_iv,
    LAST(delta, time) AS delta,
    LAST(gamma, time) AS gamma,
    LAST(theta, time) AS theta,
    LAST(vega, time) AS vega,
    COUNT(*) AS tick_count
FROM market_data_ticks
WHERE bid_price IS NOT NULL AND ask_price IS NOT NULL
GROUP BY bucket, contract_id
WITH NO DATA;

-- 5-minute aggregates for market data
CREATE MATERIALIZED VIEW market_data_5min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', time) AS bucket,
    contract_id,
    FIRST(bid_price, time) AS open_bid,
    MAX(bid_price) AS high_bid,
    MIN(bid_price) AS low_bid,
    LAST(bid_price, time) AS close_bid,
    FIRST(ask_price, time) AS open_ask,
    MAX(ask_price) AS high_ask,
    MIN(ask_price) AS low_ask,
    LAST(ask_price, time) AS close_ask,
    AVG(bid_ask_spread) AS avg_spread,
    MAX(bid_ask_spread) AS max_spread,
    MIN(bid_ask_spread) AS min_spread,
    SUM(volume) AS total_volume,
    LAST(open_interest, time) AS open_interest,
    AVG(implied_volatility) AS avg_iv,
    STDDEV(implied_volatility) AS iv_stddev,
    COUNT(*) AS tick_count
FROM market_data_ticks
WHERE bid_price IS NOT NULL AND ask_price IS NOT NULL
GROUP BY bucket, contract_id
WITH NO DATA;

-- 1-minute aggregates for stock data
CREATE MATERIALIZED VIEW stock_data_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    underlying_id,
    FIRST(price, time) AS open,
    MAX(price) AS high,
    MIN(price) AS low,
    LAST(price, time) AS close,
    SUM(volume) AS volume,
    AVG(vwap) AS avg_vwap,
    COUNT(*) AS tick_count
FROM stock_data_ticks
GROUP BY bucket, underlying_id
WITH NO DATA;

-- 1-minute aggregates for VIX data
CREATE MATERIALIZED VIEW vix_data_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    instrument_id,
    FIRST(last_price, time) AS open_price,
    MAX(last_price) AS high_price,
    MIN(last_price) AS low_price,
    LAST(last_price, time) AS close_price,
    FIRST(bid_price, time) AS open_bid,
    MAX(bid_price) AS high_bid,
    MIN(bid_price) AS low_bid,
    LAST(bid_price, time) AS close_bid,
    FIRST(ask_price, time) AS open_ask,
    MAX(ask_price) AS high_ask,
    MIN(ask_price) AS low_ask,
    LAST(ask_price, time) AS close_ask,
    AVG(bid_ask_spread) AS avg_spread,
    MAX(bid_ask_spread) AS max_spread,
    MIN(bid_ask_spread) AS min_spread,
    SUM(volume) AS total_volume,
    AVG(daily_change_pct) AS avg_daily_change_pct,
    STDDEV(last_price) AS price_volatility,
    COUNT(*) AS tick_count
FROM vix_data_ticks
WHERE last_price IS NOT NULL
GROUP BY bucket, instrument_id
WITH NO DATA;

-- 5-minute aggregates for VIX data
CREATE MATERIALIZED VIEW vix_data_5min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', time) AS bucket,
    instrument_id,
    FIRST(last_price, time) AS open_price,
    MAX(last_price) AS high_price,
    MIN(last_price) AS low_price,
    LAST(last_price, time) AS close_price,
    AVG(last_price) AS avg_price,
    STDDEV(last_price) AS price_stddev,
    AVG(bid_ask_spread) AS avg_spread,
    MAX(bid_ask_spread) AS max_spread,
    SUM(volume) AS total_volume,
    AVG(daily_change_pct) AS avg_daily_change_pct,
    MAX(daily_change_pct) AS max_daily_change_pct,
    MIN(daily_change_pct) AS min_daily_change_pct,
    COUNT(*) AS tick_count
FROM vix_data_ticks
WHERE last_price IS NOT NULL
GROUP BY bucket, instrument_id
WITH NO DATA;

-- Create refresh policies for continuous aggregates
SELECT add_continuous_aggregate_policy('market_data_1min',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

SELECT add_continuous_aggregate_policy('market_data_5min',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE
);

SELECT add_continuous_aggregate_policy('stock_data_1min',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

-- Add refresh policies for VIX continuous aggregates
SELECT add_continuous_aggregate_policy('vix_data_1min',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

SELECT add_continuous_aggregate_policy('vix_data_5min',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE
);

-- Compression policies for older data
ALTER TABLE market_data_ticks SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_segmentby = 'contract_id'
);

ALTER TABLE stock_data_ticks SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_segmentby = 'underlying_id'
);

-- Enable compression for VIX data
ALTER TABLE vix_data_ticks SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_segmentby = 'instrument_id'
);

-- Add compression policy (compress chunks older than 7 days)
SELECT add_compression_policy('market_data_ticks',
    INTERVAL '7 days',
    if_not_exists => TRUE
);

SELECT add_compression_policy('stock_data_ticks',
    INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Add compression policy for VIX data
SELECT add_compression_policy('vix_data_ticks',
    INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Retention policies (optional - uncomment to enable)
-- Drop data older than 2 years
/*
SELECT add_retention_policy('market_data_ticks',
    INTERVAL '2 years',
    if_not_exists => TRUE
);

SELECT add_retention_policy('stock_data_ticks',
    INTERVAL '2 years',
    if_not_exists => TRUE
);
*/

-- Create specialized indexes for TimescaleDB queries
CREATE INDEX IF NOT EXISTS idx_market_data_contract_bucket
    ON market_data_ticks (contract_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_stock_data_underlying_bucket
    ON stock_data_ticks (underlying_id, time DESC);

-- Create specialized indexes for VIX data queries
CREATE INDEX IF NOT EXISTS idx_vix_data_instrument_bucket
    ON vix_data_ticks (instrument_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_vix_data_price_time
    ON vix_data_ticks (last_price, time DESC) WHERE tick_type = 'REALTIME';

-- Function to get real-time spread statistics
CREATE OR REPLACE FUNCTION get_spread_stats(
    p_contract_id INTEGER,
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ
)
RETURNS TABLE (
    avg_spread DECIMAL,
    min_spread DECIMAL,
    max_spread DECIMAL,
    spread_volatility DECIMAL,
    wide_spread_pct DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        AVG(bid_ask_spread)::DECIMAL AS avg_spread,
        MIN(bid_ask_spread)::DECIMAL AS min_spread,
        MAX(bid_ask_spread)::DECIMAL AS max_spread,
        STDDEV(bid_ask_spread)::DECIMAL AS spread_volatility,
        (COUNT(*) FILTER (WHERE bid_ask_spread > 2 * AVG(bid_ask_spread) OVER()) * 100.0 / COUNT(*))::DECIMAL AS wide_spread_pct
    FROM market_data_ticks
    WHERE contract_id = p_contract_id
        AND time >= p_start_time
        AND time <= p_end_time
        AND bid_price IS NOT NULL
        AND ask_price IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

-- View for monitoring hypertable statistics
CREATE OR REPLACE VIEW hypertable_stats AS
SELECT
    hypertable_name,
    hypertable_size(format('%I.%I', hypertable_schema, hypertable_name)) AS total_size,
    (SELECT count(*) FROM show_chunks(format('%I.%I', hypertable_schema, hypertable_name))) AS chunk_count,
    (SELECT count(*) FROM show_chunks(format('%I.%I', hypertable_schema, hypertable_name))) AS compressed_chunks,
    CASE hypertable_name
        WHEN 'market_data_ticks' THEN 'Options Market Data'
        WHEN 'stock_data_ticks' THEN 'Stock Price Data'
        WHEN 'vix_data_ticks' THEN 'VIX Volatility Data'
        ELSE 'Other'
    END AS data_type
FROM timescaledb_information.hypertables
WHERE hypertable_schema = 'public';

-- Enable TimescaleDB telemetry (optional)
-- ALTER SYSTEM SET timescaledb.telemetry_level = 'off';
