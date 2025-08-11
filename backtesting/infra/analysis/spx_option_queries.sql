-- SPX Option Data Queries
-- Database: options_arbitrage
-- Description: Comprehensive SQL queries for fetching and analyzing SPX option data

-- ============================================================================
-- 1. BASIC SPX SETUP
-- ============================================================================

-- Check if SPX exists as an underlying security
SELECT * FROM underlying_securities WHERE symbol = 'SPX';

-- Insert SPX if it doesn't exist (run once)
INSERT INTO underlying_securities (symbol, name, sector, active)
VALUES ('SPX', 'S&P 500 Index', 'INDEX', true)
ON CONFLICT (symbol) DO NOTHING;

-- Get SPX underlying_id for use in other queries
-- Replace {SPX_ID} with this value in queries below
SELECT id FROM underlying_securities WHERE symbol = 'SPX';

-- ============================================================================
-- 2. SPX OPTION CONTRACTS
-- ============================================================================

-- Find all SPX option contracts
SELECT
    oc.id,
    oc.expiration_date,
    oc.strike_price,
    oc.option_type,
    oc.contract_symbol,
    oc.ib_con_id,
    oc.active
FROM option_chains oc
JOIN underlying_securities us ON oc.underlying_id = us.id
WHERE us.symbol = 'SPX'
ORDER BY oc.expiration_date, oc.strike_price, oc.option_type;

-- Find SPX options expiring in next 30 days
SELECT
    oc.expiration_date,
    oc.strike_price,
    oc.option_type,
    oc.contract_symbol,
    EXTRACT(days FROM oc.expiration_date - CURRENT_DATE) as days_to_expiry
FROM option_chains oc
JOIN underlying_securities us ON oc.underlying_id = us.id
WHERE us.symbol = 'SPX'
    AND oc.active = true
    AND oc.expiration_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '30 days'
ORDER BY oc.expiration_date, oc.strike_price, oc.option_type;

-- Find near-the-money SPX options (within 5% of current price)
WITH current_spx AS (
    SELECT
        us.id as underlying_id,
        sdt.price as current_price
    FROM stock_data_ticks sdt
    JOIN underlying_securities us ON sdt.underlying_id = us.id
    WHERE us.symbol = 'SPX'
    ORDER BY sdt.time DESC
    LIMIT 1
)
SELECT
    oc.expiration_date,
    oc.strike_price,
    oc.option_type,
    oc.contract_symbol,
    cs.current_price,
    ROUND((oc.strike_price / cs.current_price - 1) * 100, 2) as moneyness_pct
FROM option_chains oc
CROSS JOIN current_spx cs
WHERE oc.underlying_id = cs.underlying_id
    AND oc.active = true
    AND oc.strike_price BETWEEN cs.current_price * 0.95 AND cs.current_price * 1.05
    AND oc.expiration_date > CURRENT_DATE
ORDER BY oc.expiration_date, ABS(oc.strike_price - cs.current_price);

-- ============================================================================
-- 3. MARKET DATA QUERIES
-- ============================================================================

-- Latest market data for all SPX options
SELECT
    oc.contract_symbol,
    oc.expiration_date,
    oc.strike_price,
    oc.option_type,
    md.time,
    md.bid_price,
    md.ask_price,
    md.last_price,
    md.bid_ask_spread,
    md.mid_price,
    md.volume,
    md.open_interest,
    md.implied_volatility,
    md.delta,
    md.gamma,
    md.theta,
    md.vega
FROM market_data_ticks md
JOIN option_chains oc ON md.contract_id = oc.id
JOIN underlying_securities us ON oc.underlying_id = us.id
WHERE us.symbol = 'SPX'
    AND md.time > NOW() - INTERVAL '1 hour'
ORDER BY md.time DESC, oc.expiration_date, oc.strike_price, oc.option_type;

-- Get latest bid/ask for specific SPX option
SELECT
    oc.contract_symbol,
    md.time,
    md.bid_price,
    md.ask_price,
    md.bid_size,
    md.ask_size,
    md.bid_ask_spread,
    md.mid_price
FROM market_data_ticks md
JOIN option_chains oc ON md.contract_id = oc.id
JOIN underlying_securities us ON oc.underlying_id = us.id
WHERE us.symbol = 'SPX'
    AND oc.expiration_date = '2024-02-16'  -- Change date as needed
    AND oc.strike_price = 5000              -- Change strike as needed
    AND oc.option_type = 'C'                -- 'C' for call, 'P' for put
ORDER BY md.time DESC
LIMIT 10;

-- Historical option prices (1-minute bars for last day)
SELECT
    time_bucket('1 minute', md.time) as minute,
    oc.contract_symbol,
    oc.strike_price,
    oc.option_type,
    AVG(md.mid_price) as avg_mid_price,
    MAX(md.bid_price) as max_bid,
    MIN(md.ask_price) as min_ask,
    SUM(md.volume) as total_volume,
    LAST(md.open_interest, md.time) as last_oi,
    AVG(md.implied_volatility) as avg_iv
FROM market_data_ticks md
JOIN option_chains oc ON md.contract_id = oc.id
JOIN underlying_securities us ON oc.underlying_id = us.id
WHERE us.symbol = 'SPX'
    AND md.time > NOW() - INTERVAL '1 day'
    AND oc.expiration_date = '2024-02-16'  -- Change as needed
    AND oc.strike_price = 5000              -- Change as needed
    AND oc.option_type = 'C'
GROUP BY minute, oc.contract_symbol, oc.strike_price, oc.option_type
ORDER BY minute DESC;

-- ============================================================================
-- 4. OPTION CHAIN VIEWS
-- ============================================================================

-- Complete T-shaped option chain for nearest expiration
WITH nearest_expiry AS (
    SELECT MIN(expiration_date) as exp_date
    FROM option_chains oc
    JOIN underlying_securities us ON oc.underlying_id = us.id
    WHERE us.symbol = 'SPX'
        AND expiration_date > CURRENT_DATE
),
latest_data AS (
    SELECT DISTINCT ON (contract_id)
        contract_id,
        bid_price,
        ask_price,
        mid_price,
        volume,
        open_interest,
        implied_volatility,
        delta,
        gamma,
        theta,
        vega
    FROM market_data_ticks
    WHERE time > NOW() - INTERVAL '1 hour'
    ORDER BY contract_id, time DESC
)
SELECT
    oc.strike_price,
    -- Calls
    MAX(CASE WHEN oc.option_type = 'C' THEN ld.bid_price END) as call_bid,
    MAX(CASE WHEN oc.option_type = 'C' THEN ld.ask_price END) as call_ask,
    MAX(CASE WHEN oc.option_type = 'C' THEN ld.mid_price END) as call_mid,
    MAX(CASE WHEN oc.option_type = 'C' THEN ld.volume END) as call_volume,
    MAX(CASE WHEN oc.option_type = 'C' THEN ld.open_interest END) as call_oi,
    MAX(CASE WHEN oc.option_type = 'C' THEN ld.implied_volatility END) as call_iv,
    MAX(CASE WHEN oc.option_type = 'C' THEN ld.delta END) as call_delta,
    -- Puts
    MAX(CASE WHEN oc.option_type = 'P' THEN ld.bid_price END) as put_bid,
    MAX(CASE WHEN oc.option_type = 'P' THEN ld.ask_price END) as put_ask,
    MAX(CASE WHEN oc.option_type = 'P' THEN ld.mid_price END) as put_mid,
    MAX(CASE WHEN oc.option_type = 'P' THEN ld.volume END) as put_volume,
    MAX(CASE WHEN oc.option_type = 'P' THEN ld.open_interest END) as put_oi,
    MAX(CASE WHEN oc.option_type = 'P' THEN ld.implied_volatility END) as put_iv,
    MAX(CASE WHEN oc.option_type = 'P' THEN ld.delta END) as put_delta
FROM option_chains oc
JOIN underlying_securities us ON oc.underlying_id = us.id
CROSS JOIN nearest_expiry ne
LEFT JOIN latest_data ld ON oc.id = ld.contract_id
WHERE us.symbol = 'SPX'
    AND oc.expiration_date = ne.exp_date
GROUP BY oc.strike_price
ORDER BY oc.strike_price;

-- Option chain with moneyness calculation
WITH current_spx AS (
    SELECT price as spot_price
    FROM stock_data_ticks sdt
    JOIN underlying_securities us ON sdt.underlying_id = us.id
    WHERE us.symbol = 'SPX'
    ORDER BY sdt.time DESC
    LIMIT 1
),
latest_data AS (
    SELECT DISTINCT ON (contract_id)
        contract_id,
        bid_price,
        ask_price,
        mid_price,
        volume,
        open_interest,
        implied_volatility
    FROM market_data_ticks
    WHERE time > NOW() - INTERVAL '1 hour'
    ORDER BY contract_id, time DESC
)
SELECT
    oc.expiration_date,
    oc.strike_price,
    oc.option_type,
    cs.spot_price,
    ROUND((oc.strike_price / cs.spot_price - 1) * 100, 2) as moneyness_pct,
    CASE
        WHEN oc.option_type = 'C' AND oc.strike_price < cs.spot_price THEN 'ITM'
        WHEN oc.option_type = 'C' AND oc.strike_price > cs.spot_price THEN 'OTM'
        WHEN oc.option_type = 'P' AND oc.strike_price > cs.spot_price THEN 'ITM'
        WHEN oc.option_type = 'P' AND oc.strike_price < cs.spot_price THEN 'OTM'
        ELSE 'ATM'
    END as moneyness,
    ld.bid_price,
    ld.ask_price,
    ld.mid_price,
    ld.volume,
    ld.open_interest,
    ld.implied_volatility
FROM option_chains oc
JOIN underlying_securities us ON oc.underlying_id = us.id
CROSS JOIN current_spx cs
LEFT JOIN latest_data ld ON oc.id = ld.contract_id
WHERE us.symbol = 'SPX'
    AND oc.expiration_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '60 days'
    AND oc.strike_price BETWEEN cs.spot_price * 0.9 AND cs.spot_price * 1.1
ORDER BY oc.expiration_date, oc.strike_price, oc.option_type;

-- ============================================================================
-- 5. SPX INDEX PRICE DATA
-- ============================================================================

-- Latest SPX index price
SELECT
    time,
    price,
    bid_price,
    ask_price,
    volume,
    vwap,
    open_price,
    high_price,
    low_price,
    close_price
FROM stock_data_ticks sdt
JOIN underlying_securities us ON sdt.underlying_id = us.id
WHERE us.symbol = 'SPX'
ORDER BY time DESC
LIMIT 1;

-- SPX price history (5-minute bars for last day)
SELECT
    time_bucket('5 minutes', time) as bar_time,
    FIRST(price, time) as open,
    MAX(price) as high,
    MIN(price) as low,
    LAST(price, time) as close,
    SUM(volume) as volume,
    AVG(vwap) as avg_vwap
FROM stock_data_ticks sdt
JOIN underlying_securities us ON sdt.underlying_id = us.id
WHERE us.symbol = 'SPX'
    AND time > NOW() - INTERVAL '1 day'
GROUP BY bar_time
ORDER BY bar_time DESC;

-- ============================================================================
-- 6. ARBITRAGE OPPORTUNITIES
-- ============================================================================

-- Find SPX arbitrage opportunities
SELECT
    ao.strategy_type,
    ao.timestamp,
    ao.expiration_date,
    ao.call_strike,
    ao.put_strike,
    ao.stock_price,
    ao.call_bid,
    ao.call_ask,
    ao.put_bid,
    ao.put_ask,
    ao.theoretical_profit,
    ao.roi_percent,
    ao.confidence_score,
    ao.execution_attempted,
    ao.execution_successful,
    ao.actual_profit
FROM arbitrage_opportunities ao
JOIN underlying_securities us ON ao.underlying_id = us.id
WHERE us.symbol = 'SPX'
    AND ao.timestamp > NOW() - INTERVAL '1 day'
ORDER BY ao.timestamp DESC;

-- Best arbitrage opportunities by ROI
SELECT
    ao.strategy_type,
    ao.timestamp,
    ao.expiration_date,
    ao.call_strike,
    ao.put_strike,
    ao.theoretical_profit,
    ao.roi_percent,
    ao.confidence_score
FROM arbitrage_opportunities ao
JOIN underlying_securities us ON ao.underlying_id = us.id
WHERE us.symbol = 'SPX'
    AND ao.roi_percent > 0.5  -- At least 0.5% ROI
    AND ao.confidence_score > 0.7
    AND ao.timestamp > NOW() - INTERVAL '7 days'
ORDER BY ao.roi_percent DESC
LIMIT 20;

-- ============================================================================
-- 7. VOLATILITY ANALYSIS
-- ============================================================================

-- Implied volatility surface
SELECT
    oc.expiration_date,
    EXTRACT(days FROM oc.expiration_date - CURRENT_DATE) as days_to_expiry,
    oc.strike_price,
    oc.option_type,
    AVG(md.implied_volatility) as avg_iv,
    STDDEV(md.implied_volatility) as iv_stddev,
    MIN(md.implied_volatility) as min_iv,
    MAX(md.implied_volatility) as max_iv
FROM market_data_ticks md
JOIN option_chains oc ON md.contract_id = oc.id
JOIN underlying_securities us ON oc.underlying_id = us.id
WHERE us.symbol = 'SPX'
    AND md.time > NOW() - INTERVAL '1 hour'
    AND md.implied_volatility IS NOT NULL
    AND md.implied_volatility > 0
GROUP BY oc.expiration_date, oc.strike_price, oc.option_type
ORDER BY oc.expiration_date, oc.strike_price;

-- IV skew analysis (25-delta put vs 25-delta call)
WITH delta_options AS (
    SELECT DISTINCT ON (oc.expiration_date, oc.option_type)
        oc.expiration_date,
        oc.option_type,
        oc.strike_price,
        md.implied_volatility,
        ABS(md.delta - 0.25) as delta_diff
    FROM market_data_ticks md
    JOIN option_chains oc ON md.contract_id = oc.id
    JOIN underlying_securities us ON oc.underlying_id = us.id
    WHERE us.symbol = 'SPX'
        AND md.time > NOW() - INTERVAL '1 hour'
        AND md.delta IS NOT NULL
        AND ABS(ABS(md.delta) - 0.25) < 0.05  -- Within 5% of 25-delta
    ORDER BY oc.expiration_date, oc.option_type, delta_diff
)
SELECT
    p.expiration_date,
    p.strike_price as put_strike,
    p.implied_volatility as put_25d_iv,
    c.strike_price as call_strike,
    c.implied_volatility as call_25d_iv,
    p.implied_volatility - c.implied_volatility as iv_skew
FROM delta_options p
JOIN delta_options c ON p.expiration_date = c.expiration_date
WHERE p.option_type = 'P' AND c.option_type = 'C'
ORDER BY p.expiration_date;

-- ============================================================================
-- 8. LIQUIDITY ANALYSIS
-- ============================================================================

-- Most liquid SPX options by volume
SELECT
    oc.contract_symbol,
    oc.expiration_date,
    oc.strike_price,
    oc.option_type,
    SUM(md.volume) as total_volume,
    AVG(md.open_interest) as avg_oi,
    AVG(md.bid_ask_spread) as avg_spread,
    COUNT(*) as tick_count
FROM market_data_ticks md
JOIN option_chains oc ON md.contract_id = oc.id
JOIN underlying_securities us ON oc.underlying_id = us.id
WHERE us.symbol = 'SPX'
    AND md.time > NOW() - INTERVAL '1 day'
GROUP BY oc.contract_symbol, oc.expiration_date, oc.strike_price, oc.option_type
HAVING SUM(md.volume) > 0
ORDER BY total_volume DESC
LIMIT 50;

-- Open interest distribution
SELECT
    oc.expiration_date,
    oc.option_type,
    SUM(CASE WHEN oc.strike_price < s.current_price THEN md.open_interest ELSE 0 END) as itm_oi,
    SUM(CASE WHEN oc.strike_price = s.current_price THEN md.open_interest ELSE 0 END) as atm_oi,
    SUM(CASE WHEN oc.strike_price > s.current_price THEN md.open_interest ELSE 0 END) as otm_oi,
    SUM(md.open_interest) as total_oi
FROM (
    SELECT DISTINCT ON (contract_id)
        contract_id,
        open_interest
    FROM market_data_ticks
    WHERE time > NOW() - INTERVAL '1 hour'
    ORDER BY contract_id, time DESC
) md
JOIN option_chains oc ON md.contract_id = oc.id
JOIN underlying_securities us ON oc.underlying_id = us.id
CROSS JOIN (
    SELECT price as current_price
    FROM stock_data_ticks sdt
    JOIN underlying_securities us ON sdt.underlying_id = us.id
    WHERE us.symbol = 'SPX'
    ORDER BY time DESC
    LIMIT 1
) s
WHERE us.symbol = 'SPX'
GROUP BY oc.expiration_date, oc.option_type
ORDER BY oc.expiration_date, oc.option_type;

-- ============================================================================
-- 9. DATA QUALITY CHECKS
-- ============================================================================

-- Check for missing data in last hour
SELECT
    oc.contract_symbol,
    oc.expiration_date,
    oc.strike_price,
    oc.option_type,
    MAX(md.time) as last_update,
    EXTRACT(minutes FROM NOW() - MAX(md.time)) as minutes_since_update
FROM option_chains oc
JOIN underlying_securities us ON oc.underlying_id = us.id
LEFT JOIN market_data_ticks md ON oc.id = md.contract_id
    AND md.time > NOW() - INTERVAL '1 hour'
WHERE us.symbol = 'SPX'
    AND oc.active = true
    AND oc.expiration_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '30 days'
GROUP BY oc.contract_symbol, oc.expiration_date, oc.strike_price, oc.option_type
HAVING MAX(md.time) IS NULL OR MAX(md.time) < NOW() - INTERVAL '10 minutes'
ORDER BY oc.expiration_date, oc.strike_price;

-- Validate bid-ask spreads (identify potentially bad data)
SELECT
    oc.contract_symbol,
    md.time,
    md.bid_price,
    md.ask_price,
    md.bid_ask_spread,
    md.mid_price,
    (md.bid_ask_spread / NULLIF(md.mid_price, 0)) * 100 as spread_pct
FROM market_data_ticks md
JOIN option_chains oc ON md.contract_id = oc.id
JOIN underlying_securities us ON oc.underlying_id = us.id
WHERE us.symbol = 'SPX'
    AND md.time > NOW() - INTERVAL '1 hour'
    AND (
        md.bid_price > md.ask_price  -- Crossed market
        OR md.bid_ask_spread < 0     -- Negative spread
        OR (md.bid_ask_spread / NULLIF(md.mid_price, 0)) > 0.5  -- Spread > 50%
    )
ORDER BY md.time DESC;

-- ============================================================================
-- 10. PERFORMANCE QUERIES (for TimescaleDB)
-- ============================================================================

-- If using TimescaleDB, these queries leverage hypertable features

-- Continuous aggregate for 1-minute option bars (create once)
-- CREATE MATERIALIZED VIEW spx_options_1min
-- WITH (timescaledb.continuous) AS
-- SELECT
--     time_bucket('1 minute', md.time) as bucket,
--     md.contract_id,
--     AVG(md.bid_price) as avg_bid,
--     AVG(md.ask_price) as avg_ask,
--     AVG(md.mid_price) as avg_mid,
--     SUM(md.volume) as total_volume,
--     LAST(md.open_interest, md.time) as last_oi,
--     AVG(md.implied_volatility) as avg_iv
-- FROM market_data_ticks md
-- JOIN option_chains oc ON md.contract_id = oc.id
-- JOIN underlying_securities us ON oc.underlying_id = us.id
-- WHERE us.symbol = 'SPX'
-- GROUP BY bucket, md.contract_id
-- WITH NO DATA;

-- Query the continuous aggregate (much faster)
-- SELECT * FROM spx_options_1min
-- WHERE bucket > NOW() - INTERVAL '1 day'
-- ORDER BY bucket DESC;

-- ============================================================================
-- USEFUL PARAMETERS TO REPLACE:
-- - Replace dates like '2024-02-16' with actual dates
-- - Replace strike prices like 5000 with actual strikes
-- - Replace time intervals as needed ('1 hour', '1 day', etc.)
-- - Replace ROI thresholds (0.5%) with your criteria
-- ============================================================================
