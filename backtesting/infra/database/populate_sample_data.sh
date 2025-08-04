#!/bin/bash
# populate_sample_data.sh
# Script to populate the options arbitrage database with sample data for testing

set -e

# Database connection parameters
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-options_arbitrage}
DB_USER=${DB_USER:-trading_user}
DB_PASSWORD=${DB_PASSWORD:-secure_trading_password}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Options Arbitrage Database Population Script${NC}"
echo "=============================================="

# Function to execute SQL with error handling
execute_sql() {
    local sql="$1"
    local description="$2"

    echo -e "${YELLOW}$description...${NC}"

    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "$sql" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $description completed${NC}"
    else
        echo -e "${RED}✗ $description failed${NC}"
        return 1
    fi
}

# Check database connection
echo -e "${YELLOW}Testing database connection...${NC}"
if ! PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" > /dev/null 2>&1; then
    echo -e "${RED}✗ Cannot connect to database. Please ensure the container is running.${NC}"
    echo "Run: docker-compose up -d"
    exit 1
fi
echo -e "${GREEN}✓ Database connection successful${NC}"

# 1. Populate underlying securities
echo -e "\n${BLUE}1. Populating underlying securities...${NC}"

execute_sql "
INSERT INTO underlying_securities (symbol, name, sector, industry, market_cap, active) VALUES
('SPY', 'SPDR S&P 500 ETF Trust', 'ETF', 'Broad Market ETF', 400000000000, true),
('QQQ', 'Invesco QQQ Trust', 'ETF', 'Technology ETF', 180000000000, true),
('IWM', 'iShares Russell 2000 ETF', 'ETF', 'Small Cap ETF', 60000000000, true),
('AAPL', 'Apple Inc.', 'Technology', 'Consumer Electronics', 3000000000000, true),
('MSFT', 'Microsoft Corporation', 'Technology', 'Software', 2800000000000, true),
('GOOGL', 'Alphabet Inc.', 'Technology', 'Internet Services', 1700000000000, true),
('AMZN', 'Amazon.com Inc.', 'Consumer Discretionary', 'E-commerce', 1500000000000, true),
('TSLA', 'Tesla Inc.', 'Consumer Discretionary', 'Electric Vehicles', 800000000000, true),
('META', 'Meta Platforms Inc.', 'Technology', 'Social Media', 750000000000, true),
('NVDA', 'NVIDIA Corporation', 'Technology', 'Semiconductors', 1200000000000, true)
ON CONFLICT (symbol) DO UPDATE SET
    name = EXCLUDED.name,
    sector = EXCLUDED.sector,
    industry = EXCLUDED.industry,
    market_cap = EXCLUDED.market_cap,
    updated_at = CURRENT_TIMESTAMP;
" "Inserting underlying securities"

# 2. Populate option chains
echo -e "\n${BLUE}2. Populating option chains...${NC}"

execute_sql "
WITH underlying_data AS (
    SELECT id, symbol FROM underlying_securities WHERE active = true
),
expiry_dates AS (
    SELECT
        (CURRENT_DATE + INTERVAL '1 week' * s.week_offset)::date as expiry_date
    FROM generate_series(1, 8) s(week_offset)  -- Next 8 weeks
),
strikes AS (
    SELECT
        CASE
            WHEN ud.symbol IN ('SPY') THEN 400 + (s.strike_offset * 5)
            WHEN ud.symbol IN ('QQQ') THEN 350 + (s.strike_offset * 5)
            WHEN ud.symbol IN ('AAPL') THEN 180 + (s.strike_offset * 5)
            WHEN ud.symbol IN ('MSFT') THEN 400 + (s.strike_offset * 5)
            WHEN ud.symbol IN ('GOOGL') THEN 140 + (s.strike_offset * 2)
            WHEN ud.symbol IN ('AMZN') THEN 140 + (s.strike_offset * 2)
            WHEN ud.symbol IN ('TSLA') THEN 200 + (s.strike_offset * 10)
            WHEN ud.symbol IN ('META') THEN 450 + (s.strike_offset * 10)
            WHEN ud.symbol IN ('NVDA') THEN 900 + (s.strike_offset * 20)
            ELSE 100 + (s.strike_offset * 5)
        END as strike_price,
        ud.id as underlying_id,
        ud.symbol
    FROM underlying_data ud
    CROSS JOIN generate_series(-5, 5) s(strike_offset)  -- ATM ±5 strikes
)
INSERT INTO option_chains
(underlying_id, expiration_date, strike_price, option_type, contract_symbol, ib_con_id, exchange, active)
SELECT
    s.underlying_id,
    ed.expiry_date,
    s.strike_price,
    ot.option_type,
    s.symbol || to_char(ed.expiry_date, 'YYMMDD') ||
        CASE WHEN ot.option_type = 'C' THEN 'C' ELSE 'P' END ||
        LPAD(CAST(s.strike_price * 1000 AS TEXT), 8, '0') as contract_symbol,
    -- Generate fake IB contract IDs (in real scenario these come from IB API)
    (600000000 + (s.underlying_id * 100000) +
     (EXTRACT(EPOCH FROM ed.expiry_date)::bigint % 10000) +
     (s.strike_price::int % 1000) +
     CASE WHEN ot.option_type = 'C' THEN 0 ELSE 50000 END) as ib_con_id,
    'SMART' as exchange,
    true as active
FROM strikes s
CROSS JOIN expiry_dates ed
CROSS JOIN (VALUES ('C'), ('P')) ot(option_type)
WHERE ed.expiry_date > CURRENT_DATE  -- Only future expiries
ON CONFLICT (underlying_id, expiration_date, strike_price, option_type)
DO UPDATE SET
    ib_con_id = EXCLUDED.ib_con_id,
    updated_at = CURRENT_TIMESTAMP;
" "Generating option chains for all symbols"

# 3. Populate sample market data
echo -e "\n${BLUE}3. Populating sample market data...${NC}"

execute_sql "
WITH option_data AS (
    SELECT
        oc.id as contract_id,
        oc.strike_price,
        oc.option_type,
        us.symbol,
        oc.expiration_date,
        -- Calculate days to expiry
        EXTRACT(EPOCH FROM (oc.expiration_date - CURRENT_DATE)) / 86400 as days_to_expiry
    FROM option_chains oc
    JOIN underlying_securities us ON oc.underlying_id = us.id
    WHERE oc.active = true
    AND oc.expiration_date > CURRENT_DATE
    LIMIT 500  -- Limit to first 500 contracts for sample data
),
time_series AS (
    SELECT
        CURRENT_TIMESTAMP - INTERVAL '1 hour' + (INTERVAL '1 minute' * s.minute_offset) as timestamp
    FROM generate_series(0, 60, 5) s(minute_offset)  -- Every 5 minutes for last hour
)
INSERT INTO market_data_ticks
(time, contract_id, bid_price, ask_price, last_price, bid_size, ask_size, volume,
 delta, gamma, theta, vega, implied_volatility, tick_type)
SELECT
    ts.timestamp,
    od.contract_id,
    -- Generate realistic option prices based on moneyness and time to expiry
    CASE
        WHEN od.option_type = 'C' THEN
            GREATEST(0.05,
                CASE
                    WHEN od.symbol = 'SPY' THEN (520 - od.strike_price) + (random() * 2 - 1) * 5
                    WHEN od.symbol = 'QQQ' THEN (380 - od.strike_price) + (random() * 2 - 1) * 3
                    WHEN od.symbol = 'AAPL' THEN (190 - od.strike_price) + (random() * 2 - 1) * 3
                    ELSE (150 - od.strike_price) + (random() * 2 - 1) * 2
                END + (od.days_to_expiry / 365.0 * 10) + random() * 2
            )
        ELSE
            GREATEST(0.05,
                CASE
                    WHEN od.symbol = 'SPY' THEN (od.strike_price - 520) + (random() * 2 - 1) * 5
                    WHEN od.symbol = 'QQQ' THEN (od.strike_price - 380) + (random() * 2 - 1) * 3
                    WHEN od.symbol = 'AAPL' THEN (od.strike_price - 190) + (random() * 2 - 1) * 3
                    ELSE (od.strike_price - 150) + (random() * 2 - 1) * 2
                END + (od.days_to_expiry / 365.0 * 10) + random() * 2
            )
    END as bid_price,
    -- Ask price (bid + spread)
    CASE
        WHEN od.option_type = 'C' THEN
            GREATEST(0.06,
                CASE
                    WHEN od.symbol = 'SPY' THEN (520 - od.strike_price) + (random() * 2 - 1) * 5
                    WHEN od.symbol = 'QQQ' THEN (380 - od.strike_price) + (random() * 2 - 1) * 3
                    WHEN od.symbol = 'AAPL' THEN (190 - od.strike_price) + (random() * 2 - 1) * 3
                    ELSE (150 - od.strike_price) + (random() * 2 - 1) * 2
                END + (od.days_to_expiry / 365.0 * 10) + random() * 2 + 0.05
            )
        ELSE
            GREATEST(0.06,
                CASE
                    WHEN od.symbol = 'SPY' THEN (od.strike_price - 520) + (random() * 2 - 1) * 5
                    WHEN od.symbol = 'QQQ' THEN (od.strike_price - 380) + (random() * 2 - 1) * 3
                    WHEN od.symbol = 'AAPL' THEN (od.strike_price - 190) + (random() * 2 - 1) * 3
                    ELSE (od.strike_price - 150) + (random() * 2 - 1) * 2
                END + (od.days_to_expiry / 365.0 * 10) + random() * 2 + 0.05
            )
    END as ask_price,
    -- Last price (between bid and ask)
    CASE
        WHEN od.option_type = 'C' THEN
            GREATEST(0.05,
                CASE
                    WHEN od.symbol = 'SPY' THEN (520 - od.strike_price) + (random() * 2 - 1) * 5
                    WHEN od.symbol = 'QQQ' THEN (380 - od.strike_price) + (random() * 2 - 1) * 3
                    WHEN od.symbol = 'AAPL' THEN (190 - od.strike_price) + (random() * 2 - 1) * 3
                    ELSE (150 - od.strike_price) + (random() * 2 - 1) * 2
                END + (od.days_to_expiry / 365.0 * 10) + random() * 2 + 0.025
            )
        ELSE
            GREATEST(0.05,
                CASE
                    WHEN od.symbol = 'SPY' THEN (od.strike_price - 520) + (random() * 2 - 1) * 5
                    WHEN od.symbol = 'QQQ' THEN (od.strike_price - 380) + (random() * 2 - 1) * 3
                    WHEN od.symbol = 'AAPL' THEN (od.strike_price - 190) + (random() * 2 - 1) * 3
                    ELSE (od.strike_price - 150) + (random() * 2 - 1) * 2
                END + (od.days_to_expiry / 365.0 * 10) + random() * 2 + 0.025
            )
    END as last_price,
    -- Sizes
    (50 + random() * 100)::int as bid_size,
    (50 + random() * 100)::int as ask_size,
    (random() * 1000)::int as volume,
    -- Greeks (simplified calculations)
    CASE WHEN od.option_type = 'C' THEN
        LEAST(0.99, GREATEST(0.01, 0.5 + (520 - od.strike_price) * 0.01))
    ELSE
        GREATEST(-0.99, LEAST(-0.01, -0.5 - (od.strike_price - 520) * 0.01))
    END as delta,
    GREATEST(0.001, random() * 0.1) as gamma,
    -GREATEST(0.01, random() * 2) as theta,
    GREATEST(0.1, random() * 3) as vega,
    GREATEST(0.1, LEAST(3.0, 0.2 + random() * 0.5)) as implied_volatility,
    'SAMPLE' as tick_type
FROM option_data od
CROSS JOIN time_series ts
ON CONFLICT DO NOTHING;
" "Generating sample market data ticks"

# 4. Populate stock data
echo -e "\n${BLUE}4. Populating sample stock data...${NC}"

execute_sql "
WITH stock_prices AS (
    SELECT
        us.id as underlying_id,
        us.symbol,
        CASE
            WHEN us.symbol = 'SPY' THEN 520
            WHEN us.symbol = 'QQQ' THEN 380
            WHEN us.symbol = 'IWM' THEN 200
            WHEN us.symbol = 'AAPL' THEN 190
            WHEN us.symbol = 'MSFT' THEN 420
            WHEN us.symbol = 'GOOGL' THEN 145
            WHEN us.symbol = 'AMZN' THEN 145
            WHEN us.symbol = 'TSLA' THEN 250
            WHEN us.symbol = 'META' THEN 485
            WHEN us.symbol = 'NVDA' THEN 950
            ELSE 100
        END as base_price
    FROM underlying_securities us WHERE active = true
),
time_series AS (
    SELECT
        CURRENT_TIMESTAMP - INTERVAL '1 hour' + (INTERVAL '1 minute' * s.minute_offset) as timestamp
    FROM generate_series(0, 60, 1) s(minute_offset)  -- Every minute for last hour
)
INSERT INTO stock_data_ticks
(time, underlying_id, price, bid_price, ask_price, volume, tick_type)
SELECT
    ts.timestamp,
    sp.underlying_id,
    sp.base_price + (random() * 4 - 2) as price,  -- ±$2 random walk
    sp.base_price + (random() * 4 - 2) - 0.01 as bid_price,
    sp.base_price + (random() * 4 - 2) + 0.01 as ask_price,
    (random() * 10000)::int as volume,
    'SAMPLE' as tick_type
FROM stock_prices sp
CROSS JOIN time_series ts
ON CONFLICT DO NOTHING;
" "Generating sample stock data"

# 5. Add some sample arbitrage opportunities
echo -e "\n${BLUE}5. Adding sample arbitrage opportunities...${NC}"

execute_sql "
INSERT INTO arbitrage_opportunities
(strategy_type, underlying_id, timestamp, expiration_date, call_strike, put_strike,
 stock_price, call_bid, call_ask, put_bid, put_ask, theoretical_profit, max_profit,
 roi_percent, net_credit, confidence_score, discovered_by)
SELECT
    'SFR' as strategy_type,
    us.id as underlying_id,
    CURRENT_TIMESTAMP - INTERVAL '30 minutes' + (random() * INTERVAL '25 minutes') as timestamp,
    CURRENT_DATE + INTERVAL '2 weeks' as expiration_date,
    520 as call_strike,
    520 as put_strike,
    520.50 as stock_price,
    12.50 + random() * 2 as call_bid,
    12.70 + random() * 2 as call_ask,
    12.30 + random() * 2 as put_bid,
    12.50 + random() * 2 as put_ask,
    0.75 + random() * 0.5 as theoretical_profit,
    1.25 + random() * 0.5 as max_profit,
    (0.75 + random() * 0.5) / 520 * 100 as roi_percent,
    -1.20 + random() * 0.4 as net_credit,
    0.85 + random() * 0.1 as confidence_score,
    'SCANNER' as discovered_by
FROM underlying_securities us
WHERE us.symbol IN ('SPY', 'QQQ', 'AAPL')
UNION ALL
SELECT
    'SYNTHETIC' as strategy_type,
    us.id as underlying_id,
    CURRENT_TIMESTAMP - INTERVAL '45 minutes' + (random() * INTERVAL '40 minutes') as timestamp,
    CURRENT_DATE + INTERVAL '3 weeks' as expiration_date,
    CASE
        WHEN us.symbol = 'SPY' THEN 525
        WHEN us.symbol = 'QQQ' THEN 385
        ELSE 195
    END as call_strike,
    CASE
        WHEN us.symbol = 'SPY' THEN 525
        WHEN us.symbol = 'QQQ' THEN 385
        ELSE 195
    END as put_strike,
    CASE
        WHEN us.symbol = 'SPY' THEN 524.25
        WHEN us.symbol = 'QQQ' THEN 384.80
        ELSE 194.75
    END as stock_price,
    8.50 + random() * 3 as call_bid,
    8.80 + random() * 3 as call_ask,
    9.20 + random() * 3 as put_bid,
    9.50 + random() * 3 as put_ask,
    0.45 + random() * 0.3 as theoretical_profit,
    0.95 + random() * 0.3 as max_profit,
    (0.45 + random() * 0.3) / 524 * 100 as roi_percent,
    -0.80 + random() * 0.2 as net_credit,
    0.75 + random() * 0.15 as confidence_score,
    'SCANNER' as discovered_by
FROM underlying_securities us
WHERE us.symbol IN ('SPY', 'QQQ', 'AAPL');
" "Adding sample arbitrage opportunities"

# 6. Populate sample VIX data
echo -e "\n${BLUE}6. Populating sample VIX data...${NC}"

execute_sql "
WITH vix_time_series AS (
    SELECT
        CURRENT_TIMESTAMP - INTERVAL '1 hour' + (INTERVAL '1 minute' * s.minute_offset) as timestamp
    FROM generate_series(0, 60, 2) s(minute_offset)  -- Every 2 minutes for last hour
),
vix_instruments_data AS (
    SELECT id, symbol FROM vix_instruments WHERE active = true
)
INSERT INTO vix_data_ticks
(time, instrument_id, last_price, bid_price, ask_price, volume, tick_type, data_quality_score)
SELECT
    vts.timestamp,
    vi.id,
    CASE
        WHEN vi.symbol = 'VIX' THEN 18.5 + (random() * 8 - 4)  -- VIX: 14.5-22.5
        WHEN vi.symbol = 'VIX1D' THEN 16.0 + (random() * 6 - 3)  -- VIX1D: 13-19
        WHEN vi.symbol = 'VIX9D' THEN 17.5 + (random() * 7 - 3.5)  -- VIX9D: 14-21
        WHEN vi.symbol = 'VIX3M' THEN 20.0 + (random() * 8 - 4)  -- VIX3M: 16-24
        WHEN vi.symbol = 'VIX6M' THEN 21.5 + (random() * 6 - 3)  -- VIX6M: 18.5-24.5
        ELSE 18.0 + (random() * 4 - 2)
    END as last_price,
    CASE
        WHEN vi.symbol = 'VIX' THEN 18.3 + (random() * 8 - 4)
        WHEN vi.symbol = 'VIX1D' THEN 15.8 + (random() * 6 - 3)
        WHEN vi.symbol = 'VIX9D' THEN 17.3 + (random() * 7 - 3.5)
        WHEN vi.symbol = 'VIX3M' THEN 19.8 + (random() * 8 - 4)
        WHEN vi.symbol = 'VIX6M' THEN 21.3 + (random() * 6 - 3)
        ELSE 17.8 + (random() * 4 - 2)
    END as bid_price,
    CASE
        WHEN vi.symbol = 'VIX' THEN 18.7 + (random() * 8 - 4)
        WHEN vi.symbol = 'VIX1D' THEN 16.2 + (random() * 6 - 3)
        WHEN vi.symbol = 'VIX9D' THEN 17.7 + (random() * 7 - 3.5)
        WHEN vi.symbol = 'VIX3M' THEN 20.2 + (random() * 8 - 4)
        WHEN vi.symbol = 'VIX6M' THEN 21.7 + (random() * 6 - 3)
        ELSE 18.2 + (random() * 4 - 2)
    END as ask_price,
    (random() * 1000)::int as volume,
    'SAMPLE' as tick_type,
    0.95 + random() * 0.05 as data_quality_score
FROM vix_time_series vts
CROSS JOIN vix_instruments_data vi
ON CONFLICT DO NOTHING;
\" \"Generating sample VIX data ticks\"

execute_sql "
WITH vix_snapshots AS (
    SELECT
        CURRENT_TIMESTAMP - INTERVAL '30 minutes' + (INTERVAL '5 minutes' * s.minute_offset) as timestamp
    FROM generate_series(0, 6) s(minute_offset)  -- Every 5 minutes for last 30 minutes
)
INSERT INTO vix_term_structure
(timestamp, vix_1d, vix_9d, vix_30d, vix_3m, vix_6m)
SELECT
    vs.timestamp,
    16.0 + (random() * 6 - 3) as vix_1d,      -- VIX1D: 13-19
    17.5 + (random() * 7 - 3.5) as vix_9d,    -- VIX9D: 14-21
    18.5 + (random() * 8 - 4) as vix_30d,     -- VIX: 14.5-22.5
    20.0 + (random() * 8 - 4) as vix_3m,      -- VIX3M: 16-24
    21.5 + (random() * 6 - 3) as vix_6m       -- VIX6M: 18.5-24.5
FROM vix_snapshots vs
ON CONFLICT (timestamp) DO NOTHING;
\" \"Generating VIX term structure snapshots\"

# Add some sample VIX correlation records
execute_sql "
INSERT INTO vix_arbitrage_correlation
(arbitrage_opportunity_id, vix_snapshot_id, vix_level, vix_regime,
 term_structure_type, vix_spike_active, arbitrage_success,
 execution_speed_ms, profit_realized, correlation_weight)
SELECT
    ao.id,
    vts.id,
    vts.vix_30d,
    CASE
        WHEN vts.vix_30d < 15 THEN 'LOW'
        WHEN vts.vix_30d BETWEEN 15 AND 25 THEN 'MEDIUM'
        WHEN vts.vix_30d BETWEEN 25 AND 40 THEN 'HIGH'
        ELSE 'EXTREME'
    END as vix_regime,
    CASE
        WHEN vts.vix_3m > vts.vix_30d THEN 'CONTANGO'
        WHEN vts.vix_3m < vts.vix_30d THEN 'BACKWARDATION'
        ELSE 'FLAT'
    END as term_structure_type,
    vts.vix_30d > 30 as vix_spike_active,
    random() > 0.3 as arbitrage_success,  -- 70% success rate for sample data
    (50 + random() * 200)::int as execution_speed_ms,
    CASE WHEN random() > 0.3 THEN
        ao.theoretical_profit * (0.8 + random() * 0.4)  -- 80-120% of theoretical
    ELSE NULL END as profit_realized,
    1.0 as correlation_weight
FROM arbitrage_opportunities ao
CROSS JOIN vix_term_structure vts
WHERE ao.id % 3 = 0  -- Sample every 3rd arbitrage opportunity
LIMIT 50;  -- Limit to 50 correlation records
\" \"Adding sample VIX correlation records\"

# 7. Add corporate actions
echo -e "\n${BLUE}7. Adding sample corporate actions...${NC}"

execute_sql "
INSERT INTO corporate_actions
(underlying_id, action_type, ex_date, record_date, payable_date, amount, description)
SELECT
    us.id,
    'DIVIDEND' as action_type,
    CURRENT_DATE + INTERVAL '1 week' as ex_date,
    CURRENT_DATE + INTERVAL '2 weeks' as record_date,
    CURRENT_DATE + INTERVAL '4 weeks' as payable_date,
    CASE
        WHEN us.symbol = 'SPY' THEN 1.50
        WHEN us.symbol = 'QQQ' THEN 0.85
        WHEN us.symbol = 'AAPL' THEN 0.24
        WHEN us.symbol = 'MSFT' THEN 0.75
        ELSE 0.50
    END as amount,
    'Quarterly dividend payment' as description
FROM underlying_securities us
WHERE us.symbol IN ('SPY', 'QQQ', 'AAPL', 'MSFT', 'IWM')
ON CONFLICT DO NOTHING;
" "Adding sample corporate actions"

# 7. Show summary statistics
echo -e "\n${BLUE}Database Population Summary:${NC}"
echo "============================="

execute_sql "SELECT COUNT(*) as underlying_count FROM underlying_securities;" "Underlying securities"
execute_sql "SELECT COUNT(*) as option_contracts FROM option_chains;" "Option contracts"
execute_sql "SELECT COUNT(*) as market_data_points FROM market_data_ticks;" "Market data ticks"
execute_sql "SELECT COUNT(*) as stock_data_points FROM stock_data_ticks;" "Stock data points"
execute_sql "SELECT COUNT(*) as arbitrage_opportunities FROM arbitrage_opportunities;" "Arbitrage opportunities"
execute_sql "SELECT COUNT(*) as corporate_actions FROM corporate_actions;" "Corporate actions"
execute_sql "SELECT COUNT(*) as vix_instruments FROM vix_instruments;" "VIX instruments"
execute_sql "SELECT COUNT(*) as vix_data_points FROM vix_data_ticks;" "VIX data ticks"
execute_sql "SELECT COUNT(*) as vix_term_structures FROM vix_term_structure;" "VIX term structures"
execute_sql "SELECT COUNT(*) as vix_correlations FROM vix_arbitrage_correlation;" "VIX correlations"

echo -e "\n${GREEN}✅ Database population completed successfully!${NC}"
echo -e "\n${BLUE}Next steps:${NC}"
echo "• Test queries: docker exec -it alc_timescaledb psql -U trading_user -d options_arbitrage"
echo "• View hypertables: SELECT * FROM hypertable_stats;"
echo "• Check data: SELECT * FROM underlying_securities LIMIT 5;"
echo "• Start data collection with your Python collectors"

echo -e "\n${YELLOW}Sample queries to try:${NC}"
echo "SELECT symbol, COUNT(*) as option_count FROM underlying_securities us"
echo "JOIN option_chains oc ON us.id = oc.underlying_id GROUP BY symbol;"
echo ""
echo "SELECT strategy_type, COUNT(*), AVG(roi_percent) as avg_roi"
echo "FROM arbitrage_opportunities GROUP BY strategy_type;"
echo ""
echo "-- VIX Analysis Queries:"
echo "SELECT * FROM get_latest_vix_structure();"
echo ""
echo "SELECT vix_regime, COUNT(*) as opportunities, "
echo "       AVG(CASE WHEN arbitrage_success THEN 1.0 ELSE 0.0 END) * 100 as success_rate"
echo "FROM vix_arbitrage_correlation GROUP BY vix_regime;"
echo ""
echo "SELECT term_structure_type, COUNT(*) as count,"
echo "       AVG(vix_level) as avg_vix FROM vix_term_structure"
echo "WHERE term_structure_type IS NOT NULL GROUP BY term_structure_type;"
