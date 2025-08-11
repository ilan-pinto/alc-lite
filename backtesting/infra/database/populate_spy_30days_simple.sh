#!/bin/bash
# populate_spy_30days_simple.sh
# Simplified script to populate 30 days of SPY sample data

set -e

# Database connection parameters
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5433}
DB_NAME=${DB_NAME:-options_arbitrage}
DB_USER=${DB_USER:-trading_user}
DB_PASSWORD=${DB_PASSWORD:-secure_trading_password}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}SPY 30-Day Simple Data Population${NC}"
echo "================================="

# Function to execute SQL with error handling
execute_sql() {
    local sql="$1"
    local description="$2"

    echo -e "${YELLOW}$description...${NC}"

    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "$sql" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $description completed${NC}"
    else
        echo -e "${RED}✗ $description failed${NC}"
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "$sql"
        return 1
    fi
}

# Test connection
echo -e "${YELLOW}Testing database connection...${NC}"
if ! PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" > /dev/null 2>&1; then
    echo -e "${RED}✗ Cannot connect to database${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Database connection successful${NC}"

# 1. Clear existing data
execute_sql "
TRUNCATE TABLE
    vix_arbitrage_correlation,
    vix_term_structure,
    vix_data_ticks,
    arbitrage_opportunities,
    market_data_ticks,
    stock_data_ticks,
    corporate_actions,
    option_chains
CASCADE;

DELETE FROM underlying_securities
WHERE symbol NOT IN ('SPY', 'PLTR', 'TSLA');
" "Clearing existing data"

# 2. Insert Phase 1 symbols
execute_sql "
INSERT INTO underlying_securities (symbol, name, sector, industry, market_cap, active) VALUES
('SPY', 'SPDR S&P 500 ETF Trust', 'ETF', 'Broad Market ETF', 550000000000, true),
('PLTR', 'Palantir Technologies Inc.', 'Technology', 'Software - Infrastructure', 150000000000, true),
('TSLA', 'Tesla Inc.', 'Consumer Discretionary', 'Electric Vehicles', 800000000000, true)
ON CONFLICT (symbol) DO UPDATE SET
    name = EXCLUDED.name,
    updated_at = CURRENT_TIMESTAMP;
" "Inserting Phase 1 symbols"

# 3. Create simple option contracts
execute_sql "
WITH symbols AS (
    SELECT id, symbol FROM underlying_securities
    WHERE symbol IN ('SPY', 'PLTR', 'TSLA')
),
expiries AS (
    SELECT (CURRENT_DATE + (s.days || ' days')::interval)::date as expiry_date
    FROM generate_series(7, 35, 7) s(days)  -- Weekly expiries for 5 weeks
),
strikes AS (
    SELECT
        s.id as underlying_id,
        s.symbol,
        CASE
            WHEN s.symbol = 'SPY' THEN 580 + (st.strike_offset * 2)
            WHEN s.symbol = 'PLTR' THEN 85 + (st.strike_offset * 1)
            WHEN s.symbol = 'TSLA' THEN 250 + (st.strike_offset * 5)
        END as strike_price
    FROM symbols s
    CROSS JOIN generate_series(-15, 15) st(strike_offset)
)
INSERT INTO option_chains
(underlying_id, expiration_date, strike_price, option_type, contract_symbol, ib_con_id, exchange, active)
SELECT
    st.underlying_id,
    exp.expiry_date,
    st.strike_price,
    opt.option_type,
    st.symbol || to_char(exp.expiry_date, 'YYMMDD') ||
        CASE WHEN opt.option_type = 'C' THEN 'C' ELSE 'P' END ||
        LPAD((st.strike_price * 1000)::text, 8, '0') as contract_symbol,
    (600000000 + st.underlying_id * 1000000 +
     EXTRACT(DOW FROM exp.expiry_date)::int * 10000 +
     st.strike_price::int +
     CASE WHEN opt.option_type = 'C' THEN 0 ELSE 10000 END) as ib_con_id,
    'SMART' as exchange,
    true as active
FROM strikes st
CROSS JOIN expiries exp
CROSS JOIN (VALUES ('C'), ('P')) opt(option_type)
WHERE st.strike_price > 0
ON CONFLICT DO NOTHING;
" "Creating option contracts"

# 4. Generate stock data for 30 days
execute_sql "
WITH symbols AS (
    SELECT id, symbol,
        CASE
            WHEN symbol = 'SPY' THEN 580.0
            WHEN symbol = 'PLTR' THEN 85.0
            WHEN symbol = 'TSLA' THEN 250.0
        END as base_price
    FROM underlying_securities
    WHERE symbol IN ('SPY', 'PLTR', 'TSLA')
),
trading_days AS (
    SELECT
        (CURRENT_DATE - INTERVAL '30 days' + (d.day || ' days')::interval)::date as trade_date,
        d.day
    FROM generate_series(0, 29) d(day)
    WHERE EXTRACT(DOW FROM (CURRENT_DATE - INTERVAL '30 days' + (d.day || ' days')::interval)) NOT IN (0, 6)
)
INSERT INTO stock_data_ticks
(time, underlying_id, price, bid_price, ask_price, volume, open_price, high_price, low_price, close_price, tick_type)
SELECT
    (td.trade_date + TIME '16:00:00')::timestamptz,
    s.id,
    s.base_price + (td.day - 15) * 0.3 + (random() - 0.5) * 2 as price,
    s.base_price + (td.day - 15) * 0.3 + (random() - 0.5) * 2 - 0.01 as bid_price,
    s.base_price + (td.day - 15) * 0.3 + (random() - 0.5) * 2 + 0.01 as ask_price,
    CASE
        WHEN s.symbol = 'SPY' THEN (40000000 + random() * 20000000)::int
        WHEN s.symbol = 'PLTR' THEN (25000000 + random() * 15000000)::int
        WHEN s.symbol = 'TSLA' THEN (60000000 + random() * 30000000)::int
    END as volume,
    s.base_price + (td.day - 15) * 0.3 + (random() - 0.5) * 3 as open_price,
    s.base_price + (td.day - 15) * 0.3 + random() * 3 as high_price,
    s.base_price + (td.day - 15) * 0.3 - random() * 3 as low_price,
    s.base_price + (td.day - 15) * 0.3 + (random() - 0.5) * 2 as close_price,
    'EOD' as tick_type
FROM symbols s
CROSS JOIN trading_days td
ON CONFLICT DO NOTHING;
" "Generating stock data"

# 5. Generate simple option prices
execute_sql "
WITH option_data AS (
    SELECT
        oc.id,
        oc.strike_price,
        oc.option_type,
        us.symbol,
        CASE
            WHEN us.symbol = 'SPY' THEN 580.0
            WHEN us.symbol = 'PLTR' THEN 85.0
            WHEN us.symbol = 'TSLA' THEN 250.0
        END as stock_price,
        oc.expiration_date
    FROM option_chains oc
    JOIN underlying_securities us ON oc.underlying_id = us.id
    WHERE us.symbol IN ('SPY', 'PLTR', 'TSLA')
),
trading_days AS (
    SELECT (CURRENT_DATE - INTERVAL '30 days' + (d.day || ' days')::interval)::date as trade_date
    FROM generate_series(0, 29) d(day)
    WHERE EXTRACT(DOW FROM (CURRENT_DATE - INTERVAL '30 days' + (d.day || ' days')::interval)) NOT IN (0, 6)
)
INSERT INTO market_data_ticks
(time, contract_id, bid_price, ask_price, last_price, bid_size, ask_size, volume,
 delta, gamma, theta, vega, implied_volatility, tick_type)
SELECT
    (td.trade_date + TIME '16:00:00')::timestamptz,
    od.id,
    -- Simple intrinsic value calculation
    CASE
        WHEN od.option_type = 'C' THEN
            GREATEST(0.05, GREATEST(0, od.stock_price - od.strike_price) + random() * 2)
        ELSE
            GREATEST(0.05, GREATEST(0, od.strike_price - od.stock_price) + random() * 2)
    END as bid_price,
    CASE
        WHEN od.option_type = 'C' THEN
            GREATEST(0.10, GREATEST(0, od.stock_price - od.strike_price) + random() * 2 + 0.05)
        ELSE
            GREATEST(0.10, GREATEST(0, od.strike_price - od.stock_price) + random() * 2 + 0.05)
    END as ask_price,
    CASE
        WHEN od.option_type = 'C' THEN
            GREATEST(0.075, GREATEST(0, od.stock_price - od.strike_price) + random() * 2 + 0.025)
        ELSE
            GREATEST(0.075, GREATEST(0, od.strike_price - od.stock_price) + random() * 2 + 0.025)
    END as last_price,
    (50 + random() * 200)::int as bid_size,
    (50 + random() * 200)::int as ask_size,
    (random() * 1000)::int as volume,
    CASE WHEN od.option_type = 'C' THEN 0.5 + random() * 0.3 ELSE -0.5 - random() * 0.3 END as delta,
    0.02 + random() * 0.03 as gamma,
    -(0.02 + random() * 0.05) as theta,
    0.1 + random() * 0.3 as vega,
    CASE
        WHEN od.symbol = 'SPY' THEN 0.15 + random() * 0.05
        WHEN od.symbol = 'PLTR' THEN 0.45 + random() * 0.15
        WHEN od.symbol = 'TSLA' THEN 0.55 + random() * 0.20
    END as implied_volatility,
    'EOD' as tick_type
FROM option_data od
CROSS JOIN trading_days td
WHERE od.expiration_date > td.trade_date  -- Only valid options
ON CONFLICT DO NOTHING;
" "Generating option market data"

# 6. Show summary
echo -e "\n${BLUE}Summary:${NC}"
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" << EOF
SELECT 'Symbols' as metric, COUNT(*) as count
FROM underlying_securities WHERE symbol IN ('SPY', 'PLTR', 'TSLA');

SELECT 'Option Contracts' as metric, COUNT(*) as count
FROM option_chains oc
JOIN underlying_securities us ON oc.underlying_id = us.id
WHERE us.symbol IN ('SPY', 'PLTR', 'TSLA');

SELECT 'Stock Data Points' as metric, COUNT(*) as count FROM stock_data_ticks;
SELECT 'Option Data Points' as metric, COUNT(*) as count FROM market_data_ticks;

SELECT 'Date Range' as metric,
    MIN(DATE(time))::text || ' to ' || MAX(DATE(time))::text as count
FROM stock_data_ticks;
EOF

echo -e "\n${GREEN}✅ SPY 30-day data population completed!${NC}"
