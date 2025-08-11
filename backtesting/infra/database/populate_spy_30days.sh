#!/bin/bash
# populate_spy_30days.sh
# Script to populate the options arbitrage database with 30 days of SPY sample data
# Focused on Phase 1 implementation with realistic option chains

set -e

# Database connection parameters - Updated for Podman on port 5433
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

echo -e "${BLUE}SPY 30-Day Historical Data Population Script${NC}"
echo "=============================================="
echo "Phase 1: Focus on SPY, PLTR, TSLA only"
echo ""

# Function to execute SQL with error handling
execute_sql() {
    local sql="$1"
    local description="$2"

    echo -e "${YELLOW}$description...${NC}"

    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "$sql" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $description completed${NC}"
    else
        echo -e "${RED}✗ $description failed${NC}"
        # Show the actual error for debugging
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "$sql"
        return 1
    fi
}

# Check database connection
echo -e "${YELLOW}Testing database connection...${NC}"
if ! PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" > /dev/null 2>&1; then
    echo -e "${RED}✗ Cannot connect to database on port $DB_PORT${NC}"
    echo "Please ensure Podman container is running:"
    echo "  podman ps | grep timescaledb"
    exit 1
fi
echo -e "${GREEN}✓ Database connection successful on port $DB_PORT${NC}"

# 0. Clear existing data (optional - comment out if you want to keep existing data)
echo -e "\n${BLUE}0. Clearing existing sample data...${NC}"
execute_sql "
-- Clear in correct order to respect foreign keys
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

-- Keep only Phase 1 symbols
DELETE FROM underlying_securities
WHERE symbol NOT IN ('SPY', 'PLTR', 'TSLA');
" "Clearing existing data"

# 1. Populate Phase 1 underlying securities
echo -e "\n${BLUE}1. Populating Phase 1 underlying securities...${NC}"

execute_sql "
INSERT INTO underlying_securities (symbol, name, sector, industry, market_cap, active) VALUES
('SPY', 'SPDR S&P 500 ETF Trust', 'ETF', 'Broad Market ETF', 550000000000, true),
('PLTR', 'Palantir Technologies Inc.', 'Technology', 'Software - Infrastructure', 150000000000, true),
('TSLA', 'Tesla Inc.', 'Consumer Discretionary', 'Electric Vehicles', 800000000000, true)
ON CONFLICT (symbol) DO UPDATE SET
    name = EXCLUDED.name,
    sector = EXCLUDED.sector,
    industry = EXCLUDED.industry,
    market_cap = EXCLUDED.market_cap,
    updated_at = CURRENT_TIMESTAMP;
" "Inserting Phase 1 underlying securities"

# 2. Populate realistic option chains for SPY with proper expiries
echo -e "\n${BLUE}2. Populating SPY option chains (30 days)...${NC}"

execute_sql "
WITH underlying_data AS (
    SELECT id, symbol FROM underlying_securities
    WHERE symbol IN ('SPY', 'PLTR', 'TSLA')
),
-- Generate realistic expiry dates (weekly and monthly options)
expiry_dates AS (
    SELECT DISTINCT expiry_date FROM (
        -- Weekly expiries (every Friday for next 5 weeks)
        SELECT (CURRENT_DATE + (s.n || ' days')::interval)::date as expiry_date
        FROM generate_series(0, 35) s(n)
        WHERE EXTRACT(DOW FROM (CURRENT_DATE + (s.n || ' days')::interval)) = 5  -- Fridays
        AND (CURRENT_DATE + (s.n || ' days')::interval)::date > CURRENT_DATE
        AND (CURRENT_DATE + (s.n || ' days')::interval)::date <= CURRENT_DATE + INTERVAL '30 days'

        UNION

        -- Monthly expiry (3rd Friday of current and next month)
        SELECT
            DATE_TRUNC('month', CURRENT_DATE)::date + 14 +
            (5 - EXTRACT(DOW FROM DATE_TRUNC('month', CURRENT_DATE)::date + 14))::int as expiry_date
        WHERE
            DATE_TRUNC('month', CURRENT_DATE)::date + 14 +
            (5 - EXTRACT(DOW FROM DATE_TRUNC('month', CURRENT_DATE)::date + 14))::int > CURRENT_DATE

        UNION

        SELECT
            DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month')::date + 14 +
            (5 - EXTRACT(DOW FROM DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month')::date + 14))::int
    ) dates
    WHERE expiry_date >= CURRENT_DATE
    AND expiry_date <= CURRENT_DATE + INTERVAL '35 days'
    ORDER BY expiry_date
),
-- Generate realistic strikes based on current prices
strikes AS (
    SELECT
        CASE
            -- SPY: Current price ~580, $1 increments
            WHEN ud.symbol = 'SPY' THEN 580 + (s.strike_offset * 1)
            -- PLTR: Current price ~85, $1 increments
            WHEN ud.symbol = 'PLTR' THEN 85 + (s.strike_offset * 1)
            -- TSLA: Current price ~250, $5 increments
            WHEN ud.symbol = 'TSLA' THEN 250 + (s.strike_offset * 5)
        END as strike_price,
        ud.id as underlying_id,
        ud.symbol
    FROM underlying_data ud
    CROSS JOIN generate_series(-30, 30) s(strike_offset)  -- ±30 strikes for good coverage
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
    -- Generate unique IB contract IDs
    (600000000 +
     (s.underlying_id * 10000000) +
     (EXTRACT(EPOCH FROM ed.expiry_date)::bigint % 1000000) +
     (s.strike_price::int * 100) +
     CASE WHEN ot.option_type = 'C' THEN 0 ELSE 50 END) as ib_con_id,
    'SMART' as exchange,
    true as active
FROM strikes s
CROSS JOIN expiry_dates ed
CROSS JOIN (VALUES ('C'), ('P')) ot(option_type)
WHERE s.strike_price > 0  -- Ensure positive strikes
ON CONFLICT (underlying_id, expiration_date, strike_price, option_type)
DO UPDATE SET
    ib_con_id = EXCLUDED.ib_con_id,
    updated_at = CURRENT_TIMESTAMP;
" "Generating option chains for Phase 1 symbols"

# 3. Populate 30 days of historical market data (end-of-day)
echo -e "\n${BLUE}3. Populating 30 days of historical option data...${NC}"

execute_sql "
WITH option_data AS (
    SELECT
        oc.id as contract_id,
        oc.strike_price,
        oc.option_type,
        us.symbol,
        oc.expiration_date,
        CASE
            WHEN us.symbol = 'SPY' THEN 580
            WHEN us.symbol = 'PLTR' THEN 85
            WHEN us.symbol = 'TSLA' THEN 250
        END as stock_price
    FROM option_chains oc
    JOIN underlying_securities us ON oc.underlying_id = us.id
    WHERE us.symbol IN ('SPY', 'PLTR', 'TSLA')
    AND oc.active = true
),
-- Generate 30 days of end-of-day timestamps (4:00 PM ET)
time_series AS (
    SELECT
        ((CURRENT_DATE - INTERVAL '30 days' + (s.day || ' days')::interval) + TIME '16:00:00')::timestamptz as timestamp,
        s.day as day_offset
    FROM generate_series(0, 29) s(day)
    WHERE EXTRACT(DOW FROM (CURRENT_DATE - INTERVAL '30 days' + (s.day || ' days')::interval)) NOT IN (0, 6)  -- Exclude weekends
)
INSERT INTO market_data_ticks
(time, contract_id, bid_price, ask_price, last_price, bid_size, ask_size, volume,
 delta, gamma, theta, vega, implied_volatility, tick_type)
SELECT
    ts.timestamp,
    od.contract_id,
    -- Calculate realistic option prices using simplified Black-Scholes approximation
    CASE
        WHEN od.option_type = 'C' THEN
            GREATEST(0.01,
                -- Intrinsic value + time value
                GREATEST(0, od.stock_price - od.strike_price) +
                -- Time value based on days to expiry
                CASE
                    WHEN od.expiration_date > (CURRENT_DATE - INTERVAL '30 days' + (ts.day_offset || ' days')::interval)::date
                    THEN SQRT((od.expiration_date - (CURRENT_DATE - INTERVAL '30 days' + (ts.day_offset || ' days')::interval)::date)::int / 30.0) *
                         od.stock_price * 0.02 * (1 + ABS(od.stock_price - od.strike_price) / od.stock_price)
                    ELSE 0
                END +
                -- Add some random variation
                (random() - 0.5) * 0.10
            )
        ELSE -- Put option
            GREATEST(0.01,
                -- Intrinsic value + time value
                GREATEST(0, od.strike_price - od.stock_price) +
                -- Time value
                CASE
                    WHEN od.expiration_date > (CURRENT_DATE - INTERVAL '30 days' + (ts.day_offset || ' days')::interval)::date
                    THEN SQRT((od.expiration_date - (CURRENT_DATE - INTERVAL '30 days' + (ts.day_offset || ' days')::interval)::date)::int / 30.0) *
                         od.stock_price * 0.02 * (1 + ABS(od.strike_price - od.stock_price) / od.stock_price)
                    ELSE 0
                END +
                -- Add some random variation
                (random() - 0.5) * 0.10
            )
    END as bid_price,
    -- Ask price (bid + realistic spread)
    CASE
        WHEN od.option_type = 'C' THEN
            GREATEST(0.02,
                GREATEST(0, od.stock_price - od.strike_price) +
                CASE
                    WHEN od.expiration_date > (CURRENT_DATE - INTERVAL '30 days' + (ts.day_offset || ' days')::interval)::date
                    THEN SQRT((od.expiration_date - (CURRENT_DATE - INTERVAL '30 days' + (ts.day_offset || ' days')::interval)::date)::int / 30.0) *
                         od.stock_price * 0.02 * (1 + ABS(od.stock_price - od.strike_price) / od.stock_price)
                    ELSE 0
                END +
                (random() - 0.5) * 0.10 +
                -- Spread based on moneyness
                CASE
                    WHEN ABS(od.stock_price - od.strike_price) < 5 THEN 0.05  -- ATM
                    WHEN ABS(od.stock_price - od.strike_price) < 20 THEN 0.10  -- Near money
                    ELSE 0.20  -- Far OTM
                END
            )
        ELSE
            GREATEST(0.02,
                GREATEST(0, od.strike_price - od.stock_price) +
                CASE
                    WHEN od.expiration_date > (CURRENT_DATE - INTERVAL '30 days' + (ts.day_offset || ' days')::interval)::date
                    THEN SQRT(EXTRACT(EPOCH FROM (od.expiration_date - (CURRENT_DATE - INTERVAL '30 days' + (ts.day_offset || ' days')::interval)::date)) / 86400.0) *
                         od.stock_price * 0.02 * (1 + ABS(od.strike_price - od.stock_price) / od.stock_price)
                    ELSE 0
                END +
                (random() - 0.5) * 0.10 +
                CASE
                    WHEN ABS(od.stock_price - od.strike_price) < 5 THEN 0.05
                    WHEN ABS(od.stock_price - od.strike_price) < 20 THEN 0.10
                    ELSE 0.20
                END
            )
    END as ask_price,
    -- Last price (midpoint)
    CASE
        WHEN od.option_type = 'C' THEN
            GREATEST(0.015,
                GREATEST(0, od.stock_price - od.strike_price) +
                CASE
                    WHEN od.expiration_date > (CURRENT_DATE - INTERVAL '30 days' + (ts.day_offset || ' days')::interval)::date
                    THEN SQRT((od.expiration_date - (CURRENT_DATE - INTERVAL '30 days' + (ts.day_offset || ' days')::interval)::date)::int / 30.0) *
                         od.stock_price * 0.02 * (1 + ABS(od.stock_price - od.strike_price) / od.stock_price)
                    ELSE 0
                END +
                (random() - 0.5) * 0.10 + 0.025
            )
        ELSE
            GREATEST(0.015,
                GREATEST(0, od.strike_price - od.stock_price) +
                CASE
                    WHEN od.expiration_date > (CURRENT_DATE - INTERVAL '30 days' + (ts.day_offset || ' days')::interval)::date
                    THEN SQRT(EXTRACT(EPOCH FROM (od.expiration_date - (CURRENT_DATE - INTERVAL '30 days' + (ts.day_offset || ' days')::interval)::date)) / 86400.0) *
                         od.stock_price * 0.02 * (1 + ABS(od.strike_price - od.stock_price) / od.stock_price)
                    ELSE 0
                END +
                (random() - 0.5) * 0.10 + 0.025
            )
    END as last_price,
    -- Realistic sizes based on moneyness
    CASE
        WHEN ABS(od.stock_price - od.strike_price) < 5 THEN (100 + random() * 500)::int  -- High volume ATM
        WHEN ABS(od.stock_price - od.strike_price) < 20 THEN (50 + random() * 200)::int
        ELSE (10 + random() * 50)::int  -- Low volume far OTM
    END as bid_size,
    CASE
        WHEN ABS(od.stock_price - od.strike_price) < 5 THEN (100 + random() * 500)::int
        WHEN ABS(od.stock_price - od.strike_price) < 20 THEN (50 + random() * 200)::int
        ELSE (10 + random() * 50)::int
    END as ask_size,
    -- Volume based on moneyness
    CASE
        WHEN ABS(od.stock_price - od.strike_price) < 5 THEN (1000 + random() * 10000)::int
        WHEN ABS(od.stock_price - od.strike_price) < 20 THEN (100 + random() * 1000)::int
        ELSE (0 + random() * 100)::int
    END as volume,
    -- Realistic Greeks
    CASE
        WHEN od.option_type = 'C' THEN
            -- Delta for calls
            CASE
                WHEN od.stock_price - od.strike_price > 10 THEN 0.8 + random() * 0.15  -- ITM
                WHEN od.stock_price - od.strike_price > -10 THEN 0.5 + random() * 0.2  -- ATM
                ELSE 0.2 + random() * 0.2  -- OTM
            END
        ELSE
            -- Delta for puts
            CASE
                WHEN od.strike_price - od.stock_price > 10 THEN -0.8 - random() * 0.15  -- ITM
                WHEN od.strike_price - od.stock_price > -10 THEN -0.5 - random() * 0.2  -- ATM
                ELSE -0.2 - random() * 0.2  -- OTM
            END
    END as delta,
    GREATEST(0.001, 0.05 - ABS(od.stock_price - od.strike_price) * 0.001 + random() * 0.01) as gamma,
    -(0.05 + random() * 0.5) as theta,  -- Always negative
    GREATEST(0.01, 0.5 - ABS(od.stock_price - od.strike_price) * 0.01 + random() * 0.2) as vega,
    -- Implied volatility based on symbol
    CASE
        WHEN us.symbol = 'SPY' THEN 0.12 + random() * 0.08  -- 12-20% IV
        WHEN us.symbol = 'PLTR' THEN 0.35 + random() * 0.25  -- 35-60% IV
        WHEN us.symbol = 'TSLA' THEN 0.40 + random() * 0.30  -- 40-70% IV
    END as implied_volatility,
    'EOD' as tick_type
FROM option_data od
CROSS JOIN time_series ts
JOIN underlying_securities us ON us.symbol = od.symbol
WHERE od.expiration_date >= (CURRENT_DATE - INTERVAL '30 days' + (ts.day_offset || ' days')::interval)::date  -- Only valid options
ON CONFLICT DO NOTHING;
" "Generating 30 days of option market data"

# 4. Populate 30 days of stock data
echo -e "\n${BLUE}4. Populating 30 days of stock data...${NC}"

execute_sql "
WITH stock_prices AS (
    SELECT
        us.id as underlying_id,
        us.symbol,
        CASE
            WHEN us.symbol = 'SPY' THEN 580
            WHEN us.symbol = 'PLTR' THEN 85
            WHEN us.symbol = 'TSLA' THEN 250
        END as base_price
    FROM underlying_securities us
    WHERE us.symbol IN ('SPY', 'PLTR', 'TSLA')
),
-- Generate 30 days of end-of-day timestamps
time_series AS (
    SELECT
        ((CURRENT_DATE - INTERVAL '30 days' + (s.day || ' days')::interval) + TIME '16:00:00')::timestamptz as timestamp,
        s.day as day_offset
    FROM generate_series(0, 29) s(day)
    WHERE EXTRACT(DOW FROM (CURRENT_DATE - INTERVAL '30 days' + (s.day || ' days')::interval)) NOT IN (0, 6)  -- Exclude weekends
)
INSERT INTO stock_data_ticks
(time, underlying_id, price, bid_price, ask_price, volume, open_price, high_price, low_price, close_price, tick_type)
SELECT
    ts.timestamp,
    sp.underlying_id,
    -- Add realistic daily price movement
    sp.base_price + (ts.day_offset - 15) * 0.5 + (random() - 0.5) * 2 as price,
    sp.base_price + (ts.day_offset - 15) * 0.5 + (random() - 0.5) * 2 - 0.01 as bid_price,
    sp.base_price + (ts.day_offset - 15) * 0.5 + (random() - 0.5) * 2 + 0.01 as ask_price,
    CASE
        WHEN sp.symbol = 'SPY' THEN (50000000 + random() * 30000000)::int
        WHEN sp.symbol = 'PLTR' THEN (30000000 + random() * 20000000)::int
        WHEN sp.symbol = 'TSLA' THEN (80000000 + random() * 40000000)::int
    END as volume,
    sp.base_price + (ts.day_offset - 15) * 0.5 + (random() - 0.5) * 3 as open_price,
    sp.base_price + (ts.day_offset - 15) * 0.5 + (random() * 4) as high_price,
    sp.base_price + (ts.day_offset - 15) * 0.5 - (random() * 4) as low_price,
    sp.base_price + (ts.day_offset - 15) * 0.5 + (random() - 0.5) * 2 as close_price,
    'EOD' as tick_type
FROM stock_prices sp
CROSS JOIN time_series ts
ON CONFLICT DO NOTHING;
" "Generating 30 days of stock data"

# 5. Show summary statistics
echo -e "\n${BLUE}Database Population Summary:${NC}"
echo "============================="

# Use a simpler approach for statistics
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" << EOF
SELECT 'Underlying Securities' as metric, COUNT(*) as count FROM underlying_securities WHERE symbol IN ('SPY', 'PLTR', 'TSLA');
SELECT 'SPY Option Contracts' as metric, COUNT(*) as count FROM option_chains oc JOIN underlying_securities us ON oc.underlying_id = us.id WHERE us.symbol = 'SPY';
SELECT 'PLTR Option Contracts' as metric, COUNT(*) as count FROM option_chains oc JOIN underlying_securities us ON oc.underlying_id = us.id WHERE us.symbol = 'PLTR';
SELECT 'TSLA Option Contracts' as metric, COUNT(*) as count FROM option_chains oc JOIN underlying_securities us ON oc.underlying_id = us.id WHERE us.symbol = 'TSLA';
SELECT 'Market Data Points' as metric, COUNT(*) as count FROM market_data_ticks;
SELECT 'Stock Data Points' as metric, COUNT(*) as count FROM stock_data_ticks;
SELECT 'Date Range' as metric, MIN(DATE(time))::text || ' to ' || MAX(DATE(time))::text as count FROM market_data_ticks;
EOF

echo -e "\n${GREEN}✅ SPY 30-day data population completed successfully!${NC}"
echo -e "\n${BLUE}Next steps:${NC}"
echo "• Run daily collector: python backtesting/infra/data_collection/daily_collector.py"
echo "• Check data quality: SELECT * FROM check_collection_health('SPY');"
echo "• View option chains: SELECT * FROM v_active_option_contracts WHERE symbol = 'SPY' LIMIT 10;"

echo -e "\n${YELLOW}Useful verification queries:${NC}"
echo "-- Check SPY option expiries:"
echo "SELECT DISTINCT expiration_date FROM option_chains oc"
echo "JOIN underlying_securities us ON oc.underlying_id = us.id"
echo "WHERE us.symbol = 'SPY' ORDER BY expiration_date;"
echo ""
echo "-- Check data coverage by date:"
echo "SELECT DATE(time) as date, COUNT(*) as data_points"
echo "FROM market_data_ticks"
echo "GROUP BY DATE(time) ORDER BY date;"
