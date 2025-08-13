-- 05_vix_data_tables.sql
-- VIX data tables for volatility correlation analysis with arbitrage opportunities

-- VIX instruments metadata table
CREATE TABLE IF NOT EXISTS vix_instruments (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL UNIQUE, -- VIX, VIX1D, VIX9D, VIX3M, VIX6M
    name VARCHAR(100) NOT NULL,
    description TEXT,
    maturity_days INTEGER, -- Days to maturity for the volatility measurement
    ib_con_id BIGINT UNIQUE, -- Interactive Brokers contract ID
    exchange VARCHAR(10) DEFAULT 'CBOE',
    currency VARCHAR(3) DEFAULT 'USD',
    multiplier INTEGER DEFAULT 100,
    tick_size DECIMAL(8,6) DEFAULT 0.01,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Insert VIX instrument definitions
INSERT INTO vix_instruments (symbol, name, description, maturity_days) VALUES
('VIX', 'CBOE Volatility Index', 'Main volatility index measuring 30-day implied volatility of S&P 500', 30),
('VIX1D', '1-Day Volatility Index', '1-day implied volatility index', 1),
('VIX9D', '9-Day Volatility Index', '9-day implied volatility index', 9),
('VIX3M', '3-Month Volatility Index', '3-month (approximately 90-day) implied volatility index', 90),
('VIX6M', '6-Month Volatility Index', '6-month (approximately 180-day) implied volatility index', 180)
ON CONFLICT (symbol) DO UPDATE SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    maturity_days = EXCLUDED.maturity_days,
    updated_at = CURRENT_TIMESTAMP;

-- REMOVED: vix_data_ticks table - unused (empty, 0 rows)

-- REMOVED: vix_term_structure table - unused (empty, 0 rows)

-- REMOVED: vix_arbitrage_correlation table - unused

-- REMOVED: vix_historical_stats table - unused

-- Create update trigger for vix_instruments
CREATE TRIGGER update_vix_instruments_updated_at BEFORE UPDATE ON vix_instruments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Indexes for efficient VIX data queries
-- REMOVED: Indexes for unused VIX data and term structure tables

-- REMOVED: Indexes for unused VIX correlation and historical stats tables

-- Constraints for data quality
-- REMOVED: Constraints for unused VIX data and term structure tables

-- REMOVED: Function for unused VIX term structure table

-- REMOVED: Function for unused VIX correlation analysis
