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

-- High-frequency VIX data ticks table (will be converted to hypertable)
CREATE TABLE IF NOT EXISTS vix_data_ticks (
    time TIMESTAMP(6) WITH TIME ZONE NOT NULL, -- Microsecond precision
    instrument_id INTEGER NOT NULL REFERENCES vix_instruments(id),
    -- OHLC data
    open_price DECIMAL(8,4),
    high_price DECIMAL(8,4),
    low_price DECIMAL(8,4),
    close_price DECIMAL(8,4),
    -- Bid/Ask data
    bid_price DECIMAL(8,4),
    ask_price DECIMAL(8,4),
    last_price DECIMAL(8,4) NOT NULL,
    -- Volume and size data
    volume BIGINT DEFAULT 0,
    bid_size INTEGER,
    ask_size INTEGER,
    last_size INTEGER,
    -- Calculated fields
    bid_ask_spread DECIMAL(8,4) GENERATED ALWAYS AS (ask_price - bid_price) STORED,
    mid_price DECIMAL(8,4) GENERATED ALWAYS AS (
        CASE
            WHEN bid_price IS NOT NULL AND ask_price IS NOT NULL
            THEN (bid_price + ask_price) / 2
            ELSE last_price
        END
    ) STORED,
    -- VIX specific metrics
    daily_change DECIMAL(8,4), -- Change from previous day close
    daily_change_pct DECIMAL(6,4), -- Percentage change from previous day
    -- Data quality indicators
    tick_type VARCHAR(20) DEFAULT 'REALTIME', -- 'DELAYED', 'REALTIME', 'SNAPSHOT', 'HISTORICAL'
    data_quality_score DECIMAL(3,2) DEFAULT 1.0 CHECK (data_quality_score >= 0 AND data_quality_score <= 1),
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- VIX term structure snapshots (for analyzing contango/backwardation)
CREATE TABLE IF NOT EXISTS vix_term_structure (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP(6) WITH TIME ZONE NOT NULL,
    -- VIX levels at snapshot time
    vix_1d DECIMAL(8,4),
    vix_9d DECIMAL(8,4),
    vix_30d DECIMAL(8,4), -- Main VIX
    vix_3m DECIMAL(8,4),
    vix_6m DECIMAL(8,4),
    -- Calculated term structure metrics
    slope_9d_30d DECIMAL(8,4) GENERATED ALWAYS AS (
        CASE WHEN vix_9d IS NOT NULL AND vix_30d IS NOT NULL
        THEN (vix_30d - vix_9d) / vix_9d
        ELSE NULL END
    ) STORED,
    slope_30d_3m DECIMAL(8,4) GENERATED ALWAYS AS (
        CASE WHEN vix_30d IS NOT NULL AND vix_3m IS NOT NULL
        THEN (vix_3m - vix_30d) / vix_30d
        ELSE NULL END
    ) STORED,
    slope_3m_6m DECIMAL(8,4) GENERATED ALWAYS AS (
        CASE WHEN vix_3m IS NOT NULL AND vix_6m IS NOT NULL
        THEN (vix_6m - vix_3m) / vix_3m
        ELSE NULL END
    ) STORED,
    -- Market structure indicators
    term_structure_type VARCHAR(20) GENERATED ALWAYS AS (
        CASE
            WHEN vix_30d IS NOT NULL AND vix_3m IS NOT NULL THEN
                CASE WHEN vix_3m > vix_30d THEN 'CONTANGO'
                     WHEN vix_3m < vix_30d THEN 'BACKWARDATION'
                     ELSE 'FLAT'
                END
            ELSE NULL
        END
    ) STORED,
    volatility_regime VARCHAR(20) GENERATED ALWAYS AS (
        CASE
            WHEN vix_30d IS NOT NULL THEN
                CASE WHEN vix_30d < 15 THEN 'LOW'
                     WHEN vix_30d BETWEEN 15 AND 25 THEN 'MEDIUM'
                     WHEN vix_30d BETWEEN 25 AND 40 THEN 'HIGH'
                     ELSE 'EXTREME'
                END
            ELSE NULL
        END
    ) STORED,
    -- Additional metrics
    vix_spike_indicator BOOLEAN GENERATED ALWAYS AS (
        vix_30d IS NOT NULL AND vix_30d > 30
    ) STORED,
    term_structure_inversion BOOLEAN GENERATED ALWAYS AS (
        vix_9d IS NOT NULL AND vix_30d IS NOT NULL AND vix_9d > vix_30d
    ) STORED,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- VIX correlation with arbitrage opportunities
CREATE TABLE IF NOT EXISTS vix_arbitrage_correlation (
    id BIGSERIAL PRIMARY KEY,
    arbitrage_opportunity_id BIGINT NOT NULL REFERENCES arbitrage_opportunities(id),
    vix_snapshot_id BIGINT NOT NULL REFERENCES vix_term_structure(id),
    -- Timing relationship
    time_difference_ms INTEGER, -- Milliseconds between VIX snapshot and arbitrage discovery
    -- VIX context at time of arbitrage opportunity
    vix_level DECIMAL(8,4) NOT NULL, -- Main VIX level
    vix_regime VARCHAR(20) NOT NULL, -- LOW, MEDIUM, HIGH, EXTREME
    term_structure_type VARCHAR(20), -- CONTANGO, BACKWARDATION, FLAT
    vix_spike_active BOOLEAN DEFAULT false,
    -- Correlation analysis fields
    arbitrage_success BOOLEAN, -- Whether the arbitrage was successfully executed
    execution_speed_ms INTEGER, -- How quickly the opportunity was captured
    profit_realized DECIMAL(10,4), -- Actual profit if executed
    -- VIX change during arbitrage window
    vix_change_during_execution DECIMAL(8,4), -- VIX change during execution
    volatility_impact_score DECIMAL(4,3), -- Calculated impact of volatility on execution
    -- Statistical fields for analysis
    correlation_weight DECIMAL(6,4) DEFAULT 1.0, -- Weight for correlation calculations
    outlier_flag BOOLEAN DEFAULT false, -- Flag for statistical outliers
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- VIX historical statistics for backtesting
CREATE TABLE IF NOT EXISTS vix_historical_stats (
    id SERIAL PRIMARY KEY,
    calculation_date DATE NOT NULL UNIQUE,
    -- Rolling statistics (multiple periods)
    vix_1d_avg DECIMAL(8,4), vix_1d_std DECIMAL(8,4),
    vix_5d_avg DECIMAL(8,4), vix_5d_std DECIMAL(8,4),
    vix_10d_avg DECIMAL(8,4), vix_10d_std DECIMAL(8,4),
    vix_30d_avg DECIMAL(8,4), vix_30d_std DECIMAL(8,4),
    vix_90d_avg DECIMAL(8,4), vix_90d_std DECIMAL(8,4),
    -- Extreme values
    vix_daily_high DECIMAL(8,4),
    vix_daily_low DECIMAL(8,4),
    vix_intraday_range DECIMAL(8,4),
    -- Term structure statistics
    avg_slope_30d_3m DECIMAL(8,4),
    contango_days_pct DECIMAL(6,4), -- Percentage of days in contango
    backwardation_days_pct DECIMAL(6,4), -- Percentage of days in backwardation
    -- Volatility clustering metrics
    realized_vol_30d DECIMAL(8,4), -- 30-day realized volatility
    vol_of_vol DECIMAL(8,4), -- Volatility of volatility
    -- Market stress indicators
    spike_days_count INTEGER, -- Days with VIX > 30
    extreme_days_count INTEGER, -- Days with VIX > 40
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create update trigger for vix_instruments
CREATE TRIGGER update_vix_instruments_updated_at BEFORE UPDATE ON vix_instruments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Indexes for efficient VIX data queries
CREATE INDEX IF NOT EXISTS idx_vix_data_ticks_instrument_time
    ON vix_data_ticks (instrument_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_vix_data_ticks_time_bucket
    ON vix_data_ticks (time DESC) WHERE tick_type = 'REALTIME';

CREATE INDEX IF NOT EXISTS idx_vix_term_structure_timestamp
    ON vix_term_structure (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_vix_term_structure_regime
    ON vix_term_structure (volatility_regime, term_structure_type) WHERE vix_30d IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_vix_correlation_arbitrage
    ON vix_arbitrage_correlation (arbitrage_opportunity_id);

CREATE INDEX IF NOT EXISTS idx_vix_correlation_analysis
    ON vix_arbitrage_correlation (vix_regime, arbitrage_success, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_vix_historical_date
    ON vix_historical_stats (calculation_date DESC);

-- Constraints for data quality
ALTER TABLE vix_data_ticks ADD CONSTRAINT chk_vix_price_positive
    CHECK (last_price >= 0 AND (bid_price IS NULL OR bid_price >= 0) AND (ask_price IS NULL OR ask_price >= 0));

ALTER TABLE vix_data_ticks ADD CONSTRAINT chk_vix_bid_ask_order
    CHECK (bid_price IS NULL OR ask_price IS NULL OR bid_price <= ask_price);

ALTER TABLE vix_term_structure ADD CONSTRAINT chk_vix_levels_positive
    CHECK (
        (vix_1d IS NULL OR vix_1d >= 0) AND
        (vix_9d IS NULL OR vix_9d >= 0) AND
        (vix_30d IS NULL OR vix_30d >= 0) AND
        (vix_3m IS NULL OR vix_3m >= 0) AND
        (vix_6m IS NULL OR vix_6m >= 0)
    );

-- Function to get current VIX term structure
CREATE OR REPLACE FUNCTION get_latest_vix_structure()
RETURNS TABLE (
    vix_1d DECIMAL,
    vix_9d DECIMAL,
    vix_30d DECIMAL,
    vix_3m DECIMAL,
    vix_6m DECIMAL,
    regime VARCHAR,
    structure_type VARCHAR,
    snapshot_timestamp TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        vts.vix_1d,
        vts.vix_9d,
        vts.vix_30d,
        vts.vix_3m,
        vts.vix_6m,
        vts.volatility_regime,
        vts.term_structure_type,
        vts.timestamp
    FROM vix_term_structure vts
    ORDER BY vts.timestamp DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate VIX correlation with arbitrage success
CREATE OR REPLACE FUNCTION calculate_vix_arbitrage_correlation(
    p_start_date TIMESTAMPTZ DEFAULT NOW() - INTERVAL '30 days',
    p_end_date TIMESTAMPTZ DEFAULT NOW()
)
RETURNS TABLE (
    vix_regime VARCHAR,
    total_opportunities BIGINT,
    successful_opportunities BIGINT,
    success_rate DECIMAL,
    avg_profit DECIMAL,
    avg_vix_level DECIMAL,
    correlation_strength DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        vac.vix_regime,
        COUNT(*) as total_opportunities,
        COUNT(*) FILTER (WHERE vac.arbitrage_success = true) as successful_opportunities,
        (COUNT(*) FILTER (WHERE vac.arbitrage_success = true) * 100.0 / COUNT(*))::DECIMAL(5,2) as success_rate,
        AVG(vac.profit_realized) FILTER (WHERE vac.arbitrage_success = true)::DECIMAL(10,4) as avg_profit,
        AVG(vac.vix_level)::DECIMAL(8,4) as avg_vix_level,
        CORR(vac.vix_level, CASE WHEN vac.arbitrage_success THEN 1 ELSE 0 END)::DECIMAL(6,4) as correlation_strength
    FROM vix_arbitrage_correlation vac
    WHERE vac.created_at >= p_start_date
        AND vac.created_at <= p_end_date
        AND vac.outlier_flag = false
    GROUP BY vac.vix_regime
    ORDER BY total_opportunities DESC;
END;
$$ LANGUAGE plpgsql;
