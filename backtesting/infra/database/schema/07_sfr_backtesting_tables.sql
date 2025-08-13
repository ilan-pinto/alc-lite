-- 07_sfr_backtesting_tables.sql
-- SFR (Synthetic Free Risk) backtesting specific tables
-- Extends existing schema with comprehensive SFR backtesting capabilities

-- SFR backtest runs metadata (extends generic backtest_runs table)
CREATE TABLE IF NOT EXISTS sfr_backtest_runs (
    id SERIAL PRIMARY KEY,
    backtest_run_id INTEGER NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
    -- SFR specific parameters
    profit_target DECIMAL(8,4) NOT NULL, -- Minimum profit target percentage
    cost_limit DECIMAL(10,2) NOT NULL, -- Maximum cost limit in dollars
    volume_limit INTEGER DEFAULT 100, -- Minimum option volume threshold
    quantity INTEGER DEFAULT 1, -- Number of contracts per trade
    -- Strike selection parameters
    call_strike_range_days INTEGER DEFAULT 25, -- Strike range below stock price
    put_strike_range_days INTEGER DEFAULT 25, -- Strike range below stock price
    expiry_min_days INTEGER DEFAULT 19, -- Minimum days to expiration
    expiry_max_days INTEGER DEFAULT 45, -- Maximum days to expiration
    max_strike_combinations INTEGER DEFAULT 4, -- Max strike pairs to test per expiry
    max_expiry_options INTEGER DEFAULT 8, -- Max expiries to test per symbol
    -- Risk management parameters
    max_bid_ask_spread_call DECIMAL(8,4) DEFAULT 20.00, -- Maximum call bid-ask spread
    max_bid_ask_spread_put DECIMAL(8,4) DEFAULT 20.00, -- Maximum put bid-ask spread
    combo_buffer_percent DECIMAL(6,4) DEFAULT 0.00, -- Buffer for combo limit price
    data_timeout_seconds INTEGER DEFAULT 45, -- Market data collection timeout
    -- Execution simulation parameters
    slippage_model VARCHAR(20) DEFAULT 'LINEAR' CHECK (slippage_model IN ('NONE', 'LINEAR', 'SQUARE_ROOT', 'IMPACT')),
    base_slippage_bps INTEGER DEFAULT 2, -- Base slippage in basis points
    liquidity_penalty_factor DECIMAL(6,4) DEFAULT 1.0, -- Penalty for low liquidity
    commission_per_contract DECIMAL(8,4) DEFAULT 1.00, -- Commission cost per contract
    -- REMOVED: Market environment filters (VIX-related)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- SFR opportunities discovered during backtesting (hypertable candidate)
CREATE TABLE IF NOT EXISTS sfr_opportunities (
    id BIGSERIAL PRIMARY KEY,
    backtest_run_id INTEGER NOT NULL REFERENCES sfr_backtest_runs(id) ON DELETE CASCADE,
    underlying_id INTEGER NOT NULL REFERENCES underlying_securities(id),
    discovery_timestamp TIMESTAMP(6) WITH TIME ZONE NOT NULL,
    -- Option contract details
    expiry_date DATE NOT NULL,
    call_strike DECIMAL(10,2) NOT NULL,
    put_strike DECIMAL(10,2) NOT NULL,
    call_contract_id INTEGER REFERENCES option_chains(id),
    put_contract_id INTEGER REFERENCES option_chains(id),
    -- Market prices at discovery
    stock_price DECIMAL(10,4) NOT NULL,
    call_bid DECIMAL(10,4),
    call_ask DECIMAL(10,4),
    call_mid DECIMAL(10,4) GENERATED ALWAYS AS ((call_bid + call_ask) / 2) STORED,
    call_last DECIMAL(10,4),
    call_volume INTEGER DEFAULT 0,
    put_bid DECIMAL(10,4),
    put_ask DECIMAL(10,4),
    put_mid DECIMAL(10,4) GENERATED ALWAYS AS ((put_bid + put_ask) / 2) STORED,
    put_last DECIMAL(10,4),
    put_volume INTEGER DEFAULT 0,
    -- Calculated SFR metrics
    net_credit DECIMAL(10,4) NOT NULL, -- call_price - put_price
    spread DECIMAL(10,4) GENERATED ALWAYS AS (stock_price - put_strike) STORED,
    min_profit DECIMAL(10,4) NOT NULL, -- net_credit - spread
    max_profit DECIMAL(10,4) NOT NULL, -- (call_strike - put_strike) + net_credit
    min_roi DECIMAL(8,4) NOT NULL, -- (min_profit / (stock_price + net_credit)) * 100
    max_roi DECIMAL(8,4) GENERATED ALWAYS AS ((max_profit / (stock_price + net_credit)) * 100) STORED,
    -- Risk metrics
    call_moneyness DECIMAL(8,6) GENERATED ALWAYS AS (call_strike / stock_price) STORED,
    put_moneyness DECIMAL(8,6) GENERATED ALWAYS AS (put_strike / stock_price) STORED,
    call_bid_ask_spread DECIMAL(8,4) GENERATED ALWAYS AS (call_ask - call_bid) STORED,
    put_bid_ask_spread DECIMAL(8,4) GENERATED ALWAYS AS (put_ask - put_bid) STORED,
    days_to_expiry INTEGER GENERATED ALWAYS AS (expiry_date - discovery_timestamp::date) STORED,
    -- Greeks at discovery (if available)
    call_delta DECIMAL(8,6),
    call_gamma DECIMAL(8,6),
    call_theta DECIMAL(8,6),
    call_vega DECIMAL(8,6),
    call_iv DECIMAL(8,6), -- Implied volatility
    put_delta DECIMAL(8,6),
    put_gamma DECIMAL(8,6),
    put_theta DECIMAL(8,6),
    put_vega DECIMAL(8,6),
    put_iv DECIMAL(8,6), -- Implied volatility
    -- Opportunity classification
    opportunity_quality VARCHAR(20) DEFAULT 'UNKNOWN' CHECK (opportunity_quality IN ('EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'UNKNOWN')),
    execution_difficulty VARCHAR(20) DEFAULT 'UNKNOWN' CHECK (execution_difficulty IN ('EASY', 'MODERATE', 'DIFFICULT', 'VERY_DIFFICULT', 'UNKNOWN')),
    liquidity_score DECIMAL(4,3) DEFAULT 0 CHECK (liquidity_score >= 0 AND liquidity_score <= 1), -- 0-1 composite liquidity score
    -- Qualification and filtering results
    quick_viability_check BOOLEAN NOT NULL,
    viability_rejection_reason VARCHAR(50),
    conditions_check BOOLEAN NOT NULL,
    conditions_rejection_reason VARCHAR(50),
    -- Execution simulation
    simulated_execution BOOLEAN DEFAULT false,
    execution_timestamp TIMESTAMP(6) WITH TIME ZONE,
    combo_limit_price DECIMAL(10,4), -- Calculated combo limit price
    estimated_slippage DECIMAL(8,4), -- Estimated execution slippage
    estimated_commission DECIMAL(8,4), -- Estimated commission costs
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- SFR simulated trades (detailed execution simulation)
CREATE TABLE IF NOT EXISTS sfr_simulated_trades (
    id BIGSERIAL PRIMARY KEY,
    opportunity_id BIGINT NOT NULL REFERENCES sfr_opportunities(id) ON DELETE CASCADE,
    backtest_run_id INTEGER NOT NULL REFERENCES sfr_backtest_runs(id) ON DELETE CASCADE,
    trade_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    -- Trade execution details
    execution_timestamp TIMESTAMP(6) WITH TIME ZONE NOT NULL,
    quantity INTEGER NOT NULL,
    -- Individual leg execution details
    stock_execution_price DECIMAL(10,4) NOT NULL,
    stock_execution_time TIMESTAMP(6) WITH TIME ZONE NOT NULL,
    stock_slippage DECIMAL(8,4) DEFAULT 0,
    call_execution_price DECIMAL(10,4) NOT NULL,
    call_execution_time TIMESTAMP(6) WITH TIME ZONE NOT NULL,
    call_slippage DECIMAL(8,4) DEFAULT 0,
    put_execution_price DECIMAL(10,4) NOT NULL,
    put_execution_time TIMESTAMP(6) WITH TIME ZONE NOT NULL,
    put_slippage DECIMAL(8,4) DEFAULT 0,
    -- Combined trade metrics
    total_execution_time_ms INTEGER, -- Total time to execute all legs
    combo_net_credit DECIMAL(10,4) NOT NULL, -- Actual net credit received
    total_slippage DECIMAL(8,4) DEFAULT 0, -- Combined slippage impact
    total_commission DECIMAL(8,4) DEFAULT 0, -- Total commission paid
    -- Realized profit calculations
    realized_min_profit DECIMAL(10,4) NOT NULL, -- After slippage and commission
    realized_max_profit DECIMAL(10,4) NOT NULL, -- After slippage and commission
    realized_min_roi DECIMAL(8,4) NOT NULL, -- Actual ROI achieved
    realized_max_roi DECIMAL(8,4) NOT NULL, -- Actual max ROI possible
    -- Trade status and outcome
    execution_status VARCHAR(20) NOT NULL CHECK (execution_status IN ('FILLED', 'PARTIAL', 'FAILED', 'CANCELLED')),
    execution_quality VARCHAR(20) DEFAULT 'GOOD' CHECK (execution_quality IN ('EXCELLENT', 'GOOD', 'FAIR', 'POOR')),
    partial_fill_legs INTEGER DEFAULT 0, -- Number of legs with partial fills
    failed_leg VARCHAR(10), -- Which leg failed: 'STOCK', 'CALL', 'PUT'
    failure_reason TEXT, -- Detailed failure reason
    -- Position management (for tracking to expiry)
    position_opened BOOLEAN DEFAULT true,
    position_closed BOOLEAN DEFAULT false,
    close_timestamp TIMESTAMP(6) WITH TIME ZONE,
    close_reason VARCHAR(20), -- 'EXPIRY', 'EARLY_CLOSE', 'ASSIGNMENT'
    final_pnl DECIMAL(10,4), -- Final P&L if position held to close/expiry
    -- REMOVED: Market context at execution (VIX-related)
    market_stress_indicator BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- SFR performance analytics per backtest run
CREATE TABLE IF NOT EXISTS sfr_performance_analytics (
    id SERIAL PRIMARY KEY,
    backtest_run_id INTEGER NOT NULL REFERENCES sfr_backtest_runs(id) ON DELETE CASCADE UNIQUE,
    -- Opportunity discovery metrics
    total_opportunities_found INTEGER DEFAULT 0,
    opportunities_per_day DECIMAL(8,2) DEFAULT 0,
    opportunities_by_quality JSONB, -- {"EXCELLENT": 5, "GOOD": 15, "FAIR": 10, "POOR": 2}
    avg_opportunity_quality_score DECIMAL(4,3),
    -- Execution simulation results
    total_simulated_trades INTEGER DEFAULT 0,
    successful_executions INTEGER DEFAULT 0,
    failed_executions INTEGER DEFAULT 0,
    execution_success_rate DECIMAL(6,4), -- Percentage of successful executions
    partial_fills INTEGER DEFAULT 0,
    -- Profitability metrics
    total_gross_profit DECIMAL(12,4) DEFAULT 0,
    total_net_profit DECIMAL(12,4) DEFAULT 0, -- After slippage and commissions
    total_commissions_paid DECIMAL(12,4) DEFAULT 0,
    total_slippage_cost DECIMAL(12,4) DEFAULT 0,
    avg_profit_per_trade DECIMAL(10,4),
    median_profit_per_trade DECIMAL(10,4),
    -- ROI statistics
    avg_min_roi DECIMAL(8,4),
    median_min_roi DECIMAL(8,4),
    best_min_roi DECIMAL(8,4),
    worst_min_roi DECIMAL(8,4),
    roi_standard_deviation DECIMAL(8,4),
    -- Risk metrics
    max_single_trade_loss DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    max_drawdown_percent DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    sortino_ratio DECIMAL(8,4),
    calmar_ratio DECIMAL(8,4),
    -- Strike and expiry analysis
    most_profitable_strike_diff INTEGER, -- Most profitable call-put strike difference
    most_profitable_expiry_range VARCHAR(20), -- e.g., "20-30 days"
    avg_days_to_expiry DECIMAL(6,2),
    -- REMOVED: Market regime analysis (VIX-related)
    -- Time-based performance
    performance_by_hour JSONB, -- Hourly performance breakdown
    performance_by_day_of_week JSONB, -- Day of week performance
    performance_by_month JSONB, -- Monthly performance breakdown
    -- Symbol-specific results
    performance_by_symbol JSONB, -- Per-symbol performance breakdown
    most_profitable_symbol VARCHAR(10),
    least_profitable_symbol VARCHAR(10),
    -- Statistical significance tests
    t_test_significance DECIMAL(6,4), -- p-value for profit significance
    confidence_interval_lower DECIMAL(10,4), -- 95% confidence interval
    confidence_interval_upper DECIMAL(10,4),
    -- Metadata
    calculation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- REMOVED: sfr_rejection_log table - unused (empty, 0 rows, can be recreated when needed)

-- REMOVED: vix_sfr_correlation_analysis table - unused

-- REMOVED: sfr_risk_metrics table - unused

-- Indexes for efficient SFR backtesting queries
CREATE INDEX IF NOT EXISTS idx_sfr_opportunities_backtest_symbol_time
    ON sfr_opportunities (backtest_run_id, underlying_id, discovery_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_sfr_opportunities_expiry_strikes
    ON sfr_opportunities (expiry_date, call_strike, put_strike);

CREATE INDEX IF NOT EXISTS idx_sfr_opportunities_profitability
    ON sfr_opportunities (min_roi DESC, min_profit DESC) WHERE conditions_check = true;

CREATE INDEX IF NOT EXISTS idx_sfr_opportunities_quality_liquidity
    ON sfr_opportunities (opportunity_quality, liquidity_score DESC, execution_difficulty);

CREATE INDEX IF NOT EXISTS idx_sfr_simulated_trades_execution_time
    ON sfr_simulated_trades (backtest_run_id, execution_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_sfr_simulated_trades_performance
    ON sfr_simulated_trades (realized_min_roi DESC, execution_status) WHERE execution_status = 'FILLED';

-- REMOVED: Index for unused sfr_rejection_log table

-- REMOVED: Indexes for unused correlation and risk metrics tables

-- Update triggers for SFR tables
CREATE TRIGGER update_sfr_backtest_runs_updated_at BEFORE UPDATE ON sfr_backtest_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Constraints for data quality and consistency
ALTER TABLE sfr_opportunities ADD CONSTRAINT chk_sfr_strikes_valid
    CHECK (call_strike > put_strike AND call_strike > 0 AND put_strike > 0);

ALTER TABLE sfr_opportunities ADD CONSTRAINT chk_sfr_prices_positive
    CHECK (
        stock_price > 0 AND
        (call_bid IS NULL OR call_bid >= 0) AND
        (call_ask IS NULL OR call_ask >= 0) AND
        (put_bid IS NULL OR put_bid >= 0) AND
        (put_ask IS NULL OR put_ask >= 0) AND
        (call_bid IS NULL OR call_ask IS NULL OR call_bid <= call_ask) AND
        (put_bid IS NULL OR put_ask IS NULL OR put_bid <= put_ask)
    );

ALTER TABLE sfr_opportunities ADD CONSTRAINT chk_sfr_expiry_future
    CHECK (expiry_date >= discovery_timestamp::date);

ALTER TABLE sfr_simulated_trades ADD CONSTRAINT chk_sfr_execution_prices_positive
    CHECK (stock_execution_price > 0 AND call_execution_price >= 0 AND put_execution_price >= 0);

ALTER TABLE sfr_simulated_trades ADD CONSTRAINT chk_sfr_quantities_positive
    CHECK (quantity > 0);

-- Functions for SFR backtesting analysis

-- Function to calculate SFR opportunity quality score
CREATE OR REPLACE FUNCTION calculate_sfr_quality_score(
    p_min_roi DECIMAL,
    p_liquidity_score DECIMAL,
    p_call_bid_ask_spread DECIMAL,
    p_put_bid_ask_spread DECIMAL,
    p_days_to_expiry INTEGER
)
RETURNS DECIMAL(4,3) AS $$
DECLARE
    quality_score DECIMAL(4,3);
    roi_component DECIMAL(4,3);
    liquidity_component DECIMAL(4,3);
    spread_component DECIMAL(4,3);
    time_component DECIMAL(4,3);
BEGIN
    -- ROI component (0-0.4 weight) - higher ROI is better
    roi_component := LEAST(0.4, GREATEST(0, p_min_roi / 5.0 * 0.4));

    -- Liquidity component (0-0.3 weight) - direct liquidity score
    liquidity_component := COALESCE(p_liquidity_score, 0) * 0.3;

    -- Spread component (0-0.2 weight) - tighter spreads are better
    spread_component := GREATEST(0, 0.2 - ((COALESCE(p_call_bid_ask_spread, 20) + COALESCE(p_put_bid_ask_spread, 20)) / 40.0 * 0.2));

    -- Time component (0-0.1 weight) - optimal around 25-35 days
    time_component := CASE
        WHEN p_days_to_expiry BETWEEN 25 AND 35 THEN 0.1
        WHEN p_days_to_expiry BETWEEN 20 AND 40 THEN 0.08
        WHEN p_days_to_expiry BETWEEN 15 AND 45 THEN 0.05
        ELSE 0.02
    END;

    quality_score := roi_component + liquidity_component + spread_component + time_component;

    RETURN LEAST(1.0, GREATEST(0.0, quality_score));
END;
$$ LANGUAGE plpgsql;

-- Function to get SFR performance summary for a backtest run
CREATE OR REPLACE FUNCTION get_sfr_performance_summary(p_backtest_run_id INTEGER)
RETURNS TABLE (
    total_opportunities BIGINT,
    successful_trades BIGINT,
    success_rate DECIMAL,
    avg_profit DECIMAL,
    total_profit DECIMAL,
    best_roi DECIMAL,
    worst_roi DECIMAL,
    avg_roi DECIMAL,
    sharpe_ratio DECIMAL,
    max_drawdown DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*) as total_opportunities,
        COUNT(*) FILTER (WHERE st.execution_status = 'FILLED') as successful_trades,
        (COUNT(*) FILTER (WHERE st.execution_status = 'FILLED') * 100.0 / COUNT(*))::DECIMAL(6,4) as success_rate,
        AVG(st.realized_min_profit) FILTER (WHERE st.execution_status = 'FILLED')::DECIMAL(10,4) as avg_profit,
        SUM(st.realized_min_profit) FILTER (WHERE st.execution_status = 'FILLED')::DECIMAL(12,4) as total_profit,
        MAX(st.realized_min_roi) FILTER (WHERE st.execution_status = 'FILLED')::DECIMAL(8,4) as best_roi,
        MIN(st.realized_min_roi) FILTER (WHERE st.execution_status = 'FILLED')::DECIMAL(8,4) as worst_roi,
        AVG(st.realized_min_roi) FILTER (WHERE st.execution_status = 'FILLED')::DECIMAL(8,4) as avg_roi,
        pa.sharpe_ratio,
        pa.max_drawdown
    FROM sfr_opportunities so
    LEFT JOIN sfr_simulated_trades st ON so.id = st.opportunity_id
    LEFT JOIN sfr_performance_analytics pa ON so.backtest_run_id = pa.backtest_run_id
    WHERE so.backtest_run_id = p_backtest_run_id
    GROUP BY pa.sharpe_ratio, pa.max_drawdown;
END;
$$ LANGUAGE plpgsql;

-- REMOVED: analyze_sfr_vix_performance function - VIX analysis removed

-- Comments for documentation
COMMENT ON TABLE sfr_backtest_runs IS 'SFR-specific backtesting run parameters and configuration';
COMMENT ON TABLE sfr_opportunities IS 'All SFR arbitrage opportunities discovered during backtesting with detailed metrics';
COMMENT ON TABLE sfr_simulated_trades IS 'Detailed execution simulation results for SFR trades including slippage and commission modeling';
COMMENT ON TABLE sfr_performance_analytics IS 'Comprehensive performance analytics calculated per SFR backtest run';
-- REMOVED: Comment for unused sfr_rejection_log table

COMMENT ON FUNCTION calculate_sfr_quality_score IS 'Calculates a composite quality score (0-1) for SFR opportunities based on ROI, liquidity, spreads, and time to expiry';
COMMENT ON FUNCTION get_sfr_performance_summary IS 'Returns summary performance metrics for an SFR backtest run';
-- REMOVED: Comments for VIX-related functions and tables
