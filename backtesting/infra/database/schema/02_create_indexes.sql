-- 02_create_indexes.sql
-- Performance indexes for options arbitrage backtesting queries

-- Indexes for option_chains table
CREATE INDEX idx_option_chains_underlying_expiry
    ON option_chains(underlying_id, expiration_date);

CREATE INDEX idx_option_chains_strike_type
    ON option_chains(strike_price, option_type);

CREATE INDEX idx_option_chains_ib_con_id
    ON option_chains(ib_con_id)
    WHERE ib_con_id IS NOT NULL;

CREATE INDEX idx_option_chains_expiry_active
    ON option_chains(expiration_date, active)
    WHERE active = true;

-- Indexes for market_data_ticks (before converting to hypertable)
CREATE INDEX idx_market_data_contract_time
    ON market_data_ticks(contract_id, time DESC);

CREATE INDEX idx_market_data_time_only
    ON market_data_ticks(time DESC);

-- Composite index for arbitrage scanning queries
CREATE INDEX idx_market_data_arbitrage_scan
    ON market_data_ticks(time DESC, contract_id, bid_price, ask_price)
    WHERE bid_price IS NOT NULL AND ask_price IS NOT NULL;

-- Indexes for stock_data_ticks
CREATE INDEX idx_stock_data_underlying_time
    ON stock_data_ticks(underlying_id, time DESC);

CREATE INDEX idx_stock_data_time_only
    ON stock_data_ticks(time DESC);

-- Indexes for arbitrage_opportunities
CREATE INDEX idx_arbitrage_opp_strategy_time
    ON arbitrage_opportunities(strategy_type, timestamp DESC);

CREATE INDEX idx_arbitrage_opp_underlying_time
    ON arbitrage_opportunities(underlying_id, timestamp DESC);

CREATE INDEX idx_arbitrage_opp_profit
    ON arbitrage_opportunities(roi_percent DESC)
    WHERE execution_successful = true;

CREATE INDEX idx_arbitrage_opp_discovered
    ON arbitrage_opportunities(timestamp DESC, discovered_by);

-- Index for finding unexpired opportunities
CREATE INDEX idx_arbitrage_opp_active
    ON arbitrage_opportunities(expiration_date, timestamp DESC)
    WHERE execution_attempted = false;

-- Indexes for corporate_actions
CREATE INDEX idx_corp_actions_underlying_date
    ON corporate_actions(underlying_id, ex_date DESC);

CREATE INDEX idx_corp_actions_unprocessed
    ON corporate_actions(ex_date, processed)
    WHERE processed = false;

-- Indexes for market_events
CREATE INDEX idx_market_events_date
    ON market_events(event_date, event_type);

CREATE INDEX idx_market_events_underlying
    ON market_events(underlying_id, event_date)
    WHERE underlying_id IS NOT NULL;

-- Indexes for data_quality_metrics
CREATE INDEX idx_data_quality_timestamp
    ON data_quality_metrics(check_timestamp DESC);

CREATE INDEX idx_data_quality_failed
    ON data_quality_metrics(check_timestamp DESC, metric_type)
    WHERE passed = false;

-- Indexes for backtest_runs
CREATE INDEX idx_backtest_runs_status
    ON backtest_runs(status, created_at DESC);

CREATE INDEX idx_backtest_runs_strategy
    ON backtest_runs(strategy_type, created_at DESC);

-- GIN index for JSONB parameters search
CREATE INDEX idx_backtest_runs_params
    ON backtest_runs USING GIN (parameters);

-- Partial indexes for active data
CREATE INDEX idx_underlying_active
    ON underlying_securities(symbol)
    WHERE active = true;

CREATE INDEX idx_option_chains_active_near_money
    ON option_chains(underlying_id, expiration_date, strike_price)
    WHERE active = true;

-- Function-based indexes for common calculations
CREATE INDEX idx_market_data_spread_ratio
    ON market_data_ticks((bid_ask_spread / NULLIF(mid_price, 0)))
    WHERE bid_price IS NOT NULL AND ask_price IS NOT NULL;

-- Text search indexes
CREATE INDEX idx_underlying_symbol_search
    ON underlying_securities(lower(symbol));

-- Covering indexes for common queries
CREATE INDEX idx_option_complete_lookup
    ON option_chains(underlying_id, expiration_date, strike_price, option_type)
    INCLUDE (id, ib_con_id, contract_symbol);

-- Statistics target increases for better query planning
ALTER TABLE market_data_ticks ALTER COLUMN time SET STATISTICS 1000;
ALTER TABLE market_data_ticks ALTER COLUMN contract_id SET STATISTICS 1000;
ALTER TABLE option_chains ALTER COLUMN underlying_id SET STATISTICS 500;
ALTER TABLE option_chains ALTER COLUMN expiration_date SET STATISTICS 500;
