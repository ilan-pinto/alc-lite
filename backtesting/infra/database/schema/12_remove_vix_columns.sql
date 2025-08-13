-- 12_remove_vix_columns.sql
-- Remove VIX-related columns from existing tables
-- Generated: 2025-01-14

-- Remove VIX columns from sfr_backtest_runs table if they exist
ALTER TABLE sfr_backtest_runs
DROP COLUMN IF EXISTS vix_regime_filter,
DROP COLUMN IF EXISTS min_vix_level,
DROP COLUMN IF EXISTS max_vix_level,
DROP COLUMN IF EXISTS exclude_vix_spikes;

-- Remove VIX columns from sfr_simulated_trades table if they exist
ALTER TABLE sfr_simulated_trades
DROP COLUMN IF EXISTS vix_level_at_execution,
DROP COLUMN IF EXISTS vix_regime_at_execution;

-- Remove VIX columns from sfr_performance_analytics table if they exist
ALTER TABLE sfr_performance_analytics
DROP COLUMN IF EXISTS performance_by_vix_regime,
DROP COLUMN IF EXISTS performance_correlation_vix,
DROP COLUMN IF EXISTS best_vix_range_min,
DROP COLUMN IF EXISTS best_vix_range_max;

-- Remove VIX columns from sfr_rejection_log table if it exists and has VIX columns
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'sfr_rejection_log') THEN
        ALTER TABLE sfr_rejection_log
        DROP COLUMN IF EXISTS vix_level_at_rejection,
        DROP COLUMN IF EXISTS vix_regime_at_rejection;
    END IF;
END $$;

-- Drop VIX analysis function if it exists
DROP FUNCTION IF EXISTS analyze_sfr_vix_performance(INTEGER);

-- Log completion
SELECT 'VIX columns removal completed' AS cleanup_status;

-- Show updated table structures
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name IN ('sfr_backtest_runs', 'sfr_simulated_trades', 'sfr_performance_analytics')
  AND column_name LIKE '%vix%'
ORDER BY table_name, column_name;
