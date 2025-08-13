-- 11_cleanup_additional_unused_tables.sql
-- Remove additional unused tables identified as empty/inactive
-- Generated: 2025-01-14 (Phase 2 cleanup)

-- Drop additional unused tables in correct dependency order
-- These tables are all empty (0 rows) and not actively used by schedulers

-- Drop data quality and gap tracking tables (not used by active schedulers)
DROP TABLE IF EXISTS data_quality_metrics CASCADE;
DROP TABLE IF EXISTS data_gaps CASCADE;

-- Drop VIX data tables (VIX collector not used by active schedulers)
DROP TABLE IF EXISTS vix_data_ticks CASCADE;
DROP TABLE IF EXISTS vix_term_structure CASCADE;

-- Drop SFR rejection log (only used in backtests, currently empty, can be recreated)
DROP TABLE IF EXISTS sfr_rejection_log CASCADE;

-- Log additional cleanup completion
SELECT 'Phase 2: Additional unused tables cleanup completed' AS cleanup_status;

-- Show final table count
SELECT COUNT(*) AS final_table_count
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_type = 'BASE TABLE';
