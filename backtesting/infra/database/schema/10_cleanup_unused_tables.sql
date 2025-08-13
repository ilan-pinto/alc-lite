-- 10_cleanup_unused_tables.sql
-- Remove unused tables identified from code analysis
-- Generated: 2025-01-14

-- Drop unused tables in correct dependency order
-- Note: Some tables may have foreign key dependencies, so order matters

-- Drop VIX correlation tables (not used in active code)
DROP TABLE IF EXISTS vix_sfr_correlation_analysis CASCADE;
DROP TABLE IF EXISTS vix_arbitrage_correlation CASCADE;
DROP TABLE IF EXISTS vix_historical_stats CASCADE;

-- Drop unused arbitrage and market data tables
DROP TABLE IF EXISTS market_data_ticks CASCADE;
DROP TABLE IF EXISTS arbitrage_opportunities CASCADE;
DROP TABLE IF EXISTS market_events CASCADE;

-- Drop unused collection and analytics tables
DROP TABLE IF EXISTS collection_queue CASCADE;
DROP TABLE IF EXISTS collection_statistics CASCADE;
DROP TABLE IF EXISTS daily_collection_metrics CASCADE;

-- Drop unused risk and lifecycle tables
DROP TABLE IF EXISTS sfr_risk_metrics CASCADE;
DROP TABLE IF EXISTS option_contract_lifecycle CASCADE;

-- Drop superseded gap tracking table
DROP TABLE IF EXISTS historical_data_gaps CASCADE;

-- Drop unused corporate actions table
DROP TABLE IF EXISTS corporate_actions CASCADE;

-- Log cleanup completion
SELECT 'Unused tables cleanup completed' AS cleanup_status;

-- Show remaining table count
SELECT COUNT(*) AS remaining_tables
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_type = 'BASE TABLE';
