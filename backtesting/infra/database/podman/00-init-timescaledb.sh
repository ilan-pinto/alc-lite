#!/bin/bash
set -e

echo "Initializing TimescaleDB and extensions..."

# Create TimescaleDB extension and other required extensions
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create TimescaleDB extension
    CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

    -- Enable required extensions
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

    -- Set timezone to UTC for consistency
    SET timezone = 'UTC';

    -- TimescaleDB initialization completed
    SELECT 'TimescaleDB initialized successfully' as status;
EOSQL

echo "TimescaleDB initialization completed. Schema files will be executed next."
