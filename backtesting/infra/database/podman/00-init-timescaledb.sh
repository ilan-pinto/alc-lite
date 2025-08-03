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

    -- Performance tuning for TimescaleDB
    SELECT timescaledb_tune();
EOSQL

echo "TimescaleDB initialization completed. Schema files will be executed next."
