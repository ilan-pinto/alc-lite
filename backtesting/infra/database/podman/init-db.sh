#!/bin/bash
set -e

# Wait for PostgreSQL to be ready
until pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"; do
  echo "Waiting for PostgreSQL to be ready..."
  sleep 2
done

echo "PostgreSQL is ready. Initializing database..."

# Create TimescaleDB extension
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

    -- Enable required extensions
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

    -- Set timezone to UTC for consistency
    SET timezone = 'UTC';

    -- Create application user with limited privileges (optional)
    -- CREATE USER app_user WITH PASSWORD 'app_password';
    -- GRANT CONNECT ON DATABASE $POSTGRES_DB TO app_user;

    -- Performance tuning for TimescaleDB
    SELECT timescaledb_tune();
EOSQL

# Execute all SQL files in order
for sql_file in /docker-entrypoint-initdb.d/sql/*.sql; do
    if [ -f "$sql_file" ]; then
        echo "Executing $sql_file..."
        psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f "$sql_file"
    fi
done

echo "Database initialization completed successfully!"
