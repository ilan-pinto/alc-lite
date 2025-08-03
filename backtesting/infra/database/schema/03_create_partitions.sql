-- 03_create_partitions.sql
-- Partition management for time-series data

-- Function to create monthly partitions dynamically
CREATE OR REPLACE FUNCTION create_monthly_partition(
    table_name text,
    start_date date
)
RETURNS void AS $$
DECLARE
    partition_name text;
    end_date date;
BEGIN
    partition_name := table_name || '_' || to_char(start_date, 'YYYY_MM');
    end_date := start_date + interval '1 month';

    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I PARTITION OF %I
        FOR VALUES FROM (%L) TO (%L)',
        partition_name, table_name, start_date, end_date
    );

    -- Create indexes on partition
    IF table_name = 'market_data_ticks' THEN
        EXECUTE format('
            CREATE INDEX IF NOT EXISTS %I ON %I (contract_id, time DESC)',
            partition_name || '_contract_time_idx', partition_name
        );
        EXECUTE format('
            CREATE INDEX IF NOT EXISTS %I ON %I (time DESC)',
            partition_name || '_time_idx', partition_name
        );
    ELSIF table_name = 'stock_data_ticks' THEN
        EXECUTE format('
            CREATE INDEX IF NOT EXISTS %I ON %I (underlying_id, time DESC)',
            partition_name || '_underlying_time_idx', partition_name
        );
        EXECUTE format('
            CREATE INDEX IF NOT EXISTS %I ON %I (time DESC)',
            partition_name || '_time_idx', partition_name
        );
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to create partitions for a date range
CREATE OR REPLACE FUNCTION create_partition_range(
    table_name text,
    start_date date,
    end_date date
)
RETURNS void AS $$
DECLARE
    current_date date;
BEGIN
    current_date := date_trunc('month', start_date);

    WHILE current_date < end_date LOOP
        PERFORM create_monthly_partition(table_name, current_date);
        current_date := current_date + interval '1 month';
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to automatically create future partitions
CREATE OR REPLACE FUNCTION ensure_partitions_exist()
RETURNS void AS $$
DECLARE
    future_date date;
BEGIN
    future_date := date_trunc('month', CURRENT_DATE + interval '3 months');

    -- Ensure partitions exist for the next 3 months
    PERFORM create_partition_range('market_data_ticks', CURRENT_DATE, future_date);
    PERFORM create_partition_range('stock_data_ticks', CURRENT_DATE, future_date);
END;
$$ LANGUAGE plpgsql;

-- Create initial partitions (past 6 months + next 3 months)
DO $$
BEGIN
    PERFORM create_partition_range(
        'market_data_ticks',
        CURRENT_DATE - interval '6 months',
        CURRENT_DATE + interval '3 months'
    );

    PERFORM create_partition_range(
        'stock_data_ticks',
        CURRENT_DATE - interval '6 months',
        CURRENT_DATE + interval '3 months'
    );
END $$;

-- Scheduled job to create new partitions (runs monthly)
CREATE OR REPLACE FUNCTION auto_create_partitions()
RETURNS void AS $$
BEGIN
    PERFORM ensure_partitions_exist();
END;
$$ LANGUAGE plpgsql;

-- Partition maintenance function to drop old partitions
CREATE OR REPLACE FUNCTION drop_old_partitions(
    table_name text,
    retention_months integer DEFAULT 24
)
RETURNS void AS $$
DECLARE
    partition record;
    cutoff_date date;
BEGIN
    cutoff_date := CURRENT_DATE - (retention_months || ' months')::interval;

    FOR partition IN
        SELECT schemaname, tablename
        FROM pg_tables
        WHERE tablename LIKE table_name || '_%'
        AND tablename ~ '\d{4}_\d{2}$'
    LOOP
        -- Extract date from partition name
        IF to_date(right(partition.tablename, 7), 'YYYY_MM') < cutoff_date THEN
            EXECUTE format('DROP TABLE IF EXISTS %I.%I CASCADE',
                partition.schemaname, partition.tablename);
            RAISE NOTICE 'Dropped partition: %.%', partition.schemaname, partition.tablename;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Create a scheduled job for partition maintenance (requires pg_cron extension)
-- Uncomment if pg_cron is available
/*
SELECT cron.schedule('create-partitions', '0 0 1 * *', $$SELECT auto_create_partitions()$$);
SELECT cron.schedule('cleanup-partitions', '0 0 2 * *', $$SELECT drop_old_partitions('market_data_ticks', 24)$$);
SELECT cron.schedule('cleanup-stock-partitions', '0 0 2 * *', $$SELECT drop_old_partitions('stock_data_ticks', 24)$$);
*/

-- View to show partition information
CREATE OR REPLACE VIEW partition_info AS
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    pg_stat_get_live_tuples(c.oid) as live_rows,
    pg_stat_get_dead_tuples(c.oid) as dead_rows
FROM pg_tables t
JOIN pg_class c ON c.relname = t.tablename
JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = t.schemaname
WHERE tablename LIKE 'market_data_ticks_%'
   OR tablename LIKE 'stock_data_ticks_%'
ORDER BY tablename;
