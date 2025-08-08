-- 06_add_unique_constraints.sql
-- Add missing unique constraints for ON CONFLICT clauses

-- Add unique constraint for stock_data_ticks (if not already exists)
-- This ensures we don't have duplicate data for the same timestamp and underlying
DO $$
BEGIN
    -- Add unique constraint for stock_data_ticks
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uq_stock_data_time_underlying'
    ) THEN
        ALTER TABLE stock_data_ticks
        ADD CONSTRAINT uq_stock_data_time_underlying
        UNIQUE (time, underlying_id);

        RAISE NOTICE 'Added unique constraint uq_stock_data_time_underlying';
    ELSE
        RAISE NOTICE 'Constraint uq_stock_data_time_underlying already exists';
    END IF;
END $$;

-- Ensure underlying_securities has unique constraint on symbol (should already exist)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint c
        JOIN pg_attribute a ON a.attnum = ANY(c.conkey) AND a.attrelid = c.conrelid
        WHERE c.contype = 'u'
          AND c.conrelid = 'underlying_securities'::regclass
          AND a.attname = 'symbol'
    ) THEN
        ALTER TABLE underlying_securities
        ADD CONSTRAINT uq_underlying_symbol
        UNIQUE (symbol);

        RAISE NOTICE 'Added unique constraint uq_underlying_symbol';
    ELSE
        RAISE NOTICE 'Unique constraint on underlying_securities.symbol already exists';
    END IF;
END $$;

-- Ensure option_chains has the unique constraint (should already exist from schema)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname ~ 'option_chains.*underlying_id.*expiration_date.*strike_price.*option_type'
           OR conrelid = 'option_chains'::regclass
           AND contype = 'u'
           AND array_length(conkey, 1) = 4
    ) THEN
        ALTER TABLE option_chains
        ADD CONSTRAINT uq_option_chains_unique_contract
        UNIQUE (underlying_id, expiration_date, strike_price, option_type);

        RAISE NOTICE 'Added unique constraint uq_option_chains_unique_contract';
    ELSE
        RAISE NOTICE 'Unique constraint on option_chains already exists';
    END IF;
END $$;

-- Add comments
COMMENT ON CONSTRAINT uq_stock_data_time_underlying ON stock_data_ticks IS
'Ensures only one stock price record per timestamp and underlying security';

-- Note: market_data_ticks intentionally does not have a unique constraint
-- because high-frequency options data might have multiple ticks per timestamp
