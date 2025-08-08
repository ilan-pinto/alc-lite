# Database Migration Guide

## Issue: Missing Unique Constraints

The historical data loader expects unique constraints on certain table columns to handle duplicate data properly. If these constraints are missing, you'll see errors like:

```
ERROR - Error loading history for IWM: there is no unique or exclusion constraint matching the ON CONFLICT specification
```

This can happen in several places:
- `underlying_securities` table needs unique constraint on `symbol`
- `stock_data_ticks` table needs unique constraint on `(time, underlying_id)`
- `option_chains` table needs unique constraint on `(underlying_id, expiration_date, strike_price, option_type)`

## Solution

### Option 1: Apply the Migration (Recommended)

Run the following SQL migration to add the missing unique constraints:

```bash
# Connect to your database
psql -h localhost -U trading_user -d options_arbitrage

# Run the migration
\i /path/to/backtesting/infra/database/schema/06_add_unique_constraints.sql
```

Or run directly:

```sql
-- Add unique constraint for stock_data_ticks
ALTER TABLE stock_data_ticks
ADD CONSTRAINT uq_stock_data_time_underlying
UNIQUE (time, underlying_id);
```

### Option 2: Use the Updated Code (Already Applied)

The `historical_loader.py` has been updated to handle missing constraints gracefully:

1. It first tries to use `ON CONFLICT` for efficient duplicate handling
2. If the constraint doesn't exist, it falls back to manual duplicate checking
3. For options data, it handles duplicates individually since high-frequency data might have multiple ticks per timestamp

## Important Notes

### For Stock Data
- Stock data uses `(time, underlying_id)` as a unique key
- Only one price per timestamp per stock is stored
- The constraint prevents duplicate historical data

### For Options Data
- Options data might have multiple ticks per timestamp
- No unique constraint is enforced at the database level
- Duplicates are handled in the application logic

## Verification

After applying the migration, verify it worked:

```sql
-- Check if constraint exists
SELECT conname FROM pg_constraint
WHERE conname = 'uq_stock_data_time_underlying';

-- List all constraints on stock_data_ticks
\d stock_data_ticks
```

## Performance Considerations

Adding unique constraints will:
- ✅ Improve data integrity
- ✅ Make `ON CONFLICT` clauses work efficiently
- ✅ Prevent duplicate data
- ⚠️ Slightly slow down inserts (due to uniqueness checking)
- ⚠️ Require more storage space for the index

## Rollback

If you need to remove the constraint:

```sql
ALTER TABLE stock_data_ticks
DROP CONSTRAINT IF EXISTS uq_stock_data_time_underlying;
```

## Next Steps

After fixing the database constraints:

1. Run the pipeline again:
   ```bash
   python load_historical_pipeline.py --symbol SPY --days 30
   ```

2. Monitor for any remaining errors in the logs

3. If you see performance issues with large data loads, consider:
   - Temporarily disabling constraints during bulk loads
   - Using COPY instead of INSERT for very large datasets
   - Partitioning tables by date for better performance
