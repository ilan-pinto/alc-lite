#!/usr/bin/env python3
"""
Database Constraint Checker

A utility script to check if all required database constraints exist
for the historical data loading pipeline.

Usage:
    python check_database.py
    python check_database.py --fix
"""

import asyncio
from pathlib import Path

import argparse
import asyncpg
import logging
from rich.console import Console
from rich.table import Table

try:
    from ..config.config import DatabaseConfig
except ImportError:
    from backtesting.infra.data_collection.config.config import DatabaseConfig

console = Console()


async def check_constraints(db_pool: asyncpg.Pool) -> dict:
    """Check if all required constraints exist."""

    console.print("🔍 Checking database constraints...")

    results = {
        "underlying_symbol": False,
        "stock_data_unique": False,
        "option_chains_unique": False,
    }

    async with db_pool.acquire() as conn:
        # Check underlying_securities symbol constraint
        symbol_constraint = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_constraint c
                JOIN pg_attribute a ON a.attnum = ANY(c.conkey) AND a.attrelid = c.conrelid
                WHERE c.contype = 'u'
                  AND c.conrelid = 'underlying_securities'::regclass
                  AND a.attname = 'symbol'
            )
        """
        )
        results["underlying_symbol"] = symbol_constraint

        # Check stock_data_ticks unique constraint
        stock_constraint = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'uq_stock_data_time_underlying'
            )
        """
        )
        results["stock_data_unique"] = stock_constraint

        # Check option_chains unique constraint
        option_constraint = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid = 'option_chains'::regclass
                  AND contype = 'u'
                  AND array_length(conkey, 1) = 4
            )
        """
        )
        results["option_chains_unique"] = option_constraint

    return results


async def fix_constraints(db_pool: asyncpg.Pool):
    """Apply the migration to fix missing constraints."""

    console.print("🔧 Applying database migration...")

    # Read migration file
    migration_file = Path(__file__).parent / "schema" / "06_add_unique_constraints.sql"

    if not migration_file.exists():
        # Try from database directory
        migration_file = (
            Path(__file__).parent.parent
            / "database"
            / "schema"
            / "06_add_unique_constraints.sql"
        )

    if not migration_file.exists():
        console.print("[red]❌ Migration file not found![/red]")
        console.print(
            "Expected location: backtesting/infra/database/schema/06_add_unique_constraints.sql"
        )
        return False

    try:
        migration_sql = migration_file.read_text()

        async with db_pool.acquire() as conn:
            # Execute migration
            await conn.execute(migration_sql)

        console.print("[green]✅ Migration applied successfully![/green]")
        return True

    except Exception as e:
        console.print(f"[red]❌ Migration failed: {e}[/red]")
        return False


def display_results(results: dict):
    """Display constraint check results in a nice table."""

    table = Table(title="Database Constraint Status")
    table.add_column("Constraint", style="cyan")
    table.add_column("Table", style="blue")
    table.add_column("Status", justify="center")

    constraints = [
        ("Symbol Unique", "underlying_securities", results["underlying_symbol"]),
        ("Time+Underlying Unique", "stock_data_ticks", results["stock_data_unique"]),
        ("Option Contract Unique", "option_chains", results["option_chains_unique"]),
    ]

    all_good = True

    for name, table_name, exists in constraints:
        if exists:
            status = "[green]✅ EXISTS[/green]"
        else:
            status = "[red]❌ MISSING[/red]"
            all_good = False

        table.add_row(name, table_name, status)

    console.print(table)

    if all_good:
        console.print("\n[green]🎉 All constraints are properly configured![/green]")
        console.print(
            "Your database is ready for the historical data loading pipeline."
        )
    else:
        console.print("\n[yellow]⚠️  Some constraints are missing.[/yellow]")
        console.print("Run with --fix to automatically apply the migration.")
        console.print(
            "Or manually run: backtesting/infra/database/schema/06_add_unique_constraints.sql"
        )

    return all_good


async def main():
    """Main function."""

    parser = argparse.ArgumentParser(
        description="Check and fix database constraints for the historical data pipeline"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Automatically fix missing constraints"
    )
    parser.add_argument("--host", default="localhost", help="Database host")
    parser.add_argument("--port", type=int, default=5432, help="Database port")
    parser.add_argument("--database", default="options_arbitrage", help="Database name")
    parser.add_argument("--user", default="trading_user", help="Database user")
    parser.add_argument(
        "--password", default="secure_trading_password", help="Database password"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.WARNING)

    try:
        console.print("🚀 Database Constraint Checker")
        console.print(
            f"Connecting to {args.user}@{args.host}:{args.port}/{args.database}"
        )

        # Create database connection
        db_pool = await asyncpg.create_pool(
            host=args.host,
            port=args.port,
            database=args.database,
            user=args.user,
            password=args.password,
            min_size=1,
            max_size=5,
        )

        # Check constraints
        results = await check_constraints(db_pool)
        all_good = display_results(results)

        # Apply fixes if requested
        if args.fix and not all_good:
            success = await fix_constraints(db_pool)

            if success:
                # Re-check after fixing
                console.print("\n🔍 Re-checking constraints after migration...")
                results = await check_constraints(db_pool)
                display_results(results)

        await db_pool.close()

        if not all_good and not args.fix:
            exit(1)

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
