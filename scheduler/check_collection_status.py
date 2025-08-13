#!/usr/bin/env python3
"""
Collection Status Checker for Daily Options Data Collection
Provides monitoring and health check capabilities for Phase 1 implementation
"""

import asyncio
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import argparse
import asyncpg
import pytz
import yaml
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

console = Console()


class CollectionStatusChecker:
    """Check and monitor daily collection status."""

    def __init__(self, config_path: str = "scheduler/daily_schedule_israel.yaml"):
        self.config = self._load_config(config_path)
        self.israel_tz = pytz.timezone(self.config["timezone"])
        self.db_pool = None

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            config_file = Path(__file__).parent / config_path

        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    async def connect_db(self):
        """Connect to database."""
        db_config = self.config["database"]
        self.db_pool = await asyncpg.create_pool(
            host=db_config["host"],
            port=db_config["port"],
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"],
        )

    async def close_db(self):
        """Close database connection."""
        if self.db_pool:
            await self.db_pool.close()

    async def check_today_status(self) -> dict:
        """Check today's collection status."""
        async with self.db_pool.acquire() as conn:
            # Get today's collections
            today_status = await conn.fetch(
                """
                SELECT
                    symbol,
                    collection_type,
                    status,
                    started_at AT TIME ZONE 'Asia/Jerusalem' as started_at,
                    completed_at AT TIME ZONE 'Asia/Jerusalem' as completed_at,
                    records_collected,
                    contracts_discovered,
                    error_message,
                    execution_time_ms
                FROM daily_collection_status
                WHERE collection_date = CURRENT_DATE
                ORDER BY started_at DESC
            """
            )

            # Get summary statistics
            summary = await conn.fetchrow(
                """
                SELECT
                    COUNT(DISTINCT symbol) as symbols_collected,
                    SUM(records_collected) as total_records,
                    COUNT(CASE WHEN status = 'success' THEN 1 END) as successful_runs,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_runs,
                    AVG(execution_time_ms) as avg_execution_time
                FROM daily_collection_status
                WHERE collection_date = CURRENT_DATE
                AND symbol != 'ALL'
            """
            )

            return {"details": today_status, "summary": summary}

    async def check_weekly_trend(self) -> list:
        """Check collection trend for past 7 days."""
        async with self.db_pool.acquire() as conn:
            return await conn.fetch(
                """
                SELECT
                    collection_date,
                    COUNT(DISTINCT symbol) as symbols,
                    SUM(records_collected) as records,
                    COUNT(CASE WHEN status = 'success' THEN 1 END) as successes,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failures,
                    ROUND(AVG(data_quality_score)::numeric, 2) as avg_quality
                FROM daily_collection_status
                WHERE collection_date >= CURRENT_DATE - INTERVAL '7 days'
                AND symbol != 'ALL'
                GROUP BY collection_date
                ORDER BY collection_date DESC
            """
            )

    async def check_data_gaps(self) -> list:
        """Check for data gaps that need backfilling."""
        async with self.db_pool.acquire() as conn:
            return await conn.fetch(
                """
                SELECT
                    symbol,
                    gap_date,
                    gap_type,
                    detected_at AT TIME ZONE 'Asia/Jerusalem' as detected_at,
                    backfilled,
                    backfill_attempts
                FROM data_gaps
                WHERE backfilled = false
                ORDER BY gap_date DESC, symbol
                LIMIT 20
            """
            )

    async def check_active_contracts(self) -> dict:
        """Check active option contracts being tracked."""
        async with self.db_pool.acquire() as conn:
            # Get contract counts by symbol
            contract_counts = await conn.fetch(
                """
                SELECT
                    us.symbol,
                    COUNT(DISTINCT ocl.contract_id) as active_contracts,
                    COUNT(DISTINCT oc.expiration_date) as unique_expiries,
                    MIN(oc.expiration_date) as nearest_expiry,
                    MAX(oc.expiration_date) as furthest_expiry
                FROM underlying_securities us
                LEFT JOIN option_chains oc ON us.id = oc.underlying_id
                LEFT JOIN option_contract_lifecycle ocl ON oc.id = ocl.contract_id
                WHERE ocl.status = 'active'
                AND us.symbol IN ('SPY', 'PLTR', 'TSLA')
                GROUP BY us.symbol
                ORDER BY us.symbol
            """
            )

            # Get total summary
            total = await conn.fetchrow(
                """
                SELECT
                    COUNT(DISTINCT ocl.contract_id) as total_contracts,
                    COUNT(DISTINCT oc.expiration_date) as total_expiries
                FROM option_contract_lifecycle ocl
                JOIN option_chains oc ON ocl.contract_id = oc.id
                WHERE ocl.status = 'active'
            """
            )

            return {"by_symbol": contract_counts, "total": total}

    async def check_collection_health(self) -> list:
        """Check overall collection system health."""
        async with self.db_pool.acquire() as conn:
            return await conn.fetch(
                """
                SELECT * FROM check_collection_health(NULL)
            """
            )

    def display_status(self, data: dict):
        """Display status information in formatted tables."""
        current_time = datetime.now(self.israel_tz)

        # Header
        console.print(
            Panel.fit(
                f"[bold cyan]Daily Collection Status Report[/bold cyan]\n"
                f"[dim]{current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}[/dim]",
                border_style="cyan",
            )
        )

        # Today's Summary
        summary = data["today"]["summary"]
        if summary:
            summary_table = Table(title="Today's Summary", box=box.ROUNDED)
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="green")

            summary_table.add_row(
                "Symbols Collected", str(summary["symbols_collected"] or 0)
            )
            summary_table.add_row("Total Records", str(summary["total_records"] or 0))
            summary_table.add_row(
                "Successful Runs", str(summary["successful_runs"] or 0)
            )
            summary_table.add_row("Failed Runs", str(summary["failed_runs"] or 0))

            if summary["avg_execution_time"]:
                avg_time = f"{summary['avg_execution_time']:.0f}ms"
                summary_table.add_row("Avg Execution Time", avg_time)

            console.print(summary_table)
            console.print()

        # Today's Details
        if data["today"]["details"]:
            details_table = Table(title="Today's Collection Details", box=box.ROUNDED)
            details_table.add_column("Symbol", style="cyan")
            details_table.add_column("Type", style="dim")
            details_table.add_column("Status", style="bold")
            details_table.add_column("Started", style="dim")
            details_table.add_column("Records", style="green")
            details_table.add_column("Time (ms)", style="yellow")

            for row in data["today"]["details"]:
                status_style = (
                    "green"
                    if row["status"] == "success"
                    else "red" if row["status"] == "failed" else "yellow"
                )
                status_text = f"[{status_style}]{row['status']}[/{status_style}]"

                started = (
                    row["started_at"].strftime("%H:%M:%S") if row["started_at"] else "-"
                )
                records = (
                    str(row["records_collected"]) if row["records_collected"] else "-"
                )
                exec_time = (
                    f"{row['execution_time_ms']}" if row["execution_time_ms"] else "-"
                )

                details_table.add_row(
                    row["symbol"],
                    row["collection_type"],
                    status_text,
                    started,
                    records,
                    exec_time,
                )

            console.print(details_table)
            console.print()

        # Weekly Trend
        if data["weekly"]:
            trend_table = Table(title="7-Day Collection Trend", box=box.ROUNDED)
            trend_table.add_column("Date", style="cyan")
            trend_table.add_column("Symbols", style="dim")
            trend_table.add_column("Records", style="green")
            trend_table.add_column("Success", style="green")
            trend_table.add_column("Failed", style="red")
            trend_table.add_column("Quality", style="yellow")

            for row in data["weekly"]:
                trend_table.add_row(
                    str(row["collection_date"]),
                    str(row["symbols"]),
                    str(row["records"] or 0),
                    str(row["successes"]),
                    str(row["failures"]),
                    f"{row['avg_quality']:.2f}" if row["avg_quality"] else "-",
                )

            console.print(trend_table)
            console.print()

        # Active Contracts
        if data["contracts"]["by_symbol"]:
            contracts_table = Table(title="Active Option Contracts", box=box.ROUNDED)
            contracts_table.add_column("Symbol", style="cyan")
            contracts_table.add_column("Active Contracts", style="green")
            contracts_table.add_column("Unique Expiries", style="yellow")
            contracts_table.add_column("Nearest Expiry", style="dim")
            contracts_table.add_column("Furthest Expiry", style="dim")

            for row in data["contracts"]["by_symbol"]:
                contracts_table.add_row(
                    row["symbol"],
                    str(row["active_contracts"]),
                    str(row["unique_expiries"]),
                    str(row["nearest_expiry"]) if row["nearest_expiry"] else "-",
                    str(row["furthest_expiry"]) if row["furthest_expiry"] else "-",
                )

            if data["contracts"]["total"]:
                contracts_table.add_row(
                    "[bold]TOTAL[/bold]",
                    f"[bold]{data['contracts']['total']['total_contracts']}[/bold]",
                    f"[bold]{data['contracts']['total']['total_expiries']}[/bold]",
                    "-",
                    "-",
                )

            console.print(contracts_table)
            console.print()

        # Data Gaps
        if data["gaps"]:
            gaps_table = Table(title="Data Gaps (Needs Backfill)", box=box.ROUNDED)
            gaps_table.add_column("Symbol", style="cyan")
            gaps_table.add_column("Gap Date", style="yellow")
            gaps_table.add_column("Type", style="dim")
            gaps_table.add_column("Attempts", style="red")

            for row in data["gaps"][:10]:  # Show max 10 gaps
                gaps_table.add_row(
                    row["symbol"],
                    str(row["gap_date"]),
                    row["gap_type"],
                    str(row["backfill_attempts"]),
                )

            if len(data["gaps"]) > 10:
                gaps_table.add_row(
                    f"[dim]... and {len(data['gaps']) - 10} more[/dim]", "", "", ""
                )

            console.print(gaps_table)
            console.print()

        # Health Status
        if data["health"]:
            health_table = Table(title="System Health Status", box=box.ROUNDED)
            health_table.add_column("Symbol", style="cyan")
            health_table.add_column("Last Collection", style="dim")
            health_table.add_column("Hours Ago", style="yellow")
            health_table.add_column("Week Success %", style="green")
            health_table.add_column("Status", style="bold")

            for row in data["health"]:
                hours = (
                    f"{row['hours_since_collection']:.1f}"
                    if row["hours_since_collection"]
                    else "-"
                )
                week_rate = (
                    f"{row['week_success_rate']:.0f}%"
                    if row["week_success_rate"]
                    else "-"
                )

                status_style = {
                    "HEALTHY": "green",
                    "DEGRADED": "yellow",
                    "WARNING": "orange1",
                    "CRITICAL": "red",
                    "NO_DATA": "dim",
                }.get(row["health_status"], "white")

                status_text = f"[{status_style}]{row['health_status']}[/{status_style}]"

                last_collection = row["last_successful_collection"]
                if last_collection:
                    last_collection = last_collection.astimezone(
                        self.israel_tz
                    ).strftime("%Y-%m-%d %H:%M")
                else:
                    last_collection = "-"

                health_table.add_row(
                    row["symbol"], last_collection, hours, week_rate, status_text
                )

            console.print(health_table)

    async def run_check(self, watch: bool = False):
        """Run status check."""
        await self.connect_db()

        try:
            if watch:
                # Live update mode
                with Live(console=console, refresh_per_second=0.5) as live:
                    while True:
                        data = await self._collect_all_data()

                        # Clear and redraw
                        console.clear()
                        self.display_status(data)

                        # Wait before refresh
                        await asyncio.sleep(30)
            else:
                # Single check
                data = await self._collect_all_data()
                self.display_status(data)

        finally:
            await self.close_db()

    async def _collect_all_data(self) -> dict:
        """Collect all status data."""
        return {
            "today": await self.check_today_status(),
            "weekly": await self.check_weekly_trend(),
            "gaps": await self.check_data_gaps(),
            "contracts": await self.check_active_contracts(),
            "health": await self.check_collection_health(),
        }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check Daily Collection Status")
    parser.add_argument(
        "--config",
        default="scheduler/daily_schedule_israel.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode - auto-refresh every 30 seconds",
    )
    parser.add_argument(
        "--simple", action="store_true", help="Simple output without formatting"
    )

    args = parser.parse_args()

    checker = CollectionStatusChecker(args.config)
    await checker.run_check(args.watch)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Status check interrupted by user[/yellow]")
        sys.exit(0)
