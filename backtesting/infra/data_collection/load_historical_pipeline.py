#!/usr/bin/env python3
"""
Historical Data Loading Pipeline

A comprehensive pipeline script for loading historical options and stock data
from Interactive Brokers into the backtesting database.

Features:
- Command-line interface with flexible parameters
- Multi-symbol batch processing
- Progress tracking and statistics
- Data validation and quality checks
- Backfill missing data gaps
- Configurable logging and error handling
- Resume interrupted operations

Usage:
    python load_historical_pipeline.py --symbol SPY --days 30
    python load_historical_pipeline.py --symbols SPY,QQQ,AAPL --start 2024-01-01 --end 2024-01-31
    python load_historical_pipeline.py --backfill --symbols SPY --days 90

Author: AlcLite Trading System
Created: 2025-08-04
"""

import asyncio
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to Python path for absolute imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import asyncpg
import logging
import yaml
from ib_async import IB
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

try:
    # Try relative imports first (when used as module)
    from .config.config import CollectionConfig, HistoricalConfig
    from .core.collector import OptionsDataCollector
    from .core.historical_loader import HistoricalDataLoader
    from .core.validators import DataValidator
    from .core.vix_collector import VIXDataCollector
except ImportError:
    # Fall back to absolute imports (when run directly)
    from backtesting.infra.data_collection.config.config import (
        CollectionConfig,
        HistoricalConfig,
    )
    from backtesting.infra.data_collection.core.collector import OptionsDataCollector
    from backtesting.infra.data_collection.core.historical_loader import (
        HistoricalDataLoader,
    )
    from backtesting.infra.data_collection.core.validators import DataValidator
    from backtesting.infra.data_collection.core.vix_collector import VIXDataCollector

# Global console for rich output
console = Console()


class PipelineConfig:
    """Configuration class for the data loading pipeline."""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        """Load configuration from file and environment variables."""
        # Default configuration
        self.defaults = {
            "database": {
                "host": "localhost",
                "port": 5433,
                "database": "options_arbitrage",
                "user": "trading_user",
                "password": "secure_trading_password",
                "min_pool_size": 5,
                "max_pool_size": 20,
            },
            "ib_connection": {
                "host": "127.0.0.1",
                "port": 7497,
                "client_id": 1,
                "timeout": 30,
            },
            "loading": {
                "batch_size": 10,
                "request_delay": 0.5,
                "retry_attempts": 3,
                "max_days_per_request": 30,
                "bar_size": "1 min",
                "what_to_show": "MIDPOINT",
                "use_rth": True,
            },
            "validation": {
                "enabled": True,
                "max_spread_percent": 0.1,
                "quality_threshold": 0.5,
            },
            "logging": {
                "level": "INFO",
                "file": None,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        }

        # Load from file if specified
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, "r") as f:
                    if self.config_file.endswith(".yaml") or self.config_file.endswith(
                        ".yml"
                    ):
                        file_config = yaml.safe_load(f)
                    else:
                        file_config = json.load(f)

                # Merge with defaults
                self._merge_config(self.defaults, file_config)
            except Exception as e:
                console.print(f"[red]Error loading config file: {e}[/red]")
                sys.exit(1)

    def _merge_config(self, base: dict, override: dict):
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def get(self, section: str, key: str, default=None):
        """Get configuration value."""
        return self.defaults.get(section, {}).get(key, default)


class HistoricalDataPipeline:
    """Main pipeline class for historical data loading."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.db_pool = None
        self.ib = None
        self.loader = None
        self.validator = None
        self.options_collector = None
        self.vix_collector = None
        self.progress = None
        self.stats = {
            "symbols_processed": 0,
            "symbols_failed": 0,
            "total_stock_bars": 0,
            "total_option_bars": 0,
            "total_option_chains": 0,
            "total_vix_records": 0,
            "total_errors": 0,
            "start_time": None,
            "end_time": None,
        }

    async def initialize(self):
        """Initialize database and IB connections."""
        console.print("[blue]Initializing pipeline...[/blue]")

        try:
            # Database connection
            db_config = self.config.defaults["database"]
            self.db_pool = await asyncpg.create_pool(
                host=db_config["host"],
                port=db_config["port"],
                database=db_config["database"],
                user=db_config["user"],
                password=db_config["password"],
                min_size=db_config["min_pool_size"],
                max_size=db_config["max_pool_size"],
            )
            console.print("[green]✓ Database connection established[/green]")

            # IB connection
            ib_config = self.config.defaults["ib_connection"]
            self.ib = IB()
            await self.ib.connectAsync(
                host=ib_config["host"],
                port=ib_config["port"],
                clientId=ib_config["client_id"],
                timeout=ib_config["timeout"],
            )
            console.print("[green]✓ Interactive Brokers connection established[/green]")

            # Initialize components
            historical_config = HistoricalConfig()
            loading_config = self.config.defaults["loading"]
            historical_config.request_delay_seconds = loading_config["request_delay"]
            historical_config.retry_attempts = loading_config["retry_attempts"]
            historical_config.max_days_per_request = loading_config[
                "max_days_per_request"
            ]
            historical_config.bar_size = loading_config["bar_size"]
            historical_config.what_to_show = loading_config["what_to_show"]
            historical_config.use_rth = loading_config["use_rth"]

            self.loader = HistoricalDataLoader(self.db_pool, self.ib, historical_config)
            self.validator = DataValidator(self.db_pool)

            # Initialize options collector
            collection_config = CollectionConfig()
            self.options_collector = OptionsDataCollector(
                self.db_pool, self.ib, collection_config
            )

            # Initialize VIX collector
            self.vix_collector = VIXDataCollector(self.db_pool, self.ib)
            await self.vix_collector.initialize()

            console.print("[green]✓ Pipeline initialized successfully[/green]")

        except Exception as e:
            console.print(f"[red]Initialization failed: {e}[/red]")
            raise

    async def cleanup(self):
        """Clean up connections and resources."""
        console.print("[blue]Cleaning up...[/blue]")

        if self.ib and self.ib.isConnected():
            self.ib.disconnect()

        if self.db_pool:
            await self.db_pool.close()

        console.print("[green]✓ Cleanup completed[/green]")

    async def load_historical_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        validate: bool = True,
    ) -> Dict:
        """Load historical data for specified symbols and date range."""
        self.stats["start_time"] = datetime.now()
        console.print(
            f"[blue]Loading historical data for {len(symbols)} symbols "
            f"from {start_date} to {end_date}[/blue]"
        )

        # Initialize progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ) as progress:
            self.progress = progress
            main_task = progress.add_task(
                "Loading symbols...", total=len(symbols), completed=0
            )

            # Process each symbol
            for symbol in symbols:
                symbol_task = progress.add_task(
                    f"Processing {symbol}...", total=1, completed=0
                )

                try:
                    # Load data for this symbol
                    symbol_stats = await self.loader.load_symbol_history(
                        symbol, start_date, end_date
                    )

                    # Update statistics
                    self.stats["symbols_processed"] += 1
                    self.stats["total_stock_bars"] += symbol_stats["stock_bars_loaded"]
                    self.stats["total_option_bars"] += symbol_stats[
                        "option_bars_loaded"
                    ]
                    self.stats["total_errors"] += symbol_stats["errors"]

                    # Validate data if requested
                    if validate and self.config.get("validation", "enabled", True):
                        await self._validate_symbol_data(symbol, start_date, end_date)

                    progress.update(symbol_task, completed=1)
                    console.print(
                        f"[green]✓ {symbol}: {symbol_stats['stock_bars_loaded']} stock bars, "
                        f"{symbol_stats['option_bars_loaded']} option bars[/green]"
                    )

                except Exception as e:
                    self.stats["symbols_failed"] += 1
                    self.stats["total_errors"] += 1
                    progress.update(symbol_task, completed=1)
                    console.print(f"[red]✗ {symbol}: {e}[/red]")
                    logging.error(f"Failed to load {symbol}: {e}")

                progress.update(main_task, advance=1)

        self.stats["end_time"] = datetime.now()
        return self.stats

    async def backfill_missing_data(
        self, symbols: List[str], days_back: int = 30
    ) -> Dict:
        """Identify and backfill missing data for symbols."""
        console.print(
            f"[blue]Backfilling missing data for {len(symbols)} symbols "
            f"(last {days_back} days)[/blue]"
        )

        self.stats["start_time"] = datetime.now()
        backfill_stats = {"symbols_checked": 0, "gaps_filled": 0, "errors": 0}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            main_task = progress.add_task(
                "Checking for gaps...", total=len(symbols), completed=0
            )

            for symbol in symbols:
                try:
                    # Check and backfill missing data
                    await self.loader.backfill_missing_data(symbol, days_back)
                    backfill_stats["symbols_checked"] += 1
                    console.print(f"[green]✓ {symbol}: Gaps checked and filled[/green]")

                except Exception as e:
                    backfill_stats["errors"] += 1
                    console.print(f"[red]✗ {symbol}: {e}[/red]")
                    logging.error(f"Failed to backfill {symbol}: {e}")

                progress.update(main_task, advance=1)

        self.stats["end_time"] = datetime.now()
        return {**self.stats, **backfill_stats}

    async def load_complete_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        include_options: bool = True,
        include_vix: bool = False,
        validate: bool = True,
        option_expiry_days: int = 60,
        option_strike_range: float = 0.20,
        parallel_contracts: int = 5,
    ) -> Dict:
        """
        Load complete historical data including stocks, options, and VIX.

        Args:
            symbols: List of stock symbols to load
            start_date: Start date for historical data
            end_date: End date for historical data
            include_options: Whether to fetch and load option chains
            include_vix: Whether to load VIX data
            validate: Whether to validate loaded data
            option_expiry_days: Max days ahead for option expiries
            option_strike_range: Strike price range as percent of stock price
            parallel_contracts: Number of option contracts to load in parallel

        Returns:
            Dictionary with loading statistics
        """
        self.stats["start_time"] = datetime.now()
        console.print(
            f"[blue]Loading complete data for {len(symbols)} symbols "
            f"from {start_date} to {end_date}[/blue]"
        )

        # Initialize progress tracking with phases
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ) as progress:
            self.progress = progress

            # Calculate total phases per symbol
            phases_per_symbol = 1  # Stock data
            if include_options:
                phases_per_symbol += 2  # Option chains + historical
            if include_vix:
                phases_per_symbol += 1  # VIX data

            main_task = progress.add_task(
                "Processing symbols...",
                total=len(symbols) * phases_per_symbol,
                completed=0,
            )

            # Process each symbol
            for symbol in symbols:
                try:
                    console.print(f"\n[bold blue]Processing {symbol}[/bold blue]")

                    # Phase 1: Load stock data
                    stock_task = progress.add_task(
                        f"{symbol}: Loading stock data...", total=1, completed=0
                    )

                    stock_bars = await self.loader._load_stock_history(
                        symbol, start_date, end_date
                    )
                    self.stats["total_stock_bars"] += stock_bars
                    progress.update(stock_task, completed=1)
                    progress.update(main_task, advance=1)
                    console.print(
                        f"[green]✓ {symbol}: {stock_bars} stock bars loaded[/green]"
                    )

                    # Phase 2: Fetch and store option chains (if requested)
                    if include_options:
                        options_task = progress.add_task(
                            f"{symbol}: Discovering option chains...",
                            total=1,
                            completed=0,
                        )

                        # Configure options collector
                        self.options_collector.config.expiry_range_days = (
                            option_expiry_days
                        )
                        self.options_collector.config.strike_range_percent = (
                            option_strike_range
                        )

                        # Initialize option contracts for the symbol
                        await self.options_collector._initialize_symbol_contracts(
                            symbol
                        )

                        progress.update(options_task, completed=1)
                        progress.update(main_task, advance=1)

                        # Phase 3: Load historical option data
                        option_hist_task = progress.add_task(
                            f"{symbol}: Loading option history...", total=1, completed=0
                        )

                        # Get option contracts from database
                        option_contracts = (
                            await self.loader._get_historical_option_contracts(
                                symbol, start_date, end_date
                            )
                        )

                        # Load historical data for options
                        option_bars = 0

                        # Create detailed progress task for option contracts
                        contracts_task = progress.add_task(
                            f"{symbol}: Loading 0/{len(option_contracts)} option contracts...",
                            total=len(option_contracts),
                            completed=0,
                        )

                        contracts_loaded = 0
                        # Use dynamic batch size based on parallel workers
                        batch_size = min(
                            parallel_contracts * 2, 20
                        )  # 2x parallel workers, max 20
                        for contract_batch in self.loader._batch_contracts(
                            option_contracts, batch_size
                        ):
                            # Create batch progress task
                            batch_task = progress.add_task(
                                f"Batch: 0/{len(contract_batch)} contracts",
                                total=len(contract_batch),
                                completed=0,
                            )

                            # Use parallel loading for better performance
                            batch_bars = (
                                await self.loader._load_option_batch_history_parallel(
                                    contract_batch,
                                    start_date,
                                    end_date,
                                    progress,
                                    batch_task,
                                    max_concurrent=parallel_contracts,
                                )
                            )
                            option_bars += batch_bars
                            contracts_loaded += len(contract_batch)

                            # Update overall contracts progress
                            progress.update(
                                contracts_task,
                                description=f"{symbol}: Loaded {contracts_loaded}/{len(option_contracts)} option contracts",
                                completed=contracts_loaded,
                            )

                            # Remove the batch task as it's complete
                            progress.remove_task(batch_task)

                        self.stats["total_option_bars"] += option_bars
                        self.stats["total_option_chains"] += len(option_contracts)

                        # Clean up the detailed contracts progress
                        progress.remove_task(contracts_task)

                        progress.update(option_hist_task, completed=1)
                        progress.update(main_task, advance=1)
                        console.print(
                            f"[green]✓ {symbol}: {len(option_contracts)} option chains, "
                            f"{option_bars} option bars loaded[/green]"
                        )

                    # Phase 4: Load VIX data (if requested)
                    if include_vix and symbol in [
                        "SPY",
                        "SPX",
                    ]:  # VIX typically correlates with S&P
                        vix_task = progress.add_task(
                            f"{symbol}: Loading VIX data...", total=1, completed=0
                        )

                        # Calculate days for VIX historical collection
                        days_back = (end_date - start_date).days
                        vix_stats = await self.vix_collector.collect_historical_data(
                            days_back
                        )

                        # Sum up records from all VIX instruments
                        total_vix_records = sum(vix_stats.values())
                        self.stats["total_vix_records"] += total_vix_records

                        progress.update(vix_task, completed=1)
                        progress.update(main_task, advance=1)
                        console.print(
                            f"[green]✓ VIX: {total_vix_records} records loaded across {len(vix_stats)} instruments[/green]"
                        )

                    # Validate data if requested
                    if validate and self.config.get("validation", "enabled", True):
                        await self._validate_symbol_data(symbol, start_date, end_date)

                    self.stats["symbols_processed"] += 1

                except Exception as e:
                    self.stats["symbols_failed"] += 1
                    self.stats["total_errors"] += 1
                    console.print(f"[red]✗ {symbol}: {e}[/red]")
                    logging.error(f"Failed to load complete data for {symbol}: {e}")

                    # Advance progress for remaining phases of failed symbol
                    remaining_phases = phases_per_symbol - (
                        progress.tasks[main_task].completed % phases_per_symbol
                    )
                    if remaining_phases > 0:
                        progress.update(main_task, advance=remaining_phases)

        self.stats["end_time"] = datetime.now()
        return self.stats

    async def _validate_symbol_data(
        self, symbol: str, start_date: date, end_date: date
    ):
        """Validate data quality for a symbol."""
        try:
            # Run quality check for the period
            hours_back = (end_date - start_date).days * 24
            report = await self.validator.run_quality_check(hours_back=hours_back)

            if report["price_data_coverage"] < 80:
                console.print(
                    f"[yellow]⚠ {symbol}: Low data coverage "
                    f"({report['price_data_coverage']:.1f}%)[/yellow]"
                )

        except Exception as e:
            logging.warning(f"Validation failed for {symbol}: {e}")

    def print_summary(self, stats: Dict):
        """Print pipeline execution summary."""
        console.print("\n[bold blue]Pipeline Execution Summary[/bold blue]")

        # Create summary table
        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        # Add rows
        duration = (
            (stats["end_time"] - stats["start_time"]).total_seconds()
            if stats["start_time"] and stats["end_time"]
            else 0
        )

        summary_table.add_row(
            "Symbols Processed", str(stats.get("symbols_processed", 0))
        )
        summary_table.add_row("Symbols Failed", str(stats.get("symbols_failed", 0)))
        summary_table.add_row(
            "Stock Bars Loaded", str(stats.get("total_stock_bars", 0))
        )
        summary_table.add_row(
            "Option Bars Loaded", str(stats.get("total_option_bars", 0))
        )
        if stats.get("total_option_chains", 0) > 0:
            summary_table.add_row(
                "Option Chains Discovered", str(stats.get("total_option_chains", 0))
            )
        if stats.get("total_vix_records", 0) > 0:
            summary_table.add_row(
                "VIX Records Loaded", str(stats.get("total_vix_records", 0))
            )
        summary_table.add_row("Total Errors", str(stats.get("total_errors", 0)))
        summary_table.add_row("Duration", f"{duration:.1f} seconds")

        if "symbols_checked" in stats:
            summary_table.add_row("Symbols Checked", str(stats["symbols_checked"]))
            summary_table.add_row("Gaps Filled", str(stats.get("gaps_filled", 0)))

        console.print(summary_table)

        # Loader statistics
        if self.loader:
            loader_stats = self.loader.get_stats()
            console.print(f"\n[bold blue]Loader Statistics[/bold blue]")
            console.print(f"Completed Requests: {loader_stats['completed_requests']}")
            console.print(f"Failed Requests: {loader_stats['failed_requests']}")

        # Validation statistics
        if self.validator and self.config.get("validation", "enabled", True):
            validation_stats = self.validator.get_stats()
            console.print(f"\n[bold blue]Data Validation[/bold blue]")
            console.print(f"Records Validated: {validation_stats['total_validated']}")
            console.print(f"Valid: {validation_stats.get('valid_pct', 0):.1f}%")
            console.print(f"Warnings: {validation_stats.get('warning_pct', 0):.1f}%")
            console.print(f"Invalid: {validation_stats.get('invalid_pct', 0):.1f}%")


def setup_logging(config: PipelineConfig):
    """Set up logging configuration."""
    log_level = getattr(logging, config.get("logging", "level", "INFO"))
    log_format = config.get(
        "logging",
        "format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log_file = config.get("logging", "file")

    # Configure root logger
    handlers = [RichHandler(console=console, rich_tracebacks=True)]

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level, handlers=handlers, format=log_format, force=True
    )

    # Suppress noisy loggers
    logging.getLogger("ib_async").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def parse_date(date_str: str) -> date:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}")


def parse_symbols(symbols_str: str) -> List[str]:
    """Parse comma-separated symbols string."""
    return [s.strip().upper() for s in symbols_str.split(",") if s.strip()]


async def main():
    """Main pipeline entry point."""
    parser = argparse.ArgumentParser(
        description="Historical Data Loading Pipeline for Options Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete data load (stock + options + VIX)
  %(prog)s --symbol SPY --days 30 --include-vix

  # Multiple symbols with custom option filtering
  %(prog)s --symbols SPY,QQQ,AAPL --days 30 --option-expiry-days 45 --option-strike-range 0.15

  # Stock data only (skip options)
  %(prog)s --symbol TSLA --days 60 --skip-options

  # Backfill missing data
  %(prog)s --backfill --symbols SPY --days 90 --validate

  # Custom configuration
  %(prog)s --config my_config.yaml --symbol TSLA --days 60 --log pipeline.log
        """,
    )

    # Symbol selection
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument("--symbol", help="Single symbol to load")
    symbol_group.add_argument(
        "--symbols", type=parse_symbols, help="Comma-separated list of symbols"
    )

    # Date range
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--days", type=int, help="Number of days back from today to load"
    )
    date_group.add_argument("--start", type=parse_date, help="Start date (YYYY-MM-DD)")

    parser.add_argument("--end", type=parse_date, help="End date (YYYY-MM-DD)")

    # Operation mode
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill missing data instead of full load",
    )
    parser.add_argument(
        "--complete",
        action="store_true",
        help="Load complete data (stock + options + VIX)",
    )

    # Options configuration
    parser.add_argument(
        "--skip-options",
        action="store_true",
        help="Skip option chain discovery and loading",
    )
    parser.add_argument(
        "--include-vix",
        action="store_true",
        help="Include VIX data loading (for SPY/SPX)",
    )
    parser.add_argument(
        "--option-expiry-days",
        type=int,
        default=60,
        help="Max days ahead for option expiries (default: 60)",
    )
    parser.add_argument(
        "--option-strike-range",
        type=float,
        default=0.20,
        help="Strike price range as percent of stock price (default: 0.20)",
    )
    parser.add_argument(
        "--parallel-contracts",
        type=int,
        default=5,
        help="Number of option contracts to load in parallel (default: 5)",
    )

    # Configuration
    parser.add_argument("--config", help="Configuration file (YAML or JSON)")
    parser.add_argument(
        "--validate", action="store_true", help="Enable data validation"
    )
    parser.add_argument("--log", help="Log file path")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode (errors only)"
    )

    args = parser.parse_args()

    # Load configuration
    config = PipelineConfig(args.config)

    # Override logging settings from command line
    if args.verbose:
        config.defaults["logging"]["level"] = "DEBUG"
    elif args.quiet:
        config.defaults["logging"]["level"] = "ERROR"

    if args.log:
        config.defaults["logging"]["file"] = args.log

    # Set up logging
    setup_logging(config)

    # Determine symbols list
    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = args.symbols

    # Determine date range
    if args.days:
        end_date = date.today()
        start_date = end_date - timedelta(days=args.days)
    elif args.start:
        start_date = args.start
        end_date = args.end or date.today()
    else:
        # Default to last 30 days
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

    # Initialize and run pipeline
    pipeline = HistoricalDataPipeline(config)

    try:
        await pipeline.initialize()

        if args.backfill:
            days_back = args.days or (end_date - start_date).days
            stats = await pipeline.backfill_missing_data(symbols, days_back)
        elif args.complete or (not args.skip_options):
            # Use complete data loading by default or when explicitly requested
            stats = await pipeline.load_complete_data(
                symbols,
                start_date,
                end_date,
                include_options=not args.skip_options,
                include_vix=args.include_vix,
                validate=args.validate,
                option_expiry_days=args.option_expiry_days,
                option_strike_range=args.option_strike_range,
                parallel_contracts=args.parallel_contracts,
            )
        else:
            # Legacy mode: stock data only
            stats = await pipeline.load_historical_data(
                symbols, start_date, end_date, args.validate
            )

        pipeline.print_summary(stats)

        # Exit with appropriate code
        if stats.get("symbols_failed", 0) > 0 or stats.get("total_errors", 0) > 0:
            console.print("[yellow]Pipeline completed with errors[/yellow]")
            sys.exit(1)
        else:
            console.print("[green]Pipeline completed successfully[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")
        logging.exception("Pipeline execution failed")
        sys.exit(1)
    finally:
        await pipeline.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(130)
