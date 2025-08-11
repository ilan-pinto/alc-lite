#!/usr/bin/env python3
"""
Example script demonstrating how to use the SFR Backtesting Engine.

This script shows various backtesting scenarios with different configurations,
time periods, and analysis options.

Usage Examples:
    python run_sfr_backtest.py --period 1y --config conservative
    python run_sfr_backtest.py --start 2020-01-01 --end 2023-12-31 --config aggressive
    python run_sfr_backtest.py --period 5y --symbols SPY QQQ AAPL --vix-filter low
    python run_sfr_backtest.py --quick-test  # 30-day test run
"""

import asyncio
import os

# Import backtesting engine and configurations
import sys
from datetime import date, timedelta
from typing import List, Optional

import argparse
import asyncpg
import logging
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sfr_backtest_engine import (
    SFRBacktestConfig,
    SFRBacktestConfigs,
    SFRBacktestEngine,
    SlippageModel,
    VixRegime,
)

from backtesting.infra.data_collection.config.config import DatabaseConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

console = Console()


class SFRBacktestRunner:
    """Main class for running SFR backtests with various configurations."""

    def __init__(self):
        self.db_pool: Optional[asyncpg.Pool] = None
        self.db_config = DatabaseConfig()

    async def initialize(self):
        """Initialize database connection."""
        try:
            self.db_pool = await asyncpg.create_pool(
                self.db_config.connection_string, min_size=5, max_size=20
            )
            console.print("[green]✓[/green] Database connection established")
        except Exception as e:
            console.print(f"[red]✗[/red] Database connection failed: {e}")
            raise

    async def cleanup(self):
        """Clean up resources."""
        if self.db_pool:
            await self.db_pool.close()
            console.print("[blue]Database connection closed[/blue]")

    async def run_single_backtest(
        self,
        config: SFRBacktestConfig,
        start_date: date,
        end_date: date,
        symbols: List[str],
        description: str = "",
    ) -> dict:
        """Run a single backtest with progress tracking."""

        console.print(f"\n[bold blue]Starting SFR Backtest: {description}[/bold blue]")
        console.print(f"Period: {start_date} to {end_date}")
        console.print(f"Symbols: {', '.join(symbols)}")
        console.print(
            f"Config: Profit Target={config.profit_target}%, Cost Limit=${config.cost_limit}"
        )

        with Progress(console=console) as progress:
            task = progress.add_task("Running backtest...", total=100)

            try:
                # Initialize engine
                engine = SFRBacktestEngine(self.db_pool, config)
                progress.update(task, advance=10)

                # Run backtest
                results = await engine.run_backtest(
                    start_date=start_date, end_date=end_date, symbols=symbols
                )
                progress.update(task, advance=90)

                progress.update(task, completed=100)
                console.print("[green]✓[/green] Backtest completed successfully")

                return results

            except Exception as e:
                progress.update(task, completed=100)
                console.print(f"[red]✗[/red] Backtest failed: {e}")
                raise

    async def run_comparative_analysis(
        self, configs: dict, start_date: date, end_date: date, symbols: List[str]
    ) -> dict:
        """Run comparative analysis across multiple configurations."""

        console.print(f"\n[bold green]Comparative SFR Analysis[/bold green]")
        console.print(f"Comparing {len(configs)} configurations")
        console.print(f"Period: {start_date} to {end_date}")

        results = {}

        for config_name, config in configs.items():
            console.print(f"\n[yellow]Running {config_name} configuration...[/yellow]")

            try:
                result = await self.run_single_backtest(
                    config=config,
                    start_date=start_date,
                    end_date=end_date,
                    symbols=symbols,
                    description=config_name,
                )
                results[config_name] = result

            except Exception as e:
                console.print(f"[red]Failed {config_name}: {e}[/red]")
                continue

        # Display comparison table
        self._display_comparison_table(results)

        return results

    def _display_comparison_table(self, results: dict):
        """Display comparison table of backtest results."""

        table = Table(title="SFR Backtest Comparison")
        table.add_column("Configuration", style="cyan")
        table.add_column("Opportunities", justify="right")
        table.add_column("Trades", justify="right")
        table.add_column("Success Rate", justify="right")
        table.add_column("Net Profit", justify="right", style="green")
        table.add_column("Avg ROI", justify="right")
        table.add_column("Sharpe Ratio", justify="right")
        table.add_column("Max Drawdown", justify="right", style="red")

        for config_name, result in results.items():
            table.add_row(
                config_name,
                str(result["opportunities"]["total_found"]),
                str(result["trades"]["total_simulated"]),
                result["trades"]["success_rate"],
                f"${result['profitability']['total_net_profit']:.2f}",
                result["roi_metrics"]["avg_min_roi"],
                f"{result['risk_metrics']['sharpe_ratio']:.2f}",
                result["risk_metrics"]["max_drawdown_percent"],
            )

        console.print(table)

    async def run_vix_regime_analysis(
        self,
        base_config: SFRBacktestConfig,
        start_date: date,
        end_date: date,
        symbols: List[str],
    ) -> dict:
        """Run analysis across different VIX regimes."""

        console.print(f"\n[bold magenta]VIX Regime Analysis[/bold magenta]")

        # Create configs for different VIX regimes
        vix_configs = {
            "All Regimes": base_config,
            "Low VIX": SFRBacktestConfig(
                **base_config.__dict__,
                vix_regime_filter=VixRegime.LOW,
                max_vix_level=20.0,
            ),
            "Medium VIX": SFRBacktestConfig(
                **base_config.__dict__,
                vix_regime_filter=VixRegime.MEDIUM,
                min_vix_level=15.0,
                max_vix_level=25.0,
            ),
            "High VIX": SFRBacktestConfig(
                **base_config.__dict__,
                vix_regime_filter=VixRegime.HIGH,
                min_vix_level=25.0,
            ),
        }

        return await self.run_comparative_analysis(
            vix_configs, start_date, end_date, symbols
        )

    async def run_slippage_model_analysis(
        self,
        base_config: SFRBacktestConfig,
        start_date: date,
        end_date: date,
        symbols: List[str],
    ) -> dict:
        """Run analysis across different slippage models."""

        console.print(f"\n[bold cyan]Slippage Model Analysis[/bold cyan]")

        # Create configs for different slippage models
        slippage_configs = {
            "No Slippage": SFRBacktestConfig(
                **base_config.__dict__, slippage_model=SlippageModel.NONE
            ),
            "Linear Slippage": SFRBacktestConfig(
                **base_config.__dict__, slippage_model=SlippageModel.LINEAR
            ),
            "Square Root": SFRBacktestConfig(
                **base_config.__dict__, slippage_model=SlippageModel.SQUARE_ROOT
            ),
            "Market Impact": SFRBacktestConfig(
                **base_config.__dict__, slippage_model=SlippageModel.IMPACT
            ),
        }

        return await self.run_comparative_analysis(
            slippage_configs, start_date, end_date, symbols
        )

    def display_detailed_results(self, results: dict):
        """Display detailed results for a single backtest."""

        rprint(f"\n[bold green]Detailed Backtest Results[/bold green]")
        rprint(f"Backtest Run ID: {results['backtest_run_id']}")

        # Period Information
        period_table = Table(title="Backtest Period")
        period_table.add_column("Metric", style="cyan")
        period_table.add_column("Value", justify="right")

        period_table.add_row("Start Date", str(results["period"]["start_date"]))
        period_table.add_row("End Date", str(results["period"]["end_date"]))
        period_table.add_row("Total Days", str(results["period"]["total_days"]))
        period_table.add_row("Symbols", ", ".join(results["symbol_coverage"]))

        console.print(period_table)

        # Configuration
        config_table = Table(title="Configuration")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", justify="right")

        config_table.add_row("Profit Target", f"{results['config']['profit_target']}%")
        config_table.add_row("Cost Limit", f"${results['config']['cost_limit']}")
        config_table.add_row("Slippage Model", results["config"]["slippage_model"])
        config_table.add_row(
            "Commission/Contract", f"${results['config']['commission_per_contract']}"
        )

        console.print(config_table)

        # Opportunity Metrics
        opp_table = Table(title="Opportunity Discovery")
        opp_table.add_column("Metric", style="cyan")
        opp_table.add_column("Value", justify="right")

        opp_table.add_row(
            "Total Opportunities", str(results["opportunities"]["total_found"])
        )
        opp_table.add_row("Per Day", f"{results['opportunities']['per_day']:.2f}")

        # Quality breakdown
        for quality, count in results["opportunities"]["by_quality"].items():
            opp_table.add_row(f"  {quality.title()}", str(count))

        console.print(opp_table)

        # Trading Results
        trade_table = Table(title="Trading Performance")
        trade_table.add_column("Metric", style="cyan")
        trade_table.add_column("Value", justify="right")

        trade_table.add_row("Total Trades", str(results["trades"]["total_simulated"]))
        trade_table.add_row("Successful", str(results["trades"]["successful"]))
        trade_table.add_row("Failed", str(results["trades"]["failed"]))
        trade_table.add_row("Success Rate", results["trades"]["success_rate"])

        console.print(trade_table)

        # Financial Results
        profit_table = Table(title="Financial Performance")
        profit_table.add_column("Metric", style="cyan")
        profit_table.add_column("Value", justify="right")

        profit_table.add_row(
            "Gross Profit", f"${results['profitability']['total_gross_profit']:.2f}"
        )
        profit_table.add_row(
            "Net Profit",
            f"${results['profitability']['total_net_profit']:.2f}",
            style=(
                "green" if results["profitability"]["total_net_profit"] > 0 else "red"
            ),
        )
        profit_table.add_row(
            "Total Commissions", f"${results['profitability']['total_commissions']:.2f}"
        )
        profit_table.add_row(
            "Total Slippage Cost",
            f"${results['profitability']['total_slippage_cost']:.2f}",
        )
        profit_table.add_row(
            "Avg Profit/Trade",
            f"${results['profitability']['avg_profit_per_trade']:.2f}",
        )
        profit_table.add_row(
            "Median Profit/Trade",
            f"${results['profitability']['median_profit_per_trade']:.2f}",
        )

        console.print(profit_table)

        # ROI Metrics
        roi_table = Table(title="Return Metrics")
        roi_table.add_column("Metric", style="cyan")
        roi_table.add_column("Value", justify="right")

        roi_table.add_row("Average ROI", results["roi_metrics"]["avg_min_roi"])
        roi_table.add_row("Median ROI", results["roi_metrics"]["median_min_roi"])
        roi_table.add_row("Best ROI", results["roi_metrics"]["best_min_roi"])
        roi_table.add_row("Worst ROI", results["roi_metrics"]["worst_min_roi"])
        roi_table.add_row("ROI Std Dev", results["roi_metrics"]["roi_std_dev"])

        console.print(roi_table)

        # Risk Metrics
        risk_table = Table(title="Risk Metrics")
        risk_table.add_column("Metric", style="cyan")
        risk_table.add_column("Value", justify="right")

        risk_table.add_row(
            "Max Single Loss", f"${results['risk_metrics']['max_single_loss']:.2f}"
        )
        risk_table.add_row(
            "Max Drawdown", f"${results['risk_metrics']['max_drawdown']:.2f}"
        )
        risk_table.add_row(
            "Max Drawdown %", results["risk_metrics"]["max_drawdown_percent"]
        )
        risk_table.add_row(
            "Sharpe Ratio", f"{results['risk_metrics']['sharpe_ratio']:.2f}"
        )

        console.print(risk_table)


async def main():
    """Main execution function."""

    parser = argparse.ArgumentParser(
        description="Run SFR Arbitrage Strategy Backtests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --period 1y --config conservative
  %(prog)s --start 2020-01-01 --end 2023-12-31 --config aggressive
  %(prog)s --period 5y --symbols SPY QQQ AAPL --analysis vix
  %(prog)s --quick-test
        """,
    )

    # Date range options
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--period",
        choices=["30d", "3m", "6m", "1y", "3y", "5y", "10y"],
        help="Predefined time period for backtesting",
    )
    date_group.add_argument(
        "--start", type=lambda s: date.fromisoformat(s), help="Start date (YYYY-MM-DD)"
    )
    date_group.add_argument(
        "--quick-test", action="store_true", help="Run quick 30-day test (last month)"
    )

    parser.add_argument(
        "--end",
        type=lambda s: date.fromisoformat(s),
        help="End date (YYYY-MM-DD), defaults to today",
    )

    # Configuration options
    parser.add_argument(
        "--config",
        choices=["conservative", "aggressive", "low_vix", "high_vix", "custom"],
        default="conservative",
        help="Configuration preset to use",
    )

    # Symbol selection
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=[
            "SPY",
            "QQQ",
            "AAPL",
            "MSFT",
            "NVDA",
            "TSLA",
            "AMZN",
            "META",
            "GOOGL",
            "JPM",
        ],
        help="Symbols to backtest (default: top 10 target symbols)",
    )

    # Analysis options
    parser.add_argument(
        "--analysis",
        choices=["single", "comparative", "vix", "slippage"],
        default="single",
        help="Type of analysis to run",
    )

    # Custom parameters
    parser.add_argument("--profit-target", type=float, help="Custom profit target %%")
    parser.add_argument("--cost-limit", type=float, help="Custom cost limit $")
    parser.add_argument("--commission", type=float, help="Commission per contract $")
    parser.add_argument(
        "--slippage-bps", type=int, help="Base slippage in basis points"
    )

    # Output options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--export", help="Export results to JSON file")

    args = parser.parse_args()

    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Calculate date range
    end_date = args.end or date.today()

    if args.quick_test:
        start_date = end_date - timedelta(days=30)
    elif args.start:
        start_date = args.start
    else:
        # Parse period
        period_map = {
            "30d": 30,
            "3m": 90,
            "6m": 180,
            "1y": 365,
            "3y": 1095,
            "5y": 1825,
            "10y": 3650,
        }
        days_back = period_map[args.period]
        start_date = end_date - timedelta(days=days_back)

    # Get configuration
    config_map = {
        "conservative": SFRBacktestConfigs.conservative_config(),
        "aggressive": SFRBacktestConfigs.aggressive_config(),
        "low_vix": SFRBacktestConfigs.low_vix_config(),
        "high_vix": SFRBacktestConfigs.high_vix_config(),
    }

    if args.config == "custom":
        base_config = SFRBacktestConfig()
        if args.profit_target:
            base_config.profit_target = args.profit_target
        if args.cost_limit:
            base_config.cost_limit = args.cost_limit
        if args.commission:
            base_config.commission_per_contract = args.commission
        if args.slippage_bps:
            base_config.base_slippage_bps = args.slippage_bps
        config = base_config
    else:
        config = config_map[args.config]

    # Initialize runner
    runner = SFRBacktestRunner()

    try:
        await runner.initialize()

        # Run analysis based on type
        if args.analysis == "single":
            results = await runner.run_single_backtest(
                config=config,
                start_date=start_date,
                end_date=end_date,
                symbols=args.symbols,
                description=f"{args.config} configuration",
            )
            runner.display_detailed_results(results)

        elif args.analysis == "comparative":
            configs = {
                "Conservative": SFRBacktestConfigs.conservative_config(),
                "Aggressive": SFRBacktestConfigs.aggressive_config(),
                "Low VIX": SFRBacktestConfigs.low_vix_config(),
                "High VIX": SFRBacktestConfigs.high_vix_config(),
            }
            results = await runner.run_comparative_analysis(
                configs, start_date, end_date, args.symbols
            )

        elif args.analysis == "vix":
            results = await runner.run_vix_regime_analysis(
                config, start_date, end_date, args.symbols
            )

        elif args.analysis == "slippage":
            results = await runner.run_slippage_model_analysis(
                config, start_date, end_date, args.symbols
            )

        # Export results if requested
        if args.export:
            import json

            with open(args.export, "w") as f:
                json.dump(results, f, indent=2, default=str)
            console.print(f"[green]Results exported to {args.export}[/green]")

        console.print("\n[bold green]Backtesting completed successfully![/bold green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Backtesting interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error during backtesting: {e}[/red]")
        if args.verbose:
            import traceback

            traceback.print_exc()
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
