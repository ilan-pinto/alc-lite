#!/usr/bin/env python3

import argparse
import pyfiglet
import warnings

from commands.option import OptionScan
from modules.Arbitrage.common import configure_logging
from modules.Arbitrage.metrics import metrics_collector
from modules.welcome import print_welcome

warnings.simplefilter(action="ignore", category=FutureWarning)
from rich.console import Console
from rich.theme import Theme

# Version information
__version__ = "1.12.0"

# Custom theme for log levels
custom_theme = Theme(
    {
        "debug": "dim cyan",
        "info": "green",
        "warning": "yellow",
        "error": "red",
        "critical": "bold red",
    }
)

console = Console(theme=custom_theme)


# const
DEFAULT_REPORT_FOLDER = "~/dev/AlchimistProject/alchimest/report"
DEFAULT_MIN_PROFIT = 0.5


def main() -> None:
    """Main function for CLI execution"""

    print_welcome(console, __version__, DEFAULT_MIN_PROFIT)

    parser = argparse.ArgumentParser(
        description="Stock market analysis and trading tool with multiple features including scanning, analysis, and option strategies. "
        "Now featuring Global Opportunity Selection for intelligent cross-symbol arbitrage ranking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Examples:\n"
        "  Basic SFR scan:\n"
        "    %(prog)s sfr --symbols SPY QQQ --cost-limit 100 --profit 0.75\n\n"
        "  Synthetic scan with global selection:\n"
        "    %(prog)s syn --symbols AAPL MSFT GOOGL --scoring-strategy balanced\n\n"
        "  Custom scoring weights:\n"
        "    %(prog)s syn --symbols SPY QQQ --risk-reward-weight 0.5 --liquidity-weight 0.3\n\n"
        "For more examples and documentation, visit: https://github.com/ilpinto/alc-lite",
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # Synthetic free arbitrage sub-command
    parser_sfr = subparsers.add_parser(
        "sfr",
        help="Search for synthetic risk free arbitrage opportunities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_sfr.add_argument(
        "-s", "--symbols", nargs="+", help="List of symbols to scan (e.g., !MES, @SPX)"
    )
    parser_sfr.add_argument(
        "-p",
        "--profit",
        type=float,
        default=None,
        required=False,
        help=f"Minimum required ROI profit (default: {DEFAULT_MIN_PROFIT})",
    )

    parser_sfr.add_argument(
        "-l",
        "--cost-limit",
        type=float,
        default=120,
        required=False,
        help=f"the max cost paid for the option ",
    )

    parser_sfr.add_argument(
        "-q",
        "--quantity",
        type=int,
        default=1,
        required=False,
        help="Maximum number of contracts to purchase (default: 1)",
    )
    parser_sfr.add_argument(
        "--log",
        type=str,
        default=None,
        required=False,
        help="Log file path to write all logs to a text file",
    )
    parser_sfr.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (shows all log levels)",
    )
    parser_sfr.add_argument(
        "--warning",
        action="store_true",
        help="Enable warning logging (shows INFO and WARNING levels)",
    )
    parser_sfr.add_argument(
        "-f",
        "--fin",
        type=str,
        default=None,
        required=False,
        help="Finviz screener URL to extract ticker symbols from (wrap in quotes)",
    )

    # Synthetic conversion (synthetic) sub-command
    parser_syn = subparsers.add_parser(
        "syn",
        help="Search for synthetic conversion opportunities with Global Opportunity Selection - "
        "intelligently ranks and selects the best opportunity across all symbols",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Scan for synthetic arbitrage opportunities across multiple symbols. "
        "The Global Opportunity Selection system evaluates all opportunities using multi-criteria "
        "scoring (risk-reward, liquidity, time decay, market quality) to find the single best trade.",
        epilog="Strategy Examples:\n"
        "  Conservative (safety-first):\n"
        "    %(prog)s --symbols SPY QQQ IWM --scoring-strategy conservative\n\n"
        "  Aggressive (maximize returns):\n"
        "    %(prog)s --symbols TSLA NVDA --scoring-strategy aggressive --min-risk-reward 3.0\n\n"
        "  Custom scoring:\n"
        "    %(prog)s --symbols AAPL MSFT --risk-reward-weight 0.3 --liquidity-weight 0.4\n",
    )

    # Metrics reporting sub-command
    parser_metrics = subparsers.add_parser(
        "metrics",
        help="Generate performance metrics report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_metrics.add_argument(
        "--format",
        choices=["json", "console"],
        default="json",
        help="Output format (default: json)",
    )
    parser_metrics.add_argument(
        "--output", "-o", type=str, help="Output filename (optional)"
    )
    parser_metrics.add_argument(
        "--reset", action="store_true", help="Reset metrics after generating report"
    )
    parser_syn.add_argument(
        "-s", "--symbols", nargs="+", help="List of symbols to scan (e.g., !MES, @SPX)"
    )
    parser_syn.add_argument(
        "-l",
        "--cost-limit",
        type=float,
        default=120,
        required=False,
        help="Minimum price for the contract [default: 120]",
    )

    parser_syn.add_argument(
        "-ml",
        "--max-loss",
        type=float,
        default=None,
        required=False,
        help=f"defines min threshold of the *max loss* for the strategy [default: None]",
    )

    parser_syn.add_argument(
        "-mp",
        "--max-profit",
        type=float,
        default=None,
        required=False,
        help=f"defines min threshold of the *max profit* for the strategy [default: None]",
    )
    parser_syn.add_argument(
        "-pr",
        "--profit-ratio",
        type=float,
        default=None,
        required=False,
        help=f"defines min threshold of max profit to max loss [max_profit/abs(max_loss)] for the strategy [default: None]",
    )

    parser_syn.add_argument(
        "-q",
        "--quantity",
        type=int,
        default=1,
        required=False,
        help="Maximum number of contracts to purchase (default: 1)",
    )
    parser_syn.add_argument(
        "--log",
        type=str,
        default=None,
        required=False,
        help="Log file path to write all logs to a text file",
    )
    parser_syn.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (shows all log levels)",
    )
    parser_syn.add_argument(
        "--warning",
        action="store_true",
        help="Enable warning logging (shows INFO and WARNING levels)",
    )
    parser_syn.add_argument(
        "-f",
        "--fin",
        type=str,
        default=None,
        required=False,
        help="Finviz screener URL to extract ticker symbols from (wrap in quotes)",
    )

    # Global Opportunity Selection - Scoring Strategy Configuration
    parser_syn.add_argument(
        "--scoring-strategy",
        choices=["conservative", "aggressive", "balanced", "liquidity-focused"],
        default="balanced",
        help="Pre-defined scoring strategy for global opportunity selection. "
        "Conservative: prioritizes safety and liquidity (30%% risk-reward, 35%% liquidity). "
        "Aggressive: prioritizes maximum returns (50%% risk-reward, 15%% liquidity). "
        "Balanced: well-rounded approach (40%% risk-reward, 25%% liquidity). "
        "Liquidity-focused: emphasizes execution certainty (25%% risk-reward, 40%% liquidity). "
        "[default: balanced]",
    )

    # Custom Scoring Weights (Advanced Users)
    parser_syn.add_argument(
        "--risk-reward-weight",
        type=float,
        default=None,
        help="Custom weight for risk-reward ratio in scoring (0.0-1.0). "
        "Overrides strategy preset. Must sum to 1.0 with other weights.",
    )
    parser_syn.add_argument(
        "--liquidity-weight",
        type=float,
        default=None,
        help="Custom weight for liquidity scoring (0.0-1.0). "
        "Overrides strategy preset. Must sum to 1.0 with other weights.",
    )
    parser_syn.add_argument(
        "--time-decay-weight",
        type=float,
        default=None,
        help="Custom weight for time decay scoring (0.0-1.0). "
        "Overrides strategy preset. Must sum to 1.0 with other weights.",
    )
    parser_syn.add_argument(
        "--market-quality-weight",
        type=float,
        default=None,
        help="Custom weight for market quality scoring (0.0-1.0). "
        "Overrides strategy preset. Must sum to 1.0 with other weights.",
    )

    # Threshold Configuration
    parser_syn.add_argument(
        "--min-risk-reward",
        type=float,
        default=None,
        help="Minimum acceptable risk-reward ratio threshold. "
        "Opportunities below this ratio will be rejected. [default: varies by strategy]",
    )
    parser_syn.add_argument(
        "--min-liquidity",
        type=float,
        default=None,
        help="Minimum acceptable liquidity score threshold (0.0-1.0). "
        "Opportunities below this score will be rejected. [default: varies by strategy]",
    )
    parser_syn.add_argument(
        "--max-bid-ask-spread",
        type=float,
        default=None,
        help="Maximum acceptable bid-ask spread for options. "
        "Options with wider spreads will be rejected. [default: varies by strategy]",
    )
    parser_syn.add_argument(
        "--optimal-days-expiry",
        type=int,
        default=None,
        help="Optimal days to expiration for time decay scoring. "
        "Options closer to this value get higher time scores. [default: varies by strategy]",
    )

    args = parser.parse_args()

    # Configure logging with optional file output and debug/warning modes
    log_file = getattr(args, "log", None)
    debug_mode = getattr(args, "debug", False)
    warning_mode = getattr(args, "warning", False)
    configure_logging(debug=debug_mode, warning=warning_mode, log_file=log_file)

    if args.command == "sfr":
        op = OptionScan()
        op.sfr_finder(
            symbol_list=args.symbols,
            profit_target=args.profit,
            cost_limit=args.cost_limit,
            quantity=args.quantity,
            log_file=log_file,
            debug=args.debug,
            finviz_url=args.fin,
        )

    elif args.command == "syn":
        op = OptionScan()
        op.syn_finder(
            symbol_list=args.symbols,
            cost_limit=args.cost_limit,
            max_loss_threshold=args.max_loss,
            max_profit_threshold=args.max_profit,
            profit_ratio_threshold=args.profit_ratio,
            quantity=args.quantity,
            log_file=log_file,
            debug=args.debug,
            finviz_url=args.fin,
            # Global Opportunity Selection Configuration
            scoring_strategy=args.scoring_strategy,
            risk_reward_weight=args.risk_reward_weight,
            liquidity_weight=args.liquidity_weight,
            time_decay_weight=args.time_decay_weight,
            market_quality_weight=args.market_quality_weight,
            min_risk_reward=args.min_risk_reward,
            min_liquidity=args.min_liquidity,
            max_bid_ask_spread=args.max_bid_ask_spread,
            optimal_days_expiry=args.optimal_days_expiry,
        )

    elif args.command == "metrics":
        # Generate metrics report
        if args.format.lower() == "json":
            output_file = metrics_collector.export_to_json(args.output)
            print(f"Metrics exported to: {output_file}")

        # Always print summary to console
        metrics_collector.print_summary()

        # Reset if requested
        if args.reset:
            metrics_collector.reset_session()
            print("Metrics have been reset.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
