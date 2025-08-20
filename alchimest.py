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
__version__ = "1.14.0"

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
        "  Calendar spread scan - profit from time decay differential:\n"
        "    %(prog)s calendar --symbols SPY QQQ AAPL --cost-limit 300 --profit-target 0.25\n"
        "    %(prog)s calendar --symbols AAPL TSLA --iv-spread-threshold 0.04 --theta-ratio-threshold 2.0\n"
        "    %(prog)s calendar --symbols SPY IWM --front-expiry-max-days 30 --min-volume 50\n\n"
        "  Box spread scan - risk-free arbitrage from strike width differential:\n"
        "    %(prog)s box --symbols SPY QQQ --cost-limit 500 --profit-target 0.02\n"
        "    %(prog)s box --symbols AAPL TSLA --max-strike-width 10 --min-profit 0.10\n\n"
        "  Custom scoring weights:\n"
        "    %(prog)s syn --symbols SPY QQQ --risk-reward-weight 0.5 --liquidity-weight 0.3\n\n"
        "Calendar spreads are market-neutral strategies that profit when front month options\n"
        "decay faster than back month options, benefiting from time decay differential.\n\n"
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
        "--error",
        action="store_true",
        help="Enable error logging (shows INFO, WARNING, ERROR and CRITICAL levels)",
    )
    parser_sfr.add_argument(
        "-f",
        "--fin",
        type=str,
        default=None,
        required=False,
        help="Finviz screener URL to extract ticker symbols from (wrap in quotes)",
    )
    parser_sfr.add_argument(
        "--max-combinations",
        type=int,
        default=10,
        required=False,
        help="Maximum number of strike combinations to test per symbol (default: 10)",
    )
    parser_sfr.add_argument(
        "--max-strike-difference",
        type=int,
        default=5,
        required=False,
        help="Maximum strike difference to consider (default: 5)",
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

    # Calendar spread sub-command
    parser_calendar = subparsers.add_parser(
        "calendar",
        help="Search for calendar spread arbitrage opportunities - profit from time decay differential",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Scan for calendar spread opportunities that profit from the time decay differential "
        "between front and back month options. Calendar spreads are market-neutral strategies that "
        "benefit when the front month option decays faster than the back month option.",
        epilog="Calendar Spread Examples:\n"
        "  Basic calendar scan:\n"
        "    alchimest.py calendar --symbols SPY QQQ --cost-limit 300 --profit-target 0.25\n\n"
        "  High IV environment:\n"
        "    alchimest.py calendar --symbols AAPL TSLA --iv-spread-threshold 0.04 --theta-ratio-threshold 2.0\n\n"
        "  Conservative parameters:\n"
        "    alchimest.py calendar --symbols SPY IWM --front-expiry-max-days 30 --min-volume 50\n",
    )
    parser_calendar.add_argument(
        "-s",
        "--symbols",
        nargs="+",
        help="List of symbols to scan for calendar spreads",
    )
    parser_calendar.add_argument(
        "-l",
        "--cost-limit",
        type=float,
        default=300.0,
        help="Maximum net debit to pay for calendar spread (default: $300)",
    )
    parser_calendar.add_argument(
        "-p",
        "--profit-target",
        type=float,
        default=0.25,
        help="Target profit as percentage of max profit (default: 0.25 = 25 percent)",
    )
    parser_calendar.add_argument(
        "--iv-spread-threshold",
        type=float,
        default=0.01,
        help="Minimum IV spread (back - front) required (default: 0.03 = 3 percent)",
    )
    parser_calendar.add_argument(
        "--theta-ratio-threshold",
        type=float,
        default=1.5,
        help="Minimum theta ratio (front/back) required (default: 1.5)",
    )
    parser_calendar.add_argument(
        "--front-expiry-max-days",
        type=int,
        default=45,
        help="Maximum days to expiry for front month (default: 45)",
    )
    parser_calendar.add_argument(
        "--back-expiry-min-days",
        type=int,
        default=50,
        help="Minimum days to expiry for back month (default: 50)",
    )
    parser_calendar.add_argument(
        "--back-expiry-max-days",
        type=int,
        default=120,
        help="Maximum days to expiry for back month (default: 120)",
    )
    parser_calendar.add_argument(
        "--min-volume",
        type=int,
        default=10,
        help="Minimum daily volume per option leg (default: 10)",
    )
    parser_calendar.add_argument(
        "--max-bid-ask-spread",
        type=float,
        default=0.15,
        help="Maximum bid-ask spread as percent of mid price (default: 0.15 = 15 percent)",
    )
    parser_calendar.add_argument(
        "-q",
        "--quantity",
        type=int,
        default=1,
        help="Maximum number of calendar spreads to execute (default: 1)",
    )
    parser_calendar.add_argument(
        "--log",
        type=str,
        default=None,
        help="Log file path to write all logs to a text file",
    )
    parser_calendar.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (shows all log levels)",
    )
    parser_calendar.add_argument(
        "--warning",
        action="store_true",
        help="Enable warning logging (shows INFO and WARNING levels)",
    )
    parser_calendar.add_argument(
        "--error",
        action="store_true",
        help="Enable error logging (shows INFO, WARNING, ERROR and CRITICAL levels)",
    )
    parser_calendar.add_argument(
        "-f",
        "--fin",
        type=str,
        default=None,
        help="Finviz screener URL to extract ticker symbols from (wrap in quotes)",
    )

    # Box spread sub-command
    parser_box = subparsers.add_parser(
        "box",
        help="Search for box spread arbitrage opportunities - risk-free profit from strike width differential",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Scan for box spread arbitrage opportunities that provide risk-free profit when "
        "net debit < strike width. Box spreads consist of 4 legs: long call K1, short call K2, "
        "short put K1, long put K2. When executed properly, they guarantee the strike width as profit.",
        epilog="Box Spread Examples:\n"
        "  Basic box scan:\n"
        "    alchimest.py box --symbols SPY QQQ --cost-limit 500 --profit-target 0.02\n\n"
        "  Conservative parameters:\n"
        "    alchimest.py box --symbols SPY IWM --max-strike-width 10 --min-profit 0.10\n\n"
        "  High volume requirements:\n"
        "    alchimest.py box --symbols AAPL TSLA --min-volume 25 --max-spread 0.05\n",
    )
    parser_box.add_argument(
        "-s",
        "--symbols",
        nargs="+",
        help="List of symbols to scan for box spreads",
    )
    parser_box.add_argument(
        "-l",
        "--cost-limit",
        type=float,
        default=500.0,
        help="Maximum net debit to pay for box spread (default: $500)",
    )
    parser_box.add_argument(
        "-p",
        "--profit-target",
        type=float,
        default=0.01,
        help="Minimum profit target as percentage (default: 0.01 = 1 percent)",
    )
    parser_box.add_argument(
        "--min-profit",
        type=float,
        default=0.05,
        help="Minimum absolute profit per spread (default: $0.05)",
    )
    parser_box.add_argument(
        "--max-strike-width",
        type=float,
        default=50.0,
        help="Maximum strike width (K2-K1) to consider (default: $50)",
    )
    parser_box.add_argument(
        "--min-strike-width",
        type=float,
        default=1.0,
        help="Minimum strike width (K2-K1) to consider (default: $1)",
    )
    parser_box.add_argument(
        "--range",
        type=float,
        default=0.1,
        help="Price range around current stock price for strike selection (default: 0.1 = 10 percent)",
    )
    parser_box.add_argument(
        "--min-volume",
        type=int,
        default=5,
        help="Minimum daily volume per option leg (default: 5)",
    )
    parser_box.add_argument(
        "--max-spread",
        type=float,
        default=0.10,
        help="Maximum bid-ask spread as percent of mid price (default: 0.10 = 10 percent)",
    )
    parser_box.add_argument(
        "--min-days-expiry",
        type=int,
        default=1,
        help="Minimum days to expiration (default: 1)",
    )
    parser_box.add_argument(
        "--max-days-expiry",
        type=int,
        default=90,
        help="Maximum days to expiration (default: 90)",
    )
    parser_box.add_argument(
        "-q",
        "--quantity",
        type=int,
        default=1,
        help="Maximum number of box spreads to execute (default: 1)",
    )
    parser_box.add_argument(
        "--safety-buffer",
        type=float,
        default=0.02,
        help="Safety margin for pricing as percentage of net debit (default: 0.02 = 2 percent)",
    )
    parser_box.add_argument(
        "--require-risk-free",
        action="store_true",
        default=True,
        help="Only execute if truly risk-free (default: True)",
    )
    parser_box.add_argument(
        "--log",
        type=str,
        default=None,
        help="Log file path to write all logs to a text file",
    )
    parser_box.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (shows all log levels)",
    )
    parser_box.add_argument(
        "--warning",
        action="store_true",
        help="Enable warning logging (shows INFO and WARNING levels)",
    )
    parser_box.add_argument(
        "--error",
        action="store_true",
        help="Enable error logging (shows INFO, WARNING, ERROR and CRITICAL levels)",
    )
    parser_box.add_argument(
        "-f",
        "--fin",
        type=str,
        default=None,
        help="Finviz screener URL to extract ticker symbols from (wrap in quotes)",
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
        "--error",
        action="store_true",
        help="Enable error logging (shows INFO, WARNING, ERROR and CRITICAL levels)",
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
            warning=args.warning,
            error=args.error,
            finviz_url=args.fin,
            max_combinations=args.max_combinations,
            max_strike_difference=args.max_strike_difference,
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
            warning=args.warning,
            error=args.error,
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

    elif args.command == "calendar":
        op = OptionScan()
        op.calendar_finder(
            symbol_list=args.symbols,
            cost_limit=args.cost_limit,
            profit_target=args.profit_target,
            iv_spread_threshold=args.iv_spread_threshold,
            theta_ratio_threshold=args.theta_ratio_threshold,
            front_expiry_max_days=args.front_expiry_max_days,
            back_expiry_min_days=args.back_expiry_min_days,
            back_expiry_max_days=args.back_expiry_max_days,
            min_volume=args.min_volume,
            max_bid_ask_spread=args.max_bid_ask_spread,
            quantity=args.quantity,
            log_file=log_file,
            debug=args.debug,
            warning=args.warning,
            error=args.error,
            finviz_url=args.fin,
        )

    elif args.command == "box":
        op = OptionScan()
        op.box_finder(
            symbol_list=args.symbols,
            cost_limit=args.cost_limit,
            profit_target=args.profit_target,
            min_profit=args.min_profit,
            max_strike_width=args.max_strike_width,
            min_strike_width=args.min_strike_width,
            range=args.range,
            min_volume=args.min_volume,
            max_spread=args.max_spread,
            min_days_expiry=args.min_days_expiry,
            max_days_expiry=args.max_days_expiry,
            quantity=args.quantity,
            safety_buffer=args.safety_buffer,
            require_risk_free=args.require_risk_free,
            log_file=log_file,
            debug=args.debug,
            warning=args.warning,
            error=args.error,
            finviz_url=args.fin,
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
