#!/usr/bin/env python3

import argparse
import logging
import pyfiglet
import warnings

from commands.option import OptionScan
from modules.Arbitrage.metrics import metrics_collector
from modules.welcome import print_welcome

warnings.simplefilter(action="ignore", category=FutureWarning)
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Version information
__version__ = "1.5.0"

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


# Create a filter that only allows INFO-level logs
class InfoOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == logging.INFO


console = Console(theme=custom_theme)
handler = RichHandler(
    console=console,
    show_time=True,
    show_level=True,
    show_path=True,
    rich_tracebacks=True,
)

handler.setLevel(logging.INFO)
handler.addFilter(InfoOnlyFilter())  # ⬅️ This is what filters out warnings, errors, etc.


# const
DEFAULT_REPORT_FOLDER = "~/dev/AlchimistProject/alchimest/report"
DEFAULT_MIN_PROFIT = 0.5


def configure_logging(
    level: int = logging.INFO, log_file: str = None, debug: bool = False
) -> None:
    # Create console handler - remove InfoOnlyFilter if debug is enabled
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=True,
        rich_tracebacks=True,
    )
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)

    # Only add InfoOnlyFilter if not in debug mode
    if not debug:
        console_handler.addFilter(InfoOnlyFilter())

    handlers = [console_handler]

    # Add file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG if debug else level,
        format="%(message)s",
        handlers=handlers,
    )


def main() -> None:
    """Main function for CLI execution"""

    print_welcome(console, __version__, DEFAULT_MIN_PROFIT)

    parser = argparse.ArgumentParser(
        description="Stock market analysis and trading tool with multiple features including scanning, analysis, and option strategies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (shows all log levels)",
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

    # Synthetic conversion (synthetic) sub-command
    parser_syn = subparsers.add_parser(
        "syn",
        help="Search for synthetic conversion (synthetic) opportunities not risk free",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    args = parser.parse_args()

    # Configure logging with optional file output and debug mode
    log_file = getattr(args, "log", None)
    debug_mode = getattr(args, "debug", False)
    configure_logging(log_file=log_file, debug=debug_mode)

    if args.command == "sfr":
        op = OptionScan()
        op.sfr_finder(
            symbol_list=args.symbols,
            profit_target=args.profit,
            cost_limit=args.cost_limit,
            quantity=args.quantity,
            log_file=log_file,
            debug=args.debug,
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
