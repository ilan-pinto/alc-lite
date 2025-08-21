from datetime import datetime

import logging
from ib_async import Trade
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.theme import Theme

# Custom theme for log levels
CUSTOM_THEME = Theme(
    {
        "debug": "dim cyan",
        "info": "green",
        "warning": "yellow",
        "error": "red",
        "critical": "bold red",
    }
)

# Global constants
FILLED_ORDERS_FILENAME = "filled_orders.txt"


class InfoOnlyFilter(logging.Filter):
    """Filter that only allows INFO-level logs through."""

    def filter(self, record):
        return record.levelno == logging.INFO


class InfoWarningFilter(logging.Filter):
    """Filter that allows INFO and WARNING-level logs through."""

    def filter(self, record):
        return record.levelno in [logging.INFO, logging.WARNING]


class InfoWarningErrorCriticalFilter(logging.Filter):
    """Filter that allows INFO, WARNING, ERROR, and CRITICAL logs through (but not DEBUG)."""

    def filter(self, record):
        return record.levelno in [
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]


def get_console_handler(filter_type: str = "info") -> RichHandler:
    """Create and configure a RichHandler for console output.

    Args:
        filter_type: Type of filter to apply:
            - "info": Show only INFO messages (default)
            - "warning": Show INFO and WARNING messages
            - "error": Show INFO, WARNING, ERROR, and CRITICAL messages (no DEBUG)
            - "none": Show all log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        RichHandler: Configured handler with appropriate filter
    """
    console = Console(theme=CUSTOM_THEME)
    handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=True,
        rich_tracebacks=True,
    )

    handler.setLevel(logging.DEBUG)

    if filter_type == "info":
        handler.addFilter(InfoOnlyFilter())
    elif filter_type == "warning":
        handler.addFilter(InfoWarningFilter())
    elif filter_type == "error":
        handler.addFilter(InfoWarningErrorCriticalFilter())
    elif filter_type == "none":
        pass  # No filter added
    else:
        # Default to info filter for unknown types
        handler.addFilter(InfoOnlyFilter())

    return handler


def configure_logging(
    level: int = logging.INFO,
    use_info_filter: bool = True,
    debug: bool = False,
    warning: bool = False,
    error: bool = False,
    log_file: str = None,
) -> None:
    """Configure logging with Rich handler and optional file output.

    Args:
        level: Base logging level (default: INFO)
        use_info_filter: Whether to use INFO-only filter when debug/warning/error are False
        debug: If True, show all log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        warning: If True, show INFO and WARNING levels only
        error: If True, show INFO, WARNING, ERROR, and CRITICAL levels (no DEBUG)
        log_file: Optional path to log file for persistent logging

    Note:
        - debug=True takes precedence over error=True and warning=True
        - error=True takes precedence over warning=True
        - When all flags are False, uses INFO-only filter (if use_info_filter=True)
        - File logging (if enabled) applies the same filter as console logging
    """
    # Determine filter type based on flags
    if debug:
        filter_type = "none"  # Show all levels
    elif error:
        filter_type = "error"  # Show INFO, WARNING, ERROR, CRITICAL (no DEBUG)
    elif warning:
        filter_type = "warning"  # Show INFO and WARNING
    else:
        filter_type = "info" if use_info_filter else "none"  # Show INFO only or all

    console_handler = get_console_handler(filter_type)

    handlers = [console_handler]

    # Add file handler if log file is specified
    if log_file:
        from logging.handlers import RotatingFileHandler

        # Use rotating file handler to manage log file size (10MB max, 5 backups)
        file_handler = RotatingFileHandler(
            log_file, mode="a", maxBytes=10485760, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        # File handler uses more inclusive filters to capture important information
        # Always include WARNING+ messages for rejections, plus INFO for funnel tracking
        if filter_type == "info":
            # For INFO console filter, file should capture INFO + WARNING + ERROR + CRITICAL
            file_handler.addFilter(InfoWarningErrorCriticalFilter())
        elif filter_type == "warning":
            file_handler.addFilter(InfoWarningFilter())
        elif filter_type == "error":
            file_handler.addFilter(InfoWarningErrorCriticalFilter())
        # No filter for "none" type (debug mode)

        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG if debug else level,
        format="%(message)s",
        handlers=handlers,
        force=True,  # Override any existing logging configuration
    )


def get_logger(name: str = "rich") -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)


def log_order_details(
    order_details: dict, filename: str = FILLED_ORDERS_FILENAME
) -> None:
    """Log order details to file."""
    with open(filename, "a") as f:
        f.write(f"{order_details}\n")


def trade_fills_table_str(trade: Trade) -> None:
    """Return a string representation of a rich table of all fills in the trade, including price and quantity from fill.execution."""
    table = Table(title=f"Trade Fills for Order {trade.order.orderId}")
    table.add_column("secIdType", style="cyan")
    table.add_column("symbol", style="magenta")
    table.add_column("strike", style="green")
    table.add_column("right", style="yellow")
    table.add_column("comboLegs", style="blue")
    table.add_column("price", style="bold")
    table.add_column("quantity", style="bold")

    for fill in trade.fills:
        contract = fill.contract
        sec_id_type = getattr(contract, "secIdType", "") or getattr(
            contract, "secType", ""
        )
        symbol = getattr(contract, "symbol", "")
        strike = str(getattr(contract, "strike", ""))
        right = getattr(contract, "right", "")
        combo_legs = getattr(contract, "comboLegs", None)
        if combo_legs:
            combo_legs_str = ", ".join(
                [f"{leg.conId}:{leg.action}:{leg.ratio}" for leg in combo_legs]
            )
        else:
            combo_legs_str = ""
        price = str(getattr(fill.execution, "price", ""))
        quantity = str(getattr(fill.execution, "shares", ""))
        table.add_row(
            str(sec_id_type),
            str(symbol),
            str(strike),
            str(right),
            combo_legs_str,
            price,
            quantity,
        )

    console = Console(record=True)
    console.print(table)
    # return console.export_text()


def log_filled_order(trade, filename: str = FILLED_ORDERS_FILENAME) -> bool:
    """Log filled order details to file."""
    if trade.orderStatus.status == "Filled":
        # Print the table to the console
        trade_fills_table_str(trade)
        # log to file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = f" timestamp: {timestamp}  Order_id: {trade.order.orderId}, Symbol: {trade.contract.symbol}, avgFillPrice: {trade.orderStatus.avgFillPrice}, filled: {trade.orderStatus.filled}\n"
        with open(filename, "a") as f:
            f.write(row)

        return True
    else:
        return False
