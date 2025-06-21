from datetime import datetime

import logging
from rich.console import Console
from rich.logging import RichHandler
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


def get_console_handler(use_info_filter: bool = True) -> RichHandler:
    """Create and configure a RichHandler for console output."""
    console = Console(theme=CUSTOM_THEME)
    handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=True,
        rich_tracebacks=True,
    )

    handler.setLevel(logging.INFO)
    if use_info_filter:
        handler.addFilter(InfoOnlyFilter())

    return handler


def configure_logging(level: int = logging.INFO, use_info_filter: bool = True) -> None:
    """Configure logging with Rich handler."""
    handler = get_console_handler(use_info_filter)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[handler],
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


def log_filled_order(trade, filename: str = FILLED_ORDERS_FILENAME) -> None:
    """Log filled order details to file."""
    if trade.orderStatus.status == "Filled":
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = f" timestamp: {timestamp}  Order_id: {trade.order.orderId}, Symbol: {trade.contract.symbol}, avgFillPrice: {trade.orderStatus.avgFillPrice}, filled: {trade.orderStatus.filled}\n"
        with open(filename, "a") as f:
            f.write(row)
