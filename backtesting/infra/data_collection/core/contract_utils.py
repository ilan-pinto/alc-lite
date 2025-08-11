"""
Contract utilities for Interactive Brokers.
Handles creation of different contract types (stocks, indices, etc.).
"""

from typing import Optional

import logging
from ib_async import Contract, Index, Stock

logger = logging.getLogger(__name__)


class ContractFactory:
    """Factory class for creating appropriate contract types based on symbol."""

    # Known index symbols that should use Index contract type
    INDEX_SYMBOLS = {
        # US Indices
        "SPX": {"exchange": "CBOE", "currency": "USD"},  # S&P 500 Index
        "NDX": {"exchange": "NASDAQ", "currency": "USD"},  # NASDAQ 100 Index
        "RUT": {"exchange": "RUSSELL", "currency": "USD"},  # Russell 2000 Index
        "DJX": {
            "exchange": "CBOE",
            "currency": "USD",
        },  # Dow Jones Industrial Average Index
        "OEX": {"exchange": "CBOE", "currency": "USD"},  # S&P 100 Index
        "XSP": {"exchange": "CBOE", "currency": "USD"},  # Mini S&P 500 Index
        # Volatility Indices
        "VIX": {"exchange": "CBOE", "currency": "USD"},
        "VIX1D": {"exchange": "CBOE", "currency": "USD"},
        "VIX9D": {"exchange": "CBOE", "currency": "USD"},
        "VIX3M": {"exchange": "CBOE", "currency": "USD"},
        "VIX6M": {"exchange": "CBOE", "currency": "USD"},
        # Sector Indices (common ones)
        "XAU": {"exchange": "PHLX", "currency": "USD"},  # Gold/Silver Index
        "HGX": {"exchange": "PHLX", "currency": "USD"},  # Housing Index
        "SOX": {"exchange": "PHLX", "currency": "USD"},  # Semiconductor Index
        "OSX": {"exchange": "PHLX", "currency": "USD"},  # Oil Service Index
    }

    @classmethod
    def create_contract(
        cls, symbol: str, exchange: Optional[str] = None, currency: str = "USD"
    ) -> Contract:
        """
        Create appropriate contract type based on symbol.

        Args:
            symbol: Trading symbol
            exchange: Optional specific exchange (overrides defaults)
            currency: Currency (default: USD)

        Returns:
            Contract object (Stock or Index)
        """
        symbol = symbol.upper()

        if symbol in cls.INDEX_SYMBOLS:
            # Create Index contract
            default_config = cls.INDEX_SYMBOLS[symbol]
            contract_exchange = exchange or default_config["exchange"]
            contract_currency = (
                currency if currency != "USD" else default_config["currency"]
            )

            logger.info(f"Creating Index contract for {symbol} on {contract_exchange}")
            return Index(symbol, contract_exchange, contract_currency)
        else:
            # Create Stock contract (default)
            contract_exchange = exchange or "SMART"
            logger.info(f"Creating Stock contract for {symbol} on {contract_exchange}")
            return Stock(symbol, contract_exchange, currency)

    @classmethod
    def is_index_symbol(cls, symbol: str) -> bool:
        """Check if symbol is a known index."""
        return symbol.upper() in cls.INDEX_SYMBOLS

    @classmethod
    def get_exchange_for_symbol(cls, symbol: str) -> Optional[str]:
        """Get the default exchange for a symbol."""
        symbol = symbol.upper()
        if symbol in cls.INDEX_SYMBOLS:
            return cls.INDEX_SYMBOLS[symbol]["exchange"]
        return "SMART"  # Default for stocks

    @classmethod
    def add_index_symbol(cls, symbol: str, exchange: str, currency: str = "USD"):
        """Add a new index symbol to the registry."""
        cls.INDEX_SYMBOLS[symbol.upper()] = {"exchange": exchange, "currency": currency}
        logger.info(f"Added index symbol {symbol} with exchange {exchange}")
