"""
Market data collection and management for Synthetic arbitrage strategy.

This module contains:
- Global contract_ticker dictionary management
- Market data timeout and adaptive handling
- Ticker lifecycle management and utilities
- Contract data validation and cleanup
"""

import time
from typing import Dict, List, Optional, Tuple

from ib_async import Contract, Ticker

from ..common import get_logger
from .constants import ADAPTIVE_TIMEOUT_MULTIPLIER, DEFAULT_DATA_TIMEOUT

logger = get_logger()

# Global contract_ticker for use in SynExecutor and patching in tests
contract_ticker = {}


class DataCollector:
    """
    Manages market data collection and ticker lifecycle for synthetic arbitrage.
    """

    def __init__(self, symbol: str, data_timeout: float = DEFAULT_DATA_TIMEOUT):
        """
        Initialize data collector for a specific symbol.

        Args:
            symbol: Trading symbol
            data_timeout: Base timeout for data collection
        """
        self.symbol = symbol
        self.data_timeout = data_timeout
        self.data_collection_start = time.time()
        self.logger = logger

    def get_ticker(self, conId: int) -> Optional[Ticker]:
        """Get ticker for this symbol's contract using composite key"""
        return contract_ticker.get((self.symbol, conId))

    def set_ticker(self, conId: int, ticker: Ticker) -> None:
        """Set ticker for this symbol's contract using composite key"""
        contract_ticker[(self.symbol, conId)] = ticker

    def clear_symbol_tickers(self) -> int:
        """Clear all tickers for this symbol from global dictionary"""
        keys = [k for k in contract_ticker.keys() if k[0] == self.symbol]
        count = len(keys)
        for key in keys:
            del contract_ticker[key]
        self.logger.debug(
            f"[{self.symbol}] Cleared {count} contract tickers from global dictionary"
        )
        return count

    def get_symbol_contract_count(self) -> int:
        """Get count of contracts for this symbol"""
        return sum(1 for k in contract_ticker.keys() if k[0] == self.symbol)

    def check_data_timeout(self, all_contracts: List[Contract]) -> Tuple[bool, str]:
        """
        Check if data collection has timed out.

        Args:
            all_contracts: List of all contracts we're waiting for

        Returns:
            Tuple of (timed_out, timeout_message)
        """
        elapsed_time = time.time() - self.data_collection_start
        adaptive_timeout = min(
            self.data_timeout + (len(all_contracts) * ADAPTIVE_TIMEOUT_MULTIPLIER), 60.0
        )

        if elapsed_time > adaptive_timeout:
            missing_contracts = [
                c for c in all_contracts if self.get_ticker(c.conId) is None
            ]
            timeout_message = (
                f"[{self.symbol}] Data collection timeout after {elapsed_time:.1f}s "
                f"(adaptive limit: {adaptive_timeout:.1f}s). "
                f"Missing data for {len(missing_contracts)} contracts out of {len(all_contracts)}"
            )

            # Log details of missing contracts
            for c in missing_contracts[:5]:  # Log first 5 missing
                self.logger.info(
                    f"  Missing: {c.symbol} {c.right} {c.strike} {c.lastTradeDateOrContractMonth}"
                )

            return True, timeout_message

        return False, ""

    def has_all_data(self, all_contracts: List[Contract]) -> bool:
        """Check if we have data for all contracts"""
        return all(self.get_ticker(c.conId) is not None for c in all_contracts)

    def get_missing_contracts(self, all_contracts: List[Contract]) -> List[Contract]:
        """Get list of contracts missing ticker data"""
        return [c for c in all_contracts if self.get_ticker(c.conId) is None]

    def validate_ticker_data(self, ticker: Ticker, min_volume: int = 0) -> bool:
        """
        Validate ticker data quality.

        Args:
            ticker: Ticker to validate
            min_volume: Minimum volume requirement

        Returns:
            True if ticker data is valid
        """
        # Accept any ticker with valid price data, warn for low volume
        if ticker.volume > 10:
            return True
        elif ticker.volume >= min_volume and (
            ticker.bid > 0 or ticker.ask > 0 or ticker.close > 0
        ):
            # Accept low volume contracts if they have valid price data
            if ticker.volume < 10:
                self.logger.warning(
                    f"[{self.symbol}] Low volume ({ticker.volume}) for contract {ticker.contract.conId}, "
                    f"but accepting due to valid price data"
                )
            return True
        else:
            self.logger.debug(
                f"Skipping contract {ticker.contract.conId}: no valid data"
            )
            return False

    def reset_collection_timer(self):
        """Reset the data collection start time"""
        self.data_collection_start = time.time()

    def get_collection_stats(self) -> Dict:
        """Get statistics about current data collection"""
        symbol_contracts = sum(1 for k in contract_ticker.keys() if k[0] == self.symbol)
        elapsed_time = time.time() - self.data_collection_start

        return {
            "symbol": self.symbol,
            "contracts_collected": symbol_contracts,
            "elapsed_time": elapsed_time,
            "data_timeout": self.data_timeout,
        }


# Helper functions for backward compatibility
def get_ticker(symbol: str, conId: int) -> Optional[Ticker]:
    """Get ticker for a specific symbol and contract ID"""
    return contract_ticker.get((symbol, conId))


def set_ticker(symbol: str, conId: int, ticker: Ticker) -> None:
    """Set ticker for a specific symbol and contract ID"""
    contract_ticker[(symbol, conId)] = ticker


def clear_all_tickers() -> int:
    """Clear all tickers from global dictionary"""
    count = len(contract_ticker)
    contract_ticker.clear()
    logger.debug(f"Cleared all {count} contract tickers from global dictionary")
    return count


def get_global_ticker_stats() -> Dict:
    """Get statistics about the global ticker dictionary"""
    by_symbol = {}
    for (symbol, conId), _ in contract_ticker.items():
        if symbol not in by_symbol:
            by_symbol[symbol] = 0
        by_symbol[symbol] += 1

    return {
        "total_contracts": len(contract_ticker),
        "by_symbol": by_symbol,
        "symbols_count": len(by_symbol),
    }


def debug_contract_ticker_state() -> Dict:
    """Debug helper to show contract_ticker state by symbol"""
    stats = get_global_ticker_stats()
    logger.debug(f"Contract ticker state: {stats['by_symbol']}")
    return stats["by_symbol"]
