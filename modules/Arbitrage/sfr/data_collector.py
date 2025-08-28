"""
Data collection and market data management for SFR arbitrage strategy.

This module handles progressive data collection, priority-based contract management,
and market data quality assessment for the SFR strategy.
"""

import time
from typing import Dict, List, Optional, Set

from ib_async import Contract

from ..common import get_logger
from ..data_collection_metrics import (
    CollectionPhase,
    ContractPrioritizer,
    ContractPriority,
    DataCollectionMetrics,
    DataVelocityTracker,
    ProgressiveTimeoutConfig,
    log_phase_transition,
)
from .constants import (
    DEBUG_LOG_INTERVAL,
    DEFAULT_DATA_TIMEOUT,
    MIN_VOLUME_FOR_ACCEPTANCE,
    MIN_VOLUME_FOR_QUALITY,
)
from .models import ExpiryOption
from .utils import get_symbol_contract_count
from .validation import DataQualityValidator, MarketValidator

logger = get_logger()


class ContractTickerManager:
    """Manages contract ticker data with symbol-based isolation"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.contract_ticker = {}  # Global reference will be passed in

    def set_contract_ticker_reference(self, contract_ticker: dict):
        """Set reference to global contract_ticker dictionary"""
        self.contract_ticker = contract_ticker

    def get_ticker(self, conId):
        """Get ticker for this symbol's contract using composite key"""
        return self.contract_ticker.get((self.symbol, conId))

    def set_ticker(self, conId, ticker):
        """Set ticker for this symbol's contract using composite key"""
        self.contract_ticker[(self.symbol, conId)] = ticker

    def clear_symbol_tickers(self) -> int:
        """Clear all tickers for this symbol from global dictionary"""
        keys = [k for k in self.contract_ticker.keys() if k[0] == self.symbol]
        count = len(keys)
        for key in keys:
            del self.contract_ticker[key]
        logger.debug(
            f"[{self.symbol}] Cleared {count} contract tickers from global dictionary"
        )
        return count


class DataCollectionCoordinator:
    """Coordinates progressive data collection with timeout strategies"""

    def __init__(
        self,
        symbol: str,
        all_contracts: List[Contract],
        expiry_options: List[ExpiryOption],
        data_timeout: float = DEFAULT_DATA_TIMEOUT,
    ):
        self.symbol = symbol
        self.all_contracts = all_contracts
        self.expiry_options = expiry_options
        self.data_timeout = data_timeout
        self.data_collection_start = time.time()

        # Progressive collection components
        self.phase1_checked = False
        self.phase2_checked = False
        self.current_phase = CollectionPhase.INITIALIZING
        self.priority_tiers = {}
        self.velocity_tracker = DataVelocityTracker()
        self.collection_metrics = DataCollectionMetrics(
            symbol=symbol, start_time=self.data_collection_start
        )

        # Create set of valid contract IDs for fast symbol filtering
        self.valid_contract_ids = {c.conId for c in self.all_contracts}
        logger.debug(
            f"[{symbol}] Initialized with {len(self.valid_contract_ids)} valid contracts"
        )

        # Determine market hours for timeout configuration
        self.timeout_config = ProgressiveTimeoutConfig.create_for_market_conditions(
            is_market_hours=MarketValidator.is_market_hours(),
            total_contracts=len(self.all_contracts),
        )

        # Initialize components
        self.ticker_manager = ContractTickerManager(symbol)
        self.data_quality_validator = DataQualityValidator()

    def set_contract_ticker_reference(self, contract_ticker: dict):
        """Set reference to global contract_ticker dictionary"""
        self.ticker_manager.set_contract_ticker_reference(contract_ticker)

    def initialize_contract_priorities(self, stock_price: float):
        """Initialize contract priority tiers based on moneyness"""
        self.priority_tiers = ContractPrioritizer.categorize_by_moneyness(
            self.expiry_options, stock_price
        )

        # Update metrics with expected contract counts
        for priority in [
            ContractPriority.CRITICAL,
            ContractPriority.IMPORTANT,
            ContractPriority.OPTIONAL,
        ]:
            count = (
                len(self.priority_tiers[priority]) * 2
            )  # 2 contracts per expiry (call + put)

            # Add stock contract to critical priority (most important for pricing)
            if priority == ContractPriority.CRITICAL:
                count += 1  # Include the stock contract

            self.collection_metrics.contracts_expected[priority.value] = count

        logger.info(
            f"[{self.symbol}] Contract priorities initialized: "
            f"Critical={self.collection_metrics.contracts_expected['critical']}, "
            f"Important={self.collection_metrics.contracts_expected['important']}, "
            f"Optional={self.collection_metrics.contracts_expected['optional']}"
        )

    def get_contract_priority(self, contract) -> ContractPriority:
        """Get the priority of a specific contract"""
        return ContractPrioritizer.get_contract_priority(
            contract,
            self.expiry_options,
            getattr(self, "last_stock_price", 0),
        )

    def process_ticker_event(self, event) -> tuple:
        """Process ticker events and return processing statistics"""
        logger.debug(
            f"[{self.symbol}] Processing ticker event with {len(event)} contracts"
        )
        valid_processed = 0
        skipped_contracts = 0

        for tick in event:
            ticker = tick
            contract = ticker.contract

            # CRITICAL: Only process contracts that belong to this symbol
            if contract.conId not in self.valid_contract_ids:
                skipped_contracts += 1
                continue  # Skip contracts from other symbols

            valid_processed += 1

            # Update ticker data with volume-based filtering and priority tracking
            if ticker.volume >= MIN_VOLUME_FOR_ACCEPTANCE and (
                ticker.bid > 0 or ticker.ask > 0 or ticker.close > 0
            ):
                self.ticker_manager.set_ticker(contract.conId, ticker)

                # Track data arrival by priority
                priority = self.get_contract_priority(contract)
                self.collection_metrics.contracts_received[priority.value] += 1

                # Record first data arrival time
                if self.collection_metrics.time_to_first_data is None:
                    elapsed = time.time() - self.data_collection_start
                    self.collection_metrics.time_to_first_data = elapsed

                if ticker.volume < MIN_VOLUME_FOR_QUALITY:
                    logger.debug(
                        f"[{self.symbol}] Low volume ({ticker.volume}) for {contract.conId}"
                    )
            else:
                logger.debug(
                    f"Skipping contract {contract.conId}: no valid data or volume"
                )

        return valid_processed, skipped_contracts

    def update_velocity_tracking(self):
        """Update velocity tracker with current data"""
        total_received = self.collection_metrics.get_total_received()
        self.velocity_tracker.add_data_point(total_received)

    def get_elapsed_time(self) -> float:
        """Get elapsed time since data collection started"""
        return time.time() - self.data_collection_start

    def should_check_phase1(self) -> bool:
        """Check if Phase 1 evaluation should be performed"""
        return (
            self.get_elapsed_time() >= self.timeout_config.phase_1_timeout
            and not self.phase1_checked
            and self.priority_tiers
        )

    def should_check_phase2(self) -> bool:
        """Check if Phase 2 evaluation should be performed"""
        return (
            self.get_elapsed_time() >= self.timeout_config.phase_2_timeout
            and not self.phase2_checked
            and self.priority_tiers
        )

    def should_check_phase3(self) -> bool:
        """Check if Phase 3 evaluation should be performed"""
        return self.get_elapsed_time() >= self.timeout_config.phase_3_timeout

    def transition_to_phase1(self):
        """Transition to Phase 1 data collection"""
        self.phase1_checked = True
        self.current_phase = CollectionPhase.PHASE_1_CRITICAL
        log_phase_transition(
            self.symbol,
            CollectionPhase.INITIALIZING,
            self.current_phase,
            self.collection_metrics,
        )

    def transition_to_phase2(self):
        """Transition to Phase 2 data collection"""
        self.phase2_checked = True
        self.current_phase = CollectionPhase.PHASE_2_IMPORTANT
        log_phase_transition(
            self.symbol,
            CollectionPhase.PHASE_1_CRITICAL,
            self.current_phase,
            self.collection_metrics,
        )

    def transition_to_phase3(self):
        """Transition to Phase 3 data collection"""
        self.current_phase = CollectionPhase.PHASE_3_FINAL
        log_phase_transition(
            self.symbol,
            CollectionPhase.PHASE_2_IMPORTANT,
            self.current_phase,
            self.collection_metrics,
        )

    def has_sufficient_critical_data(self) -> bool:
        """Check if we have sufficient critical contract data"""
        if not self.priority_tiers:
            return False

        critical_received = self.collection_metrics.contracts_received["critical"]
        critical_expected = self.collection_metrics.contracts_expected["critical"]

        if critical_expected == 0 or critical_expected <= 0:
            return False

        percentage = critical_received / critical_expected
        return percentage >= self.timeout_config.critical_threshold

    def has_sufficient_important_data(self) -> bool:
        """Check if we have sufficient critical + important contract data"""
        if not self.priority_tiers:
            return False

        critical_received = self.collection_metrics.contracts_received["critical"]
        critical_expected = self.collection_metrics.contracts_expected["critical"]
        important_received = self.collection_metrics.contracts_received["important"]
        important_expected = self.collection_metrics.contracts_expected["important"]

        total_received = critical_received + important_received
        total_expected = critical_expected + important_expected

        if total_expected == 0 or total_expected <= 0:
            return False

        percentage = total_received / total_expected
        return percentage >= self.timeout_config.important_threshold

    def has_minimum_viable_data(self) -> bool:
        """Check if we have minimum viable data to make any decision"""
        # Need stock data
        if not self.has_stock_data():
            return False

        # Special case: If we have data overflow, we definitely have viable data
        if self.collection_metrics.get_completion_percentage() > 200:
            logger.debug(
                f"[{self.symbol}] Data overflow detected ({self.collection_metrics.get_completion_percentage():.1f}%), assuming viable data"
            )
            return True

        # Need at least 1 option pair with valid bid/ask spreads
        viable_pairs = 0

        for expiry_option in self.expiry_options:
            call_ticker = self.ticker_manager.get_ticker(
                expiry_option.call_contract.conId
            )
            put_ticker = self.ticker_manager.get_ticker(
                expiry_option.put_contract.conId
            )

            if call_ticker and put_ticker:
                # Check for valid bid/ask spreads (not volume)
                call_has_prices = (
                    hasattr(call_ticker, "bid")
                    and call_ticker.bid > 0
                    and hasattr(call_ticker, "ask")
                    and call_ticker.ask > 0
                )
                put_has_prices = (
                    hasattr(put_ticker, "bid")
                    and put_ticker.bid > 0
                    and hasattr(put_ticker, "ask")
                    and put_ticker.ask > 0
                )

                if call_has_prices and put_has_prices:
                    viable_pairs += 1

        return viable_pairs >= 1  # Allow evaluation with single expiry for testing

    def has_stock_data(self) -> bool:
        """Check if we have stock data available"""
        stock_contract = next(
            (c for c in self.all_contracts if c.secType == "STK"), None
        )
        if not stock_contract:
            return False
        return self.ticker_manager.get_ticker(stock_contract.conId) is not None

    def get_stock_ticker(self):
        """Get the stock ticker"""
        stock_contract = next(
            (c for c in self.all_contracts if c.secType == "STK"), None
        )
        if not stock_contract:
            return None
        return self.ticker_manager.get_ticker(stock_contract.conId)

    def has_data_for_option_pair(self, expiry_option: ExpiryOption) -> bool:
        """Check if we have data for both call and put contracts"""
        call_data = self.ticker_manager.get_ticker(expiry_option.call_contract.conId)
        put_data = self.ticker_manager.get_ticker(expiry_option.put_contract.conId)

        if not call_data:
            logger.debug(
                f"[{self.symbol}] No call data for {expiry_option.expiry} "
                f"strike={expiry_option.call_strike}"
            )
        if not put_data:
            logger.debug(
                f"[{self.symbol}] No put data for {expiry_option.expiry} "
                f"strike={expiry_option.put_strike}"
            )

        return call_data is not None and put_data is not None


class DataCollectionManager:
    """High-level manager for data collection operations"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.coordinator: Optional[DataCollectionCoordinator] = None

    def initialize_collection(
        self,
        all_contracts: List[Contract],
        expiry_options: List[ExpiryOption],
        data_timeout: float = DEFAULT_DATA_TIMEOUT,
    ) -> DataCollectionCoordinator:
        """Initialize data collection coordinator"""
        self.coordinator = DataCollectionCoordinator(
            symbol=self.symbol,
            all_contracts=all_contracts,
            expiry_options=expiry_options,
            data_timeout=data_timeout,
        )
        return self.coordinator

    def set_contract_ticker_reference(self, contract_ticker: dict):
        """Set reference to global contract_ticker dictionary"""
        if self.coordinator:
            self.coordinator.set_contract_ticker_reference(contract_ticker)

    def cleanup(self) -> int:
        """Clean up data collection resources"""
        if self.coordinator:
            return self.coordinator.ticker_manager.clear_symbol_tickers()
        return 0
