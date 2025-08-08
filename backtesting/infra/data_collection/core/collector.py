"""
Real-time options data collector for Interactive Brokers.
Integrates with existing ArbitrageClass architecture.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import asyncpg
import logging
import numpy as np
from ib_async import IB, Contract, Option, Stock, Ticker

try:
    # Try relative imports first (when used as module)
    from ..config.config import CollectionConfig, DatabaseConfig
    from .validators import DataValidator, MarketDataSnapshot
except ImportError:
    # Fall back to absolute imports (when run directly)
    from backtesting.infra.data_collection.config.config import (
        CollectionConfig,
        DatabaseConfig,
    )
    from backtesting.infra.data_collection.core.validators import (
        DataValidator,
        MarketDataSnapshot,
    )

logger = logging.getLogger(__name__)


class OptionsDataCollector:
    """
    High-performance options data collector for backtesting database.
    Integrates with existing IB connection from ArbitrageClass.
    """

    def __init__(
        self, db_pool: asyncpg.Pool, ib_connection: IB, config: CollectionConfig = None
    ):
        self.db_pool = db_pool
        self.ib = ib_connection
        self.config = config or CollectionConfig()

        # Data management
        self.active_contracts: Dict[int, Contract] = {}
        self.contract_tickers: Dict[int, Ticker] = {}
        self.data_buffer: List[MarketDataSnapshot] = []
        self.last_flush = time.time()

        # State management
        self.collection_active = False
        self.pending_tasks: Set[asyncio.Task] = set()
        self.error_count = 0
        self.max_errors = 100

        # Performance monitoring
        self.stats = {
            "ticks_collected": 0,
            "ticks_inserted": 0,
            "errors": 0,
            "last_error": None,
        }

        # Initialize validator
        self.validator = DataValidator(db_pool)

    async def start_collection(self, symbols: Optional[List[str]] = None):
        """Start continuous data collection for specified symbols."""
        if self.collection_active:
            logger.warning("Collection already active")
            return

        self.collection_active = True
        symbols = symbols or self.config.default_symbols

        try:
            # Initialize contracts and subscribe to market data
            await self._initialize_contracts(symbols)

            # Set up event handlers
            self.ib.pendingTickersEvent += self._on_market_data_update

            # Start background tasks
            flush_task = asyncio.create_task(self._periodic_flush())
            cleanup_task = asyncio.create_task(self._periodic_cleanup())
            monitor_task = asyncio.create_task(self._monitor_health())

            self.pending_tasks.update({flush_task, cleanup_task, monitor_task})

            logger.info(f"Started data collection for {len(symbols)} symbols")

            # Keep running until stopped
            await asyncio.gather(*self.pending_tasks)

        except Exception as e:
            logger.error(f"Data collection error: {e}")
            raise
        finally:
            await self.stop_collection()

    async def stop_collection(self):
        """Stop data collection and clean up resources."""
        self.collection_active = False

        # Cancel pending tasks
        for task in self.pending_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.pending_tasks, return_exceptions=True)

        # Flush remaining data
        await self._flush_buffer(force=True)

        # Unsubscribe from market data
        for ticker in self.contract_tickers.values():
            self.ib.cancelMktData(ticker.contract)

        # Remove event handler
        self.ib.pendingTickersEvent -= self._on_market_data_update

        logger.info(f"Stopped data collection. Stats: {self.stats}")

    async def _initialize_contracts(self, symbols: List[str]):
        """Initialize option contracts for data collection."""
        for symbol in symbols:
            try:
                await self._initialize_symbol_contracts(symbol)
                # Throttle requests to avoid overwhelming IB API
                await asyncio.sleep(self.config.request_throttle_ms / 1000.0)
            except Exception as e:
                logger.error(f"Failed to initialize contracts for {symbol}: {e}")
                self.stats["errors"] += 1

    async def _initialize_symbol_contracts(self, symbol: str):
        """Initialize all relevant option contracts for a symbol."""
        # Get underlying stock
        stock = Stock(symbol, "SMART", "USD")
        qualified_stocks = await self.ib.qualifyContractsAsync(stock)

        if not qualified_stocks:
            logger.warning(f"Could not qualify stock contract for {symbol}")
            return

        stock = qualified_stocks[0]

        # Get current stock price for strike selection
        ticker = self.ib.reqMktData(stock, "", False, False)
        await asyncio.sleep(1)  # Wait for price

        stock_price = ticker.last if ticker.last else ticker.close
        if not stock_price:
            logger.warning(f"No price available for {symbol}")
            self.ib.cancelMktData(stock)
            return

        self.ib.cancelMktData(stock)

        # Get options chains
        chains = await self.ib.reqSecDefOptParamsAsync(
            stock.symbol, "", stock.secType, stock.conId
        )

        if not chains:
            logger.warning(f"No option chains found for {symbol}")
            return

        # Register underlying security in database
        underlying_id = await self._register_underlying(symbol)

        # Process each chain
        for chain in chains[:1]:  # Limit to primary exchange
            await self._process_option_chain(symbol, underlying_id, chain, stock_price)

    async def _process_option_chain(
        self, symbol: str, underlying_id: int, chain, stock_price: float
    ):
        """Process a single option chain."""
        # Filter expiries
        current_date = datetime.now().date()
        relevant_expiries = [
            exp
            for exp in chain.expirations
            if (datetime.strptime(exp, "%Y%m%d").date() - current_date).days
            <= self.config.expiry_range_days
        ]

        # Filter strikes around current price
        strike_min = stock_price * (1 - self.config.strike_range_percent)
        strike_max = stock_price * (1 + self.config.strike_range_percent)
        relevant_strikes = [
            strike for strike in chain.strikes if strike_min <= strike <= strike_max
        ]

        logger.info(
            f"Processing {symbol}: {len(relevant_expiries)} expiries, "
            f"{len(relevant_strikes)} strikes"
        )

        # Create and register option contracts
        contracts_to_subscribe = []

        for expiry in relevant_expiries[:3]:  # Limit expiries
            for strike in relevant_strikes[:10]:  # Limit strikes per expiry
                for right in ["C", "P"]:
                    option = Option(symbol, expiry, strike, right, chain.exchange)
                    contracts_to_subscribe.append((option, underlying_id, expiry))

        # Qualify and subscribe to contracts in batches
        batch_size = 20
        for i in range(0, len(contracts_to_subscribe), batch_size):
            batch = contracts_to_subscribe[i : i + batch_size]
            await self._subscribe_contract_batch(batch)
            await asyncio.sleep(self.config.request_throttle_ms / 1000.0)

    async def _subscribe_contract_batch(self, contracts: List[Tuple[Option, int, str]]):
        """Subscribe to a batch of option contracts."""
        # Qualify contracts
        options_to_qualify = [c[0] for c in contracts]
        qualified = await self.ib.qualifyContractsAsync(*options_to_qualify)

        for i, contract in enumerate(qualified):
            if contract:
                underlying_id = contracts[i][1]
                expiry_str = contracts[i][2]

                # Register in database
                contract_id = await self._register_option_contract(
                    contract, underlying_id, expiry_str
                )

                if contract_id:
                    # Subscribe to market data
                    ticker = self.ib.reqMktData(
                        contract,
                        (
                            self.config.generic_tick_list
                            if self.config.include_greeks
                            else ""
                        ),
                        False,
                        False,
                    )

                    self.active_contracts[contract_id] = contract
                    self.contract_tickers[contract_id] = ticker

                    logger.debug(
                        f"Subscribed to {contract.symbol} {contract.strike} "
                        f"{contract.right} {expiry_str}"
                    )

    async def _register_underlying(self, symbol: str) -> Optional[int]:
        """Register underlying security in database."""
        async with self.db_pool.acquire() as conn:
            try:
                # First try with ON CONFLICT (efficient if constraint exists)
                underlying_id = await conn.fetchval(
                    """
                    INSERT INTO underlying_securities (symbol, active)
                    VALUES ($1, true)
                    ON CONFLICT (symbol) DO UPDATE
                    SET updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """,
                    symbol,
                )
                return underlying_id
            except asyncpg.exceptions.InvalidTableDefinitionError:
                # If unique constraint doesn't exist, fall back to manual check
                logger.warning(
                    f"No unique constraint on symbol column, checking manually for {symbol}"
                )

                # Check if symbol already exists
                existing_id = await conn.fetchval(
                    "SELECT id FROM underlying_securities WHERE symbol = $1", symbol
                )

                if existing_id:
                    # Update existing record
                    await conn.execute(
                        "UPDATE underlying_securities SET updated_at = CURRENT_TIMESTAMP WHERE id = $1",
                        existing_id,
                    )
                    return existing_id
                else:
                    # Insert new record
                    try:
                        new_id = await conn.fetchval(
                            """
                            INSERT INTO underlying_securities (symbol, active)
                            VALUES ($1, true)
                            RETURNING id
                            """,
                            symbol,
                        )
                        return new_id
                    except asyncpg.exceptions.UniqueViolationError:
                        # Race condition - another process inserted it
                        return await conn.fetchval(
                            "SELECT id FROM underlying_securities WHERE symbol = $1",
                            symbol,
                        )
            except Exception as e:
                logger.error(f"Failed to register underlying {symbol}: {e}")
                return None

    async def _register_option_contract(
        self, contract: Contract, underlying_id: int, expiry_str: str
    ) -> Optional[int]:
        """Register option contract in database."""
        async with self.db_pool.acquire() as conn:
            try:
                expiry_date = datetime.strptime(expiry_str, "%Y%m%d").date()

                # First try with ON CONFLICT (efficient if constraint exists)
                contract_id = await conn.fetchval(
                    """
                    INSERT INTO option_chains
                    (underlying_id, expiration_date, strike_price, option_type,
                     contract_symbol, ib_con_id, exchange, multiplier, currency)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (underlying_id, expiration_date, strike_price, option_type)
                    DO UPDATE SET
                        ib_con_id = EXCLUDED.ib_con_id,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """,
                    underlying_id,
                    expiry_date,
                    contract.strike,
                    contract.right,
                    f"{contract.symbol}{contract.conId}",
                    contract.conId,
                    contract.exchange,
                    int(contract.multiplier) if contract.multiplier else 100,
                    contract.currency,
                )

                return contract_id
            except asyncpg.exceptions.InvalidTableDefinitionError:
                # If unique constraint doesn't exist, fall back to manual check
                logger.warning(
                    f"No unique constraint on option_chains, checking manually"
                )

                # Check if contract already exists
                existing_id = await conn.fetchval(
                    """
                    SELECT id FROM option_chains
                    WHERE underlying_id = $1 AND expiration_date = $2
                      AND strike_price = $3 AND option_type = $4
                    """,
                    underlying_id,
                    expiry_date,
                    contract.strike,
                    contract.right,
                )

                if existing_id:
                    # Update existing record
                    await conn.execute(
                        """
                        UPDATE option_chains
                        SET ib_con_id = $1, updated_at = CURRENT_TIMESTAMP
                        WHERE id = $2
                        """,
                        contract.conId,
                        existing_id,
                    )
                    return existing_id
                else:
                    # Insert new record
                    try:
                        new_id = await conn.fetchval(
                            """
                            INSERT INTO option_chains
                            (underlying_id, expiration_date, strike_price, option_type,
                             contract_symbol, ib_con_id, exchange, multiplier, currency)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                            RETURNING id
                            """,
                            underlying_id,
                            expiry_date,
                            contract.strike,
                            contract.right,
                            f"{contract.symbol}{contract.conId}",
                            contract.conId,
                            contract.exchange,
                            int(contract.multiplier) if contract.multiplier else 100,
                            contract.currency,
                        )
                        return new_id
                    except asyncpg.exceptions.UniqueViolationError:
                        # Race condition - another process inserted it
                        return await conn.fetchval(
                            """
                            SELECT id FROM option_chains
                            WHERE underlying_id = $1 AND expiration_date = $2
                              AND strike_price = $3 AND option_type = $4
                            """,
                            underlying_id,
                            expiry_date,
                            contract.strike,
                            contract.right,
                        )
            except Exception as e:
                logger.error(f"Failed to register option contract: {e}")
                return None

    def _on_market_data_update(self, tickers: Set[Ticker]):
        """Handle market data updates from IB."""
        if not self.collection_active:
            return

        timestamp = datetime.now()

        for ticker in tickers:
            # Find contract ID for this ticker
            contract_id = None
            for cid, t in self.contract_tickers.items():
                if t == ticker:
                    contract_id = cid
                    break

            if contract_id:
                snapshot = self._create_snapshot(contract_id, ticker, timestamp)
                if snapshot:
                    self.data_buffer.append(snapshot)
                    self.stats["ticks_collected"] += 1

        # Check if buffer should be flushed
        if (
            len(self.data_buffer) >= self.config.batch_size
            or time.time() - self.last_flush >= self.config.flush_interval_seconds
        ):
            asyncio.create_task(self._flush_buffer())

    def _create_snapshot(
        self, contract_id: int, ticker: Ticker, timestamp: datetime
    ) -> Optional[MarketDataSnapshot]:
        """Create market data snapshot from ticker."""
        try:
            snapshot = MarketDataSnapshot(
                contract_id=contract_id,
                timestamp=timestamp,
                bid_price=ticker.bid if not np.isnan(ticker.bid) else None,
                ask_price=ticker.ask if not np.isnan(ticker.ask) else None,
                last_price=ticker.last if not np.isnan(ticker.last) else None,
                bid_size=ticker.bidSize if ticker.bidSize != -1 else None,
                ask_size=ticker.askSize if ticker.askSize != -1 else None,
                last_size=ticker.lastSize if ticker.lastSize != -1 else None,
                volume=ticker.volume if ticker.volume != -1 else None,
                open_interest=(
                    ticker.openInterest if ticker.openInterest != -1 else None
                ),
                tick_type=self.config.market_data_type,
            )

            # Add Greeks if available
            if hasattr(ticker, "modelGreeks") and ticker.modelGreeks:
                greeks = ticker.modelGreeks
                snapshot.delta = greeks.delta if greeks.delta else None
                snapshot.gamma = greeks.gamma if greeks.gamma else None
                snapshot.theta = greeks.theta if greeks.theta else None
                snapshot.vega = greeks.vega if greeks.vega else None
                snapshot.implied_volatility = (
                    greeks.impliedVol if greeks.impliedVol else None
                )

            return snapshot

        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            return None

    async def _flush_buffer(self, force: bool = False):
        """Flush data buffer to database."""
        if not self.data_buffer and not force:
            return

        buffer_copy = self.data_buffer.copy()
        self.data_buffer.clear()
        self.last_flush = time.time()

        if not buffer_copy:
            return

        try:
            # Validate data
            valid_snapshots = []
            for snapshot in buffer_copy:
                if await self.validator.validate_snapshot(snapshot):
                    valid_snapshots.append(snapshot)

            if not valid_snapshots:
                return

            # Insert into database
            async with self.db_pool.acquire() as conn:
                # Prepare data for insertion
                values = []
                for snapshot in valid_snapshots:
                    values.append(
                        (
                            snapshot.timestamp,
                            snapshot.contract_id,
                            snapshot.bid_price,
                            snapshot.ask_price,
                            snapshot.last_price,
                            snapshot.bid_size,
                            snapshot.ask_size,
                            snapshot.last_size,
                            snapshot.volume,
                            snapshot.open_interest,
                            snapshot.delta,
                            snapshot.gamma,
                            snapshot.theta,
                            snapshot.vega,
                            snapshot.rho,
                            snapshot.implied_volatility,
                            snapshot.tick_type,
                        )
                    )

                # Bulk insert
                await conn.executemany(
                    """
                    INSERT INTO market_data_ticks
                    (time, contract_id, bid_price, ask_price, last_price,
                     bid_size, ask_size, last_size, volume, open_interest,
                     delta, gamma, theta, vega, rho, implied_volatility, tick_type)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                            $11, $12, $13, $14, $15, $16, $17)
                """,
                    values,
                )

                self.stats["ticks_inserted"] += len(values)
                logger.debug(f"Inserted {len(values)} market data records")

        except Exception as e:
            logger.error(f"Failed to flush buffer: {e}")
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)
            self.error_count += 1

            if self.error_count > self.max_errors:
                logger.critical("Too many errors, stopping collection")
                self.collection_active = False

    async def _periodic_flush(self):
        """Periodically flush data buffer."""
        while self.collection_active:
            try:
                await asyncio.sleep(self.config.flush_interval_seconds)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")

    async def _periodic_cleanup(self):
        """Periodically clean up stale connections and data."""
        while self.collection_active:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Remove inactive tickers
                inactive_contracts = []
                current_time = time.time()

                for contract_id, ticker in self.contract_tickers.items():
                    if hasattr(ticker, "time") and ticker.time:
                        age = current_time - ticker.time
                        if age > self.config.stale_data_threshold_seconds:
                            inactive_contracts.append(contract_id)

                for contract_id in inactive_contracts:
                    logger.info(f"Removing stale contract {contract_id}")
                    contract = self.active_contracts.pop(contract_id, None)
                    ticker = self.contract_tickers.pop(contract_id, None)
                    if contract and ticker:
                        self.ib.cancelMktData(contract)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    async def _monitor_health(self):
        """Monitor collector health and log statistics."""
        while self.collection_active:
            try:
                await asyncio.sleep(60)  # Every minute

                logger.info(f"Collector stats: {self.stats}")
                logger.info(
                    f"Active contracts: {len(self.active_contracts)}, "
                    f"Buffer size: {len(self.data_buffer)}"
                )

                # Reset error count if no recent errors
                if self.error_count > 0 and self.stats["errors"] == 0:
                    self.error_count = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")

    def get_stats(self) -> Dict:
        """Get current collector statistics."""
        return {
            **self.stats,
            "active_contracts": len(self.active_contracts),
            "buffer_size": len(self.data_buffer),
            "collection_active": self.collection_active,
        }
