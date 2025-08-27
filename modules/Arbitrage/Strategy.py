import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import logging
import numpy as np
from eventkit import Event
from ib_async import IB, ComboLeg, Contract, FuturesOption, Index, Option, Order, Stock

from .common import get_logger, log_filled_order, log_order_details
from .metrics import metrics_collector

logger = get_logger()

# Global contract qualification cache
contract_cache = {}


class ContractCache:
    """Cache for qualified contracts with TTL to reduce IB API calls"""

    def __init__(self, ttl_seconds: int = 300):  # 5 minute TTL
        self.cache = {}
        self.ttl = ttl_seconds

    def _get_cache_key(
        self, symbol: str, expiry: str, strike: float, right: str
    ) -> str:
        """Generate cache key for contract"""
        return f"{symbol}_{expiry}_{strike}_{right}"

    def get(
        self, symbol: str, expiry: str, strike: float, right: str
    ) -> Optional[Contract]:
        """Get contract from cache if not expired"""
        key = self._get_cache_key(symbol, expiry, strike, right)
        if key in self.cache:
            contract, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                logger.debug(f"Cache hit for {key}")
                return contract
            else:
                # Expired, remove from cache
                del self.cache[key]
                logger.debug(f"Cache expired for {key}")
        return None

    def put(
        self, contract: Contract, symbol: str, expiry: str, strike: float, right: str
    ) -> None:
        """Store contract in cache with timestamp"""
        key = self._get_cache_key(symbol, expiry, strike, right)
        self.cache[key] = (contract, time.time())
        logger.debug(f"Cached contract {key}")

    def clear_expired(self) -> int:
        """Remove expired entries and return count removed"""
        current_time = time.time()
        expired_keys = []

        for key, (_, timestamp) in self.cache.items():
            if current_time - timestamp >= self.ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            logger.debug(f"Cleared {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    def size(self) -> int:
        """Return current cache size"""
        return len(self.cache)


# Global contract cache instance
contract_cache = ContractCache()


class OrderManagerClass:
    def __init__(self, ib: IB = None) -> None:
        self.ib = ib

    def _is_market_hours(self) -> bool:
        """Check if current time is during market hours (9:30 AM - 4:00 PM ET)"""
        from datetime import datetime, timedelta, timezone

        # Get current time in ET
        et_tz = timezone(timedelta(hours=-5))  # EST
        current_et = datetime.now(et_tz)

        # Check if it's a weekday
        if current_et.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Check if time is between 9:30 AM and 4:00 PM ET
        market_open = current_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_et.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= current_et <= market_close

    def _check_any_trade_exists(self) -> int:
        # Fetch the current portfolio positions
        open_trades = self.ib.openTrades()
        return len(open_trades) > 0

    def _check_trade_exists(self, combo_contract: Contract) -> bool:
        # Fetch the current portfolio positions
        positions = self.ib.openTrades()
        combo_ids = [contract.conId for contract in combo_contract.comboLegs]
        return any(pos for pos in positions if pos.contract.conId in combo_ids)

    def _check_position_exists(self, combo_contract: Contract):
        # Fetch the current portfolio positions
        positions = self.ib.portfolio()

        # Check if the combo contract is already in the portfolio
        combo_ids = [contract.conId for contract in combo_contract.comboLegs]
        combo_exists = any(
            pos
            for pos in positions
            if pos.contract.secType in ["OPT", "FOP"]
            and pos.contract.conId in combo_ids
        )
        if combo_exists:
            logger.warning(f"[{combo_contract.symbol}]contract already exists for ")
        return combo_exists

    async def place_order(self, contract, order):

        position_exists = self._check_position_exists(contract)
        any_trade = self._check_any_trade_exists()
        if not position_exists and not any_trade:
            trade = self.ib.placeOrder(contract=contract, order=order)
            logger.info(f"placed order:{ order.orderId} ")
            metrics_collector.record_order_placed()

            # Dynamic timeout based on market conditions and order type
            base_timeout = 30  # Base 30 seconds
            market_hours_bonus = (
                20 if self._is_market_hours() else 0
            )  # Extra time during market hours
            dynamic_timeout = base_timeout + market_hours_bonus

            await asyncio.sleep(dynamic_timeout)

            # Check if order was filled before cancelling
            if trade.orderStatus.status not in ["Filled", "PartiallyFilled"]:
                logger.info(
                    f"[{contract.symbol}] Order {order.orderId} not filled within {dynamic_timeout}s timeout, cancelling"
                )
                # Record rejection reason for unfilled order
                from .metrics import RejectionReason

                metrics_collector.add_rejection_reason(
                    RejectionReason.ORDER_NOT_FILLED,
                    {
                        "symbol": contract.symbol,
                        "order_id": order.orderId,
                        "order_status": trade.orderStatus.status,
                        "timeout_seconds": dynamic_timeout,
                        "filled_quantity": trade.orderStatus.filled,
                        "total_quantity": order.totalQuantity,
                    },
                )
                self.ib.cancelOrder(order)
            else:
                logger.info(
                    f"[{contract.symbol}] Order {order.orderId} filled successfully"
                )

            return trade
        else:
            logger.info(
                f"[{contract.symbol}]Conversion already exists. (position_exists: {position_exists}, any_trade: {any_trade}"
            )

            return None


class BaseExecutor:
    """
    Base executor class for arbitrage strategies with common functionality.

    This class provides shared methods for:
    1. Order building and validation
    2. Price calculations
    3. Trade logging
    4. Event handling
    """

    def __init__(
        self,
        ib: IB,
        order_manager: "OrderManagerClass",
        stock_contract: Contract,
        option_contracts: List[Contract],
        symbol: str,
        cost_limit: float,
        expiry: str,
        start_time: float = None,
    ) -> None:
        self.ib = ib
        self.order_manager = order_manager
        self.stock_contract = stock_contract
        self.option_contracts = option_contracts
        self.symbol = symbol
        self.cost_limit = cost_limit
        self.expiry = expiry
        self.start_time = start_time or time.time()
        self.contracts = [self.stock_contract] + self.option_contracts
        self.is_active = True

    def calculate_combo_limit_price(
        self,
        stock_price: float,
        call_price: float,
        put_price: float,
        buffer_percent: float = 0.02,  # 2% buffer for slippage
    ) -> float:
        """
        Calculate precise combo limit price based on individual leg target prices.

        Args:
            stock_price: Current stock price
            call_price: Target call option price (what we want to receive)
            put_price: Target put option price (what we want to pay)
            buffer_percent: Buffer percentage to account for slippage

        Returns:
            Calculated limit price for the combo order
        """
        # For conversion: Buy stock, Sell call, Buy put
        # Net cost = Stock price - Call premium + Put premium
        theoretical_cost = stock_price - call_price + put_price

        # Add buffer for market movement and slippage
        buffer_amount = theoretical_cost * buffer_percent
        limit_price = theoretical_cost + buffer_amount

        logger.info(
            f"Calculated combo limit: stock={stock_price:.2f}, call={call_price:.2f}, put={put_price:.2f}"
        )
        logger.info(
            f"Theoretical cost: {theoretical_cost:.2f}, with buffer: {limit_price:.2f}"
        )

        return round(limit_price, 2)

    def build_order(
        self,
        symbol: str,
        stock: Contract,
        call: Contract,
        put: Contract,
        lmt_price: float,
        quantity: int = 1,
        call_price: Optional[float] = None,
        put_price: Optional[float] = None,
    ) -> Tuple[Contract, Order]:
        """
        Build a conversion order with stock, call, and put legs.

        Args:
            symbol: Trading symbol
            stock: Stock contract
            call: Call option contract
            put: Put option contract
            lmt_price: Overall limit price for the combo
            quantity: Number of contracts
            call_price: Target call price (for precise limit calculation)
            put_price: Target put price (for precise limit calculation)

        Returns:
            Tuple of (conversion_contract, order)
        """
        stock_leg = ComboLeg(
            conId=stock.conId, ratio=100, action="BUY", exchange="SMART"
        )
        call_leg = ComboLeg(conId=call.conId, ratio=1, action="SELL", exchange="SMART")
        put_leg = ComboLeg(conId=put.conId, ratio=1, action="BUY", exchange="SMART")

        conversion_contract = Contract(
            symbol=symbol,
            comboLegs=[stock_leg, call_leg, put_leg],
            exchange="SMART",
            secType="BAG",
            currency="USD",
        )

        order = Order(
            orderId=self.ib.client.getReqId(),
            orderType="LMT",
            action="BUY",
            totalQuantity=quantity,
            lmtPrice=lmt_price,
            tif="DAY",
        )

        # Log the target vs actual limit price
        if call_price is not None and put_price is not None:
            logger.info(
                f"Target leg prices: call={call_price:.2f}, put={put_price:.2f}"
            )
            logger.info(f"Combo limit price: {lmt_price:.2f}")

        return conversion_contract, order

    def _extract_option_data(self, contract_ticker: Dict) -> Tuple[
        Optional[Contract],
        Optional[Contract],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        """Extract call and put contract data from ticker information."""
        call_contract = None
        put_contract = None
        call_strike = None
        put_strike = None
        call_price = None
        put_price = None

        for c in self.option_contracts:
            ticker = contract_ticker.get(c.conId)
            if not ticker:
                logger.error(f"No ticker data for option contract {c.conId}")
                return None, None, None, None, None, None

            if c.right == "C":
                call_contract = ticker.contract
                call_strike = ticker.contract.strike
                call_price = (
                    ticker.midpoint()
                    if not np.isnan(ticker.midpoint())
                    else ticker.close
                )

            elif c.right == "P":
                put_contract = ticker.contract
                put_strike = ticker.contract.strike
                put_price = ticker.ask if not np.isnan(ticker.ask) else ticker.close

        return (
            call_contract,
            put_contract,
            call_strike,
            put_strike,
            call_price,
            put_price,
        )

    def _log_trade_details(
        self,
        call_strike: float,
        call_price: float,
        put_strike: float,
        put_price: float,
        stock_price: float,
        net_credit: float,
        min_profit: float,
        max_profit: float,
        min_roi: float,
    ) -> None:
        """Log detailed trade information."""
        logger.info(
            f"min_ROI: {min_roi}. min_profit:{min_profit}. max_profit: {max_profit} "
        )

        logger.info(
            f"call_strike: {call_strike} - {call_price} . put_strike: {put_strike} - {put_price}. stock: {stock_price} "
        )

        order_details = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": self.symbol,
            "call_strike": call_strike,
            "call_price": call_price,
            "put_strike": put_strike,
            "put_price": put_price,
            "stock_price": stock_price,
            "net_credit": -net_credit,
            "min_profit": min_profit,
            "max_profit": max_profit,
            "min_roi": (min_profit / (stock_price + net_credit)) * 100,
        }

        log_order_details(order_details)

    def deactivate(self):
        """Deactivate the executor"""
        self.is_active = False

    async def executor(self, event: Event) -> None:
        """Base executor method - should be overridden by subclasses."""
        try:
            for _ in event:
                # This would be implemented by subclasses
                # with their specific logic
                pass
        except Exception as e:
            logger.error(f"Error in executor: {str(e)}")
            self.ib.pendingTickersEvent -= self.executor


class ArbitrageClass:
    def __init__(
        self,
        log_file: str = None,
        enable_vix_correlation: bool = False,
        db_pool: Optional[Any] = None,
    ) -> None:
        self.ib = IB()
        self.order_manager = OrderManagerClass(ib=self.ib)
        self.semaphore = asyncio.Semaphore(1000)
        self.active_executors: Dict[str, BaseExecutor] = {}
        self.order_filled = False  # Flag to track when an order is filled

        # Persistent state management for optimization
        self.symbol_chain_cache = {}  # Cache options chains per symbol
        self.last_scan_time = {}  # Track last scan time per symbol
        self.scan_cooldown = 30  # Minimum seconds between scans for same symbol

        # Symbol scanning throttling
        self.symbol_scan_semaphore = asyncio.Semaphore(
            5
        )  # Max 5 concurrent symbol scans

        # VIX correlation tracking (optional feature)
        self.enable_vix_correlation = enable_vix_correlation
        self.db_pool = db_pool
        self.vix_collector = None
        self.vix_correlation_data: List[Dict] = (
            []
        )  # In-memory buffer for correlation data

        # Configure file logging if specified
        if log_file:
            self._configure_file_logging(log_file)

        # Initialize VIX correlation if enabled
        if self.enable_vix_correlation and self.db_pool:
            self._initialize_vix_correlation()

    def _configure_file_logging(self, log_file: str) -> None:
        """Configure file logging for all arbitrage operations"""
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        # Add file handler to all relevant loggers
        loggers_to_update = [
            "modules.Arbitrage.SFR",
            "modules.Arbitrage.Synthetic",
            "modules.Arbitrage.Strategy",
            "modules.Arbitrage.common",
            "modules.Arbitrage.metrics",
        ]

        for logger_name in loggers_to_update:
            target_logger = logging.getLogger(logger_name)
            target_logger.addHandler(file_handler)
            target_logger.setLevel(logging.INFO)

    def onFill(self, trade):
        """Called whenever any order gets filled (partially or fully)."""
        if log_filled_order(trade):
            metrics_collector.record_order_filled()
            # Set flag to signal that an order was filled
            self.order_filled = True
            logger.info("Order filled - will exit after printing metrics")

    async def master_executor(self, event: Event) -> None:
        """
        Optimized master executor that delegates to individual symbol executors.
        This approach eliminates the need to constantly add/remove event handlers.
        """
        # Quick check for active executors to avoid unnecessary processing
        if not self.active_executors:
            return

        active_executors = [
            executor
            for executor in self.active_executors.values()
            if executor.is_active
        ]

        if not active_executors:
            # Clean up inactive executors immediately
            self.cleanup_inactive_executors()
            return

        # Process each active executor in parallel with improved error handling
        try:
            # Use asyncio.gather with return_exceptions=True for better error isolation
            tasks = [executor.executor(event) for executor in active_executors]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log any exceptions that occurred in individual executors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    executor = active_executors[i]
                    logger.warning(
                        f"Executor for {executor.symbol} failed: {str(result)}"
                    )
                    # Deactivate failed executor
                    executor.deactivate()

        except Exception as e:
            logger.error(f"Critical error in master_executor: {str(e)}")

    def deactivate_all_executors(self):
        """Deactivate all executors to stop metric collection"""
        for executor in self.active_executors.values():
            executor.deactivate()
        logger.info(
            f"Deactivated {len(self.active_executors)} executors due to order fill"
        )

    def cleanup_inactive_executors(self):
        """Remove inactive executors to prevent memory leaks"""
        inactive_symbols = [
            symbol
            for symbol, executor in self.active_executors.items()
            if not executor.is_active
        ]
        for symbol in inactive_symbols:
            del self.active_executors[symbol]

        if inactive_symbols:
            logger.info(f"Cleaned up {len(inactive_symbols)} inactive executors")

        # Also clean up expired contract cache entries
        expired_count = contract_cache.clear_expired()
        if expired_count > 0:
            logger.debug(
                f"Contract cache: removed {expired_count} expired entries, size: {contract_cache.size()}"
            )

    async def qualify_contracts_cached(self, *contracts) -> List[Contract]:
        """Qualify contracts using intelligent batching and cache"""
        cached_contracts = []
        uncached_contracts = []

        # Check cache first
        for contract in contracts:
            if hasattr(contract, "strike") and hasattr(contract, "right"):
                # Option contract - check cache
                cached_contract = contract_cache.get(
                    contract.symbol,
                    contract.lastTradeDateOrContractMonth,
                    contract.strike,
                    contract.right,
                )
                if cached_contract:
                    cached_contracts.append(cached_contract)
                else:
                    uncached_contracts.append(contract)
            else:
                # Stock/Index contract - always qualify (typically not cached)
                uncached_contracts.append(contract)

        # Qualify uncached contracts using intelligent batching
        qualified_contracts = cached_contracts.copy()
        if uncached_contracts:
            qualified_uncached = await self._qualify_contracts_in_batches(
                uncached_contracts
            )
            qualified_contracts.extend(qualified_uncached)

            # Cache the newly qualified option contracts
            for contract in qualified_uncached:
                if hasattr(contract, "strike") and hasattr(contract, "right"):
                    contract_cache.put(
                        contract,
                        contract.symbol,
                        contract.lastTradeDateOrContractMonth,
                        contract.strike,
                        contract.right,
                    )

        logger.debug(
            f"Contract qualification: {len(cached_contracts)} from cache, {len(uncached_contracts)} new"
        )
        return qualified_contracts

    def _get_contract_cache_key(self, contract) -> str:
        """Generate cache key for contract"""
        if hasattr(contract, "strike") and hasattr(contract, "right"):
            return f"{contract.symbol}_{contract.lastTradeDateOrContractMonth}_{contract.strike}_{contract.right}"
        else:
            return f"{contract.symbol}_{getattr(contract, 'exchange', 'SMART')}"

    async def _qualify_contracts_in_batches(
        self, contracts: List[Contract]
    ) -> List[Contract]:
        """Qualify contracts in optimized batches with fallback handling"""
        BATCH_SIZE = 100  # IB API handles large qualification batches efficiently
        qualified = []

        logger.debug(
            f"Qualifying {len(contracts)} contracts in batches of {BATCH_SIZE}"
        )

        for i in range(0, len(contracts), BATCH_SIZE):
            batch = contracts[i : i + BATCH_SIZE]
            try:
                # Try batch qualification first with circuit breaker protection
                if hasattr(self, "circuit_breaker"):
                    batch_qualified = await self.circuit_breaker.call_with_protection(
                        self.ib.qualifyContractsAsync, *batch
                    )
                else:
                    batch_qualified = await self.ib.qualifyContractsAsync(*batch)
                qualified.extend(batch_qualified)
                logger.debug(
                    f"Batch {i//BATCH_SIZE + 1}: qualified {len(batch_qualified)}/{len(batch)} contracts"
                )

            except Exception as e:
                logger.warning(
                    f"Batch qualification failed for batch {i//BATCH_SIZE + 1}: {e}"
                )
                # Fallback to individual qualification for this batch
                individual_qualified = await self._qualify_contracts_individually(batch)
                qualified.extend(individual_qualified)

            # Brief pause between large batches to respect API limits
            if len(batch) == BATCH_SIZE and i + BATCH_SIZE < len(contracts):
                await asyncio.sleep(0.05)

        return qualified

    async def _qualify_contracts_individually(
        self, contracts: List[Contract]
    ) -> List[Contract]:
        """Fallback method to qualify contracts individually"""
        qualified = []
        logger.debug(
            f"Falling back to individual qualification for {len(contracts)} contracts"
        )

        for contract in contracts:
            try:
                individual = await self.ib.qualifyContractsAsync(contract)
                qualified.extend(individual)
            except Exception as e:
                logger.warning(f"Failed to qualify contract {contract}: {e}")
                continue

        return qualified

    def should_scan_symbol(self, symbol: str) -> bool:
        """Check if enough time has passed since last scan for this symbol"""
        current_time = time.time()
        last_scan = self.last_scan_time.get(symbol, 0)

        if current_time - last_scan >= self.scan_cooldown:
            self.last_scan_time[symbol] = current_time
            return True
        else:
            time_remaining = self.scan_cooldown - (current_time - last_scan)
            logger.debug(
                f"Skipping {symbol} scan, cooldown: {time_remaining:.1f}s remaining"
            )
            return False

    def _get_contract_key(self, contract) -> str:
        """Generate a deterministic key for a contract based on its content"""
        try:
            symbol = getattr(contract, "symbol", "").upper()
            expiry = getattr(contract, "lastTradeDateOrContractMonth", "")
            strike = getattr(contract, "strike", 0)
            right = getattr(contract, "right", "")
            return f"{symbol}_{expiry}_{strike}_{right}"
        except Exception:
            # Fallback to id() if contract doesn't have expected attributes
            return str(id(contract))

    async def parallel_qualify_all_contracts(
        self,
        symbol: str,
        valid_expiries: List[str],
        valid_strike_pairs: List[Tuple[float, float]],
    ) -> Dict:
        """Qualify all option contracts for all expiries in parallel"""
        all_options_to_qualify = []
        expiry_contract_map = {}

        for expiry in valid_expiries:
            for call_strike, put_strike in valid_strike_pairs:
                call = Option(symbol, expiry, call_strike, "C", "SMART")
                put = Option(symbol, expiry, put_strike, "P", "SMART")

                all_options_to_qualify.extend([call, put])
                key = f"{expiry}_{call_strike}_{put_strike}"
                expiry_contract_map[key] = {
                    "call_original": call,
                    "put_original": put,
                    "expiry": expiry,
                    "call_strike": call_strike,
                    "put_strike": put_strike,
                }

        if not all_options_to_qualify:
            return {}

        # Log input contracts for debugging
        logger.debug(
            f"[{symbol}] Qualifying {len(all_options_to_qualify)} contracts in parallel:"
        )

        # Group contracts by type for logging
        calls = [
            c for c in all_options_to_qualify if hasattr(c, "right") and c.right == "C"
        ]
        puts = [
            c for c in all_options_to_qualify if hasattr(c, "right") and c.right == "P"
        ]

        logger.debug(
            f"[{symbol}] Input contracts: {len(calls)} calls, {len(puts)} puts across {len(valid_expiries)} expiries"
        )

        # Single parallel qualification for ALL contracts
        qualified_contracts = await self.qualify_contracts_cached(
            *all_options_to_qualify
        )

        # Map qualified contracts back to their original contracts
        qualified_map = {}
        matched_count = 0
        none_count = 0

        for qualified_contract in qualified_contracts:
            # Skip None contracts (failed qualification)
            if qualified_contract is None:
                none_count += 1
                continue

            # Find matching original contract by symbol, expiry, strike, right
            matched = False
            for original_contract in all_options_to_qualify:
                try:
                    # Safely compare all attributes with proper null checks
                    if (
                        hasattr(qualified_contract, "symbol")
                        and hasattr(original_contract, "symbol")
                        and qualified_contract.symbol.upper()
                        == original_contract.symbol.upper()
                        and hasattr(qualified_contract, "lastTradeDateOrContractMonth")
                        and hasattr(original_contract, "lastTradeDateOrContractMonth")
                        and qualified_contract.lastTradeDateOrContractMonth
                        == original_contract.lastTradeDateOrContractMonth
                        and hasattr(qualified_contract, "strike")
                        and hasattr(original_contract, "strike")
                        and abs(qualified_contract.strike - original_contract.strike)
                        < 0.01  # Float comparison tolerance
                        and hasattr(qualified_contract, "right")
                        and hasattr(original_contract, "right")
                        and qualified_contract.right == original_contract.right
                    ):
                        # Use content-based key instead of memory address
                        contract_key = self._get_contract_key(original_contract)
                        qualified_map[contract_key] = qualified_contract
                        matched_count += 1
                        matched = True
                        break
                except (AttributeError, TypeError) as e:
                    logger.debug(f"[{symbol}] Error comparing contracts: {e}")
                    continue

            if not matched:
                logger.debug(
                    f"[{symbol}] No match found for qualified contract: {qualified_contract.symbol} "
                    f"{qualified_contract.lastTradeDateOrContractMonth} {qualified_contract.strike} {qualified_contract.right}"
                )

        logger.debug(
            f"[{symbol}] Contract qualification results: {len(qualified_contracts)} returned, "
            f"{none_count} None, {matched_count} matched, {len(qualified_map)} in final map"
        )

        # Log the keys in qualified_map for debugging
        if qualified_map:
            logger.debug(
                f"[{symbol}] Qualified contract keys: {list(qualified_map.keys())}"
            )

        # Build final result mapping
        result = {}
        missing_calls = 0
        missing_puts = 0
        successful_pairs = 0

        for key, contract_info in expiry_contract_map.items():
            # Use content-based keys instead of memory addresses
            call_key = self._get_contract_key(contract_info["call_original"])
            put_key = self._get_contract_key(contract_info["put_original"])
            call_qualified = qualified_map.get(call_key)
            put_qualified = qualified_map.get(put_key)

            if call_qualified and put_qualified:
                # Validate that both contracts have conId (are properly qualified)
                if hasattr(call_qualified, "conId") and hasattr(put_qualified, "conId"):
                    result[key] = {
                        "call_contract": call_qualified,
                        "put_contract": put_qualified,
                        "expiry": contract_info["expiry"],
                        "call_strike": contract_info["call_strike"],
                        "put_strike": contract_info["put_strike"],
                    }
                    successful_pairs += 1
                else:
                    logger.debug(
                        f"[{symbol}] Skipping pair {key}: qualified contracts missing conId "
                        f"(call conId: {getattr(call_qualified, 'conId', 'missing')}, "
                        f"put conId: {getattr(put_qualified, 'conId', 'missing')})"
                    )
            else:
                if not call_qualified:
                    missing_calls += 1
                    logger.debug(
                        f"[{symbol}] Missing call contract for {key}: "
                        f"{contract_info['call_strike']} {contract_info['expiry']} (key: {call_key})"
                    )
                if not put_qualified:
                    missing_puts += 1
                    logger.debug(
                        f"[{symbol}] Missing put contract for {key}: "
                        f"{contract_info['put_strike']} {contract_info['expiry']} (key: {put_key})"
                    )

        logger.debug(
            f"[{symbol}] Final result: {successful_pairs} successful pairs, "
            f"{missing_calls} missing calls, {missing_puts} missing puts, "
            f"out of {len(expiry_contract_map)} attempted combinations"
        )

        logger.debug(
            f"[{symbol}] Successfully qualified {len(result)} contract pairs from {len(expiry_contract_map)} attempted"
        )
        return result

    async def scan_with_throttle(self, symbol: str, scan_func, *args):
        """Scan symbol with semaphore-based throttling"""
        async with self.symbol_scan_semaphore:
            if self.should_scan_symbol(symbol):
                return await scan_func(symbol, *args)
            else:
                return None

    def filter_expirations_within_range(
        self, expiration_dates, start_num_days=40, end_num_days=45
    ):
        """Filter expiration dates within a specific range of days."""
        today = datetime.today()
        start_range = today + timedelta(days=start_num_days)
        end_range = today + timedelta(days=end_num_days)

        filtered_dates = [
            date_str
            for date_str in expiration_dates
            if start_range <= datetime.strptime(date_str, "%Y%m%d") <= end_range
        ]

        return filtered_dates

    async def _get_chain(self, stock: Contract, exchange="CBOE"):
        """Get options chain with caching for performance"""
        cache_key = f"{stock.symbol}_{exchange}_{stock.conId}"
        current_time = time.time()

        # Check if we have cached chain data (with 5 minute TTL)
        if cache_key in self.symbol_chain_cache:
            cached_chain, cache_time = self.symbol_chain_cache[cache_key]
            if current_time - cache_time < 300:  # 5 minute cache TTL
                logger.debug(f"Using cached options chain for {stock.symbol}")
                return cached_chain

        # Fetch fresh chain data
        logger.debug(f"Fetching fresh options chain for {stock.symbol}")
        chains = await self.ib.reqSecDefOptParamsAsync(
            stock.symbol,
            "CME" if stock.secType == "IND" and not stock.symbol == "SPX" else "",
            stock.secType,
            stock.conId,
        )

        chain = next(
            c
            for c in chains
            if c.exchange == exchange and c.tradingClass == stock.symbol
        )

        # Cache the result
        self.symbol_chain_cache[cache_key] = (chain, current_time)
        return chain

    async def _get_chains(self, stock: Contract):
        chains = await self.ib.reqSecDefOptParamsAsync(
            stock.symbol,
            "CME" if stock.secType == "IND" and not stock.symbol == "SPX" else "",
            stock.secType,
            stock.conId,
        )
        return chains

    async def _get_market_data_async(self, stock):
        await self.ib.qualifyContractsAsync(stock)
        market_data = self.ib.reqMktData(stock)
        await asyncio.sleep(1)
        return market_data

    def request_market_data_batch_optimized(self, contracts: List[Contract]) -> None:
        """Request market data for multiple contracts with optimized batching"""
        try:
            # IB reqMktData is synchronous and fast - batch them efficiently
            successful_requests = 0
            failed_requests = 0

            for contract in contracts:
                try:
                    # Optimized IB request - no snapshots, no regulatory data
                    self.ib.reqMktData(contract, "", False, False)
                    successful_requests += 1
                except Exception as e:
                    failed_requests += 1
                    logger.debug(
                        f"Failed to request data for contract {contract.conId}: {str(e)}"
                    )

            logger.debug(
                f"Batch request: {successful_requests} successful, {failed_requests} failed"
            )

        except Exception as e:
            logger.error(f"Error in batch market data request: {str(e)}")

    async def request_market_data_batch(self, contracts: List[Contract]) -> None:
        """Request market data for multiple contracts with parallel processing and adaptive timeouts"""
        start_time = time.time()

        # Use optimized parallel batch request
        await self.request_market_data_parallel(contracts)

        total_time = time.time() - start_time
        logger.debug(
            f"Parallel market data batch request completed in {total_time:.3f}s"
        )

    async def request_market_data_parallel(self, contracts: List[Contract]) -> None:
        """Request market data for contracts with parallel processing and smart batching"""
        try:
            # Process contracts in optimal batches to avoid IB rate limits
            batch_size = min(50, len(contracts))  # IB typically allows ~100 req/sec
            batches = [
                contracts[i : i + batch_size]
                for i in range(0, len(contracts), batch_size)
            ]

            successful_requests = 0
            failed_requests = 0

            for batch_idx, batch in enumerate(batches):
                logger.debug(
                    f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} contracts)"
                )

                # Process batch contracts in parallel with semaphore control
                batch_tasks = []
                for contract in batch:
                    task = asyncio.create_task(
                        self._request_single_market_data(contract)
                    )
                    batch_tasks.append(task)

                # Wait for batch to complete with timeout
                batch_timeout = min(2.0 + (len(batch) * 0.02), 5.0)  # Adaptive timeout
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*batch_tasks, return_exceptions=True),
                        timeout=batch_timeout,
                    )

                    # Count results
                    for result in results:
                        if isinstance(result, Exception):
                            failed_requests += 1
                        else:
                            successful_requests += 1

                except asyncio.TimeoutError:
                    failed_requests += len(batch)
                    logger.warning(
                        f"Batch {batch_idx + 1} timed out after {batch_timeout}s"
                    )

                # Inter-batch delay to respect IB rate limits
                if batch_idx < len(batches) - 1:
                    await asyncio.sleep(0.1)

            logger.debug(
                f"Parallel request: {successful_requests} successful, {failed_requests} failed"
            )

        except Exception as e:
            logger.error(f"Error in parallel market data request: {str(e)}")

    async def _request_single_market_data(self, contract: Contract) -> bool:
        """Request market data for a single contract with error handling"""
        try:
            # Use semaphore to control concurrent requests
            async with self.semaphore:
                self.ib.reqMktData(contract, "", False, False)
                # Small delay to prevent overwhelming IB
                await asyncio.sleep(0.01)
                return True
        except Exception as e:
            logger.debug(
                f"Failed to request data for contract {contract.conId}: {str(e)}"
            )
            return False

    def _get_stock_contract(self, symbol: str):
        exchange = "SMART"
        option_type = Option

        if symbol.find("!") == 0:
            symbol = symbol.replace("!", "")

            stock = Index(symbol, exchange="CME", currency="USD")
            exchange = "CME"
            option_type = FuturesOption
        elif symbol.find("@") == 0:
            symbol = symbol.replace("@", "")
            stock = Index(symbol, exchange="CBOE", currency="USD")
            exchange = "CBOE"
        else:
            stock = Stock(symbol, "SMART", "USD")
        return exchange, option_type, stock

    def _initialize_vix_correlation(self) -> None:
        """Initialize VIX correlation tracking system."""
        try:
            # Import VIX collector here to avoid circular imports
            from backtesting.infra.data_collection.vix_collector import (
                VIXCollectionConfig,
                VIXDataCollector,
            )

            # Create VIX collector with minimal configuration for correlation tracking
            config = VIXCollectionConfig(
                tick_interval_ms=1000,  # 1 second intervals for correlation tracking
                collect_during_market_hours_only=True,
                enable_historical_backfill=False,  # Don't need historical data for real-time correlation
            )

            self.vix_collector = VIXDataCollector(self.db_pool, self.ib, config)
            logger.info("VIX correlation tracking initialized")

        except ImportError:
            logger.warning(
                "VIX collector module not available. Disabling VIX correlation."
            )
            self.enable_vix_correlation = False
        except Exception as e:
            logger.error(f"Failed to initialize VIX correlation: {e}")
            self.enable_vix_correlation = False

    async def start_vix_correlation_tracking(self) -> bool:
        """Start VIX correlation tracking if enabled."""
        if not self.enable_vix_correlation or not self.vix_collector:
            return False

        try:
            # Initialize VIX collector
            success = await self.vix_collector.initialize()
            if not success:
                logger.warning("Failed to initialize VIX collector")
                return False

            # Start real-time VIX data collection
            await self.vix_collector.start_real_time_collection()
            logger.info("VIX correlation tracking started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start VIX correlation tracking: {e}")
            return False

    async def stop_vix_correlation_tracking(self) -> None:
        """Stop VIX correlation tracking."""
        if self.vix_collector:
            try:
                await self.vix_collector.stop_real_time_collection()
                # Flush any remaining correlation data
                await self._flush_vix_correlation_data()
                logger.info("VIX correlation tracking stopped")
            except Exception as e:
                logger.error(f"Error stopping VIX correlation tracking: {e}")

    async def record_arbitrage_opportunity_with_vix(
        self,
        symbol: str,
        strategy_type: str,
        arbitrage_opportunity_id: Optional[int] = None,
        profit_potential: Optional[float] = None,
        execution_attempted: bool = False,
        success: Optional[bool] = None,
        actual_profit: Optional[float] = None,
        execution_speed_ms: Optional[int] = None,
    ) -> None:
        """
        Record arbitrage opportunity with VIX correlation context.

        Args:
            symbol: Trading symbol
            strategy_type: Type of arbitrage strategy (SFR, SYNTHETIC, etc.)
            arbitrage_opportunity_id: Database ID of the arbitrage opportunity
            profit_potential: Expected profit from the opportunity
            execution_attempted: Whether execution was attempted
            success: Whether execution was successful (if attempted)
            actual_profit: Actual profit realized (if successful)
            execution_speed_ms: Time taken to execute in milliseconds
        """
        if not self.enable_vix_correlation or not self.vix_collector:
            return

        try:
            # Get current VIX context
            vix_context = await self.vix_collector.get_vix_correlation_context()

            if not vix_context:
                logger.debug("No VIX context available for correlation recording")
                return

            # Create correlation record
            correlation_record = {
                "timestamp": datetime.now(),
                "symbol": symbol,
                "strategy_type": strategy_type,
                "arbitrage_opportunity_id": arbitrage_opportunity_id,
                "profit_potential": profit_potential,
                "execution_attempted": execution_attempted,
                "success": success,
                "actual_profit": actual_profit,
                "execution_speed_ms": execution_speed_ms,
                "vix_level": vix_context.get("vix_level"),
                "vix_regime": vix_context.get("vix_regime"),
                "term_structure_type": vix_context.get("term_structure_type"),
                "vix_spike_active": vix_context.get("vix_spike_active", False),
                "all_vix_levels": vix_context.get("all_levels", {}),
                "vix_context_timestamp": vix_context.get("timestamp"),
            }

            # Buffer the correlation data
            self.vix_correlation_data.append(correlation_record)

            logger.debug(
                f"Recorded VIX correlation for {symbol} {strategy_type}: "
                f"VIX={vix_context.get('vix_level'):.2f}, "
                f"regime={vix_context.get('vix_regime')}"
            )

            # Flush buffer if it's getting full
            if len(self.vix_correlation_data) >= 100:
                await self._flush_vix_correlation_data()

        except Exception as e:
            logger.error(f"Error recording VIX correlation: {e}")

    async def _flush_vix_correlation_data(self) -> None:
        """Flush buffered VIX correlation data to database."""
        if not self.vix_correlation_data or not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                # First, ensure we have term structure snapshots
                for record in self.vix_correlation_data:
                    if record.get("all_vix_levels"):
                        await conn.execute(
                            """
                            INSERT INTO vix_term_structure
                            (timestamp, vix_1d, vix_9d, vix_30d, vix_3m, vix_6m)
                            VALUES ($1, $2, $3, $4, $5, $6)
                            ON CONFLICT (timestamp) DO NOTHING
                        """,
                            record["vix_context_timestamp"],
                            record["all_vix_levels"].get("VIX1D"),
                            record["all_vix_levels"].get("VIX9D"),
                            record["all_vix_levels"].get("VIX"),
                            record["all_vix_levels"].get("VIX3M"),
                            record["all_vix_levels"].get("VIX6M"),
                        )

                # Then store correlation records (only for records with arbitrage_opportunity_id)
                correlation_records = [
                    r
                    for r in self.vix_correlation_data
                    if r.get("arbitrage_opportunity_id")
                ]

                if correlation_records:
                    await conn.executemany(
                        """
                        INSERT INTO vix_arbitrage_correlation
                        (arbitrage_opportunity_id, vix_snapshot_id, vix_level, vix_regime,
                         term_structure_type, vix_spike_active, arbitrage_success,
                         execution_speed_ms, profit_realized, correlation_weight)
                        SELECT $1, vts.id, $2, $3, $4, $5, $6, $7, $8, $9
                        FROM vix_term_structure vts
                        WHERE vts.timestamp = $10
                        LIMIT 1
                    """,
                        [
                            (
                                record["arbitrage_opportunity_id"],
                                record["vix_level"],
                                record["vix_regime"],
                                record["term_structure_type"],
                                record["vix_spike_active"],
                                record["success"],
                                record["execution_speed_ms"],
                                record["actual_profit"],
                                1.0,  # correlation_weight
                                record["vix_context_timestamp"],
                            )
                            for record in correlation_records
                        ],
                    )

            records_flushed = len(self.vix_correlation_data)
            self.vix_correlation_data.clear()

            logger.debug(
                f"Flushed {records_flushed} VIX correlation records to database"
            )

        except Exception as e:
            logger.error(f"Error flushing VIX correlation data: {e}")

    def get_vix_correlation_stats(self) -> Dict[str, Any]:
        """Get current VIX correlation statistics."""
        if not self.enable_vix_correlation:
            return {"vix_correlation_enabled": False}

        stats = {"vix_correlation_enabled": True}

        if self.vix_collector:
            stats.update(self.vix_collector.get_collection_stats())
            stats["current_vix_levels"] = self.vix_collector.get_current_vix_levels()
            stats["latest_term_structure"] = (
                self.vix_collector.get_latest_term_structure()
            )

        stats["correlation_buffer_size"] = len(self.vix_correlation_data)

        return stats

    async def analyze_vix_correlation_performance(
        self, days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze VIX correlation performance over specified period.

        Args:
            days_back: Number of days to analyze

        Returns:
            Dictionary containing correlation analysis results
        """
        if not self.enable_vix_correlation or not self.db_pool:
            return {"error": "VIX correlation not enabled or database not available"}

        try:
            async with self.db_pool.acquire() as conn:
                # Get correlation analysis from database function
                results = await conn.fetch(
                    """
                    SELECT * FROM calculate_vix_arbitrage_correlation(
                        $1::TIMESTAMPTZ, $2::TIMESTAMPTZ
                    )
                """,
                    datetime.now() - timedelta(days=days_back),
                    datetime.now(),
                )

                analysis = {
                    "analysis_period_days": days_back,
                    "vix_regime_performance": [],
                }

                for row in results:
                    analysis["vix_regime_performance"].append(
                        {
                            "vix_regime": row["vix_regime"],
                            "total_opportunities": int(row["total_opportunities"]),
                            "successful_opportunities": int(
                                row["successful_opportunities"]
                            ),
                            "success_rate_percent": float(row["success_rate"]),
                            "average_profit": (
                                float(row["avg_profit"]) if row["avg_profit"] else None
                            ),
                            "average_vix_level": (
                                float(row["avg_vix_level"])
                                if row["avg_vix_level"]
                                else None
                            ),
                            "correlation_strength": (
                                float(row["correlation_strength"])
                                if row["correlation_strength"]
                                else None
                            ),
                        }
                    )

                return analysis

        except Exception as e:
            logger.error(f"Error analyzing VIX correlation performance: {e}")
            return {"error": str(e)}
