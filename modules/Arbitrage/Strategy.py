import asyncio
import time
from datetime import datetime, timedelta

# from optparse import Option
from typing import Dict, List, Optional, Tuple

import numpy as np
from eventkit import Event
from ib_async import (
    IB,
    ComboLeg,
    Contract,
    FuturesOption,
    Index,
    Option,
    Order,
    OrderStatus,
    Stock,
    Ticker,
    Trade,
)
from rich.console import Console

from .common import (
    FILLED_ORDERS_FILENAME,
    get_logger,
    log_filled_order,
    log_order_details,
)
from .metrics import metrics_collector

logger = get_logger()


class OrderManagerClass:
    def __init__(self, ib: IB = None) -> None:
        self.ib = ib

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

            await asyncio.sleep(50)

            self.ib.cancelOrder(order)
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
        buffer_percent: float = 0.00,  # 2% buffer for slippage
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
            for tick in event:
                ticker: Ticker = tick
                contract = ticker.contract
                # This would be implemented by subclasses
                # with their specific logic
                pass
        except Exception as e:
            logger.error(f"Error in executor: {str(e)}")
            self.ib.pendingTickersEvent -= self.executor


class ArbitrageClass:
    def __init__(self) -> None:
        self.ib = IB()
        self.order_manager = OrderManagerClass(ib=self.ib)
        self.semaphore = asyncio.Semaphore(1000)
        self.active_executors: Dict[str, BaseExecutor] = {}

    def onFill(self, trade):
        """Called whenever any order gets filled (partially or fully)."""
        if log_filled_order(trade):
            metrics_collector.record_order_filled()
            self.ib.disconnect()

    async def master_executor(self, event: Event) -> None:
        """
        Master executor that delegates to individual symbol executors.
        This approach eliminates the need to constantly add/remove event handlers.
        """
        active_executors = [
            executor
            for executor in self.active_executors.values()
            if executor.is_active
        ]

        if not active_executors:
            return

        # Process each active executor in parallel
        try:
            tasks = [executor.executor(event) for executor in active_executors]
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in master_executor parallel processing: {str(e)}")

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
        return chain

    async def _get_chains(self, stock: Contract, exchange="CBOE"):
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
        """Request market data for multiple contracts with timing optimization"""
        start_time = time.time()

        # Use optimized synchronous batch request (IB reqMktData is not awaitable)
        self.request_market_data_batch_optimized(contracts)

        # Adaptive wait based on number of contracts
        base_wait = 0.1  # Base 100ms
        per_contract_wait = 0.01  # 10ms per contract
        max_wait = 2.0  # Maximum 2 seconds

        calculated_wait = min(
            base_wait + (len(contracts) * per_contract_wait), max_wait
        )

        logger.debug(f"Waiting {calculated_wait:.3f}s for {len(contracts)} contracts")
        await asyncio.sleep(calculated_wait)

        total_time = time.time() - start_time
        logger.debug(f"Market data batch request completed in {total_time:.3f}s")

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
