import asyncio
import logging
from optparse import Option
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from ib_async import IB, ComboLeg, Contract, FuturesOption, Index, OrderStatus, Stock, Order, Ticker
import numpy as np
from eventkit import Event
from .common import get_logger, log_order_details, log_filled_order, FILLED_ORDERS_FILENAME

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

    def order_handler(self, event):

        # orders = await self.ib.reqOpenOrdersAsync()
        orders = self.ib.openTrades()

        if event.orderStatus.status == OrderStatus.Filled:
            for trade in orders:
                logger.warning(f"Cancelling order {trade.order.orderId}")
                self.ib.cancelOrder(trade.order)

            self.ib.disconnect()
            logger.info("Completed order ")

    async def place_order(self, contract, order):

        position_exists = self._check_position_exists(contract)
        # trade_exists = self._check_trade_exists(contract)
        any_trade = self._check_any_trade_exists()
        if not position_exists and not any_trade:
            trade = self.ib.placeOrder(contract=contract, order=order)
            logger.info(f"placed order:{ order.orderId} ")

            await asyncio.sleep(50)

            # time.sleep(20)
            self.ib.cancelOrder(order)
            return trade
        else:
            logger.info(
                f"[{contract.symbol}]Conversion already exists. (position_exists: {position_exists}, any_trade: {any_trade}"
            )
            # await asyncio.sleep(45)
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
        order_manager: 'OrderManagerClass',
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

    def build_order(
        self,
        symbol: str,
        stock: Contract,
        call: Contract,
        put: Contract,
        lmt_price: float,
    ) -> Tuple[Contract, Order]:
        """Build a conversion order with stock, call, and put legs."""
        stock_leg = ComboLeg(
            conId=stock.conId, ratio=100, action="BUY", exchange="SMART"
        )
        call_leg = ComboLeg(conId=call.conId, ratio=1, action="SELL", exchange="SMART")
        put_leg = ComboLeg(
            conId=put.conId, ratio=1, action="BUY", exchange="SMART"
        )

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
            totalQuantity=1,
            lmtPrice=lmt_price,
            tif="DAY",
        )

        return conversion_contract, order

    def _extract_option_data(self, contract_ticker: Dict) -> Tuple[Optional[Contract], Optional[Contract], Optional[float], Optional[float], Optional[float], Optional[float]]:
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

        return call_contract, put_contract, call_strike, put_strike, call_price, put_price

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
        self.ib.orderStatusEvent += self.order_manager.order_handler
        self.semaphore = asyncio.Semaphore(1000)

    def onFill(self, trade):
        """Called whenever any order gets filled (partially or fully)."""
        log_filled_order(trade)

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
        chain = next(c for c in chains if c.exchange == exchange)
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

    def _get_stock_contract(self, symbol: str):
        exchange = "SMART"
        option_type = Option

        if symbol.find("!") == 0:
            symbol = symbol.replace("!", "")
            # cont = self.ib.reqMatchingSymbols(symbol)
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
