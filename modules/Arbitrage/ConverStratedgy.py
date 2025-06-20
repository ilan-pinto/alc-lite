import asyncio
import logging
from typing import List, Tuple
from ib_async import IB, ComboLeg, Contract, Order, Stock, Option, Ticker
import numpy as np
from modules.Arbitrage.Strategy import ArbitrageClass, OrderManagerClass
import time
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from eventkit import Event

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

# from modules import common
console = Console(theme=custom_theme)
handler = RichHandler(
    console=console,
    show_time=True,
    show_level=True,
    show_path=True,
    rich_tracebacks=True,
)


def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,  # Set the logging level here
        format="%(message)s",
        # datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            # logging.FileHandler("app.log"),  # Log to a file
            # logging.StreamHandler(),  # Also log to console
            handler,
        ],
    )


configure_logging()
logger = logging.getLogger("rich")


class ConversionExecutor:
    def __init__(
        self,
        ib: IB,
        order_manager: OrderManagerClass,
        stock_contract: Contract,
        option_contracts: List[Contract],
        strike,
        symbol,
        profit_target,
        limit,
        volume_limit,
        expiry,
        start_time,
    ) -> None:

        self.ib = ib
        self.order_manager = order_manager
        self.stock_contract = stock_contract
        self.option_contracts = option_contracts
        self.strike = strike
        self.symbol = symbol
        self.profit_target = profit_target
        self.limit = limit
        self.volume_limit = volume_limit
        self.expiry = expiry
        self.start_time = start_time

    def check_conditions(
        self,
        symbol,
        profit_target,
        limit,
        volume_limit,
        call_data,
        put_data,
        conversion_profit,
        lmt_price,
    ):

        if conversion_profit < profit_target:
            logger.warning(f"[{symbol}]conversion_profit< profit_target")
            return False
        elif np.isnan(lmt_price) or lmt_price > limit:
            logger.warning(f"[{symbol}]np.isnan(lmt_price) or lmt_price > limit")
            return False
        elif put_data.volume < volume_limit and call_data.volume < volume_limit:
            logger.warning(
                f"[{symbol}]  low volume. limit:{volume_limit} call volume: {call_data.volume}  put volume: {put_data.volume}"
            )
            return False
        else:
            return True

    def build_order(self, symbol, stock, call, put, lmt_price):
        stock_leg = ComboLeg(conId=stock.conId, ratio=100, action="BUY", exchange="SMART")
        call_leg = ComboLeg(conId=call.conId, ratio=1, action="SELL", exchange="SMART")
        put_data_leg = ComboLeg(conId=put.conId, ratio=1, action="BUY", exchange="SMART")

        conversion_contract = Contract(
            symbol=symbol,
            comboLegs=[stock_leg, call_leg, put_data_leg],
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
            # outsideRth=True,
        )

        return conversion_contract, order

    async def executor(self, event):
        for tick in event:
            ticker: Ticker = tick
            contract = ticker.contract
            contract_ticker[contract.conId] = ticker

            if ticker.volume > 30:
                contract_ticker[contract.conId] = ticker

            else:
                self.ib.pendingTickersEvent -= self.executor
                logger.info(f"removed {self.symbol} - {self.strike} - {self.expiry}")
                return

            self.contracts = [self.stock_contract] + self.option_contracts
            if all(contract_ticker.get(c.conId) != None for c in self.contracts):
                logger.info(f"time to execution: { time.time() - self.start_time} sec")

                # calc limit price
                conversion_contract, order = self.calc_price_and_build_order()

                if order and conversion_contract:
                    trade = await self.order_manager.place_order(conversion_contract, order)
                    # await asyncio.sleep(0.05)

    def calc_price_and_build_order(self) -> Tuple[float, float]:
        lmt_price = 0

        stock_data = self.stock_contract
        ticker = contract_ticker.get(self.stock_contract.conId)

        stock_midpoint = ticker.ask if not np.isnan(ticker.ask) else ticker.close
        lmt_price += stock_midpoint

        for c in self.option_contracts:
            ticker = contract_ticker.get(c.conId)
            if c.right == "C":
                call_data = ticker
                call_price = ticker.midpoint() if not np.isnan(ticker.midpoint()) else ticker.close
                lmt_price -= call_price
            elif c.right == "P":
                put_data = ticker
                put_price = ticker.ask if not np.isnan(ticker.ask) else ticker.close
                lmt_price += put_price

        lmt_price = round((lmt_price), 2)
        conversion_profit = call_price - put_price + (self.strike - stock_midpoint)

        roi = (conversion_profit / lmt_price) * 100
        logger.info(f"ROI: {roi}. conversion_profit:{conversion_profit}. lmt_price: {lmt_price} ")
        # logger.info(f"Call Option close Price: {call_data.close}")
        # logger.info(f"Put Option close Price: {put_data.close}")
        # logger.info(f"Call Option bid: {call_data.bid}")
        # logger.info(f"Put Option ask: {put_data.ask}")

        if self.check_conditions(
            self.symbol,
            self.profit_target,
            self.limit,
            self.volume_limit,
            call_data,
            put_data,
            conversion_profit,
            lmt_price,
        ):
            return self.build_order(
                self.symbol,
                stock_data,
                call_data.contract,
                put_data.contract,
                lmt_price,
            )

        else:
            return None, None


class Conversion(ArbitrageClass):

    async def scan(
        self,
        symbol_list,
        limit=50,
        range=0.12,
        profit_target=0.10,
        volume_limit=100,
        expiration_range=2,
        profit_target_multiplier=1.10,  #
    ):
        """
        scan for conversation and execute order

        symbol list - list of valid symbols
        limit - min price for the contract. e.g limit=50 means willing to pay up to 5000$
        range - below strike percent to scan
        profit_target - min acceptable credit e.g profit_target=0.1 means min profit of 10$
        volume_limit - min option contract volume
        profit_target_multiplier - multiplies profit target per expiry iteration
        """
        # Global
        global contract_ticker
        global stock_ticker
        contract_ticker = {}
        stock_ticker = {}

        # set configuration
        self.limit = limit
        self.range = range
        self.profit_target = profit_target
        self.volume_limit = volume_limit
        self.expiration_range = expiration_range
        self.profit_target_multiplier = profit_target_multiplier

        await self.ib.connectAsync("127.0.0.1", 7497, clientId=2)
        # self.ib.reqMarketDataType = 3

        while True:
            tasks = []
            for symbol in symbol_list:
                task = asyncio.create_task(self.scan_conv(range, symbol))
                tasks.append(task)
                await asyncio.sleep(5)
            results = await asyncio.gather(*tasks)

            await asyncio.sleep(25)
            contract_ticker = {}
            stock_ticker = {}
            self.ib.pendingTickersEvent = Event("pendingTickersEvent")

    async def scan_conv(self, range, symbol):
        exchange, option_type, stock = self._get_stock_contract(symbol)

        # Request market data for the stock
        market_data = await self._get_market_data_async(stock)

        stock_price = market_data.last if not np.isnan(market_data.last) else market_data.close

        logger.info(f"price for [{symbol}: {stock_price} ]")

        # Request options chain
        chain = await self._get_chain(stock)

        # Define parameters for the options (expiry and strike price)
        # expiry = chain.expirations[: self.expiration_range]  # Example expiration date
        valid_strikes = [
            s for s in chain.strikes if s < stock_price * (1 + range) and s > stock_price * (1 - range)
        ]  # Example strike price

        profit_target = self.profit_target  # rest profit target
        for expiry in chain.expirations[: self.expiration_range]:

            for strike in valid_strikes:
                await asyncio.sleep(0.05)
                call = Option(stock.symbol, expiry, strike, "C", "SMART")
                put = Option(stock.symbol, expiry, strike, "P", "SMART")
                # Qualify the option contracts
                valid_contracts = await self.ib.qualifyContractsAsync(call, put)
                if len(valid_contracts) == 0:
                    continue

                # Request market data for the options

                self.ib.reqMktData(stock)
                self.ib.reqMktData(call)
                self.ib.reqMktData(put)

                conversion = ConversionExecutor(
                    ib=self.ib,
                    order_manager=self.order_manager,
                    stock_contract=stock,
                    option_contracts=[call, put],
                    strike=strike,
                    symbol=symbol,
                    profit_target=profit_target,
                    limit=self.limit,
                    volume_limit=self.limit,
                    expiry=expiry,
                    start_time=time.time(),
                )

                self.ib.pendingTickersEvent += conversion.executor

            profit_target = profit_target * self.profit_target_multiplier
