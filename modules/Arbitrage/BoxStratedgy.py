import asyncio
import os
import time
from itertools import chain, permutations
from typing import List, Tuple

import logging
import numpy as np
from eventkit import Event
from ib_async import (
    IB,
    ComboLeg,
    Contract,
    Future,
    FuturesOption,
    Index,
    LimitOrder,
    Option,
    OptionChain,
    Order,
    Stock,
    Ticker,
)
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from modules.Arbitrage.Strategy import ArbitrageClass, OrderManagerClass

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
)


logging.basicConfig(
    level=logging.INFO,  # Set the logging level here
    format="%(message)s",
    # datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        handler,
    ],
)

logger = logging.getLogger("rich")
os.environ["PYTHONASYNCIODEBUG"] = "1"


class BoxExecutor:
    def __init__(
        self,
        spread,
        strike1,
        strike2,
        order_manager: OrderManagerClass,
        ib,
        bid_contracts,
        ask_contracts,
        expiry,
        profit=0.05,
    ) -> None:
        self.order_manager = order_manager
        self.strike1 = strike1
        self.strike2 = strike2
        self.bid_contracts = bid_contracts
        self.ask_contracts = ask_contracts
        self.contracts = bid_contracts + ask_contracts
        self.spread = spread
        self.ib: IB = ib
        self.profit = profit
        self.expiry = expiry
        self.start_time = time.time()
        self.ask_contracts_price = []  # debug
        self.bid_contracts_price = []  # debug

    async def executor(self, event) -> None:

        for tick in event:
            ticker = tick
            contract = ticker.contract

            # contract_ticker[contract.conId] = ticker

            # condition to check that variables are not NaN
            ticker_variables = [ticker.askSize, ticker.bidSize, ticker.bid, ticker.ask]
            all_not_nan = all(not np.isnan(x) for x in ticker_variables)
            all_positive = all(x > 0 for x in ticker_variables)

            if (
                all_not_nan == True
                and all_positive == True
                and ticker.askSize >= 5
                and ticker.bidSize >= 5
            ):
                contract_ticker[contract.conId] = ticker

            elif all_not_nan == True and ticker.askSize < 5 and ticker.bidSize < 5:
                self.ib.pendingTickersEvent -= self.executor
                # logger.info(
                #     f"removed {contract.symbol} - {self.strike1}:{self.strike2} - {self.expiry}, ticker.askSize: {ticker.askSize} ticker.bidSize: {ticker.bidSize}"
                # )
                for contract in self.contracts:
                    self.ib.cancelMktData(contract)

                return

            if all(contract_ticker.get(c.conId) != None for c in self.contracts):
                logger.info(f"time to execution: { time.time() - self.start_time} sec")

                # calc limit price
                lmt_price = 0

                # Sell Contracts
                for c in self.bid_contracts:
                    ticker: Ticker = contract_ticker.get(c.conId)
                    ticker_midpoint = ticker.bid * 0.95
                    self.bid_contracts_price.append(ticker_midpoint)
                    lmt_price -= ticker_midpoint

                # but contracts
                for c in self.ask_contracts:
                    ticker = contract_ticker.get(c.conId)
                    ticker_midpoint = ticker.ask * 1.05
                    self.ask_contracts_price.append(ticker_midpoint)
                    lmt_price += ticker_midpoint
                logger.info(
                    f"[{contract.symbol}][{self.expiry}] - strikes: {self.strike2} - {self.strike1} - spread: {self.strike2 - self.strike1}  > lmt_price: {lmt_price + self.profit}"
                )
                if (self.strike2 - self.strike1) >= round(
                    (lmt_price + self.profit), 2
                ) and lmt_price > 0:
                    # print(
                    #     f"place order. details: ask_contracts_price + {self.ask_contracts_price} . bid_contracts_price - {self.bid_contracts_price} "
                    # )

                    box_contract, order = self.build_order(
                        contract.symbol,
                        self.bid_contracts,
                        self.ask_contracts,
                        round(lmt_price, 2),
                        exchange=contract.exchange,
                    )

                    trade = await self.order_manager.place_order(box_contract, order)
                    self.ib.pendingTickersEvent -= self.executor
                    # await asyncio.sleep(0.05)
                    lmt_price = 0

                    for c in self.contracts:
                        del contract_ticker[c.conId]
                        self.ib.cancelMktData(c)

                else:
                    # delete
                    for c in self.contracts:
                        del contract_ticker[c.conId]
                        self.ib.cancelMktData(c)

    def _calc_price(self, market_data, strike, call_data, put_data):
        call_price = call_data.bid if not np.isnan(call_data.bid) else call_data.close
        put_price = put_data.ask if not np.isnan(put_data.ask) else put_data.close

        # logger.info(f"Call Option close Price: {call_data.close}")
        # logger.info(f"Put Option close Price: {put_data.close}")
        # logger.info(f"Call Option bid: {call_data.bid}")
        # logger.info(f"Put Option ask: {put_data.ask}")

        stock_midpoint = market_data.midpoint()

        conversion_profit = call_price - put_price + (strike - stock_midpoint)
        lmt_price = round((stock_midpoint - call_price + put_price), 2)
        logger.debug(f"lmt_price:{lmt_price}")
        roi = (conversion_profit / lmt_price) * 100
        logger.debug(f"ROI: {roi}")
        return conversion_profit, lmt_price

    def build_order(
        self, symbol, bid_contracts, ask_contracts, lmt_price, exchange="SMART"
    ):

        combo_legs = []
        for c in bid_contracts:
            combo_legs.append(
                ComboLeg(conId=c.conId, ratio=1, action="SELL", exchange=c.exchange)
            )

        for c in ask_contracts:
            combo_legs.append(
                ComboLeg(conId=c.conId, ratio=1, action="BUY", exchange=c.exchange)
            )

        box_contract = Contract(
            symbol=symbol,
            comboLegs=combo_legs,
            exchange=exchange,
            secType="BAG",
            currency="USD",
        )

        order = LimitOrder(
            # orderId=self.ib.client.getReqId(),
            # orderType="LMT",
            action="BUY",
            totalQuantity=1,
            lmtPrice=lmt_price,
            # tif="DAY",
            # outsideRth=True,
        )

        return box_contract, order


class Box(ArbitrageClass):

    async def box_stock_scanner(self, symbol):
        # Define the underlying stock

        exchange, option_type, stock = self.get_stock_contract(symbol)

        # Request market data for the stock
        market_data = await self._get_market_data_async(stock)

        stock_price = (
            market_data.last if not np.isnan(market_data.last) else market_data.close
        )

        logger.warning(f"Market Price for {stock.symbol}: {stock_price}")

        if symbol == "!MES":
            chains = await self._get_chains(stock, exchange=exchange)

            tasks = []
            for chain in chains:
                task = asyncio.create_task(
                    self.search_box_in_chain(chain, option_type, stock, stock_price)
                )
                tasks.append(task)
            results = await asyncio.gather(*tasks)

        else:
            # Request options chain
            chain = await self._get_chain(stock, exchange=exchange)
            await self.search_box_in_chain(chain, option_type, stock, stock_price)

    async def search_box_in_chain(
        self, chain: OptionChain, option_type, stock, stock_price
    ):
        async with self.semaphore:

            anchor_strikes = [
                s
                for s in chain.strikes
                if s < stock_price * (1 + self.range)
                and s > stock_price * (1 - self.range)
            ]  # Example strike price

            profit = self.profit_target

            expirations_range = 1

            if len(chain.expirations) >= 5:
                expirations_range = 5

            for expiry in chain.expirations[:expirations_range]:

                strike_pairs = permutations(anchor_strikes, 2)
                for strike1, strike2 in strike_pairs:
                    if abs(strike2 - strike1) <= self.max_spread and strike1 < strike2:
                        # request market data for option strikes put call
                        self.call_s1 = option_type(
                            symbol=stock.symbol,
                            lastTradeDateOrContractMonth=expiry,
                            strike=strike1,
                            right="C",
                            exchange=chain.exchange,
                            tradingClass=chain.tradingClass,
                        )
                        self.put_s1 = option_type(
                            symbol=stock.symbol,
                            lastTradeDateOrContractMonth=expiry,
                            strike=strike1,
                            right="P",
                            exchange=chain.exchange,
                            tradingClass=chain.tradingClass,
                        )
                        self.call_s2 = option_type(
                            symbol=stock.symbol,
                            lastTradeDateOrContractMonth=expiry,
                            strike=strike2,
                            right="C",
                            exchange=chain.exchange,
                            tradingClass=chain.tradingClass,
                        )
                        self.put_s2 = option_type(
                            symbol=stock.symbol,
                            lastTradeDateOrContractMonth=expiry,
                            strike=strike2,
                            right="P",
                            exchange=chain.exchange,
                            tradingClass=chain.tradingClass,
                        )

                        contracts = [
                            self.call_s1,
                            self.put_s1,
                            self.call_s2,
                            self.put_s2,
                        ]

                        bid_contracts = [self.call_s2, self.put_s1]
                        ask_contracts = [self.put_s2, self.call_s1]

                        # Qualify the option contracts
                        await self.ib.qualifyContractsAsync(*contracts)
                        # Request market data for the options

                        for contract in contracts:
                            self.ib.reqMktData(contract)

                        # await asyncio.sleep(0.05)

                        box_executor = BoxExecutor(
                            spread=self.max_spread,
                            strike1=strike1,
                            strike2=strike2,
                            ib=self.ib,
                            order_manager=self.order_manager,
                            bid_contracts=bid_contracts,
                            ask_contracts=ask_contracts,
                            expiry=expiry,
                            profit=profit,
                        )

                        self.ib.pendingTickersEvent += box_executor.executor
                profit = profit * self.profit_target_multiplier

    def get_stock_contract(self, symbol):
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

    async def scan(
        self,
        symbol_list,
        range=0.1,
        profit_target=0.1,
        max_spread=10,
        profit_target_multiplier=1.10,
        clientId=3,
    ):
        # set configuration
        self.range = range
        self.profit_target = profit_target
        self.max_spread = max_spread
        self.profit_target_multiplier = profit_target_multiplier
        await self.ib.connectAsync("127.0.0.1", 7497, clientId=clientId)

        global contract_ticker
        global stock_ticker
        contract_ticker = {}
        stock_ticker = {}

        while True:
            tasks = []
            for symbol in symbol_list:
                task = asyncio.create_task(self.box_stock_scanner(symbol))
                tasks.append(task)
            results = await asyncio.gather(*tasks)

            contract_ticker = {}
            stock_ticker = {}

            await asyncio.sleep(25)
            contract_ticker = {}
            stock_ticker = {}
            self.ib.pendingTickersEvent = Event("pendingTickersEvent")
