import asyncio
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from ib_async import IB, ComboLeg, Contract, Order, Stock, Option, Ticker
import numpy as np
from modules.Arbitrage.Strategy import ArbitrageClass, OrderManagerClass, BaseExecutor
import time
from eventkit import Event
from .common import configure_logging, get_logger

# Configure logging
configure_logging(level=logging.INFO)
logger = get_logger()


class SynExecutor(BaseExecutor):
    """
    Synthetic not free risk (Syn) Executor class that handles the execution of Syn Synthetic option strategies.

    This class is responsible for:
    1. Monitoring market data for stock and options
    2. Calculating potential arbitrage opportunities
    3. Executing trades when conditions are met
    4. Logging trade details and results

    Attributes:
        ib (IB): Interactive Brokers connection instance
        order_manager (OrderManagerClass): Manager for handling order execution
        stock_contract (Contract): The underlying stock contract
        option_contracts (List[Contract]): List of option contracts (call and put)
        symbol (str): Trading symbol
        cost_limit (float): Maximum price limit for execution
        max_loss (float): Maximum loss for execution
        max_profit (float): Maximum profit for execution
        expiry (str): Option expiration date
    """

    def __init__(
        self,
        ib: IB,
        order_manager: OrderManagerClass,
        stock_contract: Contract,
        option_contracts: List[Contract],
        symbol: str,
        cost_limit: float,
        max_loss: float,
        max_profit: float,
        expiry: str,
    ) -> None:
        """
        Initialize the Syn Executor.

        Args:
            ib: Interactive Brokers connection instance
            order_manager: Manager for handling order execution
            stock_contract: The underlying stock contract
            option_contracts: List of option contracts (call and put)
            symbol: Trading symbol
            cost_limit: Maximum price limit for execution
            max_loss: Maximum loss for execution
            max_profit: Maximum profit for execution
            expiry: Option expiration date
        """
        super().__init__(ib, order_manager, stock_contract, option_contracts, 
                        symbol, cost_limit, expiry, time.time())
        self.max_loss = max_loss
        self.max_profit = max_profit

    def check_conditions(
        self,
        symbol: str,
        profit_target: float,
        cost_limit: float,
        put_strike: float,
        lmt_price: float,
        net_credit: float,
        min_roi: float,
        stock_price: float,
        max_loss: float,
        min_profit: float,
    ) -> bool:

        spread = stock_price - put_strike

        if max_loss >= min_profit:  # no arbitrage condition
            logger.info(f"max_loss limit [{max_loss}] >  calculated max_loss [{min_profit}]")
            return False

        elif net_credit > 0:
            logger.info(f"[{symbol}] net_credit[{net_credit}] > 0")
            return False
        elif profit_target is not None and profit_target > min_roi:
            logger.info(
                f"[{symbol}]  profit_target({profit_target}) >  min_roi({min_roi}) "
            )
            return False
        elif np.isnan(lmt_price) or lmt_price > cost_limit:
            logger.info(f"[{symbol}] np.isnan(lmt_price) or lmt_price > limit")
            return False

        else:
            return True

    async def executor(self, event: Event) -> None:
        try:
            for tick in event:
                ticker: Ticker = tick
                contract = ticker.contract
                contract_ticker[contract.conId] = ticker

                if ticker.volume > 10:
                    contract_ticker[contract.conId] = ticker
                else:
                    self.ib.pendingTickersEvent -= self.executor
                    logger.debug(f"removed {self.symbol} - {tick.last} - {self.expiry}")
                    return

                self.contracts = [self.stock_contract] + self.option_contracts
                if all(
                    contract_ticker.get(c.conId) is not None for c in self.contracts
                ):
                    logger.info(
                        f"time to execution: {time.time() - self.start_time} sec"
                    )

                    self.ib.pendingTickersEvent -= self.executor

                    # calc limit price
                    conversion_contract, order = self.calc_price_and_build_order()

                    if order and conversion_contract:
                        trade = await self.order_manager.place_order(
                            conversion_contract, order
                        )

        except Exception as e:
            logger.error(f"Error in executor: {str(e)}")
            self.ib.pendingTickersEvent -= self.executor

    def calc_price_and_build_order(self) -> Tuple[Optional[Contract], Optional[Order]]:
        try:
            net_credit = 0
            stock_price = 0

            stock_data = self.stock_contract
            ticker = contract_ticker.get(self.stock_contract.conId)
            if not ticker:
                logger.error("No ticker data for stock contract")
                return None, None

            stock_midpoint = ticker.ask if not np.isnan(ticker.ask) else ticker.close
            stock_price += stock_midpoint

            # Extract option data using base class method
            call_contract, put_contract, call_strike, put_strike, call_price, put_price = (
                self._extract_option_data(contract_ticker)
            )

            if not all([call_contract, put_contract, call_strike, put_strike, call_price, put_price]):
                logger.error("Missing required option data")
                return None, None

            # Calculate net credit
            net_credit = put_price - call_price

            # temp condition
            if call_strike < put_strike:
                logger.info(f"call_strike:{call_strike} < put_strike:{put_strike}")
                return None, None

            stock_price = round(stock_price, 2)
            net_credit = round(net_credit, 2)

            spread = stock_price - put_strike

            min_profit = - (-net_credit) - spread  # max loss 
            max_profit = (call_strike - put_strike) - min_profit

            min_roi = (min_profit / (stock_price + net_credit)) * 100

            logger.info(
                f"[{self.symbol}] min_profit:{min_profit}, max_profit:{max_profit}, min_roi:[{min_roi}]"
            )

            if self.check_conditions(
                self.symbol,
                getattr(self, 'profit_target', None),
                self.cost_limit,
                put_strike,
                stock_price + net_credit,
                net_credit,
                min_roi,  # min ROI
                stock_price,
                self.max_loss,
                min_profit,
            ):
                self._log_trade_details(
                    call_strike,
                    call_price,
                    put_strike,
                    put_price,
                    stock_price,
                    net_credit,
                    min_profit,
                    max_profit,
                    min_roi,
                )

                return self.build_order(
                    self.symbol,
                    stock_data,
                    call_contract,
                    put_contract,
                    round(stock_price + net_credit, 2),
                )

            return None, None
        except Exception as e:
            logger.error(f"Error in calc_price_and_build_order: {str(e)}")
            return None, None


class Syn(ArbitrageClass):

    async def scan(
        self,
        symbol_list,
        cost_limit,
        max_loss,
        max_profit
    ):
        """
        scan for Syn and execute order

        symbol list - list of valid symbols
        cost_limit - min price for the contract. e.g limit=50 means willing to pay up to 5000$
        max_loss - max loss for the contract. e.g max_loss=50 means willing to lose up to 5000$
        max_profit - max profit for the contract. e.g max_profit=50 means willing to profit up to 5000$
        """
        # Global
        global contract_ticker
        global stock_ticker
        contract_ticker = {}
        stock_ticker = {}

        # set configuration 
        self.cost_limit = cost_limit
        self.max_loss = max_loss    
        self.max_profit = max_profit
        await self.ib.connectAsync("127.0.0.1", 7497, clientId=2)
        # self.ib.reqMarketDataType = 3

        self.ib.orderStatusEvent += self.onFill

        while True:
            tasks = []
            for symbol in symbol_list:
                task = asyncio.create_task(self.scan_syn(symbol))
                tasks.append(task)
                await asyncio.sleep(2)
            _ = await asyncio.gather(*tasks)

            contract_ticker = {}
            stock_ticker = {}
            self.ib.pendingTickersEvent = Event("pendingTickersEvent")

    async def scan_syn(self, symbol):
        exchange, option_type, stock = self._get_stock_contract(symbol)

        # Request market data for the stock
        market_data = await self._get_market_data_async(stock)

        stock_price = (
            market_data.last if not np.isnan(market_data.last) else market_data.close
        )

        logger.info(f"price for [{symbol}: {stock_price} ]")

        # Request options chain
        chain = await self._get_chain(stock)

        # Define parameters for the options (expiry and strike price)
        valid_strikes = [
            s for s in chain.strikes if s <= stock_price and s > stock_price - 10
        ]  # Example strike price

        for expiry in self.filter_expirations_within_range(chain.expirations, 19, 45):

            if len(valid_strikes) == 0:
                continue

            await asyncio.sleep(0.05)
            call = Option(stock.symbol, expiry, valid_strikes[-1], "C", "SMART")
            put = Option(stock.symbol, expiry, valid_strikes[-2], "P", "SMART")
            # Qualify the option contracts
            valid_contracts = await self.ib.qualifyContractsAsync(call, put)
            if len(valid_contracts) == 0:
                continue

            # Request market data for the options
            self.ib.reqMktData(stock)
            self.ib.reqMktData(call)
            self.ib.reqMktData(put)

            syn = SynExecutor(
                ib=self.ib,
                order_manager=self.order_manager,
                stock_contract=stock,
                option_contracts=[call, put],
                symbol=symbol,
                cost_limit=self.cost_limit,
                max_loss=self.max_loss,
                max_profit=self.max_profit,
                expiry=expiry,
            )

            self.ib.pendingTickersEvent += syn.executor
