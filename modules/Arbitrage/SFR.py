import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import logging
import numpy as np
from eventkit import Event
from ib_async import IB, ComboLeg, Contract, Option, Order, Stock, Ticker

from modules.Arbitrage.Strategy import ArbitrageClass, BaseExecutor, OrderManagerClass

from .common import configure_logging, get_logger
from .metrics import RejectionReason, metrics_collector

# Global variable to store debug mode
_debug_mode = False

# Configure logging will be done in main
logger = get_logger()

# Global contract_ticker for use in SFRExecutor and patching in tests
contract_ticker = {}


@dataclass
class ExpiryOption:
    """Data class to hold option contract information for a specific expiry"""

    expiry: str
    call_contract: Contract
    put_contract: Contract
    call_strike: float
    put_strike: float


class SFRExecutor(BaseExecutor):
    """
    Synthetic-Free-Risk (SFR) Executor class that handles the execution of SFR arbitrage strategies.

    This class is responsible for:
    1. Monitoring market data for stock and options across multiple expiries
    2. Calculating potential arbitrage opportunities
    3. Executing trades when conditions are met
    4. Logging trade details and results

    Attributes:
        ib (IB): Interactive Brokers connection instance
        order_manager (OrderManagerClass): Manager for handling order execution
        stock_contract (Contract): The underlying stock contract
        expiry_options (List[ExpiryOption]): List of option data for different expiries
        symbol (str): Trading symbol
        profit_target (float): Minimum profit target for execution
        cost_limit (float): Maximum price limit for execution
        start_time (float): Start time of the execution
        quantity (int): Quantity of contracts to execute
        all_contracts (List[Contract]): All contracts (stock + all options)
        is_active (bool): Whether the executor is currently active
        data_timeout (float): Maximum time to wait for all contract data (seconds)
    """

    def __init__(
        self,
        ib: IB,
        order_manager: OrderManagerClass,
        stock_contract: Contract,
        expiry_options: List[ExpiryOption],
        symbol: str,
        profit_target: float,
        cost_limit: float,
        start_time: float,
        quantity: int = 1,
        data_timeout: float = 30.0,  # 30 seconds timeout for data collection
    ) -> None:
        """
        Initialize the SFR Executor.

        Args:
            ib: Interactive Brokers connection instance
            order_manager: Manager for handling order execution
            stock_contract: The underlying stock contract
            expiry_options: List of option data for different expiries
            symbol: Trading symbol
            profit_target: Minimum profit target for execution
            cost_limit: Maximum price limit for execution
            start_time: Start time of the execution
            quantity: Quantity of contracts to execute
            data_timeout: Maximum time to wait for all contract data (seconds)
        """
        # Create list of all option contracts
        option_contracts = []
        for expiry_option in expiry_options:
            option_contracts.extend(
                [expiry_option.call_contract, expiry_option.put_contract]
            )

        super().__init__(
            ib,
            order_manager,
            stock_contract,
            option_contracts,
            symbol,
            cost_limit,
            expiry_options[0].expiry if expiry_options else "",
            start_time,
        )
        self.profit_target = profit_target
        self.quantity = quantity
        self.expiry_options = expiry_options
        self.all_contracts = [stock_contract] + option_contracts
        self.is_active = True
        self.data_timeout = data_timeout
        self.data_collection_start = time.time()

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
        min_profit: float,
    ) -> Tuple[bool, Optional[RejectionReason]]:

        spread = stock_price - put_strike

        if spread > net_credit:  # arbitrage condition
            logger.info(
                f"[{symbol}] spread[{spread}] > net_credit[{net_credit}] - doesn't meet conditions"
            )
            return False, RejectionReason.ARBITRAGE_CONDITION_NOT_MET

        elif net_credit < 0:
            logger.info(
                f"[{symbol}] net_credit[{net_credit}] > 0 - doesn't meet conditions"
            )
            return False, RejectionReason.NET_CREDIT_NEGATIVE
        elif profit_target is not None and profit_target > min_roi:
            logger.info(
                f"[{symbol}]  profit_target({profit_target}) >  min_roi({min_roi} - doesn't meet conditions) "
            )
            return False, RejectionReason.PROFIT_TARGET_NOT_MET
        elif np.isnan(lmt_price) or lmt_price > cost_limit:
            logger.info(
                f"[{symbol}] np.isnan(lmt_price) or lmt_price > limit - doesn't meet conditions"
            )
            return False, RejectionReason.PRICE_LIMIT_EXCEEDED

        else:
            logger.info(f"[{symbol}] meets conditions - initiating order")
            return True, None

    async def executor(self, event: Event) -> None:
        """
        Main executor method that processes market data events for all contracts.
        This method is called once per symbol and handles all expiries for that symbol.
        """
        if not self.is_active:
            return

        try:
            # Update contract_ticker with new data
            for tick in event:
                ticker: Ticker = tick
                contract = ticker.contract

                # Update ticker data - be more lenient with volume requirements
                # Accept any ticker with valid price data, log warning for low volume
                if ticker.volume > 10:
                    contract_ticker[contract.conId] = ticker
                elif ticker.volume >= 0 and (
                    ticker.bid > 0 or ticker.ask > 0 or ticker.close > 0
                ):
                    # Accept low volume contracts if they have valid price data
                    contract_ticker[contract.conId] = ticker
                    logger.warning(
                        f"[{self.symbol}] Low volume ({ticker.volume}) for contract {contract.conId}, "
                        f"but accepting due to valid price data"
                    )
                else:
                    logger.debug(f"Skipping contract {contract.conId}: no valid data")

            # Check for timeout
            elapsed_time = time.time() - self.data_collection_start
            if elapsed_time > self.data_timeout:
                missing_contracts = [
                    c
                    for c in self.all_contracts
                    if contract_ticker.get(c.conId) is None
                ]
                logger.warning(
                    f"[{self.symbol}] Data collection timeout after {elapsed_time:.1f}s. "
                    f"Missing data for {len(missing_contracts)} contracts out of {len(self.all_contracts)}"
                )
                # Log details of missing contracts
                for c in missing_contracts[:5]:  # Log first 5 missing
                    logger.info(
                        f"  Missing: {c.symbol} {c.right} {c.strike} {c.lastTradeDateOrContractMonth}"
                    )

                # Deactivate after timeout
                self.is_active = False
                # Finish scan with timeout error
                metrics_collector.finish_scan(
                    success=False, error_message="Data collection timeout"
                )
                return

            # Check if we have data for all contracts
            if all(
                contract_ticker.get(c.conId) is not None for c in self.all_contracts
            ):
                # Check if still active before proceeding
                if not self.is_active:
                    return

                logger.info(
                    f"[{self.symbol}] Fetched ticker for {len(self.all_contracts)} contracts"
                )
                execution_time = time.time() - self.start_time
                logger.info(f"time to execution: {execution_time} sec")
                metrics_collector.record_execution_time(execution_time)

                # Process all expiries
                best_opportunity = None
                best_profit = 0

                # TODO: Implement enhanced selection criteria
                # - Risk-reward ratio: max_profit / abs(min_profit)
                # - Time decay: favor closer expirations
                # - Liquidity: favor higher volume options
                # - Market spread: favor tighter bid-ask spreads
                # Example: score = min_profit * 0.5 + (max_profit/abs(min_profit)) * 0.3 + volume_score * 0.2

                for expiry_option in self.expiry_options:
                    opportunity = self.calc_price_and_build_order_for_expiry(
                        expiry_option
                    )
                    if (
                        opportunity and opportunity[2] > best_profit
                    ):  # opportunity[2] is min_profit
                        best_opportunity = opportunity
                        best_profit = opportunity[2]

                # TODO: Enhanced selection criteria could consider:
                # - Risk-reward ratio: max_profit / abs(min_profit)
                # - Time decay: favor closer expirations
                # - Liquidity: favor higher volume options
                # - Market spread: favor tighter bid-ask spreads
                # Example: score = min_profit * 0.5 + (max_profit/abs(min_profit)) * 0.3 + volume_score * 0.2

                # Execute the best opportunity
                if best_opportunity:
                    conversion_contract, order, _, trade_details = best_opportunity
                    if conversion_contract and order:
                        # Log trade details right before placing the order
                        logger.info(
                            f"[{self.symbol}] About to place trade for expiry: {trade_details['expiry']}"
                        )
                        self._log_trade_details(
                            trade_details["call_strike"],
                            trade_details["call_price"],
                            trade_details["put_strike"],
                            trade_details["put_price"],
                            trade_details["stock_price"],
                            trade_details["net_credit"],
                            trade_details["min_profit"],
                            trade_details["max_profit"],
                            trade_details["min_roi"],
                        )

                        self.is_active = (
                            False  # Deactivate to prevent multiple executions
                        )
                        trade = await self.order_manager.place_order(
                            conversion_contract, order
                        )
                        logger.info(f"Executed best opportunity for {self.symbol}")
                        metrics_collector.record_opportunity_found()
                        # Finish scan successfully when an order is placed
                        metrics_collector.finish_scan(success=True)
                        # Deactivate immediately after order placement
                        self.deactivate()

                else:
                    logger.info(f"No suitable opportunities found for {self.symbol}")
                    self.is_active = False
                    # Finish scan successfully even if no opportunities found
                    metrics_collector.finish_scan(success=True)

            else:
                # Still waiting for data from some contracts
                missing_contracts = [
                    c
                    for c in self.all_contracts
                    if contract_ticker.get(c.conId) is None
                ]
                logger.debug(
                    f"[{self.symbol}] Still waiting for data from {len(missing_contracts)} contracts "
                    f"(elapsed: {elapsed_time:.1f}s)"
                )

        except Exception as e:
            logger.error(f"Error in executor: {str(e)}")
            self.is_active = False
            # Finish scan with error
            metrics_collector.finish_scan(success=False, error_message=str(e))

    def calc_price_and_build_order_for_expiry(
        self, expiry_option: ExpiryOption
    ) -> Optional[Tuple[Contract, Order, float, Dict]]:
        """
        Calculate price and build order for a specific expiry option.
        Returns tuple of (contract, order, min_profit, trade_details) or None if no opportunity.
        """
        try:
            # Get stock data
            stock_ticker = contract_ticker.get(self.stock_contract.conId)
            if not stock_ticker:
                metrics_collector.add_rejection_reason(
                    RejectionReason.MISSING_MARKET_DATA,
                    {
                        "symbol": self.symbol,
                        "contract_type": "stock",
                        "expiry": expiry_option.expiry,
                        "reason": "no stock ticker data",
                    },
                )
                return None

            stock_price = (
                stock_ticker.ask
                if not np.isnan(stock_ticker.ask)
                else stock_ticker.close
            )

            # Get option data
            call_ticker = contract_ticker.get(expiry_option.call_contract.conId)
            put_ticker = contract_ticker.get(expiry_option.put_contract.conId)

            if not call_ticker or not put_ticker:
                metrics_collector.add_rejection_reason(
                    RejectionReason.MISSING_MARKET_DATA,
                    {
                        "symbol": self.symbol,
                        "contract_type": "options",
                        "expiry": expiry_option.expiry,
                        "call_strike": expiry_option.call_strike,
                        "put_strike": expiry_option.put_strike,
                        "missing_call_data": not call_ticker,
                        "missing_put_data": not put_ticker,
                    },
                )
                return None

            # Get validated prices for individual legs
            call_price = (
                call_ticker.bid if not np.isnan(call_ticker.bid) else call_ticker.close
            )
            put_price = (
                put_ticker.ask if not np.isnan(put_ticker.ask) else put_ticker.close
            )

            if np.isnan(call_price) or np.isnan(put_price):
                metrics_collector.add_rejection_reason(
                    RejectionReason.INVALID_CONTRACT_DATA,
                    {
                        "symbol": self.symbol,
                        "contract_type": "options",
                        "expiry": expiry_option.expiry,
                        "call_strike": expiry_option.call_strike,
                        "put_strike": expiry_option.put_strike,
                        "call_price_invalid": np.isnan(call_price),
                        "put_price_invalid": np.isnan(put_price),
                    },
                )
                return None

            # Check bid-ask spread for both call and put contracts to prevent crazy price ranges
            call_bid_ask_spread = (
                abs(call_ticker.ask - call_ticker.bid)
                if (not np.isnan(call_ticker.ask) and not np.isnan(call_ticker.bid))
                else float("inf")
            )
            put_bid_ask_spread = (
                abs(put_ticker.ask - put_ticker.bid)
                if (not np.isnan(put_ticker.ask) and not np.isnan(put_ticker.bid))
                else float("inf")
            )

            if call_bid_ask_spread > 15:
                logger.info(
                    f"[{self.symbol}] Call contract bid-ask spread too wide: {call_bid_ask_spread:.2f} > 15.00, "
                    f"expiry: {expiry_option.expiry}, strike: {expiry_option.call_strike}"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.BID_ASK_SPREAD_TOO_WIDE,
                    {
                        "symbol": self.symbol,
                        "contract_type": "call",
                        "expiry": expiry_option.expiry,
                        "strike": expiry_option.call_strike,
                        "bid_ask_spread": call_bid_ask_spread,
                        "threshold": 15.0,
                    },
                )
                return None

            if put_bid_ask_spread > 15:
                logger.info(
                    f"[{self.symbol}] Put contract bid-ask spread too wide: {put_bid_ask_spread:.2f} > 15.00, "
                    f"expiry: {expiry_option.expiry}, strike: {expiry_option.put_strike}"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.BID_ASK_SPREAD_TOO_WIDE,
                    {
                        "symbol": self.symbol,
                        "contract_type": "put",
                        "expiry": expiry_option.expiry,
                        "strike": expiry_option.put_strike,
                        "bid_ask_spread": put_bid_ask_spread,
                        "threshold": 15.0,
                    },
                )
                return None

            # Calculate net credit
            net_credit = call_price - put_price
            stock_price = round(stock_price, 2)
            net_credit = round(net_credit, 2)
            call_price = round(call_price, 2)
            put_price = round(put_price, 2)

            # temp condition
            if expiry_option.call_strike < expiry_option.put_strike:
                logger.info(
                    f"call_strike:{expiry_option.call_strike} < put_strike:{expiry_option.put_strike}"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.INVALID_STRIKE_COMBINATION,
                    {
                        "symbol": self.symbol,
                        "expiry": expiry_option.expiry,
                        "call_strike": expiry_option.call_strike,
                        "put_strike": expiry_option.put_strike,
                        "reason": "call_strike < put_strike",
                    },
                )
                return None

            spread = stock_price - expiry_option.put_strike
            min_profit = net_credit - spread
            max_profit = (
                expiry_option.call_strike - expiry_option.put_strike
            ) + net_credit
            min_roi = (min_profit / (stock_price + net_credit)) * 100

            # Calculate precise combo limit price based on target leg prices
            combo_limit_price = self.calculate_combo_limit_price(
                stock_price=stock_price,
                call_price=call_price,
                put_price=put_price,
                buffer_percent=0.00,  # 1.5% buffer for better execution
            )

            logger.info(
                f"[{self.symbol}] Expiry: {expiry_option.expiry} min_profit:{min_profit:.2f}, max_profit:{max_profit:.2f}, min_roi:{min_roi:.2f}%"
            )

            conditions_met, rejection_reason = self.check_conditions(
                self.symbol,
                self.profit_target,
                self.cost_limit,
                expiry_option.put_strike,
                combo_limit_price,  # Use calculated precise limit price
                net_credit,
                min_roi,
                stock_price,
                min_profit,
            )

            if conditions_met:
                # Build order with precise limit price and target leg prices
                conversion_contract, order = self.build_order(
                    self.symbol,
                    self.stock_contract,
                    expiry_option.call_contract,
                    expiry_option.put_contract,
                    combo_limit_price,  # Use calculated precise limit price
                    self.quantity,
                    call_price=call_price,  # Target call leg price
                    put_price=put_price,  # Target put leg price
                )

                # Prepare trade details for logging (don't log yet)
                trade_details = {
                    "call_strike": expiry_option.call_strike,
                    "call_price": call_price,
                    "put_strike": expiry_option.put_strike,
                    "put_price": put_price,
                    "stock_price": stock_price,
                    "net_credit": net_credit,
                    "min_profit": min_profit,
                    "max_profit": max_profit,
                    "min_roi": min_roi,
                    "expiry": expiry_option.expiry,
                }

                return conversion_contract, order, min_profit, trade_details
            else:
                # Record rejection reason
                if rejection_reason:
                    # Calculate profit_ratio for context
                    profit_ratio = (
                        max_profit / abs(min_profit) if min_profit != 0 else 0
                    )

                    metrics_collector.add_rejection_reason(
                        rejection_reason,
                        {
                            "symbol": self.symbol,
                            "expiry": expiry_option.expiry,
                            "call_strike": expiry_option.call_strike,
                            "put_strike": expiry_option.put_strike,
                            "stock_price": stock_price,
                            "net_credit": net_credit,
                            "min_profit": min_profit,
                            "max_profit": max_profit,
                            "min_roi": min_roi,
                            "combo_limit_price": combo_limit_price,
                            "cost_limit": self.cost_limit,
                            "profit_target": self.profit_target,
                            "spread": spread,
                            "profit_ratio": profit_ratio,
                        },
                    )

            return None
        except Exception as e:
            logger.error(f"Error in calc_price_and_build_order_for_expiry: {str(e)}")
            return None


class SFR(ArbitrageClass):
    """
    Synthetic-Free-Risk (SFR) arbitrage strategy class.
    This class uses a more efficient approach by creating one executor per symbol
    that handles all expiries, eliminating the need to constantly add/remove event handlers.
    """

    def __init__(self, log_file: str = None, debug: bool = False):
        global _debug_mode
        _debug_mode = debug
        # Reconfigure logging with debug mode
        configure_logging(level=logging.INFO, debug=debug)
        super().__init__(log_file=log_file)

    async def scan(
        self,
        symbol_list,
        cost_limit,
        profit_target=0.50,
        volume_limit=100,
        quantity=1,
    ):
        """
        scan for SFR and execute order

        symbol list - list of valid symbols
        cost_limit - min price for the contract. e.g limit=50 means willing to pay up to 5000$
        profit_target - min acceptable min roi
        volume_limit - min option contract volume
        quantity - number of contracts to trade
        """
        # Global
        global contract_ticker
        contract_ticker = {}

        # set configuration
        self.profit_target = profit_target
        self.volume_limit = volume_limit
        self.cost_limit = cost_limit
        self.quantity = quantity

        await self.ib.connectAsync("127.0.0.1", 7497, clientId=2)
        self.ib.orderStatusEvent += self.onFill

        # Set up single event handler for all symbols
        self.ib.pendingTickersEvent += self.master_executor

        while True:
            # Start cycle tracking
            cycle_metrics = metrics_collector.start_cycle(len(symbol_list))

            tasks = []
            for symbol in symbol_list:
                task = asyncio.create_task(self.scan_sfr(symbol, self.quantity))
                tasks.append(task)
                await asyncio.sleep(2)
            _ = await asyncio.gather(*tasks)

            # Clean up inactive executors
            self.cleanup_inactive_executors()

            # Finish cycle tracking
            metrics_collector.finish_cycle()

            # Print metrics summary periodically
            if len(metrics_collector.scan_metrics) > 0:
                metrics_collector.print_summary()

            # Reset for next iteration
            contract_ticker = {}
            await asyncio.sleep(30)  # Wait before next scan cycle

    async def scan_sfr(self, symbol, quantity):
        """
        Scan for SFR opportunities for a specific symbol.
        Creates a single executor per symbol that handles all expiries.
        """
        # Start metrics collection for this scan
        scan_metrics = metrics_collector.start_scan(symbol, "SFR")

        try:
            exchange, option_type, stock = self._get_stock_contract(symbol)

            # Request market data for the stock
            market_data = await self._get_market_data_async(stock)

            stock_price = (
                market_data.last
                if not np.isnan(market_data.last)
                else market_data.close
            )

            logger.info(f"price for [{symbol}: {stock_price} ]")

            # Request options chain
            chain = await self._get_chain(stock, exchange="SMART")

            # Define parameters for the options (expiry and strike price)
            valid_strikes = [
                s for s in chain.strikes if s <= stock_price and s > stock_price - 10
            ]  # Example strike price

            if len(valid_strikes) < 2:
                logger.info(
                    f"Not enough valid strikes found for {symbol} (found: {len(valid_strikes)})"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.INSUFFICIENT_VALID_STRIKES,
                    {
                        "symbol": symbol,
                        "valid_strikes_count": len(valid_strikes),
                        "required_strikes": 2,
                        "stock_price": stock_price,
                    },
                )
                return

            # Collect all expiry options
            expiry_options = []
            all_contracts = [stock]

            for expiry in self.filter_expirations_within_range(
                chain.expirations, 19, 45
            ):
                if len(valid_strikes) == 0:
                    continue

                await asyncio.sleep(0.05)

                call_strike = valid_strikes[-1]
                put_strike = valid_strikes[-2]

                call = Option(stock.symbol, expiry, call_strike, "C", "SMART")
                put = Option(stock.symbol, expiry, put_strike, "P", "SMART")

                # Qualify the option contracts
                valid_contracts = await self.ib.qualifyContractsAsync(call, put)

                # Find call and put contracts
                call_contract = None
                put_contract = None

                for contract in valid_contracts:
                    if contract.right == "C":
                        call_contract = contract
                    elif contract.right == "P":
                        put_contract = contract

                # If call contract is invalid, continue to next iteration
                if not call_contract:
                    logger.info(
                        f"Invalid call- [{call_strike}] contract for {symbol} expiry {expiry}, skipping"
                    )
                    continue

                # If put contract is invalid, try shifting one strike below
                if not put_contract:
                    put_strike_index = valid_strikes.index(put_strike)
                    if put_strike_index > 0:  # Can shift one strike below
                        new_put_strike = valid_strikes[put_strike_index - 1]
                        logger.info(
                            f"Retrying put contract for {symbol} with strike {new_put_strike}"
                        )

                        put_retry = Option(
                            stock.symbol, expiry, new_put_strike, "P", "SMART"
                        )
                        valid_contracts_retry = await self.ib.qualifyContractsAsync(
                            put_retry
                        )

                        for contract in valid_contracts_retry:
                            if contract.right == "P":
                                put_contract = contract
                                put_strike = (
                                    new_put_strike  # Update the actual strike used
                                )
                                break

                        if not put_contract:
                            logger.warning(
                                f"Still no valid put contract for {symbol} expiry {expiry} after retry, skipping"
                            )
                            continue
                    else:
                        logger.warning(
                            f"Cannot shift put strike below for {symbol} expiry {expiry}, skipping"
                        )
                        continue

                # Both contracts are valid at this point, create ExpiryOption
                if call_contract and put_contract:
                    expiry_option = ExpiryOption(
                        expiry=expiry,
                        call_contract=call_contract,
                        put_contract=put_contract,
                        call_strike=call_strike,
                        put_strike=put_strike,
                    )
                    expiry_options.append(expiry_option)
                    all_contracts.extend([call_contract, put_contract])
                else:
                    logger.warning(
                        f"Could not find both call and put contracts for {symbol} expiry {expiry}"
                    )

            if not expiry_options:
                logger.info(f"No valid expiry options found for {symbol}")
                metrics_collector.add_rejection_reason(
                    RejectionReason.INVALID_CONTRACT_DATA,
                    {
                        "symbol": symbol,
                        "reason": "no valid expiry options found",
                        "total_expiries_checked": len(
                            self.filter_expirations_within_range(
                                chain.expirations, 19, 45
                            )
                        ),
                    },
                )
                return

            # Log the total number of combinations being scanned
            logger.info(
                f"[{symbol}] Scanning {len(expiry_options)} strike combinations across "
                f"{len(set(opt.expiry for opt in expiry_options))} expiries"
            )

            # Create single executor for this symbol
            srf_executor = SFRExecutor(
                ib=self.ib,
                order_manager=self.order_manager,
                stock_contract=stock,
                expiry_options=expiry_options,
                symbol=symbol,
                profit_target=self.profit_target,
                cost_limit=self.cost_limit,
                start_time=time.time(),
                quantity=quantity,
                data_timeout=45.0,  # Give more time for data collection
            )

            # Store executor and request market data for all contracts
            self.active_executors[symbol] = srf_executor

            # Clean up any stale data in contract_ticker for this symbol's contracts
            for contract in all_contracts:
                if contract.conId in contract_ticker:
                    del contract_ticker[contract.conId]
                    logger.debug(f"Cleaned up stale data for contract {contract.conId}")

            # Request market data for all contracts with detailed logging
            logger.info(
                f"[{symbol}] Requesting market data for {len(all_contracts)} contracts:"
            )

            # Log stock contract
            logger.info(f"  Stock: {stock.symbol} (conId: {stock.conId})")

            # Log option contracts
            for expiry_option in expiry_options:
                logger.info(
                    f"  Call: {expiry_option.call_contract.symbol} {expiry_option.call_strike} "
                    f"{expiry_option.expiry} (conId: {expiry_option.call_contract.conId})"
                )
                logger.info(
                    f"  Put: {expiry_option.put_contract.symbol} {expiry_option.put_strike} "
                    f"{expiry_option.expiry} (conId: {expiry_option.put_contract.conId})"
                )

            # Request market data for all contracts in parallel
            data_collection_start = time.time()
            await self.request_market_data_batch(all_contracts)
            data_collection_time = time.time() - data_collection_start

            # Record metrics
            metrics_collector.record_contracts_count(len(all_contracts))
            metrics_collector.record_data_collection_time(data_collection_time)
            metrics_collector.record_expiries_scanned(len(expiry_options))

            logger.info(
                f"Created executor for {symbol} with {len(expiry_options)} expiry options "
                f"({len(all_contracts)} total contracts)"
            )

            # Don't finish scan here - let the executor finish it when done processing
            # The executor will call metrics_collector.finish_scan() when it's inactive

        except Exception as e:
            logger.error(f"Error in scan_sfr for {symbol}: {str(e)}")
            metrics_collector.finish_scan(success=False, error_message=str(e))
