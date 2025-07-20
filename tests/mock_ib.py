"""
Enhanced Mock Interactive Brokers API for comprehensive arbitrage testing.
This module provides realistic simulation of IB API for testing when markets are closed.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from unittest.mock import MagicMock

from eventkit import Event
from ib_async import Contract, Option, OptionChain, Stock, Ticker


class MockTicker:
    """Enhanced mock ticker with realistic market data"""

    def __init__(
        self,
        contract: Contract,
        bid: float = 0.0,
        ask: float = 0.0,
        close: float = 0.0,
        volume: int = 100,
        last: float = None,
    ):
        self.contract = contract
        self.bid = bid
        self.ask = ask
        self.close = close
        self.volume = volume
        self.last = (
            last
            if last is not None
            else (bid + ask) / 2 if bid > 0 and ask > 0 else close
        )
        self.time = datetime.now()

    def midpoint(self):
        """Calculate midpoint price"""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.close


class MockContract:
    """Enhanced mock contract that mimics IB contract behavior"""

    def __init__(
        self,
        symbol: str,
        conId: int = None,
        secType: str = "STK",
        exchange: str = "SMART",
        currency: str = "USD",
        **kwargs,
    ):
        self.symbol = symbol
        self.conId = conId or hash(f"{symbol}_{secType}_{kwargs}")
        self.secType = secType
        self.exchange = exchange
        self.currency = currency

        # Option-specific attributes
        if secType == "OPT":
            self.right = kwargs.get("right", "C")
            self.strike = kwargs.get("strike", 100.0)
            self.lastTradeDateOrContractMonth = kwargs.get("expiry", "20250830")
        else:
            self.right = None
            self.strike = None
            self.lastTradeDateOrContractMonth = None


class MarketDataGenerator:
    """Generates realistic market data for testing arbitrage scenarios"""

    @staticmethod
    def generate_stock_data(
        symbol: str, price: float, volume: int = 1000000
    ) -> MockTicker:
        """Generate stock market data"""
        spread = price * 0.001  # 0.1% spread
        bid = round(price - spread / 2, 2)
        ask = round(price + spread / 2, 2)

        contract = MockContract(symbol, secType="STK")
        # Ensure stock contracts don't have option attributes
        contract.right = None
        contract.strike = None
        contract.lastTradeDateOrContractMonth = None
        return MockTicker(
            contract, bid=bid, ask=ask, close=price, volume=volume, last=price
        )

    @staticmethod
    def generate_option_data(
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        stock_price: float,
        days_to_expiry: int = 30,
    ) -> MockTicker:
        """Generate realistic option market data"""
        contract = MockContract(
            symbol, secType="OPT", right=right, strike=strike, expiry=expiry
        )

        # Calculate intrinsic value
        if right == "C":
            intrinsic = max(0, stock_price - strike)
        else:  # Put
            intrinsic = max(0, strike - stock_price)

        # Calculate time value (simplified Black-Scholes approximation)
        time_value = max(
            0.05, min(5.0, abs(stock_price - strike) * 0.02 * (days_to_expiry / 30))
        )

        # Theoretical price
        theoretical = intrinsic + time_value

        # Add bid-ask spread (wider for options)
        spread_pct = 0.02 + (
            0.01 * abs(stock_price - strike) / stock_price
        )  # 2-3% spread
        spread = theoretical * spread_pct

        bid = max(0.01, round(theoretical - spread / 2, 2))
        ask = round(theoretical + spread / 2, 2)

        # Volume based on moneyness (ATM options have higher volume)
        moneyness = abs(stock_price - strike) / stock_price
        base_volume = 1000
        volume = max(10, int(base_volume * (1 - moneyness * 2)))

        return MockTicker(contract, bid=bid, ask=ask, close=theoretical, volume=volume)

    @classmethod
    def create_dell_arbitrage_scenario(cls) -> Dict[int, MockTicker]:
        """Create DELL scenario with known conversion arbitrage opportunity"""
        stock_price = 131.24
        tickers = {}

        # Stock data
        stock_ticker = cls.generate_stock_data("DELL", stock_price)
        tickers[stock_ticker.contract.conId] = stock_ticker

        # Create specific arbitrage opportunity: Call 132 / Put 131
        expiry = "20250221"  # ~30 days out

        # Call 132 - slightly OTM, good premium
        call_132 = cls.generate_option_data("DELL", expiry, 132.0, "C", stock_price, 30)
        # Adjust for arbitrage opportunity - make call expensive
        call_132.bid = 1.80
        call_132.ask = 2.00
        tickers[call_132.contract.conId] = call_132

        # Put 131 - ATM put, lower cost
        put_131 = cls.generate_option_data("DELL", expiry, 131.0, "P", stock_price, 30)
        # Adjust for arbitrage opportunity - make put cheaper
        put_131.bid = 1.40
        put_131.ask = 1.60
        tickers[put_131.contract.conId] = put_131

        # Add additional strikes for testing
        for strike in [128, 129, 130, 133, 134, 135]:
            for right in ["C", "P"]:
                ticker = cls.generate_option_data(
                    "DELL", expiry, strike, right, stock_price, 30
                )
                tickers[ticker.contract.conId] = ticker

        return tickers


class MockIB:
    """Comprehensive mock of Interactive Brokers API"""

    def __init__(self):
        self.client = MagicMock()
        self.client.getReqId.return_value = 123
        self.connected = False
        self.reqId_counter = 1000

        # Event system
        self.orderStatusEvent = Event()
        self.pendingTickersEvent = Event()

        # Storage for market data requests
        self.market_data_requests = {}
        self.qualified_contracts = {}
        self.option_chains = {}

        # Track background tasks for proper cleanup
        self.background_tasks = set()

        # Pre-populate with test data
        self._setup_test_data()

    def cleanup(self):
        """Cancel all background tasks to prevent asyncio warnings"""
        try:
            # Check if we're in an asyncio context
            loop = asyncio.get_running_loop()
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
        except RuntimeError:
            # Event loop is closed or not running, tasks are already cleaned up
            pass
        finally:
            self.background_tasks.clear()

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup

    def _setup_test_data(self):
        """Setup test data for common symbols"""
        # DELL arbitrage scenario
        self.test_market_data = MarketDataGenerator.create_dell_arbitrage_scenario()

    async def connectAsync(
        self, host: str = "127.0.0.1", port: int = 7497, clientId: int = 1
    ):
        """Mock connection to IB"""
        await asyncio.sleep(0.1)  # Simulate connection delay
        self.connected = True
        return True

    def disconnect(self):
        """Mock disconnection"""
        self.connected = False

    async def qualifyContractsAsync(self, *contracts) -> List[Contract]:
        """Mock contract qualification"""
        qualified = []

        for contract in contracts:
            # Create qualified contract based on input
            if hasattr(contract, "symbol"):
                qualified_contract = MockContract(
                    symbol=contract.symbol,
                    secType=getattr(contract, "secType", "STK"),
                    exchange=getattr(contract, "exchange", "SMART"),
                    right=getattr(contract, "right", None),
                    strike=getattr(contract, "strike", None),
                    expiry=getattr(contract, "lastTradeDateOrContractMonth", None),
                )
                qualified.append(qualified_contract)
                # Store for later reference
                self.qualified_contracts[qualified_contract.conId] = qualified_contract

        return qualified

    def reqMktData(
        self,
        contract: Contract,
        genericTickList: str = "",
        snapshot: bool = False,
        regulatorySnapshot: bool = False,
        mktDataOptions: List = None,
    ):
        """Mock market data request"""
        # Generate a reqId automatically (like real ib_async)
        reqId = self.reqId_counter
        self.reqId_counter += 1

        # Store the request
        self.market_data_requests[reqId] = contract

        # Simulate async market data arrival
        task = asyncio.create_task(self._deliver_market_data(reqId, contract))
        self.background_tasks.add(task)
        # Remove task from set when it completes
        task.add_done_callback(self.background_tasks.discard)

        # Return a mock ticker object immediately (like real ib_async)
        return self._create_mock_ticker_for_contract(contract)

    def _create_mock_ticker_for_contract(self, contract: Contract):
        """Create a mock ticker for the given contract"""
        # First try to find exact match in test data
        for test_ticker in self.test_market_data.values():
            if (
                test_ticker.contract.symbol == contract.symbol
                and getattr(test_ticker.contract, "right", None)
                == getattr(contract, "right", None)
                and getattr(test_ticker.contract, "strike", None)
                == getattr(contract, "strike", None)
            ):
                # Return a copy with the requested contract
                ticker = MockTicker(
                    contract,
                    test_ticker.bid,
                    test_ticker.ask,
                    test_ticker.close,
                    test_ticker.volume,
                    test_ticker.last,
                )
                return ticker

        # Get appropriate stock price for symbol
        stock_prices = {
            "AAPL": 185.50,
            "MSFT": 415.75,
            "TSLA": 245.80,
            "META": 495.25,
            "NVDA": 875.40,
            "AMZN": 155.90,
            "DELL": 131.24,
        }
        stock_price = stock_prices.get(contract.symbol, 131.24)

        if hasattr(contract, "right") and contract.right:  # Option
            # Apply symbol-specific scenarios even if strikes don't match exactly
            ticker = MarketDataGenerator.generate_option_data(
                contract.symbol,
                getattr(contract, "lastTradeDateOrContractMonth", "20250830"),
                getattr(contract, "strike", 100.0),
                getattr(contract, "right", "C"),
                stock_price,
            )

            # Override with symbol-specific profit/loss characteristics
            symbol = contract.symbol
            strike = getattr(contract, "strike", 100.0)
            right = getattr(contract, "right", "C")

            if symbol == "AAPL":
                # AAPL should be profitable
                if right == "C":  # Selling call
                    ticker.bid = 4.20  # High bid for selling
                    ticker.ask = 4.40
                else:  # Buying put
                    ticker.bid = 1.80
                    ticker.ask = 2.00  # Low ask for buying
            elif symbol == "MSFT":
                # MSFT should NOT be profitable (low net credit)
                if right == "C":  # Selling call
                    ticker.bid = 1.20  # Low bid for selling
                    ticker.ask = 1.40
                else:  # Buying put
                    ticker.bid = 0.90
                    ticker.ask = 1.10  # High ask for buying - creates low net credit
            elif symbol == "TSLA":
                # TSLA should have NEGATIVE net credit
                if right == "C":  # Selling call
                    ticker.bid = 0.80  # Very low bid for selling
                    ticker.ask = 1.00
                else:  # Buying put
                    ticker.bid = 2.20
                    ticker.ask = (
                        2.40  # Very high ask for buying - creates negative net credit
                    )
            elif symbol in ["META", "NVDA", "AMZN"]:
                # These should all be unprofitable for different reasons
                if symbol == "NVDA":
                    # NVDA: Wide spreads
                    if right == "C":
                        ticker.bid = 15.00
                        ticker.ask = 40.00  # Very wide spread > 20
                    else:
                        ticker.bid = 12.00
                        ticker.ask = 35.00  # Very wide spread > 20
                elif symbol == "META":
                    # META: No arbitrage
                    if right == "C":
                        ticker.bid = 2.10
                        ticker.ask = 2.30
                    else:
                        ticker.bid = 1.80
                        ticker.ask = 2.00  # Creates low net credit scenario
                elif symbol == "AMZN":
                    # AMZN: Negative net credit
                    if right == "C":
                        ticker.bid = 1.10  # Low call premium
                        ticker.ask = 1.30
                    else:
                        ticker.bid = 2.80
                        ticker.ask = 3.00  # Expensive put creates negative net credit
        else:  # Stock
            ticker = MarketDataGenerator.generate_stock_data(
                contract.symbol, stock_price
            )

        # Update contract reference
        ticker.contract = contract
        return ticker

    async def _deliver_market_data(self, reqId: int, contract: Contract):
        """Simulate market data delivery"""
        # Skip sleep in tests to avoid pending task warnings
        # await asyncio.sleep(0.1)  # Simulate data arrival delay

        # Find matching ticker in test data
        ticker = None

        # Ensure we have a proper contract object
        if not hasattr(contract, "symbol"):
            print(
                f"Warning: contract is not a proper Contract object: {type(contract)} = {contract}"
            )
            return

        for test_ticker in self.test_market_data.values():
            if (
                test_ticker.contract.symbol == contract.symbol
                and getattr(test_ticker.contract, "right", None)
                == getattr(contract, "right", None)
                and getattr(test_ticker.contract, "strike", None)
                == getattr(contract, "strike", None)
            ):
                ticker = test_ticker
                break

        if not ticker:
            # Generate default ticker if not found using the same logic as _create_mock_ticker_for_contract
            ticker = self._create_mock_ticker_for_contract(contract)

        # Update contract reference
        ticker.contract = contract

        # Trigger pending tickers event
        self.pendingTickersEvent.emit([ticker])

    def cancelMktData(self, reqId: int):
        """Mock cancel market data"""
        if reqId in self.market_data_requests:
            del self.market_data_requests[reqId]

    async def reqSecDefOptParamsAsync(
        self,
        underlyingSymbol: str,
        futFopExchange: str = "",
        underlyingSecType: str = "STK",
        underlyingConId: int = 0,
    ) -> List:
        """Mock option chain parameters request"""
        # Return mock option chain data
        mock_params = MagicMock()
        mock_params.exchange = "SMART"
        mock_params.underlyingConId = 12345
        mock_params.tradingClass = underlyingSymbol
        mock_params.multiplier = "100"

        # Generate strikes around current price (131.24 for DELL)
        base_price = 131.24 if underlyingSymbol == "DELL" else 100.0
        strikes = []
        for i in range(-10, 11):
            strikes.append(base_price + i)
        mock_params.strikes = strikes

        # Generate expiries (next 3 months)
        expiries = []
        current_date = datetime.now()
        for i in range(1, 4):
            future_date = current_date + timedelta(days=30 * i)
            expiry_str = future_date.strftime("%Y%m%d")
            expiries.append(expiry_str)
        mock_params.expirations = expiries

        return [mock_params]

    def reqIds(self, numIds: int = 1):
        """Mock request for valid order IDs"""
        self.reqId_counter += numIds
        return self.reqId_counter

    async def placeOrderAsync(self, contract: Contract, order) -> object:
        """Mock order placement"""
        # Create mock trade object
        trade = MagicMock()
        trade.contract = contract
        trade.order = order
        trade.orderStatus.status = "Submitted"

        # Simulate order processing
        await asyncio.sleep(0.1)
        trade.orderStatus.status = "Filled"

        return trade
