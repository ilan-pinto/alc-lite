"""
Realistic market data generators for comprehensive testing.

This module provides functions to generate realistic market data scenarios
that mirror actual trading conditions, including:
- Bid/ask spreads
- Realistic price movements
- Options pricing with proper Greeks
- Partial fill scenarios
- Market stress conditions
"""

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
from ib_async import Contract, Ticker


@dataclass
class MarketScenario:
    """Represents a complete market scenario for testing."""

    symbol: str
    stock_price: float
    stock_bid: float
    stock_ask: float
    call_price: float
    call_bid: float
    call_ask: float
    put_price: float
    put_bid: float
    put_ask: float
    spread_width: float  # Average bid-ask spread as percentage
    volatility: float  # Implied volatility for options


class MarketDataGenerator:
    """Generator for realistic market data scenarios."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed for reproducible tests."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_stock_ticker(
        self,
        symbol: str,
        base_price: float = 100.0,
        spread_bps: int = 5,  # Spread in basis points (0.05%)
    ) -> Ticker:
        """
        Generate realistic stock ticker with bid/ask spread.

        Args:
            symbol: Stock symbol
            base_price: Base stock price
            spread_bps: Bid-ask spread in basis points

        Returns:
            Mock ticker with realistic bid/ask data
        """
        # Calculate spread in dollars
        spread_dollars = base_price * (spread_bps / 10000.0)
        half_spread = spread_dollars / 2.0

        # Add some price movement
        price_movement = base_price * 0.002 * (random.random() - 0.5)  # ±0.2% movement
        mid_price = base_price + price_movement

        # Calculate bid/ask
        bid_price = round(mid_price - half_spread, 2)
        ask_price = round(mid_price + half_spread, 2)

        # Create mock ticker
        ticker = MagicMock(spec=Ticker)
        ticker.bid = bid_price
        ticker.ask = ask_price
        ticker.last = round(mid_price + random.uniform(-half_spread, half_spread), 2)
        ticker.close = round(base_price, 2)

        # Mock midpoint calculation
        ticker.midpoint = MagicMock(
            return_value=round((bid_price + ask_price) / 2.0, 2)
        )

        return ticker

    def generate_option_ticker(
        self,
        symbol: str,
        strike: float,
        option_type: str,  # 'C' for Call, 'P' for Put
        base_price: float = 5.0,
        spread_percent: float = 0.03,  # 3% spread
    ) -> Ticker:
        """
        Generate realistic option ticker with wider spreads.

        Args:
            symbol: Underlying symbol
            strike: Option strike price
            option_type: 'C' for Call, 'P' for Put
            base_price: Base option price
            spread_percent: Bid-ask spread as percentage of price

        Returns:
            Mock ticker with realistic option bid/ask data
        """
        # Options have wider spreads than stocks
        spread_dollars = base_price * spread_percent
        half_spread = spread_dollars / 2.0

        # Add some price movement based on option type and moneyness
        price_movement = base_price * 0.05 * (random.random() - 0.5)  # ±5% movement
        mid_price = max(0.01, base_price + price_movement)  # Ensure positive

        # Calculate bid/ask
        bid_price = max(0.01, round(mid_price - half_spread, 2))
        ask_price = round(mid_price + half_spread, 2)

        # Create mock ticker
        ticker = MagicMock(spec=Ticker)
        ticker.bid = bid_price
        ticker.ask = ask_price
        ticker.last = round(mid_price + random.uniform(-half_spread, half_spread), 2)
        ticker.close = round(base_price, 2)

        # Mock midpoint calculation
        ticker.midpoint = MagicMock(
            return_value=round((bid_price + ask_price) / 2.0, 2)
        )

        return ticker

    def generate_market_scenario(
        self,
        symbol: str = "SPY",
        stock_price: float = 155.0,
        call_strike: float = 155.0,
        put_strike: float = 155.0,
        market_condition: str = "normal",  # "normal", "volatile", "wide_spread"
    ) -> MarketScenario:
        """
        Generate complete market scenario with correlated stock and options data.

        Args:
            symbol: Trading symbol
            stock_price: Current stock price
            call_strike: Call option strike
            put_strike: Put option strike
            market_condition: Market condition affecting spreads and volatility

        Returns:
            MarketScenario with correlated data
        """
        # Adjust parameters based on market condition
        if market_condition == "volatile":
            stock_spread_bps = 10  # 0.1% spread
            option_spread_percent = 0.06  # 6% spread
            volatility = 0.35
        elif market_condition == "wide_spread":
            stock_spread_bps = 20  # 0.2% spread
            option_spread_percent = 0.10  # 10% spread
            volatility = 0.25
        else:  # normal
            stock_spread_bps = 5  # 0.05% spread
            option_spread_percent = 0.03  # 3% spread
            volatility = 0.20

        # Generate stock data
        stock_ticker = self.generate_stock_ticker(symbol, stock_price, stock_spread_bps)

        # Calculate option intrinsic values (simplified)
        call_intrinsic = max(0, stock_price - call_strike)
        put_intrinsic = max(0, put_strike - stock_price)

        # Add time value (simplified)
        time_value = 2.0 + random.uniform(-0.5, 0.5)
        call_base_price = call_intrinsic + time_value
        put_base_price = put_intrinsic + time_value

        # Generate option tickers
        call_ticker = self.generate_option_ticker(
            symbol, call_strike, "C", call_base_price, option_spread_percent
        )
        put_ticker = self.generate_option_ticker(
            symbol, put_strike, "P", put_base_price, option_spread_percent
        )

        return MarketScenario(
            symbol=symbol,
            stock_price=stock_ticker.midpoint(),
            stock_bid=stock_ticker.bid,
            stock_ask=stock_ticker.ask,
            call_price=call_ticker.midpoint(),
            call_bid=call_ticker.bid,
            call_ask=call_ticker.ask,
            put_price=put_ticker.midpoint(),
            put_bid=put_ticker.bid,
            put_ask=put_ticker.ask,
            spread_width=(option_spread_percent + stock_spread_bps / 10000) / 2,
            volatility=volatility,
        )

    def generate_problematic_scenario(self, symbol: str = "MU") -> MarketScenario:
        """
        Generate scenario that would trigger pricing precision issues.

        This creates prices with excessive decimal places that would cause
        the IB API validation errors we experienced with MU.

        Args:
            symbol: Trading symbol

        Returns:
            MarketScenario with problematic pricing
        """
        # Base scenario
        scenario = self.generate_market_scenario(symbol, stock_price=156.78)

        # Add excessive decimal precision that would trigger IB errors
        scenario.stock_price = 156.7823456  # Too many decimal places
        scenario.stock_bid = 156.77121
        scenario.stock_ask = 156.79334

        scenario.call_price = 16.504789  # Problematic call price
        scenario.call_bid = 16.497234
        scenario.call_ask = 16.512345

        scenario.put_price = 8.602134  # Problematic put price
        scenario.put_bid = 8.596789
        scenario.put_ask = 8.607890

        return scenario

    def generate_partial_fill_scenario(
        self, symbol: str = "AAPL", fill_probability: Dict[str, float] = None
    ) -> Tuple[MarketScenario, Dict[str, bool]]:
        """
        Generate scenario where only some legs fill quickly.

        Args:
            symbol: Trading symbol
            fill_probability: Probability each leg type will fill

        Returns:
            Tuple of (MarketScenario, fill_outcomes)
        """
        if fill_probability is None:
            fill_probability = {
                "stock": 0.9,  # Stock usually fills
                "call": 0.3,  # Call might not fill
                "put": 0.4,  # Put might not fill
            }

        scenario = self.generate_market_scenario(symbol)

        # Determine which legs will fill based on probabilities
        fill_outcomes = {
            "stock": random.random() < fill_probability["stock"],
            "call": random.random() < fill_probability["call"],
            "put": random.random() < fill_probability["put"],
        }

        return scenario, fill_outcomes


def create_realistic_ticker_dict(scenario: MarketScenario) -> Dict[tuple, object]:
    """
    Create contract_ticker dictionary from market scenario.

    Args:
        scenario: Market scenario with pricing data

    Returns:
        Dictionary mapping (symbol, contract_id) to ticker objects
    """
    # Create mock contracts
    stock_contract = MagicMock(spec=Contract)
    stock_contract.symbol = scenario.symbol
    stock_contract.secType = "STK"
    stock_contract.conId = 12345

    call_contract = MagicMock(spec=Contract)
    call_contract.symbol = scenario.symbol
    call_contract.secType = "OPT"
    call_contract.right = "C"
    call_contract.conId = 23456

    put_contract = MagicMock(spec=Contract)
    put_contract.symbol = scenario.symbol
    put_contract.secType = "OPT"
    put_contract.right = "P"
    put_contract.conId = 34567

    # Create tickers
    stock_ticker = MagicMock(spec=Ticker)
    stock_ticker.bid = scenario.stock_bid
    stock_ticker.ask = scenario.stock_ask
    stock_ticker.midpoint = MagicMock(return_value=scenario.stock_price)

    call_ticker = MagicMock(spec=Ticker)
    call_ticker.bid = scenario.call_bid
    call_ticker.ask = scenario.call_ask
    call_ticker.midpoint = MagicMock(return_value=scenario.call_price)

    put_ticker = MagicMock(spec=Ticker)
    put_ticker.bid = scenario.put_bid
    put_ticker.ask = scenario.put_ask
    put_ticker.midpoint = MagicMock(return_value=scenario.put_price)

    # Return contract_ticker dictionary
    return {
        (scenario.symbol, stock_contract.conId): stock_ticker,
        (scenario.symbol, call_contract.conId): call_ticker,
        (scenario.symbol, put_contract.conId): put_ticker,
    }


def simulate_market_movement(
    scenario: MarketScenario, time_steps: int = 10, volatility_factor: float = 1.0
) -> List[MarketScenario]:
    """
    Simulate market movement over time for dynamic testing.

    Args:
        scenario: Initial market scenario
        time_steps: Number of time steps to simulate
        volatility_factor: Multiplier for price volatility

    Returns:
        List of market scenarios showing price evolution
    """
    scenarios = [scenario]
    current_scenario = scenario

    for _ in range(time_steps):
        # Calculate price changes
        stock_change = (
            current_scenario.stock_price
            * 0.001
            * volatility_factor
            * random.gauss(0, 1)
        )

        # Update stock price
        new_stock_price = max(0.01, current_scenario.stock_price + stock_change)

        # Create new scenario with evolved prices
        generator = MarketDataGenerator()
        new_scenario = generator.generate_market_scenario(
            current_scenario.symbol, new_stock_price
        )

        scenarios.append(new_scenario)
        current_scenario = new_scenario

    return scenarios
