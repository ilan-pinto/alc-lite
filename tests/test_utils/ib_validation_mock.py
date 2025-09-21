"""
Realistic Interactive Brokers API mock with proper price validation.

This module provides mocks that simulate actual IB API behavior including:
- Price precision validation (2 decimal places for stocks/options)
- Minimum price variation errors
- Contract type validation
- Order type restrictions
"""

import re
from decimal import Decimal
from typing import Dict, List, Optional
from unittest.mock import MagicMock

from ib_async import Contract, Order, OrderStatus, Trade


class IBValidationError(Exception):
    """Exception that mimics IB API validation errors."""

    pass


class RealisticIBMock:
    """
    Mock IB connection that validates orders like the real IB API.

    This mock catches the types of errors that real IB would throw,
    particularly price precision and minimum price variation errors.
    """

    def __init__(self):
        self.client = MagicMock()
        self.client.getReqId = MagicMock(side_effect=self._get_req_id)
        self._req_id_counter = 1000
        self._placed_orders = []
        self._cancelled_orders = []

        # Configure minimum price variations
        self._min_price_variations = {
            "STK": 0.01,  # Stocks: $0.01 (1 cent)
            "OPT": 0.01,  # Options: $0.01 (1 cent) for most strikes
            "FUT": 0.0001,  # Futures: varies by contract
        }

        # Track order statuses
        self._order_statuses: Dict[int, str] = {}

    def _get_req_id(self) -> int:
        """Generate unique request IDs."""
        self._req_id_counter += 1
        return self._req_id_counter

    def _validate_price_precision(self, price: float, contract: Contract) -> None:
        """
        Validate price precision against IB requirements.

        Args:
            price: Order price to validate
            contract: Contract being traded

        Raises:
            IBValidationError: If price doesn't meet IB precision requirements
        """
        if price <= 0:
            raise IBValidationError(f"Price must be positive, got {price}")

        # Convert to Decimal for precise decimal place counting
        price_decimal = Decimal(str(price))

        # Check decimal places
        decimal_places = abs(price_decimal.as_tuple().exponent)

        # IB typically requires 2 decimal places maximum for stocks and options
        max_decimal_places = 2
        if decimal_places > max_decimal_places:
            raise IBValidationError(
                f"The price {price} does not conform to minimum price variation "
                f"for this contract. Maximum {max_decimal_places} decimal places allowed."
            )

        # Check minimum price variation
        sec_type = getattr(contract, "secType", "STK")
        min_variation = self._min_price_variations.get(sec_type, 0.01)

        # Price should be a multiple of minimum variation (with small tolerance for floating point)
        remainder = price % min_variation
        if remainder > 1e-10 and (min_variation - remainder) > 1e-10:
            raise IBValidationError(
                f"The price {price} does not conform to minimum price variation "
                f"of {min_variation} for {sec_type} contracts"
            )

    def _validate_contract(self, contract: Contract) -> None:
        """Validate contract parameters."""
        if not hasattr(contract, "symbol") or not contract.symbol:
            raise IBValidationError("Contract must have a valid symbol")

        if not hasattr(contract, "secType") or not contract.secType:
            raise IBValidationError("Contract must have a valid security type")

    def _validate_order(self, order: Order) -> None:
        """Validate order parameters."""
        if not hasattr(order, "action") or order.action not in ["BUY", "SELL"]:
            raise IBValidationError("Order action must be BUY or SELL")

        if not hasattr(order, "totalQuantity") or order.totalQuantity <= 0:
            raise IBValidationError("Order quantity must be positive")

        if not hasattr(order, "orderType") or not order.orderType:
            raise IBValidationError("Order must have a valid order type")

    def placeOrder(self, contract: Contract, order: Order) -> Trade:
        """
        Mock placeOrder that validates like real IB API.

        Args:
            contract: Trading contract
            order: Order to place

        Returns:
            Mock trade object

        Raises:
            IBValidationError: If order/contract validation fails
        """
        # Validate contract
        self._validate_contract(contract)

        # Validate order
        self._validate_order(order)

        # Validate price precision for limit orders
        if order.orderType == "LMT" and hasattr(order, "lmtPrice"):
            self._validate_price_precision(order.lmtPrice, contract)

        # Create mock trade
        trade = MagicMock(spec=Trade)
        trade.contract = contract
        trade.order = order

        # Create mock order status
        order_status = MagicMock(spec=OrderStatus)
        order_status.orderId = order.orderId
        order_status.status = "Submitted"
        order_status.filled = 0
        order_status.remaining = order.totalQuantity
        order_status.avgFillPrice = 0.0

        trade.orderStatus = order_status

        # Track order
        self._placed_orders.append((contract, order, trade))
        self._order_statuses[order.orderId] = "Submitted"

        return trade

    def cancelOrder(self, order: Order) -> None:
        """Mock order cancellation."""
        if hasattr(order, "orderId"):
            self._cancelled_orders.append(order.orderId)
            self._order_statuses[order.orderId] = "Cancelled"

    def simulate_fill(
        self, order_id: int, fill_price: float, filled_quantity: Optional[int] = None
    ) -> None:
        """
        Simulate order fill for testing.

        Args:
            order_id: Order ID to fill
            fill_price: Price at which order filled
            filled_quantity: Quantity filled (defaults to full order)
        """
        # Find the trade
        for contract, order, trade in self._placed_orders:
            if order.orderId == order_id:
                if filled_quantity is None:
                    filled_quantity = order.totalQuantity

                # Update order status
                trade.orderStatus.status = (
                    "Filled"
                    if filled_quantity == order.totalQuantity
                    else "PartiallyFilled"
                )
                trade.orderStatus.filled = filled_quantity
                trade.orderStatus.remaining = order.totalQuantity - filled_quantity
                trade.orderStatus.avgFillPrice = fill_price

                self._order_statuses[order_id] = trade.orderStatus.status
                break

    def get_placed_orders(self) -> List[tuple]:
        """Get list of all placed orders for test verification."""
        return self._placed_orders.copy()

    def get_cancelled_orders(self) -> List[int]:
        """Get list of cancelled order IDs for test verification."""
        return self._cancelled_orders.copy()

    def reset(self) -> None:
        """Reset mock state for new test."""
        self._placed_orders.clear()
        self._cancelled_orders.clear()
        self._order_statuses.clear()
        self._req_id_counter = 1000


# Utility functions for creating realistic test data
def create_realistic_stock_price(
    base_price: float = 100.0, volatility: float = 0.02
) -> float:
    """
    Create realistic stock price with proper decimal precision.

    Args:
        base_price: Base stock price
        volatility: Price volatility factor

    Returns:
        Stock price with realistic precision (2 decimal places)
    """
    import random

    variation = base_price * volatility * (random.random() - 0.5)
    raw_price = base_price + variation
    # Round to 2 decimal places (cent precision)
    return round(raw_price, 2)


def create_realistic_option_price(
    intrinsic_value: float = 5.0, time_value: float = 2.0
) -> float:
    """
    Create realistic option price with proper decimal precision.

    Args:
        intrinsic_value: Option intrinsic value
        time_value: Option time value

    Returns:
        Option price with realistic precision (2 decimal places)
    """
    import random

    # Add some random variation
    variation = 0.50 * (random.random() - 0.5)
    raw_price = intrinsic_value + time_value + variation
    # Ensure positive and round to 2 decimal places
    return max(0.01, round(raw_price, 2))


def create_problematic_price(base_price: float) -> float:
    """
    Create a price with too many decimal places that should trigger validation errors.

    Args:
        base_price: Base price to modify

    Returns:
        Price with excessive decimal precision
    """
    # Add extra decimal places that would cause IB validation errors
    return base_price + 0.001234


def create_mock_contract(symbol: str, sec_type: str = "STK", **kwargs) -> Contract:
    """
    Create a realistic mock contract.

    Args:
        symbol: Stock/option symbol
        sec_type: Security type (STK, OPT, etc.)
        **kwargs: Additional contract parameters

    Returns:
        Mock contract with realistic properties
    """
    contract = MagicMock(spec=Contract)
    contract.symbol = symbol
    contract.secType = sec_type
    contract.currency = kwargs.get("currency", "USD")
    contract.exchange = kwargs.get("exchange", "SMART")

    if sec_type == "OPT":
        contract.strike = kwargs.get("strike", 100.0)
        contract.right = kwargs.get("right", "C")  # Call or Put
        contract.expiry = kwargs.get("expiry", "20241220")

    return contract


def create_mock_order(
    action: str,
    quantity: int,
    order_type: str = "LMT",
    lmt_price: Optional[float] = None,
) -> Order:
    """
    Create a realistic mock order.

    Args:
        action: BUY or SELL
        quantity: Order quantity
        order_type: Order type (LMT, MKT, etc.)
        lmt_price: Limit price for limit orders

    Returns:
        Mock order with realistic properties
    """
    order = MagicMock(spec=Order)
    order.action = action
    order.totalQuantity = quantity
    order.orderType = order_type
    order.tif = "DAY"

    if order_type == "LMT" and lmt_price is not None:
        order.lmtPrice = lmt_price

    # Assign realistic order ID
    import time

    order.orderId = int(time.time() * 1000) % 1000000

    return order
