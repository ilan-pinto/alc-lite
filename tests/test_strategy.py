import asyncio
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from modules.Arbitrage.Strategy import BaseExecutor, OrderManagerClass


class DummyIB:
    def __init__(self):
        self.client = MagicMock()
        self.client.getReqId.return_value = 123


class DummyContract:
    def __init__(self, conId, right=None, strike=None):
        self.conId = conId
        self.right = right
        self.strike = strike


class DummyTicker:
    def __init__(self, contract, midpoint_val=np.nan, close_val=0, ask_val=np.nan):
        self.contract = contract
        self._midpoint = midpoint_val
        self.close = close_val
        self.ask = ask_val

    def midpoint(self):
        return self._midpoint


def make_executor():
    ib = DummyIB()
    order_manager = MagicMock()
    stock_contract = DummyContract(1)
    call_contract = DummyContract(2, right="C", strike=100)
    put_contract = DummyContract(3, right="P", strike=90)
    return (
        BaseExecutor(
            ib=ib,
            order_manager=order_manager,
            stock_contract=stock_contract,
            option_contracts=[call_contract, put_contract],
            symbol="TEST",
            cost_limit=10.0,
            expiry="20240101",
        ),
        call_contract,
        put_contract,
        stock_contract,
    )


@pytest.mark.unit
def test_extract_option_data_returns_correct_values():
    executor, call_contract, put_contract, _ = make_executor()
    contract_ticker = {
        call_contract.conId: DummyTicker(
            call_contract, midpoint_val=1.5, close_val=1.4
        ),
        put_contract.conId: DummyTicker(put_contract, ask_val=2.0, close_val=1.8),
    }
    result = executor._extract_option_data(contract_ticker)
    assert result[0] == call_contract
    assert result[1] == put_contract
    assert result[2] == 100
    assert result[3] == 90
    assert result[4] == 1.5
    assert result[5] == 2.0


@pytest.mark.unit
def test_extract_option_data_handles_missing_ticker():
    executor, call_contract, put_contract, _ = make_executor()
    contract_ticker = {call_contract.conId: DummyTicker(call_contract)}
    result = executor._extract_option_data(contract_ticker)
    assert result == (None, None, None, None, None, None)


@pytest.mark.unit
def test_build_order_creates_combo_contract_and_order():
    executor, call_contract, put_contract, stock_contract = make_executor()
    lmt_price = 10.5
    combo_contract, order = executor.build_order(
        symbol="TEST",
        stock=stock_contract,
        call=call_contract,
        put=put_contract,
        lmt_price=lmt_price,
    )
    assert combo_contract.symbol == "TEST"
    assert combo_contract.secType == "BAG"
    assert combo_contract.currency == "USD"
    assert len(combo_contract.comboLegs) == 3
    assert order.orderType == "LMT"
    assert order.lmtPrice == lmt_price
    assert order.action == "BUY"
    assert order.totalQuantity == 1
    assert order.tif == "DAY"


@pytest.mark.unit
def test_order_handler_filled_triggers_cancel_and_disconnect():
    ib = MagicMock()
    order_manager = OrderManagerClass(ib=ib)
    # Mock openTrades returns two trades
    trade1 = MagicMock()
    trade1.order.orderId = 1
    trade2 = MagicMock()
    trade2.order.orderId = 2
    ib.openTrades.return_value = [trade1, trade2]
    # Patch OrderStatus.Filled in the module under test
    with (
        patch("modules.Arbitrage.Strategy.OrderStatus") as MockOrderStatus,
        patch("modules.Arbitrage.Strategy.logger"),
    ):
        MockOrderStatus.Filled = "Filled"
        event = MagicMock()
        event.orderStatus.status = "Filled"
        order_manager.order_handler(event)
    # Should cancel both orders and disconnect
    assert ib.cancelOrder.call_count == 2
    assert ib.cancelOrder.call_args_list == [call(trade1.order), call(trade2.order)]
    ib.disconnect.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_place_order_places_and_cancels_order(monkeypatch):
    ib = MagicMock()
    order_manager = OrderManagerClass(ib=ib)
    contract = MagicMock()
    order = MagicMock()
    # _check_position_exists returns False, _check_any_trade_exists returns False
    order_manager._check_position_exists = MagicMock(return_value=False)
    order_manager._check_any_trade_exists = MagicMock(return_value=False)
    ib.placeOrder.return_value = "trade_obj"

    # Patch asyncio.sleep to avoid real sleep
    async def fake_sleep(x):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    # Patch logger to silence output
    with patch("modules.Arbitrage.Strategy.logger"):
        result = await order_manager.place_order(contract, order)
    ib.placeOrder.assert_called_once_with(contract=contract, order=order)
    ib.cancelOrder.assert_called_once_with(order)
    assert result == "trade_obj"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_place_order_skips_if_position_or_trade_exists(monkeypatch):
    ib = MagicMock()
    order_manager = OrderManagerClass(ib=ib)
    contract = MagicMock()
    order = MagicMock()
    # _check_position_exists returns True, _check_any_trade_exists returns True
    order_manager._check_position_exists = MagicMock(return_value=True)
    order_manager._check_any_trade_exists = MagicMock(return_value=True)
    # Patch logger to silence output
    with patch("modules.Arbitrage.Strategy.logger"):
        result = await order_manager.place_order(contract, order)
    ib.placeOrder.assert_not_called()
    ib.cancelOrder.assert_not_called()
    assert result is None
