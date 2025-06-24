from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modules.Arbitrage.SFR import SFRExecutor


@pytest.mark.unit
def test_sfr_executor_check_conditions_all_false_branches():
    # Setup dummy SFRExecutor
    sfr_executor = SFRExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(),
        option_contracts=[MagicMock(), MagicMock()],
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        expiry="20240101",
        start_time=0.0,
    )
    # 1. spread > net_credit
    assert not sfr_executor.check_conditions(
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        put_strike=90.0,
        lmt_price=50.0,
        net_credit=5.0,
        min_roi=20.0,
        stock_price=100.0,
        min_profit=10.0,
    )
    # 2. net_credit < 0
    assert not sfr_executor.check_conditions(
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        put_strike=90.0,
        lmt_price=50.0,
        net_credit=-1.0,
        min_roi=20.0,
        stock_price=90.0,
        min_profit=10.0,
    )
    # 3. profit_target > min_roi
    assert not sfr_executor.check_conditions(
        symbol="TEST",
        profit_target=30.0,
        cost_limit=100.0,
        put_strike=90.0,
        lmt_price=50.0,
        net_credit=5.0,
        min_roi=20.0,
        stock_price=90.0,
        min_profit=10.0,
    )
    # 4. np.isnan(lmt_price)
    assert not sfr_executor.check_conditions(
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        put_strike=90.0,
        lmt_price=np.nan,
        net_credit=5.0,
        min_roi=20.0,
        stock_price=90.0,
        min_profit=10.0,
    )
    # 5. lmt_price > cost_limit
    assert not sfr_executor.check_conditions(
        symbol="TEST",
        profit_target=10.0,
        cost_limit=50.0,
        put_strike=90.0,
        lmt_price=100.0,
        net_credit=5.0,
        min_roi=20.0,
        stock_price=90.0,
        min_profit=10.0,
    )


@pytest.mark.unit
def test_sfr_executor_check_conditions_true_branch():
    sfr_executor = SFRExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(),
        option_contracts=[MagicMock(), MagicMock()],
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        expiry="20240101",
        start_time=0.0,
    )
    # All conditions met
    assert sfr_executor.check_conditions(
        symbol="TEST",
        profit_target=5.0,
        cost_limit=100.0,
        put_strike=90.0,
        lmt_price=50.0,
        net_credit=10.0,
        min_roi=20.0,
        stock_price=90.0,
        min_profit=10.0,
    )


@pytest.mark.unit
def test_calc_price_and_build_order_no_stock_ticker():
    sfr_executor = SFRExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        option_contracts=[MagicMock(), MagicMock()],
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        expiry="20240101",
        start_time=0.0,
    )
    global contract_ticker
    contract_ticker = {}  # No ticker for stock
    result = sfr_executor.calc_price_and_build_order()
    assert result == (None, None)


@pytest.mark.unit
def test_calc_price_and_build_order_missing_option_data(monkeypatch):
    sfr_executor = SFRExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        option_contracts=[MagicMock(), MagicMock()],
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        expiry="20240101",
        start_time=0.0,
    )
    global contract_ticker
    contract_ticker = {1: MagicMock(ask=100.0, close=99.0)}
    monkeypatch.setattr(
        sfr_executor,
        "_extract_option_data",
        lambda _: (None, None, None, None, None, None),
    )
    result = sfr_executor.calc_price_and_build_order()
    assert result == (None, None)


@pytest.mark.unit
def test_calc_price_and_build_order_call_strike_less_than_put_strike(monkeypatch):
    sfr_executor = SFRExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        option_contracts=[MagicMock(), MagicMock()],
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        expiry="20240101",
        start_time=0.0,
    )
    global contract_ticker
    contract_ticker = {1: MagicMock(ask=100.0, close=99.0)}
    # call_strike < put_strike
    monkeypatch.setattr(
        sfr_executor,
        "_extract_option_data",
        lambda _: (MagicMock(), MagicMock(), 90.0, 100.0, 10.0, 5.0),
    )
    result = sfr_executor.calc_price_and_build_order()
    assert result == (None, None)


@pytest.mark.unit
def test_calc_price_and_build_order_check_conditions_false(monkeypatch):
    sfr_executor = SFRExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        option_contracts=[MagicMock(), MagicMock()],
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        expiry="20240101",
        start_time=0.0,
    )
    global contract_ticker
    contract_ticker = {1: MagicMock(ask=100.0, close=99.0)}
    # call_strike > put_strike, but check_conditions returns False
    monkeypatch.setattr(
        sfr_executor,
        "_extract_option_data",
        lambda _: (MagicMock(), MagicMock(), 110.0, 100.0, 10.0, 5.0),
    )
    monkeypatch.setattr(sfr_executor, "check_conditions", lambda *a, **kw: False)
    result = sfr_executor.calc_price_and_build_order()
    assert result == (None, None)


@pytest.mark.unit
def test_calc_price_and_build_order_check_conditions_true(monkeypatch):
    from modules.Arbitrage.SFR import SFRExecutor

    sfr_executor = SFRExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        option_contracts=[MagicMock(), MagicMock()],
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        expiry="20240101",
        start_time=0.0,
    )
    # Patch contract_ticker in the SFR module, not just locally
    monkeypatch.setattr(
        "modules.Arbitrage.SFR.contract_ticker", {1: MagicMock(ask=100.0, close=99.0)}
    )
    call_contract = object()
    put_contract = object()
    monkeypatch.setattr(
        sfr_executor,
        "_extract_option_data",
        lambda _: (call_contract, put_contract, 110.0, 100.0, 10.0, 5.0),
    )
    monkeypatch.setattr(sfr_executor, "check_conditions", lambda *a, **kw: True)
    with (
        patch.object(sfr_executor, "_log_trade_details") as mock_log,
        patch.object(
            sfr_executor, "build_order", return_value=("contract", "order")
        ) as mock_build,
    ):
        result = sfr_executor.calc_price_and_build_order()
        assert result == ("contract", "order")
        mock_log.assert_called()
        mock_build.assert_called()


@pytest.mark.unit
def test_sfr_executor_build_order_quantity():
    sfr_executor = SFRExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(),
        option_contracts=[MagicMock(), MagicMock()],
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        expiry="20240101",
        start_time=0.0,
        quantity=3,
    )
    # Patch contracts to have required attributes
    stock = MagicMock(conId=1)
    call = MagicMock(conId=2, right="C")
    put = MagicMock(conId=3, right="P")
    combo_contract, order = sfr_executor.build_order("TEST", stock, call, put, 99.0, 3)
    assert order.totalQuantity == 3
