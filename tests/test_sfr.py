from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modules.Arbitrage.sfr.executor import SFRExecutor
from modules.Arbitrage.sfr.models import ExpiryOption


@pytest.mark.unit
def test_sfr_executor_check_conditions_all_false_branches():
    # Setup dummy SFRExecutor
    expiry_options = [
        ExpiryOption(
            expiry="20250830",
            call_contract=MagicMock(),
            put_contract=MagicMock(),
            call_strike=100.0,
            put_strike=95.0,
        )
    ]
    sfr_executor = SFRExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(),
        expiry_options=expiry_options,
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        start_time=0.0,
        quantity=1,
    )
    # 1. spread > net_credit
    result, reason = sfr_executor.check_conditions(
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
    assert not result
    assert reason is not None

    # 2. net_credit < 0
    result, reason = sfr_executor.check_conditions(
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
    assert not result
    assert reason is not None

    # 3. profit_target > min_roi
    result, reason = sfr_executor.check_conditions(
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
    assert not result
    assert reason is not None

    # 4. np.isnan(lmt_price)
    result, reason = sfr_executor.check_conditions(
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
    assert not result
    assert reason is not None

    # 5. lmt_price > cost_limit
    result, reason = sfr_executor.check_conditions(
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
    assert not result
    assert reason is not None


@pytest.mark.unit
def test_sfr_executor_check_conditions_true_branch():
    expiry_options = [
        ExpiryOption(
            expiry="20250830",
            call_contract=MagicMock(),
            put_contract=MagicMock(),
            call_strike=100.0,
            put_strike=95.0,
        )
    ]
    sfr_executor = SFRExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(),
        expiry_options=expiry_options,
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        start_time=0.0,
        quantity=1,
    )
    # All conditions met
    result, reason = sfr_executor.check_conditions(
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
    assert result
    assert reason is None


@pytest.mark.unit
def test_calc_price_and_build_order_no_stock_ticker():
    expiry_options = [
        ExpiryOption(
            expiry="20250830",
            call_contract=MagicMock(),
            put_contract=MagicMock(),
            call_strike=100.0,
            put_strike=95.0,
        )
    ]
    sfr_executor = SFRExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        expiry_options=expiry_options,
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        start_time=0.0,
        quantity=1,
    )
    global contract_ticker
    contract_ticker = {}  # No ticker for stock
    result = sfr_executor.calc_price_and_build_order_for_expiry(expiry_options[0])
    assert result is None


@pytest.mark.unit
def test_calc_price_and_build_order_missing_option_data(monkeypatch):
    expiry_options = [
        ExpiryOption(
            expiry="20250830",
            call_contract=MagicMock(),
            put_contract=MagicMock(),
            call_strike=100.0,
            put_strike=95.0,
        )
    ]
    sfr_executor = SFRExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        expiry_options=expiry_options,
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        start_time=0.0,
        quantity=1,
    )
    global contract_ticker
    contract_ticker = {1: MagicMock(ask=100.0, close=99.0)}
    monkeypatch.setattr(
        sfr_executor,
        "_extract_option_data",
        lambda _: (None, None, None, None, None, None),
    )
    result = sfr_executor.calc_price_and_build_order_for_expiry(expiry_options[0])
    assert result is None


@pytest.mark.unit
def test_calc_price_and_build_order_call_strike_less_than_put_strike(monkeypatch):
    expiry_options = [
        ExpiryOption(
            expiry="20250830",
            call_contract=MagicMock(),
            put_contract=MagicMock(),
            call_strike=100.0,
            put_strike=95.0,
        )
    ]
    sfr_executor = SFRExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        expiry_options=expiry_options,
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        start_time=0.0,
        quantity=1,
    )
    global contract_ticker
    contract_ticker = {1: MagicMock(ask=100.0, close=99.0)}
    # call_strike < put_strike
    monkeypatch.setattr(
        sfr_executor,
        "_extract_option_data",
        lambda _: (MagicMock(), MagicMock(), 90.0, 100.0, 10.0, 5.0),
    )
    result = sfr_executor.calc_price_and_build_order_for_expiry(expiry_options[0])
    assert result is None


@pytest.mark.unit
def test_calc_price_and_build_order_check_conditions_false(monkeypatch):
    expiry_options = [
        ExpiryOption(
            expiry="20250830",
            call_contract=MagicMock(),
            put_contract=MagicMock(),
            call_strike=100.0,
            put_strike=95.0,
        )
    ]
    sfr_executor = SFRExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        expiry_options=expiry_options,
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        start_time=0.0,
        quantity=1,
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
    result = sfr_executor.calc_price_and_build_order_for_expiry(expiry_options[0])
    assert result is None


@pytest.mark.unit
def test_calc_price_and_build_order_check_conditions_true(monkeypatch):
    from modules.Arbitrage.sfr.executor import SFRExecutor

    # Create mock contracts with specific conIds
    call_contract = MagicMock(conId=2)
    put_contract = MagicMock(conId=3)
    from datetime import datetime, timedelta

    future_date = datetime.now() + timedelta(days=30)
    valid_expiry = future_date.strftime("%Y%m%d")

    expiry_options = [
        ExpiryOption(
            expiry=valid_expiry,  # Future date within valid range (30 days out)
            call_contract=call_contract,
            put_contract=put_contract,
            call_strike=100.0,
            put_strike=95.0,
        )
    ]
    sfr_executor = SFRExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        expiry_options=expiry_options,
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        start_time=0.0,
        quantity=1,
    )
    # Create mock tickers with midpoint() method
    # Make this a profitable arbitrage opportunity
    stock_ticker = MagicMock(ask=96.0, close=95.5, volume=1000, last=95.8)
    stock_ticker.midpoint.return_value = (
        95.8  # Stock below put strike for profitable conversion
    )

    call_ticker = MagicMock(bid=5.0, close=5.0, ask=5.5, volume=500, last=5.2)
    call_ticker.midpoint.return_value = 5.25  # Call premium

    put_ticker = MagicMock(ask=1.0, close=1.0, bid=0.8, volume=500, last=0.9)
    put_ticker.midpoint.return_value = 0.9  # Lower put premium for profitable spread

    # Patch contract_ticker in the SFR module with stock and option data using composite keys
    monkeypatch.setattr(
        "modules.Arbitrage.sfr.contract_ticker",
        {
            ("TEST", 1): stock_ticker,  # Stock ticker with composite key
            ("TEST", 2): call_ticker,  # Call ticker with composite key
            ("TEST", 3): put_ticker,  # Put ticker with composite key
        },
    )
    # Mock check_conditions to return (True, None) for successful execution
    monkeypatch.setattr(sfr_executor, "check_conditions", lambda *a, **kw: (True, None))
    with patch.object(
        sfr_executor, "build_order", return_value=("contract", "order")
    ) as mock_build:
        result = sfr_executor.calc_price_and_build_order_for_expiry(expiry_options[0])
        assert result is not None
        assert len(result) == 4  # (contract, order, min_profit, trade_details)
        assert result[0] == "contract"
        assert result[1] == "order"
        mock_build.assert_called()


@pytest.mark.unit
def test_sfr_executor_build_order_quantity():
    expiry_options = [
        ExpiryOption(
            expiry="20250830",
            call_contract=MagicMock(),
            put_contract=MagicMock(),
            call_strike=100.0,
            put_strike=95.0,
        )
    ]
    sfr_executor = SFRExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(),
        expiry_options=expiry_options,
        symbol="TEST",
        profit_target=10.0,
        cost_limit=100.0,
        start_time=0.0,
        quantity=3,
    )
    # Patch contracts to have required attributes
    stock = MagicMock(conId=1)
    call = MagicMock(conId=2, right="C")
    put = MagicMock(conId=3, right="P")
    combo_contract, order = sfr_executor.build_order("TEST", stock, call, put, 99.0, 3)
    assert order.totalQuantity == 3
