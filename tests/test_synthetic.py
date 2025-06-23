from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modules.Arbitrage.Synthetic import SynExecutor


@pytest.mark.unit
def test_syn_executor_check_conditions_all_false_branches():
    """Test all False branches of SynExecutor.check_conditions."""
    syn_executor = SynExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(),
        option_contracts=[MagicMock(), MagicMock()],
        symbol="TEST",
        cost_limit=100.0,
        max_loss_threshold=10.0,
        max_profit_threshold=200.0,
        profit_ratio_threshold=2.0,
        expiry="20240101",
    )
    # 1. max_loss_threshold >= min_profit
    assert not syn_executor.check_conditions(
        symbol="TEST",
        cost_limit=100.0,
        lmt_price=50.0,
        net_credit=5.0,
        min_roi=20.0,
        min_profit=10.0,
        max_profit=30.0,
    )
    # 2. net_credit < 0
    syn_executor.max_loss_threshold = None
    assert not syn_executor.check_conditions(
        symbol="TEST",
        cost_limit=100.0,
        lmt_price=50.0,
        net_credit=-1.0,
        min_roi=20.0,
        min_profit=5.0,
        max_profit=30.0,
    )
    # 3. max_profit_threshold < max_profit
    syn_executor.max_loss_threshold = None
    syn_executor.max_profit_threshold = 10.0
    assert not syn_executor.check_conditions(
        symbol="TEST",
        cost_limit=100.0,
        lmt_price=50.0,
        net_credit=5.0,
        min_roi=20.0,
        min_profit=5.0,
        max_profit=30.0,
    )
    # 4. profit_ratio_threshold > profit_ratio
    syn_executor.max_profit_threshold = None
    syn_executor.profit_ratio_threshold = 10.0
    assert not syn_executor.check_conditions(
        symbol="TEST",
        cost_limit=100.0,
        lmt_price=50.0,
        net_credit=5.0,
        min_roi=20.0,
        min_profit=5.0,
        max_profit=10.0,
    )
    # 5. np.isnan(lmt_price)
    syn_executor.profit_ratio_threshold = None
    assert not syn_executor.check_conditions(
        symbol="TEST",
        cost_limit=100.0,
        lmt_price=np.nan,
        net_credit=5.0,
        min_roi=20.0,
        min_profit=5.0,
        max_profit=10.0,
    )
    # 6. lmt_price > cost_limit
    assert not syn_executor.check_conditions(
        symbol="TEST",
        cost_limit=50.0,
        lmt_price=100.0,
        net_credit=5.0,
        min_roi=20.0,
        min_profit=5.0,
        max_profit=10.0,
    )


@pytest.mark.unit
def test_syn_executor_check_conditions_true_branch():
    """Test True branch of SynExecutor.check_conditions."""
    syn_executor = SynExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(),
        option_contracts=[MagicMock(), MagicMock()],
        symbol="TEST",
        cost_limit=100.0,
        max_loss_threshold=None,
        max_profit_threshold=None,
        profit_ratio_threshold=None,
        expiry="20240101",
    )
    assert syn_executor.check_conditions(
        symbol="TEST",
        cost_limit=100.0,
        lmt_price=50.0,
        net_credit=10.0,
        min_roi=20.0,
        min_profit=5.0,
        max_profit=20.0,
    )


@pytest.mark.unit
def test_calc_price_and_build_order_no_stock_ticker():
    """Test calc_price_and_build_order returns (None, None) if no ticker for stock contract."""
    syn_executor = SynExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        option_contracts=[MagicMock(), MagicMock()],
        symbol="TEST",
        cost_limit=100.0,
        max_loss_threshold=None,
        max_profit_threshold=None,
        profit_ratio_threshold=None,
        expiry="20240101",
    )
    global contract_ticker
    contract_ticker = {}  # No ticker for stock
    result = syn_executor.calc_price_and_build_order()
    assert result == (None, None)


@pytest.mark.unit
def test_calc_price_and_build_order_missing_option_data(monkeypatch):
    """Test calc_price_and_build_order returns (None, None) if option data is missing."""
    syn_executor = SynExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        option_contracts=[MagicMock(), MagicMock()],
        symbol="TEST",
        cost_limit=100.0,
        max_loss_threshold=None,
        max_profit_threshold=None,
        profit_ratio_threshold=None,
        expiry="20240101",
    )
    global contract_ticker
    contract_ticker = {1: MagicMock(ask=100.0, close=99.0)}
    monkeypatch.setattr(
        syn_executor,
        "_extract_option_data",
        lambda _: (None, None, None, None, None, None),
    )
    result = syn_executor.calc_price_and_build_order()
    assert result == (None, None)


@pytest.mark.unit
def test_calc_price_and_build_order_call_strike_less_than_put_strike(monkeypatch):
    """Test calc_price_and_build_order returns (None, None) if call_strike < put_strike."""
    syn_executor = SynExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        option_contracts=[MagicMock(), MagicMock()],
        symbol="TEST",
        cost_limit=100.0,
        max_loss_threshold=None,
        max_profit_threshold=None,
        profit_ratio_threshold=None,
        expiry="20240101",
    )
    global contract_ticker
    contract_ticker = {1: MagicMock(ask=100.0, close=99.0)}
    # call_strike < put_strike
    monkeypatch.setattr(
        syn_executor,
        "_extract_option_data",
        lambda _: (MagicMock(), MagicMock(), 90.0, 100.0, 10.0, 5.0),
    )
    result = syn_executor.calc_price_and_build_order()
    assert result == (None, None)


@pytest.mark.unit
def test_calc_price_and_build_order_check_conditions_false(monkeypatch):
    """Test calc_price_and_build_order returns (None, None) if check_conditions returns False."""
    syn_executor = SynExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        option_contracts=[MagicMock(), MagicMock()],
        symbol="TEST",
        cost_limit=100.0,
        max_loss_threshold=None,
        max_profit_threshold=None,
        profit_ratio_threshold=None,
        expiry="20240101",
    )
    global contract_ticker
    contract_ticker = {1: MagicMock(ask=100.0, close=99.0)}
    # call_strike > put_strike, but check_conditions returns False
    monkeypatch.setattr(
        syn_executor,
        "_extract_option_data",
        lambda _: (MagicMock(), MagicMock(), 110.0, 100.0, 10.0, 5.0),
    )
    monkeypatch.setattr(syn_executor, "check_conditions", lambda *a, **kw: False)
    result = syn_executor.calc_price_and_build_order()
    assert result == (None, None)


@pytest.mark.unit
def test_calc_price_and_build_order_check_conditions_true(monkeypatch):
    """Test calc_price_and_build_order returns build_order result if check_conditions returns True."""
    syn_executor = SynExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        option_contracts=[MagicMock(), MagicMock()],
        symbol="TEST",
        cost_limit=100.0,
        max_loss_threshold=None,
        max_profit_threshold=None,
        profit_ratio_threshold=None,
        expiry="20240101",
    )
    # Patch contract_ticker in the Synthetic module, not just locally
    monkeypatch.setattr(
        "modules.Arbitrage.Synthetic.contract_ticker",
        {1: MagicMock(ask=100.0, close=99.0)},
    )
    call_contract = object()
    put_contract = object()
    monkeypatch.setattr(
        syn_executor,
        "_extract_option_data",
        lambda _: (call_contract, put_contract, 110.0, 100.0, 10.0, 5.0),
    )
    monkeypatch.setattr(syn_executor, "check_conditions", lambda *a, **kw: True)
    with (
        patch.object(syn_executor, "_log_trade_details") as mock_log,
        patch.object(
            syn_executor, "build_order", return_value=("contract", "order")
        ) as mock_build,
    ):
        result = syn_executor.calc_price_and_build_order()
        assert result == ("contract", "order")
        mock_log.assert_called()
        mock_build.assert_called()
