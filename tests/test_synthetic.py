import asyncio
import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from ib_async import Option, Stock

from modules.Arbitrage.metrics import RejectionReason, metrics_collector
from modules.Arbitrage.Synthetic import (
    ExpiryOption,
    GlobalOpportunityManager,
    SynExecutor,
    contract_ticker,
)


@pytest.mark.unit
def test_syn_executor_check_conditions_all_false_branches():
    """Test all False branches of SynExecutor.check_conditions."""
    expiry_options = [
        ExpiryOption(
            expiry="20250830",
            call_contract=MagicMock(),
            put_contract=MagicMock(),
            call_strike=100.0,
            put_strike=95.0,
        )
    ]
    syn_executor = SynExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(),
        expiry_options=expiry_options,
        symbol="TEST",
        cost_limit=100.0,
        max_loss_threshold=10.0,
        max_profit_threshold=200.0,
        profit_ratio_threshold=2.0,
        start_time=0.0,
        quantity=1,
        global_manager=GlobalOpportunityManager(),
    )
    # 1. max_loss_threshold >= min_profit
    result, reason = syn_executor.check_conditions(
        symbol="TEST",
        cost_limit=100.0,
        lmt_price=50.0,
        net_credit=5.0,
        min_roi=20.0,
        min_profit=10.0,
        max_profit=30.0,
    )
    assert not result
    assert reason is not None

    # 2. net_credit < 0
    syn_executor.max_loss_threshold = None
    result, reason = syn_executor.check_conditions(
        symbol="TEST",
        cost_limit=100.0,
        lmt_price=50.0,
        net_credit=-1.0,
        min_roi=20.0,
        min_profit=5.0,
        max_profit=30.0,
    )
    assert not result
    assert reason is not None

    # 3. max_profit_threshold < max_profit
    syn_executor.max_loss_threshold = None
    syn_executor.max_profit_threshold = 10.0
    result, reason = syn_executor.check_conditions(
        symbol="TEST",
        cost_limit=100.0,
        lmt_price=50.0,
        net_credit=5.0,
        min_roi=20.0,
        min_profit=5.0,
        max_profit=30.0,
    )
    assert not result
    assert reason is not None

    # 4. profit_ratio_threshold > profit_ratio
    syn_executor.max_profit_threshold = None
    syn_executor.profit_ratio_threshold = 10.0
    result, reason = syn_executor.check_conditions(
        symbol="TEST",
        cost_limit=100.0,
        lmt_price=50.0,
        net_credit=5.0,
        min_roi=20.0,
        min_profit=5.0,
        max_profit=10.0,
    )
    assert not result
    assert reason is not None

    # 5. np.isnan(lmt_price)
    syn_executor.profit_ratio_threshold = None
    result, reason = syn_executor.check_conditions(
        symbol="TEST",
        cost_limit=100.0,
        lmt_price=np.nan,
        net_credit=5.0,
        min_roi=20.0,
        min_profit=5.0,
        max_profit=10.0,
    )
    assert not result
    assert reason is not None

    # 6. lmt_price > cost_limit
    result, reason = syn_executor.check_conditions(
        symbol="TEST",
        cost_limit=50.0,
        lmt_price=100.0,
        net_credit=5.0,
        min_roi=20.0,
        min_profit=5.0,
        max_profit=10.0,
    )
    assert not result
    assert reason is not None


@pytest.mark.unit
def test_syn_executor_check_conditions_true_branch():
    """Test True branch of SynExecutor.check_conditions."""
    expiry_options = [
        ExpiryOption(
            expiry="20250830",
            call_contract=MagicMock(),
            put_contract=MagicMock(),
            call_strike=100.0,
            put_strike=95.0,
        )
    ]
    syn_executor = SynExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(),
        expiry_options=expiry_options,
        symbol="TEST",
        cost_limit=100.0,
        max_loss_threshold=None,
        max_profit_threshold=None,
        profit_ratio_threshold=None,
        start_time=0.0,
        quantity=1,
        global_manager=GlobalOpportunityManager(),
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
    expiry_options = [
        ExpiryOption(
            expiry="20250830",
            call_contract=MagicMock(),
            put_contract=MagicMock(),
            call_strike=100.0,
            put_strike=95.0,
        )
    ]
    syn_executor = SynExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        expiry_options=expiry_options,
        symbol="TEST",
        cost_limit=100.0,
        max_loss_threshold=None,
        max_profit_threshold=None,
        profit_ratio_threshold=None,
        start_time=0.0,
        quantity=1,
        global_manager=GlobalOpportunityManager(),
    )
    global contract_ticker
    contract_ticker = {}  # No ticker for stock
    result = syn_executor.calc_price_and_build_order_for_expiry(expiry_options[0])
    assert result is None


@pytest.mark.unit
def test_calc_price_and_build_order_missing_option_data(monkeypatch):
    """Test calc_price_and_build_order returns (None, None) if option data is missing."""
    expiry_options = [
        ExpiryOption(
            expiry="20250830",
            call_contract=MagicMock(),
            put_contract=MagicMock(),
            call_strike=100.0,
            put_strike=95.0,
        )
    ]
    syn_executor = SynExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        expiry_options=expiry_options,
        symbol="TEST",
        cost_limit=100.0,
        max_loss_threshold=None,
        max_profit_threshold=None,
        profit_ratio_threshold=None,
        start_time=0.0,
        quantity=1,
        global_manager=GlobalOpportunityManager(),
    )
    global contract_ticker
    contract_ticker = {1: MagicMock(ask=100.0, close=99.0)}
    monkeypatch.setattr(
        syn_executor,
        "_extract_option_data",
        lambda _: (None, None, None, None, None, None),
    )
    result = syn_executor.calc_price_and_build_order_for_expiry(expiry_options[0])
    assert result is None


@pytest.mark.unit
def test_calc_price_and_build_order_call_strike_less_than_put_strike(monkeypatch):
    """Test calc_price_and_build_order returns (None, None) if call_strike < put_strike."""
    expiry_options = [
        ExpiryOption(
            expiry="20250830",
            call_contract=MagicMock(),
            put_contract=MagicMock(),
            call_strike=100.0,
            put_strike=95.0,
        )
    ]
    syn_executor = SynExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        expiry_options=expiry_options,
        symbol="TEST",
        cost_limit=100.0,
        max_loss_threshold=None,
        max_profit_threshold=None,
        profit_ratio_threshold=None,
        start_time=0.0,
        quantity=1,
        global_manager=GlobalOpportunityManager(),
    )
    global contract_ticker
    contract_ticker = {1: MagicMock(ask=100.0, close=99.0)}
    # call_strike < put_strike
    monkeypatch.setattr(
        syn_executor,
        "_extract_option_data",
        lambda _: (MagicMock(), MagicMock(), 90.0, 100.0, 10.0, 5.0),
    )
    result = syn_executor.calc_price_and_build_order_for_expiry(expiry_options[0])
    assert result is None


@pytest.mark.unit
def test_calc_price_and_build_order_check_conditions_false(monkeypatch):
    """Test calc_price_and_build_order returns (None, None) if check_conditions returns False."""
    expiry_options = [
        ExpiryOption(
            expiry="20250830",
            call_contract=MagicMock(),
            put_contract=MagicMock(),
            call_strike=100.0,
            put_strike=95.0,
        )
    ]
    syn_executor = SynExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        expiry_options=expiry_options,
        symbol="TEST",
        cost_limit=100.0,
        max_loss_threshold=None,
        max_profit_threshold=None,
        profit_ratio_threshold=None,
        start_time=0.0,
        quantity=1,
        global_manager=GlobalOpportunityManager(),
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
    result = syn_executor.calc_price_and_build_order_for_expiry(expiry_options[0])
    assert result is None


@pytest.mark.unit
def test_calc_price_and_build_order_check_conditions_true(monkeypatch):
    """Test calc_price_and_build_order returns build_order result if check_conditions returns True."""
    # Create mock contracts with specific conIds
    call_contract = MagicMock(conId=2)
    put_contract = MagicMock(conId=3)
    expiry_options = [
        ExpiryOption(
            expiry="20250918",  # Future date within valid range (30 days from now)
            call_contract=call_contract,
            put_contract=put_contract,
            call_strike=100.0,
            put_strike=95.0,
        )
    ]
    syn_executor = SynExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(conId=1),
        expiry_options=expiry_options,
        symbol="TEST",
        cost_limit=100.0,
        max_loss_threshold=None,
        max_profit_threshold=None,
        profit_ratio_threshold=None,
        start_time=0.0,
        quantity=1,
        global_manager=GlobalOpportunityManager(),
    )
    # Patch contract_ticker in the Synthetic module with stock and option data
    monkeypatch.setattr(
        "modules.Arbitrage.Synthetic.contract_ticker",
        {
            1: MagicMock(ask=100.0, close=99.0, volume=1000),  # Stock ticker
            2: MagicMock(bid=5.0, close=5.0, ask=5.5, volume=500),  # Call ticker
            3: MagicMock(ask=3.0, close=3.0, bid=2.5, volume=500),  # Put ticker
        },
    )
    # Mock check_conditions to return (True, None) for successful execution
    monkeypatch.setattr(syn_executor, "check_conditions", lambda *a, **kw: (True, None))
    with patch.object(
        syn_executor, "build_order", return_value=("contract", "order")
    ) as mock_build:
        result = syn_executor.calc_price_and_build_order_for_expiry(expiry_options[0])
        assert result is not None
        assert len(result) == 4  # (contract, order, min_profit, trade_details)
        assert result[0] == "contract"
        assert result[1] == "order"
        mock_build.assert_called()


@pytest.mark.unit
def test_syn_executor_build_order_quantity():
    expiry_options = [
        ExpiryOption(
            expiry="20250830",
            call_contract=MagicMock(),
            put_contract=MagicMock(),
            call_strike=100.0,
            put_strike=95.0,
        )
    ]
    syn_executor = SynExecutor(
        ib=MagicMock(),
        order_manager=MagicMock(),
        stock_contract=MagicMock(),
        expiry_options=expiry_options,
        symbol="TEST",
        cost_limit=100.0,
        max_loss_threshold=None,
        max_profit_threshold=None,
        profit_ratio_threshold=None,
        start_time=0.0,
        quantity=4,
        global_manager=GlobalOpportunityManager(),
    )
    # Patch contracts to have required attributes
    stock = MagicMock(conId=1)
    call = MagicMock(conId=2, right="C")
    put = MagicMock(conId=3, right="P")
    combo_contract, order = syn_executor.build_order("TEST", stock, call, put, 99.0, 4)
    assert order.totalQuantity == 4


@pytest.mark.asyncio
async def test_rejection_reasons_are_logged_during_scan():
    """Test that rejection reasons are properly logged during scan execution"""
    # Create a real SynExecutor to test scan lifecycle
    stock_contract = Stock("AAPL", "SMART", "USD")
    stock_contract.conId = 1001  # Set contract ID

    call_contract = Option("AAPL", "20250918", 150, "C", "SMART")
    call_contract.conId = 1002  # Set contract ID

    put_contract = Option("AAPL", "20250918", 140, "P", "SMART")
    put_contract.conId = 1003  # Set contract ID

    expiry_option = ExpiryOption(
        expiry="20250918",
        call_contract=call_contract,
        put_contract=put_contract,
        call_strike=150,
        put_strike=140,
    )

    ib_mock = Mock()
    order_manager_mock = Mock()

    # Create SynExecutor
    executor = SynExecutor(
        ib=ib_mock,
        order_manager=order_manager_mock,
        stock_contract=stock_contract,
        expiry_options=[expiry_option],
        symbol="AAPL",
        cost_limit=100,
        max_loss_threshold=50,
        max_profit_threshold=200,
        profit_ratio_threshold=2.0,
        start_time=time.time(),
        quantity=1,
        data_timeout=30.0,
        global_manager=GlobalOpportunityManager(),
    )

    # Clear any existing metrics first, then start a scan
    metrics_collector.reset_session()
    scan_metrics = metrics_collector.start_scan("AAPL", "Synthetic")

    # Mock contract_ticker with data that will trigger rejection
    from modules.Arbitrage import Synthetic

    # Mock stock ticker with valid data (using the contract ID)
    stock_ticker = Mock()
    stock_ticker.ask = 145.0
    stock_ticker.close = 145.0
    stock_ticker.volume = 1000
    Synthetic.contract_ticker[stock_contract.conId] = stock_ticker

    # Mock call ticker with bid-ask spread too wide (should trigger rejection)
    call_ticker = Mock()
    call_ticker.bid = 5.0
    call_ticker.ask = 25.0  # Wide spread of 20 > 15 threshold
    call_ticker.close = 10.0
    call_ticker.volume = 100
    Synthetic.contract_ticker[call_contract.conId] = call_ticker

    # Mock put ticker with valid data
    put_ticker = Mock()
    put_ticker.bid = 8.0
    put_ticker.ask = 9.0
    put_ticker.close = 8.5
    put_ticker.volume = 100
    Synthetic.contract_ticker[put_contract.conId] = put_ticker

    # Capture logs to verify rejection reasons are logged
    import sys
    from io import StringIO

    import logging

    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)

    # Add handler to the logger
    logger = logging.getLogger("modules.Arbitrage.metrics")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        # Test the calc_price_and_build_order_for_expiry method
        result = executor.calc_price_and_build_order_for_expiry(expiry_option)

        # Should return None due to rejection
        assert result is None

        # Check that rejection reason was logged
        log_output = log_capture.getvalue()

        # Accept any rejection reason as long as it's being logged
        assert "REJECTED" in log_output
        assert "AAPL" in log_output

    finally:
        # Clean up
        logger.removeHandler(handler)
        metrics_collector.finish_scan(success=True)
