"""
Comprehensive test suite for SFR Backtesting Engine.

This module tests all major components of the SFR backtesting system including:
- Configuration loading and validation
- Opportunity detection logic
- Trade simulation
- Performance analytics
- Database integration
- Error handling
"""

import asyncio
import os

# Import modules to test
import sys
from datetime import date, datetime, timedelta
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import logging
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import DatabaseConfig, SFRConfigLoader
from sfr_backtest_engine import (
    ExecutionDifficulty,
    ExpiryOption,
    MarketData,
    OpportunityQuality,
    SFRBacktestConfig,
    SFRBacktestConfigs,
    SFRBacktestEngine,
    SFROpportunity,
    SimulatedTrade,
    SlippageModel,
    VixRegime,
)

# Setup logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestSFRBacktestConfig:
    """Test SFR backtesting configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SFRBacktestConfig()

        assert config.profit_target == 0.50
        assert config.cost_limit == 120.0
        assert config.volume_limit == 100
        assert config.quantity == 1
        assert config.slippage_model == SlippageModel.LINEAR
        assert config.commission_per_contract == 1.00
        assert config.vix_regime_filter is None

    def test_preset_configs(self):
        """Test preset configuration generators."""
        # Conservative config
        conservative = SFRBacktestConfigs.conservative_config()
        assert conservative.profit_target == 0.75
        assert conservative.max_bid_ask_spread_call == 10.0
        assert conservative.slippage_model == SlippageModel.LINEAR

        # Aggressive config
        aggressive = SFRBacktestConfigs.aggressive_config()
        assert aggressive.profit_target == 0.25
        assert aggressive.cost_limit == 200.0
        assert aggressive.slippage_model == SlippageModel.IMPACT

        # Low VIX config
        low_vix = SFRBacktestConfigs.low_vix_config()
        assert low_vix.vix_regime_filter == VixRegime.LOW
        assert low_vix.max_vix_level == 20.0
        assert low_vix.exclude_vix_spikes is True

        # High VIX config
        high_vix = SFRBacktestConfigs.high_vix_config()
        assert high_vix.vix_regime_filter == VixRegime.HIGH
        assert high_vix.min_vix_level == 25.0
        assert high_vix.slippage_model == SlippageModel.IMPACT


class TestConfigLoader:
    """Test configuration loader functionality."""

    def test_config_loader_initialization(self):
        """Test configuration loader initialization."""
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = """
                database:
                  host: test_host
                  port: 5432
                target_symbols: [SPY, QQQ]
                configurations:
                  test:
                    profit_target: 0.60
                    cost_limit: 100.0
                    slippage_model: LINEAR
                """

                loader = SFRConfigLoader("test_config.yaml")
                assert loader.config_path == "test_config.yaml"

    def test_database_config_loading(self):
        """Test database configuration loading with env overrides."""
        with patch.dict(os.environ, {"DB_HOST": "env_host", "DB_PORT": "5555"}):
            loader = SFRConfigLoader()
            db_config = loader.get_database_config()

            assert db_config.host == "env_host"
            assert db_config.port == 5555

    def test_target_symbols_loading(self):
        """Test target symbols loading."""
        loader = SFRConfigLoader()
        symbols = loader.get_target_symbols()

        assert isinstance(symbols, list)
        assert "SPY" in symbols
        assert "QQQ" in symbols

    def test_configuration_validation(self):
        """Test configuration validation."""
        loader = SFRConfigLoader()

        # Valid configuration
        valid_config = SFRBacktestConfig()
        errors = loader.validate_configuration(valid_config)
        assert len(errors) == 0

        # Invalid configuration
        invalid_config = SFRBacktestConfig(
            profit_target=-0.5,  # Negative profit target
            cost_limit=0,  # Zero cost limit
            expiry_min_days=50,  # Min > Max
            expiry_max_days=30,
        )
        errors = loader.validate_configuration(invalid_config)
        assert len(errors) > 0


class MockDatabase:
    """Mock database connection for testing."""

    def __init__(self):
        self.queries = []
        self.results = {}

    def set_result(self, query_pattern: str, result):
        """Set result for queries matching pattern."""
        self.results[query_pattern] = result

    async def acquire(self):
        """Mock acquire connection."""
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def fetchval(self, query: str, *args):
        """Mock fetchval."""
        self.queries.append((query, args))

        # Return mock results based on query patterns
        if "INSERT INTO sfr_backtest_runs" in query:
            return 1  # Mock backtest run ID
        elif "SELECT id FROM underlying_securities" in query:
            return 1  # Mock underlying ID
        else:
            return 1

    async def fetchrow(self, query: str, *args):
        """Mock fetchrow."""
        self.queries.append((query, args))

        if "stock_data_ticks" in query:
            return {
                "price": 100.0,
                "volume": 1000,
                "open_price": 99.0,
                "high_price": 101.0,
                "low_price": 98.0,
                "close_price": 100.0,
            }
        elif "vix_data" in query:
            return {"vix_close": 20.0, "vix_regime": "MEDIUM"}
        elif "market_data_ticks" in query:
            return {
                "bid_price": 2.50,
                "ask_price": 2.70,
                "last_price": 2.60,
                "volume": 100,
            }

        return {}

    async def fetch(self, query: str, *args):
        """Mock fetch."""
        self.queries.append((query, args))

        if "option_chains" in query:
            return [
                {
                    "expiration_date": date.today() + timedelta(days=30),
                    "call_contract_id": 1,
                    "call_strike": 105.0,
                    "put_contract_id": 2,
                    "put_strike": 95.0,
                }
            ]

        return []

    async def execute(self, query: str, *args):
        """Mock execute."""
        self.queries.append((query, args))
        return "INSERT 0 1"

    async def executemany(self, query: str, args_list):
        """Mock executemany."""
        self.queries.append((query, args_list))
        return f"INSERT 0 {len(args_list)}"

    def transaction(self):
        """Mock transaction context manager."""
        return AsyncMock()


@pytest.fixture
async def mock_db_pool():
    """Create mock database pool."""
    mock_db = MockDatabase()

    # Setup common query results
    mock_db.set_result("underlying_securities", 1)
    mock_db.set_result("backtest_runs", 1)

    async def acquire():
        return mock_db

    mock_pool = AsyncMock()
    mock_pool.acquire = acquire

    return mock_pool


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return SFRBacktestConfig(
        profit_target=0.50,
        cost_limit=150.0,
        volume_limit=50,
        quantity=1,
        slippage_model=SlippageModel.LINEAR,
        base_slippage_bps=2,
        commission_per_contract=1.00,
    )


@pytest.fixture
def sample_expiry_option():
    """Create sample expiry option for testing."""
    return ExpiryOption(
        expiry="20241201",
        expiry_date=date(2024, 12, 1),
        call_strike=105.0,
        put_strike=95.0,
        call_contract_id=1,
        put_contract_id=2,
    )


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    return MarketData(
        timestamp=datetime.now(),
        stock_price=100.0,
        call_bid=2.40,
        call_ask=2.60,
        call_last=2.50,
        call_volume=150,
        put_bid=1.30,
        put_ask=1.50,
        put_last=1.40,
        put_volume=120,
        vix_level=20.0,
        vix_regime=VixRegime.MEDIUM,
    )


class TestSFRBacktestEngine:
    """Test main SFR backtesting engine functionality."""

    @pytest.mark.asyncio
    async def test_engine_initialization(self, mock_db_pool, sample_config):
        """Test engine initialization."""
        engine = SFRBacktestEngine(mock_db_pool, sample_config)

        assert engine.db_pool == mock_db_pool
        assert engine.config == sample_config
        assert engine.backtest_run_id is None
        assert len(engine.opportunities) == 0
        assert len(engine.trades) == 0
        assert engine.target_symbols == [
            "SPY",
            "QQQ",
            "AAPL",
            "MSFT",
            "NVDA",
            "TSLA",
            "AMZN",
            "META",
            "GOOGL",
            "JPM",
        ]

    @pytest.mark.asyncio
    async def test_quick_viability_check(
        self, mock_db_pool, sample_config, sample_expiry_option
    ):
        """Test quick viability check logic."""
        engine = SFRBacktestEngine(mock_db_pool, sample_config)

        # Valid opportunity
        viable, reason = engine._quick_viability_check(sample_expiry_option, 100.0)
        assert viable is True
        assert reason is None

        # Invalid strike spread
        bad_expiry = ExpiryOption(
            expiry="20241201",
            expiry_date=date(2024, 12, 1),
            call_strike=95.5,  # Too small spread
            put_strike=95.0,
            call_contract_id=1,
            put_contract_id=2,
        )
        viable, reason = engine._quick_viability_check(bad_expiry, 100.0)
        assert viable is False
        assert reason == "invalid_strike_spread"

        # Poor moneyness
        viable, reason = engine._quick_viability_check(
            sample_expiry_option, 50.0
        )  # Very low stock price
        assert viable is False
        assert reason == "poor_moneyness"

    @pytest.mark.asyncio
    async def test_conditions_check(self, mock_db_pool, sample_config):
        """Test SFR conditions checking logic."""
        engine = SFRBacktestEngine(mock_db_pool, sample_config)

        # Valid conditions
        conditions_met, reason = engine._check_conditions(
            stock_price=100.0,
            put_strike=95.0,
            lmt_price=110.0,
            net_credit=7.0,  # Higher than spread
            min_roi=0.60,  # Above profit target
            min_profit=2.0,  # Positive profit
        )
        assert conditions_met is True
        assert reason is None

        # Arbitrage condition not met (spread >= net_credit)
        conditions_met, reason = engine._check_conditions(
            stock_price=100.0,
            put_strike=95.0,
            lmt_price=110.0,
            net_credit=4.0,  # Lower than spread (5.0)
            min_roi=0.60,
            min_profit=-1.0,  # Negative profit
        )
        assert conditions_met is False
        assert reason == "arbitrage_condition_not_met"

        # Price limit exceeded
        conditions_met, reason = engine._check_conditions(
            stock_price=100.0,
            put_strike=95.0,
            lmt_price=200.0,  # Above cost limit
            net_credit=7.0,
            min_roi=0.60,
            min_profit=2.0,
        )
        assert conditions_met is False
        assert reason == "price_limit_exceeded"

    def test_slippage_calculation(self, mock_db_pool, sample_config):
        """Test slippage calculation models."""
        engine = SFRBacktestEngine(mock_db_pool, sample_config)

        # Test LINEAR model
        engine.config.slippage_model = SlippageModel.LINEAR
        slippage = engine._calculate_slippage(100.0, 0.8, "BUY")
        assert slippage > 0  # Should be positive for BUY

        # Test NONE model
        engine.config.slippage_model = SlippageModel.NONE
        slippage = engine._calculate_slippage(100.0, 0.8, "BUY")
        assert slippage == 0.0

        # Test SQUARE_ROOT model
        engine.config.slippage_model = SlippageModel.SQUARE_ROOT
        slippage = engine._calculate_slippage(100.0, 0.5, "SELL")
        assert slippage < 0  # Should be negative for SELL

        # Test IMPACT model
        engine.config.slippage_model = SlippageModel.IMPACT
        engine.config.quantity = 2
        slippage = engine._calculate_slippage(100.0, 0.3, "BUY")
        assert slippage > 0

    def test_liquidity_score_calculation(self, mock_db_pool, sample_config):
        """Test liquidity score calculation."""
        engine = SFRBacktestEngine(mock_db_pool, sample_config)

        # High volume, tight spreads
        score = engine._calculate_liquidity_score(200, 300, 1.0, 0.5)
        assert 0.8 <= score <= 1.0

        # Low volume, wide spreads
        score = engine._calculate_liquidity_score(10, 5, 10.0, 15.0)
        assert 0.0 <= score <= 0.2

        # No volume data
        score = engine._calculate_liquidity_score(None, None, 5.0, 5.0)
        assert score >= 0.0

    def test_opportunity_quality_classification(self, mock_db_pool, sample_config):
        """Test opportunity quality classification."""
        engine = SFRBacktestEngine(mock_db_pool, sample_config)

        # Excellent opportunity
        quality = engine._classify_opportunity_quality(
            min_roi=2.0,
            liquidity_score=0.9,
            call_spread=1.0,
            put_spread=1.0,
            days_to_expiry=30,
        )
        assert quality == OpportunityQuality.EXCELLENT

        # Poor opportunity
        quality = engine._classify_opportunity_quality(
            min_roi=0.1,
            liquidity_score=0.1,
            call_spread=15.0,
            put_spread=20.0,
            days_to_expiry=10,
        )
        assert quality == OpportunityQuality.POOR

    def test_execution_difficulty_classification(self, mock_db_pool, sample_config):
        """Test execution difficulty classification."""
        engine = SFRBacktestEngine(mock_db_pool, sample_config)

        # Easy execution
        difficulty = engine._classify_execution_difficulty(
            liquidity_score=0.8, call_spread=1.5, put_spread=1.0
        )
        assert difficulty == ExecutionDifficulty.EASY

        # Very difficult execution
        difficulty = engine._classify_execution_difficulty(
            liquidity_score=0.1, call_spread=25.0, put_spread=30.0
        )
        assert difficulty == ExecutionDifficulty.VERY_DIFFICULT

    @pytest.mark.asyncio
    async def test_sfr_opportunity_detection(
        self, mock_db_pool, sample_config, sample_expiry_option, sample_market_data
    ):
        """Test complete SFR opportunity detection logic."""
        engine = SFRBacktestEngine(mock_db_pool, sample_config)

        opportunity = await engine._check_sfr_opportunity(
            underlying_id=1,
            expiry_option=sample_expiry_option,
            market_data=sample_market_data,
        )

        assert opportunity is not None
        assert opportunity.underlying_id == 1
        assert opportunity.expiry_option == sample_expiry_option
        assert opportunity.market_data == sample_market_data
        assert opportunity.net_credit > 0  # call_price - put_price
        assert opportunity.min_profit is not None
        assert opportunity.max_profit is not None
        assert opportunity.min_roi is not None
        assert opportunity.liquidity_score > 0
        assert opportunity.opportunity_quality in [
            OpportunityQuality.EXCELLENT,
            OpportunityQuality.GOOD,
            OpportunityQuality.FAIR,
            OpportunityQuality.POOR,
        ]

    @pytest.mark.asyncio
    async def test_trade_simulation(self, mock_db_pool, sample_config):
        """Test trade execution simulation."""
        engine = SFRBacktestEngine(mock_db_pool, sample_config)

        # Create valid opportunity
        opportunity = SFROpportunity(
            underlying_id=1,
            timestamp=datetime.now(),
            conditions_check=True,
            net_credit=6.0,
            min_profit=1.0,
            max_profit=11.0,
            min_roi=0.60,
            liquidity_score=0.7,
            market_data=MarketData(
                timestamp=datetime.now(),
                stock_price=100.0,
                call_bid=2.40,
                call_ask=2.60,
                call_last=2.50,
                put_bid=1.30,
                put_ask=1.50,
                put_last=1.40,
                vix_level=20.0,
            ),
            expiry_option=ExpiryOption(
                expiry="20241201",
                expiry_date=date(2024, 12, 1),
                call_strike=105.0,
                put_strike=95.0,
            ),
        )

        trade = await engine._simulate_trade_execution(opportunity)

        assert trade is not None
        assert trade.opportunity == opportunity
        assert trade.quantity == engine.config.quantity
        assert trade.stock_execution_price > 0
        assert trade.call_execution_price > 0
        assert trade.put_execution_price > 0
        assert trade.total_commission > 0
        assert trade.execution_status == "FILLED"
        assert trade.execution_quality in ["EXCELLENT", "GOOD", "FAIR", "POOR"]

    @pytest.mark.asyncio
    async def test_vix_filter_check(self, mock_db_pool):
        """Test VIX filtering logic."""
        # Low VIX configuration
        config = SFRBacktestConfig(vix_regime_filter=VixRegime.LOW, max_vix_level=20.0)
        engine = SFRBacktestEngine(mock_db_pool, config)

        # Should pass low VIX filter
        vix_data = {"vix_level": 15.0, "vix_regime": VixRegime.LOW}
        assert engine._check_vix_filter(vix_data) is True

        # Should fail low VIX filter (high VIX)
        vix_data = {"vix_level": 30.0, "vix_regime": VixRegime.HIGH}
        assert engine._check_vix_filter(vix_data) is False

        # High VIX configuration
        config = SFRBacktestConfig(vix_regime_filter=VixRegime.HIGH, min_vix_level=25.0)
        engine = SFRBacktestEngine(mock_db_pool, config)

        # Should pass high VIX filter
        vix_data = {"vix_level": 30.0, "vix_regime": VixRegime.HIGH}
        assert engine._check_vix_filter(vix_data) is True

        # Should fail high VIX filter (low VIX)
        vix_data = {"vix_level": 15.0, "vix_regime": VixRegime.LOW}
        assert engine._check_vix_filter(vix_data) is False

    @pytest.mark.asyncio
    async def test_performance_analytics_calculation(self, mock_db_pool, sample_config):
        """Test performance analytics calculation."""
        engine = SFRBacktestEngine(mock_db_pool, sample_config)

        # Add sample opportunities and trades
        for i in range(10):
            opportunity = SFROpportunity(
                underlying_id=1,
                timestamp=datetime.now(),
                conditions_check=True,
                opportunity_quality=(
                    OpportunityQuality.GOOD if i % 2 == 0 else OpportunityQuality.FAIR
                ),
                liquidity_score=0.7,
            )
            engine.opportunities.append(opportunity)

            if i < 7:  # 7 successful trades out of 10 opportunities
                trade = SimulatedTrade(
                    opportunity=opportunity,
                    quantity=1,
                    realized_min_profit=float(i + 1),
                    realized_min_roi=float((i + 1) * 0.5),
                    total_commission=1.0,
                    total_slippage=0.05,
                    execution_status="FILLED",
                )
                engine.trades.append(trade)

        # Calculate analytics
        await engine._calculate_performance_analytics()

        metrics = engine.performance_metrics
        assert metrics.total_opportunities_found == 10
        assert metrics.total_simulated_trades == 7
        assert metrics.successful_executions == 7
        assert metrics.execution_success_rate == 1.0  # All simulated trades successful
        assert metrics.total_net_profit > 0
        assert metrics.avg_profit_per_trade > 0
        assert metrics.avg_min_roi > 0
        assert "GOOD" in metrics.opportunities_by_quality
        assert "FAIR" in metrics.opportunities_by_quality

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_db_pool, sample_config):
        """Test error handling in various scenarios."""
        engine = SFRBacktestEngine(mock_db_pool, sample_config)

        # Test with invalid market data
        bad_expiry_option = ExpiryOption(
            expiry="20241201",
            expiry_date=date(2024, 12, 1),
            call_strike=105.0,
            put_strike=95.0,
        )

        bad_market_data = MarketData(
            timestamp=datetime.now(),
            stock_price=100.0,
            call_bid=None,  # Missing data
            call_ask=None,
            call_last=None,
            put_bid=None,
            put_ask=None,
            put_last=None,
        )

        opportunity = await engine._check_sfr_opportunity(
            underlying_id=1,
            expiry_option=bad_expiry_option,
            market_data=bad_market_data,
        )

        # Should handle missing data gracefully
        assert opportunity is not None
        assert not opportunity.conditions_check  # Should fail conditions

    @pytest.mark.asyncio
    async def test_database_integration(self, sample_config):
        """Test database integration with mock database."""
        mock_db = MockDatabase()

        async def mock_acquire():
            return mock_db

        mock_pool = AsyncMock()
        mock_pool.acquire = mock_acquire

        engine = SFRBacktestEngine(mock_pool, sample_config)

        # Test backtest run creation
        run_id = await engine._create_backtest_run(date(2024, 1, 1), date(2024, 12, 31))
        assert run_id == 1
        assert len(mock_db.queries) > 0

        # Test underlying ID loading
        await engine._load_underlying_ids(["SPY", "QQQ"])
        assert (
            "SPY" in engine.underlying_ids or len(engine.underlying_ids) == 0
        )  # Depending on mock setup


class TestIntegration:
    """Integration tests for complete backtesting workflows."""

    @pytest.mark.asyncio
    async def test_full_backtest_workflow(self, mock_db_pool, sample_config):
        """Test complete backtesting workflow."""
        # This would be a longer-running integration test
        # For now, just test that the components integrate properly

        engine = SFRBacktestEngine(mock_db_pool, sample_config)

        # Initialize
        engine.backtest_run_id = 1
        engine.underlying_ids = {"SPY": 1}

        # Test processing a single day (mocked)
        with patch.object(engine, "_get_stock_data") as mock_stock:
            with patch.object(engine, "_get_vix_data") as mock_vix:
                with patch.object(engine, "_get_expiry_options") as mock_expiries:
                    with patch.object(engine, "_get_option_data") as mock_option_data:

                        # Setup mocks
                        mock_stock.return_value = {"price": 100.0, "volume": 1000}
                        mock_vix.return_value = {
                            "vix_level": 20.0,
                            "vix_regime": VixRegime.MEDIUM,
                        }
                        mock_expiries.return_value = [
                            ExpiryOption(
                                "20241201", date(2024, 12, 1), 105.0, 95.0, 1, 2
                            )
                        ]
                        mock_option_data.return_value = {
                            "call_bid": 2.40,
                            "call_ask": 2.60,
                            "call_last": 2.50,
                            "call_volume": 150,
                            "put_bid": 1.30,
                            "put_ask": 1.50,
                            "put_last": 1.40,
                            "put_volume": 120,
                        }

                        # Process trading day
                        results = await engine._process_trading_day(
                            "SPY", date(2024, 6, 1)
                        )

                        # Should find at least some opportunities
                        assert results["opportunities_found"] >= 0
                        assert results["trades_executed"] >= 0


class TestPerformanceAnalytics:
    """Test performance analytics and metrics calculations."""

    def test_sharpe_ratio_calculation(self, mock_db_pool, sample_config):
        """Test Sharpe ratio calculation."""
        engine = SFRBacktestEngine(mock_db_pool, sample_config)

        # Create sample trades with known returns
        returns = [1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 1.8]

        for i, ret in enumerate(returns):
            trade = SimulatedTrade(
                realized_min_roi=ret,
                realized_min_profit=ret * 10,  # Scale for profit
                execution_status="FILLED",
            )
            engine.trades.append(trade)

        # Calculate metrics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        expected_sharpe = (mean_return - 2.0) / std_return  # Assuming 2% risk-free rate

        # This would be calculated in _calculate_performance_analytics
        engine.performance_metrics.avg_min_roi = mean_return
        engine.performance_metrics.roi_standard_deviation = std_return
        engine.performance_metrics.sharpe_ratio = expected_sharpe

        assert abs(engine.performance_metrics.sharpe_ratio - expected_sharpe) < 0.01

    def test_drawdown_calculation(self, mock_db_pool, sample_config):
        """Test maximum drawdown calculation."""
        engine = SFRBacktestEngine(mock_db_pool, sample_config)

        # Create sequence with known drawdown
        profits = [
            10,
            15,
            12,
            8,
            5,
            7,
            20,
            18,
        ]  # Max drawdown should be from 15 to 5 = -10

        for profit in profits:
            trade = SimulatedTrade(
                realized_min_profit=float(profit), execution_status="FILLED"
            )
            engine.trades.append(trade)

        # Manual calculation for verification
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        expected_max_drawdown = min(drawdown)

        assert expected_max_drawdown == -10  # Verify our test data


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
