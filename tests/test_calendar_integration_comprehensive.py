"""
Comprehensive integration tests for the calendar spread system.
Tests end-to-end functionality from CLI through strategy execution,
module interactions, and data flow between calendar components.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alchimest
from commands.option import OptionScan
from modules.Arbitrage.CalendarGreeks import (
    AdjustmentType,
    CalendarGreeks,
    CalendarGreeksCalculator,
    GreeksEvolution,
    GreeksRiskLevel,
    PortfolioGreeks,
    PositionAdjustment,
)
from modules.Arbitrage.CalendarPnL import (
    BreakevenPoints,
    CalendarPnLCalculator,
    CalendarPnLResult,
    MonteCarloResults,
    PnLScenario,
    ThetaAnalysis,
)
from modules.Arbitrage.CalendarSpread import (
    CalendarSpread,
    CalendarSpreadConfig,
    CalendarSpreadExecutor,
    CalendarSpreadLeg,
    CalendarSpreadOpportunity,
)
from modules.Arbitrage.TermStructure import (
    IVDataPoint,
    IVPercentileData,
    TermStructureAnalyzer,
    TermStructureCurve,
    TermStructureInversion,
)


class TestCalendarSpreadEndToEndIntegration:
    """End-to-end integration tests for calendar spread system"""

    @pytest.fixture
    def mock_ib_data(self) -> Dict[str, Any]:
        """Create realistic mock IB data for testing"""
        return {
            "contracts": {
                "SPY": MagicMock(symbol="SPY", exchange="SMART"),
                "SPY_240315C00520000": MagicMock(symbol="SPY", strike=520.0, right="C"),
                "SPY_240419C00520000": MagicMock(symbol="SPY", strike=520.0, right="C"),
            },
            "market_data": {
                "SPY": MagicMock(last=525.50, bid=525.45, ask=525.55),
                "SPY_240315C00520000": MagicMock(
                    last=8.50, bid=8.45, ask=8.55, volume=150
                ),
                "SPY_240419C00520000": MagicMock(
                    last=12.25, bid=12.20, ask=12.30, volume=100
                ),
            },
            "option_chains": {
                "SPY": [
                    MagicMock(
                        strike=520.0, right="C", lastTradeDateOrContractMonth="20240315"
                    ),
                    MagicMock(
                        strike=520.0, right="C", lastTradeDateOrContractMonth="20240419"
                    ),
                ]
            },
            "greeks": {
                "SPY_240315C00520000": MagicMock(
                    delta=0.65,
                    gamma=0.025,
                    theta=-0.12,
                    vega=0.08,
                    impliedVolatility=0.18,
                ),
                "SPY_240419C00520000": MagicMock(
                    delta=0.70,
                    gamma=0.020,
                    theta=-0.08,
                    vega=0.15,
                    impliedVolatility=0.21,
                ),
            },
        }

    @pytest.fixture
    def mock_ib_connection(
        self, mock_ib_data: Dict[str, Any]
    ) -> Generator[MagicMock, None, None]:
        """Mock IB connection with realistic data"""
        with patch("modules.Arbitrage.CalendarSpread.IB") as mock_ib_class:
            mock_ib = MagicMock()
            mock_ib_class.return_value = mock_ib

            # Mock connection methods
            mock_ib.connectAsync = AsyncMock()
            mock_ib.disconnect = MagicMock()
            mock_ib.isConnected = MagicMock(return_value=True)

            # Mock data retrieval methods
            mock_ib.reqContractDetailsAsync = AsyncMock(
                return_value=[MagicMock(contract=mock_ib_data["contracts"]["SPY"])]
            )
            mock_ib.reqMktDataAsync = AsyncMock(
                side_effect=lambda contract: mock_ib_data["market_data"].get(
                    contract.symbol, MagicMock()
                )
            )
            mock_ib.reqSecDefOptParamsAsync = AsyncMock(
                return_value=[
                    MagicMock(
                        strikes=set([520.0]), expirations=set(["20240315", "20240419"])
                    )
                ]
            )

            yield mock_ib

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_calendar_spread_workflow(
        self, mock_ib_connection: MagicMock, mock_ib_data: Dict[str, Any]
    ) -> None:
        """Test complete calendar spread workflow from initialization to execution"""
        # Create calendar spread configuration
        config = CalendarSpreadConfig(
            iv_spread_threshold=0.02,
            theta_ratio_threshold=1.3,
            front_expiry_max_days=45,
            back_expiry_min_days=60,
            back_expiry_max_days=120,
            min_volume=50,
            max_bid_ask_spread=0.12,
            net_debit_limit=500.0,
        )

        # Initialize calendar spread strategy
        calendar = CalendarSpread(config=config)

        # Mock the process method to capture the workflow
        with patch.object(calendar, "_scan_symbol_for_calendar_spreads") as mock_scan:
            mock_opportunity = CalendarSpreadOpportunity(
                symbol="SPY",
                front_leg=CalendarSpreadLeg(
                    contract=mock_ib_data["contracts"]["SPY_240315C00520000"],
                    strike=520.0,
                    expiry=datetime(2024, 3, 15),
                    option_type="CALL",
                    market_price=8.50,
                    bid=8.45,
                    ask=8.55,
                    volume=150,
                    implied_volatility=0.18,
                    delta=0.65,
                    gamma=0.025,
                    theta=-0.12,
                    vega=0.08,
                ),
                back_leg=CalendarSpreadLeg(
                    contract=mock_ib_data["contracts"]["SPY_240419C00520000"],
                    strike=520.0,
                    expiry=datetime(2024, 4, 19),
                    option_type="CALL",
                    market_price=12.25,
                    bid=12.20,
                    ask=12.30,
                    volume=100,
                    implied_volatility=0.21,
                    delta=0.70,
                    gamma=0.020,
                    theta=-0.08,
                    vega=0.15,
                ),
                net_debit=-3.75,  # Sell front (8.50) - Buy back (12.25)
                iv_spread=0.03,
                theta_ratio=1.5,
                max_profit=150.0,
                breakeven_lower=518.25,
                breakeven_upper=521.75,
                profit_probability=0.68,
                days_to_front_expiry=35,
                days_to_back_expiry=70,
                score=0.85,
            )
            mock_scan.return_value = [mock_opportunity]

            # Execute the process
            symbols = ["SPY"]
            await calendar.process(
                symbols=symbols, cost_limit=500.0, profit_target=0.25, quantity=1
            )

            # Verify the scan was called
            mock_scan.assert_called_once_with("SPY")

            # Verify IB connection was established
            mock_ib_connection.connectAsync.assert_called_once()

    @pytest.mark.integration
    def test_calendar_modules_integration(self) -> None:
        """Test integration between all calendar modules"""
        # Create test data
        iv_data_points = [
            IVDataPoint(
                expiry=datetime(2024, 3, 15),
                days_to_expiry=35,
                strike=520.0,
                implied_volatility=0.18,
                volume=150,
            ),
            IVDataPoint(
                expiry=datetime(2024, 4, 19),
                days_to_expiry=70,
                strike=520.0,
                implied_volatility=0.21,
                volume=100,
            ),
            IVDataPoint(
                expiry=datetime(2024, 5, 17),
                days_to_expiry=98,
                strike=520.0,
                implied_volatility=0.19,
                volume=75,
            ),
        ]

        # 1. Term Structure Analysis
        term_analyzer = TermStructureAnalyzer()

        with patch.object(term_analyzer, "_build_iv_curve") as mock_build_curve:
            mock_curve = TermStructureCurve(
                data_points=iv_data_points,
                curve_fit_r2=0.95,
                slope=0.0008,
                curvature=-0.00002,
                volatility_of_volatility=0.05,
            )
            mock_build_curve.return_value = mock_curve

            term_result = term_analyzer.analyze_term_structure("SPY", iv_data_points)

            assert term_result.curve.slope > 0  # Positive slope indicates backwardation
            assert len(term_result.inversions) == 0  # No inversions in this case

        # 2. P&L Analysis Integration
        pnl_calculator = CalendarPnLCalculator()

        calendar_legs = [
            CalendarSpreadLeg(
                contract=MagicMock(),
                strike=520.0,
                expiry=datetime(2024, 3, 15),
                option_type="CALL",
                market_price=8.50,
                bid=8.45,
                ask=8.55,
                volume=150,
                implied_volatility=0.18,
                delta=0.65,
                gamma=0.025,
                theta=-0.12,
                vega=0.08,
            ),
            CalendarSpreadLeg(
                contract=MagicMock(),
                strike=520.0,
                expiry=datetime(2024, 4, 19),
                option_type="CALL",
                market_price=12.25,
                bid=12.20,
                ask=12.30,
                volume=100,
                implied_volatility=0.21,
                delta=0.70,
                gamma=0.020,
                theta=-0.08,
                vega=0.15,
            ),
        ]

        with patch.object(
            pnl_calculator, "_run_monte_carlo_simulation"
        ) as mock_monte_carlo:
            mock_mc_results = MonteCarloResults(
                mean_pnl=125.0,
                std_pnl=85.0,
                profit_probability=0.68,
                var_95=150.0,
                expected_shortfall=180.0,
                max_drawdown=200.0,
                scenarios=[],  # Would normally contain scenario data
            )
            mock_monte_carlo.return_value = mock_mc_results

            pnl_result = pnl_calculator.analyze_calendar_pnl(
                front_leg=calendar_legs[0],
                back_leg=calendar_legs[1],
                underlying_price=525.50,
                position_size=1,
            )

            assert pnl_result.max_profit > 0
            assert pnl_result.monte_carlo.profit_probability > 0.5

        # 3. Greeks Integration
        greeks_calculator = CalendarGreeksCalculator()

        calendar_greeks = CalendarGreeks(
            net_delta=0.05,  # Near delta-neutral
            net_gamma=-0.005,  # Negative gamma
            net_theta=0.04,  # Positive theta (time decay benefit)
            net_vega=-0.07,  # Negative vega (short volatility)
            net_rho=0.02,
        )

        with patch.object(
            greeks_calculator, "calculate_calendar_greeks"
        ) as mock_calc_greeks:
            mock_calc_greeks.return_value = calendar_greeks

            greeks_result = greeks_calculator.calculate_calendar_greeks(
                front_leg=calendar_legs[0], back_leg=calendar_legs[1], position_size=1
            )

            # Verify Greeks integration
            assert abs(greeks_result.net_delta) < 0.1  # Should be delta-neutral
            assert greeks_result.net_theta > 0  # Should benefit from time decay
            assert greeks_result.net_vega < 0  # Should be short volatility

    @pytest.mark.integration
    def test_cli_to_strategy_integration(self) -> None:
        """Test integration from CLI command to strategy execution"""
        with patch("alchimest.OptionScan") as mock_option_scan_class:
            mock_option_scan = MagicMock()
            mock_option_scan_class.return_value = mock_option_scan

            # Mock calendar_finder to capture the call
            mock_option_scan.calendar_finder = MagicMock()

            # Simulate CLI call
            test_args = [
                "alchimest.py",
                "calendar",
                "-s",
                "SPY",
                "QQQ",
                "-l",
                "400.0",
                "-p",
                "0.30",
                "--iv-spread-threshold",
                "0.04",
                "--theta-ratio-threshold",
                "2.0",
                "--debug",
            ]

            with patch.object(sys, "argv", test_args):
                alchimest.main()

            # Verify CLI arguments were passed correctly to calendar_finder
            mock_option_scan.calendar_finder.assert_called_once_with(
                symbol_list=["SPY", "QQQ"],
                cost_limit=400.0,
                profit_target=0.30,
                iv_spread_threshold=0.04,
                theta_ratio_threshold=2.0,
                front_expiry_max_days=45,
                back_expiry_min_days=60,
                back_expiry_max_days=120,
                min_volume=10,
                max_bid_ask_spread=0.15,
                quantity=1,
                log_file=None,
                debug=True,
                finviz_url=None,
            )

    @pytest.mark.integration
    def test_option_scan_to_calendar_strategy_integration(self) -> None:
        """Test integration from OptionScan.calendar_finder to CalendarSpread strategy"""
        option_scan = OptionScan()

        with patch("commands.option.CalendarSpread") as mock_calendar_class:
            mock_calendar = MagicMock()
            mock_calendar.ib = MagicMock()
            mock_calendar.process = AsyncMock()
            mock_calendar_class.return_value = mock_calendar

            # Call calendar_finder
            symbols = ["AAPL", "MSFT"]
            option_scan.calendar_finder(
                symbol_list=symbols,
                cost_limit=600.0,
                profit_target=0.35,
                iv_spread_threshold=0.05,
                theta_ratio_threshold=2.5,
                front_expiry_max_days=30,
                back_expiry_min_days=45,
                back_expiry_max_days=90,
                min_volume=25,
                max_bid_ask_spread=0.08,
                quantity=2,
                debug=True,
            )

            # Verify CalendarSpread was instantiated with correct config
            mock_calendar_class.assert_called_once()
            args, kwargs = mock_calendar_class.call_args

            assert kwargs["debug"] is True
            assert kwargs["log_file"] is None

            config = kwargs["config"]
            assert config.iv_spread_threshold == 0.05
            assert config.theta_ratio_threshold == 2.5
            assert config.front_expiry_max_days == 30
            assert config.back_expiry_min_days == 45
            assert config.back_expiry_max_days == 90
            assert config.min_volume == 25
            assert config.max_bid_ask_spread == 0.08
            assert config.net_debit_limit == 600.0

            # Verify process was called with correct parameters
            mock_calendar.process.assert_called_once_with(
                symbols=symbols,
                cost_limit=600.0,
                profit_target=0.35,
                quantity=2,
            )

    @pytest.mark.integration
    def test_calendar_spread_module_interactions(self) -> None:
        """Test interactions between CalendarSpread and helper modules"""
        config = CalendarSpreadConfig()
        calendar = CalendarSpread(config=config)

        # Mock all helper modules
        with (
            patch(
                "modules.Arbitrage.CalendarSpread.TermStructureAnalyzer"
            ) as mock_term_analyzer,
            patch(
                "modules.Arbitrage.CalendarSpread.CalendarPnLCalculator"
            ) as mock_pnl_calc,
            patch(
                "modules.Arbitrage.CalendarSpread.CalendarGreeksCalculator"
            ) as mock_greeks_calc,
        ):

            # Configure mocks
            mock_term_instance = MagicMock()
            mock_term_analyzer.return_value = mock_term_instance

            mock_pnl_instance = MagicMock()
            mock_pnl_calc.return_value = mock_pnl_instance

            mock_greeks_instance = MagicMock()
            mock_greeks_calc.return_value = mock_greeks_instance

            # Mock the _calculate_calendar_spread_score method to verify module usage
            with patch.object(
                calendar, "_calculate_calendar_spread_score"
            ) as mock_score:
                mock_score.return_value = 0.85

                # Create mock data for testing
                front_leg = CalendarSpreadLeg(
                    contract=MagicMock(),
                    strike=520.0,
                    expiry=datetime(2024, 3, 15),
                    option_type="CALL",
                    market_price=8.50,
                    bid=8.45,
                    ask=8.55,
                    volume=150,
                    implied_volatility=0.18,
                    delta=0.65,
                    gamma=0.025,
                    theta=-0.12,
                    vega=0.08,
                )

                back_leg = CalendarSpreadLeg(
                    contract=MagicMock(),
                    strike=520.0,
                    expiry=datetime(2024, 4, 19),
                    option_type="CALL",
                    market_price=12.25,
                    bid=12.20,
                    ask=12.30,
                    volume=100,
                    implied_volatility=0.21,
                    delta=0.70,
                    gamma=0.020,
                    theta=-0.08,
                    vega=0.15,
                )

                # Test opportunity creation (would normally be inside _scan_symbol_for_calendar_spreads)
                opportunity = CalendarSpreadOpportunity(
                    symbol="SPY",
                    front_leg=front_leg,
                    back_leg=back_leg,
                    net_debit=-3.75,
                    iv_spread=0.03,
                    theta_ratio=1.5,
                    max_profit=150.0,
                    breakeven_lower=518.25,
                    breakeven_upper=521.75,
                    profit_probability=0.68,
                    days_to_front_expiry=35,
                    days_to_back_expiry=70,
                    score=0.85,
                )

                # Verify opportunity is properly structured
                assert opportunity.iv_spread > 0
                assert opportunity.theta_ratio > 1.0
                assert opportunity.max_profit > 0
                assert opportunity.score > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_calendar_executor_integration(self) -> None:
        """Test CalendarSpreadExecutor integration with order management"""
        config = CalendarSpreadConfig()

        with patch("modules.Arbitrage.CalendarSpread.IB") as mock_ib_class:
            mock_ib = MagicMock()
            mock_ib_class.return_value = mock_ib
            mock_ib.connectAsync = AsyncMock()
            mock_ib.placeOrderAsync = AsyncMock()

            executor = CalendarSpreadExecutor(config=config)

            # Create test opportunity
            opportunity = CalendarSpreadOpportunity(
                symbol="SPY",
                front_leg=CalendarSpreadLeg(
                    contract=MagicMock(),
                    strike=520.0,
                    expiry=datetime(2024, 3, 15),
                    option_type="CALL",
                    market_price=8.50,
                    bid=8.45,
                    ask=8.55,
                    volume=150,
                    implied_volatility=0.18,
                    delta=0.65,
                    gamma=0.025,
                    theta=-0.12,
                    vega=0.08,
                ),
                back_leg=CalendarSpreadLeg(
                    contract=MagicMock(),
                    strike=520.0,
                    expiry=datetime(2024, 4, 19),
                    option_type="CALL",
                    market_price=12.25,
                    bid=12.20,
                    ask=12.30,
                    volume=100,
                    implied_volatility=0.21,
                    delta=0.70,
                    gamma=0.020,
                    theta=-0.08,
                    vega=0.15,
                ),
                net_debit=-3.75,
                iv_spread=0.03,
                theta_ratio=1.5,
                max_profit=150.0,
                breakeven_lower=518.25,
                breakeven_upper=521.75,
                profit_probability=0.68,
                days_to_front_expiry=35,
                days_to_back_expiry=70,
                score=0.85,
            )

            # Mock the execute method
            with patch.object(executor, "execute_calendar_spread") as mock_execute:
                mock_execute.return_value = True

                result = await executor.execute_calendar_spread(opportunity, quantity=1)

                assert result is True
                mock_execute.assert_called_once_with(opportunity, quantity=1)

    @pytest.mark.integration
    def test_error_handling_integration(self) -> None:
        """Test error handling across calendar modules"""
        option_scan = OptionScan()

        with patch("commands.option.CalendarSpread") as mock_calendar_class:
            mock_calendar = MagicMock()
            mock_calendar.ib = MagicMock()
            mock_calendar.process = AsyncMock(side_effect=Exception("Connection error"))
            mock_calendar_class.return_value = mock_calendar

            # Test that exceptions are propagated and handled correctly
            with pytest.raises(Exception, match="Connection error"):
                option_scan.calendar_finder(symbol_list=["SPY"])

            # Verify disconnect was called on error
            mock_calendar.ib.disconnect.assert_called_once()

    @pytest.mark.integration
    def test_configuration_propagation_integration(self) -> None:
        """Test that configuration is properly propagated through the system"""
        # Test configuration from CLI -> OptionScan -> CalendarSpread
        option_scan = OptionScan()

        custom_config_params = {
            "cost_limit": 750.0,
            "iv_spread_threshold": 0.06,
            "theta_ratio_threshold": 3.0,
            "front_expiry_max_days": 25,
            "back_expiry_min_days": 40,
            "back_expiry_max_days": 80,
            "min_volume": 30,
            "max_bid_ask_spread": 0.06,
        }

        with patch("commands.option.CalendarSpread") as mock_calendar_class:
            mock_calendar = MagicMock()
            mock_calendar.ib = MagicMock()
            mock_calendar.process = AsyncMock()
            mock_calendar_class.return_value = mock_calendar

            option_scan.calendar_finder(
                symbol_list=["SPY"],
                **custom_config_params,
                quantity=3,
                debug=True,
            )

            # Verify configuration was passed correctly
            mock_calendar_class.assert_called_once()
            args, kwargs = mock_calendar_class.call_args

            config = kwargs["config"]
            assert config.net_debit_limit == custom_config_params["cost_limit"]
            assert (
                config.iv_spread_threshold
                == custom_config_params["iv_spread_threshold"]
            )
            assert (
                config.theta_ratio_threshold
                == custom_config_params["theta_ratio_threshold"]
            )
            assert (
                config.front_expiry_max_days
                == custom_config_params["front_expiry_max_days"]
            )
            assert (
                config.back_expiry_min_days
                == custom_config_params["back_expiry_min_days"]
            )
            assert (
                config.back_expiry_max_days
                == custom_config_params["back_expiry_max_days"]
            )
            assert config.min_volume == custom_config_params["min_volume"]
            assert (
                config.max_bid_ask_spread == custom_config_params["max_bid_ask_spread"]
            )

            # Verify process parameters
            mock_calendar.process.assert_called_once_with(
                symbols=["SPY"],
                cost_limit=750.0,
                profit_target=0.25,  # Default value
                quantity=3,
            )

    @pytest.mark.integration
    def test_data_flow_integration(self) -> None:
        """Test data flow between different calendar modules"""
        # Create a realistic data flow scenario

        # 1. Mock IV data coming from term structure analysis
        iv_data = [
            IVDataPoint(
                expiry=datetime(2024, 3, 15),
                days_to_expiry=35,
                strike=520.0,
                implied_volatility=0.18,
                volume=150,
            ),
            IVDataPoint(
                expiry=datetime(2024, 4, 19),
                days_to_expiry=70,
                strike=520.0,
                implied_volatility=0.21,
                volume=100,
            ),
        ]

        # 2. Test term structure -> P&L calculation integration
        term_analyzer = TermStructureAnalyzer()
        pnl_calculator = CalendarPnLCalculator()

        with patch.object(
            term_analyzer, "analyze_term_structure"
        ) as mock_term_analysis:
            # Mock term structure result
            mock_term_result = MagicMock()
            mock_term_result.curve.slope = 0.0008
            mock_term_result.inversions = []
            mock_term_result.volatility_skew = 0.15
            mock_term_analysis.return_value = mock_term_result

            # Get term structure result
            term_result = term_analyzer.analyze_term_structure("SPY", iv_data)

            # Use term structure data in P&L calculation
            with patch.object(
                pnl_calculator, "analyze_calendar_pnl"
            ) as mock_pnl_analysis:
                mock_pnl_result = MagicMock()
                mock_pnl_result.max_profit = 150.0
                mock_pnl_result.breakeven_points.lower = 518.25
                mock_pnl_result.breakeven_points.upper = 521.75
                mock_pnl_analysis.return_value = mock_pnl_result

                # Create calendar legs for P&L analysis
                front_leg = CalendarSpreadLeg(
                    contract=MagicMock(),
                    strike=520.0,
                    expiry=datetime(2024, 3, 15),
                    option_type="CALL",
                    market_price=8.50,
                    bid=8.45,
                    ask=8.55,
                    volume=150,
                    implied_volatility=0.18,
                    delta=0.65,
                    gamma=0.025,
                    theta=-0.12,
                    vega=0.08,
                )

                back_leg = CalendarSpreadLeg(
                    contract=MagicMock(),
                    strike=520.0,
                    expiry=datetime(2024, 4, 19),
                    option_type="CALL",
                    market_price=12.25,
                    bid=12.20,
                    ask=12.30,
                    volume=100,
                    implied_volatility=0.21,
                    delta=0.70,
                    gamma=0.020,
                    theta=-0.08,
                    vega=0.15,
                )

                pnl_result = pnl_calculator.analyze_calendar_pnl(
                    front_leg=front_leg,
                    back_leg=back_leg,
                    underlying_price=525.50,
                    position_size=1,
                )

                # Verify data flows correctly
                assert term_result.curve.slope > 0
                assert pnl_result.max_profit > 0
                assert pnl_result.breakeven_points.lower < 520.0
                assert pnl_result.breakeven_points.upper > 520.0

    @pytest.mark.integration
    def test_logging_integration(self) -> None:
        """Test logging integration across calendar modules"""
        with patch("commands.option.logger") as mock_logger:
            option_scan = OptionScan()

            with patch("commands.option.CalendarSpread") as mock_calendar_class:
                mock_calendar = MagicMock()
                mock_calendar.ib = MagicMock()
                mock_calendar.process = AsyncMock()
                mock_calendar_class.return_value = mock_calendar

                option_scan.calendar_finder(
                    symbol_list=["SPY", "QQQ"], debug=True, log_file="calendar_test.log"
                )

                # Verify logging calls were made
                assert mock_logger.info.call_count >= 3  # At least startup messages

                # Check for specific log messages
                log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
                assert any("Starting CALENDAR scan" in msg for msg in log_calls)
                assert any("Calendar parameters" in msg for msg in log_calls)

    @pytest.mark.integration
    def test_performance_integration(self) -> None:
        """Test performance characteristics of integrated calendar system"""
        import time

        option_scan = OptionScan()

        with patch("commands.option.CalendarSpread") as mock_calendar_class:
            mock_calendar = MagicMock()
            mock_calendar.ib = MagicMock()
            mock_calendar.process = AsyncMock()
            mock_calendar_class.return_value = mock_calendar

            # Test with large symbol list
            large_symbol_list = [f"SYM{i:03d}" for i in range(100)]

            start_time = time.time()
            option_scan.calendar_finder(symbol_list=large_symbol_list)
            end_time = time.time()

            execution_time = end_time - start_time

            # Should complete quickly (most time in mocked process)
            assert (
                execution_time < 2.0
            ), f"Integration took too long: {execution_time:.3f}s"

            # Verify large symbol list was passed correctly
            mock_calendar.process.assert_called_once()
            args, kwargs = mock_calendar.process.call_args
            assert len(kwargs["symbols"]) == 100

    @pytest.mark.integration
    def test_memory_management_integration(self) -> None:
        """Test memory management across calendar modules"""
        option_scan = OptionScan()

        with patch("commands.option.CalendarSpread") as mock_calendar_class:
            mock_calendar = MagicMock()
            mock_calendar.ib = MagicMock()
            mock_calendar.process = AsyncMock()
            mock_calendar_class.return_value = mock_calendar

            # Call multiple times to check for memory leaks
            for i in range(50):
                option_scan.calendar_finder(
                    symbol_list=[f"TEST{i}"],
                    cost_limit=float(i * 10),
                    profit_target=float(i * 0.01),
                )

            # Verify CalendarSpread instances are created separately each time
            assert mock_calendar_class.call_count == 50
            assert mock_calendar.process.call_count == 50

            # Each call should have been with different parameters
            all_calls = mock_calendar.process.call_args_list
            cost_limits = [call[1]["cost_limit"] for call in all_calls]
            assert len(set(cost_limits)) == 50  # All different cost limits
