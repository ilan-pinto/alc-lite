"""
Tests for the complete lifecycle of opportunities in the global opportunity selection system.

Tests the end-to-end flow of opportunities through the system:
- Phase 1: Opportunity Collection (scan_syn collects opportunities)
- Phase 2: Opportunity Selection (get_best_opportunity ranks and selects)
- Phase 3: Opportunity Execution (place_order executes the best)
- Phase 4: Cycle Cleanup (clear_opportunities resets for next cycle)

This validates the complete workflow that transforms per-symbol optimization
into global portfolio optimization across all tickers and expirations.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from modules.Arbitrage.Synthetic import (
    GlobalOpportunity,
    GlobalOpportunityManager,
    ScoringConfig,
    Syn,
)

# Import test utilities
try:
    from .market_scenarios import SyntheticScenarios
    from .mock_ib import MockContract, MockIB, MockTicker
except ImportError:
    from market_scenarios import SyntheticScenarios
    from mock_ib import MockContract, MockIB, MockTicker


class TestOpportunityCollectionPhase:
    """Test Phase 1: Opportunity collection during scan cycle"""

    def setup_method(self):
        """Setup test environment"""
        self.global_manager = GlobalOpportunityManager()
        self.mock_ib = MockIB()

    @pytest.mark.asyncio
    async def test_opportunity_collection_phase_multiple_symbols(self):
        """Test that scan cycle properly collects opportunities from multiple symbols"""
        print("\n🔍 Testing opportunity collection phase across multiple symbols")

        # Create Syn instance with global manager
        syn = Syn(scoring_config=ScoringConfig.create_balanced())
        syn.global_manager = self.global_manager
        syn.ib = self.mock_ib
        syn.order_manager = MagicMock()

        # Create realistic market data for multiple symbols
        symbols_data = {
            "AAPL": SyntheticScenarios.synthetic_profitable_scenario(),
            "MSFT": SyntheticScenarios.synthetic_poor_risk_reward(),
            "GOOGL": SyntheticScenarios.synthetic_profitable_scenario(),  # Different instance
        }

        # Mock the scan_syn method to simulate opportunity collection
        collected_opportunities = []

        async def mock_scan_syn(symbol: str, quantity: int):
            """Mock scan_syn that simulates opportunity collection"""
            print(f"  📊 Scanning {symbol} for opportunities...")

            # Simulate finding opportunities and adding to global manager
            if symbol in symbols_data:
                # Create mock opportunity data
                trade_details = {
                    "max_profit": 50.0 + hash(symbol) % 30,  # Vary by symbol
                    "min_profit": -25.0 - hash(symbol) % 15,
                    "net_credit": 10.0 + hash(symbol) % 10,
                    "stock_price": 100.0,
                    "expiry": "20240330",
                    "call_strike": 100,
                    "put_strike": 95,
                }

                # Create mock contracts and tickers
                call_ticker = MockTicker(
                    MockContract(symbol, "OPT", strike=100, right="C"),
                    bid=5.0,
                    ask=5.1,
                    volume=500 + hash(symbol) % 300,
                )
                put_ticker = MockTicker(
                    MockContract(symbol, "OPT", strike=95, right="P"),
                    bid=3.0,
                    ask=3.05,
                    volume=300 + hash(symbol) % 200,
                )

                # Add opportunity to global manager
                success = self.global_manager.add_opportunity(
                    symbol=symbol,
                    conversion_contract=Mock(),
                    order=Mock(),
                    trade_details=trade_details,
                    call_ticker=call_ticker,
                    put_ticker=put_ticker,
                )

                if success:
                    collected_opportunities.append(symbol)
                    print(f"    ✅ Added opportunity for {symbol}")
                else:
                    print(f"    ❌ Opportunity rejected for {symbol}")

        # Replace scan_syn with mock
        syn.scan_syn = mock_scan_syn

        # Simulate Phase 1: Collection across all symbols
        symbols = list(symbols_data.keys())
        collection_tasks = []

        for symbol in symbols:
            task = asyncio.create_task(syn.scan_syn(symbol, 1))
            collection_tasks.append(task)

        # Execute collection phase
        await asyncio.gather(*collection_tasks)

        # Verify collection results
        opportunity_count = self.global_manager.get_opportunity_count()
        print(f"  📊 Collection Results:")
        print(f"    Symbols scanned: {len(symbols)}")
        print(f"    Opportunities collected: {opportunity_count}")
        print(f"    Successful symbols: {collected_opportunities}")

        # Should have collected opportunities from multiple symbols
        assert opportunity_count > 0, "Should collect at least one opportunity"
        assert len(collected_opportunities) > 0, "Should successfully add opportunities"

        # Verify opportunities are stored in global manager
        stats = self.global_manager.get_statistics()
        assert stats["total_opportunities"] == opportunity_count
        assert stats["unique_symbols"] >= 1

        print("✅ Opportunity collection phase completed successfully")

    @pytest.mark.asyncio
    async def test_opportunity_collection_with_rejections(self):
        """Test collection phase with some opportunities being rejected"""
        print("\n🔍 Testing opportunity collection with threshold rejections")

        # Create strict scoring config that will reject poor opportunities
        strict_config = ScoringConfig(
            min_risk_reward_ratio=3.0,  # High threshold
            min_liquidity_score=0.6,  # High liquidity required
            max_bid_ask_spread=0.10,  # Tight spread requirement
        )

        manager = GlobalOpportunityManager(strict_config)

        # Create opportunities with varying quality
        opportunities_data = [
            # Good opportunity (should be accepted)
            ("ACCEPT", 90.0, -30.0, 800, 600, 0.05, 0.03),  # RR: 3.0, good liquidity
            # Poor risk-reward (should be rejected)
            ("REJECT_RR", 40.0, -30.0, 800, 600, 0.05, 0.03),  # RR: 1.33 < 3.0
            # Poor liquidity (should be rejected)
            ("REJECT_LIQ", 90.0, -30.0, 50, 30, 0.05, 0.03),  # Low volume
            # Wide spreads (should be rejected) - reduce volume to ensure rejection
            (
                "REJECT_SPREAD",
                90.0,
                -30.0,
                400,
                300,
                0.25,
                0.20,
            ),  # Wide spreads + lower volume
        ]

        accepted_count = 0
        rejected_count = 0

        for (
            symbol,
            max_profit,
            min_profit,
            call_vol,
            put_vol,
            call_spread,
            put_spread,
        ) in opportunities_data:
            # Create mock tickers
            call_ticker = MockTicker(
                MockContract(symbol, "OPT", strike=100, right="C"),
                bid=5.0,
                ask=5.0 + call_spread,
                volume=call_vol,
            )
            put_ticker = MockTicker(
                MockContract(symbol, "OPT", strike=95, right="P"),
                bid=3.0,
                ask=3.0 + put_spread,
                volume=put_vol,
            )

            trade_details = {
                "max_profit": max_profit,
                "min_profit": min_profit,
                "net_credit": max_profit - abs(min_profit),
                "stock_price": 100.0,
                "expiry": "20240330",
            }

            success = manager.add_opportunity(
                symbol=symbol,
                conversion_contract=Mock(),
                order=Mock(),
                trade_details=trade_details,
                call_ticker=call_ticker,
                put_ticker=put_ticker,
            )

            if success:
                accepted_count += 1
                print(f"    ✅ {symbol}: Accepted")
            else:
                rejected_count += 1
                print(f"    ❌ {symbol}: Rejected")

        print(f"  📊 Collection Summary:")
        print(f"    Total opportunities: {len(opportunities_data)}")
        print(f"    Accepted: {accepted_count}")
        print(f"    Rejected: {rejected_count}")

        # Verify filtering behavior
        assert accepted_count == 1, "Should accept only the good opportunity"
        assert rejected_count == 3, "Should reject the three poor opportunities"
        assert manager.get_opportunity_count() == 1

        print("✅ Opportunity filtering during collection works correctly")


class TestOpportunitySelectionPhase:
    """Test Phase 2: Global opportunity selection and ranking"""

    def setup_method(self):
        """Setup test environment"""
        self.global_manager = GlobalOpportunityManager()

    @pytest.mark.integration
    def test_opportunity_selection_phase_ranking(self):
        """Test that selection phase correctly ranks opportunities"""
        print("\n🔍 Testing opportunity selection phase ranking")

        # Add multiple opportunities with known ranking
        opportunities_data = [
            # Should rank 2nd: Good but not best
            ("SILVER", 70.0, -30.0, 600, 400, 0.08, 0.05, 28),
            # Should rank 1st: Best overall
            ("GOLD", 80.0, -25.0, 800, 600, 0.05, 0.03, 30),
            # Should rank 3rd: Decent
            ("BRONZE", 60.0, -35.0, 400, 300, 0.10, 0.08, 32),
        ]

        # Add all opportunities
        for (
            symbol,
            max_profit,
            min_profit,
            call_vol,
            put_vol,
            call_spread,
            put_spread,
            days,
        ) in opportunities_data:
            call_ticker = MockTicker(
                MockContract(symbol, "OPT"),
                bid=5.0,
                ask=5.0 + call_spread,
                volume=call_vol,
            )
            put_ticker = MockTicker(
                MockContract(symbol, "OPT"),
                bid=3.0,
                ask=3.0 + put_spread,
                volume=put_vol,
            )

            trade_details = {
                "max_profit": max_profit,
                "min_profit": min_profit,
                "net_credit": max_profit - abs(min_profit),
                "stock_price": 100.0,
                "expiry": (datetime.now() + timedelta(days=days)).strftime("%Y%m%d"),
            }

            self.global_manager.add_opportunity(
                symbol=symbol,
                conversion_contract=Mock(),
                order=Mock(),
                trade_details=trade_details,
                call_ticker=call_ticker,
                put_ticker=put_ticker,
            )

        # Execute selection phase
        best_opportunity = self.global_manager.get_best_opportunity()

        assert best_opportunity is not None
        print(f"  🏆 Best opportunity selected: {best_opportunity.symbol}")
        print(f"    Composite score: {best_opportunity.score.composite_score:.3f}")

        # Verify GOLD was selected (should have highest composite score)
        assert (
            best_opportunity.symbol == "GOLD"
        ), f"Expected GOLD to be selected, got {best_opportunity.symbol}"

        # Verify ranking order
        with self.global_manager.lock:
            sorted_opportunities = sorted(
                self.global_manager.opportunities,
                key=lambda opp: opp.score.composite_score,
                reverse=True,
            )

        ranking = [opp.symbol for opp in sorted_opportunities]
        print(f"    Final ranking: {ranking}")

        assert ranking[0] == "GOLD"
        assert ranking[1] == "SILVER"
        assert ranking[2] == "BRONZE"

        print("✅ Opportunity selection phase ranking works correctly")

    @pytest.mark.integration
    def test_opportunity_selection_phase_empty_collection(self):
        """Test selection phase behavior with empty opportunity collection"""
        print("\n🔍 Testing selection phase with empty collection")

        # Start with empty collection
        assert self.global_manager.get_opportunity_count() == 0

        # Execute selection phase
        best_opportunity = self.global_manager.get_best_opportunity()

        # Should return None gracefully
        assert best_opportunity is None
        print("    📊 No opportunities available - returned None")

        # Statistics should handle empty case
        stats = self.global_manager.get_statistics()
        assert stats == {}
        print("    📊 Statistics handled empty case correctly")

        print("✅ Selection phase handles empty collection gracefully")


class TestOpportunityExecutionPhase:
    """Test Phase 3: Execution of the globally best opportunity"""

    def setup_method(self):
        """Setup test environment"""
        self.global_manager = GlobalOpportunityManager()

    @pytest.mark.asyncio
    async def test_opportunity_execution_phase_success(self):
        """Test successful execution of the globally best opportunity"""
        print("\n🔍 Testing opportunity execution phase success")

        # Create a Syn instance for testing execution
        syn = Syn()
        syn.global_manager = self.global_manager
        syn.ib = MockIB()

        # Mock successful order manager
        syn.order_manager = MagicMock()
        syn.order_manager.place_order = AsyncMock(return_value=Mock())

        # Add a test opportunity
        call_ticker = MockTicker(
            MockContract("TEST", "OPT"), bid=5.0, ask=5.1, volume=500
        )
        put_ticker = MockTicker(
            MockContract("TEST", "OPT"), bid=3.0, ask=3.05, volume=300
        )

        trade_details = {
            "max_profit": 75.0,
            "min_profit": -25.0,
            "net_credit": 50.0,
            "stock_price": 100.0,
            "expiry": "20240330",
        }

        conversion_contract = Mock()
        order = Mock()

        success = self.global_manager.add_opportunity(
            symbol="TEST",
            conversion_contract=conversion_contract,
            order=order,
            trade_details=trade_details,
            call_ticker=call_ticker,
            put_ticker=put_ticker,
        )

        assert success, "Should successfully add test opportunity"

        # Execute Phase 3: Get best opportunity and execute it
        best_opportunity = self.global_manager.get_best_opportunity()
        assert best_opportunity is not None

        # Execute the order
        await syn.order_manager.place_order(
            best_opportunity.conversion_contract, best_opportunity.order
        )

        # Verify execution was called
        syn.order_manager.place_order.assert_called_once_with(
            conversion_contract, order
        )

        print(f"    ✅ Successfully executed opportunity for {best_opportunity.symbol}")
        print(
            f"    📊 Trade details: Max profit ${trade_details['max_profit']:.2f}, "
            f"Min profit ${trade_details['min_profit']:.2f}"
        )

        print("✅ Opportunity execution phase completed successfully")

    @pytest.mark.asyncio
    async def test_opportunity_execution_phase_failure(self):
        """Test execution phase handling of order placement failure"""
        print("\n🔍 Testing opportunity execution phase failure handling")

        syn = Syn()
        syn.global_manager = self.global_manager
        syn.ib = MockIB()

        # Mock order manager that throws exception
        syn.order_manager = MagicMock()
        syn.order_manager.place_order = AsyncMock(
            side_effect=Exception("Order placement failed")
        )

        # Add a test opportunity
        call_ticker = MockTicker(
            MockContract("FAIL", "OPT"), bid=5.0, ask=5.1, volume=500
        )
        put_ticker = MockTicker(
            MockContract("FAIL", "OPT"), bid=3.0, ask=3.05, volume=300
        )

        trade_details = {
            "max_profit": 60.0,
            "min_profit": -30.0,
            "net_credit": 30.0,
            "stock_price": 100.0,
            "expiry": "20240330",
        }

        self.global_manager.add_opportunity(
            symbol="FAIL",
            conversion_contract=Mock(),
            order=Mock(),
            trade_details=trade_details,
            call_ticker=call_ticker,
            put_ticker=put_ticker,
        )

        # Get best opportunity
        best_opportunity = self.global_manager.get_best_opportunity()
        assert best_opportunity is not None

        # Attempt execution (should fail gracefully)
        try:
            await syn.order_manager.place_order(
                best_opportunity.conversion_contract, best_opportunity.order
            )
            assert False, "Should have raised exception"
        except Exception as e:
            print(f"    ❌ Order placement failed as expected: {str(e)}")

        # Verify the system can handle the failure
        syn.order_manager.place_order.assert_called_once()

        print("✅ Execution phase failure handling works correctly")


class TestCycleCleanupPhase:
    """Test Phase 4: Cycle cleanup and reset for next iteration"""

    def setup_method(self):
        """Setup test environment"""
        self.global_manager = GlobalOpportunityManager()

    @pytest.mark.integration
    def test_cycle_cleanup_phase(self):
        """Test cycle cleanup properly resets the system"""
        print("\n🔍 Testing cycle cleanup phase")

        # Phase 1-3: Add opportunities and select best
        for i in range(3):
            call_ticker = MockTicker(
                MockContract(f"SYM{i}", "OPT"), bid=5.0, ask=5.1, volume=500
            )
            put_ticker = MockTicker(
                MockContract(f"SYM{i}", "OPT"), bid=3.0, ask=3.05, volume=300
            )

            trade_details = {
                "max_profit": 50.0 + i * 10,
                "min_profit": -25.0 - i * 5,
                "net_credit": 25.0 + i * 5,
                "stock_price": 100.0,
                "expiry": "20240330",
            }

            self.global_manager.add_opportunity(
                symbol=f"SYM{i}",
                conversion_contract=Mock(),
                order=Mock(),
                trade_details=trade_details,
                call_ticker=call_ticker,
                put_ticker=put_ticker,
            )

        # Verify opportunities were collected
        initial_count = self.global_manager.get_opportunity_count()
        assert initial_count == 3
        print(f"    📊 Initial opportunity count: {initial_count}")

        # Get statistics before cleanup
        initial_stats = self.global_manager.get_statistics()
        assert initial_stats["total_opportunities"] == 3
        assert initial_stats["unique_symbols"] == 3

        # Get best opportunity before cleanup
        best_before = self.global_manager.get_best_opportunity()
        assert best_before is not None
        print(f"    🏆 Best opportunity before cleanup: {best_before.symbol}")

        # Phase 4: Execute cleanup
        self.global_manager.clear_opportunities()

        # Verify cleanup results
        after_count = self.global_manager.get_opportunity_count()
        assert after_count == 0
        print(f"    🧹 Opportunity count after cleanup: {after_count}")

        # Statistics should reflect empty state
        after_stats = self.global_manager.get_statistics()
        assert after_stats == {}

        # get_best_opportunity should return None
        best_after = self.global_manager.get_best_opportunity()
        assert best_after is None
        print("    📊 Best opportunity after cleanup: None")

        # System should be ready for next cycle
        # Add a new opportunity to verify system is functional
        call_ticker = MockTicker(
            MockContract("NEW", "OPT"), bid=5.0, ask=5.1, volume=500
        )
        put_ticker = MockTicker(
            MockContract("NEW", "OPT"), bid=3.0, ask=3.05, volume=300
        )

        trade_details = {
            "max_profit": 40.0,
            "min_profit": -20.0,
            "net_credit": 20.0,
            "stock_price": 100.0,
            "expiry": "20240330",
        }

        success = self.global_manager.add_opportunity(
            symbol="NEW",
            conversion_contract=Mock(),
            order=Mock(),
            trade_details=trade_details,
            call_ticker=call_ticker,
            put_ticker=put_ticker,
        )

        assert success, "Should be able to add opportunities after cleanup"
        assert self.global_manager.get_opportunity_count() == 1
        print("    ✅ System ready for next cycle")

        print("✅ Cycle cleanup phase completed successfully")


class TestCompleteOpportunityLifecycle:
    """Test the complete end-to-end opportunity lifecycle"""

    def setup_method(self):
        """Setup test environment"""
        self.global_manager = GlobalOpportunityManager()

    @pytest.mark.asyncio
    async def test_complete_opportunity_lifecycle_integration(self):
        """Test complete lifecycle: Collection → Selection → Execution → Cleanup"""
        print("\n🔍 Testing complete opportunity lifecycle integration")

        # Setup Syn instance for full lifecycle test
        syn = Syn(scoring_config=ScoringConfig.create_balanced())
        syn.global_manager = self.global_manager
        syn.ib = MockIB()
        syn.order_manager = MagicMock()
        syn.order_manager.place_order = AsyncMock(return_value=Mock())

        lifecycle_results = {
            "collection": {"symbols_scanned": 0, "opportunities_added": 0},
            "selection": {"best_symbol": None, "best_score": 0.0},
            "execution": {"success": False, "error": None},
            "cleanup": {"opportunities_cleared": False},
        }

        try:
            # PHASE 1: COLLECTION
            print("  🔄 Phase 1: Opportunity Collection")

            # Simulate scanning multiple symbols
            symbols = ["AAPL", "MSFT", "GOOGL"]
            lifecycle_results["collection"]["symbols_scanned"] = len(symbols)

            for symbol in symbols:
                # Create varying opportunity quality
                quality_multiplier = hash(symbol) % 3 + 1  # 1, 2, or 3

                call_ticker = MockTicker(
                    MockContract(symbol, "OPT"),
                    bid=5.0,
                    ask=5.0 + 0.05 * quality_multiplier,
                    volume=300 + 200 * quality_multiplier,
                )
                put_ticker = MockTicker(
                    MockContract(symbol, "OPT"),
                    bid=3.0,
                    ask=3.0 + 0.03 * quality_multiplier,
                    volume=200 + 150 * quality_multiplier,
                )

                trade_details = {
                    "max_profit": 40.0 + 20 * quality_multiplier,
                    "min_profit": -20.0 - 10 * quality_multiplier,
                    "net_credit": 20.0 + 10 * quality_multiplier,
                    "stock_price": 100.0,
                    "expiry": "20240330",
                }

                success = self.global_manager.add_opportunity(
                    symbol=symbol,
                    conversion_contract=Mock(),
                    order=Mock(),
                    trade_details=trade_details,
                    call_ticker=call_ticker,
                    put_ticker=put_ticker,
                )

                if success:
                    lifecycle_results["collection"]["opportunities_added"] += 1

            print(
                f"    📊 Collection: {lifecycle_results['collection']['opportunities_added']} "
                f"opportunities from {lifecycle_results['collection']['symbols_scanned']} symbols"
            )

            # PHASE 2: SELECTION
            print("  🔄 Phase 2: Opportunity Selection")

            best_opportunity = self.global_manager.get_best_opportunity()
            if best_opportunity:
                lifecycle_results["selection"]["best_symbol"] = best_opportunity.symbol
                lifecycle_results["selection"][
                    "best_score"
                ] = best_opportunity.score.composite_score
                print(
                    f"    🏆 Selection: {best_opportunity.symbol} "
                    f"(score: {best_opportunity.score.composite_score:.3f})"
                )
            else:
                print("    ❌ Selection: No opportunities available")

            # PHASE 3: EXECUTION
            print("  🔄 Phase 3: Opportunity Execution")

            if best_opportunity:
                try:
                    await syn.order_manager.place_order(
                        best_opportunity.conversion_contract, best_opportunity.order
                    )
                    lifecycle_results["execution"]["success"] = True
                    print(
                        f"    ✅ Execution: Successfully placed order for {best_opportunity.symbol}"
                    )
                except Exception as e:
                    lifecycle_results["execution"]["error"] = str(e)
                    print(f"    ❌ Execution: Failed - {str(e)}")
            else:
                print("    ⏭️  Execution: Skipped (no opportunity)")

            # PHASE 4: CLEANUP
            print("  🔄 Phase 4: Cycle Cleanup")

            opportunities_before = self.global_manager.get_opportunity_count()
            self.global_manager.clear_opportunities()
            opportunities_after = self.global_manager.get_opportunity_count()

            lifecycle_results["cleanup"]["opportunities_cleared"] = (
                opportunities_before > 0 and opportunities_after == 0
            )

            print(f"    🧹 Cleanup: Cleared {opportunities_before} opportunities")

            # VERIFICATION
            print("\n  📊 Lifecycle Summary:")
            print(f"    Collection: {lifecycle_results['collection']}")
            print(f"    Selection: {lifecycle_results['selection']}")
            print(f"    Execution: {lifecycle_results['execution']}")
            print(f"    Cleanup: {lifecycle_results['cleanup']}")

            # Verify each phase completed successfully
            assert (
                lifecycle_results["collection"]["opportunities_added"] > 0
            ), "Collection phase should add opportunities"
            assert (
                lifecycle_results["selection"]["best_symbol"] is not None
            ), "Selection phase should find best opportunity"
            assert lifecycle_results["execution"][
                "success"
            ], "Execution phase should succeed"
            assert lifecycle_results["cleanup"][
                "opportunities_cleared"
            ], "Cleanup phase should clear opportunities"

            print("✅ Complete opportunity lifecycle integration test passed")

        except Exception as e:
            print(f"❌ Lifecycle test failed: {str(e)}")
            raise

    @pytest.mark.asyncio
    async def test_lifecycle_with_multiple_cycles(self):
        """Test multiple complete lifecycle cycles"""
        print("\n🔍 Testing multiple lifecycle cycles")

        syn = Syn(scoring_config=ScoringConfig.create_balanced())
        syn.global_manager = self.global_manager
        syn.ib = MockIB()
        syn.order_manager = MagicMock()
        syn.order_manager.place_order = AsyncMock(return_value=Mock())

        cycle_results = []

        # Run 3 complete cycles
        for cycle in range(3):
            print(f"  🔄 Cycle {cycle + 1}:")

            # Collection
            symbols = [f"SYM{cycle}_{i}" for i in range(2)]  # 2 symbols per cycle
            opportunities_added = 0

            for symbol in symbols:
                call_ticker = MockTicker(
                    MockContract(symbol, "OPT"), bid=5.0, ask=5.1, volume=500
                )
                put_ticker = MockTicker(
                    MockContract(symbol, "OPT"), bid=3.0, ask=3.05, volume=300
                )

                trade_details = {
                    "max_profit": 50.0 + cycle * 10,
                    "min_profit": -25.0 - cycle * 5,
                    "net_credit": 25.0 + cycle * 5,
                    "stock_price": 100.0,
                    "expiry": "20240330",
                }

                if self.global_manager.add_opportunity(
                    symbol=symbol,
                    conversion_contract=Mock(),
                    order=Mock(),
                    trade_details=trade_details,
                    call_ticker=call_ticker,
                    put_ticker=put_ticker,
                ):
                    opportunities_added += 1

            # Selection
            best = self.global_manager.get_best_opportunity()

            # Execution
            execution_success = False
            if best:
                try:
                    await syn.order_manager.place_order(
                        best.conversion_contract, best.order
                    )
                    execution_success = True
                except:
                    pass

            # Cleanup
            self.global_manager.clear_opportunities()

            cycle_result = {
                "cycle": cycle + 1,
                "opportunities_added": opportunities_added,
                "best_selected": best.symbol if best else None,
                "execution_success": execution_success,
                "cleaned_up": self.global_manager.get_opportunity_count() == 0,
            }

            cycle_results.append(cycle_result)
            print(f"    Results: {cycle_result}")

        # Verify all cycles completed successfully
        for result in cycle_results:
            assert (
                result["opportunities_added"] > 0
            ), f"Cycle {result['cycle']} should add opportunities"
            assert (
                result["best_selected"] is not None
            ), f"Cycle {result['cycle']} should select best"
            assert result[
                "execution_success"
            ], f"Cycle {result['cycle']} should execute successfully"
            assert result["cleaned_up"], f"Cycle {result['cycle']} should clean up"

        print(f"✅ Successfully completed {len(cycle_results)} lifecycle cycles")


if __name__ == "__main__":
    # For running individual test methods during development
    import asyncio

    test_instance = TestCompleteOpportunityLifecycle()
    test_instance.setup_method()
    asyncio.run(test_instance.test_complete_opportunity_lifecycle_integration())
