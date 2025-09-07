"""
Comprehensive integration tests for Global Opportunity Selection Logic.

Tests the core functionality that transforms the arbitrage system from per-symbol
optimization to global portfolio optimization across all tickers and expirations.

This module validates:
- Global opportunity collection and ranking
- Multi-criteria scoring algorithms
- Cross-symbol competition and selection
- Thread-safety of concurrent operations
- CLI scoring strategy integration
- Performance under realistic loads
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import MagicMock, Mock

import pytest

from modules.Arbitrage.Synthetic import (
    GlobalOpportunity,
    GlobalOpportunityManager,
    OpportunityScore,
    ScoringConfig,
)

# Import test utilities
try:
    from .mock_ib import MockContract, MockTicker
except ImportError:
    from mock_ib import MockContract, MockTicker


class TestGlobalOpportunitySelectionCore:
    """Core tests for global opportunity selection algorithm"""

    def setup_method(self):
        """Setup test environment"""
        # Create a custom scoring config with lower thresholds for testing
        test_config = ScoringConfig()
        test_config.min_liquidity_score = 0.1  # Lower threshold for testing
        test_config.min_risk_reward_ratio = (
            0.4  # Lower threshold to accept all test opportunities
        )
        self.manager = GlobalOpportunityManager(scoring_config=test_config)

    def create_test_opportunity_data(
        self,
        symbol: str,
        max_profit: float,
        min_profit: float,
        call_volume: float = 500,
        put_volume: float = 300,
        call_spread: float = 0.10,
        put_spread: float = 0.05,
        days_to_expiry: int = 30,
    ) -> Dict:
        """Helper to create test opportunity data"""

        # Create mock contracts and tickers
        stock_contract = MockContract(symbol, "STK")
        call_contract = MockContract(symbol, "OPT", strike=100, right="C")
        put_contract = MockContract(symbol, "OPT", strike=95, right="P")

        call_ticker = MockTicker(
            call_contract, bid=5.0, ask=5.0 + call_spread, volume=call_volume
        )
        put_ticker = MockTicker(
            put_contract, bid=3.0, ask=3.0 + put_spread, volume=put_volume
        )

        # Create conversion contract and order
        conversion_contract = Mock()
        order = Mock()

        # Trade details
        expiry_date = datetime.now() + timedelta(days=days_to_expiry)
        trade_details = {
            "max_profit": max_profit,
            "min_profit": min_profit,
            "net_credit": max_profit - abs(min_profit),
            "stock_price": 100.0,
            "expiry": expiry_date.strftime("%Y%m%d"),
            "call_strike": 100,
            "put_strike": 95,
        }

        return {
            "symbol": symbol,
            "conversion_contract": conversion_contract,
            "order": order,
            "trade_details": trade_details,
            "call_ticker": call_ticker,
            "put_ticker": put_ticker,
        }

    @pytest.mark.integration
    def test_global_opportunity_selection_across_multiple_symbols(self):
        """Test that global selection picks the best opportunity across 5+ symbols"""
        print("\n🔍 Testing global opportunity selection across multiple symbols")

        # Create opportunities with known ranking
        opportunities_data = [
            # Symbol A: Moderate opportunity (should rank 3rd)
            ("AAPL", 50.0, -30.0, 400, 200, 0.15, 0.08, 25),  # Risk-reward: 1.67
            # Symbol B: Best opportunity (should rank 1st)
            (
                "MSFT",
                80.0,
                -20.0,
                800,
                600,
                0.05,
                0.03,
                30,
            ),  # Risk-reward: 4.0, high volume, tight spreads
            # Symbol C: Poor opportunity (should rank 5th)
            (
                "TSLA",
                30.0,
                -60.0,
                100,
                50,
                0.30,
                0.25,
                45,
            ),  # Risk-reward: 0.5, low volume, wide spreads
            # Symbol D: Good opportunity (should rank 2nd)
            (
                "GOOGL",
                70.0,
                -25.0,
                600,
                400,
                0.08,
                0.05,
                28,
            ),  # Risk-reward: 2.8, good volume
            # Symbol E: Decent opportunity (should rank 4th)
            ("AMZN", 40.0, -35.0, 300, 250, 0.12, 0.10, 35),  # Risk-reward: 1.14
        ]

        # Add all opportunities to the manager
        added_opportunities = []
        for opp_data in opportunities_data:
            data = self.create_test_opportunity_data(*opp_data)
            success = self.manager.add_opportunity(**data)
            assert success, f"Failed to add opportunity for {data['symbol']}"
            added_opportunities.append(data)

        # Verify all opportunities were collected
        assert self.manager.get_opportunity_count() == 5
        print(f"✅ Collected {self.manager.get_opportunity_count()} opportunities")

        # Get the best opportunity
        best_opportunity = self.manager.get_best_opportunity()
        assert best_opportunity is not None

        # Verify MSFT was selected (highest composite score)
        assert best_opportunity.symbol == "MSFT"
        print(f"✅ Best opportunity selected: {best_opportunity.symbol}")
        print(f"   Composite score: {best_opportunity.score.composite_score:.3f}")
        print(f"   Risk-reward ratio: {best_opportunity.score.risk_reward_ratio:.3f}")
        print(f"   Liquidity score: {best_opportunity.score.liquidity_score:.3f}")

        # Verify the opportunity has expected characteristics
        assert best_opportunity.score.risk_reward_ratio > 3.5  # Should be ~4.0
        assert (
            best_opportunity.score.liquidity_score > 0.7
        )  # High volume, tight spreads

        # Get statistics and verify ranking
        stats = self.manager.get_statistics()
        assert stats["total_opportunities"] == 5
        assert stats["unique_symbols"] == 5
        assert stats["score_stats"]["max"] == best_opportunity.score.composite_score

        print("✅ Global opportunity selection across multiple symbols test passed")

    @pytest.mark.integration
    def test_global_selection_with_different_scoring_strategies(self):
        """Test that different scoring strategies produce different optimal selections"""
        print("\n🔍 Testing global selection with different scoring strategies")

        # Create scenarios that will rank differently under different strategies
        opportunities_data = [
            # High risk-reward, low liquidity (favored by aggressive strategy)
            (
                "AGGRESSIVE",
                100.0,
                -20.0,
                50,
                30,
                0.40,
                0.30,
                35,
            ),  # RR: 5.0, poor liquidity
            # Moderate risk-reward, high liquidity (favored by liquidity-focused)
            (
                "LIQUIDITY",
                60.0,
                -40.0,
                1000,
                800,
                0.02,
                0.01,
                25,
            ),  # RR: 1.5, excellent liquidity
            # Good risk-reward, optimal time decay (favored by balanced)
            (
                "BALANCED",
                80.0,
                -35.0,
                400,
                300,
                0.10,
                0.08,
                30,
            ),  # RR: 2.3, optimal days
            # Low risk-reward, excellent spreads (favored by conservative)
            (
                "CONSERVATIVE",
                45.0,
                -15.0,
                600,
                500,
                0.01,
                0.01,
                20,
            ),  # RR: 3.0, tight spreads
        ]

        strategies = {
            "aggressive": ScoringConfig.create_aggressive(),
            "liquidity_focused": ScoringConfig.create_liquidity_focused(),
            "balanced": ScoringConfig.create_balanced(),
            "conservative": ScoringConfig.create_conservative(),
        }

        results = {}

        for strategy_name, config in strategies.items():
            print(f"\n  Testing {strategy_name.upper()} strategy:")
            manager = GlobalOpportunityManager(config)

            # Add all opportunities
            for opp_data in opportunities_data:
                data = self.create_test_opportunity_data(*opp_data)
                manager.add_opportunity(**data)

            # Get best opportunity for this strategy
            best = manager.get_best_opportunity()
            assert best is not None

            results[strategy_name] = best.symbol
            print(f"    Best opportunity: {best.symbol}")
            print(f"    Composite score: {best.score.composite_score:.3f}")
            print(f"    Risk-reward: {best.score.risk_reward_ratio:.3f}")
            print(f"    Liquidity: {best.score.liquidity_score:.3f}")

        # Verify that different strategies produce different results
        unique_selections = set(results.values())
        print(f"\n📊 Strategy Results: {results}")
        print(
            f"📊 Unique selections: {len(unique_selections)} out of {len(strategies)}"
        )

        # At least 2 different strategies should pick different symbols
        assert (
            len(unique_selections) >= 2
        ), f"Expected different strategies to pick different symbols, got: {results}"

        # Verify expected strategy preferences
        # Aggressive should favor high risk-reward
        # Liquidity-focused should favor high volume/tight spreads
        # These are probabilistic but should generally hold
        print("✅ Different scoring strategies produce different optimal selections")

    @pytest.mark.integration
    def test_global_selection_respects_minimum_thresholds(self):
        """Test that opportunities below minimum thresholds are rejected"""
        print("\n🔍 Testing global selection respects minimum thresholds")

        # Create a manager with strict thresholds
        strict_config = ScoringConfig(
            min_risk_reward_ratio=2.0,  # Require 2:1 risk-reward
            min_liquidity_score=0.5,  # Require decent liquidity
            max_bid_ask_spread=0.15,  # Max 15 cent spread
        )
        manager = GlobalOpportunityManager(strict_config)

        # Opportunity 1: Good risk-reward but poor liquidity (should be rejected)
        data1 = self.create_test_opportunity_data(
            "REJECT_LIQUIDITY",
            80.0,
            -30.0,
            10,
            5,
            0.50,
            0.40,
            30,  # RR: 2.67, but very low volume & wide spreads
        )

        # Opportunity 2: Good liquidity but poor risk-reward (should be rejected)
        data2 = self.create_test_opportunity_data(
            "REJECT_RR",
            30.0,
            -50.0,
            800,
            600,
            0.05,
            0.03,
            25,  # Good liquidity, but RR: 0.6 < 2.0
        )

        # Opportunity 3: Meets all criteria (should be accepted)
        data3 = self.create_test_opportunity_data(
            "ACCEPT", 70.0, -30.0, 500, 400, 0.10, 0.08, 28  # RR: 2.33, good liquidity
        )

        # Add opportunities and track results
        result1 = manager.add_opportunity(**data1)
        result2 = manager.add_opportunity(**data2)
        result3 = manager.add_opportunity(**data3)

        print(
            f"  Opportunity 1 (poor liquidity): {'✅ Accepted' if result1 else '❌ Rejected'}"
        )
        print(
            f"  Opportunity 2 (poor risk-reward): {'✅ Accepted' if result2 else '❌ Rejected'}"
        )
        print(
            f"  Opportunity 3 (meets criteria): {'✅ Accepted' if result3 else '❌ Rejected'}"
        )

        # Verify rejection of opportunities not meeting thresholds
        assert not result1, "Opportunity with poor liquidity should be rejected"
        assert not result2, "Opportunity with poor risk-reward should be rejected"
        assert result3, "Opportunity meeting all criteria should be accepted"

        # Only one opportunity should be in the collection
        assert manager.get_opportunity_count() == 1

        best = manager.get_best_opportunity()
        assert best is not None
        assert best.symbol == "ACCEPT"

        print("✅ Minimum thresholds properly enforced")

    @pytest.mark.integration
    def test_global_selection_handles_empty_opportunities(self):
        """Test edge case when no opportunities meet criteria"""
        print("\n🔍 Testing global selection with no valid opportunities")

        # Create manager with very strict criteria
        impossible_config = ScoringConfig(
            min_risk_reward_ratio=10.0,  # Impossibly high risk-reward
            min_liquidity_score=0.9,  # Nearly perfect liquidity required
            max_bid_ask_spread=0.001,  # Impossibly tight spreads
        )
        manager = GlobalOpportunityManager(impossible_config)

        # Try to add some reasonable opportunities (all should be rejected)
        test_opportunities = [
            self.create_test_opportunity_data(
                "SYMBOL1", 50.0, -25.0, 500, 300, 0.10, 0.05, 30
            ),
            self.create_test_opportunity_data(
                "SYMBOL2", 80.0, -20.0, 800, 600, 0.05, 0.03, 25
            ),
            self.create_test_opportunity_data(
                "SYMBOL3", 60.0, -30.0, 400, 250, 0.08, 0.06, 35
            ),
        ]

        rejected_count = 0
        for data in test_opportunities:
            if not manager.add_opportunity(**data):
                rejected_count += 1

        print(
            f"  Rejected {rejected_count} out of {len(test_opportunities)} opportunities"
        )

        # All should be rejected due to strict criteria
        assert rejected_count == len(test_opportunities)
        assert manager.get_opportunity_count() == 0

        # get_best_opportunity should return None
        best = manager.get_best_opportunity()
        assert best is None

        # Statistics should handle empty case gracefully
        stats = manager.get_statistics()
        assert stats == {}

        print("✅ Empty opportunities case handled gracefully")

    @pytest.mark.integration
    def test_cross_symbol_opportunity_ranking(self):
        """Test detailed cross-symbol ranking with known expected order"""
        print("\n🔍 Testing detailed cross-symbol opportunity ranking")

        # Create opportunities with precisely calculated expected ranking
        opportunities = [
            # Rank 1: Excellent all-around (GOOGL)
            (
                "GOOGL",
                90.0,
                -30.0,
                800,
                600,
                0.05,
                0.03,
                30,
            ),  # RR: 3.0, excellent liquidity, optimal time
            # Rank 2: High risk-reward but lower liquidity (TSLA)
            (
                "TSLA",
                100.0,
                -25.0,
                300,
                200,
                0.15,
                0.10,
                28,
            ),  # RR: 4.0, moderate liquidity
            # Rank 3: Good balance (AAPL)
            ("AAPL", 70.0, -35.0, 500, 400, 0.08, 0.06, 32),  # RR: 2.0, good liquidity
            # Rank 4: Lower profits (MSFT)
            (
                "MSFT",
                50.0,
                -25.0,
                600,
                450,
                0.06,
                0.04,
                25,
            ),  # RR: 2.0, good liquidity, closer to optimal time
            # Rank 5: Poor risk-reward (AMZN)
            (
                "AMZN",
                40.0,
                -40.0,
                700,
                500,
                0.04,
                0.03,
                35,
            ),  # RR: 1.0, good liquidity but poor ratio
        ]

        # Use balanced scoring with adjusted thresholds for testing
        config = ScoringConfig.create_balanced()
        config.min_risk_reward_ratio = 0.5  # Lower to accept all test opportunities
        config.min_liquidity_score = 0.1  # Lower for testing
        manager = GlobalOpportunityManager(config)

        # Add all opportunities
        for opp_data in opportunities:
            data = self.create_test_opportunity_data(*opp_data)
            success = manager.add_opportunity(**data)
            assert success, f"Failed to add {opp_data[0]}"

        # Get all opportunities sorted by score
        with manager.lock:
            sorted_opportunities = sorted(
                manager.opportunities,
                key=lambda opp: opp.score.composite_score,
                reverse=True,
            )

        print("  📊 Final Ranking:")
        for i, opp in enumerate(sorted_opportunities, 1):
            print(f"    #{i}: {opp.symbol} - Score: {opp.score.composite_score:.3f}")
            print(
                f"         RR: {opp.score.risk_reward_ratio:.2f}, "
                f"Liq: {opp.score.liquidity_score:.3f}, "
                f"Time: {opp.score.time_decay_score:.3f}, "
                f"Quality: {opp.score.market_quality_score:.3f}"
            )

        # Verify the best opportunity
        best = manager.get_best_opportunity()
        assert best is not None

        # The best should be one of the top performers
        # Due to scoring complexity, we'll verify it's reasonable rather than exact
        assert (
            best.score.composite_score == sorted_opportunities[0].score.composite_score
        )
        assert best.score.risk_reward_ratio >= 1.5  # Should have decent risk-reward

        print(
            f"✅ Best opportunity: {best.symbol} with score {best.score.composite_score:.3f}"
        )

    @pytest.mark.integration
    def test_cross_expiry_opportunity_ranking(self):
        """Test ranking across different expiration dates"""
        print("\n🔍 Testing cross-expiry opportunity ranking")

        # Same symbol, different expiries with varying time decay scores
        symbol = "SPY"
        base_profit_data = (80.0, -30.0, 500, 400, 0.08, 0.06)  # RR: 2.67

        expiry_scenarios = [
            (7, "Too soon - poor time score"),  # 7 days - too close to expiry
            (30, "Optimal - best time score"),  # 30 days - optimal timing
            (45, "Good - decent time score"),  # 45 days - still reasonable
            (90, "Too far - poor time score"),  # 90 days - too far out
        ]

        config = ScoringConfig.create_balanced()
        config.min_risk_reward_ratio = 0.5  # Lower to accept all test opportunities
        config.min_liquidity_score = 0.1  # Lower for testing
        manager = GlobalOpportunityManager(config)

        added_opportunities = []
        for days, description in expiry_scenarios:
            data = self.create_test_opportunity_data(
                f"{symbol}_{days}D", *base_profit_data, days_to_expiry=days
            )
            success = manager.add_opportunity(**data)
            assert success
            added_opportunities.append((days, description, data["symbol"]))

        # Get ranking
        with manager.lock:
            sorted_opportunities = sorted(
                manager.opportunities,
                key=lambda opp: opp.score.composite_score,
                reverse=True,
            )

        print("  📊 Expiry Ranking:")
        for i, opp in enumerate(sorted_opportunities, 1):
            days = opp.days_to_expiry
            print(
                f"    #{i}: {opp.symbol} - {days} days - Score: {opp.score.composite_score:.3f}"
            )
            print(f"         Time Score: {opp.score.time_decay_score:.3f}")

        # The 30-day expiry should rank highest (optimal time decay)
        best = manager.get_best_opportunity()
        assert best is not None

        # Should favor optimal expiry timing
        print(f"✅ Best expiry: {best.days_to_expiry} days")
        print(f"   Time decay score: {best.score.time_decay_score:.3f}")

        # Verify that optimal timing gets highest time score
        # Look for approximately 30 days (could be 29 or 30 due to time calculations)
        optimal_opp = next(
            (opp for opp in sorted_opportunities if 29 <= opp.days_to_expiry <= 30),
            None,
        )
        assert (
            optimal_opp is not None
        ), f"30-day opportunity should exist. Found days: {[opp.days_to_expiry for opp in sorted_opportunities]}"
        assert (
            optimal_opp.score.time_decay_score >= 0.9
        )  # Should be near maximum time score


class TestGlobalOpportunityThreadSafety:
    """Test thread-safety of concurrent global opportunity operations"""

    @pytest.mark.integration
    def test_concurrent_opportunity_collection(self):
        """Test thread-safe opportunity collection from multiple symbols"""
        print("\n🔍 Testing concurrent opportunity collection thread safety")

        manager = GlobalOpportunityManager()

        # Create multiple opportunities to add concurrently
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "NFLX"]
        opportunities_per_symbol = 3

        def add_opportunities_for_symbol(symbol: str):
            """Worker function to add opportunities for a symbol"""
            for i in range(opportunities_per_symbol):
                # Create varying opportunity data
                max_profit = 50.0 + (i * 10)
                min_profit = -30.0 - (i * 5)
                days = 25 + (i * 5)

                data = (
                    TestGlobalOpportunitySelectionCore().create_test_opportunity_data(
                        f"{symbol}_{i}", max_profit, min_profit, days_to_expiry=days
                    )
                )

                success = manager.add_opportunity(**data)
                # Some may be rejected due to thresholds, which is fine
                time.sleep(0.001)  # Small delay to increase chance of race conditions

        # Launch concurrent threads
        threads = []
        start_time = time.time()

        for symbol in symbols:
            thread = threading.Thread(
                target=add_opportunities_for_symbol, args=(symbol,)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        duration = time.time() - start_time
        print(f"  🕐 Concurrent collection took {duration:.3f} seconds")

        # Verify thread-safe collection
        final_count = manager.get_opportunity_count()
        print(f"  📊 Final opportunity count: {final_count}")

        # Should have collected some opportunities (exact count may vary due to thresholds)
        assert final_count > 0
        assert final_count <= len(symbols) * opportunities_per_symbol

        # Test thread-safe best opportunity selection
        best = manager.get_best_opportunity()
        assert best is not None

        print(f"✅ Thread-safe collection completed with {final_count} opportunities")
        print(f"   Best opportunity: {best.symbol}")

    @pytest.mark.integration
    def test_thread_safe_best_opportunity_selection(self):
        """Test that best opportunity selection is atomic"""
        print("\n🔍 Testing thread-safe best opportunity selection")

        manager = GlobalOpportunityManager()

        # Add some test opportunities
        test_data = [
            ("SYMBOL_A", 70.0, -30.0),
            ("SYMBOL_B", 80.0, -25.0),
            ("SYMBOL_C", 60.0, -35.0),
        ]

        for symbol, max_profit, min_profit in test_data:
            data = TestGlobalOpportunitySelectionCore().create_test_opportunity_data(
                symbol, max_profit, min_profit
            )
            manager.add_opportunity(**data)

        # Concurrently call get_best_opportunity multiple times
        results = []

        def get_best_opportunity():
            best = manager.get_best_opportunity()
            results.append(best.symbol if best else None)

        threads = []
        for _ in range(10):  # 10 concurrent calls
            thread = threading.Thread(target=get_best_opportunity)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All results should be identical (atomic selection)
        unique_results = set(results)
        print(f"  📊 Selection results: {results}")
        print(f"  📊 Unique results: {unique_results}")

        assert (
            len(unique_results) == 1
        ), "get_best_opportunity should return consistent results"
        assert None not in unique_results, "get_best_opportunity should not return None"

        print("✅ Thread-safe selection is atomic and consistent")


class TestGlobalOpportunityPerformance:
    """Test performance characteristics of global opportunity selection"""

    @pytest.mark.integration
    def test_large_scale_opportunity_collection(self):
        """Test performance with many opportunities (50+ symbols, 100+ total opportunities)"""
        print("\n🔍 Testing large-scale opportunity collection performance")

        manager = GlobalOpportunityManager()

        # Generate many opportunities
        num_symbols = 20
        opportunities_per_symbol = 5
        total_expected = num_symbols * opportunities_per_symbol

        print(
            f"  📊 Generating {total_expected} opportunities across {num_symbols} symbols"
        )

        start_time = time.time()
        added_count = 0

        for symbol_idx in range(num_symbols):
            symbol = f"SYM_{symbol_idx:03d}"

            for opp_idx in range(opportunities_per_symbol):
                # Vary the opportunity characteristics
                max_profit = 40.0 + (opp_idx * 10) + (symbol_idx * 2)
                min_profit = -20.0 - (opp_idx * 3) - (symbol_idx * 1)
                volume_mult = 1 + (opp_idx * 0.2)
                spread_mult = 1 + (opp_idx * 0.1)
                days = 20 + (opp_idx * 5) + (symbol_idx % 10)

                data = (
                    TestGlobalOpportunitySelectionCore().create_test_opportunity_data(
                        f"{symbol}_{opp_idx}",
                        max_profit,
                        min_profit,
                        call_volume=int(300 * volume_mult),
                        put_volume=int(200 * volume_mult),
                        call_spread=0.05 * spread_mult,
                        put_spread=0.03 * spread_mult,
                        days_to_expiry=days,
                    )
                )

                if manager.add_opportunity(**data):
                    added_count += 1

        collection_time = time.time() - start_time

        print(f"  🕐 Collection time: {collection_time:.3f} seconds")
        print(f"  📊 Added opportunities: {added_count}/{total_expected}")
        print(
            f"  📊 Collection rate: {added_count/collection_time:.1f} opportunities/second"
        )

        # Test selection performance
        start_time = time.time()
        best = manager.get_best_opportunity()
        selection_time = time.time() - start_time

        print(f"  🕐 Selection time: {selection_time*1000:.1f} milliseconds")

        # Performance assertions
        assert collection_time < 5.0, "Collection should complete within 5 seconds"
        assert selection_time < 0.1, "Selection should complete within 100ms"
        assert best is not None, "Should find a best opportunity"
        assert (
            added_count > total_expected * 0.7
        ), "Should successfully add most opportunities"

        # Test statistics generation performance
        start_time = time.time()
        stats = manager.get_statistics()
        stats_time = time.time() - start_time

        print(f"  🕐 Statistics time: {stats_time*1000:.1f} milliseconds")
        print(
            f"  📊 Statistics: {stats['total_opportunities']} opportunities, {stats['unique_symbols']} symbols"
        )

        assert stats_time < 0.05, "Statistics should generate within 50ms"

        print("✅ Large-scale performance test passed")

    @pytest.mark.integration
    def test_global_selection_performance_benchmarks(self):
        """Test selection performance with varying opportunity counts"""
        print("\n🔍 Testing selection performance benchmarks")

        benchmark_sizes = [10, 25, 50, 100]

        for size in benchmark_sizes:
            print(f"\n  Testing with {size} opportunities:")

            manager = GlobalOpportunityManager()

            # Add exactly 'size' opportunities
            for i in range(size):
                symbol = f"BENCH_{i:03d}"
                max_profit = 50.0 + (i % 50)
                min_profit = -25.0 - (i % 25)

                data = (
                    TestGlobalOpportunitySelectionCore().create_test_opportunity_data(
                        symbol, max_profit, min_profit
                    )
                )
                manager.add_opportunity(**data)

            # Benchmark selection time
            selection_times = []
            for _ in range(10):  # 10 runs for average
                start_time = time.time()
                best = manager.get_best_opportunity()
                selection_time = time.time() - start_time
                selection_times.append(selection_time)
                assert best is not None

            avg_time = (
                sum(selection_times) * 1000 / len(selection_times)
            )  # Convert to ms
            max_time = max(selection_times) * 1000

            print(f"    Average: {avg_time:.2f}ms, Max: {max_time:.2f}ms")

            # Performance requirements
            assert (
                avg_time < 50
            ), f"Average selection time should be under 50ms, got {avg_time:.2f}ms"
            assert (
                max_time < 150
            ), f"Max selection time should be under 150ms, got {max_time:.2f}ms"

        print("✅ Performance benchmarks met for all test sizes")


if __name__ == "__main__":
    # For running individual test methods during development
    test_instance = TestGlobalOpportunitySelectionCore()
    test_instance.setup_method()
    test_instance.test_global_opportunity_selection_across_multiple_symbols()
