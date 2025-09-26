"""
Tests for SFR scoring system.

This module tests the comprehensive scoring mechanism for SFR arbitrage opportunities,
including configuration validation, score calculation, ranking, and logging.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np

from modules.Arbitrage.sfr.scoring import SFRScoringConfig, SFRScoringEngine


class TestSFRScoringConfig(unittest.TestCase):
    """Test SFR scoring configuration"""

    def test_default_config_validation(self):
        """Test that default configuration is valid"""
        config = SFRScoringConfig()
        self.assertTrue(config.validate())

    def test_config_weight_normalization(self):
        """Test weight normalization functionality"""
        config = SFRScoringConfig(
            profit_weight=0.6,
            liquidity_weight=0.3,
            spread_quality_weight=0.2,
            time_decay_weight=0.1,
        )
        self.assertFalse(config.validate())  # Should be invalid (sums to 1.2)

        config.normalize_weights()
        self.assertTrue(config.validate())  # Should be valid after normalization

        # Check that weights sum to approximately 1.0
        total = (
            config.profit_weight
            + config.liquidity_weight
            + config.spread_quality_weight
            + config.time_decay_weight
        )
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_profit_focused_preset(self):
        """Test profit-focused configuration preset"""
        config = SFRScoringConfig.create_profit_focused()

        self.assertTrue(config.validate())
        self.assertGreater(config.profit_weight, 0.6)  # Should prioritize profit
        self.assertEqual(config.profit_weight, 0.70)

    def test_liquidity_focused_preset(self):
        """Test liquidity-focused configuration preset"""
        config = SFRScoringConfig.create_liquidity_focused()

        self.assertTrue(config.validate())
        self.assertEqual(config.liquidity_weight, 0.35)  # High liquidity weight
        self.assertLess(config.max_bid_ask_spread, 20.0)  # Stricter spread requirements

    def test_balanced_preset(self):
        """Test balanced configuration preset (default)"""
        config = SFRScoringConfig.create_balanced()
        default_config = SFRScoringConfig()

        self.assertEqual(config.profit_weight, default_config.profit_weight)
        self.assertEqual(config.liquidity_weight, default_config.liquidity_weight)

    def test_conservative_preset(self):
        """Test conservative configuration preset"""
        config = SFRScoringConfig.create_conservative()

        self.assertTrue(config.validate())
        self.assertGreater(
            config.min_liquidity_volume, 10
        )  # Higher volume requirements
        self.assertLess(config.max_bid_ask_spread, 15.0)  # Stricter spread requirements


class TestSFRScoringEngine(unittest.TestCase):
    """Test SFR scoring engine functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = SFRScoringConfig()
        self.engine = SFRScoringEngine(self.config)

    def test_profit_score_calculation(self):
        """Test profit score calculation"""
        # Test normal case
        score, explanation = self.engine.calculate_profit_score(
            guaranteed_profit=0.50, max_observed_profit=1.00
        )
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)
        self.assertIn("Profit $0.50", explanation)

        # Test edge case: zero max profit
        score, explanation = self.engine.calculate_profit_score(
            guaranteed_profit=0.50, max_observed_profit=0.0
        )
        self.assertEqual(score, 0.0)
        self.assertIn("No positive profits", explanation)

        # Test perfect score case
        score, explanation = self.engine.calculate_profit_score(
            guaranteed_profit=1.00, max_observed_profit=1.00
        )
        self.assertEqual(score, 1.0)

    def test_liquidity_score_calculation(self):
        """Test liquidity score calculation"""
        # Test high volume case
        score, explanation = self.engine.calculate_liquidity_score(
            call_volume=1000, put_volume=1000, stock_volume=10000
        )
        self.assertGreater(score, 0.8)  # Should be high score
        self.assertIn("Vol(C:1000,P:1000)", explanation)

        # Test low volume penalty
        score, explanation = self.engine.calculate_liquidity_score(
            call_volume=2, put_volume=2, stock_volume=1000
        )
        self.assertLess(score, 0.5)  # Should be penalized
        self.assertIn("Below minimum volume", explanation)

        # Test with open interest bonus
        score_without_oi, _ = self.engine.calculate_liquidity_score(
            call_volume=100, put_volume=100, stock_volume=1000
        )
        score_with_oi, explanation = self.engine.calculate_liquidity_score(
            call_volume=100,
            put_volume=100,
            stock_volume=1000,
            call_open_interest=500,
            put_open_interest=500,
        )
        self.assertGreater(score_with_oi, score_without_oi)
        self.assertIn("OI bonus", explanation)

    def test_spread_quality_score_calculation(self):
        """Test spread quality score calculation"""
        # Test tight spreads (good quality)
        score, explanation = self.engine.calculate_spread_quality_score(
            call_bid_ask_spread=0.05,
            put_bid_ask_spread=0.05,
            stock_bid_ask_spread=0.01,
            call_price=5.00,
            put_price=2.00,
            stock_price=100.00,
        )
        self.assertGreater(score, 0.8)  # Should be high quality score

        # Test wide spreads (poor quality)
        score, explanation = self.engine.calculate_spread_quality_score(
            call_bid_ask_spread=2.00,
            put_bid_ask_spread=1.50,
            stock_bid_ask_spread=0.50,
            call_price=5.00,
            put_price=2.00,
            stock_price=100.00,
        )
        self.assertLess(score, 0.3)  # Should be low quality score

        # Test spread exceeding threshold
        score, explanation = self.engine.calculate_spread_quality_score(
            call_bid_ask_spread=25.00,
            put_bid_ask_spread=1.00,
            stock_bid_ask_spread=0.01,
            call_price=5.00,
            put_price=2.00,
            stock_price=100.00,
        )
        self.assertIn("[WIDE:", explanation)  # Should indicate wide spread

    def test_time_decay_score_calculation(self):
        """Test time decay score calculation"""
        # Test optimal days to expiry
        optimal_days = self.config.optimal_days_to_expiry
        score, explanation = self.engine.calculate_time_decay_score(optimal_days)
        self.assertAlmostEqual(score, 1.0, places=2)  # Should be near perfect
        self.assertIn(f"{optimal_days}d (optimal={optimal_days}d)", explanation)

        # Test too short expiry
        score, explanation = self.engine.calculate_time_decay_score(10)
        self.assertLess(score, 0.8)  # Should be penalized
        self.assertIn("Too short:", explanation)

        # Test too long expiry
        score, explanation = self.engine.calculate_time_decay_score(60)
        self.assertLess(score, 1.0)  # Should be penalized
        self.assertIn("Too long:", explanation)

    def test_composite_score_calculation(self):
        """Test comprehensive composite score calculation"""
        # Create mock data for a typical opportunity
        result = self.engine.calculate_composite_score(
            symbol="TEST",
            expiry="20241220",
            guaranteed_profit=0.75,
            max_observed_profit=1.00,
            call_volume=500,
            put_volume=400,
            stock_volume=5000,
            call_bid=4.50,
            call_ask=4.60,
            put_bid=2.30,
            put_ask=2.40,
            stock_bid=99.95,
            stock_ask=100.05,
            days_to_expiry=30,
            call_strike=102.0,
            put_strike=98.0,
            call_open_interest=1000,
            put_open_interest=800,
        )

        # Verify result structure
        self.assertIn("composite_score", result)
        self.assertIn("components", result)
        self.assertIn("explanations", result)
        self.assertIn("details", result)

        # Verify component structure
        components = result["components"]
        self.assertIn("profit", components)
        self.assertIn("liquidity", components)
        self.assertIn("spread_quality", components)
        self.assertIn("time_decay", components)

        # Verify each component has required fields
        for component_name, component in components.items():
            self.assertIn("score", component)
            self.assertIn("weight", component)
            self.assertIn("weighted", component)

        # Verify composite score is reasonable
        composite = result["composite_score"]
        self.assertGreater(composite, 0.0)
        self.assertLessEqual(composite, 1.0)

        # Verify weighted sum equals composite score
        weighted_sum = sum(comp["weighted"] for comp in components.values())
        self.assertAlmostEqual(weighted_sum, composite, places=3)

    def test_opportunity_ranking(self):
        """Test opportunity ranking functionality"""
        # Create multiple opportunity results with different scores
        opportunities = []

        for i, profit in enumerate([0.20, 0.80, 0.50]):  # Different profit levels
            result = self.engine.calculate_composite_score(
                symbol="TEST",
                expiry=f"2024122{i}",
                guaranteed_profit=profit,
                max_observed_profit=1.00,
                call_volume=100 * (i + 1),  # Different volumes
                put_volume=100 * (i + 1),
                stock_volume=1000,
                call_bid=4.50,
                call_ask=4.60,
                put_bid=2.30,
                put_ask=2.40,
                stock_bid=99.95,
                stock_ask=100.05,
                days_to_expiry=30,
                call_strike=102.0,
                put_strike=98.0,
            )
            opportunities.append(result)

        # Rank the opportunities
        ranked = self.engine.rank_opportunities(opportunities)

        # Verify ranking structure
        self.assertEqual(len(ranked), 3)
        for opp in ranked:
            self.assertIn("rank", opp)

        # Verify ranking order (should be sorted by composite score, descending)
        for i in range(len(ranked) - 1):
            self.assertGreaterEqual(
                ranked[i]["composite_score"], ranked[i + 1]["composite_score"]
            )

        # Verify rank numbers are sequential
        for i, opp in enumerate(ranked):
            self.assertEqual(opp["rank"], i + 1)

    def test_scoring_with_different_configurations(self):
        """Test scoring behavior with different configuration presets"""
        # Test profit-focused configuration
        profit_config = SFRScoringConfig.create_profit_focused()
        profit_engine = SFRScoringEngine(profit_config)

        # Test liquidity-focused configuration
        liquidity_config = SFRScoringConfig.create_liquidity_focused()
        liquidity_engine = SFRScoringEngine(liquidity_config)

        # Score the same opportunity with both configurations
        common_params = {
            "symbol": "TEST",
            "expiry": "20241220",
            "guaranteed_profit": 0.50,
            "max_observed_profit": 1.00,
            "call_volume": 100,  # Moderate volume
            "put_volume": 100,
            "stock_volume": 1000,
            "call_bid": 4.50,
            "call_ask": 4.70,  # Wider spread
            "put_bid": 2.30,
            "put_ask": 2.50,
            "stock_bid": 99.95,
            "stock_ask": 100.05,
            "days_to_expiry": 30,
            "call_strike": 102.0,
            "put_strike": 98.0,
        }

        profit_result = profit_engine.calculate_composite_score(**common_params)
        liquidity_result = liquidity_engine.calculate_composite_score(**common_params)

        # Profit-focused should weight profit component more heavily
        profit_component_profit = profit_result["components"]["profit"]["weighted"]
        liquidity_component_profit = liquidity_result["components"]["profit"][
            "weighted"
        ]

        # Due to higher profit weight, profit-focused should have higher profit contribution
        # (assuming same raw profit score)
        self.assertGreater(
            profit_result["components"]["profit"]["weight"],
            liquidity_result["components"]["profit"]["weight"],
        )

        # Liquidity-focused should weight liquidity component more heavily
        self.assertGreater(
            liquidity_result["components"]["liquidity"]["weight"],
            profit_result["components"]["liquidity"]["weight"],
        )

    @patch("modules.Arbitrage.sfr.scoring.logger")
    def test_logging_configuration(self, mock_logger):
        """Test that logging works correctly with different configurations"""
        # Test with detailed logging enabled
        config = SFRScoringConfig(
            enable_detailed_logging=True, log_score_components=True, log_threshold=0.0
        )
        engine = SFRScoringEngine(config)

        # Calculate a score that should trigger logging
        result = engine.calculate_composite_score(
            symbol="TEST",
            expiry="20241220",
            guaranteed_profit=0.50,
            max_observed_profit=1.00,
            call_volume=100,
            put_volume=100,
            stock_volume=1000,
            call_bid=4.50,
            call_ask=4.60,
            put_bid=2.30,
            put_ask=2.40,
            stock_bid=99.95,
            stock_ask=100.05,
            days_to_expiry=30,
            call_strike=102.0,
            put_strike=98.0,
        )

        # Verify that logging was called
        self.assertTrue(mock_logger.info.called)

        # Check for specific log message patterns
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        scoring_logs = [log for log in log_calls if "[SFR SCORING]" in log]
        self.assertGreater(len(scoring_logs), 0)

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling"""
        # Test with zero/negative values
        result = self.engine.calculate_composite_score(
            symbol="TEST",
            expiry="20241220",
            guaranteed_profit=0.0,  # Zero profit
            max_observed_profit=0.0,  # Zero max profit
            call_volume=0,  # Zero volume
            put_volume=0,
            stock_volume=0,
            call_bid=0.0,
            call_ask=0.0,
            put_bid=0.0,
            put_ask=0.0,
            stock_bid=0.0,
            stock_ask=0.0,
            days_to_expiry=0,  # Zero days
            call_strike=100.0,
            put_strike=100.0,
        )

        # Should handle gracefully without crashing
        self.assertIsInstance(result, dict)
        self.assertIn("composite_score", result)

        # Test with very large values
        result = self.engine.calculate_composite_score(
            symbol="TEST",
            expiry="20241220",
            guaranteed_profit=10000.0,  # Very large profit
            max_observed_profit=10000.0,
            call_volume=1000000,  # Very large volume
            put_volume=1000000,
            stock_volume=1000000,
            call_bid=100.0,
            call_ask=200.0,  # Very wide spread
            put_bid=50.0,
            put_ask=150.0,
            stock_bid=1000.0,
            stock_ask=1001.0,
            days_to_expiry=365,  # Very long expiry
            call_strike=1500.0,
            put_strike=500.0,
        )

        # Should handle gracefully
        self.assertIsInstance(result, dict)
        self.assertLessEqual(result["composite_score"], 1.0)  # Should be normalized


if __name__ == "__main__":
    unittest.main()
