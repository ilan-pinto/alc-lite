"""
Dedicated tests for the scoring calculation algorithms in GlobalOpportunityManager.

Tests the mathematical correctness and edge case handling of:
- Liquidity score calculation (volume + spread components)
- Time decay score calculation (optimal days to expiry)
- Market quality score calculation (spread + credit quality)
- Composite score calculation (weighted combination)
- Risk-reward ratio edge cases (zero, negative, infinite)

These tests ensure the scoring algorithms produce mathematically correct
and consistent results that drive optimal opportunity selection.
"""

import math
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from modules.Arbitrage.Synthetic import GlobalOpportunityManager, ScoringConfig

# Import test utilities
try:
    from .mock_ib import MockContract, MockTicker
except ImportError:
    from mock_ib import MockContract, MockTicker


class TestLiquidityScoring:
    """Test liquidity score calculation algorithms"""

    def setup_method(self):
        """Setup test environment"""
        self.manager = GlobalOpportunityManager()

    @pytest.mark.unit
    def test_liquidity_score_high_volume_tight_spreads(self):
        """Test liquidity scoring with high volume and tight spreads (optimal case)"""
        # High volume: 1000+ shares, tight spreads: < 0.05
        liquidity_score = self.manager.calculate_liquidity_score(
            call_volume=800, put_volume=600, call_spread=0.03, put_spread=0.02
        )

        # Volume component: (800 + 600) / 1000 = 1.4 -> min(1.0, 1.4) = 1.0
        # Avg spread: (0.03 + 0.02) / 2 = 0.025
        # Spread component: max(0, 1.0 - (0.025 / 20.0)) = max(0, 0.99875) = 0.99875
        # Combined: (1.0 * 0.6) + (0.99875 * 0.4) = 0.6 + 0.3995 = 0.9995

        expected_volume_score = 1.0  # Capped at 1.0
        expected_spread_score = 1.0 - (0.025 / 20.0)  # Very tight spreads
        expected_liquidity = (expected_volume_score * 0.6) + (
            expected_spread_score * 0.4
        )

        assert abs(liquidity_score - expected_liquidity) < 0.001
        assert liquidity_score > 0.95  # Should be very high
        print(f"✅ High volume/tight spreads score: {liquidity_score:.4f}")

    @pytest.mark.unit
    def test_liquidity_score_low_volume_wide_spreads(self):
        """Test liquidity scoring with low volume and wide spreads (poor case)"""
        # Low volume: 50 shares total, wide spreads: > 0.30
        liquidity_score = self.manager.calculate_liquidity_score(
            call_volume=30, put_volume=20, call_spread=0.40, put_spread=0.30
        )

        # Volume component: (30 + 20) / 1000 = 0.05
        # Avg spread: (0.40 + 0.30) / 2 = 0.35
        # Spread component: max(0, 1.0 - (0.35 / 20.0)) = max(0, 0.9825) = 0.9825
        # Combined: (0.05 * 0.6) + (0.9825 * 0.4) = 0.03 + 0.393 = 0.423

        expected_volume_score = 0.05
        expected_spread_score = 1.0 - (0.35 / 20.0)
        expected_liquidity = (expected_volume_score * 0.6) + (
            expected_spread_score * 0.4
        )

        assert abs(liquidity_score - expected_liquidity) < 0.001
        assert liquidity_score < 0.5  # Should be poor
        print(f"✅ Low volume/wide spreads score: {liquidity_score:.4f}")

    @pytest.mark.unit
    def test_liquidity_score_extreme_spreads(self):
        """Test liquidity scoring with extremely wide spreads"""
        # Spreads wider than max_bid_ask_spread (20.0)
        liquidity_score = self.manager.calculate_liquidity_score(
            call_volume=500, put_volume=300, call_spread=25.0, put_spread=30.0
        )

        # Volume component: (500 + 300) / 1000 = 0.8
        # Avg spread: (25.0 + 30.0) / 2 = 27.5
        # Spread component: max(0, 1.0 - (27.5 / 20.0)) = max(0, -0.375) = 0.0
        # Combined: (0.8 * 0.6) + (0.0 * 0.4) = 0.48 + 0.0 = 0.48

        expected_volume_score = 0.8
        expected_spread_score = 0.0  # Clamped to 0 due to very wide spreads
        expected_liquidity = (expected_volume_score * 0.6) + (
            expected_spread_score * 0.4
        )

        assert abs(liquidity_score - expected_liquidity) < 0.001
        assert abs(liquidity_score - 0.48) < 0.001
        print(f"✅ Extreme spreads score: {liquidity_score:.4f}")

    @pytest.mark.unit
    def test_liquidity_score_zero_volume(self):
        """Test liquidity scoring with zero volume"""
        liquidity_score = self.manager.calculate_liquidity_score(
            call_volume=0, put_volume=0, call_spread=0.05, put_spread=0.03
        )

        # Volume component: (0 + 0) / 1000 = 0.0
        # Spread component still calculated normally
        # Should result in low but non-zero score due to good spreads

        assert liquidity_score >= 0.0
        assert liquidity_score < 0.5  # Should be low due to no volume
        print(f"✅ Zero volume score: {liquidity_score:.4f}")


class TestTimeDecayScoring:
    """Test time decay score calculation algorithms"""

    def setup_method(self):
        """Setup test environment"""
        # Use default config with optimal_days_to_expiry = 30
        self.manager = GlobalOpportunityManager()

    @pytest.mark.unit
    def test_time_decay_score_optimal_days(self):
        """Test time decay scoring at optimal days to expiry"""
        optimal_days = self.manager.scoring_config.optimal_days_to_expiry  # 30 days

        score = self.manager.calculate_time_decay_score(optimal_days)

        # At optimal days, score should be 1.0
        assert abs(score - 1.0) < 0.001
        print(f"✅ Optimal days ({optimal_days}) score: {score:.4f}")

    @pytest.mark.unit
    def test_time_decay_score_before_optimal(self):
        """Test time decay scoring before optimal days (linear increase)"""
        optimal_days = self.manager.scoring_config.optimal_days_to_expiry  # 30

        # Test various days before optimal
        test_cases = [
            (15, 15 / 30),  # Half of optimal = 0.5 score
            (20, 20 / 30),  # 2/3 of optimal = 0.667 score
            (25, 25 / 30),  # 5/6 of optimal = 0.833 score
        ]

        for days, expected_score in test_cases:
            score = self.manager.calculate_time_decay_score(days)
            assert abs(score - expected_score) < 0.001
            print(f"✅ {days} days score: {score:.4f} (expected: {expected_score:.4f})")

    @pytest.mark.unit
    def test_time_decay_score_after_optimal(self):
        """Test time decay scoring after optimal days (decreasing with penalty)"""
        optimal_days = self.manager.scoring_config.optimal_days_to_expiry  # 30

        # Test various days after optimal
        # Formula: max(0.1, 1.0 - (excess_days / (optimal_days * 2)))
        test_cases = [
            (35, 1.0 - (5 / 60)),  # 35 days: excess=5, penalty=5/60=0.083, score=0.917
            (45, 1.0 - (15 / 60)),  # 45 days: excess=15, penalty=15/60=0.25, score=0.75
            (60, 1.0 - (30 / 60)),  # 60 days: excess=30, penalty=30/60=0.5, score=0.5
            (
                90,
                1.0 - (60 / 60),
            ),  # 90 days: excess=60, penalty=60/60=1.0, score=0.1 (clamped)
        ]

        for days, expected_score in test_cases:
            score = self.manager.calculate_time_decay_score(days)
            expected_clamped = max(0.1, expected_score)  # Minimum score is 0.1
            assert abs(score - expected_clamped) < 0.001
            print(
                f"✅ {days} days score: {score:.4f} (expected: {expected_clamped:.4f})"
            )

    @pytest.mark.unit
    def test_time_decay_score_edge_cases(self):
        """Test time decay scoring edge cases"""
        # Zero or negative days
        assert self.manager.calculate_time_decay_score(0) == 0.0
        assert self.manager.calculate_time_decay_score(-5) == 0.0

        # Very far out (should hit minimum floor)
        very_far_score = self.manager.calculate_time_decay_score(200)
        assert very_far_score == 0.1  # Minimum floor

        print("✅ Edge cases handled correctly")

    @pytest.mark.unit
    def test_time_decay_with_custom_optimal_days(self):
        """Test time decay scoring with custom optimal days configuration"""
        # Create manager with different optimal days
        custom_config = ScoringConfig(optimal_days_to_expiry=45)
        custom_manager = GlobalOpportunityManager(custom_config)

        # Test that optimal score is at 45 days now
        score_45 = custom_manager.calculate_time_decay_score(45)
        score_30 = custom_manager.calculate_time_decay_score(30)
        score_60 = custom_manager.calculate_time_decay_score(60)

        assert abs(score_45 - 1.0) < 0.001  # Perfect score at 45 days
        assert score_30 < score_45  # 30 days should be less than optimal
        assert score_60 < score_45  # 60 days should be less than optimal

        print(
            f"✅ Custom optimal days (45): score_30={score_30:.3f}, score_45={score_45:.3f}, score_60={score_60:.3f}"
        )


class TestMarketQualityScoring:
    """Test market quality score calculation algorithms"""

    def setup_method(self):
        """Setup test environment"""
        self.manager = GlobalOpportunityManager()

    @pytest.mark.unit
    def test_market_quality_excellent_conditions(self):
        """Test market quality scoring with excellent conditions"""
        trade_details = {
            "net_credit": 15.0,  # Good positive credit
            "stock_price": 100.0,  # $100 stock
            # Credit quality: min(1.0, max(0.0, 15.0 / (100.0 * 0.1))) = min(1.0, 1.5) = 1.0
        }

        call_spread = 0.02  # Very tight spread
        put_spread = 0.01  # Very tight spread
        # Avg spread: 0.015
        # Spread quality: max(0.0, 1.0 - (0.015 / 20.0)) = 0.9925

        quality_score = self.manager.calculate_market_quality_score(
            trade_details, call_spread, put_spread
        )

        expected_spread_quality = 1.0 - (0.015 / 20.0)
        expected_credit_quality = 1.0  # Capped at 1.0
        expected_quality = (expected_spread_quality * 0.7) + (
            expected_credit_quality * 0.3
        )

        assert abs(quality_score - expected_quality) < 0.001
        assert quality_score > 0.9  # Should be very high
        print(f"✅ Excellent conditions score: {quality_score:.4f}")

    @pytest.mark.unit
    def test_market_quality_poor_conditions(self):
        """Test market quality scoring with poor conditions"""
        trade_details = {
            "net_credit": -5.0,  # Negative credit (poor)
            "stock_price": 50.0,  # $50 stock
            # Credit quality: min(1.0, max(0.0, -5.0 / (50.0 * 0.1))) = max(0.0, -1.0) = 0.0
        }

        call_spread = 1.50  # Wide spread
        put_spread = 1.00  # Wide spread
        # Avg spread: 1.25
        # Spread quality: max(0.0, 1.0 - (1.25 / 20.0)) = 0.9375

        quality_score = self.manager.calculate_market_quality_score(
            trade_details, call_spread, put_spread
        )

        expected_spread_quality = 1.0 - (1.25 / 20.0)
        expected_credit_quality = 0.0  # Negative credit
        expected_quality = (expected_spread_quality * 0.7) + (
            expected_credit_quality * 0.3
        )

        assert abs(quality_score - expected_quality) < 0.001
        assert quality_score < 0.7  # Should be poor due to negative credit
        print(f"✅ Poor conditions score: {quality_score:.4f}")

    @pytest.mark.unit
    def test_market_quality_extreme_spreads(self):
        """Test market quality scoring with extremely wide spreads"""
        trade_details = {
            "net_credit": 10.0,
            "stock_price": 100.0,
        }

        # Spreads wider than max threshold
        call_spread = 25.0
        put_spread = 30.0
        # Avg spread: 27.5 > 20.0 (max_bid_ask_spread)
        # Spread quality: max(0.0, 1.0 - (27.5 / 20.0)) = max(0.0, -0.375) = 0.0

        quality_score = self.manager.calculate_market_quality_score(
            trade_details, call_spread, put_spread
        )

        # Credit quality should still be good
        expected_credit_quality = min(1.0, 10.0 / 10.0)  # = 1.0
        expected_spread_quality = 0.0  # Extremely wide spreads
        expected_quality = (0.0 * 0.7) + (1.0 * 0.3)  # = 0.3

        assert abs(quality_score - 0.3) < 0.001
        print(f"✅ Extreme spreads score: {quality_score:.4f}")


class TestCompositeScoreCalculation:
    """Test composite score calculation (weighted combination)"""

    def setup_method(self):
        """Setup test environment"""
        self.manager = GlobalOpportunityManager()

    @pytest.mark.unit
    def test_composite_score_calculation_balanced(self):
        """Test composite score with balanced configuration"""
        # Use known values to test weighted combination
        trade_details = {
            "max_profit": 80.0,
            "min_profit": -40.0,  # Risk-reward ratio: 80/40 = 2.0
            "net_credit": 10.0,
            "stock_price": 100.0,
            "expiry": "20240330",
        }

        call_volume = 500
        put_volume = 300
        call_spread = 0.10
        put_spread = 0.05
        days_to_expiry = 30  # Optimal

        opportunity_score = self.manager.calculate_opportunity_score(
            trade_details,
            call_volume,
            put_volume,
            call_spread,
            put_spread,
            days_to_expiry,
        )

        # Verify individual components
        assert abs(opportunity_score.risk_reward_ratio - 2.0) < 0.001

        # Liquidity should be reasonable with 800 total volume and 0.075 avg spread
        assert opportunity_score.liquidity_score > 0.5

        # Time decay should be 1.0 (optimal 30 days)
        assert abs(opportunity_score.time_decay_score - 1.0) < 0.001

        # Market quality should be good with tight spreads and positive credit
        assert opportunity_score.market_quality_score > 0.8

        # Composite score should be weighted combination
        config = self.manager.scoring_config
        expected_composite = (
            opportunity_score.risk_reward_ratio * config.risk_reward_weight
            + opportunity_score.liquidity_score * config.liquidity_weight
            + opportunity_score.time_decay_score * config.time_decay_weight
            + opportunity_score.market_quality_score * config.market_quality_weight
        )

        assert abs(opportunity_score.composite_score - expected_composite) < 0.001
        print(f"✅ Composite score: {opportunity_score.composite_score:.4f}")
        print(
            f"   Components - RR: {opportunity_score.risk_reward_ratio:.3f}, "
            f"Liq: {opportunity_score.liquidity_score:.3f}, "
            f"Time: {opportunity_score.time_decay_score:.3f}, "
            f"Quality: {opportunity_score.market_quality_score:.3f}"
        )

    @pytest.mark.unit
    def test_composite_score_different_strategies(self):
        """Test that different scoring strategies produce different composite scores"""
        # Same opportunity data
        trade_details = {
            "max_profit": 60.0,
            "min_profit": -30.0,  # Risk-reward ratio: 2.0
            "net_credit": 8.0,
            "stock_price": 100.0,
            "expiry": "20240330",
        }

        # Moderate liquidity, optimal time
        call_volume = 300
        put_volume = 200
        call_spread = 0.15
        put_spread = 0.08
        days_to_expiry = 30

        strategies = {
            "conservative": ScoringConfig.create_conservative(),
            "aggressive": ScoringConfig.create_aggressive(),
            "balanced": ScoringConfig.create_balanced(),
            "liquidity_focused": ScoringConfig.create_liquidity_focused(),
        }

        scores = {}
        for name, config in strategies.items():
            manager = GlobalOpportunityManager(config)
            score = manager.calculate_opportunity_score(
                trade_details,
                call_volume,
                put_volume,
                call_spread,
                put_spread,
                days_to_expiry,
            )
            scores[name] = score.composite_score
            print(f"  {name}: {score.composite_score:.4f}")

        # Scores should be different (different weightings)
        unique_scores = set(scores.values())
        assert (
            len(unique_scores) > 1
        ), "Different strategies should produce different scores"

        # Aggressive should generally favor risk-reward more heavily
        # Liquidity-focused should be more conservative due to moderate liquidity
        print(f"✅ Strategy scores: {scores}")


class TestRiskRewardRatioEdgeCases:
    """Test risk-reward ratio calculation edge cases"""

    def setup_method(self):
        """Setup test environment"""
        self.manager = GlobalOpportunityManager()

    @pytest.mark.unit
    def test_risk_reward_ratio_zero_loss(self):
        """Test risk-reward ratio when min_profit is zero (no loss scenario)"""
        trade_details = {
            "max_profit": 50.0,
            "min_profit": 0.0,  # No loss possible
            "net_credit": 25.0,
            "stock_price": 100.0,
            "expiry": "20240330",
        }

        score = self.manager.calculate_opportunity_score(
            trade_details, 500, 300, 0.10, 0.05, 30
        )

        # When min_profit is 0, risk_reward_ratio should be 0 (formula handles this)
        assert score.risk_reward_ratio == 0.0
        print(f"✅ Zero loss scenario: risk_reward_ratio = {score.risk_reward_ratio}")

    @pytest.mark.unit
    def test_risk_reward_ratio_positive_min_profit(self):
        """Test risk-reward ratio when min_profit is positive (guaranteed profit)"""
        trade_details = {
            "max_profit": 80.0,
            "min_profit": 20.0,  # Guaranteed minimum profit
            "net_credit": 50.0,
            "stock_price": 100.0,
            "expiry": "20240330",
        }

        score = self.manager.calculate_opportunity_score(
            trade_details, 500, 300, 0.10, 0.05, 30
        )

        # Risk-reward ratio: 80 / abs(20) = 4.0
        expected_ratio = 80.0 / 20.0
        assert abs(score.risk_reward_ratio - expected_ratio) < 0.001
        print(
            f"✅ Guaranteed profit scenario: risk_reward_ratio = {score.risk_reward_ratio}"
        )

    @pytest.mark.unit
    def test_risk_reward_ratio_very_small_loss(self):
        """Test risk-reward ratio with very small loss (near-zero denominator)"""
        trade_details = {
            "max_profit": 100.0,
            "min_profit": -0.01,  # Very small loss
            "net_credit": 50.0,
            "stock_price": 100.0,
            "expiry": "20240330",
        }

        score = self.manager.calculate_opportunity_score(
            trade_details, 500, 300, 0.10, 0.05, 30
        )

        # Risk-reward ratio: 100 / 0.01 = 10,000 (very high)
        expected_ratio = 100.0 / 0.01
        assert (
            abs(score.risk_reward_ratio - expected_ratio) < 0.1
        )  # Allow small floating point error
        print(
            f"✅ Very small loss scenario: risk_reward_ratio = {score.risk_reward_ratio}"
        )

    @pytest.mark.unit
    def test_risk_reward_ratio_zero_profit(self):
        """Test risk-reward ratio when max_profit is zero"""
        trade_details = {
            "max_profit": 0.0,  # No profit possible
            "min_profit": -30.0,
            "net_credit": -15.0,
            "stock_price": 100.0,
            "expiry": "20240330",
        }

        score = self.manager.calculate_opportunity_score(
            trade_details, 500, 300, 0.10, 0.05, 30
        )

        # Risk-reward ratio: 0 / 30 = 0
        assert score.risk_reward_ratio == 0.0
        print(f"✅ Zero profit scenario: risk_reward_ratio = {score.risk_reward_ratio}")


class TestScoringConfigurationValidation:
    """Test scoring configuration validation and normalization"""

    @pytest.mark.unit
    def test_scoring_config_validation_valid_weights(self):
        """Test validation with valid weights that sum to 1.0"""
        config = ScoringConfig(
            risk_reward_weight=0.3,
            liquidity_weight=0.3,
            time_decay_weight=0.2,
            market_quality_weight=0.2,
        )

        assert config.validate() == True
        print("✅ Valid weights pass validation")

    @pytest.mark.unit
    def test_scoring_config_validation_invalid_weights(self):
        """Test validation with invalid weights that don't sum to 1.0"""
        config = ScoringConfig(
            risk_reward_weight=0.5,
            liquidity_weight=0.3,
            time_decay_weight=0.3,
            market_quality_weight=0.2,  # Sum = 1.3, not 1.0
        )

        assert config.validate() == False
        print("✅ Invalid weights fail validation")

    @pytest.mark.unit
    def test_scoring_config_normalization(self):
        """Test automatic weight normalization"""
        config = ScoringConfig(
            risk_reward_weight=0.5,
            liquidity_weight=0.3,
            time_decay_weight=0.3,
            market_quality_weight=0.2,  # Sum = 1.3
        )

        # Before normalization
        assert not config.validate()

        # Normalize
        config.normalize_weights()

        # After normalization should be valid
        assert config.validate()

        # Check that weights are proportionally correct
        # Original sum was 1.3, so each weight should be divided by 1.3
        assert abs(config.risk_reward_weight - (0.5 / 1.3)) < 0.001
        assert abs(config.liquidity_weight - (0.3 / 1.3)) < 0.001
        assert abs(config.time_decay_weight - (0.3 / 1.3)) < 0.001
        assert abs(config.market_quality_weight - (0.2 / 1.3)) < 0.001

        print("✅ Weight normalization works correctly")

    @pytest.mark.unit
    def test_preset_strategy_configurations(self):
        """Test that all preset strategy configurations are valid"""
        strategies = {
            "conservative": ScoringConfig.create_conservative(),
            "aggressive": ScoringConfig.create_aggressive(),
            "balanced": ScoringConfig.create_balanced(),
            "liquidity_focused": ScoringConfig.create_liquidity_focused(),
        }

        for name, config in strategies.items():
            assert config.validate(), f"{name} strategy configuration is invalid"

            # Verify weights sum to 1.0
            total_weight = (
                config.risk_reward_weight
                + config.liquidity_weight
                + config.time_decay_weight
                + config.market_quality_weight
            )
            assert abs(total_weight - 1.0) < 0.01, f"{name} weights don't sum to 1.0"

            print(f"✅ {name.capitalize()} strategy configuration is valid")


if __name__ == "__main__":
    # For running individual test methods during development
    test_instance = TestLiquidityScoring()
    test_instance.setup_method()
    test_instance.test_liquidity_score_high_volume_tight_spreads()
