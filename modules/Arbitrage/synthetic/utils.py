"""
Utility functions for Synthetic arbitrage strategy.

This module contains:
- Helper functions for calculations
- Debugging and diagnostic utilities
- Testing helper functions
- Vectorized calculation utilities
"""

from ..common import get_logger

# This will be imported from data_collector after migration
contract_ticker = {}

logger = get_logger()


def get_symbol_contract_count(symbol):
    """Get count of contracts for a specific symbol"""
    return sum(1 for k in contract_ticker.keys() if k[0] == symbol)


def debug_contract_ticker_state():
    """Debug helper to show contract_ticker state by symbol"""
    by_symbol = {}
    for (symbol, conId), _ in contract_ticker.items():
        if symbol not in by_symbol:
            by_symbol[symbol] = 0
        by_symbol[symbol] += 1
    logger.debug(f"Contract ticker state: {by_symbol}")
    return by_symbol


def test_global_opportunity_scoring():
    """Test function to verify global opportunity scoring works correctly"""
    # Import here to avoid circular imports during migration
    from .global_opportunity_manager import GlobalOpportunityManager
    from .scoring import ScoringConfig

    logger = get_logger()

    # Test different scoring configurations
    configs = {
        "conservative": ScoringConfig.create_conservative(),
        "aggressive": ScoringConfig.create_aggressive(),
        "balanced": ScoringConfig.create_balanced(),
        "liquidity_focused": ScoringConfig.create_liquidity_focused(),
    }

    logger.info("=== Testing Global Opportunity Scoring Configurations ===")

    for config_name, config in configs.items():
        logger.info(f"\n{config_name.upper()} Configuration:")
        logger.info(f"  Risk-Reward Weight: {config.risk_reward_weight:.2f}")
        logger.info(f"  Liquidity Weight: {config.liquidity_weight:.2f}")
        logger.info(f"  Time Decay Weight: {config.time_decay_weight:.2f}")
        logger.info(f"  Market Quality Weight: {config.market_quality_weight:.2f}")
        logger.info(f"  Min Risk-Reward Ratio: {config.min_risk_reward_ratio:.2f}")
        logger.info(f"  Weights Valid: {config.validate()}")

        # Test scoring calculation
        manager = GlobalOpportunityManager(config)

        # Create test trade details
        test_trade_details = {
            "max_profit": 100.0,
            "min_profit": -50.0,
            "net_credit": 25.0,
            "stock_price": 150.0,
            "expiry": "20240315",
        }

        # Test scoring with sample data
        score = manager.calculate_opportunity_score(
            test_trade_details,
            call_volume=500,
            put_volume=300,
            call_spread=2.5,
            put_spread=1.8,
            days_to_expiry=25,
        )

        logger.info(f"  Sample Score - Composite: {score.composite_score:.3f}")
        logger.info(f"    Risk-Reward: {score.risk_reward_ratio:.3f}")
        logger.info(f"    Liquidity: {score.liquidity_score:.3f}")
        logger.info(f"    Time Decay: {score.time_decay_score:.3f}")
        logger.info(f"    Market Quality: {score.market_quality_score:.3f}")

    logger.info("\n=== Global Opportunity Scoring Test Complete ===")


def create_syn_with_config(config_type: str = "balanced", **kwargs):
    """Helper function to create Syn instance with specific configuration"""
    # Import here to avoid circular imports during migration
    from .scoring import ScoringConfig
    from .strategy import Syn

    config_map = {
        "conservative": ScoringConfig.create_conservative(),
        "aggressive": ScoringConfig.create_aggressive(),
        "balanced": ScoringConfig.create_balanced(),
        "liquidity_focused": ScoringConfig.create_liquidity_focused(),
    }

    scoring_config = config_map.get(config_type, ScoringConfig.create_balanced())
    return Syn(scoring_config=scoring_config, **kwargs)
