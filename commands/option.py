import asyncio
import os

import logging
import pandas as pd
from ib_async import *

from modules.Arbitrage.SFR import SFR
from modules.Arbitrage.Synthetic import ScoringConfig, Syn
from modules.finviz_scraper import scrape_tickers_from_finviz

logger = logging.getLogger(__name__)


class OptionScan:

    def _create_scoring_config(
        self,
        scoring_strategy="balanced",
        risk_reward_weight=None,
        liquidity_weight=None,
        time_decay_weight=None,
        market_quality_weight=None,
        min_risk_reward=None,
        min_liquidity=None,
        max_bid_ask_spread=None,
        optimal_days_expiry=None,
    ) -> ScoringConfig:
        """Create scoring configuration based on user inputs with validation"""

        # Check if user provided custom weights
        custom_weights = [
            risk_reward_weight,
            liquidity_weight,
            time_decay_weight,
            market_quality_weight,
        ]
        has_custom_weights = any(w is not None for w in custom_weights)

        if has_custom_weights:
            # Validate that all weights are provided if any are provided
            if not all(w is not None for w in custom_weights):
                raise ValueError(
                    "If providing custom weights, all four weights must be specified: "
                    "--risk-reward-weight, --liquidity-weight, --time-decay-weight, --market-quality-weight"
                )

            # Validate weight values
            for i, weight in enumerate(custom_weights):
                if not (0.0 <= weight <= 1.0):
                    weight_names = [
                        "risk-reward",
                        "liquidity",
                        "time-decay",
                        "market-quality",
                    ]
                    raise ValueError(
                        f"{weight_names[i]} weight must be between 0.0 and 1.0, got {weight}"
                    )

            # Validate weights sum to approximately 1.0
            total_weight = sum(custom_weights)
            if not (0.99 <= total_weight <= 1.01):
                logger.warning(
                    f"Custom weights sum to {total_weight:.3f}, not 1.0. "
                    f"Weights will be normalized automatically."
                )

            # Create custom configuration
            config = ScoringConfig(
                risk_reward_weight=risk_reward_weight,
                liquidity_weight=liquidity_weight,
                time_decay_weight=time_decay_weight,
                market_quality_weight=market_quality_weight,
            )

            # Apply custom thresholds if provided
            if min_risk_reward is not None:
                config.min_risk_reward_ratio = min_risk_reward
            if min_liquidity is not None:
                config.min_liquidity_score = min_liquidity
            if max_bid_ask_spread is not None:
                config.max_bid_ask_spread = max_bid_ask_spread
            if optimal_days_expiry is not None:
                config.optimal_days_to_expiry = optimal_days_expiry

            logger.info(
                f"Using CUSTOM scoring configuration with weights: "
                f"risk-reward={risk_reward_weight:.2f}, liquidity={liquidity_weight:.2f}, "
                f"time-decay={time_decay_weight:.2f}, market-quality={market_quality_weight:.2f}"
            )

        else:
            # Use pre-defined strategy
            strategy_map = {
                "conservative": ScoringConfig.create_conservative,
                "aggressive": ScoringConfig.create_aggressive,
                "balanced": ScoringConfig.create_balanced,
                "liquidity-focused": ScoringConfig.create_liquidity_focused,
            }

            if scoring_strategy not in strategy_map:
                raise ValueError(f"Unknown scoring strategy: {scoring_strategy}")

            config = strategy_map[scoring_strategy]()

            # Apply custom thresholds if provided, overriding strategy defaults
            if min_risk_reward is not None:
                config.min_risk_reward_ratio = min_risk_reward
                logger.info(f"Overriding min risk-reward ratio: {min_risk_reward}")
            if min_liquidity is not None:
                config.min_liquidity_score = min_liquidity
                logger.info(f"Overriding min liquidity score: {min_liquidity}")
            if max_bid_ask_spread is not None:
                config.max_bid_ask_spread = max_bid_ask_spread
                logger.info(f"Overriding max bid-ask spread: {max_bid_ask_spread}")
            if optimal_days_expiry is not None:
                config.optimal_days_to_expiry = optimal_days_expiry
                logger.info(f"Overriding optimal days to expiry: {optimal_days_expiry}")

            logger.info(
                f"Using {scoring_strategy.upper()} scoring strategy for global opportunity selection"
            )

        return config

    def sfr_finder(
        self,
        symbol_list,
        profit_target,
        cost_limit,
        quantity=1,
        volume_limit=200,
        log_file=None,
        debug=False,
        finviz_url=None,
    ):
        sfr = SFR(log_file=log_file, debug=debug)
        default_list = [
            "SPY",
            "MRK",
            "QQQ",
            "META",
            "PLTR",
            "SPOT",
            "KO",
            "LLY",
            "INTC",
            "FIS",
            "AZN",
            "XYZ",
            "V",
            "AMD",
        ]

        if finviz_url:
            logger.info(f"Scraping ticker symbols from Finviz URL: {finviz_url}")
            scraped_symbols = scrape_tickers_from_finviz(finviz_url)
            if scraped_symbols:
                if symbol_list:
                    logger.warning(
                        "Both Finviz URL and manual symbols provided, using Finviz tickers"
                    )
                symbol_list = scraped_symbols
                logger.info(
                    f"Successfully loaded {len(symbol_list)} tickers from Finviz: {symbol_list}"
                )
            else:
                logger.error(
                    "Failed to scrape tickers from Finviz URL, falling back to provided or default symbols"
                )
                symbol_list = symbol_list if symbol_list else default_list
        elif not symbol_list:
            symbol_list = default_list

        logger.info(f"Starting SFR scan with {len(symbol_list)} symbols: {symbol_list}")

        try:
            asyncio.run(
                sfr.scan(
                    symbol_list,
                    profit_target=profit_target,
                    volume_limit=volume_limit,
                    cost_limit=cost_limit,
                    quantity=quantity,
                )
            )

        except KeyboardInterrupt:
            # Disconnect from IB
            sfr.ib.disconnect()

    def syn_finder(
        self,
        symbol_list,
        cost_limit=120,
        max_loss_threshold=None,
        max_profit_threshold=None,
        profit_ratio_threshold=None,
        quantity=1,
        log_file=None,
        debug=False,
        finviz_url=None,
        # Global Opportunity Selection Configuration
        scoring_strategy="balanced",
        risk_reward_weight=None,
        liquidity_weight=None,
        time_decay_weight=None,
        market_quality_weight=None,
        min_risk_reward=None,
        min_liquidity=None,
        max_bid_ask_spread=None,
        optimal_days_expiry=None,
    ):
        # Create scoring configuration based on user inputs
        scoring_config = self._create_scoring_config(
            scoring_strategy=scoring_strategy,
            risk_reward_weight=risk_reward_weight,
            liquidity_weight=liquidity_weight,
            time_decay_weight=time_decay_weight,
            market_quality_weight=market_quality_weight,
            min_risk_reward=min_risk_reward,
            min_liquidity=min_liquidity,
            max_bid_ask_spread=max_bid_ask_spread,
            optimal_days_expiry=optimal_days_expiry,
        )

        # Create Syn instance with scoring configuration
        syn = Syn(log_file=log_file, debug=debug, scoring_config=scoring_config)
        default_list = [
            "SPY",
            "MRK",
            "QQQ",
            "META",
            "PLTR",
            "SPOT",
            "KO",
            "LLY",
            "INTC",
            "FIS",
            "AZN",
            "XYZ",
            "V",
            "AMD",
        ]

        if finviz_url:
            logger.info(f"Scraping ticker symbols from Finviz URL: {finviz_url}")
            scraped_symbols = scrape_tickers_from_finviz(finviz_url)
            if scraped_symbols:
                if symbol_list:
                    logger.warning(
                        "Both Finviz URL and manual symbols provided, using Finviz tickers"
                    )
                symbol_list = scraped_symbols
                logger.info(
                    f"Successfully loaded {len(symbol_list)} tickers from Finviz: {symbol_list}"
                )
            else:
                logger.error(
                    "Failed to scrape tickers from Finviz URL, falling back to provided or default symbols"
                )
                symbol_list = symbol_list if symbol_list else default_list
        elif not symbol_list:
            symbol_list = default_list

        logger.info(f"Starting SYN scan with {len(symbol_list)} symbols: {symbol_list}")

        try:
            asyncio.run(
                syn.scan(
                    symbol_list,
                    cost_limit=cost_limit,
                    max_loss_threshold=max_loss_threshold,
                    max_profit_threshold=max_profit_threshold,
                    profit_ratio_threshold=profit_ratio_threshold,
                    quantity=quantity,
                )
            )
        except KeyboardInterrupt:
            syn.ib.disconnect()
