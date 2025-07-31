import asyncio
import os

import logging
import pandas as pd
from ib_async import *

from modules.Arbitrage.CalendarSpread import CalendarSpread
from modules.Arbitrage.SFR import SFR
from modules.Arbitrage.Synthetic import ScoringConfig, Syn
from modules.finviz_scraper import scrape_tickers_from_finviz

logger = logging.getLogger(__name__)


def _configure_logging_level(debug=False, warning=False, error=False):
    """Configure logging level based on CLI flags"""
    from modules.Arbitrage.common import configure_logging

    if debug:
        # Debug mode: show all levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        configure_logging(debug=True, use_info_filter=False)
        logger.debug("Debug logging enabled - all levels shown")
    elif error:
        # Error mode: show INFO, WARNING, ERROR, CRITICAL (but not DEBUG)
        configure_logging(
            level=logging.INFO, use_info_filter=False, debug=False, warning=False
        )
        logger.info(
            "Error logging enabled - INFO, WARNING, ERROR and CRITICAL messages will be shown"
        )
    elif warning:
        # Warning mode: show INFO and WARNING (original behavior)
        configure_logging(warning=True)
        logger.info("Warning logging enabled - INFO and WARNING messages will be shown")
    else:
        # Default: INFO only (original behavior)
        configure_logging(use_info_filter=True)


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
        warning=False,
        error=False,
        finviz_url=None,
    ):
        # Configure logging level based on CLI flags
        _configure_logging_level(debug=debug, warning=warning, error=error)

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
        warning=False,
        error=False,
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

        # Configure logging level based on CLI flags
        _configure_logging_level(debug=debug, warning=warning, error=error)

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

    def calendar_finder(
        self,
        symbol_list,
        cost_limit=300.0,
        profit_target=0.25,
        iv_spread_threshold=0.015,
        theta_ratio_threshold=1.5,
        front_expiry_max_days=30,
        back_expiry_min_days=50,
        back_expiry_max_days=120,
        min_volume=10,
        max_bid_ask_spread=0.15,
        quantity=1,
        log_file=None,
        debug=False,
        warning=False,
        error=False,
        finviz_url=None,
    ):
        """
        Search for calendar spread arbitrage opportunities.

        Calendar spreads profit from time decay differential between front and back month options.
        This method scans for opportunities where the front month decays faster than the back month
        and there's a favorable implied volatility spread.

        Args:
            symbol_list (List[str]): List of symbols to scan
            cost_limit (float): Maximum net debit to pay for calendar spread (default: $300)
            profit_target (float): Target profit as percentage of max profit (default: 0.25 = 25%)
            iv_spread_threshold (float): Minimum IV spread (back - front) required (default: 0.015 = 1.5%)
            theta_ratio_threshold (float): Minimum theta ratio (front/back) required (default: 1.5)
            front_expiry_max_days (int): Maximum days to expiry for front month (default: 45)
            back_expiry_min_days (int): Minimum days to expiry for back month (default: 60)
            back_expiry_max_days (int): Maximum days to expiry for back month (default: 120)
            min_volume (int): Minimum daily volume per option leg (default: 10)
            max_bid_ask_spread (float): Maximum bid-ask spread as % of mid price (default: 0.15 = 15%)
            quantity (int): Maximum number of calendar spreads to execute (default: 1)
            log_file (str): Optional log file path for detailed logging
            debug (bool): Enable debug logging (default: False)
            finviz_url (str): Optional Finviz URL to scrape symbols from

        Example:
            scanner = OptionScan()
            scanner.calendar_finder(
                symbol_list=["SPY", "QQQ", "AAPL"],
                cost_limit=500.0,
                profit_target=0.30,
                iv_spread_threshold=0.04,
                quantity=2
            )
        """
        # Configure logging level based on CLI flags
        _configure_logging_level(debug=debug, warning=warning, error=error)

        # Create CalendarSpread instance with configuration
        from modules.Arbitrage.CalendarSpread import CalendarSpreadConfig

        config = CalendarSpreadConfig(
            min_iv_spread=iv_spread_threshold * 100,  # Convert to percentage
            min_theta_ratio=theta_ratio_threshold,
            max_days_front=front_expiry_max_days,
            min_days_back=back_expiry_min_days,
            max_days_back=back_expiry_max_days,
            min_volume=min_volume,
            max_bid_ask_spread=max_bid_ask_spread,
            max_net_debit=cost_limit,
            target_profit_ratio=profit_target,
        )

        calendar = CalendarSpread(log_file=log_file)
        calendar.config = config

        # Handle symbol list logic (same as other methods)
        default_list = [
            "SPY",
            "QQQ",
            "META",
            "AAPL",
            "MSFT",
            "GOOGL",
            "TSLA",
            "NVDA",
            "AMZN",
            "NFLX",
            "AMD",
            "INTC",
            "V",
            "MA",
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

        logger.info(
            f"Starting CALENDAR scan with {len(symbol_list)} symbols: {symbol_list}"
        )
        logger.info(
            f"Calendar parameters: IV spread ≥{iv_spread_threshold:.1%}, theta ratio ≥{theta_ratio_threshold:.1f}"
        )
        logger.info(
            f"Expiry window: front ≤{front_expiry_max_days}d, back {back_expiry_min_days}-{back_expiry_max_days}d"
        )
        logger.info(
            f"Cost limit: ${cost_limit:.0f}, target profit: {profit_target:.1%}"
        )

        try:
            asyncio.run(
                calendar.scan(
                    symbol_list=symbol_list,
                    cost_limit=cost_limit,
                    profit_target=profit_target,
                    quantity=quantity,
                )
            )
        except KeyboardInterrupt:
            logger.info("Calendar spread scan interrupted by user")
            if hasattr(calendar, "ib") and calendar.ib and calendar.ib.isConnected():
                calendar.ib.disconnect()
        except Exception as e:
            logger.error(f"Error in calendar spread scan: {str(e)}")
            if hasattr(calendar, "ib") and calendar.ib and calendar.ib.isConnected():
                calendar.ib.disconnect()
        finally:
            # Ensure cleanup even if no exceptions occurred
            if hasattr(calendar, "ib") and calendar.ib and calendar.ib.isConnected():
                calendar.ib.disconnect()
