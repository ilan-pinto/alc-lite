import asyncio
import os

import logging
import pandas as pd
from ib_async import *

from modules.Arbitrage.SFR import SFR
from modules.Arbitrage.Synthetic import Syn
from modules.finviz_scraper import scrape_tickers_from_finviz

logger = logging.getLogger(__name__)


class OptionScan:

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
    ):
        syn = Syn(log_file=log_file, debug=debug)
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
