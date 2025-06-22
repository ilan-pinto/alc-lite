import asyncio
import os

import logging
import pandas as pd
from ib_async import *

from modules.Arbitrage.SFR import SFR
from modules.Arbitrage.Synthetic import Syn

logger = logging.getLogger(__name__)


class OptionScan:

    def sfr_finder(
        self,
        symbol_list,
        profit_target,
        cost_limit,
        volume_limit=200,
    ):
        sfr = SFR()
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

        if not symbol_list:
            symbol_list = default_list

        try:
            asyncio.run(
                sfr.scan(
                    symbol_list,
                    profit_target=profit_target,
                    volume_limit=volume_limit,
                    cost_limit=cost_limit,
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
    ):
        syn = Syn()
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

        if not symbol_list:
            symbol_list = default_list

        try:
            asyncio.run(
                syn.scan(
                    symbol_list,
                    cost_limit=cost_limit,
                    max_loss_threshold=max_loss_threshold,
                    max_profit_threshold=max_profit_threshold,
                    profit_ratio_threshold=profit_ratio_threshold,
                )
            )
        except KeyboardInterrupt:
            syn.ib.disconnect()
