"""
Market simulation scenarios for comprehensive arbitrage testing.
This module provides various market conditions and arbitrage opportunities for testing.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

try:
    from .mock_ib import MarketDataGenerator, MockTicker
except ImportError:
    from mock_ib import MarketDataGenerator, MockTicker


class MarketScenarios:
    """Collection of market scenarios for testing different arbitrage conditions"""

    @staticmethod
    def dell_profitable_conversion(
        stock_price: float = 131.24,
    ) -> Dict[int, MockTicker]:
        """
        DELL scenario with guaranteed conversion arbitrage opportunity.

        Setup: Call 132 / Put 131 with 1-strike difference
        Expected: Net credit > spread for profitable arbitrage
        """
        tickers = {}
        # Calculate dynamic expiry date between 19-45 days from today
        expiry_date = datetime.now() + timedelta(days=30)
        expiry = expiry_date.strftime("%Y%m%d")

        # Stock data
        stock_ticker = MarketDataGenerator.generate_stock_data(
            "DELL", stock_price, volume=500000
        )
        tickers[stock_ticker.contract.conId] = stock_ticker

        # Create profitable arbitrage: Call 132 / Put 131
        # Net credit = 2.20 - 1.40 = 0.80 (sell call 132 at bid, buy put 131 at ask)
        # Spread = 131.24 - 131 = 0.24
        # Min profit = 0.80 - 0.24 = 0.56 > 0 ✅ PROFITABLE!

        call_132 = MarketDataGenerator.generate_option_data(
            "DELL", expiry, 132.0, "C", stock_price, 30
        )
        call_132.bid = 2.20  # High bid for selling call
        call_132.ask = 2.40
        call_132.close = 2.30
        call_132.volume = 250
        tickers[call_132.contract.conId] = call_132

        put_131 = MarketDataGenerator.generate_option_data(
            "DELL", expiry, 131.0, "P", stock_price, 30
        )
        put_131.bid = 1.20
        put_131.ask = 1.40  # Low ask for buying put
        put_131.close = 1.30
        put_131.volume = 300
        tickers[put_131.contract.conId] = put_131

        # Add supporting strikes for complete option chain
        supporting_strikes = [128, 129, 130, 133, 134, 135]
        for strike in supporting_strikes:
            for right in ["C", "P"]:
                ticker = MarketDataGenerator.generate_option_data(
                    "DELL", expiry, strike, right, stock_price, 30
                )
                # Ensure reasonable volume for supporting strikes
                ticker.volume = max(50, ticker.volume // 2)
                tickers[ticker.contract.conId] = ticker

        return tickers

    @staticmethod
    def dell_no_arbitrage(stock_price: float = 131.24) -> Dict[int, MockTicker]:
        """
        DELL scenario with NO arbitrage opportunity.

        Setup: Normal market conditions where spread >= net_credit
        Expected: No profitable arbitrage detected
        """
        tickers = {}
        # Calculate dynamic expiry date
        expiry_date = datetime.now() + timedelta(days=30)
        expiry = expiry_date.strftime("%Y%m%d")

        # Stock data
        stock_ticker = MarketDataGenerator.generate_stock_data("DELL", stock_price)
        tickers[stock_ticker.contract.conId] = stock_ticker

        # Create scenario with NO arbitrage opportunity: Call 132 / Put 131
        # Stock price: 131.24, Call strike: 132, Put strike: 131
        # Spread = stock_price - put_strike = 131.24 - 131 = 0.24
        # For NO arbitrage: net_credit <= spread, so min_profit <= 0
        # Target: net_credit = 0.20, min_profit = 0.20 - 0.24 = -0.04 (NO ARBITRAGE)

        call_132 = MarketDataGenerator.generate_option_data(
            "DELL", expiry, 132.0, "C", stock_price, 30
        )
        call_132.bid = 0.80  # Selling call premium
        call_132.ask = 1.00
        call_132.close = 0.90
        tickers[call_132.contract.conId] = call_132

        put_131 = MarketDataGenerator.generate_option_data(
            "DELL", expiry, 131.0, "P", stock_price, 30
        )
        put_131.bid = 0.50
        put_131.ask = 0.60  # Buying put cost (lower than call bid)
        put_131.close = 0.55
        tickers[put_131.contract.conId] = put_131

        # Verification: net_credit = 0.80 - 0.60 = 0.20
        # spread = 131.24 - 131 = 0.24
        # min_profit = 0.20 - 0.24 = -0.04 < 0 ❌ NO ARBITRAGE

        # Add complete option chain (excluding our manually set strikes)
        for strike in range(126, 137):
            for right in ["C", "P"]:
                # Skip Call 132 and Put 131 as we manually set those above
                if (strike == 132 and right == "C") or (strike == 131 and right == "P"):
                    continue
                ticker = MarketDataGenerator.generate_option_data(
                    "DELL", expiry, strike, right, stock_price, 30
                )
                tickers[ticker.contract.conId] = ticker

        return tickers

    @staticmethod
    def dell_wide_spreads(stock_price: float = 131.24) -> Dict[int, MockTicker]:
        """
        DELL scenario with wide bid-ask spreads that should be rejected.

        Setup: Wide spreads > 20 (threshold in SFR.py)
        Expected: Rejected due to BID_ASK_SPREAD_TOO_WIDE
        """
        tickers = {}
        # Calculate dynamic expiry date
        expiry_date = datetime.now() + timedelta(days=30)
        expiry = expiry_date.strftime("%Y%m%d")

        # Stock data (normal spreads)
        stock_ticker = MarketDataGenerator.generate_stock_data("DELL", stock_price)
        tickers[stock_ticker.contract.conId] = stock_ticker

        # Create options with excessively wide spreads
        call_132 = MarketDataGenerator.generate_option_data(
            "DELL", expiry, 132.0, "C", stock_price, 30
        )
        call_132.bid = 1.00
        call_132.ask = 25.00  # 24.00 spread > 20 threshold
        call_132.close = 13.00
        call_132.volume = 5  # Low volume
        tickers[call_132.contract.conId] = call_132

        put_131 = MarketDataGenerator.generate_option_data(
            "DELL", expiry, 131.0, "P", stock_price, 30
        )
        put_131.bid = 0.50
        put_131.ask = 22.00  # 21.50 spread > 20 threshold
        put_131.close = 11.25
        put_131.volume = 3  # Very low volume
        tickers[put_131.contract.conId] = put_131

        return tickers

    @staticmethod
    def dell_negative_net_credit(stock_price: float = 131.24) -> Dict[int, MockTicker]:
        """
        DELL scenario with NEGATIVE net credit.

        Setup: Call premium < Put cost, resulting in net debit
        Expected: Rejected due to NET_CREDIT_NEGATIVE
        """
        tickers = {}
        # Calculate dynamic expiry date
        expiry_date = datetime.now() + timedelta(days=30)
        expiry = expiry_date.strftime("%Y%m%d")

        # Stock data
        stock_ticker = MarketDataGenerator.generate_stock_data("DELL", stock_price)
        tickers[stock_ticker.contract.conId] = stock_ticker

        # Create scenario with NEGATIVE net credit: Call 132 / Put 131
        # Stock price: 131.24, Call strike: 132, Put strike: 131
        # Net credit = call_bid - put_ask = 0.50 - 2.00 = -1.50 (NEGATIVE!)
        # This represents a situation where puts are very expensive (high volatility scenario)

        call_132 = MarketDataGenerator.generate_option_data(
            "DELL", expiry, 132.0, "C", stock_price, 30
        )
        call_132.bid = 0.50  # Low call premium (OTM call)
        call_132.ask = 0.70
        call_132.close = 0.60
        tickers[call_132.contract.conId] = call_132

        put_131 = MarketDataGenerator.generate_option_data(
            "DELL", expiry, 131.0, "P", stock_price, 30
        )
        put_131.bid = 1.80
        put_131.ask = 2.00  # Very expensive put (high IV scenario)
        put_131.close = 1.90
        tickers[put_131.contract.conId] = put_131

        # Verification: net_credit = 0.50 - 2.00 = -1.50 < 0 ❌ NEGATIVE NET CREDIT

        # Add complete option chain (excluding our manually set strikes)
        for strike in range(126, 137):
            for right in ["C", "P"]:
                # Skip Call 132 and Put 131 as we manually set those above
                if (strike == 132 and right == "C") or (strike == 131 and right == "P"):
                    continue
                ticker = MarketDataGenerator.generate_option_data(
                    "DELL", expiry, strike, right, stock_price, 30
                )
                tickers[ticker.contract.conId] = ticker

        return tickers

    @staticmethod
    def dell_low_volume(stock_price: float = 131.24) -> Dict[int, MockTicker]:
        """
        DELL scenario with very low volume options.

        Setup: Volume < 5 (very low volume threshold)
        Expected: Accepted but logged as debug message
        """
        tickers = {}
        # Calculate dynamic expiry date
        expiry_date = datetime.now() + timedelta(days=30)
        expiry = expiry_date.strftime("%Y%m%d")

        # Stock data
        stock_ticker = MarketDataGenerator.generate_stock_data("DELL", stock_price)
        tickers[stock_ticker.contract.conId] = stock_ticker

        # Create low volume options
        call_132 = MarketDataGenerator.generate_option_data(
            "DELL", expiry, 132.0, "C", stock_price, 30
        )
        call_132.bid = 1.80
        call_132.ask = 2.00
        call_132.volume = 2  # Very low volume
        tickers[call_132.contract.conId] = call_132

        put_131 = MarketDataGenerator.generate_option_data(
            "DELL", expiry, 131.0, "P", stock_price, 30
        )
        put_131.bid = 1.40
        put_131.ask = 1.60
        put_131.volume = 1  # Extremely low volume
        tickers[put_131.contract.conId] = put_131

        return tickers

    @staticmethod
    def spy_multiple_expiries(stock_price: float = 420.50) -> Dict[int, MockTicker]:
        """
        SPY scenario with multiple expiries for testing expiry selection logic.

        Setup: 3 expiries with different arbitrage opportunities
        Expected: Proper expiry prioritization and selection
        """
        tickers = {}

        # Stock data
        stock_ticker = MarketDataGenerator.generate_stock_data(
            "SPY", stock_price, volume=10000000
        )
        tickers[stock_ticker.contract.conId] = stock_ticker

        # Create 3 expiries with different opportunities (all within valid range)
        today = datetime.now()
        expiries = [
            (today + timedelta(days=25)).strftime("%Y%m%d"),  # 25 days
            (today + timedelta(days=35)).strftime("%Y%m%d"),  # 35 days
            (today + timedelta(days=42)).strftime("%Y%m%d"),  # 42 days
        ]

        for i, expiry in enumerate(expiries):
            days_to_expiry = 15 + (i * 20)  # 15, 35, 55 days

            # Create strike combinations for each expiry
            base_strikes = [418, 419, 420, 421, 422, 423]

            for strike in base_strikes:
                for right in ["C", "P"]:
                    ticker = MarketDataGenerator.generate_option_data(
                        "SPY", expiry, strike, right, stock_price, days_to_expiry
                    )
                    # Adjust volume based on expiry (closer = higher volume)
                    volume_multiplier = 1.5 - (i * 0.3)
                    ticker.volume = int(ticker.volume * volume_multiplier)
                    tickers[ticker.contract.conId] = ticker

            # Create one arbitrage opportunity in the first expiry
            if i == 0:
                call_421 = next(
                    t
                    for t in tickers.values()
                    if hasattr(t.contract, "strike")
                    and t.contract.strike == 421
                    and t.contract.right == "C"
                    and t.contract.lastTradeDateOrContractMonth == expiry
                )
                put_420 = next(
                    t
                    for t in tickers.values()
                    if hasattr(t.contract, "strike")
                    and t.contract.strike == 420
                    and t.contract.right == "P"
                    and t.contract.lastTradeDateOrContractMonth == expiry
                )

                # Adjust for arbitrage opportunity
                call_421.bid = 2.50
                call_421.ask = 2.70
                put_420.bid = 1.80
                put_420.ask = 2.00

        return tickers

    @staticmethod
    def multi_symbol_one_profitable() -> Dict[str, Dict[int, MockTicker]]:
        """
        Multi-symbol scenario where ONE symbol has profitable arbitrage.

        Setup: AAPL (profitable), MSFT (no arbitrage), TSLA (negative credit)
        Expected: Only AAPL should create an executor
        """
        scenarios = {}

        # AAPL: Profitable arbitrage (similar to DELL profitable but different prices)
        scenarios["AAPL"] = {}
        expiry_date = datetime.now() + timedelta(days=35)
        expiry = expiry_date.strftime("%Y%m%d")

        stock_price = 185.50
        stock_ticker = MarketDataGenerator.generate_stock_data(
            "AAPL", stock_price, volume=8000000
        )
        scenarios["AAPL"][stock_ticker.contract.conId] = stock_ticker

        # Profitable: Call 185.5 / Put 184.5 (based on stock price 185.50)
        call_185_5 = MarketDataGenerator.generate_option_data(
            "AAPL", expiry, 185.5, "C", stock_price, 35
        )
        call_185_5.bid = 3.20  # High bid for selling call
        call_185_5.ask = 3.40
        call_185_5.close = 3.30
        call_185_5.volume = 450
        scenarios["AAPL"][call_185_5.contract.conId] = call_185_5

        put_184_5 = MarketDataGenerator.generate_option_data(
            "AAPL", expiry, 184.5, "P", stock_price, 35
        )
        put_184_5.bid = 2.80
        put_184_5.ask = (
            2.60  # Net credit = 3.20 - 2.60 = 0.60, Spread = 185.50 - 184.5 = 1.00
        )
        put_184_5.close = 2.70  # Min profit = 0.60 - 1.00 = -0.40 < 0 ❌ (This is still not profitable!)
        put_184_5.volume = 380
        scenarios["AAPL"][put_184_5.contract.conId] = put_184_5

        # Fix to make actually profitable: Call 184.5 / Put 183.5 (1-strike difference)
        call_184_5 = MarketDataGenerator.generate_option_data(
            "AAPL", expiry, 184.5, "C", stock_price, 35
        )
        call_184_5.bid = 4.20  # High bid for selling call
        call_184_5.ask = 4.30  # Call spread = 0.10/4.30 = 2.3% (well within 5% limit)
        call_184_5.close = 4.30
        call_184_5.volume = 450
        scenarios["AAPL"][call_184_5.contract.conId] = call_184_5

        put_183_5 = MarketDataGenerator.generate_option_data(
            "AAPL", expiry, 183.5, "P", stock_price, 35
        )
        put_183_5.bid = 1.94
        put_183_5.ask = (
            1.98  # Net credit = 4.20 - 1.98 = 2.22, Spread = 185.50 - 183.5 = 2.00
            # Put spread = 0.04/1.98 = 2.0% (well within 5% limit)
        )
        put_183_5.close = 1.96  # Min profit = 2.22 - 2.00 = 0.22 > 0 ✅ PROFITABLE!
        put_183_5.volume = 380
        scenarios["AAPL"][put_183_5.contract.conId] = put_183_5

        # Add supporting strikes including the exact ones we need
        for strike in [182.5, 183.5, 184.5, 185.5, 186.5, 187.5]:
            for right in ["C", "P"]:
                # Skip if we already defined this strike/right combination
                if (
                    (strike == 184.5 and right == "C")
                    or (strike == 183.5 and right == "P")
                    or (strike == 185.5 and right == "C")
                    or (strike == 184.5 and right == "P")
                ):
                    continue
                ticker = MarketDataGenerator.generate_option_data(
                    "AAPL", expiry, strike, right, stock_price, 35
                )
                ticker.volume = max(100, ticker.volume // 3)
                scenarios["AAPL"][ticker.contract.conId] = ticker

        # MSFT: No arbitrage (spread >= net_credit)
        scenarios["MSFT"] = {}
        stock_price = 415.75
        stock_ticker = MarketDataGenerator.generate_stock_data(
            "MSFT", stock_price, volume=3500000
        )
        scenarios["MSFT"][stock_ticker.contract.conId] = stock_ticker

        # MSFT: No arbitrage - using strikes around stock price 415.75
        call_415_75 = MarketDataGenerator.generate_option_data(
            "MSFT", expiry, 415.75, "C", stock_price, 35
        )
        call_415_75.bid = 1.20
        call_415_75.ask = 1.40
        call_415_75.close = 1.30
        scenarios["MSFT"][call_415_75.contract.conId] = call_415_75

        put_414_75 = MarketDataGenerator.generate_option_data(
            "MSFT", expiry, 414.75, "P", stock_price, 35
        )
        put_414_75.bid = 0.90
        put_414_75.ask = (
            1.10  # Net credit = 1.20 - 1.10 = 0.10, Spread = 415.75 - 414.75 = 1.00
        )
        put_414_75.close = 1.00  # Min profit = 0.10 - 1.00 = -0.90 < 0 ❌ NO ARBITRAGE
        scenarios["MSFT"][put_414_75.contract.conId] = put_414_75

        # Add supporting strikes for MSFT including the exact ones we need
        # CRITICAL: Override auto-generated prices to ensure NO arbitrage opportunities
        for strike in [413.75, 414.75, 415.75, 416.75, 417.75]:
            for right in ["C", "P"]:
                # Skip if we already defined this strike/right combination
                if (strike == 415.75 and right == "C") or (
                    strike == 414.75 and right == "P"
                ):
                    continue
                ticker = MarketDataGenerator.generate_option_data(
                    "MSFT", expiry, strike, right, stock_price, 35
                )

                # Override with unprofitable pricing to prevent unintended arbitrage
                if (
                    right == "C"
                ):  # Calls - make bids low (hard to sell calls profitably)
                    ticker.bid = 0.05  # Very low bid
                    ticker.ask = ticker.bid + 0.20  # Wide spread
                    ticker.close = ticker.bid + 0.10
                else:  # Puts - make asks high (expensive to buy puts)
                    ticker.ask = 5.00  # Very high ask
                    ticker.bid = ticker.ask - 0.20  # Wide spread
                    ticker.close = ticker.ask - 0.10

                scenarios["MSFT"][ticker.contract.conId] = ticker

        # TSLA: Negative net credit
        scenarios["TSLA"] = {}
        stock_price = 245.80
        stock_ticker = MarketDataGenerator.generate_stock_data(
            "TSLA", stock_price, volume=12000000
        )
        scenarios["TSLA"][stock_ticker.contract.conId] = stock_ticker

        # TSLA: Negative net credit - using strikes around stock price 245.80
        call_245_8 = MarketDataGenerator.generate_option_data(
            "TSLA", expiry, 245.8, "C", stock_price, 35
        )
        call_245_8.bid = 0.80  # Low call premium
        call_245_8.ask = 1.00
        call_245_8.close = 0.90
        scenarios["TSLA"][call_245_8.contract.conId] = call_245_8

        put_244_8 = MarketDataGenerator.generate_option_data(
            "TSLA", expiry, 244.8, "P", stock_price, 35
        )
        put_244_8.bid = 2.20
        put_244_8.ask = 2.40  # Net credit = 0.80 - 2.40 = -1.60 < 0 ❌ NEGATIVE CREDIT
        put_244_8.close = 2.30
        scenarios["TSLA"][put_244_8.contract.conId] = put_244_8

        # Add supporting strikes for TSLA including the exact ones we need
        # CRITICAL: Override auto-generated prices to ensure NO arbitrage opportunities
        for strike in [243.8, 244.8, 245.8, 246.8, 247.8]:
            for right in ["C", "P"]:
                # Skip if we already defined this strike/right combination
                if (strike == 245.8 and right == "C") or (
                    strike == 244.8 and right == "P"
                ):
                    continue
                ticker = MarketDataGenerator.generate_option_data(
                    "TSLA", expiry, strike, right, stock_price, 35
                )

                # Override with unprofitable pricing to prevent unintended arbitrage
                if (
                    right == "C"
                ):  # Calls - make bids low (hard to sell calls profitably)
                    ticker.bid = 0.05  # Very low bid
                    ticker.ask = ticker.bid + 0.25  # Wide spread
                    ticker.close = ticker.bid + 0.12
                else:  # Puts - make asks high (expensive to buy puts)
                    ticker.ask = 8.00  # Very high ask
                    ticker.bid = ticker.ask - 0.25  # Wide spread
                    ticker.close = ticker.ask - 0.12

                scenarios["TSLA"][ticker.contract.conId] = ticker

        return scenarios

    @staticmethod
    def multi_symbol_none_profitable() -> Dict[str, Dict[int, MockTicker]]:
        """
        Multi-symbol scenario where NO symbols have profitable arbitrage.

        Setup: META (no arbitrage), NVDA (wide spreads), AMZN (negative credit)
        Expected: No executors should be created
        """
        scenarios = {}
        expiry_date = datetime.now() + timedelta(days=28)
        expiry = expiry_date.strftime("%Y%m%d")

        # META: No arbitrage opportunity
        scenarios["META"] = {}
        stock_price = 495.25
        stock_ticker = MarketDataGenerator.generate_stock_data(
            "META", stock_price, volume=5500000
        )
        scenarios["META"][stock_ticker.contract.conId] = stock_ticker

        call_496 = MarketDataGenerator.generate_option_data(
            "META", expiry, 496.0, "C", stock_price, 28
        )
        call_496.bid = 2.10
        call_496.ask = 2.30
        call_496.close = 2.20
        scenarios["META"][call_496.contract.conId] = call_496

        put_495 = MarketDataGenerator.generate_option_data(
            "META", expiry, 495.0, "P", stock_price, 28
        )
        put_495.bid = 1.80
        put_495.ask = (
            2.00  # Net credit = 2.10 - 2.00 = 0.10, Spread = 495.25 - 495 = 0.25
        )
        put_495.close = 1.90  # Min profit = 0.10 - 0.25 = -0.15 < 0 ❌
        scenarios["META"][put_495.contract.conId] = put_495

        # Add supporting strikes for META
        for strike in [492, 493, 494, 497, 498, 499]:
            for right in ["C", "P"]:
                ticker = MarketDataGenerator.generate_option_data(
                    "META", expiry, strike, right, stock_price, 28
                )
                scenarios["META"][ticker.contract.conId] = ticker

        # NVDA: Wide bid-ask spreads
        scenarios["NVDA"] = {}
        stock_price = 875.40
        stock_ticker = MarketDataGenerator.generate_stock_data(
            "NVDA", stock_price, volume=25000000
        )
        scenarios["NVDA"][stock_ticker.contract.conId] = stock_ticker

        call_876 = MarketDataGenerator.generate_option_data(
            "NVDA", expiry, 876.0, "C", stock_price, 28
        )
        call_876.bid = 15.00
        call_876.ask = 40.00  # 25.00 spread > 20 threshold
        call_876.close = 27.50
        call_876.volume = 8
        scenarios["NVDA"][call_876.contract.conId] = call_876

        put_875 = MarketDataGenerator.generate_option_data(
            "NVDA", expiry, 875.0, "P", stock_price, 28
        )
        put_875.bid = 12.00
        put_875.ask = 35.00  # 23.00 spread > 20 threshold
        put_875.close = 23.50
        put_875.volume = 5
        scenarios["NVDA"][put_875.contract.conId] = put_875

        # Add supporting strikes for NVDA (all with wide spreads)
        for strike in [872, 873, 874, 877, 878, 879]:
            for right in ["C", "P"]:
                ticker = MarketDataGenerator.generate_option_data(
                    "NVDA", expiry, strike, right, stock_price, 28
                )
                # Make all spreads wide
                mid_price = ticker.close
                ticker.bid = mid_price * 0.6
                ticker.ask = mid_price * 1.8  # Wide spread
                ticker.volume = max(3, ticker.volume // 5)
                scenarios["NVDA"][ticker.contract.conId] = ticker

        # AMZN: Negative net credit
        scenarios["AMZN"] = {}
        stock_price = 155.90
        stock_ticker = MarketDataGenerator.generate_stock_data(
            "AMZN", stock_price, volume=6800000
        )
        scenarios["AMZN"][stock_ticker.contract.conId] = stock_ticker

        call_156 = MarketDataGenerator.generate_option_data(
            "AMZN", expiry, 156.0, "C", stock_price, 28
        )
        call_156.bid = 1.10  # Low call premium
        call_156.ask = 1.30
        call_156.close = 1.20
        scenarios["AMZN"][call_156.contract.conId] = call_156

        put_155 = MarketDataGenerator.generate_option_data(
            "AMZN", expiry, 155.0, "P", stock_price, 28
        )
        put_155.bid = 2.80
        put_155.ask = 3.00  # Net credit = 1.10 - 3.00 = -1.90 < 0 ❌
        put_155.close = 2.90
        scenarios["AMZN"][put_155.contract.conId] = put_155

        # Add supporting strikes for AMZN
        for strike in [152, 153, 154, 157, 158, 159]:
            for right in ["C", "P"]:
                ticker = MarketDataGenerator.generate_option_data(
                    "AMZN", expiry, strike, right, stock_price, 28
                )
                scenarios["AMZN"][ticker.contract.conId] = ticker

        return scenarios

    @staticmethod
    def global_selection_test_scenarios() -> Dict[str, Dict[int, MockTicker]]:
        """
        Multi-symbol scenarios specifically designed for global opportunity selection testing.

        Creates 5 symbols with varying quality opportunities to test:
        - Cross-symbol ranking and selection
        - Scoring algorithm effectiveness
        - Global optimization vs per-symbol optimization

        Returns:
            Dict mapping symbols to their market data with known ranking characteristics
        """
        scenarios = {}
        expiry_date = datetime.now() + timedelta(days=30)
        expiry = expiry_date.strftime("%Y%m%d")

        # RANK 1: GOOGL - Excellent all-around opportunity
        # High risk-reward (3.0), excellent liquidity (800+600 volume), tight spreads (0.05+0.03)
        googl_price = 150.00
        googl_tickers = {}

        stock = MarketDataGenerator.generate_stock_data(
            "GOOGL", googl_price, volume=8000000
        )
        googl_tickers[stock.contract.conId] = stock

        # Profitable synthetic: Sell Call 150, Buy Put 149
        # Net credit = 6.50 - 3.50 = 3.00
        # Spread = 150.00 - 149 = 1.00
        # Min profit = 3.00 - 1.00 = 2.00 (max loss)
        # Max profit = (150 - 149) + 2.00 = 3.00
        # Risk-reward = 3.00 / 2.00 = 1.5 (but we'll adjust for better score)

        call_150 = MarketDataGenerator.generate_option_data(
            "GOOGL", expiry, 150.0, "C", googl_price, 30
        )
        call_150.bid = 7.50  # High bid for selling
        call_150.ask = 7.55
        call_150.volume = 800
        googl_tickers[call_150.contract.conId] = call_150

        put_149 = MarketDataGenerator.generate_option_data(
            "GOOGL", expiry, 149.0, "P", googl_price, 30
        )
        put_149.bid = 4.45
        put_149.ask = 4.50  # Low ask for buying
        put_149.volume = 600
        googl_tickers[put_149.contract.conId] = put_149

        # Net credit = 7.50 - 4.50 = 3.00
        # Min profit = 3.00 - 1.00 = 2.00, Max profit = 1.00 + 2.00 = 3.00
        # Risk-reward = 3.00 / 2.00 = 1.5 (will be enhanced by excellent liquidity score)

        scenarios["GOOGL"] = googl_tickers

        # RANK 2: TSLA - High risk-reward but moderate liquidity
        # Very high risk-reward (4.0), moderate liquidity (300+200), moderate spreads (0.15+0.10)
        tsla_price = 200.00
        tsla_tickers = {}

        stock = MarketDataGenerator.generate_stock_data(
            "TSLA", tsla_price, volume=12000000
        )
        tsla_tickers[stock.contract.conId] = stock

        # High risk-reward synthetic: Sell Call 200, Buy Put 198
        # Wider spread but better risk-reward ratio
        call_200 = MarketDataGenerator.generate_option_data(
            "TSLA", expiry, 200.0, "C", tsla_price, 28
        )
        call_200.bid = 9.50
        call_200.ask = 9.65  # Wider spread
        call_200.volume = 300
        tsla_tickers[call_200.contract.conId] = call_200

        put_198 = MarketDataGenerator.generate_option_data(
            "TSLA", expiry, 198.0, "P", tsla_price, 28
        )
        put_198.bid = 4.85
        put_198.ask = 4.95
        put_198.volume = 200
        tsla_tickers[put_198.contract.conId] = put_198

        # Net credit = 9.50 - 4.95 = 4.55
        # Spread = 200.00 - 198 = 2.00
        # Min profit = 4.55 - 2.00 = 2.55, Max profit = 2.00 + 2.55 = 4.55
        # Risk-reward = 4.55 / 2.55 = 1.78 (good, but liquidity score will be lower)

        scenarios["TSLA"] = tsla_tickers

        # RANK 3: AAPL - Good balance but sub-optimal timing
        # Decent risk-reward (2.0), good liquidity (500+400), good spreads (0.08+0.06), sub-optimal time (32 days)
        aapl_price = 175.00
        aapl_tickers = {}

        stock = MarketDataGenerator.generate_stock_data(
            "AAPL", aapl_price, volume=10000000
        )
        aapl_tickers[stock.contract.conId] = stock

        call_175 = MarketDataGenerator.generate_option_data(
            "AAPL", expiry, 175.0, "C", aapl_price, 32
        )
        call_175.bid = 8.00
        call_175.ask = 8.08
        call_175.volume = 500
        aapl_tickers[call_175.contract.conId] = call_175

        put_174 = MarketDataGenerator.generate_option_data(
            "AAPL", expiry, 174.0, "P", aapl_price, 32
        )
        put_174.bid = 5.94
        put_174.ask = 6.00
        put_174.volume = 400
        aapl_tickers[put_174.contract.conId] = put_174

        # Net credit = 8.00 - 6.00 = 2.00
        # Spread = 175.00 - 174 = 1.00
        # Min profit = 2.00 - 1.00 = 1.00, Max profit = 1.00 + 1.00 = 2.00
        # Risk-reward = 2.00 / 1.00 = 2.0 (decent but time decay score will be lower)

        scenarios["AAPL"] = aapl_tickers

        # RANK 4: MSFT - Lower profits but optimal timing
        # Lower risk-reward (2.0), good liquidity (600+450), tight spreads (0.06+0.04), optimal time (25 days)
        msft_price = 400.00
        msft_tickers = {}

        stock = MarketDataGenerator.generate_stock_data(
            "MSFT", msft_price, volume=6000000
        )
        msft_tickers[stock.contract.conId] = stock

        call_400 = MarketDataGenerator.generate_option_data(
            "MSFT", expiry, 400.0, "C", msft_price, 25
        )
        call_400.bid = 6.50
        call_400.ask = 6.56
        call_400.volume = 600
        msft_tickers[call_400.contract.conId] = call_400

        put_399 = MarketDataGenerator.generate_option_data(
            "MSFT", expiry, 399.0, "P", msft_price, 25
        )
        put_399.bid = 4.46
        put_399.ask = 4.50
        put_399.volume = 450
        msft_tickers[put_399.contract.conId] = put_399

        # Net credit = 6.50 - 4.50 = 2.00
        # Spread = 400.00 - 399 = 1.00
        # Min profit = 2.00 - 1.00 = 1.00, Max profit = 1.00 + 1.00 = 2.00
        # Risk-reward = 2.00 / 1.00 = 2.0 (same as AAPL but better timing, tighter spreads)

        scenarios["MSFT"] = msft_tickers

        # RANK 5: AMZN - Poor risk-reward but excellent liquidity
        # Poor risk-reward (1.0), excellent liquidity (700+500), tight spreads (0.04+0.03), sub-optimal time (35 days)
        amzn_price = 125.00
        amzn_tickers = {}

        stock = MarketDataGenerator.generate_stock_data(
            "AMZN", amzn_price, volume=8500000
        )
        amzn_tickers[stock.contract.conId] = stock

        call_125 = MarketDataGenerator.generate_option_data(
            "AMZN", expiry, 125.0, "C", amzn_price, 35
        )
        call_125.bid = 5.50
        call_125.ask = 5.54
        call_125.volume = 700
        amzn_tickers[call_125.contract.conId] = call_125

        put_124 = MarketDataGenerator.generate_option_data(
            "AMZN", expiry, 124.0, "P", amzn_price, 35
        )
        put_124.bid = 4.47
        put_124.ask = 4.50
        put_124.volume = 500
        amzn_tickers[put_124.contract.conId] = put_124

        # Net credit = 5.50 - 4.50 = 1.00
        # Spread = 125.00 - 124 = 1.00
        # Min profit = 1.00 - 1.00 = 0.00, Max profit = 1.00 + 0.00 = 1.00
        # Risk-reward = 1.00 / 1.00 = 1.0 (poor, but excellent liquidity)

        scenarios["AMZN"] = amzn_tickers

        return scenarios

    @staticmethod
    def global_selection_scoring_strategy_scenarios() -> (
        Dict[str, Dict[int, MockTicker]]
    ):
        """
        Scenarios designed to test different scoring strategy preferences.

        Creates opportunities that will rank differently under different scoring strategies:
        - AGGRESSIVE: High risk-reward, low liquidity
        - LIQUIDITY_FOCUSED: Moderate risk-reward, high liquidity
        - BALANCED: Good all-around
        - CONSERVATIVE: Lower risk-reward, excellent spreads

        Returns:
            Dict mapping symbols to market data optimized for strategy testing
        """
        scenarios = {}
        expiry_date = datetime.now() + timedelta(days=30)
        expiry = expiry_date.strftime("%Y%m%d")

        # AGGRESSIVE strategy favorite: High risk-reward, poor liquidity
        aggressive_price = 300.00
        aggressive_tickers = {}

        stock = MarketDataGenerator.generate_stock_data(
            "AGGRESSIVE", aggressive_price, volume=2000000
        )
        aggressive_tickers[stock.contract.conId] = stock

        # High risk-reward but poor liquidity and wide spreads
        call_300 = MarketDataGenerator.generate_option_data(
            "AGGRESSIVE", expiry, 300.0, "C", aggressive_price, 35
        )
        call_300.bid = 12.00
        call_300.ask = 12.40  # 0.40 spread
        call_300.volume = 50  # Very low volume
        aggressive_tickers[call_300.contract.conId] = call_300

        put_298 = MarketDataGenerator.generate_option_data(
            "AGGRESSIVE", expiry, 298.0, "P", aggressive_price, 35
        )
        put_298.bid = 2.70
        put_298.ask = 3.00  # 0.30 spread
        put_298.volume = 30  # Very low volume
        aggressive_tickers[put_298.contract.conId] = put_298

        # Net credit = 12.00 - 3.00 = 9.00
        # Spread = 300.00 - 298 = 2.00
        # Min profit = 9.00 - 2.00 = 7.00, Max profit = 2.00 + 7.00 = 9.00
        # Risk-reward = 9.00 / 7.00 = 1.29 (will be boosted by aggressive weighting)

        scenarios["AGGRESSIVE"] = aggressive_tickers

        # LIQUIDITY_FOCUSED strategy favorite: Moderate risk-reward, excellent liquidity
        liquidity_price = 250.00
        liquidity_tickers = {}

        stock = MarketDataGenerator.generate_stock_data(
            "LIQUIDITY", liquidity_price, volume=15000000
        )
        liquidity_tickers[stock.contract.conId] = stock

        # Moderate returns but excellent liquidity and tight spreads
        call_250 = MarketDataGenerator.generate_option_data(
            "LIQUIDITY", expiry, 250.0, "C", liquidity_price, 25
        )
        call_250.bid = 5.98
        call_250.ask = 6.00  # 0.02 spread (very tight)
        call_250.volume = 1000  # High volume
        liquidity_tickers[call_250.contract.conId] = call_250

        put_249 = MarketDataGenerator.generate_option_data(
            "LIQUIDITY", expiry, 249.0, "P", liquidity_price, 25
        )
        put_249.bid = 4.49
        put_249.ask = 4.50  # 0.01 spread (very tight)
        put_249.volume = 800  # High volume
        liquidity_tickers[put_249.contract.conId] = put_249

        # Net credit = 5.98 - 4.50 = 1.48
        # Spread = 250.00 - 249 = 1.00
        # Min profit = 1.48 - 1.00 = 0.48, Max profit = 1.00 + 0.48 = 1.48
        # Risk-reward = 1.48 / 0.48 = 3.08 (moderate but will score high on liquidity)

        scenarios["LIQUIDITY"] = liquidity_tickers

        # BALANCED strategy favorite: Good all-around metrics
        balanced_price = 100.00
        balanced_tickers = {}

        stock = MarketDataGenerator.generate_stock_data(
            "BALANCED", balanced_price, volume=5000000
        )
        balanced_tickers[stock.contract.conId] = stock

        # Well-balanced opportunity with optimal timing
        call_100 = MarketDataGenerator.generate_option_data(
            "BALANCED", expiry, 100.0, "C", balanced_price, 30
        )
        call_100.bid = 4.90
        call_100.ask = 5.00  # 0.10 spread
        call_100.volume = 400  # Good volume
        balanced_tickers[call_100.contract.conId] = call_100

        put_99 = MarketDataGenerator.generate_option_data(
            "BALANCED", expiry, 99.0, "P", balanced_price, 30
        )
        put_99.bid = 2.92
        put_99.ask = 3.00  # 0.08 spread
        put_99.volume = 300  # Good volume
        balanced_tickers[put_99.contract.conId] = put_99

        # Net credit = 4.90 - 3.00 = 1.90
        # Spread = 100.00 - 99 = 1.00
        # Min profit = 1.90 - 1.00 = 0.90, Max profit = 1.00 + 0.90 = 1.90
        # Risk-reward = 1.90 / 0.90 = 2.11 (balanced across all metrics)

        scenarios["BALANCED"] = balanced_tickers

        # CONSERVATIVE strategy favorite: Lower risk-reward, excellent spreads
        conservative_price = 500.00
        conservative_tickers = {}

        stock = MarketDataGenerator.generate_stock_data(
            "CONSERVATIVE", conservative_price, volume=4000000
        )
        conservative_tickers[stock.contract.conId] = stock

        # Lower profit but very safe with tight spreads
        call_500 = MarketDataGenerator.generate_option_data(
            "CONSERVATIVE", expiry, 500.0, "C", conservative_price, 20
        )
        call_500.bid = 3.49
        call_500.ask = 3.50  # 0.01 spread (extremely tight)
        call_500.volume = 600  # Good volume
        conservative_tickers[call_500.contract.conId] = call_500

        put_499 = MarketDataGenerator.generate_option_data(
            "CONSERVATIVE", expiry, 499.0, "P", conservative_price, 20
        )
        put_499.bid = 2.49
        put_499.ask = 2.50  # 0.01 spread (extremely tight)
        put_499.volume = 500  # Good volume
        conservative_tickers[put_499.contract.conId] = put_499

        # Net credit = 3.49 - 2.50 = 0.99
        # Spread = 500.00 - 499 = 1.00
        # Min profit = 0.99 - 1.00 = -0.01, Max profit = 1.00 + (-0.01) = 0.99
        # Risk-reward = 0.99 / 0.01 = 99.0 (very conservative, excellent spreads)

        scenarios["CONSERVATIVE"] = conservative_tickers

        return scenarios

    @staticmethod
    def get_scenario(scenario_name: str, **kwargs) -> Dict[int, MockTicker]:
        """Get a specific market scenario by name"""
        scenarios = {
            "dell_profitable": MarketScenarios.dell_profitable_conversion,
            "dell_no_arbitrage": MarketScenarios.dell_no_arbitrage,
            "dell_negative_net_credit": MarketScenarios.dell_negative_net_credit,
            "dell_wide_spreads": MarketScenarios.dell_wide_spreads,
            "dell_low_volume": MarketScenarios.dell_low_volume,
            "spy_multiple_expiries": MarketScenarios.spy_multiple_expiries,
        }

        if scenario_name not in scenarios:
            raise ValueError(
                f"Unknown scenario: {scenario_name}. Available: {list(scenarios.keys())}"
            )

        return scenarios[scenario_name](**kwargs)

    @staticmethod
    def get_multi_symbol_scenario(
        scenario_name: str,
    ) -> Dict[str, Dict[int, MockTicker]]:
        """Get a multi-symbol scenario by name"""
        scenarios = {
            "one_profitable": MarketScenarios.multi_symbol_one_profitable,
            "none_profitable": MarketScenarios.multi_symbol_none_profitable,
        }

        if scenario_name not in scenarios:
            raise ValueError(
                f"Unknown multi-symbol scenario: {scenario_name}. Available: {list(scenarios.keys())}"
            )

        return scenarios[scenario_name]()


class ArbitrageTestCases:
    """Pre-defined test cases for specific arbitrage scenarios"""

    @staticmethod
    def dell_132_131_profitable() -> Tuple[str, Dict[int, MockTicker], Dict]:
        """
        DELL Call 132 / Put 131 profitable conversion arbitrage test case.

        Returns:
            Tuple of (description, market_data, expected_results)
        """
        description = "DELL C132/P131 profitable conversion arbitrage"
        market_data = MarketScenarios.dell_profitable_conversion(131.24)

        expected_results = {
            "should_find_arbitrage": True,
            "expected_combinations": [
                (132.0, 131.0),  # High probability 1-strike difference
                (131.0, 130.0),  # Secondary option
            ],
            "min_profit_positive": True,
            "net_credit": 0.80,  # call_bid - put_ask = 2.20 - 1.40
            "spread": 0.24,  # stock_price - put_strike = 131.24 - 131
            "min_profit": 0.56,  # net_credit - spread = 0.80 - 0.24
            "rejection_reasons": [],
        }

        return description, market_data, expected_results

    @staticmethod
    def dell_no_arbitrage_normal_market() -> Tuple[str, Dict[int, MockTicker], Dict]:
        """
        DELL normal market conditions with no arbitrage opportunity.
        """
        description = "DELL normal market - no arbitrage"
        market_data = MarketScenarios.dell_no_arbitrage(131.24)

        expected_results = {
            "should_find_arbitrage": False,
            "expected_combinations": [],
            "min_profit_positive": False,
            "rejection_reasons": [
                "ARBITRAGE_CONDITION_NOT_MET"
            ],  # spread >= net_credit
            "net_credit": 0.20,  # call_bid - put_ask = 0.80 - 0.60
            "spread": 0.24,  # stock_price - put_strike = 131.24 - 131
            "min_profit": -0.04,  # net_credit - spread = 0.20 - 0.24
        }

        return description, market_data, expected_results

    @staticmethod
    def dell_negative_net_credit_rejection() -> Tuple[str, Dict[int, MockTicker], Dict]:
        """
        DELL negative net credit scenario that should be rejected.
        """
        description = "DELL negative net credit rejection"
        market_data = MarketScenarios.dell_negative_net_credit(131.24)

        expected_results = {
            "should_find_arbitrage": False,
            "expected_combinations": [],
            "min_profit_positive": False,
            "rejection_reasons": [
                "ARBITRAGE_CONDITION_NOT_MET"
            ],  # Negative net credit is caught by arbitrage condition check
            "net_credit": -1.50,  # call_bid - put_ask = 0.50 - 2.00
            "spread": 0.24,  # stock_price - put_strike = 131.24 - 131
            "min_profit": -1.74,  # net_credit - spread = -1.50 - 0.24
        }

        return description, market_data, expected_results

    @staticmethod
    def dell_low_volume_acceptance() -> Tuple[str, Dict[int, MockTicker], Dict]:
        """
        DELL low volume scenario that should be accepted with debug warnings.
        """
        description = "DELL low volume acceptance"
        market_data = MarketScenarios.dell_low_volume(131.24)

        expected_results = {
            "should_find_arbitrage": False,  # No arbitrage due to insufficient profit (min_profit < 0)
            "expected_combinations": [],
            "min_profit_positive": False,
            "rejection_reasons": [
                "ARBITRAGE_CONDITION_NOT_MET"
            ],  # Rejected due to spread >= net_credit
            "net_credit": 0.20,  # call_bid - put_ask = 1.80 - 1.60
            "spread": 0.24,  # stock_price - put_strike = 131.24 - 131
            "min_profit": -0.04,  # net_credit - spread = 0.20 - 0.24 (not profitable)
            "volume_warnings": True,  # Should log debug warnings for low volume
            "volume_accepted": True,  # Key difference: low volume contracts are still processed
        }

        return description, market_data, expected_results

    @staticmethod
    def dell_wide_spreads_rejection() -> Tuple[str, Dict[int, MockTicker], Dict]:
        """
        DELL wide bid-ask spreads that should be rejected.
        """
        description = "DELL wide spreads rejection"
        market_data = MarketScenarios.dell_wide_spreads(131.24)

        expected_results = {
            "should_find_arbitrage": False,
            "expected_combinations": [],
            "rejection_reasons": ["BID_ASK_SPREAD_TOO_WIDE"],
        }

        return description, market_data, expected_results

    @staticmethod
    def multi_symbol_one_profitable() -> (
        Tuple[str, Dict[str, Dict[int, MockTicker]], Dict]
    ):
        """
        Multi-symbol scenario where ONE symbol (AAPL) has profitable arbitrage.
        """
        description = "Multi-symbol with one profitable arbitrage (AAPL)"
        market_data = MarketScenarios.multi_symbol_one_profitable()

        expected_results = {
            "symbols_scanned": ["AAPL", "MSFT", "TSLA"],
            "profitable_symbols": ["AAPL"],
            "unprofitable_symbols": ["MSFT", "TSLA"],
            "aapl_should_find_arbitrage": True,
            "msft_should_find_arbitrage": False,
            "tsla_should_find_arbitrage": False,
            "total_opportunities": 1,
            "aapl_net_credit": 0.60,
            "aapl_spread": 0.50,
            "aapl_min_profit": 0.10,
            "msft_rejection_reason": "ARBITRAGE_CONDITION_NOT_MET",
            "tsla_rejection_reason": "ARBITRAGE_CONDITION_NOT_MET",  # Negative credit caught by arbitrage condition
        }

        return description, market_data, expected_results

    @staticmethod
    def multi_symbol_none_profitable() -> (
        Tuple[str, Dict[str, Dict[int, MockTicker]], Dict]
    ):
        """
        Multi-symbol scenario where NO symbols have profitable arbitrage.
        """
        description = "Multi-symbol with no profitable arbitrage"
        market_data = MarketScenarios.multi_symbol_none_profitable()

        expected_results = {
            "symbols_scanned": ["META", "NVDA", "AMZN"],
            "profitable_symbols": [],
            "unprofitable_symbols": ["META", "NVDA", "AMZN"],
            "meta_should_find_arbitrage": False,
            "nvda_should_find_arbitrage": False,
            "amzn_should_find_arbitrage": False,
            "total_opportunities": 0,
            "meta_rejection_reason": "ARBITRAGE_CONDITION_NOT_MET",
            "nvda_rejection_reason": "BID_ASK_SPREAD_TOO_WIDE",
            "amzn_rejection_reason": "ARBITRAGE_CONDITION_NOT_MET",  # Negative credit caught by arbitrage condition
        }

        return description, market_data, expected_results

    @staticmethod
    def get_all_multi_symbol_test_cases() -> (
        List[Tuple[str, Dict[str, Dict[int, MockTicker]], Dict]]
    ):
        """Get all multi-symbol test cases"""
        return [
            ArbitrageTestCases.multi_symbol_one_profitable(),
            ArbitrageTestCases.multi_symbol_none_profitable(),
        ]

    @staticmethod
    def get_all_test_cases() -> List[Tuple[str, Dict[int, MockTicker], Dict]]:
        """Get all predefined single-symbol test cases"""
        return [
            ArbitrageTestCases.dell_132_131_profitable(),
            ArbitrageTestCases.dell_no_arbitrage_normal_market(),
            ArbitrageTestCases.dell_negative_net_credit_rejection(),
            ArbitrageTestCases.dell_low_volume_acceptance(),
            ArbitrageTestCases.dell_wide_spreads_rejection(),
        ]

    @staticmethod
    def get_all_test_cases_including_multi_symbol() -> List:
        """Get all test cases including multi-symbol scenarios"""
        single_symbol_cases = ArbitrageTestCases.get_all_test_cases()
        multi_symbol_cases = ArbitrageTestCases.get_all_multi_symbol_test_cases()
        return single_symbol_cases + multi_symbol_cases


class SyntheticScenarios:
    """Collection of market scenarios for testing Synthetic arbitrage conditions"""

    @staticmethod
    def synthetic_profitable_scenario(
        stock_price: float = 100.0,
    ) -> Dict[int, MockTicker]:
        """
        Synthetic arbitrage scenario with profitable opportunity.
        Synthetic position: Sell Call + Buy Put at different strikes

        Setup: Stock @ $100, Call 100 / Put 99
        Expected: Good risk-reward ratio (max_profit/min_profit > threshold)
        """
        tickers = {}
        expiry_date = datetime.now() + timedelta(days=30)
        expiry = expiry_date.strftime("%Y%m%d")

        # Stock data
        stock_ticker = MarketDataGenerator.generate_stock_data(
            "TEST", stock_price, volume=1000000
        )
        tickers[stock_ticker.contract.conId] = stock_ticker

        # Synthetic position: Sell Call 100, Buy Put 99
        # Net credit = 5.50 - 4.80 = 0.70
        # Spread = 100 - 99 = 1.00
        # Min profit = 0.70 - 1.00 = -0.30 (max loss)
        # Max profit = (100 - 99) + (-0.30) = 0.70
        # Risk-reward ratio = 0.70 / 0.30 = 2.33 > threshold ✅

        call_100 = MarketDataGenerator.generate_option_data(
            "TEST", expiry, 100.0, "C", stock_price, 30
        )
        call_100.bid = 5.50  # Sell at bid
        call_100.ask = 5.70
        call_100.close = 5.60
        call_100.volume = 500
        tickers[call_100.contract.conId] = call_100

        put_99 = MarketDataGenerator.generate_option_data(
            "TEST", expiry, 99.0, "P", stock_price, 30
        )
        put_99.bid = 4.60
        put_99.ask = 4.80  # Buy at ask
        put_99.close = 4.70
        put_99.volume = 600
        tickers[put_99.contract.conId] = put_99

        return tickers

    @staticmethod
    def synthetic_poor_risk_reward() -> Dict[int, MockTicker]:
        """
        Synthetic scenario with poor risk-reward ratio.

        Setup: High cost synthetic position
        Expected: Risk-reward ratio below threshold
        """
        tickers = {}
        stock_price = 100.0
        expiry_date = datetime.now() + timedelta(days=30)
        expiry = expiry_date.strftime("%Y%m%d")

        # Stock data
        stock_ticker = MarketDataGenerator.generate_stock_data(
            "TEST", stock_price, volume=1000000
        )
        tickers[stock_ticker.contract.conId] = stock_ticker

        # Poor risk-reward: Low net credit
        # Net credit = 6.20 - 6.00 = 0.20
        # Spread = 100 - 99 = 1.00
        # Min profit = 0.20 - 1.00 = -0.80
        # Max profit = 1.00 + (-0.80) = 0.20
        # Risk-reward = 0.20 / 0.80 = 0.25 < threshold ❌

        call_100 = MarketDataGenerator.generate_option_data(
            "TEST", expiry, 100.0, "C", stock_price, 30
        )
        call_100.bid = 6.20  # Small credit
        call_100.ask = 6.40
        call_100.close = 6.30
        call_100.volume = 200
        tickers[call_100.contract.conId] = call_100

        put_99 = MarketDataGenerator.generate_option_data(
            "TEST", expiry, 99.0, "P", stock_price, 30
        )
        put_99.bid = 5.80
        put_99.ask = 6.00  # Almost same price
        put_99.close = 5.90
        put_99.volume = 150
        tickers[put_99.contract.conId] = put_99

        return tickers

    @staticmethod
    def synthetic_max_loss_exceeded() -> Dict[int, MockTicker]:
        """
        Synthetic scenario where max loss exceeds threshold.

        Setup: High cost position with large potential loss
        Expected: Rejected due to max_loss_threshold
        """
        tickers = {}
        stock_price = 150.0
        expiry_date = datetime.now() + timedelta(days=30)
        expiry = expiry_date.strftime("%Y%m%d")

        # Stock data
        stock_ticker = MarketDataGenerator.generate_stock_data(
            "TEST", stock_price, volume=800000
        )
        tickers[stock_ticker.contract.conId] = stock_ticker

        # High max loss scenario
        # Net credit = 3.00 - 12.00 = -9.00 (actually net debit)
        # Let's fix: Net credit = 12.00 - 11.00 = 1.00
        # Spread = 150 - 140 = 10.00
        # Min profit = 1.00 - 10.00 = -9.00 < max_loss_threshold (-5.00) ❌

        call_150 = MarketDataGenerator.generate_option_data(
            "TEST", expiry, 150.0, "C", stock_price, 30
        )
        call_150.bid = 12.00  # Sell at bid
        call_150.ask = 12.20
        call_150.close = 12.10
        call_150.volume = 100
        tickers[call_150.contract.conId] = call_150

        put_140 = MarketDataGenerator.generate_option_data(
            "TEST", expiry, 140.0, "P", stock_price, 30
        )
        put_140.bid = 10.80
        put_140.ask = 11.00  # Buy at ask
        put_140.close = 10.90
        put_140.volume = 120
        tickers[put_140.contract.conId] = put_140

        return tickers

    @staticmethod
    def synthetic_max_profit_exceeded() -> Dict[int, MockTicker]:
        """
        Synthetic scenario where max profit exceeds threshold.

        Setup: Position with very high profit potential
        Expected: Rejected due to max_profit_threshold
        """
        tickers = {}
        stock_price = 50.0
        expiry_date = datetime.now() + timedelta(days=45)
        expiry = expiry_date.strftime("%Y%m%d")

        # Stock data
        stock_ticker = MarketDataGenerator.generate_stock_data(
            "TEST", stock_price, volume=1500000
        )
        tickers[stock_ticker.contract.conId] = stock_ticker

        # Very favorable position with high max profit
        # Using strikes 55 (call) and 45 (put) for synthetic
        # Net credit = 8.00 - 0.50 = 7.50
        # Spread = 50 - 45 = 5.00
        # Min profit = 7.50 - 5.00 = 2.50
        # Max profit = (55 - 45) + 2.50 = 12.50 > max_profit_threshold (10.00) ❌

        call_55 = MarketDataGenerator.generate_option_data(
            "TEST", expiry, 55.0, "C", stock_price, 45
        )
        call_55.bid = 0.30
        call_55.ask = 0.50  # Cheap OTM call
        call_55.close = 0.40
        call_55.volume = 1000
        tickers[call_55.contract.conId] = call_55

        put_45 = MarketDataGenerator.generate_option_data(
            "TEST", expiry, 45.0, "P", stock_price, 45
        )
        put_45.bid = 8.00  # Expensive ITM put - SELL at bid
        put_45.ask = 8.20
        put_45.close = 8.10
        put_45.volume = 800
        tickers[put_45.contract.conId] = put_45

        # For synthetic: Sell Call 55 at 0.30, Buy Put 45 at 8.20
        # Net credit = 0.30 - 8.20 = -7.90 (net debit)
        # This is wrong, let's fix it to make it a proper synthetic
        # Actually for a high max profit scenario:
        # Sell expensive ITM put, buy cheap OTM call
        put_45.bid = 8.00  # Sell at bid
        put_45.ask = 8.20
        call_55.bid = 0.30
        call_55.ask = 0.50  # Buy at ask
        # Net credit = 8.00 - 0.50 = 7.50

        return tickers

    @staticmethod
    def synthetic_negative_credit() -> Dict[int, MockTicker]:
        """
        Synthetic scenario with negative net credit.

        Setup: Call price > Put price
        Expected: Rejected due to negative net credit
        """
        tickers = {}
        stock_price = 100.0
        expiry_date = datetime.now() + timedelta(days=30)
        expiry = expiry_date.strftime("%Y%m%d")

        # Stock data
        stock_ticker = MarketDataGenerator.generate_stock_data(
            "TEST", stock_price, volume=900000
        )
        tickers[stock_ticker.contract.conId] = stock_ticker

        # Negative net credit
        # Net credit = 2.00 - 6.00 = -4.00 < 0 ❌

        call_100 = MarketDataGenerator.generate_option_data(
            "TEST", expiry, 100.0, "C", stock_price, 30
        )
        call_100.bid = 2.00  # Cheap call (sell)
        call_100.ask = 2.20
        call_100.close = 2.10
        call_100.volume = 300
        tickers[call_100.contract.conId] = call_100

        put_100 = MarketDataGenerator.generate_option_data(
            "TEST", expiry, 100.0, "P", stock_price, 30
        )
        put_100.bid = 5.80
        put_100.ask = 6.00  # Expensive put (buy)
        put_100.close = 5.90
        put_100.volume = 250
        tickers[put_100.contract.conId] = put_100

        return tickers

    @staticmethod
    def synthetic_multi_symbol_scenarios() -> Dict[str, Dict[int, MockTicker]]:
        """
        Multi-symbol scenarios for synthetic arbitrage testing.

        Returns:
            Dict mapping symbols to their market data
        """
        scenarios = {}

        # AAPL: Profitable synthetic with good risk-reward
        aapl_price = 185.50
        aapl_tickers = {}
        expiry_date = datetime.now() + timedelta(days=35)
        expiry = expiry_date.strftime("%Y%m%d")

        stock = MarketDataGenerator.generate_stock_data(
            "AAPL", aapl_price, volume=10000000
        )
        aapl_tickers[stock.contract.conId] = stock

        # Strikes: 185 (put), 186 (call)
        # For profitable synthetic: Sell Call 186, Buy Put 185
        # Net credit = 4.20 - 3.80 = 0.40
        # Spread = 185.50 - 185 = 0.50
        # Min profit = 0.40 - 0.50 = -0.10 (small loss)
        # Max profit = (186 - 185) + (-0.10) = 0.90
        # Risk-reward = 0.90 / 0.10 = 9.0 (excellent) ✅

        call_186 = MarketDataGenerator.generate_option_data(
            "AAPL", expiry, 186.0, "C", aapl_price, 35
        )
        call_186.bid = 4.20  # Sell at bid
        call_186.ask = 4.40
        call_186.close = 4.30
        call_186.volume = 2000
        aapl_tickers[call_186.contract.conId] = call_186

        put_185 = MarketDataGenerator.generate_option_data(
            "AAPL", expiry, 185.0, "P", aapl_price, 35
        )
        put_185.bid = 3.60
        put_185.ask = 3.80  # Buy at ask
        put_185.close = 3.70
        put_185.volume = 1800
        aapl_tickers[put_185.contract.conId] = put_185

        scenarios["AAPL"] = aapl_tickers

        # MSFT: Poor risk-reward ratio
        msft_price = 415.75
        msft_tickers = {}

        stock = MarketDataGenerator.generate_stock_data(
            "MSFT", msft_price, volume=8000000
        )
        msft_tickers[stock.contract.conId] = stock

        # High cost synthetic with poor risk-reward
        # Net credit = 11.00 - 12.70 = -1.70 (net debit - actually bad)
        # Let's fix: Net credit = 12.50 - 11.20 = 1.30
        # Spread = 415.75 - 415 = 0.75
        # Min profit = 1.30 - 0.75 = 0.55
        # Max profit = (416 - 415) + 0.55 = 1.55
        # Risk-reward = 1.55 / 0.55 = 2.82 (just ok, but we'll set threshold higher)
        # Actually, let's make it worse:
        # Net credit = 12.50 - 12.20 = 0.30
        # Min profit = 0.30 - 0.75 = -0.45
        # Max profit = 1.00 + (-0.45) = 0.55
        # Risk-reward = 0.55 / 0.45 = 1.22 < 2.5 threshold ❌

        call_416 = MarketDataGenerator.generate_option_data(
            "MSFT", expiry, 416.0, "C", msft_price, 35
        )
        call_416.bid = 12.50  # Sell at bid
        call_416.ask = 12.70
        call_416.close = 12.60
        call_416.volume = 500
        msft_tickers[call_416.contract.conId] = call_416

        put_415 = MarketDataGenerator.generate_option_data(
            "MSFT", expiry, 415.0, "P", msft_price, 35
        )
        put_415.bid = 12.00
        put_415.ask = 12.20  # Buy at ask (expensive)
        put_415.close = 12.10
        put_415.volume = 450
        msft_tickers[put_415.contract.conId] = put_415

        scenarios["MSFT"] = msft_tickers

        # TSLA: Max loss exceeded
        tsla_price = 245.80
        tsla_tickers = {}

        stock = MarketDataGenerator.generate_stock_data(
            "TSLA", tsla_price, volume=15000000
        )
        tsla_tickers[stock.contract.conId] = stock

        # High cost position with large max loss
        # Net credit = 8.50 - 18.20 = -9.70 (net debit - bad)
        # Let's fix: Net credit = 18.00 - 8.70 = 9.30
        # But for max loss exceeded, we need strikes with larger spread
        # Using 246 call and 240 put
        # Spread = 245.80 - 240 = 5.80
        # Min profit = 9.30 - 5.80 = 3.50 (actually profitable)
        # Let's adjust: Net credit = 8.50 - 7.00 = 1.50
        # Min profit = 1.50 - 5.80 = -4.30
        # But we want larger loss, so use wider strikes: 246 call, 235 put
        # Spread = 245.80 - 235 = 10.80
        # Min profit = 1.50 - 10.80 = -9.30 < -5.00 threshold ❌

        call_246 = MarketDataGenerator.generate_option_data(
            "TSLA", expiry, 246.0, "C", tsla_price, 35
        )
        call_246.bid = 8.50  # Sell at bid
        call_246.ask = 8.70
        call_246.close = 8.60
        call_246.volume = 1200
        tsla_tickers[call_246.contract.conId] = call_246

        put_235 = MarketDataGenerator.generate_option_data(
            "TSLA", expiry, 235.0, "P", tsla_price, 35
        )
        put_235.bid = 6.80
        put_235.ask = 7.00  # Buy at ask
        put_235.close = 6.90
        put_235.volume = 1000
        tsla_tickers[put_235.contract.conId] = put_235

        scenarios["TSLA"] = tsla_tickers

        return scenarios
