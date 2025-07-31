import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import logging
import numpy as np
from eventkit import Event
from ib_async import IB, ComboLeg, Contract, Option, Order, Ticker

from modules.Arbitrage.Strategy import ArbitrageClass, BaseExecutor, OrderManagerClass

from .common import configure_logging, get_logger
from .metrics import RejectionReason, metrics_collector

# Global variable to store debug mode
_debug_mode = False

# Configure logging will be done in main
logger = get_logger()

# Global contract_ticker for use in CalendarSpreadExecutor and patching in tests
contract_ticker = {}


@dataclass
class CalendarSpreadLeg:
    """Data class to hold calendar spread leg information"""

    contract: Contract
    strike: float
    expiry: str
    right: str  # 'C' for call, 'P' for put
    price: float
    bid: float
    ask: float
    volume: int
    iv: float  # Implied volatility
    theta: float  # Time decay
    days_to_expiry: int


@dataclass
class CalendarSpreadOpportunity:
    """Data class to hold complete calendar spread opportunity"""

    symbol: str
    strike: float
    option_type: str  # 'CALL' or 'PUT'
    front_leg: CalendarSpreadLeg
    back_leg: CalendarSpreadLeg

    # Spread metrics
    iv_spread: float  # Back IV - Front IV (%)
    theta_ratio: float  # Front theta / Back theta
    net_debit: float  # Cost to enter position
    max_profit: float  # Maximum theoretical profit
    max_loss: float  # Maximum loss (net debit)

    # Quality metrics
    front_bid_ask_spread: float
    back_bid_ask_spread: float
    combined_liquidity_score: float
    term_structure_inversion: bool

    # Greeks analysis
    net_delta: float
    net_gamma: float
    net_vega: float

    # Scoring
    composite_score: float


@dataclass
class CalendarSpreadConfig:
    """Configuration for calendar spread detection"""

    min_iv_spread: float = 1.5  # Minimum IV spread (%)
    min_theta_ratio: float = 1.5  # Minimum theta ratio (front/back)
    max_bid_ask_spread: float = 0.15  # Maximum bid-ask spread as % of mid
    min_liquidity_score: float = 0.4  # Minimum liquidity threshold
    max_days_front: int = 45  # Maximum days to front expiry
    min_days_back: int = 60  # Minimum days to back expiry
    max_days_back: int = 120  # Maximum days to back expiry
    min_volume: int = 10  # Minimum daily volume per leg
    max_net_debit: float = 500.0  # Maximum cost to enter position
    target_profit_ratio: float = 0.3  # Target profit as % of max profit


class CalendarSpreadExecutor(BaseExecutor):
    """
    Calendar Spread Executor class that handles the execution of calendar spread strategies.

    Calendar spreads profit from time decay differential between front and back months,
    particularly when the front month decays faster due to higher implied volatility
    or term structure inversion.

    Key Strategy Elements:
    1. IV Spread Analysis - Back month IV should exceed front month by significant margin
    2. Theta Ratio Analysis - Front month should decay faster than back month
    3. Term Structure Analysis - Detect inversion opportunities
    4. Greeks Risk Assessment - Monitor delta, gamma, vega exposure
    5. Quality Filters - Ensure adequate liquidity and tight spreads
    """

    def __init__(
        self,
        ib: IB,
        order_manager: OrderManagerClass,
        stock_contract: Contract,
        opportunities: List[CalendarSpreadOpportunity],
        symbol: str,
        config: CalendarSpreadConfig,
        start_time: float,
        quantity: int = 1,
        data_timeout: float = 30.0,
    ) -> None:
        """
        Initialize the Calendar Spread Executor.

        Args:
            ib: Interactive Brokers connection instance
            order_manager: Manager for handling order execution
            stock_contract: The underlying stock contract
            opportunities: List of calendar spread opportunities
            symbol: Trading symbol
            config: Calendar spread configuration
            start_time: Start time of the execution
            quantity: Quantity of spreads to execute
            data_timeout: Maximum time to wait for all contract data
        """
        # Collect all option contracts
        option_contracts = []
        for opp in opportunities:
            option_contracts.extend([opp.front_leg.contract, opp.back_leg.contract])

        super().__init__(
            ib,
            order_manager,
            stock_contract,
            option_contracts,
            symbol,
            config.max_net_debit,
            "",  # expiry not applicable for calendar spreads
            start_time,
        )

        self.opportunities = opportunities
        self.config = config
        self.quantity = quantity
        self.data_timeout = data_timeout
        self.data_collection_start = None

    def _calculate_days_to_expiry(self, expiry_str: str) -> int:
        """Calculate days to expiry from expiry string"""
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y%m%d")
            today = datetime.now().date()
            return (expiry_date.date() - today).days
        except ValueError:
            logger.warning(f"Invalid expiry format: {expiry_str}")
            return 30  # Default assumption

    def _calculate_liquidity_score(self, leg: CalendarSpreadLeg) -> float:
        """Calculate liquidity score for a calendar spread leg"""
        if leg.volume <= 0:
            return 0.0

        # Base score from volume (normalized)
        volume_score = min(leg.volume / 100.0, 1.0)  # Cap at 100 volume = 1.0

        # Penalty for wide bid-ask spreads
        mid_price = (leg.bid + leg.ask) / 2.0 if leg.ask > 0 else leg.price
        if mid_price > 0:
            spread_penalty = min((leg.ask - leg.bid) / mid_price, 0.5)
        else:
            spread_penalty = 0.5

        spread_score = max(0.0, 1.0 - spread_penalty * 2.0)

        # Combined score
        return volume_score * 0.6 + spread_score * 0.4

    def _detect_term_structure_inversion(
        self, front_iv: float, back_iv: float, front_days: int, back_days: int
    ) -> bool:
        """
        Detect term structure inversion - when shorter expiry has higher IV than longer.
        This creates favorable conditions for calendar spreads.
        """
        if front_days >= back_days:
            return False  # Invalid: front should be shorter

        # Normalize IVs by time to expiry for comparison
        front_iv_annual = (
            front_iv * np.sqrt(365.0 / front_days) if front_days > 0 else front_iv
        )
        back_iv_annual = (
            back_iv * np.sqrt(365.0 / back_days) if back_days > 0 else back_iv
        )

        # Inversion occurs when front month IV exceeds back month
        return front_iv_annual > back_iv_annual

    def _calculate_theoretical_max_profit(
        self, strike: float, front_price: float, back_price: float, front_days: int
    ) -> float:
        """
        Calculate theoretical maximum profit for calendar spread.
        Max profit typically occurs when stock is at strike at front expiry.
        """
        # Simplified calculation - maximum occurs when front option expires worthless
        # and back option retains most of its time value
        net_debit = back_price - front_price  # We buy back, sell front

        # Estimate back option value at front expiry (simplified)
        remaining_time_value = back_price * 0.6  # Rough estimate

        return max(0.0, remaining_time_value - net_debit)

    def _build_calendar_spread_order(
        self, opportunity: CalendarSpreadOpportunity, quantity: int
    ) -> Tuple[Contract, Order]:
        """
        Build calendar spread order with front and back month legs.
        Calendar spread: Sell front month, Buy back month (net debit position)
        """
        # Front leg - SELL (shorter expiry)
        front_leg = ComboLeg(
            conId=opportunity.front_leg.contract.conId,
            ratio=1,
            action="SELL",
            exchange="SMART",
        )

        # Back leg - BUY (longer expiry)
        back_leg = ComboLeg(
            conId=opportunity.back_leg.contract.conId,
            ratio=1,
            action="BUY",
            exchange="SMART",
        )

        # Create calendar spread contract
        calendar_contract = Contract(
            symbol=opportunity.symbol,
            comboLegs=[front_leg, back_leg],
            exchange="SMART",
            secType="BAG",
            currency="USD",
        )

        # Calculate limit price (net debit)
        limit_price = opportunity.net_debit

        order = Order(
            orderId=self.ib.client.getReqId(),
            orderType="LMT",
            action="BUY",  # Buying the spread (net debit)
            totalQuantity=quantity,
            lmtPrice=limit_price,
            tif="DAY",
        )

        logger.info(
            f"Calendar spread order: {opportunity.option_type} {opportunity.strike} "
            f"Front: {opportunity.front_leg.expiry} Back: {opportunity.back_leg.expiry} "
            f"Net debit: ${limit_price:.2f}"
        )

        return calendar_contract, order

    def _log_calendar_spread_details(
        self, opportunity: CalendarSpreadOpportunity
    ) -> None:
        """Log detailed information about the calendar spread opportunity"""
        logger.info(f"=== Calendar Spread Opportunity: {opportunity.symbol} ===")
        logger.info(f"Strike: {opportunity.strike} | Type: {opportunity.option_type}")
        logger.info(
            f"Front: {opportunity.front_leg.expiry} ({opportunity.front_leg.days_to_expiry}d) "
            f"IV: {opportunity.front_leg.iv:.1f}% Theta: {opportunity.front_leg.theta:.3f}"
        )
        logger.info(
            f"Back: {opportunity.back_leg.expiry} ({opportunity.back_leg.days_to_expiry}d) "
            f"IV: {opportunity.back_leg.iv:.1f}% Theta: {opportunity.back_leg.theta:.3f}"
        )
        logger.info(
            f"IV Spread: {opportunity.iv_spread:.1f}% | Theta Ratio: {opportunity.theta_ratio:.2f}"
        )
        logger.info(
            f"Net Debit: ${opportunity.net_debit:.2f} | Max Profit: ${opportunity.max_profit:.2f}"
        )
        logger.info(
            f"Max Loss: ${opportunity.max_loss:.2f} | Score: {opportunity.composite_score:.3f}"
        )
        logger.info(f"Term Structure Inversion: {opportunity.term_structure_inversion}")

    async def executor(self, event: Event) -> None:
        """
        Main executor logic for calendar spread strategy.
        Monitors market data and executes calendar spreads when conditions are met.
        """
        try:
            if not self.is_active:
                return

            # Initialize data collection timestamp
            if self.data_collection_start is None:
                self.data_collection_start = time.time()
                logger.info(f"[{self.symbol}] Starting calendar spread data collection")

            # Check for data timeout
            elapsed_time = time.time() - self.data_collection_start
            if elapsed_time > self.data_timeout:
                logger.warning(
                    f"[{self.symbol}] Data collection timeout ({elapsed_time:.1f}s), "
                    "deactivating calendar spread executor"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.DATA_TIMEOUT,
                    {"symbol": self.symbol, "timeout_seconds": elapsed_time},
                )
                self.deactivate()
                return

            # Process each opportunity
            for opportunity in self.opportunities:
                if not self.is_active:
                    break

                try:
                    # Get fresh market data for both legs
                    front_ticker = contract_ticker.get(
                        opportunity.front_leg.contract.conId
                    )
                    back_ticker = contract_ticker.get(
                        opportunity.back_leg.contract.conId
                    )

                    if not front_ticker or not back_ticker:
                        continue  # Wait for data

                    # Update opportunity with current market data
                    self._update_opportunity_with_market_data(
                        opportunity, front_ticker, back_ticker
                    )

                    # Check if opportunity still meets criteria
                    if not self._validate_opportunity(opportunity):
                        continue

                    # Execute the calendar spread
                    await self._execute_calendar_spread(opportunity)

                    # Only execute one opportunity per cycle
                    break

                except Exception as e:
                    logger.error(
                        f"Error processing calendar spread opportunity: {str(e)}"
                    )
                    continue

        except Exception as e:
            logger.error(f"Critical error in calendar spread executor: {str(e)}")
            self.deactivate()

    def _update_opportunity_with_market_data(
        self,
        opportunity: CalendarSpreadOpportunity,
        front_ticker: Ticker,
        back_ticker: Ticker,
    ) -> None:
        """Update opportunity with current market data"""
        # Update front leg
        opportunity.front_leg.price = (
            front_ticker.midpoint()
            if not np.isnan(front_ticker.midpoint())
            else front_ticker.close
        )
        opportunity.front_leg.bid = (
            front_ticker.bid if not np.isnan(front_ticker.bid) else 0.0
        )
        opportunity.front_leg.ask = (
            front_ticker.ask if not np.isnan(front_ticker.ask) else 0.0
        )

        # Update back leg
        opportunity.back_leg.price = (
            back_ticker.midpoint()
            if not np.isnan(back_ticker.midpoint())
            else back_ticker.close
        )
        opportunity.back_leg.bid = (
            back_ticker.bid if not np.isnan(back_ticker.bid) else 0.0
        )
        opportunity.back_leg.ask = (
            back_ticker.ask if not np.isnan(back_ticker.ask) else 0.0
        )

        # Recalculate net debit (back - front for calendar spread)
        opportunity.net_debit = opportunity.back_leg.price - opportunity.front_leg.price
        opportunity.max_loss = opportunity.net_debit  # Maximum loss is net debit paid

    def _validate_opportunity(self, opportunity: CalendarSpreadOpportunity) -> bool:
        """Validate that opportunity still meets all criteria"""
        # Check IV spread
        if opportunity.iv_spread < self.config.min_iv_spread:
            metrics_collector.add_rejection_reason(
                RejectionReason.INSUFFICIENT_IV_SPREAD,
                {"symbol": self.symbol, "iv_spread": opportunity.iv_spread},
            )
            return False

        # Check theta ratio
        if opportunity.theta_ratio < self.config.min_theta_ratio:
            metrics_collector.add_rejection_reason(
                RejectionReason.INSUFFICIENT_THETA_RATIO,
                {"symbol": self.symbol, "theta_ratio": opportunity.theta_ratio},
            )
            return False

        # Check net debit limit
        if opportunity.net_debit > self.config.max_net_debit:
            metrics_collector.add_rejection_reason(
                RejectionReason.COST_LIMIT_EXCEEDED,
                {"symbol": self.symbol, "net_debit": opportunity.net_debit},
            )
            return False

        # Check bid-ask spreads
        front_spread_pct = self._calculate_spread_percentage(opportunity.front_leg)
        back_spread_pct = self._calculate_spread_percentage(opportunity.back_leg)

        if (
            front_spread_pct > self.config.max_bid_ask_spread
            or back_spread_pct > self.config.max_bid_ask_spread
        ):
            metrics_collector.add_rejection_reason(
                RejectionReason.WIDE_BID_ASK_SPREAD,
                {
                    "symbol": self.symbol,
                    "front_spread_pct": front_spread_pct,
                    "back_spread_pct": back_spread_pct,
                },
            )
            return False

        # Check liquidity
        if opportunity.combined_liquidity_score < self.config.min_liquidity_score:
            metrics_collector.add_rejection_reason(
                RejectionReason.INSUFFICIENT_LIQUIDITY,
                {
                    "symbol": self.symbol,
                    "liquidity_score": opportunity.combined_liquidity_score,
                },
            )
            return False

        return True

    def _calculate_spread_percentage(self, leg: CalendarSpreadLeg) -> float:
        """Calculate bid-ask spread as percentage of mid price"""
        if leg.ask <= leg.bid or leg.ask <= 0:
            return 1.0  # Invalid spread, return high penalty

        mid_price = (leg.bid + leg.ask) / 2.0
        if mid_price <= 0:
            return 1.0

        return (leg.ask - leg.bid) / mid_price

    async def _execute_calendar_spread(
        self, opportunity: CalendarSpreadOpportunity
    ) -> None:
        """Execute the calendar spread order"""
        try:
            # Log trade details
            self._log_calendar_spread_details(opportunity)

            # Build order
            calendar_contract, order = self._build_calendar_spread_order(
                opportunity, self.quantity
            )

            # Place order
            trade = await self.order_manager.place_order(calendar_contract, order)

            if trade:
                logger.info(
                    f"[{self.symbol}] Calendar spread order placed successfully"
                )
                # Deactivate after successful execution
                self.deactivate()
            else:
                logger.warning(
                    f"[{self.symbol}] Calendar spread order placement failed"
                )

        except Exception as e:
            logger.error(f"Error executing calendar spread: {str(e)}")


class CalendarSpread(ArbitrageClass):
    """
    Calendar Spread arbitrage strategy class.

    Calendar spreads profit from the differential time decay between front and back month options.
    The strategy is most profitable when:
    1. Front month has significantly higher IV than back month
    2. Front month decays faster (higher theta ratio)
    3. Term structure shows inversion (front > back IV)
    4. Stock remains near strike price at front expiry

    This implementation includes:
    - Advanced IV spread analysis
    - Theta ratio calculations
    - Term structure inversion detection
    - Greeks-based risk assessment
    - Comprehensive quality filters
    - Performance optimization with caching
    """

    def __init__(self, log_file: str = None) -> None:
        """Initialize Calendar Spread strategy"""
        super().__init__(log_file)
        self.config = CalendarSpreadConfig()

        # Calendar spread specific caching
        self.iv_cache = {}  # Cache for implied volatility calculations
        self.greeks_cache = {}  # Cache for Greeks calculations
        self.cache_ttl = 60  # 1 minute TTL for option Greeks

        logger.info("Calendar Spread strategy initialized")

    def _calculate_implied_volatility(
        self, ticker: Ticker, option_contract: Contract
    ) -> float:
        """
        Calculate or retrieve cached implied volatility using IB API data.

        Uses multiple IV sources from IB API with fallback hierarchy:
        1. ticker.impliedVolatility (direct IV from IB)
        2. ticker.modelGreeks.impliedVol (model-based IV)
        3. Average of bid/ask Greeks IV
        4. Fallback to bid-ask spread estimation
        """
        cache_key = f"{option_contract.conId}_{ticker.time}"

        if cache_key in self.iv_cache:
            return self.iv_cache[cache_key]

        iv_value = None
        iv_source = "unknown"

        try:
            # Priority 1: Direct implied volatility from IB API
            if (
                hasattr(ticker, "impliedVolatility")
                and ticker.impliedVolatility is not None
            ):
                if (
                    not np.isnan(ticker.impliedVolatility)
                    and ticker.impliedVolatility > 0
                ):
                    iv_value = ticker.impliedVolatility * 100.0  # Convert to percentage
                    iv_source = "direct_iv"

            # Priority 2: Model Greeks implied volatility
            elif hasattr(ticker, "modelGreeks") and ticker.modelGreeks is not None:
                if (
                    hasattr(ticker.modelGreeks, "impliedVol")
                    and ticker.modelGreeks.impliedVol is not None
                ):
                    if (
                        not np.isnan(ticker.modelGreeks.impliedVol)
                        and ticker.modelGreeks.impliedVol > 0
                    ):
                        iv_value = (
                            ticker.modelGreeks.impliedVol * 100.0
                        )  # Convert to percentage
                        iv_source = "model_greeks"

            # Priority 3: Average of bid/ask Greeks IV
            elif (
                hasattr(ticker, "bidGreeks")
                and ticker.bidGreeks is not None
                and hasattr(ticker, "askGreeks")
                and ticker.askGreeks is not None
            ):
                bid_iv = None
                ask_iv = None

                if (
                    hasattr(ticker.bidGreeks, "impliedVol")
                    and ticker.bidGreeks.impliedVol is not None
                    and not np.isnan(ticker.bidGreeks.impliedVol)
                    and ticker.bidGreeks.impliedVol > 0
                ):
                    bid_iv = ticker.bidGreeks.impliedVol * 100.0

                if (
                    hasattr(ticker.askGreeks, "impliedVol")
                    and ticker.askGreeks.impliedVol is not None
                    and not np.isnan(ticker.askGreeks.impliedVol)
                    and ticker.askGreeks.impliedVol > 0
                ):
                    ask_iv = ticker.askGreeks.impliedVol * 100.0

                if bid_iv is not None and ask_iv is not None:
                    iv_value = (bid_iv + ask_iv) / 2.0
                    iv_source = "bid_ask_average"
                elif bid_iv is not None:
                    iv_value = bid_iv
                    iv_source = "bid_greeks"
                elif ask_iv is not None:
                    iv_value = ask_iv
                    iv_source = "ask_greeks"

            # Priority 4: Single Greeks source (bid, ask, or last)
            if iv_value is None:
                for greeks_attr in ["lastGreeks", "bidGreeks", "askGreeks"]:
                    if hasattr(ticker, greeks_attr):
                        greeks = getattr(ticker, greeks_attr)
                        if (
                            greeks is not None
                            and hasattr(greeks, "impliedVol")
                            and greeks.impliedVol is not None
                            and not np.isnan(greeks.impliedVol)
                            and greeks.impliedVol > 0
                        ):
                            iv_value = greeks.impliedVol * 100.0
                            iv_source = greeks_attr.replace("Greeks", "_greeks")
                            break

        except (AttributeError, TypeError) as e:
            logger.debug(
                f"Error accessing IB IV data for {option_contract.symbol}: {e}"
            )

        # Fallback to bid-ask spread estimation if no IB IV available
        if iv_value is None:
            if ticker.ask > ticker.bid > 0:
                spread_ratio = (ticker.ask - ticker.bid) / ticker.midpoint()
                iv_value = min(100.0, max(10.0, spread_ratio * 200.0))  # 10%-100% range
                iv_source = "spread_estimation"
            else:
                iv_value = 25.0  # Default 25% IV
                iv_source = "default"

        # Validate IV is within reasonable range (5% - 200%)
        if iv_value is not None:
            iv_value = max(5.0, min(200.0, iv_value))
        else:
            iv_value = 25.0
            iv_source = "fallback_default"

        # Log IV source for debugging (only at debug level to avoid spam)
        logger.debug(
            f"IV for {option_contract.symbol} strike {option_contract.strike}: "
            f"{iv_value:.1f}% (source: {iv_source})"
        )

        self.iv_cache[cache_key] = iv_value
        return iv_value

    def _calculate_theta(
        self, ticker: Ticker, option_contract: Contract, days_to_expiry: int
    ) -> float:
        """
        Calculate or retrieve cached theta (time decay) using IB API data.

        Uses multiple theta sources from IB API with fallback hierarchy:
        1. ticker.modelGreeks.theta (model-based theta from IB)
        2. Average of bid/ask Greeks theta
        3. Single Greeks source (lastGreeks, bidGreeks, askGreeks)
        4. Fallback to time-based estimation
        """
        cache_key = (
            f"{option_contract.conId}_theta_{getattr(ticker, 'time', time.time())}"
        )

        if cache_key in self.greeks_cache:
            cached_theta, cache_time = self.greeks_cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return cached_theta

        theta_value = None
        theta_source = "unknown"

        try:
            # Priority 1: Model Greeks theta from IB API
            if hasattr(ticker, "modelGreeks") and ticker.modelGreeks is not None:
                if (
                    hasattr(ticker.modelGreeks, "theta")
                    and ticker.modelGreeks.theta is not None
                ):
                    if not np.isnan(ticker.modelGreeks.theta):
                        theta_value = ticker.modelGreeks.theta
                        theta_source = "model_greeks"

            # Priority 2: Average of bid/ask Greeks theta
            elif (
                hasattr(ticker, "bidGreeks")
                and ticker.bidGreeks is not None
                and hasattr(ticker, "askGreeks")
                and ticker.askGreeks is not None
            ):
                bid_theta = None
                ask_theta = None

                if (
                    hasattr(ticker.bidGreeks, "theta")
                    and ticker.bidGreeks.theta is not None
                    and not np.isnan(ticker.bidGreeks.theta)
                ):
                    bid_theta = ticker.bidGreeks.theta

                if (
                    hasattr(ticker.askGreeks, "theta")
                    and ticker.askGreeks.theta is not None
                    and not np.isnan(ticker.askGreeks.theta)
                ):
                    ask_theta = ticker.askGreeks.theta

                if bid_theta is not None and ask_theta is not None:
                    theta_value = (bid_theta + ask_theta) / 2.0
                    theta_source = "bid_ask_average"
                elif bid_theta is not None:
                    theta_value = bid_theta
                    theta_source = "bid_greeks"
                elif ask_theta is not None:
                    theta_value = ask_theta
                    theta_source = "ask_greeks"

            # Priority 3: Single Greeks source (last, bid, or ask)
            if theta_value is None:
                for greeks_attr in ["lastGreeks", "bidGreeks", "askGreeks"]:
                    if hasattr(ticker, greeks_attr):
                        greeks = getattr(ticker, greeks_attr)
                        if (
                            greeks is not None
                            and hasattr(greeks, "theta")
                            and greeks.theta is not None
                            and not np.isnan(greeks.theta)
                        ):
                            theta_value = greeks.theta
                            theta_source = greeks_attr.replace("Greeks", "_greeks")
                            break

        except (AttributeError, TypeError) as e:
            logger.debug(
                f"Error accessing IB theta data for {option_contract.symbol}: {e}"
            )

        # Priority 4: Fallback to time-based estimation if no IB theta available
        if theta_value is None:
            if days_to_expiry <= 0:
                theta_value = 0.0
                theta_source = "expired"
            else:
                # Rough theta estimate based on time to expiry and option price
                option_price = (
                    ticker.midpoint()
                    if not np.isnan(ticker.midpoint())
                    else ticker.close
                )
                if option_price and option_price > 0:
                    time_factor = max(
                        1.0, np.sqrt(days_to_expiry / 30.0)
                    )  # Normalized to 30 days
                    # Theta is typically negative (time decay), more negative closer to expiry
                    theta_value = -(option_price / time_factor) * 0.05  # Rough estimate
                    theta_source = "time_estimation"
                else:
                    theta_value = -0.05  # Default small negative theta
                    theta_source = "default"

        # Validate theta (should be negative for long options, representing time decay)
        if theta_value is not None:
            # Reasonable theta range: -10.0 to +0.1 (negative for time decay)
            theta_value = max(-10.0, min(0.1, theta_value))
        else:
            theta_value = -0.05  # Default fallback
            theta_source = "fallback_default"

        # Log theta source for debugging (only at debug level to avoid spam)
        logger.debug(
            f"Theta for {option_contract.symbol} strike {option_contract.strike}: "
            f"{theta_value:.4f} (source: {theta_source})"
        )

        # Cache the result
        self.greeks_cache[cache_key] = (theta_value, time.time())
        return theta_value

    def _calculate_delta(self, ticker: Ticker, option_contract: Contract) -> float:
        """
        Calculate delta (price sensitivity to underlying) using IB API data.

        Uses multiple delta sources from IB API with fallback hierarchy:
        1. ticker.modelGreeks.delta (model-based delta from IB)
        2. Average of bid/ask Greeks delta
        3. Single Greeks source (lastGreeks, bidGreeks, askGreeks)
        4. Fallback to ATM estimation based on option type
        """
        cache_key = (
            f"{option_contract.conId}_delta_{getattr(ticker, 'time', time.time())}"
        )

        if cache_key in self.greeks_cache:
            cached_delta, cache_time = self.greeks_cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return cached_delta

        delta_value = None
        delta_source = "unknown"

        try:
            # Priority 1: Model Greeks delta from IB API
            if hasattr(ticker, "modelGreeks") and ticker.modelGreeks is not None:
                if (
                    hasattr(ticker.modelGreeks, "delta")
                    and ticker.modelGreeks.delta is not None
                ):
                    if not np.isnan(ticker.modelGreeks.delta):
                        delta_value = ticker.modelGreeks.delta
                        delta_source = "model_greeks"

            # Priority 2: Average of bid/ask Greeks delta
            elif (
                hasattr(ticker, "bidGreeks")
                and ticker.bidGreeks is not None
                and hasattr(ticker, "askGreeks")
                and ticker.askGreeks is not None
            ):
                bid_delta = None
                ask_delta = None

                if (
                    hasattr(ticker.bidGreeks, "delta")
                    and ticker.bidGreeks.delta is not None
                    and not np.isnan(ticker.bidGreeks.delta)
                ):
                    bid_delta = ticker.bidGreeks.delta

                if (
                    hasattr(ticker.askGreeks, "delta")
                    and ticker.askGreeks.delta is not None
                    and not np.isnan(ticker.askGreeks.delta)
                ):
                    ask_delta = ticker.askGreeks.delta

                if bid_delta is not None and ask_delta is not None:
                    delta_value = (bid_delta + ask_delta) / 2.0
                    delta_source = "bid_ask_average"
                elif bid_delta is not None:
                    delta_value = bid_delta
                    delta_source = "bid_greeks"
                elif ask_delta is not None:
                    delta_value = ask_delta
                    delta_source = "ask_greeks"

            # Priority 3: Single Greeks source (last, bid, or ask)
            if delta_value is None:
                for greeks_attr in ["lastGreeks", "bidGreeks", "askGreeks"]:
                    if hasattr(ticker, greeks_attr):
                        greeks = getattr(ticker, greeks_attr)
                        if (
                            greeks is not None
                            and hasattr(greeks, "delta")
                            and greeks.delta is not None
                            and not np.isnan(greeks.delta)
                        ):
                            delta_value = greeks.delta
                            delta_source = greeks_attr.replace("Greeks", "_greeks")
                            break

        except (AttributeError, TypeError) as e:
            logger.debug(
                f"Error accessing IB delta data for {option_contract.symbol}: {e}"
            )

        # Priority 4: Fallback to ATM estimation if no IB delta available
        if delta_value is None:
            # Calls have positive delta, puts have negative delta
            if option_contract.right == "C":
                delta_value = 0.5  # Rough estimate for ATM call
                delta_source = "call_estimation"
            else:
                delta_value = -0.5  # Rough estimate for ATM put
                delta_source = "put_estimation"

        # Validate delta ranges
        if delta_value is not None:
            if option_contract.right == "C":
                # Calls: delta should be between 0.0 and 1.0
                delta_value = max(0.0, min(1.0, delta_value))
            else:
                # Puts: delta should be between -1.0 and 0.0
                delta_value = max(-1.0, min(0.0, delta_value))
        else:
            # Fallback default
            delta_value = 0.5 if option_contract.right == "C" else -0.5
            delta_source = "fallback_default"

        # Log delta source for debugging (only at debug level to avoid spam)
        logger.debug(
            f"Delta for {option_contract.symbol} {option_contract.right} strike {option_contract.strike}: "
            f"{delta_value:.4f} (source: {delta_source})"
        )

        # Cache the result
        self.greeks_cache[cache_key] = (delta_value, time.time())
        return delta_value

    def _calculate_gamma(self, ticker: Ticker, option_contract: Contract) -> float:
        """
        Calculate gamma (delta sensitivity) using IB API data.

        Uses multiple gamma sources from IB API with fallback hierarchy:
        1. ticker.modelGreeks.gamma (model-based gamma from IB)
        2. Average of bid/ask Greeks gamma
        3. Single Greeks source (lastGreeks, bidGreeks, askGreeks)
        4. Fallback to fixed estimation
        """
        cache_key = (
            f"{option_contract.conId}_gamma_{getattr(ticker, 'time', time.time())}"
        )

        if cache_key in self.greeks_cache:
            cached_gamma, cache_time = self.greeks_cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return cached_gamma

        gamma_value = None
        gamma_source = "unknown"

        try:
            # Priority 1: Model Greeks gamma from IB API
            if hasattr(ticker, "modelGreeks") and ticker.modelGreeks is not None:
                if (
                    hasattr(ticker.modelGreeks, "gamma")
                    and ticker.modelGreeks.gamma is not None
                ):
                    if not np.isnan(ticker.modelGreeks.gamma):
                        gamma_value = ticker.modelGreeks.gamma
                        gamma_source = "model_greeks"

            # Priority 2: Average of bid/ask Greeks gamma
            elif (
                hasattr(ticker, "bidGreeks")
                and ticker.bidGreeks is not None
                and hasattr(ticker, "askGreeks")
                and ticker.askGreeks is not None
            ):
                bid_gamma = None
                ask_gamma = None

                if (
                    hasattr(ticker.bidGreeks, "gamma")
                    and ticker.bidGreeks.gamma is not None
                    and not np.isnan(ticker.bidGreeks.gamma)
                ):
                    bid_gamma = ticker.bidGreeks.gamma

                if (
                    hasattr(ticker.askGreeks, "gamma")
                    and ticker.askGreeks.gamma is not None
                    and not np.isnan(ticker.askGreeks.gamma)
                ):
                    ask_gamma = ticker.askGreeks.gamma

                if bid_gamma is not None and ask_gamma is not None:
                    gamma_value = (bid_gamma + ask_gamma) / 2.0
                    gamma_source = "bid_ask_average"
                elif bid_gamma is not None:
                    gamma_value = bid_gamma
                    gamma_source = "bid_greeks"
                elif ask_gamma is not None:
                    gamma_value = ask_gamma
                    gamma_source = "ask_greeks"

            # Priority 3: Single Greeks source (last, bid, or ask)
            if gamma_value is None:
                for greeks_attr in ["lastGreeks", "bidGreeks", "askGreeks"]:
                    if hasattr(ticker, greeks_attr):
                        greeks = getattr(ticker, greeks_attr)
                        if (
                            greeks is not None
                            and hasattr(greeks, "gamma")
                            and greeks.gamma is not None
                            and not np.isnan(greeks.gamma)
                        ):
                            gamma_value = greeks.gamma
                            gamma_source = greeks_attr.replace("Greeks", "_greeks")
                            break

        except (AttributeError, TypeError) as e:
            logger.debug(
                f"Error accessing IB gamma data for {option_contract.symbol}: {e}"
            )

        # Priority 4: Fallback to fixed estimation if no IB gamma available
        if gamma_value is None:
            gamma_value = 0.05  # Rough estimate - gamma is typically positive for both calls and puts
            gamma_source = "fixed_estimation"

        # Validate gamma (should be positive for both calls and puts)
        if gamma_value is not None:
            # Reasonable gamma range: 0.0 to 1.0 (gamma is always positive)
            gamma_value = max(0.0, min(1.0, gamma_value))
        else:
            gamma_value = 0.05  # Default fallback
            gamma_source = "fallback_default"

        # Log gamma source for debugging (only at debug level to avoid spam)
        logger.debug(
            f"Gamma for {option_contract.symbol} strike {option_contract.strike}: "
            f"{gamma_value:.4f} (source: {gamma_source})"
        )

        # Cache the result
        self.greeks_cache[cache_key] = (gamma_value, time.time())
        return gamma_value

    def _calculate_vega(self, ticker: Ticker, option_contract: Contract) -> float:
        """
        Calculate vega (IV sensitivity) using IB API data.

        Uses multiple vega sources from IB API with fallback hierarchy:
        1. ticker.modelGreeks.vega (model-based vega from IB)
        2. Average of bid/ask Greeks vega
        3. Single Greeks source (lastGreeks, bidGreeks, askGreeks)
        4. Fallback to price-based estimation
        """
        cache_key = (
            f"{option_contract.conId}_vega_{getattr(ticker, 'time', time.time())}"
        )

        if cache_key in self.greeks_cache:
            cached_vega, cache_time = self.greeks_cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return cached_vega

        vega_value = None
        vega_source = "unknown"

        try:
            # Priority 1: Model Greeks vega from IB API
            if hasattr(ticker, "modelGreeks") and ticker.modelGreeks is not None:
                if (
                    hasattr(ticker.modelGreeks, "vega")
                    and ticker.modelGreeks.vega is not None
                ):
                    if not np.isnan(ticker.modelGreeks.vega):
                        vega_value = ticker.modelGreeks.vega
                        vega_source = "model_greeks"

            # Priority 2: Average of bid/ask Greeks vega
            elif (
                hasattr(ticker, "bidGreeks")
                and ticker.bidGreeks is not None
                and hasattr(ticker, "askGreeks")
                and ticker.askGreeks is not None
            ):
                bid_vega = None
                ask_vega = None

                if (
                    hasattr(ticker.bidGreeks, "vega")
                    and ticker.bidGreeks.vega is not None
                    and not np.isnan(ticker.bidGreeks.vega)
                ):
                    bid_vega = ticker.bidGreeks.vega

                if (
                    hasattr(ticker.askGreeks, "vega")
                    and ticker.askGreeks.vega is not None
                    and not np.isnan(ticker.askGreeks.vega)
                ):
                    ask_vega = ticker.askGreeks.vega

                if bid_vega is not None and ask_vega is not None:
                    vega_value = (bid_vega + ask_vega) / 2.0
                    vega_source = "bid_ask_average"
                elif bid_vega is not None:
                    vega_value = bid_vega
                    vega_source = "bid_greeks"
                elif ask_vega is not None:
                    vega_value = ask_vega
                    vega_source = "ask_greeks"

            # Priority 3: Single Greeks source (last, bid, or ask)
            if vega_value is None:
                for greeks_attr in ["lastGreeks", "bidGreeks", "askGreeks"]:
                    if hasattr(ticker, greeks_attr):
                        greeks = getattr(ticker, greeks_attr)
                        if (
                            greeks is not None
                            and hasattr(greeks, "vega")
                            and greeks.vega is not None
                            and not np.isnan(greeks.vega)
                        ):
                            vega_value = greeks.vega
                            vega_source = greeks_attr.replace("Greeks", "_greeks")
                            break

        except (AttributeError, TypeError) as e:
            logger.debug(
                f"Error accessing IB vega data for {option_contract.symbol}: {e}"
            )

        # Priority 4: Fallback to price-based estimation if no IB vega available
        if vega_value is None:
            option_price = (
                ticker.midpoint() if not np.isnan(ticker.midpoint()) else ticker.close
            )
            if option_price and option_price > 0:
                vega_value = option_price * 0.1  # Rough estimate - 10% of option price
                vega_source = "price_estimation"
            else:
                vega_value = 0.1  # Default small vega
                vega_source = "default"

        # Validate vega (should be positive for both calls and puts)
        if vega_value is not None:
            # Reasonable vega range: 0.0 to 100.0 (vega is always positive)
            vega_value = max(0.0, min(100.0, vega_value))
        else:
            vega_value = 0.1  # Default fallback
            vega_source = "fallback_default"

        # Log vega source for debugging (only at debug level to avoid spam)
        logger.debug(
            f"Vega for {option_contract.symbol} strike {option_contract.strike}: "
            f"{vega_value:.4f} (source: {vega_source})"
        )

        # Cache the result
        self.greeks_cache[cache_key] = (vega_value, time.time())
        return vega_value

    async def _create_calendar_spread_opportunities(
        self,
        symbol: str,
        stock_contract: Contract,
        options_data: Dict,
    ) -> List[CalendarSpreadOpportunity]:
        """
        Create calendar spread opportunities from options data.
        Analyzes all strike/expiry combinations for viable calendar spreads.
        """
        opportunities = []

        # Group options by strike and type
        strikes_data = {}
        for contract_id, ticker in options_data.items():
            if not ticker or not hasattr(ticker, "contract"):
                continue

            contract = ticker.contract
            if not hasattr(contract, "strike") or not hasattr(contract, "right"):
                continue

            strike = contract.strike
            option_type = "CALL" if contract.right == "C" else "PUT"
            expiry = contract.lastTradeDateOrContractMonth

            key = (strike, option_type)
            if key not in strikes_data:
                strikes_data[key] = []

            # Calculate option metrics
            days_to_expiry = self._calculate_days_to_expiry(expiry)
            iv = self._calculate_implied_volatility(ticker, contract)
            theta = self._calculate_theta(ticker, contract, days_to_expiry)

            leg = CalendarSpreadLeg(
                contract=contract,
                strike=strike,
                expiry=expiry,
                right=contract.right,
                price=(
                    ticker.midpoint()
                    if not np.isnan(ticker.midpoint())
                    else ticker.close
                ),
                bid=ticker.bid if not np.isnan(ticker.bid) else 0.0,
                ask=ticker.ask if not np.isnan(ticker.ask) else 0.0,
                volume=getattr(ticker, "volume", 0) or 0,
                iv=iv,
                theta=theta,
                days_to_expiry=days_to_expiry,
            )

            strikes_data[key].append(leg)

        # Create calendar spread opportunities
        total_combinations = 0
        filtered_combinations = 0
        valid_opportunities = 0

        logger.info(
            f"[{symbol}] Analyzing {len(strikes_data)} strike/type combinations for calendar spreads"
        )

        for (strike, option_type), legs in strikes_data.items():
            if len(legs) < 2:
                logger.debug(
                    f"[{symbol}] {option_type} {strike}: Only {len(legs)} expiry available, need at least 2"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.INSUFFICIENT_EXPIRY_OPTIONS,
                    {
                        "symbol": symbol,
                        "strike": strike,
                        "option_type": option_type,
                        "available_expiries": len(legs),
                        "min_required": 2,
                    },
                )
                continue  # Need at least 2 expiries

            # Sort by days to expiry
            legs.sort(key=lambda x: x.days_to_expiry)

            # Pre-filter legs based on expiry criteria to reduce combinations
            valid_front_legs = [
                leg for leg in legs if leg.days_to_expiry <= self.config.max_days_front
            ]
            valid_back_legs = [
                leg
                for leg in legs
                if self.config.min_days_back
                <= leg.days_to_expiry
                <= self.config.max_days_back
            ]

            # Skip if no valid combinations possible
            if not valid_front_legs or not valid_back_legs:
                logger.debug(
                    f"[{symbol}] {option_type} {strike}: No valid front/back leg combinations"
                )
                continue

            # Create calendar spreads (front month vs back month)
            for i in range(len(legs) - 1):
                front_leg = legs[i]

                # Skip if front leg doesn't meet criteria
                if front_leg.days_to_expiry > self.config.max_days_front:
                    logger.info(
                        f"[{symbol}] {option_type} {strike}: Front expiry too far: {front_leg.days_to_expiry}d > {self.config.max_days_front}d"
                    )
                    continue

                for j in range(i + 1, len(legs)):
                    back_leg = legs[j]
                    total_combinations += 1

                    # Validate back expiry range first (cheapest check)
                    if (
                        back_leg.days_to_expiry < self.config.min_days_back
                        or back_leg.days_to_expiry > self.config.max_days_back
                    ):
                        logger.info(
                            f"[{symbol}] {option_type} {strike}: Back expiry out of range: {back_leg.days_to_expiry}d (range: {self.config.min_days_back}-{self.config.max_days_back}d)"
                        )
                        continue

                    # Check IV spread before calculating prices
                    iv_spread = back_leg.iv - front_leg.iv
                    if iv_spread < self.config.min_iv_spread:
                        logger.info(
                            f"[{symbol}] {option_type} {strike}: IV spread insufficient: {iv_spread:.1f}% < {self.config.min_iv_spread:.1f}%"
                        )
                        metrics_collector.add_rejection_reason(
                            RejectionReason.INSUFFICIENT_IV_SPREAD,
                            {
                                "symbol": symbol,
                                "strike": strike,
                                "option_type": option_type,
                                "iv_spread": iv_spread,
                                "min_required": self.config.min_iv_spread,
                                "front_iv": front_leg.iv,
                                "back_iv": back_leg.iv,
                            },
                        )
                        continue

                    # Calculate theta ratio (still relatively cheap)
                    theta_ratio = (
                        abs(front_leg.theta / back_leg.theta)
                        if back_leg.theta != 0
                        else 0.0
                    )
                    if theta_ratio < self.config.min_theta_ratio:
                        logger.info(
                            f"[{symbol}] {option_type} {strike}: Theta ratio insufficient: {theta_ratio:.2f} < {self.config.min_theta_ratio:.2f}"
                        )
                        metrics_collector.add_rejection_reason(
                            RejectionReason.INSUFFICIENT_THETA_RATIO,
                            {
                                "symbol": symbol,
                                "strike": strike,
                                "option_type": option_type,
                                "theta_ratio": theta_ratio,
                                "min_required": self.config.min_theta_ratio,
                                "front_theta": front_leg.theta,
                                "back_theta": back_leg.theta,
                            },
                        )
                        continue

                    # Only calculate net debit after all other filters pass
                    net_debit = back_leg.price - front_leg.price
                    logger.info(
                        f"[{symbol}] {option_type} {strike}: IV spread: {iv_spread:.1f}%, Net debit: ${net_debit:.2f}"
                    )

                    # Apply price-based filters
                    if net_debit > self.config.max_net_debit or net_debit <= 0:
                        logger.info(
                            f"[{symbol}] {option_type} {strike}: Net debit invalid: ${net_debit:.2f} (max: ${self.config.max_net_debit:.0f})"
                        )
                        metrics_collector.add_rejection_reason(
                            (
                                RejectionReason.COST_LIMIT_EXCEEDED
                                if net_debit > self.config.max_net_debit
                                else RejectionReason.INVALID_CONTRACT_DATA
                            ),
                            {
                                "symbol": symbol,
                                "strike": strike,
                                "option_type": option_type,
                                "net_debit": net_debit,
                                "max_allowed": self.config.max_net_debit,
                                "front_price": front_leg.price,
                                "back_price": back_leg.price,
                            },
                        )
                        continue

                    # Check volume requirements
                    if (
                        front_leg.volume < self.config.min_volume
                        or back_leg.volume < self.config.min_volume
                    ):
                        logger.debug(
                            f"[{symbol}] {option_type} {strike}: Volume too low - Front: {front_leg.volume}, Back: {back_leg.volume} (min: {self.config.min_volume})"
                        )
                        metrics_collector.add_rejection_reason(
                            RejectionReason.INSUFFICIENT_VOLUME,
                            {
                                "symbol": symbol,
                                "strike": strike,
                                "option_type": option_type,
                                "front_volume": front_leg.volume,
                                "back_volume": back_leg.volume,
                                "min_required": self.config.min_volume,
                            },
                        )
                        logger.info(
                            f"[{symbol}] {option_type} {strike}: Volume insufficient - Front: {front_leg.volume}, Back: {back_leg.volume} (min: {self.config.min_volume})"
                        )
                        continue

                    # Calculate additional metrics
                    max_profit = self._calculate_theoretical_max_profit(
                        strike,
                        front_leg.price,
                        back_leg.price,
                        front_leg.days_to_expiry,
                    )

                    front_liquidity = self._calculate_liquidity_score(front_leg)
                    back_liquidity = self._calculate_liquidity_score(back_leg)
                    combined_liquidity = (front_liquidity + back_liquidity) / 2.0

                    if combined_liquidity < self.config.min_liquidity_score:
                        logger.debug(
                            f"[{symbol}] {option_type} {strike}: Liquidity score too low: {combined_liquidity:.3f} < {self.config.min_liquidity_score:.3f}"
                        )
                        metrics_collector.add_rejection_reason(
                            RejectionReason.INSUFFICIENT_LIQUIDITY,
                            {
                                "symbol": symbol,
                                "strike": strike,
                                "option_type": option_type,
                                "combined_liquidity_score": combined_liquidity,
                                "min_required": self.config.min_liquidity_score,
                                "front_liquidity": front_liquidity,
                                "back_liquidity": back_liquidity,
                            },
                        )
                        logger.info(
                            f"[{symbol}] {option_type} {strike}: Liquidity insufficient: {combined_liquidity:.3f} < {self.config.min_liquidity_score:.3f}"
                        )
                        continue

                    # Detect term structure inversion
                    term_structure_inversion = self._detect_term_structure_inversion(
                        front_leg.iv,
                        back_leg.iv,
                        front_leg.days_to_expiry,
                        back_leg.days_to_expiry,
                    )

                    # Calculate Greeks
                    net_delta = self._calculate_delta(
                        None, back_leg.contract
                    ) - self._calculate_delta(None, front_leg.contract)
                    net_gamma = self._calculate_gamma(
                        None, back_leg.contract
                    ) - self._calculate_gamma(None, front_leg.contract)
                    net_vega = self._calculate_vega(
                        None, back_leg.contract
                    ) - self._calculate_vega(None, front_leg.contract)

                    # Calculate composite score
                    composite_score = self._calculate_calendar_spread_score(
                        iv_spread,
                        theta_ratio,
                        combined_liquidity,
                        max_profit,
                        net_debit,
                        term_structure_inversion,
                    )

                    # Create opportunity
                    opportunity = CalendarSpreadOpportunity(
                        symbol=symbol,
                        strike=strike,
                        option_type=option_type,
                        front_leg=front_leg,
                        back_leg=back_leg,
                        iv_spread=iv_spread,
                        theta_ratio=theta_ratio,
                        net_debit=net_debit,
                        max_profit=max_profit,
                        max_loss=net_debit,
                        front_bid_ask_spread=(
                            (front_leg.ask - front_leg.bid) / front_leg.price
                            if front_leg.price > 0
                            else 0.0
                        ),
                        back_bid_ask_spread=(
                            (back_leg.ask - back_leg.bid) / back_leg.price
                            if back_leg.price > 0
                            else 0.0
                        ),
                        combined_liquidity_score=combined_liquidity,
                        term_structure_inversion=term_structure_inversion,
                        net_delta=net_delta,
                        net_gamma=net_gamma,
                        net_vega=net_vega,
                        composite_score=composite_score,
                    )

                    opportunities.append(opportunity)
                    valid_opportunities += 1

                    logger.info(
                        f"[{symbol}] ✓ Valid calendar spread: {option_type} {strike} "
                        f"IV: {iv_spread:.1f}% Theta: {theta_ratio:.2f} Score: {composite_score:.3f}"
                    )

        # Log analysis summary
        logger.info(f"[{symbol}] Calendar spread analysis complete:")
        logger.info(f"  Total combinations examined: {total_combinations}")
        logger.info(f"  Valid opportunities found: {valid_opportunities}")

        if valid_opportunities > 0:
            logger.info(
                f"  Success rate: {(valid_opportunities/total_combinations)*100:.1f}%"
            )

        # Record metrics
        metrics_collector.record_contracts_count(len(options_data))
        metrics_collector.record_expiries_scanned(
            len(set(leg.expiry for legs in strikes_data.values() for leg in legs))
        )

        # Sort by composite score and return top opportunities
        opportunities.sort(key=lambda x: x.composite_score, reverse=True)

        if opportunities:
            logger.info(f"[{symbol}] Top calendar spread opportunities:")
            for i, opp in enumerate(opportunities[:5]):
                logger.info(
                    f"  #{i+1}: {opp.option_type} {opp.strike} "
                    f"Front: {opp.front_leg.expiry} ({opp.front_leg.days_to_expiry}d) "
                    f"Back: {opp.back_leg.expiry} ({opp.back_leg.days_to_expiry}d)"
                )
                logger.info(
                    f"      IV Spread: {opp.iv_spread:.1f}% | Theta Ratio: {opp.theta_ratio:.2f} | "
                    f"Net Debit: ${opp.net_debit:.2f} | Score: {opp.composite_score:.3f}"
                )

        return opportunities[:5]  # Return top 5 opportunities

    def _calculate_days_to_expiry(self, expiry_str: str) -> int:
        """Calculate days to expiry from expiry string"""
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y%m%d")
            today = datetime.now().date()
            return (expiry_date.date() - today).days
        except ValueError:
            logger.warning(f"Invalid expiry format: {expiry_str}")
            return 30  # Default assumption

    def _calculate_calendar_spread_score(
        self,
        iv_spread: float,
        theta_ratio: float,
        liquidity_score: float,
        max_profit: float,
        net_debit: float,
        term_structure_inversion: bool,
    ) -> float:
        """Calculate composite score for calendar spread opportunity"""
        # IV spread component (0-40%)
        iv_score = min(
            1.0, (iv_spread - self.config.min_iv_spread) / 10.0
        )  # Normalize to 10% spread

        # Theta ratio component (0-30%)
        theta_score = min(
            1.0, (theta_ratio - self.config.min_theta_ratio) / 2.0
        )  # Normalize to 3.5 ratio

        # Liquidity component (0-20%)
        # liquidity_score is already 0-1

        # Profit potential component (0-20%)
        profit_ratio = max_profit / net_debit if net_debit > 0 else 0.0
        profit_score = min(1.0, profit_ratio / 2.0)  # Normalize to 200% return

        # Term structure inversion bonus (0-10%)
        inversion_bonus = 1.0 if term_structure_inversion else 0.0

        # Weighted composite score
        composite_score = (
            iv_score * 0.35
            + theta_score * 0.25
            + liquidity_score * 0.20
            + profit_score * 0.15
            + inversion_bonus * 0.05
        )

        return composite_score

    async def scan(
        self,
        symbol_list: List[str],
        cost_limit: float = 500.0,
        profit_target: float = 0.3,
        quantity: int = 1,
    ) -> None:
        """
        Main scanning method for calendar spread strategy.

        Args:
            symbol_list: List of symbols to scan
            cost_limit: Maximum cost for calendar spread
            profit_target: Target profit ratio
            quantity: Number of spreads to execute
        """
        # Update configuration
        self.config.max_net_debit = cost_limit
        self.config.target_profit_ratio = profit_target

        try:
            await self.ib.connectAsync("127.0.0.1", 7497, clientId=1)
            logger.info(
                "✓ Connected to Interactive Brokers for calendar spread scanning"
            )

            # Register order fill handler
            self.ib.orderStatusEvent += self.onFill

            # Register master executor for handling market data events
            self.ib.pendingTickersEvent += self.master_executor

            logger.info(
                f"Starting calendar spread analysis for {len(symbol_list)} symbols"
            )
            logger.info(
                f"Configuration: Max debit=${cost_limit:.0f}, Target profit={profit_target:.1%}, Quantity={quantity}"
            )

            # Start cycle tracking
            metrics_collector.start_cycle(len(symbol_list))

            total_opportunities = 0
            symbols_with_opportunities = 0

            # Process symbols using throttled scanning like SFR.py
            tasks = []
            for i, symbol in enumerate(symbol_list, 1):
                if self.order_filled:
                    logger.info("Order filled, stopping further processing")
                    break

                # Use throttled scanning instead of fixed delays
                task = asyncio.create_task(
                    self.scan_with_throttle(
                        symbol,
                        self.scan_calendar_spreads,
                        quantity,
                        i,
                        len(symbol_list),
                    )
                )
                tasks.append(task)

            # Wait for all scanning tasks to complete
            if tasks:
                logger.info(
                    f"Processing {len(tasks)} symbols with throttled scanning..."
                )
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results and aggregate statistics
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Task failed with exception: {str(result)}")
                        continue
                    elif result and isinstance(result, dict):
                        if result.get("opportunities_count", 0) > 0:
                            total_opportunities += result["opportunities_count"]
                            symbols_with_opportunities += 1

            # Log final scan summary
            logger.info(f"=== Calendar Spread Scan Summary ===")
            logger.info(f"Total symbols processed: {len(symbol_list)}")
            logger.info(f"Symbols with opportunities: {symbols_with_opportunities}")
            logger.info(f"Total opportunities found: {total_opportunities}")
            if len(symbol_list) > 0:
                logger.info(
                    f"Success rate: {(symbols_with_opportunities/len(symbol_list))*100:.1f}%"
                )
            if symbols_with_opportunities > 0:
                logger.info(
                    f"Average opportunities per symbol: {total_opportunities/symbols_with_opportunities:.1f}"
                )

            # Finish cycle tracking
            metrics_collector.finish_cycle()

            # Monitor for execution
            if self.active_executors:
                logger.info(
                    f"Monitoring {len(self.active_executors)} calendar spread executors for execution..."
                )
                await self._monitor_execution()
            else:
                logger.info(
                    "No calendar spread opportunities found - no executors to monitor"
                )

        except Exception as e:
            logger.error(f"Critical error in calendar spread processing: {str(e)}")
            metrics_collector.finish_scan(success=False, error_message=str(e))
        finally:
            await self._cleanup_and_disconnect()

    async def scan_calendar_spreads(
        self, symbol: str, quantity: int, symbol_index: int, total_symbols: int
    ) -> Optional[dict]:
        """
        Scan for calendar spread opportunities for a specific symbol.
        Creates a single executor per symbol that handles all calendar spreads.

        Args:
            symbol: Trading symbol to scan
            quantity: Number of contracts to trade
            symbol_index: Current symbol index for logging
            total_symbols: Total number of symbols being processed

        Returns:
            Dict with processing results or None if processing failed
        """
        logger.info(
            f"[{symbol_index}/{total_symbols}] Processing calendar spreads for {symbol}"
        )
        start_time = time.time()

        try:
            # Get options chain and create opportunities
            opportunities = await self._scan_symbol_for_calendar_spreads(
                symbol, quantity
            )

            scan_time = time.time() - start_time

            if opportunities:
                logger.info(
                    f"✓ [{symbol}] Found {len(opportunities)} calendar spread opportunities (scan time: {scan_time:.1f}s)"
                )

                # Create and activate executor
                executor = CalendarSpreadExecutor(
                    self.ib,
                    self.order_manager,
                    opportunities[
                        0
                    ].front_leg.contract,  # Use first opportunity's underlying
                    opportunities,
                    symbol,
                    self.config,
                    time.time(),
                    quantity,
                )

                self.active_executors[symbol] = executor
                logger.info(f"✓ [{symbol}] Activated calendar spread executor")

                return {
                    "symbol": symbol,
                    "opportunities_count": len(opportunities),
                    "scan_time": scan_time,
                    "success": True,
                }
            else:
                logger.info(
                    f"✗ [{symbol}] No calendar spread opportunities found (scan time: {scan_time:.1f}s)"
                )
                return {
                    "symbol": symbol,
                    "opportunities_count": 0,
                    "scan_time": scan_time,
                    "success": True,
                }

        except Exception as e:
            scan_time = time.time() - start_time
            logger.error(
                f"✗ [{symbol}] Error processing symbol (scan time: {scan_time:.1f}s): {str(e)}"
            )
            return {
                "symbol": symbol,
                "opportunities_count": 0,
                "scan_time": scan_time,
                "success": False,
                "error": str(e),
            }

    async def _scan_symbol_for_calendar_spreads(
        self, symbol: str, quantity: int
    ) -> List[CalendarSpreadOpportunity]:
        """Scan a single symbol for calendar spread opportunities"""
        try:
            # Start metrics collection for this scan
            metrics_collector.start_scan(symbol, "Calendar")

            logger.info(f"[{symbol}] Starting calendar spread scan with parameters:")
            logger.info(f"  Max net debit: ${self.config.max_net_debit:.0f}")
            logger.info(f"  Min IV spread: {self.config.min_iv_spread:.1f}%")
            logger.info(f"  Min theta ratio: {self.config.min_theta_ratio:.1f}")
            logger.info(f"  Front expiry max: {self.config.max_days_front}d")
            logger.info(
                f"  Back expiry range: {self.config.min_days_back}-{self.config.max_days_back}d"
            )
            logger.info(f"  Min volume: {self.config.min_volume}")

            # Create stock/index contract
            stock_contract = await self._create_underlying_contract(symbol)
            if not stock_contract:
                logger.warning(f"[{symbol}] Could not create underlying contract")
                metrics_collector.add_rejection_reason(
                    RejectionReason.INVALID_CONTRACT_DATA,
                    {
                        "symbol": symbol,
                        "reason": "failed to create underlying contract",
                    },
                )
                metrics_collector.finish_scan(
                    success=False, error_message="Invalid underlying contract"
                )
                return []

            # Get options chain data
            options_data = await self._get_options_chain_data(symbol, stock_contract)
            if not options_data:
                logger.warning(f"[{symbol}] No options chain data available")
                metrics_collector.add_rejection_reason(
                    RejectionReason.MISSING_MARKET_DATA,
                    {"symbol": symbol, "reason": "no options chain data"},
                )
                metrics_collector.finish_scan(
                    success=False, error_message="No options data"
                )
                return []

            logger.info(
                f"[{symbol}] Retrieved market data for {len(options_data)} option contracts"
            )

            # Create calendar spread opportunities
            opportunities = await self._create_calendar_spread_opportunities(
                symbol, stock_contract, options_data
            )

            if opportunities:
                logger.info(
                    f"[{symbol}] Found {len(opportunities)} calendar spread opportunities:"
                )
                for i, opp in enumerate(opportunities[:3]):  # Show top 3
                    logger.info(
                        f"  #{i+1}: {opp.option_type} {opp.strike} "
                        f"IV Spread: {opp.iv_spread:.1f}% "
                        f"Theta Ratio: {opp.theta_ratio:.2f} "
                        f"Score: {opp.composite_score:.3f}"
                    )

                metrics_collector.record_opportunity_found()
                metrics_collector.finish_scan(success=True)
            else:
                logger.info(
                    f"[{symbol}] No calendar spread opportunities found after filtering"
                )
                metrics_collector.finish_scan(success=True)

            return opportunities

        except Exception as e:
            logger.error(f"[{symbol}] Error scanning for calendar spreads: {str(e)}")
            metrics_collector.finish_scan(success=False, error_message=str(e))
            return []

    async def _create_underlying_contract(self, symbol: str) -> Optional[Contract]:
        """Create and qualify underlying contract"""
        try:
            # Handle different symbol formats
            if symbol.startswith("@"):
                # Index options
                clean_symbol = symbol[1:]
                contract = Contract(
                    symbol=clean_symbol, secType="IND", exchange="CBOE", currency="USD"
                )
            elif symbol.startswith("!"):
                # Futures options (simplified)
                clean_symbol = symbol[1:]
                contract = Contract(
                    symbol=clean_symbol, secType="FUT", exchange="CME", currency="USD"
                )
            else:
                # Stock options
                contract = Contract(
                    symbol=symbol, secType="STK", exchange="SMART", currency="USD"
                )

            qualified = await self.qualify_contracts_cached(contract)
            return qualified[0] if qualified else None

        except Exception as e:
            logger.error(f"Error creating contract for {symbol}: {str(e)}")
            return None

    async def _get_options_chain_data(
        self, symbol: str, stock_contract: Contract
    ) -> Dict:
        """Get options chain data for calendar spread analysis"""
        try:
            logger.info(f"[{symbol}] Requesting options chain data...")

            # Request options chain
            chains = await self.ib.reqSecDefOptParamsAsync(
                stock_contract.symbol, "", stock_contract.secType, stock_contract.conId
            )

            if not chains:
                logger.warning(f"[{symbol}] No options chains found")
                metrics_collector.add_rejection_reason(
                    RejectionReason.NO_OPTIONS_CHAIN, {"symbol": symbol}
                )
                return {}

            # Get the primary chain based on exchange and trading class
            try:
                exchange = (
                    stock_contract.exchange
                    if stock_contract.exchange != "SMART"
                    else "CBOE"
                )
                chain = next(
                    c
                    for c in chains
                    if c.exchange == exchange
                    and c.tradingClass == stock_contract.symbol
                )
            except StopIteration:
                # Fallback to first chain if no exact match found
                logger.warning(
                    f"[{symbol}] No chain found for exchange {exchange} and tradingClass {stock_contract.symbol}, using first available chain"
                )
                chain = chains[0]
            logger.info(
                f"[{symbol}] Found options chain with {len(chain.expirations)} expiries and {len(chain.strikes)} strikes"
            )

            # Select expiries (front and back months for calendar spreads)
            valid_expiries = self._select_calendar_expiries(chain.expirations)
            if len(valid_expiries) < 2:
                logger.warning(
                    f"[{symbol}] Insufficient expiries for calendar spreads: found {len(valid_expiries)}, need at least 2"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.NO_VALID_EXPIRIES,
                    {
                        "symbol": symbol,
                        "available_expiries": len(chain.expirations),
                        "valid_expiries": len(valid_expiries),
                        "min_required": 2,
                    },
                )
                return {}

            # Select strikes around ATM
            stock_price = await self._get_current_stock_price(stock_contract)
            if not stock_price:
                logger.warning(f"[{symbol}] Could not determine current stock price")
                metrics_collector.add_rejection_reason(
                    RejectionReason.MISSING_MARKET_DATA,
                    {"symbol": symbol, "reason": "unable to get stock price"},
                )
                return {}

            logger.info(f"[{symbol}] Current stock price: ${stock_price:.2f}")
            valid_strikes = self._select_calendar_strikes(chain.strikes, stock_price)

            logger.info(
                f"[{symbol}] Selected {len(valid_expiries)} expiries and {len(valid_strikes)} strikes for calendar analysis"
            )
            logger.info(
                f"  Expiries: {valid_expiries[:3]}{'...' if len(valid_expiries) > 3 else ''}"
            )
            logger.info(
                f"  Strike range: ${min(valid_strikes):.0f} - ${max(valid_strikes):.0f}"
            )

            # Create option contracts
            option_contracts = []
            for expiry in valid_expiries:
                for strike in valid_strikes:
                    for right in ["C", "P"]:
                        option = Option(
                            symbol=stock_contract.symbol,
                            lastTradeDateOrContractMonth=expiry,
                            strike=strike,
                            right=right,
                            exchange="SMART",
                            currency="USD",
                        )
                        option_contracts.append(option)

            # Qualify contracts
            qualified_options = await self.qualify_contracts_cached(*option_contracts)

            # Request market data
            tickers = await self._request_market_data_batch(qualified_options)

            # Build data dictionary
            options_data = {}
            for ticker in tickers:
                if ticker and hasattr(ticker, "contract"):
                    options_data[ticker.contract.conId] = ticker

            return options_data

        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {str(e)}")
            return {}

    def _select_calendar_expiries(self, all_expiries: List[str]) -> List[str]:
        """Select appropriate expiries for calendar spreads"""
        today = datetime.now().date()
        valid_expiries = []

        # For calendar spreads, we need a smart selection of expiries
        # to ensure we have good coverage for both front and back months
        front_expiries = []
        back_expiries = []

        for expiry_str in sorted(all_expiries):
            try:
                expiry_date = datetime.strptime(expiry_str, "%Y%m%d").date()
                days_to_expiry = (expiry_date - today).days

                # Skip past expiries
                if days_to_expiry < 0:
                    continue

                # Categorize expiries
                if days_to_expiry <= self.config.max_days_front:
                    front_expiries.append((expiry_str, days_to_expiry))
                elif (
                    self.config.min_days_back
                    <= days_to_expiry
                    <= self.config.max_days_back
                ):
                    back_expiries.append((expiry_str, days_to_expiry))

            except ValueError:
                continue

        # Smart selection strategy:
        # 1. For front month: Take weekly expiries (every ~7 days) if too many dailies
        # 2. For back month: Take monthly expiries (every ~30 days) or all if few

        selected_front = []
        if len(front_expiries) > 10:  # Too many daily expiries
            # Select weekly intervals
            last_selected_day = -7
            for expiry, days in front_expiries:
                if days - last_selected_day >= 7:
                    selected_front.append(expiry)
                    last_selected_day = days
        else:
            # Take all front expiries if reasonable number
            selected_front = [exp[0] for exp in front_expiries]

        selected_back = []
        if len(back_expiries) > 8:  # Too many back month expiries
            # Select monthly intervals
            last_selected_day = self.config.min_days_back - 30
            for expiry, days in back_expiries:
                if days - last_selected_day >= 21:  # ~3 weeks for better coverage
                    selected_back.append(expiry)
                    last_selected_day = days
        else:
            # Take all back expiries if reasonable number
            selected_back = [exp[0] for exp in back_expiries]

        # Combine and limit total to prevent performance issues
        valid_expiries = selected_front + selected_back

        # Log the selection
        logger.info(
            f"Selected {len(selected_front)} front month expiries (0-{self.config.max_days_front}d) "
            f"and {len(selected_back)} back month expiries ({self.config.min_days_back}-{self.config.max_days_back}d)"
        )

        # Return up to 15 expiries total for performance
        return valid_expiries[:15]

    def _select_calendar_strikes(
        self, all_strikes: List[float], stock_price: float
    ) -> List[float]:
        """Select strikes around current stock price for calendar spreads"""
        # Filter strikes within reasonable range of current price
        min_strike = stock_price * 0.85
        max_strike = stock_price * 1.15

        valid_strikes = [
            strike for strike in all_strikes if min_strike <= strike <= max_strike
        ]

        # Sort by distance from current price and take closest strikes
        valid_strikes.sort(key=lambda x: abs(x - stock_price))
        return valid_strikes[:10]  # Limit to 10 strikes for performance

    async def _get_current_stock_price(
        self, stock_contract: Contract
    ) -> Optional[float]:
        """Get current stock price"""
        try:
            ticker = self.ib.reqMktData(stock_contract, "", False, False)
            await asyncio.sleep(2)  # Wait for data

            if ticker and not np.isnan(ticker.last):
                return ticker.last
            elif ticker and not np.isnan(ticker.close):
                return ticker.close
            else:
                return None

        except Exception as e:
            logger.error(f"Error getting stock price: {str(e)}")
            return None

    async def _request_market_data_batch(
        self, contracts: List[Contract]
    ) -> List[Ticker]:
        """Request market data for multiple contracts efficiently"""
        try:
            tickers = []
            for contract in contracts:
                ticker = self.ib.reqMktData(contract, "", False, False)
                tickers.append(ticker)

            # Wait for data to populate
            await asyncio.sleep(5)

            # Update global contract_ticker for executors
            for ticker in tickers:
                if ticker and hasattr(ticker, "contract"):
                    contract_ticker[ticker.contract.conId] = ticker

            return tickers

        except Exception as e:
            logger.error(f"Error requesting market data: {str(e)}")
            return []

    async def _monitor_execution(self) -> None:
        """Monitor calendar spread execution"""
        try:
            max_monitoring_time = 300  # 5 minutes maximum
            start_time = time.time()

            while self.active_executors and not self.order_filled:
                current_time = time.time()

                # Check timeout
                if current_time - start_time > max_monitoring_time:
                    logger.info("Calendar spread monitoring timeout reached")
                    break

                # Clean up inactive executors
                self.cleanup_inactive_executors()

                # Wait before next check
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error in calendar spread monitoring: {str(e)}")

    async def _cleanup_and_disconnect(self) -> None:
        """Clean up resources and disconnect"""
        try:
            logger.info("Calendar spread scanning complete - cleaning up resources")

            # Deactivate all executors
            self.deactivate_all_executors()

            # Cancel all market data subscriptions
            self.ib.cancelMktData("")

            # Print final metrics summary
            logger.info("=== Final Calendar Spread Metrics ===")
            if len(metrics_collector.scan_metrics) > 0:
                metrics_collector.print_summary()
            else:
                logger.info("No metrics data collected")

            # Disconnect
            if self.ib.isConnected():
                self.ib.disconnect()
                logger.info("✓ Disconnected from Interactive Brokers")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


# Convenience function for external usage
async def run_calendar_spread_strategy(
    symbols: List[str],
    cost_limit: float = 500.0,
    profit_target: float = 0.3,
    quantity: int = 1,
    log_file: str = None,
) -> None:
    """
    Convenience function to run calendar spread strategy.

    Args:
        symbols: List of symbols to scan
        cost_limit: Maximum cost for calendar spread
        profit_target: Target profit ratio
        quantity: Number of spreads to execute
        log_file: Optional log file path
    """
    strategy = CalendarSpread(log_file=log_file)
    await strategy.scan(symbols, cost_limit, profit_target, quantity)
