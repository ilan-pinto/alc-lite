"""
Calendar spread strategy implementation.

This module contains the main CalendarSpread class and related functionality.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from ib_async import Contract, Option, Order, Ticker

from modules.Arbitrage.Strategy import ArbitrageClass

from ..common import get_logger
from ..metrics import RejectionReason, metrics_collector
from .models import CalendarSpreadConfig, CalendarSpreadLeg, CalendarSpreadOpportunity
from .opportunity_manager import CalendarSpreadOpportunityManager
from .utils import (
    SCIPY_AVAILABLE,
    AdaptiveCacheManager,
    PerformanceProfiler,
    TTLCache,
    VectorizedGreeksCalculator,
    _safe_isnan,
)

logger = get_logger()

# Global variable for contract ticker information (for backward compatibility)
contract_ticker = {}


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

        # Global opportunity management
        self.global_manager = CalendarSpreadOpportunityManager()

        # Calendar spread specific caching with TTL and size limits
        self.iv_cache = TTLCache(max_size=2000, ttl_seconds=120)  # 2 minute TTL for IV
        self.greeks_cache = TTLCache(
            max_size=5000, ttl_seconds=60
        )  # 1 minute TTL for Greeks
        self.cache_ttl = 60  # Legacy parameter, now handled by TTLCache

        # Adaptive cache manager for memory pressure handling
        self.cache_manager = AdaptiveCacheManager()

        # Performance profiler for monitoring and optimization
        self.profiler = PerformanceProfiler()

        logger.info(
            "Calendar Spread strategy initialized with complete performance optimizations"
        )

    def _perform_cache_maintenance(self) -> None:
        """Perform cache maintenance including memory pressure cleanup"""
        caches = [self.iv_cache, self.greeks_cache]

        # Regular cleanup of expired entries
        total_expired = sum(cache.cleanup_expired() for cache in caches)

        # Memory pressure cleanup if needed
        pressure_cleaned = self.cache_manager.cleanup_if_needed(caches)

        # Log cache statistics periodically
        iv_size = self.iv_cache.size()
        greeks_size = self.greeks_cache.size()
        memory_stats = self.cache_manager.get_memory_stats()

        if total_expired > 0 or pressure_cleaned > 0:
            logger.info(
                f"Cache maintenance: expired={total_expired}, pressure_cleaned={pressure_cleaned}, "
                f"iv_cache_size={iv_size}, greeks_cache_size={greeks_size}, "
                f"memory={memory_stats.get('memory_percent', 'N/A')}"
            )

    async def _calculate_greeks_batch(
        self, tickers: List[Ticker], stock_price: float
    ) -> Dict:
        """Calculate Greeks for multiple options in batch using vectorized operations"""

        if not tickers:
            return {}

        if not SCIPY_AVAILABLE:
            logger.warning(
                "scipy not available - using approximate Greeks calculations"
            )

        # Prepare vectorized inputs
        strikes = np.array([ticker.contract.strike for ticker in tickers])

        # Calculate days to expiry for each ticker
        expiries = []
        for ticker in tickers:
            try:
                exp_date = datetime.strptime(
                    ticker.contract.lastTradeDateOrContractMonth, "%Y%m%d"
                ).date()
                days = (exp_date - datetime.now().date()).days
                expiries.append(max(1, days) / 365.0)  # Convert to years, minimum 1 day
            except ValueError:
                expiries.append(30 / 365.0)  # Default 30 days

        expiries = np.array(expiries)

        # Option types: 1 for calls, -1 for puts
        option_types = np.array(
            [1 if ticker.contract.right == "C" else -1 for ticker in tickers]
        )

        # Use IB implied volatilities where available, fallback to estimated
        ivs = []
        for ticker in tickers:
            iv = self._get_iv_from_ticker(ticker)
            if iv is None or np.isnan(iv) or iv <= 0:
                # Fallback to estimated IV based on option price
                option_price = (
                    ticker.midpoint()
                    if not np.isnan(ticker.midpoint())
                    else ticker.close
                )
                if option_price and option_price > 0:
                    # Rough IV estimation: higher for OTM options
                    moneyness = stock_price / ticker.contract.strike
                    estimated_iv = 0.2 + abs(1 - moneyness) * 0.3  # 20-50% range
                    ivs.append(estimated_iv)
                else:
                    ivs.append(0.25)  # Default 25% IV
            else:
                ivs.append(iv / 100.0)  # Convert percentage to decimal

        ivs = np.array(ivs)

        # Stock prices array (same for all options)
        stock_prices = np.full(len(tickers), stock_price)

        # Current risk-free rate (could be made configurable)
        risk_free_rate = 0.05

        # Batch calculate Greeks using vectorized operations
        try:
            greeks = VectorizedGreeksCalculator.black_scholes_greeks_batch(
                stock_prices, strikes, expiries, risk_free_rate, ivs, option_types
            )

            # Map results back to contract IDs
            results = {}
            for i, ticker in enumerate(tickers):
                results[ticker.contract.conId] = {
                    "delta": float(greeks["delta"][i]),
                    "gamma": float(greeks["gamma"][i]),
                    "theta": float(greeks["theta"][i]),
                    "vega": float(greeks["vega"][i]),
                    "iv": ivs[i] * 100,  # Convert back to percentage
                }

            logger.debug(
                f"Calculated Greeks for {len(tickers)} options using vectorized operations"
            )
            return results

        except Exception as e:
            logger.error(
                f"Vectorized Greeks calculation failed, falling back to individual: {e}"
            )
            # Fallback to individual calculations if batch fails
            return {}

    def _get_iv_from_ticker(self, ticker: Ticker) -> Optional[float]:
        """Extract implied volatility from ticker data"""
        # Try to get IV from various ticker fields
        if hasattr(ticker, "impliedVolatility") and not np.isnan(
            ticker.impliedVolatility
        ):
            return ticker.impliedVolatility
        elif hasattr(ticker, "modelGreeks") and ticker.modelGreeks:
            if hasattr(ticker.modelGreeks, "impliedVol") and not np.isnan(
                ticker.modelGreeks.impliedVol
            ):
                return ticker.modelGreeks.impliedVol
        return None

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

    def _calculate_breakeven_boundaries(
        self,
        stock_price: float,
        strike: float,
        front_iv: float,
        back_iv: float,
        front_days: int,
        back_days: int,
        option_type: str,
        front_price: float,
        back_price: float,
        risk_free_rate: float = 0.05,
    ) -> Tuple[float, float, float]:
        """
        Calculate upper and lower breakeven boundaries for calendar spread.

        Returns:
            Tuple of (lower_breakeven, upper_breakeven, max_loss)
        """
        # Net debit of the spread (cost to enter)
        net_debit = back_price - front_price

        # Convert days to years for Black-Scholes
        front_time = front_days / 365.0
        back_time = back_days / 365.0

        # Convert IV from percentage to decimal
        front_vol = front_iv / 100.0
        back_vol = back_iv / 100.0

        # Option type for calculations (1 for call, -1 for put)
        opt_type = 1 if option_type.upper() == "CALL" else -1

        # Define search range around current stock price
        search_range = stock_price * 0.5  # +/- 50% of current price
        min_price = max(0.01, stock_price - search_range)
        max_price = stock_price + search_range

        # Create array of stock prices to test
        price_points = np.linspace(min_price, max_price, 200)

        # Calculate P&L at front expiration for each price point
        pnl_values = self._calculate_spread_pnl_at_expiration(
            price_points,
            strike,
            front_vol,
            back_vol,
            front_time,
            back_time,
            opt_type,
            net_debit,
            risk_free_rate,
        )

        # Find breakeven points (where P&L crosses zero)
        breakeven_points = []
        for i in range(len(pnl_values) - 1):
            if (
                pnl_values[i] * pnl_values[i + 1] <= 0
            ):  # Sign change indicates breakeven
                # Linear interpolation to find exact breakeven point
                if pnl_values[i + 1] != pnl_values[i]:
                    breakeven = price_points[i] - pnl_values[i] * (
                        price_points[i + 1] - price_points[i]
                    ) / (pnl_values[i + 1] - pnl_values[i])
                    breakeven_points.append(breakeven)

        # Determine lower and upper breakeven
        if len(breakeven_points) >= 2:
            lower_breakeven = min(breakeven_points)
            upper_breakeven = max(breakeven_points)
        elif len(breakeven_points) == 1:
            # Only one breakeven point found
            if breakeven_points[0] < stock_price:
                lower_breakeven = breakeven_points[0]
                upper_breakeven = max_price  # No upper breakeven
            else:
                lower_breakeven = min_price  # No lower breakeven
                upper_breakeven = breakeven_points[0]
        else:
            # No breakeven points found (unusual case)
            lower_breakeven = min_price
            upper_breakeven = max_price

        # Calculate maximum loss
        max_loss = abs(min(pnl_values))

        return lower_breakeven, upper_breakeven, max_loss

    def _calculate_spread_pnl_at_expiration(
        self,
        stock_prices: np.ndarray,
        strike: float,
        front_vol: float,
        back_vol: float,
        front_time: float,
        back_time: float,
        option_type: int,
        net_debit: float,
        risk_free_rate: float,
    ) -> np.ndarray:
        """
        Calculate calendar spread P&L at front option expiration for different stock prices.

        Calendar spread P&L = (Back option value at front expiry) - (Front option value at expiry) - Net debit
        """
        # At front expiration, front option has no time value, only intrinsic value
        if option_type == 1:  # Call
            front_value_at_expiry = np.maximum(stock_prices - strike, 0)
        else:  # Put
            front_value_at_expiry = np.maximum(strike - stock_prices, 0)

        # Back option still has remaining time value at front expiry
        remaining_time = back_time - front_time
        remaining_time = np.maximum(remaining_time, 1e-6)  # Prevent negative time

        # Calculate back option value at front expiry using Black-Scholes
        back_value_at_front_expiry = (
            VectorizedGreeksCalculator.black_scholes_price_batch(
                stock_prices,
                np.full_like(stock_prices, strike),
                np.full_like(stock_prices, remaining_time),
                risk_free_rate,
                np.full_like(stock_prices, back_vol),
                np.full_like(stock_prices, option_type),
            )
        )

        # Calendar spread P&L = Back option value - Front option payout - Net debit
        # We sold the front option, so we keep the premium but pay out if it's ITM
        # We bought the back option, so we own its remaining value
        pnl = back_value_at_front_expiry - front_value_at_expiry - net_debit

        return pnl

    def _calculate_profit_loss_curve(
        self,
        stock_price: float,
        strike: float,
        front_iv: float,
        back_iv: float,
        front_days: int,
        back_days: int,
        option_type: str,
        front_price: float,
        back_price: float,
        risk_free_rate: float = 0.05,
        price_range_pct: float = 0.4,
        num_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate detailed profit/loss curve for calendar spread across stock price range.

        Args:
            stock_price: Current stock price
            strike: Option strike price
            front_iv: Front month implied volatility (%)
            back_iv: Back month implied volatility (%)
            front_days: Days to front expiration
            back_days: Days to back expiration
            option_type: "CALL" or "PUT"
            front_price: Front option price
            back_price: Back option price
            risk_free_rate: Risk-free interest rate
            price_range_pct: Price range as percentage of stock price (0.4 = ±40%)
            num_points: Number of points in the curve

        Returns:
            Tuple of (stock_prices, pnl_values) arrays
        """
        # Net debit of the spread (cost to enter)
        net_debit = back_price - front_price

        # Convert parameters for Black-Scholes
        front_time = front_days / 365.0
        back_time = back_days / 365.0
        front_vol = front_iv / 100.0
        back_vol = back_iv / 100.0
        opt_type = 1 if option_type.upper() == "CALL" else -1

        # Define price range
        range_amount = stock_price * price_range_pct
        min_price = max(0.01, stock_price - range_amount)
        max_price = stock_price + range_amount

        # Create array of stock prices for the curve
        stock_prices = np.linspace(min_price, max_price, num_points)

        # Calculate P&L at front expiration for each price point
        pnl_values = self._calculate_spread_pnl_at_expiration(
            stock_prices,
            strike,
            front_vol,
            back_vol,
            front_time,
            back_time,
            opt_type,
            net_debit,
            risk_free_rate,
        )

        return stock_prices, pnl_values

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

        cached_iv = self.iv_cache.get(cache_key)
        if cached_iv is not None:
            return cached_iv

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

        self.iv_cache.put(cache_key, iv_value)
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

        cached_theta = self.greeks_cache.get(cache_key)
        if cached_theta is not None:
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
        self.greeks_cache.put(cache_key, theta_value)
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

        cached_delta = self.greeks_cache.get(cache_key)
        if cached_delta is not None:
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
        self.greeks_cache.put(cache_key, delta_value)
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

        cached_gamma = self.greeks_cache.get(cache_key)
        if cached_gamma is not None:
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
        self.greeks_cache.put(cache_key, gamma_value)
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

        cached_vega = self.greeks_cache.get(cache_key)
        if cached_vega is not None:
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
            option_price = None

            # Try to get price from ticker if available
            if ticker is not None:
                try:
                    if hasattr(ticker, "midpoint") and not np.isnan(ticker.midpoint()):
                        option_price = ticker.midpoint()
                    elif (
                        hasattr(ticker, "close")
                        and ticker.close
                        and not np.isnan(ticker.close)
                    ):
                        option_price = ticker.close
                    elif (
                        hasattr(ticker, "last")
                        and ticker.last
                        and not np.isnan(ticker.last)
                    ):
                        option_price = ticker.last
                except (AttributeError, TypeError):
                    option_price = None

            # If we got a valid price, estimate vega as 10% of option price
            if option_price and option_price > 0:
                vega_value = option_price * 0.1  # Rough estimate - 10% of option price
                vega_source = "price_estimation"
            else:
                # Fallback: estimate based on option type and strike distance from spot
                # This is a rough approximation when no price data is available
                try:
                    # Basic vega estimation: typically 0.05-0.30 depending on moneyness and time
                    # For ATM options with ~30-60 days, vega is usually around 0.10-0.20
                    vega_value = 0.15  # Conservative middle estimate
                    vega_source = "default_estimate"
                except Exception:
                    vega_value = 0.1  # Final fallback
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
        self.greeks_cache.put(cache_key, vega_value)
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

        # Progressive processing: early termination when sufficient opportunities found
        MAX_OPPORTUNITIES_PER_SYMBOL = 10
        opportunities_created = 0

        # Sort strikes by distance from current price for progressive processing
        # Use stock_price from the current context or get it again
        current_stock_price = await self._get_current_stock_price(stock_contract)
        if current_stock_price is None:
            logger.warning(
                f"[{symbol}] Cannot get current stock price for progressive processing"
            )
            sorted_strikes = list(strikes_data.keys())
        else:
            sorted_strikes = sorted(
                strikes_data.keys(), key=lambda sk: abs(sk[0] - current_stock_price)
            )

        for strike, option_type in sorted_strikes:
            legs = strikes_data[(strike, option_type)]

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

            # Early termination check - stop if we have enough good opportunities
            if opportunities_created >= MAX_OPPORTUNITIES_PER_SYMBOL:
                logger.info(
                    f"[{symbol}] Found {MAX_OPPORTUNITIES_PER_SYMBOL} opportunities, stopping early for performance"
                )
                break

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

                    # Apply price-based filters - check for invalid net_debit values first
                    if net_debit is None or np.isnan(net_debit) or net_debit <= 0:
                        if net_debit is None:
                            rejection_reason = "net_debit is None"
                        elif np.isnan(net_debit):
                            rejection_reason = "net_debit is NaN"
                        else:
                            rejection_reason = f"net_debit <= 0: ${net_debit:.2f}"

                        logger.info(
                            f"[{symbol}] {option_type} {strike}: Invalid net_debit - {rejection_reason}"
                        )
                        metrics_collector.add_rejection_reason(
                            RejectionReason.INVALID_CONTRACT_DATA,
                            {
                                "symbol": symbol,
                                "strike": strike,
                                "option_type": option_type,
                                "net_debit": net_debit,
                                "reason": rejection_reason,
                                "front_price": front_leg.price,
                                "back_price": back_leg.price,
                            },
                        )
                        continue

                    # Apply cost limit filter
                    if net_debit > self.config.max_net_debit:
                        logger.info(
                            f"[{symbol}] {option_type} {strike}: Net debit exceeds limit: ${net_debit:.2f} (max: ${self.config.max_net_debit:.0f})"
                        )
                        metrics_collector.add_rejection_reason(
                            RejectionReason.COST_LIMIT_EXCEEDED,
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

                    # Apply profit target filtering
                    profit_ratio = max_profit / net_debit if net_debit > 0 else 0.0
                    if profit_ratio < self.config.target_profit_ratio:
                        logger.info(
                            f"[{symbol}] {option_type} {strike}: Profit ratio insufficient: {profit_ratio:.2f} < {self.config.target_profit_ratio:.2f} (max_profit: ${max_profit:.2f}, net_debit: ${net_debit:.2f})"
                        )
                        metrics_collector.add_rejection_reason(
                            RejectionReason.PROFIT_RATIO_THRESHOLD_NOT_MET,
                            {
                                "symbol": symbol,
                                "strike": strike,
                                "option_type": option_type,
                                "profit_ratio": profit_ratio,
                                "target_ratio": self.config.target_profit_ratio,
                                "max_profit": max_profit,
                                "net_debit": net_debit,
                            },
                        )
                        continue

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

                    # Calculate profitability boundaries
                    lower_breakeven = None
                    upper_breakeven = None
                    profitability_range = None

                    if current_stock_price is not None:
                        try:
                            lower_breakeven, upper_breakeven, boundary_max_loss = (
                                self._calculate_breakeven_boundaries(
                                    current_stock_price,
                                    strike,
                                    front_leg.iv,
                                    back_leg.iv,
                                    front_leg.days_to_expiry,
                                    back_leg.days_to_expiry,
                                    option_type,
                                    front_leg.price,
                                    back_leg.price,
                                )
                            )

                            # Calculate profitability range
                            if (
                                lower_breakeven is not None
                                and upper_breakeven is not None
                            ):
                                profitability_range = upper_breakeven - lower_breakeven

                            logger.debug(
                                f"[{symbol}] {option_type} {strike} boundaries: "
                                f"Lower: ${lower_breakeven:.2f}, Upper: ${upper_breakeven:.2f}, Range: ${profitability_range:.2f}"
                            )

                        except Exception as e:
                            logger.debug(
                                f"[{symbol}] {option_type} {strike}: Boundary calculation failed: {e}"
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
                        lower_breakeven=lower_breakeven,
                        upper_breakeven=upper_breakeven,
                        profitability_range=profitability_range,
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
                    opportunities_created += 1

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
        # Use target_profit_ratio as the baseline, giving higher scores for exceeding target
        profit_score = min(
            1.0, profit_ratio / (self.config.target_profit_ratio * 2.0)
        )  # Normalize to 2x target ratio

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

    def _build_calendar_spread_order(
        self, opportunity: CalendarSpreadOpportunity, quantity: int
    ) -> Tuple[Contract, Order]:
        """
        Build calendar spread order with front and back month legs.
        Calendar spread: Sell front month, Buy back month (net debit position)
        """
        # Import ComboLeg here to avoid potential circular imports
        from ib_async import ComboLeg

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

        # Calculate limit price (net debit) - use optimized pricing
        limit_price = self.calculate_optimized_limit_price(opportunity)

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

    def _get_time_adjustment_factor(self) -> float:
        """Get pricing adjustment based on US market time (adjusted for Israel timezone)"""
        from datetime import datetime

        import pytz

        # Get current time in Israel timezone
        israel_tz = pytz.timezone("Asia/Jerusalem")
        now_israel = datetime.now(israel_tz)

        # Convert to US Eastern time (market timezone)
        us_eastern = pytz.timezone("US/Eastern")
        now_eastern = now_israel.astimezone(us_eastern)

        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now_eastern.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_eastern.replace(hour=16, minute=0, second=0, microsecond=0)

        if market_open <= now_eastern <= market_close:
            # During market hours
            if now_eastern.hour < 11:  # Morning (9:30-11:00)
                return 1.1  # More aggressive pricing
            elif now_eastern.hour >= 15:  # Afternoon (3:00-4:00)
                return 1.2  # Even more aggressive near close
            else:  # Midday (11:00-3:00)
                return 1.0  # Standard pricing
        else:
            # Outside market hours - more conservative
            return 0.9

    async def scan(
        self,
        symbol_list: List[str],
        cost_limit: float = 500.0,
        profit_target: float = 0.3,
        quantity: int = 1,
    ) -> None:
        """
        Main scanning method for calendar spread strategy with continuous execution.
        Runs until an order is filled, similar to Synthetic.py pattern.

        Args:
            symbol_list: List of symbols to scan
            cost_limit: Maximum cost for calendar spread
            profit_target: Target profit ratio
            quantity: Number of spreads to execute
        """
        # Global contract ticker for calendar spreads
        global contract_ticker
        contract_ticker = {}

        # Update configuration
        self.config.max_net_debit = cost_limit

        # Validate and set profit target
        if profit_target <= 0.0:
            logger.warning(
                f"Invalid profit_target {profit_target}, using default 0.3 (30%)"
            )
            self.config.target_profit_ratio = 0.3
        elif profit_target > 3.0:
            logger.warning(
                f"Profit target {profit_target} seems high (>300%), using 3.0"
            )
            self.config.target_profit_ratio = 3.0
        else:
            self.config.target_profit_ratio = profit_target

        self.quantity = quantity

        try:
            # Optimized IB connection setup
            await self._setup_optimized_ib_connection()
            logger.info(
                "✓ Connected to Interactive Brokers for calendar spread scanning with optimized settings"
            )

            # Register order fill handler
            self.ib.orderStatusEvent += self.onFill

            # Register master executor for handling market data events
            self.ib.pendingTickersEvent += self.master_executor

            logger.info(
                f"Starting continuous calendar spread analysis for {len(symbol_list)} symbols"
            )
            logger.info(
                f"Configuration: Max debit=${cost_limit:.0f}, Target profit={profit_target:.1%}, Quantity={quantity}"
            )

            # Continuous scanning until order is filled
            while not self.order_filled:
                # Start cycle tracking and performance monitoring
                metrics_collector.start_cycle(len(symbol_list))
                self.profiler.start_timer("full_scan_cycle")

                # Perform cache maintenance periodically
                self._perform_cache_maintenance()

                # Clear opportunities from previous cycle
                self.global_manager.clear_opportunities()
                logger.info(
                    f"Starting new calendar spread cycle: scanning {len(symbol_list)} symbols for global best opportunity"
                )

                # Phase 1: Collect opportunities from all symbols
                tasks = []
                for symbol in symbol_list:
                    # Check if order was filled during symbol processing
                    if self.order_filled:
                        break

                    # Use throttled scanning instead of fixed delays
                    task = asyncio.create_task(
                        self.scan_with_throttle(
                            symbol, self.scan_calendar_spreads, self.quantity
                        )
                    )
                    tasks.append(task)
                    # Minimal delay for API rate limiting
                    await asyncio.sleep(0.1)

                # Wait for all symbols to complete their opportunity scanning
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Log any exceptions from scanning
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Error scanning {symbol_list[i]}: {str(result)}")

                # Phase 2: Global opportunity selection and execution
                opportunity_count = self.global_manager.get_opportunity_count()
                logger.info(
                    f"Collected {opportunity_count} calendar spread opportunities across all symbols"
                )

                # Log detailed cycle summary
                self.global_manager.log_cycle_summary()

                if opportunity_count > 0:
                    # Get the globally best opportunity
                    best_opportunity = self.global_manager.get_best_opportunity()
                    if best_opportunity:
                        # Execute the globally best opportunity
                        logger.info(
                            f"Executing globally best calendar spread: [{best_opportunity.symbol}] "
                            f"with composite score: {best_opportunity.composite_score:.3f}"
                        )

                        # Log detailed trade information
                        logger.info(
                            f"[{best_opportunity.symbol}] Global best calendar spread details:"
                        )
                        logger.info(f"  Strike: {best_opportunity.strike}")
                        logger.info(f"  Option Type: {best_opportunity.option_type}")
                        logger.info(
                            f"  Front Expiry: {best_opportunity.front_leg.expiry} ({best_opportunity.front_leg.days_to_expiry}d)"
                        )
                        logger.info(
                            f"  Back Expiry: {best_opportunity.back_leg.expiry} ({best_opportunity.back_leg.days_to_expiry}d)"
                        )
                        logger.info(f"  IV Spread: {best_opportunity.iv_spread:.1f}%")
                        logger.info(
                            f"  Theta Ratio: {best_opportunity.theta_ratio:.2f}"
                        )
                        logger.info(f"  Net Debit: ${best_opportunity.net_debit:.2f}")
                        logger.info(f"  Max Profit: ${best_opportunity.max_profit:.2f}")

                        # Display boundary information if available
                        if (
                            best_opportunity.lower_breakeven is not None
                            and best_opportunity.upper_breakeven is not None
                        ):
                            logger.info(
                                f"  Breakeven Range: ${best_opportunity.lower_breakeven:.2f} - ${best_opportunity.upper_breakeven:.2f} "
                                f"(Range: ${best_opportunity.profitability_range:.2f})"
                            )
                        else:
                            logger.info("  Breakeven Range: Not calculated")

                        logger.info(
                            f"  Liquidity Score: {best_opportunity.combined_liquidity_score:.3f}"
                        )

                        try:
                            # Build and execute the calendar spread order
                            calendar_contract, order = (
                                self._build_calendar_spread_order(
                                    best_opportunity, quantity
                                )
                            )

                            # Execute the trade
                            await self.order_manager.place_order(
                                calendar_contract, order
                            )
                            logger.info(
                                f"Successfully executed global best calendar spread for {best_opportunity.symbol}"
                            )

                        except Exception as e:
                            logger.error(
                                f"Failed to execute global best calendar spread: {str(e)}"
                            )
                    else:
                        logger.warning(
                            "No best opportunity returned despite having opportunities"
                        )
                else:
                    logger.info("No calendar spread opportunities found this cycle")

                # Finish cycle tracking
                metrics_collector.finish_cycle()

                # Print cycle metrics summary
                if len(metrics_collector.scan_metrics) > 0:
                    metrics_collector.print_summary()

                # Check if order was filled before continuing
                if self.order_filled:
                    logger.info("Order filled - exiting calendar spread scan loop")
                    break

                # End performance timing for this cycle
                cycle_duration = self.profiler.end_timer("full_scan_cycle")
                logger.debug(f"Scan cycle completed in {cycle_duration:.2f} seconds")

                # Reset for next iteration
                contract_ticker = {}
                await asyncio.sleep(5)  # Wait between cycles

        except Exception as e:
            logger.error(f"Error in calendar spread scan loop: {str(e)}")
        finally:
            # Always print final metrics summary before exiting
            logger.info(
                "Calendar spread scanning complete - printing final metrics summary"
            )
            if len(metrics_collector.scan_metrics) > 0:
                metrics_collector.print_summary()

            # Print performance profiling report
            self.profiler.log_performance_report()

            # Deactivate all executors and disconnect from IB
            logger.info("Deactivating all executors and disconnecting from IB")
            self.deactivate_all_executors()
            if self.ib.isConnected():
                self.ib.disconnect()
                logger.info("✓ Disconnected from Interactive Brokers")

    async def scan_calendar_spreads(self, symbol: str, quantity: int) -> Optional[dict]:
        """
        Scan for calendar spread opportunities for a specific symbol.
        Reports opportunities to global manager instead of executing directly.

        Args:
            symbol: Trading symbol to scan
            quantity: Number of contracts to trade

        Returns:
            Dict with processing results or None if processing failed
        """
        start_time = time.time()

        try:
            # Get options chain and create opportunities
            opportunities = await self._scan_symbol_for_calendar_spreads(symbol)

            scan_time = time.time() - start_time
            opportunities_reported = 0

            if opportunities:
                logger.info(
                    f"✓ [{symbol}] Found {len(opportunities)} calendar spread opportunities (scan time: {scan_time:.1f}s)"
                )

                # Report all opportunities to global manager
                for opportunity in opportunities:
                    success = self.global_manager.add_opportunity(symbol, opportunity)
                    if success:
                        opportunities_reported += 1
                        profit_ratio = (
                            opportunity.max_profit / opportunity.net_debit
                            if opportunity.net_debit > 0
                            else 0.0
                        )
                        logger.info(
                            f"[{symbol}] Reported calendar spread opportunity: "
                            f"{opportunity.option_type} {opportunity.strike} "
                            f"IV: {opportunity.iv_spread:.1f}% Profit ratio: {profit_ratio:.1%} Score: {opportunity.composite_score:.3f}"
                        )

                logger.info(
                    f"[{symbol}] Reported {opportunities_reported} calendar spread opportunities to global manager"
                )

                return {
                    "symbol": symbol,
                    "opportunities_count": opportunities_reported,
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
        self, symbol: str
    ) -> List[CalendarSpreadOpportunity]:
        """Scan a single symbol for calendar spread opportunities"""
        try:
            # Start metrics collection for this scan
            metrics_collector.start_scan(symbol, "Calendar")

            logger.info(f"[{symbol}] Starting calendar spread scan with parameters:")
            logger.info(f"  Max net debit: ${self.config.max_net_debit:.0f}")
            logger.info(f"  Target profit ratio: {self.config.target_profit_ratio:.1%}")
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
        """Select expiries using optimized approach for calendar spreads"""
        today = datetime.now().date()

        # Target specific DTE ranges that historically perform well for calendars
        front_targets = [14, 21, 30, 35, 42]  # Weekly/monthly cycles
        back_targets = [60, 75, 90, 105, 120]  # Back month targets

        # Parse all expiries with DTE calculation
        expiry_data = []
        for expiry_str in all_expiries:
            try:
                expiry_date = datetime.strptime(expiry_str, "%Y%m%d").date()
                days_to_expiry = (expiry_date - today).days

                # Skip past expiries and very short-term (< 7 days)
                if days_to_expiry < 7:
                    continue

                expiry_data.append((expiry_str, days_to_expiry))
            except ValueError:
                logger.warning(f"Invalid expiry format: {expiry_str}")
                continue

        if not expiry_data:
            logger.warning("No valid expiries found")
            return []

        selected_expiries = []

        # Find closest expiry to each target DTE
        for target_dte in front_targets + back_targets:
            if expiry_data:
                closest_expiry = min(
                    expiry_data, key=lambda exp: abs(exp[1] - target_dte)
                )

                # Only add if not already selected and within reasonable range
                if (
                    closest_expiry[0] not in selected_expiries
                    and abs(closest_expiry[1] - target_dte) <= 10
                ):  # Within 10 days of target
                    selected_expiries.append(closest_expiry[0])

        # If we don't have enough expiries, add the most liquid ones
        if len(selected_expiries) < 6:
            # Sort by DTE and add missing ones
            remaining = [
                exp[0]
                for exp in sorted(expiry_data, key=lambda x: x[1])
                if exp[0] not in selected_expiries
            ]
            selected_expiries.extend(remaining[: 12 - len(selected_expiries)])

        # Limit to 12 expiries total for optimal performance
        final_selection = selected_expiries[:12]

        logger.debug(
            f"Optimized expiry selection: {len(final_selection)} expiries from {len(all_expiries)} total"
        )

        return final_selection

    def _select_calendar_strikes(
        self, all_strikes: List[float], stock_price: float
    ) -> List[float]:
        """Select strikes with highest calendar spread probability using optimized filtering"""
        # Focus on ATM and slightly OTM strikes (better time decay characteristics)
        optimal_range = (stock_price * 0.92, stock_price * 1.08)  # ±8% range

        candidates = [
            s for s in all_strikes if optimal_range[0] <= s <= optimal_range[1]
        ]

        if not candidates:
            # Fallback to wider range if no strikes in optimal range
            fallback_range = (stock_price * 0.85, stock_price * 1.15)
            candidates = [
                s for s in all_strikes if fallback_range[0] <= s <= fallback_range[1]
            ]

        # Sort by distance from current price, prioritize slightly OTM
        def strike_priority(strike):
            distance = abs(strike - stock_price)
            # Slight preference for strikes 2-5% OTM (better calendar spread characteristics)
            otm_bonus = (
                -0.1 if stock_price * 1.02 <= strike <= stock_price * 1.05 else 0
            )
            return distance + otm_bonus

        candidates.sort(key=strike_priority)

        # Limit to 8 most promising strikes for optimal performance
        selected = candidates[:8]

        logger.debug(
            f"Strike selection: {len(selected)} strikes from {len(all_strikes)} total "
            f"(optimal range: {optimal_range[0]:.2f}-{optimal_range[1]:.2f})"
        )

        return selected

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
        """Request market data for multiple contracts with parallel processing"""
        try:
            # Optimal batch size for IB API (expert recommendation)
            BATCH_SIZE = 50

            tickers = []

            # Process contracts in batches to respect API limits
            for i in range(0, len(contracts), BATCH_SIZE):
                batch = contracts[i : i + BATCH_SIZE]

                # Create all ticker requests in parallel for this batch
                batch_tickers = [
                    self.ib.reqMktData(contract, "", False, False) for contract in batch
                ]
                tickers.extend(batch_tickers)

                # Brief delay between batches for API rate limiting
                if i + BATCH_SIZE < len(contracts):
                    await asyncio.sleep(0.1)

            # Wait for market data with intelligent timeout
            await self._wait_for_market_data_smart(tickers)

            # Update global contract_ticker for executors
            for ticker in tickers:
                if ticker and hasattr(ticker, "contract"):
                    contract_ticker[ticker.contract.conId] = ticker

            return tickers

        except Exception as e:
            logger.error(f"Error requesting market data: {str(e)}")
            return []

    async def _wait_for_market_data_smart(self, tickers: List[Ticker]) -> None:
        """Smart waiting with adaptive timeout based on data readiness"""
        if not tickers:
            return

        max_wait_time = 30.0
        check_interval = 0.5
        min_ready_percentage = 0.75  # Proceed when 75% of data is ready

        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            ready_count = sum(
                1
                for ticker in tickers
                if ticker
                and not _safe_isnan(ticker.midpoint())
                and ticker.midpoint() > 0
            )

            ready_percentage = ready_count / len(tickers) if tickers else 0

            if ready_percentage >= min_ready_percentage:
                logger.info(
                    f"Market data ready: {ready_percentage:.1%} ({ready_count}/{len(tickers)})"
                )
                break

            await asyncio.sleep(check_interval)

        final_ready = sum(
            1
            for ticker in tickers
            if ticker and not _safe_isnan(ticker.midpoint()) and ticker.midpoint() > 0
        )
        logger.info(
            f"Final market data status: {final_ready}/{len(tickers)} contracts ready"
        )

    async def _setup_optimized_ib_connection(self) -> None:
        """Setup IB connection with optimal parameters for performance"""
        import os

        # Use environment variables or defaults for connection parameters
        ib_host = os.getenv("IB_HOST", "127.0.0.1")
        ib_port = int(os.getenv("IB_PORT", "7497"))
        client_id = int(os.getenv("IB_CLIENT_ID", "1"))

        # Connect to IB
        await self.ib.connectAsync(ib_host, ib_port, clientId=client_id)

        # Configure optimal settings for calendar spread scanning
        try:
            # Increase request limits for better performance
            if hasattr(self.ib.client, "MaxRequests"):
                self.ib.client.MaxRequests = 100

            # Use delayed frozen data for faster response in scanning mode
            # This is acceptable for calendar spread analysis as we need relative pricing
            self.ib.reqMarketDataType(3)  # Delayed frozen data

            # Clear any pending requests to start fresh
            self.ib.reqGlobalCancel()

            # Brief pause to allow settings to take effect
            await asyncio.sleep(0.5)

            logger.debug("IB connection optimized for calendar spread performance")

        except Exception as e:
            logger.info(
                f"Some IB optimization settings failed (continuing anyway): {e}"
            )

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
