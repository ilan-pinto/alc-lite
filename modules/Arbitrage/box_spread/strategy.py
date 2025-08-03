"""
Box spread strategy implementation.

This module contains the main BoxSpread class and related functionality.
Follows the established patterns from CalendarSpread while implementing
box spread specific business logic.
"""

import asyncio
import time
from datetime import datetime
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np
from ib_async import Contract, FuturesOption, Index, Option, OptionChain, Stock, Ticker

from modules.Arbitrage.Strategy import ArbitrageClass

from ..common import get_logger
from ..metrics import RejectionReason, metrics_collector
from .executor import BoxExecutor
from .models import BoxSpreadConfig, BoxSpreadLeg, BoxSpreadOpportunity
from .opportunity_manager import BoxOpportunityManager
from .utils import (
    AdaptiveCacheManager,
    PerformanceProfiler,
    TTLCache,
    VectorizedGreeksCalculator,
    _safe_isnan,
)

logger = get_logger()

# Global variable for contract ticker information (for backward compatibility)
contract_ticker = {}


class BoxSpread(ArbitrageClass):
    """
    Box Spread arbitrage strategy class.

    Box spreads are risk-free arbitrage opportunities when:
    1. Net debit < strike width (K2 - K1)
    2. All options have same expiry
    3. Sufficient liquidity exists for execution

    The strategy consists of 4 legs:
    - Long call at K1 (lower strike)
    - Short call at K2 (higher strike)
    - Short put at K1 (lower strike)
    - Long put at K2 (higher strike)

    This implementation includes:
    - Comprehensive opportunity detection
    - Risk-free validation
    - Global opportunity selection
    - Performance optimization with caching
    - Integration with existing order management
    """

    def __init__(self, log_file: str = None) -> None:
        """Initialize Box Spread strategy"""
        super().__init__(log_file)
        self.config = BoxSpreadConfig()

        # Global opportunity management
        self.global_manager = BoxOpportunityManager(self.config)

        # Box spread specific caching with TTL and size limits
        self.pricing_cache = TTLCache(max_size=2000, ttl_seconds=60)  # 1 minute TTL
        self.greeks_cache = TTLCache(max_size=5000, ttl_seconds=30)  # 30 second TTL
        self.leg_cache = TTLCache(max_size=3000, ttl_seconds=45)  # 45 second TTL

        # Adaptive cache manager for memory pressure handling
        self.cache_manager = AdaptiveCacheManager()

        # Performance profiler for monitoring and optimization
        self.profiler = PerformanceProfiler()

        # Strategy configuration
        self.range = 0.1  # Price range for strike selection
        self.profit_target = 0.01  # Minimum profit target (1%)
        self.max_spread = 50.0  # Maximum strike width
        self.profit_target_multiplier = 1.10  # Profit multiplier per iteration

        logger.info(
            "Box Spread strategy initialized with complete performance optimizations"
        )

    def _perform_cache_maintenance(self) -> None:
        """Perform cache maintenance including memory pressure cleanup"""
        caches = [self.pricing_cache, self.greeks_cache, self.leg_cache]

        # Regular cleanup of expired entries
        total_expired = sum(cache.cleanup_expired() for cache in caches)

        # Memory pressure cleanup if needed
        pressure_cleaned = self.cache_manager.cleanup_if_needed(caches)

        # Log cache statistics periodically
        pricing_size = self.pricing_cache.size()
        greeks_size = self.greeks_cache.size()
        leg_size = self.leg_cache.size()
        memory_stats = self.cache_manager.get_memory_stats()

        if total_expired > 0 or pressure_cleaned > 0:
            logger.info(
                f"Cache maintenance: expired={total_expired}, pressure_cleaned={pressure_cleaned}, "
                f"pricing_cache_size={pricing_size}, greeks_cache_size={greeks_size}, "
                f"leg_cache_size={leg_size}, memory={memory_stats.get('memory_percent', 'N/A')}"
            )

    async def box_stock_scanner(self, symbol: str) -> None:
        """
        Scan for box spread opportunities in a single stock.

        Args:
            symbol: Stock symbol to scan
        """
        try:
            # Define the underlying stock
            exchange, option_type, stock = self.get_stock_contract(symbol)

            # Request market data for the stock
            market_data = await self._get_market_data_async(stock)

            stock_price = (
                market_data.last
                if not _safe_isnan(market_data.last)
                else market_data.close
            )

            logger.info(
                f"Scanning box spreads for {symbol} at price ${stock_price:.2f}"
            )

            if symbol.startswith("!"):
                # Handle futures options (like !MES)
                chains = await self._get_chains(stock, exchange=exchange)
                tasks = []
                for chain in chains:
                    task = asyncio.create_task(
                        self.search_box_in_chain(chain, option_type, stock, stock_price)
                    )
                    tasks.append(task)
                await asyncio.gather(*tasks)
            else:
                # Handle regular stock options
                chain = await self._get_chain(stock, exchange=exchange)
                await self.search_box_in_chain(chain, option_type, stock, stock_price)

        except Exception as e:
            logger.error(f"Error scanning box spreads for {symbol}: {e}")
            metrics_collector.record_rejection(
                strategy="box_spread",
                symbol=symbol,
                reason=RejectionReason.SCAN_ERROR,
                details=str(e),
            )

    async def search_box_in_chain(
        self, chain: OptionChain, option_type, stock: Contract, stock_price: float
    ) -> None:
        """
        Search for box spread opportunities in an option chain.

        Args:
            chain: Option chain to search
            option_type: Option contract type (Option or FuturesOption)
            stock: Underlying stock contract
            stock_price: Current stock price
        """
        async with self.semaphore:
            try:
                # Select strikes around current stock price
                anchor_strikes = self._select_anchor_strikes(chain.strikes, stock_price)

                logger.debug(
                    f"Searching {len(anchor_strikes)} strikes for {stock.symbol}"
                )

                # Limit expiries to process (performance optimization)
                expirations_range = min(3, len(chain.expirations))  # Max 3 expiries

                for expiry in chain.expirations[:expirations_range]:
                    await self._process_expiry_for_box_spreads(
                        chain, expiry, anchor_strikes, option_type, stock
                    )

            except Exception as e:
                logger.error(
                    f"Error searching box spreads in chain for {stock.symbol}: {e}"
                )

    def _select_anchor_strikes(
        self, available_strikes: List[float], stock_price: float
    ) -> List[float]:
        """
        Select relevant strikes for box spread scanning.

        Args:
            available_strikes: All available strikes
            stock_price: Current stock price

        Returns:
            List of strikes to use for box spread combinations
        """
        # Filter strikes within range of current stock price
        anchor_strikes = [
            s
            for s in available_strikes
            if s < stock_price * (1 + self.range) and s > stock_price * (1 - self.range)
        ]

        # Sort strikes for consistent processing
        anchor_strikes.sort()

        # Limit number of strikes to avoid combinatorial explosion
        max_strikes = 10  # Reasonable limit for performance
        if len(anchor_strikes) > max_strikes:
            # Take strikes centered around stock price
            mid_idx = len(anchor_strikes) // 2
            half_range = max_strikes // 2
            start_idx = max(0, mid_idx - half_range)
            end_idx = min(len(anchor_strikes), start_idx + max_strikes)
            anchor_strikes = anchor_strikes[start_idx:end_idx]

        return anchor_strikes

    async def _process_expiry_for_box_spreads(
        self,
        chain: OptionChain,
        expiry: str,
        anchor_strikes: List[float],
        option_type,
        stock: Contract,
    ) -> None:
        """
        Process a single expiry for box spread opportunities.

        Args:
            chain: Option chain
            expiry: Expiry date to process
            anchor_strikes: List of strikes to consider
            option_type: Option contract type
            stock: Underlying stock contract
        """
        # Generate strike pairs (K1, K2) where K1 < K2
        strike_pairs = [
            (k1, k2)
            for k1, k2 in permutations(anchor_strikes, 2)
            if k1 < k2 and (k2 - k1) <= self.max_spread
        ]

        logger.debug(f"Processing {len(strike_pairs)} strike pairs for expiry {expiry}")

        # Process strike pairs in batches for performance
        batch_size = 5
        for i in range(0, len(strike_pairs), batch_size):
            batch = strike_pairs[i : i + batch_size]

            tasks = []
            for k1, k2 in batch:
                task = asyncio.create_task(
                    self._evaluate_box_spread_opportunity(
                        chain, expiry, k1, k2, option_type, stock
                    )
                )
                tasks.append(task)

            # Wait for batch to complete
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _evaluate_box_spread_opportunity(
        self,
        chain: OptionChain,
        expiry: str,
        k1_strike: float,
        k2_strike: float,
        option_type,
        stock: Contract,
    ) -> None:
        """
        Evaluate a specific box spread opportunity.

        Args:
            chain: Option chain
            expiry: Expiry date
            k1_strike: Lower strike (K1)
            k2_strike: Upper strike (K2)
            option_type: Option contract type
            stock: Underlying stock contract
        """
        try:
            # Create option contracts for the 4 legs
            contracts = await self._create_box_spread_contracts(
                stock.symbol, expiry, k1_strike, k2_strike, option_type, chain
            )

            if not contracts:
                return

            long_call_k1, short_call_k2, short_put_k1, long_put_k2 = contracts

            # Qualify contracts
            await self.ib.qualifyContractsAsync(*contracts)

            # Request market data
            for contract in contracts:
                self.ib.reqMktData(contract)

            # Create and start executor
            await self._create_and_start_executor(
                stock.symbol, k1_strike, k2_strike, expiry, contracts
            )

        except Exception as e:
            logger.debug(
                f"Error evaluating box spread {stock.symbol} K1={k1_strike} K2={k2_strike}: {e}"
            )

    async def _create_box_spread_contracts(
        self,
        symbol: str,
        expiry: str,
        k1_strike: float,
        k2_strike: float,
        option_type,
        chain: OptionChain,
    ) -> Optional[Tuple[Contract, Contract, Contract, Contract]]:
        """
        Create the 4 option contracts for a box spread.

        Returns:
            Tuple of (long_call_k1, short_call_k2, short_put_k1, long_put_k2)
        """
        try:
            # Long call at K1
            long_call_k1 = option_type(
                symbol=symbol,
                lastTradeDateOrContractMonth=expiry,
                strike=k1_strike,
                right="C",
                exchange=chain.exchange,
                tradingClass=chain.tradingClass,
            )

            # Short call at K2
            short_call_k2 = option_type(
                symbol=symbol,
                lastTradeDateOrContractMonth=expiry,
                strike=k2_strike,
                right="C",
                exchange=chain.exchange,
                tradingClass=chain.tradingClass,
            )

            # Short put at K1
            short_put_k1 = option_type(
                symbol=symbol,
                lastTradeDateOrContractMonth=expiry,
                strike=k1_strike,
                right="P",
                exchange=chain.exchange,
                tradingClass=chain.tradingClass,
            )

            # Long put at K2
            long_put_k2 = option_type(
                symbol=symbol,
                lastTradeDateOrContractMonth=expiry,
                strike=k2_strike,
                right="P",
                exchange=chain.exchange,
                tradingClass=chain.tradingClass,
            )

            return long_call_k1, short_call_k2, short_put_k1, long_put_k2

        except Exception as e:
            logger.debug(f"Error creating box spread contracts: {e}")
            return None

    async def _create_and_start_executor(
        self,
        symbol: str,
        k1_strike: float,
        k2_strike: float,
        expiry: str,
        contracts: Tuple[Contract, Contract, Contract, Contract],
    ) -> None:
        """
        Create and start a BoxExecutor for the opportunity.

        This creates a temporary opportunity object for the executor.
        The actual opportunity evaluation happens in the executor.
        """
        try:
            # Create a minimal opportunity object for the executor
            # The executor will do the real evaluation when market data arrives
            long_call_k1, short_call_k2, short_put_k1, long_put_k2 = contracts

            # Create placeholder legs (real data will come from market data)
            placeholder_leg = lambda contract, strike, right, action: BoxSpreadLeg(
                contract=contract,
                strike=strike,
                expiry=expiry,
                right=right,
                action=action,
                price=0.0,
                bid=0.0,
                ask=0.0,
                volume=0,
                iv=0.0,
                delta=0.0,
                gamma=0.0,
                theta=0.0,
                vega=0.0,
                days_to_expiry=30,
            )

            opportunity = BoxSpreadOpportunity(
                symbol=symbol,
                lower_strike=k1_strike,
                upper_strike=k2_strike,
                expiry=expiry,
                long_call_k1=placeholder_leg(long_call_k1, k1_strike, "C", "BUY"),
                short_call_k2=placeholder_leg(short_call_k2, k2_strike, "C", "SELL"),
                short_put_k1=placeholder_leg(short_put_k1, k1_strike, "P", "SELL"),
                long_put_k2=placeholder_leg(long_put_k2, k2_strike, "P", "BUY"),
                strike_width=k2_strike - k1_strike,
                net_debit=0.0,
                theoretical_value=k2_strike - k1_strike,
                arbitrage_profit=0.0,
                profit_percentage=0.0,
                max_profit=0.0,
                max_loss=0.0,
                risk_free=False,
                total_bid_ask_spread=0.0,
                combined_liquidity_score=0.0,
                execution_difficulty=0.0,
                net_delta=0.0,
                net_gamma=0.0,
                net_theta=0.0,
                net_vega=0.0,
                composite_score=0.0,
            )

            # Create and register executor
            box_executor = BoxExecutor(
                opportunity=opportunity,
                ib=self.ib,
                order_manager=self.order_manager,
                config=self.config,
            )

            # Register executor with pending tickers event
            self.ib.pendingTickersEvent += box_executor.executor

        except Exception as e:
            logger.error(f"Error creating box executor: {e}")

    def get_stock_contract(self, symbol: str) -> Tuple[str, type, Contract]:
        """
        Get stock contract based on symbol type.

        Args:
            symbol: Trading symbol (may have prefixes like ! or @)

        Returns:
            Tuple of (exchange, option_type, stock_contract)
        """
        exchange = "SMART"
        option_type = Option

        if symbol.startswith("!"):
            symbol = symbol[1:]  # Remove ! prefix
            stock = Index(symbol, exchange="CME", currency="USD")
            exchange = "CME"
            option_type = FuturesOption
        elif symbol.startswith("@"):
            symbol = symbol[1:]  # Remove @ prefix
            stock = Index(symbol, exchange="CBOE", currency="USD")
            exchange = "CBOE"
        else:
            stock = Stock(symbol, "SMART", "USD")

        return exchange, option_type, stock

    async def scan(
        self,
        symbol_list: List[str],
        range: float = 0.1,
        profit_target: float = 0.01,
        max_spread: float = 50.0,
        profit_target_multiplier: float = 1.10,
        clientId: int = 3,
    ) -> None:
        """
        Main scan method for box spread opportunities.

        Args:
            symbol_list: List of symbols to scan
            range: Price range for strike selection (default 10%)
            profit_target: Minimum profit target (default 1%)
            max_spread: Maximum strike width (default $50)
            profit_target_multiplier: Profit multiplier per iteration
            clientId: IB client ID
        """
        # Set configuration
        self.range = range
        self.profit_target = profit_target
        self.max_spread = max_spread
        self.profit_target_multiplier = profit_target_multiplier

        # Update config object
        self.config.min_arbitrage_profit = profit_target
        self.config.max_strike_width = max_spread

        # Connect to IB
        await self.ib.connectAsync("127.0.0.1", 7497, clientId=clientId)

        # Initialize global contract ticker
        global contract_ticker
        contract_ticker = {}

        # Start opportunity manager scan
        self.global_manager.start_scan()

        logger.info(f"Starting box spread scan for {len(symbol_list)} symbols")

        try:
            while True:
                # Scan all symbols in parallel
                tasks = []
                for symbol in symbol_list:
                    task = asyncio.create_task(self.box_stock_scanner(symbol))
                    tasks.append(task)

                await asyncio.gather(*tasks, return_exceptions=True)

                # Perform cache maintenance
                self._perform_cache_maintenance()

                # Log scan progress
                scan_summary = self.global_manager.get_scan_summary()
                logger.info(f"Scan iteration complete: {scan_summary}")

                # Clear contract ticker and reset for next iteration
                contract_ticker.clear()

                # Wait before next iteration
                await asyncio.sleep(30)

                # Increase profit target for next iteration
                self.profit_target *= self.profit_target_multiplier

        except KeyboardInterrupt:
            logger.info("Box spread scan interrupted by user")
        except Exception as e:
            logger.error(f"Error in box spread scan: {e}")
        finally:
            # Final cleanup
            scan_summary = self.global_manager.get_scan_summary()
            logger.info(f"Box spread scan completed: {scan_summary}")


# Convenience function for running box spread strategy
async def run_box_spread_strategy(
    symbol_list: List[str],
    range: float = 0.1,
    profit_target: float = 0.01,
    max_spread: float = 50.0,
    client_id: int = 3,
) -> None:
    """
    Convenience function to run box spread strategy.

    Args:
        symbol_list: List of symbols to scan
        range: Price range for strike selection
        profit_target: Minimum profit target
        max_spread: Maximum strike width
        client_id: IB client ID
    """
    strategy = BoxSpread()
    await strategy.scan(
        symbol_list=symbol_list,
        range=range,
        profit_target=profit_target,
        max_spread=max_spread,
        clientId=client_id,
    )
