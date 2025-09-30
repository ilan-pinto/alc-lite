"""
SFR (Synthetic-Free-Risk) arbitrage strategy class.

This module contains the main SFR strategy class that implements
the Synthetic-Free-Risk arbitrage strategy.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from ib_async import IB, Option

from ..common import get_logger
from ..metrics import RejectionReason, metrics_collector
from ..Strategy import ArbitrageClass
from .executor import SFRExecutor
from .models import ExpiryOption
from .validation import StrikeValidator

# Global contract_ticker for use in SFRExecutor
contract_ticker = {}

logger = get_logger()


class SFR(ArbitrageClass):
    """
    Synthetic-Free-Risk (SFR) arbitrage strategy class.
    This class uses a modular approach with improved maintainability and reusability.
    """

    def __init__(self, log_file: str = None):
        super().__init__(log_file=log_file)
        # Default strike selection parameters for backward compatibility with tests
        self.max_combinations = 10
        self.max_strike_difference = 5
        self.strike_validator = StrikeValidator()

    def cleanup_inactive_executors(self):
        """Enhanced cleanup that properly cancels market data subscriptions"""
        inactive_symbols = [
            symbol
            for symbol, executor in self.active_executors.items()
            if not executor.is_active
        ]

        # Ensure inactive executors have been properly deactivated
        for symbol in inactive_symbols:
            executor = self.active_executors[symbol]
            if hasattr(executor, "deactivate") and executor.is_active:
                # Call deactivate to clean up market data subscriptions
                executor.deactivate()

        # Call parent cleanup method
        super().cleanup_inactive_executors()

        if inactive_symbols:
            logger.info(
                f"[SFR] Enhanced cleanup completed for {len(inactive_symbols)} inactive executors"
            )

    async def scan(
        self,
        symbol_list,
        cost_limit,
        profit_target=0.50,
        volume_limit=100,
        quantity=1,
        max_combinations=10,
        max_strike_difference=5,
    ):
        """
        scan for SFR and execute order

        symbol list - list of valid symbols
        cost_limit - min price for the contract. e.g limit=50 means willing to pay up to 5000$
        profit_target - min acceptable min roi
        volume_limit - min option contract volume
        quantity - number of contracts to trade
        """
        # Global
        global contract_ticker
        contract_ticker = {}

        # set configuration
        self.profit_target = profit_target
        self.volume_limit = volume_limit
        self.cost_limit = cost_limit
        self.quantity = quantity
        self.max_combinations = max_combinations
        self.max_strike_difference = max_strike_difference

        await self.ib.connectAsync("127.0.0.1", 7497, clientId=2)
        self.ib.orderStatusEvent += self.onFill

        # Set up single event handler for all symbols
        self.ib.pendingTickersEvent += self.master_executor

        try:
            while not self.order_filled:
                # Check if executor is paused per ADR-003
                current_symbol = getattr(self, "current_scanning_symbol", None)
                if self.is_paused(current_symbol):
                    logger.debug(
                        "Executor is paused - waiting during parallel execution"
                    )
                    await asyncio.sleep(0.1)  # Small delay while paused
                    continue

                # Start cycle tracking
                _ = metrics_collector.start_cycle(len(symbol_list))

                tasks = []
                for symbol in symbol_list:
                    # Check if order was filled during symbol processing
                    if self.order_filled:
                        break

                    # Check if this symbol's executor should be paused per ADR-003
                    if self.is_paused(symbol):
                        logger.debug(f"[{symbol}] Skipping scan - executor is paused")
                        continue

                    # Use throttled scanning instead of fixed delays
                    task = asyncio.create_task(
                        self.scan_with_throttle(
                            symbol,
                            self.scan_sfr,
                            self.quantity,
                            self.profit_target,
                            self.cost_limit,
                        )
                    )
                    tasks.append(task)
                    # Minimal delay for API rate limiting
                    await asyncio.sleep(0.1)

                # Wait for all tasks to complete
                _ = await asyncio.gather(*tasks, return_exceptions=True)

                # Clean up inactive executors
                self.cleanup_inactive_executors()

                # Finish cycle tracking
                metrics_collector.finish_cycle()

                # Print metrics summary periodically
                if len(metrics_collector.scan_metrics) > 0:
                    metrics_collector.print_summary()

                # Check if order was filled before continuing
                # For parallel execution, wait until all legs are complete
                if self.order_filled:
                    if self.parallel_execution_in_progress:
                        if self.parallel_execution_complete:
                            logger.info(
                                "All parallel legs completed - exiting scan loop"
                            )
                            break
                        else:
                            logger.info(
                                f"Order filled for {self.active_parallel_symbol} but parallel execution still in progress - continuing scan"
                            )
                            # Reset order_filled to false so scan continues until parallel execution completes
                            self.order_filled = False
                    else:
                        logger.info("Order filled (sequential) - exiting scan loop")
                        break

                # Reset for next iteration
                contract_ticker = {}
                await asyncio.sleep(2)  # Reduced wait time for faster cycles
        except Exception as e:
            logger.error(f"Error in scan loop: {str(e)}")
        finally:
            # Always print final metrics summary before exiting
            logger.info("Scanning complete - printing final metrics summary")
            if len(metrics_collector.scan_metrics) > 0:
                metrics_collector.print_summary()

            # Deactivate all executors and disconnect from IB
            logger.info("Deactivating all executors and disconnecting from IB")
            self.deactivate_all_executors()
            self.ib.disconnect()

    def find_stock_position_in_strikes(
        self, stock_price: float, valid_strikes: List[float]
    ) -> int:
        """
        Find the position of stock price within valid strikes array.
        Returns the index of the strike closest to or just below the stock price.
        """
        return self.strike_validator.find_stock_position_in_strikes(
            stock_price, valid_strikes
        )

    async def validate_strikes_for_expiry(
        self, symbol: str, expiry: str, potential_strikes: List[float]
    ) -> List[float]:
        """
        Validate which strikes actually exist for a specific expiry.

        Args:
            symbol: Trading symbol
            expiry: Specific expiry date (e.g., '20250912')
            potential_strikes: List of strikes to validate

        Returns:
            List of strikes that actually exist for this expiry
        """
        # Initialize cache if not exists
        if not hasattr(self, "expiry_strike_cache"):
            self.expiry_strike_cache = {}

        # Check cache first (5 minute TTL for strike validation)
        cache_key = f"{symbol}_{expiry}"
        current_time = time.time()
        if hasattr(self, "cache_timestamps"):
            cached_time = self.cache_timestamps.get(cache_key, 0)
            if current_time - cached_time < 300:  # 5 minute cache
                cached_strikes = self.expiry_strike_cache.get(symbol, {}).get(expiry)
                if cached_strikes is not None:
                    logger.debug(
                        f"[{symbol}] Using cached strikes for {expiry}: {len(cached_strikes)} strikes"
                    )
                    return list(cached_strikes)

        # Limit strikes to reasonable range to avoid overwhelming API
        # We'll get the stock price when this is called from scan_sfr
        nearby_strikes = potential_strikes[:20]  # Limit to first 20 for API efficiency

        if not nearby_strikes:
            return []

        logger.debug(
            f"[{symbol}] Validating {len(nearby_strikes)} strikes for expiry {expiry}"
        )

        # Create test contracts for batch validation
        test_contracts = []
        for strike in nearby_strikes:
            # Test with call options (calls usually have same availability as puts)
            test_contract = Option(symbol, expiry, strike, "C", "SMART")
            test_contracts.append(test_contract)

        valid_strikes = []

        try:
            # Use qualifyContractsAsync for batch validation
            qualified_contracts = await self.ib.qualifyContractsAsync(*test_contracts)

            for i, qualified in enumerate(qualified_contracts):
                if qualified and hasattr(qualified, "conId") and qualified.conId:
                    valid_strikes.append(nearby_strikes[i])

        except Exception as e:
            logger.warning(f"[{symbol}] Strike validation failed for {expiry}: {e}")
            # Fallback: assume all strikes are valid (better than blocking)
            valid_strikes = nearby_strikes

        # Cache the result
        if symbol not in self.expiry_strike_cache:
            self.expiry_strike_cache[symbol] = {}
        self.expiry_strike_cache[symbol][expiry] = set(valid_strikes)

        if not hasattr(self, "cache_timestamps"):
            self.cache_timestamps = {}
        self.cache_timestamps[cache_key] = current_time

        logger.info(
            f"[{symbol}] Expiry {expiry}: {len(valid_strikes)}/{len(nearby_strikes)} strikes are valid"
        )
        if len(valid_strikes) < len(nearby_strikes):
            invalid_strikes = [s for s in nearby_strikes if s not in valid_strikes]
            logger.debug(f"[{symbol}] Invalid strikes for {expiry}: {invalid_strikes}")

        return valid_strikes

    async def parallel_qualify_all_contracts_with_validation(
        self,
        symbol: str,
        strikes_by_expiry: Dict[str, List[float]],
        valid_strike_pairs: List[Tuple[float, float]],
    ) -> Dict:
        """
        Qualify option contracts using expiry-specific strike validation.
        Only creates contracts for strikes that actually exist for each expiry.
        """
        all_options_to_qualify = []
        expiry_contract_map = {}

        for expiry, valid_strikes_for_expiry in strikes_by_expiry.items():
            valid_strikes_set = set(valid_strikes_for_expiry)

            for call_strike, put_strike in valid_strike_pairs:
                # Only create contracts if BOTH strikes exist for this specific expiry
                if call_strike in valid_strikes_set and put_strike in valid_strikes_set:
                    call = Option(symbol, expiry, call_strike, "C", "SMART")
                    put = Option(symbol, expiry, put_strike, "P", "SMART")
                    all_options_to_qualify.extend([call, put])

                    key = f"{expiry}_{call_strike}_{put_strike}"
                    expiry_contract_map[key] = {
                        "call_original": call,
                        "put_original": put,
                        "expiry": expiry,
                        "call_strike": call_strike,
                        "put_strike": put_strike,
                    }
                else:
                    # Log when we skip invalid combinations
                    missing_strikes = []
                    if call_strike not in valid_strikes_set:
                        missing_strikes.append(f"call {call_strike}")
                    if put_strike not in valid_strikes_set:
                        missing_strikes.append(f"put {put_strike}")
                    logger.debug(
                        f"[{symbol}] Skipping {expiry} - missing strikes: {', '.join(missing_strikes)}"
                    )

        if not all_options_to_qualify:
            logger.warning(
                f"[{symbol}] No valid option contracts to qualify after expiry-specific filtering"
            )
            return {}

        logger.info(
            f"[{symbol}] Qualifying {len(all_options_to_qualify)} contracts ({len(all_options_to_qualify)//2} strike pairs) using expiry-specific validation"
        )

        try:
            # Single parallel qualification for ALL validated contracts
            qualified_contracts = await self.qualify_contracts_cached(
                *all_options_to_qualify
            )
        except Exception as e:
            logger.error(f"[{symbol}] Contract qualification failed: {e}")
            return {}

        # Map qualified contracts back to their original contracts
        qualified_map = {}
        original_to_qualified = {}

        # Build mapping from original to qualified contracts
        for i, qualified in enumerate(qualified_contracts):
            if i < len(all_options_to_qualify):
                original_to_qualified[id(all_options_to_qualify[i])] = qualified

        # Build final result mapping with strike verification
        rejected_mismatches = 0
        for key, contract_info in expiry_contract_map.items():
            call_qualified = original_to_qualified.get(
                id(contract_info["call_original"])
            )
            put_qualified = original_to_qualified.get(id(contract_info["put_original"]))

            if call_qualified and put_qualified:
                # Critical: Verify strikes match exactly after IB qualification
                call_strike_match = (
                    abs(call_qualified.strike - contract_info["call_strike"]) < 0.01
                )
                put_strike_match = (
                    abs(put_qualified.strike - contract_info["put_strike"]) < 0.01
                )

                if not call_strike_match:
                    logger.warning(
                        f"[{symbol}] STRIKE MISMATCH REJECTED: Call requested {contract_info['call_strike']}, "
                        f"IB returned {call_qualified.strike} for {contract_info['expiry']}"
                    )
                    rejected_mismatches += 1
                    continue

                if not put_strike_match:
                    logger.warning(
                        f"[{symbol}] STRIKE MISMATCH REJECTED: Put requested {contract_info['put_strike']}, "
                        f"IB returned {put_qualified.strike} for {contract_info['expiry']}"
                    )
                    rejected_mismatches += 1
                    continue

                # Only add to map if strikes match exactly
                qualified_map[key] = {
                    "call_contract": call_qualified,
                    "put_contract": put_qualified,
                    "expiry": contract_info["expiry"],
                    "call_strike": contract_info["call_strike"],
                    "put_strike": contract_info["put_strike"],
                }

        if rejected_mismatches > 0:
            logger.warning(
                f"[{symbol}] Rejected {rejected_mismatches} contracts due to strike mismatches from IB qualification"
            )

        logger.info(
            f"[{symbol}] Successfully qualified {len(qualified_map)} strike combinations after expiry and strike validation"
        )
        return qualified_map

    async def scan_sfr(self, symbol, quantity=1, profit_target=0.50, cost_limit=120.0):
        """
        Scan for SFR opportunities for a specific symbol.
        Creates a single executor per symbol that handles all expiries.

        Args:
            symbol: Trading symbol to scan
            quantity: Number of contracts to trade
            profit_target: Minimum profit target (default: 0.50%)
            cost_limit: Maximum cost limit (default: $120.0)
        """
        # Start metrics collection for this scan
        _ = metrics_collector.start_scan(symbol, "SFR")

        # Set configuration for this scan
        self.profit_target = profit_target
        self.cost_limit = cost_limit

        # Ensure strike selection parameters have defaults (for backward compatibility)
        if not hasattr(self, "max_combinations"):
            self.max_combinations = 10
        if not hasattr(self, "max_strike_difference"):
            self.max_strike_difference = 5

        try:
            _, _, stock = self._get_stock_contract(symbol)

            # Request market data for the stock
            market_data = await self._get_market_data_async(stock)

            stock_price = (
                market_data.last
                if not np.isnan(market_data.last)
                else market_data.close
            )

            logger.info(f"price for [{symbol}: {stock_price} ]")

            # Early filter: Skip stocks above cost limit to avoid unnecessary API calls
            if stock_price > cost_limit:
                logger.info(
                    f"[{symbol}] Stock price ${stock_price:.2f} exceeds cost limit ${cost_limit:.2f} - skipping scan"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.STOCK_PRICE_ABOVE_LIMIT,
                    {
                        "symbol": symbol,
                        "stock_price": stock_price,
                        "cost_limit": cost_limit,
                    },
                )
                # Don't call finish_scan here - we haven't actually started scanning yet
                return

            # Request options chain
            chain = await self._get_chain(stock, exchange="SMART")

            # PyPy optimization: Cache stock_price as local variable and use optimized
            # list comprehension (PyPy's JIT heavily optimizes list comprehensions)
            stock_price_local = stock_price
            chain_strikes = chain.strikes
            potential_strikes = [
                s for s in chain_strikes if abs(s - stock_price_local) <= 25
            ]

            logger.info(
                f"[{symbol}] Chain has {len(chain_strikes)} total strikes across all expiries"
            )
            logger.info(
                f"[{symbol}] {len(potential_strikes)} strikes within $25 of stock price (${stock_price:.2f})"
            )

            # Get valid expiries first
            valid_expiries = self.filter_expirations_within_range(
                chain.expirations, 15, 50
            )

            if len(valid_expiries) == 0:
                logger.warning(
                    f"No valid expiries found for {symbol} in range 15-50 days, skipping scan"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.NO_VALID_EXPIRIES,
                    {
                        "available_expiries": len(chain.expirations),
                        "days_range": "15-50",
                    },
                )
                return

            # Validate strikes for each expiry (this is the key fix!)
            strikes_by_expiry = {}
            total_valid_strikes = 0

            for expiry in valid_expiries:
                valid_strikes_for_expiry = await self.validate_strikes_for_expiry(
                    symbol, expiry, potential_strikes
                )
                if valid_strikes_for_expiry:
                    strikes_by_expiry[expiry] = valid_strikes_for_expiry
                    total_valid_strikes += len(valid_strikes_for_expiry)

            if total_valid_strikes < 2:
                logger.info(
                    f"Not enough valid strikes found for {symbol} across all expiries (found: {total_valid_strikes})"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.INSUFFICIENT_VALID_STRIKES,
                    {
                        "symbol": symbol,
                        "valid_strikes_count": total_valid_strikes,
                        "required_strikes": 2,
                        "stock_price": stock_price,
                        "expiries_tested": len(valid_expiries),
                    },
                )
                return

            logger.info(
                f"[{symbol}] Strike validation complete: {len(strikes_by_expiry)} expiries with valid strikes"
            )

            # Now we need to update the logic to work with expiry-specific strikes
            # For now, let's combine all valid strikes to maintain existing logic
            all_valid_strikes = set()
            for expiry_strikes in strikes_by_expiry.values():
                all_valid_strikes.update(expiry_strikes)

            combined_valid_strikes = sorted(list(all_valid_strikes))

            if len(combined_valid_strikes) < 2:
                logger.info(
                    f"Not enough combined valid strikes for {symbol}, skipping parallel qualification"
                )
                return

            # Adaptive strike position logic for conversion arbitrage
            valid_strike_pairs = []

            # Sort strikes for position-based selection
            sorted_strikes = sorted(combined_valid_strikes)

            # Find stock price position within valid strikes
            stock_position = self.find_stock_position_in_strikes(
                stock_price, sorted_strikes
            )

            # Adaptive strike ranges based on position (not dollar amounts)
            # Call candidates: stock position ± 3 (expanded for more opportunities)
            call_start = max(0, stock_position - 3)
            call_end = min(len(sorted_strikes), stock_position + 4)
            call_candidates = sorted_strikes[call_start:call_end]

            # Put candidates: stock position -2 to +2 (expanded range for flexibility)
            put_start = max(0, stock_position - 2)
            put_end = min(len(sorted_strikes), stock_position + 3)
            put_candidates = sorted_strikes[put_start:put_end]

            # Generate combinations with expanded strike differences
            priority_combinations = []  # Strike difference = 1-2 (highest priority)
            secondary_combinations = []  # Strike difference = 3-5 (lower priority)

            for call_strike in call_candidates:
                for put_strike in put_candidates:
                    # Enforce call_strike > put_strike for proper conversion arbitrage
                    if call_strike > put_strike:
                        # Calculate strike difference in position terms
                        call_idx = sorted_strikes.index(call_strike)
                        put_idx = sorted_strikes.index(put_strike)
                        strike_diff = call_idx - put_idx

                        combination = (call_strike, put_strike, strike_diff)

                        if strike_diff in [1, 2]:
                            # 1-2 strike difference: highest probability of trade
                            priority_combinations.append(combination)
                        elif strike_diff <= self.max_strike_difference:
                            # 3+ strike difference: configurable secondary options
                            secondary_combinations.append(combination)

            # Sort by strike difference (lower differences have higher priority) for optimal trade probability
            priority_combinations.sort(key=lambda x: x[2])
            secondary_combinations.sort(key=lambda x: x[2])

            # Combine and limit to configurable number of best combinations
            all_combinations = priority_combinations + secondary_combinations
            valid_strike_pairs = [
                (call, put)
                for call, put, _ in all_combinations[: self.max_combinations]
            ]

            # Track strike selection effectiveness
            total_strikes = len(chain.strikes)
            combinations_generated = len(all_combinations)
            combinations_tested = len(valid_strike_pairs)
            metrics_collector.record_strike_effectiveness(
                symbol,
                total_strikes,
                len(combined_valid_strikes),
                combinations_generated,
                combinations_tested,
            )

            logger.info(
                f"[{symbol}] Testing {len(valid_strike_pairs)} conversion-optimized strike combinations "
                f"(stock position: {stock_position}, price: ${stock_price:.2f})"
            )

            if priority_combinations:
                logger.info(
                    f"[{symbol}] Found {len(priority_combinations)} high-probability 1-2 strike difference combinations"
                )
            if secondary_combinations:
                logger.info(
                    f"[{symbol}] Found {len(secondary_combinations)} secondary 3-{self.max_strike_difference} strike difference combinations"
                )

            # Parallel qualification of all contracts using expiry-specific strikes
            qualified_contracts_map = (
                await self.parallel_qualify_all_contracts_with_validation(
                    symbol, strikes_by_expiry, valid_strike_pairs
                )
            )

            # Build expiry options from qualified contracts
            expiry_options = []
            all_contracts = [stock]

            # Limit expiry options to avoid too many low volume contracts
            max_expiry_options = (
                12  # Expanded limit to accommodate more strike combinations
            )

            for expiry in valid_expiries:
                if len(expiry_options) >= max_expiry_options:
                    logger.debug(
                        f"[{symbol}] Reached maximum expiry options limit ({max_expiry_options})"
                    )
                    break

                # Try all strike combinations for this expiry (prioritized by volume)
                found_valid_combination = False
                for call_strike, put_strike in valid_strike_pairs:
                    key = f"{expiry}_{call_strike}_{put_strike}"
                    if key in qualified_contracts_map:
                        contract_info = qualified_contracts_map[key]
                        expiry_option = ExpiryOption(
                            expiry=contract_info["expiry"],
                            call_contract=contract_info["call_contract"],
                            put_contract=contract_info["put_contract"],
                            call_strike=contract_info["call_strike"],
                            put_strike=contract_info["put_strike"],
                        )
                        expiry_options.append(expiry_option)
                        all_contracts.extend(
                            [
                                contract_info["call_contract"],
                                contract_info["put_contract"],
                            ]
                        )
                        found_valid_combination = True
                        logger.debug(
                            f"Using strike combination C{call_strike}/P{put_strike} for {symbol} expiry {expiry}"
                        )
                        break  # Found valid combination, move to next expiry

                if not found_valid_combination:
                    logger.debug(
                        f"No valid contract pair found for {symbol} expiry {expiry}"
                    )

            if not expiry_options:
                logger.info(f"No valid expiry options found for {symbol}")
                metrics_collector.add_rejection_reason(
                    RejectionReason.INVALID_CONTRACT_DATA,
                    {
                        "symbol": symbol,
                        "reason": "no valid expiry options found",
                        "total_expiries_checked": len(
                            self.filter_expirations_within_range(
                                chain.expirations, 19, 45
                            )
                        ),
                    },
                )
                return

            # Log the total number of combinations being scanned
            logger.info(
                f"[{symbol}] Scanning {len(expiry_options)} strike combinations across "
                f"{len(set(opt.expiry for opt in expiry_options))} expiries"
            )

            # Clean up any existing executor for this symbol before creating new one
            if symbol in self.active_executors:
                old_executor = self.active_executors[symbol]
                logger.info(
                    f"[{symbol}] Cleaning up existing executor before creating new one"
                )
                old_executor.deactivate()
                del self.active_executors[symbol]

            # Create single executor for this symbol using the modular SFRExecutor
            srf_executor = SFRExecutor(
                ib=self.ib,
                order_manager=self.order_manager,
                stock_contract=stock,
                expiry_options=expiry_options,
                symbol=symbol,
                profit_target=self.profit_target,
                cost_limit=self.cost_limit,
                start_time=time.time(),
                quantity=quantity,
                data_timeout=5.0,  # Give more time for data collection
                strategy=self,
            )

            # Set the global contract_ticker reference
            srf_executor.set_contract_ticker_reference(contract_ticker)

            # Store executor and request market data for all contracts
            self.active_executors[symbol] = srf_executor

            # Request market data for all contracts
            logger.info(
                f"[{symbol}] Requesting market data for {len(all_contracts)} contracts "
                f"({len(expiry_options)} expiry options)"
            )

            # Request market data for all contracts in parallel
            data_collection_start = time.time()
            await self.request_market_data_batch(all_contracts)
            data_collection_time = time.time() - data_collection_start

            # Record metrics
            metrics_collector.record_contracts_count(len(all_contracts))
            metrics_collector.record_data_collection_time(data_collection_time)
            metrics_collector.record_expiries_scanned(len(expiry_options))

            logger.info(
                f"Created executor for {symbol} with {len(expiry_options)} expiry options "
                f"({len(all_contracts)} total contracts)"
            )

            # Don't finish scan here - let the executor finish it when done processing
            # The executor will call metrics_collector.finish_scan() when it's inactive

        except Exception as e:
            logger.error(f"Error in scan_sfr for {symbol}: {str(e)}")
            metrics_collector.finish_scan(success=False, error_message=str(e))
