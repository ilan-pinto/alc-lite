"""
Main strategy orchestration for Synthetic arbitrage.

This module contains the main Syn class responsible for:
- Symbol scanning coordination
- Strike validation and contract qualification
- Integration with global opportunity management
- Global opportunity execution
"""

import asyncio
import time
from typing import Dict, List, Tuple

import numpy as np
from ib_async import Option

from ..common import get_logger
from ..metrics import RejectionReason, metrics_collector
from ..Strategy import ArbitrageClass
from .constants import CACHE_TTL
from .data_collector import contract_ticker
from .executor import SynExecutor
from .global_opportunity_manager import GlobalOpportunityManager
from .models import ExpiryOption, GlobalOpportunity
from .scoring import ScoringConfig
from .validation import ValidationEngine, strike_cache

logger = get_logger()


class Syn(ArbitrageClass):
    """
    Synthetic arbitrage strategy class.
    This class uses a global opportunity manager to find the best opportunities
    across all symbols and expiries, then executes only the globally optimal trade.
    """

    def __init__(
        self,
        log_file: str = None,
        scoring_config: ScoringConfig = None,
    ):
        super().__init__(log_file=log_file)
        self.active_executors: Dict[str, SynExecutor] = {}
        self.global_manager = GlobalOpportunityManager(scoring_config)

    async def scan(
        self,
        symbol_list: List[str],
        cost_limit: float,
        max_loss_threshold: float,
        max_profit_threshold: float,
        profit_ratio_threshold: float,
        quantity=1,
    ) -> None:
        """
        scan for Syn and execute order

        symbol list - list of valid symbols
        cost_limit - min price for the contract. e.g limit=50 means willing to pay up to 5000$
        max_loss - max loss for the contract. e.g max_loss=50 means willing to lose up to 5000$
        max_profit - max profit for the contract. e.g max_profit=50 means willing to profit up to 5000$
        quantity - number of contracts to trade
        """
        # Global
        global contract_ticker
        contract_ticker = {}

        # set configuration
        self.cost_limit = cost_limit
        self.max_loss_threshold = max_loss_threshold
        self.max_profit_threshold = max_profit_threshold
        self.profit_ratio_threshold = profit_ratio_threshold
        self.quantity = quantity
        await self.ib.connectAsync("127.0.0.1", 7497, clientId=2)
        self.ib.orderStatusEvent += self.onFill

        # Set up single event handler for all symbols
        self.ib.pendingTickersEvent += self.master_executor

        try:
            while not self.order_filled:
                # Start cycle tracking
                metrics_collector.start_cycle(len(symbol_list))

                # Clear opportunities from previous cycle
                self.global_manager.clear_opportunities()
                logger.info(
                    f"Starting new cycle: scanning {len(symbol_list)} symbols for global best opportunity"
                )

                # Phase 1: Collect opportunities from all symbols
                tasks = []
                for symbol in symbol_list:
                    # Check if order was filled during symbol processing
                    if self.order_filled:
                        break

                    # Use throttled scanning instead of fixed delays
                    task = asyncio.create_task(
                        self.scan_with_throttle(symbol, self.scan_syn, self.quantity)
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
                    f"Collected {opportunity_count} opportunities across all symbols"
                )

                # Log detailed cycle summary
                self.global_manager.log_cycle_summary()

                if opportunity_count > 0:
                    # Get the globally best opportunity
                    best_opportunity = self.global_manager.get_best_opportunity()

                    if best_opportunity:
                        # Execute the globally best opportunity
                        logger.info(
                            f"Executing globally best opportunity: [{best_opportunity.symbol}] "
                            f"with composite score: {best_opportunity.score.composite_score:.3f}"
                        )

                        # Log detailed trade information
                        trade_details = best_opportunity.trade_details
                        logger.info(
                            f"[{best_opportunity.symbol}] Global best trade details:"
                        )
                        logger.info(f"  Expiry: {trade_details.get('expiry', 'N/A')}")
                        logger.info(
                            f"  Max Profit: ${trade_details.get('max_profit', 0):.2f}"
                        )
                        logger.info(
                            f"  Min Profit: ${trade_details.get('min_profit', 0):.2f}"
                        )
                        logger.info(
                            f"  Risk-Reward Ratio: {best_opportunity.score.risk_reward_ratio:.3f}"
                        )
                        logger.info(
                            f"  Liquidity Score: {best_opportunity.score.liquidity_score:.3f}"
                        )
                        logger.info(
                            f"  Time Decay Score: {best_opportunity.score.time_decay_score:.3f}"
                        )
                        logger.info(
                            f"  Market Quality: {best_opportunity.score.market_quality_score:.3f}"
                        )

                        try:
                            # Execute the trade
                            await self.order_manager.place_order(
                                best_opportunity.conversion_contract,
                                best_opportunity.order,
                            )
                            logger.info(
                                f"Successfully executed global best opportunity for {best_opportunity.symbol}"
                            )

                            # Log the trade details
                            self._log_trade_details_from_opportunity(best_opportunity)

                        except Exception as e:
                            logger.error(
                                f"Failed to execute global best opportunity: {str(e)}"
                            )
                    else:
                        logger.warning(
                            "No best opportunity returned despite having opportunities"
                        )
                else:
                    logger.info(
                        "No opportunities found across all symbols in this cycle"
                    )

                # Clean up inactive executors
                self.cleanup_inactive_executors()

                # Finish cycle tracking
                metrics_collector.finish_cycle()

                # Print metrics summary periodically
                if len(metrics_collector.scan_metrics) > 0:
                    metrics_collector.print_summary()

                # Check if order was filled before continuing
                if self.order_filled:
                    logger.info("Order filled - exiting scan loop")
                    break

                # Reset for next iteration
                contract_ticker = {}
                await asyncio.sleep(5)  # Reduced wait time for faster cycles
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

    def _log_trade_details_from_opportunity(self, opportunity: GlobalOpportunity):
        """Helper method to log trade details from a global opportunity"""
        trade_details = opportunity.trade_details
        logger.info(f"[{opportunity.symbol}] Trade Details:")
        logger.info(
            f"  Call Strike: {trade_details.get('call_strike', 'N/A')}, "
            f"Call Price: ${trade_details.get('call_price', 0):.2f}"
        )
        logger.info(
            f"  Put Strike: {trade_details.get('put_strike', 'N/A')}, "
            f"Put Price: ${trade_details.get('put_price', 0):.2f}"
        )
        logger.info(f"  Stock Price: ${trade_details.get('stock_price', 0):.2f}")
        logger.info(f"  Net Credit: ${trade_details.get('net_credit', 0):.2f}")
        logger.info(f"  Min ROI: {trade_details.get('min_roi', 0):.2f}%")

    async def scan_syn(self, symbol: str, quantity: int) -> None:
        """
        Scan for Syn opportunities for a specific symbol.
        Creates a single executor per symbol that handles all expiries.
        """
        # Start metrics collection for this scan
        metrics_collector.start_scan(symbol, "Synthetic")

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
            if stock_price > self.cost_limit:
                logger.info(
                    f"[{symbol}] Stock price ${stock_price:.2f} exceeds cost limit ${self.cost_limit:.2f} - skipping scan"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.STOCK_PRICE_ABOVE_LIMIT,
                    {
                        "symbol": symbol,
                        "stock_price": stock_price,
                        "cost_limit": self.cost_limit,
                    },
                )
                # Don't call finish_scan here - we haven't actually started scanning yet
                return

            # Request options chain
            chain = await self._get_chain(stock)

            # PyPy optimization: Cache values as local variables and use optimized
            # list comprehensions (PyPy's JIT heavily optimizes list comprehensions)
            stock_price_local = stock_price
            chain_strikes = chain.strikes

            # Define parameters for the options (expiry and strike price)
            valid_strikes = [
                s
                for s in chain_strikes
                if s <= stock_price_local and s > stock_price_local - 10
            ]  # Example strike price

            if len(valid_strikes) < 2:
                logger.info(
                    f"Not enough valid strikes found for {symbol} (found: {len(valid_strikes)})"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.INSUFFICIENT_VALID_STRIKES,
                    {
                        "symbol": symbol,
                        "valid_strikes_count": len(valid_strikes),
                        "required_strikes": 2,
                        "stock_price": stock_price,
                    },
                )
                return

            # Prepare for parallel contract qualification with expiry-specific validation
            valid_expiries = self.filter_expirations_within_range(
                chain.expirations, 19, 45
            )

            if len(valid_strikes) < 2:
                logger.info(
                    f"Not enough valid strikes for {symbol}, skipping parallel qualification"
                )
                return

            # Get potential strikes for validation (within reasonable range)
            potential_strikes = [
                s for s in chain_strikes if abs(s - stock_price_local) <= 25
            ]

            logger.info(
                f"[{symbol}] Validating strikes for {len(valid_expiries)} expiries"
            )

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
                else:
                    logger.warning(f"[{symbol}] No valid strikes for expiry {expiry}")

            if not strikes_by_expiry:
                logger.warning(
                    f"[{symbol}] No valid strikes found for any expiry after validation"
                )
                return

            logger.info(
                f"[{symbol}] Validation complete: {total_valid_strikes} total valid strikes across {len(strikes_by_expiry)} expiries"
            )

            # Use validated strikes for strike selection
            # Get a representative set of strikes (from first expiry that has data)
            first_expiry_strikes = list(strikes_by_expiry.values())[0]
            first_expiry_strikes.sort()

            # Apply original strike selection logic to validated strikes
            filtered_strikes = [
                s
                for s in first_expiry_strikes
                if s <= stock_price and s > stock_price - 10
            ]

            if len(filtered_strikes) < 2:
                logger.info(
                    f"Not enough filtered strikes for {symbol} after validation, using all validated strikes"
                )
                filtered_strikes = first_expiry_strikes

            call_strike = filtered_strikes[-1]
            put_strike = filtered_strikes[-2]
            valid_strike_pairs = [(call_strike, put_strike)]

            # Add retry strike pair if available
            if len(filtered_strikes) >= 3:
                retry_put_strike = filtered_strikes[-3]
                valid_strike_pairs.append((call_strike, retry_put_strike))

            logger.info(
                f"[{symbol}] Using strike pairs: {valid_strike_pairs} across {len(strikes_by_expiry)} expiries"
            )

            # Parallel qualification with expiry-specific validation
            qualified_contracts_map = (
                await self.parallel_qualify_all_contracts_with_validation(
                    symbol, strikes_by_expiry, valid_strike_pairs
                )
            )

            # Build expiry options from qualified contracts
            expiry_options = []
            all_contracts = [stock]

            for expiry in valid_expiries:
                # Try primary strike combination first
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
                        [contract_info["call_contract"], contract_info["put_contract"]]
                    )
                    continue

                # Try retry strike combination if available
                if len(valid_strikes) >= 3:
                    retry_put_strike = valid_strikes[-3]
                    retry_key = f"{expiry}_{call_strike}_{retry_put_strike}"
                    if retry_key in qualified_contracts_map:
                        contract_info = qualified_contracts_map[retry_key]
                        logger.info(
                            f"Using retry put strike {retry_put_strike} for {symbol} expiry {expiry}"
                        )
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
                        continue

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

            # Create single executor for this symbol
            syn_executor = SynExecutor(
                ib=self.ib,
                order_manager=self.order_manager,
                stock_contract=stock,
                expiry_options=expiry_options,
                symbol=symbol,
                cost_limit=self.cost_limit,
                max_loss_threshold=self.max_loss_threshold,
                max_profit_threshold=self.max_profit_threshold,
                profit_ratio_threshold=self.profit_ratio_threshold,
                start_time=time.time(),
                global_manager=self.global_manager,
                quantity=quantity,
                data_timeout=45.0,  # Give more time for data collection
            )

            # Store executor and request market data for all contracts
            self.active_executors[symbol] = syn_executor

            # Clean up any stale data in contract_ticker for this symbol's contracts
            cleared_count = syn_executor._clear_symbol_tickers()
            if cleared_count > 0:
                logger.debug(
                    f"[{symbol}] Cleaned up {cleared_count} stale ticker entries"
                )

            # Request market data for all contracts with detailed logging
            logger.info(
                f"[{symbol}] Requesting market data for {len(all_contracts)} contracts:"
            )

            # Log stock contract
            logger.info(f"  Stock: {stock.symbol} (conId: {stock.conId})")

            # Log option contracts
            for expiry_option in expiry_options:
                logger.info(
                    f"  Call: {expiry_option.call_contract.symbol} {expiry_option.call_strike} "
                    f"{expiry_option.expiry} (conId: {expiry_option.call_contract.conId})"
                )
                logger.info(
                    f"  Put: {expiry_option.put_contract.symbol} {expiry_option.put_strike} "
                    f"{expiry_option.expiry} (conId: {expiry_option.put_contract.conId})"
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
            logger.error(f"Error in scan_syn for {symbol}: {str(e)}")
            metrics_collector.finish_scan(success=False, error_message=str(e))

    async def validate_strikes_for_expiry(
        self, symbol: str, expiry: str, potential_strikes: List[float]
    ) -> List[float]:
        """
        Validate which strikes actually exist for a specific expiry.
        Uses global caching to improve performance.

        Args:
            symbol: Trading symbol
            expiry: Specific expiry date (e.g., '20250912')
            potential_strikes: List of strikes to validate

        Returns:
            List of strikes that actually exist for this expiry
        """
        # Delegate to the validation engine
        validator = ValidationEngine(self.ib)
        return await validator.validate_strikes_for_expiry(
            symbol, expiry, potential_strikes
        )

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

        # Build final result mapping
        for key, contract_info in expiry_contract_map.items():
            call_qualified = original_to_qualified.get(
                id(contract_info["call_original"])
            )
            put_qualified = original_to_qualified.get(id(contract_info["put_original"]))

            if call_qualified and put_qualified:
                qualified_map[key] = {
                    "call_contract": call_qualified,
                    "put_contract": put_qualified,
                    "expiry": contract_info["expiry"],
                    "call_strike": contract_info["call_strike"],
                    "put_strike": contract_info["put_strike"],
                }

        logger.info(
            f"[{symbol}] Successfully qualified {len(qualified_map)} strike combinations after expiry validation"
        )
        return qualified_map
