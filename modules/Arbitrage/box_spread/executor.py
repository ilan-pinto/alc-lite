"""
Box spread execution engine.

This module provides the BoxExecutor class which handles the execution
of box spread arbitrage opportunities, following the BaseExecutor patterns
from the existing codebase.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from ib_async import IB, ComboLeg, Contract, LimitOrder, Order, Ticker

from ..common import get_logger
from ..metrics import RejectionReason, metrics_collector
from ..Strategy import OrderManagerClass
from .models import BoxSpreadConfig, BoxSpreadOpportunity
from .utils import _safe_isnan

logger = get_logger()

# Global contract ticker for compatibility with existing patterns
contract_ticker = {}


class BoxExecutor:
    """
    Executes box spread arbitrage opportunities.

    Follows the established patterns from the existing codebase while providing
    enhanced functionality for box spread specific execution logic.
    """

    def __init__(
        self,
        opportunity: BoxSpreadOpportunity,
        ib: IB,
        order_manager: OrderManagerClass,
        config: BoxSpreadConfig = None,
    ) -> None:
        self.opportunity = opportunity
        self.ib = ib
        self.order_manager = order_manager
        self.config = config or BoxSpreadConfig()

        # Execution state
        self.start_time = time.time()
        self.is_active = True
        self.execution_completed = False

        # Track contracts and their market data
        self.contracts = [
            opportunity.long_call_k1.contract,
            opportunity.short_call_k2.contract,
            opportunity.short_put_k1.contract,
            opportunity.long_put_k2.contract,
        ]

        # Required data for execution decision
        self.required_tickers = {contract.conId: None for contract in self.contracts}

        logger.info(f"BoxExecutor initialized for {opportunity.symbol} box spread")

    async def executor(self, event) -> None:
        """
        Main executor method that processes market data events.

        This follows the established pattern from the original BoxExecutor
        but with improved error handling and validation.
        """
        if not self.is_active or self.execution_completed:
            return

        try:
            for tick in event:
                ticker = tick
                contract = ticker.contract

                # Update global contract_ticker for compatibility
                if self._is_valid_ticker_data(ticker):
                    contract_ticker[contract.conId] = ticker
                    self.required_tickers[contract.conId] = ticker

                elif self._should_cancel_due_to_poor_liquidity(ticker):
                    logger.warning(
                        f"[{self.opportunity.symbol}] REJECTED - Poor liquidity: "
                        f"bid/ask sizes below minimum threshold"
                    )
                    metrics_collector.add_rejection_reason(
                        RejectionReason.MISSING_MARKET_DATA,
                        {"symbol": self.opportunity.symbol, "reason": "Poor liquidity"},
                    )
                    await self._cancel_execution("Poor liquidity")
                    return

                # Check if we have all required data
                if self._have_all_required_data():
                    execution_time = time.time() - self.start_time
                    logger.info(f"Box spread data complete in {execution_time:.2f}s")

                    # Evaluate execution opportunity
                    should_execute, limit_price = self._evaluate_execution_opportunity()

                    if should_execute:
                        await self._execute_box_spread(limit_price)
                    else:
                        logger.warning(
                            f"[{self.opportunity.symbol}] REJECTED - Execution criteria not met: "
                            f"arbitrage conditions no longer favorable"
                        )
                        metrics_collector.add_rejection_reason(
                            RejectionReason.ARBITRAGE_CONDITION_NOT_MET,
                            {
                                "symbol": self.opportunity.symbol,
                                "reason": "Execution criteria not met",
                            },
                        )
                        await self._cancel_execution("Execution criteria not met")

        except Exception as e:
            logger.error(f"Error in box spread executor: {e}")
            await self._cancel_execution(f"Executor error: {str(e)}")

    def _is_valid_ticker_data(self, ticker: Ticker) -> bool:
        """
        Validate ticker data quality.

        Args:
            ticker: Market data ticker

        Returns:
            True if ticker data is valid for execution
        """
        required_fields = [ticker.askSize, ticker.bidSize, ticker.bid, ticker.ask]

        # Check for NaN values
        if any(_safe_isnan(x) for x in required_fields):
            return False

        # Check for positive values
        if any(x <= 0 for x in required_fields):
            return False

        # Check minimum liquidity requirements
        min_size = self.config.min_volume_per_leg
        if ticker.askSize < min_size or ticker.bidSize < min_size:
            return False

        return True

    def _should_cancel_due_to_poor_liquidity(self, ticker: Ticker) -> bool:
        """
        Determine if execution should be cancelled due to poor liquidity.

        Args:
            ticker: Market data ticker

        Returns:
            True if liquidity is too poor to continue
        """
        # Cancel if bid/ask sizes fall below minimum thresholds
        if not _safe_isnan(ticker.askSize) and not _safe_isnan(ticker.bidSize):
            if ticker.askSize < 5 and ticker.bidSize < 5:
                return True

        return False

    def _have_all_required_data(self) -> bool:
        """Check if we have valid market data for all required contracts"""
        return all(ticker is not None for ticker in self.required_tickers.values())

    def _evaluate_execution_opportunity(self) -> Tuple[bool, float]:
        """
        Evaluate whether the current market conditions justify execution.

        Returns:
            Tuple of (should_execute, limit_price)
        """
        try:
            # Get current market prices
            current_prices = self._get_current_execution_prices()
            if not current_prices:
                return False, 0.0

            long_call_price, short_call_price, short_put_price, long_put_price = (
                current_prices
            )

            # Calculate current net debit
            current_net_debit = (long_call_price + long_put_price) - (
                short_call_price + short_put_price
            )

            # Calculate current arbitrage profit
            strike_width = self.opportunity.upper_strike - self.opportunity.lower_strike
            current_arbitrage_profit = strike_width - current_net_debit

            # Apply safety buffer
            safety_buffer = current_net_debit * self.config.safety_buffer
            adjusted_profit = current_arbitrage_profit - safety_buffer

            # Check if still profitable
            if adjusted_profit <= self.config.min_absolute_profit:
                logger.warning(
                    f"[{self.opportunity.symbol}] REJECTED - Profit target not met: "
                    f"adjusted_profit=${adjusted_profit:.4f} <= min_absolute_profit=${self.config.min_absolute_profit:.4f}"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.PROFIT_TARGET_NOT_MET,
                    {
                        "symbol": self.opportunity.symbol,
                        "adjusted_profit": adjusted_profit,
                        "min_absolute_profit": self.config.min_absolute_profit,
                    },
                )
                return False, 0.0

            # Check profit percentage
            if current_net_debit > 0:
                profit_pct = (adjusted_profit / current_net_debit) * 100
                if profit_pct < self.config.min_arbitrage_profit * 100:
                    logger.warning(
                        f"[{self.opportunity.symbol}] REJECTED - ROI too low: "
                        f"profit_pct={profit_pct:.2f}% < min_required={self.config.min_arbitrage_profit * 100:.2f}%"
                    )
                    metrics_collector.add_rejection_reason(
                        RejectionReason.MIN_ROI_NOT_MET,
                        {
                            "symbol": self.opportunity.symbol,
                            "profit_pct": profit_pct,
                            "min_arbitrage_profit": self.config.min_arbitrage_profit
                            * 100,
                        },
                    )
                    return False, 0.0

            # Calculate limit price (net debit we're willing to pay)
            limit_price = current_net_debit + safety_buffer

            logger.info(
                f"Box spread execution approved: "
                f"net_debit={current_net_debit:.4f}, "
                f"profit={adjusted_profit:.4f}, "
                f"limit_price={limit_price:.4f}"
            )

            return True, limit_price

        except Exception as e:
            logger.error(f"Error evaluating execution opportunity: {e}")
            return False, 0.0

    def _get_current_execution_prices(
        self,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Get current execution prices for all legs.

        Returns:
            Tuple of (long_call_price, short_call_price, short_put_price, long_put_price)
            or None if data not available
        """
        try:
            tickers = list(self.required_tickers.values())
            if len(tickers) != 4 or any(t is None for t in tickers):
                return None

            # Map tickers to legs (order matters!)
            long_call_ticker = self.required_tickers[
                self.opportunity.long_call_k1.contract.conId
            ]
            short_call_ticker = self.required_tickers[
                self.opportunity.short_call_k2.contract.conId
            ]
            short_put_ticker = self.required_tickers[
                self.opportunity.short_put_k1.contract.conId
            ]
            long_put_ticker = self.required_tickers[
                self.opportunity.long_put_k2.contract.conId
            ]

            # Use conservative pricing: pay ask for longs, receive bid for shorts
            # Apply small slippage buffer (5% move towards mid)
            long_call_price = long_call_ticker.ask * 1.05  # Pay slightly more
            short_call_price = short_call_ticker.bid * 0.95  # Receive slightly less
            short_put_price = short_put_ticker.bid * 0.95  # Receive slightly less
            long_put_price = long_put_ticker.ask * 1.05  # Pay slightly more

            return long_call_price, short_call_price, short_put_price, long_put_price

        except Exception as e:
            logger.error(f"Error getting current execution prices: {e}")
            return None

    async def _execute_box_spread(self, limit_price: float) -> None:
        """
        Execute the box spread using a combo order.

        Args:
            limit_price: Maximum net debit to pay
        """
        try:
            logger.info(
                f"Executing box spread for {self.opportunity.symbol} at limit {limit_price:.4f}"
            )

            # Build the combo contract and order
            box_contract, order = self._build_box_order(limit_price)

            # Place the order
            trade = await self.order_manager.place_order(box_contract, order)

            if trade:
                logger.info(f"Box spread order placed successfully: {trade}")
                metrics_collector.record_opportunity_found()
                self.execution_completed = True
            else:
                logger.warning(
                    f"[{self.opportunity.symbol}] REJECTED - Order placement failed: "
                    f"unable to place box spread combo order"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.ORDER_REJECTED,
                    {
                        "symbol": self.opportunity.symbol,
                        "reason": "Order placement failed",
                    },
                )

            # Cleanup
            await self._cleanup_execution()

        except Exception as e:
            logger.error(f"Error executing box spread: {e}")
            await self._cancel_execution(f"Execution error: {str(e)}")

    def _build_box_order(self, limit_price: float) -> Tuple[Contract, Order]:
        """
        Build the combo contract and order for the box spread.

        Args:
            limit_price: Net debit limit price

        Returns:
            Tuple of (box_contract, order)
        """
        # Create combo legs for the box spread
        combo_legs = []

        # Long call at K1 (buy)
        combo_legs.append(
            ComboLeg(
                conId=self.opportunity.long_call_k1.contract.conId,
                ratio=1,
                action="BUY",
                exchange=self.opportunity.long_call_k1.contract.exchange,
            )
        )

        # Short call at K2 (sell)
        combo_legs.append(
            ComboLeg(
                conId=self.opportunity.short_call_k2.contract.conId,
                ratio=1,
                action="SELL",
                exchange=self.opportunity.short_call_k2.contract.exchange,
            )
        )

        # Short put at K1 (sell)
        combo_legs.append(
            ComboLeg(
                conId=self.opportunity.short_put_k1.contract.conId,
                ratio=1,
                action="SELL",
                exchange=self.opportunity.short_put_k1.contract.exchange,
            )
        )

        # Long put at K2 (buy)
        combo_legs.append(
            ComboLeg(
                conId=self.opportunity.long_put_k2.contract.conId,
                ratio=1,
                action="BUY",
                exchange=self.opportunity.long_put_k2.contract.exchange,
            )
        )

        # Create combo contract
        box_contract = Contract(
            symbol=self.opportunity.symbol,
            comboLegs=combo_legs,
            exchange="SMART",
            secType="BAG",
            currency="USD",
        )

        # Create limit order
        order = LimitOrder(
            action="BUY",  # We're buying the box spread (net debit)
            totalQuantity=1,
            lmtPrice=round(limit_price, self.config.price_precision_decimals),
        )

        return box_contract, order

    async def _cancel_execution(self, reason: str) -> None:
        """
        Cancel the execution and cleanup resources.

        Args:
            reason: Reason for cancellation
        """
        logger.info(f"Cancelling box spread execution: {reason}")
        self.is_active = False
        await self._cleanup_execution()

    async def _cleanup_execution(self) -> None:
        """Cleanup market data subscriptions and event handlers"""
        try:
            # Remove executor from pending tickers event
            if hasattr(self.ib, "pendingTickersEvent"):
                self.ib.pendingTickersEvent -= self.executor

            # Cancel market data for all contracts
            for contract in self.contracts:
                self.ib.cancelMktData(contract)

            # Clear from global contract_ticker
            for contract in self.contracts:
                if contract.conId in contract_ticker:
                    del contract_ticker[contract.conId]

            logger.debug("Box spread execution cleanup completed")

        except Exception as e:
            logger.error(f"Error during box spread cleanup: {e}")

    def get_execution_status(self) -> Dict:
        """
        Get current execution status information.

        Returns:
            Dictionary with execution status details
        """
        return {
            "symbol": self.opportunity.symbol,
            "lower_strike": self.opportunity.lower_strike,
            "upper_strike": self.opportunity.upper_strike,
            "is_active": self.is_active,
            "execution_completed": self.execution_completed,
            "runtime_seconds": time.time() - self.start_time,
            "data_received": len(
                [t for t in self.required_tickers.values() if t is not None]
            ),
            "data_required": len(self.required_tickers),
            "arbitrage_profit": self.opportunity.arbitrage_profit,
            "net_debit": self.opportunity.net_debit,
        }
