import asyncio
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import logging
import numpy as np
from eventkit import Event
from ib_async import IB, Contract, Option, Order, Ticker

from modules.Arbitrage.Strategy import ArbitrageClass, BaseExecutor, OrderManagerClass

from .common import get_logger
from .metrics import RejectionReason, metrics_collector

# Global contract_ticker for use in SynExecutor and patching in tests
contract_ticker = {}

# Global strike cache for validated strikes per expiry
strike_cache = {}
CACHE_TTL = 300  # 5 minutes


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


# Configure logging will be done in main
logger = get_logger()


@dataclass
class ExpiryOption:
    """Data class to hold option contract information for a specific expiry"""

    expiry: str
    call_contract: Contract
    put_contract: Contract
    call_strike: float
    put_strike: float


@dataclass
class OpportunityScore:
    """Data class to hold scoring components for an opportunity"""

    risk_reward_ratio: float
    liquidity_score: float
    time_decay_score: float
    market_quality_score: float
    composite_score: float


@dataclass
class GlobalOpportunity:
    """Data class to hold a complete arbitrage opportunity with scoring"""

    symbol: str
    conversion_contract: Contract
    order: Order
    trade_details: Dict
    score: OpportunityScore
    timestamp: float

    # Additional metadata for decision making
    call_volume: float
    put_volume: float
    call_bid_ask_spread: float
    put_bid_ask_spread: float
    days_to_expiry: int


@dataclass
class ScoringConfig:
    """Configuration for opportunity scoring weights"""

    risk_reward_weight: float = 0.40
    liquidity_weight: float = 0.25
    time_decay_weight: float = 0.20
    market_quality_weight: float = 0.15

    # Thresholds
    min_liquidity_score: float = 0.3
    min_risk_reward_ratio: float = 1.5
    max_bid_ask_spread: float = 20.0
    optimal_days_to_expiry: int = 30

    @classmethod
    def create_conservative(cls) -> "ScoringConfig":
        """Create conservative scoring configuration - prioritizes safety"""
        return cls(
            risk_reward_weight=0.30,
            liquidity_weight=0.35,
            time_decay_weight=0.25,
            market_quality_weight=0.10,
            min_liquidity_score=0.5,
            min_risk_reward_ratio=2.0,
            max_bid_ask_spread=15.0,
            optimal_days_to_expiry=25,
        )

    @classmethod
    def create_aggressive(cls) -> "ScoringConfig":
        """Create aggressive scoring configuration - prioritizes returns"""
        return cls(
            risk_reward_weight=0.50,
            liquidity_weight=0.15,
            time_decay_weight=0.20,
            market_quality_weight=0.15,
            min_liquidity_score=0.2,
            min_risk_reward_ratio=1.2,
            max_bid_ask_spread=25.0,
            optimal_days_to_expiry=35,
        )

    @classmethod
    def create_balanced(cls) -> "ScoringConfig":
        """Create balanced scoring configuration - default settings"""
        return cls()

    @classmethod
    def create_liquidity_focused(cls) -> "ScoringConfig":
        """Create liquidity-focused configuration - prioritizes execution certainty"""
        return cls(
            risk_reward_weight=0.25,
            liquidity_weight=0.40,
            time_decay_weight=0.15,
            market_quality_weight=0.20,
            min_liquidity_score=0.6,
            min_risk_reward_ratio=1.3,
            max_bid_ask_spread=12.0,
            optimal_days_to_expiry=28,
        )

    def validate(self) -> bool:
        """Validate that weights sum to approximately 1.0"""
        total_weight = (
            self.risk_reward_weight
            + self.liquidity_weight
            + self.time_decay_weight
            + self.market_quality_weight
        )
        return abs(total_weight - 1.0) < 0.01

    def normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total = (
            self.risk_reward_weight
            + self.liquidity_weight
            + self.time_decay_weight
            + self.market_quality_weight
        )
        if total > 0:
            self.risk_reward_weight /= total
            self.liquidity_weight /= total
            self.time_decay_weight /= total
            self.market_quality_weight /= total


class GlobalOpportunityManager:
    """
    Manages collection, scoring, and selection of arbitrage opportunities across all symbols.
    Thread-safe implementation to handle concurrent opportunity submissions.
    """

    def __init__(self, scoring_config: ScoringConfig = None):
        self.scoring_config = scoring_config or ScoringConfig()
        self.opportunities: List[GlobalOpportunity] = []
        self.lock = threading.Lock()
        self.logger = get_logger()

        # Validate and normalize configuration
        if not self.scoring_config.validate():
            self.logger.warning(
                "Scoring configuration weights don't sum to 1.0, normalizing..."
            )
            self.scoring_config.normalize_weights()

        # Log the configuration being used
        self.logger.info(f"Initialized GlobalOpportunityManager with scoring config:")
        self.logger.info(f"  Risk-Reward: {self.scoring_config.risk_reward_weight:.2f}")
        self.logger.info(f"  Liquidity: {self.scoring_config.liquidity_weight:.2f}")
        self.logger.info(f"  Time Decay: {self.scoring_config.time_decay_weight:.2f}")
        self.logger.info(
            f"  Market Quality: {self.scoring_config.market_quality_weight:.2f}"
        )
        self.logger.info(
            f"  Min Risk-Reward Ratio: {self.scoring_config.min_risk_reward_ratio:.2f}"
        )
        self.logger.info(
            f"  Min Liquidity Score: {self.scoring_config.min_liquidity_score:.2f}"
        )

    def clear_opportunities(self):
        """Clear all collected opportunities for new cycle"""
        with self.lock:
            self.opportunities.clear()
            self.logger.debug("Cleared all opportunities for new cycle")

    def calculate_liquidity_score(
        self,
        call_volume: float,
        put_volume: float,
        call_spread: float,
        put_spread: float,
    ) -> float:
        """Calculate liquidity score based on volume and bid-ask spreads"""
        # Volume component (normalized to 0-1 scale)
        volume_score = min(1.0, (call_volume + put_volume) / 1000.0)

        # Spread component (inverted - tighter spreads are better)
        avg_spread = (call_spread + put_spread) / 2.0
        spread_score = max(
            0.0, 1.0 - (avg_spread / self.scoring_config.max_bid_ask_spread)
        )

        # Combined liquidity score
        return (volume_score * 0.6) + (spread_score * 0.4)

    def calculate_time_decay_score(self, days_to_expiry: int) -> float:
        """Calculate time decay score - favor optimal time to expiration"""
        optimal_days = self.scoring_config.optimal_days_to_expiry

        if days_to_expiry <= 0:
            return 0.0

        # Score peaks at optimal days, decreases as we move away
        if days_to_expiry <= optimal_days:
            return days_to_expiry / optimal_days
        else:
            # Penalty for being too far out
            excess_days = days_to_expiry - optimal_days
            return max(0.1, 1.0 - (excess_days / (optimal_days * 2)))

    def calculate_market_quality_score(
        self, trade_details: Dict, call_spread: float, put_spread: float
    ) -> float:
        """Calculate market quality score based on spreads and pricing"""
        net_credit = trade_details.get("net_credit", 0)
        stock_price = trade_details.get("stock_price", 1)

        # Spread quality (tighter is better)
        avg_spread = (call_spread + put_spread) / 2.0
        spread_quality = max(
            0.0, 1.0 - (avg_spread / self.scoring_config.max_bid_ask_spread)
        )

        # Credit quality (positive credit is better)
        credit_quality = min(1.0, max(0.0, net_credit / (stock_price * 0.1)))

        return (spread_quality * 0.7) + (credit_quality * 0.3)

    def calculate_opportunity_score(
        self,
        trade_details: Dict,
        call_volume: float,
        put_volume: float,
        call_spread: float,
        put_spread: float,
        days_to_expiry: int,
    ) -> OpportunityScore:
        """Calculate comprehensive opportunity score"""

        # Risk-reward ratio
        max_profit = trade_details.get("max_profit", 0)
        min_profit = trade_details.get("min_profit", -1)
        risk_reward_ratio = max_profit / abs(min_profit) if min_profit != 0 else 0

        # Component scores
        liquidity_score = self.calculate_liquidity_score(
            call_volume, put_volume, call_spread, put_spread
        )
        time_decay_score = self.calculate_time_decay_score(days_to_expiry)
        market_quality_score = self.calculate_market_quality_score(
            trade_details, call_spread, put_spread
        )

        # Weighted composite score
        composite_score = (
            risk_reward_ratio * self.scoring_config.risk_reward_weight
            + liquidity_score * self.scoring_config.liquidity_weight
            + time_decay_score * self.scoring_config.time_decay_weight
            + market_quality_score * self.scoring_config.market_quality_weight
        )

        return OpportunityScore(
            risk_reward_ratio=risk_reward_ratio,
            liquidity_score=liquidity_score,
            time_decay_score=time_decay_score,
            market_quality_score=market_quality_score,
            composite_score=composite_score,
        )

    def add_opportunity(
        self,
        symbol: str,
        conversion_contract: Contract,
        order: Order,
        trade_details: Dict,
        call_ticker: Ticker,
        put_ticker: Ticker,
    ) -> bool:
        """Add an opportunity to the global collection"""

        # Calculate additional metadata
        call_volume = getattr(call_ticker, "volume", 0)
        put_volume = getattr(put_ticker, "volume", 0)
        call_spread = (
            abs(call_ticker.ask - call_ticker.bid)
            if (not np.isnan(call_ticker.ask) and not np.isnan(call_ticker.bid))
            else float("inf")
        )
        put_spread = (
            abs(put_ticker.ask - put_ticker.bid)
            if (not np.isnan(put_ticker.ask) and not np.isnan(put_ticker.bid))
            else float("inf")
        )

        # Calculate days to expiry
        try:
            expiry_str = trade_details.get("expiry", "")
            expiry_date = datetime.strptime(expiry_str, "%Y%m%d")
            days_to_expiry = (expiry_date - datetime.now()).days
        except (ValueError, TypeError):
            days_to_expiry = 0

        # Calculate opportunity score
        score = self.calculate_opportunity_score(
            trade_details,
            call_volume,
            put_volume,
            call_spread,
            put_spread,
            days_to_expiry,
        )

        # Apply minimum thresholds
        if (
            score.liquidity_score < self.scoring_config.min_liquidity_score
            or score.risk_reward_ratio < self.scoring_config.min_risk_reward_ratio
        ):
            self.logger.debug(
                f"[{symbol}] Opportunity rejected due to minimum thresholds: "
                f"liquidity={score.liquidity_score:.3f}, risk_reward={score.risk_reward_ratio:.3f}"
            )
            return False

        # Create global opportunity
        global_opportunity = GlobalOpportunity(
            symbol=symbol,
            conversion_contract=conversion_contract,
            order=order,
            trade_details=trade_details,
            score=score,
            timestamp=time.time(),
            call_volume=call_volume,
            put_volume=put_volume,
            call_bid_ask_spread=call_spread,
            put_bid_ask_spread=put_spread,
            days_to_expiry=days_to_expiry,
        )

        # Thread-safe addition
        with self.lock:
            self.opportunities.append(global_opportunity)
            self.logger.info(
                f"[{symbol}] Added opportunity with composite score: {score.composite_score:.3f} "
                f"(risk_reward: {score.risk_reward_ratio:.3f}, liquidity: {score.liquidity_score:.3f}, "
                f"time: {score.time_decay_score:.3f}, quality: {score.market_quality_score:.3f})"
            )

        return True

    def get_best_opportunity(self) -> Optional[GlobalOpportunity]:
        """Get the best opportunity based on composite score"""
        with self.lock:
            if not self.opportunities:
                return None

            # Sort by composite score (highest first)
            sorted_opportunities = sorted(
                self.opportunities,
                key=lambda opp: opp.score.composite_score,
                reverse=True,
            )

            best = sorted_opportunities[0]

            # Log comparison details
            self.logger.info(
                f"Global opportunity selection from {len(self.opportunities)} opportunities:"
            )
            for i, opp in enumerate(sorted_opportunities[:5]):  # Show top 5
                self.logger.info(
                    f"  #{i+1}: [{opp.symbol}] Score: {opp.score.composite_score:.3f} "
                    f"Expiry: {opp.trade_details.get('expiry', 'N/A')} "
                    f"Profit: ${opp.trade_details.get('max_profit', 0):.2f}"
                )

            return best

    def get_opportunity_count(self) -> int:
        """Get current number of collected opportunities"""
        with self.lock:
            return len(self.opportunities)

    def log_cycle_summary(self):
        """Log detailed summary of all opportunities in current cycle"""
        with self.lock:
            if not self.opportunities:
                self.logger.info("No opportunities collected in this cycle")
                return

            # Group opportunities by symbol
            by_symbol = defaultdict(list)
            for opp in self.opportunities:
                by_symbol[opp.symbol].append(opp)

            self.logger.info(
                f"=== CYCLE SUMMARY: {len(self.opportunities)} opportunities across {len(by_symbol)} symbols ==="
            )

            # Summary statistics
            scores = [opp.score.composite_score for opp in self.opportunities]
            risk_rewards = [opp.score.risk_reward_ratio for opp in self.opportunities]

            self.logger.info(
                f"Score Range: {min(scores):.3f} - {max(scores):.3f} (avg: {sum(scores)/len(scores):.3f})"
            )
            self.logger.info(
                f"Risk-Reward Range: {min(risk_rewards):.3f} - {max(risk_rewards):.3f} (avg: {sum(risk_rewards)/len(risk_rewards):.3f})"
            )

            # Per-symbol breakdown
            for symbol, symbol_opps in by_symbol.items():
                best_symbol_opp = max(
                    symbol_opps, key=lambda x: x.score.composite_score
                )
                self.logger.info(
                    f"  [{symbol}]: {len(symbol_opps)} opportunities, "
                    f"best score: {best_symbol_opp.score.composite_score:.3f} "
                    f"(expiry: {best_symbol_opp.trade_details.get('expiry', 'N/A')})"
                )

    def get_statistics(self) -> Dict:
        """Get statistical summary of current opportunities"""
        with self.lock:
            if not self.opportunities:
                return {}

            scores = [opp.score.composite_score for opp in self.opportunities]
            risk_rewards = [opp.score.risk_reward_ratio for opp in self.opportunities]
            liquidity_scores = [opp.score.liquidity_score for opp in self.opportunities]

            by_symbol = defaultdict(int)
            for opp in self.opportunities:
                by_symbol[opp.symbol] += 1

            return {
                "total_opportunities": len(self.opportunities),
                "unique_symbols": len(by_symbol),
                "score_stats": {
                    "min": min(scores),
                    "max": max(scores),
                    "avg": sum(scores) / len(scores),
                },
                "risk_reward_stats": {
                    "min": min(risk_rewards),
                    "max": max(risk_rewards),
                    "avg": sum(risk_rewards) / len(risk_rewards),
                },
                "liquidity_stats": {
                    "min": min(liquidity_scores),
                    "max": max(liquidity_scores),
                    "avg": sum(liquidity_scores) / len(liquidity_scores),
                },
                "opportunities_per_symbol": dict(by_symbol),
            }


def test_global_opportunity_scoring():
    """Test function to verify global opportunity scoring works correctly"""
    logger = get_logger()

    # Test different scoring configurations
    configs = {
        "conservative": ScoringConfig.create_conservative(),
        "aggressive": ScoringConfig.create_aggressive(),
        "balanced": ScoringConfig.create_balanced(),
        "liquidity_focused": ScoringConfig.create_liquidity_focused(),
    }

    logger.info("=== Testing Global Opportunity Scoring Configurations ===")

    for config_name, config in configs.items():
        logger.info(f"\n{config_name.upper()} Configuration:")
        logger.info(f"  Risk-Reward Weight: {config.risk_reward_weight:.2f}")
        logger.info(f"  Liquidity Weight: {config.liquidity_weight:.2f}")
        logger.info(f"  Time Decay Weight: {config.time_decay_weight:.2f}")
        logger.info(f"  Market Quality Weight: {config.market_quality_weight:.2f}")
        logger.info(f"  Min Risk-Reward Ratio: {config.min_risk_reward_ratio:.2f}")
        logger.info(f"  Weights Valid: {config.validate()}")

        # Test scoring calculation
        manager = GlobalOpportunityManager(config)

        # Create test trade details
        test_trade_details = {
            "max_profit": 100.0,
            "min_profit": -50.0,
            "net_credit": 25.0,
            "stock_price": 150.0,
            "expiry": "20240315",
        }

        # Test scoring with sample data
        score = manager.calculate_opportunity_score(
            test_trade_details,
            call_volume=500,
            put_volume=300,
            call_spread=2.5,
            put_spread=1.8,
            days_to_expiry=25,
        )

        logger.info(f"  Sample Score - Composite: {score.composite_score:.3f}")
        logger.info(f"    Risk-Reward: {score.risk_reward_ratio:.3f}")
        logger.info(f"    Liquidity: {score.liquidity_score:.3f}")
        logger.info(f"    Time Decay: {score.time_decay_score:.3f}")
        logger.info(f"    Market Quality: {score.market_quality_score:.3f}")

    logger.info("\n=== Global Opportunity Scoring Test Complete ===")


def create_syn_with_config(config_type: str = "balanced", **kwargs):
    """Helper function to create Syn instance with specific configuration"""
    config_map = {
        "conservative": ScoringConfig.create_conservative(),
        "aggressive": ScoringConfig.create_aggressive(),
        "balanced": ScoringConfig.create_balanced(),
        "liquidity_focused": ScoringConfig.create_liquidity_focused(),
    }

    scoring_config = config_map.get(config_type, ScoringConfig.create_balanced())
    return Syn(scoring_config=scoring_config, **kwargs)


class SynExecutor(BaseExecutor):
    """
    Synthetic not free risk (Syn) Executor class that handles the execution of Syn Synthetic option strategies.

    This class is responsible for:
    1. Monitoring market data for stock and options across multiple expiries
    2. Calculating potential arbitrage opportunities
    3. Reporting opportunities to the global manager (no longer executes directly)
    4. Logging trade details and results

    Attributes:
        ib (IB): Interactive Brokers connection instance
        order_manager (OrderManagerClass): Manager for handling order execution
        stock_contract (Contract): The underlying stock contract
        expiry_options (List[ExpiryOption]): List of option data for different expiries
        symbol (str): Trading symbol
        cost_limit (float): Maximum price limit for execution
        max_loss_threshold (float): Maximum loss for execution
        max_profit_threshold (float): Maximum profit for execution
        profit_ratio_threshold (float): Maximum profit to loss ratio for execution
        start_time (float): Start time of the execution
        quantity (int): Quantity of contracts to execute
        all_contracts (List[Contract]): All contracts (stock + all options)
        is_active (bool): Whether the executor is currently active
        data_timeout (float): Maximum time to wait for all contract data (seconds)
        global_manager (GlobalOpportunityManager): Manager for global opportunity collection
    """

    def __init__(
        self,
        ib: IB,
        order_manager: OrderManagerClass,
        stock_contract: Contract,
        expiry_options: List[ExpiryOption],
        symbol: str,
        cost_limit: float,
        max_loss_threshold: float,
        max_profit_threshold: float,
        profit_ratio_threshold: float,
        start_time: float,
        global_manager: GlobalOpportunityManager,
        quantity: int = 1,
        data_timeout: float = 30.0,  # 30 seconds timeout for data collection
    ) -> None:
        """
        Initialize the Syn Executor.

        Args:
            ib: Interactive Brokers connection instance
            order_manager: Manager for handling order execution
            stock_contract: The underlying stock contract
            expiry_options: List of option data for different expiries
            symbol: Trading symbol
            cost_limit: Maximum price limit for execution
            max_loss_threshold: Maximum loss for execution
            max_profit_threshold: Maximum profit for execution
            profit_ratio_threshold: Maximum profit to loss ratio for execution
            start_time: Start time of the execution
            global_manager: Manager for global opportunity collection
            quantity: Quantity of contracts to execute
            data_timeout: Maximum time to wait for all contract data (seconds)
        """
        # Create list of all option contracts
        option_contracts = []
        for expiry_option in expiry_options:
            option_contracts.extend(
                [expiry_option.call_contract, expiry_option.put_contract]
            )

        super().__init__(
            ib,
            order_manager,
            stock_contract,
            option_contracts,
            symbol,
            cost_limit,
            expiry_options[0].expiry if expiry_options else "",
            start_time,
        )
        self.max_loss_threshold = max_loss_threshold
        self.max_profit_threshold = max_profit_threshold
        self.profit_ratio_threshold = profit_ratio_threshold
        self.quantity = quantity
        self.expiry_options = expiry_options
        self.all_contracts = [stock_contract] + option_contracts
        self.is_active = True
        self.data_timeout = data_timeout
        self.data_collection_start = time.time()
        self.global_manager = global_manager

    def _get_ticker(self, conId):
        """Get ticker for this symbol's contract using composite key"""
        return contract_ticker.get((self.symbol, conId))

    def _set_ticker(self, conId, ticker):
        """Set ticker for this symbol's contract using composite key"""
        contract_ticker[(self.symbol, conId)] = ticker

    def _clear_symbol_tickers(self):
        """Clear all tickers for this symbol from global dictionary"""
        keys = [k for k in contract_ticker.keys() if k[0] == self.symbol]
        count = len(keys)
        for key in keys:
            del contract_ticker[key]
        logger.debug(
            f"[{self.symbol}] Cleared {count} contract tickers from global dictionary"
        )
        return count

    def quick_viability_check(
        self, expiry_option: ExpiryOption, stock_price: float
    ) -> Tuple[bool, Optional[str]]:
        """Fast pre-filtering to eliminate non-viable opportunities early"""
        # Quick strike spread check
        strike_spread = expiry_option.call_strike - expiry_option.put_strike
        if strike_spread < 1.0 or strike_spread > 50.0:
            return False, "invalid_strike_spread"

        # Quick time to expiry check
        from datetime import datetime

        try:
            expiry_date = datetime.strptime(expiry_option.expiry, "%Y%m%d")
            days_to_expiry = (expiry_date - datetime.now()).days
            if days_to_expiry < 15 or days_to_expiry > 50:
                return False, "expiry_out_of_range"
        except ValueError:
            return False, "invalid_expiry_format"

        # Quick moneyness check for synthetics (more lenient than SFR)
        call_moneyness = expiry_option.call_strike / stock_price
        put_moneyness = expiry_option.put_strike / stock_price
        if (
            call_moneyness < 0.90
            or call_moneyness > 1.2
            or put_moneyness < 0.80
            or put_moneyness > 1.1
        ):
            return False, "poor_moneyness"

        return True, None

    def check_conditions(
        self,
        symbol: str,
        cost_limit: float,
        lmt_price: float,
        net_credit: float,
        min_roi: float,
        min_profit: float,
        max_profit: float,
    ) -> Tuple[bool, Optional[RejectionReason]]:

        profit_ratio = max_profit / abs(min_profit)

        if (
            self.max_loss_threshold is not None
            and self.max_loss_threshold >= min_profit
        ):  # no arbitrage condition
            logger.info(
                f"max_loss limit [{self.max_loss_threshold}] >  calculated max_loss [{min_profit}] - <doesn't meet conditions>"
            )
            return False, RejectionReason.MAX_LOSS_THRESHOLD_EXCEEDED

        elif net_credit < 0:
            logger.info(
                f"[{symbol}] net_credit[{net_credit}] < 0 - doesn't meet conditions"
            )
            return False, RejectionReason.NET_CREDIT_NEGATIVE

        elif (
            self.max_profit_threshold is not None
            and self.max_profit_threshold < max_profit
        ):
            logger.info(
                f"[{symbol}] max_profit threshold [{self.max_profit_threshold }] < max_profit [{max_profit}] - doesn't meet conditions"
            )
            return False, RejectionReason.MAX_PROFIT_THRESHOLD_NOT_MET

        elif (
            self.profit_ratio_threshold is not None
            and self.profit_ratio_threshold > profit_ratio
        ):
            logger.info(
                f"[{symbol}] profit_ratio_threshold  [{self.profit_ratio_threshold }] > profit_ratio [{profit_ratio}] - doesn't meet conditions"
            )
            return False, RejectionReason.PROFIT_RATIO_THRESHOLD_NOT_MET

        elif np.isnan(lmt_price) or lmt_price > cost_limit:
            logger.info(
                f"[{symbol}] np.isnan(lmt_price) or lmt_price > limit - doesn't meet conditions"
            )
            return False, RejectionReason.PRICE_LIMIT_EXCEEDED

        else:
            logger.info(
                f"[{symbol}] meets conditions - initiating order. [profit_ratio: {profit_ratio}]"
            )
            return True, None

    async def executor(self, event: Event) -> None:
        """
        Main executor method that processes market data events for all contracts.
        This method is called once per symbol and handles all expiries for that symbol.
        """
        if not self.is_active:
            return

        try:
            # Update contract_ticker with new data
            for tick in event:
                ticker: Ticker = tick
                contract = ticker.contract

                # Update ticker data - be more lenient with volume requirements
                # Accept any ticker with valid price data, log warning for low volume
                if ticker.volume > 10:
                    self._set_ticker(contract.conId, ticker)
                elif ticker.volume >= 0 and (
                    ticker.bid > 0 or ticker.ask > 0 or ticker.close > 0
                ):
                    # Accept low volume contracts if they have valid price data
                    self._set_ticker(contract.conId, ticker)
                    logger.warning(
                        f"[{self.symbol}] Low volume ({ticker.volume}) for contract {contract.conId}, "
                        f"but accepting due to valid price data"
                    )
                else:
                    logger.debug(f"Skipping contract {contract.conId}: no valid data")

            # Check for timeout with adaptive timeout based on contract count
            elapsed_time = time.time() - self.data_collection_start
            adaptive_timeout = min(
                self.data_timeout + (len(self.all_contracts) * 0.1), 60.0
            )
            if elapsed_time > adaptive_timeout:
                missing_contracts = [
                    c for c in self.all_contracts if self._get_ticker(c.conId) is None
                ]
                logger.warning(
                    f"[{self.symbol}] Data collection timeout after {elapsed_time:.1f}s (adaptive limit: {adaptive_timeout:.1f}s). "
                    f"Missing data for {len(missing_contracts)} contracts out of {len(self.all_contracts)}"
                )
                # Log details of missing contracts
                for c in missing_contracts[:5]:  # Log first 5 missing
                    logger.info(
                        f"  Missing: {c.symbol} {c.right} {c.strike} {c.lastTradeDateOrContractMonth}"
                    )

                # Deactivate after timeout
                self.is_active = False
                # Finish scan with timeout error
                metrics_collector.finish_scan(
                    success=False, error_message="Data collection timeout"
                )
                return

            # Check if we have data for all contracts
            if all(self._get_ticker(c.conId) is not None for c in self.all_contracts):
                # Check if still active before proceeding
                if not self.is_active:
                    return

                logger.info(
                    f"[{self.symbol}] Fetched ticker for {len(self.all_contracts)} contracts"
                )
                execution_time = time.time() - self.start_time
                logger.info(f"time to execution: {execution_time} sec")
                metrics_collector.record_execution_time(execution_time)

                # Process all expiries and report opportunities to global manager
                opportunities_found = 0

                for expiry_option in self.expiry_options:
                    opportunity = self.calc_price_and_build_order_for_expiry(
                        expiry_option
                    )
                    if opportunity:
                        conversion_contract, order, _, trade_details = opportunity

                        # Get ticker data for scoring
                        call_ticker = self._get_ticker(
                            expiry_option.call_contract.conId
                        )
                        put_ticker = self._get_ticker(expiry_option.put_contract.conId)

                        if call_ticker and put_ticker:
                            # Report opportunity to global manager instead of executing
                            success = self.global_manager.add_opportunity(
                                symbol=self.symbol,
                                conversion_contract=conversion_contract,
                                order=order,
                                trade_details=trade_details,
                                call_ticker=call_ticker,
                                put_ticker=put_ticker,
                            )

                            if success:
                                opportunities_found += 1
                                max_profit = trade_details["max_profit"]
                                min_profit = trade_details["min_profit"]
                                risk_reward_ratio = (
                                    max_profit / abs(min_profit)
                                    if min_profit != 0
                                    else 0
                                )

                                logger.info(
                                    f"[{self.symbol}] Reported opportunity for expiry: {trade_details['expiry']} - "
                                    f"Risk-Reward Ratio: {risk_reward_ratio:.3f} "
                                    f"(max_profit: {max_profit:.2f}, min_profit: {min_profit:.2f})"
                                )

                # Log summary for this symbol
                if opportunities_found > 0:
                    logger.info(
                        f"[{self.symbol}] Reported {opportunities_found} opportunities to global manager"
                    )
                    metrics_collector.record_opportunity_found()
                else:
                    logger.info(f"[{self.symbol}] No suitable opportunities found")

                # Always finish scan successfully - global manager will handle execution
                self.is_active = False
                metrics_collector.finish_scan(success=True)

            else:
                # Still waiting for data from some contracts
                missing_contracts = [
                    c for c in self.all_contracts if self._get_ticker(c.conId) is None
                ]
                # Only log debug message every 5 seconds to reduce noise
                if int(elapsed_time) % 5 == 0:
                    logger.debug(
                        f"[{self.symbol}] Still waiting for data from {len(missing_contracts)} contracts "
                        f"(elapsed: {elapsed_time:.1f}s)"
                    )

        except Exception as e:
            logger.error(f"Error in executor: {str(e)}")
            self.is_active = False
            # Finish scan with error
            metrics_collector.finish_scan(success=False, error_message=str(e))

    def calc_price_and_build_order_for_expiry(
        self, expiry_option: ExpiryOption
    ) -> Optional[Tuple[Contract, Order, float, Dict]]:
        """
        Calculate price and build order for a specific expiry option.
        Returns tuple of (contract, order, min_profit, trade_details) or None if no opportunity.
        """
        try:
            # Fast pre-filtering to eliminate non-viable opportunities early
            stock_ticker = self._get_ticker(self.stock_contract.conId)
            if not stock_ticker:
                return None

            stock_price = (
                stock_ticker.ask
                if not np.isnan(stock_ticker.ask)
                else stock_ticker.close
            )

            viable, reason = self.quick_viability_check(expiry_option, stock_price)
            if not viable:
                logger.debug(
                    f"[{self.symbol}] Quick rejection for {expiry_option.expiry}: {reason}"
                )
                return None

            # Get option data
            call_ticker = self._get_ticker(expiry_option.call_contract.conId)
            put_ticker = self._get_ticker(expiry_option.put_contract.conId)

            if not call_ticker or not put_ticker:
                metrics_collector.add_rejection_reason(
                    RejectionReason.MISSING_MARKET_DATA,
                    {
                        "symbol": self.symbol,
                        "contract_type": "options",
                        "expiry": expiry_option.expiry,
                        "call_strike": expiry_option.call_strike,
                        "put_strike": expiry_option.put_strike,
                        "missing_call_data": not call_ticker,
                        "missing_put_data": not put_ticker,
                    },
                )
                return None

            # Get validated prices for individual legs
            call_price = (
                call_ticker.bid if not np.isnan(call_ticker.bid) else call_ticker.close
            )
            put_price = (
                put_ticker.ask if not np.isnan(put_ticker.ask) else put_ticker.close
            )

            if np.isnan(call_price) or np.isnan(put_price):
                metrics_collector.add_rejection_reason(
                    RejectionReason.INVALID_CONTRACT_DATA,
                    {
                        "symbol": self.symbol,
                        "contract_type": "options",
                        "expiry": expiry_option.expiry,
                        "call_strike": expiry_option.call_strike,
                        "put_strike": expiry_option.put_strike,
                        "call_price_invalid": np.isnan(call_price),
                        "put_price_invalid": np.isnan(put_price),
                    },
                )
                return None

            # Check bid-ask spread for both call and put contracts to prevent crazy price ranges
            call_bid_ask_spread = (
                abs(call_ticker.ask - call_ticker.bid)
                if (not np.isnan(call_ticker.ask) and not np.isnan(call_ticker.bid))
                else float("inf")
            )
            put_bid_ask_spread = (
                abs(put_ticker.ask - put_ticker.bid)
                if (not np.isnan(put_ticker.ask) and not np.isnan(put_ticker.bid))
                else float("inf")
            )

            if call_bid_ask_spread > 20:
                logger.info(
                    f"[{self.symbol}] Call contract bid-ask spread too wide: {call_bid_ask_spread:.2f} > 15.00, "
                    f"expiry: {expiry_option.expiry}, strike: {expiry_option.call_strike}"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.BID_ASK_SPREAD_TOO_WIDE,
                    {
                        "symbol": self.symbol,
                        "contract_type": "call",
                        "expiry": expiry_option.expiry,
                        "strike": expiry_option.call_strike,
                        "bid_ask_spread": call_bid_ask_spread,
                        "threshold": 20.0,
                    },
                )
                return None

            if put_bid_ask_spread > 20:
                logger.info(
                    f"[{self.symbol}] Put contract bid-ask spread too wide: {put_bid_ask_spread:.2f} > 15.00, "
                    f"expiry: {expiry_option.expiry}, strike: {expiry_option.put_strike}"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.BID_ASK_SPREAD_TOO_WIDE,
                    {
                        "symbol": self.symbol,
                        "contract_type": "put",
                        "expiry": expiry_option.expiry,
                        "strike": expiry_option.put_strike,
                        "bid_ask_spread": put_bid_ask_spread,
                        "threshold": 20.0,
                    },
                )
                return None

            # Calculate net credit
            net_credit = call_price - put_price
            stock_price = round(stock_price, 2)
            net_credit = round(net_credit, 2)
            call_price = round(call_price, 2)
            put_price = round(put_price, 2)

            # temp condition
            if expiry_option.call_strike < expiry_option.put_strike:
                logger.info(
                    f"call_strike:{expiry_option.call_strike} < put_strike:{expiry_option.put_strike}"
                )
                metrics_collector.add_rejection_reason(
                    RejectionReason.INVALID_STRIKE_COMBINATION,
                    {
                        "symbol": self.symbol,
                        "expiry": expiry_option.expiry,
                        "call_strike": expiry_option.call_strike,
                        "put_strike": expiry_option.put_strike,
                        "reason": "call_strike < put_strike",
                    },
                )
                return None

            spread = stock_price - expiry_option.put_strike
            min_profit = net_credit - spread  # max loss
            max_profit = (
                expiry_option.call_strike - expiry_option.put_strike
            ) + min_profit
            min_roi = (min_profit / (stock_price + net_credit)) * 100

            # Calculate precise combo limit price based on target leg prices
            combo_limit_price = self.calculate_combo_limit_price(
                stock_price=stock_price,
                call_price=call_price,
                put_price=put_price,
                buffer_percent=0.00,  # 1.5% buffer for better execution
            )

            logger.info(
                f"[{self.symbol}] Expiry: {expiry_option.expiry} min_profit:{min_profit:.2f}, max_profit:{max_profit:.2f}, min_roi:{min_roi:.2f}%"
            )

            conditions_met, rejection_reason = self.check_conditions(
                self.symbol,
                self.cost_limit,
                combo_limit_price,  # Use calculated precise limit price
                net_credit,
                min_roi,
                min_profit,
                max_profit,
            )

            if conditions_met:
                # Build order with precise limit price and target leg prices
                conversion_contract, order = self.build_order(
                    self.symbol,
                    self.stock_contract,
                    expiry_option.call_contract,
                    expiry_option.put_contract,
                    combo_limit_price,  # Use calculated precise limit price
                    self.quantity,
                    call_price=call_price,  # Target call leg price
                    put_price=put_price,  # Target put leg price
                )

                # Prepare trade details for logging (don't log yet)
                trade_details = {
                    "call_strike": expiry_option.call_strike,
                    "call_price": call_price,
                    "put_strike": expiry_option.put_strike,
                    "put_price": put_price,
                    "stock_price": stock_price,
                    "net_credit": net_credit,
                    "min_profit": min_profit,
                    "max_profit": max_profit,
                    "min_roi": min_roi,
                    "expiry": expiry_option.expiry,
                }

                return conversion_contract, order, min_profit, trade_details
            else:
                # Record rejection reason
                if rejection_reason:
                    # Calculate profit_ratio for context
                    profit_ratio = (
                        max_profit / abs(min_profit) if min_profit != 0 else 0
                    )

                    metrics_collector.add_rejection_reason(
                        rejection_reason,
                        {
                            "symbol": self.symbol,
                            "expiry": expiry_option.expiry,
                            "call_strike": expiry_option.call_strike,
                            "put_strike": expiry_option.put_strike,
                            "stock_price": stock_price,
                            "net_credit": net_credit,
                            "min_profit": min_profit,
                            "max_profit": max_profit,
                            "min_roi": min_roi,
                            "combo_limit_price": combo_limit_price,
                            "cost_limit": self.cost_limit,
                            "max_loss_threshold": self.max_loss_threshold,
                            "max_profit_threshold": self.max_profit_threshold,
                            "profit_ratio_threshold": self.profit_ratio_threshold,
                            "spread": spread,
                            "profit_ratio": profit_ratio,
                        },
                    )

            return None
        except Exception as e:
            logger.error(f"Error in calc_price_and_build_order_for_expiry: {str(e)}")
            return None

    def calculate_all_opportunities_vectorized(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Calculate all synthetic arbitrage opportunities in parallel using NumPy.
        Returns arrays of profits and metadata for all expiry options.
        """
        # Gather all data into NumPy arrays
        num_options = len(self.expiry_options)

        # Pre-allocate arrays for all data
        call_bids = np.zeros(num_options)
        call_asks = np.zeros(num_options)
        put_bids = np.zeros(num_options)
        put_asks = np.zeros(num_options)
        call_strikes = np.zeros(num_options)
        put_strikes = np.zeros(num_options)
        stock_bids = np.zeros(num_options)
        stock_asks = np.zeros(num_options)

        # Populate arrays (this is the only loop needed)
        valid_mask = np.zeros(num_options, dtype=bool)

        for i, expiry_option in enumerate(self.expiry_options):
            call_ticker = self._get_ticker(expiry_option.call_contract.conId)
            put_ticker = self._get_ticker(expiry_option.put_contract.conId)
            stock_ticker = self._get_ticker(self.stock_contract.conId)

            if call_ticker and put_ticker and stock_ticker:
                # Check data validity
                if (
                    hasattr(call_ticker, "bid")
                    and call_ticker.bid > 0
                    and hasattr(call_ticker, "ask")
                    and call_ticker.ask > 0
                    and hasattr(put_ticker, "bid")
                    and put_ticker.bid > 0
                    and hasattr(put_ticker, "ask")
                    and put_ticker.ask > 0
                ):

                    call_bids[i] = call_ticker.bid
                    call_asks[i] = call_ticker.ask
                    put_bids[i] = put_ticker.bid
                    put_asks[i] = put_ticker.ask
                    call_strikes[i] = expiry_option.call_strike
                    put_strikes[i] = expiry_option.put_strike
                    stock_bids[i] = (
                        stock_ticker.bid if stock_ticker.bid > 0 else stock_ticker.last
                    )
                    stock_asks[i] = (
                        stock_ticker.ask if stock_ticker.ask > 0 else stock_ticker.last
                    )
                    valid_mask[i] = True

        # VECTORIZED CALCULATIONS - All opportunities calculated at once!
        # For synthetic: net_credit = call_price - put_price
        # min_profit = net_credit - (stock_price - put_strike) [this is max loss]
        # max_profit = (call_strike - put_strike) + min_profit

        # Execution prices (what we actually get/pay)
        exec_net_credits = (
            call_bids - put_asks
        )  # We sell call (get bid), buy put (pay ask)
        exec_stock_prices = stock_asks  # We buy stock (pay ask)
        exec_spreads = exec_stock_prices - put_strikes  # Stock price - put strike

        # Calculate min profits (actually max losses, but kept as min_profit for consistency)
        min_profits = exec_net_credits - exec_spreads

        # Calculate max profits
        strike_spreads = call_strikes - put_strikes
        max_profits = strike_spreads + min_profits

        # Apply validity mask
        min_profits[~valid_mask] = -np.inf
        max_profits[~valid_mask] = -np.inf

        return (
            min_profits,  # Actually max losses
            max_profits,
            {
                "call_bids": call_bids,
                "call_asks": call_asks,
                "put_bids": put_bids,
                "put_asks": put_asks,
                "call_strikes": call_strikes,
                "put_strikes": put_strikes,
                "stock_bids": stock_bids,
                "stock_asks": stock_asks,
                "valid_mask": valid_mask,
                "net_credits": exec_net_credits,
                "spreads": exec_spreads,
            },
        )

    async def evaluate_with_available_data_vectorized(self) -> Optional[Dict]:
        """
        Vectorized evaluation of all synthetic opportunities at once.
        Similar to SFR but adapted for synthetic strategy parameters.
        """
        logger.info(
            f"[{self.symbol}] Starting vectorized synthetic evaluation with {len(self.expiry_options)} options"
        )

        # Step 1: Calculate all opportunities in parallel
        min_profits, max_profits, market_data = (
            self.calculate_all_opportunities_vectorized()
        )

        # Step 2: Apply filters based on synthetic strategy thresholds
        profit_ratios = np.where(min_profits != 0, max_profits / np.abs(min_profits), 0)

        # Apply synthetic-specific filters
        viable_mask = (
            market_data["valid_mask"]
            & (
                min_profits >= -abs(self.max_loss_threshold)
                if self.max_loss_threshold is not None
                else True
            )
            & (
                max_profits <= self.max_profit_threshold
                if self.max_profit_threshold is not None
                else True
            )
            & (
                profit_ratios >= self.profit_ratio_threshold
                if self.profit_ratio_threshold is not None
                else True
            )
            & (market_data["net_credits"] >= 0)  # Positive net credit
        )

        # Step 3: Find best opportunity
        if not np.any(viable_mask):
            logger.info(
                f"[{self.symbol}] No viable synthetic opportunities found after vectorized evaluation"
            )
            return None

        # Create composite score (higher is better for synthetic)
        # Focus on profit ratio and credit quality
        composite_scores = np.zeros(len(self.expiry_options))
        composite_scores[viable_mask] = (
            profit_ratios[viable_mask] * 0.6  # Profit ratio weight
            + (
                market_data["net_credits"][viable_mask]
                / np.mean(market_data["net_credits"][viable_mask])
            )
            * 0.4  # Credit quality
        )

        best_idx = np.argmax(composite_scores)
        best_opportunity = self.expiry_options[best_idx]

        logger.info(
            f"[{self.symbol}] Best synthetic opportunity found: "
            f"Expiry {best_opportunity.expiry}, "
            f"Min profit: ${min_profits[best_idx]:.2f}, "
            f"Max profit: ${max_profits[best_idx]:.2f}, "
            f"Profit ratio: {profit_ratios[best_idx]:.3f}"
        )

        # Build the order for the best opportunity
        combo_limit_price = self.calculate_combo_limit_price(
            stock_price=market_data["stock_asks"][best_idx],
            call_price=market_data["call_bids"][best_idx],
            put_price=market_data["put_asks"][best_idx],
            buffer_percent=0.01,
        )

        conversion_contract, order = self.build_order(
            self.symbol,
            self.stock_contract,
            best_opportunity.call_contract,
            best_opportunity.put_contract,
            combo_limit_price,
            self.quantity,
            call_price=market_data["call_bids"][best_idx],
            put_price=market_data["put_asks"][best_idx],
        )

        return {
            "contract": conversion_contract,
            "order": order,
            "min_profit": min_profits[best_idx],
            "trade_details": {
                "expiry": best_opportunity.expiry,
                "call_strike": best_opportunity.call_strike,
                "put_strike": best_opportunity.put_strike,
                "call_price": market_data["call_bids"][best_idx],
                "put_price": market_data["put_asks"][best_idx],
                "stock_price": market_data["stock_asks"][best_idx],
                "net_credit": market_data["net_credits"][best_idx],
                "min_profit": min_profits[best_idx],
                "max_profit": max_profits[best_idx],
                "profit_ratio": profit_ratios[best_idx],
            },
            "expiry_option": best_opportunity,
        }

    def deactivate(self):
        """Deactivate the executor"""
        self.is_active = False


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

            # Request options chain
            chain = await self._get_chain(stock)

            # Define parameters for the options (expiry and strike price)
            valid_strikes = [
                s for s in chain.strikes if s <= stock_price and s > stock_price - 10
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
            potential_strikes = [s for s in chain.strikes if abs(s - stock_price) <= 25]

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
        global strike_cache

        # Check cache first (5 minute TTL for strike validation)
        cache_key = f"{symbol}_{expiry}"
        current_time = time.time()

        if cache_key in strike_cache:
            cached_data = strike_cache[cache_key]
            if current_time - cached_data["timestamp"] < CACHE_TTL:
                logger.debug(
                    f"[{symbol}] Using cached strikes for {expiry}: {len(cached_data['strikes'])} strikes"
                )
                return list(cached_data["strikes"])

        # Limit strikes to reasonable range to avoid overwhelming API
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
        strike_cache[cache_key] = {
            "strikes": set(valid_strikes),
            "timestamp": current_time,
        }

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
