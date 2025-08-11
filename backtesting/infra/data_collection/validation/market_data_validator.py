"""
General market data validator for consistency and quality checks.

Provides validation for market data beyond SFR-specific requirements,
including general data integrity, price consistency, and technical validation.
"""

import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MarketDataValidationResult:
    """Result of general market data validation."""

    symbol: str
    validation_timestamp: datetime
    data_type: str  # 'stock' or 'option'
    period_start: date
    period_end: date

    # Overall validation
    is_valid: bool
    quality_score: float  # 0-1

    # Specific validation checks
    price_consistency_passed: bool
    volume_consistency_passed: bool
    temporal_consistency_passed: bool
    outlier_detection_passed: bool
    coverage_validation_passed: bool

    # Detailed metrics
    total_records: int
    suspicious_records: int
    price_outliers: int
    volume_outliers: int
    temporal_gaps: int

    # Issues found
    errors: List[str]
    warnings: List[str]
    info_messages: List[str]

    # Statistical summaries
    price_stats: Dict[str, float]
    volume_stats: Dict[str, float]
    temporal_stats: Dict[str, float]


class MarketDataValidator:
    """
    General market data validator.

    Performs comprehensive validation of market data including:
    - Price consistency and outlier detection
    - Volume validation and anomaly detection
    - Temporal consistency and gap analysis
    - Data coverage and completeness
    - Technical data integrity checks
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

        # Validation thresholds
        self.thresholds = {
            # Price validation
            "max_daily_price_change_percent": 20.0,  # Max 20% daily change
            "price_outlier_std_multiplier": 4.0,  # 4 standard deviations
            "min_price_value": 0.01,  # Minimum valid price
            "max_price_jump_percent": 50.0,  # Max single jump
            # Volume validation
            "volume_outlier_multiplier": 10.0,  # 10x median volume
            "min_volume_value": 0,  # Minimum volume
            "volume_consistency_threshold": 0.95,  # 95% of records should have volume
            # Temporal validation
            "max_gap_minutes_intraday": 60,  # Max 60 minute gap during market hours
            "max_gap_hours_daily": 48,  # Max 48 hour gap for daily data
            "market_hours_start": 9.5,  # 9:30 AM
            "market_hours_end": 16.0,  # 4:00 PM
            # Coverage validation
            "min_coverage_percent": 90.0,  # Min 90% time coverage
            "min_records_per_day": 100,  # Min records per trading day
            # General thresholds
            "max_suspicious_record_percent": 5.0,  # Max 5% suspicious records
            "quality_score_pass_threshold": 0.7,  # Min quality score to pass
        }

    async def validate_stock_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        detailed_validation: bool = True,
    ) -> MarketDataValidationResult:
        """
        Validate stock market data for a symbol and date range.

        Args:
            symbol: Stock symbol to validate
            start_date: Start date for validation period
            end_date: End date for validation period
            detailed_validation: Whether to perform detailed analysis

        Returns:
            MarketDataValidationResult with validation details
        """
        logger.info(
            f"Validating stock data for {symbol} from {start_date} to {end_date}"
        )

        # Initialize result
        result = MarketDataValidationResult(
            symbol=symbol,
            validation_timestamp=datetime.now(),
            data_type="stock",
            period_start=start_date,
            period_end=end_date,
            is_valid=False,
            quality_score=0.0,
            price_consistency_passed=False,
            volume_consistency_passed=False,
            temporal_consistency_passed=False,
            outlier_detection_passed=False,
            coverage_validation_passed=False,
            total_records=0,
            suspicious_records=0,
            price_outliers=0,
            volume_outliers=0,
            temporal_gaps=0,
            errors=[],
            warnings=[],
            info_messages=[],
            price_stats={},
            volume_stats={},
            temporal_stats={},
        )

        try:
            # Get underlying ID
            underlying_id = await self._get_underlying_id(symbol)
            if not underlying_id:
                result.errors.append(f"Symbol {symbol} not found in database")
                return result

            # Load stock data for validation
            stock_data = await self._load_stock_data_for_validation(
                underlying_id, start_date, end_date
            )

            if not stock_data:
                result.errors.append("No stock data found for validation period")
                return result

            result.total_records = len(stock_data)
            result.info_messages.append(f"Loaded {len(stock_data)} stock data records")

            # Run validation checks
            await self._validate_price_consistency(stock_data, result)
            await self._validate_volume_data(stock_data, result)
            await self._validate_temporal_consistency(stock_data, result)

            if detailed_validation:
                await self._detect_price_outliers(stock_data, result)
                await self._validate_data_coverage(
                    stock_data, start_date, end_date, result
                )

            # Calculate overall quality score
            result.quality_score = self._calculate_quality_score(result)
            result.is_valid = (
                result.quality_score >= self.thresholds["quality_score_pass_threshold"]
            )

            # Generate summary statistics
            self._generate_summary_statistics(stock_data, result)

            logger.info(
                f"Stock validation completed for {symbol}: "
                f"Valid={result.is_valid}, Quality={result.quality_score:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Stock validation failed for {symbol}: {e}")
            result.errors.append(f"Validation error: {str(e)}")
            return result

    async def validate_option_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        strike: Optional[float] = None,
        expiry: Optional[date] = None,
        option_type: Optional[str] = None,
    ) -> MarketDataValidationResult:
        """
        Validate option market data.

        Args:
            symbol: Underlying symbol
            start_date: Start date for validation
            end_date: End date for validation
            strike: Specific strike to validate (optional)
            expiry: Specific expiry to validate (optional)
            option_type: Option type 'C' or 'P' (optional)

        Returns:
            MarketDataValidationResult for option data
        """
        logger.info(f"Validating option data for {symbol}")

        # Initialize result
        result = MarketDataValidationResult(
            symbol=symbol,
            validation_timestamp=datetime.now(),
            data_type="option",
            period_start=start_date,
            period_end=end_date,
            is_valid=False,
            quality_score=0.0,
            price_consistency_passed=False,
            volume_consistency_passed=False,
            temporal_consistency_passed=False,
            outlier_detection_passed=False,
            coverage_validation_passed=False,
            total_records=0,
            suspicious_records=0,
            price_outliers=0,
            volume_outliers=0,
            temporal_gaps=0,
            errors=[],
            warnings=[],
            info_messages=[],
            price_stats={},
            volume_stats={},
            temporal_stats={},
        )

        try:
            # Get underlying ID
            underlying_id = await self._get_underlying_id(symbol)
            if not underlying_id:
                result.errors.append(f"Symbol {symbol} not found in database")
                return result

            # Load option data
            option_data = await self._load_option_data_for_validation(
                underlying_id, start_date, end_date, strike, expiry, option_type
            )

            if not option_data:
                result.errors.append("No option data found for validation period")
                return result

            result.total_records = len(option_data)

            # Run option-specific validations
            await self._validate_option_price_consistency(option_data, result)
            await self._validate_option_spreads(option_data, result)
            await self._validate_option_volume_data(option_data, result)
            await self._validate_option_greeks(option_data, result)

            # General validations
            await self._validate_temporal_consistency(option_data, result)

            # Calculate quality score
            result.quality_score = self._calculate_option_quality_score(result)
            result.is_valid = (
                result.quality_score >= self.thresholds["quality_score_pass_threshold"]
            )

            self._generate_option_statistics(option_data, result)

            return result

        except Exception as e:
            logger.error(f"Option validation failed for {symbol}: {e}")
            result.errors.append(f"Validation error: {str(e)}")
            return result

    async def _get_underlying_id(self, symbol: str) -> Optional[int]:
        """Get underlying security ID."""
        async with self.db_pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT id FROM underlying_securities WHERE symbol = $1", symbol
            )

    async def _load_stock_data_for_validation(
        self, underlying_id: int, start_date: date, end_date: date
    ) -> List[Dict[str, Any]]:
        """Load stock data for validation."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    time,
                    price,
                    volume,
                    bid_price,
                    ask_price,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    vwap
                FROM stock_data_ticks
                WHERE underlying_id = $1
                  AND time >= $2
                  AND time <= $3
                ORDER BY time
                """,
                underlying_id,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )

            return [dict(row) for row in rows]

    async def _load_option_data_for_validation(
        self,
        underlying_id: int,
        start_date: date,
        end_date: date,
        strike: Optional[float] = None,
        expiry: Optional[date] = None,
        option_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load option data for validation."""
        async with self.db_pool.acquire() as conn:
            where_conditions = [
                "oc.underlying_id = $1",
                "mdt.time >= $2",
                "mdt.time <= $3",
            ]
            params = [
                underlying_id,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            ]

            if strike is not None:
                where_conditions.append(f"oc.strike_price = ${len(params) + 1}")
                params.append(strike)

            if expiry is not None:
                where_conditions.append(f"oc.expiration_date = ${len(params) + 1}")
                params.append(expiry)

            if option_type is not None:
                where_conditions.append(f"oc.option_type = ${len(params) + 1}")
                params.append(option_type)

            query = f"""
                SELECT
                    mdt.time,
                    mdt.bid_price,
                    mdt.ask_price,
                    mdt.last_price,
                    mdt.volume,
                    mdt.bid_ask_spread,
                    mdt.delta,
                    mdt.gamma,
                    mdt.theta,
                    mdt.vega,
                    mdt.implied_volatility,
                    oc.strike_price,
                    oc.expiration_date,
                    oc.option_type
                FROM market_data_ticks mdt
                JOIN option_chains oc ON mdt.contract_id = oc.id
                WHERE {' AND '.join(where_conditions)}
                ORDER BY mdt.time
            """

            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]

    async def _validate_price_consistency(
        self, data: List[Dict[str, Any]], result: MarketDataValidationResult
    ):
        """Validate price consistency in stock data."""
        if not data:
            return

        price_issues = 0
        previous_price = None

        for record in data:
            price = record.get("price") or record.get("close_price")

            if price is None:
                price_issues += 1
                continue

            # Check minimum price
            if price < self.thresholds["min_price_value"]:
                price_issues += 1
                result.warnings.append(f"Price below minimum: {price}")

            # Check for unrealistic price jumps
            if previous_price is not None:
                price_change_percent = (
                    abs(price - previous_price) / previous_price * 100
                )
                if price_change_percent > self.thresholds["max_price_jump_percent"]:
                    price_issues += 1
                    result.warnings.append(
                        f"Large price jump: {price_change_percent:.1f}% "
                        f"from {previous_price} to {price}"
                    )

            # Validate OHLC consistency
            open_price = record.get("open_price")
            high_price = record.get("high_price")
            low_price = record.get("low_price")
            close_price = record.get("close_price")

            if all([open_price, high_price, low_price, close_price]):
                if not (
                    low_price <= open_price <= high_price
                    and low_price <= close_price <= high_price
                ):
                    price_issues += 1
                    result.warnings.append("OHLC consistency violation")

            previous_price = price

        # Determine if price consistency passed
        suspicious_percent = (price_issues / len(data)) * 100
        result.price_consistency_passed = (
            suspicious_percent <= self.thresholds["max_suspicious_record_percent"]
        )
        result.suspicious_records += price_issues

        if not result.price_consistency_passed:
            result.errors.append(
                f"Price consistency failed: {suspicious_percent:.1f}% suspicious records"
            )

    async def _validate_volume_data(
        self, data: List[Dict[str, Any]], result: MarketDataValidationResult
    ):
        """Validate volume data consistency."""
        if not data:
            return

        volume_issues = 0
        volumes = []

        for record in data:
            volume = record.get("volume")

            if volume is None:
                volume_issues += 1
                continue

            if volume < self.thresholds["min_volume_value"]:
                volume_issues += 1

            volumes.append(volume)

        if volumes:
            # Statistical outlier detection for volume
            median_volume = statistics.median(volumes)
            volume_outliers = [
                v
                for v in volumes
                if v > median_volume * self.thresholds["volume_outlier_multiplier"]
            ]

            result.volume_outliers = len(volume_outliers)

            # Volume consistency check
            records_with_volume = len(volumes)
            volume_coverage = records_with_volume / len(data)

            result.volume_consistency_passed = (
                volume_coverage >= self.thresholds["volume_consistency_threshold"]
            )

            if not result.volume_consistency_passed:
                result.warnings.append(f"Volume coverage only {volume_coverage:.1%}")
        else:
            result.volume_consistency_passed = False
            result.errors.append("No valid volume data found")

    async def _validate_temporal_consistency(
        self, data: List[Dict[str, Any]], result: MarketDataValidationResult
    ):
        """Validate temporal consistency and gaps."""
        if len(data) < 2:
            result.temporal_consistency_passed = True
            return

        gaps = 0
        large_gaps = []
        previous_time = None

        for record in data:
            current_time = record["time"]

            if previous_time is not None:
                gap_minutes = (current_time - previous_time).total_seconds() / 60

                # Check for intraday gaps during market hours
                if self._is_market_hours(current_time) and self._is_market_hours(
                    previous_time
                ):
                    if gap_minutes > self.thresholds["max_gap_minutes_intraday"]:
                        gaps += 1
                        large_gaps.append(
                            {
                                "start": previous_time,
                                "end": current_time,
                                "gap_minutes": gap_minutes,
                            }
                        )

                # Check for very large gaps
                if gap_minutes > self.thresholds["max_gap_hours_daily"] * 60:
                    gaps += 1
                    result.warnings.append(
                        f"Large temporal gap: {gap_minutes/60:.1f} hours"
                    )

            previous_time = current_time

        result.temporal_gaps = gaps
        result.temporal_consistency_passed = gaps <= len(data) * 0.01  # Allow 1% gaps

        if not result.temporal_consistency_passed:
            result.warnings.append(f"Found {gaps} temporal gaps in data")

    async def _detect_price_outliers(
        self, data: List[Dict[str, Any]], result: MarketDataValidationResult
    ):
        """Detect price outliers using statistical methods."""
        prices = []
        for record in data:
            price = record.get("price") or record.get("close_price")
            if price is not None:
                prices.append(price)

        if len(prices) < 10:  # Need minimum data for statistical analysis
            result.outlier_detection_passed = True
            return

        # Calculate daily returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                returns.append((prices[i] - prices[i - 1]) / prices[i - 1])

        if returns:
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 0

            outlier_threshold = (
                self.thresholds["price_outlier_std_multiplier"] * std_return
            )

            outliers = [r for r in returns if abs(r - mean_return) > outlier_threshold]

            result.price_outliers = len(outliers)
            result.outlier_detection_passed = (
                len(outliers) <= len(returns) * 0.02
            )  # Allow 2% outliers

            if not result.outlier_detection_passed:
                result.warnings.append(
                    f"Found {len(outliers)} price outliers "
                    f"({len(outliers)/len(returns):.1%} of returns)"
                )

    async def _validate_data_coverage(
        self,
        data: List[Dict[str, Any]],
        start_date: date,
        end_date: date,
        result: MarketDataValidationResult,
    ):
        """Validate data coverage and completeness."""
        if not data:
            result.coverage_validation_passed = False
            result.errors.append("No data for coverage validation")
            return

        # Calculate expected vs actual coverage
        expected_days = self._count_trading_days(start_date, end_date)

        # Count actual days with data
        actual_days = len(set(record["time"].date() for record in data))

        coverage_percent = (actual_days / expected_days) * 100

        # Check records per day
        avg_records_per_day = len(data) / max(actual_days, 1)

        result.coverage_validation_passed = (
            coverage_percent >= self.thresholds["min_coverage_percent"]
            and avg_records_per_day >= self.thresholds["min_records_per_day"]
        )

        if not result.coverage_validation_passed:
            if coverage_percent < self.thresholds["min_coverage_percent"]:
                result.warnings.append(
                    f"Coverage {coverage_percent:.1f}% below threshold "
                    f"{self.thresholds['min_coverage_percent']:.1f}%"
                )

            if avg_records_per_day < self.thresholds["min_records_per_day"]:
                result.warnings.append(
                    f"Average {avg_records_per_day:.0f} records/day below threshold "
                    f"{self.thresholds['min_records_per_day']}"
                )

    async def _validate_option_price_consistency(
        self, data: List[Dict[str, Any]], result: MarketDataValidationResult
    ):
        """Validate option price consistency."""
        price_issues = 0

        for record in data:
            bid = record.get("bid_price")
            ask = record.get("ask_price")
            last = record.get("last_price")

            # Check bid <= ask
            if bid is not None and ask is not None:
                if bid > ask:
                    price_issues += 1
                    result.warnings.append(f"Bid {bid} > Ask {ask}")
                elif bid < 0 or ask < 0:
                    price_issues += 1
                    result.warnings.append("Negative bid or ask price")

            # Check last price reasonableness
            if last is not None and bid is not None and ask is not None:
                if not (bid <= last <= ask):
                    # Allow some tolerance for stale last prices
                    mid_price = (bid + ask) / 2
                    if abs(last - mid_price) > mid_price * 0.1:  # 10% tolerance
                        price_issues += 1

        suspicious_percent = (price_issues / len(data)) * 100
        result.price_consistency_passed = (
            suspicious_percent <= self.thresholds["max_suspicious_record_percent"]
        )
        result.suspicious_records += price_issues

    async def _validate_option_spreads(
        self, data: List[Dict[str, Any]], result: MarketDataValidationResult
    ):
        """Validate option bid-ask spreads."""
        spread_issues = 0
        spreads = []

        for record in data:
            spread = record.get("bid_ask_spread")
            bid = record.get("bid_price")
            ask = record.get("ask_price")

            if spread is not None:
                spreads.append(spread)

                # Check for negative spreads
                if spread < 0:
                    spread_issues += 1
                    result.warnings.append("Negative bid-ask spread")

            elif bid is not None and ask is not None:
                calculated_spread = ask - bid
                spreads.append(calculated_spread)

                if calculated_spread < 0:
                    spread_issues += 1

        if spreads:
            # Check for unrealistic spreads
            median_spread = statistics.median(spreads)
            for spread in spreads:
                if spread > median_spread * 10:  # 10x median spread
                    spread_issues += 1

        suspicious_percent = (spread_issues / len(data)) * 100 if data else 0
        spread_ok = (
            suspicious_percent <= self.thresholds["max_suspicious_record_percent"]
        )

        if not spread_ok:
            result.warnings.append(
                f"Spread issues in {suspicious_percent:.1f}% of records"
            )

    async def _validate_option_volume_data(
        self, data: List[Dict[str, Any]], result: MarketDataValidationResult
    ):
        """Validate option volume data."""
        # Similar to stock volume validation but with option-specific checks
        await self._validate_volume_data(data, result)

    async def _validate_option_greeks(
        self, data: List[Dict[str, Any]], result: MarketDataValidationResult
    ):
        """Validate option Greeks for reasonableness."""
        greek_issues = 0

        for record in data:
            delta = record.get("delta")
            gamma = record.get("gamma")
            theta = record.get("theta")
            vega = record.get("vega")
            iv = record.get("implied_volatility")

            option_type = record.get("option_type")

            # Delta bounds checking
            if delta is not None:
                if option_type == "C":  # Call
                    if not (0 <= delta <= 1):
                        greek_issues += 1
                elif option_type == "P":  # Put
                    if not (-1 <= delta <= 0):
                        greek_issues += 1

            # Gamma should be positive
            if gamma is not None and gamma < 0:
                greek_issues += 1

            # Theta should be negative for long options
            if theta is not None and theta > 0:
                greek_issues += 1

            # Vega should be positive
            if vega is not None and vega < 0:
                greek_issues += 1

            # IV bounds checking
            if iv is not None:
                if not (0 < iv < 5):  # 0% to 500% IV range
                    greek_issues += 1

        greek_quality_ok = (greek_issues / len(data)) <= 0.05  # Allow 5% issues

        if not greek_quality_ok:
            result.warnings.append(
                f"Greeks validation issues in {greek_issues} records"
            )

    def _calculate_quality_score(self, result: MarketDataValidationResult) -> float:
        """Calculate overall quality score for stock data."""
        # Component scores
        price_score = 1.0 if result.price_consistency_passed else 0.5
        volume_score = 1.0 if result.volume_consistency_passed else 0.7
        temporal_score = 1.0 if result.temporal_consistency_passed else 0.6
        outlier_score = 1.0 if result.outlier_detection_passed else 0.8
        coverage_score = 1.0 if result.coverage_validation_passed else 0.4

        # Weighted average
        weights = {
            "price": 0.3,
            "volume": 0.2,
            "temporal": 0.2,
            "outlier": 0.15,
            "coverage": 0.15,
        }

        quality_score = (
            price_score * weights["price"]
            + volume_score * weights["volume"]
            + temporal_score * weights["temporal"]
            + outlier_score * weights["outlier"]
            + coverage_score * weights["coverage"]
        )

        # Penalty for errors
        error_penalty = len(result.errors) * 0.1
        quality_score = max(0.0, quality_score - error_penalty)

        return min(1.0, quality_score)

    def _calculate_option_quality_score(
        self, result: MarketDataValidationResult
    ) -> float:
        """Calculate quality score for option data."""
        # Similar to stock but with different weights
        price_score = 1.0 if result.price_consistency_passed else 0.4
        volume_score = (
            1.0 if result.volume_consistency_passed else 0.8
        )  # Volume less critical for options
        temporal_score = 1.0 if result.temporal_consistency_passed else 0.7

        weights = {"price": 0.5, "volume": 0.2, "temporal": 0.3}

        quality_score = (
            price_score * weights["price"]
            + volume_score * weights["volume"]
            + temporal_score * weights["temporal"]
        )

        error_penalty = len(result.errors) * 0.15  # Higher penalty for options
        quality_score = max(0.0, quality_score - error_penalty)

        return min(1.0, quality_score)

    def _generate_summary_statistics(
        self, data: List[Dict[str, Any]], result: MarketDataValidationResult
    ):
        """Generate summary statistics for stock data."""
        prices = []
        volumes = []

        for record in data:
            price = record.get("price") or record.get("close_price")
            volume = record.get("volume")

            if price is not None:
                prices.append(price)
            if volume is not None:
                volumes.append(volume)

        if prices:
            result.price_stats = {
                "min": min(prices),
                "max": max(prices),
                "mean": statistics.mean(prices),
                "median": statistics.median(prices),
                "std": statistics.stdev(prices) if len(prices) > 1 else 0,
            }

        if volumes:
            result.volume_stats = {
                "min": min(volumes),
                "max": max(volumes),
                "mean": statistics.mean(volumes),
                "median": statistics.median(volumes),
                "total": sum(volumes),
            }

    def _generate_option_statistics(
        self, data: List[Dict[str, Any]], result: MarketDataValidationResult
    ):
        """Generate summary statistics for option data."""
        bid_prices = []
        ask_prices = []
        spreads = []
        ivs = []

        for record in data:
            if record.get("bid_price") is not None:
                bid_prices.append(record["bid_price"])
            if record.get("ask_price") is not None:
                ask_prices.append(record["ask_price"])
            if record.get("bid_ask_spread") is not None:
                spreads.append(record["bid_ask_spread"])
            if record.get("implied_volatility") is not None:
                ivs.append(record["implied_volatility"])

        result.price_stats = {}
        if bid_prices:
            result.price_stats["bid_mean"] = statistics.mean(bid_prices)
            result.price_stats["bid_median"] = statistics.median(bid_prices)
        if ask_prices:
            result.price_stats["ask_mean"] = statistics.mean(ask_prices)
            result.price_stats["ask_median"] = statistics.median(ask_prices)
        if spreads:
            result.price_stats["spread_mean"] = statistics.mean(spreads)
            result.price_stats["spread_median"] = statistics.median(spreads)
        if ivs:
            result.price_stats["iv_mean"] = statistics.mean(ivs)
            result.price_stats["iv_median"] = statistics.median(ivs)

    def _is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during regular market hours."""
        hour = timestamp.hour + timestamp.minute / 60.0
        weekday = timestamp.weekday()

        return (
            weekday < 5  # Monday-Friday
            and self.thresholds["market_hours_start"]
            <= hour
            <= self.thresholds["market_hours_end"]
        )

    def _count_trading_days(self, start_date: date, end_date: date) -> int:
        """Count trading days between dates."""
        days = 0
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:
                days += 1
            current += timedelta(days=1)
        return days
