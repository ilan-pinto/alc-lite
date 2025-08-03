"""
Data validation utilities for options market data.
Ensures data integrity and quality for backtesting.
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import logging
import numpy as np

from .collector import MarketDataSnapshot

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Validation result status."""

    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"


class ValidationRule:
    """Base class for validation rules."""

    def __init__(self, name: str, description: str, critical: bool = False):
        self.name = name
        self.description = description
        self.critical = critical  # If true, failure means data should be rejected

    async def validate(
        self, data: Any, context: Dict = None
    ) -> Tuple[ValidationResult, str]:
        """
        Validate data against this rule.

        Returns:
            Tuple of (result, message)
        """
        raise NotImplementedError


class PriceSanityRule(ValidationRule):
    """Validate price sanity (bid < ask, positive values, etc.)."""

    def __init__(self):
        super().__init__(
            "price_sanity",
            "Validate basic price relationships and bounds",
            critical=True,
        )

    async def validate(
        self, snapshot: MarketDataSnapshot, context: Dict = None
    ) -> Tuple[ValidationResult, str]:
        issues = []

        # Check bid/ask relationship
        if snapshot.bid_price is not None and snapshot.ask_price is not None:

            if snapshot.bid_price >= snapshot.ask_price:
                issues.append(
                    f"Bid ({snapshot.bid_price}) >= Ask ({snapshot.ask_price})"
                )

            # Check spread reasonableness
            spread = snapshot.ask_price - snapshot.bid_price
            mid_price = (snapshot.bid_price + snapshot.ask_price) / 2

            if spread > mid_price * 0.5:  # Spread > 50% of mid
                issues.append(
                    f"Excessive spread: {spread:.4f} ({spread/mid_price*100:.1f}% of mid)"
                )

        # Check for negative prices
        prices = [
            ("bid", snapshot.bid_price),
            ("ask", snapshot.ask_price),
            ("last", snapshot.last_price),
        ]

        for price_type, price in prices:
            if price is not None and price <= 0:
                issues.append(f"Non-positive {price_type} price: {price}")

        # Check sizes
        sizes = [
            ("bid_size", snapshot.bid_size),
            ("ask_size", snapshot.ask_size),
            ("volume", snapshot.volume),
        ]

        for size_type, size in sizes:
            if size is not None and size < 0:
                issues.append(f"Negative {size_type}: {size}")

        if issues:
            return ValidationResult.INVALID, "; ".join(issues)

        return ValidationResult.VALID, "Prices valid"


class GreeksBoundsRule(ValidationRule):
    """Validate Greeks are within reasonable bounds."""

    def __init__(self):
        super().__init__(
            "greeks_bounds",
            "Validate option Greeks within theoretical bounds",
            critical=False,
        )

    async def validate(
        self, snapshot: MarketDataSnapshot, context: Dict = None
    ) -> Tuple[ValidationResult, str]:
        issues = []
        warnings = []

        # Delta bounds
        if snapshot.delta is not None:
            if not (-1.0 <= snapshot.delta <= 1.0):
                issues.append(f"Delta out of bounds: {snapshot.delta}")
            elif abs(snapshot.delta) > 0.99:
                warnings.append(f"Extreme delta: {snapshot.delta}")

        # Gamma bounds (should be positive for long options)
        if snapshot.gamma is not None:
            if snapshot.gamma < 0:
                issues.append(f"Negative gamma: {snapshot.gamma}")
            elif snapshot.gamma > 1.0:
                warnings.append(f"High gamma: {snapshot.gamma}")

        # Theta (usually negative for long options)
        if snapshot.theta is not None:
            if abs(snapshot.theta) > 1.0:
                warnings.append(f"High theta: {snapshot.theta}")

        # Vega bounds
        if snapshot.vega is not None:
            if snapshot.vega < 0:
                warnings.append(f"Negative vega: {snapshot.vega}")
            elif snapshot.vega > 2.0:
                warnings.append(f"High vega: {snapshot.vega}")

        # Implied volatility bounds
        if snapshot.implied_volatility is not None:
            if not (0.001 <= snapshot.implied_volatility <= 10.0):  # 0.1% to 1000%
                issues.append(f"IV out of bounds: {snapshot.implied_volatility}")
            elif snapshot.implied_volatility > 3.0:
                warnings.append(f"Very high IV: {snapshot.implied_volatility}")

        if issues:
            return ValidationResult.INVALID, "; ".join(issues)
        elif warnings:
            return ValidationResult.WARNING, "; ".join(warnings)

        return ValidationResult.VALID, "Greeks within bounds"


class TimeConsistencyRule(ValidationRule):
    """Validate timestamp consistency."""

    def __init__(self):
        super().__init__(
            "time_consistency", "Validate timestamp is reasonable", critical=True
        )

    async def validate(
        self, snapshot: MarketDataSnapshot, context: Dict = None
    ) -> Tuple[ValidationResult, str]:
        now = datetime.now()
        age = (now - snapshot.timestamp).total_seconds()

        # Data from the future
        if age < -60:  # Allow 1 minute clock skew
            return ValidationResult.INVALID, f"Future timestamp: {snapshot.timestamp}"

        # Very old data
        if age > 86400:  # 24 hours
            return ValidationResult.WARNING, f"Old data: {age/3600:.1f} hours old"

        # Weekend data during weekdays (might be stale)
        if snapshot.timestamp.weekday() >= 5 and now.weekday() < 5:
            return ValidationResult.WARNING, "Weekend data on weekday"

        return ValidationResult.VALID, "Timestamp valid"


class VolumeConsistencyRule(ValidationRule):
    """Validate volume and open interest consistency."""

    def __init__(self, db_pool: asyncpg.Pool):
        super().__init__(
            "volume_consistency",
            "Validate volume progression and open interest",
            critical=False,
        )
        self.db_pool = db_pool

    async def validate(
        self, snapshot: MarketDataSnapshot, context: Dict = None
    ) -> Tuple[ValidationResult, str]:
        if snapshot.volume is None:
            return ValidationResult.VALID, "No volume data"

        try:
            async with self.db_pool.acquire() as conn:
                # Get previous volume for comparison
                prev_volume = await conn.fetchval(
                    """
                    SELECT volume
                    FROM market_data_ticks
                    WHERE contract_id = $1
                      AND time < $2
                      AND volume IS NOT NULL
                    ORDER BY time DESC
                    LIMIT 1
                """,
                    snapshot.contract_id,
                    snapshot.timestamp,
                )

                if prev_volume is not None:
                    # Volume should generally increase during trading day
                    if snapshot.volume < prev_volume:
                        # Check if this is a new trading day
                        prev_day = await conn.fetchval(
                            """
                            SELECT DATE(time)
                            FROM market_data_ticks
                            WHERE contract_id = $1
                              AND volume = $2
                            LIMIT 1
                        """,
                            snapshot.contract_id,
                            prev_volume,
                        )

                        current_day = snapshot.timestamp.date()

                        if prev_day == current_day:
                            return (
                                ValidationResult.WARNING,
                                f"Volume decreased: {prev_volume} -> {snapshot.volume}",
                            )

        except Exception as e:
            logger.error(f"Error in volume validation: {e}")
            return ValidationResult.WARNING, f"Volume validation error: {e}"

        return ValidationResult.VALID, "Volume consistent"


class DataValidator:
    """
    Comprehensive data validator for options market data.
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

        # Initialize validation rules
        self.rules = [
            PriceSanityRule(),
            GreeksBoundsRule(),
            TimeConsistencyRule(),
            VolumeConsistencyRule(db_pool),
        ]

        # Statistics
        self.stats = {
            "total_validated": 0,
            "valid": 0,
            "warnings": 0,
            "invalid": 0,
            "rule_failures": {},
        }

    async def validate_snapshot(
        self, snapshot: MarketDataSnapshot, context: Dict = None
    ) -> bool:
        """
        Validate a market data snapshot.

        Args:
            snapshot: Market data to validate
            context: Additional context for validation

        Returns:
            True if data is valid (warnings allowed), False if invalid
        """
        self.stats["total_validated"] += 1

        valid = True
        warnings = []
        errors = []

        for rule in self.rules:
            try:
                result, message = await rule.validate(snapshot, context)

                if result == ValidationResult.INVALID:
                    if rule.critical:
                        valid = False
                    errors.append(f"{rule.name}: {message}")

                    # Track rule failures
                    if rule.name not in self.stats["rule_failures"]:
                        self.stats["rule_failures"][rule.name] = 0
                    self.stats["rule_failures"][rule.name] += 1

                elif result == ValidationResult.WARNING:
                    warnings.append(f"{rule.name}: {message}")

            except Exception as e:
                logger.error(f"Validation rule {rule.name} failed: {e}")
                if rule.critical:
                    valid = False
                errors.append(f"{rule.name}: validation error")

        # Update statistics
        if valid:
            if warnings:
                self.stats["warnings"] += 1
                logger.debug(f"Data validation warnings: {'; '.join(warnings)}")
            else:
                self.stats["valid"] += 1
        else:
            self.stats["invalid"] += 1
            logger.warning(f"Data validation failed: {'; '.join(errors)}")

        # Log quality issues to database
        if errors or warnings:
            await self._log_quality_issue(snapshot, errors + warnings, not valid)

        return valid

    async def validate_batch(self, snapshots: List[MarketDataSnapshot]) -> List[bool]:
        """Validate a batch of snapshots."""
        results = []

        for snapshot in snapshots:
            result = await self.validate_snapshot(snapshot)
            results.append(result)

        return results

    async def _log_quality_issue(
        self, snapshot: MarketDataSnapshot, issues: List[str], is_error: bool
    ):
        """Log data quality issue to database."""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO data_quality_metrics
                    (check_timestamp, metric_type, metric_value, passed, details)
                    VALUES ($1, $2, $3, $4, $5)
                """,
                    snapshot.timestamp,
                    "VALIDATION",
                    len(issues),
                    not is_error,
                    {"contract_id": snapshot.contract_id, "issues": issues},
                )
        except Exception as e:
            logger.error(f"Failed to log quality issue: {e}")

    async def run_quality_check(
        self, contract_id: Optional[int] = None, hours_back: int = 24
    ) -> Dict[str, Any]:
        """
        Run comprehensive data quality check.

        Args:
            contract_id: Specific contract to check (None for all)
            hours_back: Hours of data to analyze

        Returns:
            Quality report dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        async with self.db_pool.acquire() as conn:
            # Base query conditions
            where_conditions = ["time >= $1"]
            params = [cutoff_time]

            if contract_id:
                where_conditions.append("contract_id = $2")
                params.append(contract_id)

            where_clause = " AND ".join(where_conditions)

            # Count total records
            total_records = await conn.fetchval(
                f"""
                SELECT COUNT(*) FROM market_data_ticks WHERE {where_clause}
            """,
                *params,
            )

            # Count records with bid/ask data
            price_records = await conn.fetchval(
                f"""
                SELECT COUNT(*) FROM market_data_ticks
                WHERE {where_clause}
                  AND bid_price IS NOT NULL
                  AND ask_price IS NOT NULL
            """,
                *params,
            )

            # Find wide spreads
            wide_spreads = await conn.fetchval(
                f"""
                SELECT COUNT(*) FROM market_data_ticks
                WHERE {where_clause}
                  AND bid_price IS NOT NULL
                  AND ask_price IS NOT NULL
                  AND bid_ask_spread > mid_price * 0.1  -- 10% spread
            """,
                *params,
            )

            # Count stale data (no updates in last hour)
            stale_contracts = await conn.fetchval(
                f"""
                SELECT COUNT(DISTINCT contract_id) FROM market_data_ticks t1
                WHERE {where_clause}
                  AND NOT EXISTS (
                      SELECT 1 FROM market_data_ticks t2
                      WHERE t2.contract_id = t1.contract_id
                        AND t2.time > NOW() - INTERVAL '1 hour'
                  )
            """,
                *params,
            )

            # Get contracts with missing Greeks
            missing_greeks = await conn.fetchval(
                f"""
                SELECT COUNT(DISTINCT contract_id) FROM market_data_ticks
                WHERE {where_clause}
                  AND (delta IS NULL OR gamma IS NULL OR theta IS NULL)
            """,
                *params,
            )

            # Find data gaps
            gaps = await conn.fetch(
                f"""
                WITH time_series AS (
                    SELECT contract_id,
                           time,
                           LAG(time) OVER (PARTITION BY contract_id ORDER BY time) as prev_time
                    FROM market_data_ticks
                    WHERE {where_clause}
                )
                SELECT contract_id, COUNT(*) as gap_count
                FROM time_series
                WHERE EXTRACT(EPOCH FROM (time - prev_time)) > 300  -- 5 minute gaps
                GROUP BY contract_id
                HAVING COUNT(*) > 5
                ORDER BY gap_count DESC
                LIMIT 10
            """,
                *params,
            )

        report = {
            "period_hours": hours_back,
            "total_records": total_records,
            "price_data_coverage": price_records / max(total_records, 1) * 100,
            "wide_spread_pct": wide_spreads / max(price_records, 1) * 100,
            "stale_contracts": stale_contracts,
            "missing_greeks_contracts": missing_greeks,
            "contracts_with_gaps": len(gaps),
            "validation_stats": self.stats.copy(),
            "top_gappy_contracts": [dict(g) for g in gaps],
        }

        return report

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.stats["total_validated"]
        if total == 0:
            return self.stats

        return {
            **self.stats,
            "valid_pct": self.stats["valid"] / total * 100,
            "warning_pct": self.stats["warnings"] / total * 100,
            "invalid_pct": self.stats["invalid"] / total * 100,
        }

    def reset_stats(self):
        """Reset validation statistics."""
        self.stats = {
            "total_validated": 0,
            "valid": 0,
            "warnings": 0,
            "invalid": 0,
            "rule_failures": {},
        }
