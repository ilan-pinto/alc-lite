#!/usr/bin/env python3
"""
Custom Validation Example

This example demonstrates how to create and use custom validation rules
with the historical data loading pipeline.

Usage:
    python custom_validation.py
"""

import asyncio
from datetime import date, datetime, timedelta
from typing import Dict, Tuple

import logging

try:
    # Try importing from parent directory (when run from examples/)
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from load_historical_pipeline import (
        HistoricalDataPipeline,
        PipelineConfig,
        setup_logging,
    )
    from validators import (
        DataValidator,
        MarketDataSnapshot,
        ValidationResult,
        ValidationRule,
    )
except ImportError:
    # Try package import (when installed as package)
    from backtesting.infra.data_collection.load_historical_pipeline import (
        HistoricalDataPipeline,
        PipelineConfig,
        setup_logging,
    )
    from backtesting.infra.data_collection.validators import (
        DataValidator,
        MarketDataSnapshot,
        ValidationResult,
        ValidationRule,
    )


class TradingHoursRule(ValidationRule):
    """Validate that data timestamps are within trading hours."""

    def __init__(self):
        super().__init__(
            "trading_hours",
            "Validate data is within market trading hours",
            critical=False,
        )
        # NYSE trading hours (9:30 AM - 4:00 PM ET)
        self.market_open = datetime.strptime("09:30", "%H:%M").time()
        self.market_close = datetime.strptime("16:00", "%H:%M").time()

    async def validate(
        self, snapshot: MarketDataSnapshot, context: Dict = None
    ) -> Tuple[ValidationResult, str]:
        """Validate timestamp is within trading hours."""

        # Check if it's a weekend
        if snapshot.timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return ValidationResult.WARNING, "Weekend data"

        # Check trading hours (assuming ET timezone)
        time_of_day = snapshot.timestamp.time()

        if not (self.market_open <= time_of_day <= self.market_close):
            return ValidationResult.WARNING, f"Outside trading hours: {time_of_day}"

        return ValidationResult.VALID, "Within trading hours"


class PriceMovementRule(ValidationRule):
    """Validate reasonable price movements between ticks."""

    def __init__(self, db_pool):
        super().__init__(
            "price_movement", "Validate reasonable price movements", critical=False
        )
        self.db_pool = db_pool
        self.max_movement_percent = 0.10  # 10% max movement

    async def validate(
        self, snapshot: MarketDataSnapshot, context: Dict = None
    ) -> Tuple[ValidationResult, str]:
        """Validate price movement is reasonable."""

        if not snapshot.last_price:
            return ValidationResult.VALID, "No price data"

        try:
            async with self.db_pool.acquire() as conn:
                # Get previous price for comparison
                prev_price = await conn.fetchval(
                    """
                    SELECT last_price
                    FROM market_data_ticks
                    WHERE contract_id = $1
                      AND time < $2
                      AND last_price IS NOT NULL
                    ORDER BY time DESC
                    LIMIT 1
                """,
                    snapshot.contract_id,
                    snapshot.timestamp,
                )

                if prev_price:
                    # Calculate percentage change
                    price_change = abs(snapshot.last_price - prev_price) / prev_price

                    if price_change > self.max_movement_percent:
                        return ValidationResult.WARNING, (
                            f"Large price movement: {price_change*100:.1f}% "
                            f"({prev_price} -> {snapshot.last_price})"
                        )

        except Exception as e:
            return ValidationResult.WARNING, f"Price movement check failed: {e}"

        return ValidationResult.VALID, "Price movement reasonable"


class OptionsPricingRule(ValidationRule):
    """Validate options pricing relationships."""

    def __init__(self):
        super().__init__(
            "options_pricing", "Validate options pricing relationships", critical=False
        )

    async def validate(
        self, snapshot: MarketDataSnapshot, context: Dict = None
    ) -> Tuple[ValidationResult, str]:
        """Validate options pricing makes sense."""

        # This is a simplified example - in practice you'd need more context
        # about the option (strike, expiry, underlying price, etc.)

        if not (snapshot.bid_price and snapshot.ask_price):
            return ValidationResult.VALID, "Incomplete price data"

        # Basic intrinsic value check (would need more context for real implementation)
        if snapshot.bid_price > snapshot.ask_price:
            return ValidationResult.INVALID, "Bid > Ask"

        # Check for reasonable option prices (simplified)
        mid_price = (snapshot.bid_price + snapshot.ask_price) / 2

        if mid_price < 0.01:
            return ValidationResult.WARNING, "Very low option price"

        if mid_price > 1000:
            return ValidationResult.WARNING, "Very high option price"

        return ValidationResult.VALID, "Option pricing reasonable"


class VolumeAnomalyRule(ValidationRule):
    """Detect volume anomalies."""

    def __init__(self):
        super().__init__(
            "volume_anomaly", "Detect unusual volume patterns", critical=False
        )

    async def validate(
        self, snapshot: MarketDataSnapshot, context: Dict = None
    ) -> Tuple[ValidationResult, str]:
        """Detect volume anomalies."""

        if not snapshot.volume:
            return ValidationResult.VALID, "No volume data"

        # Simple anomaly detection - in practice you'd use historical patterns
        if snapshot.volume > 1000000:  # Very high volume
            return (
                ValidationResult.WARNING,
                f"Unusually high volume: {snapshot.volume:,}",
            )

        if (
            snapshot.volume == 0
            and snapshot.timestamp.time() > datetime.strptime("10:00", "%H:%M").time()
        ):
            return ValidationResult.WARNING, "Zero volume during trading hours"

        return ValidationResult.VALID, "Volume normal"


class CustomValidationPipeline(HistoricalDataPipeline):
    """Extended pipeline with custom validation rules."""

    async def initialize(self):
        """Initialize with custom validation rules."""
        await super().initialize()

        # Add custom validation rules
        custom_rules = [
            TradingHoursRule(),
            PriceMovementRule(self.db_pool),
            OptionsPricingRule(),
            VolumeAnomalyRule(),
        ]

        # Add to existing validator
        self.validator.rules.extend(custom_rules)

        print(f"✅ Added {len(custom_rules)} custom validation rules")
        print(f"   Total validation rules: {len(self.validator.rules)}")


async def custom_validation_example():
    """Example using custom validation rules."""

    config = PipelineConfig()
    # Enable debug logging to see validation details
    config.defaults["logging"]["level"] = "DEBUG"
    setup_logging(config)

    # Use custom pipeline with additional validation
    pipeline = CustomValidationPipeline(config)

    try:
        print("🔍 Starting custom validation example...")

        await pipeline.initialize()

        # Load data for a symbol known to have interesting price movements
        symbols = ["TSLA"]  # Tesla often has significant price movements
        end_date = date.today()
        start_date = end_date - timedelta(days=3)

        print(f"📊 Loading {symbols[0]} data with custom validation...")

        # Load with enhanced validation
        stats = await pipeline.load_historical_data(
            symbols=symbols, start_date=start_date, end_date=end_date, validate=True
        )

        print(f"\n✅ Loading with custom validation completed!")

        # Show validation statistics
        validation_stats = pipeline.validator.get_stats()

        print(f"\n🔍 Custom Validation Results:")
        print(f"   Total records validated: {validation_stats['total_validated']}")
        print(
            f"   Valid records: {validation_stats['valid']} ({validation_stats.get('valid_pct', 0):.1f}%)"
        )
        print(
            f"   Records with warnings: {validation_stats['warnings']} ({validation_stats.get('warning_pct', 0):.1f}%)"
        )
        print(
            f"   Invalid records: {validation_stats['invalid']} ({validation_stats.get('invalid_pct', 0):.1f}%)"
        )

        # Show rule-specific failures
        if validation_stats["rule_failures"]:
            print(f"\n⚠️  Rule-specific Issues:")
            for rule_name, count in validation_stats["rule_failures"].items():
                print(f"   {rule_name}: {count} failures")

        # Run comprehensive quality check
        print(f"\n📊 Running comprehensive quality check...")
        quality_report = await pipeline.validator.run_quality_check(hours_back=72)

        print(f"   Data coverage: {quality_report['price_data_coverage']:.1f}%")
        print(f"   Wide spread percentage: {quality_report['wide_spread_pct']:.1f}%")
        print(f"   Stale contracts: {quality_report['stale_contracts']}")

        pipeline.print_summary(stats)

    except Exception as e:
        print(f"❌ Error during custom validation: {e}")
        logging.exception("Custom validation failed")

    finally:
        await pipeline.cleanup()


async def validation_rule_testing():
    """Test individual validation rules."""

    print("🧪 Testing individual validation rules...")

    # Create sample data for testing
    test_snapshot = MarketDataSnapshot(
        contract_id=12345,
        timestamp=datetime.now(),
        bid_price=150.50,
        ask_price=150.55,
        last_price=150.52,
        bid_size=100,
        ask_size=100,
        last_size=50,
        volume=1000,
        open_interest=5000,
        delta=0.65,
        gamma=0.02,
        theta=-0.05,
        vega=0.15,
        implied_volatility=0.25,
    )

    # Test trading hours rule
    trading_rule = TradingHoursRule()
    result, message = await trading_rule.validate(test_snapshot)
    print(f"📅 Trading Hours Rule: {result.value} - {message}")

    # Test options pricing rule
    options_rule = OptionsPricingRule()
    result, message = await options_rule.validate(test_snapshot)
    print(f"💰 Options Pricing Rule: {result.value} - {message}")

    # Test volume anomaly rule
    volume_rule = VolumeAnomalyRule()
    result, message = await volume_rule.validate(test_snapshot)
    print(f"📈 Volume Anomaly Rule: {result.value} - {message}")

    # Test with problematic data
    print(f"\n🚨 Testing with problematic data...")

    bad_snapshot = MarketDataSnapshot(
        contract_id=12346,
        timestamp=datetime.now().replace(hour=2),  # 2 AM - outside trading hours
        bid_price=100.00,
        ask_price=95.00,  # Bid > Ask (invalid)
        last_price=97.50,
        volume=2000000,  # Very high volume
        tick_type="REALTIME",
    )

    result, message = await trading_rule.validate(bad_snapshot)
    print(f"📅 Trading Hours (bad): {result.value} - {message}")

    result, message = await options_rule.validate(bad_snapshot)
    print(f"💰 Options Pricing (bad): {result.value} - {message}")

    result, message = await volume_rule.validate(bad_snapshot)
    print(f"📈 Volume Anomaly (bad): {result.value} - {message}")


def main():
    """Main entry point for custom validation examples."""

    import argparse

    parser = argparse.ArgumentParser(description="Custom validation examples")
    parser.add_argument(
        "--mode",
        choices=["full", "test"],
        default="full",
        help="Run full pipeline or just test rules",
    )

    args = parser.parse_args()

    try:
        if args.mode == "full":
            asyncio.run(custom_validation_example())
        elif args.mode == "test":
            asyncio.run(validation_rule_testing())

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
