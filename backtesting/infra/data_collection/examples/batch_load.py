#!/usr/bin/env python3
"""
Batch Historical Data Loading Example

This example demonstrates loading historical data for multiple symbols
with custom configuration and progress monitoring.

Usage:
    python batch_load.py
"""

import asyncio
from datetime import date, timedelta
from pathlib import Path

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
except ImportError:
    # Try package import (when installed as package)
    from backtesting.infra.data_collection.load_historical_pipeline import (
        HistoricalDataPipeline,
        PipelineConfig,
        setup_logging,
    )


class BatchLoadConfig(PipelineConfig):
    """Custom configuration for batch loading."""

    def __init__(self):
        super().__init__()
        # Override some defaults for batch processing
        self.defaults["loading"]["batch_size"] = 5
        self.defaults["loading"]["request_delay"] = 0.3
        self.defaults["validation"]["enabled"] = True
        self.defaults["logging"]["level"] = "INFO"


async def batch_load_example():
    """Example of batch loading multiple symbols."""

    # Use custom configuration
    config = BatchLoadConfig()
    setup_logging(config)

    # Initialize pipeline
    pipeline = HistoricalDataPipeline(config)

    try:
        print("🚀 Starting batch historical data load example...")

        # Initialize connections
        await pipeline.initialize()

        # Define symbols and date range
        symbols = [
            "SPY",  # S&P 500 ETF
            "QQQ",  # NASDAQ 100 ETF
            "IWM",  # Russell 2000 ETF
            "AAPL",  # Apple
            "MSFT",  # Microsoft
        ]

        # Load last 14 days
        end_date = date.today()
        start_date = end_date - timedelta(days=14)

        print(
            f"📊 Loading data for {len(symbols)} symbols from {start_date} to {end_date}"
        )
        print(f"   Symbols: {', '.join(symbols)}")

        # Load historical data with validation
        stats = await pipeline.load_historical_data(
            symbols=symbols, start_date=start_date, end_date=end_date, validate=True
        )

        # Analyze results
        success_rate = (stats["symbols_processed"] / len(symbols)) * 100

        print(f"\n✅ Batch loading completed!")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Total stock bars: {stats['total_stock_bars']:,}")
        print(f"   Total option bars: {stats['total_option_bars']:,}")

        if stats["symbols_failed"] > 0:
            print(f"   ⚠️  Failed symbols: {stats['symbols_failed']}")

        # Show performance metrics
        duration = (stats["end_time"] - stats["start_time"]).total_seconds()
        if duration > 0:
            symbols_per_minute = (stats["symbols_processed"] / duration) * 60
            print(f"   Processing rate: {symbols_per_minute:.1f} symbols/minute")

        # Detailed summary
        pipeline.print_summary(stats)

        # Show loader statistics
        loader_stats = pipeline.loader.get_stats()
        print(f"\n📈 API Request Statistics:")
        print(f"   Completed requests: {loader_stats['completed_requests']}")
        print(f"   Failed requests: {loader_stats['failed_requests']}")

        if loader_stats["failed_requests"] > 0:
            failure_rate = (
                loader_stats["failed_requests"]
                / (loader_stats["completed_requests"] + loader_stats["failed_requests"])
            ) * 100
            print(f"   Request failure rate: {failure_rate:.1f}%")

    except Exception as e:
        print(f"❌ Error during batch loading: {e}")
        logging.exception("Batch load failed")

    finally:
        await pipeline.cleanup()


async def batch_load_with_backfill():
    """Example combining regular load with backfill operation."""

    config = BatchLoadConfig()
    setup_logging(config)
    pipeline = HistoricalDataPipeline(config)

    try:
        print("🔄 Starting batch load with backfill example...")

        await pipeline.initialize()

        symbols = ["SPY", "QQQ"]

        # First, do a regular load for recent data
        print("📊 Phase 1: Loading recent data...")
        recent_stats = await pipeline.load_historical_data(
            symbols=symbols,
            start_date=date.today() - timedelta(days=3),
            end_date=date.today(),
            validate=True,
        )

        # Then backfill any missing data from the last 30 days
        print("\n🔍 Phase 2: Backfilling missing data...")
        backfill_stats = await pipeline.backfill_missing_data(
            symbols=symbols, days_back=30
        )

        # Combined results
        print(f"\n✅ Combined operation completed!")
        print(f"   Recent data - Stock bars: {recent_stats['total_stock_bars']}")
        print(f"   Recent data - Option bars: {recent_stats['total_option_bars']}")
        print(
            f"   Backfill - Symbols checked: {backfill_stats.get('symbols_checked', 0)}"
        )

        pipeline.print_summary(recent_stats)

    except Exception as e:
        print(f"❌ Error during combined operation: {e}")
        logging.exception("Combined operation failed")

    finally:
        await pipeline.cleanup()


def main():
    """Main entry point with multiple examples."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch loading examples")
    parser.add_argument(
        "--mode",
        choices=["batch", "backfill"],
        default="batch",
        help="Example mode to run",
    )

    args = parser.parse_args()

    try:
        if args.mode == "batch":
            asyncio.run(batch_load_example())
        elif args.mode == "backfill":
            asyncio.run(batch_load_with_backfill())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
