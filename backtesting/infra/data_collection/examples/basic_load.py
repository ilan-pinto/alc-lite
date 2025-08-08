#!/usr/bin/env python3
"""
Basic Historical Data Loading Example

This example demonstrates the simplest way to load historical data
for a single symbol using the pipeline programmatically.

Usage:
    python basic_load.py
"""

import asyncio
from datetime import date, timedelta

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


async def basic_load_example():
    """Basic example of loading historical data for SPY."""

    # Load default configuration
    config = PipelineConfig()

    # Set up logging
    setup_logging(config)

    # Initialize pipeline
    pipeline = HistoricalDataPipeline(config)

    try:
        print("🚀 Starting basic historical data load example...")

        # Initialize connections
        await pipeline.initialize()

        # Define parameters
        symbols = ["SPY"]  # Single symbol
        end_date = date.today()
        start_date = end_date - timedelta(days=7)  # Last 7 days

        print(f"📊 Loading data for {symbols[0]} from {start_date} to {end_date}")

        # Load historical data
        stats = await pipeline.load_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            validate=True,  # Enable validation
        )

        # Print results
        print(f"\n✅ Loading completed!")
        print(f"   Stock bars loaded: {stats['total_stock_bars']}")
        print(f"   Option bars loaded: {stats['total_option_bars']}")
        print(f"   Symbols processed: {stats['symbols_processed']}")
        print(f"   Errors: {stats['total_errors']}")

        # Show detailed summary
        pipeline.print_summary(stats)

    except Exception as e:
        print(f"❌ Error during loading: {e}")
        logging.exception("Basic load failed")

    finally:
        # Clean up connections
        await pipeline.cleanup()


def main():
    """Main entry point."""
    try:
        asyncio.run(basic_load_example())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
