#!/usr/bin/env python3
"""
Test script for the unified data loading pipeline.
"""

import asyncio
import sys
from datetime import date, timedelta

from ..pipelines.load_historical_pipeline import HistoricalDataPipeline, PipelineConfig


async def test_unified_pipeline():
    """Test the unified pipeline with sample data."""
    print("🚀 Testing Unified Data Loading Pipeline")
    print("=" * 50)

    # Create configuration
    config = PipelineConfig()

    # Create pipeline
    pipeline = HistoricalDataPipeline(config)

    try:
        # Initialize
        print("📡 Initializing pipeline...")
        await pipeline.initialize()

        # Test parameters
        symbols = ["SPY"]
        end_date = date.today()
        start_date = end_date - timedelta(days=5)  # Small test dataset

        print(f"📊 Testing complete data load for {symbols}")
        print(f"📅 Date range: {start_date} to {end_date}")

        # Test complete data loading
        stats = await pipeline.load_complete_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            include_options=True,
            include_vix=True,
            validate=False,  # Skip validation for quick test
            option_expiry_days=30,  # Shorter range for test
            option_strike_range=0.10,  # Narrow range for test
        )

        # Print results
        pipeline.print_summary(stats)

        print("\n✅ Test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        await pipeline.cleanup()


async def test_basic_pipeline():
    """Test the basic pipeline functionality."""
    print("🧪 Testing Basic Pipeline")
    print("=" * 30)

    config = PipelineConfig()
    pipeline = HistoricalDataPipeline(config)

    try:
        await pipeline.initialize()

        symbols = ["SPY"]
        end_date = date.today()
        start_date = end_date - timedelta(days=2)

        print(f"📈 Testing stock-only load for {symbols}")

        # Test stock-only loading
        stats = await pipeline.load_historical_data(
            symbols=symbols, start_date=start_date, end_date=end_date, validate=False
        )

        pipeline.print_summary(stats)
        print("✅ Basic test completed!")
        return True

    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

    finally:
        await pipeline.cleanup()


async def main():
    """Run all tests."""
    print("🔧 Unified Pipeline Test Suite")
    print("=" * 40)

    # Test 1: Basic functionality
    print("\n1️⃣ Running basic pipeline test...")
    basic_success = await test_basic_pipeline()

    if not basic_success:
        print("❌ Basic test failed, skipping unified test")
        sys.exit(1)

    # Test 2: Unified functionality
    print("\n2️⃣ Running unified pipeline test...")
    unified_success = await test_unified_pipeline()

    # Results
    print("\n" + "=" * 40)
    print("📋 Test Results:")
    print(f"   Basic Pipeline: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"   Unified Pipeline: {'✅ PASS' if unified_success else '❌ FAIL'}")

    if basic_success and unified_success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
