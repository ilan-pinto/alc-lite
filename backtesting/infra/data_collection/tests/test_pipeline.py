#!/usr/bin/env python3
"""
Pipeline Test Script

Quick test script to verify the historical data loading pipeline works
without requiring a full database setup.

Usage:
    python test_pipeline.py
"""

import asyncio
import sys
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import logging

from ..pipelines.load_historical_pipeline import (
    HistoricalDataPipeline,
    PipelineConfig,
    setup_logging,
)


async def test_pipeline_initialization():
    """Test pipeline initialization with mocked connections."""

    print("🧪 Testing pipeline initialization...")

    config = PipelineConfig()
    pipeline = HistoricalDataPipeline(config)

    # Mock database and IB connections
    with patch("asyncpg.create_pool") as mock_db, patch("ib_async.IB") as mock_ib:

        # Configure mocks
        mock_db.return_value = AsyncMock()
        mock_ib_instance = AsyncMock()
        mock_ib.return_value = mock_ib_instance
        mock_ib_instance.connectAsync = AsyncMock()

        try:
            await pipeline.initialize()
            print("✅ Pipeline initialization successful")

            # Verify connections were attempted
            mock_db.assert_called_once()
            mock_ib_instance.connectAsync.assert_called_once()

            return True

        except Exception as e:
            print(f"❌ Pipeline initialization failed: {e}")
            return False

        finally:
            await pipeline.cleanup()


async def test_configuration_loading():
    """Test configuration loading and validation."""

    print("🧪 Testing configuration loading...")

    try:
        # Test default config
        config = PipelineConfig()

        # Verify some key settings
        assert config.get("database", "host") == "localhost"
        assert config.get("database", "port") == 5432
        assert config.get("ib_connection", "port") == 7497
        assert config.get("loading", "batch_size") == 10

        print("✅ Configuration loading successful")
        return True

    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


def test_cli_argument_parsing():
    """Test command-line argument parsing."""

    print("🧪 Testing CLI argument parsing...")

    try:
        # Test various argument combinations
        from load_historical_pipeline import parse_date, parse_symbols

        # Test symbol parsing
        symbols = parse_symbols("SPY,QQQ,AAPL")
        assert symbols == ["SPY", "QQQ", "AAPL"]

        # Test date parsing
        test_date = parse_date("2024-01-01")
        assert test_date == date(2024, 1, 1)

        print("✅ CLI argument parsing successful")
        return True

    except Exception as e:
        print(f"❌ CLI argument parsing failed: {e}")
        return False


async def test_data_validation():
    """Test data validation rules."""

    print("🧪 Testing data validation...")

    try:
        from datetime import datetime

        from ..core.collector import MarketDataSnapshot
        from ..core.validators import PriceSanityRule, ValidationResult

        # Create test data
        valid_snapshot = MarketDataSnapshot(
            contract_id=12345,
            timestamp=datetime.now(),
            bid_price=150.00,
            ask_price=150.05,
            last_price=150.02,
            bid_size=100,
            ask_size=100,
            volume=1000,
            tick_type="TEST",
        )

        # Test price sanity rule
        rule = PriceSanityRule()
        result, message = await rule.validate(valid_snapshot)

        assert result == ValidationResult.VALID

        # Test invalid data
        invalid_snapshot = MarketDataSnapshot(
            contract_id=12346,
            timestamp=datetime.now(),
            bid_price=150.05,  # Bid > Ask (invalid)
            ask_price=150.00,
            last_price=150.02,
            tick_type="TEST",
        )

        result, message = await rule.validate(invalid_snapshot)
        assert result == ValidationResult.INVALID

        print("✅ Data validation testing successful")
        return True

    except Exception as e:
        print(f"❌ Data validation testing failed: {e}")
        return False


async def test_mock_data_loading():
    """Test data loading with completely mocked data."""

    print("🧪 Testing mock data loading...")

    config = PipelineConfig()
    pipeline = HistoricalDataPipeline(config)

    # Mock all external dependencies
    with (
        patch("asyncpg.create_pool") as mock_db,
        patch("ib_async.IB") as mock_ib,
        patch.object(pipeline, "loader") as mock_loader,
    ):

        # Configure mocks
        mock_db.return_value = AsyncMock()
        mock_ib_instance = AsyncMock()
        mock_ib.return_value = mock_ib_instance
        mock_ib_instance.connectAsync = AsyncMock()
        mock_ib_instance.isConnected.return_value = True
        mock_ib_instance.disconnect = MagicMock()

        # Mock loader with successful statistics
        mock_loader.load_symbol_history = AsyncMock(
            return_value={
                "symbol": "SPY",
                "stock_bars_loaded": 100,
                "option_bars_loaded": 500,
                "errors": 0,
            }
        )

        try:
            await pipeline.initialize()

            # Test data loading
            symbols = ["SPY"]
            start_date = date.today() - timedelta(days=7)
            end_date = date.today()

            stats = await pipeline.load_historical_data(
                symbols, start_date, end_date, validate=False
            )

            # Verify results
            assert stats["symbols_processed"] == 1
            assert stats["symbols_failed"] == 0
            assert stats["total_stock_bars"] == 100
            assert stats["total_option_bars"] == 500

            print("✅ Mock data loading successful")
            return True

        except Exception as e:
            print(f"❌ Mock data loading failed: {e}")
            return False

        finally:
            await pipeline.cleanup()


def run_syntax_checks():
    """Run basic syntax and import checks."""

    print("🧪 Running syntax checks...")

    try:
        # Try importing all main modules
        from ..config import config
        from ..core.collector import MarketDataSnapshot, OptionsDataCollector
        from ..core.historical_loader import HistoricalDataLoader
        from ..core.validators import DataValidator
        from ..core.vix_collector import VIXDataCollector
        from ..pipelines import load_historical_pipeline

        print("✅ All imports successful")

        # Check key classes can be instantiated
        config_obj = load_historical_pipeline.PipelineConfig()
        assert config_obj is not None

        print("✅ Syntax checks passed")
        return True

    except Exception as e:
        print(f"❌ Syntax check failed: {e}")
        return False


async def run_all_tests():
    """Run all tests and report results."""

    print("🚀 Starting pipeline test suite...\n")

    test_results = []

    # Run tests
    tests = [
        ("Syntax Checks", run_syntax_checks),
        ("Configuration Loading", test_configuration_loading),
        ("CLI Argument Parsing", test_cli_argument_parsing),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Data Validation", test_data_validation),
        ("Mock Data Loading", test_mock_data_loading),
    ]

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")

        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            test_results.append((test_name, result))

        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            test_results.append((test_name, False))

    # Report results
    print(f"\n{'='*50}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*50}")

    passed = 0
    failed = 0

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {len(test_results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n🎉 All tests passed! Pipeline is ready to use.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review and fix issues.")
        return False


def main():
    """Main test entry point."""

    # Set up basic logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Test suite crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
