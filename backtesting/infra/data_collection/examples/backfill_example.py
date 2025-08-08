#!/usr/bin/env python3
"""
Data Backfill Example

This example demonstrates how to identify and fill missing data gaps
in the database using the backfill functionality.

Usage:
    python backfill_example.py
    python backfill_example.py --symbol AAPL --days 60
"""

import asyncio
from datetime import date, timedelta

import argparse
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


async def analyze_data_gaps(pipeline, symbols, days_back=30):
    """Analyze data gaps before backfilling."""

    print(
        f"🔍 Analyzing data gaps for {len(symbols)} symbols (last {days_back} days)..."
    )

    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    gap_analysis = {}

    for symbol in symbols:
        try:
            # Check for gaps using the database
            async with pipeline.db_pool.acquire() as conn:
                # Find missing trading days
                gaps = await conn.fetch(
                    """
                    WITH date_series AS (
                        SELECT generate_series(
                            $2::date, $3::date, '1 day'::interval
                        )::date AS trading_date
                    ),
                    existing_dates AS (
                        SELECT DISTINCT DATE(time) as data_date
                        FROM stock_data_ticks st
                        JOIN underlying_securities us ON st.underlying_id = us.id
                        WHERE us.symbol = $1
                          AND time >= $2
                          AND time < $3 + interval '1 day'
                    )
                    SELECT ds.trading_date
                    FROM date_series ds
                    LEFT JOIN existing_dates ed ON ds.trading_date = ed.data_date
                    WHERE ed.data_date IS NULL
                      AND EXTRACT(dow FROM ds.trading_date) NOT IN (0, 6)  -- Exclude weekends
                    ORDER BY ds.trading_date
                """,
                    symbol,
                    start_date,
                    end_date,
                )

                gap_count = len(gaps)
                gap_analysis[symbol] = {
                    "gaps": gap_count,
                    "gap_dates": [
                        gap["trading_date"] for gap in gaps[:5]
                    ],  # First 5 gaps
                }

                if gap_count > 0:
                    print(f"   📊 {symbol}: {gap_count} missing days")
                    if gap_count <= 5:
                        gap_dates = [str(gap["trading_date"]) for gap in gaps]
                        print(f"      Missing: {', '.join(gap_dates)}")
                    else:
                        recent_gaps = [str(gap["trading_date"]) for gap in gaps[:3]]
                        print(
                            f"      Recent gaps: {', '.join(recent_gaps)} (+{gap_count-3} more)"
                        )
                else:
                    print(f"   ✅ {symbol}: No gaps found")

        except Exception as e:
            print(f"   ❌ {symbol}: Error analyzing gaps - {e}")
            gap_analysis[symbol] = {"gaps": -1, "error": str(e)}

    return gap_analysis


async def backfill_with_analysis():
    """Backfill example with gap analysis."""

    config = PipelineConfig()
    setup_logging(config)
    pipeline = HistoricalDataPipeline(config)

    try:
        print("🔄 Starting intelligent backfill example...")

        await pipeline.initialize()

        # Symbols to check
        symbols = ["SPY", "QQQ", "AAPL", "MSFT"]
        days_back = 14

        # Phase 1: Analyze existing data gaps
        gap_analysis = await analyze_data_gaps(pipeline, symbols, days_back)

        # Phase 2: Prioritize symbols that need backfill
        symbols_needing_backfill = [
            symbol for symbol, data in gap_analysis.items() if data.get("gaps", 0) > 0
        ]

        if not symbols_needing_backfill:
            print("\n✅ No gaps found - all data is complete!")
            return

        print(f"\n🔧 Backfilling {len(symbols_needing_backfill)} symbols with gaps...")

        # Phase 3: Perform backfill
        backfill_stats = await pipeline.backfill_missing_data(
            symbols=symbols_needing_backfill, days_back=days_back
        )

        print(f"\n✅ Backfill completed!")
        print(f"   Symbols checked: {backfill_stats.get('symbols_checked', 0)}")
        print(f"   Errors: {backfill_stats.get('errors', 0)}")

        # Phase 4: Verify backfill results
        print(f"\n🔍 Verifying backfill results...")
        post_gap_analysis = await analyze_data_gaps(
            pipeline, symbols_needing_backfill, days_back
        )

        for symbol in symbols_needing_backfill:
            before = gap_analysis.get(symbol, {}).get("gaps", 0)
            after = post_gap_analysis.get(symbol, {}).get("gaps", 0)

            if after < before:
                filled = before - after
                print(f"   ✅ {symbol}: Filled {filled} gaps ({after} remaining)")
            elif after == 0:
                print(f"   ✅ {symbol}: All gaps filled!")
            else:
                print(f"   ⚠️  {symbol}: Still has {after} gaps")

        pipeline.print_summary(backfill_stats)

    except Exception as e:
        print(f"❌ Error during backfill: {e}")
        logging.exception("Backfill failed")

    finally:
        await pipeline.cleanup()


async def targeted_backfill(symbol, days_back):
    """Backfill for a specific symbol with detailed reporting."""

    config = PipelineConfig()
    # Enable more detailed logging for single symbol
    config.defaults["logging"]["level"] = "DEBUG"
    setup_logging(config)

    pipeline = HistoricalDataPipeline(config)

    try:
        print(f"🎯 Starting targeted backfill for {symbol} (last {days_back} days)...")

        await pipeline.initialize()

        # Detailed analysis for this symbol
        print(f"📊 Analyzing data completeness for {symbol}...")
        gap_analysis = await analyze_data_gaps(pipeline, [symbol], days_back)

        symbol_gaps = gap_analysis.get(symbol, {}).get("gaps", 0)

        if symbol_gaps == 0:
            print(f"✅ {symbol} has complete data - no backfill needed!")
            return

        print(f"🔧 Found {symbol_gaps} gaps - proceeding with backfill...")

        # Perform backfill with statistics
        start_time = asyncio.get_event_loop().time()

        stats = await pipeline.backfill_missing_data([symbol], days_back)

        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time

        print(f"\n✅ Targeted backfill completed in {duration:.1f} seconds")

        # Verify results
        print(f"🔍 Verifying results...")
        post_analysis = await analyze_data_gaps(pipeline, [symbol], days_back)
        remaining_gaps = post_analysis.get(symbol, {}).get("gaps", 0)

        if remaining_gaps == 0:
            print(f"🎉 Perfect! All {symbol_gaps} gaps have been filled!")
        else:
            filled = symbol_gaps - remaining_gaps
            print(
                f"📈 Progress: {filled}/{symbol_gaps} gaps filled ({remaining_gaps} remaining)"
            )

        # Show loader performance
        loader_stats = pipeline.loader.get_stats()
        if loader_stats["completed_requests"] > 0:
            avg_time_per_request = duration / loader_stats["completed_requests"]
            print(f"⚡ Performance: {avg_time_per_request:.2f}s per API request")

        pipeline.print_summary(stats)

    except Exception as e:
        print(f"❌ Error during targeted backfill: {e}")
        logging.exception("Targeted backfill failed")

    finally:
        await pipeline.cleanup()


async def comprehensive_backfill():
    """Comprehensive backfill for multiple timeframes."""

    config = PipelineConfig()
    setup_logging(config)
    pipeline = HistoricalDataPipeline(config)

    try:
        print("🌟 Starting comprehensive backfill example...")

        await pipeline.initialize()

        symbols = ["SPY", "QQQ"]
        timeframes = [7, 30, 90]  # Different lookback periods

        overall_stats = {
            "total_symbols_checked": 0,
            "total_gaps_found": 0,
            "total_gaps_filled": 0,
        }

        for days_back in timeframes:
            print(f"\n📅 Checking {days_back}-day lookback period...")

            # Analyze gaps for this timeframe
            gap_analysis = await analyze_data_gaps(pipeline, symbols, days_back)

            # Count total gaps
            timeframe_gaps = sum(data.get("gaps", 0) for data in gap_analysis.values())

            if timeframe_gaps > 0:
                print(f"🔧 Backfilling {timeframe_gaps} gaps...")

                # Backfill for this timeframe
                stats = await pipeline.backfill_missing_data(symbols, days_back)

                overall_stats["total_symbols_checked"] += stats.get(
                    "symbols_checked", 0
                )
                overall_stats["total_gaps_found"] += timeframe_gaps

            else:
                print(f"✅ No gaps found for {days_back}-day period")

        print(f"\n🏁 Comprehensive backfill completed!")
        print(f"   Total symbols processed: {overall_stats['total_symbols_checked']}")
        print(f"   Total gaps identified: {overall_stats['total_gaps_found']}")

    except Exception as e:
        print(f"❌ Error during comprehensive backfill: {e}")
        logging.exception("Comprehensive backfill failed")

    finally:
        await pipeline.cleanup()


def main():
    """Main entry point with different backfill modes."""

    parser = argparse.ArgumentParser(
        description="Historical data backfill examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Standard backfill with analysis
  %(prog)s --symbol AAPL --days 60   # Targeted backfill for AAPL
  %(prog)s --mode comprehensive      # Multi-timeframe backfill
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["standard", "targeted", "comprehensive"],
        default="standard",
        help="Backfill mode to run",
    )

    parser.add_argument("--symbol", help="Symbol for targeted backfill")

    parser.add_argument(
        "--days", type=int, default=30, help="Number of days back to check"
    )

    args = parser.parse_args()

    try:
        if args.mode == "standard":
            asyncio.run(backfill_with_analysis())
        elif args.mode == "targeted":
            if not args.symbol:
                print("❌ --symbol required for targeted mode")
                return
            asyncio.run(targeted_backfill(args.symbol.upper(), args.days))
        elif args.mode == "comprehensive":
            asyncio.run(comprehensive_backfill())

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
