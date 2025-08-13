#!/usr/bin/env python3
"""
Python wrapper for historical data collection
This bypasses shell script environment issues
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import asyncpg

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
os.chdir(PROJECT_ROOT)

# Set up environment
os.environ["PYTHONPATH"] = f"{PROJECT_ROOT}:{os.environ.get('PYTHONPATH', '')}"
os.environ["PYTHONUNBUFFERED"] = "1"


def log_message(msg):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


async def verify_collection_success(
    collection_type: str, symbols: list
) -> tuple[bool, dict]:
    """
    Verify collection success by checking database records.
    Returns (success: bool, details: dict)
    """
    try:
        conn = await asyncpg.connect(
            "postgresql://trading_user:secure_trading_password@localhost:5433/options_arbitrage"
        )

        # Check the most recent collection run
        run_record = await conn.fetchrow(
            """
            SELECT id, status, contracts_requested, contracts_successful,
                   bars_collected, bars_updated, bars_skipped, errors,
                   completed_at, error_details
            FROM intraday_collection_runs
            WHERE run_type = $1
            AND run_date = CURRENT_DATE
            ORDER BY started_at DESC
            LIMIT 1
            """,
            collection_type,
        )

        await conn.close()

        if not run_record:
            return False, {"error": "No collection run found in database"}

        # Determine success based on database status and data collected
        db_status = run_record["status"]
        contracts_successful = run_record["contracts_successful"] or 0
        bars_collected = run_record["bars_collected"] or 0
        errors = run_record["errors"] or 0

        details = {
            "db_status": db_status,
            "contracts_successful": contracts_successful,
            "bars_collected": bars_collected,
            "bars_updated": run_record["bars_updated"] or 0,
            "bars_skipped": run_record["bars_skipped"] or 0,
            "errors": errors,
            "completed_at": run_record["completed_at"],
            "error_details": run_record["error_details"],
        }

        # Collection is successful if:
        # 1. Database status is 'success' OR 'partial'
        # 2. Some data was collected (bars_collected > 0)
        # 3. Not too many errors relative to successful operations
        is_success = (
            db_status in ["success", "partial"]
            and bars_collected > 0
            and (errors == 0 or contracts_successful > errors)
        )

        return is_success, details

    except Exception as e:
        return False, {"error": f"Database verification failed: {str(e)}"}


def classify_collection_result(
    exit_code: int, db_verified: bool, db_details: dict
) -> tuple[str, str, str]:
    """
    Classify collection result into status, message, and CSS class.
    Returns (status_text, status_class, detailed_message)
    """
    if db_verified:
        if db_details.get("db_status") == "success":
            return "✅ SUCCESS", "success", "Collection completed successfully"
        elif db_details.get("db_status") == "partial":
            bars_collected = db_details.get("bars_collected", 0)
            errors = db_details.get("errors", 0)
            return (
                "⚠️ PARTIAL",
                "warning",
                f"Partial success: {bars_collected} bars collected, {errors} errors",
            )
        else:
            return "❌ FAILED", "error", "Collection failed - no data collected"
    else:
        if exit_code == 0:
            return (
                "✅ SUCCESS",
                "success",
                "Process completed successfully (unable to verify database)",
            )
        else:
            error_msg = db_details.get("error", f"Exit code {exit_code}")
            return "❌ FAILED", "error", f"Process failed: {error_msg}"


def create_stats_html(
    collection_type,
    symbols,
    duration,
    start_time,
    end_time,
    exit_code,
    error_message=None,
    db_details=None,
    status_text=None,
    status_class=None,
):
    """Create HTML stats page with enhanced database details"""
    duration_seconds = int((end_time - start_time).total_seconds())

    # Use provided status or fall back to simple exit code logic
    if status_text is None or status_class is None:
        if exit_code == 0:
            status_text = "✅ SUCCESS"
            status_class = "success"
        else:
            status_text = "❌ FAILED"
            status_class = "error"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Collection Stats - {collection_type}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #007acc; padding-bottom: 10px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007acc; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #007acc; }}
        .stat-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .success {{ border-left-color: #28a745; }} .success .stat-value {{ color: #28a745; }}
        .warning {{ border-left-color: #ffc107; }} .warning .stat-value {{ color: #ffc107; }}
        .error {{ border-left-color: #dc3545; }} .error .stat-value {{ color: #dc3545; }}
        .details {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px; }}
        .refresh-btn {{ background: #007acc; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Collection Statistics</h1>

        <div class="stats-grid">
            <div class="stat-card success">
                <div class="stat-value">{collection_type}</div>
                <div class="stat-label">Collection Type</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{duration_seconds}s</div>
                <div class="stat-label">Duration</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{' '.join(symbols)}</div>
                <div class="stat-label">Symbols</div>
            </div>
            <div class="stat-card {status_class}">
                <div class="stat-value">{status_text}</div>
                <div class="stat-label">Status</div>
            </div>
        </div>

        <div class="details">
            <h3>📋 Collection Details</h3>
            <p><strong>Started:</strong> {start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Completed:</strong> {end_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Duration Requested:</strong> {duration}</p>
            <p><strong>Method:</strong> Python Wrapper (bypasses shell script issues)</p>
            <p><strong>Process Exit Code:</strong> {exit_code}</p>
            {f'<p><strong>Process Error:</strong> <code>{error_message}</code></p>' if error_message else ''}

            {f'''
            <h3>📊 Database Collection Results</h3>
            <p><strong>Database Status:</strong> {db_details.get('db_status', 'Unknown')}</p>
            <p><strong>Contracts Successful:</strong> {db_details.get('contracts_successful', 'N/A')}</p>
            <p><strong>Bars Collected:</strong> {db_details.get('bars_collected', 'N/A')}</p>
            <p><strong>Bars Updated:</strong> {db_details.get('bars_updated', 'N/A')}</p>
            <p><strong>Bars Skipped:</strong> {db_details.get('bars_skipped', 'N/A')}</p>
            <p><strong>Errors:</strong> {db_details.get('errors', 'N/A')}</p>
            ''' + (f'<p><strong>Database Completed:</strong> {db_details["completed_at"]}</p>' if db_details.get("completed_at") else '') + '''
            ''' if db_details and 'error' not in db_details else ''}

            {f'<p><strong>Database Error:</strong> <code>{db_details["error"]}</code></p>' if db_details and 'error' in db_details else ''}
        </div>

        <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>

        <script>
            // Auto-refresh every 30 seconds
            setTimeout(() => location.reload(), 30000);
        </script>
    </div>
</body>
</html>"""

    return html_content


def open_stats_page(
    collection_type,
    symbols,
    duration,
    start_time,
    end_time,
    exit_code,
    error_message=None,
    db_details=None,
    status_text=None,
    status_class=None,
):
    """Create and open HTML stats page"""
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = logs_dir / f"collection_stats_{timestamp}.html"

    # Create HTML content
    html_content = create_stats_html(
        collection_type,
        symbols,
        duration,
        start_time,
        end_time,
        exit_code,
        error_message,
        db_details,
        status_text,
        status_class,
    )

    # Write to file
    with open(stats_file, "w") as f:
        f.write(html_content)

    log_message(f"Opening stats page: {stats_file}")

    # Open in default browser
    try:
        subprocess.run(["open", str(stats_file)], check=True)
        log_message("✅ Stats page opened successfully")
    except subprocess.CalledProcessError as e:
        log_message(f"⚠️ Failed to open stats page: {e}")

    return stats_file


async def main():
    collection_type = sys.argv[1] if len(sys.argv) > 1 else "manual"

    # Collection-specific defaults
    defaults = {
        "morning": {"symbols": ["SPY", "QQQ"], "duration": "1800 S"},
        "midday": {"symbols": ["SPY"], "duration": "10800 S"},
        "afternoon": {"symbols": ["SPY", "QQQ", "TSLA", "AAPL"], "duration": "18000 S"},
        "eod": {"symbols": ["SPY", "QQQ", "TSLA", "AAPL", "MSFT"], "duration": "1 D"},
        "late_night": {"symbols": ["SPY"], "duration": "1 D"},
        "gap_fill": {"symbols": ["SPY"], "duration": "1 D"},
        "manual": {"symbols": ["SPY"], "duration": "1 D"},
    }

    config = defaults.get(collection_type, defaults["manual"])
    symbols = config["symbols"]
    duration = config["duration"]

    log_message("===========================================")
    log_message("Python Historical Collection Wrapper")
    log_message(f"Collection Type: {collection_type}")
    log_message(f"Symbols: {' '.join(symbols)}")
    log_message(f"Duration: {duration}")
    log_message("===========================================")

    # Build Python command
    cmd = (
        [
            "python",
            "backtesting/infra/data_collection/historical_bars_collector.py",
            "--type",
            collection_type,
            "--symbols",
        ]
        + symbols
        + [
            "--duration",
            duration,
            "--connection-timeout",
            "60",
            "--connection-retries",
            "3",
            "--verbose",
        ]
    )

    log_message(f"Executing: {' '.join(cmd)}")
    log_message("")

    # Track execution time
    start_time = datetime.now()
    error_message = None

    # Execute with proper environment
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            text=True,
            capture_output=False,  # Let output stream directly
        )

        end_time = datetime.now()

        if result.returncode == 0:
            log_message("✅ Collection process completed successfully")
        else:
            log_message(
                f"⚠️ Collection process ended with exit code {result.returncode}"
            )
            error_message = f"Exit code {result.returncode}"

        # Wait a moment for async cleanup to complete
        log_message("⏳ Waiting for database operations to complete...")
        await asyncio.sleep(5)  # Give async cleanup time to finish

        # Verify collection success via database
        log_message("🔍 Verifying collection results in database...")
        db_verified, db_details = await verify_collection_success(
            collection_type, symbols
        )

        # Classify the overall result
        status_text, status_class, detailed_message = classify_collection_result(
            result.returncode, db_verified, db_details
        )

        log_message(f"📊 Collection result: {detailed_message}")

        # Always open stats page on completion
        log_message("")
        log_message("Opening collection statistics page...")
        open_stats_page(
            collection_type=collection_type,
            symbols=symbols,
            duration=duration,
            start_time=start_time,
            end_time=end_time,
            exit_code=result.returncode,
            error_message=error_message,
            db_details=db_details,
            status_text=status_text,
            status_class=status_class,
        )

        return result.returncode

    except Exception as e:
        end_time = datetime.now()
        error_message = f"Exception: {str(e)}"
        log_message(f"❌ Exception during execution: {e}")

        # Try to get database details even on exception
        try:
            log_message("🔍 Attempting to verify database status despite exception...")
            db_verified, db_details = await verify_collection_success(
                collection_type, symbols
            )
            status_text, status_class, detailed_message = classify_collection_result(
                1, db_verified, db_details
            )
        except:
            db_details = {"error": f"Database verification failed after exception: {e}"}
            status_text, status_class, detailed_message = (
                "❌ FAILED",
                "error",
                f"Process exception: {e}",
            )

        # Open stats page even on exception
        log_message("Opening error statistics page...")
        open_stats_page(
            collection_type=collection_type,
            symbols=symbols,
            duration=duration,
            start_time=start_time,
            end_time=end_time,
            exit_code=1,
            error_message=error_message,
            db_details=db_details,
            status_text=status_text,
            status_class=status_class,
        )

        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
