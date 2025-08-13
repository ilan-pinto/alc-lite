#!/usr/bin/env python3
"""
Python wrapper for Israel daily options data collection
Bypasses shell script environment issues that cause timeout problems
"""

import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
os.chdir(PROJECT_ROOT)

# Set up environment
os.environ["PYTHONPATH"] = f"{PROJECT_ROOT}:{os.environ.get('PYTHONPATH', '')}"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TZ"] = "Asia/Jerusalem"


def log_message(msg):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def create_stats_html(
    collection_type,
    symbols,
    force,
    truncate,
    verbose,
    start_time,
    end_time,
    exit_code,
    error_message=None,
):
    """Create HTML stats page for Israel collection"""
    duration_seconds = int((end_time - start_time).total_seconds())

    # Determine status and styling
    if exit_code == 0:
        status_text = "✅ SUCCESS"
        status_class = "success"
    else:
        status_text = "❌ FAILED"
        status_class = "error"

    # Format flags
    flags = []
    if force:
        flags.append("--force")
    if truncate:
        flags.append("--truncate")
    if verbose:
        flags.append("--verbose")
    flags_text = " ".join(flags) if flags else "None"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Israel Daily Collection Stats - {collection_type}</title>
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
        .israel-flag {{ background: linear-gradient(to bottom, #0038b8 33%, white 33%, white 66%, #0038b8 66%); height: 20px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🇮🇱 Israel Daily Collection Statistics</h1>
        <div class="israel-flag"></div>

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
                <div class="stat-value">{symbols or "Default"}</div>
                <div class="stat-label">Symbols</div>
            </div>
            <div class="stat-card {status_class}">
                <div class="stat-value">{status_text}</div>
                <div class="stat-label">Status</div>
            </div>
        </div>

        <div class="details">
            <h3>📋 Collection Details</h3>
            <p><strong>Started:</strong> {start_time.strftime('%Y-%m-%d %H:%M:%S')} IDT</p>
            <p><strong>Completed:</strong> {end_time.strftime('%Y-%m-%d %H:%M:%S')} IDT</p>
            <p><strong>Flags:</strong> {flags_text}</p>
            <p><strong>Method:</strong> Python Wrapper (bypasses shell script issues)</p>
            <p><strong>Script:</strong> daily_collector.py</p>
            {f'<p><strong>Error:</strong> <code>{error_message}</code></p>' if error_message else ''}
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
    force,
    truncate,
    verbose,
    start_time,
    end_time,
    exit_code,
    error_message=None,
):
    """Create and open HTML stats page for Israel collection"""
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = logs_dir / f"israel_collection_stats_{timestamp}.html"

    # Create HTML content
    html_content = create_stats_html(
        collection_type,
        symbols,
        force,
        truncate,
        verbose,
        start_time,
        end_time,
        exit_code,
        error_message,
    )

    # Write to file
    with open(stats_file, "w") as f:
        f.write(html_content)

    log_message(f"Opening Israel collection stats page: {stats_file}")

    # Open in default browser
    try:
        subprocess.run(["open", str(stats_file)], check=True)
        log_message("✅ Stats page opened successfully")
    except subprocess.CalledProcessError as e:
        log_message(f"⚠️ Failed to open stats page: {e}")

    return stats_file


def main():
    # Parse command line arguments (same as shell script)
    collection_type = "end_of_day"  # Default
    force = False
    truncate = False
    verbose = False
    symbols = None

    # Parse arguments
    for arg in sys.argv[1:]:
        if arg in ["end_of_day", "friday_expiry_check", "morning_check"]:
            collection_type = arg
        elif arg == "--force":
            force = True
        elif arg == "--truncate":
            truncate = True
        elif arg == "--verbose":
            verbose = True
        elif arg.startswith("--symbols="):
            symbols = arg.split("=", 1)[1]
        elif arg == "--help":
            print("Python wrapper for Israel daily options data collection")
            print("")
            print(
                "Usage: python scheduler/run_israel_collection_python.py [collection_type] [options]"
            )
            print("")
            print("Collection types:")
            print("  end_of_day         - Primary daily collection (default)")
            print("  friday_expiry_check - Friday expiry data collection")
            print("  morning_check      - Morning health check")
            print("")
            print("Options:")
            print("  --force           - Force collection even if already done")
            print(
                "  --truncate        - Delete today's data before collection (use with --force)"
            )
            print("  --verbose         - Enable verbose logging")
            print("  --symbols=X,Y,Z   - Override symbols (comma-separated)")
            print("  --help            - Show this help message")
            print("")
            print("Examples:")
            print("  python scheduler/run_israel_collection_python.py")
            print(
                "  python scheduler/run_israel_collection_python.py friday_expiry_check"
            )
            print(
                "  python scheduler/run_israel_collection_python.py end_of_day --force"
            )
            print(
                "  python scheduler/run_israel_collection_python.py --force --truncate"
            )
            print(
                "  python scheduler/run_israel_collection_python.py end_of_day --symbols=SPY"
            )
            return 0

    log_message("==========================================")
    log_message("Israel Daily Collection - Python Wrapper")
    log_message("==========================================")

    # Display current times (like shell script)
    israel_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S IDT")
    et_time = datetime.now().astimezone().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    log_message(f"Current Israel Time: {israel_time}")
    log_message(f"Current ET Time: {et_time}")
    log_message(f"Collection Type: {collection_type}")

    if force:
        log_message("Force mode: Will run even if already collected today")
    if truncate:
        log_message("Truncate mode: Will delete today's data before collection")
    if verbose:
        log_message("Verbose mode: Detailed logging enabled")
    if symbols:
        log_message(f"Custom symbols: {symbols}")

    log_message("==========================================")

    # Build command
    cmd = [
        "python",
        "backtesting/infra/data_collection/daily_collector.py",
        "--config",
        "scheduler/daily_schedule_israel.yaml",
        "--type",
        collection_type,
    ]

    if force:
        cmd.append("--force")
    if truncate:
        cmd.append("--truncate")
    if verbose:
        cmd.append("--verbose")
    if symbols:
        # Convert comma-separated to space-separated
        symbols_list = symbols.replace(",", " ").split()
        cmd.extend(["--symbols"] + symbols_list)

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
            log_message("✅ Israel collection completed successfully")
        else:
            log_message(
                f"❌ Israel collection failed with exit code {result.returncode}"
            )
            error_message = f"Exit code {result.returncode}"

        # Always open stats page on completion
        log_message("")
        log_message("Opening Israel collection statistics page...")
        open_stats_page(
            collection_type=collection_type,
            symbols=symbols,
            force=force,
            truncate=truncate,
            verbose=verbose,
            start_time=start_time,
            end_time=end_time,
            exit_code=result.returncode,
            error_message=error_message,
        )

        return result.returncode

    except Exception as e:
        end_time = datetime.now()
        error_message = f"Exception: {str(e)}"
        log_message(f"❌ Exception during execution: {e}")

        # Open stats page even on exception
        log_message("Opening error statistics page...")
        open_stats_page(
            collection_type=collection_type,
            symbols=symbols,
            force=force,
            truncate=truncate,
            verbose=verbose,
            start_time=start_time,
            end_time=end_time,
            exit_code=1,
            error_message=error_message,
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())
