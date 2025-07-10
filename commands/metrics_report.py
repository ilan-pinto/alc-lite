"""
Metrics reporting command for generating performance reports.
"""

import sys
from pathlib import Path

import argparse

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from modules.Arbitrage.metrics import metrics_collector


def generate_metrics_report(export_format: str = "json", filename: str = None) -> None:
    """Generate and export metrics report"""

    if export_format.lower() == "json":
        output_file = metrics_collector.export_to_json(filename)
        print(f"Metrics exported to: {output_file}")

    # Always print summary to console
    metrics_collector.print_summary()


def main():
    """Main function for metrics reporting command"""
    parser = argparse.ArgumentParser(
        description="Generate performance metrics report for arbitrage scanner"
    )

    parser.add_argument(
        "--format",
        choices=["json", "console"],
        default="json",
        help="Output format (default: json)",
    )

    parser.add_argument("--output", "-o", type=str, help="Output filename (optional)")

    parser.add_argument(
        "--reset", action="store_true", help="Reset metrics after generating report"
    )

    args = parser.parse_args()

    # Generate report
    generate_metrics_report(args.format, args.output)

    # Reset if requested
    if args.reset:
        metrics_collector.reset_session()
        print("Metrics have been reset.")


if __name__ == "__main__":
    main()
