#!/usr/bin/env python3
"""
Email Notification System for Historical Data Collection
Sends formatted HTML emails with collection statistics and status
Created: 2025-08-12
"""

import asyncio
import json
import os
import smtplib
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import argparse

# Import email modules with fallback handling
try:
    from email import encoders
    from email.mime.base import MIMEBase
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
except ImportError:
    # Fallback - disable SMTP functionality
    MIMEText = None
    MIMEMultipart = None
    MIMEBase = None
    encoders = None

try:
    import asyncpg
except ImportError:
    asyncpg = None

# Email configuration
EMAIL_TO = "pint12@gmail.com"
EMAIL_FROM = "alc-lite@localhost"
SMTP_SERVER = "localhost"
SMTP_PORT = 587

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class CollectionNotifier:
    """Handles email notifications for data collection events."""

    def __init__(self, email_to: str = EMAIL_TO):
        self.email_to = email_to
        self.email_from = EMAIL_FROM

    async def send_success_notification(
        self, collection_type: str, statistics: Dict[str, Any], log_file: str = None
    ):
        """Send success notification with collection statistics."""
        subject = f"✅ Data Collection Success - {collection_type.title()}"

        # Get additional statistics from database
        db_stats = await self._get_database_statistics()

        # Create HTML email content
        html_content = self._create_success_html(collection_type, statistics, db_stats)

        await self._send_email(subject, html_content, is_html=True)

    async def send_failure_notification(
        self,
        collection_type: str,
        error_message: str,
        log_file: str = None,
        statistics: Dict[str, Any] = None,
    ):
        """Send failure notification with error details and logs."""
        subject = f"❌ Data Collection FAILED - {collection_type.title()}"

        # Create HTML email content
        html_content = self._create_failure_html(
            collection_type, error_message, statistics
        )

        # Attach log file if provided
        attachments = []
        if log_file and Path(log_file).exists():
            attachments.append(log_file)

        await self._send_email(
            subject, html_content, is_html=True, attachments=attachments
        )

    def _create_success_html(
        self, collection_type: str, statistics: Dict[str, Any], db_stats: Dict[str, Any]
    ) -> str:
        """Create HTML content for success notification."""

        # Calculate performance metrics
        duration = statistics.get("duration_seconds", 0)
        bars_collected = statistics.get("bars_collected", 0)
        bars_per_second = bars_collected / duration if duration > 0 else 0

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ color: #28a745; font-size: 24px; margin-bottom: 20px; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                .metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px; }}
                .metric-box {{ background-color: #e9ecef; padding: 15px; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 28px; font-weight: bold; color: #007bff; }}
                .metric-label {{ font-size: 14px; color: #6c757d; margin-top: 5px; }}
                .section {{ margin-bottom: 25px; }}
                .section-title {{ font-size: 18px; font-weight: bold; color: #495057; margin-bottom: 10px; }}
                .status-good {{ color: #28a745; font-weight: bold; }}
                .timestamp {{ color: #6c757d; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">✅ Data Collection Completed Successfully</div>

            <div class="summary">
                <strong>Collection Type:</strong> {collection_type.title()}<br>
                <strong>Completed:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S IDT')}<br>
                <strong>Duration:</strong> {duration:.1f} seconds ({duration/60:.1f} minutes)
            </div>

            <div class="section">
                <div class="section-title">Collection Statistics</div>
                <div class="metrics">
                    <div class="metric-box">
                        <div class="metric-value">{statistics.get('contracts_successful', 0)}</div>
                        <div class="metric-label">Contracts Collected</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{bars_collected:,}</div>
                        <div class="metric-label">5-Min Bars</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{bars_per_second:.1f}</div>
                        <div class="metric-label">Bars/Second</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{statistics.get('rate_limit_hits', 0)}</div>
                        <div class="metric-label">Rate Limit Hits</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <div class="section-title">Data Quality</div>
                <ul>
                    <li><span class="status-good">Success Rate:</span> {statistics.get('contracts_successful', 0)}/{statistics.get('contracts_requested', 0)} contracts ({statistics.get('contracts_successful', 0)/max(statistics.get('contracts_requested', 1), 1)*100:.1f}%)</li>
                    <li><span class="status-good">Errors:</span> {statistics.get('errors', 0)}</li>
                    <li><span class="status-good">Bars Skipped:</span> {statistics.get('bars_skipped', 0)} (duplicates)</li>
                </ul>
            </div>

            <div class="section">
                <div class="section-title">Database Status (Today)</div>
                <ul>
                    <li><strong>Total Bars Today:</strong> {db_stats.get('total_bars_today', 0):,}</li>
                    <li><strong>Unique Contracts:</strong> {db_stats.get('contracts_today', 0)}</li>
                    <li><strong>Time Range:</strong> {db_stats.get('first_bar', 'N/A')} to {db_stats.get('last_bar', 'N/A')}</li>
                    <li><strong>Latest Collection:</strong> {db_stats.get('latest_collection', 'N/A')}</li>
                </ul>
            </div>

            <div class="section">
                <div class="section-title">Next Scheduled Collection</div>
                <p>{self._get_next_collection_info()}</p>
            </div>

            <div class="timestamp">
                Generated by ALC-Lite Historical Data Collector<br>
                {datetime.now().strftime('%Y-%m-%d %H:%M:%S IDT')}
            </div>
        </body>
        </html>
        """

        return html

    def _create_failure_html(
        self,
        collection_type: str,
        error_message: str,
        statistics: Dict[str, Any] = None,
    ) -> str:
        """Create HTML content for failure notification."""

        stats = statistics or {}

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ color: #dc3545; font-size: 24px; margin-bottom: 20px; }}
                .summary {{ background-color: #f8d7da; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #f5c6cb; }}
                .error-box {{ background-color: #721c24; color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-family: monospace; }}
                .section {{ margin-bottom: 25px; }}
                .section-title {{ font-size: 18px; font-weight: bold; color: #495057; margin-bottom: 10px; }}
                .status-bad {{ color: #dc3545; font-weight: bold; }}
                .timestamp {{ color: #6c757d; font-size: 12px; }}
                .action-items {{ background-color: #fff3cd; padding: 15px; border-radius: 8px; border: 1px solid #ffeaa7; }}
            </style>
        </head>
        <body>
            <div class="header">❌ Data Collection Failed</div>

            <div class="summary">
                <strong>Collection Type:</strong> {collection_type.title()}<br>
                <strong>Failed At:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S IDT')}<br>
                <strong>Status:</strong> <span class="status-bad">FAILED</span>
            </div>

            <div class="section">
                <div class="section-title">Error Details</div>
                <div class="error-box">
                    {error_message}
                </div>
            </div>

            <div class="section">
                <div class="section-title">Partial Statistics</div>
                <ul>
                    <li><strong>Contracts Requested:</strong> {stats.get('contracts_requested', 0)}</li>
                    <li><strong>Contracts Successful:</strong> {stats.get('contracts_successful', 0)}</li>
                    <li><strong>Bars Collected:</strong> {stats.get('bars_collected', 0)}</li>
                    <li><strong>Errors:</strong> {stats.get('errors', 0)}</li>
                    <li><strong>Rate Limit Hits:</strong> {stats.get('rate_limit_hits', 0)}</li>
                </ul>
            </div>

            <div class="section">
                <div class="section-title">Troubleshooting Steps</div>
                <div class="action-items">
                    <ol>
                        <li><strong>Check IB Gateway:</strong> Ensure TWS/IB Gateway is running on port 7497</li>
                        <li><strong>Check Database:</strong> Verify TimescaleDB is running on port 5433</li>
                        <li><strong>Review Logs:</strong> Check attached log files for detailed error information</li>
                        <li><strong>Network Issues:</strong> Verify internet connectivity and IB server status</li>
                        <li><strong>Rate Limiting:</strong> Wait 5-10 minutes before manual retry if rate limited</li>
                    </ol>
                </div>
            </div>

            <div class="section">
                <div class="section-title">Manual Recovery</div>
                <p><strong>To manually run collection:</strong><br>
                <code>python backtesting/infra/data_collection/historical_bars_collector.py --symbols SPY --duration "1 D" --verbose</code></p>
            </div>

            <div class="section">
                <div class="section-title">Next Retry</div>
                <p>Automatic retry will occur at the next scheduled collection time: {self._get_next_collection_info()}</p>
            </div>

            <div class="timestamp">
                Generated by ALC-Lite Historical Data Collector<br>
                {datetime.now().strftime('%Y-%m-%d %H:%M:%S IDT')}
            </div>
        </body>
        </html>
        """

        return html

    def _get_next_collection_info(self) -> str:
        """Get information about the next scheduled collection."""
        now = datetime.now()
        hour = now.hour

        if hour < 19:
            return "Today at 7:30 PM IDT (Midday Collection)"
        elif hour < 23:
            return "Today at 11:45 PM IDT (End of Day Collection)"
        else:
            return "Tomorrow at 1:00 AM IDT (Late Night Backfill)"

    async def _get_database_statistics(self) -> Dict[str, Any]:
        """Get current database statistics."""
        if not asyncpg:
            return {
                "contracts_today": "N/A",
                "total_bars_today": "N/A",
                "first_bar": "asyncpg not available",
                "last_bar": "Install asyncpg for database stats",
                "latest_collection": "N/A",
            }

        try:
            conn = await asyncpg.connect(
                host="localhost",
                port=5433,
                database="options_arbitrage",
                user="trading_user",
                password="secure_trading_password",
            )

            # Get today's data statistics
            today_stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(DISTINCT contract_id) as contracts,
                    COUNT(*) as total_bars,
                    MIN(time) as first_bar,
                    MAX(time) as last_bar
                FROM option_bars_5min
                WHERE DATE(time) = CURRENT_DATE
            """
            )

            # Get latest collection run
            latest_run = await conn.fetchrow(
                """
                SELECT completed_at, status, run_type
                FROM intraday_collection_runs
                WHERE run_date = CURRENT_DATE
                ORDER BY started_at DESC
                LIMIT 1
            """
            )

            await conn.close()

            return {
                "contracts_today": today_stats["contracts"] or 0,
                "total_bars_today": today_stats["total_bars"] or 0,
                "first_bar": (
                    str(today_stats["first_bar"]) if today_stats["first_bar"] else "N/A"
                ),
                "last_bar": (
                    str(today_stats["last_bar"]) if today_stats["last_bar"] else "N/A"
                ),
                "latest_collection": (
                    f"{latest_run['completed_at']} ({latest_run['run_type']} - {latest_run['status']})"
                    if latest_run and latest_run["completed_at"]
                    else "N/A"
                ),
            }

        except Exception as e:
            return {
                "contracts_today": "Error",
                "total_bars_today": "Error",
                "first_bar": "Database connection failed",
                "last_bar": str(e),
                "latest_collection": "N/A",
            }

    async def _send_email(
        self,
        subject: str,
        content: str,
        is_html: bool = False,
        attachments: List[str] = None,
    ):
        """Send email using multiple delivery methods."""
        attachments = attachments or []

        # Method 1: Try macOS notification center
        if await self._try_macos_notification(subject, content):
            print(f"✅ Notification sent via macOS Notification Center")

        # Method 2: Try system mail command (may not work for external emails)
        if self._try_system_mail(subject, content, is_html, attachments):
            print(f"✅ Email queued via system mail to {self.email_to}")
        else:
            print(f"⚠️ System mail failed - likely needs external SMTP configuration")

        # Method 3: Try simple webhook notification (if configured)
        await self._try_webhook_notification(subject, content)

        # Always log to file as backup
        self._log_email_to_file(subject, content)
        print(f"📝 Email content logged to file as backup")

    def _try_system_mail(
        self, subject: str, content: str, is_html: bool, attachments: List[str]
    ) -> bool:
        """Try to send email using system mail command."""
        try:
            # Build mail command
            mail_cmd = ["mail", "-s", subject]

            # Add HTML content type if needed
            if is_html:
                mail_cmd.extend(["-a", "Content-Type: text/html"])

            # Add attachments
            for attachment in attachments:
                if Path(attachment).exists():
                    mail_cmd.extend(["-A", attachment])

            mail_cmd.append(self.email_to)

            # Send email
            process = subprocess.run(
                mail_cmd, input=content, text=True, capture_output=True, timeout=30
            )

            return process.returncode == 0

        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    async def _send_smtp_email(
        self, subject: str, content: str, is_html: bool, attachments: List[str]
    ):
        """Send email using SMTP (requires SMTP server configuration)."""
        if not MIMEMultipart or not MIMEText:
            raise Exception("Email MIME modules not available - using file fallback")

        msg = MIMEMultipart()
        msg["From"] = self.email_from
        msg["To"] = self.email_to
        msg["Subject"] = subject

        # Add body
        if is_html:
            msg.attach(MIMEText(content, "html"))
        else:
            msg.attach(MIMEText(content, "plain"))

        # Add attachments
        for attachment_path in attachments:
            if Path(attachment_path).exists():
                with open(attachment_path, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())

                if encoders:
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename= {Path(attachment_path).name}",
                    )
                    msg.attach(part)

        # This would require SMTP server configuration
        # For now, just raise an exception to fall back to file logging
        raise Exception("SMTP not configured - using file fallback")

    def _log_email_to_file(self, subject: str, content: str):
        """Log email content to file as fallback when email sending fails."""
        log_dir = Path("logs/email_notifications")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"notification_{timestamp}.txt"

        with open(log_file, "w") as f:
            f.write(f"Subject: {subject}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"To: {self.email_to}\n")
            f.write("=" * 50 + "\n")
            f.write(content)

        print(f"📝 Email content logged to: {log_file}")

    async def _try_macos_notification(self, subject: str, content: str) -> bool:
        """Send macOS system notification."""
        try:
            # Extract plain text from HTML content for notification
            import re

            if "<html>" in content:
                # Simple HTML to text conversion
                text_content = re.sub(r"<[^>]+>", "", content)
                text_content = re.sub(r"\s+", " ", text_content).strip()
                # Limit to first 200 characters for notification
                text_content = (
                    text_content[:200] + "..."
                    if len(text_content) > 200
                    else text_content
                )
            else:
                text_content = content[:200] + "..." if len(content) > 200 else content

            # Send macOS notification
            process = subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'display notification "{text_content}" with title "{subject}" subtitle "ALC-Lite Data Collector"',
                ],
                capture_output=True,
                timeout=10,
            )

            return process.returncode == 0

        except Exception as e:
            print(f"⚠️ macOS notification failed: {e}")
            return False

    async def _try_webhook_notification(self, subject: str, content: str):
        """Try to send notification via webhook (future implementation)."""
        # This could be extended to send to Slack, Discord, Telegram, etc.
        # For now, just create a template for future webhook integration
        webhook_url = os.environ.get("NOTIFICATION_WEBHOOK_URL")

        if webhook_url:
            try:
                # This would require requests library
                # For now, just log that webhook would be attempted
                print(f"🔗 Would send webhook notification to {webhook_url[:50]}...")
            except Exception as e:
                print(f"⚠️ Webhook notification failed: {e}")

        # Log that webhook could be configured
        if not webhook_url:
            webhook_log = Path("logs/webhook_config.txt")
            if not webhook_log.exists():
                with open(webhook_log, "w") as f:
                    f.write("# ALC-Lite Webhook Configuration\n")
                    f.write(
                        "# Set NOTIFICATION_WEBHOOK_URL environment variable to enable webhook notifications\n"
                    )
                    f.write("# Examples:\n")
                    f.write(
                        "# export NOTIFICATION_WEBHOOK_URL='https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'\n"
                    )
                    f.write(
                        "# export NOTIFICATION_WEBHOOK_URL='https://discord.com/api/webhooks/YOUR/DISCORD/WEBHOOK'\n"
                    )
                    f.write(f"# Generated: {datetime.now()}\n")


async def main():
    """Main entry point for notification script."""
    parser = argparse.ArgumentParser(description="Send data collection notifications")
    parser.add_argument(
        "--type",
        required=True,
        choices=["success", "failure"],
        help="Notification type",
    )
    parser.add_argument(
        "--collection-type",
        default="manual",
        help="Collection type (e.g., morning, eod, manual)",
    )
    parser.add_argument("--stats-file", help="JSON file with collection statistics")
    parser.add_argument("--error", help="Error message for failure notifications")
    parser.add_argument("--log-file", help="Log file to attach")
    parser.add_argument("--email-to", default=EMAIL_TO, help="Email recipient")

    args = parser.parse_args()

    notifier = CollectionNotifier(args.email_to)

    # Load statistics if provided
    statistics = {}
    if args.stats_file and Path(args.stats_file).exists():
        with open(args.stats_file, "r") as f:
            statistics = json.load(f)

    if args.type == "success":
        await notifier.send_success_notification(
            args.collection_type, statistics, args.log_file
        )
    elif args.type == "failure":
        await notifier.send_failure_notification(
            args.collection_type,
            args.error or "Collection failed - no error message provided",
            args.log_file,
            statistics,
        )


if __name__ == "__main__":
    asyncio.run(main())
