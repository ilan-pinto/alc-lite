#!/usr/bin/env python3
"""
View email notifications in browser since external email isn't configured
"""

import sys
import tempfile
import webbrowser
from pathlib import Path


def view_latest_notification():
    """Open the latest email notification in browser."""
    notifications_dir = Path("logs/email_notifications")

    if not notifications_dir.exists():
        print("❌ No email notifications found")
        return

    # Find the latest notification file
    notification_files = list(notifications_dir.glob("notification_*.txt"))
    if not notification_files:
        print("❌ No notification files found")
        return

    latest_file = max(notification_files, key=lambda x: x.stat().st_mtime)
    print(f"📧 Opening latest notification: {latest_file.name}")

    # Read the notification content
    with open(latest_file, "r") as f:
        content = f.read()

    # Extract the HTML content (after the header)
    lines = content.split("\n")
    html_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("<html>"):
            html_start = i
            break

    if html_start is None:
        print("❌ No HTML content found in notification")
        return

    # Get the HTML content
    html_content = "\n".join(lines[html_start:])

    # Create a temporary HTML file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html_content)
        temp_file = f.name

    # Open in browser
    webbrowser.open(f"file://{temp_file}")
    print(f"✅ Email notification opened in your default browser")
    print(f"📝 Subject: {lines[0].replace('Subject: ', '')}")
    print(
        f"📧 To: {[line for line in lines if line.startswith('To:')][0].replace('To: ', '')}"
    )


if __name__ == "__main__":
    view_latest_notification()
