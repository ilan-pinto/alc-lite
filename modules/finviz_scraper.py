import re
from typing import List, Optional

import logging
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)


def validate_and_clean_finviz_url(url: str) -> Optional[str]:
    """
    Validate and clean a Finviz URL by removing backslashes and checking format.

    Args:
        url: Raw URL string that may contain escaped characters

    Returns:
        Cleaned URL if valid, None if invalid
    """
    if not url:
        return None

    # Remove backslashes that may have been added by shell escaping
    cleaned_url = url.replace("\\", "")

    # Basic validation - must be a finviz screener URL
    if not cleaned_url.startswith("http"):
        return None

    if "finviz.com" not in cleaned_url:
        logger.error(f"URL must be from finviz.com, got: {cleaned_url}")
        return None

    if "screener.ashx" not in cleaned_url:
        logger.error(f"URL must be a Finviz screener URL, got: {cleaned_url}")
        return None

    logger.debug(f"Cleaned URL: {cleaned_url}")
    return cleaned_url


def scrape_tickers_from_finviz(url: str, timeout: int = 15) -> Optional[List[str]]:
    """
    Scrape ticker symbols from Finviz screener table.

    Args:
        url: Finviz screener URL (may contain escaped characters)
        timeout: Maximum time to wait for page load (seconds)

    Returns:
        List of ticker symbols or None if scraping fails
    """
    # Validate and clean the URL first
    cleaned_url = validate_and_clean_finviz_url(url)
    if not cleaned_url:
        logger.error(f"Invalid Finviz URL provided: {url}")
        return None

    driver = None
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        logger.info(f"Loading Finviz URL: {cleaned_url}")
        driver.get(cleaned_url)

        wait = WebDriverWait(driver, timeout)
        table = wait.until(EC.presence_of_element_located((By.ID, "screener-table")))

        rows = table.find_elements(By.TAG_NAME, "tr")
        tickers = []

        for row in rows[1:]:  # Skip header row
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) >= 2:
                ticker = cells[1].text.strip()
                if ticker and ticker.isalpha():
                    tickers.append(ticker)

        logger.info(f"Successfully scraped {len(tickers)} tickers from Finviz")
        logger.info(f"Scraped tickers: {tickers}")

        return tickers if tickers else None

    except TimeoutException:
        logger.error(f"Timeout waiting for Finviz page to load: {cleaned_url}")
        return None
    except WebDriverException as e:
        logger.error(f"WebDriver error while scraping Finviz: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while scraping Finviz: {e}")
        return None
    finally:
        if driver:
            try:
                driver.quit()
            except Exception as e:
                logger.warning(f"Error closing WebDriver: {e}")
