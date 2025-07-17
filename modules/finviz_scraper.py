import re
import time
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


def parse_screener_total(driver) -> Optional[int]:
    """
    Parse the total number of results from the screener-total div.

    Args:
        driver: Selenium WebDriver instance

    Returns:
        Total number of results or None if parsing fails
    """
    try:
        total_element = driver.find_element(By.ID, "screener-total")
        total_text = total_element.text.strip()
        logger.debug(f"Found screener-total text: {total_text}")

        # Extract number from format like "#1 / 117 Total"
        match = re.search(r"#\d+\s*/\s*(\d+)\s*Total", total_text)
        if match:
            total = int(match.group(1))
            logger.info(f"Found total results: {total}")
            return total
        else:
            logger.warning(f"Could not parse total from: {total_text}")
            return None
    except Exception as e:
        logger.debug(f"Could not find or parse screener-total: {e}")
        return None


def check_and_navigate_pagination(driver, current_page: int) -> bool:
    """
    Check if pagination exists and navigate to the next page.

    Args:
        driver: Selenium WebDriver instance
        current_page: Current page number (1-based)

    Returns:
        True if successfully navigated to next page, False if no more pages
    """
    try:
        # Check if pagination element exists
        pagination_element = driver.find_element(By.ID, "screener_pagination")
        logger.debug(f"Found pagination element on page {current_page}")

        # Look for next page link - typically the last clickable link that's not "..."
        # or look for a link containing the next page number
        next_page_num = current_page + 1

        # Try to find a link with the next page number
        pagination_links = pagination_element.find_elements(By.TAG_NAME, "a")

        for link in pagination_links:
            link_text = link.text.strip()
            if link_text.isdigit() and int(link_text) == next_page_num:
                logger.info(f"Navigating to page {next_page_num}")
                driver.execute_script("arguments[0].click();", link)

                # Wait for page to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "screener-table"))
                )
                return True

        # If no specific page number found, try looking for "next" type links
        # This is a fallback for different pagination styles
        for link in pagination_links:
            link_text = link.text.strip().lower()
            if "next" in link_text or ">" in link_text:
                logger.info(f"Navigating to next page using '{link_text}' link")
                driver.execute_script("arguments[0].click();", link)

                # Wait for page to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "screener-table"))
                )
                return True

        logger.debug(f"No more pages found after page {current_page}")
        return False

    except Exception as e:
        logger.debug(f"No pagination found or error navigating: {e}")
        return False


def scrape_tickers_from_current_page(driver) -> List[str]:
    """
    Scrape ticker symbols from the current page's screener table.

    Args:
        driver: Selenium WebDriver instance

    Returns:
        List of ticker symbols found on current page
    """
    tickers = []
    try:
        table = driver.find_element(By.ID, "screener-table")
        rows = table.find_elements(By.TAG_NAME, "tr")

        for row in rows[1:]:  # Skip header row
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) >= 2:
                ticker = cells[1].text.strip()
                if ticker and ticker.isalpha():
                    tickers.append(ticker)

        logger.debug(f"Found {len(tickers)} tickers on current page: {tickers}")
        return tickers

    except Exception as e:
        logger.error(f"Error scraping tickers from current page: {e}")
        return []


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


def scrape_tickers_from_finviz(
    url: str, timeout: int = 15, max_pages: int = None
) -> Optional[List[str]]:
    """
    Scrape ticker symbols from Finviz screener table with pagination support.

    Args:
        url: Finviz screener URL (may contain escaped characters)
        timeout: Maximum time to wait for page load (seconds)
        max_pages: Maximum number of pages to scrape (None for unlimited)

    Returns:
        List of ticker symbols from all pages or None if scraping fails
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
        wait.until(EC.presence_of_element_located((By.ID, "screener-table")))

        # Parse total count for validation
        expected_total = parse_screener_total(driver)
        if expected_total:
            logger.info(f"Expected total results: {expected_total}")

        all_tickers = []
        current_page = 1
        max_iterations = 100  # Safety limit to prevent infinite loops

        while current_page <= max_iterations:
            # Check if we've hit the max pages limit
            if max_pages and current_page > max_pages:
                logger.info(f"Reached max pages limit ({max_pages}), stopping")
                break

            logger.info(f"Scraping page {current_page}...")

            try:
                # Scrape tickers from current page
                page_tickers = scrape_tickers_from_current_page(driver)

                if page_tickers:
                    all_tickers.extend(page_tickers)
                    logger.info(
                        f"Page {current_page}: Found {len(page_tickers)} tickers (Total so far: {len(all_tickers)})"
                    )
                else:
                    logger.warning(f"No tickers found on page {current_page}")

                # Try to navigate to next page
                if not check_and_navigate_pagination(driver, current_page):
                    logger.info(f"No more pages after page {current_page}")
                    break

            except Exception as e:
                logger.error(f"Error processing page {current_page}: {e}")
                # Continue to next page even if current page fails
                if not check_and_navigate_pagination(driver, current_page):
                    logger.info(f"No more pages after failed page {current_page}")
                    break

            current_page += 1

            # Add a small delay between page loads
            time.sleep(1)

        # Check if we hit the safety limit
        if current_page > max_iterations:
            logger.warning(
                f"Hit safety limit of {max_iterations} pages, stopping pagination"
            )

        # Validation
        if expected_total and len(all_tickers) != expected_total:
            logger.warning(
                f"Scraped {len(all_tickers)} tickers but expected {expected_total}"
            )

        logger.info(
            f"Successfully scraped {len(all_tickers)} tickers from {current_page} page(s)"
        )
        logger.debug(f"All scraped tickers: {all_tickers}")

        return all_tickers if all_tickers else None

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
