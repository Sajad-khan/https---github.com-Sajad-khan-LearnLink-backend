import requests
from bs4 import BeautifulSoup
import json
import os
import time
import re
from urllib.parse import urljoin, urlparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException

# Configure logging - remove emojis to avoid encoding issues on Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
START_URL = "https://nitsri.ac.in/"
ALLOWED_DOMAIN = "nitsri.ac.in"
MAX_DEPTH = 3  # Increased depth
DATA_PATH = "data/scraped_data.json"
MAX_WORKERS = 3  # Number of parallel workers
MAX_PAGES_PER_DOMAIN = 5000  # Increased page limit
REQUEST_TIMEOUT = 15  # Seconds for requests
PAGE_LOAD_TIMEOUT = 15  # Seconds for Selenium

# User agents to rotate
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0'
]

# Global state
visited_urls = set()
url_queue = []
data_count = 0
selenium_driver = None

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def init_selenium():
    """Initialize Selenium WebDriver with optimized settings"""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--log-level=3")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-images")  # Don't load images
    
    # Use a random user agent
    options.add_argument(f"user-agent={get_random_user_agent()}")
    
    # Use eager page load strategy
    options.page_load_strategy = 'eager'
    
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    
    return driver

def clean_text(text):
    """Clean extracted text"""
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_page_title(soup):
    """Extract page title from BeautifulSoup object"""
    if not soup:
        return "No title found"
        
    # Try to get title from title tag
    title_tag = soup.find("title")
    if title_tag and title_tag.text.strip():
        return title_tag.text.strip()
    
    # Try to get from h1
    h1 = soup.find("h1")
    if h1 and h1.text.strip():
        return h1.text.strip()
    
    # Try to get from header
    header = soup.find(["header", "div", "span"], class_=re.compile(r'title|header', re.I))
    if header and header.text.strip():
        return header.text.strip()
    
    return "No title found"

def extract_content_from_html(html, url):
    """Extract content from HTML using BeautifulSoup"""
    if not html:
        return None
        
    soup = BeautifulSoup(html, "html.parser")
    
    # Common content areas in ASP.NET sites
    content_selectors = [
        "#ctl00_ContentPlaceHolder1_pnlContent",
        "#ContentPlaceHolder1_pnlContent", 
        "#ctl00_ContentPlaceHolder1",
        "#ContentPlaceHolder1",
        "#maincontent",
        ".content-area",
        "#content",
        "article",
        ".panel-body",
        "main",
        "#mainContent",
        ".main-content"
    ]
    
    # Try each selector
    for selector in content_selectors:
        content_area = soup.select_one(selector)
        if content_area:
            # Remove non-content elements
            for tag in content_area.select('script, style, iframe'):
                tag.decompose()
                
            text = content_area.get_text(separator=" ", strip=True)
            if text and len(text.split()) > 10:
                return {
                    "url": url,
                    "title": get_page_title(soup),
                    "content": clean_text(text),
                    "source": selector
                }
    
    # If no content area found, try whole page with filtering
    # Remove navigation, scripts, etc.
    for tag in soup.select('script, style, header, footer, nav, .navigation, .menu, .sidebar, aside'):
        tag.decompose()
        
    # Get the main content
    body = soup.select_one('body') or soup
    text = body.get_text(separator=" ", strip=True)
    
    # Only use if it has substantial content
    if text and len(text.split()) > 10:
        return {
            "url": url,
            "title": get_page_title(soup),
            "content": clean_text(text),
            "source": "body"
        }
        
    return None

def extract_links(html, base_url):
    """Extract links from HTML content"""
    links = []
    try:
        soup = BeautifulSoup(html, "html.parser")
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href')
            if href and not href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                # Handle ASP.NET JavaScript links
                if "__doPostBack" in href:
                    continue  # Skip these for simplicity
                
                # Build absolute URL
                absolute_url = urljoin(base_url, href)
                
                # Filter by domain and file types
                parsed = urlparse(absolute_url)
                if parsed.netloc in ("", ALLOWED_DOMAIN, f"www.{ALLOWED_DOMAIN}") and not absolute_url.lower().endswith((
                    '.pdf', '.doc', '.docx', '.jpg', '.jpeg', '.png', '.gif', '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.rar'
                )):
                    links.append(absolute_url)
    except Exception as e:
        logger.error(f"Error extracting links from {base_url}: {e}")
    
    return links

def get_url_id(url):
    """Create consistent identifier for URLs"""
    parsed = urlparse(url)
    path = parsed.path.lower().rstrip('/')
    if not path:
        path = 'home'
    
    # Normalize ASP.NET paths and query strings
    path = re.sub(r'\.aspx$', '', path)
    
    # Include query parameters for ASP.NET pages which often use them for navigation
    if parsed.query:
        # Sort query params for consistency
        query_items = sorted(parsed.query.split('&'))
        path = f"{path}?{'&'.join(query_items)}"
    
    return f"{parsed.netloc}{path}"

def save_data(data_item):
    """Save extracted data to file"""
    global data_count
    
    if not data_item:
        return
        
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    with open(DATA_PATH, "a", encoding="utf-8") as f:
        json.dump(data_item, f, ensure_ascii=False)
        f.write("\n")
    
    data_count += 1
    logger.info(f"Saved item #{data_count}: {data_item['title']}")

def fetch_url_with_requests(url):
    """Fetch URL using requests library"""
    try:
        headers = {
            'User-Agent': get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response.text
        else:
            logger.warning(f"Failed to fetch {url} with status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching {url} with requests: {e}")
        return None

def fetch_url_with_selenium(url):
    """Fetch URL using Selenium as fallback"""
    global selenium_driver
    
    if selenium_driver is None:
        try:
            selenium_driver = init_selenium()
        except Exception as e:
            logger.error(f"Failed to initialize Selenium: {e}")
            return None
    
    try:
        selenium_driver.get(url)
        time.sleep(2)  # Brief pause
        return selenium_driver.page_source
    except TimeoutException:
        logger.warning(f"Selenium timed out for {url}")
        return None
    except Exception as e:
        logger.error(f"Selenium error for {url}: {e}")
        return None

def process_url(url, depth):
    """Process a single URL with multiple fallback methods"""
    url_id = get_url_id(url)
    
    if url_id in visited_urls:
        return []
    
    visited_urls.add(url_id)
    logger.info(f"Visiting [{depth}/{MAX_DEPTH}]: {url}")
    
    # Try with requests first (faster)
    html_content = fetch_url_with_requests(url)
    
    # If requests fails, try with Selenium
    if not html_content:
        logger.info(f"Falling back to Selenium for {url}")
        html_content = fetch_url_with_selenium(url)
    
    if not html_content:
        logger.warning(f"Failed to fetch {url} with all methods")
        return []
    
    # Extract and save content
    content_data = extract_content_from_html(html_content, url)
    if content_data:
        preview = content_data["content"][:100].replace("\n", " ")
        logger.info(f"Found content: {content_data['title']}: {preview}...")
        save_data(content_data)
    
    # Stop at max depth
    if depth >= MAX_DEPTH:
        return []
    
    # Extract links for next level
    links = extract_links(html_content, url)
    
    # Return discovered links along with their depth
    return [(link, depth + 1) for link in links]

def main():
    # Clear or create data file
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        pass
    
    global url_queue
    url_queue = [(START_URL, 0)]  # (url, depth)
    
    try:
        logger.info(f"Starting web scraping of {START_URL}")
        logger.info(f"Max depth: {MAX_DEPTH}, Max pages: {MAX_PAGES_PER_DOMAIN}, Saving to: {DATA_PATH}")
        
        processed_count = 0
        
        # Process URLs in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Process URLs until queue is empty or we reach max pages
            while url_queue and processed_count < MAX_PAGES_PER_DOMAIN:
                # Take a batch of URLs to process
                batch = url_queue[:min(MAX_WORKERS, len(url_queue))]
                url_queue = url_queue[min(MAX_WORKERS, len(url_queue)):]
                
                # Submit batch to executor
                future_to_url = {executor.submit(process_url, url, depth): url for url, depth in batch}
                
                for future in as_completed(future_to_url):
                    processed_count += 1
                    # Get new URLs discovered and add to queue
                    new_urls = future.result()
                    for new_url, new_depth in new_urls:
                        if get_url_id(new_url) not in visited_urls:
                            url_queue.append((new_url, new_depth))
                
                # Small delay between batches to be gentle on the server
                time.sleep(1)
                
                # Safety check
                if processed_count >= MAX_PAGES_PER_DOMAIN:
                    logger.warning(f"Reached maximum page limit ({MAX_PAGES_PER_DOMAIN})")
                    break
                
                # Progress report every 50 pages
                if processed_count % 50 == 0:
                    logger.info(f"Progress: {processed_count} pages processed, {data_count} content items saved, {len(url_queue)} URLs in queue")
                
    except KeyboardInterrupt:
        logger.info("Scraping stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        # Clean up Selenium driver if used
        if selenium_driver:
            logger.info("Closing Selenium driver")
            selenium_driver.quit()
        
        logger.info(f"Done. Processed {processed_count} pages. Saved {data_count} content items to {DATA_PATH}")

if __name__ == "__main__":
    main()