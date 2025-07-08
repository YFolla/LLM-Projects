from pydantic import BaseModel
from typing import List, Dict, Self, Optional
from bs4 import BeautifulSoup
import re
import feedparser
from tqdm import tqdm
import requests
import time
import logging
from urllib.parse import urljoin, urlparse

# Configure logging for debugging network issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RSS feed URLs for deal scraping - focused on electronics and home categories
# These feeds are from DealNews and provide structured deal information
feeds = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/c238/Automotive/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
    "https://www.dealnews.com/c196/Home-Garden/?rss=1",
]

# Configuration constants
MAX_ENTRIES_PER_FEED = 10  # Limit entries per feed to avoid overwhelming the system
REQUEST_TIMEOUT = 15       # Increased timeout for HTTP requests in seconds
RATE_LIMIT_DELAY = 1.0     # Increased delay between requests to be more respectful
MAX_RETRIES = 3            # Maximum number of retry attempts for failed requests
RETRY_DELAY = 2.0          # Delay between retry attempts

def extract(html_snippet: str) -> str:
    """
    Clean up HTML snippet and extract useful text using Beautiful Soup.
    
    This function specifically looks for DealNews snippet divs and extracts
    the clean text content, removing HTML tags and normalizing whitespace.
    
    Args:
        html_snippet (str): Raw HTML content from RSS feed summary
        
    Returns:
        str: Cleaned text with HTML tags removed and whitespace normalized
    """
    if not html_snippet:
        return ""
    
    try:
        soup = BeautifulSoup(html_snippet, 'html.parser')
        snippet_div = soup.find('div', class_='snippet summary')
        
        if snippet_div:
            # Extract text from the specific snippet div
            description = snippet_div.get_text(strip=True)
            # Double-parse to handle any nested HTML entities
            description = BeautifulSoup(description, 'html.parser').get_text()
            # Remove any remaining HTML tags with regex
            description = re.sub('<[^<]+?>', '', description)
            result = description.strip()
        else:
            # Fallback: clean the entire HTML snippet
            result = BeautifulSoup(html_snippet, 'html.parser').get_text(strip=True)
        
        # Normalize whitespace - replace newlines with spaces
        return result.replace('\n', ' ')
    
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {e}")
        # Return original text with basic cleaning as fallback
        return re.sub('<[^<]+?>', '', html_snippet).replace('\n', ' ')

class ScrapedDeal:
    """
    A class to represent a Deal retrieved from an RSS feed.
    
    This class handles the complete lifecycle of deal data:
    1. Parsing RSS feed entries
    2. Extracting deal summaries
    3. Scraping full deal details from the source website
    4. Extracting product features when available
    
    Attributes:
        category (str): Deal category (inferred from feed source)
        title (str): Deal title/headline
        summary (str): Cleaned summary text from RSS
        url (str): Direct link to the deal page
        details (str): Full deal description scraped from the page
        features (str): Product features if available
    """
    
    def __init__(self, entry: Dict[str, str], category: str = "Unknown"):
        """
        Initialize a ScrapedDeal from an RSS feed entry.
        
        Args:
            entry (Dict[str, str]): RSS feed entry containing title, summary, links
            category (str): Deal category for classification
            
        Raises:
            ValueError: If required fields are missing from entry
            requests.RequestException: If unable to fetch deal details
        """
        # Validate required fields
        if not entry.get('title'):
            raise ValueError("Entry must contain a title")
        if not entry.get('links') or not entry['links']:
            raise ValueError("Entry must contain at least one link")
        
        self.category = category
        self.title = entry['title']
        self.summary = extract(entry.get('summary', ''))
        self.url = entry['links'][0]['href']
        self.details = ""
        self.features = ""
        
        # Scrape full deal details from the source page
        self._scrape_deal_details()

    def _scrape_deal_details(self) -> None:
        """
        Scrape detailed information from the deal's source page with retry logic.
        
        This method fetches the full page content and extracts:
        - Detailed deal description
        - Product features (if available)
        
        The method handles network errors gracefully with exponential backoff
        and falls back to using the RSS summary if all attempts fail.
        """
        # If we already have a good summary, we might not need to scrape
        if len(self.summary) > 100:  # If summary is detailed enough
            self.details = self.summary
            logger.info(f"Using RSS summary for {self.url} (sufficient detail)")
            return
            
        for attempt in range(MAX_RETRIES):
            try:
                # Validate URL format
                parsed_url = urlparse(self.url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    logger.warning(f"Invalid URL format: {self.url}")
                    self.details = self.summary
                    return
                    
                # Enhanced headers to appear more like a real browser
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
                
                # Create a session for connection pooling
                session = requests.Session()
                session.headers.update(headers)
                
                # Fetch page content with increased timeout
                response = session.get(
                    self.url, 
                    timeout=REQUEST_TIMEOUT,
                    allow_redirects=True
                )
                response.raise_for_status()  # Raise an exception for bad status codes
                
                # Parse the HTML content
                soup = BeautifulSoup(response.content, 'html.parser')
                content_div = soup.find('div', class_='content-section')
                
                if content_div:
                    content = content_div.get_text()
                    # Clean up the content
                    content = content.replace('\nmore', '').replace('\n', ' ')
                    
                    # Split into details and features if features section exists
                    if "Features" in content:
                        self.details, self.features = content.split("Features", 1)
                    else:
                        self.details = content
                        self.features = ""
                    
                    logger.info(f"Successfully scraped details for {self.url}")
                    return  # Success - exit retry loop
                    
                else:
                    # Fallback to summary if content section not found
                    logger.warning(f"Content section not found for {self.url}, using summary")
                    self.details = self.summary
                    return
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{MAX_RETRIES} for {self.url}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    logger.error(f"All retry attempts failed for {self.url} - using summary")
                    self.details = self.summary
                    
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error on attempt {attempt + 1}/{MAX_RETRIES} for {self.url}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    logger.error(f"Connection failed after all retries for {self.url} - using summary")
                    self.details = self.summary
                    
            except requests.RequestException as e:
                logger.warning(f"Request error on attempt {attempt + 1}/{MAX_RETRIES} for {self.url}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    logger.error(f"Request failed after all retries for {self.url} - using summary")
                    self.details = self.summary
                    
            except Exception as e:
                logger.error(f"Unexpected error scraping {self.url}: {e}")
                self.details = self.summary  # Fallback to RSS summary
                return

    def __repr__(self) -> str:
        """
        Return a concise string representation of the deal.
        
        Returns:
            str: Deal title wrapped in angle brackets
        """
        return f"<{self.title}>"

    def describe(self) -> str:
        """
        Return a comprehensive description of the deal for model processing.
        
        This method formats all deal information into a structured string
        that can be easily processed by language models for analysis.
        
        Returns:
            str: Formatted deal description with all available information
        """
        return (
            f"Title: {self.title}\n"
            f"Details: {self.details.strip()}\n"
            f"Features: {self.features.strip()}\n"
            f"URL: {self.url}"
        )

    @classmethod
    def fetch(cls, show_progress: bool = False) -> List[Self]:
        """
        Retrieve all deals from the configured RSS feeds.
        
        This method processes multiple RSS feeds in sequence, with rate limiting
        to be respectful to the source servers. It handles network errors
        gracefully and continues processing other feeds if one fails.
        
        Args:
            show_progress (bool): Whether to show a progress bar during fetching
            
        Returns:
            List[Self]: List of successfully scraped deals
        """
        deals = []
        feed_iter = tqdm(feeds, desc="Processing feeds") if show_progress else feeds
        
        for feed_url in feed_iter:
            try:
                logger.info(f"Processing feed: {feed_url}")
                
                # Parse RSS feed with timeout
                feed = feedparser.parse(feed_url)
                
                # Check if feed was parsed successfully
                if hasattr(feed, 'bozo') and feed.bozo:
                    logger.warning(f"Feed parsing issues for {feed_url}: {feed.bozo_exception}")
                
                if not hasattr(feed, 'entries') or not feed.entries:
                    logger.warning(f"No entries found in feed: {feed_url}")
                    continue
                
                # Extract category from feed URL for classification
                category = cls._extract_category_from_url(feed_url)
                
                # Process limited number of entries per feed
                entries_to_process = feed.entries[:MAX_ENTRIES_PER_FEED]
                
                if show_progress:
                    entries_iter = tqdm(entries_to_process, desc=f"Processing {category}", leave=False)
                else:
                    entries_iter = entries_to_process
                
                for entry in entries_iter:
                    try:
                        deal = cls(entry, category)
                        deals.append(deal)
                        
                        # Rate limiting to be respectful to servers
                        time.sleep(RATE_LIMIT_DELAY)
                        
                    except Exception as e:
                        logger.error(f"Error processing entry '{entry.get('title', 'Unknown')}': {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error processing feed {feed_url}: {e}")
                continue
        
        logger.info(f"Successfully fetched {len(deals)} deals from {len(feeds)} feeds")
        return deals

    @staticmethod
    def _extract_category_from_url(url: str) -> str:
        """
        Extract category name from RSS feed URL.
        
        Args:
            url (str): RSS feed URL
            
        Returns:
            str: Extracted category name or "Unknown" if not found
        """
        # Simple regex to extract category from DealNews URLs
        match = re.search(r'/c\d+/([^/?]+)', url)
        if match:
            return match.group(1).replace('-', ' ')
        
        # Try to extract from feed title patterns
        match = re.search(r'/f\d+/([^/?]+)', url)
        if match:
            return match.group(1).replace('-', ' ')
            
        return "Unknown"

class Deal(BaseModel):
    """
    A structured representation of a deal with essential information.
    
    This Pydantic model ensures data validation and provides a clean
    interface for deal information used by other system components.
    
    Attributes:
        product_description (str): Clean description of the product/deal
        price (float): Deal price in USD (must be positive)
        url (str): Direct link to the deal page
    """
    product_description: str
    price: float
    url: str
    
    class Config:
        """Pydantic configuration for the Deal model."""
        # Enable validation on assignment
        validate_assignment = True
        # Use enum values for serialization
        use_enum_values = True

class DealSelection(BaseModel):
    """
    A collection of deals, typically representing filtered or curated results.
    
    This model is used to group related deals together for processing
    or presentation to users.
    
    Attributes:
        deals (List[Deal]): List of Deal objects
    """
    deals: List[Deal]
    
    def __len__(self) -> int:
        """Return the number of deals in this selection."""
        return len(self.deals)
    
    def filter_by_max_price(self, max_price: float) -> 'DealSelection':
        """
        Filter deals by maximum price.
        
        Args:
            max_price (float): Maximum price threshold
            
        Returns:
            DealSelection: New selection with deals under the price threshold
        """
        filtered_deals = [deal for deal in self.deals if deal.price <= max_price]
        return DealSelection(deals=filtered_deals)

class Opportunity(BaseModel):
    """
    A deal opportunity representing potential savings.
    
    This model identifies deals where the actual price is significantly
    lower than the estimated market value, representing good opportunities
    for consumers.
    
    Attributes:
        deal (Deal): The underlying deal information
        estimate (float): Estimated market value of the product
        discount (float): Calculated discount percentage (0-100)
    """
    deal: Deal
    estimate: float
    discount: float
    
    def __post_init__(self):
        """Validate that discount calculation is reasonable."""
        if self.discount < 0 or self.discount > 100:
            raise ValueError("Discount must be between 0 and 100 percent")
    
    @property
    def savings_amount(self) -> float:
        """Calculate the absolute savings amount in dollars."""
        return self.estimate - self.deal.price
    
    def __str__(self) -> str:
        """Return a human-readable description of the opportunity."""
        return (
            f"Deal: {self.deal.product_description[:50]}... "
            f"(${self.deal.price:.2f} vs ${self.estimate:.2f} estimated, "
            f"{self.discount:.1f}% off)"
        )