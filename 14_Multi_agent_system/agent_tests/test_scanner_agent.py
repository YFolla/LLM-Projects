#!/usr/bin/env python3
"""
Test script for the Scanner Agent system.

This script demonstrates how to use the ScannerAgent to fetch and analyze deals
from RSS feeds, providing a clean example of the system's capabilities.
"""

import os
import sys
import json
from typing import List, Optional
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.deals import ScrapedDeal, DealSelection, Opportunity
from agents.scanner_agent import ScannerAgent
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """
    Set up the environment variables and validate configuration.
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        load_dotenv(override=True)
        
        # Validate required environment variables
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            return False
        
        # Set the environment variable for OpenAI
        os.environ['OPENAI_API_KEY'] = api_key
        logger.info("Environment setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up environment: {e}")
        return False

def test_deal_fetching():
    """
    Test the deal fetching functionality from RSS feeds.
    
    Returns:
        List[ScrapedDeal]: List of fetched deals, or empty list if failed
    """
    try:
        logger.info("Testing deal fetching from RSS feeds...")
        deals = ScrapedDeal.fetch(show_progress=True)
        
        logger.info(f"Successfully fetched {len(deals)} deals")
        
        # Display sample deal information
        if deals:
            sample_deal = deals[0]
            logger.info(f"Sample deal: {sample_deal.title}")
            logger.info(f"Category: {sample_deal.category}")
            logger.info(f"URL: {sample_deal.url}")
            logger.info(f"Summary length: {len(sample_deal.summary)} characters")
            logger.info(f"Details length: {len(sample_deal.details)} characters")
        
        return deals
        
    except Exception as e:
        logger.error(f"Error fetching deals: {e}")
        return []

def test_scanner_agent(memory: List[str] = None):
    """
    Test the ScannerAgent functionality with deal analysis.
    
    Args:
        memory (List[str], optional): List of URLs to exclude from processing
        
    Returns:
        Optional[DealSelection]: Selected deals or None if failed
    """
    try:
        logger.info("Testing ScannerAgent functionality...")
        
        # Initialize the scanner agent
        agent = ScannerAgent()
        
        # Use empty memory if none provided
        if memory is None:
            memory = []
        
        # Scan for deals
        result = agent.scan(memory=memory)
        
        if result:
            logger.info(f"ScannerAgent successfully selected {len(result.deals)} deals")
            
            # Display selected deals
            for i, deal in enumerate(result.deals, 1):
                logger.info(f"\nDeal {i}:")
                logger.info(f"  Description: {deal.product_description[:100]}...")
                logger.info(f"  Price: ${deal.price:.2f}")
                logger.info(f"  URL: {deal.url}")
        else:
            logger.warning("ScannerAgent returned no deals")
        
        return result
        
    except Exception as e:
        logger.error(f"Error testing ScannerAgent: {e}")
        return None

def display_deal_analysis(deals: List[ScrapedDeal]):
    """
    Display analysis of the fetched deals.
    
    Args:
        deals (List[ScrapedDeal]): List of deals to analyze
    """
    if not deals:
        logger.info("No deals to analyze")
        return
    
    logger.info("\n" + "="*50)
    logger.info("DEAL ANALYSIS")
    logger.info("="*50)
    
    # Category analysis
    categories = {}
    for deal in deals:
        categories[deal.category] = categories.get(deal.category, 0) + 1
    
    logger.info(f"\nDeals by Category:")
    for category, count in sorted(categories.items()):
        logger.info(f"  {category}: {count} deals")
    
    # Content quality analysis
    deals_with_features = sum(1 for deal in deals if deal.features.strip())
    deals_with_details = sum(1 for deal in deals if deal.details.strip())
    
    logger.info(f"\nContent Quality:")
    logger.info(f"  Deals with features: {deals_with_features}/{len(deals)}")
    logger.info(f"  Deals with details: {deals_with_details}/{len(deals)}")
    
    # Sample deal details
    logger.info(f"\nSample Deal Details:")
    if deals:
        sample = deals[0]
        logger.info(f"  Title: {sample.title}")
        logger.info(f"  Category: {sample.category}")
        logger.info(f"  Summary: {sample.summary[:200]}...")
        if sample.features:
            logger.info(f"  Features: {sample.features[:200]}...")

def save_results(selected_deals: Optional[DealSelection], filename: str = "selected_deals.json"):
    """
    Save the selected deals to a JSON file for further analysis.
    
    Args:
        selected_deals (Optional[DealSelection]): Deals to save
        filename (str): Output filename
    """
    if not selected_deals:
        logger.info("No deals to save")
        return
    
    try:
        # Convert to dictionary for JSON serialization
        deals_data = {
            "deals": [
                {
                    "product_description": deal.product_description,
                    "price": deal.price,
                    "url": deal.url
                }
                for deal in selected_deals.deals
            ],
            "total_deals": len(selected_deals.deals),
            "timestamp": None  # Could add timestamp if needed
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(deals_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def main():
    """
    Main test function that runs all tests in sequence.
    """
    logger.info("Starting Scanner Agent Test Suite")
    logger.info("="*60)
    
    # Setup environment
    if not setup_environment():
        logger.error("Environment setup failed. Exiting.")
        return
    
    # Test 1: Fetch deals from RSS feeds
    logger.info("\n1. Testing deal fetching...")
    deals = test_deal_fetching()
    
    if not deals:
        logger.error("No deals fetched. Cannot continue with scanner agent test.")
        return
    
    # Test 2: Analyze fetched deals
    logger.info("\n2. Analyzing fetched deals...")
    display_deal_analysis(deals)
    
    # Test 3: Test scanner agent with empty memory
    logger.info("\n3. Testing ScannerAgent with empty memory...")
    selected_deals = test_scanner_agent()
    
    if selected_deals:
        # Test 4: Save results
        logger.info("\n4. Saving results...")
        save_results(selected_deals)
        
        # Test 5: Test scanner agent with memory (should return fewer/different deals)
        logger.info("\n5. Testing ScannerAgent with memory...")
        memory_urls = [deal.url for deal in selected_deals.deals[:2]]  # Use first 2 URLs as memory
        logger.info(f"Using {len(memory_urls)} URLs in memory")
        
        selected_deals_with_memory = test_scanner_agent(memory=memory_urls)
        
        if selected_deals_with_memory:
            logger.info(f"With memory: {len(selected_deals_with_memory.deals)} deals selected")
        else:
            logger.info("No new deals found with memory filtering")
    
    logger.info("\n" + "="*60)
    logger.info("Scanner Agent Test Suite Completed")

if __name__ == "__main__":
    main() 