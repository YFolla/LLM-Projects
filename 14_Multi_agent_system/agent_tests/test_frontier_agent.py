"""
Test script for the Frontier Agent.

This script demonstrates how to use the FrontierAgent with ChromaDB for RAG-based
price estimation using OpenAI's GPT-4o-mini. It includes multiple test cases,
proper ChromaDB initialization, and comprehensive error handling.
"""

import logging
import sys
import time
import os
import chromadb
from dotenv import load_dotenv

# Add parent directory to path to import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.frontier_agent import FrontierAgent

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Load environment variables
load_dotenv(override=True)

# Ensure OpenAI API key is available
if not os.getenv('OPENAI_API_KEY'):
    print("❌ Error: OPENAI_API_KEY not found in environment variables")
    print("   Please set your OpenAI API key in the .env file")
    sys.exit(1)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Configure logging to show agent messages with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

# ============================================================================
# CHROMADB SETUP
# ============================================================================

def setup_chromadb():
    """
    Initialize ChromaDB client and collection for testing.
    
    Returns:
        chromadb.Collection: The products collection for RAG search
    """
    try:
        # Initialize ChromaDB client (go up one directory to find the vectorstore)
        DB_PATH = "../products_vectorstore"
        client = chromadb.PersistentClient(path=DB_PATH)
        
        # Get or create the products collection
        collection = client.get_or_create_collection('products')
        
        # Check if collection has data
        count = collection.count()
        if count == 0:
            print("⚠️  Warning: ChromaDB collection is empty")
            print("   The agent will not have context for RAG search")
            print("   Consider running the vectorization script first")
        else:
            print(f"✅ ChromaDB collection loaded with {count} products")
        
        return collection
        
    except Exception as e:
        print(f"❌ Error setting up ChromaDB: {str(e)}")
        print("   Make sure the products_vectorstore directory exists")
        print("   and contains vectorized product data")
        sys.exit(1)

# ============================================================================
# TEST CASES
# ============================================================================

# Define test cases with various product descriptions
TEST_CASES = [
    {
        "name": "Audio Equipment",
        "description": "Quadcast HyperX condenser mic, connects via usb-c to your computer for crystal clear audio",
        "expected_range": (100, 200)  # Expected price range for validation
    },
    {
        "name": "Smartphone",
        "description": "iPhone 15 Pro Max 256GB Space Black with titanium design and advanced camera system",
        "expected_range": (1000, 1300)
    },
    {
        "name": "Laptop",
        "description": "MacBook Pro 14-inch M3 chip with 16GB RAM and 512GB SSD, perfect for professional work",
        "expected_range": (2000, 2500)
    },
    {
        "name": "Gaming Console",
        "description": "PlayStation 5 console with DualSense controller and 825GB SSD storage",
        "expected_range": (400, 600)
    },
    {
        "name": "Headphones",
        "description": "Sony WH-1000XM5 wireless noise-canceling headphones with 30-hour battery life",
        "expected_range": (300, 400)
    },
    {
        "name": "Smart Watch",
        "description": "Apple Watch Series 9 GPS 45mm with Sport Band and health monitoring features",
        "expected_range": (350, 450)
    },
    {
        "name": "Graphics Card",
        "description": "NVIDIA RTX 4070 Ti graphics card with 12GB GDDR6X memory for gaming and AI workloads",
        "expected_range": (700, 900)
    },
    {
        "name": "Kitchen Appliance",
        "description": "KitchenAid stand mixer 5-quart with multiple attachments for baking and cooking",
        "expected_range": (300, 500)
    }
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_separator(title: str = ""):
    """Print a visual separator with optional title."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print(f"{'='*60}")

def validate_price(price: float, expected_range: tuple, item_name: str) -> bool:
    """
    Validate if the predicted price falls within expected range.
    
    Args:
        price: Predicted price
        expected_range: Tuple of (min_price, max_price)
        item_name: Name of the item for logging
    
    Returns:
        bool: True if price is within expected range
    """
    min_price, max_price = expected_range
    is_valid = min_price <= price <= max_price
    
    if is_valid:
        print(f"✅ {item_name}: Price ${price:.2f} is within expected range ${min_price}-${max_price}")
    else:
        print(f"⚠️  {item_name}: Price ${price:.2f} is outside expected range ${min_price}-${max_price}")
    
    return is_valid

def run_single_test(agent: FrontierAgent, test_case: dict) -> tuple:
    """
    Run a single test case and return results.
    
    Args:
        agent: FrontierAgent instance
        test_case: Test case dictionary with name, description, and expected_range
    
    Returns:
        tuple: (success: bool, price: float, duration: float)
    """
    try:
        print(f"\n📝 Testing: {test_case['name']}")
        print(f"   Description: {test_case['description'][:60]}...")
        
        # Measure execution time
        start_time = time.time()
        price = agent.price(test_case['description'])
        duration = time.time() - start_time
        
        print(f"   💰 Predicted Price: ${price:.2f}")
        print(f"   ⏱️  Execution Time: {duration:.2f} seconds")
        
        # Validate the result
        is_valid = validate_price(price, test_case['expected_range'], test_case['name'])
        
        return True, price, duration
        
    except Exception as e:
        print(f"❌ Error testing {test_case['name']}: {str(e)}")
        return False, 0.0, 0.0

# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def main():
    """
    Main test function that runs all test cases and provides summary.
    """
    print_separator("FRONTIER AGENT TEST SUITE")
    print("Testing the OpenAI GPT-4o-mini RAG-based pricing agent")
    print("This will test multiple product descriptions using ChromaDB for context.")
    
    # ========================================================================
    # CHROMADB INITIALIZATION
    # ========================================================================
    
    print_separator("INITIALIZING CHROMADB")
    collection = setup_chromadb()
    
    # ========================================================================
    # AGENT INITIALIZATION
    # ========================================================================
    
    print_separator("INITIALIZING AGENT")
    
    try:
        # Create the frontier agent with ChromaDB collection
        agent = FrontierAgent(collection)
        print("✅ Frontier Agent initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize agent: {str(e)}")
        print("   Make sure OpenAI API key is set and ChromaDB is accessible.")
        sys.exit(1)
    
    # ========================================================================
    # RUNNING TEST CASES
    # ========================================================================
    
    print_separator("RUNNING TEST CASES")
    
    results = []
    total_duration = 0.0
    successful_tests = 0
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n[Test {i}/{len(TEST_CASES)}]")
        success, price, duration = run_single_test(agent, test_case)
        
        results.append({
            'name': test_case['name'],
            'success': success,
            'price': price,
            'duration': duration,
            'expected_range': test_case['expected_range']
        })
        
        if success:
            successful_tests += 1
            total_duration += duration
        
        # Add a small delay between tests to avoid overwhelming the API
        if i < len(TEST_CASES):
            time.sleep(1)
    
    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    
    print_separator("TEST RESULTS SUMMARY")
    
    print(f"📊 Test Results:")
    print(f"   • Total Tests: {len(TEST_CASES)}")
    print(f"   • Successful: {successful_tests}")
    print(f"   • Failed: {len(TEST_CASES) - successful_tests}")
    print(f"   • Success Rate: {(successful_tests/len(TEST_CASES)*100):.1f}%")
    
    if successful_tests > 0:
        avg_duration = total_duration / successful_tests
        print(f"   • Average Response Time: {avg_duration:.2f} seconds")
    
    print(f"\n📋 Detailed Results:")
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        if result['success']:
            print(f"   {status} {result['name']}: ${result['price']:.2f} ({result['duration']:.2f}s)")
        else:
            print(f"   {status} {result['name']}: Test failed")
    
    # ========================================================================
    # PERFORMANCE NOTES
    # ========================================================================
    
    print_separator("PERFORMANCE NOTES")
    print("🔍 Performance Insights:")
    print("   • RAG search adds context but may increase response time")
    print("   • ChromaDB vector search is typically fast (< 1 second)")
    print("   • OpenAI API calls dominate the total response time")
    print("   • Similar products improve price estimation accuracy")
    
    # ========================================================================
    # RAG ANALYSIS
    # ========================================================================
    
    print_separator("RAG SYSTEM ANALYSIS")
    print("🧠 RAG System Insights:")
    print("   • Agent uses sentence-transformers/all-MiniLM-L6-v2 for embeddings")
    print("   • Retrieves 5 most similar products for context")
    print("   • GPT-4o-mini processes context to estimate prices")
    print("   • Seed=42 ensures reproducible results")
    
    print_separator()
    print("🎉 Test suite completed!")
    
    # Return exit code based on success rate
    if successful_tests == len(TEST_CASES):
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print(f"⚠️  {len(TEST_CASES) - successful_tests} test(s) failed.")
        sys.exit(1)

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    main() 