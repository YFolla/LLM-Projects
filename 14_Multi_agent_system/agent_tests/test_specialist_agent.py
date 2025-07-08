"""
Test script for the Specialist Agent.

This script demonstrates how to use the SpecialistAgent to get price estimates
from the Modal-deployed fine-tuned model. It includes multiple test cases and
proper error handling to validate the agent's functionality.
"""

import logging
import sys
import time
import os

# Add parent directory to path to import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.specialist_agent import SpecialistAgent

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

def run_single_test(agent: SpecialistAgent, test_case: dict) -> tuple:
    """
    Run a single test case and return results.
    
    Args:
        agent: SpecialistAgent instance
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
    print_separator("SPECIALIST AGENT TEST SUITE")
    print("Testing the Modal-deployed pricing service through the SpecialistAgent")
    print("This will test multiple product descriptions and validate the results.")
    
    # ========================================================================
    # AGENT INITIALIZATION
    # ========================================================================
    
    print_separator("INITIALIZING AGENT")
    
    try:
        # Create the specialist agent (this will connect to Modal)
        agent = SpecialistAgent()
        print("✅ Agent initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize agent: {str(e)}")
        print("   Make sure the Modal service is deployed and accessible.")
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
        
        # Add a small delay between tests to avoid overwhelming the service
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
    print("   • First request may be slower due to cold start")
    print("   • Subsequent requests should be faster with warm containers")
    print("   • Response time depends on model complexity and network latency")
    print("   • Modal automatically scales based on demand")
    
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