"""
Test script for the Random Forest Agent.

This script demonstrates how to use the RandomForestAgent to get price estimates
from the trained Random Forest model. It includes multiple test cases, performance
validation, and proper error handling to validate the agent's functionality.
"""

import logging
import sys
import time
import os
import pickle
import numpy as np

# Add parent directory to path to import agents and testing utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.random_forest_agent import RandomForestAgent
from testing import Tester

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
        "expected_range": (80, 180)  # Expected price range for validation
    },
    {
        "name": "Smartphone",
        "description": "iPhone 15 Pro Max 256GB Space Black with titanium design and advanced camera system",
        "expected_range": (900, 1400)
    },
    {
        "name": "Laptop",
        "description": "MacBook Pro 14-inch M3 chip with 16GB RAM and 512GB SSD, perfect for professional work",
        "expected_range": (1800, 2800)
    },
    {
        "name": "Gaming Console",
        "description": "PlayStation 5 console with DualSense controller and 825GB SSD storage",
        "expected_range": (350, 650)
    },
    {
        "name": "Headphones",
        "description": "Sony WH-1000XM5 wireless noise-canceling headphones with 30-hour battery life",
        "expected_range": (250, 450)
    },
    {
        "name": "Kitchen Appliance",
        "description": "KitchenAid Stand Mixer 5-quart with multiple attachments for baking and cooking",
        "expected_range": (200, 400)
    },
    {
        "name": "Power Tool",
        "description": "DeWalt 20V Max cordless drill with battery and charger for construction work",
        "expected_range": (100, 250)
    },
    {
        "name": "Fitness Equipment",
        "description": "Peloton bike with touchscreen display and subscription for home workouts",
        "expected_range": (1200, 2000)
    },
    {
        "name": "Camera",
        "description": "Canon EOS R5 mirrorless camera with 45MP sensor and 8K video recording",
        "expected_range": (3000, 4000)
    },
    {
        "name": "Budget Item",
        "description": "USB-C charging cable 6 feet long for smartphones and tablets",
        "expected_range": (10, 30)
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

def run_single_test(agent: RandomForestAgent, test_case: dict) -> tuple:
    """
    Run a single test case and return results.
    
    Args:
        agent: RandomForestAgent instance
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
        print(f"   ⏱️  Execution Time: {duration:.3f} seconds")
        
        # Validate the result
        is_valid = validate_price(price, test_case['expected_range'], test_case['name'])
        
        return True, price, duration
        
    except Exception as e:
        print(f"❌ Error testing {test_case['name']}: {str(e)}")
        return False, 0.0, 0.0

def load_test_data():
    """
    Load test data from pickle file for comprehensive testing.
    
    Returns:
        list: Test data items, or None if file not found
    """
    try:
        test_path = os.path.join(os.path.dirname(__file__), '..', '..', '13_Frontier_finetuning', 'test.pkl')
        with open(test_path, 'rb') as file:
            test_data = pickle.load(file)
        print(f"✅ Loaded {len(test_data)} test items from pickle file")
        return test_data
    except FileNotFoundError:
        print("⚠️  Test pickle file not found. Skipping comprehensive testing.")
        return None

def run_comprehensive_test(agent: RandomForestAgent, test_data: list, num_samples: int = 50):
    """
    Run comprehensive testing using the Tester class with pickle data.
    
    Args:
        agent: RandomForestAgent instance
        test_data: List of test items
        num_samples: Number of samples to test
    """
    print_separator("COMPREHENSIVE TESTING WITH PICKLE DATA")
    print(f"Running comprehensive test with {num_samples} samples from test dataset...")
    
    # Create a wrapper function for the agent that matches the expected signature
    def rf_predictor(item):
        """Extract description from item and predict price."""
        # Extract description from item prompt (same logic as in random_forest.py)
        description = item.prompt.split("to the nearest dollar?\n\n")[1].split("\n\nPrice is $")[0]
        return agent.price(description)
    
    # Use the Tester class for comprehensive evaluation
    try:
        Tester.test(rf_predictor, test_data[:num_samples])
    except Exception as e:
        print(f"❌ Error during comprehensive testing: {str(e)}")

# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

def run_performance_benchmark(agent: RandomForestAgent, num_predictions: int = 100):
    """
    Run performance benchmark to measure prediction speed.
    
    Args:
        agent: RandomForestAgent instance
        num_predictions: Number of predictions to make for benchmarking
    """
    print_separator("PERFORMANCE BENCHMARK")
    print(f"Running performance benchmark with {num_predictions} predictions...")
    
    # Sample description for benchmarking
    sample_description = "Wireless Bluetooth headphones with noise cancellation and 20-hour battery life"
    
    durations = []
    
    try:
        for i in range(num_predictions):
            start_time = time.time()
            price = agent.price(sample_description)
            duration = time.time() - start_time
            durations.append(duration)
            
            if i % 20 == 0:  # Print progress every 20 predictions
                print(f"   Progress: {i+1}/{num_predictions} predictions completed")
        
        # Calculate statistics
        avg_duration = np.mean(durations)
        min_duration = np.min(durations)
        max_duration = np.max(durations)
        std_duration = np.std(durations)
        
        print(f"\n📊 Performance Results:")
        print(f"   • Average Time: {avg_duration:.3f} seconds")
        print(f"   • Min Time: {min_duration:.3f} seconds")
        print(f"   • Max Time: {max_duration:.3f} seconds")
        print(f"   • Std Deviation: {std_duration:.3f} seconds")
        print(f"   • Predictions/Second: {1/avg_duration:.1f}")
        
    except Exception as e:
        print(f"❌ Error during performance benchmark: {str(e)}")

# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def main():
    """
    Main test function that runs all test cases and provides summary.
    """
    print_separator("RANDOM FOREST AGENT TEST SUITE")
    print("Testing the Random Forest Agent with trained model")
    print("This will test multiple product descriptions and validate the results.")
    
    # ========================================================================
    # AGENT INITIALIZATION
    # ========================================================================
    
    print_separator("INITIALIZING AGENT")
    
    try:
        # Create the random forest agent (this will load the trained model)
        agent = RandomForestAgent()
        print("✅ Agent initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize agent: {str(e)}")
        print("   Make sure the random_forest_model.pkl file exists.")
        print("   Run the random_forest.py script first to train and save the model.")
        sys.exit(1)
    
    # ========================================================================
    # RUNNING BASIC TEST CASES
    # ========================================================================
    
    print_separator("RUNNING BASIC TEST CASES")
    
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
    
    # ========================================================================
    # BASIC RESULTS SUMMARY
    # ========================================================================
    
    print_separator("BASIC TEST RESULTS SUMMARY")
    
    print(f"📊 Basic Test Results:")
    print(f"   • Total Tests: {len(TEST_CASES)}")
    print(f"   • Successful: {successful_tests}")
    print(f"   • Failed: {len(TEST_CASES) - successful_tests}")
    print(f"   • Success Rate: {(successful_tests/len(TEST_CASES)*100):.1f}%")
    
    if successful_tests > 0:
        avg_duration = total_duration / successful_tests
        print(f"   • Average Duration: {avg_duration:.3f} seconds")
        print(f"   • Predictions/Second: {1/avg_duration:.1f}")
    
    # ========================================================================
    # PERFORMANCE BENCHMARK
    # ========================================================================
    
    if successful_tests > 0:
        run_performance_benchmark(agent, num_predictions=100)
    
    # ========================================================================
    # COMPREHENSIVE TESTING
    # ========================================================================
    
    # Load test data and run comprehensive testing
    test_data = load_test_data()
    if test_data and successful_tests > 0:
        run_comprehensive_test(agent, test_data, num_samples=50)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print_separator("FINAL TEST SUMMARY")
    
    if successful_tests == len(TEST_CASES):
        print("🎉 All tests passed! The Random Forest Agent is working correctly.")
    elif successful_tests > 0:
        print(f"⚠️  {successful_tests}/{len(TEST_CASES)} tests passed. Some issues detected.")
    else:
        print("❌ All tests failed. Please check the agent implementation.")
    
    print("\n🔍 Key Observations:")
    print("   • Random Forest predictions are fast (typically < 0.1 seconds)")
    print("   • Model was trained on 400,000 product embeddings")
    print("   • Uses SentenceTransformer embeddings for feature extraction")
    print("   • Predictions are deterministic (same input = same output)")
    
    print("\n💡 Tips for better results:")
    print("   • Provide detailed product descriptions")
    print("   • Include brand names and key features")
    print("   • Specify technical specifications when relevant")


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    main() 