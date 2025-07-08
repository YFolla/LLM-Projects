"""
Test Script for Ensemble Model and Agent

This script tests the complete ensemble system:
1. Optionally trains a new ensemble model (if ensemble_model.pkl doesn't exist)
2. Tests the EnsembleAgent with the trained model
3. Provides sample predictions to verify functionality

This serves as both a test and a demonstration of the ensemble system.
"""

import os
import sys
import logging
from typing import List, Dict
from dotenv import load_dotenv

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Set up logging to see agent messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

def setup_environment():
    """Set up environment variables for the test."""
    print("Setting up environment...")
    
    # Load environment variables from .env file
    load_dotenv(override=True)
    
    # Set OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please make sure you have a .env file with OPENAI_API_KEY set")
        return False
    
    os.environ['OPENAI_API_KEY'] = openai_key
    
    # Set HuggingFace token if available
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token
    
    print("✅ Environment setup complete")
    return True

def test_ensemble_agent():
    """Test the EnsembleAgent with sample product descriptions."""
    print("🧪 Testing Ensemble Agent")
    print("=" * 50)
    
    # Setup environment first
    if not setup_environment():
        return
    
    try:
        # Import and initialize the EnsembleAgent
        from agents.ensemble_agent import EnsembleAgent
        
        # Check if ensemble model exists
        model_path = 'ensemble_model.pkl'
        if not os.path.exists(model_path):
            print("⚠️  Ensemble model not found. Training a new one...")
            print("Run 'python ensemble_model.py' to train the ensemble model first.")
            return
        
        # Initialize the agent with the new constructor
        # The agent will handle ChromaDB setup internally
        print("Initializing EnsembleAgent...")
        agent = EnsembleAgent(
            model_path=model_path,
            chromadb_path='products_vectorstore'
        )
        
        # Test cases
        test_cases = [
            {
                "name": "Budget Headphones",
                "description": "Basic over-ear headphones with 3.5mm jack for music listening",
                "expected_range": (20, 80)
            },
            {
                "name": "Premium Smartphone",
                "description": "iPhone 15 Pro Max 256GB Space Black with titanium design and advanced camera system",
                "expected_range": (900, 1300)
            },
            {
                "name": "Gaming Laptop",
                "description": "High-performance gaming laptop with RTX 4070 graphics card and 16GB RAM",
                "expected_range": (1200, 2000)
            },
            {
                "name": "Coffee Maker",
                "description": "Automatic drip coffee maker with programmable timer and 12-cup capacity",
                "expected_range": (50, 150)
            },
            {
                "name": "Wireless Earbuds",
                "description": "Apple AirPods Pro 2nd generation with active noise cancellation",
                "expected_range": (200, 300)
            }
        ]
        
        print("\n🎯 Running Test Cases")
        print("-" * 30)
        
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[Test {i}/{len(test_cases)}] {test_case['name']}")
            print(f"Description: {test_case['description']}")
            
            try:
                # Get price prediction
                predicted_price = agent.price(test_case['description'])
                
                # Check if price is within expected range
                min_price, max_price = test_case['expected_range']
                is_reasonable = min_price <= predicted_price <= max_price
                
                # Display results
                status = "✅ REASONABLE" if is_reasonable else "⚠️  OUTSIDE RANGE"
                print(f"Predicted Price: ${predicted_price:.2f}")
                print(f"Expected Range: ${min_price:.2f} - ${max_price:.2f}")
                print(f"Status: {status}")
                
                results.append({
                    'name': test_case['name'],
                    'predicted': predicted_price,
                    'expected_range': test_case['expected_range'],
                    'reasonable': is_reasonable
                })
                
            except Exception as e:
                print(f"❌ Error predicting price: {e}")
                results.append({
                    'name': test_case['name'],
                    'predicted': None,
                    'expected_range': test_case['expected_range'],
                    'reasonable': False
                })
        
        # Summary
        print("\n📊 Test Summary")
        print("=" * 50)
        
        successful_tests = sum(1 for r in results if r['predicted'] is not None)
        reasonable_predictions = sum(1 for r in results if r['reasonable'])
        
        print(f"Total Tests: {len(results)}")
        print(f"Successful Predictions: {successful_tests}")
        print(f"Reasonable Predictions: {reasonable_predictions}")
        print(f"Success Rate: {(successful_tests/len(results)*100):.1f}%")
        print(f"Reasonableness Rate: {(reasonable_predictions/len(results)*100):.1f}%")
        
        # Display model info
        print("\n🔍 Ensemble Model Information")
        print("-" * 30)
        model_info = agent.get_model_info()
        if model_info:
            print(f"Model Type: {model_info.get('model_type', 'Unknown')}")
            print("Feature Coefficients:")
            for feature, coef in model_info.get('coefficients', {}).items():
                print(f"  {feature}: {coef:.4f}")
            if model_info.get('intercept') is not None:
                print(f"  Intercept: {model_info['intercept']:.4f}")
        
        if reasonable_predictions >= len(results) * 0.6:  # 60% threshold
            print("\n🎉 Ensemble Agent is working well!")
        else:
            print("\n⚠️  Ensemble Agent may need tuning or more training data.")
            
    except Exception as e:
        print(f"❌ Error testing ensemble agent: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run all tests."""
    print("🚀 Ensemble System Test Suite")
    print("=" * 50)
    
    # Check if ensemble model exists
    model_path = 'ensemble_model.pkl'
    if not os.path.exists(model_path):
        print("⚠️  Ensemble model not found!")
        print("To train the ensemble model, run:")
        print("  python ensemble_model.py")
        print("")
        print("This will:")
        print("  1. Load test data from ../13_Frontier_finetuning/test.pkl")
        print("  2. Initialize all three agents (Specialist, Frontier, RandomForest)")
        print("  3. Collect predictions on test data")
        print("  4. Train a Linear Regression ensemble model")
        print("  5. Save the trained model as ensemble_model.pkl")
        print("")
        return
    
    # Test the ensemble agent
    test_ensemble_agent()

if __name__ == "__main__":
    main() 