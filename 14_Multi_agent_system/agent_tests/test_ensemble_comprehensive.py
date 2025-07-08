"""
Comprehensive Ensemble Model Testing Script

This script evaluates the EnsembleAgent on the full test dataset using the 
existing Tester class to generate performance metrics and visualization charts.
"""

import os
import sys
import pickle
from dotenv import load_dotenv

# Import the testing framework and ensemble agent
from testing import Tester
from agents.ensemble_agent import EnsembleAgent

def setup_environment():
    """Set up environment variables for testing."""
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

def load_test_data():
    """Load the test dataset from the fine-tuning project."""
    print("Loading test data...")
    
    try:
        # Load test data from the fine-tuning project
        test_path = '../13_Frontier_finetuning/test.pkl'
        with open(test_path, 'rb') as file:
            test_data = pickle.load(file)
        
        print(f"✅ Successfully loaded {len(test_data)} test items")
        return test_data
    
    except FileNotFoundError:
        print(f"❌ Error: Could not find test data at {test_path}")
        print("Please ensure the test.pkl file exists in the 13_Frontier_finetuning directory")
        return None
    except Exception as e:
        print(f"❌ Error loading test data: {str(e)}")
        return None

def extract_description(item):
    """
    Extract product description from item prompt.
    
    Args:
        item: Product item with prompt attribute
    
    Returns:
        str: Clean product description
    """
    try:
        # Extract description from the structured prompt format
        # Format: "... to the nearest dollar?\n\n[DESCRIPTION]\n\nPrice is $..."
        description = item.prompt.split("to the nearest dollar?\n\n")[1].split("\n\nPrice is $")[0]
        return description.strip()
    except (IndexError, AttributeError):
        # Fallback for different prompt formats
        return str(item.prompt)

def create_ensemble_predictor(agent):
    """
    Create a predictor function that matches the Tester class interface.
    
    Args:
        agent: EnsembleAgent instance
    
    Returns:
        function: Predictor function for the Tester class
    """
    def ensemble_predictor(item):
        """Extract description from item and predict price using ensemble model."""
        description = extract_description(item)
        return agent.price(description)
    
    return ensemble_predictor

def main():
    """Main function to run comprehensive ensemble testing."""
    print("🚀 Comprehensive Ensemble Model Testing")
    print("=" * 60)
    
    # Setup environment
    if not setup_environment():
        return
    
    # Load test data
    test_data = load_test_data()
    if not test_data:
        return
    
    # Check if ensemble model exists
    model_path = 'ensemble_model.pkl'
    if not os.path.exists(model_path):
        print("❌ Ensemble model not found!")
        print("Please run 'python ensemble_model.py' first to train the ensemble model")
        return
    
    try:
        # Initialize the EnsembleAgent
        print("Initializing EnsembleAgent...")
        agent = EnsembleAgent(
            model_path=model_path,
            chromadb_path='products_vectorstore'
        )
        
        # Create predictor function for the Tester
        ensemble_predictor = create_ensemble_predictor(agent)
        
        # Run comprehensive testing with the Tester class
        print("\n🎯 Running Comprehensive Testing")
        print("=" * 60)
        print("This will evaluate the ensemble model on 250 test samples")
        print("and generate a performance chart showing:")
        print("• Individual predictions vs ground truth")
        print("• Color-coded accuracy (green=good, orange=ok, red=poor)")
        print("• Overall performance metrics (Error, RMSLE, Hit Rate)")
        print("\nStarting evaluation...")
        
        # Use the Tester class to evaluate the ensemble model
        # This will generate the scatter plot chart you're looking for
        Tester.test(ensemble_predictor, test_data)
        
        print("\n🎉 Comprehensive testing completed!")
        print("The performance chart shows how well the ensemble model")
        print("combines predictions from all three specialist agents.")
        
    except Exception as e:
        print(f"❌ Error during comprehensive testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 