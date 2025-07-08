"""
Ensemble Model Training Script

This script trains an ensemble model that combines predictions from three specialist agents:
1. SpecialistAgent - Fine-tuned model for price prediction
2. FrontierAgent - RAG-based agent using ChromaDB + OpenAI GPT-4o-mini
3. RandomForestAgent - Random Forest model on product embeddings

The ensemble uses Linear Regression to learn optimal weights for combining
the individual agent predictions, creating a more robust pricing system.
"""

# Core Python libraries
import os
import pickle
from tqdm import tqdm

# Third-party libraries
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from sklearn.linear_model import LinearRegression
import joblib
import chromadb

# Local imports
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.random_forest_agent import RandomForestAgent

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """Set up environment variables and authenticate with external services."""
    print("Setting up environment...")
    
    # Load environment variables from .env file
    load_dotenv(override=True)
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

    # Authenticate with HuggingFace Hub
    hf_token = os.environ['HF_TOKEN']
    login(hf_token, add_to_git_credential=True)
    
    print("✅ Environment setup complete")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_data():
    """
    Load test data from pickle file for ensemble training.
    
    Returns:
        list: Test data items containing product information
    """
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
        raise
    except Exception as e:
        print(f"❌ Error loading test data: {str(e)}")
        raise

def setup_chromadb_for_frontier():
    """
    Set up ChromaDB collection specifically for the FrontierAgent.
    Only called when FrontierAgent is needed.
    
    Returns:
        chromadb.Collection: ChromaDB collection with product embeddings
    """
    print("Setting up ChromaDB collection for FrontierAgent...")
    
    try:
        # Initialize ChromaDB client
        db_path = "products_vectorstore"
        client = chromadb.PersistentClient(path=db_path)
        
        # Get the products collection
        collection = client.get_or_create_collection('products')
        
        print("✅ ChromaDB collection ready for FrontierAgent")
        return collection
    
    except Exception as e:
        print(f"❌ Error setting up ChromaDB: {str(e)}")
        print("Make sure to run vectorize.py first to create the product embeddings")
        raise

# ============================================================================
# AGENT INITIALIZATION
# ============================================================================

def initialize_agents():
    """
    Initialize the three specialist agents for ensemble training.
    
    Returns:
        tuple: (specialist_agent, frontier_agent, random_forest_agent)
    """
    print("Initializing specialist agents...")
    
    try:
        # Initialize agents that don't need ChromaDB first
        print("  - Initializing SpecialistAgent (Modal connection)...")
        specialist = SpecialistAgent()
        
        print("  - Initializing RandomForestAgent (local model)...")
        random_forest = RandomForestAgent()
        
        # Initialize ChromaDB only for FrontierAgent
        print("  - Setting up ChromaDB for FrontierAgent...")
        collection = setup_chromadb_for_frontier()
        
        print("  - Initializing FrontierAgent (RAG + OpenAI)...")
        frontier = FrontierAgent(collection)
        
        print("✅ All agents initialized successfully")
        return specialist, frontier, random_forest
    
    except Exception as e:
        print(f"❌ Error initializing agents: {str(e)}")
        raise

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_description(item):
    """
    Extract product description from item prompt.
    
    Args:
        item: Product item with prompt attribute
    
    Returns:
        str: Clean product description
    """
    # Extract description from the structured prompt format
    # Format: "... to the nearest dollar?\n\n[DESCRIPTION]\n\nPrice is $..."
    try:
        description = item.prompt.split("to the nearest dollar?\n\n")[1].split("\n\nPrice is $")[0]
        return description.strip()
    except (IndexError, AttributeError):
        # Fallback for different prompt formats
        return str(item.prompt)

# ============================================================================
# ENSEMBLE TRAINING
# ============================================================================

def collect_predictions(agents, test_data, start_idx=1000, end_idx=1250):
    """
    Collect predictions from all three agents on test data.
    
    Args:
        agents: Tuple of (specialist, frontier, random_forest) agents
        test_data: List of test items
        start_idx: Starting index for test data slice
        end_idx: Ending index for test data slice
    
    Returns:
        tuple: (predictions_df, actual_prices)
    """
    specialist, frontier, random_forest = agents
    
    print(f"Collecting predictions on test data slice [{start_idx}:{end_idx}]...")
    
    # Initialize prediction lists
    specialist_predictions = []
    frontier_predictions = []
    random_forest_predictions = []
    actual_prices = []
    
    # Collect predictions from each agent
    test_slice = test_data[start_idx:end_idx]
    
    for item in tqdm(test_slice, desc="Collecting predictions"):
        description = extract_description(item)
        
        # Get prediction from each agent
        specialist_pred = specialist.price(description)
        frontier_pred = frontier.price(description)
        random_forest_pred = random_forest.price(description)
        
        # Store predictions and actual price
        specialist_predictions.append(specialist_pred)
        frontier_predictions.append(frontier_pred)
        random_forest_predictions.append(random_forest_pred)
        actual_prices.append(item.price)
    
    # Create feature matrix with individual predictions and derived features
    min_predictions = [min(s, f, r) for s, f, r in zip(specialist_predictions, frontier_predictions, random_forest_predictions)]
    max_predictions = [max(s, f, r) for s, f, r in zip(specialist_predictions, frontier_predictions, random_forest_predictions)]
    
    # Create DataFrame with all features
    predictions_df = pd.DataFrame({
        'Specialist': specialist_predictions,
        'Frontier': frontier_predictions,
        'RandomForest': random_forest_predictions,
        'Min': min_predictions,
        'Max': max_predictions,
    })
    
    print(f"✅ Collected {len(predictions_df)} predictions from all agents")
    return predictions_df, actual_prices

def train_ensemble_model(X, y):
    """
    Train the Linear Regression ensemble model.
    
    Args:
        X: Feature matrix with agent predictions
        y: Target values (actual prices)
    
    Returns:
        sklearn.LinearRegression: Trained ensemble model
    """
    print("Training ensemble Linear Regression model...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Convert target to pandas Series for consistency
    y_series = pd.Series(y)
    
    # Train Linear Regression model
    ensemble_model = LinearRegression()
    ensemble_model.fit(X, y_series)
    
    # Display model coefficients
    print("\n📊 Ensemble Model Coefficients:")
    feature_names = X.columns.tolist()
    for feature, coef in zip(feature_names, ensemble_model.coef_):
        print(f"   {feature}: {coef:.4f}")
    print(f"   Intercept: {ensemble_model.intercept_:.4f}")
    
    print("✅ Ensemble model training complete")
    return ensemble_model

def save_ensemble_model(model, filepath='ensemble_model.pkl'):
    """
    Save the trained ensemble model to disk.
    
    Args:
        model: Trained ensemble model
        filepath: Path where to save the model
    """
    print(f"Saving ensemble model to {filepath}...")
    
    try:
        joblib.dump(model, filepath)
        print(f"✅ Ensemble model saved successfully to {filepath}")
    except Exception as e:
        print(f"❌ Error saving model: {str(e)}")
        raise

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main function to train the ensemble model."""
    print("🚀 Starting Ensemble Model Training")
    print("=" * 50)
    
    try:
        # Step 1: Environment setup
        setup_environment()
        
        # Step 2: Load test data
        test_data = load_test_data()
        
        # Step 3: Initialize agents (ChromaDB only set up for FrontierAgent)
        agents = initialize_agents()
        
        # Step 4: Collect predictions
        predictions_df, actual_prices = collect_predictions(agents, test_data)
        
        # Step 5: Train ensemble model
        ensemble_model = train_ensemble_model(predictions_df, actual_prices)
        
        # Step 6: Save the model
        save_ensemble_model(ensemble_model)
        
        print("\n🎉 Ensemble model training completed successfully!")
        print("The model is now ready to be used by EnsembleAgent")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()