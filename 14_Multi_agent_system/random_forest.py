# Essential imports for model training and data handling
import pickle
import numpy as np
import chromadb
import joblib
from sklearn.ensemble import RandomForestRegressor
from sentence_transformers import SentenceTransformer

# CONSTANTS
DB = "products_vectorstore"

def train_and_save_model():
    """
    Train a Random Forest model on product embeddings and save it to disk.
    
    This function:
    1. Loads product data from ChromaDB
    2. Trains a Random Forest model on embeddings -> prices
    3. Saves the trained model for use by RandomForestAgent
    """
    print("Loading product data from ChromaDB...")
    
    # Initialize ChromaDB client and collection
    client = chromadb.PersistentClient(path=DB)
    collection = client.get_or_create_collection('products')
    
    # Retrieve all vectors, documents, and metadata from the collection
    result = collection.get(include=['embeddings', 'documents', 'metadatas'])
    vectors = np.array(result['embeddings'])  # Product embeddings
    prices = [metadata['price'] for metadata in result['metadatas']]  # Extract prices
    
    print(f"Training Random Forest model on {len(vectors)} products...")
    
    # Train Random Forest model on product embeddings to predict prices
    # Using 100 estimators for good performance, random_state for reproducibility
    # n_jobs=6 uses 6 out of 10 available cores - fast but leaves cores free for system
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=8)
    rf_model.fit(vectors, prices)
    
    # Save the trained model to disk for use by RandomForestAgent
    model_path = 'random_forest_model.pkl'
    joblib.dump(rf_model, model_path)
    print(f"Model saved to {model_path}")
    
    return rf_model

if __name__ == "__main__":
    # Train and save the model when this script is run directly
    model = train_and_save_model()
    print("Random Forest model training completed!")
    print("The model can now be used by RandomForestAgent for price predictions.")