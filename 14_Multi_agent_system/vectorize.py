# ================================
# VECTOR DATABASE INITIALIZATION
# ================================
"""
This script creates a vector database from product data using ChromaDB and SentenceTransformers.
It loads pre-processed product data, generates embeddings, and stores them for similarity search.
"""

# Core Python imports
import os
import pickle
from tqdm import tqdm

# Environment and authentication
from dotenv import load_dotenv
from huggingface_hub import login

# Machine learning and vector database imports
from sentence_transformers import SentenceTransformer
import chromadb

# ================================
# ENVIRONMENT SETUP
# ================================

# Load environment variables from .env file
load_dotenv(override=True)

# Set up API keys for external services
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# Database configuration
DB = "products_vectorstore"  # Local ChromaDB directory name

# ================================
# HUGGING FACE AUTHENTICATION
# ================================

# Authenticate with HuggingFace Hub for model access
hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)

# ================================
# IMPORT CUSTOM CLASSES
# ================================

# Import the Item class definition
from items import Item

# ================================
# UTILITY FUNCTIONS
# ================================

def description(item):
    """
    Extract the product description from an Item object.
    
    Args:
        item (Item): An Item object containing product information
        
    Returns:
        str: Clean product description without pricing information
    """
    # Remove the question prompt from the item's text
    text = item.prompt.replace("How much does this cost to the nearest dollar?\n\n", "")
    # Split on the price indicator and return only the description part
    return text.split("\n\nPrice is $")[0]

# ================================
# DATA LOADING
# ================================

# Load the pre-processed training data from the 13_Frontier_finetuning directory
# This file contains pickled Item objects with product information
train_data_path = os.path.join('..', '13_Frontier_finetuning', 'train.pkl')

try:
    with open(train_data_path, 'rb') as file:
        train = pickle.load(file)
    print(f"Successfully loaded {len(train)} items from {train_data_path}")
except FileNotFoundError:
    print(f"Error: Could not find {train_data_path}")
    print("Please ensure the train.pkl file exists in the 13_Frontier_finetuning directory")
    exit(1)

# ================================
# VECTOR DATABASE SETUP
# ================================

# Initialize ChromaDB client with persistent storage
client = chromadb.PersistentClient(path=DB)

# Collection configuration
collection_name = "products"

# Clean slate: remove existing collection if it exists
# This ensures we start fresh with the current data
existing_collection_names = client.list_collections()

if collection_name in existing_collection_names:
    client.delete_collection(collection_name)
    print(f"Deleted existing collection: {collection_name}")

# Create new collection for product embeddings
collection = client.create_collection(collection_name)
print(f"Created new collection: {collection_name}")

# ================================
# EMBEDDING MODEL SETUP
# ================================

# Initialize the sentence transformer model for creating embeddings
# This model converts text descriptions into numerical vectors
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Initialized SentenceTransformer model")

# ================================
# BATCH PROCESSING CONFIGURATION
# ================================

# Set the number of documents to process
# Full dataset processing - adjust NUMBER_OF_DOCUMENTS for testing
# NUMBER_OF_DOCUMENTS = len(train)

# For testing with a smaller subset, uncomment the line below:
NUMBER_OF_DOCUMENTS = 100000

print(f"Processing {NUMBER_OF_DOCUMENTS} documents")

# ================================
# VECTOR GENERATION AND STORAGE
# ================================

# Process documents in batches for memory efficiency
# Batch size of 1000 provides good balance between memory usage and processing speed
BATCH_SIZE = 1000

print("Starting batch processing and vector generation...")

for i in tqdm(range(0, NUMBER_OF_DOCUMENTS, BATCH_SIZE), desc="Processing batches"):
    # Calculate the end index for this batch
    batch_end = min(i + BATCH_SIZE, NUMBER_OF_DOCUMENTS)
    current_batch = train[i:batch_end]
    
    # Generate text descriptions for each item in the batch
    documents = [description(item) for item in current_batch]
    
    # Create embeddings for all documents in the batch
    # Convert to float and list format required by ChromaDB
    vectors = model.encode(documents).astype(float).tolist()
    
    # Prepare metadata for each document (category and price information)
    metadatas = [{"category": item.category, "price": item.price} for item in current_batch]
    
    # Generate unique IDs for each document
    ids = [f"doc_{j}" for j in range(i, i + len(documents))]
    
    # Add the batch to the ChromaDB collection
    collection.add(
        ids=ids,                # Unique identifiers for each document
        documents=documents,     # Original text descriptions
        embeddings=vectors,      # Vector embeddings for similarity search
        metadatas=metadatas     # Additional searchable metadata
    )

print(f"Successfully processed and stored {NUMBER_OF_DOCUMENTS} documents in ChromaDB")
print(f"Vector database created at: {DB}")
print("Vectorization complete!")