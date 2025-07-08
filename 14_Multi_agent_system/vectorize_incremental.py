# ================================
# INCREMENTAL VECTOR DATABASE LOADING
# ================================
"""
This script adds the remaining 300k products to the existing ChromaDB collection.
It continues from where the previous vectorization left off (starting at index 100,000).
"""

# Core Python imports
import os
import pickle
import time
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

# Connect to existing ChromaDB client
client = chromadb.PersistentClient(path=DB)

# Get the existing collection (don't create new one)
collection_name = "products"

try:
    collection = client.get_collection(collection_name)
    current_count = collection.count()
    print(f"Connected to existing collection: {collection_name}")
    print(f"Current collection size: {current_count}")
except Exception as e:
    print(f"Error: Could not connect to existing collection: {e}")
    print("Please run the main vectorize.py script first to create the initial collection")
    exit(1)

# ================================
# EMBEDDING MODEL SETUP
# ================================

# Initialize the sentence transformer model (should be cached from previous run)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Initialized SentenceTransformer model (using cached version)")

# ================================
# INCREMENTAL PROCESSING CONFIGURATION
# ================================

# Continue from where we left off
START_INDEX = 100000  # We already processed 0-99,999
TOTAL_DOCUMENTS = len(train)
REMAINING_DOCUMENTS = TOTAL_DOCUMENTS - START_INDEX

print(f"Starting incremental processing from index {START_INDEX}")
print(f"Processing remaining {REMAINING_DOCUMENTS} documents")

# ================================
# INCREMENTAL VECTOR GENERATION AND STORAGE
# ================================

# Process documents in batches for memory efficiency
BATCH_SIZE = 1000

print("Starting incremental batch processing...")
start_time = time.time()

for i in tqdm(range(START_INDEX, TOTAL_DOCUMENTS, BATCH_SIZE), desc="Processing remaining batches"):
    # Calculate the end index for this batch
    batch_end = min(i + BATCH_SIZE, TOTAL_DOCUMENTS)
    current_batch = train[i:batch_end]
    
    # Generate text descriptions for each item in the batch
    documents = [description(item) for item in current_batch]
    
    # Create embeddings for all documents in the batch
    # Convert to float and list format required by ChromaDB
    vectors = model.encode(documents).astype(float).tolist()
    
    # Prepare metadata for each document (category and price information)
    metadatas = [{"category": item.category, "price": item.price} for item in current_batch]
    
    # Generate unique IDs for each document (continuing from where we left off)
    ids = [f"doc_{j}" for j in range(i, i + len(documents))]
    
    # Add the batch to the existing ChromaDB collection
    collection.add(
        ids=ids,                # Unique identifiers for each document
        documents=documents,     # Original text descriptions
        embeddings=vectors,      # Vector embeddings for similarity search
        metadatas=metadatas     # Additional searchable metadata
    )

# ================================
# COMPLETION SUMMARY
# ================================

end_time = time.time()
processing_time = end_time - start_time
final_count = collection.count()

print(f"\n{'='*50}")
print(f"INCREMENTAL PROCESSING COMPLETE!")
print(f"{'='*50}")
print(f"Documents processed: {REMAINING_DOCUMENTS:,}")
print(f"Processing time: {processing_time:.2f} seconds ({processing_time/60:.1f} minutes)")
print(f"Final collection size: {final_count:,}")
print(f"Average processing speed: {REMAINING_DOCUMENTS/processing_time:.1f} docs/second")
print(f"Vector database location: {DB}")
print(f"{'='*50}") 