"""
GPT-based RAG (Retrieval-Augmented Generation) System for Price Estimation

This script implements a RAG system that uses ChromaDB for vector storage and retrieval,
combined with OpenAI's GPT-4o-mini model to estimate product prices based on similar items.

The system works by:
1. Vectorizing product descriptions using sentence transformers
2. Finding similar products from a vector database
3. Using GPT-4o-mini to estimate prices based on similar items as context
"""

# Core Python libraries
import os
import re
import pickle

# Third-party libraries
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

# Local imports
from testing import Tester
from items import Item

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Load environment variables from .env file
load_dotenv(override=True)

# Set up API keys for OpenAI and HuggingFace
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# Authenticate with HuggingFace Hub
hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)

# Initialize OpenAI client
openai = OpenAI()

# ============================================================================
# DATA LOADING
# ============================================================================

# Load test data from pickle file
# Note: The test.pkl file contains product items for price estimation testing
# See the "Back to the PKL files" section in the day2.0 notebook for details
with open('../13_Frontier_finetuning/test.pkl', 'rb') as file:
    test = pickle.load(file)

# ============================================================================
# RAG SYSTEM COMPONENTS
# ============================================================================

# ChromaDB configuration
DB = "products_vectorstore"  # Database path for vector storage

# Initialize ChromaDB client and collection
client = chromadb.PersistentClient(path=DB)
collection = client.get_or_create_collection('products')

# Initialize sentence transformer model for text embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def make_context(similars, prices):
    """
    Create contextual message with similar products and their prices.
    
    Args:
        similars (list): List of similar product descriptions
        prices (list): List of corresponding prices
        
    Returns:
        str: Formatted context message for the GPT model
    """
    message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
    
    for similar, price in zip(similars, prices):
        message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
    
    return message

def messages_for(item, similars, prices):
    """
    Construct the message array for GPT API call.
    
    Args:
        item: The item object to estimate price for
        similars (list): List of similar product descriptions
        prices (list): List of corresponding prices
        
    Returns:
        list: Formatted messages for OpenAI Chat API
    """
    system_message = "You estimate prices of items. Reply only with the price, no explanation"
    
    # Build user prompt with context and the item to estimate
    user_prompt = make_context(similars, prices)
    user_prompt += "And now the question for you:\n\n"
    
    # Clean the item's test prompt by removing price-related text
    cleaned_prompt = item.test_prompt().replace(" to the nearest dollar","").replace("\n\nPrice is $","")
    user_prompt += cleaned_prompt
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "Price is $"}
    ]

def description(item):
    """
    Extract clean product description from item prompt.
    
    Args:
        item: Item object containing the prompt
        
    Returns:
        str: Clean product description without price information
    """
    # Remove the question prompt and price information
    text = item.prompt.replace("How much does this cost to the nearest dollar?\n\n", "")
    return text.split("\n\nPrice is $")[0]

def vector(item):
    """
    Generate vector embedding for an item's description.
    
    Args:
        item: Item object to vectorize
        
    Returns:
        numpy.ndarray: Vector embedding of the item description
    """
    return model.encode([description(item)])

def find_similars(item):
    """
    Find similar products from the vector database.
    
    Args:
        item: Item object to find similars for
        
    Returns:
        tuple: (documents, prices) - Similar product descriptions and their prices
    """
    # Query the vector database for similar items
    results = collection.query(
        query_embeddings=vector(item).astype(float).tolist(), 
        n_results=5
    )
    
    # Extract documents and prices from query results
    documents = results['documents'][0][:]
    prices = [m['price'] for m in results['metadatas'][0][:]]
    
    return documents, prices

def get_price(s):
    """
    Extract numerical price from a string response.
    
    Args:
        s (str): String containing price information
        
    Returns:
        float: Extracted price value, or 0 if no valid price found
    """
    # Remove currency symbols and commas
    s = s.replace('$','').replace(',','')
    
    # Find first number (integer or decimal) in the string
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    
    return float(match.group()) if match else 0

# ============================================================================
# MAIN RAG FUNCTION
# ============================================================================

def gpt_4o_mini_rag(item):
    """
    Main RAG function that estimates item price using GPT-4o-mini with context.
    
    This function implements the complete RAG pipeline:
    1. Find similar items from vector database
    2. Create context with similar items and their prices
    3. Query GPT-4o-mini for price estimation
    4. Extract and return the estimated price
    
    Args:
        item: Item object to estimate price for
        
    Returns:
        float: Estimated price for the item
    """
    # Step 1: Retrieve similar products and their prices
    documents, prices = find_similars(item)
    
    # Step 2: Generate price estimation using GPT-4o-mini
    response = openai.chat.completions.create(
        model="gpt-4o-mini", 
        messages=messages_for(item, documents, prices),
        seed=42,          # For reproducible results
        max_tokens=5      # Limit response to just the price
    )
    
    # Step 3: Extract the response and convert to price
    reply = response.choices[0].message.content
    return get_price(reply)

# ============================================================================
# TESTING
# ============================================================================

# Run the RAG system test
Tester.test(gpt_4o_mini_rag, test)