"""
Frontier Agent - OpenAI GPT-4o-mini with RAG for Price Estimation

This agent uses OpenAI's GPT-4o-mini model combined with Retrieval-Augmented Generation (RAG)
to estimate product prices. It searches for similar products in a ChromaDB vector database
and uses them as context to improve price estimation accuracy.

Key Features:
- Vector similarity search using sentence transformers
- Context-aware price estimation with similar products
- OpenAI GPT-4o-mini integration for natural language processing
"""

import os
import re
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from agents.agent import Agent


class FrontierAgent(Agent):
    """
    Frontier Agent that uses OpenAI GPT-4o-mini with RAG for price estimation.
    
    This agent combines vector similarity search with language model inference
    to provide accurate price estimates based on similar product context.
    """

    name = "Frontier Agent"
    color = Agent.BLUE
    MODEL = "gpt-4o-mini"
    
    def __init__(self, collection):
        """
        Initialize the Frontier Agent with OpenAI client and ChromaDB collection.
        
        Args:
            collection: ChromaDB collection containing product embeddings and metadata
        """
        self.log("Initializing Frontier Agent")
        
        # Initialize OpenAI client
        self.client = OpenAI()
        self.log("Frontier Agent connected to OpenAI")
        
        # Set up ChromaDB collection and sentence transformer model
        self.collection = collection
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        self.log("Frontier Agent is ready")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        """
        Create contextual message with similar products and their prices.
        
        Args:
            similars: List of similar product descriptions
            prices: List of corresponding prices for similar products
            
        Returns:
            str: Formatted context message for the language model
        """
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
            
        return message

    def messages_for(self, description: str, similars: List[str], prices: List[float]) -> List[Dict[str, str]]:
        """
        Create the message array for OpenAI Chat API call.
        
        Args:
            description: Product description to estimate price for
            similars: List of similar product descriptions
            prices: List of corresponding prices for similar products
            
        Returns:
            List[Dict[str, str]]: Formatted messages for OpenAI Chat API
        """
        system_message = "You estimate prices of items. Reply only with the price, no explanation"
        
        # Build user prompt with context and target product
        user_prompt = self.make_context(similars, prices)
        user_prompt += "And now the question for you:\n\n"
        user_prompt += "How much does this cost?\n\n" + description
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Price is $"}
        ]

    def find_similars(self, description: str) -> tuple:
        """
        Find similar products from the ChromaDB vector database.
        
        Args:
            description: Product description to find similars for
            
        Returns:
            tuple: (documents, prices) - Similar product descriptions and their prices
        """
        self.log("Frontier Agent is performing a RAG search of the ChromaDB datastore to find 5 similar products")
        
        # Generate vector embedding for the description
        vector = self.model.encode([description])
        
        # Query ChromaDB for similar products
        results = self.collection.query(
            query_embeddings=vector.astype(float).tolist(), 
            n_results=5
        )
        
        # Extract documents and prices from results
        documents = results['documents'][0][:]
        prices = [m['price'] for m in results['metadatas'][0][:]]
        
        self.log("Frontier Agent has found similar products")
        return documents, prices

    def get_price(self, s: str) -> float:
        """
        Extract numerical price from a string response.
        
        Args:
            s: String containing price information
            
        Returns:
            float: Extracted price value, or 0.0 if no valid price found
        """
        # Remove currency symbols and commas
        s = s.replace('$', '').replace(',', '')
        
        # Find first number (integer or decimal) in the string
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        
        return float(match.group()) if match else 0.0

    def price(self, description: str) -> float:
        """
        Estimate the price of a product using OpenAI GPT-4o-mini with RAG context.
        
        This method implements the complete RAG pipeline:
        1. Find similar products from vector database
        2. Create context with similar products and prices
        3. Query GPT-4o-mini for price estimation
        4. Extract and return the estimated price
        
        Args:
            description: Product description to estimate price for
            
        Returns:
            float: Estimated price for the product
        """
        # Step 1: Retrieve similar products and their prices
        documents, prices = self.find_similars(description)
        
        # Step 2: Generate price estimation using GPT-4o-mini
        self.log(f"Frontier Agent is about to call {self.MODEL} with context including 5 similar products")
        
        response = self.client.chat.completions.create(
            model=self.MODEL, 
            messages=self.messages_for(description, documents, prices),
            seed=42,          # For reproducible results
            max_tokens=5      # Limit response to just the price
        )
        
        # Step 3: Extract the response and convert to price
        reply = response.choices[0].message.content
        result = self.get_price(reply)
        
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result
        