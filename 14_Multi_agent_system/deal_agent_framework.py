#!/usr/bin/env python3
"""
Deal Agent Framework - Multi-Agent System for Deal Discovery and Analysis

This framework orchestrates multiple specialized agents to:
1. Discover deals from RSS feeds
2. Analyze and price products using machine learning models
3. Identify opportunities with significant discounts
4. Maintain persistent memory of discovered deals

The system uses ChromaDB for vector storage and supports visualization
of product embeddings in 3D space.
"""

import os
import sys
import logging
import json
from typing import List, Optional
from dotenv import load_dotenv
import chromadb
from agents.planning_agent import PlanningAgent
from agents.deals import Opportunity
from sklearn.manifold import TSNE
import numpy as np

# ANSI color codes for enhanced logging visibility
BG_BLUE = '\033[44m'    # Blue background
WHITE = '\033[37m'      # White text
RESET = '\033[0m'       # Reset to default

# Configuration for 3D visualization plot
# Product categories mapped to distinct colors for clear visualization
CATEGORIES = [
    'Appliances', 'Automotive', 'Cell_Phones_and_Accessories', 
    'Electronics', 'Musical_Instruments', 'Office_Products', 
    'Tools_and_Home_Improvement', 'Toys_and_Games'
]
COLORS = ['red', 'blue', 'brown', 'orange', 'yellow', 'green', 'purple', 'cyan']

def init_logging():
    """
    Initialize logging configuration for the agent framework.
    
    Sets up structured logging with timestamps and component identification
    to help track multi-agent system operations and debug issues.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    
    # Create console handler with custom formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

class DealAgentFramework:
    """
    Main orchestrator for the multi-agent deal discovery system.
    
    This class manages the lifecycle of the agent system, including:
    - Initialization of ChromaDB vector database
    - Persistent memory management for discovered opportunities
    - Coordination of the planning agent workflow
    - Data visualization utilities
    
    Attributes:
        DB (str): Path to ChromaDB persistent storage
        MEMORY_FILENAME (str): JSON file for opportunity persistence
        collection: ChromaDB collection for product embeddings
        memory: List of previously discovered opportunities
        planner: PlanningAgent instance for coordinating other agents
    """

    # Class constants for configuration
    DB = "products_vectorstore"     # ChromaDB database directory
    MEMORY_FILENAME = "memory.json" # Persistent memory storage file

    def __init__(self):
        """
        Initialize the Deal Agent Framework.
        
        Sets up logging, loads environment variables, initializes ChromaDB,
        and loads persistent memory from previous runs.
        """
        # Initialize logging system
        init_logging()
        
        # Load environment variables from .env file
        load_dotenv()
        
        # Initialize ChromaDB client with persistent storage
        client = chromadb.PersistentClient(path=self.DB)
        
        # Load persistent memory of previous opportunities
        self.memory = self.read_memory()
        
        # Get or create the products collection for vector storage
        self.collection = client.get_or_create_collection('products')
        
        # Defer planner initialization until needed (lazy loading)
        self.planner = None

    def init_agents_as_needed(self):
        """
        Initialize the agent system with lazy loading.
        
        Creates the PlanningAgent instance only when needed to optimize
        resource usage and startup time. The PlanningAgent coordinates
        Scanner, Ensemble, and Messaging agents.
        """
        if not self.planner:
            self.log("Initializing Agent Framework")
            # Create planning agent with default model and database paths
            self.planner = PlanningAgent()
            self.log("Agent Framework is ready")
        
    def read_memory(self) -> List[Opportunity]:
        """
        Load persistent memory from JSON file.
        
        Reads previously discovered opportunities from disk to maintain
        state across framework restarts. This prevents rediscovering
        the same deals and enables tracking of deal patterns over time.
        
        Returns:
            List[Opportunity]: List of previously discovered opportunities,
                              empty list if no memory file exists
        """
        if os.path.exists(self.MEMORY_FILENAME):
            try:
                with open(self.MEMORY_FILENAME, "r") as file:
                    data = json.load(file)
                # Reconstruct Opportunity objects from saved data
                opportunities = [Opportunity(**item) for item in data]
                self.log(f"Loaded {len(opportunities)} opportunities from memory")
                return opportunities
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                self.log(f"Error loading memory file: {e}")
                return []
        return []

    def write_memory(self) -> None:
        """
        Persist current memory to JSON file.
        
        Saves all discovered opportunities to disk for persistence across
        framework restarts. Uses Pydantic's dict() method for clean
        serialization of opportunity objects.
        """
        try:
            # Convert Opportunity objects to dictionaries for JSON serialization
            data = [opportunity.dict() for opportunity in self.memory]
            with open(self.MEMORY_FILENAME, "w") as file:
                json.dump(data, file, indent=2)
            self.log(f"Saved {len(self.memory)} opportunities to memory")
        except (IOError, TypeError) as e:
            self.log(f"Error saving memory file: {e}")

    def log(self, message: str):
        """
        Log a message with framework-specific formatting.
        
        Adds visual distinction to framework messages using ANSI colors
        and consistent labeling for easy identification in logs.
        
        Args:
            message (str): The message to log
        """
        # Format message with blue background and white text
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)

    def run(self) -> List[Opportunity]:
        """
        Execute the main deal discovery workflow.
        
        This method orchestrates the complete deal discovery process:
        1. Initialize agents if needed
        2. Execute planning agent workflow
        3. Save any new opportunities to memory
        4. Return all discovered opportunities
        
        Returns:
            List[Opportunity]: All opportunities in memory, including any
                              newly discovered ones from this run
        """
        # Ensure agents are initialized
        self.init_agents_as_needed()
        
        # Execute the planning workflow
        logging.info("Kicking off Planning Agent")
        result = self.planner.plan(memory=self.memory)
        logging.info(f"Planning Agent has completed and returned: {result}")
        
        # Save any new opportunities to memory
        if result:
            self.memory.append(result)
            self.write_memory()
            self.log(f"Added new opportunity with discount: ${result.discount:.2f}")
        
        return self.memory

    @classmethod
    def get_plot_data(cls, max_datapoints=10000):
        """
        Generate data for 3D visualization of product embeddings.
        
        Retrieves product embeddings from ChromaDB and applies t-SNE
        dimensionality reduction to enable 3D visualization of product
        relationships in embedding space.
        
        Args:
            max_datapoints (int): Maximum number of products to include
                                 in visualization (default: 10000)
        
        Returns:
            tuple: (documents, reduced_vectors, colors) where:
                - documents: List of product descriptions
                - reduced_vectors: 3D coordinates from t-SNE reduction
                - colors: Color codes for category visualization
        """
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=cls.DB)
        collection = client.get_or_create_collection('products')
        
        # Retrieve embeddings and metadata
        result = collection.get(
            include=['embeddings', 'documents', 'metadatas'], 
            limit=max_datapoints
        )
        
        # Extract data for processing
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        categories = [metadata['category'] for metadata in result['metadatas']]
        
        # Map categories to colors for visualization
        colors = [COLORS[CATEGORIES.index(c)] for c in categories]
        
        # Apply t-SNE dimensionality reduction for 3D visualization
        # n_jobs=-1 uses all available CPU cores for faster processing
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        reduced_vectors = tsne.fit_transform(vectors)
        
        return documents, reduced_vectors, colors


if __name__ == "__main__":
    # Entry point for running the deal agent framework
    # Creates an instance and executes the main workflow
    framework = DealAgentFramework()
    opportunities = framework.run()
    
    # Log summary of results
    if opportunities:
        best_discount = max(opportunities, key=lambda x: x.discount)
        print(f"\nFramework completed successfully!")
        print(f"Total opportunities discovered: {len(opportunities)}")
        print(f"Best discount found: ${best_discount.discount:.2f}")
    else:
        print("\nFramework completed - no new opportunities discovered this run.")