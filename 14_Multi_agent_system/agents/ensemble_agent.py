"""
Ensemble Agent - Multi-Agent Price Prediction System

This agent combines predictions from three specialist agents using a trained
Linear Regression model to provide more accurate and robust price estimates.

The ensemble approach leverages the strengths of each individual agent:
1. SpecialistAgent - Fine-tuned model for specialized price prediction
2. FrontierAgent - RAG-based agent using ChromaDB + OpenAI GPT-4o-mini  
3. RandomForestAgent - Random Forest model on product embeddings

The Linear Regression ensemble model learns optimal weights for combining
these predictions, often outperforming individual agents.
"""

import pandas as pd
import joblib
import os
import chromadb

from agents.agent import Agent
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.random_forest_agent import RandomForestAgent

class EnsembleAgent(Agent):
    """
    Ensemble Agent that combines predictions from multiple specialist agents.
    
    This agent uses a trained Linear Regression model to optimally weight
    predictions from three different pricing agents, creating a more robust
    and accurate price estimation system.
    """

    name = "Ensemble Agent"
    color = Agent.YELLOW
    
    def __init__(self, model_path='ensemble_model.pkl', chromadb_path='products_vectorstore'):
        """
        Initialize the Ensemble Agent by loading component agents and ensemble model.
        
        Args:
            model_path: Path to the trained ensemble model file
            chromadb_path: Path to ChromaDB database (only needed for FrontierAgent)
        """
        self.log("Initializing Ensemble Agent")
        
        try:
            # Initialize agents that don't need ChromaDB first
            self.log("Loading SpecialistAgent (Modal connection)...")
            self.specialist = SpecialistAgent()
            
            self.log("Loading RandomForestAgent (local model)...")
            self.random_forest = RandomForestAgent()
            
            # Set up ChromaDB only for FrontierAgent
            self.log("Setting up ChromaDB for FrontierAgent...")
            client = chromadb.PersistentClient(path=chromadb_path)
            collection = client.get_or_create_collection('products')
            
            self.log("Loading FrontierAgent (RAG + OpenAI)...")
            self.frontier = FrontierAgent(collection)
            
            # Load the trained ensemble model
            self.log(f"Loading ensemble model from {model_path}...")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Ensemble model not found at {model_path}")
            
            self.model = joblib.load(model_path)
            
            # Verify the model has the expected features
            expected_features = ['Specialist', 'Frontier', 'RandomForest', 'Min', 'Max']
            if hasattr(self.model, 'feature_names_in_'):
                model_features = list(self.model.feature_names_in_)
                if model_features != expected_features:
                    self.log(f"Warning: Model features {model_features} don't match expected {expected_features}")
            
            self.log("✅ Ensemble Agent initialized successfully")
            
        except Exception as e:
            self.log(f"❌ Error initializing Ensemble Agent: {str(e)}")
            raise

    def price(self, description: str) -> float:
        """
        Estimate the price of a product using the ensemble model.
        
        This method:
        1. Gets predictions from all three specialist agents
        2. Calculates derived features (min/max of predictions)
        3. Uses the trained Linear Regression model to combine predictions
        4. Returns the ensemble prediction (ensuring non-negative result)
        
        Args:
            description: Product description text
            
        Returns:
            float: Estimated price in dollars
        """
        try:
            self.log("🤖 Running Ensemble Agent - collaborating with specialist agents")
            
            # Get predictions from each specialist agent
            self.log("Collecting predictions from specialist agents...")
            specialist_pred = self.specialist.price(description)
            frontier_pred = self.frontier.price(description)
            random_forest_pred = self.random_forest.price(description)
            
            # Log individual predictions for transparency
            self.log(f"Individual predictions: Specialist=${specialist_pred:.2f}, "
                    f"Frontier=${frontier_pred:.2f}, RandomForest=${random_forest_pred:.2f}")
            
            # Calculate derived features
            min_pred = min(specialist_pred, frontier_pred, random_forest_pred)
            max_pred = max(specialist_pred, frontier_pred, random_forest_pred)
            
            # Create feature matrix for ensemble model
            # Note: Feature names must match those used during training
            feature_matrix = pd.DataFrame({
                'Specialist': [specialist_pred],
                'Frontier': [frontier_pred],
                'RandomForest': [random_forest_pred],
                'Min': [min_pred],
                'Max': [max_pred],
            })
            
            # Get ensemble prediction
            ensemble_prediction = self.model.predict(feature_matrix)[0]
            
            # Ensure non-negative price (prices can't be negative)
            final_price = max(0, ensemble_prediction)
            
            self.log(f"✅ Ensemble Agent complete - returning ${final_price:.2f}")
            return final_price
            
        except Exception as e:
            self.log(f"❌ Error in ensemble prediction: {str(e)}")
            # Fallback to average of individual predictions if ensemble fails
            try:
                fallback_price = (specialist_pred + frontier_pred + random_forest_pred) / 3
                self.log(f"🔄 Using fallback average price: ${fallback_price:.2f}")
                return max(0, fallback_price)
            except:
                self.log("❌ Complete prediction failure - returning default price")
                return 0.0

    def get_model_info(self) -> dict:
        """
        Get information about the ensemble model.
        
        Returns:
            dict: Model information including coefficients and features
        """
        try:
            info = {
                'model_type': type(self.model).__name__,
                'features': ['Specialist', 'Frontier', 'RandomForest', 'Min', 'Max'],
                'coefficients': {},
                'intercept': None
            }
            
            if hasattr(self.model, 'coef_'):
                for feature, coef in zip(info['features'], self.model.coef_):
                    info['coefficients'][feature] = float(coef)
                    
            if hasattr(self.model, 'intercept_'):
                info['intercept'] = float(self.model.intercept_)
                
            return info
            
        except Exception as e:
            self.log(f"Error getting model info: {str(e)}")
            return {}