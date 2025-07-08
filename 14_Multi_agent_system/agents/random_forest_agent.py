# Essential imports for the RandomForestAgent
import os
from sentence_transformers import SentenceTransformer
import joblib
from agents.agent import Agent



class RandomForestAgent(Agent):

    name = "Random Forest Agent"
    color = Agent.MAGENTA

    def __init__(self):
        """
        Initialize this object by loading in the saved model weights
        and the SentenceTransformer vector encoding model
        """
        self.log("Random Forest Agent is initializing")
        
        # Initialize the same SentenceTransformer model used for training
        self.vectorizer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load the saved Random Forest model
        # Use relative path to go up one directory to find the model file
        model_path = os.path.join(os.path.dirname(__file__), '..', 'random_forest_model.pkl')
        self.model = joblib.load(model_path)
        
        self.log("Random Forest Agent is ready")

    def price(self, description: str) -> float:
        """
        Use a Random Forest model to estimate the price of the described item
        
        Args:
            description: the product description to be estimated
            
        Returns:
            float: the predicted price as a float
        """        
        self.log("Random Forest Agent is starting a prediction")
        
        # Convert description to embedding using the same model as training
        vector = self.vectorizer.encode([description])
        
        # Use the trained Random Forest model to predict price
        result = max(0, self.model.predict(vector)[0])
        
        self.log(f"Random Forest Agent completed - predicting ${result:.2f}")
        return result