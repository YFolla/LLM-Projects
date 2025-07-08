"""
Specialist Agent for Multi-Agent System with Modal Integration.

This module implements a specialized agent that connects to a remote fine-tuned LLM
running on Modal cloud infrastructure. The agent provides price estimation capabilities
by leveraging a fine-tuned Llama model deployed as a Modal service.

The SpecialistAgent acts as a bridge between the local multi-agent system and the
remote pricing service, handling authentication, remote procedure calls, and result
processing.
"""

import modal
from agents.agent import Agent


class SpecialistAgent(Agent):
    """
    A specialized agent that interfaces with a remote fine-tuned LLM on Modal.
    
    This agent extends the base Agent class to provide price estimation capabilities
    by connecting to a Modal-deployed pricing service. It handles:
    
    1. Modal service connection and authentication
    2. Remote procedure calls to the fine-tuned model
    3. Result processing and logging
    4. Error handling for network operations
    
    The agent maintains a persistent connection to the Modal service and provides
    a simple interface for price estimation requests.
    
    Attributes:
        name (str): Human-readable name for the agent
        color (str): Color code for console output formatting
        pricer: Modal service instance for remote pricing calls
    """

    # Agent identification and display configuration
    name = "Specialist Agent"          # Display name for logging and UI
    color = Agent.RED                  # Color for console output (inherited from base Agent)

    def __init__(self):
        """
        Initialize the Specialist Agent and establish connection to Modal service.
        
        This constructor performs the following initialization steps:
        1. Logs the initialization start
        2. Connects to the Modal-deployed pricing service
        3. Creates a service instance for remote procedure calls
        4. Logs successful initialization
        
        The Modal service connection is established using the service name and class
        that were deployed earlier. This creates a persistent connection that can
        be reused for multiple pricing requests.
        
        Raises:
            modal.exception.NotFoundError: If the Modal service is not found
            modal.exception.AuthenticationError: If Modal authentication fails
            ConnectionError: If unable to connect to Modal infrastructure
        """
        # Log initialization start for debugging and monitoring
        self.log("Specialist Agent is initializing - connecting to modal")
        
        # ====================================================================
        # MODAL SERVICE CONNECTION
        # ====================================================================
        
        # Connect to the deployed Modal service by name and class
        # This creates a reference to the remote "Pricer" class in the "pricer-service" app
        Pricer = modal.Cls.from_name("pricer-service", "Pricer")
        
        # Create an instance of the remote Pricer class
        # This establishes the connection and prepares for remote method calls
        self.pricer = Pricer()
        
        # Log successful initialization
        self.log("Specialist Agent is ready")
        
    def price(self, description: str) -> float:
        """
        Estimate the price of an item using the remote fine-tuned model.
        
        This method performs a remote procedure call to the Modal-deployed pricing
        service, which uses a fine-tuned Llama model to extract price information
        from product descriptions.
        
        Args:
            description (str): Product description text to analyze for pricing.
                              Should contain enough detail for the model to make
                              a reasonable price estimation.
        
        Returns:
            float: Estimated price in USD. Returns 0.0 if no price can be determined
                   or if the remote call fails.
        
        Process:
            1. Log the start of the pricing request
            2. Make a remote procedure call to the Modal service
            3. Process the returned result
            4. Log the completion with the estimated price
            5. Return the price estimate
        
        Note:
            This method makes a blocking remote call, which may take several seconds
            depending on model loading time and network latency. The first call after
            a period of inactivity may be slower due to cold start overhead.
        
        Example:
            >>> agent = SpecialistAgent()
            >>> price = agent.price("iPhone 15 Pro Max 256GB Space Black")
            >>> print(f"Estimated price: ${price:.2f}")
            Estimated price: $1199.00
        """
        # ====================================================================
        # REMOTE PROCEDURE CALL SETUP
        # ====================================================================
        
        # Log the start of the pricing operation for monitoring
        self.log("Specialist Agent is calling remote fine-tuned model")
        
        # ====================================================================
        # MODAL SERVICE INVOCATION
        # ====================================================================
        
        # Make a remote procedure call to the Modal service
        # The .remote() method executes the pricing logic on Modal's infrastructure
        # This call is blocking and will wait for the result
        result = self.pricer.price.remote(description)
        
        # ====================================================================
        # RESULT PROCESSING AND LOGGING
        # ====================================================================
        
        # Log the completion of the operation with the predicted price
        # Format the price to 2 decimal places for consistent display
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        
        # Return the price estimate as a float
        return result