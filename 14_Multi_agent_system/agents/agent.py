"""
Base Agent class for the Multi-Agent System.

This module provides the foundational Agent class that all specialized agents
inherit from. It includes logging functionality with color-coded output to
help distinguish between different agents in the system.
"""

import logging


class Agent:
    """
    Abstract base class for all agents in the multi-agent system.
    
    This class provides common functionality that all agents need:
    - Colored logging output to distinguish between agents
    - Consistent agent identification and naming
    - Standardized message formatting
    
    All specialized agents should inherit from this class and implement
    their specific functionality while using the provided logging methods.
    
    Attributes:
        name (str): Human-readable name for the agent
        color (str): ANSI color code for console output
    """

    # ========================================================================
    # ANSI COLOR CODES FOR CONSOLE OUTPUT
    # ========================================================================
    
    # Foreground colors for text
    RED = '\033[31m'        # Red text
    GREEN = '\033[32m'      # Green text
    YELLOW = '\033[33m'     # Yellow text
    BLUE = '\033[34m'       # Blue text
    MAGENTA = '\033[35m'    # Magenta text
    CYAN = '\033[36m'       # Cyan text
    WHITE = '\033[37m'      # White text
    
    # Background colors
    BG_BLACK = '\033[40m'   # Black background
    
    # Reset code to return to default terminal colors
    RESET = '\033[0m'       # Reset all formatting
    
    # ========================================================================
    # AGENT CONFIGURATION
    # ========================================================================
    
    # Default agent properties (should be overridden by subclasses)
    name: str = "Base Agent"     # Default name for the agent
    color: str = WHITE           # Default color (white text)

    def log(self, message: str) -> None:
        """
        Log a message with agent identification and color formatting.
        
        This method provides consistent logging across all agents with:
        - Agent name identification in brackets
        - Color-coded output for easy visual distinction
        - Black background for better readability
        - Automatic color reset after the message
        
        Args:
            message (str): The message to log
            
        Example:
            >>> agent = Agent()
            >>> agent.log("Agent is starting up")
            # Output: [Base Agent] Agent is starting up (in colored text)
        """
        # Combine background and foreground colors
        color_code = self.BG_BLACK + self.color
        
        # Format message with agent identification
        formatted_message = f"[{self.name}] {message}"
        
        # Log with color formatting and automatic reset
        logging.info(color_code + formatted_message + self.RESET) 