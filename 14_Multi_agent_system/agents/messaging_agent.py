import os
import http.client
import urllib
from agents.deals import Opportunity
from agents.agent import Agent

class MessagingAgent(Agent):
    """
    Agent responsible for sending push notifications about deal opportunities.
    Uses Pushover API for delivering notifications to mobile devices.
    """

    name = "Messaging Agent"
    color = Agent.WHITE

    def __init__(self):
        """
        Initialize the Messaging Agent with Pushover credentials.
        
        Retrieves Pushover user and token from environment variables.
        Falls back to placeholder values if environment variables are not set.
        """
        self.log(f"Messaging Agent is initializing")
        
        # Initialize Pushover credentials from environment variables
        self.pushover_user = os.getenv('PUSHOVER_USER')
        self.pushover_token = os.getenv('PUSHOVER_TOKEN')
        
        self.log("Messaging Agent has initialized Pushover")

    def push(self, text):
        """
        Send a push notification using the Pushover API.
        
        Args:
            text (str): The message content to send in the push notification
            
        The notification is sent with a "cash register" sound to indicate
        it's a deal alert notification.
        """
        self.log("Messaging Agent is sending a push notification")
        
        # Establish HTTPS connection to Pushover API
        conn = http.client.HTTPSConnection("api.pushover.net:443")
        
        # Send POST request with notification data
        conn.request("POST", "/1/messages.json",
          urllib.parse.urlencode({
            "token": self.pushover_token,      # API token for authentication
            "user": self.pushover_user,        # User key for message delivery
            "message": text,                   # Message content
            "sound": "cashregister"            # Sound to play for deal alerts
          }), { "Content-type": "application/x-www-form-urlencoded" })
        
        # Execute the request
        conn.getresponse()

    def alert(self, opportunity: Opportunity):
        """
        Send an alert notification about a deal opportunity.
        
        Args:
            opportunity (Opportunity): The deal opportunity to alert about
            
        Formats the opportunity details into a concise message and sends
        it via push notification to notify the user of the deal.
        """
        # Format the deal information into a concise alert message
        text = f"Deal Alert! Price=${opportunity.deal.price:.2f}, "
        text += f"Estimate=${opportunity.estimate:.2f}, "
        text += f"Discount=${opportunity.discount:.2f} :"
        text += opportunity.deal.product_description[:10]+'... '  # First 10 chars of description
        text += opportunity.deal.url
        
        # Send push notification
        self.push(text)
        
        self.log("Messaging Agent has completed")
        
        