#!/usr/bin/env python3
"""
Gradio Web Interface for "The Price is Right" Multi-Agent Deal Discovery System

This module provides a web-based user interface for the multi-agent deal hunting system.
The interface allows users to monitor autonomous agents that continuously discover and
analyze online deals, providing real-time updates on the best opportunities found.

Key Components:
- Multi-agent framework coordination (deal discovery, pricing, messaging)
- Real-time deal monitoring with automatic refresh
- Interactive deal selection for notifications
- Beautiful, responsive web interface using Gradio
- Live logging and progress tracking

The system integrates multiple specialized agents:
1. Scanner Agent: Scrapes RSS feeds for deals
2. Ensemble Agent: Combines 3 pricing models for accuracy
3. Messaging Agent: Sends notifications via Pushover API
4. Planning Agent: Orchestrates the entire workflow

Architecture:
Web Interface (Gradio) -> Deal Agent Framework -> Planning Agent -> [Scanner, Ensemble, Messaging] Agents
"""

import gradio as gr
import logging
import io
import sys
import threading
import time
from datetime import datetime
from deal_agent_framework import DealAgentFramework
from agents.deals import Opportunity, Deal


class LogCapture:
    """
    Custom logging handler that captures log messages for display in the UI.
    
    This class intercepts log messages from the agent system and stores them
    in a buffer that can be displayed in the Gradio interface, providing
    real-time visibility into system operations.
    """
    
    def __init__(self):
        """Initialize the log capture system."""
        self.logs = []
        self.max_logs = 100  # Keep last 100 log entries
        self.lock = threading.Lock()
        
    def add_log(self, message: str, level: str = "INFO"):
        """
        Add a log message to the capture buffer.
        
        Args:
            message (str): The log message to add
            level (str): Log level (INFO, WARNING, ERROR, etc.)
        """
        with self.lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] [{level}] {message}"
            self.logs.append(log_entry)
            
            # Keep only the most recent logs
            if len(self.logs) > self.max_logs:
                self.logs.pop(0)
    
    def get_logs(self) -> str:
        """
        Get all captured logs as a formatted string.
        
        Returns:
            str: Formatted log messages for display
        """
        with self.lock:
            return "\n".join(self.logs) if self.logs else "No logs yet..."
    
    def clear_logs(self):
        """Clear all captured logs."""
        with self.lock:
            self.logs.clear()


class CustomLogHandler(logging.Handler):
    """
    Custom logging handler that sends messages to our LogCapture instance.
    """
    
    def __init__(self, log_capture: LogCapture):
        super().__init__()
        self.log_capture = log_capture
        
    def emit(self, record):
        """
        Handle a log record by sending it to the log capture.
        
        Args:
            record: The log record to handle
        """
        try:
            message = self.format(record)
            self.log_capture.add_log(message, record.levelname)
        except Exception:
            # Don't let logging errors break the application
            pass


class App:
    """
    Main application class that creates and manages the Gradio web interface.
    
    This class handles the complete user interface lifecycle, including:
    - Initial system setup and agent initialization
    - Periodic deal discovery execution
    - Real-time display updates
    - User interactions and notifications
    - Live logging and progress tracking
    
    The interface provides a clean, modern UI that displays discovered deals
    in a tabular format with automatic refresh functionality and real-time
    logging for transparency.
    
    Attributes:
        agent_framework (DealAgentFramework): The core multi-agent system instance
        log_capture (LogCapture): System for capturing and displaying logs
        is_running (bool): Flag indicating if a discovery cycle is in progress
        last_update (str): Timestamp of the last successful update
    """

    def __init__(self):
        """
        Initialize the application with no active agent framework.
        
        The agent framework is initialized lazily when the interface starts
        to optimize startup time and resource usage.
        """
        # Agent framework is initialized on first UI load for better performance
        self.agent_framework = None
        
        # Initialize logging capture system
        self.log_capture = LogCapture()
        self.is_running = False
        self.last_update = "Never"
        
        # Set up custom logging handler
        self._setup_logging()
        
    def _setup_logging(self):
        """
        Set up custom logging to capture agent system logs.
        
        This method configures the logging system to send all log messages
        to our custom handler, which displays them in the UI.
        """
        # Create custom handler
        handler = CustomLogHandler(self.log_capture)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(name)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        
        # Initial log message
        self.log_capture.add_log("🚀 Application starting up...", "INFO")

    def run(self):
        """
        Create and launch the Gradio web interface.
        
        This method sets up the complete UI including:
        1. Header with system description
        2. Real-time deal monitoring table
        3. Live logging display
        4. Status indicators and progress tracking
        5. Automatic refresh timer
        6. Interactive deal selection capabilities
        
        The interface uses a clean, professional design with:
        - Responsive layout that adapts to different screen sizes
        - Color-coded deal information
        - Real-time status updates
        - Automatic updates every 60 seconds
        - Click-to-select functionality for deal notifications
        """
        
        # Create the main Gradio interface with custom styling
        with gr.Blocks(title="The Price is Right", fill_width=True) as ui:
            
            # ================================================================
            # INTERNAL HELPER FUNCTIONS
            # ================================================================
            
            def table_for(opportunities):
                """
                Convert list of opportunities to table format for Gradio display.
                
                This function transforms the structured Opportunity objects into
                a format suitable for the Gradio Dataframe component, extracting
                key information and formatting prices with proper currency symbols.
                
                Args:
                    opportunities (List[Opportunity]): List of deal opportunities
                    
                Returns:
                    List[List[str]]: Formatted table rows with columns:
                        - Product Description
                        - Deal Price (formatted as currency)
                        - Estimated Value (formatted as currency)
                        - Discount Amount (formatted as currency)
                        - Deal URL
                """
                return [[
                    opp.deal.product_description,    # Product details
                    f"${opp.deal.price:.2f}",       # Current deal price
                    f"${opp.estimate:.2f}",         # AI-estimated value
                    f"${opp.discount:.2f}",         # Calculated discount
                    opp.deal.url                     # Link to deal page
                ] for opp in opportunities]
            
            def get_status_info():
                """
                Get current system status information.
                
                Returns:
                    tuple: (status_message, is_running_flag, last_update_time)
                """
                if self.is_running:
                    return "🔄 Discovering deals...", True, self.last_update
                else:
                    return "✅ Ready", False, self.last_update
            
            # ================================================================
            # AGENT SYSTEM INTERFACE FUNCTIONS
            # ================================================================
            
            def start():
                """
                Initialize the multi-agent system and load existing opportunities.
                
                This function is called when the Gradio interface first loads.
                It performs the following initialization steps:
                1. Creates the DealAgentFramework instance
                2. Initializes all required agents (Scanner, Ensemble, Messaging)
                3. Loads persistent memory of previously discovered deals
                4. Formats the data for initial display
                
                The lazy initialization approach ensures the UI loads quickly
                while the agents are being set up in the background.
                
                Returns:
                    tuple: (table_data, status_message, logs, last_update)
                """
                self.log_capture.add_log("🔧 Initializing multi-agent system...", "INFO")
                
                try:
                    # Initialize the core multi-agent framework
                    self.agent_framework = DealAgentFramework()
                    self.log_capture.add_log("✅ DealAgentFramework created", "INFO")
                    
                    # Initialize all agent components as needed
                    # This creates: PlanningAgent -> [ScannerAgent, EnsembleAgent, MessagingAgent]
                    self.log_capture.add_log("🤖 Initializing agents...", "INFO")
                    self.agent_framework.init_agents_as_needed()
                    self.log_capture.add_log("✅ All agents initialized successfully", "INFO")
                    
                    # Load existing opportunities from persistent storage
                    opportunities = self.agent_framework.memory
                    self.log_capture.add_log(f"📚 Loaded {len(opportunities)} opportunities from memory", "INFO")
                    
                    # Format opportunities for display in the UI table
                    table = table_for(opportunities)
                    
                    # Update status
                    self.last_update = datetime.now().strftime("%H:%M:%S")
                    status_msg, _, _ = get_status_info()
                    
                    self.log_capture.add_log("🎉 System initialization complete!", "INFO")
                    
                    return table, status_msg, self.log_capture.get_logs(), self.last_update
                    
                except Exception as e:
                    error_msg = f"❌ Initialization failed: {str(e)}"
                    self.log_capture.add_log(error_msg, "ERROR")
                    return [], "❌ Initialization failed", self.log_capture.get_logs(), "Error"
            
            def go():
                """
                Execute a complete deal discovery cycle and update the display.
                
                This function is called periodically (every 60 seconds) to run
                the full agent workflow:
                1. Scanner Agent: Scrapes RSS feeds for new deals
                2. Ensemble Agent: Estimates prices using 3 ML models
                3. Evaluates deals against discount threshold ($30)
                4. Messaging Agent: Sends notifications for qualifying deals
                5. Updates persistent memory with new opportunities
                
                The function provides a seamless user experience by automatically
                refreshing the display with newly discovered deals.
                
                Returns:
                    tuple: (table_data, status_message, logs, last_update)
                """
                if not self.agent_framework:
                    self.log_capture.add_log("⚠️ System not initialized yet", "WARNING")
                    return [], "⚠️ Not initialized", self.log_capture.get_logs(), "Error"
                
                self.is_running = True
                self.log_capture.add_log("🔄 Starting deal discovery cycle...", "INFO")
                
                try:
                    # Execute the complete multi-agent workflow
                    self.log_capture.add_log("🕷️ Scanner Agent: Searching for deals...", "INFO")
                    self.agent_framework.run()
                    
                    # Retrieve updated opportunities including any new discoveries
                    new_opportunities = self.agent_framework.memory
                    self.log_capture.add_log(f"📊 Discovery complete! Found {len(new_opportunities)} total opportunities", "INFO")
                    
                    # Format updated data for display
                    table = table_for(new_opportunities)
                    
                    # Update status
                    self.last_update = datetime.now().strftime("%H:%M:%S")
                    self.is_running = False
                    
                    status_msg, _, _ = get_status_info()
                    
                    return table, status_msg, self.log_capture.get_logs(), self.last_update
                    
                except Exception as e:
                    error_msg = f"❌ Discovery cycle failed: {str(e)}"
                    self.log_capture.add_log(error_msg, "ERROR")
                    self.is_running = False
                    return [], "❌ Discovery failed", self.log_capture.get_logs(), "Error"
            
            def do_select(selected_index: gr.SelectData):
                """
                Handle user selection of a deal for manual notification.
                
                This function is triggered when a user clicks on a row in the
                deals table. It allows users to manually trigger notifications
                for deals that may not have met the automatic threshold but
                are still of interest.
                
                Args:
                    selected_index (gr.SelectData): Gradio selection event containing
                                                   the row index of the selected deal
                
                Returns:
                    str: Updated logs with notification status
                """
                if not self.agent_framework:
                    self.log_capture.add_log("⚠️ Cannot send notification - system not initialized", "WARNING")
                    return self.log_capture.get_logs()
                
                try:
                    # Get the current list of opportunities from memory
                    opportunities = self.agent_framework.memory
                    
                    # Extract the row index from the selection event
                    row = selected_index.index[0]
                    
                    # Get the specific opportunity that was selected
                    opportunity = opportunities[row]
                    
                    self.log_capture.add_log(f"📱 Sending notification for: {opportunity.deal.product_description[:50]}...", "INFO")
                    
                    # Send notification for the selected deal via Messaging Agent
                    # This uses the same notification system as automatic alerts
                    self.agent_framework.planner.messenger.alert(opportunity)
                    
                    self.log_capture.add_log("✅ Notification sent successfully!", "INFO")
                    
                except Exception as e:
                    error_msg = f"❌ Failed to send notification: {str(e)}"
                    self.log_capture.add_log(error_msg, "ERROR")
                
                return self.log_capture.get_logs()
            
            def clear_logs():
                """
                Clear the log display.
                
                Returns:
                    str: Empty log display
                """
                self.log_capture.clear_logs()
                self.log_capture.add_log("🧹 Logs cleared", "INFO")
                return self.log_capture.get_logs()
        
            # ================================================================
            # USER INTERFACE LAYOUT
            # ================================================================
            
            # Main header with system branding and description
            with gr.Row():
                gr.Markdown('''
                    <div style="text-align: center; font-size: 24px; font-weight: bold; color: #2c3e50;">
                        "The Price is Right" - Deal Hunting Agentic AI
                    </div>
                ''')
            
            # System description with technical details
            with gr.Row():
                gr.Markdown('''
                    <div style="text-align: center; font-size: 14px; color: #7f8c8d; margin-bottom: 10px;">
                        Autonomous agent framework that finds online deals, collaborating with a proprietary 
                        fine-tuned LLM deployed on Modal, and a RAG pipeline with a frontier model and Chroma.
                    </div>
                ''')
            
            # Status and control panel
            with gr.Row():
                with gr.Column(scale=2):
                    status_display = gr.Textbox(
                        label="System Status",
                        value="🔄 Initializing...",
                        interactive=False,
                        container=True
                    )
                with gr.Column(scale=1):
                    last_update_display = gr.Textbox(
                        label="Last Update",
                        value="Never",
                        interactive=False,
                        container=True
                    )
                with gr.Column(scale=1):
                    clear_logs_btn = gr.Button("Clear Logs", variant="secondary")
            
            # Instructions for user interaction
            with gr.Row():
                gr.Markdown('''
                    <div style="text-align: center; font-size: 14px; color: #34495e; margin-bottom: 15px;">
                        <strong>Deals surfaced so far:</strong> Click on any row to send a notification for that deal.
                    </div>
                ''')
            
            # Main content area with deals table and logs
            with gr.Row():
                with gr.Column(scale=2):
                    # Interactive dataframe for displaying deal opportunities
                    opportunities_dataframe = gr.Dataframe(
                        headers=["Description", "Price", "Estimate", "Discount", "URL"],
                        wrap=True,                    # Enable text wrapping for long descriptions
                        column_widths=[4, 1, 1, 1, 2],  # Proportional column widths
                        row_count=10,                 # Show up to 10 deals at once
                        col_count=5,                  # Fixed number of columns
                        max_height=400,               # Limit table height with scrolling
                        interactive=True              # Enable row selection
                    )
                
                with gr.Column(scale=1):
                    # Live logging display
                    logs_display = gr.Textbox(
                        label="System Logs",
                        value="Starting up...",
                        lines=20,
                        max_lines=20,
                        interactive=False,
                        container=True,
                        show_copy_button=True
                    )
            
            # ================================================================
            # EVENT HANDLERS AND AUTOMATION
            # ================================================================
            
            # Initialize the system when the interface loads
            # This triggers the start() function to set up agents and load existing data
            ui.load(
                start, 
                inputs=[], 
                outputs=[opportunities_dataframe, status_display, logs_display, last_update_display]
            )

            # Set up automatic refresh every 60 seconds
            # This creates a persistent background process that runs deal discovery
            timer = gr.Timer(value=60)  # 60-second interval
            timer.tick(
                go, 
                inputs=[], 
                outputs=[opportunities_dataframe, status_display, logs_display, last_update_display]
            )

            # Enable row selection for manual notifications
            # Users can click on any deal row to trigger a notification
            opportunities_dataframe.select(
                do_select,
                outputs=[logs_display]
            )
            
            # Clear logs button functionality
            clear_logs_btn.click(
                clear_logs,
                outputs=[logs_display]
            )
        
        # ================================================================
        # LAUNCH CONFIGURATION
        # ================================================================
        
        # Launch the web interface with optimized settings
        ui.launch(
            share=False,     # Don't create public tunnel (for security)
            inbrowser=True   # Automatically open in default browser
        )


# ========================================================================
# APPLICATION ENTRY POINT
# ========================================================================

if __name__ == "__main__":
    """
    Main entry point for the Deal Hunting Agentic AI web interface.
    
    This script can be run directly to start the web application:
    python gradio_interface.py
    
    The application will:
    1. Initialize the multi-agent system
    2. Load any existing deal history
    3. Launch the web interface in the default browser
    4. Begin continuous deal monitoring and discovery
    5. Provide real-time logging and status updates
    
    Requirements:
    - All agent dependencies must be properly installed
    - Environment variables must be configured (.env file)
    - ChromaDB vector database must be available
    - OpenAI API key must be valid
    - Modal service must be deployed and accessible
    """
    # Create and run the application instance
    App().run()