#!/usr/bin/env python3
"""
Script to run live data tests for the PlanningAgent.

This script demonstrates how to run the complete test suite including live data tests
that make actual API calls. It includes proper environment setup and cost warnings.

Usage:
    python run_live_tests.py

Environment Variables Required:
    - OPENAI_API_KEY: Your OpenAI API key for the EnsembleAgent
    - PUSHOVER_USER_KEY: (Optional) For MessagingAgent tests
    - PUSHOVER_APP_TOKEN: (Optional) For MessagingAgent tests

Cost Warning:
    Live tests make actual API calls and will incur costs. Estimate: $0.01-$0.05 per test run.
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the path so we can import from agents
sys.path.append(str(Path(__file__).parent.parent))

# Import the test functions
from test_planning_agent import run_all_tests, logger

def check_environment():
    """Check if the required environment variables are set."""
    required_vars = ['OPENAI_API_KEY']
    optional_vars = ['PUSHOVER_USER_KEY', 'PUSHOVER_APP_TOKEN']
    
    missing_required = [var for var in required_vars if not os.getenv(var)]
    missing_optional = [var for var in optional_vars if not os.getenv(var)]
    
    if missing_required:
        logger.error(f"❌ Missing required environment variables: {missing_required}")
        logger.error("Please set these variables before running live tests.")
        return False
    
    if missing_optional:
        logger.warning(f"⚠️  Missing optional environment variables: {missing_optional}")
        logger.warning("Some messaging tests may be skipped.")
    
    logger.info("✅ Environment variables are properly configured.")
    return True

def show_cost_warning():
    """Show cost warning and get user confirmation."""
    print("\n" + "="*60)
    print("⚠️  COST WARNING - LIVE DATA TESTS")
    print("="*60)
    print("These tests will make actual API calls to:")
    print("• OpenAI API (for price estimation)")
    print("• Pushover API (for notifications)")
    print("• RSS feeds (for deal scanning)")
    print()
    print("Estimated cost: $0.01 - $0.05 per test run")
    print("="*60)
    
    response = input("Do you want to proceed with live tests? (y/N): ").strip().lower()
    return response in ['y', 'yes']

def main():
    """Main function to run live tests with proper setup."""
    logger.info("PlanningAgent Live Data Test Runner")
    logger.info("=" * 50)
    
    # Check environment
    if not check_environment():
        return 1
    
    # Show cost warning
    if not show_cost_warning():
        logger.info("Live tests cancelled by user.")
        return 0
    
    # Set environment variable to enable live tests
    os.environ['RUN_LIVE_TESTS'] = '1'
    
    try:
        logger.info("Starting live data tests...")
        success = run_all_tests()
        
        if success:
            logger.info("🎉 All live tests completed successfully!")
            return 0
        else:
            logger.error("💥 Some live tests failed.")
            return 1
            
    except Exception as e:
        logger.error(f"Error running live tests: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 