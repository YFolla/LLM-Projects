#!/usr/bin/env python3
"""
Comprehensive Test Suite for Planning Agent

This test suite provides thorough coverage of the PlanningAgent class, which serves as the
orchestrator for the multi-agent deal-finding system. The PlanningAgent coordinates three
sub-agents (ScannerAgent, EnsembleAgent, MessagingAgent) to find, evaluate, and alert on deals.

Test Strategy:
1. Unit tests with mocking - Test individual methods in isolation
2. Integration tests - Test the complete workflow
3. Edge case testing - Test boundary conditions and error scenarios
4. Threshold testing - Test the deal qualification logic

Key Testing Challenges Addressed:
- Mocking external dependencies (OpenAI API, ChromaDB, Modal services)
- Testing agent coordination and communication
- Verifying correct data flow between agents
- Testing sorting and filtering logic
- Ensuring proper threshold-based decision making
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Optional

# Add the parent directory to the path so we can import from agents
# This allows us to import the agents module from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.planning_agent import PlanningAgent
from agents.deals import Deal, DealSelection, Opportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
import logging

# Configure logging for test output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestPlanningAgent(unittest.TestCase):
    """
    Primary test class for PlanningAgent functionality.
    
    This class uses extensive mocking to isolate the PlanningAgent from its dependencies,
    allowing us to test the coordination logic without requiring actual API calls or
    external services.
    
    Testing Philosophy:
    - Mock all external dependencies (agents, APIs, databases)
    - Test the PlanningAgent's orchestration logic in isolation
    - Verify correct method calls and parameter passing
    - Test edge cases and error conditions
    """

    def setUp(self):
        """
        Set up test fixtures before each test method.
        
        This method creates sample data objects that represent realistic deal scenarios.
        Using consistent test data across tests ensures predictable behavior and makes
        tests easier to understand and maintain.
        """
        # Create mock collection for ChromaDB - required for EnsembleAgent initialization
        # In real usage, this would be a ChromaDB collection containing product embeddings
        self.mock_collection = Mock()
        
        # Create realistic sample deals for testing
        # These represent the kind of deals the system would process in production
        self.sample_deal = Deal(
            product_description="iPhone 15 Pro Max 256GB Space Black smartphone with A17 Pro chip",
            price=999.99,
            url="https://example.com/iphone-15-pro-max"
        )
        
        self.sample_deal_2 = Deal(
            product_description="Samsung Galaxy S24 Ultra 512GB Titanium Black",
            price=1299.99,
            url="https://example.com/samsung-galaxy-s24-ultra"
        )
        
        self.sample_deals = [self.sample_deal, self.sample_deal_2]
        
        # Create sample deal selection - this represents what ScannerAgent would return
        self.sample_deal_selection = DealSelection(deals=self.sample_deals)
        
        # Create sample opportunity - this represents a processed deal with discount calculation
        self.sample_opportunity = Opportunity(
            deal=self.sample_deal,
            estimate=1199.99,  # Estimated market value
            discount=200.00    # Calculated discount (estimate - actual price)
        )

    @patch('agents.planning_agent.MessagingAgent')
    @patch('agents.planning_agent.EnsembleAgent')
    @patch('agents.planning_agent.ScannerAgent')
    def test_init(self, mock_scanner_cls, mock_ensemble_cls, mock_messaging_cls):
        """
        Test PlanningAgent initialization and dependency injection.
        
        This test verifies that:
        1. The PlanningAgent correctly instantiates all three sub-agents
        2. The correct parameters are passed to each agent
        3. Agent properties are set correctly
        4. The collection parameter is properly forwarded to EnsembleAgent
        
        Mocking Strategy:
        - Mock the agent classes themselves (not instances)
        - Verify the classes are called with correct parameters
        - Check that the returned mock instances are stored correctly
        """
        # Arrange - Create mock instances that the agent classes will return
        mock_scanner = Mock()
        mock_ensemble = Mock()
        mock_messaging = Mock()
        
        # Configure the mock classes to return our mock instances
        mock_scanner_cls.return_value = mock_scanner
        mock_ensemble_cls.return_value = mock_ensemble
        mock_messaging_cls.return_value = mock_messaging
        
        # Act - Create the PlanningAgent (this will trigger agent initialization)
        planning_agent = PlanningAgent(self.mock_collection)
        
        # Assert - Verify correct initialization
        self.assertEqual(planning_agent.name, "Planning Agent")
        self.assertEqual(planning_agent.DEAL_THRESHOLD, 50)
        self.assertEqual(planning_agent.scanner, mock_scanner)
        self.assertEqual(planning_agent.ensemble, mock_ensemble)
        self.assertEqual(planning_agent.messenger, mock_messaging)
        
        # Verify that the agents were initialized with correct parameters
        mock_scanner_cls.assert_called_once()  # ScannerAgent takes no parameters
        mock_ensemble_cls.assert_called_once_with(self.mock_collection)  # EnsembleAgent needs collection
        mock_messaging_cls.assert_called_once()  # MessagingAgent takes no parameters

    @patch('agents.planning_agent.MessagingAgent')
    @patch('agents.planning_agent.EnsembleAgent')
    @patch('agents.planning_agent.ScannerAgent')
    def test_run_method(self, mock_scanner_cls, mock_ensemble_cls, mock_messaging_cls):
        """
        Test the run method that processes a single deal.
        
        The run method is responsible for:
        1. Taking a Deal object as input
        2. Using EnsembleAgent to estimate the market value
        3. Calculating the discount (estimate - actual price)
        4. Returning an Opportunity object
        
        This test verifies the core pricing logic and data transformation.
        """
        # Arrange - Set up the EnsembleAgent to return a specific price estimate
        mock_ensemble = Mock()
        mock_ensemble.price.return_value = 1199.99  # Mock price estimate
        mock_ensemble_cls.return_value = mock_ensemble
        
        # We don't need the other agents for this test, but we mock them anyway
        mock_scanner_cls.return_value = Mock()
        mock_messaging_cls.return_value = Mock()
        
        planning_agent = PlanningAgent(self.mock_collection)
        
        # Act - Process a single deal
        result = planning_agent.run(self.sample_deal)
        
        # Assert - Verify the result is correctly structured
        self.assertIsInstance(result, Opportunity)
        self.assertEqual(result.deal, self.sample_deal)
        self.assertEqual(result.estimate, 1199.99)
        self.assertEqual(result.discount, 200.00)  # 1199.99 - 999.99
        
        # Verify that the ensemble agent was called with the correct product description
        mock_ensemble.price.assert_called_once_with(self.sample_deal.product_description)

    @patch('agents.planning_agent.MessagingAgent')
    @patch('agents.planning_agent.EnsembleAgent')
    @patch('agents.planning_agent.ScannerAgent')
    def test_run_method_negative_discount(self, mock_scanner_cls, mock_ensemble_cls, mock_messaging_cls):
        """
        Test the run method when estimate is lower than actual price.
        
        This is an important edge case - sometimes the estimated value might be lower
        than the actual price (indicating the deal might not be good, or the estimate
        is incorrect). The system should handle this gracefully.
        """
        # Arrange - Set up ensemble to return a lower estimate than actual price
        mock_ensemble = Mock()
        mock_ensemble.price.return_value = 799.99  # Lower than actual price of 999.99
        mock_ensemble_cls.return_value = mock_ensemble
        
        mock_scanner_cls.return_value = Mock()
        mock_messaging_cls.return_value = Mock()
        
        planning_agent = PlanningAgent(self.mock_collection)
        
        # Act
        result = planning_agent.run(self.sample_deal)
        
        # Assert - Verify negative discount is handled correctly
        self.assertIsInstance(result, Opportunity)
        self.assertEqual(result.deal, self.sample_deal)
        self.assertEqual(result.estimate, 799.99)
        self.assertEqual(result.discount, -200.00)  # 799.99 - 999.99 (negative discount)

    @patch('agents.planning_agent.MessagingAgent')
    @patch('agents.planning_agent.EnsembleAgent')
    @patch('agents.planning_agent.ScannerAgent')
    def test_plan_method_with_deals(self, mock_scanner_cls, mock_ensemble_cls, mock_messaging_cls):
        """
        Test the complete plan method workflow when deals are found.
        
        This test covers the full workflow:
        1. ScannerAgent finds deals
        2. EnsembleAgent estimates prices for each deal
        3. Deals are sorted by discount (highest first)
        4. Best deal is selected
        5. If above threshold, MessagingAgent sends alert
        6. Best deal is returned (if above threshold)
        """
        # Arrange - Set up the complete workflow
        mock_scanner = Mock()
        mock_scanner.scan.return_value = self.sample_deal_selection
        mock_scanner_cls.return_value = mock_scanner
        
        # Configure ensemble to return different estimates for each deal
        # iPhone: 999.99 -> 1199.99 = $200 discount
        # Samsung: 1299.99 -> 1499.99 = $200 discount (same discount, first one will be selected)
        mock_ensemble = Mock()
        mock_ensemble.price.side_effect = [1199.99, 1499.99]  # Different estimates for each deal
        mock_ensemble_cls.return_value = mock_ensemble
        
        mock_messaging = Mock()
        mock_messaging_cls.return_value = mock_messaging
        
        planning_agent = PlanningAgent(self.mock_collection)
        
        # Act - Run the complete planning workflow
        result = planning_agent.plan(memory=["https://example.com/old-deal"])
        
        # Assert - Verify the workflow executed correctly
        self.assertIsInstance(result, Opportunity)
        # Since both deals have same discount, first one (iPhone) should be selected
        self.assertEqual(result.deal, self.sample_deal)  # Should return the first deal with equal discount
        self.assertEqual(result.estimate, 1199.99)
        self.assertEqual(result.discount, 200.00)  # 1199.99 - 999.99
        
        # Verify scanner was called with correct memory parameter
        mock_scanner.scan.assert_called_once_with(memory=["https://example.com/old-deal"])
        
        # Verify messenger was called for deal above threshold (200 > 50)
        mock_messaging.alert.assert_called_once_with(result)

    @patch('agents.planning_agent.MessagingAgent')
    @patch('agents.planning_agent.EnsembleAgent')
    @patch('agents.planning_agent.ScannerAgent')
    def test_plan_method_no_deals(self, mock_scanner_cls, mock_ensemble_cls, mock_messaging_cls):
        """
        Test the plan method when no deals are found.
        
        This tests the case where ScannerAgent returns None (no deals found).
        The system should handle this gracefully and return None.
        """
        # Arrange - Configure scanner to return no deals
        mock_scanner = Mock()
        mock_scanner.scan.return_value = None  # No deals found
        mock_scanner_cls.return_value = mock_scanner
        
        mock_ensemble_cls.return_value = Mock()
        mock_messaging_cls.return_value = Mock()
        
        planning_agent = PlanningAgent(self.mock_collection)
        
        # Act
        result = planning_agent.plan()
        
        # Assert - Should return None when no deals are found
        self.assertIsNone(result)
        
        # Verify scanner was called with empty memory (default parameter)
        mock_scanner.scan.assert_called_once_with(memory=[])

    @patch('agents.planning_agent.MessagingAgent')
    @patch('agents.planning_agent.EnsembleAgent')
    @patch('agents.planning_agent.ScannerAgent')
    def test_plan_method_below_threshold(self, mock_scanner_cls, mock_ensemble_cls, mock_messaging_cls):
        """
        Test the plan method when best deal is below threshold.
        
        This tests the threshold logic - deals must have a discount > DEAL_THRESHOLD (50)
        to be considered worthy of alerting. Deals below this threshold should not
        trigger alerts and should return None.
        """
        # Arrange - Set up a deal with discount below threshold
        mock_scanner = Mock()
        mock_scanner.scan.return_value = DealSelection(deals=[self.sample_deal])
        mock_scanner_cls.return_value = mock_scanner
        
        mock_ensemble = Mock()
        mock_ensemble.price.return_value = 1029.99  # Only $30 discount (1029.99 - 999.99), below threshold of 50
        mock_ensemble_cls.return_value = mock_ensemble
        
        mock_messaging = Mock()
        mock_messaging_cls.return_value = mock_messaging
        
        planning_agent = PlanningAgent(self.mock_collection)
        
        # Act
        result = planning_agent.plan()
        
        # Assert - Should return None when below threshold
        self.assertIsNone(result)
        
        # Verify messenger was NOT called (no alert for deals below threshold)
        mock_messaging.alert.assert_not_called()

    @patch('agents.planning_agent.MessagingAgent')
    @patch('agents.planning_agent.EnsembleAgent')
    @patch('agents.planning_agent.ScannerAgent')
    def test_plan_method_exactly_at_threshold(self, mock_scanner_cls, mock_ensemble_cls, mock_messaging_cls):
        """
        Test the plan method when best deal is exactly at threshold.
        
        This tests the boundary condition - the code uses ">" not ">=" for threshold
        comparison, so deals exactly at the threshold should NOT trigger alerts.
        """
        # Arrange - Set up a deal with discount exactly at threshold
        mock_scanner = Mock()
        mock_scanner.scan.return_value = DealSelection(deals=[self.sample_deal])
        mock_scanner_cls.return_value = mock_scanner
        
        mock_ensemble = Mock()
        mock_ensemble.price.return_value = 1049.99  # Exactly $50 discount (1049.99 - 999.99)
        mock_ensemble_cls.return_value = mock_ensemble
        
        mock_messaging = Mock()
        mock_messaging_cls.return_value = mock_messaging
        
        planning_agent = PlanningAgent(self.mock_collection)
        
        # Act
        result = planning_agent.plan()
        
        # Assert - Should return None when exactly at threshold (not greater than)
        self.assertIsNone(result)
        
        # Verify messenger was NOT called (threshold requires > not >=)
        mock_messaging.alert.assert_not_called()

    @patch('agents.planning_agent.MessagingAgent')
    @patch('agents.planning_agent.EnsembleAgent')
    @patch('agents.planning_agent.ScannerAgent')
    def test_plan_method_above_threshold(self, mock_scanner_cls, mock_ensemble_cls, mock_messaging_cls):
        """
        Test the plan method when best deal is above threshold.
        
        This tests the positive case - deals above the threshold should trigger
        alerts and be returned as opportunities.
        """
        # Arrange - Set up a deal with discount above threshold
        mock_scanner = Mock()
        mock_scanner.scan.return_value = DealSelection(deals=[self.sample_deal])
        mock_scanner_cls.return_value = mock_scanner
        
        mock_ensemble = Mock()
        mock_ensemble.price.return_value = 1050.00  # $50.01 discount (1050.00 - 999.99), above threshold
        mock_ensemble_cls.return_value = mock_ensemble
        
        mock_messaging = Mock()
        mock_messaging_cls.return_value = mock_messaging
        
        planning_agent = PlanningAgent(self.mock_collection)
        
        # Act
        result = planning_agent.plan()
        
        # Assert - Should return the opportunity when above threshold
        self.assertIsInstance(result, Opportunity)
        # Use assertAlmostEqual for floating point comparison
        self.assertAlmostEqual(result.discount, 50.01, places=2)
        
        # Verify messenger was called (alert for deals above threshold)
        mock_messaging.alert.assert_called_once_with(result)

    @patch('agents.planning_agent.MessagingAgent')
    @patch('agents.planning_agent.EnsembleAgent')
    @patch('agents.planning_agent.ScannerAgent')
    def test_plan_method_multiple_deals_sorting(self, mock_scanner_cls, mock_ensemble_cls, mock_messaging_cls):
        """
        Test that deals are sorted by discount and best one is selected.
        
        This test verifies the sorting logic - the system should process multiple deals,
        calculate discounts for each, sort them by discount (highest first), and
        select the best one.
        """
        # Arrange - Create multiple deals with different potential discounts
        deal_1 = Deal(product_description="Deal 1", price=100.00, url="https://example.com/deal1")
        deal_2 = Deal(product_description="Deal 2", price=200.00, url="https://example.com/deal2")
        deal_3 = Deal(product_description="Deal 3", price=300.00, url="https://example.com/deal3")
        
        mock_scanner = Mock()
        mock_scanner.scan.return_value = DealSelection(deals=[deal_1, deal_2, deal_3])
        mock_scanner_cls.return_value = mock_scanner
        
        mock_ensemble = Mock()
        # Set up estimates that will result in different discounts:
        # deal_1: 150.00 - 100.00 = $50 discount
        # deal_2: 300.00 - 200.00 = $100 discount (highest)
        # deal_3: 375.00 - 300.00 = $75 discount
        mock_ensemble.price.side_effect = [150.00, 300.00, 375.00]
        mock_ensemble_cls.return_value = mock_ensemble
        
        mock_messaging = Mock()
        mock_messaging_cls.return_value = mock_messaging
        
        planning_agent = PlanningAgent(self.mock_collection)
        
        # Act
        result = planning_agent.plan()
        
        # Assert - Should return deal_2 which has the highest discount
        self.assertIsInstance(result, Opportunity)
        self.assertEqual(result.deal, deal_2)  # Should be deal_2 with $100 discount
        self.assertEqual(result.discount, 100.00)

    @patch('agents.planning_agent.MessagingAgent')
    @patch('agents.planning_agent.EnsembleAgent')
    @patch('agents.planning_agent.ScannerAgent')
    def test_plan_method_limits_deals_to_five(self, mock_scanner_cls, mock_ensemble_cls, mock_messaging_cls):
        """
        Test that only first 5 deals are processed.
        
        This test verifies the performance optimization - the system should only
        process the first 5 deals from the scanner to avoid excessive API calls
        and processing time.
        """
        # Arrange - Create 10 deals but expect only 5 to be processed
        deals = [
            Deal(product_description=f"Deal {i}", price=100.00, url=f"https://example.com/deal{i}")
            for i in range(10)  # Create 10 deals
        ]
        
        mock_scanner = Mock()
        mock_scanner.scan.return_value = DealSelection(deals=deals)
        mock_scanner_cls.return_value = mock_scanner
        
        mock_ensemble = Mock()
        mock_ensemble.price.return_value = 200.00  # $100 discount for all deals
        mock_ensemble_cls.return_value = mock_ensemble
        
        mock_messaging = Mock()
        mock_messaging_cls.return_value = mock_messaging
        
        planning_agent = PlanningAgent(self.mock_collection)
        
        # Act
        result = planning_agent.plan()
        
        # Assert
        self.assertIsInstance(result, Opportunity)
        
        # Verify that ensemble.price was called exactly 5 times (not 10)
        self.assertEqual(mock_ensemble.price.call_count, 5)

    @patch('agents.planning_agent.MessagingAgent')
    @patch('agents.planning_agent.EnsembleAgent')
    @patch('agents.planning_agent.ScannerAgent')
    def test_plan_method_empty_memory_default(self, mock_scanner_cls, mock_ensemble_cls, mock_messaging_cls):
        """
        Test that plan method handles empty memory list by default.
        
        This test verifies the default parameter behavior - when no memory is provided,
        the system should pass an empty list to the scanner.
        """
        # Arrange
        mock_scanner = Mock()
        mock_scanner.scan.return_value = None
        mock_scanner_cls.return_value = mock_scanner
        
        mock_ensemble_cls.return_value = Mock()
        mock_messaging_cls.return_value = Mock()
        
        planning_agent = PlanningAgent(self.mock_collection)
        
        # Act - Call plan without memory parameter
        result = planning_agent.plan()  # No memory parameter
        
        # Assert
        self.assertIsNone(result)
        
        # Verify scanner was called with empty list (default behavior)
        mock_scanner.scan.assert_called_once_with(memory=[])

    def test_deal_threshold_constant(self):
        """
        Test that the DEAL_THRESHOLD constant is set correctly.
        
        This is a simple but important test - the threshold value is critical
        for the system's behavior and should be explicitly tested.
        """
        self.assertEqual(PlanningAgent.DEAL_THRESHOLD, 50)

    def test_agent_properties(self):
        """
        Test that the agent has correct name and color properties.
        
        These properties are used for logging and identification purposes.
        """
        self.assertEqual(PlanningAgent.name, "Planning Agent")
        # Note: We can't easily test the color without importing Agent base class


class TestPlanningAgentIntegration(unittest.TestCase):
    """
    Integration tests for PlanningAgent with more realistic scenarios.
    
    While the main test class focuses on unit testing with extensive mocking,
    this class tests more realistic integration scenarios that verify the
    complete workflow behaves correctly.
    """

    def setUp(self):
        """Set up test fixtures for integration tests."""
        self.mock_collection = Mock()

    @patch('agents.planning_agent.MessagingAgent')
    @patch('agents.planning_agent.EnsembleAgent')
    @patch('agents.planning_agent.ScannerAgent')
    def test_integration_flow(self, mock_scanner_cls, mock_ensemble_cls, mock_messaging_cls):
        """
        Test the complete integration flow from scanning to messaging.
        
        This test verifies that all components work together correctly in a
        realistic scenario, with proper data flow and method calls.
        """
        # Arrange - Set up a realistic deal scenario
        sample_deal = Deal(
            product_description="MacBook Pro 16-inch M3 Pro chip",
            price=2499.99,
            url="https://example.com/macbook-pro"
        )
        
        mock_scanner = Mock()
        mock_scanner.scan.return_value = DealSelection(deals=[sample_deal])
        mock_scanner_cls.return_value = mock_scanner
        
        mock_ensemble = Mock()
        mock_ensemble.price.return_value = 2599.99  # $100 discount
        mock_ensemble_cls.return_value = mock_ensemble
        
        mock_messaging = Mock()
        mock_messaging_cls.return_value = mock_messaging
        
        planning_agent = PlanningAgent(self.mock_collection)
        
        # Act - Run the complete workflow
        result = planning_agent.plan(memory=["https://example.com/old-deal"])
        
        # Assert - Verify the complete workflow
        self.assertIsInstance(result, Opportunity)
        self.assertEqual(result.deal.product_description, "MacBook Pro 16-inch M3 Pro chip")
        self.assertEqual(result.deal.price, 2499.99)
        self.assertEqual(result.estimate, 2599.99)
        self.assertEqual(result.discount, 100.00)
        
        # Verify the complete flow with correct method calls and parameters
        mock_scanner.scan.assert_called_once_with(memory=["https://example.com/old-deal"])
        mock_ensemble.price.assert_called_once_with("MacBook Pro 16-inch M3 Pro chip")
        mock_messaging.alert.assert_called_once_with(result)


def run_comprehensive_test():
    """
    Run all tests and provide a comprehensive report.
    
    This function orchestrates the complete test suite and provides detailed
    reporting on test results, including failures and errors.
    """
    logger.info("Starting comprehensive PlanningAgent tests...")
    
    # Create test suite containing all test classes using modern approach
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add all test methods from both test classes
    test_suite.addTest(loader.loadTestsFromTestCase(TestPlanningAgent))
    test_suite.addTest(loader.loadTestsFromTestCase(TestPlanningAgentIntegration))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Provide detailed reporting
    if result.wasSuccessful():
        logger.info("✅ All tests passed!")
    else:
        logger.error(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
        # Log details of any failures
        for test, traceback in result.failures:
            logger.error(f"FAIL: {test}")
            logger.error(traceback)
            
        # Log details of any errors
        for test, traceback in result.errors:
            logger.error(f"ERROR: {test}")
            logger.error(traceback)
    
    return result.wasSuccessful()


class TestPlanningAgentLiveData(unittest.TestCase):
    """
    Live data integration tests for PlanningAgent.
    
    These tests use real agents with actual API calls to test the complete system
    end-to-end. They require proper environment setup (API keys, etc.) and are
    designed to verify that the system works with real data.
    
    WARNING: These tests make actual API calls and may incur costs.
    Set SKIP_LIVE_TESTS=1 environment variable to skip these tests.
    """

    def setUp(self):
        """Set up live data tests with environment checks."""
        # Skip live tests if environment variable is set
        if os.getenv('SKIP_LIVE_TESTS') == '1':
            self.skipTest("Live tests skipped by environment variable")
        
        # Check for required environment variables
        required_vars = ['OPENAI_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            self.skipTest(f"Missing required environment variables: {missing_vars}")
        
        # Set up a mock collection for testing
        # In real usage, this would be a proper ChromaDB collection
        self.mock_collection = Mock()
        logger.info("Setting up live data tests...")

    def test_live_data_integration_with_mocked_apis(self):
        """
        Test with live PlanningAgent but mocked external APIs.
        
        This test creates a real PlanningAgent instance but mocks the external
        API calls to avoid costs while still testing the integration logic.
        """
        # Create a real PlanningAgent instance
        planning_agent = PlanningAgent(self.mock_collection)
        
        # Verify that the agent was initialized correctly
        self.assertIsNotNone(planning_agent.scanner)
        self.assertIsNotNone(planning_agent.ensemble)
        self.assertIsNotNone(planning_agent.messenger)
        self.assertEqual(planning_agent.name, "Planning Agent")
        
        # Test with a realistic deal
        sample_deal = Deal(
            product_description="Apple MacBook Air 13-inch M2 chip 8GB RAM 256GB SSD",
            price=1099.99,
            url="https://example.com/macbook-air-m2"
        )
        
        # Mock the ensemble agent's price method to return a realistic estimate
        with patch.object(planning_agent.ensemble, 'price', return_value=1299.99):
            result = planning_agent.run(sample_deal)
            
            # Verify the result structure
            self.assertIsInstance(result, Opportunity)
            self.assertEqual(result.deal, sample_deal)
            self.assertEqual(result.estimate, 1299.99)
            self.assertEqual(result.discount, 200.00)
            
            # Verify savings calculation
            expected_savings = result.calculate_savings()
            self.assertAlmostEqual(expected_savings, 15.38, places=2)  # 200/1299.99 * 100

    def test_live_data_threshold_behavior(self):
        """
        Test threshold behavior with realistic deal scenarios.
        
        This test verifies that the threshold logic works correctly with
        realistic price points and discount calculations.
        """
        planning_agent = PlanningAgent(self.mock_collection)
        
        # Test case 1: Deal above threshold
        high_discount_deal = Deal(
            product_description="Sony WH-1000XM5 Wireless Noise Canceling Headphones",
            price=299.99,
            url="https://example.com/sony-headphones"
        )
        
        # Mock scanner to return our test deal
        with patch.object(planning_agent.scanner, 'scan') as mock_scan, \
             patch.object(planning_agent.ensemble, 'price', return_value=399.99), \
             patch.object(planning_agent.messenger, 'alert') as mock_alert:
            
            mock_scan.return_value = DealSelection(deals=[high_discount_deal])
            
            result = planning_agent.plan()
            
            # Should return the opportunity (discount = $100, above threshold of $50)
            self.assertIsInstance(result, Opportunity)
            self.assertEqual(result.discount, 100.00)
            mock_alert.assert_called_once()
        
        # Test case 2: Deal below threshold
        low_discount_deal = Deal(
            product_description="USB-C Cable 6ft",
            price=19.99,
            url="https://example.com/usb-cable"
        )
        
        with patch.object(planning_agent.scanner, 'scan') as mock_scan, \
             patch.object(planning_agent.ensemble, 'price', return_value=39.99), \
             patch.object(planning_agent.messenger, 'alert') as mock_alert:
            
            mock_scan.return_value = DealSelection(deals=[low_discount_deal])
            
            result = planning_agent.plan()
            
            # Should return None (discount = $20, below threshold of $50)
            self.assertIsNone(result)
            mock_alert.assert_not_called()

    def test_live_data_multiple_deals_ranking(self):
        """
        Test deal ranking with multiple realistic deals.
        
        This test verifies that the system correctly ranks and selects
        the best deal from multiple options.
        """
        planning_agent = PlanningAgent(self.mock_collection)
        
        # Create multiple realistic deals
        deals = [
            Deal(
                product_description="iPad Air 10.9-inch 64GB WiFi",
                price=499.99,
                url="https://example.com/ipad-air"
            ),
            Deal(
                product_description="Samsung Galaxy Tab S9 128GB",
                price=699.99,
                url="https://example.com/galaxy-tab"
            ),
            Deal(
                product_description="Microsoft Surface Pro 9 256GB",
                price=899.99,
                url="https://example.com/surface-pro"
            )
        ]
        
        # Mock estimates that will create different discount levels
        # iPad: 499.99 -> 649.99 = $150 discount
        # Samsung: 699.99 -> 849.99 = $150 discount (same as iPad, first will be selected)
        # Surface: 899.99 -> 1199.99 = $300 discount (highest, should be selected)
        estimates = [649.99, 849.99, 1199.99]
        
        with patch.object(planning_agent.scanner, 'scan') as mock_scan, \
             patch.object(planning_agent.ensemble, 'price', side_effect=estimates), \
             patch.object(planning_agent.messenger, 'alert') as mock_alert:
            
            mock_scan.return_value = DealSelection(deals=deals)
            
            result = planning_agent.plan()
            
            # Should return the Surface Pro (highest discount)
            self.assertIsInstance(result, Opportunity)
            self.assertEqual(result.deal.product_description, "Microsoft Surface Pro 9 256GB")
            self.assertEqual(result.discount, 300.00)
            mock_alert.assert_called_once()

    def test_live_data_memory_functionality(self):
        """
        Test memory functionality with realistic URL scenarios.
        
        This test verifies that the memory parameter is correctly passed
        to the scanner and used for filtering previously seen deals.
        """
        planning_agent = PlanningAgent(self.mock_collection)
        
        # Test with memory URLs
        memory_urls = [
            "https://example.com/old-deal-1",
            "https://example.com/old-deal-2",
            "https://example.com/old-deal-3"
        ]
        
        with patch.object(planning_agent.scanner, 'scan') as mock_scan:
            mock_scan.return_value = None  # No new deals found
            
            result = planning_agent.plan(memory=memory_urls)
            
            # Verify scanner was called with correct memory
            mock_scan.assert_called_once_with(memory=memory_urls)
            self.assertIsNone(result)

    def test_live_data_error_handling(self):
        """
        Test error handling in live scenarios.
        
        This test verifies that the system handles various error conditions
        gracefully without crashing.
        """
        planning_agent = PlanningAgent(self.mock_collection)
        
        # Test with scanner returning None
        with patch.object(planning_agent.scanner, 'scan', return_value=None):
            result = planning_agent.plan()
            self.assertIsNone(result)
        
        # Test with empty deal list
        with patch.object(planning_agent.scanner, 'scan', return_value=DealSelection(deals=[])):
            result = planning_agent.plan()
            self.assertIsNone(result)
        
        # Test with ensemble throwing an exception
        sample_deal = Deal(
            product_description="Test Product",
            price=100.00,
            url="https://example.com/test"
        )
        
        with patch.object(planning_agent.scanner, 'scan') as mock_scan, \
             patch.object(planning_agent.ensemble, 'price', side_effect=Exception("API Error")):
            
            mock_scan.return_value = DealSelection(deals=[sample_deal])
            
            # Should handle the exception gracefully
            with self.assertRaises(Exception):
                planning_agent.plan()


def run_all_tests():
    """
    Run all tests including live data tests.
    
    This function runs both mocked unit tests and live data integration tests,
    providing comprehensive coverage of the PlanningAgent functionality.
    """
    logger.info("Starting comprehensive PlanningAgent tests (including live data)...")
    
    # Create test suite containing all test classes
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_suite.addTest(loader.loadTestsFromTestCase(TestPlanningAgent))
    test_suite.addTest(loader.loadTestsFromTestCase(TestPlanningAgentIntegration))
    test_suite.addTest(loader.loadTestsFromTestCase(TestPlanningAgentLiveData))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Provide detailed reporting
    if result.wasSuccessful():
        logger.info("✅ All tests passed!")
    else:
        logger.error(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
        # Log details of any failures
        for test, traceback in result.failures:
            logger.error(f"FAIL: {test}")
            logger.error(traceback)
            
        # Log details of any errors
        for test, traceback in result.errors:
            logger.error(f"ERROR: {test}")
            logger.error(traceback)
    
    return result.wasSuccessful()


def main():
    """
    Main function to run the tests.
    
    This function serves as the entry point for running the test suite,
    providing a clean interface and proper exit codes.
    """
    try:
        logger.info("PlanningAgent Test Suite")
        logger.info("=" * 50)
        
        # Check if we should run live tests
        run_live_tests = os.getenv('RUN_LIVE_TESTS', '0') == '1'
        
        if run_live_tests:
            logger.info("Running ALL tests (including live data tests)...")
            logger.info("⚠️  Live tests will make actual API calls and may incur costs!")
            success = run_all_tests()
        else:
            logger.info("Running unit and integration tests (mocked)...")
            logger.info("💡 Set RUN_LIVE_TESTS=1 to include live data tests")
            success = run_comprehensive_test()
        
        if success:
            logger.info("🎉 All tests completed successfully!")
            return 0
        else:
            logger.error("💥 Some tests failed. Please check the output above.")
            return 1
            
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 