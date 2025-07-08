# PlanningAgent Test Suite

This directory contains comprehensive tests for the `PlanningAgent` class, which orchestrates the multi-agent deal-finding system.

## Test Structure

The test suite is organized into three main categories:

### 1. Unit Tests (`TestPlanningAgent`)
- **13 test methods** covering individual functionality
- **Extensive mocking** to isolate the PlanningAgent from dependencies
- **Fast execution** (no external API calls)
- **Comprehensive coverage** of all methods and edge cases

### 2. Integration Tests (`TestPlanningAgentIntegration`)
- **End-to-end workflow testing** with realistic scenarios
- **Agent coordination verification**
- **Data flow validation** between components

### 3. Live Data Tests (`TestPlanningAgentLiveData`)
- **Real agent instances** with actual API calls
- **Cost-aware testing** with environment variable controls
- **Realistic deal scenarios** and threshold testing

## Running Tests

### Quick Start (Mocked Tests Only)
```bash
# Run unit and integration tests (no API calls)
python test_planning_agent.py
```

### All Tests Including Live Data
```bash
# Run all tests including live data (requires API keys)
RUN_LIVE_TESTS=1 python test_planning_agent.py

# Or use the dedicated script with safety checks
python run_live_tests.py
```

### Environment Variables

#### Required for Live Tests
- `OPENAI_API_KEY`: Your OpenAI API key for price estimation

#### Optional
- `PUSHOVER_USER_KEY`: For messaging agent tests
- `PUSHOVER_APP_TOKEN`: For messaging agent tests
- `SKIP_LIVE_TESTS=1`: Skip live tests even if `RUN_LIVE_TESTS=1`

## Test Coverage

### Core Functionality
- ✅ Agent initialization and dependency injection
- ✅ Deal processing and price estimation
- ✅ Opportunity creation and discount calculation
- ✅ Threshold-based decision making
- ✅ Deal sorting and selection logic

### Edge Cases
- ✅ Negative discounts (estimate < actual price)
- ✅ No deals found scenarios
- ✅ Empty memory handling
- ✅ Deals exactly at threshold boundary
- ✅ Performance optimization (5-deal limit)

### Integration Scenarios
- ✅ Complete workflow from scanning to messaging
- ✅ Multiple deal ranking and selection
- ✅ Memory functionality with URL filtering
- ✅ Error handling and graceful degradation

## Test Philosophy

### Isolation Through Mocking
The test suite uses extensive mocking to isolate the `PlanningAgent` from its dependencies:

```python
@patch('agents.planning_agent.MessagingAgent')
@patch('agents.planning_agent.EnsembleAgent')
@patch('agents.planning_agent.ScannerAgent')
def test_method(self, mock_scanner_cls, mock_ensemble_cls, mock_messaging_cls):
    # Test logic here
```

### Realistic Test Data
Tests use realistic deal scenarios that mirror production use cases:

```python
sample_deal = Deal(
    product_description="iPhone 15 Pro Max 256GB Space Black smartphone",
    price=999.99,
    url="https://example.com/iphone-15-pro-max"
)
```

### Comprehensive Assertions
Each test verifies multiple aspects:
- Return value types and structure
- Method call verification
- Parameter passing validation
- Business logic correctness

## Key Test Cases

### Threshold Testing
```python
# Tests the $50 threshold logic
test_plan_method_below_threshold()     # $30 discount -> None
test_plan_method_exactly_at_threshold() # $50 discount -> None  
test_plan_method_above_threshold()     # $50.01 discount -> Opportunity
```

### Deal Ranking
```python
# Tests sorting by discount (highest first)
test_plan_method_multiple_deals_sorting()
test_live_data_multiple_deals_ranking()
```

### Performance Optimization
```python
# Tests 5-deal processing limit
test_plan_method_limits_deals_to_five()
```

## Cost Considerations

### Live Tests
- **Estimated cost**: $0.01 - $0.05 per test run
- **API calls made**: OpenAI (price estimation), Pushover (notifications)
- **Safety measures**: Environment variable controls, user confirmation prompts

### Mocked Tests
- **No cost**: All external dependencies are mocked
- **Fast execution**: Typical run time < 1 second
- **Safe for CI/CD**: No external API dependencies

## Continuous Integration

### Recommended CI Configuration
```yaml
# Run mocked tests in CI
- name: Run Planning Agent Tests
  run: |
    cd agent_tests
    python test_planning_agent.py
  env:
    SKIP_LIVE_TESTS: 1
```

### Manual Live Testing
```bash
# Set up environment
export OPENAI_API_KEY="your-key-here"
export PUSHOVER_USER_KEY="your-user-key"
export PUSHOVER_APP_TOKEN="your-app-token"

# Run live tests
python run_live_tests.py
```

## Test Output

### Successful Run
```
INFO:__main__:PlanningAgent Test Suite
INFO:__main__:==================================================
INFO:__main__:Running unit and integration tests (mocked)...
INFO:__main__:💡 Set RUN_LIVE_TESTS=1 to include live data tests

...

----------------------------------------------------------------------
Ran 14 tests in 0.006s

OK
INFO:__main__:✅ All tests passed!
INFO:__main__:🎉 All tests completed successfully!
```

### With Live Tests
```
INFO:__main__:Running ALL tests (including live data tests)...
INFO:__main__:⚠️  Live tests will make actual API calls and may incur costs!

...

----------------------------------------------------------------------
Ran 19 tests in 0.125s

OK (skipped=5)  # Skipped due to missing API keys
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the correct directory
   cd agent_tests
   python test_planning_agent.py
   ```

2. **Missing Dependencies**
   ```bash
   # Install required packages
   pip install unittest-mock
   ```

3. **API Key Issues**
   ```bash
   # Check environment variables
   echo $OPENAI_API_KEY
   
   # Set if missing
   export OPENAI_API_KEY="your-key-here"
   ```

4. **Live Tests Skipped**
   - This is expected behavior when API keys are not set
   - Use `python run_live_tests.py` for guided setup

## Contributing

When adding new tests:

1. **Follow the existing pattern**: Use the same mocking and assertion style
2. **Add comprehensive comments**: Explain the test purpose and strategy
3. **Test both success and failure cases**: Include edge cases and error scenarios
4. **Use realistic test data**: Mirror production scenarios
5. **Update this README**: Document any new test categories or requirements

## Files

- `test_planning_agent.py`: Main test suite
- `run_live_tests.py`: Script for running live tests with safety checks
- `README.md`: This documentation file 