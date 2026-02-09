"""Unit tests for the base Agent interface"""

import pytest
from src.agents.base import (
    Agent,
    AgentInput,
    AgentOutput,
    AgentExecutionError,
    RetryPolicy,
    BackoffStrategy,
)


class MockInput(AgentInput):
    """Mock input for testing"""
    def __init__(self, value: str):
        self.value = value


class MockOutput(AgentOutput):
    """Mock output for testing"""
    def __init__(self, result: str):
        self.result = result


class MockAgent(Agent):
    """Mock agent implementation for testing"""
    
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.execution_count = 0
    
    def execute(self, input_data: AgentInput) -> AgentOutput:
        self.execution_count += 1
        
        if not self.validate_input(input_data):
            raise AgentExecutionError(
                error_code="INVALID_INPUT",
                message="Input validation failed",
                context={"input": str(input_data)}
            )
        
        if self.should_fail:
            raise AgentExecutionError(
                error_code="MOCK_FAILURE",
                message="Mock agent configured to fail"
            )
        
        return MockOutput(result=f"processed: {input_data.value}")
    
    def validate_input(self, input_data: AgentInput) -> bool:
        return isinstance(input_data, MockInput) and hasattr(input_data, 'value')
    
    def get_retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay_seconds=1.0,
            retryable_errors=["MOCK_FAILURE"]
        )


@pytest.mark.unit
class TestAgentInterface:
    """Test suite for Agent interface"""
    
    def test_agent_execute_success(self):
        """Test successful agent execution"""
        agent = MockAgent()
        input_data = MockInput(value="test")
        
        output = agent.execute(input_data)
        
        assert isinstance(output, MockOutput)
        assert output.result == "processed: test"
        assert agent.execution_count == 1
    
    def test_agent_validate_input_valid(self):
        """Test input validation with valid input"""
        agent = MockAgent()
        input_data = MockInput(value="test")
        
        assert agent.validate_input(input_data) is True
    
    def test_agent_validate_input_invalid(self):
        """Test input validation with invalid input"""
        agent = MockAgent()
        
        class WrongInput(AgentInput):
            pass
        
        input_data = WrongInput()
        
        assert agent.validate_input(input_data) is False
    
    def test_agent_execution_error(self):
        """Test agent execution error handling"""
        agent = MockAgent(should_fail=True)
        input_data = MockInput(value="test")
        
        with pytest.raises(AgentExecutionError) as exc_info:
            agent.execute(input_data)
        
        assert exc_info.value.error_code == "MOCK_FAILURE"
        assert "Mock agent configured to fail" in exc_info.value.message
    
    def test_agent_get_retry_policy(self):
        """Test retry policy retrieval"""
        agent = MockAgent()
        policy = agent.get_retry_policy()
        
        assert isinstance(policy, RetryPolicy)
        assert policy.max_attempts == 3
        assert policy.backoff_strategy == BackoffStrategy.EXPONENTIAL
        assert policy.base_delay_seconds == 1.0
        assert "MOCK_FAILURE" in policy.retryable_errors


@pytest.mark.unit
class TestRetryPolicy:
    """Test suite for RetryPolicy"""
    
    def test_retry_policy_defaults(self):
        """Test RetryPolicy with default values"""
        policy = RetryPolicy()
        
        assert policy.max_attempts == 1
        assert policy.backoff_strategy == BackoffStrategy.EXPONENTIAL
        assert policy.base_delay_seconds == 1.0
        assert policy.max_delay_seconds == 60.0
        assert policy.retryable_errors == []
    
    def test_retry_policy_custom_values(self):
        """Test RetryPolicy with custom values"""
        policy = RetryPolicy(
            max_attempts=5,
            backoff_strategy=BackoffStrategy.LINEAR,
            base_delay_seconds=2.0,
            max_delay_seconds=120.0,
            retryable_errors=["ERROR_1", "ERROR_2"]
        )
        
        assert policy.max_attempts == 5
        assert policy.backoff_strategy == BackoffStrategy.LINEAR
        assert policy.base_delay_seconds == 2.0
        assert policy.max_delay_seconds == 120.0
        assert policy.retryable_errors == ["ERROR_1", "ERROR_2"]


@pytest.mark.unit
class TestAgentExecutionError:
    """Test suite for AgentExecutionError"""
    
    def test_error_with_context(self):
        """Test error creation with context"""
        error = AgentExecutionError(
            error_code="TEST_ERROR",
            message="Test error message",
            context={"key": "value"}
        )
        
        assert error.error_code == "TEST_ERROR"
        assert error.message == "Test error message"
        assert error.context == {"key": "value"}
        assert "[TEST_ERROR]" in str(error)
    
    def test_error_without_context(self):
        """Test error creation without context"""
        error = AgentExecutionError(
            error_code="TEST_ERROR",
            message="Test error message"
        )
        
        assert error.error_code == "TEST_ERROR"
        assert error.message == "Test error message"
        assert error.context == {}
