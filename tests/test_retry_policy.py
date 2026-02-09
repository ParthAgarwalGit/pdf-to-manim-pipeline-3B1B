"""Unit tests for retry policy implementation.

Tests cover:
- Backoff delay calculation for different strategies
- Error classification (retryable vs non-retryable)
- Retry execution logic
- Policy creation for different agent types
"""

import pytest
import time
from unittest.mock import Mock, patch

from src.agents.base import (
    AgentExecutionError,
    RetryPolicy,
    BackoffStrategy
)
from src.orchestrator.retry_policy import (
    calculate_backoff_delay,
    is_retryable_error,
    execute_with_retry,
    create_retry_policy,
    RETRYABLE_ERROR_CODES,
    NON_RETRYABLE_ERROR_CODES,
)


class TestBackoffDelayCalculation:
    """Test backoff delay calculation for different strategies."""
    
    def test_exponential_backoff(self):
        """Test exponential backoff: delay = base * 2^attempt."""
        # Attempt 0: 1.0 * 2^0 = 1.0
        assert calculate_backoff_delay(0, BackoffStrategy.EXPONENTIAL, 1.0, 60.0) == 1.0
        
        # Attempt 1: 1.0 * 2^1 = 2.0
        assert calculate_backoff_delay(1, BackoffStrategy.EXPONENTIAL, 1.0, 60.0) == 2.0
        
        # Attempt 2: 1.0 * 2^2 = 4.0
        assert calculate_backoff_delay(2, BackoffStrategy.EXPONENTIAL, 1.0, 60.0) == 4.0
        
        # Attempt 3: 1.0 * 2^3 = 8.0
        assert calculate_backoff_delay(3, BackoffStrategy.EXPONENTIAL, 1.0, 60.0) == 8.0
    
    def test_exponential_backoff_capped_at_max(self):
        """Test exponential backoff is capped at max_delay."""
        # Attempt 10: 1.0 * 2^10 = 1024.0, but capped at 60.0
        assert calculate_backoff_delay(10, BackoffStrategy.EXPONENTIAL, 1.0, 60.0) == 60.0
        
        # Attempt 6: 1.0 * 2^6 = 64.0, but capped at 60.0
        assert calculate_backoff_delay(6, BackoffStrategy.EXPONENTIAL, 1.0, 60.0) == 60.0
    
    def test_linear_backoff(self):
        """Test linear backoff: delay = base * (attempt + 1)."""
        # Attempt 0: 1.0 * 1 = 1.0
        assert calculate_backoff_delay(0, BackoffStrategy.LINEAR, 1.0, 60.0) == 1.0
        
        # Attempt 1: 1.0 * 2 = 2.0
        assert calculate_backoff_delay(1, BackoffStrategy.LINEAR, 1.0, 60.0) == 2.0
        
        # Attempt 2: 1.0 * 3 = 3.0
        assert calculate_backoff_delay(2, BackoffStrategy.LINEAR, 1.0, 60.0) == 3.0
        
        # Attempt 9: 1.0 * 10 = 10.0
        assert calculate_backoff_delay(9, BackoffStrategy.LINEAR, 1.0, 60.0) == 10.0
    
    def test_linear_backoff_capped_at_max(self):
        """Test linear backoff is capped at max_delay."""
        # Attempt 100: 1.0 * 101 = 101.0, but capped at 60.0
        assert calculate_backoff_delay(100, BackoffStrategy.LINEAR, 1.0, 60.0) == 60.0
    
    def test_constant_backoff(self):
        """Test constant backoff: delay = base (always)."""
        # All attempts should return base_delay
        assert calculate_backoff_delay(0, BackoffStrategy.CONSTANT, 1.0, 60.0) == 1.0
        assert calculate_backoff_delay(1, BackoffStrategy.CONSTANT, 1.0, 60.0) == 1.0
        assert calculate_backoff_delay(10, BackoffStrategy.CONSTANT, 1.0, 60.0) == 1.0
        assert calculate_backoff_delay(100, BackoffStrategy.CONSTANT, 1.0, 60.0) == 1.0
    
    def test_different_base_delays(self):
        """Test backoff with different base delays."""
        # Base delay 2.0
        assert calculate_backoff_delay(0, BackoffStrategy.EXPONENTIAL, 2.0, 60.0) == 2.0
        assert calculate_backoff_delay(1, BackoffStrategy.EXPONENTIAL, 2.0, 60.0) == 4.0
        assert calculate_backoff_delay(2, BackoffStrategy.EXPONENTIAL, 2.0, 60.0) == 8.0
        
        # Base delay 0.5
        assert calculate_backoff_delay(0, BackoffStrategy.EXPONENTIAL, 0.5, 60.0) == 0.5
        assert calculate_backoff_delay(1, BackoffStrategy.EXPONENTIAL, 0.5, 60.0) == 1.0
        assert calculate_backoff_delay(2, BackoffStrategy.EXPONENTIAL, 0.5, 60.0) == 2.0


class TestErrorClassification:
    """Test error classification as retryable or non-retryable."""
    
    def test_retryable_error_with_explicit_policy(self):
        """Test error is retryable when in policy's retryable_errors list."""
        policy = RetryPolicy(
            max_attempts=3,
            retryable_errors=['API_TIMEOUT', 'API_RATE_LIMIT']
        )
        
        error = AgentExecutionError('API_TIMEOUT', 'Request timed out')
        assert is_retryable_error(error, policy) is True
        
        error = AgentExecutionError('API_RATE_LIMIT', 'Rate limit exceeded')
        assert is_retryable_error(error, policy) is True
    
    def test_non_retryable_error_with_explicit_policy(self):
        """Test error is not retryable when not in policy's retryable_errors list."""
        policy = RetryPolicy(
            max_attempts=3,
            retryable_errors=['API_TIMEOUT']
        )
        
        error = AgentExecutionError('INVALID_INPUT', 'Invalid input format')
        assert is_retryable_error(error, policy) is False
    
    def test_retryable_error_with_default_policy(self):
        """Test error is retryable when in default RETRYABLE_ERROR_CODES."""
        policy = RetryPolicy(max_attempts=3, retryable_errors=[])
        
        # Test common retryable errors
        for error_code in ['API_TIMEOUT', 'API_RATE_LIMIT', 'NETWORK_ERROR', 'LLM_TIMEOUT']:
            error = AgentExecutionError(error_code, 'Test error')
            assert is_retryable_error(error, policy) is True, f"{error_code} should be retryable"
    
    def test_non_retryable_error_always_non_retryable(self):
        """Test errors in NON_RETRYABLE_ERROR_CODES are never retried."""
        # Even with permissive policy, non-retryable errors should not retry
        policy = RetryPolicy(
            max_attempts=3,
            retryable_errors=list(NON_RETRYABLE_ERROR_CODES)  # Try to make them retryable
        )
        
        # Test common non-retryable errors
        for error_code in ['INVALID_MIME_TYPE', 'SCHEMA_VIOLATION', 'CORRUPTED_PDF']:
            error = AgentExecutionError(error_code, 'Test error')
            assert is_retryable_error(error, policy) is False, f"{error_code} should never be retryable"
    
    def test_error_without_error_code_not_retryable(self):
        """Test errors without error_code attribute are not retried."""
        policy = RetryPolicy(max_attempts=3, retryable_errors=['API_TIMEOUT'])
        
        # Standard exception without error_code
        error = ValueError("Some error")
        assert is_retryable_error(error, policy) is False
    
    def test_unknown_error_code_not_retryable(self):
        """Test unknown error codes are not retried by default."""
        policy = RetryPolicy(max_attempts=3, retryable_errors=[])
        
        error = AgentExecutionError('UNKNOWN_ERROR_CODE', 'Unknown error')
        assert is_retryable_error(error, policy) is False


class TestExecuteWithRetry:
    """Test retry execution logic."""
    
    def test_success_on_first_attempt(self):
        """Test function succeeds on first attempt without retry."""
        mock_func = Mock(return_value="success")
        policy = RetryPolicy(max_attempts=3, retryable_errors=['API_TIMEOUT'])
        
        result = execute_with_retry(mock_func, policy, "test operation")
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_success_after_retries(self):
        """Test function succeeds after retryable failures."""
        # Fail twice, then succeed
        mock_func = Mock(side_effect=[
            AgentExecutionError('API_TIMEOUT', 'Timeout 1'),
            AgentExecutionError('API_TIMEOUT', 'Timeout 2'),
            "success"
        ])
        policy = RetryPolicy(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.CONSTANT,
            base_delay_seconds=0.01,  # Short delay for testing
            retryable_errors=['API_TIMEOUT']
        )
        
        result = execute_with_retry(mock_func, policy, "test operation")
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_failure_after_max_attempts(self):
        """Test function fails after exhausting all retry attempts."""
        # Always fail
        mock_func = Mock(side_effect=AgentExecutionError('API_TIMEOUT', 'Always timeout'))
        policy = RetryPolicy(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.CONSTANT,
            base_delay_seconds=0.01,
            retryable_errors=['API_TIMEOUT']
        )
        
        with pytest.raises(AgentExecutionError) as exc_info:
            execute_with_retry(mock_func, policy, "test operation")
        
        assert exc_info.value.error_code == 'API_TIMEOUT'
        assert mock_func.call_count == 3
    
    def test_non_retryable_error_no_retry(self):
        """Test non-retryable errors are not retried."""
        mock_func = Mock(side_effect=AgentExecutionError('INVALID_INPUT', 'Bad input'))
        policy = RetryPolicy(
            max_attempts=3,
            retryable_errors=['API_TIMEOUT']
        )
        
        with pytest.raises(AgentExecutionError) as exc_info:
            execute_with_retry(mock_func, policy, "test operation")
        
        assert exc_info.value.error_code == 'INVALID_INPUT'
        assert mock_func.call_count == 1  # No retry
    
    def test_backoff_delay_applied(self):
        """Test backoff delay is applied between retries."""
        # Fail twice, then succeed
        mock_func = Mock(side_effect=[
            AgentExecutionError('API_TIMEOUT', 'Timeout 1'),
            AgentExecutionError('API_TIMEOUT', 'Timeout 2'),
            "success"
        ])
        policy = RetryPolicy(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.CONSTANT,
            base_delay_seconds=0.1,  # 100ms delay
            retryable_errors=['API_TIMEOUT']
        )
        
        start_time = time.time()
        result = execute_with_retry(mock_func, policy, "test operation")
        elapsed_time = time.time() - start_time
        
        assert result == "success"
        # Should have 2 delays of ~0.1s each = ~0.2s total
        assert elapsed_time >= 0.2, f"Expected at least 0.2s delay, got {elapsed_time:.3f}s"
    
    def test_max_attempts_one_no_retry(self):
        """Test max_attempts=1 means no retry."""
        mock_func = Mock(side_effect=AgentExecutionError('API_TIMEOUT', 'Timeout'))
        policy = RetryPolicy(
            max_attempts=1,
            retryable_errors=['API_TIMEOUT']
        )
        
        with pytest.raises(AgentExecutionError):
            execute_with_retry(mock_func, policy, "test operation")
        
        assert mock_func.call_count == 1


class TestCreateRetryPolicy:
    """Test retry policy creation for different agent types."""
    
    def test_validation_agent_no_retry(self):
        """Test validation agent has no retry (deterministic)."""
        policy = create_retry_policy('validation', max_attempts=1)
        
        assert policy.max_attempts == 1
        assert policy.retryable_errors == []
    
    def test_json_sanitizer_no_retry(self):
        """Test JSON sanitizer has no retry (deterministic)."""
        policy = create_retry_policy('json_sanitizer', max_attempts=1)
        
        assert policy.max_attempts == 1
        assert policy.retryable_errors == []
    
    def test_vision_ocr_agent_retries_api_errors(self):
        """Test Vision+OCR agent retries API errors."""
        policy = create_retry_policy('vision_ocr', max_attempts=3)
        
        assert policy.max_attempts == 3
        assert 'API_TIMEOUT' in policy.retryable_errors
        assert 'API_RATE_LIMIT' in policy.retryable_errors
        assert 'LLM_TIMEOUT' in policy.retryable_errors
        assert policy.backoff_strategy == BackoffStrategy.EXPONENTIAL
    
    def test_script_architect_retries_llm_errors(self):
        """Test Script Architect retries LLM errors."""
        policy = create_retry_policy('script_architect', max_attempts=3)
        
        assert policy.max_attempts == 3
        assert 'LLM_TIMEOUT' in policy.retryable_errors
        assert 'LLM_RATE_LIMIT' in policy.retryable_errors
        assert 'API_TIMEOUT' in policy.retryable_errors
    
    def test_animation_generator_retries_llm_errors(self):
        """Test Animation Generator retries LLM errors."""
        policy = create_retry_policy('animation_generator', max_attempts=3)
        
        assert policy.max_attempts == 3
        assert 'LLM_TIMEOUT' in policy.retryable_errors
        assert 'LLM_RATE_LIMIT' in policy.retryable_errors
    
    def test_persistence_agent_retries_disk_errors(self):
        """Test Persistence agent retries transient disk errors."""
        policy = create_retry_policy('persistence', max_attempts=3)
        
        assert policy.max_attempts == 3
        assert 'TEMPORARY_DISK_FULL' in policy.retryable_errors
        assert 'TEMPORARY_RESOURCE_UNAVAILABLE' in policy.retryable_errors
    
    def test_custom_retryable_errors(self):
        """Test custom retryable errors override defaults."""
        custom_errors = ['CUSTOM_ERROR_1', 'CUSTOM_ERROR_2']
        policy = create_retry_policy('vision_ocr', max_attempts=3, retryable_errors=custom_errors)
        
        assert policy.max_attempts == 3
        assert policy.retryable_errors == custom_errors
    
    def test_default_backoff_parameters(self):
        """Test default backoff parameters are reasonable."""
        policy = create_retry_policy('vision_ocr', max_attempts=3)
        
        assert policy.backoff_strategy == BackoffStrategy.EXPONENTIAL
        assert policy.base_delay_seconds == 1.0
        assert policy.max_delay_seconds == 60.0


class TestRetryPolicyIntegration:
    """Integration tests for retry policy with realistic scenarios."""
    
    def test_api_timeout_with_exponential_backoff(self):
        """Test realistic API timeout scenario with exponential backoff."""
        call_count = 0
        
        def flaky_api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise AgentExecutionError('API_TIMEOUT', f'Timeout on attempt {call_count}')
            return {'status': 'success', 'data': 'result'}
        
        policy = RetryPolicy(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay_seconds=0.05,  # Short for testing
            max_delay_seconds=1.0,
            retryable_errors=['API_TIMEOUT']
        )
        
        result = execute_with_retry(flaky_api_call, policy, "API call")
        
        assert result == {'status': 'success', 'data': 'result'}
        assert call_count == 3
    
    def test_validation_error_aborts_immediately(self):
        """Test validation errors abort without retry."""
        call_count = 0
        
        def validate_input():
            nonlocal call_count
            call_count += 1
            raise AgentExecutionError('INVALID_MIME_TYPE', 'Not a PDF')
        
        policy = RetryPolicy(
            max_attempts=3,
            retryable_errors=['API_TIMEOUT']  # Validation errors not in list
        )
        
        with pytest.raises(AgentExecutionError) as exc_info:
            execute_with_retry(validate_input, policy, "validation")
        
        assert exc_info.value.error_code == 'INVALID_MIME_TYPE'
        assert call_count == 1  # No retry
    
    def test_mixed_errors_retry_only_retryable(self):
        """Test mixed retryable and non-retryable errors."""
        call_count = 0
        
        def mixed_errors():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise AgentExecutionError('API_TIMEOUT', 'Timeout')
            elif call_count == 2:
                raise AgentExecutionError('INVALID_INPUT', 'Bad input')
            return 'success'
        
        policy = RetryPolicy(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.CONSTANT,
            base_delay_seconds=0.01,
            retryable_errors=['API_TIMEOUT']
        )
        
        # First call: API_TIMEOUT (retryable) -> retry
        # Second call: INVALID_INPUT (non-retryable) -> abort
        with pytest.raises(AgentExecutionError) as exc_info:
            execute_with_retry(mixed_errors, policy, "operation")
        
        assert exc_info.value.error_code == 'INVALID_INPUT'
        assert call_count == 2  # Retried once, then aborted


class TestRetryPolicyEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_base_delay(self):
        """Test zero base delay (immediate retry)."""
        delay = calculate_backoff_delay(0, BackoffStrategy.EXPONENTIAL, 0.0, 60.0)
        assert delay == 0.0
    
    def test_very_large_attempt_number(self):
        """Test very large attempt number doesn't overflow."""
        delay = calculate_backoff_delay(1000, BackoffStrategy.EXPONENTIAL, 1.0, 60.0)
        assert delay == 60.0  # Capped at max
    
    def test_max_delay_less_than_base_delay(self):
        """Test max_delay less than base_delay."""
        # Should cap at max_delay even on first attempt
        delay = calculate_backoff_delay(0, BackoffStrategy.EXPONENTIAL, 10.0, 5.0)
        assert delay == 5.0
    
    def test_empty_retryable_errors_list(self):
        """Test empty retryable_errors list uses defaults."""
        policy = RetryPolicy(max_attempts=3, retryable_errors=[])
        
        # Should use default RETRYABLE_ERROR_CODES
        error = AgentExecutionError('API_TIMEOUT', 'Timeout')
        assert is_retryable_error(error, policy) is True
    
    def test_none_retryable_errors_uses_empty_list(self):
        """Test None retryable_errors is converted to empty list."""
        policy = RetryPolicy(max_attempts=3)
        assert policy.retryable_errors == []
