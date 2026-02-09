"""Retry policy implementation for the PDF to Manim pipeline.

This module provides retry logic with exponential backoff for handling
transient failures in agent execution. It distinguishes between retryable
errors (API timeouts, rate limits) and non-retryable errors (invalid input,
schema violations).

The retry system supports:
- Exponential, linear, and constant backoff strategies
- Configurable max attempts and delay bounds
- Error code-based retry decisions
- Structured logging of retry attempts
"""

import time
import logging
from typing import Callable, TypeVar, Any, Optional
from dataclasses import dataclass

from src.agents.base import (
    Agent,
    AgentInput,
    AgentOutput,
    AgentExecutionError,
    RetryPolicy,
    BackoffStrategy
)


logger = logging.getLogger(__name__)


# Type variable for generic retry function
T = TypeVar('T')


# Common retryable error codes
RETRYABLE_ERROR_CODES = {
    # Network and API errors
    'API_TIMEOUT',
    'API_RATE_LIMIT',
    'API_UNAVAILABLE',
    'NETWORK_ERROR',
    'CONNECTION_TIMEOUT',
    'SERVICE_UNAVAILABLE',
    
    # Transient resource errors
    'TEMPORARY_DISK_FULL',
    'TEMPORARY_RESOURCE_UNAVAILABLE',
    
    # LLM-specific errors
    'LLM_TIMEOUT',
    'LLM_RATE_LIMIT',
    'LLM_OVERLOADED',
}


# Non-retryable error codes (deterministic failures)
NON_RETRYABLE_ERROR_CODES = {
    # Validation errors
    'INVALID_MIME_TYPE',
    'INVALID_FILE_SIZE',
    'CORRUPTED_PDF',
    'FILE_UNREADABLE',
    
    # Schema errors
    'SCHEMA_VIOLATION',
    'INVALID_JSON',
    'MISSING_REQUIRED_FIELD',
    
    # Input errors
    'INVALID_INPUT',
    'MALFORMED_PAYLOAD',
    'FILE_NOT_FOUND',
    
    # Logic errors
    'INVALID_CONFIGURATION',
    'UNSUPPORTED_OPERATION',
}


@dataclass
class RetryContext:
    """Context information for retry attempts.
    
    Attributes:
        agent_name: Name of the agent being retried
        attempt: Current attempt number (0-indexed)
        max_attempts: Maximum number of attempts
        last_error: Last error encountered
        total_delay: Total delay accumulated across retries
    """
    agent_name: str
    attempt: int
    max_attempts: int
    last_error: Optional[Exception] = None
    total_delay: float = 0.0


def calculate_backoff_delay(
    attempt: int,
    strategy: BackoffStrategy,
    base_delay: float,
    max_delay: float
) -> float:
    """Calculate backoff delay for retry attempt.
    
    Args:
        attempt: Current attempt number (0-indexed)
        strategy: Backoff strategy to use
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Delay in seconds, capped at max_delay
        
    Examples:
        >>> calculate_backoff_delay(0, BackoffStrategy.EXPONENTIAL, 1.0, 60.0)
        1.0
        >>> calculate_backoff_delay(1, BackoffStrategy.EXPONENTIAL, 1.0, 60.0)
        2.0
        >>> calculate_backoff_delay(2, BackoffStrategy.EXPONENTIAL, 1.0, 60.0)
        4.0
        >>> calculate_backoff_delay(10, BackoffStrategy.EXPONENTIAL, 1.0, 60.0)
        60.0
    """
    if strategy == BackoffStrategy.EXPONENTIAL:
        # Exponential: base_delay * 2^attempt
        delay = base_delay * (2 ** attempt)
    elif strategy == BackoffStrategy.LINEAR:
        # Linear: base_delay * (attempt + 1)
        delay = base_delay * (attempt + 1)
    else:  # CONSTANT
        # Constant: always base_delay
        delay = base_delay
    
    # Cap at max_delay
    return min(delay, max_delay)


def is_retryable_error(error: Exception, retry_policy: RetryPolicy) -> bool:
    """Determine if an error is retryable based on retry policy.
    
    Args:
        error: Exception to check
        retry_policy: Retry policy with retryable error codes
        
    Returns:
        True if error should be retried, False otherwise
        
    Logic:
        1. If error has error_code attribute, check against policy's retryable_errors
        2. If policy has empty retryable_errors list, use default RETRYABLE_ERROR_CODES
        3. If error_code is in NON_RETRYABLE_ERROR_CODES, never retry
        4. Otherwise, don't retry unknown errors
    """
    # Extract error code if available
    error_code = getattr(error, 'error_code', None)
    
    if error_code is None:
        # No error code, don't retry
        return False
    
    # Never retry explicitly non-retryable errors
    if error_code in NON_RETRYABLE_ERROR_CODES:
        return False
    
    # Check against policy's retryable errors
    if retry_policy.retryable_errors:
        return error_code in retry_policy.retryable_errors
    
    # If policy has no specific retryable errors, use defaults
    return error_code in RETRYABLE_ERROR_CODES


def execute_with_retry(
    func: Callable[[], T],
    retry_policy: RetryPolicy,
    context_name: str = "operation"
) -> T:
    """Execute a function with retry logic.
    
    Args:
        func: Function to execute (should take no arguments)
        retry_policy: Retry policy to apply
        context_name: Name for logging context
        
    Returns:
        Result of successful function execution
        
    Raises:
        Exception: Last exception if all retries exhausted
        
    Example:
        >>> policy = RetryPolicy(
        ...     max_attempts=3,
        ...     backoff_strategy=BackoffStrategy.EXPONENTIAL,
        ...     base_delay_seconds=1.0,
        ...     retryable_errors=['API_TIMEOUT']
        ... )
        >>> result = execute_with_retry(
        ...     lambda: api_call(),
        ...     policy,
        ...     "API call"
        ... )
    """
    last_error = None
    total_delay = 0.0
    
    for attempt in range(retry_policy.max_attempts):
        try:
            # Execute function
            result = func()
            
            # Log success if this was a retry
            if attempt > 0:
                logger.info(
                    f"{context_name} succeeded on attempt {attempt + 1} "
                    f"after {total_delay:.2f}s total delay"
                )
            
            return result
            
        except Exception as e:
            last_error = e
            
            # Check if we should retry
            is_last_attempt = (attempt == retry_policy.max_attempts - 1)
            should_retry = (
                not is_last_attempt
                and is_retryable_error(e, retry_policy)
            )
            
            if not should_retry:
                # Don't retry - either last attempt or non-retryable error
                if is_last_attempt:
                    logger.error(
                        f"{context_name} failed after {retry_policy.max_attempts} attempts"
                    )
                else:
                    error_code = getattr(e, 'error_code', 'UNKNOWN')
                    logger.error(
                        f"{context_name} failed with non-retryable error: {error_code}"
                    )
                raise
            
            # Calculate backoff delay
            delay = calculate_backoff_delay(
                attempt,
                retry_policy.backoff_strategy,
                retry_policy.base_delay_seconds,
                retry_policy.max_delay_seconds
            )
            total_delay += delay
            
            # Log retry
            error_code = getattr(e, 'error_code', 'UNKNOWN')
            logger.warning(
                f"{context_name} failed with {error_code}, "
                f"retrying in {delay:.2f}s (attempt {attempt + 2}/{retry_policy.max_attempts})"
            )
            
            # Wait before retry
            time.sleep(delay)
    
    # Should never reach here, but raise last error if we do
    raise last_error


def create_retry_policy(
    agent_type: str,
    max_attempts: int = 1,
    retryable_errors: Optional[list] = None
) -> RetryPolicy:
    """Create a retry policy for a specific agent type.
    
    Args:
        agent_type: Type of agent (for determining defaults)
        max_attempts: Maximum number of attempts
        retryable_errors: List of retryable error codes (None = use defaults)
        
    Returns:
        RetryPolicy configured for the agent type
        
    Agent-specific defaults:
        - Ingestion: Retry transient network/storage errors
        - Validation: No retry (deterministic)
        - Vision+OCR: Retry API timeouts and rate limits
        - ScriptArchitect: Retry LLM timeouts and rate limits
        - JSONSanitizer: No retry (deterministic)
        - SceneController: Retry per-scene failures
        - AnimationGenerator: Retry LLM and syntax errors
        - Persistence: Retry transient disk errors
    """
    # Default: no retry
    if max_attempts == 1:
        return RetryPolicy(
            max_attempts=1,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay_seconds=1.0,
            max_delay_seconds=60.0,
            retryable_errors=[]
        )
    
    # Determine retryable errors based on agent type
    if retryable_errors is None:
        if agent_type in ['ingestion', 'vision_ocr', 'script_architect', 'animation_generator']:
            # API-based agents: retry network and API errors
            retryable_errors = [
                'API_TIMEOUT',
                'API_RATE_LIMIT',
                'API_UNAVAILABLE',
                'NETWORK_ERROR',
                'CONNECTION_TIMEOUT',
                'LLM_TIMEOUT',
                'LLM_RATE_LIMIT',
                'LLM_OVERLOADED',
            ]
        elif agent_type == 'persistence':
            # Persistence: retry transient disk errors
            retryable_errors = [
                'TEMPORARY_DISK_FULL',
                'TEMPORARY_RESOURCE_UNAVAILABLE',
            ]
        else:
            # Default: use common retryable errors
            retryable_errors = list(RETRYABLE_ERROR_CODES)
    
    return RetryPolicy(
        max_attempts=max_attempts,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        base_delay_seconds=1.0,
        max_delay_seconds=60.0,
        retryable_errors=retryable_errors
    )


class RetryableOperation:
    """Context manager for retryable operations.
    
    Provides a convenient way to wrap operations with retry logic.
    
    Example:
        >>> policy = create_retry_policy('vision_ocr', max_attempts=3)
        >>> with RetryableOperation(policy, "OCR extraction") as retry:
        ...     result = perform_ocr()
    """
    
    def __init__(self, retry_policy: RetryPolicy, operation_name: str):
        """Initialize retryable operation.
        
        Args:
            retry_policy: Retry policy to apply
            operation_name: Name for logging
        """
        self.retry_policy = retry_policy
        self.operation_name = operation_name
        self.attempt = 0
        self.total_delay = 0.0
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if exc_type is None:
            # Success
            if self.attempt > 0:
                logger.info(
                    f"{self.operation_name} succeeded on attempt {self.attempt + 1} "
                    f"after {self.total_delay:.2f}s total delay"
                )
            return True
        
        # Check if we should retry
        is_last_attempt = (self.attempt == self.retry_policy.max_attempts - 1)
        should_retry = (
            not is_last_attempt
            and is_retryable_error(exc_val, self.retry_policy)
        )
        
        if not should_retry:
            # Don't suppress exception
            return False
        
        # Calculate backoff delay
        delay = calculate_backoff_delay(
            self.attempt,
            self.retry_policy.backoff_strategy,
            self.retry_policy.base_delay_seconds,
            self.retry_policy.max_delay_seconds
        )
        self.total_delay += delay
        
        # Log retry
        error_code = getattr(exc_val, 'error_code', 'UNKNOWN')
        logger.warning(
            f"{self.operation_name} failed with {error_code}, "
            f"retrying in {delay:.2f}s (attempt {self.attempt + 2}/{self.retry_policy.max_attempts})"
        )
        
        # Wait before retry
        time.sleep(delay)
        self.attempt += 1
        
        # Suppress exception to allow retry
        return True
