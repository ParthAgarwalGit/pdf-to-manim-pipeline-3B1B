"""Base Agent interface for the PDF to Manim pipeline

This module defines the core Agent interface that all pipeline agents must implement.
Each agent has a single responsibility and explicit input/output contracts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class BackoffStrategy(Enum):
    """Retry backoff strategies"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"


@dataclass
class RetryPolicy:
    """Retry policy configuration for agent execution
    
    Attributes:
        max_attempts: Maximum number of execution attempts (including initial attempt)
        backoff_strategy: Strategy for calculating retry delays
        base_delay_seconds: Base delay for backoff calculation
        max_delay_seconds: Maximum delay between retries
        retryable_errors: List of error codes that should trigger retry
    """
    max_attempts: int = 1
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    retryable_errors: List[str] = None
    
    def __post_init__(self):
        if self.retryable_errors is None:
            self.retryable_errors = []


class AgentInput(ABC):
    """Base class for agent input data
    
    All agent-specific input classes should inherit from this base class.
    """
    pass


class AgentOutput(ABC):
    """Base class for agent output data
    
    All agent-specific output classes should inherit from this base class.
    """
    pass


class AgentExecutionError(Exception):
    """Exception raised for unrecoverable agent execution failures
    
    Attributes:
        error_code: Machine-readable error code
        message: Human-readable error message
        context: Additional context about the failure
    """
    
    def __init__(self, error_code: str, message: str, context: Optional[Dict[str, Any]] = None):
        self.error_code = error_code
        self.message = message
        self.context = context or {}
        super().__init__(f"[{error_code}] {message}")


class Agent(ABC):
    """Base interface for all pipeline agents
    
    Each agent implements a single stage of the PDF-to-Manim pipeline with
    explicit inputs, outputs, and failure modes. Agents are designed to be
    composable and testable.
    
    The agent interface provides three core methods:
    - execute(): Perform the agent's primary task
    - validate_input(): Verify input conforms to expected schema
    - get_retry_policy(): Define retry behavior for transient failures
    """
    
    @abstractmethod
    def execute(self, input_data: AgentInput) -> AgentOutput:
        """Execute the agent's primary task
        
        Args:
            input_data: Agent-specific input object conforming to expected schema
            
        Returns:
            Agent-specific output object OR error object
            
        Raises:
            AgentExecutionError: For unrecoverable failures that should abort pipeline
        """
        pass
    
    def validate_input(self, input_data: AgentInput) -> bool:
        """Validate input conforms to expected schema
        
        Args:
            input_data: Agent-specific input object to validate
            
        Returns:
            True if input is valid, False otherwise
            
        Note:
            This method should perform schema validation only, not business logic.
            It should be fast and deterministic.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement validate_input()"
        )
    
    def get_retry_policy(self) -> RetryPolicy:
        """Return retry policy for this agent
        
        Returns:
            RetryPolicy object defining retry behavior
            
        Note:
            Agents with deterministic failures (e.g., validation) should return
            a policy with max_attempts=1. Agents with transient failures (e.g.,
            API calls) should return a policy with exponential backoff.
        """
        return RetryPolicy(max_attempts=1)
