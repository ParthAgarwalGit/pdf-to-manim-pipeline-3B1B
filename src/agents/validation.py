"""Validation Agent for PDF file verification

This module implements the Validation Agent which verifies file type, size,
and structural integrity before processing.
"""

import os
from typing import Union

from src.agents.base import Agent, AgentExecutionError, AgentInput, AgentOutput, RetryPolicy
from src.schemas.file_reference import FileReference


class ValidationError(AgentOutput):
    """Error object returned when validation fails
    
    Attributes:
        error_code: Machine-readable error code
        reason: Human-readable failure reason
        file_path: Path to the file that failed validation
    """
    
    def __init__(self, error_code: str, reason: str, file_path: str):
        self.error_code = error_code
        self.reason = reason
        self.file_path = file_path
    
    def __repr__(self):
        return f"ValidationError(error_code='{self.error_code}', reason='{self.reason}', file_path='{self.file_path}')"


class ValidationAgent(Agent):
    """Agent responsible for verifying PDF file type and structural integrity
    
    The ValidationAgent performs the following checks:
    1. Verify file has PDF MIME type (check for %PDF- magic bytes)
    2. Verify file size is within bounds [100KB, 50MB]
    3. Verify PDF structure is not corrupted (check for %PDF- header and %%EOF trailer)
    
    If all checks pass, returns the validated FileReference.
    If any check fails, returns a ValidationError with specific failure reason.
    """
    
    # File size bounds
    MIN_FILE_SIZE = 100 * 1024  # 100KB
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    # PDF magic bytes and markers
    PDF_MAGIC_BYTES = b'%PDF-'
    PDF_TRAILER = b'%%EOF'
    
    def execute(self, input_data: AgentInput) -> AgentOutput:
        """Execute validation checks on the PDF file
        
        Args:
            input_data: FileReference object from Ingestion Agent
            
        Returns:
            Validated FileReference on success OR ValidationError on failure
            
        Raises:
            AgentExecutionError: For unrecoverable failures (e.g., file not readable)
        """
        if not self.validate_input(input_data):
            raise AgentExecutionError(
                error_code="INVALID_INPUT",
                message="Input must be a FileReference object",
                context={"input_type": type(input_data).__name__}
            )
        
        file_ref: FileReference = input_data
        file_path = file_ref.file_path
        
        # Check 1: Verify file is readable
        if not os.path.exists(file_path):
            return ValidationError(
                error_code="FILE_NOT_FOUND",
                reason=f"File does not exist: {file_path}",
                file_path=file_path
            )
        
        if not os.path.isfile(file_path):
            return ValidationError(
                error_code="NOT_A_FILE",
                reason=f"Path is not a file: {file_path}",
                file_path=file_path
            )
        
        try:
            with open(file_path, 'rb') as f:
                # Read first 1024 bytes for header check
                header = f.read(1024)
                
                # Check 2: Verify PDF MIME type (magic bytes)
                if not header.startswith(self.PDF_MAGIC_BYTES):
                    return ValidationError(
                        error_code="INVALID_MIME_TYPE",
                        reason=f"File does not have PDF magic bytes (%PDF-): {file_path}",
                        file_path=file_path
                    )
                
                # Check 3: Verify file size is within bounds
                file_size = file_ref.size_bytes
                if file_size < self.MIN_FILE_SIZE:
                    return ValidationError(
                        error_code="INVALID_FILE_SIZE",
                        reason=f"File size {file_size} bytes is below minimum {self.MIN_FILE_SIZE} bytes",
                        file_path=file_path
                    )
                
                if file_size > self.MAX_FILE_SIZE:
                    return ValidationError(
                        error_code="INVALID_FILE_SIZE",
                        reason=f"File size {file_size} bytes exceeds maximum {self.MAX_FILE_SIZE} bytes",
                        file_path=file_path
                    )
                
                # Check 4: Verify PDF structure (check for %%EOF trailer)
                # Read last 1024 bytes to find trailer
                f.seek(max(0, file_size - 1024))
                trailer = f.read()
                
                if self.PDF_TRAILER not in trailer:
                    return ValidationError(
                        error_code="CORRUPTED_PDF",
                        reason=f"PDF structure is corrupted (missing %%EOF trailer): {file_path}",
                        file_path=file_path
                    )
        
        except PermissionError:
            return ValidationError(
                error_code="FILE_UNREADABLE",
                reason=f"Permission denied reading file: {file_path}",
                file_path=file_path
            )
        except OSError as e:
            return ValidationError(
                error_code="FILE_UNREADABLE",
                reason=f"Error reading file: {str(e)}",
                file_path=file_path
            )
        
        # All checks passed - return validated FileReference
        return file_ref
    
    def validate_input(self, input_data: AgentInput) -> bool:
        """Validate input is a FileReference object
        
        Args:
            input_data: Input object to validate
            
        Returns:
            True if input is a valid FileReference, False otherwise
        """
        return isinstance(input_data, FileReference)
    
    def get_retry_policy(self) -> RetryPolicy:
        """Return retry policy for validation
        
        Validation failures are deterministic and indicate invalid input,
        so no retry is performed.
        
        Returns:
            RetryPolicy with max_attempts=1 (no retry)
        """
        return RetryPolicy(
            max_attempts=1,
            retryable_errors=[]
        )
