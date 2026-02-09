"""File reference schema for PDF ingestion."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FileReference(BaseModel):
    """
    File reference object produced by Ingestion Agent.
    
    Represents metadata about an uploaded PDF file.
    """
    
    file_path: str = Field(
        ...,
        description="Full path to the file in cloud storage or local filesystem"
    )
    
    filename: str = Field(
        ...,
        description="Name of the file (without path)"
    )
    
    size_bytes: int = Field(
        ...,
        ge=0,
        description="File size in bytes"
    )
    
    upload_timestamp: datetime = Field(
        ...,
        description="ISO8601 timestamp of when the file was uploaded"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the file (e.g., content type, user info)"
    )
    
    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Ensure filename is not empty."""
        if not v or not v.strip():
            raise ValueError("filename cannot be empty")
        return v
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Ensure file_path is not empty."""
        if not v or not v.strip():
            raise ValueError("file_path cannot be empty")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_path": "s3://bucket/uploads/document.pdf",
                "filename": "document.pdf",
                "size_bytes": 1048576,
                "upload_timestamp": "2024-01-15T10:30:00Z",
                "metadata": {
                    "content_type": "application/pdf",
                    "user_id": "user123"
                }
            }
        }
    )
