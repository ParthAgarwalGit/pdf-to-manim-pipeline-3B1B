"""Unit tests for ValidationAgent

Tests cover:
- PDF MIME type verification (magic bytes check)
- File size validation (100KB to 50MB bounds)
- PDF structure validation (header and trailer)
- Error handling for various failure modes
- Property-based tests for validation correctness
"""

import os
import tempfile
from datetime import datetime

import pytest
from hypothesis import given, strategies as st, settings

from src.agents.validation import ValidationAgent, ValidationError
from src.schemas.file_reference import FileReference


class TestValidationAgent:
    """Test suite for ValidationAgent"""
    
    @pytest.fixture
    def agent(self):
        """Create a ValidationAgent instance"""
        return ValidationAgent()
    
    @pytest.fixture
    def valid_pdf_file(self):
        """Create a temporary valid PDF file for testing"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            # Write minimal valid PDF structure
            f.write(b'%PDF-1.4\n')
            f.write(b'1 0 obj\n')
            f.write(b'<< /Type /Catalog /Pages 2 0 R >>\n')
            f.write(b'endobj\n')
            f.write(b'2 0 obj\n')
            f.write(b'<< /Type /Pages /Kids [] /Count 0 >>\n')
            f.write(b'endobj\n')
            f.write(b'xref\n')
            f.write(b'0 3\n')
            f.write(b'0000000000 65535 f\n')
            f.write(b'0000000009 65535 n\n')
            f.write(b'0000000058 65535 n\n')
            f.write(b'trailer\n')
            f.write(b'<< /Size 3 /Root 1 0 R >>\n')
            f.write(b'startxref\n')
            f.write(b'116\n')
            # Pad to meet minimum size requirement (100KB)
            current_size = f.tell()
            padding_needed = ValidationAgent.MIN_FILE_SIZE - current_size - len(b'%%EOF\n')
            if padding_needed > 0:
                f.write(b'%' + b' ' * (padding_needed - 1) + b'\n')
            f.write(b'%%EOF\n')
            
            file_path = f.name
        
        yield file_path
        
        # Cleanup
        if os.path.exists(file_path):
            os.unlink(file_path)
    
    @pytest.fixture
    def file_reference(self, valid_pdf_file):
        """Create a FileReference for the valid PDF"""
        file_size = os.path.getsize(valid_pdf_file)
        return FileReference(
            file_path=valid_pdf_file,
            filename=os.path.basename(valid_pdf_file),
            size_bytes=file_size,
            upload_timestamp=datetime.now(),
            metadata={"content_type": "application/pdf"}
        )
    
    def test_valid_pdf_passes_validation(self, agent, file_reference):
        """Test that a valid PDF passes all validation checks"""
        result = agent.execute(file_reference)
        
        # Should return the same FileReference object
        assert isinstance(result, FileReference)
        assert result.file_path == file_reference.file_path
        assert result.filename == file_reference.filename
        assert result.size_bytes == file_reference.size_bytes
    
    def test_invalid_input_type_raises_error(self, agent):
        """Test that invalid input type raises AgentExecutionError"""
        from src.agents.base import AgentExecutionError
        
        with pytest.raises(AgentExecutionError) as exc_info:
            agent.execute("not a FileReference")
        
        assert exc_info.value.error_code == "INVALID_INPUT"
    
    def test_file_not_found_returns_error(self, agent):
        """Test that non-existent file returns FILE_NOT_FOUND error"""
        file_ref = FileReference(
            file_path="/nonexistent/path/file.pdf",
            filename="file.pdf",
            size_bytes=1000000,
            upload_timestamp=datetime.now(),
            metadata={}
        )
        
        result = agent.execute(file_ref)
        
        assert isinstance(result, ValidationError)
        assert result.error_code == "FILE_NOT_FOUND"
        assert "does not exist" in result.reason
    
    def test_directory_path_returns_error(self, agent):
        """Test that directory path returns NOT_A_FILE error"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_ref = FileReference(
                file_path=tmpdir,
                filename="directory",
                size_bytes=0,
                upload_timestamp=datetime.now(),
                metadata={}
            )
            
            result = agent.execute(file_ref)
            
            assert isinstance(result, ValidationError)
            assert result.error_code == "NOT_A_FILE"
            assert "not a file" in result.reason
    
    def test_non_pdf_file_returns_invalid_mime_type(self, agent):
        """Test that non-PDF file returns INVALID_MIME_TYPE error"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.txt') as f:
            f.write(b'This is not a PDF file\n')
            file_path = f.name
        
        try:
            file_size = os.path.getsize(file_path)
            file_ref = FileReference(
                file_path=file_path,
                filename=os.path.basename(file_path),
                size_bytes=file_size,
                upload_timestamp=datetime.now(),
                metadata={}
            )
            
            result = agent.execute(file_ref)
            
            assert isinstance(result, ValidationError)
            assert result.error_code == "INVALID_MIME_TYPE"
            assert "magic bytes" in result.reason
        finally:
            os.unlink(file_path)
    
    def test_file_too_small_returns_invalid_file_size(self, agent):
        """Test that file smaller than 100KB returns INVALID_FILE_SIZE error"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            # Write minimal PDF (less than 100KB)
            f.write(b'%PDF-1.4\n')
            f.write(b'%%EOF\n')
            file_path = f.name
        
        try:
            file_size = os.path.getsize(file_path)
            assert file_size < ValidationAgent.MIN_FILE_SIZE
            
            file_ref = FileReference(
                file_path=file_path,
                filename=os.path.basename(file_path),
                size_bytes=file_size,
                upload_timestamp=datetime.now(),
                metadata={}
            )
            
            result = agent.execute(file_ref)
            
            assert isinstance(result, ValidationError)
            assert result.error_code == "INVALID_FILE_SIZE"
            assert "below minimum" in result.reason
        finally:
            os.unlink(file_path)
    
    def test_file_too_large_returns_invalid_file_size(self, agent):
        """Test that file larger than 50MB returns INVALID_FILE_SIZE error"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            f.write(b'%PDF-1.4\n')
            f.write(b'%%EOF\n')
            file_path = f.name
        
        try:
            # Create FileReference with size exceeding maximum
            file_ref = FileReference(
                file_path=file_path,
                filename=os.path.basename(file_path),
                size_bytes=ValidationAgent.MAX_FILE_SIZE + 1,
                upload_timestamp=datetime.now(),
                metadata={}
            )
            
            result = agent.execute(file_ref)
            
            assert isinstance(result, ValidationError)
            assert result.error_code == "INVALID_FILE_SIZE"
            assert "exceeds maximum" in result.reason
        finally:
            os.unlink(file_path)
    
    def test_corrupted_pdf_missing_trailer_returns_error(self, agent):
        """Test that PDF without %%EOF trailer returns CORRUPTED_PDF error"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            # Write PDF without %%EOF trailer
            f.write(b'%PDF-1.4\n')
            # Pad to meet minimum size requirement
            padding_needed = ValidationAgent.MIN_FILE_SIZE - len(b'%PDF-1.4\n')
            f.write(b'x' * padding_needed)
            file_path = f.name
        
        try:
            file_size = os.path.getsize(file_path)
            file_ref = FileReference(
                file_path=file_path,
                filename=os.path.basename(file_path),
                size_bytes=file_size,
                upload_timestamp=datetime.now(),
                metadata={}
            )
            
            result = agent.execute(file_ref)
            
            assert isinstance(result, ValidationError)
            assert result.error_code == "CORRUPTED_PDF"
            assert "missing %%EOF trailer" in result.reason
        finally:
            os.unlink(file_path)
    
    def test_validate_input_accepts_file_reference(self, agent, file_reference):
        """Test that validate_input accepts FileReference objects"""
        assert agent.validate_input(file_reference) is True
    
    def test_validate_input_rejects_non_file_reference(self, agent):
        """Test that validate_input rejects non-FileReference objects"""
        assert agent.validate_input("not a FileReference") is False
        assert agent.validate_input(123) is False
        assert agent.validate_input(None) is False
        assert agent.validate_input({}) is False
    
    def test_retry_policy_no_retry(self, agent):
        """Test that ValidationAgent has no-retry policy"""
        policy = agent.get_retry_policy()
        
        assert policy.max_attempts == 1
        assert len(policy.retryable_errors) == 0
    
    def test_validation_error_representation(self):
        """Test ValidationError string representation"""
        error = ValidationError(
            error_code="TEST_ERROR",
            reason="Test reason",
            file_path="/path/to/file.pdf"
        )
        
        repr_str = repr(error)
        assert "TEST_ERROR" in repr_str
        assert "Test reason" in repr_str
        assert "/path/to/file.pdf" in repr_str
    
    def test_pdf_with_trailer_in_middle_passes(self, agent):
        """Test that PDF with %%EOF in content but also at end passes validation"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            # Write PDF with %%EOF in middle and at end
            f.write(b'%PDF-1.4\n')
            f.write(b'Some content with %%EOF in the middle\n')
            # Pad to meet minimum size
            current_size = f.tell()
            padding_needed = ValidationAgent.MIN_FILE_SIZE - current_size - len(b'%%EOF\n')
            if padding_needed > 0:
                f.write(b'x' * padding_needed)
            f.write(b'%%EOF\n')
            file_path = f.name
        
        try:
            file_size = os.path.getsize(file_path)
            file_ref = FileReference(
                file_path=file_path,
                filename=os.path.basename(file_path),
                size_bytes=file_size,
                upload_timestamp=datetime.now(),
                metadata={}
            )
            
            result = agent.execute(file_ref)
            
            # Should pass validation
            assert isinstance(result, FileReference)
        finally:
            os.unlink(file_path)
    
    def test_minimum_valid_size_pdf_passes(self, agent):
        """Test that PDF exactly at minimum size (100KB) passes validation"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            # Write PDF exactly at minimum size
            f.write(b'%PDF-1.4\n')
            # Pad to exactly minimum size
            padding_size = ValidationAgent.MIN_FILE_SIZE - len(b'%PDF-1.4\n') - len(b'%%EOF\n')
            f.write(b'x' * padding_size)
            f.write(b'%%EOF\n')
            file_path = f.name
        
        try:
            file_size = os.path.getsize(file_path)
            assert file_size >= ValidationAgent.MIN_FILE_SIZE
            
            file_ref = FileReference(
                file_path=file_path,
                filename=os.path.basename(file_path),
                size_bytes=file_size,
                upload_timestamp=datetime.now(),
                metadata={}
            )
            
            result = agent.execute(file_ref)
            
            # Should pass validation
            assert isinstance(result, FileReference)
        finally:
            os.unlink(file_path)
    
    def test_maximum_valid_size_pdf_passes(self, agent):
        """Test that PDF exactly at maximum size (50MB) passes validation"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            # Write PDF header
            f.write(b'%PDF-1.4\n')
            # Pad to near maximum size
            current_size = f.tell()
            padding_needed = ValidationAgent.MAX_FILE_SIZE - current_size - len(b'%%EOF\n')
            if padding_needed > 0:
                f.write(b'x' * padding_needed)
            # Write trailer at end
            f.write(b'%%EOF\n')
            file_path = f.name
        
        try:
            file_size = os.path.getsize(file_path)
            # Create FileReference with actual file size
            file_ref = FileReference(
                file_path=file_path,
                filename=os.path.basename(file_path),
                size_bytes=file_size,
                upload_timestamp=datetime.now(),
                metadata={}
            )
            
            result = agent.execute(file_ref)
            
            # Should pass validation
            assert isinstance(result, FileReference)
        finally:
            os.unlink(file_path)
    
    def test_empty_file_returns_invalid_mime_type(self, agent):
        """Test that empty file (0 bytes) returns INVALID_MIME_TYPE error"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            # Write nothing - empty file
            file_path = f.name
        
        try:
            file_size = os.path.getsize(file_path)
            assert file_size == 0
            
            file_ref = FileReference(
                file_path=file_path,
                filename=os.path.basename(file_path),
                size_bytes=file_size,
                upload_timestamp=datetime.now(),
                metadata={}
            )
            
            result = agent.execute(file_ref)
            
            # Empty file should fail MIME type check (no magic bytes)
            assert isinstance(result, ValidationError)
            assert result.error_code == "INVALID_MIME_TYPE"
        finally:
            os.unlink(file_path)
    
    def test_file_with_partial_header_returns_invalid_mime_type(self, agent):
        """Test that file with partial PDF header returns INVALID_MIME_TYPE error"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            # Write partial header (missing version)
            f.write(b'%PD')
            file_path = f.name
        
        try:
            file_size = os.path.getsize(file_path)
            file_ref = FileReference(
                file_path=file_path,
                filename=os.path.basename(file_path),
                size_bytes=file_size,
                upload_timestamp=datetime.now(),
                metadata={}
            )
            
            result = agent.execute(file_ref)
            
            assert isinstance(result, ValidationError)
            assert result.error_code == "INVALID_MIME_TYPE"
            assert "magic bytes" in result.reason
        finally:
            os.unlink(file_path)
    
    def test_pdf_with_only_header_no_trailer_returns_corrupted_pdf(self, agent):
        """Test that PDF with header but no trailer returns CORRUPTED_PDF error"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            # Write valid header
            f.write(b'%PDF-1.4\n')
            # Add content to meet size requirement but no trailer
            padding_needed = ValidationAgent.MIN_FILE_SIZE - len(b'%PDF-1.4\n')
            # Use exact padding to stay within size limits
            f.write(b'x' * padding_needed)
            file_path = f.name
        
        try:
            file_size = os.path.getsize(file_path)
            file_ref = FileReference(
                file_path=file_path,
                filename=os.path.basename(file_path),
                size_bytes=file_size,
                upload_timestamp=datetime.now(),
                metadata={}
            )
            
            result = agent.execute(file_ref)
            
            assert isinstance(result, ValidationError)
            assert result.error_code == "CORRUPTED_PDF"
            assert "missing %%EOF trailer" in result.reason
        finally:
            os.unlink(file_path)
    
    def test_different_pdf_versions_pass_validation(self, agent):
        """Test that PDFs with different version numbers pass validation"""
        versions = [b'%PDF-1.0', b'%PDF-1.1', b'%PDF-1.2', b'%PDF-1.3', 
                   b'%PDF-1.4', b'%PDF-1.5', b'%PDF-1.6', b'%PDF-1.7', b'%PDF-2.0']
        
        for version in versions:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
                # Write PDF with specific version
                f.write(version + b'\n')
                # Pad to meet minimum size
                current_size = f.tell()
                padding_needed = ValidationAgent.MIN_FILE_SIZE - current_size - len(b'%%EOF\n')
                if padding_needed > 0:
                    f.write(b'x' * padding_needed)
                f.write(b'%%EOF\n')
                file_path = f.name
            
            try:
                file_size = os.path.getsize(file_path)
                file_ref = FileReference(
                    file_path=file_path,
                    filename=os.path.basename(file_path),
                    size_bytes=file_size,
                    upload_timestamp=datetime.now(),
                    metadata={}
                )
                
                result = agent.execute(file_ref)
                
                # All PDF versions should pass validation
                assert isinstance(result, FileReference), \
                    f"PDF version {version.decode()} should pass validation"
            finally:
                os.unlink(file_path)
    
    def test_file_with_case_sensitive_trailer_variations(self, agent):
        """Test that only exact %%EOF trailer is accepted (case-sensitive)"""
        # Test lowercase variation - should fail
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            f.write(b'%PDF-1.4\n')
            padding_needed = ValidationAgent.MIN_FILE_SIZE - len(b'%PDF-1.4\n') - len(b'%%eof\n')
            if padding_needed > 0:
                f.write(b'x' * padding_needed)
            f.write(b'%%eof\n')  # lowercase - should fail
            file_path = f.name
        
        try:
            file_size = os.path.getsize(file_path)
            file_ref = FileReference(
                file_path=file_path,
                filename=os.path.basename(file_path),
                size_bytes=file_size,
                upload_timestamp=datetime.now(),
                metadata={}
            )
            
            result = agent.execute(file_ref)
            
            # Should fail - trailer is case-sensitive
            assert isinstance(result, ValidationError)
            assert result.error_code == "CORRUPTED_PDF"
        finally:
            os.unlink(file_path)
    
    def test_boundary_size_just_below_minimum_fails(self, agent):
        """Test that file with size just below 100KB (e.g., 100KB - 1 byte) fails"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            # Write PDF that's exactly 1 byte below minimum
            f.write(b'%PDF-1.4\n')
            target_size = ValidationAgent.MIN_FILE_SIZE - 1
            padding_size = target_size - len(b'%PDF-1.4\n') - len(b'%%EOF\n')
            f.write(b'x' * padding_size)
            f.write(b'%%EOF\n')
            file_path = f.name
        
        try:
            file_size = os.path.getsize(file_path)
            # Verify we're actually below minimum
            assert file_size < ValidationAgent.MIN_FILE_SIZE
            
            file_ref = FileReference(
                file_path=file_path,
                filename=os.path.basename(file_path),
                size_bytes=file_size,
                upload_timestamp=datetime.now(),
                metadata={}
            )
            
            result = agent.execute(file_ref)
            
            assert isinstance(result, ValidationError)
            assert result.error_code == "INVALID_FILE_SIZE"
            assert "below minimum" in result.reason
        finally:
            os.unlink(file_path)
    
    def test_boundary_size_just_above_maximum_fails(self, agent):
        """Test that file with size just above 50MB (e.g., 50MB + 1 byte) fails"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            # Write minimal PDF (we'll fake the size in FileReference)
            f.write(b'%PDF-1.4\n')
            f.write(b'%%EOF\n')
            file_path = f.name
        
        try:
            # Create FileReference with size just above maximum
            file_ref = FileReference(
                file_path=file_path,
                filename=os.path.basename(file_path),
                size_bytes=ValidationAgent.MAX_FILE_SIZE + 1,
                upload_timestamp=datetime.now(),
                metadata={}
            )
            
            result = agent.execute(file_ref)
            
            assert isinstance(result, ValidationError)
            assert result.error_code == "INVALID_FILE_SIZE"
            assert "exceeds maximum" in result.reason
        finally:
            os.unlink(file_path)


class TestValidationAgentProperties:
    """Property-based tests for ValidationAgent
    
    These tests use Hypothesis to verify universal correctness properties
    across all valid inputs.
    """
    
    @given(
        # Generate file size within valid bounds [100KB, 50MB]
        # Use smaller range for faster testing
        file_size=st.integers(
            min_value=ValidationAgent.MIN_FILE_SIZE,
            max_value=min(ValidationAgent.MIN_FILE_SIZE * 10, ValidationAgent.MAX_FILE_SIZE)
        ),
        # Generate PDF content padding (small size for performance)
        pdf_content=st.binary(min_size=0, max_size=100),
        # Generate filename
        filename=st.text(
            alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'),
                whitelist_characters='.-_'
            ),
            min_size=1,
            max_size=50
        ).filter(lambda x: x.strip() != '').map(lambda x: x + '.pdf')
    )
    @settings(deadline=None, max_examples=50)
    def test_property_valid_pdfs_pass_validation(self, file_size, pdf_content, filename):
        """**Validates: Requirements 3.1, 3.2, 3.3, 3.5**
        
        Property 1: Valid PDFs pass validation
        
        For any file with valid PDF structure (correct header, trailer, MIME type,
        size in bounds), validation should succeed and return the same FileReference.
        """
        # Create agent instance for this test
        agent = ValidationAgent()
        
        # Create a temporary file with valid PDF structure
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            # Write PDF header (magic bytes)
            f.write(b'%PDF-1.4\n')
            
            # Calculate how much padding we need to reach target file size
            header_size = len(b'%PDF-1.4\n')
            trailer_size = len(b'%%EOF\n')
            content_size = len(pdf_content)
            
            # Calculate padding needed to reach target file size
            current_size = header_size + content_size + trailer_size
            if current_size < file_size:
                padding_needed = file_size - current_size
                # Write the generated content
                f.write(pdf_content)
                # Add padding to reach target size
                f.write(b'x' * padding_needed)
            else:
                # If content is too large, truncate it
                max_content = file_size - header_size - trailer_size
                if max_content > 0:
                    f.write(pdf_content[:max_content])
            
            # Write PDF trailer (required for valid structure)
            f.write(b'%%EOF\n')
            
            file_path = f.name
        
        try:
            # Get actual file size
            actual_size = os.path.getsize(file_path)
            
            # Create FileReference with the file metadata
            file_ref = FileReference(
                file_path=file_path,
                filename=filename,
                size_bytes=actual_size,
                upload_timestamp=datetime.now(),
                metadata={"content_type": "application/pdf"}
            )
            
            # Execute validation
            result = agent.execute(file_ref)
            
            # Property: Valid PDFs should pass validation and return the same FileReference
            assert isinstance(result, FileReference), \
                f"Expected FileReference, got {type(result).__name__}"
            assert not isinstance(result, ValidationError), \
                f"Valid PDF should not return ValidationError: {result}"
            
            # Verify the returned FileReference matches the input
            assert result.file_path == file_ref.file_path, \
                "Returned FileReference should have same file_path"
            assert result.filename == file_ref.filename, \
                "Returned FileReference should have same filename"
            assert result.size_bytes == file_ref.size_bytes, \
                "Returned FileReference should have same size_bytes"
            
        finally:
            # Cleanup
            if os.path.exists(file_path):
                os.unlink(file_path)
