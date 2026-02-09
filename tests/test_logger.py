"""Unit tests for structured JSON logger.

Tests verify that the logger correctly writes structured JSON log entries
to pipeline.log in the output directory, following the required format.

Requirements tested:
- 12.1: Log agent_start events with agent_name, timestamp, input_summary
- 12.2: Log agent_complete events with agent_name, timestamp, duration_ms, output_summary, status
- 12.3: Log agent_failure events with agent_name, timestamp, error_message, error_code, input_context, retry_attempt
- 12.4: Write all logs to pipeline.log in output directory using structured JSON format
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.orchestrator.logger import StructuredJSONLogger


class TestStructuredJSONLogger:
    """Test suite for StructuredJSONLogger."""
    
    def test_logger_creates_log_file(self):
        """Test that logger creates pipeline.log in output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredJSONLogger(output_directory=tmpdir)
            
            # Verify log file was created
            log_file = Path(tmpdir) / "pipeline.log"
            assert log_file.exists()
            
            logger.close()
    
    def test_log_agent_start_format(self):
        """Test agent_start log entry format.
        
        Requirement 12.1: Log agent_start events with agent_name, timestamp, input_summary
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredJSONLogger(output_directory=tmpdir)
            
            # Log agent start
            logger.log_agent_start(
                agent_name="TestAgent",
                input_summary="Test input summary",
                retry_attempt=0
            )
            
            logger.close()
            
            # Read log file
            log_file = Path(tmpdir) / "pipeline.log"
            with open(log_file, 'r') as f:
                log_line = f.readline()
            
            # Parse JSON
            log_entry = json.loads(log_line)
            
            # Verify required fields
            assert log_entry["event"] == "agent_start"
            assert log_entry["agent_name"] == "TestAgent"
            assert "timestamp" in log_entry
            assert log_entry["input_summary"] == "Test input summary"
            assert log_entry["retry_attempt"] == 0
    
    def test_log_agent_complete_format(self):
        """Test agent_complete log entry format.
        
        Requirement 12.2: Log agent_complete events with agent_name, timestamp,
        duration_ms, output_summary, status
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredJSONLogger(output_directory=tmpdir)
            
            # Log agent completion
            logger.log_agent_complete(
                agent_name="TestAgent",
                duration_ms=123.45,
                output_summary="Test output summary",
                status="SUCCESS",
                retry_attempt=0
            )
            
            logger.close()
            
            # Read log file
            log_file = Path(tmpdir) / "pipeline.log"
            with open(log_file, 'r') as f:
                log_line = f.readline()
            
            # Parse JSON
            log_entry = json.loads(log_line)
            
            # Verify required fields
            assert log_entry["event"] == "agent_complete"
            assert log_entry["agent_name"] == "TestAgent"
            assert "timestamp" in log_entry
            assert log_entry["duration_ms"] == 123.45
            assert log_entry["output_summary"] == "Test output summary"
            assert log_entry["status"] == "SUCCESS"
            assert log_entry["retry_attempt"] == 0
    
    def test_log_agent_failure_format(self):
        """Test agent_failure log entry format.
        
        Requirement 12.3: Log agent_failure events with agent_name, timestamp,
        error_message, error_code, input_context, retry_attempt
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredJSONLogger(output_directory=tmpdir)
            
            # Log agent failure
            logger.log_agent_failure(
                agent_name="TestAgent",
                error_message="Test error message",
                error_code="TEST_ERROR",
                input_context="Test input context",
                retry_attempt=1,
                duration_ms=50.0
            )
            
            logger.close()
            
            # Read log file
            log_file = Path(tmpdir) / "pipeline.log"
            with open(log_file, 'r') as f:
                log_line = f.readline()
            
            # Parse JSON
            log_entry = json.loads(log_line)
            
            # Verify required fields
            assert log_entry["event"] == "agent_failure"
            assert log_entry["agent_name"] == "TestAgent"
            assert "timestamp" in log_entry
            assert log_entry["error_message"] == "Test error message"
            assert log_entry["error_code"] == "TEST_ERROR"
            assert log_entry["input_context"] == "Test input context"
            assert log_entry["retry_attempt"] == 1
            assert log_entry["duration_ms"] == 50.0
    
    def test_multiple_log_entries(self):
        """Test that multiple log entries are written correctly.
        
        Requirement 12.4: Write all logs to pipeline.log using structured JSON format
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredJSONLogger(output_directory=tmpdir)
            
            # Log multiple events
            logger.log_agent_start("Agent1", "Input 1")
            logger.log_agent_complete("Agent1", 100.0, "Output 1")
            logger.log_agent_start("Agent2", "Input 2")
            logger.log_agent_failure("Agent2", "Error", "ERROR_CODE", "Context")
            
            logger.close()
            
            # Read all log lines
            log_file = Path(tmpdir) / "pipeline.log"
            with open(log_file, 'r') as f:
                log_lines = f.readlines()
            
            # Verify we have 4 log entries
            assert len(log_lines) == 4
            
            # Verify each line is valid JSON
            for line in log_lines:
                log_entry = json.loads(line)
                assert "event" in log_entry
                assert "timestamp" in log_entry
    
    def test_log_pipeline_start(self):
        """Test pipeline_start log entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredJSONLogger(output_directory=tmpdir)
            
            logger.log_pipeline_start(
                pdf_file_path="/path/to/test.pdf",
                config={"verbosity": "medium", "depth": "intermediate"}
            )
            
            logger.close()
            
            # Read log file
            log_file = Path(tmpdir) / "pipeline.log"
            with open(log_file, 'r') as f:
                log_line = f.readline()
            
            log_entry = json.loads(log_line)
            
            assert log_entry["event"] == "pipeline_start"
            assert log_entry["pdf_file_path"] == "/path/to/test.pdf"
            assert log_entry["config"]["verbosity"] == "medium"
    
    def test_log_pipeline_complete(self):
        """Test pipeline_complete log entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredJSONLogger(output_directory=tmpdir)
            
            logger.log_pipeline_complete(
                duration_seconds=45.67,
                output_directory="/output/test",
                status="SUCCESS",
                scene_success_rate=0.95
            )
            
            logger.close()
            
            # Read log file
            log_file = Path(tmpdir) / "pipeline.log"
            with open(log_file, 'r') as f:
                log_line = f.readline()
            
            log_entry = json.loads(log_line)
            
            assert log_entry["event"] == "pipeline_complete"
            assert log_entry["duration_seconds"] == 45.67
            assert log_entry["output_directory"] == "/output/test"
            assert log_entry["status"] == "SUCCESS"
            assert log_entry["scene_success_rate"] == 0.95
    
    def test_log_pipeline_error(self):
        """Test pipeline_error log entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredJSONLogger(output_directory=tmpdir)
            
            logger.log_pipeline_error(
                error_type="ValidationError",
                error_message="Invalid PDF",
                stage="Validation"
            )
            
            logger.close()
            
            # Read log file
            log_file = Path(tmpdir) / "pipeline.log"
            with open(log_file, 'r') as f:
                log_line = f.readline()
            
            log_entry = json.loads(log_line)
            
            assert log_entry["event"] == "pipeline_error"
            assert log_entry["error_type"] == "ValidationError"
            assert log_entry["error_message"] == "Invalid PDF"
            assert log_entry["stage"] == "Validation"
    
    def test_context_manager(self):
        """Test that logger works as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with StructuredJSONLogger(output_directory=tmpdir) as logger:
                logger.log_agent_start("TestAgent", "Input")
            
            # Verify log file was written and closed
            log_file = Path(tmpdir) / "pipeline.log"
            assert log_file.exists()
            
            # Should be able to read the file (not locked)
            with open(log_file, 'r') as f:
                log_line = f.readline()
                log_entry = json.loads(log_line)
                assert log_entry["event"] == "agent_start"
    
    def test_logger_without_output_directory(self):
        """Test that logger works without output directory (console only)."""
        # Should not raise an error
        logger = StructuredJSONLogger(output_directory=None)
        
        # Should be able to log (to console only)
        logger.log_agent_start("TestAgent", "Input")
        logger.log_agent_complete("TestAgent", 100.0, "Output")
        
        logger.close()
    
    def test_json_format_one_line_per_entry(self):
        """Test that each log entry is on a single line.
        
        Requirement 12.4: Write logs using structured JSON format (one JSON object per line)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredJSONLogger(output_directory=tmpdir)
            
            # Log an entry with nested data
            logger.log_pipeline_start(
                pdf_file_path="/path/to/test.pdf",
                config={
                    "verbosity": "high",
                    "depth": "advanced",
                    "nested": {"key": "value"}
                }
            )
            
            logger.close()
            
            # Read log file
            log_file = Path(tmpdir) / "pipeline.log"
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Should have exactly one newline at the end
            lines = content.strip().split('\n')
            assert len(lines) == 1
            
            # Should be valid JSON
            log_entry = json.loads(lines[0])
            assert log_entry["event"] == "pipeline_start"
    
    def test_timestamp_format(self):
        """Test that timestamps are in ISO8601 format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredJSONLogger(output_directory=tmpdir)
            
            logger.log_agent_start("TestAgent", "Input")
            
            logger.close()
            
            # Read log file
            log_file = Path(tmpdir) / "pipeline.log"
            with open(log_file, 'r') as f:
                log_line = f.readline()
            
            log_entry = json.loads(log_line)
            
            # Verify timestamp is ISO8601 format (contains 'T' and timezone)
            timestamp = log_entry["timestamp"]
            assert 'T' in timestamp
            assert '+' in timestamp or 'Z' in timestamp or timestamp.endswith('00:00')
    
    def test_retry_attempt_tracking(self):
        """Test that retry attempts are correctly tracked in logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredJSONLogger(output_directory=tmpdir)
            
            # Log multiple retry attempts
            logger.log_agent_start("TestAgent", "Input", retry_attempt=0)
            logger.log_agent_failure("TestAgent", "Error", "ERROR", "Context", retry_attempt=0)
            logger.log_agent_start("TestAgent", "Input", retry_attempt=1)
            logger.log_agent_failure("TestAgent", "Error", "ERROR", "Context", retry_attempt=1)
            logger.log_agent_start("TestAgent", "Input", retry_attempt=2)
            logger.log_agent_complete("TestAgent", 100.0, "Output", retry_attempt=2)
            
            logger.close()
            
            # Read all log lines
            log_file = Path(tmpdir) / "pipeline.log"
            with open(log_file, 'r') as f:
                log_lines = f.readlines()
            
            # Verify retry attempts are tracked
            assert len(log_lines) == 6
            
            log_entries = [json.loads(line) for line in log_lines]
            
            # First attempt
            assert log_entries[0]["retry_attempt"] == 0
            assert log_entries[1]["retry_attempt"] == 0
            
            # Second attempt
            assert log_entries[2]["retry_attempt"] == 1
            assert log_entries[3]["retry_attempt"] == 1
            
            # Third attempt (success)
            assert log_entries[4]["retry_attempt"] == 2
            assert log_entries[5]["retry_attempt"] == 2
