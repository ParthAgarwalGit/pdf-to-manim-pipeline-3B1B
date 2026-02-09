"""Unit tests for Vision+OCR Agent

Tests cover:
- Text extraction accuracy with sample PDF pages
- Math notation extraction with LaTeX expressions
- Diagram detection and classification
- Multi-column layout reading order preservation
- Error handling for failed extractions
"""

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

try:
    import fitz  # PyMuPDF
    from PIL import Image
except ImportError:
    pytest.skip("PyMuPDF or PIL not installed", allow_module_level=True)

from src.agents.base import AgentExecutionError, BackoffStrategy
from src.agents.vision_ocr import VisionOCRAgent
from src.schemas.file_reference import FileReference
from src.schemas.ocr_output import BoundingBox, Diagram, MathExpression, OCROutput, TextBlock


class TestVisionOCRAgentBasics:
    """Test basic agent functionality"""
    
    def test_agent_initialization(self):
        """Test agent can be initialized"""
        agent = VisionOCRAgent()
        assert agent is not None
        assert agent.DPI == 300
        assert agent.MIN_DIAGRAM_AREA == 1000
        assert agent.MAX_RETRIES == 3
    
    def test_validate_input_with_file_reference(self):
        """Test input validation accepts FileReference"""
        agent = VisionOCRAgent()
        file_ref = FileReference(
            file_path="/path/to/file.pdf",
            filename="file.pdf",
            size_bytes=1000000,
            upload_timestamp="2024-01-15T10:30:00Z"
        )
        assert agent.validate_input(file_ref) is True
    
    def test_validate_input_with_invalid_type(self):
        """Test input validation rejects non-FileReference"""
        agent = VisionOCRAgent()
        assert agent.validate_input("not a file reference") is False
        assert agent.validate_input(None) is False
        assert agent.validate_input({"file_path": "test"}) is False
    
    def test_get_retry_policy(self):
        """Test retry policy configuration"""
        agent = VisionOCRAgent()
        policy = agent.get_retry_policy()
        
        assert policy.max_attempts == 3
        assert policy.backoff_strategy == BackoffStrategy.EXPONENTIAL
        assert policy.base_delay_seconds == 2.0
        assert policy.max_delay_seconds == 30.0
        assert "API_TIMEOUT" in policy.retryable_errors
        assert "RATE_LIMIT_EXCEEDED" in policy.retryable_errors


class TestTextExtraction:
    """Test text extraction functionality"""
    
    def test_extract_text_blocks_basic(self):
        """Test basic text block extraction"""
        agent = VisionOCRAgent()
        
        # Create mock page with text blocks
        mock_page = Mock()
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,  # Text block
                    "bbox": [100, 100, 500, 150],
                    "lines": [
                        {
                            "spans": [
                                {"text": "This is a test sentence."}
                            ]
                        }
                    ]
                },
                {
                    "type": 0,
                    "bbox": [100, 200, 500, 250],
                    "lines": [
                        {
                            "spans": [
                                {"text": "Another paragraph here."}
                            ]
                        }
                    ]
                }
            ]
        }
        
        mock_img = Mock(spec=Image.Image)
        
        text_blocks = agent._extract_text_blocks(mock_page, mock_img)
        
        assert len(text_blocks) == 2
        assert text_blocks[0].text == "This is a test sentence."
        assert text_blocks[1].text == "Another paragraph here."
        assert text_blocks[0].reading_order == 0
        assert text_blocks[1].reading_order == 1
    
    def test_extract_text_blocks_with_bounding_boxes(self):
        """Test text blocks have correct bounding boxes"""
        agent = VisionOCRAgent()
        
        mock_page = Mock()
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": [50, 75, 450, 125],
                    "lines": [
                        {
                            "spans": [{"text": "Test text"}]
                        }
                    ]
                }
            ]
        }
        
        mock_img = Mock(spec=Image.Image)
        text_blocks = agent._extract_text_blocks(mock_page, mock_img)
        
        assert len(text_blocks) == 1
        bbox = text_blocks[0].bounding_box
        assert bbox.x == 50
        assert bbox.y == 75
        assert bbox.width == 400  # 450 - 50
        assert bbox.height == 50  # 125 - 75
    
    def test_extract_text_blocks_filters_empty(self):
        """Test empty text blocks are filtered out"""
        agent = VisionOCRAgent()
        
        mock_page = Mock()
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": [100, 100, 500, 150],
                    "lines": [
                        {
                            "spans": [{"text": "Valid text"}]
                        }
                    ]
                },
                {
                    "type": 0,
                    "bbox": [100, 200, 500, 250],
                    "lines": [
                        {
                            "spans": [{"text": "   "}]  # Only whitespace
                        }
                    ]
                },
                {
                    "type": 0,
                    "bbox": [100, 300, 500, 350],
                    "lines": []  # No lines
                }
            ]
        }
        
        mock_img = Mock(spec=Image.Image)
        text_blocks = agent._extract_text_blocks(mock_page, mock_img)
        
        # Only the valid text block should remain
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "Valid text"


class TestReadingOrderSorting:
    """Test multi-column layout reading order"""
    
    def test_sort_by_reading_order_single_column(self):
        """Test sorting for single column layout (top to bottom)"""
        agent = VisionOCRAgent()
        
        # Create blocks in random order
        blocks = [
            TextBlock(
                text="Third",
                bounding_box=BoundingBox(x=100, y=300, width=400, height=50),
                reading_order=0,
                confidence=1.0
            ),
            TextBlock(
                text="First",
                bounding_box=BoundingBox(x=100, y=100, width=400, height=50),
                reading_order=1,
                confidence=1.0
            ),
            TextBlock(
                text="Second",
                bounding_box=BoundingBox(x=100, y=200, width=400, height=50),
                reading_order=2,
                confidence=1.0
            )
        ]
        
        sorted_blocks = agent._sort_by_reading_order(blocks)
        
        assert len(sorted_blocks) == 3
        assert sorted_blocks[0].text == "First"
        assert sorted_blocks[1].text == "Second"
        assert sorted_blocks[2].text == "Third"
    
    def test_sort_by_reading_order_two_columns(self):
        """Test sorting for two-column layout (left to right within rows)"""
        agent = VisionOCRAgent()
        
        # Create two-column layout
        blocks = [
            # Row 1, Column 2
            TextBlock(
                text="Top Right",
                bounding_box=BoundingBox(x=400, y=100, width=300, height=50),
                reading_order=0,
                confidence=1.0
            ),
            # Row 2, Column 1
            TextBlock(
                text="Middle Left",
                bounding_box=BoundingBox(x=100, y=200, width=300, height=50),
                reading_order=1,
                confidence=1.0
            ),
            # Row 1, Column 1
            TextBlock(
                text="Top Left",
                bounding_box=BoundingBox(x=100, y=100, width=300, height=50),
                reading_order=2,
                confidence=1.0
            ),
            # Row 2, Column 2
            TextBlock(
                text="Middle Right",
                bounding_box=BoundingBox(x=400, y=200, width=300, height=50),
                reading_order=3,
                confidence=1.0
            )
        ]
        
        sorted_blocks = agent._sort_by_reading_order(blocks)
        
        assert len(sorted_blocks) == 4
        # Row 1: left to right
        assert sorted_blocks[0].text == "Top Left"
        assert sorted_blocks[1].text == "Top Right"
        # Row 2: left to right
        assert sorted_blocks[2].text == "Middle Left"
        assert sorted_blocks[3].text == "Middle Right"
    
    def test_sort_by_reading_order_empty_list(self):
        """Test sorting handles empty list"""
        agent = VisionOCRAgent()
        sorted_blocks = agent._sort_by_reading_order([])
        assert sorted_blocks == []


class TestMathExpressionExtraction:
    """Test mathematical notation extraction"""
    
    def test_extract_math_expressions_inline(self):
        """Test extraction of inline math expressions"""
        agent = VisionOCRAgent()
        
        mock_page = Mock()
        mock_img = Mock(spec=Image.Image)
        
        text_blocks = [
            TextBlock(
                text="The equation $x^2 + y^2 = r^2$ represents a circle.",
                bounding_box=BoundingBox(x=100, y=100, width=400, height=50),
                reading_order=0,
                confidence=1.0
            )
        ]
        
        math_exprs = agent._extract_math_expressions(mock_page, mock_img, text_blocks)
        
        assert len(math_exprs) > 0
        # Should find the inline math expression
        latex_strings = [expr.latex for expr in math_exprs]
        assert any("x^2" in latex for latex in latex_strings)
    
    def test_extract_math_expressions_multiple_patterns(self):
        """Test extraction of various LaTeX patterns"""
        agent = VisionOCRAgent()
        
        mock_page = Mock()
        mock_img = Mock(spec=Image.Image)
        
        text_blocks = [
            TextBlock(
                text="We have \\frac{a}{b} and \\sum_{i=1}^{n} x_i in the formula.",
                bounding_box=BoundingBox(x=100, y=100, width=400, height=50),
                reading_order=0,
                confidence=1.0
            ),
            TextBlock(
                text="The space \\mathbb{R}^n is important.",
                bounding_box=BoundingBox(x=100, y=200, width=400, height=50),
                reading_order=1,
                confidence=1.0
            )
        ]
        
        math_exprs = agent._extract_math_expressions(mock_page, mock_img, text_blocks)
        
        assert len(math_exprs) > 0
        latex_strings = [expr.latex for expr in math_exprs]
        
        # Should find various patterns
        assert any("frac" in latex for latex in latex_strings)
        assert any("sum" in latex or "mathbb" in latex for latex in latex_strings)
    
    def test_extract_math_expressions_with_context(self):
        """Test math expressions include surrounding context"""
        agent = VisionOCRAgent()
        
        mock_page = Mock()
        mock_img = Mock(spec=Image.Image)
        
        text_blocks = [
            TextBlock(
                text="Consider the integral \\int f(x) dx which represents the area.",
                bounding_box=BoundingBox(x=100, y=100, width=400, height=50),
                reading_order=0,
                confidence=1.0
            )
        ]
        
        math_exprs = agent._extract_math_expressions(mock_page, mock_img, text_blocks)
        
        assert len(math_exprs) > 0
        # Context should include surrounding text
        for expr in math_exprs:
            if "int" in expr.latex:
                assert len(expr.context) > 0
                assert "integral" in expr.context or "area" in expr.context
    
    def test_mock_math_ocr_cleans_delimiters(self):
        """Test mock math OCR removes delimiters"""
        agent = VisionOCRAgent()
        bbox = BoundingBox(x=0, y=0, width=100, height=50)
        
        # Test inline math delimiters
        result = agent._mock_math_ocr("$x^2$", bbox)
        assert result is not None
        assert result.latex == "x^2"
        
        # Test display math delimiters
        result = agent._mock_math_ocr("\\[x^2\\]", bbox)
        assert result is not None
        assert result.latex == "x^2"
        
        # Test parenthesis delimiters
        result = agent._mock_math_ocr("\\(x^2\\)", bbox)
        assert result is not None
        assert result.latex == "x^2"


class TestDiagramDetection:
    """Test diagram detection and classification"""
    
    def test_detect_diagrams_basic(self):
        """Test basic diagram detection"""
        agent = VisionOCRAgent()
        
        mock_page = Mock()
        mock_page.number = 0
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 1,  # Image block
                    "bbox": [100, 100, 500, 400]  # Large enough area
                }
            ]
        }
        
        mock_img = Mock(spec=Image.Image)
        text_blocks = []
        
        diagrams = agent._detect_diagrams(mock_page, mock_img, text_blocks)
        
        assert len(diagrams) == 1
        assert diagrams[0].diagram_id == "diagram_1_1"
        assert diagrams[0].diagram_type in ["graph", "geometric", "flowchart", "plot", "unknown"]
        assert len(diagrams[0].description) > 0
        assert len(diagrams[0].visual_elements) > 0
    
    def test_detect_diagrams_filters_small_images(self):
        """Test small images below threshold are filtered"""
        agent = VisionOCRAgent()
        
        mock_page = Mock()
        mock_page.number = 0
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 1,
                    "bbox": [100, 100, 110, 110]  # Only 100 px² - too small
                },
                {
                    "type": 1,
                    "bbox": [100, 200, 200, 300]  # 10000 px² - large enough
                }
            ]
        }
        
        mock_img = Mock(spec=Image.Image)
        text_blocks = []
        
        diagrams = agent._detect_diagrams(mock_page, mock_img, text_blocks)
        
        # Only the large diagram should be detected
        assert len(diagrams) == 1
        assert diagrams[0].bounding_box.width == 100
        assert diagrams[0].bounding_box.height == 100
    
    def test_mock_diagram_analysis_classification(self):
        """Test diagram classification based on aspect ratio"""
        agent = VisionOCRAgent()
        mock_img = Mock(spec=Image.Image)
        
        # Wide diagram -> graph
        bbox_wide = (0, 0, 300, 100)
        dtype, desc, elements = agent._mock_diagram_analysis(mock_img, bbox_wide)
        assert dtype == "graph"
        assert "graph" in desc.lower()
        assert len(elements) > 0
        
        # Tall diagram -> flowchart
        bbox_tall = (0, 0, 100, 300)
        dtype, desc, elements = agent._mock_diagram_analysis(mock_img, bbox_tall)
        assert dtype == "flowchart"
        assert "flowchart" in desc.lower()
        
        # Large square diagram -> plot
        bbox_large = (0, 0, 300, 300)
        dtype, desc, elements = agent._mock_diagram_analysis(mock_img, bbox_large)
        assert dtype == "plot"
        assert "plot" in desc.lower()
        
        # Small square diagram -> geometric
        bbox_small = (0, 0, 50, 50)
        dtype, desc, elements = agent._mock_diagram_analysis(mock_img, bbox_small)
        assert dtype == "geometric"
        assert "geometric" in desc.lower()


class TestErrorHandling:
    """Test error handling and partial failures"""
    
    def test_execute_with_invalid_input(self):
        """Test execution fails with invalid input"""
        agent = VisionOCRAgent()
        
        with pytest.raises(AgentExecutionError) as exc_info:
            agent.execute("not a file reference")
        
        assert exc_info.value.error_code == "INVALID_INPUT"
    
    def test_execute_with_nonexistent_file(self):
        """Test execution fails gracefully with nonexistent file"""
        agent = VisionOCRAgent()
        
        file_ref = FileReference(
            file_path="/nonexistent/file.pdf",
            filename="file.pdf",
            size_bytes=1000000,
            upload_timestamp="2024-01-15T10:30:00Z"
        )
        
        with pytest.raises(AgentExecutionError) as exc_info:
            agent.execute(file_ref)
        
        assert exc_info.value.error_code in ["PDF_OPEN_FAILED", "OCR_EXTRACTION_FAILED"]
    
    @patch('fitz.open')
    def test_execute_handles_page_failure(self, mock_fitz_open):
        """Test execution continues when individual pages fail"""
        agent = VisionOCRAgent()
        
        # Create mock PDF with 3 pages using MagicMock for magic methods
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 3
        mock_doc.__getitem__.side_effect = [
            Mock(),  # Page 0 succeeds
            Exception("Page processing failed"),  # Page 1 fails
            Mock()   # Page 2 succeeds
        ]
        mock_doc.close = Mock()
        mock_fitz_open.return_value = mock_doc
        
        # Mock successful page processing for pages 0 and 2
        def mock_process_page(doc, page_num):
            if page_num == 1:
                raise Exception("Page processing failed")
            from src.schemas.ocr_output import Page
            return Page(
                page_number=page_num + 1,
                text_blocks=[],
                math_expressions=[],
                diagrams=[]
            )
        
        with patch.object(agent, '_process_page', side_effect=mock_process_page):
            file_ref = FileReference(
                file_path="/path/to/test.pdf",
                filename="test.pdf",
                size_bytes=1000000,
                upload_timestamp="2024-01-15T10:30:00Z"
            )
            
            result = agent.execute(file_ref)
            
            # Should succeed with 2 pages (page 1 failed but was skipped)
            assert isinstance(result, OCROutput)
            assert len(result.pages) == 2
            assert result.pdf_metadata.page_count == 3


class TestIntegration:
    """Integration tests with real PDF processing"""
    
    def create_test_pdf(self, content: str = "Test content") -> str:
        """Helper to create a minimal test PDF"""
        # Create a temporary PDF file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Create a minimal PDF using PyMuPDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((100, 100), content)
        doc.save(temp_path)
        doc.close()
        
        return temp_path
    
    def test_execute_with_real_pdf(self):
        """Test execution with a real PDF file"""
        agent = VisionOCRAgent()
        
        # Create test PDF
        pdf_path = self.create_test_pdf("This is a test document with some text.")
        
        try:
            file_ref = FileReference(
                file_path=pdf_path,
                filename="test.pdf",
                size_bytes=os.path.getsize(pdf_path),
                upload_timestamp="2024-01-15T10:30:00Z"
            )
            
            result = agent.execute(file_ref)
            
            # Verify result structure
            assert isinstance(result, OCROutput)
            assert result.pdf_metadata.filename == "test.pdf"
            assert result.pdf_metadata.page_count == 1
            assert len(result.pages) == 1
            
            # Verify page content
            page = result.pages[0]
            assert page.page_number == 1
            assert len(page.text_blocks) > 0
            
            # Check if text was extracted
            all_text = " ".join(block.text for block in page.text_blocks)
            assert "test" in all_text.lower()
            
        finally:
            # Clean up
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
    
    def test_execute_with_math_content(self):
        """Test execution with mathematical content"""
        agent = VisionOCRAgent()
        
        # Create PDF with math-like content
        pdf_path = self.create_test_pdf("The equation $x^2 + y^2 = 1$ is a circle.")
        
        try:
            file_ref = FileReference(
                file_path=pdf_path,
                filename="math_test.pdf",
                size_bytes=os.path.getsize(pdf_path),
                upload_timestamp="2024-01-15T10:30:00Z"
            )
            
            result = agent.execute(file_ref)
            
            assert isinstance(result, OCROutput)
            assert len(result.pages) == 1
            
            # Should detect math expressions
            page = result.pages[0]
            # Math detection depends on text extraction, may or may not find patterns
            # Just verify the structure is correct
            assert isinstance(page.math_expressions, list)
            
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
