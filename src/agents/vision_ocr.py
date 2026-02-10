"""Vision + OCR Agent for PDF content extraction

This module implements the Vision+OCR Agent which extracts text, mathematical notation,
and diagrams from PDF files using OCR and multimodal vision models.
"""

import io
import logging
import re
import time
import os
import base64
from typing import List, Optional, Tuple, Any
from io import BytesIO

# Try importing dependencies, handle if missing (for initial setup)
try:
    from PIL import Image
    import fitz  # PyMuPDF
    import openai
except ImportError:
    pass

from src.agents.base import (
    Agent,
    AgentExecutionError,
    AgentInput,
    AgentOutput,
    BackoffStrategy,
    RetryPolicy,
)
from src.schemas.file_reference import FileReference
from src.schemas.ocr_output import (
    BoundingBox,
    Diagram,
    MathExpression,
    OCROutput,
    Page,
    PDFMetadata,
    TextBlock,
)

logger = logging.getLogger(__name__)

class VisionOCRAgent(Agent):
    """Agent responsible for extracting text, math notation, and diagrams from PDFs"""
    
    # Configuration constants
    DPI = 300  # Resolution for PDF rendering
    MIN_DIAGRAM_AREA = 1000  # Minimum area in pixels² for diagram detection
    MAX_RETRIES = 3  # Maximum retry attempts for API calls
    
    def __init__(self):
        super().__init__()
        # Initialize OpenAI Client safely
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = None
            logger.warning("OPENAI_API_KEY not found. Vision features will return mocks.")
    
    def execute(self, input_data: AgentInput) -> AgentOutput:
        """Execute OCR extraction on the PDF file"""
        if not self.validate_input(input_data):
            raise AgentExecutionError(
                error_code="INVALID_INPUT",
                message="Input must be a FileReference object",
                context={"input_type": type(input_data).__name__}
            )
        
        file_ref: FileReference = input_data
        file_path = file_ref.file_path
        
        logger.info(f"Starting OCR extraction for: {file_path}")
        
        try:
            # Open PDF document
            pdf_document = fitz.open(file_path)
            page_count = len(pdf_document)
            
            # Create PDF metadata
            pdf_metadata = PDFMetadata(
                filename=file_ref.filename,
                page_count=page_count,
                file_size_bytes=file_ref.size_bytes
            )
            
            # Process each page
            pages: List[Page] = []
            for page_num in range(page_count):
                logger.info(f"Processing page {page_num + 1}/{page_count}")
                
                try:
                    page_data = self._process_page(pdf_document, page_num)
                    pages.append(page_data)
                except Exception as e:
                    logger.error(f"Failed to process page {page_num + 1}: {str(e)}")
                    # Continue with remaining pages (partial failure handling)
                    continue
            
            pdf_document.close()
            
            if not pages:
                raise AgentExecutionError(
                    error_code="ALL_PAGES_FAILED",
                    message="Failed to process any pages from PDF",
                    context={"file_path": file_path}
                )
            
            # Assemble OCR output
            ocr_output = OCROutput(
                pdf_metadata=pdf_metadata,
                pages=pages
            )
            
            logger.info(f"OCR extraction completed: {len(pages)}/{page_count} pages processed")
            return ocr_output
            
        except fitz.FileDataError as e:
            raise AgentExecutionError(
                error_code="PDF_OPEN_FAILED",
                message=f"Failed to open PDF file: {str(e)}",
                context={"file_path": file_path}
            )
        except Exception as e:
            raise AgentExecutionError(
                error_code="OCR_EXTRACTION_FAILED",
                message=f"Unexpected error during OCR extraction: {str(e)}",
                context={"file_path": file_path}
            )
    
    def _process_page(self, pdf_document: 'fitz.Document', page_num: int) -> Page:
        """Process a single PDF page"""
        page = pdf_document[page_num]
        
        # Render page to image at high resolution
        pix = page.get_pixmap(matrix=fitz.Matrix(self.DPI / 72, self.DPI / 72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Extract text blocks with bounding boxes
        text_blocks = self._extract_text_blocks(page, img)
        
        # Extract mathematical expressions
        math_expressions = self._extract_math_expressions(page, img, text_blocks)
        
        # Detect and describe diagrams
        diagrams = self._detect_diagrams(page, img, text_blocks)
        
        return Page(
            page_number=page_num + 1,  # 1-indexed for output
            text_blocks=text_blocks,
            math_expressions=math_expressions,
            diagrams=diagrams
        )
    
    def _extract_text_blocks(self, page: 'fitz.Page', img: Image.Image) -> List[TextBlock]:
        """Extract text blocks with bounding boxes and reading order"""
        text_blocks: List[TextBlock] = []
        
        # Extract text with bounding boxes using PyMuPDF
        blocks = page.get_text("dict")["blocks"]
        
        for block_idx, block in enumerate(blocks):
            if block.get("type") == 0:  # Text block
                # Get bounding box
                bbox = block["bbox"]  # (x0, y0, x1, y1)
                
                # Extract text from lines
                lines = block.get("lines", [])
                text_content = []
                for line in lines:
                    spans = line.get("spans", [])
                    line_text = " ".join(span.get("text", "") for span in spans)
                    text_content.append(line_text)
                
                text = " ".join(text_content).strip()
                
                if text:  # Only add non-empty text blocks
                    text_blocks.append(TextBlock(
                        text=text,
                        bounding_box=BoundingBox(
                            x=bbox[0],
                            y=bbox[1],
                            width=bbox[2] - bbox[0],
                            height=bbox[3] - bbox[1]
                        ),
                        reading_order=block_idx,
                        confidence=0.95  # Mock confidence score
                    ))
        
        # Sort by reading order (spatial sorting for multi-column layouts)
        text_blocks = self._sort_by_reading_order(text_blocks)
        
        # Update reading_order indices after sorting
        for idx, block in enumerate(text_blocks):
            block.reading_order = idx
        
        return text_blocks
    
    def _sort_by_reading_order(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """Sort text blocks by reading order for multi-column layouts"""
        if not text_blocks:
            return text_blocks
        
        # Group blocks into rows based on vertical overlap
        rows: List[List[TextBlock]] = []
        
        for block in text_blocks:
            placed = False
            block_y = block.bounding_box.y
            block_height = block.bounding_box.height
            
            # Try to place in existing row
            for row in rows:
                # Check if block overlaps vertically with row
                row_y = min(b.bounding_box.y for b in row)
                row_height = max(b.bounding_box.y + b.bounding_box.height for b in row) - row_y
                
                # Blocks are in same row if they overlap vertically by at least 50%
                overlap = min(block_y + block_height, row_y + row_height) - max(block_y, row_y)
                if overlap > 0.5 * min(block_height, row_height):
                    row.append(block)
                    placed = True
                    break
            
            if not placed:
                rows.append([block])
        
        # Sort rows by vertical position (top to bottom)
        rows.sort(key=lambda row: min(b.bounding_box.y for b in row))
        
        # Within each row, sort blocks by horizontal position (left to right)
        for row in rows:
            row.sort(key=lambda b: b.bounding_box.x)
        
        # Flatten rows into single list
        sorted_blocks = [block for row in rows for block in row]
        
        return sorted_blocks
    
    def _extract_math_expressions(
        self,
        page: 'fitz.Page',
        img: Image.Image,
        text_blocks: List[TextBlock]
    ) -> List[MathExpression]:
        """Extract mathematical notation in LaTeX format"""
        math_expressions: List[MathExpression] = []
        
        for block in text_blocks:
            # Look for LaTeX-like patterns
            latex_patterns = [
                r'\$[^\$]+\$',  # Inline math: $...$
                r'\\\([^\)]+\\\)',  # Inline math: \(...\)
                r'\\\[[^\]]+\\\]',  # Display math: \[...\]
                r'\\begin\{equation\}.*?\\end\{equation\}',  # Equation environment
                r'\\frac\{[^\}]+\}\{[^\}]+\}',  # Fractions
                r'\\sum|\\int|\\prod|\\lim',  # Math operators
                r'\\mathbb\{[A-Z]\}',  # Blackboard bold
                r'[a-zA-Z]\^[0-9\{]',  # Superscripts
                r'[a-zA-Z]_[0-9\{]',  # Subscripts
            ]
            
            text = block.text
            for pattern in latex_patterns:
                matches = re.finditer(pattern, text, re.DOTALL)
                for match in matches:
                    latex_str = match.group(0)
                    
                    # Get context (surrounding text)
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    try:
                        math_expr = self._mock_math_ocr(latex_str, block.bounding_box)
                        if math_expr:
                            math_expr.context = context
                            math_expressions.append(math_expr)
                    except Exception as e:
                        logger.warning(f"Math extraction failed for pattern: {latex_str[:50]}")
                        # Mark as failed but continue
                        math_expressions.append(MathExpression(
                            latex=latex_str,
                            bounding_box=block.bounding_box,
                            context=context,
                            confidence=0.0,
                            extraction_failed=True
                        ))
        
        return math_expressions
    
    def _mock_math_ocr(self, text: str, bbox: BoundingBox) -> Optional[MathExpression]:
        """Mock math OCR implementation"""
        # Clean up the LaTeX string
        latex = text.strip()
        
        # Remove inline math delimiters
        latex = re.sub(r'^\$|\$$', '', latex)
        latex = re.sub(r'^\\\(|\\\)$', '', latex)
        latex = re.sub(r'^\\\[|\\\]$', '', latex)
        
        if not latex:
            return None
        
        return MathExpression(
            latex=latex,
            bounding_box=bbox,
            context="",
            confidence=0.85,
            extraction_failed=False
        )
    
    def _detect_diagrams(
        self,
        page: 'fitz.Page',
        img: Image.Image,
        text_blocks: List[TextBlock]
    ) -> List[Diagram]:
        """Detect and describe diagrams in the page"""
        diagrams: List[Diagram] = []
        
        blocks = page.get_text("dict")["blocks"]
        diagram_count = 0
        
        for block in blocks:
            if block.get("type") == 1:  # Image block
                bbox = block["bbox"]  # (x0, y0, x1, y1)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                
                # Only process diagrams larger than minimum area
                if area >= self.MIN_DIAGRAM_AREA:
                    diagram_count += 1
                    diagram_id = f"diagram_{page.number + 1}_{diagram_count}"
                    
                    # Use multimodal vision model (GPT-4o)
                    diagram_type, description, visual_elements = self._analyze_diagram_vision(
                        img, bbox
                    )
                    
                    diagrams.append(Diagram(
                        diagram_id=diagram_id,
                        bounding_box=BoundingBox(
                            x=bbox[0],
                            y=bbox[1],
                            width=width,
                            height=height
                        ),
                        diagram_type=diagram_type,
                        description=description,
                        visual_elements=visual_elements,
                        confidence=0.80
                    ))
        
        return diagrams

    def _analyze_diagram_vision(
        self, 
        img: Image.Image, 
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[str, str, List[str]]:
        """
        Processes a specific cropped area of the page image using GPT-4o Vision.
        """
        # Safety fallback
        if not self.client:
            return "unknown", "Visual analysis unavailable (No API Key)", []

        # 1. Crop the image to the diagram's bounding box (x0, y0, x1, y1)
        cropped_img = img.crop(bbox)
        
        # 2. Convert PIL Image to Base64 string for OpenAI
        buffered = BytesIO()
        cropped_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 3. Call the Vision Model
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Identify the type of this mathematical diagram (graph, flowchart, geometric, or plot) and provide a detailed description for a Manim animator."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                    ]
                }],
                max_tokens=300
            )
            
            full_description = response.choices[0].message.content
            
            # 4. Parse the response
            diagram_type = "geometric" # Default
            if "graph" in full_description.lower(): diagram_type = "graph"
            elif "flow" in full_description.lower(): diagram_type = "flowchart"
            elif "plot" in full_description.lower(): diagram_type = "plot"
            
            return diagram_type, full_description, ["extracted_via_vision"]

        except Exception as e:
            logger.error(f"Vision API call failed: {e}")
            return "unknown", f"Failed to extract description: {e}", []

    def validate_input(self, input_data: AgentInput) -> bool:
        """Validate input is a FileReference object"""
        return isinstance(input_data, FileReference)
    
    def get_retry_policy(self) -> RetryPolicy:
        """Return retry policy for Vision+OCR Agent"""
        return RetryPolicy(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay_seconds=2.0,
            max_delay_seconds=30.0,
            retryable_errors=[
                "API_TIMEOUT",
                "RATE_LIMIT_EXCEEDED",
                "SERVICE_UNAVAILABLE",
                "NETWORK_ERROR"
            ]
        )
