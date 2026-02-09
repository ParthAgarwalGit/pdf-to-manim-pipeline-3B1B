"""OCR output schema for Vision+OCR Agent."""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BoundingBox(BaseModel):
    """Bounding box coordinates for text blocks, math expressions, and diagrams."""
    
    x: float = Field(..., description="X coordinate of top-left corner")
    y: float = Field(..., description="Y coordinate of top-left corner")
    width: float = Field(..., ge=0, description="Width of bounding box")
    height: float = Field(..., ge=0, description="Height of bounding box")


class TextBlock(BaseModel):
    """Text block extracted from a PDF page."""
    
    text: str = Field(..., description="Extracted text content")
    bounding_box: BoundingBox = Field(..., description="Position of text block on page")
    reading_order: int = Field(..., ge=0, description="Sequential order for reading")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="OCR confidence score (0-1)"
    )


class MathExpression(BaseModel):
    """Mathematical expression extracted from a PDF page."""
    
    latex: str = Field(..., description="LaTeX representation of the math expression")
    bounding_box: BoundingBox = Field(..., description="Position of math expression on page")
    context: str = Field(
        default="",
        description="Surrounding text providing context for the expression"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score (0-1)"
    )
    extraction_failed: bool = Field(
        default=False,
        description="Flag indicating if extraction failed for this expression"
    )


class Diagram(BaseModel):
    """Diagram extracted from a PDF page."""
    
    diagram_id: str = Field(..., description="Unique identifier for the diagram")
    bounding_box: BoundingBox = Field(..., description="Position of diagram on page")
    diagram_type: str = Field(
        ...,
        description="Type of diagram (graph|geometric|flowchart|plot|unknown)"
    )
    description: str = Field(..., description="Textual description of the diagram")
    visual_elements: List[str] = Field(
        default_factory=list,
        description="Array of detected visual elements in the diagram"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Detection confidence score (0-1)"
    )
    
    @field_validator('diagram_type')
    @classmethod
    def validate_diagram_type(cls, v: str) -> str:
        """Validate diagram type is one of the allowed values."""
        allowed_types = {'graph', 'geometric', 'flowchart', 'plot', 'unknown'}
        if v not in allowed_types:
            raise ValueError(
                f"diagram_type must be one of {allowed_types}, got '{v}'"
            )
        return v


class Page(BaseModel):
    """Single page of OCR output."""
    
    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    text_blocks: List[TextBlock] = Field(
        default_factory=list,
        description="Text blocks extracted from the page"
    )
    math_expressions: List[MathExpression] = Field(
        default_factory=list,
        description="Mathematical expressions extracted from the page"
    )
    diagrams: List[Diagram] = Field(
        default_factory=list,
        description="Diagrams extracted from the page"
    )


class PDFMetadata(BaseModel):
    """Metadata about the source PDF file."""
    
    filename: str = Field(..., description="Name of the PDF file")
    page_count: int = Field(..., ge=1, description="Total number of pages in the PDF")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")


class OCROutput(BaseModel):
    """
    Complete OCR output from Vision+OCR Agent.
    
    Contains extracted text, mathematical notation, and diagrams from all pages.
    """
    
    pdf_metadata: PDFMetadata = Field(..., description="Metadata about the source PDF")
    pages: List[Page] = Field(..., description="OCR output for each page")
    
    @field_validator('pages')
    @classmethod
    def validate_pages_count(cls, v: List[Page], info) -> List[Page]:
        """Ensure pages list is not empty."""
        if not v:
            raise ValueError("pages list cannot be empty")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pdf_metadata": {
                    "filename": "linear_algebra.pdf",
                    "page_count": 5,
                    "file_size_bytes": 2097152
                },
                "pages": [
                    {
                        "page_number": 1,
                        "text_blocks": [
                            {
                                "text": "Introduction to Vector Spaces",
                                "bounding_box": {"x": 100, "y": 50, "width": 400, "height": 30},
                                "reading_order": 0,
                                "confidence": 0.98
                            }
                        ],
                        "math_expressions": [
                            {
                                "latex": "\\mathbb{R}^n",
                                "bounding_box": {"x": 150, "y": 200, "width": 50, "height": 20},
                                "context": "vector space over the real numbers",
                                "confidence": 0.95
                            }
                        ],
                        "diagrams": []
                    }
                ]
            }
        }
    )
