"""Source segment schema for document coverage tracking."""

from typing import List

from pydantic import BaseModel, Field, model_validator


class SourceSegment(BaseModel):
    """A deterministic, ordered slice of OCR source material."""

    segment_id: str = Field(..., description="Stable id, e.g. 'seg_001'")
    order: int = Field(..., ge=0, description="Zero-based reading order")
    page_start: int = Field(..., ge=1, description="First source page covered")
    page_end: int = Field(..., ge=1, description="Last source page covered")
    heading: str = Field(default="", description="Nearby heading, if detected")
    text: str = Field(default="", description="OCR text covered by this segment")
    math_latex: List[str] = Field(default_factory=list, description="LaTeX expressions in this segment")

    @model_validator(mode="after")
    def _validate_page_order(self) -> "SourceSegment":
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        return self
