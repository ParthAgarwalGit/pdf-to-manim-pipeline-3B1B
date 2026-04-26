"""Reusable OCROutput fixtures for agent tests.

Keeps test setup compact and consistent across test files.
"""

from src.schemas.ocr_output import (
    BoundingBox,
    MathExpression,
    OCROutput,
    Page,
    PDFMetadata,
    TextBlock,
)


def minimal_trig_ocr_output() -> OCROutput:
    """A small OCROutput covering basic trigonometry — sine, cosine, unit circle.

    Designed so the script-LLM has enough material to introduce ≥ 2 distinct
    concepts but stays small enough for fast tests.
    """
    return OCROutput(
        pdf_metadata=PDFMetadata(
            filename="trig.pdf", page_count=1, file_size_bytes=10_000,
        ),
        pages=[
            Page(
                page_number=1,
                text_blocks=[
                    TextBlock(
                        text=(
                            "Definition: Given a right triangle with hypotenuse h, "
                            "opposite side o, and adjacent side a, the sine of an "
                            "angle theta is sin(theta) = o / h. "
                            "The cosine is cos(theta) = a / h. "
                            "On the unit circle, the point at angle theta has "
                            "coordinates (cos(theta), sin(theta))."
                        ),
                        bounding_box=BoundingBox(x=0, y=0, width=600, height=200),
                        reading_order=0,
                        confidence=0.97,
                    ),
                ],
                math_expressions=[
                    MathExpression(
                        latex=r"\sin(\theta) = \frac{o}{h}",
                        bounding_box=BoundingBox(x=0, y=210, width=200, height=40),
                        confidence=0.95,
                        extraction_failed=False,
                    ),
                    MathExpression(
                        latex=r"\cos(\theta) = \frac{a}{h}",
                        bounding_box=BoundingBox(x=0, y=260, width=200, height=40),
                        confidence=0.95,
                        extraction_failed=False,
                    ),
                ],
                diagrams=[],
            ),
        ],
    )
