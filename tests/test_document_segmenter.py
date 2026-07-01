"""Tests for deterministic OCR -> SourceSegment partitioning."""

from src.agents.document_segmenter import (
    DocumentSegmenterAgent,
    DocumentSegmenterInput,
)
from src.schemas.ocr_output import (
    BoundingBox,
    MathExpression,
    OCROutput,
    Page,
    PDFMetadata,
    TextBlock,
)


def _block(text: str, order: int) -> TextBlock:
    return TextBlock(
        text=text,
        bounding_box=BoundingBox(x=0, y=order * 10, width=100, height=10),
        reading_order=order,
        confidence=0.95,
    )


def test_segmenter_covers_pages_in_order_without_dropping_text():
    ocr = OCROutput(
        pdf_metadata=PDFMetadata(filename="paper.pdf", page_count=3, file_size_bytes=1000),
        pages=[
            Page(
                page_number=1,
                text_blocks=[
                    _block("Section 1 Introduction to vectors.", 0),
                    _block("Vectors can be added and scaled.", 1),
                ],
                math_expressions=[],
                diagrams=[],
            ),
            Page(
                page_number=2,
                text_blocks=[
                    _block("Section 2 Bases and coordinates.", 0),
                    _block("A basis expresses every vector uniquely.", 1),
                ],
                math_expressions=[
                    MathExpression(
                        latex=r"v = \sum_i c_i b_i",
                        bounding_box=BoundingBox(x=0, y=25, width=100, height=10),
                    )
                ],
                diagrams=[],
            ),
            Page(
                page_number=3,
                text_blocks=[
                    _block("Section 3 Linear maps.", 0),
                    _block("Matrices represent linear maps after choosing bases.", 1),
                ],
                math_expressions=[],
                diagrams=[],
            ),
        ],
    )

    result = DocumentSegmenterAgent().execute(
        DocumentSegmenterInput(ocr_output=ocr, target_words_per_segment=6)
    )

    segments = result.segments
    assert [segment.segment_id for segment in segments] == [
        f"seg_{index:03d}" for index in range(1, len(segments) + 1)
    ]
    assert [segment.order for segment in segments] == list(range(len(segments)))
    assert segments[0].page_start == 1
    assert segments[-1].page_end == 3

    source_words = " ".join(
        block.text for page in ocr.pages for block in sorted(page.text_blocks, key=lambda b: b.reading_order)
    ).split()
    segmented_words = " ".join(segment.text for segment in segments).split()
    assert segmented_words == source_words
    assert any(r"v = \sum_i c_i b_i" in segment.math_latex for segment in segments)


def test_segmenter_splits_oversized_blocks_and_merges_tiny_tail():
    long_text = " ".join(f"word{i}" for i in range(25))
    ocr = OCROutput(
        pdf_metadata=PDFMetadata(filename="long.pdf", page_count=1, file_size_bytes=1000),
        pages=[
            Page(
                page_number=1,
                text_blocks=[
                    _block(long_text, 0),
                    _block("tail", 1),
                ],
                math_expressions=[],
                diagrams=[],
            )
        ],
    )

    segments = DocumentSegmenterAgent().execute(
        DocumentSegmenterInput(ocr_output=ocr, target_words_per_segment=10)
    ).segments

    assert len(segments) == 3
    assert segments[-1].text.endswith("word24 tail")
    assert all(segment.page_start == segment.page_end == 1 for segment in segments)
