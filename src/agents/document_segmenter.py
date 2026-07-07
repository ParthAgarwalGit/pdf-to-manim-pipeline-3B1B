"""Deterministic OCR-to-source-segment partitioning."""

from dataclasses import dataclass
import re
from typing import List, Optional

from src.agents.base import Agent, AgentExecutionError, AgentInput, AgentOutput, BackoffStrategy, RetryPolicy
from src.schemas.ocr_output import OCROutput
from src.schemas.source_segment import SourceSegment


# Words-per-segment by the --granularity dial.
GRANULARITY_WORD_TARGETS = {"coarse": 300, "balanced": 150, "fine": 90}
# Hard ceiling on segment count. Without this, a long PDF at fine granularity
# explodes: a ~10k-word doc at 90 words/segment yields ~110 segments -> 100+
# scenes -> hours of render for an unwatchably fragmented video.
MAX_SEGMENTS = 40

# Estimated finished duration of one scene, used to convert a requested video
# length (--target-minutes) into a scene budget. Matches ScriptArchitect's
# TARGET_SCENE_DURATION so the segment floor and the chunker ceiling agree.
SECONDS_PER_SCENE_ESTIMATE = 25.0


def scenes_for_minutes(target_minutes: float) -> int:
    """Scene budget implied by a requested finished-video length (>=1)."""
    seconds = max(0.0, float(target_minutes)) * 60.0
    return max(1, round(seconds / SECONDS_PER_SCENE_ESTIMATE))


def target_words_for(
    granularity: str, total_words: int, max_segments: int = MAX_SEGMENTS
) -> int:
    """Words-per-segment target, adaptive to document length.

    Honours the granularity dial for short documents, but for long ones raises
    the target so the segment count stays under ``max_segments`` (default
    ``MAX_SEGMENTS``; a length-governed run passes a tighter cap). Short PDFs
    are unaffected; long PDFs get proportionally larger segments instead of an
    unbounded scene explosion.
    """
    base = GRANULARITY_WORD_TARGETS.get(granularity, 150)
    cap = max(1, int(max_segments))
    if total_words <= 0:
        return base
    cap_target = -(-total_words // cap)  # ceil(total_words / cap)
    return max(base, cap_target)


def total_ocr_words(ocr_output: OCROutput) -> int:
    """Total word count across all OCR text blocks (drives adaptive sizing)."""
    return sum(
        len(block.text.split())
        for page in ocr_output.pages
        for block in page.text_blocks
    )


class DocumentSegmenterInput(AgentInput):
    """Input for the deterministic document segmenter."""

    def __init__(
        self,
        ocr_output: OCROutput,
        target_words_per_segment: int = 150,
    ):
        self.ocr_output = ocr_output
        self.target_words_per_segment = max(1, int(target_words_per_segment))


@dataclass
class DocumentSegmenterOutput(AgentOutput):
    """Segmenter output."""

    segments: List[SourceSegment]


class DocumentSegmenterAgent(Agent):
    """Partition OCR text into ordered, non-overlapping source segments."""

    MIN_TAIL_RATIO = 0.4

    def execute(self, input_data: AgentInput) -> DocumentSegmenterOutput:
        if not self.validate_input(input_data):
            raise AgentExecutionError(
                "INVALID_INPUT",
                "Input must be a DocumentSegmenterInput object.",
            )

        segment_input: DocumentSegmenterInput = input_data
        segments = self.segment_ocr(
            segment_input.ocr_output,
            segment_input.target_words_per_segment,
        )
        return DocumentSegmenterOutput(segments=segments)

    def validate_input(self, input_data: AgentInput) -> bool:
        return (
            isinstance(input_data, DocumentSegmenterInput)
            and isinstance(input_data.ocr_output, OCROutput)
        )

    def get_retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_attempts=1,
            backoff_strategy=BackoffStrategy.CONSTANT,
            retryable_errors=[],
        )

    def segment_ocr(
        self,
        ocr_output: OCROutput,
        target_words_per_segment: int = 150,
    ) -> List[SourceSegment]:
        target_words_per_segment = max(1, int(target_words_per_segment))
        units = []
        for page in sorted(ocr_output.pages, key=lambda p: p.page_number):
            blocks = sorted(page.text_blocks, key=lambda b: b.reading_order)
            for block in blocks:
                text = block.text.strip()
                if not text:
                    continue
                units.extend(
                    {
                        "text": part,
                        "page": page.page_number,
                        "heading": self._heading_from_text(part),
                    }
                    for part in self._split_text(text, target_words_per_segment)
                )

        if not units:
            return [
                SourceSegment(
                    segment_id="seg_001",
                    order=0,
                    page_start=1,
                    page_end=ocr_output.pdf_metadata.page_count,
                    text="",
                )
            ]

        raw_segments = []
        current_texts: List[str] = []
        current_pages: List[int] = []
        current_heading = ""
        current_words = 0

        for unit in units:
            words = unit["text"].split()
            starts_heading = bool(unit["heading"])
            should_flush_for_heading = starts_heading and current_texts and current_words >= target_words_per_segment // 2
            should_flush_for_size = current_texts and current_words + len(words) > target_words_per_segment
            if should_flush_for_heading or should_flush_for_size:
                raw_segments.append((current_texts, current_pages, current_heading))
                current_texts, current_pages, current_heading, current_words = [], [], "", 0

            current_texts.append(unit["text"])
            current_pages.append(unit["page"])
            current_words += len(words)
            if not current_heading and unit["heading"]:
                current_heading = unit["heading"]

        if current_texts:
            raw_segments.append((current_texts, current_pages, current_heading))

        if len(raw_segments) > 1:
            tail_words = len(" ".join(raw_segments[-1][0]).split())
            if tail_words < target_words_per_segment * self.MIN_TAIL_RATIO:
                prev_texts, prev_pages, prev_heading = raw_segments[-2]
                tail_texts, tail_pages, _tail_heading = raw_segments[-1]
                raw_segments[-2] = (prev_texts + tail_texts, prev_pages + tail_pages, prev_heading)
                raw_segments.pop()

        segments = [
            SourceSegment(
                segment_id=f"seg_{index:03d}",
                order=index - 1,
                page_start=min(pages),
                page_end=max(pages),
                heading=heading,
                text=" ".join(texts),
            )
            for index, (texts, pages, heading) in enumerate(raw_segments, start=1)
        ]
        self._attach_math(segments, ocr_output)
        return segments

    def _split_text(self, text: str, target_words_per_segment: int) -> List[str]:
        words = text.split()
        if len(words) <= target_words_per_segment:
            return [text]
        return [
            " ".join(words[index:index + target_words_per_segment])
            for index in range(0, len(words), target_words_per_segment)
        ]

    def _heading_from_text(self, text: str) -> str:
        first_sentence = re.split(r"[.\n]", text.strip(), maxsplit=1)[0].strip()
        if re.match(r"^(section|chapter|definition|theorem|lemma|proposition|example)\b", first_sentence, re.I):
            return first_sentence[:120]
        return ""

    def _attach_math(self, segments: List[SourceSegment], ocr_output: OCROutput) -> None:
        for page in ocr_output.pages:
            math_latex = [
                expr.latex
                for expr in page.math_expressions
                if not getattr(expr, "extraction_failed", False)
            ]
            if not math_latex:
                continue
            target: Optional[SourceSegment] = next(
                (
                    segment
                    for segment in segments
                    if segment.page_start <= page.page_number <= segment.page_end
                ),
                None,
            )
            if target is not None:
                target.math_latex.extend(math_latex)
