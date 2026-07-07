"""Regression tests for the two failures a real fine-granularity run exposed:

1. Scene-count explosion — adaptive segment sizing caps long documents.
2. Visual-intent ~95% fallback — objects missing object_type/content are coerced
   and validated item-by-item instead of dropping the whole scene.
"""

import pytest

from src.agents.document_segmenter import (
    GRANULARITY_WORD_TARGETS,
    MAX_SEGMENTS,
    target_words_for,
    total_ocr_words,
)
from src.agents.script_architect import (
    ScriptArchitectAgent,
    _coerce_visual_object,
    _validate_items,
)
from src.schemas.storyboard import MathematicalObject


# ---------------------------------------------------------------------------
# Fix B — adaptive segment sizing / scene cap
# ---------------------------------------------------------------------------

class TestAdaptiveSizing:
    def test_long_doc_is_capped_under_max_segments(self):
        # 10k words at fine (90) would be ~110 segments; adaptive must cap it.
        target = target_words_for("fine", 10_000)
        approx_segments = -(-10_000 // target)  # ceil
        assert approx_segments <= MAX_SEGMENTS
        assert target > GRANULARITY_WORD_TARGETS["fine"]  # raised above the base

    def test_short_doc_honours_granularity_dial(self):
        # A short doc stays under the cap already, so the dial is respected.
        assert target_words_for("fine", 500) == GRANULARITY_WORD_TARGETS["fine"]
        assert target_words_for("coarse", 500) == GRANULARITY_WORD_TARGETS["coarse"]

    def test_coarser_dial_is_never_finer_than_base(self):
        assert target_words_for("coarse", 10_000) >= GRANULARITY_WORD_TARGETS["coarse"]

    def test_zero_words_returns_base(self):
        assert target_words_for("balanced", 0) == GRANULARITY_WORD_TARGETS["balanced"]

    def test_total_ocr_words_counts_all_blocks(self):
        from src.schemas.ocr_output import (
            BoundingBox, OCROutput, Page, PDFMetadata, TextBlock,
        )
        bb = BoundingBox(x=0, y=0, width=1, height=1)
        ocr = OCROutput(
            pdf_metadata=PDFMetadata(filename="t.pdf", page_count=1, file_size_bytes=1),
            pages=[Page(page_number=1, text_blocks=[
                TextBlock(text="one two three", reading_order=0, bounding_box=bb),
                TextBlock(text="four five", reading_order=1, bounding_box=bb),
            ])],
        )
        assert total_ocr_words(ocr) == 5


# ---------------------------------------------------------------------------
# Fix A — visual-intent coercion + item-by-item validation
# ---------------------------------------------------------------------------

class TestVisualObjectCoercion:
    def test_infers_geometric_shape_from_primitive(self):
        o = _coerce_visual_object({"object_id": "c0", "primitive": "Circle",
                                   "args": {"radius": 2.0}})
        assert o["object_type"] == "geometric_shape"
        assert o["content"] == ""
        MathematicalObject.model_validate(o)  # must not raise

    def test_infers_equation_and_pulls_tex_content(self):
        o = _coerce_visual_object({"primitive": "MathTex", "args": {"tex": r"e^{i\pi}"}})
        assert o["object_type"] == "equation"
        assert o["content"] == r"e^{i\pi}"

    def test_infers_axes_and_pi_creature(self):
        assert _coerce_visual_object({"primitive": "Axes"})["object_type"] == "axes"
        assert _coerce_visual_object(
            {"primitive": "create_pi_teacher"}
        )["object_type"] == "pi_creature"

    def test_existing_object_type_is_preserved(self):
        o = _coerce_visual_object({"object_type": "text", "content": "hi", "primitive": "Text"})
        assert o["object_type"] == "text"
        assert o["content"] == "hi"

    def test_non_dict_passthrough(self):
        assert _coerce_visual_object("not a dict") == "not a dict"


class TestValidateItems:
    def test_drops_invalid_keeps_valid(self):
        items = [
            {"object_type": "geometric_shape", "content": "", "primitive": "Circle",
             "args": {"radius": 1.0}},
            {"object_type": "geometric_shape", "content": "", "primitive": "NotReal"},
        ]
        out = _validate_items(items, MathematicalObject)
        assert len(out) == 1
        assert out[0].primitive == "Circle"

    def test_non_list_returns_empty(self):
        assert _validate_items(None, MathematicalObject) == []


class TestVisualIntentFromJson:
    def _agent(self):
        return ScriptArchitectAgent.__new__(ScriptArchitectAgent)

    def test_object_missing_type_and_content_is_rescued(self):
        # The exact shape that caused ~95% fallback in the real run.
        data = {"objects": [{
            "object_id": "c0", "primitive": "Circle", "args": {"radius": 2.0},
            "bbox": {"x": 0, "y": 0, "w": 2, "h": 2},
            "interval": {"t_in": 0, "t_out": 5},
        }]}
        intent = self._agent()._visual_intent_from_json(data, "narration text")
        ids = [o.object_id for o in intent.mathematical_objects]
        assert "c0" in ids  # rescued, not degraded to fallback
        obj = next(o for o in intent.mathematical_objects if o.object_id == "c0")
        assert obj.object_type == "geometric_shape"
        assert obj.primitive == "Circle"

    def test_mixed_valid_and_invalid_keeps_valid_only(self):
        data = {"objects": [
            {"object_id": "good", "primitive": "Circle", "args": {"radius": 2.0}},
            {"object_id": "bad", "primitive": "NotAPrimitive"},
        ]}
        intent = self._agent()._visual_intent_from_json(data, "n")
        ids = [o.object_id for o in intent.mathematical_objects]
        assert ids == ["good"]  # one bad object doesn't void the scene

    def test_all_invalid_falls_back(self):
        data = {"objects": [{"object_id": "bad", "primitive": "NotAPrimitive"}]}
        intent = self._agent()._visual_intent_from_json(data, "n")
        ids = [o.object_id for o in intent.mathematical_objects]
        assert "bad" not in ids
        assert intent.mathematical_objects  # fallback still produced something


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
