"""Review gates that measure what the viewer sees: the planned-video
storyboard rubric, the render-time audit of sentences on screen and
under-filled frames, and the frame verifier's defect codes. Keyless."""

import json
import shutil
from types import SimpleNamespace

from src.agents import frame_verifier
from src.agents.frame_verifier import FrameVerifierAgent, verdicts_report
from src.agents.storyboard_grader import (
    StoryboardGraderAgent,
    StoryboardGraderInput,
    StoryboardGraderOutput,
)
from src.prompts.review import PLANNED_CRITERIA_WEIGHTS
from src.schemas.storyboard import Configuration, PDFMetadataStoryboard, StoryboardJSON
from src.schemas.video_script import Beat
from src.utils.layout_audit import audit_feedback, content_fill, long_texts, read_audit
from tests.test_lesson_plan import PLAN, _plan
from tests.test_storyboard_grader_coverage import _scene, _segments

LEGACY_CRITERIA = ["pedagogical_flow", "visual_richness", "narration_quality", "completeness"]


# ── storyboard grader ───────────────────────────────────────────────────

def _storyboard() -> StoryboardJSON:
    return StoryboardJSON(
        pdf_metadata=PDFMetadataStoryboard(filename="paper.pdf", page_count=2),
        configuration=Configuration(verbosity="medium", depth="intermediate", audience="undergraduate"),
        concept_hierarchy=[],
        scenes=[_scene("scene_001", 0, ["seg_001"]), _scene("scene_002", 1, ["seg_002"])],
    )


class _FakeClient:
    """Stands in for ``openai.OpenAI``: records each call, answers with all criteria at 8."""

    def __init__(self, fail: bool = False):
        self.calls = []
        self.fail = fail
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("API down")
        names = list(PLANNED_CRITERIA_WEIGHTS) + ["source_coverage"] + LEGACY_CRITERIA
        body = {name: {"score": 8, "feedback": "ok"} for name in names}
        body.update(summary="fine", improvement_suggestions=[])
        message = SimpleNamespace(content=json.dumps(body))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _agent(client) -> StoryboardGraderAgent:
    agent = StoryboardGraderAgent()  # keyless: client is None until set here
    agent.client = client
    return agent


def _prompt(client) -> str:
    return client.calls[-1]["messages"][0]["content"]


def test_planned_rubric_parses_and_weights():
    agent = StoryboardGraderAgent.__new__(StoryboardGraderAgent)
    report = agent._parse_planned_response({
        "learning_path": {"score": 8, "feedback": "a"},
        "narration_economy": {"score": 6},
        "visual_design": {"score": "bad"},
        "hook_and_payoff": {"score": 14},
        "source_coverage": {"score": 3, "feedback": "c"},
        "summary": "s",
        "improvement_suggestions": ["x", "", 3],
    })
    expected = {
        "learning_path": 8.0, "narration_economy": 6.0, "visual_design": 5.0,
        "hook_and_payoff": 10.0, "accuracy": 5.0,
    }
    scores = {c.name: c.score for c in report.criteria}
    for name, score in expected.items():
        assert scores[name] == score
    assert report.overall_score == round(
        sum(expected[name] * weight for name, weight in PLANNED_CRITERIA_WEIGHTS.items()), 2
    )
    assert [c.name for c in report.criteria] == list(PLANNED_CRITERIA_WEIGHTS) + ["source_coverage"]
    coverage = report.criteria[-1]
    assert coverage.weight == 0.0 and coverage.score == 3.0 and coverage.feedback == "c"
    assert report.summary == "s"
    assert report.improvement_suggestions == ["x"]


def test_planned_grading_uses_the_plan_prompt():
    client = _FakeClient()
    storyboard = _storyboard()
    out = _agent(client).execute(StoryboardGraderInput(
        storyboard=storyboard, source_segments=_segments(2), lesson_plan=_plan(),
    ))
    assert isinstance(out, StoryboardGraderOutput)
    assert client.calls[-1]["model"] == "gpt-5.4-mini"
    prompt = _prompt(client)
    assert "learning_path" in prompt
    assert PLAN["central_question"] in prompt
    assert storyboard.scenes[0].narration in prompt
    assert out.report.overall_score == 8.0


def test_unplanned_grading_is_unchanged():
    client = _FakeClient()
    _agent(client).execute(StoryboardGraderInput(storyboard=_storyboard(), source_segments=_segments(2)))
    prompt = _prompt(client)
    assert "completeness" in prompt
    assert "learning_path" not in prompt


def test_planned_grading_falls_back_to_heuristic_on_error():
    client = _FakeClient(fail=True)
    out = _agent(client).execute(StoryboardGraderInput(
        storyboard=_storyboard(), source_segments=_segments(2), lesson_plan=_plan(),
    ))
    assert len(client.calls) == 1
    assert [c.name for c in out.report.criteria] == LEGACY_CRITERIA


def test_without_visuals_drops_visual_design():
    agent = StoryboardGraderAgent.__new__(StoryboardGraderAgent)
    data = {name: {"score": score} for name, score in zip(PLANNED_CRITERIA_WEIGHTS, (9, 7, 6, 2, 8))}
    data["source_coverage"] = {"score": 1}
    out = agent._without_visuals(agent._parse_planned_response(data))
    names = [c.name for c in out.criteria]
    assert "visual_design" not in names
    assert names == [n for n in PLANNED_CRITERIA_WEIGHTS if n != "visual_design"] + ["source_coverage"]
    kept = [c for c in out.criteria if c.weight > 0]
    expected = sum(c.score * c.weight for c in kept) / sum(c.weight for c in kept)
    assert out.overall_score == round(expected, 2)


def test_scene_for_review_fields():
    scene = _scene("scene_007", 6, ["seg_001"])
    scene.narration = "First we stretch the plane. Then we rotate it back."
    scene.source_beats = [
        Beat(beat_id="beat_001", narration="First we stretch the plane.", visual_hint="", act_id="act_02"),
        Beat(beat_id="beat_002", narration="Then we rotate it back.",
             visual_hint="the grid turns", act_id="act_03"),
    ]
    scene.visual_plan_source = "fallback"
    out = StoryboardGraderAgent._scene_for_review(scene)
    assert out["scene_id"] == "scene_007"
    assert out["narration"] == "First we stretch the plane. Then we rotate it back."
    assert out["visual_hints"] == ["the grid turns"]
    assert out["plan_source"] == "fallback"
    assert out["act_id"] == "act_02"
    assert out["scene_type"] == "manim"
    assert out["seconds"] == 30.0
    assert out["primitives"] == [] and out["transformations"] == 0


# ── render-time layout audit ────────────────────────────────────────────

def test_long_texts_flags_sentences_not_labels_or_formulas(tmp_path):
    from manim import MathTex, Text, tempconfig

    with tempconfig({"media_dir": str(tmp_path)}):
        sentence = Text("That is enough to reconstruct the whole matrix A")
        label = Text("top singular vector")
        mobjects = [sentence, label]
        if shutil.which("latex"):
            mobjects.append(MathTex(r"A = U \Sigma V^T = \sum_i \sigma_i u_i v_i^T + 0 + 0 + 0"))
        found = long_texts(mobjects)
    assert [item["words"] for item in found] == [9]
    assert found[0]["text"].startswith("That is enough")


def test_content_fill():
    from manim import Rectangle

    quarter = Rectangle(width=7.11, height=4.0)
    assert 0.24 <= content_fill([quarter], 14.22, 8.0) <= 0.26
    assert content_fill([quarter, Rectangle(width=14.22, height=8.0)], 14.22, 8.0) == 1.0
    assert content_fill([], 14.22, 8.0) == 0.0
    outside = Rectangle(width=6.0, height=4.0).move_to([9.0, 0.0, 0.0])  # x from 6 to 12
    clipped = content_fill([outside], 14.22, 8.0)
    assert 0.0 < clipped <= 1.0
    assert clipped < 0.05  # only x 6..7.11 is inside; unclipped it would be 0.21


def _audit(tmp_path, **data):
    path = tmp_path / "layout_audit.json"
    path.write_text(json.dumps({"overlaps": [], "off_frame": [], **data}), encoding="utf-8")
    return read_audit(path)


def test_read_audit_reports_sentences_and_underfill(tmp_path):
    long_text = [{"text": "That is enough to reconstruct A", "words": 10, "t": 1.0}]
    issues = _audit(tmp_path, long_text=long_text, fill={"max": 0.18, "observations": 12})
    assert len(issues) == 2
    assert "10-word sentence" in issues[0]
    assert "18% of the frame" in issues[1]
    few = _audit(tmp_path, long_text=long_text, fill={"max": 0.18, "observations": 2})
    assert not any("of the frame" in issue for issue in few)
    full = _audit(tmp_path, long_text=long_text, fill={"max": 0.6, "observations": 12})
    assert not any("of the frame" in issue for issue in full)


def test_read_audit_old_files_unchanged(tmp_path):
    path = tmp_path / "layout_audit.json"
    path.write_text(json.dumps({
        "overlaps": [{"a": "x", "b": "y", "ratio": 0.5, "t": 2.0}],
        "off_frame": [],
    }), encoding="utf-8")
    assert read_audit(path) == ['Text "x" overlaps text "y" (50% of the smaller one) at 2.0s.']


def test_audit_feedback_mentions_labels_and_fill():
    feedback = audit_feedback(["x"])
    assert any("at most four words" in line for line in feedback)
    assert any("fill the frame" in line for line in feedback)


# ── frame verifier ──────────────────────────────────────────────────────

def test_frame_verifier_collects_issue_codes(monkeypatch):
    prompts = []

    def fake(prompt, images):
        prompts.append(prompt)
        return {"easiest": "manim", "legible": True, "reason": "r",
                "issues": ["underfilled_frame", "made_up_code"]}

    monkeypatch.setattr(frame_verifier, "sample_frames", lambda video, out_dir, label: ["f.png"])
    verdicts = FrameVerifierAgent(vision=fake).review([
        {"scene_id": "scene_001", "frame": "manim", "video": "v.mp4", "narration": "n"},
    ])
    assert len(verdicts) == 1
    verdict = verdicts[0]
    assert verdict.issues == ["underfilled_frame"]
    assert verdict.action == "keep"
    prompt = prompts[0]
    assert "underfilled_frame" in prompt
    assert "Manim is for VISUALIZATION" in prompt
    assert "EASIEST TO UNDERSTAND" in prompt
    assert verdicts_report([verdict])["issues"] == {"scene_001": ["underfilled_frame"]}
