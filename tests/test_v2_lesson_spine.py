"""WPB: lesson spine, spoken narration rules, script review and code-gen style in V2."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli import pdf2manim
from src.pipeline_v2 import PipelineV2, PipelineV2Config, RunStatus
from src.pipeline_v2.generation import generation_request
from src.pipeline_v2.model_config import TEXT_MODEL
from src.pipeline_v2.models import (
    LessonPlan,
    LessonScene,
    NarratedScene,
    NarrationBeat,
    NarrationPlan,
    RunResult,
    SourceDocument,
    SourceFact,
)
from src.pipeline_v2.planning import (
    NARRATION_PROMPT_VERSION,
    PROMPT_VERSION,
    PlanningService,
)
from src.pipeline_v2.prompt_text import CODEGEN_STYLE_RULES, REVIEW_PASS_SCORE
from src.pipeline_v2.providers import (
    MissingProviderKey,
    ProviderResponse,
    ProviderRouter,
    _google_compatible_schema,
)
from src.pipeline_v2.script_review import review_script
from src.prompts.review import PLANNED_CRITERIA_WEIGHTS


FIXTURE = Path(__file__).parent / "fixtures" / "test_trig_3pages.pdf"


def _without_keys(monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _source(pages: int = 4) -> SourceDocument:
    return SourceDocument(
        filename="notes.pdf", sha256="a", page_count=pages,
        facts=[
            SourceFact(fact_id=f"p{page:03d}-f000", page_number=page, text=f"Fact on page {page}.")
            for page in range(1, pages + 1)
        ],
    )


def _lesson_dict() -> dict:
    """An old-shape lesson: no spine keys at all."""
    return {
        "title": "Right triangles",
        "audience": "undergraduate",
        "objectives": ["See why a^2 + b^2 = c^2"],
        "scenes": [
            {
                "scene_id": "scene_001", "title": "Squares", "description": "Build squares",
                "learning_goal": "Areas add up", "source_fact_ids": ["p001-f000"],
            },
            {
                "scene_id": "scene_002", "title": "Rearrange", "description": "Move pieces",
                "learning_goal": "The big square is fixed", "source_fact_ids": ["p003-f000"],
            },
        ],
    }


def _spine_dict() -> dict:
    lesson = _lesson_dict()
    lesson["scenes"][1]["viewer_question"] = "Why do the leftover areas match?"
    lesson.update({
        "central_question": "Why do two small squares always fill the big one exactly?",
        "hook": "Two squares, one big square, and they always match. Why?",
        "takeaway": "Rearranging four triangles leaves the same empty area both ways.",
        "running_example": {
            "description": "A 3-4-5 right triangle",
            "values": {"a": "3", "b": "4", "c": "5"},
        },
        "motifs": [
            {"motif_id": "triangle", "what": "the 3-4-5 triangle", "color": "BLUE_B",
             "stands_for": "the right triangle"},
            {"motif_id": "big_square", "what": "the square on c"},
        ],
        "coverage": [{"page_number": 1, "treatment": "covered", "reason": ""}],
    })
    return lesson


def _review_json(score: float, suggestions=("Open on the triangle, not a definition.",)) -> str:
    data = {
        name: {"score": score, "feedback": f"{name} feedback"} for name in PLANNED_CRITERIA_WEIGHTS
    }
    data["source_coverage"] = {"score": 6, "feedback": "Covers the main line."}
    data["summary"] = "One sentence."
    data["improvement_suggestions"] = list(suggestions)
    return json.dumps(data)


def _response(text: str) -> ProviderResponse:
    return ProviderResponse(
        text=text, provider="google", model=TEXT_MODEL, input_units=1, output_units=1,
        exact_usage=True, raw={},
    )


class CapturingRouter:
    """Records structured calls and answers from a handler or the fallback."""

    strict_keys = False

    def __init__(self, narration=None):
        self.calls: list[dict] = []
        self.narration = narration
        self.artifacts = SimpleNamespace(event=lambda *args, **kwargs: None)

    async def structured(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["stage"] == "narration" and self.narration:
            return self.narration(kwargs)
        return kwargs["fallback"]()

    def prompts(self, stage: str) -> list[str]:
        return [call["prompt"] for call in self.calls if call["stage"] == stage]


def _payload(prompt: str) -> dict:
    start = prompt.index('{"lesson_title"')
    return json.JSONDecoder().raw_decode(prompt, start)[0]


# 1 ───────────────────────────────────────────────────────────────────────────


def test_lesson_plan_spine_fields_default_and_round_trip():
    old = LessonPlan.model_validate(_lesson_dict())
    assert old.central_question == old.hook == old.takeaway == ""
    assert old.running_example.description == "" and old.running_example.values == {}
    assert old.motifs == [] and old.coverage == []
    assert old.scenes[0].viewer_question == ""
    assert old.visual_world() == []

    full = LessonPlan.model_validate(_spine_dict())
    assert LessonPlan.model_validate_json(full.model_dump_json()) == full
    assert full.scenes[1].viewer_question == "Why do the leftover areas match?"
    assert full.visual_world() == [
        "Running example: A 3-4-5 right triangle",
        "Value: a = 3",
        "Value: b = 4",
        "Value: c = 5",
        "Motif triangle: the 3-4-5 triangle; color BLUE_B; stands for the right triangle",
        "Motif big_square: the square on c",
    ]
    spine = full.spine()
    assert list(spine) == ["central_question", "hook", "takeaway", "running_example", "motifs"]
    assert spine["running_example"] == {
        "description": "A 3-4-5 right triangle", "values": {"a": "3", "b": "4", "c": "5"},
    }
    assert spine["motifs"][1] == {
        "motif_id": "big_square", "what": "the square on c", "color": "", "stands_for": "",
    }
    json.dumps(spine)  # plain JSON-ready values


# 2 ───────────────────────────────────────────────────────────────────────────


def test_google_schema_accepts_lesson_plan():
    schema = _google_compatible_schema(LessonPlan.model_json_schema())

    assert "central_question" in schema["properties"]
    assert "motifs" in schema["properties"]
    assert "running_example" in schema["properties"]
    assert "coverage" in schema["properties"]
    assert "maxItems" not in json.dumps(schema)


# 3 ───────────────────────────────────────────────────────────────────────────


def test_lesson_prompt_uses_spine_instructions():
    router = CapturingRouter()
    planner = PlanningService(router, audience="graduate")

    asyncio.run(planner.lesson(_source()))

    prompt = router.prompts("lesson")[0]
    assert prompt.startswith(f"PROMPT_VERSION={PROMPT_VERSION}\n")
    assert PROMPT_VERSION == "v2.5.0-spine"
    assert "ONE CENTRAL QUESTION" in prompt
    assert '"page_count": 4' in prompt
    assert "Prefer 4-6 substantial scenes" in prompt  # scene guidance kept
    assert "PROFESSOR DIRECTIVES (binding" in prompt
    assert "- Audience: graduate." in prompt
    assert "presenter speaks" not in prompt

    presenter_router = CapturingRouter()
    asyncio.run(PlanningService(presenter_router, presenter_opens=True).lesson(_source()))
    assert "presenter speaks" in presenter_router.prompts("lesson")[0]


# 4 ───────────────────────────────────────────────────────────────────────────


def test_spine_normalisation_fills_coverage():
    source = _source(4)
    planned = _lesson_dict()
    planned["coverage"] = [
        {"page_number": 3, "treatment": "omitted", "reason": ""},
        {"page_number": 3, "treatment": "covered", "reason": "A later duplicate."},
        {"page_number": 7, "treatment": "covered", "reason": "Out of range."},
    ]
    planned["motifs"] = [
        {"motif_id": f"m{index}", "what": "dots", "color": "blue b"} for index in range(5)
    ]

    class PlannedRouter(CapturingRouter):
        async def structured(self, **kwargs):
            self.calls.append(kwargs)
            return LessonPlan.model_validate(planned)

    lesson = asyncio.run(PlanningService(PlannedRouter()).lesson(source))

    assert [(item.page_number, item.treatment, item.reason) for item in lesson.coverage] == [
        (1, "covered", "Cited by a scene."),
        (2, "omitted", "The lesson plan did not place this page."),
        (3, "omitted", "No reason given by the planner."),
        (4, "omitted", "The lesson plan did not place this page."),
    ]
    assert [motif.motif_id for motif in lesson.motifs] == ["m0", "m1", "m2", "m3"]
    assert {motif.color for motif in lesson.motifs} == {"BLUE_B"}

    fallback = PlanningService(CapturingRouter())._fallback_lesson(source)
    assert fallback.spine() == LessonPlan.model_validate(_lesson_dict()).spine()


# 5 ───────────────────────────────────────────────────────────────────────────


def test_narration_prompt_has_spoken_rules_and_spine():
    router = CapturingRouter()
    lesson = LessonPlan.model_validate(_spine_dict())

    asyncio.run(PlanningService(router).narration(lesson, _source()))

    prompt = router.prompts("narration")[0]
    assert prompt.startswith(f"PROMPT_VERSION={NARRATION_PROMPT_VERSION}\nWrite the requested")
    assert NARRATION_PROMPT_VERSION == "v2.6.0-spoken"
    assert "SPOKEN NARRATION" in prompt
    assert "calm teaching pace" in prompt  # pace guidance kept
    assert "A REVIEWER REJECTED" not in prompt
    payload = _payload(prompt)
    assert payload["lesson_spine"]["running_example"]["description"] == "A 3-4-5 right triangle"
    assert payload["lesson_spine"]["central_question"].startswith("Why do two small squares")
    assert [entry["viewer_question"] for entry in payload["global_scene_outline"]] == [
        "", "Why do the leftover areas match?",
    ]
    assert payload["scenes_to_write"][1]["viewer_question"] == "Why do the leftover areas match?"

    feedback_router = CapturingRouter()
    asyncio.run(PlanningService(feedback_router).narration(
        lesson, _source(), review_feedback=["Cut the recap in scene_002.", "Open on the triangle."],
    ))
    assert feedback_router.prompts("narration")[0].startswith(
        f"PROMPT_VERSION={NARRATION_PROMPT_VERSION}\n"
        "A REVIEWER REJECTED THE PREVIOUS SCRIPT. Fix every point:\n"
        "- Cut the recap in scene_002.\n- Open on the triangle.\n"
        "Write the requested"
    )


# 6 ───────────────────────────────────────────────────────────────────────────


def test_narration_strips_snake_case_ids():
    lesson = LessonPlan.model_validate(_lesson_dict()).model_copy(
        update={"scenes": LessonPlan.model_validate(_lesson_dict()).scenes[:1]},
    )

    def leaky(_call):
        return NarrationPlan(scenes=[NarratedScene(
            scene_id="scene_001", narration="the best_fit_line moves",
            beats=[NarrationBeat(
                beat_id="scene_001_b1", text="the best_fit_line moves",
                source_fact_ids=["p001-f000"],
            )],
        )])

    narration = asyncio.run(PlanningService(CapturingRouter(narration=leaky)).narration(
        lesson, _source(),
    ))

    scene = narration.scenes[0]
    assert scene.beats[0].text == "the best fit line moves"
    assert scene.beats[0].beat_id == "scene_001_b1"
    assert scene.narration == "the best fit line moves"


# 7 ───────────────────────────────────────────────────────────────────────────


def test_script_review_scores_and_parses():
    calls = []
    raw = {
        "learning_path": {"score": 8, "feedback": "Clear path."},
        "accuracy": {"score": "12", "feedback": "Fine."},          # clamped to 10
        "narration_economy": {"feedback": "No score given."},      # default 5
        "visual_design": {"score": "bad"},                         # default 5
        "hook_and_payoff": {"score": -2, "feedback": "Generic."},  # clamped to 0
        "source_coverage": {"score": 3, "feedback": "Thin."},
        "summary": "Decent.",
        "improvement_suggestions": ["Name the triangle.", "", 4, "Cut scene 2's recap."],
    }
    expected = {
        "learning_path": 8.0, "accuracy": 10.0, "narration_economy": 5.0,
        "visual_design": 5.0, "hook_and_payoff": 0.0,
    }

    class Router:
        artifacts = SimpleNamespace(event=lambda *args, **kwargs: None)

        async def text(self, **kwargs):
            calls.append(kwargs)
            return _response("```json\n" + json.dumps(raw) + "\n```")

    lesson = LessonPlan.model_validate(_spine_dict())
    narration = PlanningService(CapturingRouter())._fallback_narration(lesson, _source())

    review = asyncio.run(review_script(Router(), lesson, narration, _source(), attempt=2))

    overall = round(sum(expected[name] * weight for name, weight in PLANNED_CRITERIA_WEIGHTS.items()), 2)
    assert review.overall_score == overall
    assert review.passed == (overall >= REVIEW_PASS_SCORE)
    assert {name: entry["score"] for name, entry in review.criteria.items()} == {
        **expected, "source_coverage": 3.0,
    }
    assert review.criteria["source_coverage"]["weight"] == 0.0
    assert review.criteria["learning_path"]["weight"] == PLANNED_CRITERIA_WEIGHTS["learning_path"]
    assert review.improvement_suggestions == ["Name the triangle.", "Cut scene 2's recap."]
    assert review.summary == "Decent." and review.attempt == 2
    assert review.weakest_criterion()[0] == "hook_and_payoff"

    call = calls[0]
    assert (call["stage"], call["task_id"], call["provider"], call["model"]) == (
        "script_review", "script-review-2", "google", TEXT_MODEL,
    )
    assert call["system"] == "You are a demanding editor of short educational videos."
    assert call["max_output_tokens"] == 3000
    assert "Why do two small squares always fill the big one exactly?" in call["prompt"]
    assert '"omitted": 0' in call["prompt"] and '"covered": 1' in call["prompt"]

    passing = dict(raw, **{name: {"score": 9} for name in PLANNED_CRITERIA_WEIGHTS})
    calls.clear()
    raw.clear()
    raw.update(passing)
    assert asyncio.run(review_script(Router(), lesson, narration, _source())).passed is True


# 8 ───────────────────────────────────────────────────────────────────────────


REWRITE_MARK = "Rewritten after the review"


def _scripted_router(review_scores: dict[str, float], prompts: list[str]):
    """A real router whose model answers are scripted; unscripted stages are keyless."""

    class ScriptedRouter(ProviderRouter):
        async def text(self, *, stage, task_id, prompt, **kwargs):
            if stage == "script_review":
                return _response(_review_json(review_scores[task_id]))
            if stage == "narration":
                prompts.append(prompt)
                if prompt.startswith(f"PROMPT_VERSION={NARRATION_PROMPT_VERSION}\nA REVIEWER"):
                    payload = _payload(prompt)
                    return _response(json.dumps({"scenes": [
                        {
                            "scene_id": item["scene_id"],
                            "narration": f"{REWRITE_MARK}.",
                            "beats": [{
                                "beat_id": f"{item['scene_id']}_b1",
                                "text": f"{REWRITE_MARK}.",
                                "source_fact_ids": item["source_fact_ids"][:1],
                            }],
                        }
                        for item in payload["scenes_to_write"]
                    ]}))
            raise MissingProviderKey("GEMINI_API_KEY is not configured")

    return ScriptedRouter


@pytest.mark.parametrize(
    ("second_score", "rewrite_kept"),
    [(7.5, True), (4.0, False)],
)
def test_script_review_revise_keeps_better_attempt(monkeypatch, tmp_path, second_score, rewrite_kept):
    _without_keys(monkeypatch)
    prompts: list[str] = []
    monkeypatch.setattr(
        "src.pipeline_v2.pipeline.ProviderRouter",
        _scripted_router({"script-review-1": 5.0, "script-review-2": second_score}, prompts),
    )
    config = PipelineV2Config(
        output_dir=tmp_path, max_scenes=2, dry_run=True,
        enable_tts=False, enable_render=False, repair="off",
    )
    assert config.script_review == "revise"

    result = PipelineV2(config).run(FIXTURE)

    assert result.status == RunStatus.SUCCESS, result.message
    run_dir = Path(result.run_directory)
    assert len(prompts) == 2
    assert prompts[1].startswith(
        f"PROMPT_VERSION={NARRATION_PROMPT_VERSION}\n"
        "A REVIEWER REJECTED THE PREVIOUS SCRIPT. Fix every point:\n"
        "- Open on the triangle, not a definition.\n"
    )
    attempts = json.loads((run_dir / "script_review.json").read_text(encoding="utf-8"))
    assert [(item["attempt"], item["overall_score"]) for item in attempts] == [
        (1, 5.0), (2, second_score),
    ]
    storyboard = json.loads((run_dir / "storyboard.json").read_text(encoding="utf-8"))
    spoken = " ".join(scene["narration"] for scene in storyboard["narration"]["scenes"])
    assert (REWRITE_MARK in spoken) is rewrite_kept
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    reviewed = [event for event in events if event["event"] == "script_reviewed"]
    assert len(reviewed) == 1 and reviewed[0]["rewritten"] is rewrite_kept
    report = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Script review" in report
    assert f"Score {max(5.0, second_score):.2f}/10" in report


def test_script_review_report_mode_runs_once(monkeypatch, tmp_path):
    _without_keys(monkeypatch)
    prompts: list[str] = []
    monkeypatch.setattr(
        "src.pipeline_v2.pipeline.ProviderRouter",
        _scripted_router({"script-review-1": 5.0}, prompts),
    )
    config = PipelineV2Config(
        output_dir=tmp_path, max_scenes=2, dry_run=True, script_review="report",
        enable_tts=False, enable_render=False, repair="off",
    )

    result = PipelineV2(config).run(FIXTURE)

    assert result.status == RunStatus.SUCCESS, result.message
    attempts = json.loads((Path(result.run_directory) / "script_review.json").read_text())
    assert [item["overall_score"] for item in attempts] == [5.0]
    assert len(prompts) == 1  # no rewrite in report mode


# 9 ───────────────────────────────────────────────────────────────────────────


def test_script_review_keyless_is_skipped(monkeypatch, tmp_path):
    _without_keys(monkeypatch)

    events = []

    class KeylessRouter:
        artifacts = SimpleNamespace(event=lambda name, **kwargs: events.append((name, kwargs)))

        async def text(self, **kwargs):
            raise MissingProviderKey("GEMINI_API_KEY is not configured")

    lesson = LessonPlan.model_validate(_spine_dict())
    narration = PlanningService(CapturingRouter())._fallback_narration(lesson, _source())
    assert asyncio.run(review_script(KeylessRouter(), lesson, narration, _source())) is None
    assert [name for name, _ in events] == ["script_review_skipped"]
    assert "GEMINI_API_KEY" in events[0][1]["message"]

    config = PipelineV2Config(
        output_dir=tmp_path, max_scenes=2, dry_run=True,
        enable_tts=False, enable_render=False, repair="off",
    )
    result = PipelineV2(config).run(FIXTURE)

    assert result.status == RunStatus.SUCCESS, result.message
    run_dir = Path(result.run_directory)
    logged = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert any(event["event"] == "script_review_skipped" for event in logged)
    assert not (run_dir / "script_review.json").exists()
    assert result.estimated_cost_usd == 0.0


# 10 ──────────────────────────────────────────────────────────────────────────


def test_codegen_request_has_style_rules_and_visual_world():
    scene = NarratedScene(
        scene_id="scene_001", narration="Watch the dots.",
        beats=[NarrationBeat(
            beat_id="scene_001_b1", text="Watch the dots.", source_fact_ids=["p001-f000"],
            end_seconds=2.0,
        )],
    )
    contract = PlanningService._fallback_contract(scene)
    facts = _source(1).facts

    system, prompt = generation_request(
        scene=scene, contract=contract, source_facts=facts, class_name="V2Scene001",
        visual_world=["Motif m: dots"],
    )
    assert system.endswith(CODEGEN_STYLE_RULES)
    assert "-style" in system.splitlines()[0]
    assert json.loads(prompt)["visual_world"] == ["Motif m: dots"]

    plain_system, plain_prompt = generation_request(
        scene=scene, contract=contract, source_facts=facts, class_name="V2Scene001",
    )
    assert plain_system.endswith(CODEGEN_STYLE_RULES)
    assert "visual_world" not in json.loads(plain_prompt)


@pytest.mark.parametrize("with_spine", [True, False])
def test_pipeline_passes_visual_world_to_code_generation(tmp_path, with_spine):
    lesson = LessonPlan.model_validate(_spine_dict() if with_spine else _lesson_dict())
    source = _source()
    narrated = PlanningService(CapturingRouter())._fallback_narration(lesson, source).scenes[0]
    contract = PlanningService._fallback_contract(narrated)
    generated: list[dict] = []
    task_inputs: dict[str, dict] = {}

    class Generator:
        async def generate(self, *_args, **kwargs):
            generated.append(kwargs)
            return "from manim import *\n"

    class Artifacts:
        def write_json(self, path, payload):
            target = tmp_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload), encoding="utf-8")

        def event(self, *_args, **_kwargs):
            return None

    async def exercise():
        pipeline = PipelineV2(PipelineV2Config(enable_render=False))
        pipeline.artifacts = Artifacts()

        async def execute(task_id, _stage, _scene_id, inputs, operation, **_kwargs):
            task_inputs[task_id] = inputs
            return await operation()

        pipeline._task = execute
        return await pipeline._process_scene(
            input_path=FIXTURE, source=source, lesson=lesson, original_scene=narrated,
            timed_scene_task=asyncio.create_task(
                asyncio.sleep(0, result=narrated.model_dump(mode="json")),
            ),
            contract_task=asyncio.create_task(asyncio.sleep(
                0, result={"contracts": [contract.model_dump(mode="json")]},
            )),
            scene_dir=tmp_path / narrated.scene_id, generator=Generator(),
        )

    result = asyncio.run(exercise())

    assert result.success and result.status == "CODE_GENERATED"
    code_inputs = task_inputs[f"code-{narrated.scene_id}"]
    if with_spine:
        assert generated[0]["visual_world"] == lesson.visual_world() != []
        assert code_inputs["visual_world"] == lesson.visual_world()
    else:
        assert "visual_world" not in generated[0]
        assert "visual_world" not in code_inputs


# 11 ──────────────────────────────────────────────────────────────────────────


def test_cli_script_review_option(tmp_path):
    def convert(*args: str) -> PipelineV2Config:
        pipeline = MagicMock()
        pipeline.run.return_value = RunResult(
            run_id="test", run_directory=str(tmp_path), status=RunStatus.SUCCESS,
        )
        with patch("src.cli.PipelineV2", return_value=pipeline) as factory:
            result = CliRunner().invoke(pdf2manim, ["convert", str(FIXTURE), *args])
        assert result.exit_code == 0, result.output
        return factory.call_args.args[0]

    assert convert("--script-review", "report").script_review == "report"
    assert convert("--script-review", "off").script_review == "off"
    assert convert().script_review == "revise"
    assert convert().presenter == "off"


def test_slightly_off_spine_is_cleaned_not_rejected():
    """A page 0 coverage entry or seven motifs must not make the whole plan invalid."""
    from src.pipeline_v2.models import LessonPlan

    plan = LessonPlan.model_validate({
        "title": "T", "audience": "u", "objectives": ["o"],
        "scenes": [{"scene_id": "scene_01", "title": "t", "description": "d",
                    "learning_goal": "g", "source_fact_ids": ["p001-001"]}],
        "coverage": [{"page_number": 0, "treatment": "covered"}],
        "motifs": [{"motif_id": f"m{i}"} for i in range(7)],
    })
    assert len(plan.motifs) == 7 and plan.coverage[0].page_number == 0

