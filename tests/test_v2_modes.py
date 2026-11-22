"""Two video modes: the brief concept visualizer and the in-depth lecture."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src import cli
from src.cli import pdf2manim
from src.pipeline_v2 import PipelineV2, PipelineV2Config, RunStatus
from src.pipeline_v2.generation import _lesson_brief
from src.pipeline_v2.models import (
    DEEP_MAX_COST_USD,
    MAX_V2_SCENES,
    LessonPlan,
    LessonScene,
    NarratedScene,
    NarrationBeat,
    RunResult,
    SourceDocument,
    SourceFact,
)
from src.pipeline_v2.planning import (
    DEEP_LESSON_OUTPUT_TOKENS,
    DEEP_NARRATION_OUTPUT_TOKENS,
    PlanningService,
    _normalise_spine,
    deep_scene_words,
)
from src.pipeline_v2.prompt_text import MODE_LABELS
from src.pipeline_v2.providers import ProviderResponse
from src.pipeline_v2.script_review import (
    DEEP_REVIEW_OUTPUT_TOKENS,
    DEEP_SOURCE_SUMMARY_CHARS,
    review_script,
)
from src.prompts.review import storyboard_review_prompt

FIXTURE = Path(__file__).parent / "fixtures" / "test_trig_3pages.pdf"


def _source(pages: int = 12) -> SourceDocument:
    return SourceDocument(
        filename="notes.pdf", sha256="a", page_count=pages,
        facts=[SourceFact(fact_id=f"p{page:03d}-f000", page_number=page, text=f"Fact {page}. " * 400)
               for page in range(1, pages + 1)],
    )


def _lesson(scenes: int = 2) -> LessonPlan:
    return LessonPlan.model_validate({
        "title": "T", "audience": "undergraduate", "objectives": ["o"],
        "central_question": "Why?", "hook": "", "takeaway": "Because.",
        "scenes": [
            {"scene_id": f"scene_{i:03d}", "title": "S", "description": "d", "learning_goal": "g",
             "source_fact_ids": [f"p{i:03d}-f000"], "chapter": "  First   chapter ", "role": " Derive"}
            for i in range(1, scenes + 1)
        ],
    })


class _Router:
    strict_keys = False

    def __init__(self):
        self.calls: list[dict] = []
        self.artifacts = SimpleNamespace(event=lambda *args, **kwargs: None)

    async def structured(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["fallback"]()


# ── settings ────────────────────────────────────────────────────────────────


def test_deep_mode_gets_more_scenes_budget_and_render_time_unless_set():
    brief = PipelineV2Config()
    assert (brief.mode, brief.max_scenes, brief.max_cost_usd) == ("brief", 12, 4.0)
    deep = PipelineV2Config(mode="deep")
    assert (deep.max_scenes, deep.max_cost_usd) == (MAX_V2_SCENES, DEEP_MAX_COST_USD)
    assert deep.preview_timeout_seconds > brief.preview_timeout_seconds
    assert deep.render_timeout_seconds > brief.render_timeout_seconds
    chosen = PipelineV2Config(mode="deep", max_scenes=10, max_cost_usd=6.0)
    assert (chosen.max_scenes, chosen.max_cost_usd) == (10, 6.0)
    resumed = PipelineV2Config.model_validate(json.loads(deep.model_dump_json()))
    assert (resumed.mode, resumed.max_scenes) == ("deep", MAX_V2_SCENES)
    assert PipelineV2Config.model_validate({}).mode == "brief"  # old run.json


def _convert(tmp_path, *args: str, input_text=None):
    pipeline = MagicMock()
    pipeline.run.return_value = RunResult(run_id="t", run_directory=str(tmp_path), status=RunStatus.SUCCESS)
    with patch("src.cli.PipelineV2", return_value=pipeline) as factory:
        result = CliRunner().invoke(pdf2manim, ["convert", str(FIXTURE), *args], input=input_text)
    assert result.exit_code == 0, result.output
    return factory.call_args.args[0], result.output


def test_cli_mode_option_and_labels(tmp_path):
    config, output = _convert(tmp_path)
    assert config.mode == "brief" and config.max_cost_usd == 4.0 and config.max_scenes == 12
    assert f"Mode:        {MODE_LABELS['brief']}" in output
    config, output = _convert(tmp_path, "--mode", "deep")
    assert (config.mode, config.max_scenes, config.max_cost_usd) == ("deep", MAX_V2_SCENES, DEEP_MAX_COST_USD)
    assert f"Mode:        {MODE_LABELS['deep']}" in output and "API budget:  $15.00" in output
    config, _ = _convert(tmp_path, "--mode", "deep", "--max-cost", "7", "--max-scenes", "9")
    assert (config.max_cost_usd, config.max_scenes) == (7.0, 9)


def test_cli_asks_for_the_mode_at_the_start_in_a_terminal(monkeypatch):
    monkeypatch.setattr(cli.click, "get_text_stream", lambda name: SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(cli.click, "prompt", lambda *args, **kwargs: 2)
    assert cli._choose_mode(None) == "deep"
    assert cli._choose_mode("brief") == "brief"
    monkeypatch.setattr(cli.click, "get_text_stream", lambda name: SimpleNamespace(isatty=lambda: False))
    assert cli._choose_mode(None) == "brief"


# ── planning ────────────────────────────────────────────────────────────────


def test_deep_lesson_prompt_asks_for_the_whole_document_with_more_room():
    router = _Router()
    planner = PlanningService(router, max_scenes=24, mode="deep", presenter_opens=True)
    asyncio.run(planner.lesson(_source(12)))
    call = router.calls[0]
    prompt = call["prompt"]
    assert "in-depth animated lecture" in prompt and "between 6 and 24 scenes" in prompt
    assert '"role"' in prompt and '"chapter"' in prompt and "derive" in prompt
    assert "Prerequisites the document assumes are not scenes" in prompt
    assert "Full-coverage mode" not in prompt
    assert call["max_output_tokens"] == DEEP_LESSON_OUTPUT_TOKENS


def test_brief_lesson_prompt_is_the_short_explainer():
    router = _Router()
    asyncio.run(PlanningService(router, presenter_opens=True).lesson(_source(4)))
    prompt = router.calls[0]["prompt"]
    assert "short animated explainer" in prompt and "in-depth" not in prompt


def test_deep_narration_prompt_has_lecture_rules_and_word_range():
    router = _Router()
    lesson = _lesson(scenes=2)
    asyncio.run(PlanningService(router, mode="deep").narration(lesson, _source(2)))
    call = next(item for item in router.calls if item["stage"] == "narration")
    low, high = deep_scene_words(2)
    assert f"Each scene's narration is {low} to {high} words" in call["prompt"]
    assert "IN-DEPTH LECTURE" in call["prompt"] and call["max_output_tokens"] == DEEP_NARRATION_OUTPUT_TOKENS
    assert deep_scene_words(24)[1] < deep_scene_words(6)[1]  # the whole lecture stays bounded


def test_deep_mode_never_falls_back_to_a_rule_based_plan_when_a_key_is_set(monkeypatch):
    planner = PlanningService(_Router(), mode="deep")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert planner._lesson_fallback(_source(2)).scenes  # offline: the fallback is all there is
    monkeypatch.setenv("GEMINI_API_KEY", "set")
    with pytest.raises(RuntimeError, match="in-depth lecture's lesson plan"):
        planner._lesson_fallback(_source(2))
    assert PlanningService(_Router())._lesson_fallback(_source(2)).scenes  # brief keeps falling back


def test_scene_roles_and_chapters_are_tidied():
    lesson = _normalise_spine(_lesson(), _source(2))
    assert {(scene.role, scene.chapter) for scene in lesson.scenes} == {("derive", "First chapter")}
    odd = _lesson().model_copy(update={"scenes": [
        _lesson().scenes[0].model_copy(update={"role": "lecture"})
    ]})
    assert _normalise_spine(odd, _source(2)).scenes[0].role == ""


def test_code_generation_sees_the_scene_role():
    scene = NarratedScene(scene_id="s", narration="x", beats=[
        NarrationBeat(beat_id="b", text="x", source_fact_ids=["f"])])
    lesson_scene = LessonScene(scene_id="s", title="t", description="d", learning_goal="g",
                               source_fact_ids=["f"], role="apply", chapter="Two")
    assert _lesson_brief(lesson_scene, scene)["role"] == "apply"
    plain = lesson_scene.model_copy(update={"role": "", "chapter": ""})
    assert "role" not in _lesson_brief(plain, scene)


# ── review ──────────────────────────────────────────────────────────────────


def test_deep_review_judges_a_lecture_with_more_of_the_source():
    assert "IN-DEPTH LECTURE" in storyboard_review_prompt({}, [], {}, mode="deep")
    assert "IN-DEPTH LECTURE" not in storyboard_review_prompt({}, [], {})
    calls = []

    class Router:
        artifacts = SimpleNamespace(event=lambda *args, **kwargs: None)

        async def text(self, **kwargs):
            calls.append(kwargs)
            return ProviderResponse(text="{}", provider="google", model="m", input_units=1,
                                    output_units=1, exact_usage=True, raw={})

    lesson = _normalise_spine(_lesson(), _source(2))
    narration = PlanningService(_Router())._fallback_narration(lesson, _source(2))
    asyncio.run(review_script(Router(), lesson, narration, _source(12), mode="deep"))
    asyncio.run(review_script(Router(), lesson, narration, _source(12)))
    deep, brief = calls
    assert deep["max_output_tokens"] == DEEP_REVIEW_OUTPUT_TOKENS and brief["max_output_tokens"] == 3000
    assert len(deep["prompt"]) - len(brief["prompt"]) > DEEP_SOURCE_SUMMARY_CHARS // 2


# ── end to end, offline ─────────────────────────────────────────────────────


def test_a_deep_dry_run_is_labelled_and_keyed_by_mode(monkeypatch, tmp_path):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    config = PipelineV2Config(output_dir=tmp_path, mode="deep", dry_run=True,
                              enable_tts=False, enable_render=False, repair="off")

    result = PipelineV2(config).run(FIXTURE)

    assert result.status == RunStatus.SUCCESS, result.message
    run_dir = Path(result.run_directory)
    assert json.loads((run_dir / "run.json").read_text())["config"]["mode"] == "deep"
    report = (run_dir / "report.html").read_text(encoding="utf-8")
    assert f"Mode: <strong>{MODE_LABELS['deep']}" in report
