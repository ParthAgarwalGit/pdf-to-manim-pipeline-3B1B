"""Guardrails for a logical, followable script: the hook is a chain the
listener can follow, and the script review gates on sentences that do not
follow from the one before."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.pipeline_v2 import PipelineV2
from src.pipeline_v2.models import LessonPlan, NarrationPlan, SourceDocument, SourceFact
from src.pipeline_v2.observability import RunArtifacts
from src.pipeline_v2.planning import (
    BEAT_FIX_PROMPT_VERSION,
    HOOK_PROMPT_VERSION,
    PlanningService,
    hook_problems,
)
from src.pipeline_v2.prompt_text import (
    HOOK_MAX_WORDS_PRESENTER,
    BEAT_FIX_MAX_EXTRA_WORDS,
    CODEGEN_STYLE_RULES,
    HOOK_CHAIN,
    beat_fix_instructions,
    hook_check_instructions,
    lesson_instructions,
    narration_instructions,
)
from src.pipeline_v2.report import _script_review_html
from src.pipeline_v2.script_review import locate_flow_breaks, parse_review, splice_beats
from src.pipeline_v2.state import RunStore
from src.prompts.review import PLANNED_CRITERIA_WEIGHTS, storyboard_review_prompt

# The hook that shipped in the logistic-regression video: it never says what
# is being predicted, and its claim about 5.5 hours is false for the data.
BROKEN_HOOK = (
    "Suppose eight students take an exam and we only record whether they passed or "
    "failed—ones and zeros. If a new student studies for 5.5 hours, a straight line breaks "
    "probability rules. How can we fit a smooth probability curve to binary 0-and-1 outcomes "
    "without our predictions ever shooting below 0 or above 1?"
)
CHAIN_HOOK = (
    "Put a square on each side of a three, four, five triangle. The small squares hold nine "
    "and sixteen tiles. The big one holds twenty five. Why do they always fill it?"
)


def _source() -> SourceDocument:
    return SourceDocument(
        filename="notes.pdf", sha256="a", page_count=2,
        facts=[
            SourceFact(fact_id=f"p{page:03d}-f000", page_number=page, text=f"Fact on page {page}.")
            for page in (1, 2)
        ],
    )


def _lesson(hook: str = "Two squares, one big square, and they always match. Why?") -> LessonPlan:
    return LessonPlan.model_validate({
        "title": "Right triangles",
        "audience": "undergraduate",
        "objectives": ["See why a^2 + b^2 = c^2"],
        "central_question": "Why do two small squares always fill the big one exactly?",
        "hook": hook,
        "takeaway": "Rearranging four triangles leaves the same empty area both ways.",
        "running_example": {"description": "A 3-4-5 right triangle",
                            "values": {"a": "3", "b": "4", "c": "5"}},
        "scenes": [
            {"scene_id": "scene_001", "title": "Squares", "description": "Build squares",
             "learning_goal": "Areas add up", "source_fact_ids": ["p001-f000"],
             "viewer_question": "Do the areas really add up?"},
            {"scene_id": "scene_002", "title": "Rearrange", "description": "Move pieces",
             "learning_goal": "The big square is fixed", "source_fact_ids": ["p002-f000"]},
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


def _pipeline(tmp_path, script_review: str = "revise") -> PipelineV2:
    pipeline = PipelineV2.__new__(PipelineV2)
    pipeline.store = RunStore(tmp_path / "run.db", 10.0)
    pipeline.artifacts = RunArtifacts(tmp_path / "run", "run")
    pipeline.config = SimpleNamespace(script_review=script_review)
    return pipeline


def _events(pipeline: PipelineV2, name: str) -> list[dict]:
    lines = pipeline.artifacts.events_path.read_text(encoding="utf-8").splitlines()
    return [record for record in map(json.loads, lines) if record["event"] == name]


class _Planner:
    presenter_opens = True

    def __init__(self, new_hook: str, flow_problems=()):
        self.new_hook = new_hook
        self.flow_problems = list(flow_problems)
        self.checked = 0
        self.problems: list[str] = []

    async def check_hook(self, lesson):
        self.checked += 1
        return list(self.flow_problems)

    async def rewrite_hook(self, lesson, problems):
        self.problems = list(problems)
        return self.new_hook


def _review(**extra) -> dict:
    data = {name: {"score": 9, "feedback": "ok"} for name in PLANNED_CRITERIA_WEIGHTS}
    data["source_coverage"] = {"score": 8}
    data.update(extra)
    return data


# ── the hook's checkable shape ───────────────────────────────────────────────


def test_hook_problems_catch_the_shipped_hook_and_pass_a_chain():
    problems = hook_problems(BROKEN_HOOK, presenter_opens=True)
    assert any("53 words" in problem and f"limit is {HOOK_MAX_WORDS_PRESENTER}" in problem for problem in problems)
    assert any("dash" in problem for problem in problems)
    assert hook_problems(CHAIN_HOOK, presenter_opens=True) == []
    assert hook_problems("", presenter_opens=True) == []


def test_hook_problems_shape_rules():
    two_questions = "We have a triangle. Its squares look odd. Why? And how?"
    assert any("exactly one question" in p for p in hook_problems(two_questions, True))
    no_question = "We have a triangle. Its squares look odd. They match."
    assert any("exactly one question" in p for p in hook_problems(no_question, True))
    too_short = "Two squares always match the big one. Why?"
    assert any("at least three sentences" in p for p in hook_problems(too_short, True))
    words_42 = "One two three four five six seven eight nine ten. " * 4 + "Why not?"
    assert any(f"limit is {HOOK_MAX_WORDS_PRESENTER}" in p for p in hook_problems(words_42, presenter_opens=True))
    assert not any("limit" in p for p in hook_problems(words_42, presenter_opens=False))


# ── prompts ──────────────────────────────────────────────────────────────────


def test_prompts_carry_the_flow_rules():
    for presenter_opens in (True, False):
        lesson_text = lesson_instructions("Use 4 scenes.", presenter_opens)
        assert HOOK_CHAIN in lesson_text
        assert "Read the hook aloud as someone who has never seen the document" in lesson_text
        narration_text = narration_instructions("Pace.", 135, presenter_opens)
        assert "FLOW (the listener hears each sentence once" in narration_text
        assert "Introduce before you use" in narration_text
        assert "a sentence says at most two numbers" in narration_text
    assert "never hold the frame unchanged for more than about five seconds" in CODEGEN_STYLE_RULES
    review_text = storyboard_review_prompt({}, [], {})
    assert "logical_flow" in review_text and '"flow_breaks"' in review_text


def test_narration_without_a_target_length_gets_a_scene_word_range():
    router = _Router()
    asyncio.run(PlanningService(router).narration(_lesson(), _source()))
    prompt = next(call["prompt"] for call in router.calls if call["stage"] == "narration")
    assert "Each scene's narration is 80 to 130 words" in prompt


def test_rewrite_hook_prompt_lists_the_problems_and_falls_back_to_the_old_hook():
    router = _Router()
    planner = PlanningService(router, presenter_opens=True)
    lesson = _lesson(BROKEN_HOOK)
    problems = hook_problems(BROKEN_HOOK, True)

    hook = asyncio.run(planner.rewrite_hook(lesson, problems))

    assert hook == " ".join(BROKEN_HOOK.split())  # keyless: the fallback keeps the old hook
    (call,) = router.calls
    assert (call["stage"], call["task_id"]) == ("lesson", "hook-rewrite")
    assert call["prompt"].startswith(f"PROMPT_VERSION={HOOK_PROMPT_VERSION}\n")
    for problem in problems:
        assert f"- {problem}" in call["prompt"]
    assert HOOK_CHAIN in call["prompt"]
    assert '"current_hook"' in call["prompt"] and '"running_example"' in call["prompt"]


# ── the review gate ──────────────────────────────────────────────────────────


def test_a_scene_flow_break_fails_review_and_becomes_a_fix():
    review = parse_review(_review(flow_breaks=[
        {"where": "scene_002", "sentence": "So the slope is beta one.",
         "problem": "Beta one was never introduced."},
        "not an object", {"where": "scene_001", "sentence": "  "},
    ]))
    assert review.overall_score >= 6.5
    assert review.passed is False
    assert review.flow_breaks == [{
        "where": "scene_002", "sentence": "So the slope is beta one.",
        "problem": "Beta one was never introduced.",
    }]
    assert any(
        'In scene_002, the sentence "So the slope is beta one."' in item
        and "Beta one was never introduced." in item
        for item in review.improvement_suggestions
    )


def test_low_flow_score_fails_review_even_with_a_high_overall():
    data = _review()
    data["logical_flow"] = {"score": 5.5}
    review = parse_review(data)
    assert review.overall_score >= 6.5 and review.passed is False


def test_a_hook_only_break_passes_the_narration_but_is_reported():
    review = parse_review(_review(flow_breaks=[
        {"where": "Hook", "sentence": "A straight line breaks probability rules.",
         "problem": "Nothing has said what is being predicted."},
    ]))
    assert review.passed is True
    assert review.hook_breaks() == review.flow_breaks
    assert not any("A straight line" in item for item in review.improvement_suggestions)


# ── the pipeline acts on them ────────────────────────────────────────────────


def test_rewrite_hook_keeps_a_rewrite_that_fixes_the_problems(tmp_path):
    pipeline = _pipeline(tmp_path)
    planner = _Planner(CHAIN_HOOK)
    lesson = _lesson(BROKEN_HOOK)
    problems = hook_problems(BROKEN_HOOK, True)

    updated = asyncio.run(pipeline._rewrite_hook(planner, lesson, problems, "hook-rewrite"))

    assert updated.hook == CHAIN_HOOK
    assert planner.problems == problems
    (event,) = _events(pipeline, "hook_rewritten")
    assert event["before"] == BROKEN_HOOK and event["after"] == CHAIN_HOOK
    pipeline.store.close()


def test_rewrite_hook_rejects_a_rewrite_that_fails_more_checks(tmp_path):
    pipeline = _pipeline(tmp_path)
    worse = BROKEN_HOOK + " And why does it matter (really)?"
    planner = _Planner(worse)
    lesson = _lesson(CHAIN_HOOK)

    updated = asyncio.run(pipeline._rewrite_hook(
        planner, lesson, ["The sentence does not follow."], "hook-rewrite-review",
    ))

    assert updated.hook == CHAIN_HOOK
    assert len(_events(pipeline, "hook_rewrite_rejected")) == 1
    pipeline.store.close()


# A hook with a fine shape whose middle sentence does not follow.
JUMPING_HOOK = (
    "Eight students take an exam. A straight line breaks probability rules at five and a half "
    "hours. How do we fit a curve?"
)
FLOW_PROBLEM = (
    'The sentence "A straight line breaks probability rules at five and a half hours." does not '
    "follow: nothing says what is predicted, and the line gives about 0.64 there."
)


def test_check_hook_prompt_and_keyless_fallback():
    router = _Router()
    planner = PlanningService(router, presenter_opens=True)

    assert asyncio.run(planner.check_hook(_lesson(JUMPING_HOOK))) == []

    (call,) = router.calls
    assert (call["stage"], call["task_id"]) == ("lesson", "hook-check")
    assert hook_check_instructions() in call["prompt"]
    assert JUMPING_HOOK in call["prompt"] and '"values"' in call["prompt"]
    assert asyncio.run(planner.check_hook(_lesson(""))) == []
    assert len(router.calls) == 1  # an empty hook is not sent


def test_settle_hook_rewrites_a_hook_whose_sentences_do_not_follow(tmp_path):
    pipeline = _pipeline(tmp_path)
    assert hook_problems(JUMPING_HOOK, True) == []
    planner = _Planner(CHAIN_HOOK, flow_problems=[FLOW_PROBLEM])

    settled = asyncio.run(pipeline._settle_hook(planner, _lesson(JUMPING_HOOK)))

    assert settled.hook == CHAIN_HOOK
    assert planner.checked == 1 and planner.problems == [FLOW_PROBLEM]
    (checked,) = _events(pipeline, "hook_checked")
    assert checked["problems"] == [FLOW_PROBLEM]
    assert len(_events(pipeline, "hook_rewritten")) == 1
    pipeline.store.close()


def test_settle_hook_keeps_a_hook_that_passes_every_check(tmp_path):
    pipeline = _pipeline(tmp_path)
    planner = _Planner("unused")

    settled = asyncio.run(pipeline._settle_hook(planner, _lesson(CHAIN_HOOK)))

    assert settled.hook == CHAIN_HOOK and planner.checked == 1 and planner.problems == []
    pipeline.store.close()


def test_settle_hook_in_report_mode_logs_flow_but_does_not_rewrite(tmp_path):
    pipeline = _pipeline(tmp_path, script_review="report")
    planner = _Planner(CHAIN_HOOK, flow_problems=[FLOW_PROBLEM])

    settled = asyncio.run(pipeline._settle_hook(planner, _lesson(JUMPING_HOOK)))

    assert settled.hook == JUMPING_HOOK and planner.problems == []
    assert _events(pipeline, "hook_checked")[0]["problems"] == [FLOW_PROBLEM]
    pipeline.store.close()


def test_settle_hook_with_review_off_still_enforces_the_shape(tmp_path):
    pipeline = _pipeline(tmp_path, script_review="off")
    planner = _Planner(CHAIN_HOOK, flow_problems=[FLOW_PROBLEM])

    settled = asyncio.run(pipeline._settle_hook(planner, _lesson(BROKEN_HOOK)))

    assert planner.checked == 0  # no model check with the review off
    assert planner.problems == hook_problems(BROKEN_HOOK, True)
    assert settled.hook == CHAIN_HOOK
    pipeline.store.close()


def test_review_reports_hook_breaks_and_never_rewrites_the_hook(monkeypatch, tmp_path):
    pipeline = _pipeline(tmp_path)
    lesson = _lesson()
    narration = PlanningService(_Router())._fallback_narration(lesson, _source())

    async def fake_review(router, plan, candidate, source, attempt=1):
        return parse_review(_review(flow_breaks=[{
            "where": "hook", "sentence": "Two squares, one big square, and they always match.",
            "problem": "Which squares? Nothing has been drawn or named yet.",
        }]), attempt)

    monkeypatch.setattr("src.pipeline_v2.pipeline.review_script", fake_review)
    planner = _Planner(CHAIN_HOOK)

    kept, checks = asyncio.run(pipeline._review_script(
        None, planner, lesson, narration, _source(), {},
    ))

    assert kept is narration and checks is None
    assert planner.problems == []
    (reported,) = _events(pipeline, "hook_flow_reported")
    assert reported["breaks"][0]["where"] == "hook"
    pipeline.store.close()


LONG_LOGICAL_HOOK = (
    "We record study hours for eight students, and whether each one fails at zero or passes at "
    "one. To predict the chance of passing, we might fit a straight line through those points. "
    "However, that line keeps going below zero and above one. How can we fit a curve that "
    "stays between zero and one?"
)


class _SequencePlanner(_Planner):
    """Answers each rewrite request with the next hook in ``hooks``."""

    def __init__(self, hooks, flow_problems=()):
        super().__init__("", flow_problems)
        self.hooks = list(hooks)
        self.requests: list[tuple[str, list[str]]] = []

    async def rewrite_hook(self, lesson, problems):
        self.requests.append((lesson.hook, list(problems)))
        return self.hooks[len(self.requests) - 1]


def test_a_rewrite_that_misses_the_shape_gets_one_more_request(tmp_path):
    pipeline = _pipeline(tmp_path)
    long_problems = hook_problems(LONG_LOGICAL_HOOK, True)
    assert long_problems and all(f"limit is {HOOK_MAX_WORDS_PRESENTER}" in p for p in long_problems)
    planner = _SequencePlanner([LONG_LOGICAL_HOOK, CHAIN_HOOK], flow_problems=[FLOW_PROBLEM])

    settled = asyncio.run(pipeline._settle_hook(planner, _lesson(JUMPING_HOOK)))

    assert settled.hook == CHAIN_HOOK
    assert planner.requests == [
        (JUMPING_HOOK, [FLOW_PROBLEM]),
        (LONG_LOGICAL_HOOK, long_problems),  # the second request fixes only the shape
    ]
    (event,) = _events(pipeline, "hook_rewritten")
    assert [item["hook"] for item in event["attempts"]] == [LONG_LOGICAL_HOOK, CHAIN_HOOK]
    pipeline.store.close()


def test_a_hook_that_does_not_follow_is_replaced_even_by_a_long_rewrite(tmp_path):
    pipeline = _pipeline(tmp_path)
    planner = _SequencePlanner([LONG_LOGICAL_HOOK, LONG_LOGICAL_HOOK], flow_problems=[FLOW_PROBLEM])

    settled = asyncio.run(pipeline._settle_hook(planner, _lesson(JUMPING_HOOK)))

    assert settled.hook == LONG_LOGICAL_HOOK
    assert len(planner.requests) == 2
    pipeline.store.close()


def test_a_shape_only_rewrite_must_not_fail_more_checks(tmp_path):
    pipeline = _pipeline(tmp_path, script_review="off")
    too_short = "Two squares always match the big one. Why?"  # one shape problem
    worse = "Two squares (and one big square) always match. Why? How?"
    planner = _SequencePlanner([worse, worse])

    settled = asyncio.run(pipeline._settle_hook(planner, _lesson(too_short)))

    assert settled.hook == too_short
    assert len(_events(pipeline, "hook_rewrite_rejected")) == 1
    pipeline.store.close()


# ── targeted fixes of the sentences the review flags ─────────────────────────


OPENING = "Put a square on each side of a three, four, five triangle."
BROKEN = "So the slope is beta one."
FIXED_BEAT = (
    "The steepness of the slid edge, which we call beta one, is the slope. "
    "The empty area stays the same."
)


def _narration() -> NarrationPlan:
    scenes = {
        "scene_001": [
            ("scene_001_b01", OPENING, "Grow three squares."),
            ("scene_001_b02", "The small squares hold nine and sixteen tiles.", "Fill tiles."),
        ],
        "scene_002": [
            ("scene_002_b01", "Slide the four triangles into the corners.", "Slide triangles."),
            ("scene_002_b02", f"{BROKEN} The empty area stays the same.", "Shade the gap."),
        ],
    }
    return NarrationPlan.model_validate({"scenes": [
        {
            "scene_id": scene_id,
            "narration": " ".join(text for _, text, _ in beats),
            "beats": [
                {"beat_id": beat_id, "text": text, "visual_action": action,
                 "source_fact_ids": [f"p00{scene_id[-1]}-f000"]}
                for beat_id, text, action in beats
            ],
        }
        for scene_id, beats in scenes.items()
    ]})


def _scene_break(sentence: str = BROKEN, where: str = "scene_002") -> dict:
    return {"where": where, "sentence": sentence, "problem": "Beta one was never introduced."}


def test_locate_flow_breaks_groups_breaks_by_beat_in_narration_order():
    narration = _narration()
    placed, unplaced = locate_flow_breaks(narration, [
        _scene_break("  so the SLOPE   is beta one "),  # case, spacing, no full stop
        _scene_break("The empty area stays the same."),  # the same beat again
        _scene_break("The small squares hold nine and sixteen tiles.", where="scene_009"),
        _scene_break("A sentence the script never says."),
    ])

    assert list(placed) == [("scene_001", "scene_001_b02"), ("scene_002", "scene_002_b02")]
    assert len(placed[("scene_002", "scene_002_b02")]) == 2  # one fix request carries both
    assert [item["sentence"] for item in unplaced] == ["A sentence the script never says."]


def test_splice_beats_replaces_one_beat_and_leaves_the_rest_untouched():
    narration = _narration()
    before = narration.model_dump(mode="json")

    patched = splice_beats(narration, {("scene_002", "scene_002_b02"): FIXED_BEAT})

    assert narration.model_dump(mode="json") == before  # the original is not changed
    assert patched.scenes[0] is narration.scenes[0]
    assert patched.scenes[1].beats[0] is narration.scenes[1].beats[0]
    fixed, old = patched.scenes[1].beats[1], narration.scenes[1].beats[1]
    assert fixed.text == FIXED_BEAT
    assert fixed.model_dump(exclude={"text"}) == old.model_dump(exclude={"text"})
    assert patched.scenes[1].narration == (
        f"Slide the four triangles into the corners. {FIXED_BEAT}"
    )


def test_fix_beat_prompt_carries_the_scene_and_falls_back_to_the_old_text():
    router = _Router()
    narration = _narration()
    problem = f'The sentence "{BROKEN}" does not follow: Beta one was never introduced.'

    text = asyncio.run(PlanningService(router).fix_beat(
        narration.scenes[1], "scene_002_b02", narration.scenes[0].beats[-1].text,
        [problem], _source(),
    ))

    assert text == narration.scenes[1].beats[1].text  # keyless: the fallback keeps the beat
    (call,) = router.calls
    assert (call["stage"], call["task_id"], call["scene_id"]) == (
        "narration", "beat-fix-scene_002_b02", "scene_002",
    )
    prompt = call["prompt"]
    assert prompt.startswith(f"PROMPT_VERSION={BEAT_FIX_PROMPT_VERSION}\n")
    assert beat_fix_instructions([problem]) in prompt and f"- {problem}" in prompt
    assert f"at most {BEAT_FIX_MAX_EXTRA_WORDS} words longer" in prompt
    payload = json.loads(prompt[prompt.index("\n\n{") + 2:])
    assert payload["beat_to_fix"] == {"beat_id": "scene_002_b02", "text": f"{BROKEN} The empty area stays the same."}
    assert [beat["beat_id"] for beat in payload["scene_beats"]] == ["scene_002_b01", "scene_002_b02"]
    assert payload["previous_scene_last_beat"] == "The small squares hold nine and sixteen tiles."
    assert payload["source_facts"] == {"p002-f000": "Fact on page 2."}


class _FixPlanner:
    """Answers beat fixes from ``fixes``; records whole rewrites."""

    presenter_opens = True
    omitted_concepts: list = []

    def __init__(self, fixes=None, rewrite=None):
        self.fixes = dict(fixes or {})
        self.rewrite = rewrite
        self.fix_calls: list[tuple] = []
        self.rewrites: list[list[str]] = []
        self.length_budget = self.length_report = None
        self.compliance_checks: list = []

    async def fix_beat(self, scene, beat_id, previous_beat, problems, source):
        self.fix_calls.append((scene.scene_id, beat_id, previous_beat, list(problems)))
        return self.fixes.get(beat_id, next(b.text for b in scene.beats if b.beat_id == beat_id))

    async def narration(self, lesson, source, review_feedback=None):
        self.rewrites.append(list(review_feedback or []))
        return self.rewrite


def _review_pipeline(tmp_path) -> PipelineV2:
    pipeline = _pipeline(tmp_path)
    pipeline.config = SimpleNamespace(script_review="revise", pacing="normal")
    pipeline.directives = None
    return pipeline


def _scripted_reviews(monkeypatch, reviews: list[dict]) -> list[NarrationPlan]:
    """Answer each script review with the next scripted reviewer JSON; record what was read."""

    seen: list[NarrationPlan] = []

    async def fake_review(router, plan, candidate, source, attempt=1):
        seen.append(candidate)
        return parse_review(reviews[len(seen) - 1], attempt)

    monkeypatch.setattr("src.pipeline_v2.pipeline.review_script", fake_review)
    return seen


def _scored(score: float, **extra) -> dict:
    data = {name: {"score": score} for name in PLANNED_CRITERIA_WEIGHTS}
    data.update(extra)
    return data


@pytest.mark.parametrize(
    ("second", "kept_fix"),
    [
        (_scored(8.0), True),  # passes
        (_scored(8.0, flow_breaks=[_scene_break("The empty area stays the same.")]), True),
        (_scored(6.0), False),  # fails and scores lower
    ],
    ids=["passes", "fails-but-scores-higher", "fails-and-scores-lower"],
)
def test_a_flow_break_gets_only_its_beat_rewritten(monkeypatch, tmp_path, second, kept_fix):
    pipeline = _review_pipeline(tmp_path)
    narration = _narration()
    seen = _scripted_reviews(monkeypatch, [_scored(7.0, flow_breaks=[_scene_break()]), second])
    planner = _FixPlanner(fixes={"scene_002_b02": FIXED_BEAT})

    kept, checks = asyncio.run(pipeline._review_script(
        None, planner, _lesson(), narration, _source(), {},
    ))

    assert planner.rewrites == []  # never the whole script
    ((scene_id, beat_id, previous, problems),) = planner.fix_calls
    assert (scene_id, beat_id) == ("scene_002", "scene_002_b02")
    assert previous == "The small squares hold nine and sixteen tiles."
    assert problems == [f'The sentence "{BROKEN}" does not follow: Beta one was never introduced.']
    # The re-review read the fixed script, with the scene narration rebuilt.
    assert len(seen) == 2 and BROKEN not in seen[1].scenes[1].narration
    assert FIXED_BEAT in seen[1].scenes[1].narration
    attempts = json.loads((pipeline.artifacts.run_dir / "script_review.json").read_text())
    assert [item["attempt"] for item in attempts] == [1, 2]
    (reviewed,) = _events(pipeline, "script_reviewed")
    assert reviewed["fix"] == "beats" and reviewed["rewritten"] is kept_fix
    if kept_fix:
        assert checks is not None
        assert kept.scenes[1].beats[1].text == FIXED_BEAT
        assert kept.scenes[0] == narration.scenes[0]
        assert kept.scenes[1].beats[0] == narration.scenes[1].beats[0]
    else:
        assert kept is narration and checks is None
    pipeline.store.close()


@pytest.mark.parametrize(
    "fix",
    [
        f"{BROKEN} The empty area stays the same.",  # unchanged
        FIXED_BEAT + " word" * (BEAT_FIX_MAX_EXTRA_WORDS + 6),  # far longer than allowed
    ],
    ids=["unchanged", "too-long"],
)
def test_a_dropped_beat_fix_keeps_the_script_without_a_second_review(monkeypatch, tmp_path, fix):
    pipeline = _review_pipeline(tmp_path)
    narration = _narration()
    seen = _scripted_reviews(monkeypatch, [_scored(7.0, flow_breaks=[_scene_break()])])
    planner = _FixPlanner(fixes={"scene_002_b02": fix})

    kept, checks = asyncio.run(pipeline._review_script(
        None, planner, _lesson(), narration, _source(), {},
    ))

    assert kept is narration and checks is None
    assert len(seen) == 1 and planner.rewrites == []
    assert len(_events(pipeline, "beat_fix_rejected")) == 1
    assert len(_events(pipeline, "script_review_fix_skipped")) == 1
    pipeline.store.close()


def test_the_beat_that_speaks_the_hook_is_never_rewritten(monkeypatch, tmp_path):
    pipeline = _review_pipeline(tmp_path)
    narration = _narration()
    seen = _scripted_reviews(monkeypatch, [
        _scored(7.0, flow_breaks=[_scene_break(OPENING, where="scene_001")]),
    ])
    planner = _FixPlanner(fixes={"scene_001_b01": "A different opening line."})

    kept, checks = asyncio.run(pipeline._review_script(
        None, planner, _lesson(hook=OPENING), narration, _source(), {},
    ))

    assert kept is narration and checks is None
    assert planner.fix_calls == [] and len(seen) == 1
    assert len(_events(pipeline, "beat_fix_skipped")) == 1
    pipeline.store.close()


def test_report_shows_a_kept_fix_that_passes_with_a_lower_score(tmp_path):
    first = parse_review(_scored(8.65, flow_breaks=[_scene_break()]), 1)
    second = parse_review(_scored(8.0), 2)
    assert not first.passed and second.passed
    (tmp_path / "script_review.json").write_text(json.dumps(
        [first.model_dump(mode="json"), second.model_dump(mode="json")],
    ))

    assert "Score 8.00/10 (passed, 2 review(s))" in _script_review_html(tmp_path)


def test_a_failure_without_flow_breaks_falls_back_to_the_whole_rewrite(monkeypatch, tmp_path):
    pipeline = _review_pipeline(tmp_path)
    narration = _narration()
    low_flow = _scored(7.0, improvement_suggestions=["Open on the triangle."])
    low_flow["logical_flow"] = {"score": 5.0}
    seen = _scripted_reviews(monkeypatch, [low_flow, _scored(8.0)])
    rewrite = splice_beats(narration, {("scene_001", "scene_001_b01"): "A rewritten opening."})
    planner = _FixPlanner(rewrite=rewrite)

    kept, checks = asyncio.run(pipeline._review_script(
        None, planner, _lesson(), narration, _source(), {},
    ))

    assert planner.fix_calls == []
    assert planner.rewrites == [["Open on the triangle."]]
    assert len(seen) == 2 and checks is not None
    assert kept.scenes[0].beats[0].text == "A rewritten opening."
    (reviewed,) = _events(pipeline, "script_reviewed")
    assert reviewed["fix"] == "rewrite" and reviewed["rewritten"] is True
    pipeline.store.close()
