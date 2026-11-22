"""Grounded lesson, narration, timing, and artifact-contract stages."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel

from src.prompts.narrative import spoken_plain
from src.schemas.directives import ProfessorDirectives

from .directives import (
    ComplianceCheck,
    ConceptPlan,
    LengthBudget,
    build_length_budget,
    effective_audience,
    effective_target_minutes,
    enforce_narration_directives,
    lesson_directive_block,
    narration_directive_block,
    scene_cap_for,
    trim_to_budget,
    truncate_sentences,
)
from .layout import layout_prompt_guide, regions_for
from .models import (
    ArtifactCategory,
    ArtifactSpec,
    BeatComposition,
    LessonPlan,
    LessonScene,
    LayoutRegion,
    LayoutTemplate,
    NarratedScene,
    NarrationBeat,
    NarrationPlan,
    OCRBatch,
    PageCoverage,
    SourceDocument,
    SourceFact,
    VisualContract,
    VisualContractBatch,
)
from .model_config import OCR_MODEL, TEXT_MODEL
from .prompt_text import (
    HOOK_MAX_WORDS_NARRATOR,
    HOOK_MAX_WORDS_PRESENTER,
    beat_fix_instructions,
    hook_check_instructions,
    hook_rewrite_instructions,
    lesson_instructions,
    narration_instructions,
)
from .providers import MissingProviderKey, ProviderRouter, clean_json_text

try:
    import pymupdf
except ImportError:  # pragma: no cover - compatibility with older PyMuPDF
    import fitz as pymupdf


# Prompt versions, per video mode. A planning task's input hash includes its
# mode's version, so changing one mode's prompts re-plans only that mode's runs
# on resume. Bump a mode's entry when its prompts change, and both entries when
# text the two modes share changes. "planning" covers OCR, lesson and visual
# contract prompts; "narration" covers the narration prompt.
PROMPT_VERSIONS = {
    "brief": {"planning": "v2.7.1-brief-coverage", "narration": "v2.8.1-voice"},
    "deep": {"planning": "v2.7.0-modes", "narration": "v2.8.0-voice"},
}
# Other names a mode's current prompts went by: a task whose stored hash used
# one of them is still reused. Deep runs planned while both modes shared one
# version (070d988) carry that commit's brief-only bump. Drop a mode's aliases
# when its version is bumped.
PROMPT_VERSION_ALIASES = {
    "deep": {"planning": ("v2.7.1-brief-coverage",), "narration": ("v2.8.1-voice",)},
}


def prompt_version(mode: str, kind: str) -> str:
    """The prompt version of ``kind`` ("planning" or "narration") for a video mode."""

    return PROMPT_VERSIONS[mode][kind]


def prompt_version_aliases(mode: str, kind: str) -> tuple[str, ...]:
    return PROMPT_VERSION_ALIASES.get(mode, {}).get(kind, ())


HOOK_PROMPT_VERSION = "hook-3-no-spoilers"
BEAT_FIX_PROMPT_VERSION = "beat-fix-1"
NARRATION_BATCH_SIZE = 5
MAX_SPINE_MOTIFS = 4
# OCR output budgets: first try, then one retry with room for the model's thinking.
OCR_OUTPUT_BUDGETS = (5000, 16000)
PLANNING_PROVIDER = "google"
PLANNING_MODEL = TEXT_MODEL


def extract_source(pdf_path: Path) -> SourceDocument:
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    facts: list[SourceFact] = []
    low_confidence: list[int] = []
    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            blocks = page.get_text("blocks")
            page_chars = 0
            fact_index = 0
            for block in blocks:
                text = re.sub(r"\s+", " ", str(block[4])).strip()
                if not text:
                    continue
                page_chars += len(text)
                kind = "equation" if _looks_like_equation(text) else "text"
                facts.append(SourceFact(
                    fact_id=f"p{page_number:03d}-f{fact_index:03d}",
                    page_number=page_number,
                    text=text,
                    kind=kind,
                    bbox=(float(block[0]), float(block[1]), float(block[2]), float(block[3])),
                    confidence=1.0,
                ))
                fact_index += 1
            for image_index, _image in enumerate(page.get_images(full=True)):
                facts.append(SourceFact(
                    fact_id=f"p{page_number:03d}-img{image_index:03d}",
                    page_number=page_number,
                    text=f"Embedded source image {image_index + 1} on page {page_number}",
                    kind="image",
                    confidence=0.8,
                ))
            if page_chars < 40:
                low_confidence.append(page_number)
        page_count = len(document)
    if not facts:
        facts.append(SourceFact(
            fact_id="p001-f000", page_number=1,
            text=f"Document titled {pdf_path.stem}", confidence=0.2,
        ))
    return SourceDocument(
        filename=pdf_path.name,
        sha256=digest,
        page_count=page_count,
        facts=facts,
        low_confidence_pages=low_confidence,
    )


def _looks_like_equation(text: str) -> bool:
    symbols = sum(text.count(symbol) for symbol in ("=", "+", "−", "-", "∑", "∫", "√"))
    return symbols >= 2 and len(text) < 300


class PlanningService:
    def __init__(
        self,
        router: ProviderRouter,
        max_scenes: int = 12,
        min_scenes: int | None = None,
        pacing: str = "standard",
        *,
        audience: Optional[str] = None,
        directives: Optional[ProfessorDirectives] = None,
        concept_plan: Optional[ConceptPlan] = None,
        target_minutes: Optional[float] = None,
        presenter_opens: bool = False,
        mode: str = "brief",
    ):
        if min_scenes is not None and min_scenes > max_scenes:
            raise ValueError("min_scenes cannot exceed max_scenes")
        self.router = router
        self.presenter_opens = presenter_opens
        self.mode = mode
        self.max_scenes = max_scenes
        self.min_scenes = min_scenes
        self.pacing = pacing
        self.audience = effective_audience(directives, audience)
        self.directives = directives
        self.concept_plan = concept_plan
        self.target_minutes = effective_target_minutes(directives, target_minutes)
        # Known before lesson() runs, so a resumed run that reuses the lesson
        # task still reports what the professor left out.
        self.omitted_concepts: list[str] = (
            [entry.concept for entry in concept_plan.skipped()] if concept_plan else []
        )
        # Set by narration().
        self.length_budget: Optional[LengthBudget] = None
        self.length_report: Optional[dict[str, Any]] = None
        self.compliance_checks: list[ComplianceCheck] = []

    def _scene_ceiling(self) -> int:
        if self.target_minutes is None:
            return self.max_scenes
        return min(self.max_scenes, scene_cap_for(self.target_minutes))

    def _required_min_scenes(self, source: SourceDocument) -> int:
        if self.min_scenes is not None:
            floor = self.min_scenes
        elif self.mode == "deep":
            floor = max(DEEP_MIN_SCENES, source.page_count // DEEP_PAGES_PER_SCENE)
        elif self.max_scenes > 12:
            floor = _full_coverage_scene_floor(source.page_count, self.max_scenes)
        else:
            floor = 1
        return min(floor, self._scene_ceiling())

    def _scene_guidance(
        self, lesson: LessonPlan, source: SourceDocument,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Per-scene depth and binding notes from the plan, or from the brief."""

        plan = self.concept_plan
        if plan is None and self.directives is not None:
            plan = ConceptPlan.from_lesson(lesson, source, self.directives)
        if plan is None:
            return {}, {}
        depths: dict[str, str] = {}
        notes: dict[str, str] = {}
        for scene in lesson.scenes:
            depth = plan.depth_for(scene.scene_id)
            # A scene that is still in the lesson has to be narrated.
            depths[scene.scene_id] = "brief" if depth == "skip" else depth
            if plan.notes_for(scene.scene_id):
                notes[scene.scene_id] = plan.notes_for(scene.scene_id)
        return depths, notes

    async def enrich_low_confidence_pages(
        self,
        source: SourceDocument,
        pdf_path: Path,
        artifact_dir: Path,
    ) -> SourceDocument:
        if not source.low_confidence_pages:
            return source
        artifact_dir.mkdir(parents=True, exist_ok=True)
        additions: list[SourceFact] = []
        with pymupdf.open(pdf_path) as document:
            batches = [source.low_confidence_pages[index:index + 3]
                       for index in range(0, len(source.low_confidence_pages), 3)]
            for batch_index, pages in enumerate(batches):
                image_paths = []
                for page_number in pages:
                    path = artifact_dir / f"page-{page_number:03d}.png"
                    pixmap = document[page_number - 1].get_pixmap(
                        matrix=pymupdf.Matrix(1.6, 1.6), alpha=False,
                    )
                    pixmap.save(str(path))
                    image_paths.append(path)
                async def read(read_pages: list[int], paths: list, task_id: str, budgets) -> Optional[OCRBatch]:
                    """Transcribe ``read_pages``; None when every attempt came back unreadable."""

                    prompt = (
                        f"PROMPT_VERSION={prompt_version(self.mode, 'planning')}\n"
                        "Transcribe the educational content in these PDF pages in image order. "
                        f"Their one-based page numbers are {read_pages}. Preserve equations exactly. "
                        "Return JSON only as {\"facts\":[{\"page_number\":1,\"text\":\"...\","
                        "\"kind\":\"text|equation|image|table\"}]} and do not infer unseen content."
                    )
                    for attempt, budget in enumerate(budgets, start=1):
                        response = await self.router.vision(
                            stage="ocr", task_id=task_id, scene_id="source", model=OCR_MODEL,
                            system="You are a precise OCR system for mathematical source documents.",
                            prompt=prompt, image_paths=paths, max_output_tokens=budget,
                        )
                        try:
                            return OCRBatch.model_validate_json(clean_json_text(response.text))
                        except ValueError as exc:
                            self.router.artifacts.event(
                                "ocr_batch_unreadable", stage="ocr", task_id=task_id,
                                pages=read_pages, attempt=attempt,
                                message=f"OCR answer was not valid JSON: {str(exc)[:160]}",
                            )
                    return None

                try:
                    # The model's thinking shares the output budget, so a batch of dense
                    # pages can come back empty: retry it with more room, then read its
                    # pages one at a time. A page still unreadable keeps the PDF's own
                    # text rather than failing the run.
                    batch_id = f"ocr-{batch_index:03d}"
                    parsed = await read(pages, image_paths, batch_id, OCR_OUTPUT_BUDGETS)
                    readings = [parsed] if parsed is not None else []
                    if parsed is None and len(pages) > 1:
                        for page_number, path in zip(pages, image_paths):
                            single = await read(
                                [page_number], [path], f"{batch_id}-p{page_number:03d}",
                                OCR_OUTPUT_BUDGETS[-1:],
                            )
                            if single is not None:
                                readings.append(single)
                    for reading in readings:
                        for index, fact in enumerate(reading.facts):
                            if fact.page_number not in pages:
                                continue
                            additions.append(SourceFact(
                                fact_id=f"p{fact.page_number:03d}-ocr{index:03d}",
                                page_number=fact.page_number, text=fact.text,
                                kind=fact.kind, confidence=0.92,
                            ))
                except MissingProviderKey as exc:
                    if self.router.strict_keys:
                        raise
                    self.router.artifacts.event(
                        "ocr_fallback", stage="ocr", task_id=f"ocr-{batch_index:03d}",
                        message=str(exc), pages=pages,
                    )
                    break
        if not additions:
            return source
        return source.model_copy(update={"facts": source.facts + additions})

    async def lesson(self, source: SourceDocument) -> LessonPlan:
        if self.concept_plan is not None:
            lesson, self.omitted_concepts = self.concept_plan.to_lesson(source)
            return lesson
        facts = [_lesson_fact_payload(fact) for fact in source.facts]
        max_scenes = self._scene_ceiling()
        required_min = self._required_min_scenes(source)
        if self.mode == "deep":
            scene_guidance = (
                f"This document has {source.page_count} pages. Plan between {required_min} and "
                f"{max_scenes} scenes: as many as its important content needs and no more, since a "
                "short document needs fewer. A scene is about a minute to a minute and a half of "
                "narration, so the whole lecture stays under about 25 minutes. Keep every scene "
                "record concise: a short title, a one-sentence description, learning goal and "
                "visual objective, and at most two do_not_repeat items. "
            )
        elif self.min_scenes is not None or self.max_scenes > 12:
            coverage_floor = required_min
            scene_range = (
                f"exactly {max_scenes}"
                if coverage_floor == max_scenes
                else f"between {coverage_floor} and {max_scenes}"
            )
            scene_guidance = (
                f"Full-coverage mode is requested for this {source.page_count}-page document. "
                f"Produce {scene_range} substantial scenes. "
                "Cover every substantive source page and every distinct definition, derivation, "
                "algorithm, example family, and conclusion. Every substantive page must be "
                "represented by at least one cited source_fact_id. Keep related consecutive pages "
                "together, but never merge unrelated topics merely to reduce scene count. "
                "Give each scene one primary teaching objective. Split a scene whenever it would "
                "otherwise contain multiple independent concepts, derivations, or worked examples. "
                "Keep every scene record concise: use a short title, one-sentence description, "
                "one-sentence learning goal, one-sentence visual objective, and at most two "
                "do_not_repeat items. "
            )
        else:
            scene_guidance = (
                "Prefer 4-6 substantial scenes for a short document and use at most "
                f"{max_scenes} scenes. "
            )
        prompt = (
            f"PROMPT_VERSION={prompt_version(self.mode, 'planning')}\n"
            + lesson_instructions(
                scene_guidance, self.presenter_opens, self.mode, self.audience or "undergraduate",
            ) + "\n\n"
            + json.dumps(
                {"document": source.filename, "page_count": source.page_count, "facts": facts},
                ensure_ascii=False,
            )
        )
        prompt += lesson_directive_block(
            self.directives, self.audience,
            max_scenes if self.target_minutes is not None else None,
        )
        lesson = await self.router.structured(
            stage="lesson", task_id="lesson", provider=PLANNING_PROVIDER,
            model=PLANNING_MODEL, system="You are a rigorous educational lesson director.",
            prompt=prompt, schema=LessonPlan, fallback=lambda: self._lesson_fallback(source),
            max_output_tokens=self._lesson_output_tokens(),
        )
        lesson = _normalise_spine(lesson, source)
        if len(lesson.scenes) >= required_min:
            return lesson

        artifacts = getattr(self.router, "artifacts", None)
        if artifacts is not None:
            artifacts.event(
                "lesson_scene_floor_missed", stage="lesson", task_id="lesson-expand",
                message=(
                    f"Lesson returned {len(lesson.scenes)} scenes; at least "
                    f"{required_min} were requested. Regenerating once."
                ),
                actual_scenes=len(lesson.scenes), required_scenes=required_min,
            )
        retry_prompt = (
            prompt
            + "\n\nThe previous structurally valid plan contained only "
            + f"{len(lesson.scenes)} scenes. That is below the required minimum of "
            + f"{required_min}. Regenerate the complete plan with between {required_min} and "
            + f"{max_scenes} scenes. Do not merely duplicate scenes: split dense material "
            + "along concept, derivation, and worked-example boundaries while preserving full "
            + "source coverage."
        )
        expanded = await self.router.structured(
            stage="lesson", task_id="lesson-expand", provider=PLANNING_PROVIDER,
            model=PLANNING_MODEL, system="You are a rigorous educational lesson director.",
            prompt=retry_prompt, schema=LessonPlan,
            fallback=lambda: self._lesson_fallback(source),
            max_output_tokens=self._lesson_output_tokens(),
        )
        expanded = _normalise_spine(expanded, source)
        if len(expanded.scenes) < required_min:
            raise RuntimeError(
                f"Lesson planner returned {len(expanded.scenes)} scenes after regeneration; "
                f"at least {required_min} are required."
            )
        return expanded

    def _narration_fallback(self, lesson: LessonPlan, source: SourceDocument, scene_words):
        self._deep_needs_model("narration")
        return self._fallback_narration(lesson, source, scene_words)

    def _lesson_output_tokens(self) -> int:
        if self.mode == "deep":
            return DEEP_LESSON_OUTPUT_TOKENS
        return min(16000, max(7000, self.max_scenes * 750))

    def _deep_needs_model(self, what: str) -> None:
        """An in-depth lecture is never built on the rule-based fallback.

        Without a key (tests, offline dry runs) the fallback is all there is.
        With one, falling back means the model's answer was unreadable even
        after a retry, and a rule-based plan would quietly replace the lecture.
        """

        if self.mode == "deep" and os.getenv("GEMINI_API_KEY"):
            raise RuntimeError(
                f"The in-depth lecture's {what} came back unreadable after a retry; refusing "
                "to replace it with a rule-based one. Resume the run to try again."
            )

    def _lesson_fallback(self, source: SourceDocument) -> LessonPlan:
        self._deep_needs_model("lesson plan")
        return self._fallback_lesson(source)

    async def check_hook(self, lesson: LessonPlan) -> list[str]:
        """Sentences of the hook a first-time listener could not follow, as problems.

        Empty when the hook is empty, every sentence follows, or the check
        cannot run (the deterministic checks in hook_problems still apply).
        """

        if not (lesson.hook or "").strip():
            return []
        payload = {
            "hook": lesson.hook,
            "central_question": lesson.central_question,
            "running_example": lesson.running_example.model_dump(mode="json"),
        }
        prompt = (
            f"PROMPT_VERSION={HOOK_PROMPT_VERSION}\n"
            + hook_check_instructions() + "\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        result = await self.router.structured(
            stage="lesson", task_id="hook-check", provider=PLANNING_PROVIDER,
            model=PLANNING_MODEL, system="You are a careful listener checking a script.",
            prompt=prompt, schema=HookCheck, fallback=lambda: HookCheck(breaks=[]),
            max_output_tokens=1500,
        )
        return [
            f'The sentence "{" ".join(item.sentence.split())}" does not follow: '
            f'{" ".join(item.problem.split())}'
            for item in result.breaks
            if item.sentence.strip()
        ]

    async def rewrite_hook(self, lesson: LessonPlan, problems: list[str]) -> str:
        """A new hook for ``lesson`` that fixes ``problems``; the old hook if the call fails."""

        first = lesson.scenes[0] if lesson.scenes else None
        payload = {
            "central_question": lesson.central_question,
            "current_hook": lesson.hook,
            "running_example": lesson.running_example.model_dump(mode="json"),
            "first_scene": (
                {
                    "title": first.title,
                    "viewer_question": first.viewer_question,
                    "visual_objective": first.visual_objective,
                }
                if first else None
            ),
        }
        prompt = (
            f"PROMPT_VERSION={HOOK_PROMPT_VERSION}\n"
            + hook_rewrite_instructions(self.presenter_opens, problems) + "\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        result = await self.router.structured(
            stage="lesson", task_id="hook-rewrite", provider=PLANNING_PROVIDER,
            model=PLANNING_MODEL, system="You write the opening lines of explainer videos.",
            prompt=prompt, schema=HookRewrite,
            fallback=lambda: HookRewrite(hook=lesson.hook),
            max_output_tokens=1500,
        )
        return " ".join(result.hook.split())

    async def fix_beat(
        self,
        scene: NarratedScene,
        beat_id: str,
        previous_beat: Optional[str],
        problems: list[str],
        source: SourceDocument,
    ) -> str:
        """Beat ``beat_id`` of ``scene`` rewritten to fix ``problems``; its old text if the call fails.

        ``previous_beat`` is the previous scene's last beat (``None`` for the
        first scene). Only the text comes back: the caller keeps the beat's id
        and cited facts.
        """

        beat = next(item for item in scene.beats if item.beat_id == beat_id)
        facts = {fact.fact_id: fact.text for fact in source.facts}
        payload = {
            "beat_to_fix": {"beat_id": beat.beat_id, "text": beat.text},
            "scene_beats": [
                {"beat_id": item.beat_id, "text": item.text, "visual_action": item.visual_action}
                for item in scene.beats
            ],
            "previous_scene_last_beat": previous_beat,
            "source_facts": {
                fact_id: facts[fact_id] for fact_id in beat.source_fact_ids if fact_id in facts
            },
        }
        prompt = (
            f"PROMPT_VERSION={BEAT_FIX_PROMPT_VERSION}\n"
            + beat_fix_instructions(problems) + "\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        result = await self.router.structured(
            stage="narration", task_id=f"beat-fix-{beat_id}", scene_id=scene.scene_id,
            provider=PLANNING_PROVIDER, model=PLANNING_MODEL,
            system="You edit spoken educational scripts.",
            prompt=prompt, schema=BeatFix, fallback=lambda: BeatFix(text=beat.text),
            max_output_tokens=1500,
        )
        return spoken_plain(" ".join(result.text.split()))

    async def narration(
        self,
        lesson: LessonPlan,
        source: SourceDocument,
        review_feedback: Optional[list[str]] = None,
    ) -> NarrationPlan:
        facts = {fact.fact_id: fact.text for fact in source.facts}
        feedback_block = (
            "A REVIEWER REJECTED THE PREVIOUS SCRIPT. Fix every point:\n"
            + "".join(f"- {point}\n" for point in review_feedback)
            if review_feedback else ""
        )
        relaxed = self.pacing == "relaxed"
        target_wpm = 115 if relaxed else 135
        pace_guidance = (
            "Use a relaxed pace of 110-120 words per minute. Leave room for the learner to inspect "
            "each construction. Use 3-5 short semantic beats when the material needs them, but put "
            "at most one major visual change and one small emphasis action in each beat. "
            if relaxed else
            "Use a calm teaching pace of 125-140 words per minute. Split each scene into 2-4 "
            "semantic beats. "
        )
        # The brief visualizer has no per-scene word cap: each scene takes what its
        # idea needs. A professor's target length still sets a budget (below).
        if self.target_minutes is None and self.mode == "deep":
            low, high = deep_scene_words(len(lesson.scenes))
            pace_guidance += (
                f"Each scene's narration is {low} to {high} words; derivation and worked-example "
                "scenes use the upper end. Split each scene into 3-6 semantic beats. "
            )
        outline = [
            {
                "scene_id": item.scene_id,
                "title": item.title,
                "learning_goal": item.learning_goal,
                "new_visual_fact": item.new_visual_fact,
                "viewer_question": item.viewer_question,
            }
            for item in lesson.scenes
        ]
        depth_by_scene, notes_by_scene = self._scene_guidance(lesson, source)
        self.length_budget = (
            build_length_budget(lesson, source, self.target_minutes, target_wpm, depth_by_scene)
            if self.target_minutes is not None else None
        )
        scene_words = self.length_budget.scene_words if self.length_budget else {}
        narrated: list[NarratedScene] = []
        for offset in range(0, len(lesson.scenes), NARRATION_BATCH_SIZE):
            batch = lesson.scenes[offset:offset + NARRATION_BATCH_SIZE]
            batch_lesson = lesson.model_copy(update={"scenes": batch})
            expected_ids = [item.scene_id for item in batch]
            relevant_ids = {
                fact_id
                for item in batch
                for fact_id in item.source_fact_ids
            }
            relevant_facts = {
                fact_id: facts[fact_id]
                for fact_id in relevant_ids
                if fact_id in facts
            }
            previous = narrated[-1] if narrated else None
            next_scene = (
                lesson.scenes[offset + len(batch)]
                if offset + len(batch) < len(lesson.scenes)
                else None
            )
            payload = {
                "lesson_title": lesson.title,
                "lesson_spine": lesson.spine(),
                "global_scene_outline": outline,
                "required_scene_ids": expected_ids,
                "scenes_to_write": [item.model_dump(mode="json") for item in batch],
                "source_facts": relevant_facts,
                "continuity": {
                    "previous_scene": (
                        {
                            "scene_id": previous.scene_id,
                            "ending": previous.beats[-1].text[-400:],
                            "transition_out": previous.transition_out,
                        }
                        if previous else None
                    ),
                    "next_scene": (
                        {
                            "scene_id": next_scene.scene_id,
                            "title": next_scene.title,
                            "learning_goal": next_scene.learning_goal,
                        }
                        if next_scene else None
                    ),
                },
            }
            prompt = (
                f"PROMPT_VERSION={prompt_version(self.mode, 'narration')}\n"
                + feedback_block
                + narration_instructions(
                    pace_guidance, target_wpm, self.presenter_opens, self.mode,
                    self.audience or "undergraduate",
                ) + "\n\n"
                + json.dumps(payload, ensure_ascii=False)
            )
            prompt += narration_directive_block(
                self.directives,
                {key: scene_words[key] for key in expected_ids if key in scene_words},
                {key: depth_by_scene[key] for key in expected_ids if key in depth_by_scene},
                {key: notes_by_scene[key] for key in expected_ids if key in notes_by_scene},
                audience=self.audience,
            )
            task_id = f"narration-{offset // NARRATION_BATCH_SIZE:03d}"
            batch_narration = await self.router.structured(
                stage="narration", task_id=task_id, provider=PLANNING_PROVIDER,
                model=PLANNING_MODEL, system="You write clear spoken educational scripts.",
                prompt=prompt, schema=NarrationPlan,
                fallback=lambda batch_lesson=batch_lesson: self._narration_fallback(
                    batch_lesson, source, scene_words,
                ),
                max_output_tokens=DEEP_NARRATION_OUTPUT_TOKENS if self.mode == "deep" else 7000,
            )
            actual_ids = [item.scene_id for item in batch_narration.scenes]
            if actual_ids != expected_ids:
                raise RuntimeError(
                    f"Narration batch {task_id} returned scene IDs {actual_ids}; "
                    f"expected {expected_ids}."
                )
            narrated.extend(
                _spoken_scene(item).model_copy(update={"target_words_per_minute": target_wpm})
                for item in batch_narration.scenes
            )
        plan = NarrationPlan(scenes=narrated)
        self.compliance_checks = []
        if self.directives is not None or self.omitted_concepts:
            plan, self.compliance_checks = enforce_narration_directives(
                plan, self.directives, self.omitted_concepts,
            )
        self.length_report = None
        if self.length_budget is not None:
            plan, self.length_report = trim_to_budget(plan, self.length_budget)
        return plan

    async def visual_contracts(
        self,
        scenes: list[NarratedScene],
        lesson: LessonPlan,
        source: SourceDocument,
    ) -> list[VisualContract]:
        contracts: list[VisualContract] = []
        fact_map = {fact.fact_id: fact.text for fact in source.facts}
        for offset in range(0, len(scenes), 3):
            batch = scenes[offset:offset + 3]
            task_id = f"visual-{offset // 3:03d}"
            relevant = {
                fact_id: fact_map[fact_id]
                for scene in batch for beat in scene.beats for fact_id in beat.source_fact_ids
                if fact_id in fact_map
            }
            prompt = (
                f"PROMPT_VERSION={prompt_version(self.mode, 'planning')}\n"
                "For each scene, first define a semantic artifact_inventory, then compose those "
                "same artifacts into beat-level composition_plan entries. Separate primary visuals, "
                "semantic text, target-linked annotations, and transient effects. Every artifact needs "
                "a real teaching purpose, source facts, supported beats, and explicit lifecycle. "
                "Titles and equations use reserved text regions; diagrams use the remaining canvas. "
                "Do not add filler. Geometry may overlap intentionally; text may not overlap text or "
                "the frame. Persistent objects require next_scene carry_policy. Populate footprint, "
                "overflow_policy, minimum_scale, region_padding, and overlap_policy deliberately.\n\n"
                + layout_prompt_guide() + "\n\n"
                + json.dumps({
                    "lesson_title": lesson.title,
                    "scenes": [scene.model_dump(mode="json") for scene in batch],
                    "source_facts": relevant,
                }, ensure_ascii=False)
            )
            result = await self.router.structured(
                stage="visual_contract", task_id=task_id, provider="google",
                model=TEXT_MODEL, system="You are a Manim visual artifact director.",
                prompt=prompt, schema=VisualContractBatch,
                fallback=lambda batch=batch: VisualContractBatch(
                    contracts=[self._fallback_contract(scene) for scene in batch]
                ), max_output_tokens=6000,
            )
            contracts.extend(result.contracts)
        return contracts

    async def visual_contract_batch(
        self,
        batch: list[NarratedScene],
        lesson: LessonPlan,
        source: SourceDocument,
        batch_index: int,
    ) -> VisualContractBatch:
        fact_map = {fact.fact_id: fact.text for fact in source.facts}
        relevant = {
            fact_id: fact_map[fact_id]
            for scene in batch for beat in scene.beats for fact_id in beat.source_fact_ids
            if fact_id in fact_map
        }
        task_id = f"visual-{batch_index:03d}"
        prompt = (
            f"PROMPT_VERSION={prompt_version(self.mode, 'planning')}\n"
            "For each scene, first define a semantic artifact_inventory, then compose those "
            "same artifacts into beat-level composition_plan entries. Separate primary visuals, "
            "semantic text, target-linked annotations, and transient effects. Every artifact needs "
            "a real teaching purpose, source facts, supported beats, and explicit lifecycle. "
            "Titles and equations use reserved text regions; diagrams use the remaining canvas. "
            "Do not add filler. Geometry may overlap intentionally; text may not overlap text or "
            "the frame. Persistent objects require next_scene carry_policy. Populate footprint, "
            "overflow_policy, minimum_scale, region_padding, and overlap_policy deliberately.\n\n"
            + layout_prompt_guide() + "\n\n"
            + json.dumps({
                "lesson_title": lesson.title,
                "scenes": [scene.model_dump(mode="json") for scene in batch],
                "source_facts": relevant,
            }, ensure_ascii=False)
        )
        return await self.router.structured(
            stage="visual_contract", task_id=task_id, provider="google",
            model=TEXT_MODEL, system="You are a Manim visual artifact director.",
            prompt=prompt, schema=VisualContractBatch,
            fallback=lambda: VisualContractBatch(
                contracts=[self._fallback_contract(scene) for scene in batch]
            ), max_output_tokens=6000,
        )

    def _fallback_lesson(self, source: SourceDocument) -> LessonPlan:
        if self.min_scenes is not None or self.max_scenes > 12:
            target = self._required_min_scenes(source)
        else:
            target = min(self._scene_ceiling(), max(1, min(6, source.page_count)))
        chunks = _chunks(source.facts, target)
        scenes = []
        for index, chunk in enumerate(chunks):
            excerpt = chunk[0].text[:90]
            scenes.append(LessonScene(
                scene_id=f"scene_{index + 1:03d}", title=f"Part {index + 1}",
                description=excerpt, learning_goal=f"Understand {excerpt}",
                source_fact_ids=[fact.fact_id for fact in chunk],
            ))
        return LessonPlan(
            title=Path(source.filename).stem.replace("_", " "),
            audience=self.audience or "undergraduate",
            objectives=[scene.learning_goal for scene in scenes[:4]], scenes=scenes,
        )

    def _fallback_narration(
        self,
        lesson: LessonPlan,
        source: SourceDocument,
        scene_words: Optional[dict[str, int]] = None,
    ) -> NarrationPlan:
        facts = {fact.fact_id: fact.text for fact in source.facts}
        if scene_words is None and self.target_minutes is not None:
            # Called directly for a replacement lesson: budget that lesson.
            self.length_budget = build_length_budget(
                lesson, source, self.target_minutes,
                115 if self.pacing == "relaxed" else 135,
            )
            scene_words = self.length_budget.scene_words
        scenes = []
        for scene in lesson.scenes:
            selected = [facts[fact_id] for fact_id in scene.source_fact_ids if fact_id in facts]
            narration = " ".join(selected)[:2400] or scene.description
            sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", narration) if part.strip()]
            if not sentences:
                sentences = [narration]
            if scene_words and scene.scene_id in scene_words:
                sentences = truncate_sentences(sentences, scene_words[scene.scene_id])
            groups = _chunks(sentences, min(4, len(sentences)))
            beats = [NarrationBeat(
                beat_id=f"{scene.scene_id}_beat_{index + 1:02d}",
                text=" ".join(group), source_fact_ids=scene.source_fact_ids,
                role=("introduce" if index == 0 else
                      "conclude" if index == len(groups) - 1 else "explain"),
                pause_after_seconds=0.7 if index == len(groups) - 1 else 0.35,
                visual_action=("Introduce the scene's central visual." if index == 0 else
                               "Transform or highlight the existing visual with new information."),
            ) for index, group in enumerate(groups)]
            scenes.append(NarratedScene(
                scene_id=scene.scene_id, narration=" ".join(beat.text for beat in beats), beats=beats,
                transition_in=("" if not scenes else "Continue from the previous visual state."),
                transition_out="Pause before the next idea.",
                target_words_per_minute=115 if self.pacing == "relaxed" else 135,
            ))
        return NarrationPlan(scenes=scenes)

    @staticmethod
    def _fallback_contract(scene: NarratedScene) -> VisualContract:
        first = scene.beats[0]
        last = scene.beats[-1]
        primary_id = f"{scene.scene_id}_primary"
        title_id = f"{scene.scene_id}_title"
        artifacts = [
            ArtifactSpec(
                artifact_id=primary_id, category=ArtifactCategory.PRIMARY_VISUAL,
                role="concept diagram", purpose="Depict the scene's central relationship",
                source_fact_ids=sorted({item for beat in scene.beats for item in beat.source_fact_ids}),
                beat_ids=[beat.beat_id for beat in scene.beats],
                enter_beat=first.beat_id, exit_beat=last.beat_id,
                region=LayoutRegion.CONTENT_FULL, carry_policy="scene_end",
            ),
            ArtifactSpec(
                artifact_id=title_id, category=ArtifactCategory.SEMANTIC_TEXT,
                role="scene title", purpose="Orient the learner to the current concept",
                content=first.text[:90], source_fact_ids=first.source_fact_ids,
                beat_ids=[first.beat_id], enter_beat=first.beat_id, exit_beat=first.beat_id,
                region=LayoutRegion.HEADER, layer=2,
            ),
        ]
        return VisualContract(
            scene_id=scene.scene_id, layout_template=LayoutTemplate.FULL_VISUAL,
            artifact_inventory=artifacts,
            composition_plan=[BeatComposition(
                beat_id=beat.beat_id,
                active_artifact_ids=[primary_id] + ([title_id] if index == 0 else []),
                focus_artifact_ids=[primary_id],
            ) for index, beat in enumerate(scene.beats)],
            visual_progression=[beat.visual_action for beat in scene.beats if beat.visual_action],
            style_notes=["Use a dark background and high-contrast 3Blue1Brown-inspired colors."],
        )


def apply_audio_timing(scene: NarratedScene, duration: float) -> NarratedScene:
    """Give each beat its window in a recording of ``duration`` seconds.

    With measured speech spans (per-beat TTS), a beat's window opens when its
    first word is spoken and closes when the next beat's does, so the visuals
    timed to it land on its words. Without them, windows are shared out by
    word count.
    """

    spans = scene.beat_speech_spans
    if spans and len(spans) == len(scene.beats):
        starts = [0.0] + [min(duration, max(0.0, float(span[0]))) for span in spans[1:]]
        ends = starts[1:] + [duration]
        return scene.model_copy(update={
            "beats": [
                beat.model_copy(update={
                    "start_seconds": round(start, 3), "end_seconds": round(max(start, end), 3),
                })
                for beat, start, end in zip(scene.beats, starts, ends)
            ],
            "audio_duration_seconds": duration,
        })
    weights =[max(1, len(beat.text.split())) for beat in scene.beats]
    total = sum(weights)
    pauses = [beat.pause_after_seconds for beat in scene.beats]
    pause_total = min(sum(pauses), duration * 0.3)
    pause_scale = pause_total / sum(pauses) if sum(pauses) else 0.0
    speech_duration = max(0.0, duration - pause_total)
    cursor = 0.0
    beats = []
    for index, (beat, weight) in enumerate(zip(scene.beats, weights)):
        allocated_pause = beat.pause_after_seconds * pause_scale
        end = (
            duration
            if index == len(scene.beats) - 1
            else cursor + speech_duration * weight / total + allocated_pause
        )
        beats.append(beat.model_copy(update={
            "start_seconds": round(cursor, 3), "end_seconds": round(end, 3),
        }))
        cursor = end
    return scene.model_copy(update={
        "beats": beats, "audio_duration_seconds": duration,
    })


def validate_grounding(
    lesson: LessonPlan,
    narration: NarrationPlan,
    source: SourceDocument,
) -> list[str]:
    known = {fact.fact_id for fact in source.facts}
    errors = []
    for scene in lesson.scenes:
        unknown = set(scene.source_fact_ids) - known
        if unknown:
            errors.append(f"{scene.scene_id} cites unknown source facts: {sorted(unknown)}")
    lesson_ids = {scene.scene_id for scene in lesson.scenes}
    for scene in narration.scenes:
        if scene.scene_id not in lesson_ids:
            errors.append(f"Narration contains unknown scene: {scene.scene_id}")
        for beat in scene.beats:
            unknown = set(beat.source_fact_ids) - known
            if unknown:
                errors.append(f"{beat.beat_id} cites unknown source facts: {sorted(unknown)}")
    return errors


def validate_visual_contract(
    contract: VisualContract,
    scene: NarratedScene,
    source: SourceDocument,
) -> list[str]:
    errors: list[str] = []
    known_facts = {fact.fact_id for fact in source.facts}
    beat_ids = [beat.beat_id for beat in scene.beats]
    known_beats = set(beat_ids)
    if contract.scene_id != scene.scene_id:
        errors.append(f"contract scene_id {contract.scene_id!r} does not match {scene.scene_id!r}")
    available_regions = set(regions_for(contract.layout_template))
    composition_beats = [beat.beat_id for beat in contract.composition_plan]
    if composition_beats != beat_ids:
        errors.append("composition_plan must contain every narration beat once and in order")
    for artifact in contract.artifact_inventory:
        if artifact.region not in available_regions:
            errors.append(
                f"{artifact.artifact_id} uses region {artifact.region.value!r}, which is unavailable "
                f"in layout {contract.layout_template.value!r}"
            )
        if artifact.footprint.value == "large" and artifact.region not in {
            LayoutRegion.CONTENT_FULL, LayoutRegion.CONTENT_CENTER,
        }:
            errors.append(
                f"{artifact.artifact_id} is large but is assigned to narrow region "
                f"{artifact.region.value!r}"
            )
        unknown_facts = set(artifact.source_fact_ids) - known_facts
        unknown_beats = set(artifact.beat_ids) - known_beats
        if unknown_facts:
            errors.append(f"{artifact.artifact_id} cites unknown facts: {sorted(unknown_facts)}")
        if unknown_beats:
            errors.append(f"{artifact.artifact_id} cites unknown beats: {sorted(unknown_beats)}")
        if artifact.enter_beat not in known_beats or artifact.exit_beat not in known_beats:
            errors.append(f"{artifact.artifact_id} has an invalid lifecycle boundary")
        elif beat_ids.index(artifact.enter_beat) > beat_ids.index(artifact.exit_beat):
            errors.append(f"{artifact.artifact_id} exits before it enters")
        if artifact.carry_policy == "next_scene" and artifact.exit_beat != beat_ids[-1]:
            errors.append(f"{artifact.artifact_id} carries forward but exits before scene end")
    artifacts = {item.artifact_id: item for item in contract.artifact_inventory}
    for composition in contract.composition_plan:
        owners: dict[LayoutRegion, list[str]] = {}
        for artifact_id in composition.active_artifact_ids:
            artifact = artifacts.get(artifact_id)
            if artifact is None:
                continue
            attached = (
                artifact.category in {ArtifactCategory.ANNOTATION, ArtifactCategory.TRANSIENT_EFFECT}
                and bool(artifact.parent_id or artifact.target_id)
            )
            if not attached:
                owners.setdefault(artifact.region, []).append(artifact_id)
        for region, artifact_ids in owners.items():
            if len(artifact_ids) > 1:
                errors.append(
                    f"{composition.beat_id} assigns multiple independent artifacts to "
                    f"{region.value}: {sorted(artifact_ids)}; combine related content into one VGroup"
                )
    return errors


def _spoken_scene(scene: NarratedScene) -> NarratedScene:
    """Turn leaked snake_case ids in beat text into words; narration follows the beats."""

    beats = [
        beat.model_copy(update={"text": spoken_plain(beat.text)}) for beat in scene.beats
    ]
    return scene.model_copy(update={
        "beats": beats, "narration": " ".join(beat.text for beat in beats),
    })


class HookRewrite(BaseModel):
    hook: str


class HookBreak(BaseModel):
    sentence: str
    problem: str = ""


class HookCheck(BaseModel):
    breaks: list[HookBreak] = []


class BeatFix(BaseModel):
    text: str


# In-depth lecture: the whole narration stays under about 25 minutes, and a
# scene is at least long enough to teach one step properly.
DEEP_TOTAL_WORDS = 3800
DEEP_SCENE_WORDS = (110, 260)
DEEP_MIN_SCENES = 6
DEEP_PAGES_PER_SCENE = 3
# Argon's thinking shares the output budget; a long plan or batch needs room.
DEEP_LESSON_OUTPUT_TOKENS = 32000
DEEP_NARRATION_OUTPUT_TOKENS = 16000
LESSON_ROLES = frozenset({"motivate", "intuition", "derive", "apply", "connect"})


def deep_scene_words(scene_count: int) -> tuple[int, int]:
    """Words per scene for an in-depth lecture of ``scene_count`` scenes."""

    low, high = DEEP_SCENE_WORDS
    high = max(low + 30, min(high, DEEP_TOTAL_WORDS // max(1, scene_count)))
    return low, high
_HOOK_PUNCTUATION = ("\u2014", "\u2013", ";", "(", ")")


def hook_problems(hook: str, presenter_opens: bool) -> list[str]:
    """What is checkably wrong with a hook, as sentences a rewrite can act on.

    The hook must be a short chain (situation, snag, question) ending on one
    question. Whether each sentence follows from the last is judged by the
    script review; this catches the shapes that cannot flow when spoken.
    """

    text = " ".join((hook or "").split())
    if not text:
        return []
    problems = []
    limit = HOOK_MAX_WORDS_PRESENTER if presenter_opens else HOOK_MAX_WORDS_NARRATOR
    words = len(text.split())
    if words > limit:
        problems.append(f"It is {words} words long; the limit is {limit}.")
    if text.count("?") != 1 or not text.endswith("?"):
        problems.append(
            "It must end on exactly one question, the central question, and ask nothing before it."
        )
    if any(mark in text for mark in _HOOK_PUNCTUATION) or " - " in text:
        problems.append(
            "It uses a dash, semicolon or parentheses, which a listener cannot hear; "
            "say it as separate sentences."
        )
    sentences = [part for part in re.split(r"(?<=[.?!])\s+", text) if part.strip()]
    if len(sentences) < 3:
        problems.append(
            "It is not a chain of situation, snag and question: it needs at least three sentences."
        )
    return problems


def _normalise_spine(lesson: LessonPlan, source: SourceDocument) -> LessonPlan:
    """One coverage entry per source page, a reason for every page left out, tidy motifs."""

    fact_pages = {fact.fact_id: fact.page_number for fact in source.facts}
    cited_pages = {
        fact_pages[fact_id]
        for scene in lesson.scenes
        for fact_id in scene.source_fact_ids
        if fact_id in fact_pages
    }
    by_page: dict[int, PageCoverage] = {}
    for entry in lesson.coverage:
        if 1 <= entry.page_number <= source.page_count and entry.page_number not in by_page:
            if entry.treatment != "covered" and not entry.reason.strip():
                entry = entry.model_copy(update={"reason": "No reason given by the planner."})
            by_page[entry.page_number] = entry
    coverage = []
    for page in range(1, source.page_count + 1):
        if page in by_page:
            coverage.append(by_page[page])
        elif page in cited_pages:
            coverage.append(PageCoverage(
                page_number=page, treatment="covered", reason="Cited by a scene.",
            ))
        else:
            coverage.append(PageCoverage(
                page_number=page, treatment="omitted",
                reason="The lesson plan did not place this page.",
            ))
    motifs = [
        motif.model_copy(update={"color": motif.color.strip().upper().replace(" ", "_")})
        for motif in lesson.motifs[:MAX_SPINE_MOTIFS]
    ]
    scenes = [
        scene.model_copy(update={
            "role": role if (role := scene.role.strip().lower()) in LESSON_ROLES else "",
            "chapter": " ".join(scene.chapter.split()),
        })
        for scene in lesson.scenes
    ]
    return lesson.model_copy(update={"coverage": coverage, "motifs": motifs, "scenes": scenes})


def _lesson_fact_payload(fact: SourceFact) -> dict[str, object]:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", fact.text)
    return {
        "fact_id": fact.fact_id,
        "page_number": fact.page_number,
        "kind": fact.kind,
        "text": " ".join(text.split()),
    }


def _chunks(items: Iterable, count: int) -> list[list]:
    values = list(items)
    if not values:
        return []
    count = max(1, min(count, len(values)))
    return [values[index * len(values) // count:(index + 1) * len(values) // count] for index in range(count)]


def _full_coverage_scene_floor(page_count: int, max_scenes: int) -> int:
    """Reserve enough scenes for dense lecture decks without making one scene per page."""

    return min(max_scenes, max(8, (page_count * 3 + 4) // 5))
