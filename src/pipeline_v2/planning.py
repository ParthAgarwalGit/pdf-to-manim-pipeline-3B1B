"""Grounded lesson, narration, timing, and artifact-contract stages."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

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
    SourceDocument,
    SourceFact,
    VisualContract,
    VisualContractBatch,
)
from .providers import MissingProviderKey, ProviderRouter, clean_json_text

try:
    import pymupdf
except ImportError:  # pragma: no cover - compatibility with older PyMuPDF
    import fitz as pymupdf


PROMPT_VERSION = "v2.4.2"
NARRATION_PROMPT_VERSION = "v2.5.0"
NARRATION_BATCH_SIZE = 5
PLANNING_PROVIDER = "google"
PLANNING_MODEL = "gemini-3.7-flash"


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
    ):
        if min_scenes is not None and min_scenes > max_scenes:
            raise ValueError("min_scenes cannot exceed max_scenes")
        self.router = router
        self.max_scenes = max_scenes
        self.min_scenes = min_scenes
        self.pacing = pacing

    def _required_min_scenes(self, source: SourceDocument) -> int:
        if self.min_scenes is not None:
            return self.min_scenes
        if self.max_scenes > 12:
            return _full_coverage_scene_floor(source.page_count, self.max_scenes)
        return 1

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
                prompt = (
                    f"PROMPT_VERSION={PROMPT_VERSION}\n"
                    "Transcribe the educational content in these PDF pages in image order. "
                    f"Their one-based page numbers are {pages}. Preserve equations exactly. "
                    "Return JSON only as {\"facts\":[{\"page_number\":1,\"text\":\"...\","
                    "\"kind\":\"text|equation|image|table\"}]} and do not infer unseen content."
                )
                try:
                    response = await self.router.vision(
                        stage="ocr", task_id=f"ocr-{batch_index:03d}", scene_id="source",
                        model="gemini-3.7-flash",
                        system="You are a precise OCR system for mathematical source documents.",
                        prompt=prompt, image_paths=image_paths, max_output_tokens=5000,
                    )
                    parsed = OCRBatch.model_validate_json(clean_json_text(response.text))
                    for index, fact in enumerate(parsed.facts):
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
        facts = [_lesson_fact_payload(fact) for fact in source.facts]
        required_min = self._required_min_scenes(source)
        if self.min_scenes is not None or self.max_scenes > 12:
            coverage_floor = required_min
            scene_range = (
                f"exactly {self.max_scenes}"
                if coverage_floor == self.max_scenes
                else f"between {coverage_floor} and {self.max_scenes}"
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
                f"{self.max_scenes} scenes. "
            )
        prompt = (
            f"PROMPT_VERSION={PROMPT_VERSION}\n"
            "Create a source-grounded educational lesson plan. "
            + scene_guidance
            + "Merge "
            "adjacent scenes that teach the same relationship; do not fragment one example into "
            "several tiny scenes. Every scene must cite source_fact_ids from the input. For every "
            "scene set visual_objective, the one new_visual_fact it contributes, and do_not_repeat "
            "items already demonstrated by earlier scenes. Do not invent facts, examples, or "
            "equations. Set delivery_mode to manim.\n\n"
            + json.dumps({"document": source.filename, "facts": facts}, ensure_ascii=False)
        )
        lesson = await self.router.structured(
            stage="lesson", task_id="lesson", provider=PLANNING_PROVIDER,
            model=PLANNING_MODEL, system="You are a rigorous educational lesson director.",
            prompt=prompt, schema=LessonPlan, fallback=lambda: self._fallback_lesson(source),
            max_output_tokens=min(16000, max(7000, self.max_scenes * 750)),
        )
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
            + f"{self.max_scenes} scenes. Do not merely duplicate scenes: split dense material "
            + "along concept, derivation, and worked-example boundaries while preserving full "
            + "source coverage."
        )
        expanded = await self.router.structured(
            stage="lesson", task_id="lesson-expand", provider=PLANNING_PROVIDER,
            model=PLANNING_MODEL, system="You are a rigorous educational lesson director.",
            prompt=retry_prompt, schema=LessonPlan,
            fallback=lambda: self._fallback_lesson(source),
            max_output_tokens=min(16000, max(7000, self.max_scenes * 750)),
        )
        if len(expanded.scenes) < required_min:
            raise RuntimeError(
                f"Lesson planner returned {len(expanded.scenes)} scenes after regeneration; "
                f"at least {required_min} are required."
            )
        return expanded

    async def narration(self, lesson: LessonPlan, source: SourceDocument) -> NarrationPlan:
        facts = {fact.fact_id: fact.text for fact in source.facts}
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
        outline = [
            {
                "scene_id": item.scene_id,
                "title": item.title,
                "learning_goal": item.learning_goal,
                "new_visual_fact": item.new_visual_fact,
            }
            for item in lesson.scenes
        ]
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
                f"PROMPT_VERSION={NARRATION_PROMPT_VERSION}\n"
                "Write the requested consecutive section of one continuity-aware spoken lesson. "
                + pace_guidance
                + "Return exactly the required_scene_ids once each and in the supplied order; do "
                "not write narration for any other scene. Keep one main idea per sentence. Avoid "
                "dense clauses, breathless lists, repeated definitions, and abrupt openings. The "
                "first beat of each scene must connect naturally to what the learner just saw; the "
                "final beat must either pause for reflection or bridge to the next scene. Populate "
                "role, pause_after_seconds (normally 0.25-0.55, or 0.6-0.9 for reflection and scene "
                "endings), visual_action, continuity_note, transition_in, and transition_out. Every "
                "visual_action must introduce, transform, compare, highlight, or remove visual "
                "information; never request the same complete construction more than once. Every "
                "beat must cite only supplied source_fact_ids. Preserve mathematical notation and "
                f"do not add unsupported claims. Set target_words_per_minute to {target_wpm}. "
                "Timing fields must be zero; the pipeline computes them after TTS.\n\n"
                + json.dumps(payload, ensure_ascii=False)
            )
            task_id = f"narration-{offset // NARRATION_BATCH_SIZE:03d}"
            batch_narration = await self.router.structured(
                stage="narration", task_id=task_id, provider=PLANNING_PROVIDER,
                model=PLANNING_MODEL, system="You write clear spoken educational scripts.",
                prompt=prompt, schema=NarrationPlan,
                fallback=lambda batch_lesson=batch_lesson: self._fallback_narration(
                    batch_lesson, source,
                ),
                max_output_tokens=7000,
            )
            actual_ids = [item.scene_id for item in batch_narration.scenes]
            if actual_ids != expected_ids:
                raise RuntimeError(
                    f"Narration batch {task_id} returned scene IDs {actual_ids}; "
                    f"expected {expected_ids}."
                )
            narrated.extend(
                item.model_copy(update={"target_words_per_minute": target_wpm})
                for item in batch_narration.scenes
            )
        return NarrationPlan(scenes=narrated)

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
                f"PROMPT_VERSION={PROMPT_VERSION}\n"
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
                stage="visual_contract", task_id=task_id, provider="openai",
                model="gpt-5.4-mini", system="You are a Manim visual artifact director.",
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
            f"PROMPT_VERSION={PROMPT_VERSION}\n"
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
            stage="visual_contract", task_id=task_id, provider="openai",
            model="gpt-5.4-mini", system="You are a Manim visual artifact director.",
            prompt=prompt, schema=VisualContractBatch,
            fallback=lambda: VisualContractBatch(
                contracts=[self._fallback_contract(scene) for scene in batch]
            ), max_output_tokens=6000,
        )

    def _fallback_lesson(self, source: SourceDocument) -> LessonPlan:
        if self.min_scenes is not None or self.max_scenes > 12:
            target = self._required_min_scenes(source)
        else:
            target = min(self.max_scenes, max(1, min(6, source.page_count)))
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
            title=Path(source.filename).stem.replace("_", " "), audience="undergraduate",
            objectives=[scene.learning_goal for scene in scenes[:4]], scenes=scenes,
        )

    def _fallback_narration(
        self, lesson: LessonPlan, source: SourceDocument,
    ) -> NarrationPlan:
        facts = {fact.fact_id: fact.text for fact in source.facts}
        scenes = []
        for scene in lesson.scenes:
            selected = [facts[fact_id] for fact_id in scene.source_fact_ids if fact_id in facts]
            narration = " ".join(selected)[:2400] or scene.description
            sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", narration) if part.strip()]
            if not sentences:
                sentences = [narration]
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
    weights = [max(1, len(beat.text.split())) for beat in scene.beats]
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
