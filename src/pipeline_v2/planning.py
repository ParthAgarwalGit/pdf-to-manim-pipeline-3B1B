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


PROMPT_VERSION = "v2.1.0"
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
    def __init__(self, router: ProviderRouter, max_scenes: int = 12):
        self.router = router
        self.max_scenes = max_scenes

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
        facts = [fact.model_dump(mode="json") for fact in source.facts]
        prompt = (
            f"PROMPT_VERSION={PROMPT_VERSION}\n"
            "Create a source-grounded educational lesson plan. Use 6-12 substantial scenes "
            "when the source supports them. Every scene must cite source_fact_ids from the input. "
            "Do not invent facts, examples, or equations. Set delivery_mode to manim.\n\n"
            + json.dumps({"document": source.filename, "facts": facts}, ensure_ascii=False)
        )
        return await self.router.structured(
            stage="lesson", task_id="lesson", provider=PLANNING_PROVIDER,
            model=PLANNING_MODEL, system="You are a rigorous educational lesson director.",
            prompt=prompt, schema=LessonPlan, fallback=lambda: self._fallback_lesson(source),
            max_output_tokens=7000,
        )

    async def narration(self, lesson: LessonPlan, source: SourceDocument) -> NarrationPlan:
        facts = {fact.fact_id: fact.text for fact in source.facts}
        prompt = (
            f"PROMPT_VERSION={PROMPT_VERSION}\n"
            "Write one continuous, concise narration for this lesson. Split every scene into "
            "2-5 semantic beats. Every beat must cite only supplied source_fact_ids. Preserve "
            "mathematical notation and do not add unsupported claims. Timing fields must be zero; "
            "the pipeline computes them after TTS.\n\n"
            + json.dumps({"lesson": lesson.model_dump(mode="json"), "facts": facts}, ensure_ascii=False)
        )
        return await self.router.structured(
            stage="narration", task_id="narration", provider=PLANNING_PROVIDER,
            model=PLANNING_MODEL, system="You write clear spoken educational scripts.",
            prompt=prompt, schema=NarrationPlan,
            fallback=lambda: self._fallback_narration(lesson, source), max_output_tokens=9000,
        )

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
        target = min(self.max_scenes, max(1, min(8, max(source.page_count, 6))))
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

    @staticmethod
    def _fallback_narration(lesson: LessonPlan, source: SourceDocument) -> NarrationPlan:
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
            ) for index, group in enumerate(groups)]
            scenes.append(NarratedScene(
                scene_id=scene.scene_id, narration=" ".join(beat.text for beat in beats), beats=beats,
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
            style_notes=["Use a dark background and high-contrast 3Blue1Brown-inspired colors."],
        )


def apply_audio_timing(scene: NarratedScene, duration: float) -> NarratedScene:
    weights = [max(1, len(beat.text.split())) for beat in scene.beats]
    total = sum(weights)
    cursor = 0.0
    beats = []
    for index, (beat, weight) in enumerate(zip(scene.beats, weights)):
        end = duration if index == len(scene.beats) - 1 else cursor + duration * weight / total
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


def _chunks(items: Iterable, count: int) -> list[list]:
    values = list(items)
    if not values:
        return []
    count = max(1, min(count, len(values)))
    return [values[index * len(values) // count:(index + 1) * len(values) // count] for index in range(count)]
