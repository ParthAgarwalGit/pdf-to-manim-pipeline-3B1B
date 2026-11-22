"""Compact creative raw-Manim generation with automatic instrumentation."""

from __future__ import annotations

import json
import keyword
import re
from pathlib import Path

from .instrumentation import deinstrument_scene_code, instrument_scene_code
from .layout import region_bounds
from .model_config import CODE_MODEL
from .models import LessonScene, NarratedScene, SourceFact, VerificationFinding, VisualContract
from .prompt_text import CODEGEN_STYLE_RULES, REPAIR_RECIPES
from .providers import MissingProviderKey, ProviderRouter


CODE_PROMPT_VERSION = "v2-creative-code-5-argon-style-layout"
# Argon's thinking shares the output budget; a long scene's code needs room.
CODE_OUTPUT_TOKENS = 16000
# Findings shown to the repair model, most serious first.
REPAIR_FINDINGS_SHOWN = 12
REPAIR_PRIORITY = (
    # Seen in the rendered video and confirmed: what a viewer would notice first.
    "VISUAL_TEXT_OVERLAP", "VISUAL_CUT_OFF",
    "TEXT_FRAME_COLLISION", "OFFSCREEN_OBJECT", "TEXT_TEXT_OVERLAP", "TEXT_GEOMETRY_OVERLAP",
    "VISUAL_CROWDED", "TEXT_TOO_SMALL", "VISUAL_DENSITY_HIGH",
)


class CodeGenerationService:
    def __init__(self, router: ProviderRouter):
        self.router = router

    async def generate(
        self,
        scene: NarratedScene,
        contract: VisualContract,
        source_facts: list[SourceFact],
        lesson_scene: LessonScene | None = None,
        visual_history: list[dict[str, object]] | None = None,
        retry_attempt: int = 0,
        notation_rules: list[str] | None = None,
        visual_world: list[str] | None = None,
        avoid_defects: list[str] | None = None,
    ) -> str:
        class_name = class_name_for(scene.scene_id)
        system, prompt = generation_request(
            scene=scene,
            contract=contract,
            source_facts=source_facts,
            class_name=class_name,
            lesson_scene=lesson_scene,
            visual_history=visual_history,
            retry_attempt=retry_attempt,
            notation_rules=notation_rules,
            visual_world=visual_world,
            avoid_defects=avoid_defects,
        )
        try:
            response = await self.router.text(
                stage="code_generation", task_id=f"code-{scene.scene_id}", scene_id=scene.scene_id,
                provider="google", model=CODE_MODEL, system=system,
                prompt=prompt, max_output_tokens=CODE_OUTPUT_TOKENS,
            )
            raw_code = clean_code(response.text)
            return instrument_scene_code(
                raw_code, class_name=class_name, scene=scene, contract=contract,
            ).code
        except MissingProviderKey:
            if self.router.strict_keys:
                raise
            code = fallback_code(scene, contract)
            self.router.artifacts.record_provider_call(
                stage="code_generation", call_id=f"code-local-{scene.scene_id}",
                scene_id=scene.scene_id,
                request={
                    "provider": "google", "model": CODE_MODEL,
                    "system": system, "prompt": prompt,
                },
                response={
                    "provider": "local", "model": "deterministic-manim-fallback",
                    "code": code,
                },
            )
            return code

    async def repair(
        self,
        *,
        scene: NarratedScene,
        contract: VisualContract,
        code: str,
        findings: list[VerificationFinding],
        frame_paths: list[str],
        lesson_scene: LessonScene | None = None,
        visual_history: list[dict[str, object]] | None = None,
        visual_world: list[str] | None = None,
    ) -> str:
        class_name = class_name_for(scene.scene_id)
        system = f"""PROMPT_VERSION={CODE_PROMPT_VERSION}
You repair one self-contained Manim Community Python animation using rendered evidence.
Return one complete executable Python script only. Define exactly one Scene subclass named
{class_name}. Make the smallest visual correction that fixes the reported problem. Preserve
correct mathematics, the explanatory idea, and every self.next_section("beat_id") marker.
Do not add tracking, layout-engine, artifact-registration, or audio-timing code; the runtime
adds those automatically. Do not claim success; the pipeline will render and measure the result.
{CODEGEN_STYLE_RULES}
{REPAIR_RECIPES}"""
        raw_code = deinstrument_scene_code(code)
        repair_payload: dict[str, object] = {
            "brief": _lesson_brief(lesson_scene, scene),
            "visual_history": visual_history or [],
            "narration_beats": _beat_payload(scene),
            "findings": [item.model_dump(mode="json") for item in repair_findings(findings)],
            "existing_code": raw_code,
        }
        if visual_world:
            repair_payload["visual_world"] = visual_world
        prompt = json.dumps(repair_payload, ensure_ascii=False)
        response = await self.router.vision(
            stage="vision_repair", task_id=f"repair-{scene.scene_id}", scene_id=scene.scene_id,
            model=CODE_MODEL, system=system, prompt=prompt,
            image_paths=[Path(path) for path in frame_paths[:8]], max_output_tokens=CODE_OUTPUT_TOKENS,
        )
        repaired = clean_code(response.text)
        return instrument_scene_code(
            repaired, class_name=class_name, scene=scene, contract=contract,
        ).code

    def reinstrument(self, code: str, scene: NarratedScene, contract: VisualContract) -> str:
        """Re-time existing scene code against new narration timing; no model call."""

        return instrument_scene_code(
            deinstrument_scene_code(code), class_name=class_name_for(scene.scene_id),
            scene=scene, contract=contract,
        ).code

    async def revise(
        self,
        *,
        scene: NarratedScene,
        contract: VisualContract,
        code: str,
        instruction: str,
        lesson_scene: LessonScene | None = None,
        visual_history: list[dict[str, object]] | None = None,
        notation_rules: list[str] | None = None,
        visual_world: list[str] | None = None,
    ) -> str:
        """Apply a professor's visual correction to one scene's existing code."""

        class_name = class_name_for(scene.scene_id)
        system = f"""PROMPT_VERSION={CODE_PROMPT_VERSION}
You revise one self-contained Manim Community Python animation according to a correction from the
professor who owns the lesson. Return one complete executable Python script only. Define exactly
one Scene subclass named {class_name}. Apply the correction exactly and change nothing else:
preserve correct mathematics, the explanatory idea, and every self.next_section("beat_id") marker.
Allowed imports are `from manim import *` and `import numpy as np`. Do not add tracking,
layout-engine, artifact-registration, or audio-timing code; the runtime adds those automatically.
{CODEGEN_STYLE_RULES}"""
        payload: dict[str, object] = {
            "professor_correction": instruction,
            "brief": _lesson_brief(lesson_scene, scene),
            "visual_history": visual_history or [],
            "narration_beats": _beat_payload(scene),
            "existing_code": deinstrument_scene_code(code),
        }
        if notation_rules:
            payload["notation_rules"] = notation_rules
        if visual_world:
            payload["visual_world"] = visual_world
        try:
            response = await self.router.text(
                stage="code_generation", task_id=f"code-{scene.scene_id}", scene_id=scene.scene_id,
                provider="google", model=CODE_MODEL, system=system,
                prompt=json.dumps(payload, ensure_ascii=False), max_output_tokens=CODE_OUTPUT_TOKENS,
            )
        except MissingProviderKey:
            if self.router.strict_keys:
                raise
            # Without a model the correction cannot be applied; keep the scene
            # renderable and say so instead of silently claiming a fix.
            self.router.artifacts.event(
                "revision_instruction_not_applied", stage="code_generation",
                scene_id=scene.scene_id, task_id=f"code-{scene.scene_id}",
                message="GEMINI_API_KEY is not configured; the existing visuals were kept.",
            )
            return self.reinstrument(code, scene, contract)
        return instrument_scene_code(
            clean_code(response.text), class_name=class_name, scene=scene, contract=contract,
        ).code


def repair_findings(findings: list[VerificationFinding]) -> list[VerificationFinding]:
    """The findings worth a repair, most serious first, one per kind and object.

    The verifier reports the same overlap at every checkpoint and dozens of
    density warnings; sending the first few as they came could hide the one
    error that matters.
    """

    def rank(item: VerificationFinding) -> tuple[int, int]:
        kind = REPAIR_PRIORITY.index(item.code) if item.code in REPAIR_PRIORITY else len(REPAIR_PRIORITY)
        return (0 if item.severity == "error" else 1, kind)

    chosen: list[VerificationFinding] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for item in sorted(findings, key=rank):
        ids = tuple(sorted(item.object_ids or item.artifact_ids))
        if not ids and item.code.startswith("VISUAL_"):
            # Seen in frames, so named in words, not object ids: each is its own defect.
            ids = (item.message,)
        key = (item.code, ids)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(item)
        if len(chosen) == REPAIR_FINDINGS_SHOWN:
            break
    return chosen


def generation_request(
    *,
    scene: NarratedScene,
    contract: VisualContract,
    source_facts: list[SourceFact],
    class_name: str,
    lesson_scene: LessonScene | None = None,
    visual_history: list[dict[str, object]] | None = None,
    retry_attempt: int = 0,
    notation_rules: list[str] | None = None,
    visual_world: list[str] | None = None,
    avoid_defects: list[str] | None = None,
) -> tuple[str, str]:
    """Build the deliberately small model-facing creative request."""

    system = f"""PROMPT_VERSION={CODE_PROMPT_VERSION}
You create one self-contained Manim Community Edition Python animation.
Return only executable Python, without Markdown or explanation. Define exactly one Scene
subclass named {class_name}. Allowed imports are `from manim import *` and `import numpy as np`.
Do not read files, call external services, or assume helper modules.

Use ordinary Manim creatively. Prefer transforming a small number of meaningful visual objects
over presenting slide-like paragraphs. Develop one clear visual explanation through the supplied
narration beats. At the start of each beat call self.next_section("beat_id"); do not calculate exact
audio timing because the runtime retimes those sections. Keep visible content inside the frame,
keep unrelated text separated, and use axes.c2p() for mathematical graph coordinates.

Treat the scene as a progression, not a slideshow or replay. Each beat must preserve useful state
and introduce, transform, compare, highlight, or remove information according to visual_action.
Never rebuild or replay a complete construction listed in visual_history or do_not_repeat. Refer
back to it briefly and begin from the new contribution instead. Do not repeat one animation merely
to fill narration time; use a calm hold or a focused highlight when no new construction is needed.
{CODEGEN_STYLE_RULES}"""
    payload = {
        "brief": _lesson_brief(lesson_scene, scene),
        "target_duration_seconds": round(_target_duration(scene), 3),
        "narration": scene.narration,
        "narration_beats": _beat_payload(scene),
        "scene_transition": {
            "in": scene.transition_in,
            "out": scene.transition_out,
        },
        "visual_history": visual_history or [],
        "regeneration_attempt": retry_attempt,
        "regeneration_instruction": (
            "Create a fresh visual composition for this retry; do not reproduce the previous "
            "candidate's layout verbatim."
            if retry_attempt else ""
        ),
        "do_not_repeat": lesson_scene.do_not_repeat if lesson_scene else [],
        "required_new_visual_fact": lesson_scene.new_visual_fact if lesson_scene else "",
        "grounded_facts": [fact.text[:500] for fact in source_facts[:8]],
        "suggested_visual_direction": {
            "archetype": contract.archetype,
            "ingredients": _creative_ingredients(contract),
            "style": contract.style_notes[:3],
        },
    }
    if notation_rules:
        # Added only when the professor set notation, so requests (and their
        # cache keys) for runs without directives are unchanged.
        payload["notation_rules"] = notation_rules
    if visual_world:
        payload["visual_world"] = visual_world
    if avoid_defects:
        # A redesign after repairs could not clear what a viewer would see.
        payload["regeneration_instruction"] = (
            "An earlier design of this scene still showed the layout defects in "
            "avoid_defects in its rendered video after repair. Create a fresh composition "
            "with a different arrangement in which none of them can happen: give every "
            "label and formula its own free space for the whole scene, including while "
            "objects move."
        )
        payload["avoid_defects"] = avoid_defects
    return system, json.dumps(payload, ensure_ascii=False)


def _lesson_brief(lesson_scene: LessonScene | None, scene: NarratedScene) -> dict[str, str]:
    if lesson_scene is None:
        return {"title": scene.scene_id, "description": scene.narration}
    values = {
        "title": lesson_scene.title,
        "description": lesson_scene.description,
        "learning_goal": lesson_scene.learning_goal,
        "visual_objective": lesson_scene.visual_objective,
        "new_visual_fact": lesson_scene.new_visual_fact,
    }
    if lesson_scene.example:
        values["example"] = lesson_scene.example
    # In-depth lecture only: "derive" and "apply" scenes follow their own layout rules.
    if lesson_scene.role:
        values["role"] = lesson_scene.role
    if lesson_scene.chapter:
        values["chapter"] = lesson_scene.chapter
    return values


def _beat_payload(scene: NarratedScene) -> list[dict[str, object]]:
    return [
        {
            "beat_id": beat.beat_id,
            "text": beat.text,
            "role": beat.role,
            "pause_after_seconds": round(float(beat.pause_after_seconds), 3),
            "visual_action": beat.visual_action,
            "continuity_note": beat.continuity_note,
            "start_seconds": round(float(beat.start_seconds), 3),
            "end_seconds": round(float(beat.end_seconds), 3),
        }
        for beat in scene.beats
    ]


def _creative_ingredients(contract: VisualContract) -> list[str]:
    values = []
    for artifact in sorted(contract.artifact_inventory, key=lambda item: item.priority, reverse=True):
        value = artifact.content.strip() or artifact.purpose.strip()
        if value and value not in values:
            values.append(value[:180])
        if len(values) == 6:
            break
    return values


def _target_duration(scene: NarratedScene) -> float:
    if scene.audio_duration_seconds:
        return float(scene.audio_duration_seconds)
    return max(1.5, len(scene.narration.split()) / 2.8)


def clean_code(value: str) -> str:
    text = value.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip() + "\n"


def class_name_for(scene_id: str) -> str:
    pieces = re.findall(r"[A-Za-z0-9]+", scene_id)
    name = "V2" + "".join(piece[:1].upper() + piece[1:] for piece in pieces)
    if not name or name[0].isdigit() or keyword.iskeyword(name):
        name = "V2GeneratedScene"
    return name


def fallback_code(scene: NarratedScene, contract: VisualContract) -> str:
    """Deterministic, tracked fallback used by offline tests and missing-key runs."""
    class_name = class_name_for(scene.scene_id)
    lines = [
        "from manim import *",
        "import numpy as np",
        "from src.pipeline_v2.tracked_scene import TrackedScene",
        "",
        f"class {class_name}(TrackedScene):",
        "    def construct(self):",
        f"        self.set_layout({contract.layout_template.value!r})",
        "        objects = {}",
    ]
    for artifact in contract.artifact_inventory:
        box = region_bounds(contract.layout_template, artifact.region)
        text = (artifact.content or artifact.role or artifact.purpose)[:72]
        if artifact.category.value == "primary_visual":
            expression = (
                f"VGroup(RoundedRectangle(width={box.width * 0.88:.3f}, "
                f"height={box.height * 0.78:.3f}, corner_radius=0.12, color=BLUE), "
                f"Text({text!r}, font_size=28, color=WHITE).scale_to_fit_width({box.width * 0.75:.3f}))"
            )
        elif artifact.category.value == "annotation":
            expression = f"Text({text!r}, font_size=24, color=YELLOW)"
        elif artifact.category.value == "transient_effect":
            expression = "Circle(radius=0.35, color=YELLOW, stroke_width=6)"
        else:
            expression = f"Text({text!r}, font_size=30, color=WHITE)"
        variable = _variable(artifact.artifact_id)
        lines.extend([
            f"        {variable} = {expression}",
            f"        self.place_in_region({variable}, {artifact.region.value!r}, "
            f"padding={artifact.region_padding!r}, minimum_scale={artifact.minimum_scale!r}, "
            f"overflow_policy={artifact.overflow_policy.value!r})",
            f"        self.register_artifact({artifact.artifact_id!r}, {variable}, "
            f"category={artifact.category.value!r}, purpose={artifact.purpose!r}, "
            f"beat_ids={artifact.beat_ids!r})",
            f"        objects[{artifact.artifact_id!r}] = {variable}",
        ])
    previous: set[str] = set()
    beat_times = {beat.beat_id: max(0.25, beat.end_seconds - beat.start_seconds) for beat in scene.beats}
    for composition in contract.composition_plan:
        current = set(composition.active_artifact_ids)
        entering = sorted(current - previous)
        leaving = sorted(previous - current)
        duration = beat_times.get(composition.beat_id, 1.5)
        run_time = min(0.6, max(0.2, duration * 0.35))
        lines.append(f"        self.set_beat({composition.beat_id!r})")
        actions = [f"FadeOut(objects[{item!r}])" for item in leaving]
        actions += [f"FadeIn(objects[{item!r}])" for item in entering]
        if actions:
            lines.append(f"        self.play({', '.join(actions)}, run_time={run_time:.3f})")
            remaining = max(0.1, duration - run_time)
        else:
            remaining = max(0.1, duration)
        lines.append(f"        self.wait({remaining:.3f})")
        previous = current
    return "\n".join(lines) + "\n"


def _variable(artifact_id: str) -> str:
    value = "obj_" + re.sub(r"[^A-Za-z0-9_]", "_", artifact_id)
    return value if not keyword.iskeyword(value) else value + "_obj"
