"""Artifact-contract-driven raw Manim generation and targeted repair."""

from __future__ import annotations

import json
import keyword
import re
from pathlib import Path

from .layout import layout_prompt_guide, region_bounds
from .models import NarratedScene, SourceFact, VerificationFinding, VisualContract
from .providers import MissingProviderKey, ProviderRouter


CODE_PROMPT_VERSION = "v2-code-3"


class CodeGenerationService:
    def __init__(self, router: ProviderRouter):
        self.router = router

    async def generate(
        self,
        scene: NarratedScene,
        contract: VisualContract,
        source_facts: list[SourceFact],
    ) -> str:
        class_name = class_name_for(scene.scene_id)
        payload = {
            "scene": scene.model_dump(mode="json"),
            "visual_contract": contract.model_dump(mode="json"),
            "source_facts": [fact.model_dump(mode="json") for fact in source_facts],
        }
        system = f"""PROMPT_VERSION={CODE_PROMPT_VERSION}
You write one complete executable Manim Community Python script.
Use raw Manim creatively, but the class must be named {class_name} and inherit TrackedScene.
Allowed imports are exactly:
from manim import *
import numpy as np
from src.pipeline_v2.tracked_scene import TrackedScene

For every contract artifact, construct one top-level mobject and immediately call:
self.register_artifact("artifact_id", mobject, category="...", purpose="...", beat_ids=[...])
Call self.set_beat("beat_id") before animating every beat. The composition_plan is authoritative:
objects must enter and leave so the active set at each checkpoint matches it. Text uses reserved
regions and must stay inside x=+/-6.9, y=+/-3.8. Call self.set_layout("layout_template") once before
creating artifacts. After composing each top-level artifact, call self.place_in_region(mobject,
"region", padding=region_padding, minimum_scale=minimum_scale,
overflow_policy="overflow_policy"). Do not create unregistered visible mobjects.
Prefer relational layout with arrange(), next_to(), align_to(), and to_edge() over guessed text
coordinates. Keep unrelated text boxes separated by at least 0.2 Manim units at stable states.
VGroup children are traced individually, so grouping objects does not hide internal collisions.
Use axes.c2p() for data coordinates. Preserve source mathematics exactly. Match measured audio
duration without empty padding. Return only the complete Python script, never a patch or markdown.

{layout_prompt_guide()}
"""
        try:
            response = await self.router.text(
                stage="code_generation", task_id=f"code-{scene.scene_id}", scene_id=scene.scene_id,
                provider="google", model="gemini-3.7-flash", system=system,
                prompt=json.dumps(payload, ensure_ascii=False), max_output_tokens=9000,
            )
            return clean_code(response.text)
        except MissingProviderKey:
            if self.router.strict_keys:
                raise
            code = fallback_code(scene, contract)
            self.router.artifacts.record_provider_call(
                stage="code_generation", call_id=f"code-local-{scene.scene_id}",
                scene_id=scene.scene_id,
                request={
                    "provider": "google", "model": "gemini-3.7-flash",
                    "system": system, "prompt": payload,
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
    ) -> str:
        system = f"""PROMPT_VERSION={CODE_PROMPT_VERSION}
You repair one complete raw-Manim TrackedScene using deterministic evidence.
Make the smallest changes that fix the reported issues. Preserve mathematical meaning, artifact IDs,
narration timing, and correct visuals. Return one complete executable Python script only, not a patch.
Keep self.set_layout() and place every top-level artifact with self.place_in_region() using its contract
region and fit policy. If an artifact cannot fit above minimum_scale, simplify its internal composition
or distribute the explanation across beats. Do not claim success; the pipeline will re-render and score
the replacement objectively.

{layout_prompt_guide()}"""
        prompt = json.dumps({
            "scene": scene.model_dump(mode="json"),
            "visual_contract": contract.model_dump(mode="json"),
            "findings": [item.model_dump(mode="json") for item in findings],
            "screenshots": frame_paths,
            "existing_code": code,
        }, ensure_ascii=False)
        response = await self.router.vision(
            stage="vision_repair", task_id=f"repair-{scene.scene_id}", scene_id=scene.scene_id,
            model="gemini-3.7-flash", system=system, prompt=prompt,
            image_paths=[Path(path) for path in frame_paths[:8]], max_output_tokens=9000,
        )
        return clean_code(response.text)


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
