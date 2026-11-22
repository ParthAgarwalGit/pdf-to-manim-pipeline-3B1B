"""Comprehension verifier: watches rendered hybrid scenes and checks that
each one uses the frame type that is easiest to understand.

For a split scene both renderings exist (the split composite and the plain
Manim render), so the verifier runs a true A/B comparison: frames of both,
with the narration, go to a vision model that picks the easier one. If the
full-screen animation wins (or the half-size visual is not legible), the
scene is switched to manim; that costs nothing. For presenter and manim
scenes there is no free alternative, so the verifier judges the rendered
frames and records a recommendation in frame_review.json.
"""

import base64
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.prompts.review import FRAME_ISSUE_CODES, frame_issue_instructions
from src.utils.ffmpeg_finder import get_ffmpeg

logger = logging.getLogger(__name__)

VERIFIER_MODEL = "gpt-5.4-mini"
FRAME_POSITIONS = (0.35, 0.75)  # sampled points within each scene
FRAME_WIDTH = 640

Vision = Callable[[str, List[str]], Dict[str, Any]]

_DEFINITIONS = (
    "Frame types in this lecture video (rule: Manim is for VISUALIZATION, the presenter "
    "is for EXPLANATION):\n"
    "- manim: full-screen animation, for ideas understood by seeing them.\n"
    "- presenter: a talking presenter alone, for motivation, transitions and verbal explanation.\n"
    "- split: presenter on the left, a simple animation on the right, for introducing a concept.\n"
)


@dataclass
class FrameVerdict:
    scene_id: str
    chosen: str
    easiest: str
    legible: Optional[bool]
    reason: str
    action: str  # keep | switch_to_manim | recommend
    compared: bool = False
    issues: List[str] = field(default_factory=list)


def default_vision() -> Optional[Vision]:
    """JSON-mode OpenAI vision call, or None without a key."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    import openai

    client = openai.OpenAI(max_retries=3)

    def call(prompt: str, images: List[str]) -> Dict[str, Any]:
        from src.agents.script_editor import ensure_json_instruction

        content: List[Dict[str, Any]] = [{"type": "text", "text": ensure_json_instruction(prompt)}]
        for path in images:
            data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
        response = client.chat.completions.create(
            model=VERIFIER_MODEL,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    return call


def sample_frames(video: str, out_dir: Path, label: str) -> List[str]:
    """Stills at FRAME_POSITIONS of the clip, scaled to FRAME_WIDTH."""
    from src.utils.media_probe import probe

    info = probe(video)
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    frames = []
    for i, position in enumerate(FRAME_POSITIONS):
        target = out_dir / f"{label}_{i}.png"
        subprocess.run(
            [get_ffmpeg(), "-v", "error", "-y", "-ss", f"{max(0.0, duration * position):.2f}", "-i", video,
             "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-2", str(target)],
            capture_output=True, timeout=60, stdin=subprocess.DEVNULL,
        )
        if target.is_file():
            frames.append(str(target))
    return frames


class FrameVerifierAgent:
    def __init__(self, vision: Optional[Vision] = None, use_default_vision: bool = True):
        self.vision = vision if vision is not None else (default_vision() if use_default_vision else None)

    def review(self, scenes: List[Dict[str, Any]]) -> List[FrameVerdict]:
        """``scenes``: dicts with scene_id, frame, narration, video (rendered
        clip) and, for split scenes, manim_video (the plain render)."""
        if self.vision is None:
            logger.info("Frame verifier skipped: no vision model configured.")
            return []
        verdicts = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for scene in scenes:
                try:
                    verdicts.append(self._review_one(scene, tmp_path))
                except Exception as exc:  # never fatal: the plan stands
                    logger.warning("Frame verifier could not review %s: %s", scene.get("scene_id"), exc)
        return verdicts

    def _review_one(self, scene: Dict[str, Any], tmp: Path) -> FrameVerdict:
        sid, chosen = scene["scene_id"], scene["frame"]
        narration = (scene.get("narration") or "")[:700]
        frames = sample_frames(scene["video"], tmp, f"{sid}_a")
        alternative = scene.get("manim_video") if chosen == "split" else None
        if alternative and Path(alternative).is_file():
            frames_b = sample_frames(alternative, tmp, f"{sid}_b")
            prompt = (
                _DEFINITIONS
                + f"\nNarration of this scene: \"{narration}\"\n"
                f"The first {len(frames)} image(s) are option A (split: presenter left, animation right); "
                f"the next {len(frames_b)} are option B (the same animation full-screen, narration as "
                "voice-over). Which option makes this scene EASIER TO UNDERSTAND for a student? Judge "
                "whether the animation is legible in the half-size split, and whether seeing the "
                "presenter adds anything here.\n"
                'Respond as {"easiest": "A" or "B", "legible_in_split": true|false, "reason": "one sentence"}'
            )
            data = self.vision(prompt, frames + frames_b) or {}
            pick = str(data.get("easiest", "")).strip().upper()
            easiest = "manim" if pick == "B" else "split"
            legible = data.get("legible_in_split")
            legible = legible if isinstance(legible, bool) else None
            action = "switch_to_manim" if easiest == "manim" or legible is False else "keep"
            return FrameVerdict(sid, chosen, easiest, legible, str(data.get("reason") or ""), action, compared=True)

        prompt = (
            _DEFINITIONS
            + f"\nNarration of this scene: \"{narration}\"\n"
            f"The images are stills of the scene as rendered, frame type '{chosen}'. Which frame type "
            "would make this scene EASIEST TO UNDERSTAND? Is any on-screen visual legible?\n"
            'Respond as {"easiest": "manim|presenter|split", "legible": true|false, "reason": "one sentence"}'
        ) + frame_issue_instructions()
        data = self.vision(prompt, frames) or {}
        issues = [str(c) for c in (data.get("issues") or []) if str(c) in FRAME_ISSUE_CODES]
        easiest = str(data.get("easiest", chosen)).strip().lower()
        if easiest not in ("manim", "presenter", "split"):
            easiest = chosen
        legible = data.get("legible") if isinstance(data.get("legible"), bool) else None
        action = "keep" if easiest == chosen else "recommend"
        return FrameVerdict(
            sid, chosen, easiest, legible, str(data.get("reason") or ""), action, issues=issues,
        )


def verdicts_report(verdicts: List[FrameVerdict]) -> Dict[str, Any]:
    return {
        "model": VERIFIER_MODEL,
        "verdicts": [asdict(v) for v in verdicts],
        "agreed": sum(1 for v in verdicts if v.action == "keep"),
        "switched": [v.scene_id for v in verdicts if v.action == "switch_to_manim"],
        "recommended": {v.scene_id: v.easiest for v in verdicts if v.action == "recommend"},
        "issues": {v.scene_id: v.issues for v in verdicts if v.issues},
    }
