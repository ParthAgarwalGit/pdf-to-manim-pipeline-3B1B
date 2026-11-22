"""Look at a rendered scene the way a viewer does.

The trace-based verifier measures object bounds at animation checkpoints, but
its pixel masks are too coarse to judge a label crossed by a line, and its
frames are taken at beat ends, so overlaps that happen while things move go
unseen. This check samples frames through the whole scene, mid-animation
included, and asks a vision model for the defects a viewer would notice.

A first look at small frames over the whole scene often names defects that
are not there (measured: about half of them on a 22-minute lecture), so each
one is confirmed on its clearest frame at full size before it counts. Each
confirmed defect becomes an error finding with its time and frame, so it goes
to the vision repair with the evidence attached, and the repaired preview is
checked the same way.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.utils.ffmpeg_finder import get_ffmpeg

from .model_config import TEXT_MODEL
from .models import VerificationFinding
from .prompt_text import visual_qa_confirm_instructions, visual_qa_instructions
from .providers import clean_json_text

VISUAL_QA_PROMPT_VERSION = "visual-qa-3-confirm-across-frames"
# A frame every 1.5 s catches anything that stays visible for a beat; long
# scenes are sampled more sparsely so one request stays a reasonable size.
VISUAL_QA_INTERVAL_SECONDS = 1.5
VISUAL_QA_MAX_FRAMES = 48
VISUAL_QA_FRAME_WIDTH = 640
CONFIRM_FRAME_WIDTH = 1280
VISUAL_QA_OUTPUT_TOKENS = 6000
CONFIRM_OUTPUT_TOKENS = 1500
# At most this many reported defects are confirmed per look; the rest are
# reported as unconfirmed warnings so a flood of guesses cannot run up cost.
MAX_CONFIRMATIONS = 8
# A defect is confirmed on its clearest frame and on the first and last frames
# it was seen in: a moving overlap can be plain in one and gone in another
# (a real box-through-label overlap was rejected on the one frame it got).
CONFIRM_MAX_FRAMES = 3
FINDING_CODES = {
    "text_overlap": ("VISUAL_TEXT_OVERLAP", "error"),
    "line_through_text": ("VISUAL_TEXT_OVERLAP", "error"),
    "cut_off": ("VISUAL_CUT_OFF", "error"),
    "pile_up": ("VISUAL_CROWDED", "warning"),
}
UNCONFIRMED_CODE = "VISUAL_UNCONFIRMED"
# Confirmed defects a viewer would see: they send a scene to repair, and one
# still there in the final video marks the run for review.
VISUAL_ERROR_CODES = frozenset(code for code, severity in FINDING_CODES.values() if severity == "error")
VISUAL_CODES = frozenset(code for code, _ in FINDING_CODES.values())


class VisualCheckUnreadable(RuntimeError):
    """The first look's answer could not be read, so the scene is not recorded as clean."""


@dataclass
class Defect:
    kind: str
    images: list[int]
    best: int
    what: str


def qa_timestamps(duration: float) -> list[float]:
    """Evenly spaced sample times through the scene, at most VISUAL_QA_MAX_FRAMES."""

    if duration <= 0.5:
        return [max(0.0, duration / 2)]
    interval = max(VISUAL_QA_INTERVAL_SECONDS, duration / VISUAL_QA_MAX_FRAMES)
    # Each frame stands in the middle of its own interval, through to the end.
    count = min(VISUAL_QA_MAX_FRAMES, max(1, math.floor(duration / interval + 0.5)))
    return [round(interval * (index + 0.5), 2) for index in range(count)]


def extract_frame(video: Path, seconds: float, path: Path, width: int) -> Optional[str]:
    result = subprocess.run(
        [get_ffmpeg(), "-v", "error", "-y", "-ss", f"{seconds:.3f}", "-i", str(video),
         "-frames:v", "1", "-vf", f"scale={width}:-2", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    return str(path) if result.returncode == 0 and path.is_file() else None


def parse_defects(data: Any, image_count: int) -> list[Defect]:
    """The first look's defects, with image numbers checked and a clearest image chosen."""

    defects = data.get("defects") if isinstance(data, dict) else None
    parsed: list[Defect] = []
    for item in defects if isinstance(defects, list) else []:
        if not isinstance(item, dict) or item.get("kind") not in FINDING_CODES:
            continue
        images = sorted({
            int(value) for value in item.get("images") or []
            if isinstance(value, (int, float)) and 1 <= int(value) <= image_count
        })
        best = item.get("best")
        best = int(best) if isinstance(best, (int, float)) and 1 <= int(best) <= image_count else None
        if best is not None and best not in images:
            images = sorted({*images, best})
        if not images:
            continue
        parsed.append(Defect(
            kind=item["kind"], images=images, best=best or images[len(images) // 2],
            what=" ".join(str(item.get("what") or item["kind"]).split())[:300],
        ))
    return parsed


def confirm_images(defect: Defect) -> list[int]:
    """Image numbers for the close look: the clearest, then the first and last seen."""

    return list(dict.fromkeys([defect.best, defect.images[0], defect.images[-1]]))[:CONFIRM_MAX_FRAMES]


def to_finding(defect: Defect, timestamps: list[float], screenshot: Optional[str], confirmed: bool) -> VerificationFinding:
    code, severity = FINDING_CODES[defect.kind]
    first, last = timestamps[defect.images[0] - 1], timestamps[defect.images[-1] - 1]
    when = f"{first:.1f}s" if first == last else f"{first:.1f}-{last:.1f}s"
    return VerificationFinding(
        code=code if confirmed else UNCONFIRMED_CODE,
        severity=severity if confirmed else "warning",
        message=f"Seen at {when}, clearest at {timestamps[defect.best - 1]:.1f}s "
                f"({defect.kind.replace('_', ' ')}): {defect.what}",
        timestamp=timestamps[defect.best - 1],
        screenshot_path=screenshot,
    )


async def _ask(router: Any, *, scene_id: str, task_id: str, prompt: str, images: list[str], tokens: int) -> Any:
    """The model's JSON answer, or None when it cannot be read twice.

    The model's thinking shares the output budget, so an answer can be cut
    short; the second request has twice the room (and is not served from
    the provider cache, whose key includes the budget).
    """

    for budget in (tokens, tokens * 2):
        response = await router.vision(
            stage="visual_qa", task_id=task_id, scene_id=scene_id, model=TEXT_MODEL,
            system="You are a careful visual quality checker for educational animations.",
            prompt=prompt, image_paths=[Path(path) for path in images], max_output_tokens=budget,
        )
        try:
            return json.loads(clean_json_text(response.text))
        except (ValueError, TypeError):
            continue
    return None


async def review_frames(
    router: Any,
    *,
    scene_id: str,
    video_path: Path,
    duration: float,
    out_dir: Path,
    task_id: str,
) -> list[VerificationFinding]:
    """Sample the scene, find visible layout defects, confirm each, return them as findings."""

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamps, frames = [], []
    for index, seconds in enumerate(qa_timestamps(duration)):
        path = extract_frame(video_path, seconds, out_dir / f"qa-{index:03d}.png", VISUAL_QA_FRAME_WIDTH)
        if path:
            timestamps.append(seconds)
            frames.append(path)
    if not frames:
        return []
    data = await _ask(router, scene_id=scene_id, task_id=task_id, prompt=visual_qa_instructions(timestamps),
                      images=frames, tokens=VISUAL_QA_OUTPUT_TOKENS)
    if not isinstance(data, dict) or not isinstance(data.get("defects"), list):
        # Never "clean" by default: the caller records a failed check instead.
        raise VisualCheckUnreadable("The visual check's answer could not be read.")
    findings: list[VerificationFinding] = []
    for number, defect in enumerate(parse_defects(data, len(frames))):
        if number >= MAX_CONFIRMATIONS:
            findings.append(to_finding(defect, timestamps, frames[defect.best - 1], confirmed=False))
            continue
        picks = confirm_images(defect)
        shown = [
            extract_frame(video_path, timestamps[image - 1],
                          out_dir / f"confirm-{number:02d}-{index}.png", CONFIRM_FRAME_WIDTH)
            or frames[image - 1]
            for index, image in enumerate(picks)
        ]
        answer = await _ask(
            router, scene_id=scene_id, task_id=f"{task_id}-confirm-{number:02d}",
            prompt=visual_qa_confirm_instructions(defect.kind, defect.what, len(shown)),
            images=shown, tokens=CONFIRM_OUTPUT_TOKENS,
        )
        # Only a clear "not visible" clears a reported defect; an answer that
        # cannot be read keeps it.
        confirmed = not (isinstance(answer, dict) and answer.get("visible") is False)
        if confirmed and not (isinstance(answer, dict) and answer.get("visible") is True):
            defect.what = f"{defect.what} (kept: the close look could not be read)"
        seen_in = answer.get("frame") if isinstance(answer, dict) else None
        index = int(seen_in) - 1 if isinstance(seen_in, (int, float)) and 1 <= int(seen_in) <= len(shown) else 0
        defect.best = picks[index]  # the frame the close look saw it in
        findings.append(to_finding(defect, timestamps, shown[index], confirmed))
    return findings
