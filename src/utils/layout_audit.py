"""Render-time layout audit: what actually overlaps on screen.

The planner's boxes are nudged apart before codegen, but generated code
places objects its own way, so overlaps and off-frame text only exist in the
rendered scene. This module runs inside the Manim subprocess: after every
``play`` and ``wait`` it measures the visible text-like mobjects and records

- ``overlaps``: two texts whose boxes intersect by more than OVERLAP_RATIO of
  the smaller one (labels on top of labels, formulas over titles);
- ``off_frame``: text extending past the frame edge by more than FRAME_SLACK;
- ``long_text``: prose shown as text that is long enough to be a sentence
  (LONG_TEXT_WORDS or more words) rather than a label;
- ``fill``: the largest share of the frame the picture ever reaches, and how
  many times the scene was measured.

It is switched on by ``PDF2MANIM_LAYOUT_AUDIT=<json path>`` through the
``layout_audit_site`` sitecustomize that the native renderer puts on the
subprocess PYTHONPATH, so it covers every scene without editing its code.
The renderer reads the JSON back (``read_audit``) and the pipeline
regenerates offending scenes with the findings as feedback.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

ENV_VAR = "PDF2MANIM_LAYOUT_AUDIT"
SITE_DIR = Path(__file__).resolve().parent / "layout_audit_site"
OVERLAP_RATIO = 0.12   # of the smaller box's area
FRAME_SLACK = 0.05     # scene units past the frame edge
MAX_FINDINGS = 12
LONG_TEXT_WORDS = 7    # a Text with this many words is a sentence, not a label
MIN_FILL = 0.30        # share of the frame the picture should reach at least once
MIN_FILL_OBSERVATIONS = 3
# Tex (but not its parent MathTex) is prose too: Title and BulletedList are built on it.
_PROSE_CLASSES = ("Text", "MarkupText", "Paragraph", "Tex", "Title", "BulletedList")

_TEXT_CLASSES = (
    "Text", "MarkupText", "Paragraph", "Tex", "MathTex", "SingleStringMathTex",
    "DecimalNumber", "Integer", "Title", "BulletedList",
)


def _is_text(mob) -> bool:
    return any(cls.__name__ in _TEXT_CLASSES for cls in type(mob).__mro__)


def _label(mob) -> str:
    raw = (
        getattr(mob, "original_text", None) or getattr(mob, "text", None)
        or getattr(mob, "tex_string", None) or type(mob).__name__
    )
    return " ".join(str(raw).split())[:48]


def _visible(mob) -> bool:
    try:
        family = mob.get_family()
        return any(
            (m.get_fill_opacity() if hasattr(m, "get_fill_opacity") else 1.0) > 0.05
            or (m.get_stroke_opacity() if hasattr(m, "get_stroke_opacity") else 0.0) > 0.05
            for m in family
        )
    except Exception:
        return True


def _box(mob):
    try:
        left, right = float(mob.get_left()[0]), float(mob.get_right()[0])
        bottom, top = float(mob.get_bottom()[1]), float(mob.get_top()[1])
    except Exception:
        return None
    if right - left <= 1e-6 or top - bottom <= 1e-6:
        return None
    return left, right, bottom, top


def visible_texts(mobjects) -> List[Any]:
    """Outermost visible text-like mobjects (a MathTex counts once, not per glyph)."""
    found: List[Any] = []

    def walk(mob):
        if _is_text(mob):
            if _visible(mob):
                found.append(mob)
            return
        for sub in getattr(mob, "submobjects", []) or []:
            walk(sub)

    for mob in mobjects:
        walk(mob)
    return found


def _raw_text(mob) -> str:
    # Tex-based prose (Tex, Title, BulletedList) keeps its words in tex_string only.
    raw = (
        getattr(mob, "original_text", None) or getattr(mob, "text", None)
        or getattr(mob, "tex_string", None) or ""
    )
    return " ".join(str(raw).split())


def long_texts(mobjects) -> List[Dict[str, Any]]:
    """Visible prose mobjects (not formulas) long enough to be a sentence."""
    found = []
    for mob in visible_texts(mobjects):
        if not any(cls.__name__ in _PROSE_CLASSES for cls in type(mob).__mro__):
            continue
        words = len(_raw_text(mob).split())
        if words >= LONG_TEXT_WORDS:
            found.append({"text": _label(mob), "words": words})
    return found


def content_fill(mobjects, frame_width: float, frame_height: float) -> float:
    """Share of the frame covered by the bounding box of everything visible (0..1)."""
    boxes = [_box(m) for m in mobjects if _visible(m)]
    boxes = [b for b in boxes if b is not None]
    if not boxes or frame_width <= 0 or frame_height <= 0:
        return 0.0
    hw, hh = frame_width / 2, frame_height / 2
    left = max(-hw, min(b[0] for b in boxes))
    right = min(hw, max(b[1] for b in boxes))
    bottom = max(-hh, min(b[2] for b in boxes))
    top = min(hh, max(b[3] for b in boxes))
    if right <= left or top <= bottom:
        return 0.0
    return round((right - left) * (top - bottom) / (frame_width * frame_height), 3)


def check(mobjects, frame_width: float, frame_height: float) -> Dict[str, List[Dict[str, Any]]]:
    """Overlapping and off-frame text among ``mobjects`` (pure geometry)."""
    texts = [(m, _box(m)) for m in visible_texts(mobjects)]
    texts = [(m, b) for m, b in texts if b is not None]
    overlaps, off_frame = [], []
    for i in range(len(texts)):
        a, (al, ar, ab, at) = texts[i]
        for j in range(i + 1, len(texts)):
            b, (bl, br, bb, bt) = texts[j]
            w = min(ar, br) - max(al, bl)
            h = min(at, bt) - max(ab, bb)
            if w <= 0 or h <= 0:
                continue
            smaller = min((ar - al) * (at - ab), (br - bl) * (bt - bb))
            ratio = (w * h) / smaller if smaller > 0 else 0.0
            if ratio > OVERLAP_RATIO:
                overlaps.append({"a": _label(a), "b": _label(b), "ratio": round(ratio, 2)})
        hw, hh = frame_width / 2 + FRAME_SLACK, frame_height / 2 + FRAME_SLACK
        if al < -hw or ar > hw or ab < -hh or at > hh:
            off_frame.append({"text": _label(a), "box": [round(v, 2) for v in (al, ar, ab, at)]})
    return {"overlaps": overlaps, "off_frame": off_frame}


class _Recorder:
    def __init__(self, path: str):
        self.path = path
        self.overlaps: Dict[tuple, Dict[str, Any]] = {}
        self.off_frame: Dict[str, Dict[str, Any]] = {}
        self.long_text: Dict[str, Dict[str, Any]] = {}
        self.max_fill = 0.0
        self.observations = 0

    def observe(self, scene) -> None:
        try:
            from manim import config

            found = check(scene.mobjects, config.frame_width, config.frame_height)
            long = long_texts(scene.mobjects)
            fill = content_fill(scene.mobjects, config.frame_width, config.frame_height)
        except Exception:
            return
        t = round(float(getattr(getattr(scene, "renderer", None), "time", 0.0) or 0.0), 2)
        self.observations += 1
        self.max_fill = max(self.max_fill, fill)
        for item in long:
            self.long_text.setdefault(item["text"], {**item, "t": t})
        for item in found["overlaps"]:
            key = tuple(sorted((item["a"], item["b"])))
            if key not in self.overlaps or item["ratio"] > self.overlaps[key]["ratio"]:
                self.overlaps[key] = {**item, "t": t}
        for item in found["off_frame"]:
            self.off_frame.setdefault(item["text"], {**item, "t": t})

    def dump(self) -> None:
        overlaps = sorted(self.overlaps.values(), key=lambda x: -x["ratio"])[:MAX_FINDINGS]
        off = list(self.off_frame.values())[:MAX_FINDINGS]
        Path(self.path).write_text(json.dumps({
            "overlaps": overlaps,
            "off_frame": off,
            "long_text": list(self.long_text.values())[:MAX_FINDINGS],
            "fill": {"max": round(self.max_fill, 3), "observations": self.observations},
        }, indent=2), encoding="utf-8")


def install(path: Optional[str] = None) -> bool:
    """Patch manim's Scene so every play/wait is audited; dump on tear_down."""
    path = path or os.environ.get(ENV_VAR)
    if not path:
        return False
    try:
        from manim import Scene
    except Exception:
        return False
    if getattr(Scene, "_pdf2manim_audited", False):
        return True
    recorder = _Recorder(path)
    original_play, original_wait, original_tear_down = Scene.play, Scene.wait, Scene.tear_down

    def play(self, *args, **kwargs):
        result = original_play(self, *args, **kwargs)
        recorder.observe(self)
        return result

    def wait(self, *args, **kwargs):
        result = original_wait(self, *args, **kwargs)
        recorder.observe(self)
        return result

    def tear_down(self, *args, **kwargs):
        try:
            recorder.observe(self)
            recorder.dump()
        finally:
            return original_tear_down(self, *args, **kwargs)

    Scene.play, Scene.wait, Scene.tear_down = play, wait, tear_down
    Scene._pdf2manim_audited = True
    return True


def read_audit(path: Path) -> List[str]:
    """Findings as plain sentences (empty when the audit found nothing)."""
    path = Path(path)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    issues = [
        f"Text \"{o['a']}\" overlaps text \"{o['b']}\" ({int(o['ratio'] * 100)}% of the smaller one) at {o.get('t', 0)}s."
        for o in data.get("overlaps") or []
    ]
    issues += [
        f"Text \"{o['text']}\" runs off the frame edge at {o.get('t', 0)}s."
        for o in data.get("off_frame") or []
    ]
    issues += [
        f"Text \"{o['text']}\" is a {o.get('words', '?')}-word sentence on screen at "
        f"{o.get('t', 0)}s; on-screen text must be a label of at most four words."
        for o in data.get("long_text") or []
    ]
    fill = data.get("fill")
    if (
        isinstance(fill, dict)
        and fill.get("observations", 0) >= MIN_FILL_OBSERVATIONS
        and float(fill.get("max", 1.0)) < MIN_FILL
    ):
        issues.append(
            f"The picture never covers more than {int(float(fill['max']) * 100)}% of the frame; "
            "the main visual must be much larger."
        )
    return issues


def audit_feedback(issues: List[str]) -> List[str]:
    """Codegen feedback for a scene whose render had layout problems."""
    return [
        "The rendered scene had these layout problems; fix every one of them:",
        *[f"- {issue}" for issue in issues[:MAX_FINDINGS]],
        "Keep all text inside the frame (titles at most about 40 characters per line, "
        "font_size at most 36). Never place a label or formula on top of other text: use "
        "LayoutTracker.place_label for labels, arrange stacked text with .arrange(DOWN, buff=0.3), "
        "and FadeOut text before putting new text in the same place.",
        "Remove every sentence, title or caption from the screen: the narrator says it. "
        "On-screen text is a label of at most four words, or a formula.",
        "Make the main visual fill the frame: a coordinate system with x_length 9 to 11 and "
        "y_length 5.5 to 6.5, or scale the main group with scale_to_fit_height(5) and centre it.",
    ]
