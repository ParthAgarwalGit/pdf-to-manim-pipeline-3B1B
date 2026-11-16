"""Frame director: decides how each scene of a hybrid video is shown.

Principle: **Manim is for visualization, the presenter is for explanation.**

- ``manim``: full-screen animation. For concepts understood by *seeing*
  them: geometry, transformations, plots, worked examples, derivations.
- ``presenter``: the talking presenter alone. For motivation, transitions
  between concepts, intuition in words, recaps: where a picture adds little.
- ``split``: presenter on the left, a simple animation on the right. For
  introducing a concept that a single clear picture supports while the
  presenter explains it.

Each scene gets a visual-need score from deterministic features (beat roles,
the planned visual's richness, visual verbs in the narration). An LLM, when
configured, then reviews every decision with the scene's narration and plan;
its choice wins unless it breaks a constraint. Constraints keep Manim the
backbone (presenter + split at most ``max_explainer_share`` of the runtime)
and respect the presenter-seconds budget (D-ID bills by video seconds).
Bridges are always presenter scenes. The plan is written to frame_plan.json.
"""

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

FRAME_TYPES = ("manim", "presenter", "split")
SPOKEN_WORDS_PER_SECOND = 2.5
# Score bands: >= MANIM_THRESHOLD needs a full-screen animation; below
# PRESENTER_THRESHOLD a picture adds little; in between, split if the scene
# introduces a concept, else manim.
MANIM_THRESHOLD = 0.55
PRESENTER_THRESHOLD = 0.30

ROLE_VISUAL_NEED = {
    "derivation": 0.85,
    "example": 0.75,
    "core": 0.55,
    "intuition": 0.45,
    "analogy": 0.35,
    "recap": 0.10,
    "bridge": 0.0,
}
TEXT_PRIMITIVES = {"Text", "Tex", "MathTex", "Paragraph", "Title", "BulletedList"}
TEXT_OBJECT_TYPES = {"text", "equation", "label", "title"}
VISUAL_CUES = re.compile(
    r"(?i)\b(rotat\w*|stretch\w*|project\w*|plot\w*|graph\w*|ax[ie]s|vector\w*|ellips\w*|"
    r"circle\w*|transform\w*|geometr\w*|picture|look at|watch|shape\w*|direction\w*|angle\w*|"
    r"plane\w*|arrow\w*|grid|squish\w*|flip\w*|spin\w*|line\w* up|point\w* (?:to|at))\b"
)

SPLIT_CODEGEN_HINT = (
    "This scene is shown in the RIGHT HALF of a split screen next to the presenter "
    "(about 640x360 on a 1280x720 frame). Use at most 3-4 large, high-contrast objects, "
    "font_size >= 44 for any text, keep everything centered with generous margins, "
    "and avoid small labels or dense equations."
)


@dataclass
class FrameDecision:
    scene_id: str
    frame: str
    visual_need: float
    seconds: float
    reason: str
    source: str  # rules | llm | constraint
    rules_frame: str = ""


@dataclass
class FramePlan:
    decisions: List[FrameDecision] = field(default_factory=list)
    explainer_seconds: float = 0.0
    total_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def frame_of(self, scene_id: str) -> Optional[str]:
        return next((d.frame for d in self.decisions if d.scene_id == scene_id), None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "principle": "manim for visualization, presenter for explanation",
            "decisions": [asdict(d) for d in self.decisions],
            "explainer_seconds": round(self.explainer_seconds, 1),
            "total_seconds": round(self.total_seconds, 1),
            "explainer_share": round(self.explainer_seconds / self.total_seconds, 3) if self.total_seconds else 0.0,
            "counts": {f: sum(1 for d in self.decisions if d.frame == f) for f in FRAME_TYPES},
            "notes": list(self.notes),
        }


def _seconds(scene) -> float:
    return len((scene.narration or "").split()) / SPOKEN_WORDS_PER_SECOND


def _roles(scene) -> List[str]:
    return [getattr(b, "role", "core") or "core" for b in getattr(scene, "source_beats", []) or []]


def visual_need(scene) -> float:
    """0 (words suffice) .. 1 (must be seen), from roles, plan and narration."""
    roles = _roles(scene)
    role_score = sum(ROLE_VISUAL_NEED.get(r, 0.5) for r in roles) / len(roles) if roles else 0.5
    intent = getattr(scene, "visual_intent", None)
    objects = list(getattr(intent, "mathematical_objects", []) or [])
    transforms = list(getattr(intent, "transformations", []) or [])
    pictorial = [
        o for o in objects
        if (getattr(o, "primitive", None) or "") not in TEXT_PRIMITIVES
        and (getattr(o, "object_type", "") or "").lower() not in TEXT_OBJECT_TYPES
    ]
    intent_score = min(1.0, (len(pictorial) + len(transforms)) / 6.0)
    cue_score = min(1.0, len(VISUAL_CUES.findall(scene.narration or "")) / 4.0)
    return round(0.5 * role_score + 0.3 * intent_score + 0.2 * cue_score, 3)


class FrameDirectorAgent:
    def __init__(
        self,
        llm: Optional[Callable[[str], Dict[str, Any]]] = None,
        use_default_llm: bool = True,
        max_presenter_seconds: float = 180.0,
        max_explainer_share: float = 0.45,
        max_split_seconds: Optional[float] = None,
    ):
        if llm is None and use_default_llm:
            from src.agents.script_editor import default_llm

            llm = default_llm()
        self.llm = llm
        self.max_presenter_seconds = max_presenter_seconds
        self.max_explainer_share = max_explainer_share
        # A presenter engine with a clip-length limit (Gemini Omni: 15 s) can
        # only voice a split scene that fits one clip.
        self.max_split_seconds = max_split_seconds

    # ── rules ───────────────────────────────────────────────────────────

    @staticmethod
    def _rule(scene, need: float, index: int) -> (str, str):
        roles = _roles(scene)
        if getattr(scene, "is_presenter", False) or (roles and all(r == "bridge" for r in roles)):
            return "presenter", "Bridge between concepts: explained by the presenter."
        introduces = bool(getattr(scene, "concepts_introduced", None)) or any(
            getattr(b, "introduces_concepts", None) for b in getattr(scene, "source_beats", []) or []
        )
        if need >= MANIM_THRESHOLD:
            return "manim", f"Visual need {need:.2f}: the idea has to be seen, so full-screen animation."
        if need < PRESENTER_THRESHOLD:
            return "presenter", f"Visual need {need:.2f}: words carry it; the presenter explains."
        if introduces or index == 0:
            return "split", (
                f"Visual need {need:.2f} while introducing a concept: presenter explains beside a simple picture."
            )
        return "manim", f"Visual need {need:.2f}: a supporting animation helps more than a speaker."

    # ── LLM review ──────────────────────────────────────────────────────

    def _llm_review(self, scenes, rules: Dict[str, str], needs: Dict[str, float]) -> Dict[str, Dict[str, str]]:
        if self.llm is None:
            return {}
        listing = []
        for scene in scenes:
            intent = getattr(scene, "visual_intent", None)
            objects = [
                getattr(o, "primitive", None) or getattr(o, "object_type", "")
                for o in getattr(intent, "mathematical_objects", []) or []
            ][:8]
            listing.append(
                f"- {scene.scene_id} | roles={','.join(_roles(scene)) or 'core'} | "
                f"visual_need={needs[scene.scene_id]:.2f} | rule_choice={rules[scene.scene_id]} | "
                f"planned_visual={objects} | narration: {(scene.narration or '')[:400]}"
            )
        prompt = (
            "You direct a lecture video that mixes Manim animation with a talking presenter. "
            "Rule: Manim is for VISUALIZATION, the presenter is for EXPLANATION.\n"
            "Frame types:\n"
            "- manim: full-screen animation; for anything understood by seeing it "
            "(geometry, transformations, plots, worked examples, derivations).\n"
            "- presenter: the presenter alone; for motivation, transitions between concepts, "
            "verbal intuition and recaps, where a picture adds little.\n"
            "- split: presenter left, one simple animation right; for INTRODUCING a concept "
            "that one clear picture supports while the presenter explains.\n"
            "For each scene choose the frame type that makes it EASIEST TO UNDERSTAND. Keep Manim "
            "the backbone of the video; avoid more than two presenter-only scenes in a row.\n"
            + "\n".join(listing)
            + '\nRespond as {"scenes": [{"scene_id": "...", "frame": "manim|presenter|split", '
            '"reason": "one sentence"}]}'
        )
        try:
            data = self.llm(prompt) or {}
        except Exception as exc:
            logger.warning("Frame director LLM review failed (%s); using rules.", exc)
            return {}
        out = {}
        for item in data.get("scenes") or []:
            if not isinstance(item, dict):
                continue
            frame = str(item.get("frame", "")).lower()
            if item.get("scene_id") in rules and frame in FRAME_TYPES:
                out[item["scene_id"]] = {"frame": frame, "reason": str(item.get("reason") or "").strip()}
        return out

    # ── plan ────────────────────────────────────────────────────────────

    def plan(self, storyboard) -> FramePlan:
        scenes = list(getattr(storyboard, "scenes", []) or [])
        needs = {s.scene_id: visual_need(s) for s in scenes}
        rules = {}
        rule_reasons = {}
        for i, scene in enumerate(scenes):
            rules[scene.scene_id], rule_reasons[scene.scene_id] = self._rule(scene, needs[scene.scene_id], i)
        review = self._llm_review(scenes, rules, needs)

        plan = FramePlan(total_seconds=sum(_seconds(s) for s in scenes))
        for scene in scenes:
            sid = scene.scene_id
            bridge = rules[sid] == "presenter" and rule_reasons[sid].startswith("Bridge")
            if sid in review and not bridge:
                frame, reason, source = review[sid]["frame"], review[sid]["reason"] or rule_reasons[sid], "llm"
            else:
                frame, reason, source = rules[sid], rule_reasons[sid], "rules"
            plan.decisions.append(FrameDecision(
                scene_id=sid, frame=frame, visual_need=needs[sid], seconds=round(_seconds(scene), 1),
                reason=reason, source=source, rules_frame=rules[sid],
            ))
        self._enforce_split_length(plan)
        self._enforce_budget(plan)
        plan.explainer_seconds = sum(d.seconds for d in plan.decisions if d.frame != "manim")
        return plan

    def _enforce_split_length(self, plan: FramePlan) -> None:
        if not self.max_split_seconds:
            return
        for decision in plan.decisions:
            if decision.frame == "split" and decision.seconds > self.max_split_seconds:
                decision.frame, decision.source = "manim", "constraint"
                decision.reason = (
                    f"Was split; {decision.seconds:.0f}s of narration is longer than the presenter "
                    f"clip can hold in a split ({self.max_split_seconds:.0f}s), so shown as manim."
                )
                plan.notes.append(f"{decision.scene_id}: split -> manim (clip length)")

    def _enforce_budget(self, plan: FramePlan) -> None:
        """Demote the most visual explainer scenes to manim until the
        presenter seconds and the explainer share fit. Bridges stay."""
        cap = min(self.max_presenter_seconds, self.max_explainer_share * plan.total_seconds)

        def explainer_seconds() -> float:
            return sum(d.seconds for d in plan.decisions if d.frame != "manim")

        candidates = sorted(
            (d for d in plan.decisions if d.frame != "manim" and not d.reason.startswith("Bridge")),
            key=lambda d: (d.frame != "split", -d.visual_need),  # splits first, most visual first
        )
        for decision in candidates:
            if explainer_seconds() <= cap:
                break
            old = decision.frame
            decision.frame = "manim"
            decision.source = "constraint"
            decision.reason = (
                f"Was {old}; shown as manim to keep presenter time within "
                f"{cap:.0f}s (visual need {decision.visual_need:.2f})."
            )
            plan.notes.append(f"{decision.scene_id}: {old} -> manim (presenter budget)")
        if explainer_seconds() > cap:
            plan.notes.append(
                f"Bridges alone use {explainer_seconds():.0f}s of presenter time (cap {cap:.0f}s)."
            )


def apply_frame_plan(storyboard, plan: FramePlan) -> None:
    """Write the decisions onto the storyboard's scenes.

    Presenter scenes become ``scene_type='presenter'`` (no Manim code);
    split scenes stay Manim scenes and are composited after rendering.
    """
    by_id = {d.scene_id: d for d in plan.decisions}
    for scene in getattr(storyboard, "scenes", []) or []:
        decision = by_id.get(scene.scene_id)
        if decision is None:
            continue
        scene.frame_type = decision.frame
        scene.frame_reason = decision.reason
        if decision.frame == "presenter":
            scene.scene_type = "presenter"
        elif scene.scene_type == "presenter" and decision.frame != "presenter":
            scene.scene_type = "manim"
