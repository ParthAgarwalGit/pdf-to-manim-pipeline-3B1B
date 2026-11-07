"""Script/Storyboard Architect Agent for scene generation

This module implements the Script/Storyboard Architect Agent which generates
pedagogically sound scene sequences with narration and visual intent from OCR output.

Note: This implementation uses mock/stub implementations for LLM calls
since we don't have actual API keys. The focus is on agent structure,
concept extraction, dependency graph building, and data flow.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import openai


from src.agents.base import (
    Agent,
    AgentExecutionError,
    AgentInput,
    AgentOutput,
    BackoffStrategy,
    RetryPolicy,
)
from src.agents.document_segmenter import DocumentSegmenterAgent, DocumentSegmenterInput
from src.schemas.ocr_output import OCROutput
from src.schemas.source_segment import SourceSegment
from src.schemas.video_script import MAX_BEAT_DEPTH, Analogy
from src.schemas.storyboard import (
    Configuration,
    ConceptHierarchy,
    EmphasisPoint,
    MathematicalObject,
    ObjectStyle,
    PDFMetadataStoryboard,
    Scene,
    SceneState,
    StoryboardJSON,
    Transformation,
    VisualBeat,
    VisualIntent,
)
from src.utils.fingerprints import beat_fingerprint, beats_fingerprint
from src.utils.manim_engine_contract import build_visual_intent_contract_prompt, CANONICAL_RENDER_HINTS, normalize_render_hint
from src.utils.manim_primitives import catalog_for_prompt
from src.utils.visual_archetypes import archetype_prompt_block, classify_concept

# Per-task model tiering (see docs). The unified whole-document script — one call
# per document that sets breadth coverage — runs on Claude Sonnet 5 (best
# reasoning + instruction-following per dollar). The high-volume per-segment and
# visual-intent calls run on the cheap gpt-5.4-mini tier: Pydantic schema
# enforcement + retry are a safety net, so raw capability matters less there.
UNIFIED_SCRIPT_MODEL = "claude-sonnet-5"   # Anthropic
SEGMENT_SCRIPT_MODEL = "gpt-5.4-mini"      # OpenAI
VISUAL_INTENT_MODEL = "gpt-5.4-mini"       # OpenAI

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# F1 mitigation — concept-label canonicalization
#
# The script LLM might emit "sine" in beat 1 and "Sine Function" in beat 4.
# Without normalization both pass through the ledger as distinct concepts
# and the dedup rule loses force. The alias map covers the common drift
# patterns we have seen on real PDFs; extend conservatively.
# ─────────────────────────────────────────────────────────────────────────

_CONCEPT_ALIASES = {
    "sin": "sine",
    "cos": "cosine",
    "tan": "tangent",
    "sine_function": "sine",
    "cosine_function": "cosine",
    "tangent_function": "tangent",
    "trig_function": "trigonometric_function",
    "trig_functions": "trigonometric_function",
    "trigonometric_functions": "trigonometric_function",
    "unit_circ": "unit_circle",
    "the_unit_circle": "unit_circle",
}


def _canonicalize_concept(label: str) -> str:
    """Return a normalized canonical form of a concept label.

    Lowercases, strips, replaces internal whitespace and dashes with
    underscores, and applies the alias map.
    """
    if not label:
        return ""
    s = label.strip().lower().replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return _CONCEPT_ALIASES.get(s, s)


# ─────────────────────────────────────────────────────────────────────────
# Typed beats — role / depth / analogy tags from the script LLM. They let a
# professor edit "the pizza analogy" or "the optional depth" instead of raw
# beat ids. "bridge" beats belong to the presenter layer, never this LLM.
# ─────────────────────────────────────────────────────────────────────────

_LLM_BEAT_ROLES = frozenset({"core", "analogy", "example", "derivation", "intuition", "recap"})

TYPED_BEAT_PROMPT_RULES = """BEAT TAGS:
- "role": core | analogy | example | derivation | intuition | recap.
  core = the essential argument; analogy = introduces or calls back an analogy;
  example = a worked instance; derivation = step-by-step working;
  intuition = an informal "why"; recap = a short summary.
- "depth": 0 when the beat is essential to the argument; 1-3 for increasingly
  optional elaboration.
- "analogy_id": id of the analogy this beat introduces or calls back, else null.
  Declare every new analogy in "analogies" (analogy_id, label, target_concepts,
  mapping from analogy term to concept). Callbacks reuse the existing id.
  Return "analogies": [] when there is no new analogy."""


def _analogy_id_slug(raw: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(raw or "").strip().lower()).strip("_")
    if not slug:
        return ""
    return slug if slug.startswith("analogy_") else f"analogy_{slug}"


def _parse_analogies(raw_items: Any, known: Dict[str, Analogy]) -> Dict[str, str]:
    """Register the analogies one LLM response declared.

    Mutates ``known`` (analogy_id -> Analogy, declaration order) and returns a
    map from the ids the response used to registered ids. Re-declaring a known
    id with the same (or no) label is a callback; a different label under the
    same id is a different analogy and gets a numeric suffix.
    """
    id_map: Dict[str, str] = {}
    for item in raw_items if isinstance(raw_items, list) else []:
        if not isinstance(item, dict):
            continue
        local = str(item.get("analogy_id") or item.get("label") or "")
        base = _analogy_id_slug(local)
        if not base:
            continue
        label = str(item.get("label") or "").strip()
        existing = known.get(base)
        if existing is not None and (not label or existing.label.lower() == label.lower()):
            id_map[local] = base
            continue
        analogy_id, suffix = base, 2
        while analogy_id in known:
            analogy_id, suffix = f"{base}_{suffix}", suffix + 1
        concepts = [
            canonical
            for concept in (item.get("target_concepts") or [])
            if isinstance(concept, str) and (canonical := _canonicalize_concept(concept))
        ]
        raw_mapping = item.get("mapping")
        mapping = (
            {str(k): str(v) for k, v in raw_mapping.items()}
            if isinstance(raw_mapping, dict) else {}
        )
        known[analogy_id] = Analogy(
            analogy_id=analogy_id,
            label=label or base.removeprefix("analogy_").replace("_", " "),
            target_concepts=concepts,
            mapping=mapping,
        )
        id_map[local] = analogy_id
    return id_map


def _typed_beat_fields(raw: Dict[str, Any], analogy_id_map: Dict[str, str]) -> Dict[str, Any]:
    """Coerce the LLM's role/depth/analogy tags; bad values fall back to defaults."""
    role = str(raw.get("role") or "core").strip().lower()
    if role not in _LLM_BEAT_ROLES:
        role = "core"
    try:
        depth = int(raw.get("depth") or 0)
    except (TypeError, ValueError):
        depth = 0
    analogy_id = None
    local = raw.get("analogy_id")
    if isinstance(local, str) and local.strip():
        analogy_id = analogy_id_map.get(local) or _analogy_id_slug(local) or None
    return {
        "role": role,
        "depth": min(max(depth, 0), MAX_BEAT_DEPTH),
        "analogy_id": analogy_id,
    }


def _extract_json_object(text: str) -> str:
    """Return the JSON-object substring from an LLM text response.

    OpenAI's ``response_format=json_object`` guarantees clean JSON, but Claude
    returns free text — so strip ``` fences and slice from the first ``{`` to
    the last ``}``. Returns the input unchanged when no object braces are found.
    """
    s = text.strip()
    if s.startswith("```"):
        # Drop the opening fence line (``` or ```json) and any trailing fence.
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end + 1]
    return s


# Primitive -> canonical object_type, used to backfill the required
# ``object_type`` field when the visual planner omits it (the #1 cause of
# visual-intent schema failures — models emit primitive/args/bbox/interval but
# drop object_type/content).
_PRIMITIVE_TO_OBJECT_TYPE = {
    "MathTex": "equation", "Tex": "equation",
    "Text": "text", "DecimalNumber": "text",
    "Axes": "axes", "NumberPlane": "axes", "NumberLine": "axes",
    "Circle": "geometric_shape", "Square": "geometric_shape",
    "Rectangle": "geometric_shape", "RoundedRectangle": "geometric_shape",
    "Polygon": "geometric_shape", "Triangle": "geometric_shape",
    "Line": "geometric_shape", "DashedLine": "geometric_shape",
    "Arrow": "geometric_shape", "DoubleArrow": "geometric_shape",
    "Vector": "geometric_shape", "Arc": "geometric_shape",
    "Dot": "geometric_shape", "Ellipse": "geometric_shape",
    "create_pi_teacher": "pi_creature", "create_pi_students": "pi_creature",
    "create_speech_bubble": "pi_creature",
}


# Manim safe area (frame is x ∈ [-7.1, 7.1], y ∈ [-4.0, 4.0]; margin keeps
# content off the bezel). Plan bboxes are clamped into this box.
SAFE_X, SAFE_Y = 6.5, 3.5


def _coerce_visual_object(raw: Any) -> Any:
    """Backfill required fields and clamp the bbox into the safe area.

    The visual planner often emits an object with primitive/args/bbox/interval
    but no ``object_type`` and no ``content`` — both required by
    ``MathematicalObject``. Infer object_type from the primitive (or render_hint)
    and pull content from the tex/text args, so the object validates instead of
    dropping the whole scene to the deterministic fallback.

    Screen-space guard: planners occasionally place centres outside the frame
    or emit frame-sized boxes; clamp centre into ±(SAFE_X, SAFE_Y) and cap
    w/h to the frame so nothing is planned off-screen.
    """
    if not isinstance(raw, dict):
        return raw
    o = dict(raw)
    if not o.get("object_type"):
        prim, hint = o.get("primitive"), o.get("render_hint")
        o["object_type"] = (
            _PRIMITIVE_TO_OBJECT_TYPE.get(prim)
            or _PRIMITIVE_TO_OBJECT_TYPE.get(hint)
            or ("geometric_shape" if prim else "text")
        )
    if o.get("content") in (None, ""):
        args = o.get("args") if isinstance(o.get("args"), dict) else {}
        o["content"] = args.get("tex") or args.get("text") or o.get("notation_key") or ""

    bbox = o.get("bbox")
    if isinstance(bbox, dict):
        b = dict(bbox)
        try:
            if isinstance(b.get("w"), (int, float)):
                b["w"] = min(float(b["w"]), 2 * SAFE_X)
            if isinstance(b.get("h"), (int, float)):
                b["h"] = min(float(b["h"]), 2 * SAFE_Y)
            if isinstance(b.get("x"), (int, float)):
                b["x"] = max(-SAFE_X, min(SAFE_X, float(b["x"])))
            if isinstance(b.get("y"), (int, float)):
                b["y"] = max(-SAFE_Y, min(SAFE_Y, float(b["y"])))
            o["bbox"] = b
        except (TypeError, ValueError):
            pass
    return o


def _intervals_overlap(a: Optional[Any], b: Optional[Any]) -> bool:
    """Whether two visibility windows intersect (missing window = always on)."""
    if a is None or b is None:
        return True
    return a.t_in < b.t_out and b.t_in < a.t_out


def _resolve_bbox_overlaps(objs: List[Any], iterations: int = 4) -> None:
    """Deterministically nudge time-overlapping planned bboxes apart.

    The useful core of the retired (x, y, t) validator, without its LLM retry
    loop: for each pair of objects visible at the same time whose boxes
    intersect, push the SECOND one along the axis of least penetration and
    clamp back into the safe area. A few sweeps resolve typical plans; hard
    cases are left for the LayoutTracker at render time.
    """
    timed = [o for o in objs if o.bbox is not None]
    for _ in range(iterations):
        moved = False
        for i in range(len(timed)):
            for j in range(i + 1, len(timed)):
                a, b = timed[i], timed[j]
                if not _intervals_overlap(a.interval, b.interval):
                    continue
                dx = (a.bbox.w + b.bbox.w) / 2 - abs(a.bbox.x - b.bbox.x)
                dy = (a.bbox.h + b.bbox.h) / 2 - abs(a.bbox.y - b.bbox.y)
                if dx <= 0 or dy <= 0:
                    continue  # no spatial intersection
                if dx <= dy:  # push along x (smaller penetration)
                    direction = 1.0 if b.bbox.x >= a.bbox.x else -1.0
                    b.bbox.x = max(-SAFE_X, min(SAFE_X, b.bbox.x + direction * (dx + 0.1)))
                else:
                    direction = 1.0 if b.bbox.y >= a.bbox.y else -1.0
                    b.bbox.y = max(-SAFE_Y, min(SAFE_Y, b.bbox.y + direction * (dy + 0.1)))
                moved = True
        if not moved:
            return


from functools import lru_cache


@lru_cache(maxsize=1)
def _visual_intent_static_prompt() -> str:
    """The byte-identical prefix of every visual-intent prompt.

    OpenAI prefix caching only fires on identical leading tokens — the old
    prompt put the (per-scene) narration FIRST, so ~45 planner calls per run
    paid full price for the same ~2K-token contract every time. All static
    content lives here, in one stable string; per-scene data is appended
    after it.
    """
    engine_contract = build_visual_intent_contract_prompt()
    catalog_block = catalog_for_prompt()
    return (
        "You are a 3Blue1Brown animation architect. You plan the visuals for "
        "ONE scene of a longer video; the scene's narration and context follow "
        "after these standing rules.\n\n"
        f"{engine_contract}\n"
        "CLOSED VOCABULARY:\n"
        f"{catalog_block}\n\n"
        "For every mathematical_object you emit, ALWAYS include these REQUIRED "
        "fields: `object_type` (one of: equation, text, axes, graph, "
        "geometric_shape, pi_creature) and `content` (the LaTeX/text string, "
        "or \"\" for a pure shape). Then set `primitive` to one of "
        "the MOBJECT names above and `args` to its kwargs. Free-form "
        "`content` is ONLY permitted for MathTex / Tex / Text (where it "
        "goes into the `tex` or `text` arg).\n\n"
        "PURPOSE (required): every object must include a concise `purpose` "
        "explaining what the viewer learns from it and `supports_beat_ids` "
        "listing the narration/visual beat ids it serves. Never emit generic "
        "decoration, filler, or an object with no beat-level job.\n\n"
        "VISUAL IMPACT (non-negotiable): this is a VISUALIZATION tool, not "
        "a slideshow. At least ONE object MUST be a non-text primitive "
        "(Axes, NumberLine, Vector, Arrow, Circle, Polygon, Line, Dot, ...) "
        "that visually DEPICTS the concept. Text/MathTex are for labels and "
        "the key formula only — never the whole scene.\n"
        "AT MOST 15 objects across the whole scene and AT MOST 7 visible at "
        "the same time. Sequence reveals and reuse existing objects rather "
        "than creating an unreadable collage.\n\n"
        "LAYOUT (x, y, t):\n"
        "  - For every object also emit `bbox` = {x, y, w, h} centred on the\n"
        "    object's intended position in Manim units AND `interval` =\n"
        "    {t_in, t_out} — when in scene-time the object is visible.\n"
        "  - SAFE AREA: keep bbox.x in [-6.5, 6.5] and bbox.y in [-3.5, 3.5].\n"
        "  - Two objects may share the same (x, y) only if their intervals\n"
        "    are disjoint (one fades out before the other appears).\n\n"
        "Return JSON with keys: objects (list), transformations (list), "
        "beats (list), scene_state_in (object), scene_state_out (object), "
        "emphasis_points (list).\n\n"
        "───────────────  SCENE-SPECIFIC CONTEXT BELOW  ───────────────\n\n"
    )


def _validate_items(items: Any, model: Any) -> list:
    """Validate a list against a Pydantic model, dropping (not raising on) the
    items that fail — one malformed object never voids the whole scene."""
    out = []
    if not isinstance(items, list):
        return out
    for item in items:
        try:
            out.append(model.model_validate(item))
        except Exception as exc:
            logger.debug("Dropping invalid %s item: %s", getattr(model, "__name__", model), exc)
    return out


class ScriptArchitectInput(AgentInput):
    """Input for Script/Storyboard Architect Agent"""
    
    def __init__(
        self,
        ocr_output: OCROutput,
        configuration: Configuration,
        segments: Optional[List[SourceSegment]] = None,
        review_feedback: Optional[List[str]] = None,
    ):
        """Initialize input
        
        Args:
            ocr_output: OCR output from Vision+OCR Agent
            configuration: Configuration parameters for storyboard generation
            segments: Ordered source segments covering the OCR output
            review_feedback: Optional critique to apply on a regeneration pass
        """
        
        self.ocr_output = ocr_output
        self.configuration = configuration
        self.segments = segments or []
        self.review_feedback = review_feedback or []


class ScriptArchitectAgent(Agent):
    """Agent that generates a chunked storyboard from a unified video script.

    Pipeline (see docs/superpowers/plans/2026-04-26-unified-script-chunked-render.md):
      1. _generate_video_script — single LLM call produces a continuous
         narration arc with an explicit concept-introduction ledger.
      2. _chunk_script_into_scenes — deterministic chunker groups beats
         into ~30-90s scenes; beats are atomic.
      3. _generate_visual_intent_for_chunk — per-chunk visual planner
         (currently a deterministic fallback; Task 7 replaces it with an
         LLM-driven planner that receives the narrative briefing instead
         of the full PDF).
      4. _assemble_storyboard_from_script — derives concept_hierarchy from
         the script's ledger and packages everything into a StoryboardJSON.

    Determinism: scene ids are sequential (scene_001, scene_002, ...). The
    chunker is fully deterministic given the same VideoScript.
    """
    
    # Configuration constants
    # Visual-first pacing (ADR-005): scenes are short, dense vignettes — one
    # focused visual idea each — not lecture-length segments. MIN matches the
    # animation generator's MIN_SCENE_SECONDS floor.
    MIN_SCENE_DURATION = 8.0  # Minimum scene duration in seconds
    MAX_SCENE_DURATION = 45.0  # Maximum scene duration in seconds
    TARGET_SCENE_DURATION = 25.0  # Target scene duration in seconds
    # Hard ceiling on scenes per video. The chunker escalates its packing
    # target until the count fits (fewer, slightly longer scenes beat a
    # choppy 90-cut video for engagement, render time, and cost).
    MAX_SCENES = 45

    # Primitives / object families that put words (not pictures) on screen.
    # A scene whose every object is text-like violates the visualization-first
    # thesis ("text wall") and gets one retry with explicit feedback.
    TEXT_ONLY_PRIMITIVES = frozenset({"Text", "Tex", "MathTex"})
    TEXT_ONLY_OBJECT_TYPES = frozenset({"text", "equation"})
    TEXT_ONLY_RETRY_FEEDBACK = (
        "Your previous plan for this scene contained ONLY text/equation objects "
        "— a text wall. Replace it with a plan where at least one object is a "
        "non-text primitive (Axes + plotted idea, Vector, Circle, Polygon, "
        "Arrow, NumberLine, Dot, ...) that visually depicts the concept; keep "
        "Text/MathTex only as labels."
    )
    WORDS_PER_SECOND = 3.0  # Average narration speed used by prompt + TTS estimates
    MAX_JSON_RETRIES = 3  # Maximum retries for invalid JSON
    MAX_SOURCE_CHARS = 120_000
    # ~45 words ≈ 15 s of narration — one beat is one vignette-sized idea.
    # Smaller beats => more beats per document => more, shorter videos that
    # cover the same breadth without compressing topics together.
    WORDS_PER_BEAT = 45
    GRANULARITY_WORD_TARGETS = {
        "coarse": 300,
        "balanced": 150,
        "fine": 90,
    }
    
    # Default configuration values
    DEFAULT_VERBOSITY = "medium"
    DEFAULT_DEPTH = "intermediate"
    DEFAULT_AUDIENCE = "undergraduate"
    
    def __init__(self, strict: bool = False):
        """Initialize the Script/Storyboard Architect Agent.

        Args:
            strict: If True, raise immediately when OPENAI_API_KEY is missing
                    instead of falling back to mock narration/visual generation.
        """
        self.retry_count = 0
        self.strict = strict
        self.last_video_script = None
        super().__init__()
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            # max_retries=4 (default 2): the per-segment / visual-intent loop
            # fires many calls and saw transient RemoteProtocolError disconnects
            # that were degrading scenes to the fallback. Retry them first.
            self.client = openai.OpenAI(api_key=api_key, max_retries=4)
        else:
            if self.strict:
                raise AgentExecutionError(
                    "MISSING_API_KEY",
                    "OPENAI_API_KEY is required in --strict mode for ScriptArchitect"
                )
            self.client = None
            logger.warning("OPENAI_API_KEY not found. ScriptArchitect will use mock fallbacks.")

        # The unified whole-document script runs on Claude Sonnet 5. Best-effort:
        # only constructed when an Anthropic key is present; the unified path falls
        # back to the OpenAI client (gpt-5.4) when it is absent.
        self.anthropic_client = None
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic()
            except Exception as exc:  # pragma: no cover — optional dependency / init
                logger.warning("Anthropic client unavailable for unified script: %s", exc)
    
    def execute(self, input_data: AgentInput) -> AgentOutput:
        """Execute storyboard generation via the unified-script + chunker flow.

        Phase 1 — Generate ONE continuous video script (single LLM call).
        Phase 2 — Deterministically chunk beats into render-sized scenes.
        Phase 3 — Per-chunk visual planner attaches a VisualIntent to each
                  scene. Failures fall back per-chunk so one bad chunk
                  doesn't abort the whole storyboard (F6 mitigation).

        Args:
            input_data: ScriptArchitectInput with OCR output and configuration

        Returns:
            StoryboardJSON object with chunked scene specifications and a
            concept hierarchy derived from the script's ledger.

        Raises:
            AgentExecutionError: For unrecoverable failures.
        """
        if not self.validate_input(input_data):
            raise AgentExecutionError(
                error_code="INVALID_INPUT",
                message="Input must be a ScriptArchitectInput object",
                context={"input_type": type(input_data).__name__},
            )

        script_input: ScriptArchitectInput = input_data
        ocr_output = script_input.ocr_output
        config = script_input.configuration
        review_feedback = script_input.review_feedback or []
        segments = script_input.segments or self._segment_ocr_output(ocr_output, config)

        logger.info(f"Starting storyboard generation for: {ocr_output.pdf_metadata.filename}")
        logger.info(
            f"Configuration: verbosity={config.verbosity}, depth={config.depth}, "
            f"audience={config.audience}"
        )
        if review_feedback:
            logger.info(f"Applying {len(review_feedback)} storyboard revision hints")

        try:
            # Phase 1: ONE unified video script.
            video_script = self._generate_video_script(
                ocr_output,
                config,
                review_feedback,
                segments=segments,
            )
            logger.info(
                f"Generated video script: {len(video_script.beats)} beats, "
                f"{len(video_script.concept_ledger())} unique concepts"
            )
            # The orchestrator persists this as lesson_script.json — the
            # professor-editable source a revision run starts from.
            self.last_video_script = video_script

            # Phase 2: deterministic chunker (Task 4).
            scenes = self._chunk_script_into_scenes(
                video_script, max_scenes=self._effective_max_scenes(config)
            )
            logger.info(f"Chunked script into {len(scenes)} scenes")

            # Phase 3: per-chunk visual intent. F6: any per-chunk failure is
            # isolated and never aborts the whole storyboard.
            self._plan_visual_intents(scenes, video_script, config, review_feedback)
            self._renumber_scenes(scenes)

            storyboard = self._assemble_storyboard_from_script(
                ocr_output, config, video_script, scenes, segments,
            )
            logger.info(f"Storyboard generation completed: {len(scenes)} scenes")
            return storyboard

        except AgentExecutionError:
            raise
        except Exception as e:
            raise AgentExecutionError(
                error_code="STORYBOARD_GENERATION_FAILED",
                message=f"Failed to generate storyboard: {str(e)}",
                context={
                    "filename": ocr_output.pdf_metadata.filename,
                    "error": str(e),
                },
            )

    def _plan_visual_intents(
        self,
        scenes: List[Scene],
        video_script,
        config: Configuration,
        review_feedback: List[str],
        keep_ids: frozenset = frozenset(),
    ) -> None:
        """Attach a VisualIntent to each scene, in order, mutating ``scenes``.

        Overloaded multi-beat scenes may be split in place. Scenes whose id is
        in ``keep_ids`` already carry a plan (a revision run reuses them): they
        are not re-planned, but still hand their on-screen objects to the next
        scene so continuity is preserved.
        """
        visual_failures: List[Dict[str, str]] = []
        text_only_scenes: List[str] = []
        # Visual continuity: intents are generated in scene order so each
        # scene can be told what the previous one left on screen.
        carried_objects: List[MathematicalObject] = []
        scene_index = 0
        while scene_index < len(scenes):
            scene = scenes[scene_index]
            if scene.scene_id in keep_ids:
                carried_objects = self._carried_objects_from_intent(scene.visual_intent)
                scene_index += 1
                continue
            try:
                constraint_errors: List[str] = []
                for purpose_attempt in range(self.MAX_PURPOSE_RETRIES + 1):
                    purpose_feedback = list(review_feedback)
                    if constraint_errors:
                        purpose_feedback.append(
                            "The prior visual plan failed object-purpose validation:\n"
                            + "\n".join(f"- {item}" for item in constraint_errors)
                            + "\nReturn a corrected plan; do not drop narrated content."
                        )
                    scene.visual_intent = self._generate_visual_intent_for_chunk(
                        scene=scene,
                        video_script=video_script,
                        config=config,
                        review_feedback=purpose_feedback,
                        carried_objects=carried_objects,
                    )
                    constraint_errors = self._visual_intent_constraint_errors(
                        scene,
                        scene.visual_intent,
                    )
                    if not constraint_errors:
                        break
                if constraint_errors:
                    # Splitting adds a scene, so it must respect the
                    # same ceiling the chunker enforces (target_minutes /
                    # MAX_SCENES); past it, fall back instead.
                    if (
                        self._has_object_budget_error(constraint_errors)
                        and len(scene.source_beats) > 1
                        and len(scenes) < self._effective_max_scenes(config)
                    ):
                        replacements = self._split_scene_at_beat_boundary(scene)
                        scenes[scene_index:scene_index + 1] = replacements
                        logger.info(
                            "Split overloaded %s into two beat-aligned scenes.",
                            scene.scene_id,
                        )
                        continue
                    raise ValueError("; ".join(constraint_errors))
                # Visual-impact invariant: a scene of nothing but text /
                # equations is a slideshow, not a visualization. Retry the
                # planner ONCE with explicit feedback. Only meaningful on
                # the LLM path — the no-key fallback is deterministic, so
                # retrying it would reproduce the same intent.
                if (
                    self._is_text_only_intent(scene.visual_intent)
                    and self.client is not None
                    and os.getenv("OPENAI_API_KEY")
                ):
                    validated_intent = scene.visual_intent
                    retry_intent = self._generate_visual_intent_for_chunk(
                        scene=scene,
                        video_script=video_script,
                        config=config,
                        review_feedback=review_feedback + [self.TEXT_ONLY_RETRY_FEEDBACK],
                        carried_objects=carried_objects,
                    )
                    retry_errors = self._visual_intent_constraint_errors(
                        scene,
                        retry_intent,
                    )
                    if retry_errors:
                        # Keep the plan that already passed validation; a
                        # failed retry must not discard it for the generic
                        # narration fallback.
                        logger.warning(
                            "Text-only retry for %s failed validation (%s); "
                            "keeping the validated first plan.",
                            scene.scene_id, "; ".join(retry_errors),
                        )
                        scene.visual_intent = validated_intent
                    else:
                        scene.visual_intent = retry_intent
                    if self._is_text_only_intent(scene.visual_intent):
                        text_only_scenes.append(scene.scene_id)
            except Exception as exc:
                if self.strict:
                    raise
                logger.warning(
                    "Visual intent failed for %s: %s — falling back to narration-derived intent",
                    scene.scene_id, exc,
                )
                scene.visual_intent = self._generate_fallback_visual_intent(scene.narration)
                visual_failures.append({"scene_id": scene.scene_id, "error": str(exc)})
            carried_objects = self._carried_objects_from_intent(scene.visual_intent)
            scene_index += 1

        if visual_failures:
            logger.warning(
                "Visual intent fallback used for %d/%d scenes",
                len(visual_failures), len(scenes),
            )
        if text_only_scenes:
            logger.warning(
                "Text-only visual intent persisted after retry for %d/%d scenes: %s",
                len(text_only_scenes), len(scenes), ", ".join(text_only_scenes),
            )

    def revise_storyboard(
        self,
        base_storyboard: StoryboardJSON,
        script,
    ) -> Tuple[StoryboardJSON, Dict[str, str]]:
        """Re-derive a storyboard from an edited lesson script.

        Beats are grouped into runs of consecutive beats from one source
        segment. A run whose beats fingerprint-match a base run keeps that
        run's scenes verbatim, visual plans included. Any other run is
        re-chunked on its own (the chunker never crosses a segment boundary,
        so no other run moves), and a re-chunked scene whose beats exactly
        match a base scene still reuses it. Only the remaining scenes are
        re-planned.

        Returns the storyboard and a map from new scene_id to the base
        scene_id each reused scene came from.
        """
        from types import SimpleNamespace

        config = base_storyboard.configuration
        base_by_fp: Dict[str, Scene] = {}
        base_runs: Dict[Tuple[str, Tuple[str, ...]], List[Scene]] = {}
        for segment_id, run_scenes in self._scene_runs(base_storyboard.scenes):
            fps = tuple(
                beat_fingerprint(b) for s in run_scenes for b in s.source_beats
            )
            base_runs.setdefault((segment_id, fps), run_scenes)
            for s in run_scenes:
                if s.source_beats:
                    base_by_fp.setdefault(beats_fingerprint(s.source_beats), s)

        scenes: List[Scene] = []
        reused_from: Dict[str, str] = {}  # temporary scene_id -> base scene_id

        def take(base_scene: Scene, beats) -> None:
            reused = base_scene.model_copy(deep=True)
            # Fingerprints match, so only provenance fields (author, locked)
            # can differ; carry the edited script's beats.
            reused.source_beats = list(beats)
            reused.scene_id = f"rev_{len(scenes):03d}"
            reused_from[reused.scene_id] = base_scene.scene_id
            scenes.append(reused)

        for segment_id, run_beats in self._beat_runs(script.beats):
            fps = tuple(beat_fingerprint(b) for b in run_beats)
            matched = base_runs.pop((segment_id, fps), None)
            if matched is not None:
                cursor = 0
                for base_scene in matched:
                    count = len(base_scene.source_beats)
                    take(base_scene, run_beats[cursor:cursor + count])
                    cursor += count
                continue
            for chunk in self._chunk_script_into_scenes(SimpleNamespace(beats=run_beats)):
                base_scene = base_by_fp.get(beats_fingerprint(chunk.source_beats))
                if base_scene is not None:
                    take(base_scene, chunk.source_beats)
                else:
                    chunk.scene_id = f"rev_{len(scenes):03d}"
                    scenes.append(chunk)

        self._plan_visual_intents(
            scenes, script, config, review_feedback=[],
            keep_ids=frozenset(reused_from),
        )
        pairs = [(s, reused_from.get(s.scene_id)) for s in scenes]
        self._renumber_scenes(scenes)
        reuse_map = {s.scene_id: base_id for s, base_id in pairs if base_id}

        ledger = script.concept_ledger()
        self._repair_scene_concepts(scenes, ledger)
        storyboard = StoryboardJSON(
            pdf_metadata=base_storyboard.pdf_metadata,
            configuration=config,
            concept_hierarchy=self._concept_hierarchy_from_ledger(ledger),
            scenes=scenes,
        )
        logger.info(
            "Revised storyboard: %d scenes, %d reused, %d re-planned",
            len(scenes), len(reuse_map), len(scenes) - len(reuse_map),
        )
        return storyboard, reuse_map

    @staticmethod
    def _beat_runs(beats) -> List[Tuple[str, List[Any]]]:
        """Group consecutive beats that share a source segment."""
        runs: List[Tuple[str, List[Any]]] = []
        for beat in beats:
            segment_id = getattr(beat, "source_segment_id", "") or ""
            if runs and runs[-1][0] == segment_id:
                runs[-1][1].append(beat)
            else:
                runs.append((segment_id, [beat]))
        return runs

    @staticmethod
    def _scene_runs(scenes: List[Scene]) -> List[Tuple[str, List[Scene]]]:
        """Group consecutive scenes whose beats come from one source segment."""
        runs: List[Tuple[str, List[Scene]]] = []
        for scene in scenes:
            beats = scene.source_beats or []
            segment_id = (getattr(beats[0], "source_segment_id", "") or "") if beats else ""
            if runs and runs[-1][0] == segment_id:
                runs[-1][1].append(scene)
            else:
                runs.append((segment_id, [scene]))
        return runs

    @staticmethod
    def _repair_scene_concepts(scenes: List[Scene], ledger: List[str]) -> None:
        """Point every scene's concept_id at a concept the script still introduces.

        A reused scene can name a concept whose introducing beat was edited
        away, and a re-chunked run with no introductions falls back to a beat
        id; both would fail StoryboardJSON's concept cross-check.
        """
        valid = set(ledger)
        if not valid:
            return
        previous = ledger[0]
        for scene in scenes:
            if scene.concept_id not in valid:
                scene.concept_id = next(
                    (
                        c for b in scene.source_beats
                        for c in b.introduces_concepts if c in valid
                    ),
                    previous,
                )
            previous = scene.concept_id

    @staticmethod
    def _concept_hierarchy_from_ledger(ledger: List[str]) -> List[ConceptHierarchy]:
        """One entry per concept in first-introduction order; each depends on
        every concept introduced before it (mirrors the linear narrative)."""
        return [
            ConceptHierarchy(
                concept_id=concept,
                concept_name=concept.replace("_", " ").title(),
                dependencies=ledger[:i],
            )
            for i, concept in enumerate(ledger)
        ]

    def _assemble_storyboard_from_script(
        self,
        ocr_output: OCROutput,
        config: Configuration,
        video_script,
        scenes: List[Scene],
        segments: Optional[List[SourceSegment]] = None,
    ) -> StoryboardJSON:
        """Build a StoryboardJSON from a unified VideoScript + chunked scenes.

        ``concept_hierarchy`` is derived from the script's deduped concept
        ledger (see ``_concept_hierarchy_from_ledger``).
        """
        pdf_metadata = PDFMetadataStoryboard(
            filename=ocr_output.pdf_metadata.filename,
            page_count=ocr_output.pdf_metadata.page_count,
        )
        concept_hierarchy = self._concept_hierarchy_from_ledger(
            video_script.concept_ledger()
        )
        self._log_segment_coverage(scenes, segments or [])
        return StoryboardJSON(
            pdf_metadata=pdf_metadata,
            configuration=config,
            concept_hierarchy=concept_hierarchy,
            scenes=scenes,
        )

    def _segment_ocr_output(
        self,
        ocr_output: OCROutput,
        config: Configuration,
    ) -> List[SourceSegment]:
        from src.agents.document_segmenter import target_words_for, total_ocr_words

        target_words = target_words_for(
            config.granularity, total_ocr_words(ocr_output)
        )
        result = DocumentSegmenterAgent().execute(
            DocumentSegmenterInput(
                ocr_output=ocr_output,
                target_words_per_segment=target_words,
            )
        )
        return result.segments

    def _log_segment_coverage(
        self,
        scenes: List[Scene],
        segments: List[SourceSegment],
    ) -> None:
        """Check the two real coverage invariants of the vignette design.

        WARN on genuine defects only:
          - a segment covered by NO scene (content dropped), or
          - a beat appearing in MORE than one scene (content duplicated).

        A segment spanning multiple scenes is *by design* — the chunker splits
        a segment's beats into vignette-sized scenes — so it is reported as an
        INFO statistic, never a warning (the old check flagged it and spammed
        one false-positive warning per segment on real runs).
        """
        if not segments:
            return
        expected = {segment.segment_id for segment in segments}
        scene_segment_ids = [
            segment_id
            for scene in scenes
            for segment_id in getattr(scene, "source_segment_ids", [])
        ]
        covered = set(scene_segment_ids)
        missing = sorted(expected - covered)
        if missing:
            logger.warning("Storyboard missing source segments: %s", ", ".join(missing))

        beat_ids = [
            beat.beat_id
            for scene in scenes
            for beat in getattr(scene, "source_beats", []) or []
        ]
        duplicated = sorted({b for b in beat_ids if beat_ids.count(b) > 1})
        if duplicated:
            logger.warning(
                "Beats duplicated across scenes (content repeated): %s",
                ", ".join(duplicated),
            )

        if covered:
            logger.info(
                "Segment coverage: %d/%d segments across %d scenes "
                "(avg %.1f scenes/segment).",
                len(covered), len(expected), len(scenes),
                len(scene_segment_ids) / max(len(covered), 1),
            )

    # How many objects a scene may hand to its successor. Carrying everything
    # forward re-creates the clutter problem in the next scene; the anchor
    # visual plus a label or two is what makes the cut feel continuous.
    MAX_CARRY_OVER_OBJECTS = 3
    # Hard cap on planned objects per vignette scene — a real run averaged
    # 16.7 objects/scene, which drove both the overlaps and the text walls.
    MAX_OBJECTS_PER_SCENE = 15
    MAX_CONCURRENT_OBJECTS = 7
    MAX_PURPOSE_RETRIES = 2

    @classmethod
    def _derive_carry_out_ids(cls, objs: List[MathematicalObject]) -> List[str]:
        """Objects still on screen at scene end, by their planned intervals.

        An object whose ``interval.t_out`` reaches (within 0.5s of) the scene's
        final timestamp is considered carried over. Non-text visuals are
        preferred — carrying the anchor diagram matters more than carrying a
        caption — and the list is capped at MAX_CARRY_OVER_OBJECTS.
        """
        timed = [o for o in objs if o.interval is not None and o.object_id]
        if not timed:
            return []
        scene_end = max(o.interval.t_out for o in timed)
        surviving = [o for o in timed if o.interval.t_out >= scene_end - 0.5]
        # Anchor visuals first, then text, preserving plan order within each.
        text_like = cls.TEXT_ONLY_PRIMITIVES
        ordered = (
            [o for o in surviving if (o.primitive or "") not in text_like]
            + [o for o in surviving if (o.primitive or "") in text_like]
        )
        return [o.object_id for o in ordered[: cls.MAX_CARRY_OVER_OBJECTS]]

    @staticmethod
    def _carried_objects_from_intent(intent: VisualIntent) -> List[MathematicalObject]:
        """The objects a finished scene hands to its successor.

        Resolves ``scene_state_out.active_object_ids`` back to the full
        MathematicalObject records so the next scene's prompt can name their
        primitive and position, not just an id.
        """
        active = set(intent.scene_state_out.active_object_ids or [])
        if not active:
            return []
        return [o for o in intent.mathematical_objects if o.object_id in active]

    @classmethod
    def _is_text_only_intent(cls, intent: VisualIntent) -> bool:
        """True when the plan puts nothing but words on screen.

        An object is text-like when its catalog ``primitive`` is Text/Tex/
        MathTex, or — for legacy objects without a primitive — when its
        ``object_type`` is text/equation. An empty object list also counts:
        a blank screen is no more a visualization than a text wall.
        """
        objs = intent.mathematical_objects
        if not objs:
            return True
        for obj in objs:
            primitive = getattr(obj, "primitive", None)
            if primitive is not None:
                if primitive not in cls.TEXT_ONLY_PRIMITIVES:
                    return False
            elif obj.object_type not in cls.TEXT_ONLY_OBJECT_TYPES:
                return False
        return True

    def _generate_visual_intent_for_chunk(
        self,
        scene: Scene,
        video_script,
        config: Configuration,
        review_feedback: List[str],
        carried_objects: Optional[List[MathematicalObject]] = None,
    ) -> VisualIntent:
        """Plan visuals for one chunk via a chunk-scoped LLM call.

        The prompt receives:
          - the chunk's narration (audio for THIS scene only),
          - the visual hints written by the script architect for this chunk,
          - the list of concepts introduced in EARLIER chunks (the briefing —
            the viewer has already seen these; reference, do not redefine),
          - the engine contract (canonical object types, render hints,
            continuity actions, …).

        The full PDF / OCR text is intentionally NOT passed; that data is
        the script generator's responsibility. Empty / malformed LLM output
        falls back to the deterministic narration-derived intent so a single
        bad chunk never aborts the storyboard (F6 mitigation).
        """
        if not os.getenv("OPENAI_API_KEY") or self.client is None:
            if self.strict:
                raise AgentExecutionError(
                    "MISSING_API_KEY",
                    "OPENAI_API_KEY is required in --strict mode for visual intent",
                )
            return self._generate_fallback_visual_intent(scene.narration)

        chunk_beats = list(getattr(scene, "source_beats", []) or [])
        chunk_concepts: List[str] = []
        for b in chunk_beats:
            for c in b.introduces_concepts:
                if c not in chunk_concepts:
                    chunk_concepts.append(c)

        # Concepts taught in earlier chunks (everything before this scene's first beat).
        chunk_beat_ids = {b.beat_id for b in chunk_beats}
        earlier_concepts: List[str] = []
        for b in video_script.beats:
            if b.beat_id in chunk_beat_ids:
                break
            for c in b.introduces_concepts:
                if c not in earlier_concepts and c not in chunk_concepts:
                    earlier_concepts.append(c)

        visual_hints = (
            "\n".join(f"- {b.beat_id}: {b.visual_hint}" for b in chunk_beats)
            or "- (no specific hints)"
        )
        valid_beat_ids = [beat.beat_id for beat in chunk_beats]

        word_count = len(scene.narration.split())
        target = round(word_count / self.WORDS_PER_SECOND, 1)

        review_block = ""
        if review_feedback:
            review_block = (
                "\nREVISION FEEDBACK:\n"
                + "\n".join(f"- {r}" for r in review_feedback[:8])
                + "\n"
            )

        earlier_block = ""
        if earlier_concepts:
            earlier_block = (
                "\nCONCEPTS ALREADY INTRODUCED in EARLIER scenes (the viewer has "
                "seen these — reference them, do NOT redefine):\n"
                + "\n".join(f"  - {c}" for c in earlier_concepts)
                + "\n"
            )

        # Concept → archetype: give the planner a proven visual pattern for
        # this chunk's concept family instead of a blank page. Underscored
        # concept labels are space-joined so "unit_circle" matches "circle".
        classification_text = " ".join([
            scene.narration,
            " ".join(c.replace("_", " ") for c in chunk_concepts),
            visual_hints,
        ])
        archetype = classify_concept(classification_text)
        archetype_block = (
            archetype_prompt_block(archetype) + "\n" if archetype else ""
        )

        # Visual continuity: describe what the previous scene left on screen so
        # this scene starts from it instead of cutting to black.
        carry_block = ""
        if carried_objects:
            carry_lines = []
            for o in carried_objects:
                pos = f"({o.bbox.x:.1f}, {o.bbox.y:.1f})" if o.bbox else "(keep position)"
                carry_lines.append(
                    f"  - object_id={o.object_id}  {o.primitive or o.object_type}  at {pos}"
                )
            carry_block = (
                "PREVIOUS SCENE ENDS WITH THESE OBJECTS STILL ON SCREEN:\n"
                + "\n".join(carry_lines)
                + "\nVISUAL CONTINUITY (important): start THIS scene from that "
                "arrangement. Re-emit each carried object FIRST in `objects` — "
                "SAME object_id, same primitive and args, same bbox position, "
                "interval.t_in = 0 — then transform it or build around it. Do "
                "not restart from an empty screen unless the topic changes "
                "completely. List the carried ids in "
                "scene_state_in.active_object_ids.\n\n"
            )

        # Static contract first (byte-identical across all scenes → provider
        # prefix caching), per-scene context after.
        prompt = _visual_intent_static_prompt() + (
            f"{archetype_block}"
            "--- CHUNK NARRATION (audio for THIS scene only) ---\n"
            f"{scene.narration}\n"
            "--- END NARRATION ---\n\n"
            f"NEW CONCEPTS INTRODUCED IN THIS CHUNK: "
            f"{', '.join(chunk_concepts) or '(none)'}\n"
            f"VALID BEAT IDS FOR supports_beat_ids: "
            f"{', '.join(valid_beat_ids) or 'beat_001'}\n"
            f"{earlier_block}"
            f"{carry_block}"
            "VISUAL HINTS FROM THE SCRIPT WRITER:\n"
            f"{visual_hints}\n\n"
            f"TIMING: {word_count} words at ~3 wps => visuals should fill ~{target}s.\n"
            f"{review_block}"
            "\nNow plan THIS scene following every standing rule above. "
            "Return ONLY the JSON object."
        )

        try:
            response = self.client.chat.completions.create(
                model=VISUAL_INTENT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
        except Exception as exc:  # malformed JSON / API error -> deterministic fallback
            logger.warning(
                "Visual-intent LLM call failed for %s: %s. Using fallback.",
                scene.scene_id, exc,
            )
            return self._generate_fallback_visual_intent(scene.narration)

        return self._visual_intent_from_json(data, fallback_narration=scene.narration)

    def _visual_intent_from_json(self, data: Dict[str, Any], fallback_narration: str) -> VisualIntent:
        """Parse LLM JSON into a VisualIntent.

        Accepts either ``objects`` or ``mathematical_objects`` for the object
        list (LLMs drift between the two). Empty / malformed payloads fall
        back to ``_generate_fallback_visual_intent`` so the pipeline never
        ships a blank scene (F6 mitigation).
        """
        from src.schemas.storyboard import EmphasisPoint, SceneState, VisualBeat

        if not isinstance(data, dict):
            return self._generate_fallback_visual_intent(fallback_narration)

        raw_objects = data.get("objects") or data.get("mathematical_objects") or []
        raw_trans = data.get("transformations") or []
        raw_beats = data.get("beats") or []
        raw_emphasis = data.get("emphasis_points") or []
        raw_in = data.get("scene_state_in") or {}
        raw_out = data.get("scene_state_out") or {}

        if not raw_objects:
            # An empty visual intent is unrenderable — degrade to the
            # narration-derived placeholder so the scene at least produces
            # something on screen.
            return self._generate_fallback_visual_intent(fallback_narration)

        # Coerce missing required fields, then validate item-by-item so a single
        # malformed object doesn't void the entire scene's plan (F6). Only fall
        # back when NO objects survive.
        objs = _validate_items(
            [_coerce_visual_object(o) for o in raw_objects], MathematicalObject
        )
        if not objs:
            logger.warning(
                "Visual-intent objects all failed validation; using fallback."
            )
            return self._generate_fallback_visual_intent(fallback_narration)

        # Deterministic de-overlap of the planned layout (time-aware).
        _resolve_bbox_overlaps(objs)

        trans = _validate_items(raw_trans, Transformation)
        beats = _validate_items(raw_beats, VisualBeat)
        emphasis = _validate_items(raw_emphasis, EmphasisPoint)
        try:
            state_in = SceneState.model_validate(raw_in if isinstance(raw_in, dict) else {})
        except Exception:
            state_in = SceneState()
        try:
            state_out = SceneState.model_validate(raw_out if isinstance(raw_out, dict) else {})
        except Exception:
            state_out = SceneState()

        # Continuity backbone: planners reliably leave scene_state_out empty
        # (a real run: 0/90 scenes carried state → every scene restarted from
        # black). Don't trust the LLM for it — derive it from the plan: the
        # objects still visible at scene end are the carry-over set.
        if not state_out.active_object_ids:
            state_out.active_object_ids = self._derive_carry_out_ids(objs)

        return VisualIntent(
            mathematical_objects=objs,
            transformations=trans,
            beats=beats,
            scene_state_in=state_in,
            scene_state_out=state_out,
            emphasis_points=emphasis,
        )

    def _visual_intent_constraint_errors(
        self,
        scene: Scene,
        intent: VisualIntent,
    ) -> List[str]:
        """Return precise feedback for unjustified or overloaded visual plans."""
        errors: List[str] = []
        source_beat_ids = {
            beat.beat_id for beat in (getattr(scene, "source_beats", None) or [])
        }
        visual_beat_ids = {beat.beat_id for beat in intent.beats}
        valid_beat_ids = source_beat_ids | visual_beat_ids
        object_ids = {obj.object_id for obj in intent.mathematical_objects if obj.object_id}
        vague_purposes = {"decoration", "decorative", "filler", "visual", "shape"}

        for obj in intent.mathematical_objects:
            label = obj.object_id or "<missing object_id>"
            if not obj.object_id:
                errors.append("Every object needs a stable object_id.")
            purpose = obj.purpose.strip()
            if not purpose or purpose.lower() in vague_purposes:
                errors.append(f"Object {label} needs a specific instructional purpose.")
            if not obj.supports_beat_ids:
                errors.append(f"Object {label} must support at least one beat id.")
            unknown = sorted(set(obj.supports_beat_ids) - valid_beat_ids)
            if unknown:
                errors.append(f"Object {label} references unknown beat ids: {unknown}.")

        if len(intent.mathematical_objects) > self.MAX_OBJECTS_PER_SCENE:
            errors.append(
                f"Scene has {len(intent.mathematical_objects)} objects; maximum is "
                f"{self.MAX_OBJECTS_PER_SCENE}. Sequence or reuse them."
            )

        concurrent = self._max_concurrent_objects(intent.mathematical_objects)
        if concurrent > self.MAX_CONCURRENT_OBJECTS:
            errors.append(
                f"Scene has {concurrent} concurrently visible objects; maximum is "
                f"{self.MAX_CONCURRENT_OBJECTS}. Stagger their intervals."
            )

        for trans in intent.transformations:
            if trans.source_object_id not in object_ids:
                errors.append(
                    f"Transformation references missing source object {trans.source_object_id}."
                )
            if trans.target_object_id and trans.target_object_id not in object_ids:
                errors.append(
                    f"Transformation references missing target object {trans.target_object_id}."
                )
        return errors

    @staticmethod
    def _max_concurrent_objects(objects: List[MathematicalObject]) -> int:
        if not objects:
            return 0
        boundaries = {0.0}
        for obj in objects:
            if obj.interval:
                boundaries.add(obj.interval.t_in)
                boundaries.add(max(obj.interval.t_in, obj.interval.t_out - 1e-6))
        maximum = 0
        for timestamp in boundaries:
            visible = sum(
                1
                for obj in objects
                if obj.interval is None
                or obj.interval.t_in <= timestamp < obj.interval.t_out
            )
            maximum = max(maximum, visible)
        return maximum

    @staticmethod
    def _has_object_budget_error(errors: List[str]) -> bool:
        return any(
            "objects; maximum" in error or "concurrently visible objects" in error
            for error in errors
        )

    def _split_scene_at_beat_boundary(self, scene: Scene) -> List[Scene]:
        """Split an overloaded multi-beat scene without rewriting narration."""
        midpoint = max(1, len(scene.source_beats) // 2)
        groups = [scene.source_beats[:midpoint], scene.source_beats[midpoint:]]
        replacements: List[Scene] = []
        for suffix, beats in zip(("a", "b"), groups):
            narration = "\n\n".join(beat.narration for beat in beats)
            word_count = sum(len(beat.narration.split()) for beat in beats)
            concept_id = next(
                (
                    concept
                    for beat in beats
                    for concept in beat.introduces_concepts
                ),
                scene.concept_id,
            )
            segment_ids: List[str] = []
            for beat in beats:
                segment_id = getattr(beat, "source_segment_id", "")
                if segment_id and segment_id not in segment_ids:
                    segment_ids.append(segment_id)
            replacements.append(Scene(
                scene_id=f"{scene.scene_id}_{suffix}",
                scene_index=scene.scene_index,
                concept_id=concept_id,
                narration=narration,
                visual_intent=self._empty_visual_intent_placeholder(),
                duration_estimate=round(
                    max(self.MIN_SCENE_DURATION, word_count / self.WORDS_PER_SECOND),
                    1,
                ),
                dependencies=list(scene.dependencies),
                difficulty_level=scene.difficulty_level,
                source_beats=list(beats),
                source_segment_ids=segment_ids,
            ))
        return replacements

    @staticmethod
    def _renumber_scenes(scenes: List[Scene]) -> None:
        for index, scene in enumerate(scenes):
            scene.scene_index = index
            scene.scene_id = f"scene_{index + 1:03d}"
            scene.dependencies = [f"scene_{index:03d}"] if index > 0 else []
    
    def _generate_video_script(
        self,
        ocr_output: OCROutput,
        config: Configuration,
        review_feedback: Optional[List[str]] = None,
        segments: Optional[List[SourceSegment]] = None,
    ):
        """Generate ONE continuous video script with a concept-introduction ledger.

        This replaces the per-concept narration loop. The viewer sees a
        single coherent storyline; later beats reference earlier ones instead
        of re-explaining them.

        F1 mitigation: every concept label is canonicalized via
        ``_canonicalize_concept`` so aliases collapse before the
        ``VideoScript`` ledger is built.

        Returns a :class:`src.schemas.video_script.VideoScript`.
        """
        from src.schemas.video_script import VideoScript, Beat

        # Phase-1 fallback: no API key. Used by tests and offline runs.
        if not os.getenv("OPENAI_API_KEY"):
            if self.strict:
                raise AgentExecutionError(
                    "MISSING_API_KEY",
                    "OPENAI_API_KEY is required in --strict mode for video script generation",
                )
            logger.warning("No OpenAI API key found — using mock VideoScript fallback.")
            return self._mock_video_script_fallback(ocr_output, config, segments=segments)

        segments = segments or self._segment_ocr_output(ocr_output, config)
        if segments:
            return self._generate_segmented_video_script(
                segments=segments,
                config=config,
                review_feedback=review_feedback or [],
            )

        # Build full source-material context — only this call sees the full PDF.
        # Downstream chunks see only their slice + the narrative briefing
        # (Tasks 7-8 enforce that).
        source_text = ""
        math_expressions: List[str] = []
        for page in ocr_output.pages:
            source_text += " ".join(b.text for b in page.text_blocks) + "\n"
            for expr in page.math_expressions:
                if not expr.extraction_failed:
                    math_expressions.append(expr.latex)

        math_block = ""
        if math_expressions:
            math_block = (
                "\n--- KEY FORMULAS ---\n"
                + "\n".join(f"  - {e}" for e in math_expressions)
                + "\n"
            )

        source_prompt_text = source_text
        if len(source_text) > self.MAX_SOURCE_CHARS:
            logger.warning(
                "Source text %d chars truncated to %d; use document segmentation for full coverage.",
                len(source_text),
                self.MAX_SOURCE_CHARS,
            )
            source_prompt_text = source_text[:self.MAX_SOURCE_CHARS]

        review_block = ""
        if review_feedback:
            review_block = (
                "\nREVISION FEEDBACK:\n"
                + "\n".join(f"- {r}" for r in review_feedback[:8])
                + "\n"
            )

        audience_line = {
            "high-school": "Use real-world analogies; avoid jargon.",
            "graduate": "Lean into theoretical structure and generalizations.",
        }.get(config.audience, "Balance intuition with mathematical structure.")

        verbosity_line = {
            "low": "Be terse — only the core ideas.",
            "high": "Be detailed; explain why each step matters.",
        }.get(config.verbosity, "Balanced explanation.")

        prompt = f"""You are a Grant Sanderson–style educational scriptwriter (3Blue1Brown).

--- SOURCE MATERIAL ---
{source_prompt_text}
{math_block}
--- END SOURCE MATERIAL ---

TASK: Write ONE continuous video script that explains the source material as a single
coherent story. The output is a list of BEATS. Each beat is one paragraph of spoken
narration paired with a short visual description.

CRITICAL RULES:
1. Every concept is introduced AT MOST ONCE. Once introduced, later beats may
   reference it but MUST NOT redefine or re-derive it.
2. Beats build on each other. A later beat may say "recall the unit circle from
   earlier" but must not re-explain what the unit circle is.
3. Mark `introduces_concepts` for each beat with the canonical labels first
   taught in that beat (e.g. "sine", "unit_circle"). Use lowercase, underscored.
4. Mark `references_prior_beats` with the beat_ids whose visuals/ideas this
   beat builds on.
5. Cover every formula and definition in the source material exactly once.
6. Produce approximately {self._target_beat_count(source_text)} beats (plus or minus 20%). Each beat MUST be 30-60 words of narration.
   if a topic needs more, split into multiple beats with references_prior_beats.

{TYPED_BEAT_PROMPT_RULES}

STYLE:
- Audience ({config.audience}): {audience_line}
- Verbosity ({config.verbosity}): {verbosity_line}
- Tone: Friendly, intuitive, conversational.
- Visual hint per beat: 1–2 sentences describing what is on screen.
{review_block}

Return JSON in this EXACT shape:
{{
  "title": "...",
  "analogies": [
    {{
      "analogy_id": "analogy_short_name",
      "label": "what the analogy is",
      "target_concepts": ["concept_label"],
      "mapping": {{"analogy term": "concept"}}
    }}
  ],
  "beats": [
    {{
      "beat_id": "beat_001",
      "narration": "spoken text...",
      "visual_hint": "what is on screen during this beat",
      "introduces_concepts": ["concept_label", ...],
      "references_prior_beats": ["beat_001", ...],
      "role": "core",
      "depth": 0,
      "analogy_id": null
    }}
  ]
}}
"""

        data = self._complete_unified_script_json(prompt)

        known_analogies: Dict[str, Analogy] = {}
        analogy_id_map = _parse_analogies(data.get("analogies"), known_analogies)
        beats = []
        for b in data.get("beats", []):
            beats.append(Beat(
                beat_id=b["beat_id"],
                narration=b["narration"],
                visual_hint=b.get("visual_hint", ""),
                # F1: canonicalize labels here so the ledger is consistent
                # even if the LLM emitted "Sine Function" / "sin" / "sine".
                introduces_concepts=[
                    _canonicalize_concept(c) for c in b.get("introduces_concepts", [])
                    if _canonicalize_concept(c)
                ],
                references_prior_beats=b.get("references_prior_beats", []),
                **_typed_beat_fields(b, analogy_id_map),
            ))

        return VideoScript(
            title=data.get("title", ""),
            beats=beats,
            analogies=list(known_analogies.values()),
        )

    def _complete_unified_script_json(self, prompt: str) -> Dict[str, Any]:
        """Run the unified whole-document script call and return parsed JSON.

        Prefers Claude Sonnet 5 (per-task model tiering — this single call sets
        breadth coverage and benefits most from strong reasoning). Falls back to
        the OpenAI client (gpt-5.4) when no Anthropic client is configured, so a
        key-only-OpenAI deployment keeps working.
        """
        client = getattr(self, "anthropic_client", None)
        if client is not None:
            response = client.messages.create(
                model=UNIFIED_SCRIPT_MODEL,
                max_tokens=8000,
                # Cost-conscious: no adaptive thinking for a structured JSON task.
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                getattr(block, "text", "")
                for block in response.content
                if getattr(block, "type", None) == "text"
            )
            return json.loads(_extract_json_object(text))

        response = self.client.chat.completions.create(
            model="gpt-5.4",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def _generate_segmented_video_script(
        self,
        segments: List[SourceSegment],
        config: Configuration,
        review_feedback: List[str],
    ):
        """Generate a global VideoScript through one LLM call per source segment."""
        from src.schemas.video_script import Beat, VideoScript

        all_beats: List[Beat] = []
        covered_concepts: List[str] = []
        known_analogies: Dict[str, Analogy] = {}
        title = ""

        for segment in segments:
            prompt = self._build_segment_script_prompt(
                segment=segment,
                config=config,
                covered_concepts=covered_concepts,
                review_feedback=review_feedback,
                known_analogies=list(known_analogies.values()),
            )
            # Failure isolation: a single bad segment response (API error, invalid
            # JSON, or a malformed beat) must never abort the whole storyboard —
            # skip it and keep the rest. LLMs occasionally emit `narration: null`
            # or drop a required field; that one beat is dropped, not the run.
            try:
                response = self.client.chat.completions.create(
                    model=SEGMENT_SCRIPT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                data = json.loads(response.choices[0].message.content)
            except Exception as exc:
                logger.warning(
                    "Segment %s script generation failed (%s); skipping segment.",
                    segment.segment_id, exc,
                )
                continue

            if not isinstance(data, dict):
                continue
            if not title:
                title = data.get("title") or ""

            pending_beats = data.get("beats")
            if not isinstance(pending_beats, list):
                continue
            analogy_id_map = _parse_analogies(data.get("analogies"), known_analogies)

            # Pass 1 — keep the usable beats and pre-assign global ids so
            # forward references WITHIN the segment resolve too (the old
            # incremental map only saw beats already processed, leaking
            # "unknown or forward beat" warnings from the schema validator).
            usable: List[Dict[str, Any]] = []
            local_to_global: Dict[str, str] = {}
            next_index = len(all_beats) + 1
            for raw in pending_beats:
                if not isinstance(raw, dict):
                    continue
                narration = raw.get("narration")
                if not isinstance(narration, str) or not narration.strip():
                    logger.warning(
                        "Segment %s returned a beat with missing/invalid narration; dropping it.",
                        segment.segment_id,
                    )
                    continue
                local_id = str(raw.get("beat_id") or f"__auto_{next_index}")
                local_to_global[local_id] = f"beat_{next_index:03d}"
                usable.append(raw)
                next_index += 1

            # Pass 2 — build beats, resolving refs against this segment's map
            # plus all previously emitted global ids. Anything else is a
            # hallucinated / cross-segment-unknowable id: drop it here at
            # DEBUG so the VideoScript validator's WARNING only fires for
            # genuine anomalies.
            known_global_ids = {b.beat_id for b in all_beats}
            for raw in usable:
                local_id = str(raw.get("beat_id") or "")
                global_id = local_to_global.get(local_id) or f"beat_{len(all_beats) + 1:03d}"

                concepts = [
                    canonical
                    for label in (raw.get("introduces_concepts") or [])
                    if isinstance(label, str) and (canonical := _canonicalize_concept(label))
                ]
                for concept in concepts:
                    if concept not in covered_concepts:
                        covered_concepts.append(concept)

                refs: List[str] = []
                for ref in (raw.get("references_prior_beats") or []):
                    if not isinstance(ref, str):
                        continue
                    resolved = local_to_global.get(ref) or (ref if ref in known_global_ids else None)
                    # Keep only *backward* references — a forward ref would be
                    # dropped (with a WARNING) by the VideoScript validator.
                    if resolved and resolved < global_id:
                        refs.append(resolved)
                    else:
                        logger.debug(
                            "Segment %s: dropping unresolvable/forward beat reference %r.",
                            segment.segment_id, ref,
                        )
                visual_hint = raw.get("visual_hint")
                if not isinstance(visual_hint, str):
                    visual_hint = ""

                try:
                    all_beats.append(Beat(
                        beat_id=global_id,
                        narration=raw["narration"],
                        visual_hint=visual_hint,
                        introduces_concepts=concepts,
                        references_prior_beats=refs,
                        source_segment_id=segment.segment_id,
                        **_typed_beat_fields(raw, analogy_id_map),
                    ))
                except Exception as exc:
                    logger.warning(
                        "Dropping malformed beat in segment %s: %s", segment.segment_id, exc,
                    )

        # VideoScript requires >= 1 beat. If every segment failed, ship a minimal
        # placeholder so the pipeline degrades gracefully instead of aborting.
        if not all_beats:
            logger.warning("All segment script calls produced no usable beats; using placeholder.")
            all_beats = [Beat(
                beat_id="beat_001",
                narration="This video introduces the key ideas from the source material.",
                visual_hint="Title card with the main concept.",
                introduces_concepts=["overview"],
                references_prior_beats=[],
                source_segment_id=segments[0].segment_id if segments else "",
            )]

        # Keep only analogies some surviving beat uses (a dropped segment or
        # beat can orphan a declaration).
        used = {b.analogy_id for b in all_beats if b.analogy_id}
        analogies = [a for a in known_analogies.values() if a.analogy_id in used]
        return VideoScript(title=title, beats=all_beats, analogies=analogies)

    def _build_segment_script_prompt(
        self,
        segment: SourceSegment,
        config: Configuration,
        covered_concepts: List[str],
        review_feedback: List[str],
        known_analogies: Optional[List[Analogy]] = None,
    ) -> str:
        audience_line = {
            "high-school": "Use real-world analogies; avoid jargon.",
            "graduate": "Lean into theoretical structure and generalizations.",
        }.get(config.audience, "Balance intuition with mathematical structure.")

        verbosity_line = {
            "low": "Be terse - only the core ideas.",
            "high": "Be detailed; explain why each step matters.",
        }.get(config.verbosity, "Balanced explanation.")

        math_block = ""
        if segment.math_latex:
            math_block = (
                "\nFORMULAS IN THIS SEGMENT:\n"
                + "\n".join(f"- {expr}" for expr in segment.math_latex)
                + "\n"
            )
        already_block = ""
        if covered_concepts:
            already_block = (
                "\nCONCEPTS ALREADY COVERED:\n"
                + "\n".join(f"- {concept}" for concept in covered_concepts)
                + "\nReference these by name only; do not redefine them.\n"
            )
        review_block = ""
        if review_feedback:
            review_block = (
                "\nREVISION FEEDBACK:\n"
                + "\n".join(f"- {item}" for item in review_feedback[:8])
                + "\n"
            )
        analogies_block = ""
        if known_analogies:
            analogies_block = (
                "\nANALOGIES ALREADY INTRODUCED (call back by analogy_id; do not re-declare):\n"
                + "\n".join(f"- {a.analogy_id}: {a.label}" for a in known_analogies)
                + "\n"
            )

        target_beats = max(1, self._target_beat_count(segment.text))

        return f"""You are a Grant Sanderson-style educational scriptwriter (3Blue1Brown).

SOURCE SEGMENT:
- segment_id: {segment.segment_id}
- order: {segment.order}
- pages: {segment.page_start}-{segment.page_end}
- heading: {segment.heading or "(none)"}

TEXT:
{segment.text}
{math_block}{already_block}{analogies_block}{review_block}
TASK: Explain THIS segment fully as part of a larger continuous video.
Do not preview or re-explain other sections. If a concept was already covered,
reference it by name without redefining it.

TARGET: Produce approximately {target_beats} beat(s). Each beat should be 30-60
words of narration — one vignette-sized visual idea. Split only when this
segment needs multiple visual/narrative steps.

{TYPED_BEAT_PROMPT_RULES}

STYLE:
- Audience ({config.audience}): {audience_line}
- Verbosity ({config.verbosity}): {verbosity_line}
- Tone: Friendly, intuitive, conversational.

Return JSON in this exact shape:
{{
  "title": "...",
  "analogies": [
    {{
      "analogy_id": "analogy_short_name",
      "label": "what the analogy is",
      "target_concepts": ["concept_label"],
      "mapping": {{"analogy term": "concept"}}
    }}
  ],
  "beats": [
    {{
      "beat_id": "beat_local_001",
      "narration": "spoken text...",
      "visual_hint": "what is on screen during this beat",
      "introduces_concepts": ["concept_label"],
      "references_prior_beats": [],
      "role": "core",
      "depth": 0,
      "analogy_id": null
    }}
  ]
}}
"""

    def _target_beat_count(self, source_text: str) -> int:
        return max(1, round(len(source_text.split()) / self.WORDS_PER_BEAT))

    def _effective_max_scenes(self, config) -> int:
        """Scene ceiling for this run: the static MAX_SCENES unless the user
        requested a target length, in which case the tighter length-derived
        budget wins (min of the two — never exceed the explosion guard)."""
        target_minutes = getattr(config, "target_minutes", None)
        if not target_minutes:
            return self.MAX_SCENES
        budget = max(1, round(float(target_minutes) * 60.0 / self.TARGET_SCENE_DURATION))
        return min(self.MAX_SCENES, budget)

    def _chunk_script_into_scenes(self, script, max_scenes: Optional[int] = None) -> List[Scene]:
        """Group beats into ~8-45 s vignette scenes (Task 4 chunker, ADR-005).

        Greedy packer: walk the beats in order, accumulating beats into the
        current chunk until adding the next beat would push the chunk over
        ``MAX_SCENE_DURATION``. A beat is atomic — never split. If a single
        beat exceeds ``MAX_SCENE_DURATION``, it becomes its own oversized
        chunk and a WARNING is logged (F3 mitigation).

        Source beats are stored on ``Scene.source_beats`` as a real Pydantic
        field so they survive ``model_dump()`` round-trips (F5 mitigation).
        """
        from src.schemas.video_script import Beat, VideoScript

        def beat_duration(b) -> float:
            return len(b.narration.split()) / self.WORDS_PER_SECOND

        def pack(target_dur: float, max_dur: float) -> List[List[Beat]]:
            """Greedy pack beats into chunks, never crossing a segment boundary."""
            packed: List[List[Beat]] = []
            cur: List[Beat] = []
            cur_dur = 0.0
            for beat in script.beats:
                bd = beat_duration(beat)
                beat_segment = getattr(beat, "source_segment_id", "")
                cur_segment_ids = {
                    getattr(existing, "source_segment_id", "")
                    for existing in cur
                    if getattr(existing, "source_segment_id", "")
                }
                crosses_segment = bool(
                    cur and beat_segment and cur_segment_ids
                    and beat_segment not in cur_segment_ids
                )
                if cur and (crosses_segment or cur_dur + bd > max_dur):
                    packed.append(cur)
                    cur, cur_dur = [], 0.0
                cur.append(beat)
                cur_dur += bd
                if cur_dur >= target_dur:
                    packed.append(cur)
                    cur, cur_dur = [], 0.0
            if cur:
                packed.append(cur)
            return packed

        for beat in script.beats:
            bd = beat_duration(beat)
            if bd > self.MAX_SCENE_DURATION:
                # F3: oversized beat — log a warning and ship it as a single chunk.
                logger.warning(
                    "Beat %s narration is %.1fs (>%.1fs MAX_SCENE_DURATION); "
                    "shipping as oversized chunk. Tighten the script-LLM prompt "
                    "to keep beats under 80 words.",
                    beat.beat_id, bd, self.MAX_SCENE_DURATION,
                )

        # Scene-count governance: segments were capped upstream, but beats per
        # segment can still explode the scene count (a real 14-page run yielded
        # 90 scenes — choppy, slow, expensive). Escalate the packing target
        # until the count fits MAX_SCENES. Floor = number of distinct segments
        # (chunks never cross segment boundaries), which the segmenter caps.
        scene_cap = self.MAX_SCENES if max_scenes is None else max(1, int(max_scenes))
        target = float(self.TARGET_SCENE_DURATION)
        max_dur = float(self.MAX_SCENE_DURATION)
        chunks = pack(target, max_dur)
        while len(chunks) > scene_cap and target < 120.0:
            target *= 1.4
            max_dur = max(max_dur, target * 1.8)
            chunks = pack(target, max_dur)
        if len(chunks) > scene_cap:
            logger.warning(
                "Chunker could not reduce scene count below cap=%d "
                "(got %d) — segment boundaries dominate.",
                scene_cap, len(chunks),
            )

        # Concept tracking: each scene's concept_id is the first canonical
        # concept introduced in its beats. Reference-only chunks inherit the
        # most-recently introduced concept so StoryboardJSON's cross-validator
        # stays satisfied.
        most_recent_concept: Optional[str] = None
        # Pre-compute the global ledger so we have a fallback for purely-
        # reference chunks at the very start of the video (none introduced yet).
        for beat in script.beats:
            for c in beat.introduces_concepts:
                most_recent_concept = c
                break
            if most_recent_concept:
                break
        # Reset for the actual loop
        most_recent_concept_iter = None

        scenes: List[Scene] = []
        for idx, chunk_beats in enumerate(chunks):
            narration = "\n\n".join(b.narration for b in chunk_beats)
            word_count = sum(len(b.narration.split()) for b in chunk_beats)
            # Honour the floor (MIN_SCENE_DURATION) but allow oversized
            # chunks to keep their natural duration.
            duration = max(self.MIN_SCENE_DURATION, word_count / self.WORDS_PER_SECOND)
            scene_id = f"scene_{idx + 1:03d}"

            # Pick the chunk's primary concept: first newly introduced in this
            # chunk, else the most-recent introduced earlier, else the global
            # first-introduced (the oldest chunk-1 case).
            chunk_first_concept: Optional[str] = None
            for b in chunk_beats:
                for c in b.introduces_concepts:
                    chunk_first_concept = c
                    break
                if chunk_first_concept:
                    break
            scene_concept_id = (
                chunk_first_concept
                or most_recent_concept_iter
                or most_recent_concept
                or chunk_beats[0].beat_id  # absolute fallback (no concepts anywhere)
            )
            if chunk_first_concept:
                most_recent_concept_iter = chunk_first_concept
            source_segment_ids: List[str] = []
            for beat in chunk_beats:
                segment_id = getattr(beat, "source_segment_id", "")
                if segment_id and segment_id not in source_segment_ids:
                    source_segment_ids.append(segment_id)

            scenes.append(Scene(
                scene_id=scene_id,
                scene_index=idx,
                concept_id=scene_concept_id,
                narration=narration,
                visual_intent=self._empty_visual_intent_placeholder(),
                duration_estimate=round(duration, 1),
                dependencies=[f"scene_{idx:03d}"] if idx > 0 else [],
                source_beats=chunk_beats,
                source_segment_ids=source_segment_ids,
            ))

        return scenes

    def _empty_visual_intent_placeholder(self) -> VisualIntent:
        """Visual intent left empty by the chunker; populated in Task 7
        by the per-chunk visual planner."""
        return VisualIntent(
            mathematical_objects=[],
            transformations=[],
            beats=[],
            scene_state_in=SceneState(),
            scene_state_out=SceneState(),
            emphasis_points=[],
        )

    def _mock_video_script_fallback(
        self,
        ocr_output,
        config,
        segments: Optional[List[SourceSegment]] = None,
    ):
        """Deterministic fallback for runs without OPENAI_API_KEY.

        Returns a minimal 2-beat script so the pipeline degrades gracefully
        instead of aborting. Real PDFs running without a key will produce a
        2-chunk video; that's intentional — set OPENAI_API_KEY for real output.
        """
        from src.schemas.video_script import VideoScript, Beat
        title_text = ocr_output.pdf_metadata.filename
        if segments:
            beats = []
            previous_beat_id = ""
            for index, segment in enumerate(segments, start=1):
                beat_id = f"beat_{index:03d}"
                heading = segment.heading or f"source segment {index}"
                preview = segment.text.strip().split(".")[0][:160] or heading
                beats.append(Beat(
                    beat_id=beat_id,
                    narration=(
                        f"This part covers {heading}. "
                        f"The key idea is: {preview}."
                    ),
                    visual_hint="Show the source idea as a clear visual summary.",
                    introduces_concepts=[f"segment_{index:03d}"],
                    references_prior_beats=[previous_beat_id] if previous_beat_id else [],
                    source_segment_id=segment.segment_id,
                ))
                previous_beat_id = beat_id
            return VideoScript(title=f"Overview of {title_text}", beats=beats)

        return VideoScript(
            title=f"Overview of {title_text}",
            beats=[
                Beat(
                    beat_id="beat_001",
                    narration=(
                        f"This video introduces the core concepts in {title_text}. "
                        "We start with the basic ideas and build up step by step."
                    ),
                    visual_hint="Title card with the document name and a short outline.",
                    introduces_concepts=["overview"],
                    references_prior_beats=[],
                ),
                Beat(
                    beat_id="beat_002",
                    narration=(
                        "Now we walk through the most important formulas and what "
                        "they mean intuitively."
                    ),
                    visual_hint="Show the key formulas one at a time, with simple geometric illustrations.",
                    introduces_concepts=["main_formulas"],
                    references_prior_beats=["beat_001"],
                ),
            ],
        )

    def _generate_fallback_visual_intent(self, narration: str) -> VisualIntent:
        """Build a minimal visual intent from the narration text.

        Emits closed-vocabulary catalog primitives (Task 12) so the
        animation generator's catalog path handles the fallback the same
        way it handles real LLM output. The legacy if/elif heuristic chain
        in ``AnimationCodeGeneratorAgent._generate_mathematical_objects``
        is no longer reached for fallback intents.

        Layout (Task 13): every emitted object carries a bbox + interval
        so the layout validator can run unchanged. Positions are picked
        from a small canonical stack (title on top, equation in centre,
        archetype visual below) inside the safe area.

        Visual-impact: when the narration classifies to a visual archetype,
        its ``fallback_primitive`` is emitted too, so even degraded/keyless
        runs draw something rather than shipping a text-only slide.
        """
        from src.schemas.storyboard import BBox, TimeInterval

        first_sentence = (
            narration.split(".")[0].strip() if narration else "Introduction"
        )
        title_text = first_sentence[:80] or "Introduction"

        # Title object — Text primitive at the top of the safe area.
        objs = [
            MathematicalObject(
                object_id="intro_text",
                object_role="title",
                purpose="Introduce the scene's narrated idea in the first beat",
                supports_beat_ids=["beat_001"],
                object_type="text",
                content=title_text,
                primitive="Text",
                args={"text": title_text, "font_size": 36},
                bbox=BBox(x=0.0, y=2.5, w=10.0, h=1.0),
                interval=TimeInterval(t_in=0.0, t_out=15.0),
                style=ObjectStyle(color="BLUE", position="top"),
            )
        ]

        # If narration mentions trig/equations, add a centred MathTex.
        import re as _re
        latex_hints = _re.findall(
            r"(sin|cos|tan|theta|pi|equation|formula)", narration.lower()
        )
        if latex_hints:
            tex = r"\sin(\theta) = \frac{\text{opposite}}{\text{hypotenuse}}"
            objs.append(MathematicalObject(
                object_id="main_equation",
                object_role="main_equation",
                purpose="Show the equation explicitly referenced by the narration",
                supports_beat_ids=["beat_001"],
                object_type="equation",
                notation_key="main_equation",
                content=tex,
                primitive="MathTex",
                args={"tex": tex},
                bbox=BBox(x=0.0, y=0.0, w=8.0, h=1.5),
                interval=TimeInterval(t_in=0.0, t_out=15.0),
                style=ObjectStyle(color="YELLOW", position="center"),
            ))

        # Archetype visual — the concept family's proven primitive, placed in
        # the lower half of the safe area so degraded runs still DRAW the idea.
        archetype = classify_concept(narration)
        if archetype is not None:
            primitive_name, primitive_args, object_type = archetype.fallback_primitive
            objs.append(MathematicalObject(
                object_id="archetype_visual",
                object_role="concept_visual",
                purpose="Depict the narrated concept with a non-text visual",
                supports_beat_ids=["beat_001"],
                object_type=object_type,
                content=archetype.description,
                primitive=primitive_name,
                args=dict(primitive_args),
                bbox=BBox(x=0.0, y=-2.0, w=6.0, h=2.5),
                interval=TimeInterval(t_in=0.0, t_out=15.0),
                style=ObjectStyle(color="GREEN", position="bottom"),
            ))

        active_object_ids = [obj.object_id for obj in objs if obj.object_id]
        trans = [
            Transformation(
                transformation_type="fade_in",
                source_object_id=active_object_ids[0] if active_object_ids else "intro_text",
                target_object_id=active_object_ids[0] if active_object_ids else "intro_text",
                continuity_action="create",
                beat_id="beat_001",
                parameters={"animation_hint": "Write"},
                timing=2.0,
            )
        ]

        return VisualIntent(
            mathematical_objects=objs,
            transformations=trans,
            beats=[
                VisualBeat(
                    beat_id="beat_001",
                    narration_excerpt=title_text,
                    start_time=0.0,
                    end_time=5.0,
                    active_object_ids=active_object_ids,
                    camera_action="focus",
                    beat_goal="Introduce the primary visual elements for the scene",
                )
            ],
            scene_state_in=SceneState(),
            scene_state_out=SceneState(active_object_ids=active_object_ids),
            emphasis_points=[],
        )
    
    def validate_input(self, input_data: AgentInput) -> bool:
        """Validate input is a ScriptArchitectInput object
        
        Args:
            input_data: Input object to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        if not isinstance(input_data, ScriptArchitectInput):
            return False
        
        # Validate OCR output and configuration are present
        if not hasattr(input_data, 'ocr_output') or not hasattr(input_data, 'configuration'):
            return False
        
        return isinstance(input_data.ocr_output, OCROutput) and isinstance(input_data.configuration, Configuration)
    
    def get_retry_policy(self) -> RetryPolicy:
        """Return retry policy for Script/Storyboard Architect Agent
        
        LLM calls may fail due to invalid JSON generation or API issues.
        Use retry with schema enforcement for JSON issues.
        
        Returns:
            RetryPolicy with limited retries
        """
        return RetryPolicy(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.CONSTANT,
            base_delay_seconds=1.0,
            max_delay_seconds=5.0,
            retryable_errors=[
                "INVALID_JSON",
                "SCHEMA_VIOLATION",
                "API_TIMEOUT",
                "RATE_LIMIT_EXCEEDED"
            ]
        )
