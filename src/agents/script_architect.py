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
from typing import Any, Dict, List, Optional

import openai


from src.agents.base import (
    Agent,
    AgentExecutionError,
    AgentInput,
    AgentOutput,
    BackoffStrategy,
    RetryPolicy,
)
from src.schemas.ocr_output import OCROutput
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
from src.utils.manim_engine_contract import build_visual_intent_contract_prompt, CANONICAL_RENDER_HINTS, normalize_render_hint
from src.utils.manim_primitives import catalog_for_prompt

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


class ScriptArchitectInput(AgentInput):
    """Input for Script/Storyboard Architect Agent"""
    
    def __init__(
        self,
        ocr_output: OCROutput,
        configuration: Configuration,
        review_feedback: Optional[List[str]] = None,
    ):
        """Initialize input
        
        Args:
            ocr_output: OCR output from Vision+OCR Agent
            configuration: Configuration parameters for storyboard generation
            review_feedback: Optional critique to apply on a regeneration pass
        """
        
        self.ocr_output = ocr_output
        self.configuration = configuration
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
    MIN_SCENE_DURATION = 30.0  # Minimum scene duration in seconds
    MAX_SCENE_DURATION = 90.0  # Maximum scene duration in seconds
    TARGET_SCENE_DURATION = 60.0  # Target scene duration in seconds
    WORDS_PER_SECOND = 3.0  # Average narration speed used by prompt + TTS estimates
    MAX_JSON_RETRIES = 3  # Maximum retries for invalid JSON
    
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
        super().__init__()
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            if self.strict:
                raise AgentExecutionError(
                    "MISSING_API_KEY",
                    "OPENAI_API_KEY is required in --strict mode for ScriptArchitect"
                )
            self.client = None
            logger.warning("OPENAI_API_KEY not found. ScriptArchitect will use mock fallbacks.")
    
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

        logger.info(f"Starting storyboard generation for: {ocr_output.pdf_metadata.filename}")
        logger.info(
            f"Configuration: verbosity={config.verbosity}, depth={config.depth}, "
            f"audience={config.audience}"
        )
        if review_feedback:
            logger.info(f"Applying {len(review_feedback)} storyboard revision hints")

        try:
            # Phase 1: ONE unified video script.
            video_script = self._generate_video_script(ocr_output, config, review_feedback)
            logger.info(
                f"Generated video script: {len(video_script.beats)} beats, "
                f"{len(video_script.concept_ledger())} unique concepts"
            )

            # Phase 2: deterministic chunker (Task 4).
            scenes = self._chunk_script_into_scenes(video_script)
            logger.info(f"Chunked script into {len(scenes)} scenes")

            # Phase 3: per-chunk visual intent.
            # Until Task 7 lands the chunk-aware visual planner, we use the
            # narration-derived fallback so the pipeline produces a valid
            # storyboard end-to-end. F6: any per-chunk failure is isolated
            # and never aborts the whole storyboard.
            visual_failures: List[Dict[str, str]] = []
            for scene in scenes:
                try:
                    scene.visual_intent = self._generate_visual_intent_for_chunk(
                        scene=scene,
                        video_script=video_script,
                        config=config,
                        review_feedback=review_feedback,
                    )
                except Exception as exc:
                    logger.warning(
                        "Visual intent failed for %s: %s — falling back to narration-derived intent",
                        scene.scene_id, exc,
                    )
                    scene.visual_intent = self._generate_fallback_visual_intent(scene.narration)
                    visual_failures.append({"scene_id": scene.scene_id, "error": str(exc)})

            if visual_failures:
                logger.warning(
                    "Visual intent fallback used for %d/%d scenes",
                    len(visual_failures), len(scenes),
                )

            storyboard = self._assemble_storyboard_from_script(
                ocr_output, config, video_script, scenes,
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

    def _assemble_storyboard_from_script(
        self,
        ocr_output: OCROutput,
        config: Configuration,
        video_script,
        scenes: List[Scene],
    ) -> StoryboardJSON:
        """Build a StoryboardJSON from a unified VideoScript + chunked scenes.

        ``concept_hierarchy`` is derived from the script's deduped concept
        ledger — one entry per canonical concept in first-introduction order,
        with each concept depending on every concept introduced before it
        (a flat dependency line that mirrors the linear narrative).
        """
        pdf_metadata = PDFMetadataStoryboard(
            filename=ocr_output.pdf_metadata.filename,
            page_count=ocr_output.pdf_metadata.page_count,
        )
        ledger = video_script.concept_ledger()
        concept_hierarchy = [
            ConceptHierarchy(
                concept_id=concept,
                concept_name=concept.replace("_", " ").title(),
                dependencies=ledger[:i],
            )
            for i, concept in enumerate(ledger)
        ]
        return StoryboardJSON(
            pdf_metadata=pdf_metadata,
            configuration=config,
            concept_hierarchy=concept_hierarchy,
            scenes=scenes,
        )

    def _generate_visual_intent_for_chunk(
        self,
        scene: Scene,
        video_script,
        config: Configuration,
        review_feedback: List[str],
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
            "\n".join(f"- {b.visual_hint}" for b in chunk_beats)
            or "- (no specific hints)"
        )

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

        engine_contract = build_visual_intent_contract_prompt()
        catalog_block = catalog_for_prompt()

        prompt = (
            "You are a 3Blue1Brown animation architect.\n\n"
            "--- CHUNK NARRATION (audio for THIS scene only) ---\n"
            f"{scene.narration}\n"
            "--- END NARRATION ---\n\n"
            f"NEW CONCEPTS INTRODUCED IN THIS CHUNK: "
            f"{', '.join(chunk_concepts) or '(none)'}\n"
            f"{earlier_block}"
            "VISUAL HINTS FROM THE SCRIPT WRITER:\n"
            f"{visual_hints}\n\n"
            f"TIMING: {word_count} words at ~3 wps => visuals should fill ~{target}s.\n"
            f"{review_block}"
            f"{engine_contract}\n"
            "CLOSED VOCABULARY:\n"
            f"{catalog_block}\n\n"
            "For every mathematical_object you emit, set `primitive` to one of "
            "the MOBJECT names above and `args` to its kwargs. Free-form "
            "`content` is ONLY permitted for MathTex / Tex / Text (where it "
            "goes into the `tex` or `text` arg).\n\n"
            "LAYOUT (x, y, t):\n"
            "  - For every object also emit `bbox` = {x, y, w, h} centred on the\n"
            "    object's intended position in Manim units AND `interval` =\n"
            "    {t_in, t_out} — when in scene-time the object is visible.\n"
            "  - SAFE AREA: keep bbox.x in [-6.5, 6.5] and bbox.y in [-3.5, 3.5].\n"
            "  - Two objects may share the same (x, y) only if their intervals\n"
            "    are disjoint (one fades out before the other appears).\n\n"
            "Return JSON with keys: objects (list), transformations (list), "
            "beats (list), scene_state_in (object), scene_state_out (object), "
            "emphasis_points (list)."
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-5",
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

        try:
            objs = [MathematicalObject.model_validate(o) for o in raw_objects]
            trans = [Transformation.model_validate(t) for t in raw_trans]
            beats = [VisualBeat.model_validate(b) for b in raw_beats]
            emphasis = [EmphasisPoint.model_validate(e) for e in raw_emphasis]
            state_in = SceneState.model_validate(raw_in if isinstance(raw_in, dict) else {})
            state_out = SceneState.model_validate(raw_out if isinstance(raw_out, dict) else {})
        except Exception as exc:
            logger.warning(
                "Visual-intent JSON failed schema validation (%s); using fallback.",
                exc,
            )
            return self._generate_fallback_visual_intent(fallback_narration)

        return VisualIntent(
            mathematical_objects=objs,
            transformations=trans,
            beats=beats,
            scene_state_in=state_in,
            scene_state_out=state_out,
            emphasis_points=emphasis,
        )
    
    def _generate_video_script(
        self,
        ocr_output: OCROutput,
        config: Configuration,
        review_feedback: Optional[List[str]] = None,
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
            return self._mock_video_script_fallback(ocr_output, config)

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
                + "\n".join(f"  • {e}" for e in math_expressions[:30])
                + "\n"
            )

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
{source_text[:8000]}
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
6. Aim for 8–20 beats total. Each beat MUST be 40–80 words of narration —
   if a topic needs more, split into multiple beats with references_prior_beats.

STYLE:
- Audience ({config.audience}): {audience_line}
- Verbosity ({config.verbosity}): {verbosity_line}
- Tone: Friendly, intuitive, conversational.
- Visual hint per beat: 1–2 sentences describing what is on screen.
{review_block}

Return JSON in this EXACT shape:
{{
  "title": "...",
  "beats": [
    {{
      "beat_id": "beat_001",
      "narration": "spoken text...",
      "visual_hint": "what is on screen during this beat",
      "introduces_concepts": ["concept_label", ...],
      "references_prior_beats": ["beat_001", ...]
    }}
  ]
}}
"""

        response = self.client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)

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
            ))

        return VideoScript(title=data.get("title", ""), beats=beats)

    def _chunk_script_into_scenes(self, script) -> List[Scene]:
        """Group beats into ~30-90 s scenes (Task 4 chunker).

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

        chunks: List[List[Beat]] = []
        current: List[Beat] = []
        current_dur = 0.0

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
            # If adding this beat overflows the current chunk, flush first.
            if current and current_dur + bd > self.MAX_SCENE_DURATION:
                chunks.append(current)
                current, current_dur = [], 0.0
            current.append(beat)
            current_dur += bd
            # Close the chunk once we've crossed TARGET to avoid trailing tinies.
            if current_dur >= self.TARGET_SCENE_DURATION:
                chunks.append(current)
                current, current_dur = [], 0.0
        if current:
            chunks.append(current)

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

            scenes.append(Scene(
                scene_id=scene_id,
                scene_index=idx,
                concept_id=scene_concept_id,
                narration=narration,
                visual_intent=self._empty_visual_intent_placeholder(),
                duration_estimate=round(duration, 1),
                dependencies=[f"scene_{idx:03d}"] if idx > 0 else [],
                source_beats=chunk_beats,
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

    def _mock_video_script_fallback(self, ocr_output, config):
        """Deterministic fallback for runs without OPENAI_API_KEY.

        Returns a minimal 2-beat script so the pipeline degrades gracefully
        instead of aborting. Real PDFs running without a key will produce a
        2-chunk video; that's intentional — set OPENAI_API_KEY for real output.
        """
        from src.schemas.video_script import VideoScript, Beat
        title_text = ocr_output.pdf_metadata.filename
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
        from a small "canonical 2-object stack" (title on top, equation
        in centre) inside the safe area.
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
                object_type="equation",
                notation_key="main_equation",
                content=tex,
                primitive="MathTex",
                args={"tex": tex},
                bbox=BBox(x=0.0, y=0.0, w=8.0, h=1.5),
                interval=TimeInterval(t_in=0.0, t_out=15.0),
                style=ObjectStyle(color="YELLOW", position="center"),
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
