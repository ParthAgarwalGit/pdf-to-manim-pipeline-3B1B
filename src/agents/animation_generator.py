"""Animation Code Generator Agent for converting scene specifications to Manim code.

This agent generates executable Python/Manim code from scene specifications,
including mathematical objects, transformations, and timing synchronization.
"""

import ast
import re
import anthropic 
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.agents.base import Agent, AgentExecutionError, AgentInput, AgentOutput, BackoffStrategy, RetryPolicy
from src.schemas.animation_code import AnimationCodeObject
from src.schemas.scene_processing import SceneProcessingInput, SharedContext
from src.utils.manim_engine_contract import (
    build_animation_codegen_contract_prompt,
    build_3b1b_quality_prompt,
)

# Per-task model tiering (see docs). Manim code generation is the quality- and
# retry-critical path, so it runs on Claude Sonnet 5 — best coding/instruction-
# following per dollar; higher first-pass validity means fewer paid re-renders.
CODE_GEN_MODEL = "claude-sonnet-5"


from functools import lru_cache


@lru_cache(maxsize=1)
def _codegen_static_system_prompt() -> str:
    """The byte-identical contract block of every code-gen system prompt.

    Sent as the FIRST system block with ``cache_control`` so Anthropic's
    prompt cache serves it at ~0.1x input price on every scene after the
    first. Per-scene data (class name, pacing, briefing, colors) goes in a
    second, uncached block — nothing here may vary between scenes.
    """
    engine_contract = build_animation_codegen_contract_prompt()
    quality_contract = build_3b1b_quality_prompt()
    return (
        "You are an expert Manim Community v0.19 developer who creates\n"
        "3Blue1Brown-quality educational animations. The scene's class name,\n"
        "pacing targets, and context follow AFTER these standing rules.\n\n"
        f"{engine_contract}\n\n{quality_contract}\n"
        + """
═══════════════════════  SCREEN LAYOUT (NO OVERLAP)  ═══════════════════════
A LayoutTracker is imported for you.  Use it to REGISTER every Mobject after
you place it, and to PLACE LABELS so they never overlap other objects.

    layout = LayoutTracker()          # create once at the top of construct()

    # After creating and positioning any object, register it:
    triangle = Polygon(A, B, C, color=WHITE)
    triangle.shift(LEFT * 2)
    layout.register("triangle", triangle)

    # To place a label near a registered object without overlap:
    opp_label = Text("opposite = 6m", font_size=24, color=GREEN)
    layout.place_label(opp_label, "opp_label", anchor="triangle",
                       preferred=RIGHT, buff=0.3)
    #   → tries RIGHT first; if that overlaps something it auto-tries
    #     UP, LEFT, DOWN, UR, UL, DR, DL until it finds free space.
    #   → stays within max_distance=1.5 of the anchor (labels stay close).

    # For angle arcs at a vertex — place arc at the vertex, register,
    # then use place_label for the θ symbol:
    arc = Arc(radius=0.5, start_angle=0, angle=PI/4, color=YELLOW)
    arc.shift(triangle.get_vertices()[0])      # put arc AT the vertex
    layout.register("theta_arc", arc)
    theta = MathTex(r"\\theta", color=YELLOW)
    layout.place_label(theta, "theta", anchor="theta_arc",
                       preferred=RIGHT, buff=0.2)

    # When you FadeOut objects, unregister them to free space:
    layout.unregister("triangle")
    layout.unregister("opp_label")

RULES:
1. Create `layout = LayoutTracker()` as the FIRST line of construct().
2. Call `layout.register(name, mob)` after EVERY Mobject is positioned.
3. ALWAYS use `layout.place_label(...)` for Text/MathTex labels — never
   raw `.next_to()` or `.move_to()` for labels.  The tracker prevents overlap.
4. Call `layout.unregister(name)` before or after FadeOut so space is freed.
5. Keep font sizes readable:  Text(font_size=28..36), MathTex default is fine.
6. Scale graphs/axes to fit reasonably (x_length ≤ 5, y_length ≤ 3).
7. Angle arcs: position the arc at the ACTUAL VERTEX of the shape using
   `.shift(vertex_point)`, then register it, then place_label the θ symbol
   anchored to the arc.  This keeps the angle annotation touching the shape.
8. FRAME GUARD — after composing any VGroup or calling .arrange(...), clamp it:
       if group.width > 12: group.scale_to_fit_width(12)
       if group.height > 6.5: group.scale_to_fit_height(6.5)
   then re-position it inside the safe area (|x| <= 6.5, |y| <= 3.5). Nothing
   may extend past the frame edge at any point of the animation, including
   mid-shift positions.

═══════════════════════  NARRATOR MASCOT (SIG)  ═══════════════════════
If and only if the storyboard intent includes a `pi_creature` object, use the
pre-built blob mascot helpers (already imported — do NOT draw a character from
raw shapes yourself):

    sig = create_pi_teacher(height=1.2).to_corner(DL, buff=0.4)
    bubble = create_speech_bubble("Key insight text", sig)

    self.play(FadeIn(sig, shift=UP * 0.3), run_time=0.8)
    self.play(Write(bubble), run_time=1.0)
    self.wait(2)
    self.play(FadeOut(bubble), FadeOut(sig), run_time=0.8)

RULES:
- At most ONE mascot appearance per scene, in a corner, small (height <= 1.4).
- The mascot never overlaps the math content and leaves before dense content
  animates.
- Do not invent helper objects that are not present in the visual intent.

═══════════════════════  STYLE  ═══════════════════════
- Black background (default).  Use BLUE, YELLOW, GREEN, RED for emphasis.
- Use MathTex(r"...") for ALL equations.  Use Text("...") for plain labels.
- Prefer Write / Create for new objects, ReplacementTransform for morphing.
- Smooth pacing: run_time between 1.0 and 3.0 for most animations.
- After every major visual change, add a brief self.wait() so the reveal lands.

═══════════════════════  VISUAL PLAN (HONOR IT)  ═══════════════════════
Each entry in `objects` is a PRE-PLANNED visual. Treat it as the intended
design, not a loose suggestion:
- `primitive` + `args`  → construct THIS Manim class with these kwargs
  (e.g. primitive="Vector", args={"direction": [2,1,0]} → Vector([2,1,0])).
  You may add helpers/labels around it, but do not silently swap the primitive
  for a plainer one (e.g. don't downgrade a planned Vector to a static Line).
- `position` {x, y}   → place the object there: mob.move_to([x, y, 0]).
- `visible` {t_in, t_out} → reveal it around t_in and remove/fade it near
  t_out so the screen stays uncluttered.
- `id` / `role`         → use `id` as the variable name; `role` tells you its
  job (main_equation, shared_axes, label, …).
- `notation_key` / `coordinate_system_id` → REUSE the matching helper instead
  of redrawing; these mark objects shared across scenes.
When an object has no `primitive`, fall back to the type/content and choose the
most visually intuitive Manim primitive yourself.

═══════════════════════  GEOMETRIC ANCHORING (NO MENTAL MATH)  ═══════════════
An endpoint that refers to another object may NEVER be a hardcoded coordinate.
Eyeballed numbers land midway or in the wrong direction. Helpers are imported:
- Arrow/line BETWEEN two objects  ->  connect(obj_a, obj_b)
- A point ON axes                 ->  on_axes(axes, x, y)  (never guess pixels)
- Distance point-to-line/plane    ->  drop_perpendicular(dot, line) — computes
  the TRUE projected foot and adds the right-angle tick. NEVER draw this with
  a hand-placed DashedLine.
- The projection point itself     ->  perpendicular_foot(p, line_start, line_end)
- Anchoring to a part of an object -> obj.get_center() / get_top() / get_corner(UR)
Hardcoded coordinates are ONLY acceptable for free-standing placement of a
single object (its move_to position from the plan).

═══════════════════════  SCENE ENDING (CONTINUITY)  ═══════════════════════
`carry_out` in the scene data lists object ids that MUST still be visible when
this scene ends — the NEXT scene begins from exactly that arrangement:
- NEVER FadeOut / Uncreate / clear a carry_out object at the end of construct().
- NEVER use FadeOut(*self.mobjects) or self.clear() when carry_out is non-empty.
- Objects NOT in carry_out should be faded out before the scene ends so the
  handoff is clean.
- Leave each carry_out object at its planned position in the final frame.

═══════════════════════  CONTENT FIDELITY  ═══════════════════════
The narration is extracted from an academic PDF.  Your animation MUST
show every formula, definition, and example the narration mentions:
- If the narration says "sin θ = opposite / hypotenuse", you MUST show
  that equation on screen using MathTex.
- If the narration describes a right triangle with specific sides,
  you MUST draw that triangle with labeled sides.
- If the narration mentions the unit circle, you MUST draw it.
Do NOT omit PDF content.  Do NOT replace it with unrelated content.

═══════════════════════  OUTPUT RULES  ═══════════════════════
- Return ONLY the class definition.  No module docstrings, no imports,
  no markdown fences, no commentary outside the class body.
- Every self.play() call MUST have an explicit run_time parameter.
- Do NOT use deprecated Manim APIs (e.g. ShowCreation → Create).
"""
    )


class AnimationGeneratorInput(AgentInput):
    """Input for Animation Code Generator Agent.
    
    This is an alias for SceneProcessingInput for clarity.
    """
    
    def __init__(
        self,
        scene_processing_input: SceneProcessingInput,
        review_feedback: Optional[List[str]] = None,
    ):
        self.scene_processing_input = scene_processing_input
        self.review_feedback = review_feedback or []


class AnimationGeneratorOutput(AgentOutput):
    """Output from Animation Code Generator Agent.
    
    Attributes:
        animation_code: Generated AnimationCodeObject
    """
    
    def __init__(self, animation_code: AnimationCodeObject):
        self.animation_code = animation_code


class AnimationGeneratorError(AgentOutput):
    """Error output when code generation fails.
    
    Attributes:
        scene_id: Scene identifier
        error: Error message
    """
    
    def __init__(self, scene_id: str, error: str):
        self.scene_id = scene_id
        self.error = error


class AnimationCodeGeneratorAgent(Agent):
    """Agent responsible for generating executable Manim animation code.
    
    The Animation Code Generator acts as a scene compiler:
    - It receives an engine-aware storyboard contract
    - Maps canonical object families onto concrete Manim primitives
    - Generates Scene.construct() code with explicit Mobject lifecycle
    - Validates generated code syntax before persistence
    
    Note: This is a mock implementation. In production, this would use
    an LLM (GPT-4, Claude 3.5 Sonnet) with temperature=0 for code generation.
    """
    
    # Visual-first pacing floor. This is a VISUALIZATION tool, not a narrated
    # lecture: scene length is driven by how much visual work is on screen,
    # not by how many words of narration exist. We only guarantee a minimum
    # so a scene doesn't flash by — there is deliberately no narration-derived
    # maximum that we pad up to (that produced long, static "dead-air" videos).
    MIN_SCENE_SECONDS = 8.0

    def __init__(self, max_retries: int = 3, strict: bool = False):
        super().__init__()
        self.max_retries = max_retries
        self.strict = strict

        # Initialize Anthropic Client
        # This looks for os.environ["ANTHROPIC_API_KEY"]
        if os.getenv("ANTHROPIC_API_KEY"):
            self.client = anthropic.Anthropic()
        else:
            if self.strict:
                raise AgentExecutionError(
                    "MISSING_API_KEY",
                    "ANTHROPIC_API_KEY is required in --strict mode for AnimationCodeGenerator"
                )
            self.client = None
    
    def execute(self, input_data: AnimationGeneratorInput) -> AgentOutput:
        """Execute animation code generation.
        
        Args:
            input_data: AnimationGeneratorInput with scene processing input
            
        Returns:
            AnimationGeneratorOutput with generated code OR AnimationGeneratorError
            
        Raises:
            AgentExecutionError: For unrecoverable failures
        """
        if not self.validate_input(input_data):
            raise AgentExecutionError(
                "INVALID_INPUT",
                "Input must be AnimationGeneratorInput",
                {"input_type": type(input_data).__name__}
            )
        
        scene_input = input_data.scene_processing_input
        scene_spec = scene_input.scene_spec
        context = scene_input.context
        review_feedback = input_data.review_feedback or []

        try:
            # Generate code with retry logic
            for attempt in range(self.max_retries):
                try:
                    code = self._generate_code(
                        scene_spec, context,
                        review_feedback=review_feedback,
                    )
                    
                    # Validate syntax
                    self._validate_syntax(code)
                    
                    # Format code
                    formatted_code = self._format_code(code, scene_spec, context)
                    
                    # Create animation code object
                    animation_code = AnimationCodeObject(
                        scene_id=scene_spec.scene_id,
                        code=formatted_code,
                        imports=self._get_imports(),
                        class_name=self._get_class_name(scene_spec.scene_id),
                        narration=scene_spec.narration,
                        duration_estimate=scene_spec.duration_estimate,
                        generation_timestamp=datetime.now()
                    )
                    
                    return AnimationGeneratorOutput(animation_code=animation_code)
                    
                except SyntaxError as e:
                    if attempt < self.max_retries - 1:
                        # Retry with syntax error feedback
                        continue
                    else:
                        raise
        
        except Exception as e:
            return AnimationGeneratorError(
                scene_id=scene_spec.scene_id,
                error=str(e)
            )
    
    def validate_input(self, input_data: AgentInput) -> bool:
        """Validate input is AnimationGeneratorInput.
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(input_data, AnimationGeneratorInput):
            return False
        if not isinstance(input_data.scene_processing_input, SceneProcessingInput):
            return False
        return True
    
    def get_retry_policy(self) -> RetryPolicy:
        """Return retry policy for Animation Code Generator.
        
        Returns:
            RetryPolicy with exponential backoff for transient failures
        """
        return RetryPolicy(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay_seconds=1.0,
            max_delay_seconds=30.0,
            retryable_errors=["SYNTAX_ERROR", "API_TIMEOUT", "TRANSIENT_FAILURE"]
        )
    
    def _generate_code(
        self,
        scene_spec: Any,
        context: SharedContext,
        review_feedback: Optional[List[str]] = None,
    ) -> str:
        """Generate Manim code using Anthropic Claude, visual-first (ADR-005).

        The structured visual plan (primitive/args, (x,y) position, visibility
        window, continuity keys) is forwarded verbatim to the LLM, which owns
        the motion design. Narration length is a pacing *hint* only; the scene
        runs as long as its visual work needs, floored at MIN_SCENE_SECONDS —
        never padded up to narration length.
        """

        # Safety: Fallback if no key (in strict mode, constructor already raised)
        if not os.getenv("ANTHROPIC_API_KEY"):
            if self.strict:
                raise AgentExecutionError(
                    "MISSING_API_KEY",
                    "ANTHROPIC_API_KEY is required in --strict mode"
                )
            return self._mock_generate_code(scene_spec, context)

        class_name = self._get_class_name(scene_spec.scene_id)

        # ── Compute narration-driven duration ────────────────────────
        narration = scene_spec.narration or ""
        word_count = len(narration.split())
        # Reference pacing only — a rough guide for the LLM, NOT a quota we pad
        # up to. The visuals drive the real length; narration is fit to them
        # downstream by the composer.
        narration_seconds = round(word_count / 3.0, 1)  # ~3 words/sec TTS
        min_seconds = self.MIN_SCENE_SECONDS
        target_duration = round(max(min_seconds, narration_seconds), 1)

        # ── Prepare structured scene data for the LLM ────────────────
        # Continuity contract: these object ids must still be on screen when
        # the scene ends — the NEXT scene starts from them.
        carry_out = list(
            getattr(scene_spec.visual_intent.scene_state_out, "active_object_ids", [])
            or []
        )

        scene_data = {
            "narration": narration,
            "target_duration_seconds": target_duration,
            "word_count": word_count,
            "carry_out": carry_out,
            "objects": [
                self._object_plan_for_prompt(obj)
                for obj in scene_spec.visual_intent.mathematical_objects
            ],
            "transformations": [
                {
                    "type": t.transformation_type,
                    "target": t.target_object,
                    "timing": t.timing,
                    "beat_index": t.parameters.get("beat_index", 0)
                        if hasattr(t, "parameters") and t.parameters else 0,
                }
                for t in scene_spec.visual_intent.transformations
            ],
        }

        # ── Narrative briefing (Task 8) + opening-state instructions (Task 9) ──
        # Built from context.narrative so the LLM knows what the viewer has
        # already seen and which objects are still on screen from the prior
        # chunk. Empty narrative => empty blocks => no token waste.
        briefing_block, opening_state_block = self._build_narrative_blocks(context)

        # Variable -> color bindings carried across scenes via SharedContext.
        # The ColorManager renders them as a tex_to_color_map hint so the LLM
        # uses the SAME color for the same variable in every scene of the
        # video (the 3B1B "one variable, one color" rule).
        color_block = self._build_color_block(context)

        # ── SYSTEM PROMPT: cached static contract + per-scene dynamic tail ──
        static_system = _codegen_static_system_prompt()
        dynamic_system = f"""CLASS NAME: {class_name}  (must inherit from Scene)
{briefing_block}{opening_state_block}
═══════════════════════  PACING (VISUAL-FIRST)  ═══════════════════════
This is a VISUALIZATION, not a narrated lecture. Let the VISUALS decide the
length. The narration is ~{word_count} words (~{narration_seconds}s of audio),
which is a rough guide only. Aim for a tight, visually dense scene of about
{target_duration}s.
- Fill time with MOTION, TRANSFORMATION and progressive reveal — never with
  empty waits.
- Do NOT pad to hit a number. A shorter scene full of animation is better than
  a long scene sitting on a static frame.
- Minimum scene length: {min_seconds}s. Every second on screen must be doing
  visual work.
- Use self.wait() only to let a specific reveal land (typically 0.5-2s), not to
  burn time.

At the END of construct(), add a comment:
  # Total animation time: Xs
with your manually-computed running total.
{color_block}"""
        user_prompt = f"""Generate the full class for this scene:

{json.dumps(scene_data, indent=2)}
"""
        if review_feedback:
            feedback_block = "\n".join(f"- {item}" for item in review_feedback[:8])
            user_prompt += (
                "\nREVISION FEEDBACK TO FIX IN THIS REWRITE:\n"
                f"{feedback_block}\n"
                "Address the issues above while preserving the narration content and mathematical intent.\n"
            )

        try:
            response = self.client.messages.create(
                model=CODE_GEN_MODEL,
                max_tokens=12000,
                # The static contract is byte-identical for every scene of a
                # run — cache_control serves it from Anthropic's prompt cache
                # (~0.1x input price) on scenes 2..N.
                system=[
                    {
                        "type": "text",
                        "text": static_system,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": dynamic_system},
                ],
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                # Sonnet 5 rejects non-default sampling params (temperature/top_p/
                # top_k → 400) and runs adaptive thinking by default. Disable
                # thinking to keep the prior deterministic no-thinking behaviour
                # and leave the full max_tokens budget for code (the new tokenizer
                # emits ~30% more tokens, so the budget is raised from 8192).
                thinking={"type": "disabled"},
            )

            raw_code = response.content[0].text
            cleaned = self._clean_llm_code(raw_code, class_name)
            # Duration sync (P1.4): grow the video to the narration/audio
            # length (floored at MIN_SCENE_SECONDS) by distributing the slack
            # across existing waits, so the composer never has to freeze the
            # tail. target_duration already = max(min_seconds, narration_seconds).
            return self._distribute_duration_slack(cleaned, target_duration)

        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"LLM generation failed: {e}")
            return self._mock_generate_code(scene_spec, context)

    def _object_plan_for_prompt(self, obj: Any) -> Dict[str, Any]:
        """Serialize ONE planned math object for the code LLM.

        The upstream visual planner already chose a concrete Manim primitive,
        its kwargs, an on-screen position (bbox centre) and a visibility
        window. Earlier this method dropped all of that and passed only
        type/content/style, forcing the code LLM to re-invent the visual from
        scratch — the "plan then discard" gap. We now forward the structured
        plan (omitting empty fields to keep the prompt tight) so the render
        honours the plan instead of diverging from it.
        """
        plan: Dict[str, Any] = {
            "type": obj.object_type,
            "content": obj.content,
        }
        # Identity / role — lets the LLM reuse a stable variable name and know
        # what job the object does in the scene.
        if getattr(obj, "object_id", None):
            plan["id"] = obj.object_id
        if getattr(obj, "object_role", None):
            plan["role"] = obj.object_role
        # Concrete primitive + kwargs chosen by the planner.
        if getattr(obj, "primitive", None):
            plan["primitive"] = obj.primitive
            if getattr(obj, "args", None):
                plan["args"] = obj.args
        render_hint = getattr(obj, "render_hint", None)
        if render_hint:
            plan["render_hint"] = render_hint
        # Continuity keys so shared notation / axes are reused, not redrawn.
        if getattr(obj, "notation_key", None):
            plan["notation_key"] = obj.notation_key
        if getattr(obj, "coordinate_system_id", None):
            plan["coordinate_system_id"] = obj.coordinate_system_id
        if getattr(obj, "character_role", None):
            plan["character_role"] = obj.character_role
        # (x, y, t) placement — where and when the object should be on screen.
        bbox = getattr(obj, "bbox", None)
        if bbox is not None:
            b = bbox.model_dump() if hasattr(bbox, "model_dump") else bbox
            plan["position"] = {"x": round(b["x"], 3), "y": round(b["y"], 3)}
        interval = getattr(obj, "interval", None)
        if interval is not None:
            iv = interval.model_dump() if hasattr(interval, "model_dump") else interval
            plan["visible"] = {"t_in": iv["t_in"], "t_out": iv["t_out"]}
        style = obj.style.model_dump() if hasattr(obj.style, "model_dump") else obj.style
        if style:
            plan["style"] = style
        return plan

    def _build_narrative_blocks(self, context: SharedContext) -> tuple[str, str]:
        """Build the NARRATIVE BRIEFING (Task 8) and OPENING STATE (Task 9)
        prompt sections from ``context.narrative``.

        Returns a (briefing, opening_state) pair. Either may be empty when
        the corresponding narrative field is empty so the prompt does not
        carry dead headers for the first scene of a video.
        """
        nc = getattr(context, "narrative", None)
        if nc is None:
            return "", ""

        briefing_lines: List[str] = []
        if nc.concepts_introduced:
            briefing_lines.append(
                "Concepts the viewer has ALREADY been taught (REFERENCE — do not "
                "re-explain):\n  - "
                + "\n  - ".join(nc.concepts_introduced)
            )
        if nc.live_objects_on_screen:
            briefing_lines.append(
                "Objects on screen at the END of the previous scene (may carry "
                "forward visually):\n  - "
                + "\n  - ".join(f"{k}: {v}" for k, v in nc.live_objects_on_screen.items())
            )
        if nc.running_storyline:
            briefing_lines.append(f"Running storyline: {nc.running_storyline}")
        if nc.prior_scene_summaries:
            recent = nc.prior_scene_summaries[-3:]
            briefing_lines.append(
                "Most recent prior scenes:\n  - " + "\n  - ".join(recent)
            )

        briefing = ""
        if briefing_lines:
            briefing = (
                "\n═══════════════════════  NARRATIVE BRIEFING  ═══════════════════════\n"
                + "\n\n".join(briefing_lines)
                + "\n\nTHIS SCENE IS PART OF A LONGER VIDEO. The viewer has seen "
                "everything\nabove. Build on it — do not redefine concepts already "
                "introduced.\n"
            )

        opening_state = ""
        if nc.live_objects_on_screen:
            live_lines = "\n".join(
                f"  - {k}  ({v})" for k, v in nc.live_objects_on_screen.items()
            )
            opening_state = (
                "\n═══════════════════════  OPENING STATE  ═══════════════════════\n"
                "The previous scene ended with these objects still on screen. To "
                "make the cut\nseamless, your construct() MUST begin by INSTANTLY "
                "re-creating them:\n\n"
                f"{live_lines}\n\n"
                "Use either:\n"
                "    self.add(<mobject>)              # no animation, no run_time\n"
                "or:\n"
                "    self.play(FadeIn(<mobject>), run_time=0)\n\n"
                "Position them where they were at the end of the previous scene "
                "(use sensible\ndefaults if unknown). Only AFTER the carry-over add "
                "do you start the new\nbeats for this scene.\n"
            )

        return briefing, opening_state

    def _build_color_block(self, context: SharedContext) -> str:
        """Render ``context.color_bindings`` as a `tex_to_color_map` hint.

        Empty bindings => empty string so the prompt does not carry a dead
        header for the first scene of a video.
        """
        bindings = getattr(context, "color_bindings", None) or {}
        if not bindings:
            return ""

        from src.utils.color_manager import ColorManager

        cm = ColorManager.from_context_bindings(bindings)
        tex_map = cm.build_tex_color_map()
        if not tex_map:
            return ""

        lines = "\n".join(f"  {var!r}: {color!r}" for var, color in tex_map.items())
        return (
            "\n═══════════════════════  COLOR CONTINUITY  ═══════════════════════\n"
            "These variables already have a color from earlier scenes — REUSE the\n"
            "same color so the viewer's eye keeps tracking them across the video:\n\n"
            f"  tex_to_color_map = {{\n{lines}\n  }}\n\n"
            "Pass this dict into MathTex(..., tex_to_color_map=tex_to_color_map)\n"
            "for any equation that mentions one of these variables.\n"
        )

    MAX_SINGLE_WAIT = 4.0  # cap per pause so no single wait becomes dead air

    def _distribute_duration_slack(self, code: str, target_seconds: float) -> str:
        """Grow a short scene to ``target_seconds`` by DISTRIBUTING the slack
        across its existing ``self.wait()`` calls — not by freezing the tail.

        Duration sync by construction (P1.4): the composed scene runs
        ``max(video, narration_audio)`` seconds. When the video is shorter, the
        composer holds the final frame for the remainder — a "static scene"
        (58 sync findings + much of the static-frame population in a real run).
        Here ``target_seconds`` is the narration/audio length; we spread the
        missing time across the pauses already in the code so those seconds
        read as breathing room between reveals. Each wait is capped at
        ``MAX_SINGLE_WAIT`` so no single pause dominates; any residual goes to
        one bounded tail wait. A scene already at/over target is left untouched.
        """
        import re as _re3

        run_time_total = sum(
            float(m.group(1)) for m in _re3.finditer(r'run_time\s*=\s*([0-9.]+)', code)
        )
        wait_matches = list(_re3.finditer(r'self\.wait\s*\(\s*([0-9.]+)\s*\)', code))
        total = run_time_total + sum(float(m.group(1)) for m in wait_matches)

        gap = round(target_seconds - total, 1)
        if gap <= 0.5:
            return code  # already fills the audio — no freeze-frame to avoid

        if wait_matches:
            # Absorb the gap into existing waits up to the per-wait cap,
            # proportional to each wait's remaining headroom.
            headrooms = [
                max(0.0, self.MAX_SINGLE_WAIT - float(m.group(1))) for m in wait_matches
            ]
            capacity = sum(headrooms)
            absorbed = min(gap, capacity)
            adds = [
                (absorbed * (hr / capacity)) if capacity > 0 else 0.0
                for hr in headrooms
            ]
            new_code = code
            for m, add in sorted(
                zip(wait_matches, adds), key=lambda t: t[0].start(), reverse=True
            ):
                new_val = round(float(m.group(1)) + add, 1)
                new_code = new_code[:m.start()] + f"self.wait({new_val})" + new_code[m.end():]
            remaining = round(gap - absorbed, 1)
            if remaining > 0.5:
                lines = new_code.rstrip().split("\n")
                lines.append(f"        self.wait({remaining})  # duration sync tail")
                new_code = "\n".join(lines) + "\n"
            return new_code

        # No existing waits — a single bounded tail wait is the only lever.
        lines = code.rstrip().split("\n")
        lines.append(f"        self.wait({gap})  # duration sync")
        return "\n".join(lines) + "\n"

    def _clean_llm_code(self, raw_code: str, class_name: str) -> str:
        """Strip markdown, duplicate headers, and ensure class structure.

        Handles three LLM output shapes:
          A) Full class with construct  → return as-is
          B) Just ``def construct(self):`` + body → wrap in class
          C) Bare code (no class, no def) → wrap in class + def
        """
        # Remove markdown fences if present
        clean = raw_code.replace("```python", "").replace("```", "").strip()

        # Remove module-level docstrings the LLM may have added
        # (these will be re-added by _format_code)
        import re as _re2
        clean = _re2.sub(r'^"""[\s\S]*?"""\s*', '', clean).strip()
        clean = _re2.sub(r"^'''[\s\S]*?'''\s*", '', clean).strip()

        # Remove ALL import lines — imports are added by _format_code so we
        # don't want LLM-generated duplicates.
        lines = clean.split("\n")
        lines = [l for l in lines
                 if not l.strip().startswith("from manim import")
                 and not l.strip().startswith("import manim")
                 and not l.strip().startswith("from src.utils.")]
        clean = "\n".join(lines).strip()

        # ── Case A: full class already present ────────────────────
        if f"class {class_name}" in clean:
            return clean

        # Check if ANY Scene subclass is present (LLM may have used a
        # different name).  If so, rename it.
        import re as _re
        class_match = _re.search(
            r"class\s+(\w+)\s*\(\s*(?:Scene|ThreeDScene|MovingCameraScene)\s*\)", clean
        )
        if class_match:
            old_name = class_match.group(1)
            return clean.replace(f"class {old_name}", f"class {class_name}", 1)

        # ── Case B: starts with ``def construct(self):`` ──────────
        has_def = _re.match(r"^\s*def\s+construct\s*\(\s*self\s*\)", clean)
        if has_def:
            # Ensure body is indented 4 spaces under the class
            indented = "\n".join("    " + line for line in clean.split("\n"))
            return f"class {class_name}(Scene):\n{indented}"

        # ── Case C: bare code body ────────────────────────────────
        # Indent to 8 spaces (inside class > def construct)
        indented_body = "\n".join(
            "        " + line if line.strip() else ""
            for line in clean.split("\n")
        )
        return (
            f"class {class_name}(Scene):\n"
            f"    def construct(self):\n"
            f"{indented_body}"
        )
    
    def _generate_mathematical_objects(
        self,
        math_objects: List[Dict[str, Any]],
        context: SharedContext
    ) -> str:
        """Generate code for mathematical objects.
        
        Args:
            math_objects: List of mathematical object specifications
            context: Shared context with visual style
            
        Returns:
            Generated code string
        """
        if not math_objects:
            return "        # No mathematical objects"
        
        code_lines = []
        colors = context.visual_style.colors
        
        for i, obj in enumerate(math_objects):
            if hasattr(obj, "model_dump"):
                obj_data = obj.model_dump()
            elif isinstance(obj, dict):
                obj_data = obj
            else:
                obj_data = {
                    "object_type": getattr(obj, "object_type", "text"),
                    "content": getattr(obj, "content", ""),
                    "render_hint": getattr(obj, "render_hint", None),
                    "style": getattr(obj, "style", {}),
                }

            obj_type = obj_data.get("object_type", "text")
            content = obj_data.get("content", "")
            render_hint = (obj_data.get("render_hint") or "").strip().lower()
            style = obj_data.get("style", {})
            if hasattr(style, "model_dump"):
                style = style.model_dump()
            color = style.get("color") or colors.get("primary") or "BLUE"
            
            var_name = self._object_var_name(obj_data, i)
            notation_key = obj_data.get("notation_key")
            notation_key_repr = repr(notation_key) if notation_key else "None"

            # Catalog path (Task 12) — when the visual planner emitted a
            # closed-vocabulary primitive, the animation generator becomes a
            # thin emitter and the legacy if/elif chain is bypassed entirely.
            if obj_data.get("primitive"):
                code_lines.append(self._emit_primitive(var_name, obj_data))
                bbox_line = self._emit_bbox_placement(var_name, obj_data)
                if bbox_line:
                    code_lines.append(bbox_line)
                continue

            # ─── LEGACY PATH (deprecated) ─────────────────────────────────
            # No `primitive` was set on this object. As of the closed-catalog
            # ADR (003), every code path that produces a MathematicalObject
            # — real LLM, mock fallback, hand-built test fixture — should
            # emit `primitive` + `args`. Hitting this branch means a producer
            # was missed. Logged so CI / dev runs surface the regression
            # before it reaches production. The branch itself is retained
            # as a safety net until catalog adoption is proven on real PDFs
            # (target: delete after 2-3 successful real-API runs).
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Animation generator falling back to legacy if/elif chain for "
                "object %s (object_type=%s). Producer should set "
                "MathematicalObject.primitive — see ADR-003.",
                var_name, obj_type,
            )

            if obj_type == "equation":
                # Use create_notation so the object carries a stable notation_key
                latex = self._escape_latex(content)
                code_lines.append(
                    f'        {var_name} = create_notation('
                    f'tex=r"{latex}", notation_key={notation_key_repr}, color="{color.upper()}")'
                )

            elif obj_type == "text":
                font_size = context.visual_style.font_size
                code_lines.append(
                    f'        {var_name} = create_text_label({content!r}, '
                    f'color="{color.upper()}", font_size={font_size})'
                )

            elif obj_type in {"axes", "graph"}:
                # Derive helper kind from coordinate_system_id or render_hint
                cs_id = obj_data.get("coordinate_system_id") or ""
                if cs_id:
                    helper_kind = cs_id
                elif render_hint in {"numberplane", "number_plane"}:
                    helper_kind = "number_plane"
                elif render_hint in {"complexplane", "complex_plane"}:
                    helper_kind = "complex_plane"
                else:
                    helper_kind = "axes"
                code_lines.append(
                    f'        {var_name} = create_coordinate_system('
                    f'kind="{helper_kind}", x_length=5.0, y_length=3.0)'
                )
                if obj_type == "graph":
                    code_lines.append(
                        f'        graph_{i} = plot_function_on_system('
                        f'{var_name}, lambda x: x**2, color="{color.upper()}")'
                    )

            elif obj_type == "geometric_shape":
                if render_hint == "arrow":
                    code_lines.append(f'        {var_name} = Arrow(LEFT, RIGHT, color={color.upper()})')
                elif render_hint == "line":
                    code_lines.append(f'        {var_name} = Line(LEFT, RIGHT, color={color.upper()})')
                elif render_hint == "arc":
                    code_lines.append(f'        {var_name} = Arc(radius=1.0, angle=PI / 2, color={color.upper()})')
                elif render_hint == "polygon" or "triangle" in content.lower():
                    code_lines.append(f'        {var_name} = Polygon(LEFT, RIGHT, UP, color={color.upper()})')
                elif "circle" in content.lower() or render_hint == "circle":
                    code_lines.append(f'        {var_name} = Circle(color={color.upper()})')
                elif "square" in content.lower() or "rectangle" in content.lower():
                    code_lines.append(f'        {var_name} = Square(color={color.upper()})')
                elif "line" in content.lower():
                    code_lines.append(f'        {var_name} = Line(color={color.upper()})')
                else:
                    code_lines.append(f'        {var_name} = Circle(color={color.upper()})')

            elif obj_type == "pi_creature":
                # Use role-aware helper; fall back to teacher if no role given
                character_role = obj_data.get("character_role") or "teacher"
                code_lines.append(
                    f'        {var_name} = create_character_for_role("{character_role}")'
                )
        
        return "\n".join(code_lines)
    
    def _generate_transformations(
        self,
        transformations: List[Dict[str, Any]],
        context: SharedContext
    ) -> str:
        """Generate code for transformations.
        
        Args:
            transformations: List of transformation specifications
            context: Shared context with animation speed
            
        Returns:
            Generated code string
        """
        if not transformations:
            return "        # No transformations"
        
        code_lines = []
        animation_speed = context.visual_style.animation_speed
        
        for trans in transformations:
            if hasattr(trans, "model_dump"):
                trans_data = trans.model_dump()
            elif isinstance(trans, dict):
                trans_data = trans
            else:
                trans_data = {
                    "transformation_type": getattr(trans, "transformation_type", "fade_in"),
                    "target_object": getattr(trans, "target_object", "obj_0"),
                    "timing": getattr(trans, "timing", 0.0),
                    "parameters": getattr(trans, "parameters", {}) or {},
                }

            trans_type = trans_data.get("transformation_type", "fade_in")
            target = trans_data.get("target_object", "obj_0")
            timing = trans_data.get("timing", 0.0)
            
            # Map target_object reference to variable name
            # Simplified: assume target_object is like "equation_0" -> "obj_0"
            if "_" in target:
                parts = target.split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    var_name = f"obj_{parts[1]}"
                else:
                    var_name = "obj_0"
            else:
                var_name = "obj_0"
            
            # Generate transformation code
            if trans_type == "fade_in":
                code_lines.append(f'        self.play(FadeIn({var_name}), run_time={1.0/animation_speed})')
            
            elif trans_type == "fade_out":
                code_lines.append(f'        self.play(FadeOut({var_name}), run_time={1.0/animation_speed})')
            
            elif trans_type == "write":
                code_lines.append(f'        self.play(Write({var_name}), run_time={2.0/animation_speed})')
            
            elif trans_type == "morph":
                # Simplified morph (would need target object)
                code_lines.append(f'        self.play(Transform({var_name}, {var_name}.copy()), run_time={1.5/animation_speed})')
            
            elif trans_type == "highlight":
                code_lines.append(f'        self.play(Indicate({var_name}), run_time={0.5/animation_speed})')
            
            elif trans_type == "move":
                code_lines.append(f'        self.play({var_name}.animate.shift(UP), run_time={1.0/animation_speed})')
            
            elif trans_type == "scale":
                params = trans_data.get("parameters", {})
                scale_factor = params.get("scale_factor", 1.5)
                code_lines.append(f'        self.play({var_name}.animate.scale({scale_factor}), run_time={1.0/animation_speed})')
            
            # Add wait for timing
            if timing > 0:
                code_lines.append(f'        self.wait({timing})')
        
        return "\n".join(code_lines)
    
    def _validate_syntax(self, code: str) -> None:
        """Validate Python syntax using AST parser.
        
        Args:
            code: Python code string to validate
            
        Raises:
            SyntaxError: If code has syntax errors
        """
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise SyntaxError(f"Generated code has syntax error: {e}")
    
    def _format_code(
        self,
        code: str,
        scene_spec: Any,
        context: SharedContext
    ) -> str:
        """Format code with header comment and consistent style.
        
        Args:
            code: Generated code string
            scene_spec: Scene specification
            context: Shared context
            
        Returns:
            Formatted code string
        """
        # Add header comment
        header = f'''"""
Scene: {scene_spec.scene_id}
Narration: {scene_spec.narration[:100]}{"..." if len(scene_spec.narration) > 100 else ""}
Duration: {scene_spec.duration_estimate}s
Generated: {datetime.now().isoformat()}
"""

'''
        
        # Add imports
        imports = "\n".join(self._get_imports())
        
        # Combine
        formatted = f"{header}{imports}\n\n{code}"
        
        return formatted
    
    def _get_imports(self) -> List[str]:
        """Get required import statements.

        Returns:
            List of import statements
        """
        return [
            "from manim import *",
            "from src.utils.coordinate_system_helpers import attach_axis_labels, create_coordinate_system, plot_function_on_system",
            "from src.utils.notation_helpers import create_notation, create_notation_transform_target, create_text_label",
            "from src.utils.pi_characters import create_character_for_role, create_character_lineup, create_pi_teacher, create_pi_students, create_speech_bubble",
            "from src.utils.animation_patterns import progressive_write, dim_and_focus, staged_transform, execute_animation_steps",
            "from src.utils.layout_tracker import LayoutTracker",
            "from src.utils.geometry_helpers import connect, drop_perpendicular, on_axes, perpendicular_foot",
        ]

    def _emit_primitive(self, var_name: str, obj_data: dict) -> str:
        """Emit one line of code from a closed-vocabulary primitive (Task 12).

        ``obj_data["primitive"]`` selects the catalog entry and ``args``
        carries its kwargs. Color from ``style.color`` is appended when the
        spec accepts it. The Pydantic model_validator on ``MathematicalObject``
        already rejected unknown primitives / args, so this method trusts
        its input.
        """
        from src.utils.manim_primitives import MOBJECT_CATALOG

        primitive = obj_data["primitive"]
        spec = MOBJECT_CATALOG.get(primitive)
        args = dict(obj_data.get("args", {}))
        style = obj_data.get("style") or {}
        if hasattr(style, "model_dump"):
            style = style.model_dump()
        color = style.get("color")

        parts = [f"{k}={self._py_repr(v)}" for k, v in args.items()]
        if spec is not None and spec.accepts_color and color and "color" not in args:
            parts.append(f"color={color.upper()}")
        return f"        {var_name} = {primitive}({', '.join(parts)})"

    def _emit_bbox_placement(self, var_name: str, obj_data: dict) -> str:
        """Emit ``move_to`` (and a width fit when sensible) for a primitive
        that ships with a bbox (Task 13). Returns "" when no bbox is set
        so the caller can simply append non-empty results.
        """
        bbox = obj_data.get("bbox")
        if not bbox:
            return ""
        if hasattr(bbox, "model_dump"):
            bbox = bbox.model_dump()
        x = bbox.get("x")
        y = bbox.get("y")
        if x is None or y is None:
            return ""
        return f"        {var_name}.move_to([{x}, {y}, 0])"

    @staticmethod
    def _py_repr(value: Any) -> str:
        """Compact Python literal for catalog kwargs (lists/tuples/strings)."""
        if isinstance(value, str):
            # Tex/Text content goes in raw strings so backslashes survive.
            return "r" + repr(value)
        return repr(value)

    def _object_var_name(self, obj_data: dict, index: int) -> str:
        """Derive a stable Python variable name from an object spec.

        Prefers ``object_id`` when present so the name is stable across retries
        and matches the storyboard's continuity identifiers.

        Args:
            obj_data: Object specification dict.
            index: Fallback positional index.

        Returns:
            Valid Python identifier string.
        """
        raw = obj_data.get("object_id") or f"obj_{index}"
        return re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    
    def _get_class_name(self, scene_id: str) -> str:
        """Generate valid Python class name from scene_id.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Valid Python class name
        """
        # Replace non-alphanumeric characters with underscores
        class_name = re.sub(r'[^a-zA-Z0-9_]', '_', scene_id)
        
        # Ensure it starts with a letter or underscore
        if class_name and class_name[0].isdigit():
            class_name = f"Scene_{class_name}"
        elif not class_name:
            class_name = "Scene_default"
        
        # Capitalize first letter
        class_name = class_name[0].upper() + class_name[1:] if len(class_name) > 1 else class_name.upper()
        
        return class_name
    
    def _escape_latex(self, latex: str) -> str:
        """Escape LaTeX string for Python raw string literal.
        
        Args:
            latex: LaTeX string
            
        Returns:
            Escaped LaTeX string safe for r"..." literals
        """
        # For raw strings (r"..."), we need to:
        # 1. Replace double quotes with single quotes to avoid breaking the literal
        # 2. Handle trailing backslashes (can't end raw string with \)
        result = latex.replace('"', "'")

        # If string ends with odd number of backslashes, add a space
        # to avoid syntax error in raw string
        if result.endswith('\\') and not result.endswith('\\\\'):
            result += ' '

        return result

    def _mock_generate_code(self, scene_spec: Any, context: SharedContext) -> str:
        """Fallback mock generator when API is unavailable."""
        class_name = self._get_class_name(scene_spec.scene_id)
        math_code = self._generate_mathematical_objects(
            scene_spec.visual_intent.mathematical_objects, context
        )
        transform_code = self._generate_transformations(
            scene_spec.visual_intent.transformations, context
        )

        body_blocks = [
            "        # MOCK GENERATION - API KEY NOT FOUND",
            f"        # Narration: {scene_spec.narration}",
        ]
        if math_code and "No mathematical objects" not in math_code:
            body_blocks.append(math_code)
        if transform_code and "No transformations" not in transform_code:
            body_blocks.append(transform_code)
        if len(body_blocks) == 2:
            body_blocks.append("        pass")

        return (
            f"class {class_name}(Scene):\n"
            f"    def construct(self):\n"
            + "\n".join(body_blocks)
            + "\n"
        )