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
        scene_layout = scene_input.scene_layout

        try:
            # Generate code with retry logic
            for attempt in range(self.max_retries):
                try:
                    code = self._generate_code(
                        scene_spec, context,
                        review_feedback=review_feedback,
                        scene_layout=scene_layout,
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
        scene_layout: Optional[Any] = None,
    ) -> str:
        """Generate Manim code using Anthropic Claude, synchronized to narration.

        The narration is the single source of truth for duration.  We compute
        target_duration = word_count / 3.0 (TTS rate) and instruct the LLM to
        produce animations whose total run_time + wait() gaps equal that value.
        When a pre-validated scene_layout is provided, the LLM is instructed to
        place objects at the given coordinates rather than inventing positions.
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
        target_duration = round(word_count / 3.0, 1)  # ~3 words/sec TTS
        if target_duration < 15:
            target_duration = 15.0  # floor

        # ── Prepare structured scene data for the LLM ────────────────
        scene_data = {
            "narration": narration,
            "target_duration_seconds": target_duration,
            "word_count": word_count,
            "objects": [
                {
                    "type": obj.object_type,
                    "content": obj.content,
                    "render_hint": getattr(obj, "render_hint", None),
                    "style": obj.style.model_dump(),
                }
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

        engine_contract = build_animation_codegen_contract_prompt()
        quality_contract = build_3b1b_quality_prompt()

        # ── Narrative briefing (Task 8) + opening-state instructions (Task 9) ──
        # Built from context.narrative so the LLM knows what the viewer has
        # already seen and which objects are still on screen from the prior
        # chunk. Empty narrative => empty blocks => no token waste.
        briefing_block, opening_state_block = self._build_narrative_blocks(context)

        # ── THE NARRATION-AWARE 3B1B SYSTEM PROMPT ───────────────────
        system_prompt = f"""You are an expert Manim Community v0.19 developer who creates
3Blue1Brown-quality educational animations.

CLASS NAME: {class_name}  (must inherit from Scene)

{engine_contract}

{quality_contract}
{briefing_block}{opening_state_block}
═══════════════════════  DURATION  ═══════════════════════
The narration is {word_count} words.  TTS will produce ~{target_duration}s of
audio.  Your total animation time (sum of all run_time + self.wait) MUST equal
{target_duration}s (±2 s).  Add self.wait() calls between animations so the
viewer can absorb what's on screen while the narrator speaks.

At the END of construct(), add a comment:
  # Total animation time: Xs  (target: {target_duration}s)
with your manually-computed running total so you can self-check.

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

═══════════════════════  PI CHARACTERS  ═══════════════════════
If and only if the storyboard intent includes a `pi_creature` object or a
PiCreature* render_hint, use the pre-built Pi character helpers (already
imported for you):

    teacher = create_pi_teacher().shift(LEFT * 3)
    students = create_pi_students().shift(RIGHT * 3)
    bubble = create_speech_bubble("Key insight text", teacher)

    self.play(FadeIn(teacher), FadeIn(students), run_time=1.0)
    self.play(Write(bubble), run_time=1.0)
    self.wait(3)
    self.play(FadeOut(bubble), FadeOut(teacher), FadeOut(students), run_time=1.0)

Use this pattern only when the storyboard explicitly asks for that helper beat.
Do not invent helper objects that are not present in the visual intent.

═══════════════════════  STYLE  ═══════════════════════
- Black background (default).  Use BLUE, YELLOW, GREEN, RED for emphasis.
- Use MathTex(r"...") for ALL equations.  Use Text("...") for plain labels.
- Prefer Write / Create for new objects, ReplacementTransform for morphing.
- Smooth pacing: run_time between 1.0 and 3.0 for most animations.
- After every major visual change, add self.wait(2) to self.wait(5)
  so the narrator can explain.

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
        # ── Inject pre-validated layout constraints ───────────────────
        if scene_layout is not None and getattr(scene_layout, "objects", None):
            layout_lines = [
                "\n═══════════════════════  PRE-VALIDATED LAYOUT (MANDATORY)  ═══════════════════════",
                "A spatial layout has been computed and validated for zero overlap.",
                "You MUST position each object at the coordinates listed below.",
                "Use mob.move_to(np.array([cx, cy, 0])) where cx/cy is the centre",
                "computed from the bounding box corners.  Do NOT deviate from these",
                "positions — the validator has confirmed they do not overlap.\n",
            ]
            for bbox in scene_layout.objects:
                cx = round((bbox.x_min + bbox.x_max) / 2, 3)
                cy = round((bbox.y_min + bbox.y_max) / 2, 3)
                layout_lines.append(
                    f"  object_id={bbox.object_id!r}  type={bbox.object_type}"
                    f"  centre=({cx}, {cy})  size={bbox.width:.2f}×{bbox.height:.2f}"
                    f"  visible_for={bbox.duration:.1f}s"
                )
            system_prompt += "\n".join(layout_lines) + "\n"

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
                model="claude-sonnet-4-6",
                max_tokens=8192,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
            )

            raw_code = response.content[0].text
            cleaned = self._clean_llm_code(raw_code, class_name)
            return self._enforce_duration(cleaned, target_duration)

        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"LLM generation failed: {e}")
            return self._mock_generate_code(scene_spec, context)

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

    def _enforce_duration(self, code: str, target_duration: float) -> str:
        """Parse run_time and self.wait values; distribute padding waits if short.

        Sums every numeric ``run_time=X`` and ``self.wait(X)`` in the code.
        If the total is significantly less than *target_duration*, it distributes
        extra wait time by boosting existing self.wait() calls proportionally
        and adding a small tail wait if needed.
        """
        import re as _re3

        total = 0.0
        for m in _re3.finditer(r'run_time\s*=\s*([0-9.]+)', code):
            total += float(m.group(1))
        for m in _re3.finditer(r'self\.wait\s*\(\s*([0-9.]+)\s*\)', code):
            total += float(m.group(1))

        gap = target_duration - total
        if gap <= 3.0:
            return code  # close enough

        # Strategy: find all existing self.wait(X) calls and scale them up
        # proportionally, then add any remainder as a tail wait.
        wait_calls = list(_re3.finditer(r'self\.wait\s*\(\s*([0-9.]+)\s*\)', code))
        existing_wait_total = sum(float(m.group(1)) for m in wait_calls)

        if existing_wait_total > 0 and wait_calls:
            # Scale factor: how much to multiply each wait by
            scale = (existing_wait_total + gap) / existing_wait_total
            # Cap individual waits at 8s to avoid absurdly long pauses
            max_single_wait = 8.0

            new_code = code
            # Process in reverse so indices stay valid
            for m in reversed(wait_calls):
                old_val = float(m.group(1))
                new_val = min(round(old_val * scale, 1), max_single_wait)
                new_code = (new_code[:m.start()]
                            + f"self.wait({new_val})"
                            + new_code[m.end():])

            # Recompute total after scaling
            new_total = 0.0
            for m2 in _re3.finditer(r'run_time\s*=\s*([0-9.]+)', new_code):
                new_total += float(m2.group(1))
            for m2 in _re3.finditer(r'self\.wait\s*\(\s*([0-9.]+)\s*\)', new_code):
                new_total += float(m2.group(1))

            remaining = target_duration - new_total
            if remaining > 2.0:
                lines = new_code.rstrip().split("\n")
                lines.append(f"        self.wait({round(remaining, 1)})  # duration padding")
                new_code = "\n".join(lines) + "\n"

            return new_code
        else:
            # No existing waits — just add a tail wait
            lines = code.rstrip().split("\n")
            lines.append(f"        self.wait({round(gap, 1)})  # duration padding")
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
            "from src.utils.layout_tracker import LayoutTracker",
        ]

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