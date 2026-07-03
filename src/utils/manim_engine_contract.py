"""Shared Manim engine contract for schema validation and prompt authoring.

This module keeps our storyboard schema, LLM prompts, and code generator
aligned around the same engine-aware vocabulary.

The structure is inspired by the 3Blue1Brown `manimlib` codebase:
- `Scene.construct()` is the unit of execution
- visuals are concrete `Mobject` families
- changes happen through animation primitives plus `self.wait(...)`

Our runtime still targets Manim Community, so these hints describe the engine
shape without importing `manimlib` at runtime.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# 3Blue1Brown Timing Contract
# ---------------------------------------------------------------------------
# Maps animation families to (min_run_time, max_run_time) in seconds.
# These come from studying 3B1B videos: equations appear at ~2s, fades at
# ~0.5-1s, transforms at ~1-2s.  Violations are flagged by the reviewer.

TIMING_CONTRACT: Dict[str, Tuple[float, float]] = {
    # Introductory / creation animations. Minimums were relaxed 1.0 -> 0.6 for
    # vignette pacing (ADR-005): short scenes legitimately write labels in
    # ~0.8s, and the old floor produced dozens of noise findings per run.
    "Write": (0.6, 3.0),
    "Create": (0.6, 2.5),
    "ShowCreation": (0.6, 2.5),
    "DrawBorderThenFill": (0.6, 2.5),
    "GrowFromCenter": (0.5, 1.5),
    "GrowArrow": (0.5, 1.5),
    # Fade family
    "FadeIn": (0.5, 1.5),
    "FadeOut": (0.5, 1.5),
    "FadeTransform": (0.8, 2.0),
    "FadeTransformPieces": (0.8, 2.0),
    # Transform / morph family
    "Transform": (1.0, 2.5),
    "ReplacementTransform": (1.0, 2.5),
    "TransformFromCopy": (1.0, 2.5),
    # Emphasis / highlight family
    "Indicate": (0.5, 1.5),
    "Flash": (0.3, 1.0),
    "Circumscribe": (0.5, 1.5),
    "FocusOn": (0.5, 1.5),
    "Wiggle": (0.5, 1.5),
    # Motion family
    "MoveToTarget": (0.8, 2.0),
    "MoveAlongPath": (1.0, 3.0),
    "Rotate": (0.8, 2.0),
    "ApplyWave": (0.8, 2.0),
}

# Minimum self.wait() durations (seconds)
WAIT_CONTRACT = {
    "between_animations": 0.5,      # Minimum pause between consecutive self.play()
    "after_major_reveal": 1.0,      # After showing a key equation or concept
    "scene_breathing_room": 0.3,    # Minimum wait at scene boundaries
    "after_transform": 0.5,         # After a morph / ReplacementTransform
}


# ---------------------------------------------------------------------------
# 3Blue1Brown Color Palette
# ---------------------------------------------------------------------------
# Derived from 3B1B's manimlib/constants.py and video analysis.
# Names map to hex strings usable in Manim Community's ManimColor.

THREE_BLUE_ONE_BROWN_PALETTE = {
    # Core identity colors
    "BLUE_D": "#1C758A",
    "BLUE_C": "#29ABCA",
    "BLUE_B": "#58C4DD",       # 3B1B signature blue
    "BLUE_A": "#C7E9F1",
    "TEAL_D": "#49A88F",
    "TEAL_C": "#5CD0B3",
    "GREEN_D": "#77B05D",
    "GREEN_C": "#83C167",
    "YELLOW_D": "#E8C11C",
    "YELLOW_C": "#FFFF00",
    "GOLD_D": "#C78D46",
    "GOLD_C": "#E1A158",
    "RED_D": "#CF5044",
    "RED_C": "#FC6255",
    "MAROON_D": "#94424F",
    "MAROON_C": "#A24D61",
    "PURPLE_D": "#644172",
    "PURPLE_C": "#9A72AC",
    # Semantic aliases used in 3B1B videos
    "PRIMARY": "#58C4DD",          # main concept / equation
    "SECONDARY": "#83C167",        # supporting concept
    "ACCENT": "#FFFF00",           # highlight / emphasis
    "WARNING": "#FC6255",          # error / contradiction
    "BACKGROUND": "#1C1C1C",       # 3B1B dark background
    "TEXT_DEFAULT": "#FFFFFF",      # default text color
}

# Ordered list for automatic variable-to-color assignment
COLOR_CYCLE = [
    "#58C4DD",  # BLUE_B
    "#83C167",  # GREEN_C
    "#FFFF00",  # YELLOW_C
    "#FC6255",  # RED_C
    "#9A72AC",  # PURPLE_C
    "#E1A158",  # GOLD_C
    "#5CD0B3",  # TEAL_C
    "#A24D61",  # MAROON_C
    "#29ABCA",  # BLUE_C
    "#C78D46",  # GOLD_D
]


CANONICAL_OBJECT_TYPES = (
    "equation",
    "text",
    "axes",
    "graph",
    "geometric_shape",
    "pi_creature",
)

CANONICAL_COORDINATE_SYSTEM_KINDS = (
    "axes",
    "number_plane",
    "complex_plane",
    "number_line",
    "three_d_axes",
)

PREFERRED_HELPER_PATTERNS = (
    "create_coordinate_system(kind=...) for reusable axes, planes, and number lines",
    "attach_axis_labels(system, x_label=..., y_label=...) for deterministic axis labelling",
    "plot_function_on_system(system, fn, color=...) when graphing on a reusable coordinate frame",
)

CANONICAL_CONTINUITY_ACTIONS = (
    "create",
    "reuse",
    "transform",
    "remove",
    "highlight",
)

CANONICAL_CHARACTER_ROLES = (
    "teacher",
    "student_left",
    "student_center",
    "student_right",
    "narrator_proxy",
)

CANONICAL_CAMERA_ACTIONS = (
    "static",
    "focus",
    "zoom_to_region",
    "pan_to_object",
    "follow_object",
    "restore",
)

OBJECT_TYPE_ALIASES = {
    "mathtex": "equation",
    "tex": "equation",
    "formula": "equation",
    "equations": "equation",
    "label": "text",
    "labels": "text",
    "caption": "text",
    "subtitle": "text",
    "speech_bubble": "text",
    "number_plane": "axes",
    "coordinate_plane": "axes",
    "plane": "axes",
    "coordinate_axes": "axes",
    "plot": "graph",
    "curve": "graph",
    "vector": "geometric_shape",
    "line": "geometric_shape",
    "arrow": "geometric_shape",
    "circle": "geometric_shape",
    "polygon": "geometric_shape",
    "triangle": "geometric_shape",
    "rectangle": "geometric_shape",
    "square": "geometric_shape",
    "arc": "geometric_shape",
    "brace": "geometric_shape",
    "pi_character": "pi_creature",
    "pi_characters": "pi_creature",
    "character": "pi_creature",
}

CANONICAL_RENDER_HINTS = (
    "MathTex",
    "Tex",
    "Text",
    "DecimalNumber",
    "Integer",
    "Axes",
    "ThreeDAxes",
    "NumberPlane",
    "ComplexPlane",
    "NumberLine",
    "Line",
    "DashedLine",
    "Arrow",
    "Vector",
    "DoubleArrow",
    "Polygon",
    "Circle",
    "Ellipse",
    "Arc",
    "Rectangle",
    "RoundedRectangle",
    "Square",
    "Triangle",
    "Dot",
    "Brace",
    "ParametricCurve",
    "Graph",
    "DiGraph",
    "VGroup",
    "Group",
    "PiCreatureTeacher",
    "PiCreatureStudents",
    "SpeechBubble",
)

RENDER_HINT_ALIASES = {
    "mathtex": "MathTex",
    "tex": "Tex",
    "text": "Text",
    "decimalnumber": "DecimalNumber",
    "decimal_number": "DecimalNumber",
    "integer": "Integer",
    "axes": "Axes",
    "threedaxes": "ThreeDAxes",
    "three_d_axes": "ThreeDAxes",
    "numberplane": "NumberPlane",
    "number_plane": "NumberPlane",
    "complexplane": "ComplexPlane",
    "complex_plane": "ComplexPlane",
    "numberline": "NumberLine",
    "number_line": "NumberLine",
    "line": "Line",
    "dashedline": "DashedLine",
    "dashed_line": "DashedLine",
    "arrow": "Arrow",
    "vector": "Vector",
    "doublearrow": "DoubleArrow",
    "double_arrow": "DoubleArrow",
    "polygon": "Polygon",
    "triangle": "Polygon",
    "circle": "Circle",
    "ellipse": "Ellipse",
    "arc": "Arc",
    "rectangle": "Rectangle",
    "roundedrectangle": "RoundedRectangle",
    "rounded_rectangle": "RoundedRectangle",
    "square": "Square",
    "dot": "Dot",
    "brace": "Brace",
    "parametriccurve": "ParametricCurve",
    "parametric_curve": "ParametricCurve",
    "graph": "Graph",
    "digraph": "DiGraph",
    "di_graph": "DiGraph",
    "vgroup": "VGroup",
    "group": "Group",
    "pi_creature_teacher": "PiCreatureTeacher",
    "picreatureteacher": "PiCreatureTeacher",
    "pi_creature_students": "PiCreatureStudents",
    "picreaturestudents": "PiCreatureStudents",
    "speech_bubble": "SpeechBubble",
    "speechbubble": "SpeechBubble",
}

CANONICAL_TRANSFORMATION_TYPES = (
    "fade_in",
    "fade_out",
    "morph",
    "highlight",
    "move",
    "scale",
)

CANONICAL_ANIMATION_HINTS = (
    "Create",
    "Write",
    "ShowCreation",
    "DrawBorderThenFill",
    "GrowFromCenter",
    "GrowArrow",
    "FadeIn",
    "Transform",
    "ReplacementTransform",
    "TransformFromCopy",
    "MoveToTarget",
    "Indicate",
    "Flash",
    "Circumscribe",
    "FocusOn",
    "Wiggle",
    "MoveAlongPath",
    "Rotate",
    "ApplyWave",
    "FadeOut",
    "FadeTransform",
    "FadeTransformPieces",
)


def normalize_object_type(value: str) -> str:
    """Normalize an LLM-provided object type to the canonical schema value."""
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return OBJECT_TYPE_ALIASES.get(normalized, normalized)


def normalize_render_hint(value: Optional[str]) -> Optional[str]:
    """Normalize a render hint to a canonical Manim primitive/helper name."""
    if value is None:
        return None
    normalized = value.strip().replace("-", "_").replace(" ", "_")
    if not normalized:
        return None
    lookup_key = normalized.lower()
    return RENDER_HINT_ALIASES.get(lookup_key, normalized)


def build_visual_intent_contract_prompt() -> str:
    """Prompt block shared by visual-planning agents."""
    render_hints = ", ".join(CANONICAL_RENDER_HINTS)
    continuity_actions = ", ".join(CANONICAL_CONTINUITY_ACTIONS)
    character_roles = ", ".join(CANONICAL_CHARACTER_ROLES)
    coordinate_kinds = ", ".join(CANONICAL_COORDINATE_SYSTEM_KINDS)
    camera_actions = ", ".join(CANONICAL_CAMERA_ACTIONS)
    animation_hints = ", ".join(CANONICAL_ANIMATION_HINTS)
    return """
ENGINE CONTRACT (inspired by 3Blue1Brown's manimlib):
- Think in Scene -> Mobject -> Animation terms, not generic media assets.
- A scene is a timed sequence inside construct() built from concrete visual primitives.
- Prefer explicit visual primitives over vague labels like "diagram" or "graphic".

CANONICAL OBJECT TYPES:
- equation        -> MathTex / Tex style mathematical expressions
- text            -> Text / short explanatory labels
- axes            -> Axes or NumberPlane
- graph           -> a plotted curve or function anchored to axes
- geometric_shape -> Line, Arrow, Circle, Arc, Polygon, Rectangle, Triangle
- pi_creature     -> use repo helpers for the 3Blue1Brown-style teacher/students beat

OPTIONAL render_hint VALUES:
- {render_hints}

SEMANTIC CONTINUITY ENUMS:
- continuity_action: {continuity_actions}
- character_role: {character_roles}
- coordinate_system_kind: {coordinate_kinds}
- camera_action: {camera_actions}

PREFERRED HELPERS:
- {preferred_helper_1}
- {preferred_helper_2}
- {preferred_helper_3}

RULES:
- Do not invent new object types.
- Do not output pi_character; use pi_creature instead.
- Use render_hint only when it materially clarifies the intended Manim primitive.
- pi_creature (the narrator mascot) is a SPICE, not a staple: use it ONLY for
  an intro, recap, or brief aside beat — at most ONE per scene, and NEVER in a
  scene that is presenting equations, graphs, or geometric constructions.
  For abstract narration beats without a natural visual, prefer a simple
  diagrammatic metaphor over the mascot.
- Every object should use a stable object_id and object_role when continuity matters.
- Prefer reusing a named coordinate system or notation key instead of redrawing it from scratch.

CANONICAL TRANSFORMATION TYPES:
- fade_in
- fade_out
- morph
- highlight
- move
- scale

If you want to reference a concrete Manim animation class, put it in
`animation_hint`, but `type` must still be one of the canonical values above.
Allowed animation_hint values include: {animation_hints}
""".format(
        render_hints=render_hints,
        continuity_actions=continuity_actions,
        character_roles=character_roles,
        coordinate_kinds=coordinate_kinds,
        camera_actions=camera_actions,
        preferred_helper_1=PREFERRED_HELPER_PATTERNS[0],
        preferred_helper_2=PREFERRED_HELPER_PATTERNS[1],
        preferred_helper_3=PREFERRED_HELPER_PATTERNS[2],
        animation_hints=animation_hints,
    )


def build_3b1b_quality_prompt() -> str:
    """Prompt block injecting 3B1B quality rules into code generation.

    Covers timing constraints, progressive reveal, color consistency,
    smooth morphing, and breathing room — the five pillars of 3B1B style.
    """
    timing_lines = []
    for anim, (lo, hi) in sorted(TIMING_CONTRACT.items()):
        timing_lines.append(f"  {anim}: {lo}–{hi}s")
    timing_block = "\n".join(timing_lines)

    wait_lines = []
    for ctx, val in WAIT_CONTRACT.items():
        wait_lines.append(f"  {ctx}: >= {val}s")
    wait_block = "\n".join(wait_lines)

    palette_lines = []
    for name, hex_val in list(THREE_BLUE_ONE_BROWN_PALETTE.items())[:6]:
        palette_lines.append(f"  {name}: {hex_val}")
    palette_block = "\n".join(palette_lines)

    return f"""
3B1B QUALITY CONTRACT:

1. TIMING CONSTRAINTS (run_time per animation):
{timing_block}

2. BREATHING ROOM (minimum self.wait() durations):
{wait_block}
- ALWAYS call self.wait() between consecutive self.play() calls.
- After revealing a major equation or concept, wait >= 1.0s.

3. PROGRESSIVE REVEAL:
- NEVER Write() a complex equation (>3 terms) in a single shot.
- Instead, build it up: show LHS first, then relation, then RHS.
- Use progressive_write() helper or isolate sub-expressions and animate sequentially.
- When updating an equation, use ReplacementTransform on the changed parts only.

4. COLOR CONSISTENCY:
- One mathematical variable = one color throughout the entire video.
- Use the 3B1B palette for variable coloring:
{palette_block}
- Pass tex_to_color_map to MathTex for consistent variable coloring.
- Never change a variable's color between scenes.

5. SMOOTH MORPHING:
- Prefer ReplacementTransform over FadeOut→FadeIn for equation updates.
- When an equation evolves (e.g., simplification), morph the parts that change.
- Use TransformFromCopy when showing a derivation step that keeps the original.

6. ANTI-PATTERNS TO AVOID:
- FadeOut immediately followed by FadeIn of a related object (use Transform instead).
- Multiple self.play() calls with no self.wait() in between.
- More than 3 Text objects visible simultaneously (creates a "text wall").
- Creating a new Axes/NumberPlane when one is already active (reuse it).
"""


def build_animation_codegen_contract_prompt() -> str:
    """Prompt block shared by code-generation agents."""
    render_hints = ", ".join(CANONICAL_RENDER_HINTS)
    animation_hints = ", ".join(CANONICAL_ANIMATION_HINTS)
    preferred_helper_1 = PREFERRED_HELPER_PATTERNS[0]
    preferred_helper_2 = PREFERRED_HELPER_PATTERNS[1]
    preferred_helper_3 = PREFERRED_HELPER_PATTERNS[2]
    return """
ENGINE MODEL (inspired by 3Blue1Brown's manimlib Scene/Mobject/Animation design):
- All logic lives inside Scene.construct().
- Create explicit Mobjects first, then animate them with self.play(...).
- Use self.wait(...) to hold the frame while narration continues.
- Manage object lifecycle deliberately: create -> animate -> transform -> fade out / remove.

PREFERRED MOBJECT FAMILIES:
- equation        -> MathTex (or Tex when plain TeX text is clearer)
- text            -> Text
- axes            -> Axes or NumberPlane
- graph           -> plot attached to Axes/NumberPlane
- geometric_shape -> Line, Arrow, Polygon, Circle, Arc, Rectangle, Triangle
- pi_creature     -> create_pi_teacher(), create_pi_students(), create_speech_bubble()

PREFERRED HELPERS:
- {preferred_helper_1}
- {preferred_helper_2}
- {preferred_helper_3}

PREFERRED ANIMATION FAMILIES:
- fade_in         -> FadeIn / Write / Create
- fade_out        -> FadeOut / Uncreate
- morph           -> Transform / ReplacementTransform
- highlight       -> Indicate / Circumscribe / Flash
- move            -> .animate.shift(...), .animate.move_to(...), Rotate when motion is the goal
- scale           -> .animate.scale(...)

COMMUNITY-MANIM TARGET:
- Generate code for `from manim import *`, not `manimlib` imports.
- Use the repo helpers already imported for Pi creatures and layout tracking.
- If visual intent includes `render_hint`, treat it as the preferred primitive unless the scene content makes it invalid.
- Allowed render_hint values: {render_hints}
- Allowed animation_hint values: {animation_hints}
""".format(
        render_hints=render_hints,
        animation_hints=animation_hints,
        preferred_helper_1=preferred_helper_1,
        preferred_helper_2=preferred_helper_2,
        preferred_helper_3=preferred_helper_3,
    )
