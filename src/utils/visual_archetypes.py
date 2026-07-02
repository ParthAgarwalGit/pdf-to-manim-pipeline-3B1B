"""Concept → visual-archetype library.

The distilled "how would 3Blue1Brown draw this?" knowledge base. Each
archetype is a proven visual pattern for a family of mathematical concepts:
what to put on screen (closed-catalog primitives only) and how it should
MOVE so the concept becomes visible rather than described.

Consumers:
  - ``ScriptArchitectAgent._generate_visual_intent_for_chunk`` classifies each
    chunk and injects the matched archetype's design block into the planner
    prompt, so the LLM starts from a proven pattern instead of a blank page.
  - ``ScriptArchitectAgent._generate_fallback_visual_intent`` uses the matched
    archetype's ``fallback_primitive`` to put a real (non-text) visual on
    screen even in keyless/degraded runs.

Classification is deterministic keyword scoring — hermetic, testable, and
cheap. It is a *recommendation* signal: prompts tell the LLM to adapt, and a
no-match simply omits the block.

Adding an archetype: append to ``ARCHETYPE_REGISTRY``. Every name listed in
``primitives`` and the ``fallback_primitive`` pair must exist in
``MOBJECT_CATALOG`` (enforced by tests), so the library can never recommend
something the schema would reject.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class VisualArchetype:
    """One reusable visual design pattern for a concept family."""

    name: str
    description: str  # what concept family this depicts
    keywords: Tuple[str, ...]  # word-boundary matched, lowercase
    primitives: Tuple[str, ...]  # MOBJECT_CATALOG names to suggest
    motion: str  # how the visual should move (prompt prose)
    # (primitive_name, kwargs, canonical object_type) for the deterministic
    # fallback intent. kwargs must validate against allowed_args_for(name).
    fallback_primitive: Tuple[str, Dict[str, Any], str]


ARCHETYPE_REGISTRY: Dict[str, VisualArchetype] = {
    "function_graph": VisualArchetype(
        name="function_graph",
        description="A function, curve, or plotted relationship between quantities.",
        keywords=(
            "function", "graph", "curve", "plot", "axis", "axes",
            "domain", "range", "parabola", "exponential", "logarithm",
            "sine wave", "cosine wave",
        ),
        primitives=("Axes", "Line", "Dot", "MathTex"),
        motion=(
            "Draw the axes first, then trace the curve progressively with "
            "Create so the viewer watches the relationship emerge. Slide a "
            "Dot along the curve to connect input and output; label only the "
            "key point, not every feature."
        ),
        fallback_primitive=(
            "Axes",
            {"x_range": [-4, 4, 1], "y_range": [-2, 2, 1], "x_length": 5.0, "y_length": 3.0},
            "axes",
        ),
    ),
    "rate_of_change": VisualArchetype(
        name="rate_of_change",
        description="Derivatives, slopes, velocity — instantaneous change of one quantity.",
        keywords=(
            "derivative", "slope", "rate of change", "velocity", "tangent",
            "instantaneous", "differentiat", "gradient", "steepness",
        ),
        primitives=("Axes", "Line", "Dot", "DashedLine", "DecimalNumber"),
        motion=(
            "Slide a Dot along a curve while a tangent Line pivots to match "
            "the local slope; a DecimalNumber tracking the slope value makes "
            "the change legible. Steep vs. flat regions should be FELT by "
            "the tangent's rotation speed."
        ),
        fallback_primitive=(
            "Line",
            {"start": (-2.0, -1.0, 0.0), "end": (2.0, 1.0, 0.0)},
            "geometric_shape",
        ),
    ),
    "accumulation_series": VisualArchetype(
        name="accumulation_series",
        description="Sums, series, integrals-as-area — a total built from pieces.",
        keywords=(
            "sum", "series", "accumulate", "total", "integral", "area under",
            "add up", "partial sum", "riemann", "converges to",
        ),
        primitives=("Rectangle", "Axes", "MathTex", "DecimalNumber"),
        motion=(
            "Build the total piece by piece: Rectangles appear one at a time "
            "(LaggedStart feel) while a running DecimalNumber accumulates. "
            "For integrals, thin the rectangles to suggest the limit."
        ),
        fallback_primitive=(
            "Rectangle",
            {"width": 1.0, "height": 2.0},
            "geometric_shape",
        ),
    ),
    "vector_space": VisualArchetype(
        name="vector_space",
        description="Vectors, directions, forces, spans, and linear combinations.",
        keywords=(
            "vector", "direction", "magnitude", "force", "velocity vector",
            "span", "basis", "component", "dot product", "cross product",
            "linear combination",
        ),
        primitives=("NumberPlane", "Vector", "Arrow", "DashedLine", "Dot"),
        motion=(
            "Ground everything on a NumberPlane. Grow Vectors from the origin "
            "with GrowArrow; show addition tip-to-tail by moving a copy, and "
            "scaling by stretching the vector while its label updates."
        ),
        fallback_primitive=(
            "Vector",
            {"direction": (2.0, 1.0, 0.0)},
            "geometric_shape",
        ),
    ),
    "transformation_morph": VisualArchetype(
        name="transformation_morph",
        description="Linear maps, matrix actions, rotations — space or objects transforming.",
        keywords=(
            "transform", "matrix", "linear map", "rotation", "rotate",
            "scaling", "reflection", "mapping", "eigenvector", "eigenvalue",
            "determinant",
        ),
        primitives=("NumberPlane", "Square", "Vector", "MathTex"),
        motion=(
            "Show the BEFORE state, then morph the whole picture — grid lines "
            "bending, a unit Square carried along — in one smooth Transform. "
            "Highlight what stays fixed (eigenvectors, origin) after the "
            "motion settles."
        ),
        fallback_primitive=(
            "Square",
            {"side_length": 2.0},
            "geometric_shape",
        ),
    ),
    "geometric_relation": VisualArchetype(
        name="geometric_relation",
        description="Shapes, angles, and theorems about their relationships.",
        keywords=(
            "triangle", "circle", "angle", "theorem", "polygon", "radius",
            "hypotenuse", "perpendicular", "parallel", "congruent", "similar",
            "pythagor", "geometr",
        ),
        primitives=("Polygon", "Circle", "Arc", "Line", "Dot", "MathTex"),
        motion=(
            "Construct the figure the way a compass-and-straightedge proof "
            "would: base shape first, then auxiliary Lines/Arcs one at a "
            "time. Use Indicate on the parts a relation connects, never all "
            "labels at once."
        ),
        fallback_primitive=(
            "Circle",
            {"radius": 1.8},
            "geometric_shape",
        ),
    ),
    "limit_convergence": VisualArchetype(
        name="limit_convergence",
        description="Limits, convergence, sequences approaching a value.",
        keywords=(
            "limit", "converge", "approach", "tends to", "epsilon",
            "sequence", "infinity", "arbitrarily close", "asymptote",
        ),
        primitives=("NumberLine", "Dot", "DashedLine", "MathTex"),
        motion=(
            "Place successive Dots on a NumberLine bunching toward the limit "
            "point, each hop visibly smaller. A DashedLine marks the limit; "
            "zoom or scale toward it so 'arbitrarily close' is seen, not said."
        ),
        fallback_primitive=(
            "NumberLine",
            {"x_range": [-5, 5, 1]},
            "axes",
        ),
    ),
    "probability_sampling": VisualArchetype(
        name="probability_sampling",
        description="Probability, randomness, distributions, expected values.",
        keywords=(
            "probability", "random", "sample", "distribution", "expected",
            "chance", "likelihood", "outcome", "dice", "coin", "bayes",
        ),
        primitives=("Dot", "Rectangle", "Circle", "DecimalNumber"),
        motion=(
            "Rain Dots into regions whose sizes encode probabilities; let the "
            "counts settle into the expected proportions while a "
            "DecimalNumber tracks the running frequency. Randomness should "
            "look random — jitter the drops."
        ),
        fallback_primitive=(
            "Dot",
            {"point": (0.0, 0.0, 0.0)},
            "geometric_shape",
        ),
    ),
    "comparison_contrast": VisualArchetype(
        name="comparison_contrast",
        description="Two quantities, methods, or cases set against each other.",
        keywords=(
            "compare", "versus", "difference between", "greater", "smaller",
            "faster", "slower", "better", "worse", "trade-off", "contrast",
        ),
        primitives=("Rectangle", "Line", "MathTex", "Arrow"),
        motion=(
            "Split the frame: build the left case fully, then the right, "
            "using the same sub-layout so eyes can diff them. Finish by "
            "highlighting exactly the part that differs — nothing else."
        ),
        fallback_primitive=(
            "Rectangle",
            {"width": 2.5, "height": 1.5},
            "geometric_shape",
        ),
    ),
    "process_flow": VisualArchetype(
        name="process_flow",
        description="Algorithms, procedures, pipelines — ordered steps with flow.",
        keywords=(
            "algorithm", "step", "procedure", "pipeline", "iterate",
            "process", "sequence of steps", "workflow", "recursion", "loop",
        ),
        primitives=("RoundedRectangle", "Arrow", "Text"),
        motion=(
            "Reveal stages left-to-right: RoundedRectangle appears, then a "
            "GrowArrow into the next stage. For iteration, send a Dot around "
            "the loop path so the cycle is watched, not narrated."
        ),
        fallback_primitive=(
            "RoundedRectangle",
            {"width": 2.5, "height": 1.2, "corner_radius": 0.2},
            "geometric_shape",
        ),
    ),
    "equation_evolution": VisualArchetype(
        name="equation_evolution",
        description="An identity or formula being derived, rearranged, or simplified.",
        keywords=(
            "equation", "identity", "formula", "simplify", "solve",
            "substitut", "rearrange", "derive", "proof", "both sides",
        ),
        primitives=("MathTex", "Rectangle", "Arrow"),
        motion=(
            "Morph the equation between steps with ReplacementTransform so "
            "unchanged terms visibly persist and only the rewritten part "
            "moves. Box (Rectangle) the term being acted on before each step. "
            "Anchor with a small geometric illustration of what the equation "
            "MEANS whenever one exists."
        ),
        fallback_primitive=(
            "Rectangle",
            {"width": 4.0, "height": 1.2},
            "geometric_shape",
        ),
    ),
    "set_grouping": VisualArchetype(
        name="set_grouping",
        description="Sets, membership, unions/intersections, classifications.",
        keywords=(
            "set", "subset", "element", "union", "intersection", "belongs",
            "member", "venn", "category", "classify", "partition",
        ),
        primitives=("Circle", "Ellipse", "Dot", "Text"),
        motion=(
            "Draw overlapping Circles/Ellipses as regions, then fly Dots "
            "(elements) into their homes one group at a time. Highlight the "
            "overlap region when discussing intersection — motion into the "
            "region IS the definition."
        ),
        fallback_primitive=(
            "Ellipse",
            {"width": 3.0, "height": 2.0},
            "geometric_shape",
        ),
    ),
}


def classify_concept(text: str) -> Optional[VisualArchetype]:
    """Match text (narration + concept labels + hints) to the best archetype.

    Deterministic: scores each archetype by the number of DISTINCT keywords
    that hit (word-boundary, case-insensitive); highest score wins, ties go
    to registry order. Returns None when nothing matches — the caller simply
    omits the archetype block.
    """
    if not text:
        return None
    lowered = text.lower()

    best: Optional[VisualArchetype] = None
    best_score = 0
    for archetype in ARCHETYPE_REGISTRY.values():
        score = sum(
            1 for kw in archetype.keywords
            if re.search(r"\b" + re.escape(kw), lowered)
        )
        if score > best_score:
            best, best_score = archetype, score
    return best


def archetype_prompt_block(archetype: VisualArchetype) -> str:
    """Render one archetype as a compact prompt section for the visual planner."""
    return (
        f"RECOMMENDED VISUAL ARCHETYPE: {archetype.name}\n"
        f"  Concept family: {archetype.description}\n"
        f"  Suggested primitives: {', '.join(archetype.primitives)}\n"
        f"  Motion design: {archetype.motion}\n"
        "  This is a proven pattern for this concept family — start from it "
        "and adapt to the narration; do not force it where it fights the content.\n"
    )
