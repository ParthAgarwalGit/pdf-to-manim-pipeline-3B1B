"""Prompts for the per-scene visual planner.

``visual_plan_static_prompt`` is the byte-identical prefix of every planner
call (provider prefix caching depends on that: no per-scene data in it).
``lesson_context_block`` and ``beat_schedule_block`` are per-scene and go
after it.
"""

from typing import Any, Iterable, List, Optional

# Budgets quoted to the planner. They are deliberately one below the limits
# the validator enforces (ScriptArchitectAgent.MAX_CONCURRENT_OBJECTS = 7,
# MAX_OBJECTS_PER_SCENE = 15) so a plan that miscounts by one still passes.
PROMPT_MAX_OBJECTS = 12
PROMPT_MAX_CONCURRENT = 6


def visual_plan_static_prompt(engine_contract: str, catalog_block: str) -> str:
    """Standing rules for planning one scene's visuals."""
    return (
        "You design the visuals for ONE scene of a longer animated explainer. A code "
        "generator animates exactly what you plan, and a narrator speaks over it. Plan a "
        "shot that moves, not a slide. The scene's narration, its place in the lesson and "
        "its timing follow after these standing rules.\n\n"
        f"{engine_contract}\n"
        "CLOSED VOCABULARY:\n"
        f"{catalog_block}\n\n"
        "OBJECT FIELDS (every object, always):\n"
        "- object_id: a stable snake_case id. When the object is one of the video's motifs "
        "(listed in VISUAL WORLD below, when present), use the motif_id as the object_id.\n"
        "- object_type: one of equation, text, axes, graph, geometric_shape, pi_creature.\n"
        "- primitive: one of the MOBJECT names above. args: its kwargs, using only the "
        "parameters listed for it (plus color).\n"
        "- content: the LaTeX or text string for MathTex / Tex / Text (the same string as "
        "args.tex or args.text); \"\" for a pure shape.\n"
        "- purpose: what the viewer learns from this object, said specifically.\n"
        "- supports_beat_ids: the narration beat ids it serves, taken from VALID BEAT IDS.\n"
        "- bbox {x, y, w, h}: centre and size in Manim units. interval {t_in, t_out}: when "
        "it is visible, in scene seconds.\n"
        "- style: {\"color\": \"<Manim color name>\"}.\n"
        "- coordinate_system_id: for anything drawn ON an Axes, NumberPlane or NumberLine, "
        "the object_id of that host.\n\n"
        "SHOT DESIGN, in this order:\n"
        "1. HERO. Choose the ONE visual this scene is about: a plane with vectors on it, a "
        "curve, a shape being transformed, a diagram of boxes and arrows. It is built from "
        "non-text primitives, it sits in the middle of the frame and it is BIG: its bbox is "
        "at least 7 wide or 4.5 tall. A coordinate system that is the hero has x_length 9 to "
        "11 and y_length 5.5 to 6.5; when a formula shares the frame, 6 to 7.5 by 4.5 to 5.5. "
        "Points with coordinates, vectors, lines through the origin and curves are drawn ON a "
        "coordinate system: include the NumberPlane or Axes as an object (it is part of the "
        "hero), keep it for the whole scene, and set coordinate_system_id on everything drawn "
        "on it. A set of points is one object per point only when the points act separately; "
        "otherwise describe the set once and give its coordinates in `purpose`.\n"
        "2. MOTION. For each narration beat decide what CHANGES on the hero while the words "
        "are spoken: something rotates, stretches, slides, splits, lines up, lights up. Put "
        "that in `transformations` (at least one per beat; prefer morph, move, scale and "
        "highlight over fade_in) and state it in `beats[].beat_goal` as a verb phrase. A beat "
        "in which nothing changes is a slide.\n"
        "3. SUPPORT. Add only what the narration needs to be understood: at most ONE formula "
        "and at most THREE short labels visible at any moment.\n"
        "4. STAGING. The hero is on screen from the first two seconds to the end of the scene "
        "(interval 0 to scene end). Things change on it. Do not clear the screen in the middle "
        "of a scene to start a different picture.\n\n"
        f"BUDGET (hard): at most {PROMPT_MAX_OBJECTS} objects in the scene and at most "
        f"{PROMPT_MAX_CONCURRENT} visible at the same moment. Count them before you answer. A "
        "plan over the budget is cut down automatically, labels first, so anything beyond it is "
        "lost. Reuse and transform objects instead of adding new ones.\n\n"
        "TEXT ON SCREEN:\n"
        "- The narration is spoken. Never put a narration sentence, a title, a heading or a "
        "caption that restates the voice on screen.\n"
        "- A Text object is a label: at most four words, naming something the viewer is "
        "looking at, placed beside it.\n"
        "- Mathematics goes in MathTex. Show a formula only after the picture has given its "
        "parts a meaning, and never more than one new formula in a scene.\n"
        "- VISUAL IMPACT (non-negotiable): at least ONE object MUST be a non-text primitive "
        "(Axes, NumberPlane, NumberLine, Vector, Arrow, Circle, Ellipse, Polygon, Line, Dot, "
        "...) that DEPICTS the idea. A scene of text and formulas alone is a slide.\n\n"
        "LAYOUT (x, y, t):\n"
        "- The frame is 14.2 wide and 8 tall with the origin at the centre. SAFE AREA: keep "
        "bbox.x in [-6.3, 6.3] and bbox.y in [-3.4, 3.4], including the object's own extent.\n"
        "- Things that live ON another object (points, vectors and curves on a coordinate "
        "system; the axes of an ellipse; a brace under a term) get the bbox of where they "
        "really are, even though it overlaps the host. That overlap is expected.\n"
        "- Labels sit beside what they name. Two text objects may never share space while "
        "both are visible; give them different positions or disjoint intervals.\n"
        "- When a formula shares the frame, give it its own zone: a strip along the top "
        "(y about 3.0) or a column on the right (x about 4.5), with the hero in the rest.\n\n"
        "CONTINUITY:\n"
        "- When the scene context lists objects the previous scene left on screen, this "
        "scene starts from them: re-emit each one first, with the same object_id, primitive, "
        "args and position, t_in = 0.\n"
        "- `scene_state_out.active_object_ids` lists what is still on screen at the end and "
        "should open the next scene: the hero and at most two other objects, never labels.\n\n"
        "MOTION PLAN SHAPES (exact; an item with a missing field is discarded):\n"
        "\"transformations\": [{\"transformation_type\": \"morph|move|scale|highlight|fade_in|fade_out\", "
        "\"source_object_id\": \"<an object_id>\", \"target_object_id\": \"<an object_id, or null>\", "
        "\"beat_id\": \"<a valid beat id>\", \"timing\": <seconds from scene start>, "
        "\"continuity_action\": \"create|reuse|transform|remove|highlight\", "
        "\"parameters\": {\"description\": \"one sentence: what the viewer sees happen\", "
        "\"animation_hint\": \"<optional animation name from the list above>\"}}]\n"
        "\"beats\": [{\"beat_id\": \"<a valid beat id>\", \"narration_excerpt\": \"first words of the beat\", "
        "\"start_time\": <seconds>, \"end_time\": <seconds>, \"active_object_ids\": [\"<object_id>\"], "
        "\"camera_action\": \"static\", \"beat_goal\": \"verb phrase: what changes on screen\"}]\n"
        "\"emphasis_points\": [{\"timestamp\": <seconds>, \"description\": \"what to draw the eye to\"}]\n"
        "\"scene_state_in\" and \"scene_state_out\": {\"active_object_ids\": [\"<object_id>\"]}\n\n"
        "Return JSON with keys: objects (list), transformations (list), "
        "beats (list), scene_state_in (object), scene_state_out (object), "
        "emphasis_points (list).\n\n"
        "───────────────  SCENE-SPECIFIC CONTEXT BELOW  ───────────────\n\n"
    )


def lesson_context_block(plan: Optional[Any], act_id: str = "") -> str:
    """Where this scene sits in the lesson, and the look the whole video shares.

    Returns "" when there is no plan, so legacy runs keep their prompt.
    """
    if plan is None:
        return ""
    lines: List[str] = [
        "LESSON CONTEXT (the whole video; this scene is one step of it):",
        f"  Central question: {plan.central_question}",
    ]
    act = plan.act(act_id) if act_id else None
    if act is not None:
        lines.append(
            f"  This scene belongs to {act.act_id} ({act.kind}). Viewer question: "
            f"{act.viewer_question or '(none)'} Key idea: {act.key_idea}"
        )
        if act.visual:
            lines.append(f"  Planned visual for this act: {act.visual}")
    world = plan.visual_world_lines()
    if world:
        lines.append(
            "VISUAL WORLD (the same in every scene: same object ids, same colors, same values):"
        )
        lines.extend(f"  - {line}" for line in world)
        lines.append(
            "  When this scene shows a motif, use its motif_id as the object_id and its color "
            "in style.color, and draw the running example with exactly these values."
        )
    return "\n".join(lines) + "\n\n"


def beat_schedule_block(beats: Iterable[Any], words_per_second: float) -> str:
    """Estimated start and end of each beat, so intervals line up with the words."""
    lines: List[str] = []
    cursor = 0.0
    for beat in beats:
        words = len((getattr(beat, "narration", "") or "").split())
        seconds = words / words_per_second if words_per_second > 0 else 0.0
        opening = " ".join((getattr(beat, "narration", "") or "").split()[:8])
        lines.append(
            f"  {beat.beat_id}: {cursor:.1f}s to {cursor + seconds:.1f}s  \"{opening}...\""
        )
        cursor += seconds
    if not lines:
        return ""
    return (
        "BEAT SCHEDULE (estimated; use these times for intervals, transformation timing and "
        "beats[].start_time / end_time):\n" + "\n".join(lines) + "\n\n"
    )
