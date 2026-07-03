"""Sig — the friendly blob narrator mascot (replaces the Pi creatures).

The old characters were a π MathTex glyph inside a box with dot "eyes"
floating over the glyph — structurally incapable of looking good. Sig is a
rounded blob with large expressive eyes and a simple mouth: ~7 primitives,
no LaTeX (immune to Tex render failures), and hard to make ugly.

The module keeps the historical name and the full public API so generated
code and prompts keep working unchanged:

    teacher  = create_pi_teacher()                       # big blue Sig
    students = create_pi_students()                      # small green Sigs
    who      = create_character_for_role("teacher")
    bubble   = create_speech_bubble("Key insight!", teacher)

Usage guidance (enforced by the prompts): the mascot appears for intro /
recap / aside beats only — at most one per scene, never inside dense math.
"""

import numpy as np
from manim import (
    BLACK,
    BLUE_B,
    BLUE_E,
    DARK_GRAY,
    DOWN,
    GREEN_C,
    LEFT,
    PI,
    RIGHT,
    TEAL_C,
    UP,
    WHITE,
    Arc,
    Circle,
    Dot,
    Ellipse,
    Line,
    Polygon,
    RoundedRectangle,
    Text,
    VGroup,
)


# ---------------------------------------------------------------------------
# The blob
# ---------------------------------------------------------------------------

def _make_blob(
    color,
    height: float = 1.2,
    emotion: str = "happy",
    stroke_color=None,
) -> VGroup:
    """A likeable blob: rounded body, two big eyes with pupils, simple mouth.

    ``emotion`` ∈ {"happy", "neutral", "thinking"}:
      - happy    → smile arc, pupils centered
      - neutral  → short flat mouth, pupils centered
      - thinking → flat mouth, pupils looking up-right
    """
    width = height * 0.92
    body = RoundedRectangle(
        corner_radius=width * 0.32,
        width=width,
        height=height,
        color=stroke_color or BLUE_E,
        fill_color=color,
        fill_opacity=1.0,
        stroke_width=3,
    )
    center = body.get_center()

    # Eyes — large whites with pupils, in the upper half of the body.
    eye_dx = width * 0.22
    eye_y = center[1] + height * 0.14
    eye_w, eye_h = width * 0.22, height * 0.26
    eyes = VGroup()
    pupil_shift = (
        np.array([eye_w * 0.12, eye_h * 0.16, 0.0])
        if emotion == "thinking"
        else np.array([0.0, 0.0, 0.0])
    )
    for side in (-1.0, 1.0):
        eye_center = np.array([center[0] + side * eye_dx, eye_y, 0.0])
        white = Ellipse(width=eye_w, height=eye_h, color=BLACK, stroke_width=1.5)
        white.set_fill(WHITE, opacity=1.0)
        white.move_to(eye_center)
        pupil = Dot(point=eye_center + pupil_shift, radius=eye_w * 0.22, color=BLACK)
        eyes.add(VGroup(white, pupil))

    # Mouth — below the eyes.
    mouth_y = center[1] - height * 0.18
    if emotion == "happy":
        mouth = Arc(
            radius=width * 0.22,
            start_angle=-PI * 0.8,
            angle=PI * 0.6,
            color=BLACK,
            stroke_width=2.5,
        )
        mouth.move_to(np.array([center[0], mouth_y, 0.0]))
    else:
        half = width * 0.14
        mouth = Line(
            np.array([center[0] - half, mouth_y, 0.0]),
            np.array([center[0] + half, mouth_y, 0.0]),
            color=BLACK,
            stroke_width=2.5,
        )

    return VGroup(body, eyes, mouth)


# ---------------------------------------------------------------------------
# Public builders (historical names preserved)
# ---------------------------------------------------------------------------

def create_pi_teacher(height: float = 1.4) -> VGroup:
    """The narrator: a big, blue, happy Sig.

    Returns a VGroup you can position with .shift() / .move_to().
    """
    return _make_blob(BLUE_B, height=height, emotion="happy")


def create_pi_students(count: int = 3, height: float = 0.8) -> VGroup:
    """A column of smaller student Sigs in green/teal tones."""
    palette = [GREEN_C, TEAL_C, BLUE_B]
    emotions = ["neutral", "thinking", "happy"]
    students = VGroup()
    for i in range(count):
        students.add(_make_blob(
            palette[i % len(palette)],
            height=height,
            emotion=emotions[i % len(emotions)],
        ))
    students.arrange(DOWN, buff=0.35)
    return students


# ---------------------------------------------------------------------------
# Role-aware helpers
# ---------------------------------------------------------------------------

#: Canonical ordering for deterministic left-to-right character lineups.
ROLE_ORDER: dict = {
    "teacher": 0,
    "student_left": 1,
    "student_center": 2,
    "student_right": 3,
    "narrator_proxy": 4,
}

_ROLE_EMOTIONS = {
    "student_left": "neutral",
    "student_center": "thinking",
    "student_right": "happy",
}


def create_character_for_role(role: str, height: float | None = None) -> VGroup:
    """Return the canonical mascot for *role*.

    Args:
        role: One of ``"teacher"``, ``"student_left"``, ``"student_center"``,
            ``"student_right"``, ``"narrator_proxy"``, or any custom string
            (falls back to the narrator style).
        height: Override the default character height in Manim units.

    Returns:
        VGroup representing the character.
    """
    if role == "teacher":
        return create_pi_teacher(height=height or 1.4)
    if role in {"student_left", "student_center", "student_right"}:
        return _make_blob(
            GREEN_C, height=height or 0.8, emotion=_ROLE_EMOTIONS[role],
        )
    # narrator_proxy and unknown roles → narrator style, slightly smaller
    return create_pi_teacher(height=height or 1.0)


def create_character_lineup(roles: list, spacing: float = 1.1) -> VGroup:
    """Create a left-to-right lineup of mascots for the given *roles*.

    Characters are sorted by ``ROLE_ORDER`` so the teacher always appears
    on the far left and students are ordered left → center → right, regardless
    of the order ``roles`` is given in.
    """
    ordered_roles = sorted(roles, key=lambda r: ROLE_ORDER.get(r, 99))
    characters = [create_character_for_role(role) for role in ordered_roles]
    lineup = VGroup(*characters)
    lineup.arrange(RIGHT, buff=spacing)
    return lineup


def create_speech_bubble(text: str, speaker: VGroup,
                         max_width: float = 4.5,
                         font_size: int = 24) -> VGroup:
    """Create a speech bubble next to *speaker* containing *text*.

    Auto-sizes to the text, points a small triangle at the speaker, and nudges
    itself back inside the visible frame if the speaker sits near an edge.
    """
    label = Text(text, font_size=font_size, color=WHITE)
    if label.width > max_width:
        label.scale(max_width / label.width)

    bg = RoundedRectangle(
        corner_radius=0.2,
        width=label.width + 0.6,
        height=label.height + 0.4,
        color=WHITE,
        fill_color=DARK_GRAY,
        fill_opacity=0.85,
        stroke_width=1.5,
    )
    label.move_to(bg.get_center())

    tip_start = bg.get_left() + LEFT * 0.05
    pointer = Polygon(
        tip_start,
        tip_start + LEFT * 0.3 + DOWN * 0.15,
        tip_start + DOWN * 0.25,
        color=WHITE,
        fill_color=DARK_GRAY,
        fill_opacity=0.85,
        stroke_width=1.5,
    )

    bubble = VGroup(bg, pointer, label)
    bubble.next_to(speaker, RIGHT, buff=0.4)

    # Frame guard: keep the bubble inside the safe area.
    right_edge = bubble.get_right()[0]
    if right_edge > 6.9:
        bubble.shift(LEFT * (right_edge - 6.9))
    top_edge = bubble.get_top()[1]
    if top_edge > 3.8:
        bubble.shift(DOWN * (top_edge - 3.8))
    bottom_edge = bubble.get_bottom()[1]
    if bottom_edge < -3.8:
        bubble.shift(UP * (-3.8 - bottom_edge))
    return bubble
