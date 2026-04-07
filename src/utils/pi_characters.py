"""Reusable 3Blue1Brown-style Pi character helpers for Manim scenes.

The teacher is a large BLUE π symbol with eyes and a mouth.
The students are 3 smaller DARK BROWN π symbols with eyes.
These are recognizable Pi CREATURES, not plain circles.

Usage inside generated scene code:

    from src.utils.pi_characters import create_pi_teacher, create_pi_students, create_speech_bubble

    teacher = create_pi_teacher().shift(LEFT * 3)
    students = create_pi_students().shift(RIGHT * 3)
    bubble = create_speech_bubble("Key insight here", teacher)

    self.play(FadeIn(teacher), FadeIn(students))
    self.play(Write(bubble))
    self.wait(3)
    self.play(FadeOut(bubble), FadeOut(teacher), FadeOut(students))
"""

from manim import *


def _make_pi_body(color, height: float = 1.2) -> VGroup:
    """Build a π-shaped creature body: a large π glyph with a rounded
    background capsule so it reads as a character, not just a symbol."""

    # The π glyph IS the body
    pi_glyph = MathTex(r"\pi", color=color, font_size=round(height * 80))

    # Soft rounded background so it looks like a creature, not floating text
    bg = RoundedRectangle(
        corner_radius=0.25,
        width=pi_glyph.width + 0.35,
        height=pi_glyph.height + 0.3,
        color=color,
        fill_color=color,
        fill_opacity=0.25,
        stroke_width=2,
    )
    bg.move_to(pi_glyph.get_center())

    return VGroup(bg, pi_glyph)


def create_pi_teacher(height: float = 1.4) -> VGroup:
    """Create a large blue Pi creature teacher with eyes and a mouth.

    Returns a VGroup you can position with .shift() / .move_to().
    """
    body = _make_pi_body(BLUE, height)
    body_center = body.get_center()

    # Eyes — two white dots placed above the horizontal bar of π
    eye_y = body_center[1] + height * 0.28
    left_eye = Dot(
        point=np.array([body_center[0] - 0.2, eye_y, 0]),
        radius=0.07, color=WHITE,
    )
    right_eye = Dot(
        point=np.array([body_center[0] + 0.2, eye_y, 0]),
        radius=0.07, color=WHITE,
    )
    # Pupils
    left_pupil = Dot(
        point=left_eye.get_center() + RIGHT * 0.02,
        radius=0.03, color=BLACK,
    )
    right_pupil = Dot(
        point=right_eye.get_center() + RIGHT * 0.02,
        radius=0.03, color=BLACK,
    )

    # Mouth — small arc below the glyph
    mouth = Arc(
        radius=0.15,
        start_angle=-PI * 0.75,
        angle=PI * 0.5,
        color=WHITE,
        stroke_width=2,
    )
    mouth.move_to(np.array([body_center[0], body_center[1] - height * 0.32, 0]))

    character = VGroup(body, left_eye, right_eye, left_pupil, right_pupil, mouth)
    return character


def create_pi_students(count: int = 3, height: float = 0.8) -> VGroup:
    """Create a column of small dark-brown Pi creature students.

    Returns a VGroup of *count* students stacked vertically.
    """
    students = VGroup()
    browns = ["#8B4513", "#A0522D", "#6B3410"]

    for i in range(count):
        color = browns[i % len(browns)]
        body = _make_pi_body(color, height)
        body_center = body.get_center()

        eye_y = body_center[1] + height * 0.25
        left_eye = Dot(
            point=np.array([body_center[0] - 0.12, eye_y, 0]),
            radius=0.04, color=WHITE,
        )
        right_eye = Dot(
            point=np.array([body_center[0] + 0.12, eye_y, 0]),
            radius=0.04, color=WHITE,
        )

        student = VGroup(body, left_eye, right_eye)
        students.add(student)

    students.arrange(DOWN, buff=0.35)
    return students


def create_speech_bubble(text: str, speaker: VGroup,
                         max_width: float = 4.5,
                         font_size: int = 24) -> VGroup:
    """Create a speech bubble next to *speaker* containing *text*.

    The bubble auto-sizes to the text and adds a small triangular pointer
    towards the speaker.
    """
    label = Text(text, font_size=font_size, color=WHITE)
    if label.width > max_width:
        label.scale(max_width / label.width)

    # Rounded rectangle background
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

    # Small triangle pointer towards the speaker
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
    return bubble
