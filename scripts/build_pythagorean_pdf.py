from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas


PAGE_W = 960
PAGE_H = 540
OUTPUT = Path("output/pdf/pythagorean_visual_proof_5pages.pdf")

NAVY = HexColor("#132238")
INK = HexColor("#1E293B")
MUTED = HexColor("#64748B")
PALE = HexColor("#F4F7FA")
LINE = HexColor("#D8E1EA")
TEAL = HexColor("#0F8B8D")
TEAL_PALE = HexColor("#DDF4F2")
ORANGE = HexColor("#F28C28")
ORANGE_PALE = HexColor("#FFF0DE")
BLUE = HexColor("#2E6FBB")
BLUE_PALE = HexColor("#E5EFFB")
RED = HexColor("#D94A4A")


def text(c, value, x, y, size, color=INK, font="Helvetica", align="left"):
    c.setFillColor(color)
    c.setFont(font, size)
    if align == "center":
        c.drawCentredString(x, y, value)
    elif align == "right":
        c.drawRightString(x, y, value)
    else:
        c.drawString(x, y, value)


def wrapped(c, value, x, y, width, size, leading, color=INK, font="Helvetica"):
    words = value.split()
    current = ""
    lines = []
    for word in words:
        candidate = f"{current} {word}".strip()
        if c.stringWidth(candidate, font, size) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    for index, line in enumerate(lines):
        text(c, line, x, y - index * leading, size, color, font)


def box(c, x, y, width, height, fill=PALE, stroke=LINE, radius=12, line_width=1.2):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(line_width)
    c.roundRect(x, y, width, height, radius, fill=1, stroke=1)


def page_frame(c, page, section):
    c.setFillColor(white)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.rect(0, PAGE_H - 8, PAGE_W, 8, fill=1, stroke=0)
    text(c, "A VISUAL PROOF OF PYTHAGORAS", 54, PAGE_H - 38, 10, MUTED, "Helvetica-Bold")
    text(c, section.upper(), PAGE_W - 54, PAGE_H - 38, 10, MUTED, "Helvetica-Bold", "right")
    c.setStrokeColor(LINE)
    c.setLineWidth(1)
    c.line(54, 36, PAGE_W - 54, 36)
    text(c, "Geometry through area and rearrangement", 54, 18, 9, MUTED)
    text(c, f"{page} / 5", PAGE_W - 54, 18, 9, MUTED, align="right")


def heading(c, eyebrow, title_value, subtitle):
    text(c, eyebrow.upper(), 62, 438, 12, TEAL, "Helvetica-Bold")
    text(c, title_value, 62, 392, 34, NAVY, "Helvetica-Bold")
    wrapped(c, subtitle, 62, 356, 836, 16, 22, MUTED)


def polygon(c, points, fill, stroke, line_width=2):
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    path.close()
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(line_width)
    c.drawPath(path, fill=1, stroke=1)


def right_angle(c, x, y, size=16, color=MUTED):
    c.setStrokeColor(color)
    c.setLineWidth(2)
    c.line(x, y + size, x + size, y + size)
    c.line(x + size, y + size, x + size, y)


def right_triangle(c, x, y, a, b, fill=PALE, labels=True):
    points = [(x, y), (x + a, y), (x, y + b)]
    polygon(c, points, fill, NAVY, 2.4)
    right_angle(c, x + 5, y + 5, 15)
    if labels:
        text(c, "a", x + a / 2, y - 26, 19, TEAL, "Helvetica-Bold", "center")
        text(c, "b", x - 24, y + b / 2 - 6, 19, ORANGE, "Helvetica-Bold", "center")
        text(c, "c", x + a * 0.56, y + b * 0.55, 19, BLUE, "Helvetica-Bold", "center")
    return points


def page_one(c):
    page_frame(c, 1, "The question")
    heading(
        c,
        "One triangle, one famous relationship",
        "Can We Measure a Diagonal Without Measuring It?",
        "For every right triangle, the two shorter sides determine the longest side exactly.",
    )
    right_triangle(c, 112, 105, 310, 190, TEAL_PALE)
    text(c, "right angle", 152, 130, 11, MUTED, "Helvetica-Bold")
    box(c, 545, 121, 330, 176, NAVY, stroke=NAVY)
    text(c, "The Pythagorean theorem", 575, 258, 13, TEAL_PALE, "Helvetica-Bold")
    text(c, "a^2 + b^2 = c^2", 710, 202, 31, white, "Helvetica-Bold", "center")
    wrapped(c, "The square on the longest side has the same area as the other two squares combined.",
            575, 165, 270, 14, 20, white)
    text(c, "Why should lengths obey an equation about squares?", 480, 70, 16, NAVY,
         "Helvetica-Bold", "center")


def page_two(c):
    page_frame(c, 2, "Build the areas")
    heading(
        c,
        "Turn lengths into areas",
        "Build a Square on Every Side",
        "A side of length a makes area a^2. The same idea gives areas b^2 and c^2.",
    )
    x, y = 365, 200
    a, b = 145, 100
    # Square on side a, below the triangle.
    c.setFillColor(TEAL_PALE)
    c.setStrokeColor(TEAL)
    c.setLineWidth(2)
    c.rect(x, y - a, a, a, fill=1, stroke=1)
    text(c, "area a^2", x + a / 2, y - a / 2 - 7, 19, TEAL, "Helvetica-Bold", "center")
    # Square on side b, left of the triangle.
    c.setFillColor(ORANGE_PALE)
    c.setStrokeColor(ORANGE)
    c.rect(x - b, y, b, b, fill=1, stroke=1)
    text(c, "area", x - b / 2, y + b / 2 + 7, 15, ORANGE, "Helvetica-Bold", "center")
    text(c, "b^2", x - b / 2, y + b / 2 - 18, 19, ORANGE, "Helvetica-Bold", "center")
    # Triangle and an offset schematic square on the hypotenuse.
    right_triangle(c, x, y, a, b, white)
    box(c, 645, 132, 230, 190, BLUE_PALE, stroke=BLUE, radius=8, line_width=2)
    text(c, "square on c", 760, 274, 13, BLUE, "Helvetica-Bold", "center")
    text(c, "area c^2", 760, 208, 27, BLUE, "Helvetica-Bold", "center")
    text(c, "The theorem is an area claim:", 760, 164, 13, MUTED, "Helvetica-Bold", "center")
    text(c, "teal + orange = blue", 760, 140, 13, NAVY, "Helvetica-Bold", "center")


def page_three(c):
    page_frame(c, 3, "Rearrange")
    heading(
        c,
        "Four identical triangles",
        "Arrange the Same Pieces Inside One Large Square",
        "The outer square has side a + b. Four copies of the right triangle leave a tilted square in the center.",
    )
    x, y = 124, 66
    a, b = 156, 104
    length = a + b
    c.setStrokeColor(NAVY)
    c.setLineWidth(3)
    c.rect(x, y, length, length, fill=0, stroke=1)
    center = [(x + a, y), (x + length, y + a), (x + b, y + length), (x, y + b)]
    triangles = [
        [(x, y), (x + a, y), (x, y + b)],
        [(x + a, y), (x + length, y), (x + length, y + a)],
        [(x + length, y + a), (x + length, y + length), (x + b, y + length)],
        [(x + b, y + length), (x, y + length), (x, y + b)],
    ]
    fills = [TEAL_PALE, ORANGE_PALE, TEAL_PALE, ORANGE_PALE]
    strokes = [TEAL, ORANGE, TEAL, ORANGE]
    for points, fill, stroke in zip(triangles, fills, strokes):
        polygon(c, points, fill, stroke, 1.8)
    polygon(c, center, BLUE_PALE, BLUE, 2.5)
    text(c, "c^2", x + length / 2, y + length / 2 - 8, 28, BLUE, "Helvetica-Bold", "center")
    text(c, "a + b", x + length / 2, y - 25, 14, NAVY, "Helvetica-Bold", "center")
    box(c, 514, 92, 362, 238, PALE, stroke=LINE)
    text(c, "Area of the outer square", 542, 290, 13, MUTED, "Helvetica-Bold")
    text(c, "(a + b)^2", 695, 249, 27, NAVY, "Helvetica-Bold", "center")
    c.setStrokeColor(LINE)
    c.line(542, 224, 848, 224)
    text(c, "Area of four triangles", 542, 198, 13, MUTED, "Helvetica-Bold")
    text(c, "4 x (ab / 2) = 2ab", 695, 157, 22, ORANGE, "Helvetica-Bold", "center")
    text(c, "What remains in blue must be c^2.", 695, 116, 14, BLUE, "Helvetica-Bold", "center")


def page_four(c):
    page_frame(c, 4, "Reveal the proof")
    heading(
        c,
        "Subtract the four triangles",
        "The Extra Area Cancels Perfectly",
        "The central blue square equals the outer area minus the area of the four identical triangles.",
    )
    steps = [
        ("c^2", "=", "(a + b)^2 - 2ab", BLUE),
        ("c^2", "=", "a^2 + 2ab + b^2 - 2ab", NAVY),
        ("c^2", "=", "a^2 + b^2", TEAL),
    ]
    ys = [284, 204, 124]
    for index, ((left, equals, right, color), y) in enumerate(zip(steps, ys)):
        fill = TEAL_PALE if index == 2 else PALE
        stroke = TEAL if index == 2 else LINE
        box(c, 102, y - 30, 756, 64, fill, stroke=stroke, radius=9, line_width=1.5)
        text(c, left, 214, y - 6, 24, color, "Helvetica-Bold", "center")
        text(c, equals, 330, y - 6, 24, MUTED, "Helvetica-Bold", "center")
        text(c, right, 582, y - 6, 23, color, "Helvetica-Bold", "center")
    text(c, "The 2ab added by expanding the outer square is exactly removed by the four triangles.",
         480, 72, 14, NAVY, "Helvetica-Bold", "center")


def page_five(c):
    page_frame(c, 5, "Use it")
    heading(
        c,
        "A real measurement",
        "Find the Diagonal of a 6 m by 8 m Hall",
        "The floor edges form the two shorter sides of a right triangle. The diagonal is the hypotenuse.",
    )
    x, y = 92, 102
    width, height = 365, 220
    c.setFillColor(PALE)
    c.setStrokeColor(NAVY)
    c.setLineWidth(2.5)
    c.rect(x, y, width, height, fill=1, stroke=1)
    c.setStrokeColor(BLUE)
    c.setLineWidth(5)
    c.line(x, y, x + width, y + height)
    text(c, "8 m", x + width / 2, y - 28, 18, TEAL, "Helvetica-Bold", "center")
    text(c, "6 m", x - 31, y + height / 2 - 6, 18, ORANGE, "Helvetica-Bold", "center")
    text(c, "diagonal c", x + width / 2 + 20, y + height / 2 + 28, 17, BLUE,
         "Helvetica-Bold", "center")
    right_angle(c, x + 8, y + 8, 19)
    box(c, 542, 111, 336, 211, NAVY, stroke=NAVY)
    text(c, "6^2 + 8^2 = c^2", 710, 273, 23, white, "Helvetica-Bold", "center")
    text(c, "36 + 64 = c^2", 710, 226, 20, white, "Helvetica-Bold", "center")
    text(c, "100 = c^2", 710, 181, 20, white, "Helvetica-Bold", "center")
    text(c, "c = 10 m", 710, 132, 29, TEAL_PALE, "Helvetica-Bold", "center")
    text(c, "One visual proof becomes a practical measuring tool.", 650, 55, 15, NAVY,
         "Helvetica-Bold", "center")


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("A Visual Proof of the Pythagorean Theorem")
    c.setAuthor("PDF-to-Manim Pipeline Demonstration")
    for draw_page in (page_one, page_two, page_three, page_four, page_five):
        draw_page(c)
        c.showPage()
    c.save()
    print(OUTPUT.resolve())


if __name__ == "__main__":
    build()
