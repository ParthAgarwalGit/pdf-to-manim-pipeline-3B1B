from __future__ import annotations

import math
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


PAGE_W = 960
PAGE_H = 540
OUTPUT_DIR = Path("output/pdf")

NAVY = HexColor("#14213D")
INK = HexColor("#1F2937")
MUTED = HexColor("#667085")
PALE = HexColor("#F5F7FA")
LINE = HexColor("#D8E0E8")
TEAL = HexColor("#168A8A")
TEAL_PALE = HexColor("#DDF4F1")
ORANGE = HexColor("#EF8A22")
ORANGE_PALE = HexColor("#FFF0DD")
BLUE = HexColor("#3777C8")
BLUE_PALE = HexColor("#E5EFFB")
RED = HexColor("#D95555")
RED_PALE = HexColor("#FBE7E7")
YELLOW = HexColor("#E5B93C")
YELLOW_PALE = HexColor("#FFF7D6")
PURPLE = HexColor("#7A5CC7")
PURPLE_PALE = HexColor("#EEE9FA")


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
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if stringWidth(candidate, font, size) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    for index, line in enumerate(lines):
        text(c, line, x, y - index * leading, size, color, font)


def box(c, x, y, width, height, fill=PALE, stroke=LINE, radius=10, line_width=1.2):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(line_width)
    c.roundRect(x, y, width, height, radius, fill=1, stroke=1)


def line(c, x1, y1, x2, y2, color=NAVY, width=2):
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)


def arrow(c, x1, y1, x2, y2, color=TEAL, width=2.5):
    line(c, x1, y1, x2, y2, color, width)
    angle = math.atan2(y2 - y1, x2 - x1)
    for offset in (2.55, -2.55):
        line(
            c,
            x2,
            y2,
            x2 + 11 * math.cos(angle + offset),
            y2 + 11 * math.sin(angle + offset),
            color,
            width,
        )


def polygon(c, points, fill, stroke=NAVY, width=2):
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    path.close()
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(width)
    c.drawPath(path, fill=1, stroke=1)


def page_frame(c, topic, tagline, section, page):
    c.setFillColor(white)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.rect(0, PAGE_H - 8, PAGE_W, 8, fill=1, stroke=0)
    text(c, topic.upper(), 52, PAGE_H - 38, 10, MUTED, "Helvetica-Bold")
    text(c, section.upper(), PAGE_W - 52, PAGE_H - 38, 10, MUTED, "Helvetica-Bold", "right")
    line(c, 52, 36, PAGE_W - 52, 36, LINE, 1)
    text(c, tagline, 52, 18, 9, MUTED)
    text(c, f"{page} / 5", PAGE_W - 52, 18, 9, MUTED, align="right")


def heading(c, eyebrow, title_value, subtitle):
    text(c, eyebrow.upper(), 60, 438, 12, TEAL, "Helvetica-Bold")
    text(c, title_value, 60, 393, 33, NAVY, "Helvetica-Bold")
    wrapped(c, subtitle, 60, 357, 830, 15, 21, MUTED)


def right_triangle(c, x, y, width, height, fill=TEAL_PALE, stroke=TEAL):
    polygon(c, [(x, y), (x + width, y), (x, y + height)], fill, stroke, 2.2)
    line(c, x + 8, y + 8, x + 24, y + 8, MUTED, 1.5)
    line(c, x + 24, y + 8, x + 24, y + 24, MUTED, 1.5)
    line(c, x + 24, y + 24, x + 8, y + 24, MUTED, 1.5)


def number_chip(c, value, x, y, fill=PALE, color=INK, width=76, height=52):
    box(c, x, y, width, height, fill, fill, 8)
    text(c, str(value), x + width / 2, y + 17, 20, color, "Helvetica-Bold", "center")


def save_pdf(path, title_value, page_functions):
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle(title_value)
    c.setAuthor("PDF-to-Manim Visual Lesson Set")
    for draw_page in page_functions:
        draw_page(c)
        c.showPage()
    c.save()
    print(path.resolve())


# Similar triangles and shadows


def similar_1(c):
    page_frame(c, "Similar triangles", "Same shape, different scale", "The idea", 1)
    heading(c, "Geometry that keeps its shape", "Same Angles, Different Sizes", "Similar triangles have equal matching angles. Their matching side lengths grow or shrink by one common scale factor.")
    right_triangle(c, 115, 102, 190, 125, TEAL_PALE, TEAL)
    right_triangle(c, 470, 102, 330, 218, ORANGE_PALE, ORANGE)
    text(c, "3", 210, 76, 17, TEAL, "Helvetica-Bold", "center")
    text(c, "2", 91, 158, 17, TEAL, "Helvetica-Bold", "center")
    text(c, "6", 635, 76, 17, ORANGE, "Helvetica-Bold", "center")
    text(c, "4", 446, 204, 17, ORANGE, "Helvetica-Bold", "center")
    arrow(c, 330, 181, 430, 181, BLUE)
    box(c, 327, 114, 106, 46, BLUE_PALE, BLUE)
    text(c, "scale x 2", 380, 130, 14, BLUE, "Helvetica-Bold", "center")


def similar_2(c):
    page_frame(c, "Similar triangles", "Same shape, different scale", "Match the sides", 2)
    heading(c, "Corresponding sides", "Every Matching Ratio Is Equal", "Place each side beside the side in the same position. The ratios agree because one scale factor controls the whole triangle.")
    right_triangle(c, 105, 105, 250, 150, TEAL_PALE, TEAL)
    right_triangle(c, 530, 105, 300, 180, ORANGE_PALE, ORANGE)
    text(c, "small", 230, 285, 14, TEAL, "Helvetica-Bold", "center")
    text(c, "large", 680, 315, 14, ORANGE, "Helvetica-Bold", "center")
    box(c, 326, 98, 308, 88, NAVY, NAVY)
    text(c, "large / small", 480, 154, 12, TEAL_PALE, "Helvetica-Bold", "center")
    text(c, "6/5 = 3.6/3 = 4.8/4 = 1.2", 480, 117, 19, white, "Helvetica-Bold", "center")
    text(c, "If two angles match, the side ratios must match too.", 480, 65, 15, NAVY, "Helvetica-Bold", "center")


def similar_3(c):
    page_frame(c, "Similar triangles", "Same shape, different scale", "Shadows", 3)
    heading(c, "A shared Sun angle", "A Person and a Tree Make Similar Triangles", "At the same moment, sunlight reaches the ground at the same angle. Vertical heights and horizontal shadows form matching right triangles.")
    line(c, 72, 96, 888, 96, NAVY, 3)
    line(c, 190, 96, 190, 246, TEAL, 8)
    c.setFillColor(TEAL)
    c.circle(190, 263, 18, fill=1, stroke=0)
    line(c, 190, 96, 340, 96, TEAL, 4)
    line(c, 190, 281, 340, 96, YELLOW, 3)
    line(c, 605, 96, 605, 330, ORANGE, 13)
    c.setFillColor(ORANGE)
    c.circle(605, 350, 24, fill=1, stroke=0)
    line(c, 605, 96, 840, 96, ORANGE, 4)
    line(c, 605, 374, 840, 96, YELLOW, 3)
    text(c, "person height", 166, 177, 13, TEAL, "Helvetica-Bold", "right")
    text(c, "person shadow", 265, 70, 13, TEAL, "Helvetica-Bold", "center")
    text(c, "tree height", 580, 221, 13, ORANGE, "Helvetica-Bold", "right")
    text(c, "tree shadow", 723, 70, 13, ORANGE, "Helvetica-Bold", "center")


def similar_4(c):
    page_frame(c, "Similar triangles", "Same shape, different scale", "Worked example", 4)
    heading(c, "Measure without climbing", "How Tall Is the Tree?", "A 1.5 m student casts a 2 m shadow. At the same time, a nearby tree casts an 8 m shadow.")
    box(c, 72, 124, 342, 176, TEAL_PALE, TEAL)
    text(c, "student", 243, 268, 14, TEAL, "Helvetica-Bold", "center")
    text(c, "height = 1.5 m", 102, 213, 20, NAVY, "Helvetica-Bold")
    text(c, "shadow = 2 m", 102, 165, 20, NAVY, "Helvetica-Bold")
    box(c, 546, 124, 342, 176, ORANGE_PALE, ORANGE)
    text(c, "tree", 717, 268, 14, ORANGE, "Helvetica-Bold", "center")
    text(c, "height = h", 576, 213, 20, NAVY, "Helvetica-Bold")
    text(c, "shadow = 8 m", 576, 165, 20, NAVY, "Helvetica-Bold")
    box(c, 247, 60, 466, 50, NAVY, NAVY)
    text(c, "1.5 / 2 = h / 8  ->  h = 6 m", 480, 77, 20, white, "Helvetica-Bold", "center")


def similar_5(c):
    page_frame(c, "Similar triangles", "Same shape, different scale", "Check the answer", 5)
    heading(c, "Reason before calculating", "Scale the Whole Triangle", "The tree shadow is four times the student shadow, so every matching length must also be four times as large.")
    box(c, 86, 158, 216, 132, TEAL_PALE, TEAL)
    text(c, "shadow", 194, 253, 12, TEAL, "Helvetica-Bold", "center")
    text(c, "2 m", 194, 198, 28, TEAL, "Helvetica-Bold", "center")
    arrow(c, 326, 224, 466, 224, BLUE, 3)
    text(c, "x 4", 396, 243, 20, BLUE, "Helvetica-Bold", "center")
    box(c, 490, 158, 216, 132, ORANGE_PALE, ORANGE)
    text(c, "shadow", 598, 253, 12, ORANGE, "Helvetica-Bold", "center")
    text(c, "8 m", 598, 198, 28, ORANGE, "Helvetica-Bold", "center")
    box(c, 729, 158, 145, 132, PALE, LINE)
    text(c, "height", 801, 253, 12, MUTED, "Helvetica-Bold", "center")
    text(c, "1.5 x 4", 801, 211, 19, NAVY, "Helvetica-Bold", "center")
    text(c, "= 6 m", 801, 178, 23, TEAL, "Helvetica-Bold", "center")
    text(c, "Similar triangles turn an inaccessible length into a proportion.", 480, 84, 16, NAVY, "Helvetica-Bold", "center")


# Quadratic vertex form


def axes(c, x, y, width, height):
    line(c, x, y + height / 2, x + width, y + height / 2, LINE, 1.5)
    line(c, x + width / 2, y, x + width / 2, y + height, LINE, 1.5)


def parabola(c, center_x, base_y, scale_x, scale_y, h=0, k=0, color=BLUE):
    points = []
    for step in range(-40, 41):
        value = step / 10
        px = center_x + (value + h) * scale_x
        py = base_y + (value * value + k) * scale_y
        points.append((px, py))
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    c.setStrokeColor(color)
    c.setLineWidth(3)
    c.drawPath(path, fill=0, stroke=1)


def quadratic_1(c):
    page_frame(c, "Quadratic functions", "Reading a parabola through its vertex", "The curve", 1)
    heading(c, "A smooth U-shaped graph", "Meet the Parabola", "The basic quadratic y = x^2 is symmetric. Its lowest point is the vertex, and every input x produces a squared output.")
    axes(c, 90, 80, 500, 250)
    parabola(c, 340, 115, 52, 13, color=BLUE)
    c.setFillColor(ORANGE)
    c.circle(340, 115, 7, fill=1, stroke=0)
    text(c, "vertex (0, 0)", 340, 88, 13, ORANGE, "Helvetica-Bold", "center")
    box(c, 650, 124, 240, 170, NAVY, NAVY)
    text(c, "y = x^2", 770, 246, 28, white, "Helvetica-Bold", "center")
    text(c, "x = 2  ->  y = 4", 770, 199, 16, TEAL_PALE, "Helvetica-Bold", "center")
    text(c, "x = -2 ->  y = 4", 770, 160, 16, TEAL_PALE, "Helvetica-Bold", "center")
    text(c, "mirror symmetry", 770, 132, 12, ORANGE_PALE, "Helvetica-Bold", "center")


def quadratic_2(c):
    page_frame(c, "Quadratic functions", "Reading a parabola through its vertex", "Vertex form", 2)
    heading(c, "Read the movement directly", "y = (x - h)^2 + k", "Vertex form starts with y = x^2 and moves its vertex to the point (h, k). The shape stays the same.")
    box(c, 80, 198, 260, 112, BLUE_PALE, BLUE)
    text(c, "horizontal move", 210, 275, 13, BLUE, "Helvetica-Bold", "center")
    text(c, "x - h", 210, 226, 27, NAVY, "Helvetica-Bold", "center")
    box(c, 350, 198, 260, 112, TEAL_PALE, TEAL)
    text(c, "vertical move", 480, 275, 13, TEAL, "Helvetica-Bold", "center")
    text(c, "+ k", 480, 226, 27, NAVY, "Helvetica-Bold", "center")
    box(c, 620, 198, 260, 112, ORANGE_PALE, ORANGE)
    text(c, "new vertex", 750, 275, 13, ORANGE, "Helvetica-Bold", "center")
    text(c, "(h, k)", 750, 226, 27, NAVY, "Helvetica-Bold", "center")
    text(c, "Example: y = (x - 3)^2 + 2 has vertex (3, 2).", 480, 132, 18, NAVY, "Helvetica-Bold", "center")
    text(c, "The minus sign inside means the graph moves right.", 480, 92, 14, MUTED, "Helvetica-Bold", "center")


def quadratic_3(c):
    page_frame(c, "Quadratic functions", "Reading a parabola through its vertex", "Transformations", 3)
    heading(c, "Compare three graphs", "Slide the Parabola Around", "Changing h moves the graph left or right. Changing k moves it up or down. The vertex tracks both movements.")
    panels = [(65, "y = x^2", "(0, 0)", BLUE), (355, "y = (x - 2)^2", "(2, 0)", TEAL), (645, "y = (x - 2)^2 + 2", "(2, 2)", ORANGE)]
    for x, equation, vertex, color in panels:
        box(c, x, 88, 250, 232, PALE, LINE)
        axes(c, x + 20, 120, 210, 150)
        vertex_x = x + 125 + (0 if color == BLUE else 40)
        vertex_y = 150 + (26 if color == ORANGE else 0)
        parabola(c, vertex_x, vertex_y, 20, 9, color=color)
        text(c, equation, x + 125, 290, 14, color, "Helvetica-Bold", "center")
        text(c, f"vertex {vertex}", x + 125, 101, 12, NAVY, "Helvetica-Bold", "center")


def quadratic_4(c):
    page_frame(c, "Quadratic functions", "Reading a parabola through its vertex", "Worked example", 4)
    heading(c, "Read first, calculate second", "Analyze y = (x - 2)^2 - 3", "The equation already reveals the graph's key point. Then a few symmetric values sketch the curve.")
    box(c, 68, 126, 336, 190, NAVY, NAVY)
    text(c, "vertex", 100, 277, 13, TEAL_PALE, "Helvetica-Bold")
    text(c, "(2, -3)", 370, 265, 25, white, "Helvetica-Bold", "right")
    text(c, "axis of symmetry", 100, 220, 13, TEAL_PALE, "Helvetica-Bold")
    text(c, "x = 2", 370, 208, 25, white, "Helvetica-Bold", "right")
    text(c, "minimum value", 100, 163, 13, TEAL_PALE, "Helvetica-Bold")
    text(c, "y = -3", 370, 151, 25, white, "Helvetica-Bold", "right")
    box(c, 482, 126, 410, 190, PALE, LINE)
    text(c, "symmetric points", 687, 277, 14, NAVY, "Helvetica-Bold", "center")
    for index, value in enumerate(["x=0, y=1", "x=1, y=-2", "x=2, y=-3", "x=3, y=-2", "x=4, y=1"]):
        text(c, value, 535 + (index % 2) * 180, 230 - (index // 2) * 48, 16, TEAL if index == 2 else INK, "Helvetica-Bold")
    text(c, "Equal distances from x = 2 give equal heights.", 480, 76, 15, ORANGE, "Helvetica-Bold", "center")


def quadratic_5(c):
    page_frame(c, "Quadratic functions", "Reading a parabola through its vertex", "Application", 5)
    heading(c, "A simple flight path", "When Does the Ball Reach Maximum Height?", "Suppose a ball's height is H(t) = -(t - 2)^2 + 5 metres. The negative sign turns the parabola upside down.")
    axes(c, 74, 84, 520, 238)
    points = []
    for step in range(0, 81):
        t = step / 20
        height = -(t - 2) ** 2 + 5
        points.append((118 + t * 105, 105 + height * 36))
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    c.setStrokeColor(ORANGE)
    c.setLineWidth(4)
    c.drawPath(path, fill=0, stroke=1)
    c.setFillColor(TEAL)
    c.circle(328, 285, 7, fill=1, stroke=0)
    text(c, "maximum (2 s, 5 m)", 328, 306, 13, TEAL, "Helvetica-Bold", "center")
    box(c, 646, 130, 244, 162, BLUE_PALE, BLUE)
    text(c, "vertex", 768, 252, 13, BLUE, "Helvetica-Bold", "center")
    text(c, "(2, 5)", 768, 210, 28, NAVY, "Helvetica-Bold", "center")
    text(c, "highest after 2 s", 768, 164, 15, NAVY, "Helvetica-Bold", "center")


# Circumference and pi


def circle_page_1(c):
    page_frame(c, "Circumference and pi", "How far around a circle?", "Circle parts", 1)
    heading(c, "Start at the centre", "Radius, Diameter, Circumference", "The radius runs from the centre to the edge. The diameter crosses the entire circle and is exactly twice the radius.")
    c.setFillColor(BLUE_PALE)
    c.setStrokeColor(BLUE)
    c.setLineWidth(4)
    c.circle(300, 190, 125, fill=1, stroke=1)
    c.setFillColor(NAVY)
    c.circle(300, 190, 5, fill=1, stroke=0)
    line(c, 300, 190, 425, 190, TEAL, 4)
    line(c, 175, 160, 425, 160, ORANGE, 4)
    text(c, "radius r", 362, 205, 15, TEAL, "Helvetica-Bold", "center")
    text(c, "diameter d = 2r", 300, 132, 15, ORANGE, "Helvetica-Bold", "center")
    box(c, 570, 124, 310, 178, NAVY, NAVY)
    text(c, "distance around", 725, 263, 13, TEAL_PALE, "Helvetica-Bold", "center")
    text(c, "circumference C", 725, 217, 24, white, "Helvetica-Bold", "center")
    text(c, "d = 2r", 725, 167, 22, ORANGE_PALE, "Helvetica-Bold", "center")
    text(c, "One circle, three useful lengths.", 725, 138, 12, TEAL_PALE, "Helvetica-Bold", "center")


def circle_page_2(c):
    page_frame(c, "Circumference and pi", "How far around a circle?", "Discover pi", 2)
    heading(c, "Measure many circles", "The Same Ratio Appears Every Time", "For every circle, circumference divided by diameter is approximately 3.14159. We call this constant pi.")
    data = [("coin", "2.4 cm", "7.54 cm"), ("plate", "26 cm", "81.68 cm"), ("wheel", "70 cm", "219.91 cm")]
    for index, (name, diameter, circumference) in enumerate(data):
        x = 64 + index * 300
        box(c, x, 128, 260, 178, [TEAL_PALE, ORANGE_PALE, BLUE_PALE][index], [TEAL, ORANGE, BLUE][index])
        text(c, name, x + 130, 273, 14, [TEAL, ORANGE, BLUE][index], "Helvetica-Bold", "center")
        text(c, f"d = {diameter}", x + 24, 224, 16, NAVY, "Helvetica-Bold")
        text(c, f"C = {circumference}", x + 24, 186, 16, NAVY, "Helvetica-Bold")
        text(c, "C / d ~= 3.14", x + 130, 145, 18, [TEAL, ORANGE, BLUE][index], "Helvetica-Bold", "center")
    text(c, "pi = C / d", 480, 76, 25, NAVY, "Helvetica-Bold", "center")


def circle_page_3(c):
    page_frame(c, "Circumference and pi", "How far around a circle?", "The formula", 3)
    heading(c, "Unroll the boundary", "Circumference Is pi Diameters Long", "Imagine cutting the circular boundary and laying it flat. Its length is pi times the diameter.")
    c.setStrokeColor(BLUE)
    c.setLineWidth(6)
    c.circle(215, 203, 92, fill=0, stroke=1)
    arrow(c, 330, 203, 446, 203, TEAL)
    line(c, 505, 203, 855, 203, BLUE, 6)
    for index in range(4):
        x = 505 + index * 100
        line(c, x, 190, x, 216, ORANGE, 2)
    text(c, "circular boundary", 215, 87, 14, BLUE, "Helvetica-Bold", "center")
    text(c, "about 3.14 diameters", 680, 168, 16, ORANGE, "Helvetica-Bold", "center")
    box(c, 250, 62, 460, 70, NAVY, NAVY)
    text(c, "C = pi d = 2 pi r", 480, 85, 27, white, "Helvetica-Bold", "center")


def circle_page_4(c):
    page_frame(c, "Circumference and pi", "How far around a circle?", "Wheel example", 4)
    heading(c, "One turn, one circumference", "How Far Does a Bicycle Wheel Travel?", "A wheel has radius 0.35 m. Each complete revolution moves the bicycle forward by one circumference.")
    c.setFillColor(TEAL_PALE)
    c.setStrokeColor(TEAL)
    c.setLineWidth(5)
    c.circle(240, 184, 105, fill=1, stroke=1)
    line(c, 240, 184, 345, 184, ORANGE, 4)
    text(c, "r = 0.35 m", 292, 201, 14, ORANGE, "Helvetica-Bold", "center")
    arrow(c, 375, 184, 480, 184, BLUE, 3)
    box(c, 532, 116, 340, 200, NAVY, NAVY)
    text(c, "one revolution", 702, 278, 13, TEAL_PALE, "Helvetica-Bold", "center")
    text(c, "C = 2 pi (0.35)", 702, 231, 20, white, "Helvetica-Bold", "center")
    text(c, "C ~= 2.20 m", 702, 188, 25, ORANGE_PALE, "Helvetica-Bold", "center")
    text(c, "10 turns ~= 22.0 m", 702, 141, 18, TEAL_PALE, "Helvetica-Bold", "center")


def circle_page_5(c):
    page_frame(c, "Circumference and pi", "How far around a circle?", "Arc length", 5)
    heading(c, "Take only part of the circle", "A Quarter Circle Uses a Quarter Circumference", "Arc length is the same fraction of the circumference as its central angle is of 360 degrees.")
    c.setFillColor(BLUE_PALE)
    c.setStrokeColor(BLUE)
    c.setLineWidth(2)
    c.wedge(90, 82, 390, 382, 0, 90, fill=1, stroke=1)
    c.setStrokeColor(ORANGE)
    c.setLineWidth(7)
    c.arc(90, 82, 390, 382, 0, 90)
    text(c, "90 degrees", 235, 183, 18, NAVY, "Helvetica-Bold", "center")
    box(c, 505, 112, 370, 210, PALE, LINE)
    text(c, "arc fraction", 690, 280, 13, MUTED, "Helvetica-Bold", "center")
    text(c, "90 / 360 = 1 / 4", 690, 238, 21, NAVY, "Helvetica-Bold", "center")
    text(c, "arc = (1/4)(2 pi r)", 690, 190, 21, BLUE, "Helvetica-Bold", "center")
    text(c, "arc = pi r / 2", 690, 142, 25, ORANGE, "Helvetica-Bold", "center")


# Two-dice probability


def die(c, x, y, value, size=52, fill=white, stroke=NAVY):
    box(c, x, y, size, size, fill, stroke, 8, 1.7)
    positions = {
        1: [(0.5, 0.5)],
        2: [(0.28, 0.72), (0.72, 0.28)],
        3: [(0.28, 0.72), (0.5, 0.5), (0.72, 0.28)],
        4: [(0.28, 0.72), (0.72, 0.72), (0.28, 0.28), (0.72, 0.28)],
        5: [(0.28, 0.72), (0.72, 0.72), (0.5, 0.5), (0.28, 0.28), (0.72, 0.28)],
        6: [(0.28, 0.74), (0.72, 0.74), (0.28, 0.5), (0.72, 0.5), (0.28, 0.26), (0.72, 0.26)],
    }
    c.setFillColor(stroke)
    for px, py in positions[value]:
        c.circle(x + px * size, y + py * size, size * 0.055, fill=1, stroke=0)


def dice_1(c):
    page_frame(c, "Two-dice probability", "Why some sums appear more often", "Sample space", 1)
    heading(c, "Roll two fair dice", "There Are 36 Equally Likely Outcomes", "The first die has six possibilities and the second die has six possibilities. An outcome records both values in order.")
    die(c, 145, 176, 2, 82, TEAL_PALE, TEAL)
    text(c, "+", 265, 201, 30, MUTED, "Helvetica-Bold", "center")
    die(c, 305, 176, 5, 82, ORANGE_PALE, ORANGE)
    text(c, "outcome (2, 5)", 266, 136, 16, NAVY, "Helvetica-Bold", "center")
    box(c, 520, 136, 350, 182, NAVY, NAVY)
    text(c, "6 choices x 6 choices", 695, 268, 17, TEAL_PALE, "Helvetica-Bold", "center")
    text(c, "= 36 outcomes", 695, 218, 28, white, "Helvetica-Bold", "center")
    text(c, "Each ordered pair has probability 1/36.", 695, 168, 14, ORANGE_PALE, "Helvetica-Bold", "center")


def dice_2(c):
    page_frame(c, "Two-dice probability", "Why some sums appear more often", "Sum table", 2)
    heading(c, "Add each ordered pair", "Build the Table of Possible Sums", "Rows show the first die and columns show the second. The diagonals of the table collect equal sums.")
    x0, y0, cell = 270, 25, 40
    for value in range(1, 7):
        text(c, str(value), x0 + value * cell + cell / 2, y0 + 7 * cell + 15, 13, TEAL, "Helvetica-Bold", "center")
        text(c, str(value), x0 + cell / 2, y0 + (6 - value + 1) * cell + 15, 13, ORANGE, "Helvetica-Bold", "center")
    for row in range(1, 7):
        for col in range(1, 7):
            total = row + col
            x = x0 + col * cell
            y = y0 + (6 - row + 1) * cell
            fill = YELLOW_PALE if total == 7 else PALE
            stroke = YELLOW if total == 7 else LINE
            box(c, x, y, cell - 2, cell - 2, fill, stroke, 3, 1)
            text(c, str(total), x + (cell - 2) / 2, y + 14, 13, ORANGE if total == 7 else INK, "Helvetica-Bold", "center")
    text(c, "first die", 180, 225, 13, ORANGE, "Helvetica-Bold", "center")
    text(c, "second die", 430, 420, 13, TEAL, "Helvetica-Bold", "center")
    box(c, 650, 122, 230, 190, YELLOW_PALE, YELLOW)
    text(c, "highlighted diagonal", 765, 270, 12, ORANGE, "Helvetica-Bold", "center")
    text(c, "sum = 7", 765, 222, 27, NAVY, "Helvetica-Bold", "center")
    text(c, "6 matching cells", 765, 174, 18, ORANGE, "Helvetica-Bold", "center")
    text(c, "probability = 6/36", 765, 141, 14, NAVY, "Helvetica-Bold", "center")


def dice_3(c):
    page_frame(c, "Two-dice probability", "Why some sums appear more often", "Why seven wins", 3)
    heading(c, "Count the routes", "Six Different Outcomes Produce a Sum of 7", "A sum is more likely when more ordered pairs lead to it. Seven has the largest number of routes.")
    pairs = [(1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1)]
    for index, (left, right) in enumerate(pairs):
        x = 66 + (index % 3) * 300
        y = 226 - (index // 3) * 115
        box(c, x, y, 260, 88, PALE, LINE)
        die(c, x + 26, y + 18, left, 52, TEAL_PALE, TEAL)
        text(c, "+", x + 130, y + 35, 18, MUTED, "Helvetica-Bold", "center")
        die(c, x + 182, y + 18, right, 52, ORANGE_PALE, ORANGE)
        text(c, "= 7", x + 148, y + 35, 16, NAVY, "Helvetica-Bold")
    text(c, "P(sum = 7) = 6 / 36 = 1 / 6", 480, 66, 22, NAVY, "Helvetica-Bold", "center")


def dice_4(c):
    page_frame(c, "Two-dice probability", "Why some sums appear more often", "Distribution", 4)
    heading(c, "A triangular pattern", "Middle Sums Have More Routes", "The number of ways rises from sum 2 to sum 7, then falls symmetrically toward sum 12.")
    counts = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
    chart_x, chart_y = 85, 92
    bar_w = 60
    for index, count in enumerate(counts):
        x = chart_x + index * 72
        height = count * 35
        color = ORANGE if count == 6 else TEAL
        c.setFillColor(ORANGE_PALE if count == 6 else TEAL_PALE)
        c.setStrokeColor(color)
        c.setLineWidth(1.5)
        c.rect(x, chart_y, bar_w, height, fill=1, stroke=1)
        text(c, str(count), x + bar_w / 2, chart_y + height + 10, 12, color, "Helvetica-Bold", "center")
        text(c, str(index + 2), x + bar_w / 2, chart_y - 23, 11, MUTED, "Helvetica-Bold", "center")
    text(c, "sum", 480, 50, 12, MUTED, "Helvetica-Bold", "center")


def dice_5(c):
    page_frame(c, "Two-dice probability", "Why some sums appear more often", "Simple game", 5)
    heading(c, "Use the same counting idea", "What Is the Chance of Rolling 10 or More?", "Winning sums are 10, 11, and 12. Count the ordered pairs for each sum, then combine them.")
    cards = [("sum 10", "3 ways", TEAL_PALE, TEAL), ("sum 11", "2 ways", ORANGE_PALE, ORANGE), ("sum 12", "1 way", BLUE_PALE, BLUE)]
    for index, (label, ways, fill, color) in enumerate(cards):
        x = 72 + index * 296
        box(c, x, 176, 256, 132, fill, color)
        text(c, label, x + 128, 265, 15, color, "Helvetica-Bold", "center")
        text(c, ways, x + 128, 214, 27, NAVY, "Helvetica-Bold", "center")
    box(c, 230, 74, 500, 70, NAVY, NAVY)
    text(c, "P(sum >= 10) = (3 + 2 + 1) / 36 = 1 / 6", 480, 98, 20, white, "Helvetica-Bold", "center")


def build():
    save_pdf(
        OUTPUT_DIR / "similar_triangles_and_shadows_5pages.pdf",
        "Similar Triangles and Shadows",
        [similar_1, similar_2, similar_3, similar_4, similar_5],
    )
    save_pdf(
        OUTPUT_DIR / "quadratic_vertex_form_5pages.pdf",
        "Quadratic Functions and Vertex Form",
        [quadratic_1, quadratic_2, quadratic_3, quadratic_4, quadratic_5],
    )
    save_pdf(
        OUTPUT_DIR / "circumference_and_pi_5pages.pdf",
        "Circumference and Pi",
        [circle_page_1, circle_page_2, circle_page_3, circle_page_4, circle_page_5],
    )
    save_pdf(
        OUTPUT_DIR / "two_dice_probability_5pages.pdf",
        "Two-Dice Probability",
        [dice_1, dice_2, dice_3, dice_4, dice_5],
    )


if __name__ == "__main__":
    build()
