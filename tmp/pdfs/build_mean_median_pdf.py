from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


PAGE_W = 960
PAGE_H = 540
OUTPUT = Path("output/pdf/mean_median_outliers_5pages.pdf")

NAVY = HexColor("#132238")
INK = HexColor("#1E293B")
MUTED = HexColor("#64748B")
PALE = HexColor("#F4F7FA")
LINE = HexColor("#D8E1EA")
TEAL = HexColor("#0F8B8D")
TEAL_PALE = HexColor("#DDF4F2")
ORANGE = HexColor("#F28C28")
ORANGE_PALE = HexColor("#FFF0DE")
RED = HexColor("#D94A4A")
BLUE = HexColor("#2E6FBB")


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
    return y - len(lines) * leading


def rounded_box(c, x, y, width, height, fill, stroke=LINE, radius=12, line_width=1):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(line_width)
    c.roundRect(x, y, width, height, radius, fill=1, stroke=1)


def page_frame(c, page_number, section):
    c.setFillColor(white)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.rect(0, PAGE_H - 8, PAGE_W, 8, fill=1, stroke=0)
    text(c, "MEAN, MEDIAN & OUTLIERS", 54, PAGE_H - 38, 10, MUTED, "Helvetica-Bold")
    text(c, section.upper(), PAGE_W - 54, PAGE_H - 38, 10, MUTED, "Helvetica-Bold", "right")
    c.setStrokeColor(LINE)
    c.line(54, 36, PAGE_W - 54, 36)
    text(c, "A visual guide to choosing a useful average", 54, 18, 9, MUTED)
    text(c, f"{page_number} / 5", PAGE_W - 54, 18, 9, MUTED, align="right")


def title_block(c, eyebrow, title_value, subtitle):
    text(c, eyebrow.upper(), 62, 438, 12, TEAL, "Helvetica-Bold")
    text(c, title_value, 62, 392, 34, NAVY, "Helvetica-Bold")
    wrapped(c, subtitle, 62, 357, 820, 16, 22, MUTED)


def number_chip(c, value, x, y, fill=PALE, color=INK, width=88, height=66):
    rounded_box(c, x, y, width, height, fill, stroke=fill, radius=10)
    text(c, str(value), x + width / 2, y + 21, 25, color, "Helvetica-Bold", "center")


def draw_arrow(c, x1, y1, x2, y2, color=TEAL, width=3):
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)
    direction = 1 if x2 >= x1 else -1
    c.line(x2, y2, x2 - direction * 10, y2 + 6)
    c.line(x2, y2, x2 - direction * 10, y2 - 6)


def page_one(c):
    page_frame(c, 1, "The question")
    title_block(
        c,
        "A familiar word with two meanings",
        "When the Average Lies",
        "Mean and median both describe a center, but one unusual value can make them tell very different stories.",
    )
    text(c, "Attendance at five seminars", 62, 291, 14, INK, "Helvetica-Bold")
    values = [62, 68, 70, 72, 74]
    for index, value in enumerate(values):
        number_chip(c, value, 62 + index * 106, 197, TEAL_PALE if value == 70 else PALE,
                    TEAL if value == 70 else INK)
    rounded_box(c, 630, 168, 268, 122, ORANGE_PALE, stroke=ORANGE, radius=12, line_width=1.5)
    text(c, "The practical question", 654, 255, 12, ORANGE, "Helvetica-Bold")
    wrapped(c, "What number best represents a typical seminar?", 654, 226, 220, 18, 23, NAVY,
            "Helvetica-Bold")
    text(c, "The answer depends on the shape of the data.", 62, 132, 16, NAVY, "Helvetica-Bold")
    text(c, "First, compare the two most common measures of center.", 62, 105, 13, MUTED)


def page_two(c):
    page_frame(c, 2, "The mean")
    title_block(
        c,
        "Method 1",
        "Mean: Share the Total Equally",
        "Add every value, then divide by how many values there are. The mean uses every observation.",
    )
    rounded_box(c, 62, 230, 836, 92, PALE, stroke=LINE)
    text(c, "62 + 68 + 70 + 72 + 74", 280, 276, 22, NAVY, "Helvetica-Bold", "center")
    text(c, "=", 448, 276, 24, MUTED, "Helvetica-Bold", "center")
    text(c, "346", 532, 276, 22, NAVY, "Helvetica-Bold", "center")
    c.setStrokeColor(NAVY)
    c.setLineWidth(1.5)
    c.line(510, 263, 554, 263)
    text(c, "5", 532, 239, 18, NAVY, "Helvetica-Bold", "center")
    text(c, "= 69.2", 675, 263, 30, TEAL, "Helvetica-Bold", "center")
    text(c, "A balance point on the number line", 62, 189, 13, INK, "Helvetica-Bold")
    c.setStrokeColor(LINE)
    c.setLineWidth(4)
    c.line(100, 125, 820, 125)
    for value in [60, 65, 70, 75]:
        x = 100 + (value - 60) / 15 * 720
        c.setStrokeColor(MUTED)
        c.setLineWidth(1)
        c.line(x, 116, x, 134)
        text(c, str(value), x, 94, 10, MUTED, align="center")
    mean_x = 100 + (69.2 - 60) / 15 * 720
    c.setFillColor(TEAL)
    c.circle(mean_x, 125, 10, fill=1, stroke=0)
    text(c, "mean = 69.2", mean_x, 153, 13, TEAL, "Helvetica-Bold", "center")
    text(c, "Strength: every value contributes.", 62, 67, 12, BLUE, "Helvetica-Bold")
    text(c, "Risk: one extreme value can pull the balance point.", 320, 67, 12, RED, "Helvetica-Bold")


def page_three(c):
    page_frame(c, 3, "The median")
    title_block(
        c,
        "Method 2",
        "Median: Find the Middle Position",
        "Sort the values from smallest to largest. The median is the middle value, so distance from the center matters less.",
    )
    text(c, "Step 1: sort", 62, 294, 13, INK, "Helvetica-Bold")
    values = [62, 68, 70, 72, 74]
    for index, value in enumerate(values):
        fill = TEAL_PALE if index == 2 else PALE
        color = TEAL if index == 2 else INK
        number_chip(c, value, 170 + index * 124, 207, fill, color, width=102, height=74)
    draw_arrow(c, 469, 187, 469, 154, TEAL, 3)
    text(c, "third value = median = 70", 469, 124, 19, TEAL, "Helvetica-Bold", "center")
    rounded_box(c, 62, 62, 836, 52, PALE, stroke=LINE, radius=8)
    text(c, "With five values, two lie below 70 and two lie above 70.", 480, 80, 14, NAVY,
         "Helvetica-Bold", "center")


def page_four(c):
    page_frame(c, 4, "The outlier")
    title_block(
        c,
        "Change one value",
        "One Exceptional Event Changes the Story",
        "Suppose the final seminar is unusually popular: attendance rises from 74 to 160 while the other four values stay fixed.",
    )
    values = [62, 68, 70, 72, 160]
    for index, value in enumerate(values):
        fill = ORANGE_PALE if value == 160 else PALE
        color = ORANGE if value == 160 else INK
        number_chip(c, value, 62 + index * 106, 230, fill, color)
    text(c, "outlier", 62 + 4 * 106 + 44, 210, 11, ORANGE, "Helvetica-Bold", "center")
    rounded_box(c, 628, 215, 270, 96, NAVY, stroke=NAVY)
    text(c, "Same four ordinary events", 763, 270, 13, white, "Helvetica-Bold", "center")
    text(c, "One very different event", 763, 240, 13, ORANGE, "Helvetica-Bold", "center")
    rounded_box(c, 62, 89, 382, 88, TEAL_PALE, stroke=TEAL)
    text(c, "Median", 88, 140, 12, TEAL, "Helvetica-Bold")
    text(c, "70", 410, 119, 32, TEAL, "Helvetica-Bold", "right")
    text(c, "unchanged", 88, 110, 13, MUTED)
    rounded_box(c, 516, 89, 382, 88, ORANGE_PALE, stroke=ORANGE)
    text(c, "Mean", 542, 140, 12, ORANGE, "Helvetica-Bold")
    text(c, "86.4", 864, 119, 32, ORANGE, "Helvetica-Bold", "right")
    text(c, "pulled upward by 17.2", 542, 110, 13, MUTED)
    text(c, "The mean answers 'How much in total?' The median answers 'What was typical?'", 480, 59,
         13, NAVY, "Helvetica-Bold", "center")


def page_five(c):
    page_frame(c, 5, "A practical choice")
    title_block(
        c,
        "Worked example",
        "Which Average Should We Report?",
        "Five class sections contain 28, 30, 31, 32, and 79 students. One unusually large section changes the mean.",
    )
    values = [28, 30, 31, 32, 79]
    for index, value in enumerate(values):
        number_chip(c, value, 62 + index * 106, 248, ORANGE_PALE if value == 79 else PALE,
                    ORANGE if value == 79 else INK)
    rounded_box(c, 630, 230, 268, 102, PALE, stroke=LINE)
    text(c, "Mean = 40", 654, 288, 17, ORANGE, "Helvetica-Bold")
    text(c, "Median = 31", 654, 252, 17, TEAL, "Helvetica-Bold")
    rounded_box(c, 62, 96, 392, 100, TEAL_PALE, stroke=TEAL)
    text(c, "Use the median", 86, 157, 14, TEAL, "Helvetica-Bold")
    wrapped(c, "to describe a typical section when the distribution is skewed.", 86, 132, 338, 13,
            18, INK)
    rounded_box(c, 506, 96, 392, 100, ORANGE_PALE, stroke=ORANGE)
    text(c, "Use the mean", 530, 157, 14, ORANGE, "Helvetica-Bold")
    wrapped(c, "for totals, budgeting, and balanced data where every value should count.", 530, 132,
            338, 13, 18, INK)
    text(c, "Best practice: show both when one number could hide the shape of the data.", 480, 64,
         14, NAVY, "Helvetica-Bold", "center")


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("Mean, Median, and Outliers")
    c.setAuthor("PDF-to-Manim Pipeline Demonstration")
    for draw_page in (page_one, page_two, page_three, page_four, page_five):
        draw_page(c)
        c.showPage()
    c.save()
    print(OUTPUT.resolve())


if __name__ == "__main__":
    build()
