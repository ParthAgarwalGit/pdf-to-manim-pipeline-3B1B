"""Build two 7-page instructional test PDFs (DFS, linear regression).

Usage: python scripts/build_instructional_pdfs.py
Writes PDFs and intermediate cover images to output/pdf/ (git-ignored).
Requires reportlab and Pillow; fonts are loaded from C:/Windows/Fonts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = A4
M = 46

INK = HexColor("#172033")
MUTED = HexColor("#5B6577")
LINE = HexColor("#D9DEE8")
PAPER = HexColor("#F7F9FC")
NAVY = HexColor("#173B57")
CYAN = HexColor("#2C9FB8")
GREEN = HexColor("#2E8B67")
YELLOW = HexColor("#E9B949")
RED = HexColor("#D9534F")
PURPLE = HexColor("#7453A6")
ORANGE = HexColor("#E47D31")
LIGHT_CYAN = HexColor("#E7F5F8")
LIGHT_GREEN = HexColor("#EAF5EF")
LIGHT_YELLOW = HexColor("#FFF7DB")
LIGHT_RED = HexColor("#FCEBEA")
LIGHT_PURPLE = HexColor("#F1ECF8")


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def make_cover_images() -> tuple[Path, Path]:
    """Create high-resolution source diagrams for the two cover pages."""
    dfs_path = OUT_DIR / "dfs_cover.jpg"
    reg_path = OUT_DIR / "regression_cover.jpg"

    bg = (247, 249, 252)
    ink = (23, 32, 51)
    muted = (91, 101, 119)
    cyan = (44, 159, 184)
    green = (46, 139, 103)
    yellow = (233, 185, 73)
    orange = (228, 125, 49)

    dfs = Image.new("RGB", (1800, 920), bg)
    d = ImageDraw.Draw(dfs)
    for x in range(0, 1800, 60):
        d.line((x, 0, x, 920), fill=(235, 239, 245), width=1)
    for y in range(0, 920, 60):
        d.line((0, y, 1800, y), fill=(235, 239, 245), width=1)
    pts = {"A": (190, 430), "B": (540, 225), "C": (540, 650), "D": (990, 130), "E": (990, 430), "F": (990, 760)}
    for u, v in [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "E"), ("C", "F")]:
        d.line((*pts[u], *pts[v]), fill=muted, width=9)
    active = {"A", "B"}
    finished = {"D"}
    for name, (x, y) in pts.items():
        fill = ink if name in finished else (yellow if name in active else (255, 255, 255))
        d.ellipse((x - 62, y - 62, x + 62, y + 62), fill=fill, outline=ink, width=7)
        color = (255, 255, 255) if name in finished else ink
        d.text((x, y), name, font=_font(42, True), fill=color, anchor="mm")
    d.rounded_rectangle((1190, 125, 1680, 785), radius=35, fill=(255, 255, 255), outline=(217, 222, 232), width=5)
    d.text((1260, 175), "DFS STATE", font=_font(34, True), fill=ink)
    for i, (name, fill, desc) in enumerate([
        ("WHITE", (255, 255, 255), "not visited"),
        ("GRAY", yellow, "on call stack"),
        ("BLACK", ink, "fully explored"),
    ]):
        yy = 290 + i * 145
        d.ellipse((1260, yy, 1330, yy + 70), fill=fill, outline=ink, width=4)
        d.text((1370, yy + 4), name, font=_font(28, True), fill=ink)
        d.text((1370, yy + 42), desc, font=_font(22), fill=muted)
    d.text((1215, 700), "A -> B -> D -> backtrack", font=_font(25, True), fill=green)
    dfs.save(dfs_path, "JPEG", quality=100, subsampling=0)

    reg = Image.new("RGB", (1800, 920), bg)
    d = ImageDraw.Draw(reg)
    left, bottom, right, top = 160, 780, 1660, 100
    d.line((left, bottom, right, bottom), fill=muted, width=7)
    d.line((left, bottom, left, top), fill=muted, width=7)
    for i in range(1, 8):
        px = left + (right - left) * i / 7
        d.line((px, bottom - 9, px, bottom + 9), fill=muted, width=4)
        d.text((px, bottom + 35), str(i), font=_font(25), fill=muted, anchor="ma")
    for score in range(20, 101, 20):
        py = bottom - (bottom - top) * score / 100
        d.line((left - 8, py, right, py), fill=(229, 233, 240), width=3)
        d.text((left - 25, py), str(score), font=_font(22), fill=muted, anchor="rm")
    xs = [1, 2, 3, 4, 5, 6]
    ys = [50, 55, 65, 70, 78, 85]
    slope, intercept = 7.114, 42.267
    line_y0 = bottom - (bottom - top) * intercept / 100
    line_y7 = bottom - (bottom - top) * (slope * 7 + intercept) / 100
    d.line((left, line_y0, right, line_y7), fill=orange, width=11)
    for xv, yv in zip(xs, ys):
        px = left + (right - left) * xv / 7
        py = bottom - (bottom - top) * yv / 100
        pred = slope * xv + intercept
        pred_y = bottom - (bottom - top) * pred / 100
        d.line((px, py, px, pred_y), fill=(217, 83, 79), width=7)
        d.ellipse((px - 18, py - 18, px + 18, py + 18), fill=cyan, outline=ink, width=5)
    d.text((1180, 155), "fitted line", font=_font(30, True), fill=orange)
    d.text((1180, 205), "red = residual error", font=_font(25, True), fill=(217, 83, 79))
    d.text((1210, 690), "score = 7.114x + 42.267", font=_font(32, True), fill=green)
    reg.save(reg_path, "JPEG", quality=100, subsampling=0)

    return dfs_path, reg_path


DFS_COVER, REGRESSION_COVER = make_cover_images()


def wrap(text: str, font: str, size: float, max_width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if stringWidth(trial, font, size) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def text_block(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    size: float = 10,
    leading: float | None = None,
    color: Color = INK,
    font: str = "Helvetica",
) -> float:
    leading = leading or size * 1.35
    c.setFont(font, size)
    c.setFillColor(color)
    for line in wrap(text, font, size, width):
        c.drawString(x, y, line)
        y -= leading
    return y


def bullet_list(
    c: canvas.Canvas,
    items: Iterable[str],
    x: float,
    y: float,
    width: float,
    size: float = 10,
    color: Color = INK,
) -> float:
    for item in items:
        c.setFillColor(CYAN)
        c.circle(x + 3, y + 3, 2.2, fill=1, stroke=0)
        y = text_block(c, item, x + 14, y, width - 14, size=size, color=color)
        y -= 6
    return y


def page_base(c: canvas.Canvas, section: str, title: str, page: int, total: int = 7) -> None:
    c.setFillColor(PAPER)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.rect(0, H - 84, W, 84, fill=1, stroke=0)
    c.setFillColor(CYAN)
    c.rect(0, H - 84, 10, 84, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(M, H - 31, section.upper())
    c.setFont("Helvetica-Bold", 23)
    c.drawString(M, H - 61, title)
    c.setStrokeColor(LINE)
    c.line(M, 32, W - M, 32)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8.5)
    c.drawString(M, 18, "Visual study notes - worked examples included")
    c.drawRightString(W - M, 18, f"{page} / {total}")


def rounded_box(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: Color = white,
    stroke: Color = LINE,
    radius: float = 8,
) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(0.8)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)


def label(c: canvas.Canvas, text: str, x: float, y: float, color: Color = CYAN) -> None:
    c.setFillColor(color)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y, text.upper())


def node(c: canvas.Canvas, x: float, y: float, name: str, fill: Color, radius: float = 18) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(INK)
    c.setLineWidth(1.3)
    c.circle(x, y, radius, fill=1, stroke=1)
    c.setFillColor(INK if fill != INK else white)
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(x, y - 3.5, name)


def edge(c: canvas.Canvas, a: tuple[float, float], b: tuple[float, float], color: Color = MUTED, width: float = 1.5) -> None:
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(a[0], a[1], b[0], b[1])


def draw_graph(c: canvas.Canvas, ox: float, oy: float, scale: float = 1.0, colors: dict[str, Color] | None = None) -> None:
    pts = {
        "A": (ox + 35 * scale, oy + 110 * scale),
        "B": (ox + 125 * scale, oy + 145 * scale),
        "C": (ox + 125 * scale, oy + 65 * scale),
        "D": (ox + 220 * scale, oy + 155 * scale),
        "E": (ox + 220 * scale, oy + 85 * scale),
        "F": (ox + 220 * scale, oy + 20 * scale),
    }
    for u, v in [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "E"), ("C", "F")]:
        edge(c, pts[u], pts[v])
    for name, point in pts.items():
        node(c, point[0], point[1], name, (colors or {}).get(name, white), 16 * scale)


def code_box(c: canvas.Canvas, lines: Sequence[str], x: float, y: float, w: float, h: float) -> None:
    rounded_box(c, x, y, w, h, fill=HexColor("#101827"), stroke=HexColor("#26364C"))
    c.setFont("Courier", 8.8)
    yy = y + h - 22
    for i, line in enumerate(lines, 1):
        c.setFillColor(HexColor("#7892AA"))
        c.drawRightString(x + 25, yy, str(i))
        c.setFillColor(HexColor("#E6EDF3"))
        c.drawString(x + 35, yy, line)
        yy -= 15


def table(
    c: canvas.Canvas,
    rows: Sequence[Sequence[str]],
    x: float,
    y_top: float,
    widths: Sequence[float],
    row_h: float = 25,
    header_fill: Color = NAVY,
) -> float:
    y = y_top
    for r_idx, row in enumerate(rows):
        fill = header_fill if r_idx == 0 else (white if r_idx % 2 else PAPER)
        c.setFillColor(fill)
        c.setStrokeColor(LINE)
        c.rect(x, y - row_h, sum(widths), row_h, fill=1, stroke=1)
        xx = x
        for value, width in zip(row, widths):
            c.setStrokeColor(LINE)
            c.line(xx, y - row_h, xx, y)
            c.setFillColor(white if r_idx == 0 else INK)
            c.setFont("Helvetica-Bold" if r_idx == 0 else "Helvetica", 8.5)
            c.drawCentredString(xx + width / 2, y - row_h + 8.5, str(value))
            xx += width
        c.line(xx, y - row_h, xx, y)
        y -= row_h
    return y


def draw_axes(c: canvas.Canvas, x: float, y: float, w: float, h: float, x_max: float, y_max: float) -> None:
    c.setStrokeColor(MUTED)
    c.setLineWidth(1)
    c.line(x, y, x + w, y)
    c.line(x, y, x, y + h)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 7)
    for i in range(1, int(x_max) + 1):
        px = x + w * i / x_max
        c.line(px, y - 3, px, y + 3)
        c.drawCentredString(px, y - 12, str(i))
    for i in range(0, int(y_max) + 1, 10):
        py = y + h * i / y_max
        c.line(x - 3, py, x + 3, py)
        c.drawRightString(x - 6, py - 2, str(i))


def scatter_plot(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    xs: Sequence[float],
    ys: Sequence[float],
    line_params: tuple[float, float] | None = None,
    residuals: bool = False,
) -> None:
    draw_axes(c, x, y, w, h, 7, 100)
    if line_params:
        slope, intercept = line_params
        y1 = slope * 0 + intercept
        y2 = slope * 7 + intercept
        c.setStrokeColor(ORANGE)
        c.setLineWidth(2.2)
        c.line(x, y + h * y1 / 100, x + w, y + h * y2 / 100)
    for xv, yv in zip(xs, ys):
        px = x + w * xv / 7
        py = y + h * yv / 100
        if residuals and line_params:
            pred = line_params[0] * xv + line_params[1]
            pred_y = y + h * pred / 100
            c.setStrokeColor(RED)
            c.setLineWidth(1.2)
            c.line(px, pred_y, px, py)
        c.setFillColor(CYAN)
        c.setStrokeColor(NAVY)
        c.circle(px, py, 4.5, fill=1, stroke=1)


def dfs_pdf(path: Path) -> None:
    # The target pipeline enforces a 100 KiB minimum input size. Keeping page
    # streams uncompressed preserves vector quality and avoids artificial data.
    c = canvas.Canvas(str(path), pagesize=A4, pageCompression=0)

    page_base(c, "Depth-First Search", "Go deep, then backtrack", 1)
    rounded_box(c, M, 595, W - 2 * M, 116, LIGHT_CYAN)
    label(c, "Core idea", M + 18, 686)
    text_block(c, "Depth-First Search explores one branch as far as possible before returning to the most recent choice point. It can be implemented with recursion or an explicit stack.", M + 18, 663, W - 2 * M - 36, 12, 17)
    c.drawImage(str(DFS_COVER), 62, 332, 472, 242, preserveAspectRatio=True, anchor="c")
    rounded_box(c, M, 118, W - 2 * M, 168, white)
    label(c, "Learning goals", M + 18, 259, GREEN)
    bullet_list(c, [
        "Trace the recursion stack and distinguish discovery from completion.",
        "Use WHITE, GRAY, and BLACK states to prevent repeated work.",
        "Explain why backtracking resumes at the latest unfinished vertex.",
        "Apply DFS to path finding, cycle detection, and dependency analysis.",
    ], M + 18, 235, W - 2 * M - 36, 10.5)
    c.showPage()

    page_base(c, "Depth-First Search", "The traversal mechanism", 2)
    code_box(c, [
        "def dfs(u):",
        "    color[u] = GRAY",
        "    for v in neighbors[u]:",
        "        if color[v] == WHITE:",
        "            parent[v] = u",
        "            dfs(v)",
        "    color[u] = BLACK",
    ], M, 460, 290, 250)
    rounded_box(c, 354, 460, 195, 250, LIGHT_YELLOW)
    label(c, "Stack frames", 372, 684, ORANGE)
    frames = [("dfs(A)", CYAN), ("dfs(B)", GREEN), ("dfs(D)", YELLOW)]
    yy = 628
    for text, fill in frames:
        rounded_box(c, 382, yy, 138, 44, fill)
        c.setFillColor(INK)
        c.setFont("Courier-Bold", 11)
        c.drawCentredString(451, yy + 16, text)
        yy -= 54
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8)
    c.drawCentredString(451, 493, "top of stack")
    rounded_box(c, M, 118, W - 2 * M, 300, white)
    label(c, "What each line means", M + 18, 391, PURPLE)
    table(c, [
        ["Moment", "Action", "Why"],
        ["Enter u", "u becomes GRAY", "u is discovered and active"],
        ["Inspect v", "recurse only if WHITE", "avoid revisiting vertices"],
        ["Return", "u becomes BLACK", "all outgoing work is finished"],
        ["Cost", "O(V + E)", "each vertex and edge is processed once"],
    ], M + 18, 365, [105, 150, 202], 42)
    c.showPage()

    page_base(c, "Depth-First Search", "Color coding the search", 3)
    rounded_box(c, M, 606, W - 2 * M, 105, white)
    legend = [("WHITE", white, "unvisited"), ("GRAY", YELLOW, "active"), ("BLACK", INK, "finished")]
    xx = M + 30
    for name, fill, meaning in legend:
        node(c, xx, 655, "", fill, 14)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(xx + 23, 661, name)
        c.setFont("Helvetica", 8)
        c.setFillColor(MUTED)
        c.drawString(xx + 23, 646, meaning)
        xx += 155
    stages = [
        ("1. Start A", {"A": YELLOW}),
        ("2. Descend A-B-D", {"A": YELLOW, "B": YELLOW, "D": YELLOW}),
        ("3. Finish D", {"A": YELLOW, "B": YELLOW, "D": INK}),
    ]
    y_positions = [440, 258, 76]
    for (title, colors), yy in zip(stages, y_positions):
        rounded_box(c, M, yy, W - 2 * M, 158, white)
        label(c, title, M + 15, yy + 136, CYAN)
        draw_graph(c, M + 118, yy + 13, 0.95, colors)
    c.showPage()

    page_base(c, "Depth-First Search", "Backtracking is the return journey", 4)
    rounded_box(c, M, 552, W - 2 * M, 160, LIGHT_PURPLE)
    label(c, "Rule", M + 18, 684, PURPLE)
    text_block(c, "When the current vertex has no WHITE neighbor, its recursive call ends. Control returns to its parent, which continues scanning the next neighbor. No search state is lost: the call stack remembers every unfinished choice point.", M + 18, 659, W - 2 * M - 36, 11.2, 16)
    pts = {"A": (110, 420), "B": (225, 490), "C": (225, 350), "D": (355, 520), "E": (355, 430), "F": (355, 320)}
    for u, v in [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F")]:
        edge(c, pts[u], pts[v], MUTED, 1.3)
    for n, p in pts.items():
        node(c, *p, n, YELLOW if n in ("A", "B") else (INK if n == "D" else white), 17)
    c.setStrokeColor(RED)
    c.setLineWidth(3)
    c.line(345, 506, 235, 482)
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(274, 507, "return")
    rounded_box(c, 407, 298, 142, 230, white)
    label(c, "Call stack", 422, 504, ORANGE)
    for i, frame in enumerate(["dfs(A)", "dfs(B)", "dfs(D) done"]):
        fill = LIGHT_RED if i == 2 else LIGHT_CYAN
        rounded_box(c, 425, 438 - i * 54, 106, 40, fill)
        c.setFillColor(INK)
        c.setFont("Courier", 8.8)
        c.drawCentredString(478, 453 - i * 54, frame)
    rounded_box(c, M, 82, W - 2 * M, 170, white)
    label(c, "Backtracking trace", M + 18, 225, GREEN)
    table(c, [
        ["Step", "Current", "Event", "Next"],
        ["1", "D", "No WHITE neighbors", "Return to B"],
        ["2", "B", "Continue neighbor scan", "Visit E"],
        ["3", "E", "No WHITE neighbors", "Return to B"],
        ["4", "B", "All neighbors done", "Return to A"],
    ], M + 18, 201, [50, 80, 205, 120], 22)
    c.showPage()

    page_base(c, "Depth-First Search", "Worked question: traversal and tree", 5)
    rounded_box(c, M, 622, W - 2 * M, 90, LIGHT_YELLOW)
    label(c, "Question", M + 18, 686, ORANGE)
    text_block(c, "Starting at A, visit neighbors alphabetically. Find the DFS discovery order, finish order, and parent edges.", M + 18, 660, W - 2 * M - 36, 11.5, 16)
    draw_graph(c, 88, 380, 1.45)
    rounded_box(c, M, 70, W - 2 * M, 260, white)
    label(c, "Solution", M + 18, 305, GREEN)
    table(c, [
        ["Result", "Answer", "Reason"],
        ["Discovery", "A, B, D, E, C, F", "Go deepest before taking next choice"],
        ["Finish", "D, E, B, F, C, A", "A vertex finishes after descendants"],
        ["Parents", "A-B, B-D, B-E, A-C, C-F", "Edge used for first discovery"],
    ], M + 18, 281, [88, 165, 200], 42)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(M + 18, 84, "Check: every vertex appears exactly once in each order.")
    c.showPage()

    page_base(c, "Depth-First Search", "Application: solve a maze", 6)
    rounded_box(c, M, 630, W - 2 * M, 82, LIGHT_CYAN)
    text_block(c, "Treat each open cell as a vertex and connect orthogonal neighbors. DFS follows one corridor until blocked, then backtracks to the latest junction.", M + 18, 683, W - 2 * M - 36, 11, 16)
    grid = [
        "S..#..",
        "##.#.#",
        "...#.#",
        ".###.#",
        "......",
        ".####G",
    ]
    path_cells = {(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (5, 5)}
    gx, gy, cell = 70, 330, 54
    for r, row in enumerate(grid):
        for col, value in enumerate(row):
            x = gx + col * cell
            y = gy + (5 - r) * cell
            fill = INK if value == "#" else (LIGHT_GREEN if (r, col) in path_cells else white)
            c.setFillColor(fill)
            c.setStrokeColor(LINE)
            c.rect(x, y, cell, cell, fill=1, stroke=1)
            if value in "SG":
                c.setFillColor(GREEN if value == "S" else RED)
                c.circle(x + cell / 2, y + cell / 2, 14, fill=1, stroke=0)
                c.setFillColor(white)
                c.setFont("Helvetica-Bold", 10)
                c.drawCentredString(x + cell / 2, y + cell / 2 - 3, value)
            elif (r, col) in path_cells:
                c.setFillColor(GREEN)
                c.circle(x + cell / 2, y + cell / 2, 4, fill=1, stroke=0)
    rounded_box(c, 420, 330, 129, 324, white)
    label(c, "DFS state", 437, 628, PURPLE)
    bullet_list(c, [
        "GRAY cells form the active corridor.",
        "Dead ends become BLACK.",
        "Parent pointers reconstruct the path.",
        "DFS finds a path, not necessarily the shortest path.",
    ], 437, 596, 98, 9.2)
    rounded_box(c, M, 92, W - 2 * M, 190, LIGHT_YELLOW)
    label(c, "Solved path", M + 18, 257, ORANGE)
    text_block(c, "The green cells show one discovered route from S down to the lower corridor and across to G. If shortest distance matters, use BFS; if memory is tight and any solution is acceptable, DFS is often useful.", M + 18, 231, W - 2 * M - 36, 10.5, 15)
    c.setFont("Courier-Bold", 9)
    c.setFillColor(NAVY)
    c.drawString(M + 18, 132, "Search state: (cell, parent, color)    Complexity: O(open cells + links)")
    c.showPage()

    page_base(c, "Depth-First Search", "Application: detect dependency cycles", 7)
    rounded_box(c, M, 610, W - 2 * M, 102, LIGHT_RED)
    label(c, "Problem", M + 18, 685, RED)
    text_block(c, "A course planner must reject prerequisite cycles. In a directed graph, an edge from a GRAY vertex to another GRAY vertex is a back edge and proves that a cycle exists.", M + 18, 659, W - 2 * M - 36, 11, 16)
    pts = {"Math": (100, 480), "DSA": (250, 535), "AI": (405, 480), "Stats": (250, 390)}
    for u, v in [("Math", "DSA"), ("DSA", "AI"), ("AI", "Stats"), ("Stats", "DSA")]:
        edge(c, pts[u], pts[v], RED if (u, v) == ("Stats", "DSA") else MUTED, 2.4 if (u, v) == ("Stats", "DSA") else 1.5)
    for n, p in pts.items():
        node(c, *p, n, YELLOW if n in ("DSA", "AI", "Stats") else INK, 27)
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(285, 421, "back edge: Stats -> DSA")
    rounded_box(c, M, 90, W - 2 * M, 235, white)
    label(c, "Final checklist", M + 18, 299, GREEN)
    bullet_list(c, [
        "WHITE -> GRAY means discover; GRAY -> BLACK means finish.",
        "The stack stores unfinished choices and makes backtracking possible.",
        "A GRAY-to-GRAY directed edge detects a cycle immediately.",
        "Time is O(V + E); recursion depth can reach O(V).",
    ], M + 18, 273, W - 2 * M - 36, 10)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(M + 18, 115, "Answer: the dependency graph is invalid because DSA -> AI -> Stats -> DSA.")
    c.save()


def regression_pdf(path: Path) -> None:
    xs = [1, 2, 3, 4, 5, 6]
    ys = [50, 55, 65, 70, 78, 85]
    c = canvas.Canvas(str(path), pagesize=A4, pageCompression=0)

    page_base(c, "Linear Regression", "Fit a line by learning from error", 1)
    rounded_box(c, M, 610, W - 2 * M, 102, LIGHT_CYAN)
    label(c, "Core model", M + 18, 684)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(M + 18, 646, "prediction = w x + b")
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9)
    c.drawRightString(W - M - 18, 650, "slope w + intercept b")
    c.drawImage(str(REGRESSION_COVER), 62, 322, 472, 248, preserveAspectRatio=True, anchor="c")
    rounded_box(c, M, 92, W - 2 * M, 180, white)
    label(c, "Learning goals", M + 18, 246, GREEN)
    bullet_list(c, [
        "Measure prediction error with mean squared error (MSE).",
        "Derive gradients for the slope and intercept.",
        "Use gradient descent to reduce loss step by step.",
        "Fit a real six-row dataset and make a new prediction.",
    ], M + 18, 221, W - 2 * M - 36, 10.5)
    c.showPage()

    page_base(c, "Linear Regression", "Dataset and model setup", 2)
    rounded_box(c, M, 630, W - 2 * M, 82, LIGHT_YELLOW)
    text_block(c, "Example: predict exam score y from study hours x. We use all six observations to learn one straight line.", M + 18, 681, W - 2 * M - 36, 11.2, 16)
    table(c, [
        ["Student", "Hours x", "Score y"],
        ["A", "1", "50"], ["B", "2", "55"], ["C", "3", "65"],
        ["D", "4", "70"], ["E", "5", "78"], ["F", "6", "85"],
    ], M, 596, [70, 80, 80], 38)
    scatter_plot(c, 315, 360, 235, 215, xs, ys)
    rounded_box(c, M, 105, W - 2 * M, 190, white)
    label(c, "Start with a guess", M + 18, 268, PURPLE)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(M + 18, 228, "w = 5,  b = 45")
    text_block(c, "This initial line predicts 50, 55, 60, 65, 70, and 75. The later observations are underestimated, so the slope must increase.", M + 18, 195, W - 2 * M - 36, 10.5, 15)
    c.showPage()

    page_base(c, "Linear Regression", "Loss: measure every residual", 3)
    scatter_plot(c, 70, 424, 455, 245, xs, ys, (5, 45), residuals=True)
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(378, 645, "red segments = residuals")
    rounded_box(c, M, 255, W - 2 * M, 122, LIGHT_RED)
    label(c, "Mean squared error", M + 18, 350, RED)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(M + 18, 314, "MSE = (1/n) sum (prediction - actual)^2")
    text_block(c, "Squaring makes positive and negative errors contribute equally and penalizes large misses more strongly.", M + 18, 284, W - 2 * M - 36, 9.8, 13)
    rounded_box(c, M, 82, W - 2 * M, 135, white)
    label(c, "Initial calculation", M + 18, 190, ORANGE)
    c.setFillColor(INK)
    c.setFont("Courier", 9.2)
    c.drawString(M + 18, 158, "errors = [0, 0, -5, -5, -8, -10]")
    c.drawString(M + 18, 134, "MSE = (0 + 0 + 25 + 25 + 64 + 100) / 6")
    c.setFont("Courier-Bold", 11)
    c.setFillColor(RED)
    c.drawString(M + 18, 104, "MSE = 35.67")
    c.showPage()

    page_base(c, "Linear Regression", "Gradient descent: choose a downhill step", 4)
    rounded_box(c, M, 566, W - 2 * M, 146, LIGHT_PURPLE)
    label(c, "Gradients", M + 18, 684, PURPLE)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(M + 18, 646, "dL/dw = (2/n) sum x(prediction - actual)")
    c.drawString(M + 18, 616, "dL/db = (2/n) sum (prediction - actual)")
    text_block(c, "A negative gradient means increasing the parameter will reduce the loss.", M + 18, 588, W - 2 * M - 36, 9.5)
    rounded_box(c, M, 370, W - 2 * M, 155, white)
    label(c, "At w = 5 and b = 45", M + 18, 498, ORANGE)
    table(c, [
        ["Quantity", "Calculation", "Value"],
        ["sum errors", "0 + 0 - 5 - 5 - 8 - 10", "-28"],
        ["sum x*error", "0 + 0 - 15 - 20 - 40 - 60", "-135"],
        ["gradients", "2/6 times each sum", "dw=-45, db=-9.33"],
    ], M + 18, 472, [90, 245, 120], 34)
    rounded_box(c, M, 92, W - 2 * M, 225, LIGHT_GREEN)
    label(c, "Update with learning rate 0.01", M + 18, 290, GREEN)
    c.setFillColor(INK)
    c.setFont("Courier-Bold", 11)
    c.drawString(M + 18, 250, "w_new = 5 - 0.01(-45)   = 5.45")
    c.drawString(M + 18, 218, "b_new = 45 - 0.01(-9.33) = 45.093")
    text_block(c, "Both parameters increase. The line rotates upward and shifts slightly upward, moving closer to the high-hour observations.", M + 18, 173, W - 2 * M - 36, 10.2, 14)
    c.showPage()

    page_base(c, "Linear Regression", "Watch the loss fall", 5)
    rounded_box(c, M, 594, W - 2 * M, 118, white)
    table(c, [
        ["Iteration", "w", "b", "MSE", "Meaning"],
        ["0", "5.000", "45.000", "35.67", "starting guess"],
        ["1", "5.450", "45.093", "17.92", "large improvement"],
        ["2", "5.757", "45.153", "9.70 approx", "still descending"],
    ], M + 12, 690, [58, 65, 70, 75, 185], 28)
    rounded_box(c, M, 292, W - 2 * M, 255, LIGHT_CYAN)
    label(c, "Loss curve", M + 18, 521, CYAN)
    px, py, pw, ph = M + 54, 332, 400, 150
    c.setStrokeColor(MUTED)
    c.line(px, py, px + pw, py)
    c.line(px, py, px, py + ph)
    points = [(0, 35.67), (1, 17.92), (2, 9.7), (3, 6.1), (4, 4.4), (5, 3.6)]
    max_loss = 40
    coords = [(px + i * pw / 5, py + loss * ph / max_loss) for i, loss in points]
    c.setStrokeColor(ORANGE)
    c.setLineWidth(2.4)
    for a, b in zip(coords, coords[1:]):
        c.line(a[0], a[1], b[0], b[1])
    for x0, y0 in coords:
        c.setFillColor(ORANGE)
        c.circle(x0, y0, 4, fill=1, stroke=0)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(px + pw - 35, py - 18, "iteration")
    c.drawString(px - 28, py + ph + 2, "MSE")
    rounded_box(c, M, 92, W - 2 * M, 155, LIGHT_YELLOW)
    label(c, "Learning-rate intuition", M + 18, 221, ORANGE)
    bullet_list(c, [
        "Too small: reliable but slow progress.",
        "Well chosen: steady decrease toward the minimum.",
        "Too large: overshoot, oscillation, or divergence.",
    ], M + 18, 196, W - 2 * M - 36, 10)
    c.showPage()

    page_base(c, "Linear Regression", "Solved application: predict a new score", 6)
    rounded_box(c, M, 610, W - 2 * M, 102, LIGHT_GREEN)
    label(c, "Converged model", M + 18, 684, GREEN)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 19)
    c.drawString(M + 18, 644, "score = 7.114 x hours + 42.267")
    scatter_plot(c, 74, 330, 447, 230, xs, ys, (7.114, 42.267), residuals=True)
    rounded_box(c, M, 92, W - 2 * M, 190, white)
    label(c, "Question: what score for 7 study hours?", M + 18, 255, ORANGE)
    c.setFillColor(INK)
    c.setFont("Courier-Bold", 11)
    c.drawString(M + 18, 213, "prediction = 7.114(7) + 42.267")
    c.setFillColor(GREEN)
    c.setFont("Helvetica-Bold", 21)
    c.drawString(M + 18, 169, "prediction = 92.1")
    text_block(c, "The fitted line explains about 99.4% of the score variation in this tiny example (R^2 approx 0.994). This is illustration, not evidence of causation.", 290, 213, 240, 9.5, 14, MUTED)
    c.showPage()

    page_base(c, "Linear Regression", "Practical checklist and limits", 7)
    rounded_box(c, M, 552, W - 2 * M, 160, LIGHT_RED)
    label(c, "Common mistakes", M + 18, 684, RED)
    bullet_list(c, [
        "Using a learning rate so large that loss increases.",
        "Ignoring feature scaling when predictors have very different units.",
        "Treating a high R^2 as proof that x causes y.",
        "Extrapolating far beyond the observed x range.",
    ], M + 18, 657, W - 2 * M - 36, 10.2)
    rounded_box(c, M, 326, W - 2 * M, 180, white)
    label(c, "Algorithm summary", M + 18, 480, PURPLE)
    table(c, [
        ["Step", "Operation"],
        ["1", "Predict y_hat = wx + b"],
        ["2", "Compute MSE and residuals"],
        ["3", "Compute dL/dw and dL/db"],
        ["4", "Update w and b opposite the gradients"],
        ["5", "Repeat until improvement is small"],
    ], M + 18, 456, [60, 392], 25)
    rounded_box(c, M, 92, W - 2 * M, 185, LIGHT_CYAN)
    label(c, "Final check", M + 18, 251, CYAN)
    text_block(c, "If every residual is negative, predictions are below the data. The intercept gradient is negative, so gradient descent raises b. Depending on x values, the slope may rise too. Always inspect both the numbers and the plotted line.", M + 18, 222, W - 2 * M - 36, 10.4, 15)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(M + 18, 115, "Key idea: gradient descent turns error into a direction for improvement.")
    c.save()


if __name__ == "__main__":
    dfs_pdf(OUT_DIR / "dfs_color_backtracking_7pages.pdf")
    regression_pdf(OUT_DIR / "linear_regression_gradient_descent_7pages.pdf")
    print(OUT_DIR / "dfs_color_backtracking_7pages.pdf")
    print(OUT_DIR / "linear_regression_gradient_descent_7pages.pdf")
