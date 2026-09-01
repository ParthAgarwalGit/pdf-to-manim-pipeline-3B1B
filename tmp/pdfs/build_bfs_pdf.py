from __future__ import annotations

from pathlib import Path
from typing import Iterable

from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "bfs_queue_shortest_paths_7pages.pdf"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

W, H = A4
M = 44
INK = HexColor("#172033")
MUTED = HexColor("#5C6678")
NAVY = HexColor("#173B57")
CYAN = HexColor("#2C9FB8")
GREEN = HexColor("#2E8B67")
YELLOW = HexColor("#E9B949")
RED = HexColor("#D9534F")
PURPLE = HexColor("#7453A6")
ORANGE = HexColor("#E47D31")
LINE = HexColor("#D9DEE8")
PAPER = HexColor("#F7F9FC")
PALE_BLUE = HexColor("#E7F5F8")
PALE_GREEN = HexColor("#EAF5EF")
PALE_YELLOW = HexColor("#FFF7DB")
PALE_RED = HexColor("#FCEBEA")
PALE_PURPLE = HexColor("#F1ECF8")


def wrap(text: str, font: str, size: float, width: float) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = word if not current else f"{current} {word}"
        if stringWidth(trial, font, size) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def paragraph(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    size: float = 10,
    color: Color = INK,
    font: str = "Helvetica",
    leading: float | None = None,
) -> float:
    leading = leading or size * 1.36
    c.setFillColor(color)
    c.setFont(font, size)
    for line in wrap(text, font, size, width):
        c.drawString(x, y, line)
        y -= leading
    return y


def bullets(c: canvas.Canvas, items: Iterable[str], x: float, y: float, width: float) -> float:
    for item in items:
        c.setFillColor(CYAN)
        c.circle(x + 3, y + 3, 2.2, fill=1, stroke=0)
        y = paragraph(c, item, x + 14, y, width - 14, size=9.5)
        y -= 6
    return y


def page(c: canvas.Canvas, section: str, title: str, number: int) -> None:
    c.setFillColor(PAPER)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.rect(0, H - 82, W, 82, fill=1, stroke=0)
    c.setFillColor(CYAN)
    c.rect(0, H - 82, 10, 82, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(M, H - 29, section.upper())
    c.setFont("Helvetica-Bold", 22)
    c.drawString(M, H - 59, title)
    c.setStrokeColor(LINE)
    c.line(M, 31, W - M, 31)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(M, 17, "Breadth-First Search - queue, levels, shortest paths, applications")
    c.drawRightString(W - M, 17, f"{number} / 7")


def box(c: canvas.Canvas, x: float, y: float, w: float, h: float, fill: Color = white) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(LINE)
    c.setLineWidth(0.8)
    c.roundRect(x, y, w, h, 7, fill=1, stroke=1)


def heading(c: canvas.Canvas, text: str, x: float, y: float, color: Color = NAVY) -> None:
    c.setFillColor(color)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, text)


def node(c: canvas.Canvas, x: float, y: float, name: str, fill: Color = white, r: float = 17) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(INK)
    c.setLineWidth(1.5)
    c.circle(x, y, r, fill=1, stroke=1)
    c.setFillColor(white if fill == INK else INK)
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(x, y - 3.5, name)


def edge(c: canvas.Canvas, a: tuple[float, float], b: tuple[float, float], color: Color = MUTED) -> None:
    c.setStrokeColor(color)
    c.setLineWidth(1.7)
    c.line(a[0], a[1], b[0], b[1])


def queue(c: canvas.Canvas, values: list[str], x: float, y: float, cell: float = 38) -> None:
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Bold", 8)
    span = max(1, len(values)) * cell
    if len(values) == 1:
        c.drawString(x, y + 27, "F")
        c.drawRightString(x + span, y + 27, "B")
    else:
        c.drawString(x, y + 27, "FRONT")
        c.drawRightString(x + span, y + 27, "BACK")
    if not values:
        c.setStrokeColor(LINE)
        c.rect(x, y, cell * 2, 22, fill=0, stroke=1)
        c.setFillColor(MUTED)
        c.setFont("Helvetica-Oblique", 9)
        c.drawCentredString(x + cell, y + 7, "empty")
        return
    for index, value in enumerate(values):
        fill = PALE_YELLOW if index == 0 else PALE_BLUE
        c.setFillColor(fill)
        c.setStrokeColor(CYAN)
        c.rect(x + index * cell, y, cell, 22, fill=1, stroke=1)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(x + index * cell + cell / 2, y + 7, value)


def graph(c: canvas.Canvas, positions: dict[str, tuple[float, float]], links: list[tuple[str, str]], fills=None) -> None:
    fills = fills or {}
    for left, right in links:
        edge(c, positions[left], positions[right])
    for name, point in positions.items():
        node(c, *point, name, fills.get(name, white))


def page_one(c: canvas.Canvas) -> None:
    page(c, "Core idea", "BFS explores a graph one layer at a time", 1)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 30)
    c.drawString(M, H - 135, "Breadth-First Search")
    paragraph(
        c,
        "BFS starts at a source vertex, visits every vertex at distance 1, then distance 2, "
        "and continues outward. A first-in, first-out queue preserves this layer order.",
        M, H - 162, W - 2 * M, size=12, color=MUTED,
    )
    positions = {
        "S": (W / 2, 505), "A": (185, 425), "B": (W / 2, 425), "C": (410, 425),
        "D": (130, 325), "E": (250, 325), "F": (350, 325), "G": (470, 325),
    }
    links = [("S", "A"), ("S", "B"), ("S", "C"), ("A", "D"), ("A", "E"),
             ("B", "E"), ("B", "F"), ("C", "F"), ("C", "G")]
    fills = {"S": YELLOW, "A": PALE_BLUE, "B": PALE_BLUE, "C": PALE_BLUE,
             "D": PALE_GREEN, "E": PALE_GREEN, "F": PALE_GREEN, "G": PALE_GREEN}
    graph(c, positions, links, fills)
    for y, text, color in [(535, "LEVEL 0", YELLOW), (455, "LEVEL 1", CYAN), (355, "LEVEL 2", GREEN)]:
        c.setStrokeColor(color)
        c.setDash(3, 3)
        c.line(75, y - 20, W - 75, y - 20)
        c.setDash()
        c.setFillColor(color)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(76, y - 14, text)
    box(c, M, 82, W - 2 * M, 175, PALE_BLUE)
    heading(c, "Three facts to remember", M + 16, 232)
    bullets(c, [
        "Queue order: remove the oldest discovered vertex first.",
        "Visit-on-enqueue: mark a vertex when it enters the queue, preventing duplicates.",
        "Shortest-path guarantee: in an unweighted graph, the first discovery gives the fewest edges from S.",
    ], M + 16, 207, W - 2 * M - 32)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(W / 2, 56, "Think ripple, not tunnel: BFS expands an entire frontier before moving deeper.")


def page_two(c: canvas.Canvas) -> None:
    page(c, "Mechanics", "The queue controls the exploration order", 2)
    box(c, M, 455, 250, 275, white)
    heading(c, "Canonical pseudocode", M + 16, 706)
    code = [
        "BFS(G, s):",
        "  for each vertex v:",
        "    visited[v] = false",
        "    dist[v] = infinity",
        "    parent[v] = none",
        "  visited[s] = true",
        "  dist[s] = 0",
        "  enqueue(Q, s)",
        "  while Q is not empty:",
        "    u = dequeue(Q)",
        "    for each neighbor v of u:",
        "      if not visited[v]:",
        "        visited[v] = true",
        "        dist[v] = dist[u] + 1",
        "        parent[v] = u",
        "        enqueue(Q, v)",
    ]
    c.setFont("Courier", 8.6)
    c.setFillColor(INK)
    y = 682
    for line in code:
        c.drawString(M + 17, y, line)
        y -= 14
    box(c, 315, 455, W - 315 - M, 275, PALE_YELLOW)
    heading(c, "Why mark on enqueue?", 331, 706, ORANGE)
    paragraph(c, "Suppose A and B both point to C. If C is marked only when dequeued, both A and B may enqueue C. Marking C immediately reserves it for its first, shortest discovery.", 331, 680, W - 315 - M - 32, size=9.5)
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(331, 592, "Wrong")
    queue(c, ["C", "C"], 331, 550, 45)
    c.setFillColor(GREEN)
    c.drawString(331, 507, "Correct")
    queue(c, ["C"], 331, 465, 45)
    box(c, M, 85, W - 2 * M, 340, white)
    heading(c, "Color states and their meaning", M + 16, 398)
    states = [
        ("WHITE", white, "Undiscovered", "No queue entry and no known path yet."),
        ("GRAY", YELLOW, "Discovered", "In the queue or waiting to be processed."),
        ("BLACK", INK, "Finished", "Dequeued; all outgoing neighbors were examined."),
    ]
    y = 348
    for name, fill, title, description in states:
        node(c, M + 42, y, name[0], fill, 16)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(M + 72, y + 8, f"{name}: {title}")
        paragraph(c, description, M + 72, y - 8, W - 2 * M - 90, size=9, color=MUTED)
        y -= 76
    c.setFillColor(PALE_PURPLE)
    c.roundRect(M + 16, 106, W - 2 * M - 32, 57, 6, fill=1, stroke=0)
    c.setFillColor(PURPLE)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(M + 29, 142, "Invariant")
    paragraph(c, "The queue contains discovered vertices in nondecreasing distance from the source.", M + 29, 124, W - 2 * M - 58, size=9.5)


def page_three(c: canvas.Canvas) -> None:
    page(c, "Trace", "Worked traversal: every queue transition", 3)
    positions = {"A": (105, 640), "B": (210, 690), "C": (210, 590), "D": (330, 700),
                 "E": (330, 620), "F": (330, 540), "G": (455, 620)}
    links = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"),
             ("C", "E"), ("C", "F"), ("E", "G"), ("F", "G")]
    graph(c, positions, links, {"A": YELLOW})
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9)
    c.drawRightString(W - M, 705, "Assume neighbors are read alphabetically.")
    rows = [
        ("0", "Start", "[A]", "A", "0", "none"),
        ("1", "Process A", "[B, C]", "B, C", "1", "A"),
        ("2", "Process B", "[C, D, E]", "D, E", "2", "B"),
        ("3", "Process C", "[D, E, F]", "F", "2", "C"),
        ("4", "Process D", "[E, F]", "-", "-", "-"),
        ("5", "Process E", "[F, G]", "G", "3", "E"),
        ("6", "Process F", "[G]", "-", "-", "-"),
        ("7", "Process G", "[]", "-", "-", "-"),
    ]
    x0, y0 = M, 480
    widths = [34, 82, 118, 85, 50, 70]
    headers = ["Step", "Action", "Queue after step", "New vertices", "Distance", "Parent"]
    c.setFillColor(NAVY)
    c.rect(x0, y0, sum(widths), 28, fill=1, stroke=0)
    x = x0
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 8)
    for header, width in zip(headers, widths):
        c.drawString(x + 5, y0 + 10, header)
        x += width
    y = y0 - 28
    for index, row in enumerate(rows):
        c.setFillColor(white if index % 2 == 0 else PALE_BLUE)
        c.rect(x0, y, sum(widths), 28, fill=1, stroke=0)
        x = x0
        c.setFillColor(INK)
        c.setFont("Helvetica", 8.5)
        for value, width in zip(row, widths):
            c.drawString(x + 5, y + 10, value)
            x += width
        y -= 28
    box(c, M, 82, W - 2 * M, 135, PALE_GREEN)
    heading(c, "Result", M + 16, 192, GREEN)
    paragraph(c, "Traversal order: A, B, C, D, E, F, G. Distances are A:0, B:1, C:1, D:2, E:2, F:2, G:3.", M + 16, 167, W - 2 * M - 32, size=10)
    paragraph(c, "The parent links form a BFS tree. One shortest route from A to G is A -> B -> E -> G. Because G was first discovered at distance 3, no route using fewer than 3 edges exists.", M + 16, 130, W - 2 * M - 32, size=10)


def page_four(c: canvas.Canvas) -> None:
    page(c, "Guarantee", "Why BFS finds shortest unweighted paths", 4)
    box(c, M, 565, W - 2 * M, 165, PALE_PURPLE)
    heading(c, "Layer invariant", M + 18, 702, PURPLE)
    paragraph(c, "When BFS dequeues a vertex u with distance d, every vertex already removed has distance at most d, and every queued vertex has distance d or d + 1.", M + 18, 675, W - 2 * M - 36, size=11)
    c.setFillColor(PURPLE)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(W / 2, 605, "dist[v] = dist[u] + 1 when v is first discovered from u")
    box(c, M, 335, W - 2 * M, 200, white)
    heading(c, "Proof sketch by induction on layers", M + 18, 507)
    bullets(c, [
        "Base: the source s has distance 0, which is optimal.",
        "Inductive step: assume all vertices through layer d have correct distances.",
        "A new vertex v discovered from layer d receives distance d + 1, so a path of that length exists.",
        "If a shorter path to v existed, its predecessor would lie in an earlier layer and would have discovered v first. Contradiction.",
    ], M + 18, 480, W - 2 * M - 36)
    box(c, M, 82, 244, 220, PALE_YELLOW)
    heading(c, "When the guarantee applies", M + 16, 275, ORANGE)
    bullets(c, [
        "Unweighted graphs.",
        "Or every edge has equal cost.",
        "Directed and undirected graphs both work.",
    ], M + 16, 247, 212)
    box(c, 306, 82, W - 306 - M, 220, PALE_RED)
    heading(c, "When BFS is not enough", 322, 275, RED)
    paragraph(c, "If edges have different positive weights, fewest edges may not mean minimum total cost.", 322, 247, W - 306 - M - 32, size=9.5)
    c.setFillColor(INK)
    c.setFont("Courier-Bold", 9.5)
    c.drawString(322, 187, "S --10--> A --1--> T")
    c.drawString(322, 158, "S --2--> B --2--> C --2--> T")
    paragraph(c, "BFS chooses the 2-edge route through A (cost 11), but Dijkstra chooses the 3-edge route through B and C (cost 6).", 322, 126, W - 306 - M - 32, size=9.2)


def page_five(c: canvas.Canvas) -> None:
    page(c, "Application", "Shortest path through a maze", 5)
    paragraph(c, "Treat each open cell as a vertex. Connect cells that share an edge. BFS finds the minimum number of moves from S to T.", M, H - 112, W - 2 * M, size=10.5, color=MUTED)
    grid = [
        "S..#...",
        ".#.#.#.",
        ".#...#.",
        ".###.#.",
        ".....#T",
        ".#####.",
        ".......",
    ]
    path_cells = {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2),
                  (4, 3), (4, 4), (5, 0), (6, 0), (6, 1), (6, 2), (6, 3),
                  (6, 4), (6, 5), (6, 6), (5, 6), (4, 6)}
    gx, gy, cell = M + 14, 365, 42
    for row, values in enumerate(grid):
        for col, value in enumerate(values):
            x, y = gx + col * cell, gy + (6 - row) * cell
            fill = INK if value == "#" else PALE_GREEN if (row, col) in path_cells else white
            if value in {"S", "T"}:
                fill = YELLOW if value == "S" else ORANGE
            c.setFillColor(fill)
            c.setStrokeColor(LINE)
            c.rect(x, y, cell, cell, fill=1, stroke=1)
            if value in {"S", "T"}:
                c.setFillColor(INK)
                c.setFont("Helvetica-Bold", 12)
                c.drawCentredString(x + cell / 2, y + 14, value)
    box(c, 370, 365, W - 370 - M, 294, white)
    heading(c, "State representation", 386, 632)
    paragraph(c, "Vertex: (row, column)", 386, 606, 150, size=9.5)
    paragraph(c, "Neighbors: up, down, left, right", 386, 584, 150, size=9.5)
    paragraph(c, "Blocked cells are never enqueued.", 386, 548, 150, size=9.5)
    heading(c, "Queue frontier", 386, 505, CYAN)
    queue(c, ["(4,4)", "(6,2)"], 386, 462, 58)
    paragraph(c, "Each cell is discovered once. parent[next] = current records how to reconstruct the final route.", 386, 420, 150, size=9.2)
    box(c, M, 82, W - 2 * M, 245, PALE_BLUE)
    heading(c, "Worked result", M + 16, 300)
    bullets(c, [
        "The green cells show one shortest route from S to T.",
        "The target is reached after 18 moves, so dist[T] = 18.",
        "To recover the route, follow parent pointers backward from T to S, then reverse the sequence.",
        "Stopping when T is first discovered is safe because the queue processes cells by increasing distance.",
    ], M + 16, 273, W - 2 * M - 32)


def page_six(c: canvas.Canvas) -> None:
    page(c, "Performance", "Complexity, variants, and common mistakes", 6)
    box(c, M, 580, W - 2 * M, 150, PALE_GREEN)
    heading(c, "Time and space with adjacency lists", M + 18, 702, GREEN)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(M + 18, 656, "Time: O(V + E)")
    c.drawString(310, 656, "Space: O(V)")
    paragraph(c, "Every vertex is enqueued at most once. Every adjacency-list edge entry is examined once; in an undirected graph each edge appears twice, which changes only the constant factor.", M + 18, 620, W - 2 * M - 36, size=9.5)
    cards = [
        (M, 390, "Connected components", "Start BFS from each still-white vertex. Each run reveals one component.", CYAN, PALE_BLUE),
        (306, 390, "Multi-source BFS", "Enqueue all sources at distance 0. The result is distance to the nearest source.", GREEN, PALE_GREEN),
        (M, 205, "Bipartite testing", "Color each new neighbor opposite its parent. A same-color edge proves an odd cycle.", PURPLE, PALE_PURPLE),
        (306, 205, "Implicit graphs", "Generate neighbors on demand for mazes, puzzles, word ladders, and game states.", ORANGE, PALE_YELLOW),
    ]
    for x, y, title, description, color, fill in cards:
        box(c, x, y, 244, 155, fill)
        heading(c, title, x + 16, y + 126, color)
        paragraph(c, description, x + 16, y + 96, 212, size=9.5)
    box(c, M, 82, W - 2 * M, 90, PALE_RED)
    heading(c, "Common mistakes", M + 16, 145, RED)
    paragraph(c, "Marking too late causes duplicate queue entries. Forgetting parent loses the path. Using BFS on weighted edges gives the wrong optimization target. Reusing visited across separate queries can silently skip valid vertices.", M + 16, 119, W - 2 * M - 32, size=9.3)


def page_seven(c: canvas.Canvas) -> None:
    page(c, "Challenge", "Solve and verify a shortest-route problem", 7)
    paragraph(c, "Question: Starting at S, find a shortest path to T. Read neighbors alphabetically. Show the queue states, distances, and parent chain.", M, H - 112, W - 2 * M, size=10.5, color=MUTED)
    positions = {"S": (95, 610), "A": (190, 680), "B": (190, 545), "C": (300, 700),
                 "D": (300, 610), "E": (300, 520), "F": (415, 655), "G": (415, 545),
                 "T": (505, 610)}
    links = [("S", "A"), ("S", "B"), ("A", "C"), ("A", "D"), ("B", "D"),
             ("B", "E"), ("C", "F"), ("D", "F"), ("D", "G"), ("E", "G"),
             ("F", "T"), ("G", "T")]
    graph(c, positions, links, {"S": YELLOW, "T": ORANGE})
    box(c, M, 315, W - 2 * M, 175, white)
    heading(c, "Queue trace", M + 16, 463)
    traces = [
        ("Start", "[S]"), ("after S", "[A, B]"), ("after A", "[B, C, D]"),
        ("after B", "[C, D, E]"), ("after C", "[D, E, F]"),
        ("after D", "[E, F, G]"), ("after E", "[F, G]"),
        ("after F", "[G, T]"),
    ]
    x, y = M + 16, 430
    for index, (label_text, values) in enumerate(traces):
        col = index % 2
        row = index // 2
        xx = x + col * 250
        yy = y - row * 33
        c.setFillColor(MUTED)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(xx, yy, label_text)
        c.setFillColor(CYAN if "T" not in values else GREEN)
        c.setFont("Courier-Bold", 9)
        c.drawString(xx + 72, yy, values)
    box(c, M, 82, W - 2 * M, 200, PALE_GREEN)
    heading(c, "Solution", M + 16, 255, GREEN)
    bullets(c, [
        "Distances: S:0; A,B:1; C,D,E:2; F,G:3; T:4.",
        "First discovery of T occurs from F, so parent[T] = F.",
        "Parent chain backward: T <- F <- C <- A <- S.",
        "Reverse it to obtain a shortest path: S -> A -> C -> F -> T.",
        "Another 4-edge route may exist, but no route can use fewer edges because T belongs to level 4.",
    ], M + 16, 227, W - 2 * M - 32)


def build() -> Path:
    c = canvas.Canvas(str(OUTPUT), pagesize=A4, pageCompression=1)
    c.setTitle("Breadth-First Search: Queue, Levels, and Shortest Paths")
    c.setAuthor("pdf2manim instructional fixture")
    for renderer in (page_one, page_two, page_three, page_four, page_five, page_six, page_seven):
        renderer(c)
        c.showPage()
    c.save()
    return OUTPUT


if __name__ == "__main__":
    print(build())
