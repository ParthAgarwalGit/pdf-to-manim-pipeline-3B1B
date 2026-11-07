"""Re-time a generated Manim scene by adjusting its ``self.wait(...)`` holds.

The delivery gate's cheapest repair: when a scene's rendered video is
shorter or longer than its narration, spread the measured difference
across the scene's existing pauses and re-render. No LLM call is needed.

Only waits that run exactly once are adjusted: waits inside loops or
nested helper functions could execute any number of times, so changing
them would move the duration by an unknown amount.
"""

import ast
from typing import List, Optional, Tuple

MAX_SINGLE_WAIT = 4.0   # no single pause grows into dead air
MIN_SINGLE_WAIT = 0.1   # shrinking never removes a pause entirely
_SCENE_BASES = ("Scene", "ThreeDScene", "MovingCameraScene")
_TAIL_COMMENT = "# delivery gate: hold to match narration"


class _WaitCollector(ast.NodeVisitor):
    """Collect single-execution ``self.wait(<number>)`` calls in construct()."""

    def __init__(self) -> None:
        self.calls: List[Tuple[ast.Call, float]] = []

    def visit_For(self, node):        # loop bodies run an unknown number of times
        return

    visit_While = visit_AsyncFor = visit_For

    def visit_FunctionDef(self, node):  # helpers may be called repeatedly
        return

    visit_AsyncFunctionDef = visit_Lambda = visit_FunctionDef

    def visit_Call(self, node: ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "wait"
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
            and node.lineno == node.end_lineno
        ):
            value = _wait_value(node)
            if value is not None:
                self.calls.append((node, value))
        self.generic_visit(node)


def _wait_value(node: ast.Call) -> Optional[float]:
    if not node.args and not node.keywords:
        return 1.0  # Manim's default
    arg = node.args[0] if node.args else next(
        (kw.value for kw in node.keywords if kw.arg == "duration"), None,
    )
    if len(node.args) + len(node.keywords) != 1:
        return None
    if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) and not isinstance(arg.value, bool):
        return float(arg.value)
    return None


def _construct(tree: ast.Module) -> Optional[ast.FunctionDef]:
    fallback = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "construct":
                if any(isinstance(b, ast.Name) and b.id in _SCENE_BASES for b in node.bases):
                    return item
                fallback = fallback or item
    return fallback


def _splice(lines: List[bytes], node: ast.Call, replacement: str) -> None:
    # ast column offsets are UTF-8 byte offsets.
    index = node.lineno - 1
    line = lines[index]
    lines[index] = line[:node.col_offset] + replacement.encode("utf-8") + line[node.end_col_offset:]


def adjust_waits(code: str, delta_seconds: float) -> Tuple[str, float]:
    """Lengthen (positive delta) or shorten (negative) the scene's holds.

    Returns the new code and the delta actually applied. Lengthening never
    fails: what the existing waits cannot absorb becomes one tail wait.
    Shortening stops at ``MIN_SINGLE_WAIT`` per pause.
    """
    delta = round(float(delta_seconds), 2)
    if abs(delta) < 0.05:
        return code, 0.0
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, 0.0
    construct = _construct(tree)
    if construct is None:
        return code, 0.0
    collector = _WaitCollector()
    for statement in construct.body:
        collector.visit(statement)
    calls = collector.calls

    if delta > 0:
        rooms = [max(0.0, MAX_SINGLE_WAIT - value) for _, value in calls]
    else:
        rooms = [max(0.0, value - MIN_SINGLE_WAIT) for _, value in calls]
    capacity = sum(rooms)
    absorbed = min(abs(delta), capacity)
    sign = 1.0 if delta > 0 else -1.0

    lines = code.encode("utf-8").splitlines(keepends=True)
    applied = 0.0
    # Splice right-to-left so earlier offsets on the same line stay valid.
    for (node, value), room in sorted(
        zip(calls, rooms), key=lambda item: (item[0][0].lineno, item[0][0].col_offset), reverse=True,
    ):
        if capacity <= 0 or room <= 0:
            continue
        change = round(sign * absorbed * room / capacity, 2)
        if change == 0:
            continue
        new_value = round(value + change, 2)
        _splice(lines, node, f"self.wait({new_value})")
        applied += new_value - value

    remaining = round(delta - applied, 2)
    if delta > 0 and remaining >= 0.05:
        last = construct.body[-1]
        indent = " " * construct.body[0].col_offset
        tail = f"{indent}self.wait({remaining})  {_TAIL_COMMENT}\n".encode("utf-8")
        insert_at = last.end_lineno
        if insert_at > 0 and not lines[insert_at - 1].endswith(b"\n"):
            lines[insert_at - 1] += b"\n"
        lines.insert(insert_at, tail)
        applied += remaining

    new_code = b"".join(lines).decode("utf-8")
    try:
        ast.parse(new_code)
    except SyntaxError:
        return code, 0.0
    return new_code, round(applied, 2)
