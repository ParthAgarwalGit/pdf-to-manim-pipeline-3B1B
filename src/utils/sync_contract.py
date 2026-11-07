"""Inject and lint the narration-sync contract in generated scene code.

``inject_sync_contract`` writes the measured timing track into the scene
class (``BEAT_TIMES``, ``NARRATION_END``) and guarantees ``construct`` ends
with ``sync_to_end(self)``. It is deterministic and idempotent, so it can
also re-time persisted code after the narration changes, with no LLM call.

``lint_sync_calls`` checks that every beat has a ``sync_to`` call, in order.
"""

import ast
from typing import Dict, List, Optional

_SCENE_BASES = ("Scene", "ThreeDScene", "MovingCameraScene")
_CONTRACT_NAMES = ("BEAT_TIMES", "NARRATION_END")
SYNC_IMPORT = "from src.utils.narration_sync import sync_to, sync_to_end"


def _scene_class(tree: ast.Module, class_name: Optional[str]) -> Optional[ast.ClassDef]:
    fallback = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if class_name and node.name == class_name:
            return node
        if any(isinstance(b, ast.Name) and b.id in _SCENE_BASES for b in node.bases):
            fallback = fallback or node
    return fallback


def _construct(cls: ast.ClassDef) -> Optional[ast.FunctionDef]:
    return next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "construct"),
        None,
    )


def _is_call_to(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == name
    )


def inject_sync_contract(
    code: str,
    beat_times: Dict[str, float],
    narration_end: Optional[float],
    class_name: Optional[str] = None,
) -> str:
    """Return ``code`` with the timing track injected (replacing any old one).

    Unparseable code, or code without a Scene class, is returned unchanged.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    cls = _scene_class(tree, class_name)
    if cls is None or not cls.body:
        return code
    lines = code.splitlines(keepends=True)
    newline = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"

    # Drop an earlier contract (re-injection after the narration changed).
    stale = [
        node for node in cls.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in _CONTRACT_NAMES for t in node.targets)
    ]
    for node in sorted(stale, key=lambda n: n.lineno, reverse=True):
        del lines[node.lineno - 1:node.end_lineno]
    if stale:
        return inject_sync_contract("".join(lines), beat_times, narration_end, class_name)

    construct = _construct(cls)
    if construct is not None and not _is_call_to(construct.body[-1], "sync_to_end"):
        indent = " " * construct.body[0].col_offset
        at = construct.body[-1].end_lineno
        if not lines[at - 1].endswith(("\n", "\r")):
            lines[at - 1] += newline
        lines.insert(at, f"{indent}sync_to_end(self){newline}")

    first = cls.body[0]
    body_start = first.lineno - 1
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        body_start = first.end_lineno  # keep the class docstring first
    indent = " " * first.col_offset
    times = ", ".join(f"{beat!r}: {round(float(t), 3)}" for beat, t in beat_times.items())
    contract = [f"{indent}BEAT_TIMES = {{{times}}}{newline}"]
    if narration_end is not None:
        contract.append(f"{indent}NARRATION_END = {round(float(narration_end), 3)}{newline}")
    lines[body_start:body_start] = contract

    new_code = "".join(lines)
    try:
        ast.parse(new_code)
    except SyntaxError:
        return code
    return new_code


def lint_sync_calls(code: str, beat_ids: List[str], class_name: Optional[str] = None) -> List[str]:
    """Problems with the scene's sync calls; empty when the contract holds."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["Code does not parse."]
    cls = _scene_class(tree, class_name)
    construct = _construct(cls) if cls else None
    if construct is None:
        return ["No construct() method found."]
    called: List[str] = []
    ends = False
    for node in ast.walk(construct):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id == "sync_to_end":
            ends = True
        elif (
            node.func.id == "sync_to"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            called.append((node.lineno, node.col_offset, node.args[1].value))
    order = [beat for _, _, beat in sorted(called)]
    issues: List[str] = []
    missing = [b for b in beat_ids if b not in order]
    if missing:
        issues.append(f"No sync_to call for beat(s): {', '.join(missing)}.")
    unknown = [b for b in order if b not in beat_ids]
    if unknown:
        issues.append(f"sync_to names unknown beat(s): {', '.join(unknown)}.")
    known = [b for b in order if b in beat_ids]
    if known != sorted(known, key=beat_ids.index):
        issues.append("sync_to calls are out of narration order.")
    if not ends:
        issues.append("construct() does not end with sync_to_end(self).")
    return issues
