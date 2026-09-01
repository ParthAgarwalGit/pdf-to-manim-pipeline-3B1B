"""Add V2 observability and narration timing to ordinary generated Manim."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from .models import NarratedScene, VisualContract


INSTRUMENTATION_MARKER = "Instrumented automatically by PDF-to-Manim V2."
_SCENE_BASES = {"Scene", "TrackedScene"}
_INJECTED_METHODS = {"auto_register", "configure_generated_scene"}


@dataclass(frozen=True)
class InstrumentationSummary:
    code: str
    section_timing_applied: bool
    registered_names: tuple[str, ...]


def instrument_scene_code(
    source: str,
    *,
    class_name: str,
    scene: NarratedScene,
    contract: VisualContract,
) -> InstrumentationSummary:
    """Convert one ordinary Scene into a tracked, audio-timed Scene.

    Invalid Python is returned unchanged so the static verifier can report the
    original syntax error and the normal repair path can handle it.
    """

    clean = deinstrument_scene_code(source)
    try:
        tree = ast.parse(clean)
    except SyntaxError:
        return InstrumentationSummary(clean, False, ())

    scene_classes = _scene_classes(tree)
    if len(scene_classes) != 1:
        return InstrumentationSummary(clean, False, ())

    target = scene_classes[0]
    target.name = class_name
    target.bases = [ast.Name(id="TrackedScene", ctx=ast.Load())]
    construct = next(
        (
            item
            for item in target.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == "construct"
        ),
        None,
    )
    if construct is None:
        return InstrumentationSummary(clean, False, ())

    _ensure_tracked_import(tree)
    section_timing = _retime_construct(construct, scene)
    injector = _RegistrationInjector()
    construct.body = injector.visit_statement_list(construct.body)
    configure = ast.Expr(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr="configure_generated_scene",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[
                ast.keyword(
                    arg="layout_template",
                    value=ast.Constant(value=contract.layout_template.value),
                ),
                ast.keyword(
                    arg="target_duration",
                    value=ast.Constant(value=_target_duration(scene)),
                ),
                ast.keyword(
                    arg="beat_schedule",
                    value=ast.List(
                        elts=[
                            ast.Tuple(
                                elts=[
                                    ast.Constant(value=beat.beat_id),
                                    ast.Constant(value=float(beat.start_seconds)),
                                    ast.Constant(value=float(beat.end_seconds)),
                                ],
                                ctx=ast.Load(),
                            )
                            for beat in scene.beats
                        ],
                        ctx=ast.Load(),
                    ),
                ),
            ],
        )
    )
    construct.body.insert(0, configure)
    ast.fix_missing_locations(tree)
    rendered = ast.unparse(tree).strip() + "\n"
    return InstrumentationSummary(
        f"# {INSTRUMENTATION_MARKER}\n{rendered}",
        section_timing,
        tuple(injector.registered_names),
    )


def deinstrument_scene_code(source: str) -> str:
    """Remove injected tracking calls before showing code to a repair model."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source.strip() + "\n"

    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module == "src.pipeline_v2.tracked_scene"
        )
    ]
    for node in _scene_classes(tree):
        node.bases = [
            ast.Name(id="Scene", ctx=ast.Load())
            if isinstance(base, ast.Name) and base.id == "TrackedScene"
            else base
            for base in node.bases
        ]
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "construct":
                item.body = _remove_injected_statements(item.body)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).strip() + "\n"


class _RegistrationInjector(ast.NodeTransformer):
    def __init__(self) -> None:
        self.registered_names: list[str] = []

    def visit_statement_list(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        output: list[ast.stmt] = []
        for statement in statements:
            statement = self.visit(statement)
            values = statement if isinstance(statement, list) else [statement]
            output.extend(item for item in values if item is not None)
        return output

    def visit_Assign(self, node: ast.Assign):  # noqa: N802 - ast visitor API
        self.generic_visit(node)
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        return [node, *[self._registration(name) for name in names]]

    def visit_AnnAssign(self, node: ast.AnnAssign):  # noqa: N802 - ast visitor API
        self.generic_visit(node)
        if isinstance(node.target, ast.Name) and node.value is not None:
            return [node, self._registration(node.target.id)]
        return node

    def visit_If(self, node: ast.If):  # noqa: N802 - ast visitor API
        node.test = self.visit(node.test)
        node.body = self.visit_statement_list(node.body)
        node.orelse = self.visit_statement_list(node.orelse)
        return node

    def visit_For(self, node: ast.For):  # noqa: N802 - ast visitor API
        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        node.body = self.visit_statement_list(node.body)
        node.orelse = self.visit_statement_list(node.orelse)
        return node

    def visit_While(self, node: ast.While):  # noqa: N802 - ast visitor API
        node.test = self.visit(node.test)
        node.body = self.visit_statement_list(node.body)
        node.orelse = self.visit_statement_list(node.orelse)
        return node

    def visit_With(self, node: ast.With):  # noqa: N802 - ast visitor API
        for item in node.items:
            item.context_expr = self.visit(item.context_expr)
            if item.optional_vars is not None:
                item.optional_vars = self.visit(item.optional_vars)
        node.body = self.visit_statement_list(node.body)
        return node

    def visit_Try(self, node: ast.Try):  # noqa: N802 - ast visitor API
        node.body = self.visit_statement_list(node.body)
        node.orelse = self.visit_statement_list(node.orelse)
        node.finalbody = self.visit_statement_list(node.finalbody)
        for handler in node.handlers:
            handler.body = self.visit_statement_list(handler.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):  # noqa: N802 - ast visitor API
        # Nested helper functions can construct mobjects, but injecting self
        # there is only valid when they close over the scene. Leave them alone.
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def _registration(self, name: str) -> ast.Expr:
        self.registered_names.append(name)
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr="auto_register",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=name), ast.Name(id=name, ctx=ast.Load())],
                keywords=[],
            )
        )


def _scene_classes(tree: ast.AST) -> list[ast.ClassDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(isinstance(base, ast.Name) and base.id in _SCENE_BASES for base in node.bases)
    ]


def _ensure_tracked_import(tree: ast.Module) -> None:
    if any(
        isinstance(node, ast.ImportFrom)
        and node.module == "src.pipeline_v2.tracked_scene"
        and any(alias.name == "TrackedScene" for alias in node.names)
        for node in tree.body
    ):
        return
    index = 0
    while index < len(tree.body) and isinstance(
        tree.body[index], (ast.Import, ast.ImportFrom)
    ):
        index += 1
    tree.body.insert(
        index,
        ast.ImportFrom(
            module="src.pipeline_v2.tracked_scene",
            names=[ast.alias(name="TrackedScene")],
            level=0,
        ),
    )


def _remove_injected_statements(statements: list[ast.stmt]) -> list[ast.stmt]:
    output = []
    for statement in statements:
        if _is_injected_call(statement):
            continue
        for field in ("body", "orelse", "finalbody"):
            nested = getattr(statement, field, None)
            if isinstance(nested, list):
                setattr(statement, field, _remove_injected_statements(nested))
        if isinstance(statement, ast.Try):
            for handler in statement.handlers:
                handler.body = _remove_injected_statements(handler.body)
        output.append(statement)
    return output


def _is_injected_call(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    function = statement.value.func
    return isinstance(function, ast.Attribute) and function.attr in _INJECTED_METHODS


def _retime_construct(construct: ast.FunctionDef, scene: NarratedScene) -> bool:
    sections = _section_ranges(construct.body)
    beats = {beat.beat_id: beat for beat in scene.beats}
    expected_order = [beat.beat_id for beat in scene.beats]
    if [beat_id for beat_id, _ in sections] == expected_order:
        for beat_id, statements in sections:
            beat = beats[beat_id]
            _retime_calls(statements, max(0.1, beat.end_seconds - beat.start_seconds))
        return True
    _retime_calls(construct.body, _target_duration(scene))
    return False


def _section_ranges(statements: list[ast.stmt]) -> list[tuple[str, list[ast.stmt]]]:
    sections: list[tuple[str, list[ast.stmt]]] = []
    active_id: str | None = None
    active: list[ast.stmt] = []
    for statement in statements:
        marker = _section_marker(statement)
        if marker is not None:
            if active_id is not None:
                sections.append((active_id, active))
            active_id = marker
            active = []
        elif active_id is not None:
            active.append(statement)
    if active_id is not None:
        sections.append((active_id, active))
    return sections


def _section_marker(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if not call.args or not isinstance(call.args[0], ast.Constant):
        return None
    function = call.func
    if (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "self"
        and function.attr == "next_section"
        and isinstance(call.args[0].value, str)
    ):
        return call.args[0].value
    return None


def _retime_calls(statements: list[ast.stmt], target: float) -> None:
    calls = sorted(
        (
            item
            for statement in statements
            for item in _timed_calls(statement)
        ),
        key=lambda item: (
            getattr(item[0], "lineno", 0), getattr(item[0], "col_offset", 0)
        ),
    )
    if not calls:
        statements.append(_wait_statement(target))
        return
    weights = [_call_duration(call) for call, _ in calls]
    total = sum(weight * repetitions for weight, (_, repetitions) in zip(weights, calls))
    total = total or float(len(calls))
    durations = [target * weight / total for weight in weights]
    actual = sum(duration * repetitions for duration, (_, repetitions) in zip(durations, calls))
    durations[-1] += (target - actual) / calls[-1][1]
    for (call, _), duration in zip(calls, durations):
        _set_call_duration(call, max(0.01, duration))


def _timed_calls(node: ast.AST, repetitions: int = 1) -> list[tuple[ast.Call, int]]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return []
    if isinstance(node, ast.For):
        loop_repetitions = _range_repetitions(node.iter)
        values = [
            item
            for statement in node.body
            for item in _timed_calls(statement, repetitions * loop_repetitions)
        ]
        values.extend(
            item
            for statement in node.orelse
            for item in _timed_calls(statement, repetitions)
        )
        return values
    if isinstance(node, ast.Call) and _self_method(node) in {"play", "wait"}:
        return [(node, repetitions)]
    return [
        item
        for child in ast.iter_child_nodes(node)
        for item in _timed_calls(child, repetitions)
    ]


def _range_repetitions(node: ast.AST) -> int:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and 1 <= len(node.args) <= 3
    ):
        return 1
    try:
        values = [int(ast.literal_eval(argument)) for argument in node.args]
        return max(1, len(range(*values)))
    except (TypeError, ValueError):
        return 1


def _self_method(call: ast.Call) -> str | None:
    function = call.func
    if (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "self"
    ):
        return function.attr
    return None


def _call_duration(call: ast.Call) -> float:
    method = _self_method(call)
    if method == "play":
        for keyword in call.keywords:
            if keyword.arg == "run_time":
                return _number(keyword.value, 1.0)
        return 1.0
    if call.args:
        return _number(call.args[0], 1.0)
    return 1.0


def _set_call_duration(call: ast.Call, duration: float) -> None:
    value = ast.Constant(value=round(float(duration), 4))
    if _self_method(call) == "play":
        for keyword in call.keywords:
            if keyword.arg == "run_time":
                keyword.value = value
                return
        call.keywords.append(ast.keyword(arg="run_time", value=value))
        return
    if call.args:
        call.args[0] = value
    else:
        call.args.append(value)


def _number(value: ast.AST, fallback: float) -> float:
    if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
        return max(0.01, float(value.value))
    return fallback


def _wait_statement(duration: float) -> ast.Expr:
    return ast.Expr(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()), attr="wait", ctx=ast.Load()
            ),
            args=[ast.Constant(value=round(max(0.01, duration), 4))],
            keywords=[],
        )
    )


def _target_duration(scene: NarratedScene) -> float:
    if scene.audio_duration_seconds:
        return float(scene.audio_duration_seconds)
    if scene.beats:
        final = max(float(beat.end_seconds) for beat in scene.beats)
        if final > 0:
            return final
    return max(1.5, len(scene.narration.split()) / 2.8)
