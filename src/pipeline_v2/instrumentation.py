"""Add V2 observability and narration timing to ordinary generated Manim."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from .models import NarratedScene, VisualContract


INSTRUMENTATION_MARKER = "Instrumented automatically by PDF-to-Manim V2."
_SCENE_BASES = {"Scene", "TrackedScene"}
_INJECTED_METHODS = {
    "auto_register", "configure_generated_scene", "finish_narration_sync",
}

# The narration-sync contract. Static retiming only nudges a section towards
# its beat: durations are never multiplied by less than 1 / MAX_SPEEDUP or by
# more than MAX_STRETCH. Whatever is left over is handled at render time by
# TrackedScene, which holds until a beat starts and reports beats that start late.
MAX_SPEEDUP = 1.5
MAX_STRETCH = 1.5
SCENE_SCALE_KEY = "scene"


@dataclass(frozen=True)
class InstrumentationSummary:
    code: str
    section_timing_applied: bool
    registered_names: tuple[str, ...]
    # (beat_id, applied scale) per retimed section, or one ("scene", scale)
    # entry when the section markers did not match the beats.
    timing_scales: tuple[tuple[str, float], ...] = ()
    # Keys of timing_scales whose required scale was outside the caps.
    capped_beats: tuple[str, ...] = ()


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
    authored_timing = _authored_timing(construct.body)
    section_timing, timing_scales, capped_beats = _retime_construct(construct, scene)
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
                # Every duration as the model wrote it, in source order, so
                # deinstrument_scene_code can undo the retiming exactly.
                # Without it each repair or revision pass would scale already
                # scaled durations and compound the speed-up past the cap.
                ast.keyword(
                    arg="authored_timing",
                    value=ast.List(
                        elts=[ast.Constant(value=item) for item in authored_timing],
                        ctx=ast.Load(),
                    ),
                ),
            ],
        )
    )
    construct.body.insert(0, configure)
    construct.body.append(_self_call_statement("finish_narration_sync"))
    ast.fix_missing_locations(tree)
    rendered = ast.unparse(tree).strip() + "\n"
    return InstrumentationSummary(
        f"# {INSTRUMENTATION_MARKER}\n{rendered}",
        section_timing,
        tuple(injector.registered_names),
        tuple(timing_scales),
        tuple(capped_beats),
    )


def deinstrument_scene_code(source: str) -> str:
    """Remove injected tracking calls before showing code to a repair model.

    Retimed durations are put back exactly as they were written, so
    instrumenting the result again scales from the authored durations.
    """

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
                _restore_authored_timing(item)
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


def _retime_construct(
    construct: ast.FunctionDef | ast.AsyncFunctionDef, scene: NarratedScene,
) -> tuple[bool, list[tuple[str, float]], list[str]]:
    """Clamp-scale each beat section; returns (section timing, scales, capped)."""

    sections = _section_ranges(construct.body)
    beats = {beat.beat_id: beat for beat in scene.beats}
    expected_order = [beat.beat_id for beat in scene.beats]
    scales: list[tuple[str, float]] = []
    capped: list[str] = []
    if [beat_id for beat_id, _ in sections] == expected_order:
        for beat_id, statements in sections:
            beat = beats[beat_id]
            scale, was_capped = _retime_calls(
                statements, max(0.1, beat.end_seconds - beat.start_seconds),
            )
            scales.append((beat_id, scale))
            if was_capped:
                capped.append(beat_id)
        return True, scales, capped
    scale, was_capped = _retime_calls(construct.body, _target_duration(scene))
    return False, [(SCENE_SCALE_KEY, scale)], [SCENE_SCALE_KEY] if was_capped else []


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


def _retime_calls(statements: list[ast.stmt], target: float) -> tuple[float, bool]:
    """Scale the play/wait calls towards ``target``; returns (scale, capped).

    The scale is clamped, so the calls need not add up to the target: a
    section that ends early is held at render time until the next beat, and
    one that is still too long starts the next beat late and is reported.
    """

    calls = _ordered_timed_calls(statements)
    if not calls:
        return 1.0, False
    weights = [_call_duration(call) for call, _ in calls]
    total = sum(weight * repetitions for weight, (_, repetitions) in zip(weights, calls))
    required = target / total
    scale = min(MAX_STRETCH, max(1.0 / MAX_SPEEDUP, required))
    for (call, _), weight in zip(calls, weights):
        _set_call_duration(call, max(0.01, weight * scale))
    return scale, abs(required - scale) > 1e-9


def _ordered_timed_calls(statements: list[ast.stmt]) -> list[tuple[ast.Call, int]]:
    return sorted(
        (
            item
            for statement in statements
            for item in _timed_calls(statement)
        ),
        key=lambda item: (
            getattr(item[0], "lineno", 0), getattr(item[0], "col_offset", 0)
        ),
    )


def _authored_timing(statements: list[ast.stmt]) -> list[str | None]:
    """Source of every play/wait duration in order; None where none was written."""

    values: list[str | None] = []
    for call, _ in _ordered_timed_calls(statements):
        node = _duration_node(call)
        values.append(None if node is None else ast.unparse(node))
    return values


def _restore_authored_timing(construct: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    recorded = _recorded_authored_timing(construct.body)
    calls = _ordered_timed_calls(construct.body)
    # Code instrumented before the sync contract records nothing; it is left
    # as it is, like anything that no longer lines up with the record.
    if recorded is None or len(recorded) != len(calls):
        return
    try:
        nodes = [
            None if value is None else ast.parse(value, mode="eval").body
            for value in recorded
        ]
    except SyntaxError:
        return
    for (call, _), node in zip(calls, nodes):
        _restore_duration_node(call, node)


def _recorded_authored_timing(statements: list[ast.stmt]) -> list[str | None] | None:
    for statement in statements:
        if not _is_injected_call(statement):
            continue
        call = statement.value
        if call.func.attr != "configure_generated_scene":
            continue
        for keyword in call.keywords:
            if keyword.arg != "authored_timing":
                continue
            try:
                values = ast.literal_eval(keyword.value)
            except (ValueError, SyntaxError):
                return None
            if isinstance(values, list) and all(
                item is None or isinstance(item, str) for item in values
            ):
                return values
            return None
    return None


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


def _duration_node(call: ast.Call) -> ast.expr | None:
    """The expression that sets how long a self.play / self.wait call lasts."""

    if _self_method(call) == "play":
        return next(
            (keyword.value for keyword in call.keywords if keyword.arg == "run_time"),
            None,
        )
    if call.args:
        return call.args[0]
    return next(
        (keyword.value for keyword in call.keywords if keyword.arg == "duration"),
        None,
    )


def _call_duration(call: ast.Call) -> float:
    node = _duration_node(call)
    if node is not None:
        return _number(node, 1.0)
    if _self_method(call) == "play":
        # self.play(Create(shape, run_time=3)) lasts 3 s, not the 1 s default.
        # The play-level run_time written below overrides it, so assuming 1 s
        # would turn the retiming into a hidden 3x speed-up.
        nested = [
            float(keyword.value.value)
            for argument in call.args
            for inner in ast.walk(argument)
            if isinstance(inner, ast.Call)
            for keyword in inner.keywords
            if keyword.arg == "run_time"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, (int, float))
        ]
        if nested:
            return max(0.01, max(nested))
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
        return
    for keyword in call.keywords:
        if keyword.arg == "duration":
            keyword.value = value
            return
    call.args.append(value)


def _restore_duration_node(call: ast.Call, node: ast.expr | None) -> None:
    if node is None:
        # Nothing was written originally, so the retimed value was added by us.
        if _self_method(call) == "play":
            call.keywords = [item for item in call.keywords if item.arg != "run_time"]
        elif call.args:
            del call.args[0]
        return
    if _self_method(call) == "play":
        for keyword in call.keywords:
            if keyword.arg == "run_time":
                keyword.value = node
    elif call.args:
        call.args[0] = node
    else:
        for keyword in call.keywords:
            if keyword.arg == "duration":
                keyword.value = node


def _number(value: ast.AST, fallback: float) -> float:
    if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
        return max(0.01, float(value.value))
    return fallback


def _self_call_statement(method: str) -> ast.Expr:
    return ast.Expr(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()), attr=method, ctx=ast.Load()
            ),
            args=[],
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
