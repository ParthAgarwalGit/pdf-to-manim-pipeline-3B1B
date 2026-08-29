"""Conservative AST safety gate for generated raw Manim Python.

This is deliberately a gate, not a sandbox. Production execution still needs
an isolated container with no network, a non-root user, resource limits, and a
single writable output directory.
"""

from __future__ import annotations

import ast
from typing import Iterable, Optional

from src.manim_verifier.models import VerificationIssue, VerificationReport


class StaticManimVerifier:
    """Reject generated source that is unsafe or not a single Manim scene."""

    APPROVED_IMPORTS = {
        ("from", "manim", "*", None),
        ("import", "manim", None, None),
        ("import", "numpy", None, "np"),
    }
    FORBIDDEN_CALLS = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
    }
    FORBIDDEN_NAMES = {
        "os",
        "subprocess",
        "pathlib",
        "socket",
        "requests",
        "urllib",
        "http",
        "shutil",
        "tempfile",
        "multiprocessing",
        "builtins",
        "__builtins__",
    }
    FORBIDDEN_ATTRIBUTES = {
        "environ",
        "getenv",
        "system",
        "popen",
        "Popen",
        "spawn",
        "fork",
        "connect",
        "request",
        "urlopen",
        "read_text",
        "write_text",
        "read_bytes",
        "write_bytes",
        "open",
        "exec",
        "eval",
        "compile",
    }
    SCENE_BASES = {"Scene", "ThreeDScene", "MovingCameraScene"}
    VISIBLE_CONSTRUCTORS = {
        "Angle", "AnnularSector", "Annulus", "Arc", "ArcBetweenPoints", "Arrow",
        "Axes", "Brace", "BraceBetweenPoints", "BulletedList", "Circle", "Code",
        "Cross", "Cube", "CurvedArrow", "DashedLine", "DecimalNumber", "Dot",
        "DoubleArrow", "Ellipse", "FunctionGraph", "ImageMobject", "ImplicitFunction",
        "Integer", "Line", "MathTex", "Matrix", "NumberLine", "NumberPlane",
        "Paragraph", "ParametricFunction", "Polygon", "Prism", "Rectangle",
        "RegularPolygon", "RightAngle", "RoundedRectangle", "Sector", "Sphere",
        "Square", "Star", "Surface", "SurroundingRectangle", "Table", "Tex", "Text",
        "Title", "Triangle", "Underline", "VGroup", "Vector",
        "attach_axis_labels", "connect", "create_character_for_role",
        "create_character_lineup", "create_coordinate_system", "create_notation",
        "create_notation_transform_target", "create_pi_students", "create_pi_teacher",
        "create_speech_bubble", "create_text_label", "drop_perpendicular",
        "plot", "plot_function_on_system",
    }

    def __init__(
        self,
        max_script_chars: int = 100_000,
        trusted_import_roots: Optional[set[str]] = None,
        require_purpose_comments: bool = False,
    ):
        self.max_script_chars = max_script_chars
        self.trusted_import_roots = trusted_import_roots or set()
        self.require_purpose_comments = require_purpose_comments

    def verify(self, source: str) -> VerificationReport:
        issues: list[VerificationIssue] = []
        if len(source) > self.max_script_chars:
            issues.append(self._issue(
                "SCRIPT_TOO_LARGE",
                f"Script has {len(source)} characters; limit is {self.max_script_chars}.",
            ))

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            issues.append(self._issue(
                "SYNTAX_ERROR",
                exc.msg,
                lines=[exc.lineno] if exc.lineno else [],
            ))
            return VerificationReport(
                safe=False,
                syntax_valid=False,
                issues=issues,
            )

        self._check_imports(tree, issues)
        self._check_dangerous_nodes(tree, issues)
        self._check_coordinate_literals(tree, issues)
        if self.require_purpose_comments:
            self._check_purpose_comments(tree, source, issues)
        scene_classes = self._scene_classes(tree)
        if len(scene_classes) != 1:
            lines = [node.lineno for node in scene_classes]
            issues.append(self._issue(
                "SCENE_CLASS_COUNT",
                f"Expected exactly one Manim Scene subclass; found {len(scene_classes)}.",
                lines=lines,
            ))
        elif not any(
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == "construct"
            for item in scene_classes[0].body
        ):
            issues.append(self._issue(
                "MISSING_CONSTRUCT",
                "The Scene subclass must define construct().",
                lines=[scene_classes[0].lineno],
            ))

        safe = not any(issue.severity == "error" for issue in issues)
        return VerificationReport(
            safe=safe,
            syntax_valid=True,
            issues=issues,
            scene_class=scene_classes[0].name if len(scene_classes) == 1 else None,
        )

    def _check_imports(
        self,
        tree: ast.AST,
        issues: list[VerificationIssue],
    ) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    signature = ("import", alias.name, None, alias.asname)
                    if signature not in self.APPROVED_IMPORTS and not self._trusted(alias.name):
                        issues.append(self._issue(
                            "IMPORT_NOT_ALLOWED",
                            f"Import '{alias.name}' is not approved.",
                            lines=[node.lineno],
                        ))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    issues.append(self._issue(
                        "IMPORT_NOT_ALLOWED",
                        "Relative imports are not approved.",
                        lines=[node.lineno],
                    ))
                    continue
                for alias in node.names:
                    signature = ("from", node.module, alias.name, alias.asname)
                    if signature not in self.APPROVED_IMPORTS and not self._trusted(node.module):
                        issues.append(self._issue(
                            "IMPORT_NOT_ALLOWED",
                            f"Import 'from {node.module} import {alias.name}' is not approved.",
                            lines=[node.lineno],
                        ))

    def _trusted(self, module: Optional[str]) -> bool:
        if not module:
            return False
        return any(module == root or module.startswith(f"{root}.") for root in self.trusted_import_roots)

    def _check_dangerous_nodes(
        self,
        tree: ast.AST,
        issues: list[VerificationIssue],
    ) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = self._call_name(node.func)
                root = name.split(".", 1)[0]
                tail = name.rsplit(".", 1)[-1]
                if name in self.FORBIDDEN_CALLS or root in self.FORBIDDEN_NAMES:
                    issues.append(self._issue(
                        "DANGEROUS_CALL",
                        f"Call '{name}' is not allowed in generated Manim code.",
                        lines=[node.lineno],
                    ))
                elif tail in self.FORBIDDEN_ATTRIBUTES:
                    issues.append(self._issue(
                        "DANGEROUS_ATTRIBUTE",
                        f"Attribute call '{name}' is not allowed.",
                        lines=[node.lineno],
                    ))
            elif isinstance(node, ast.Name) and node.id in self.FORBIDDEN_NAMES:
                issues.append(self._issue(
                    "DANGEROUS_NAME",
                    f"Name '{node.id}' is not allowed.",
                    lines=[node.lineno],
                ))
            elif isinstance(node, ast.Attribute) and node.attr in self.FORBIDDEN_ATTRIBUTES:
                issues.append(self._issue(
                    "DANGEROUS_ATTRIBUTE",
                    f"Attribute '{node.attr}' is not allowed.",
                    lines=[node.lineno],
                ))
            elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                issues.append(self._issue(
                    "DANGEROUS_ATTRIBUTE",
                    f"Dunder attribute '{node.attr}' is not allowed.",
                    lines=[node.lineno],
                ))
            elif isinstance(node, ast.While) and self._is_true(node.test):
                issues.append(self._issue(
                    "OBVIOUS_INFINITE_LOOP",
                    "An unconditional while loop is not allowed.",
                    lines=[node.lineno],
                ))

    def _check_purpose_comments(
        self,
        tree: ast.AST,
        source: str,
        issues: list[VerificationIssue],
    ) -> None:
        lines = source.splitlines()
        covered_calls: set[int] = set()
        candidates: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            value = None
            if isinstance(node, ast.Assign):
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                value = node.value
            if not isinstance(value, ast.Call):
                continue
            constructor = self._call_name(value.func).rsplit(".", 1)[-1]
            if constructor not in self.VISIBLE_CONSTRUCTORS:
                continue
            covered_calls.add(id(value))
            candidates.append((node.lineno, constructor))

        # Also catch inline objects such as Create(Circle()), which otherwise
        # have neither a stable identity nor a nearby purpose annotation.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or id(node) in covered_calls:
                continue
            constructor = self._call_name(node.func).rsplit(".", 1)[-1]
            if constructor in self.VISIBLE_CONSTRUCTORS:
                candidates.append((node.lineno, constructor))

        for line_number, constructor in dict.fromkeys(candidates):
            previous = line_number - 2
            while previous >= 0 and not lines[previous].strip():
                previous -= 1
            comment = lines[previous].strip() if previous >= 0 else ""
            valid_comment = False
            if comment.startswith("# Purpose [") and "]:" in comment:
                beat_text, purpose = comment[len("# Purpose ["):].split("]:", 1)
                valid_comment = bool(beat_text.strip() and purpose.strip())
            if not valid_comment:
                issues.append(VerificationIssue(
                    code="PURPOSE_COMMENT_MISSING",
                    severity="error",
                    message=(
                        f"Visible {constructor} created at line {line_number} needs an "
                        "immediately preceding, non-empty "
                        "'# Purpose [beat_ids]: explanation' comment."
                    ),
                    code_lines=[line_number],
                ))

    def _check_coordinate_literals(
        self,
        tree: ast.AST,
        issues: list[VerificationIssue],
    ) -> None:
        """Report obvious off-frame and axes-relative coordinate mistakes."""
        has_axes = any(
            isinstance(node, ast.Call)
            and self._call_name(node.func).rsplit(".", 1)[-1]
            in {"Axes", "NumberPlane", "create_coordinate_system"}
            for node in ast.walk(tree)
        )
        coordinate_constructors = {"Arrow", "Dot", "Line", "Polygon", "Vector"}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = self._call_name(node.func).rsplit(".", 1)[-1]
            vectors: list[tuple[float, float]] = []
            if call_name == "move_to" and node.args:
                vector = self._literal_xy(node.args[0])
                if vector is not None:
                    vectors.append(vector)
            elif call_name in coordinate_constructors:
                for argument in node.args:
                    vector = self._literal_xy(argument)
                    if vector is not None:
                        vectors.append(vector)
                for keyword in node.keywords:
                    if keyword.arg in {"point", "start", "end"}:
                        vector = self._literal_xy(keyword.value)
                        if vector is not None:
                            vectors.append(vector)

            if any(abs(x) > 6.5 or abs(y) > 3.5 for x, y in vectors):
                issues.append(VerificationIssue(
                    code="OFFSCREEN_LITERAL_COORDINATE",
                    severity="warning",
                    message=(
                        f"Literal coordinates at line {node.lineno} exceed the "
                        "conservative Manim safe area x=+/-6.5, y=+/-3.5."
                    ),
                    code_lines=[node.lineno],
                ))
            if has_axes and call_name in coordinate_constructors and vectors:
                issues.append(VerificationIssue(
                    code="RAW_COORDINATE_WITH_AXES",
                    severity="warning",
                    message=(
                        f"{call_name} at line {node.lineno} uses raw scene coordinates "
                        "while axes exist; verify that axes.c2p() or on_axes() is intended."
                    ),
                    code_lines=[node.lineno],
                ))

    @classmethod
    def _literal_xy(cls, node: ast.AST) -> Optional[tuple[float, float]]:
        if not isinstance(node, (ast.List, ast.Tuple)) or len(node.elts) < 2:
            return None
        x = cls._literal_number(node.elts[0])
        y = cls._literal_number(node.elts[1])
        if x is None or y is None:
            return None
        return x, y

    @staticmethod
    def _literal_number(node: ast.AST) -> Optional[float]:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant) and isinstance(
                node.operand.value,
                (int, float),
            ):
                return -float(node.operand.value)
        return None

    def _scene_classes(self, tree: ast.AST) -> list[ast.ClassDef]:
        classes: list[ast.ClassDef] = []
        for node in tree.body if isinstance(tree, ast.Module) else []:
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                name = self._call_name(base)
                if name in self.SCENE_BASES or name in {
                    f"manim.{base_name}" for base_name in self.SCENE_BASES
                }:
                    classes.append(node)
                    break
        return classes

    @staticmethod
    def _call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = StaticManimVerifier._call_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""

    @staticmethod
    def _is_true(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return bool(node.value)
        return (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.Not)
            and isinstance(node.operand, ast.Constant)
            and not bool(node.operand.value)
        )

    @staticmethod
    def _issue(
        code: str,
        message: str,
        *,
        lines: Optional[Iterable[int]] = None,
    ) -> VerificationIssue:
        return VerificationIssue(
            code=code,
            severity="error",
            message=message,
            code_lines=list(lines or []),
        )
