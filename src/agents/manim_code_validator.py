"""Manim Code Validator Agent — validates generated animation code.

This agent performs multi-level validation on generated Manim scene code:
  1. AST Parse     — catches syntax errors
  2. Import Check  — verifies referenced Manim classes exist
  3. Dry Run       — optionally runs `manim --dry_run` via subprocess

If validation fails, the error information is returned so the orchestrator
can send it back to the Animation Generator for retry.
"""

import ast
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agents.base import (
    Agent,
    AgentExecutionError,
    AgentInput,
    AgentOutput,
    BackoffStrategy,
    RetryPolicy,
)
from src.schemas.animation_code import AnimationCodeObject

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known Manim classes / functions (subset for import checking)
# ---------------------------------------------------------------------------

KNOWN_MANIM_CLASSES = {
    # Core
    "Scene", "ThreeDScene", "MovingCameraScene",
    # Mobjects — text
    "Text", "Tex", "MathTex", "Title", "BulletedList", "Paragraph",
    # Mobjects — shapes
    "Circle", "Square", "Rectangle", "Triangle", "Line", "Arrow",
    "Dot", "Polygon", "Ellipse", "Arc", "Annulus", "Star",
    "DashedLine", "DoubleArrow", "RoundedRectangle", "RegularPolygon",
    # Mobjects — coordinate
    "Axes", "NumberPlane", "NumberLine", "ComplexPlane",
    "CoordinateSystem", "ThreeDAxes",
    # Mobjects — graph
    "Graph", "DiGraph",
    # Mobjects — groups
    "VGroup", "Group", "VDict",
    # Mobjects — 3D
    "Surface", "Sphere", "Cube", "Cylinder", "Cone", "Torus",
    # Animations — creation
    "Create", "Write", "DrawBorderThenFill", "ShowCreation",
    "FadeIn", "FadeOut", "GrowFromCenter", "GrowArrow",
    "SpinInFromNothing", "SpiralIn",
    # Animations — transform
    "Transform", "ReplacementTransform", "TransformFromCopy",
    "MoveToTarget", "ApplyMethod", "AnimationGroup",
    "Succession", "LaggedStart", "LaggedStartMap",
    # Animations — indication
    "Indicate", "Flash", "Circumscribe", "ShowPassingFlash",
    "FocusOn", "Wiggle",
    # Animations — movement
    "MoveAlongPath", "Rotate", "ApplyWave",
    # Animations — fading
    "FadeTransform", "FadeTransformPieces",
    # Constants / utilities
    "UP", "DOWN", "LEFT", "RIGHT", "ORIGIN", "UL", "UR", "DL", "DR",
    "PI", "TAU", "DEGREES",
    "RED", "BLUE", "GREEN", "YELLOW", "WHITE", "BLACK",
    "ORANGE", "PINK", "PURPLE", "TEAL", "GOLD", "GREY", "GRAY",
    "BLUE_A", "BLUE_B", "BLUE_C", "BLUE_D", "BLUE_E",
    "RED_A", "RED_B", "RED_C", "RED_D", "RED_E",
    "GREEN_A", "GREEN_B", "GREEN_C", "GREEN_D", "GREEN_E",
    "config", "tempconfig",
    # Rate functions
    "linear", "smooth", "rush_into", "rush_from", "there_and_back",
    "running_start", "rate_functions",
}


# ---------------------------------------------------------------------------
# Security gate — generated scene code is EXECUTED locally by `manim`
# ---------------------------------------------------------------------------
# The code under validation comes from an LLM whose prompt embeds text
# extracted from an arbitrary user-supplied PDF. A malicious document can
# prompt-inject instructions ("...also call os.system(...)") that would then
# run on the user's machine at render time. Scene code has no legitimate need
# for filesystem, network, process, or reflection APIs, so anything outside
# this allowlist fails validation as an ERROR — the orchestrator feeds the
# message back to the generator for a clean rewrite.

ALLOWED_IMPORT_ROOTS = {
    "manim",   # the engine itself
    "numpy",   # positions/paths (the prompt suggests np.array)
    "math",    # inline trig for plotted functions
    "random",  # occasionally used for scatter/jitter effects
    "src",     # repo helpers (pi characters, layout tracker, notation)
}

FORBIDDEN_CALL_NAMES = {
    "eval", "exec", "compile", "__import__", "open", "input", "breakpoint",
}

FORBIDDEN_ATTRIBUTES = {
    "__globals__", "__builtins__", "__subclasses__", "__mro__",
    "__code__", "__reduce__", "__loader__",
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result for a single scene validation."""
    scene_id: str
    passed: bool
    syntax_ok: bool
    imports_ok: bool
    dry_run_ok: Optional[bool]
    security_ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# I/O Contracts
# ---------------------------------------------------------------------------

class ManimValidatorInput(AgentInput):
    """Input to the Manim Code Validator."""

    def __init__(
        self,
        animation_code_objects: List[AnimationCodeObject],
        enable_dry_run: bool = False,
    ):
        self.animation_code_objects = animation_code_objects
        self.enable_dry_run = enable_dry_run


class ManimValidatorOutput(AgentOutput):
    """Output: all scenes passed validation."""

    def __init__(
        self,
        validated_objects: List[AnimationCodeObject],
        results: List[ValidationResult],
    ):
        self.validated_objects = validated_objects
        self.results = results


class ManimValidatorFailure(AgentOutput):
    """Output: one or more scenes failed validation."""

    def __init__(
        self,
        passed_objects: List[AnimationCodeObject],
        failed_objects: List[AnimationCodeObject],
        results: List[ValidationResult],
    ):
        self.passed_objects = passed_objects
        self.failed_objects = failed_objects
        self.results = results


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ManimCodeValidatorAgent(Agent):
    """Validates generated Manim code via AST, import checks, and optional dry run."""

    def execute(self, input_data: ManimValidatorInput) -> AgentOutput:
        if not self.validate_input(input_data):
            raise AgentExecutionError("INVALID_INPUT", "Expected ManimValidatorInput.")

        results: List[ValidationResult] = []
        passed: List[AnimationCodeObject] = []
        failed: List[AnimationCodeObject] = []

        for code_obj in input_data.animation_code_objects:
            result = self._validate_scene(code_obj, input_data.enable_dry_run)
            results.append(result)
            if result.passed:
                passed.append(code_obj)
            else:
                failed.append(code_obj)

        if not failed:
            logger.info(f"All {len(passed)} scenes passed validation.")
            return ManimValidatorOutput(validated_objects=passed, results=results)
        else:
            logger.warning(f"{len(failed)}/{len(results)} scenes failed validation.")
            return ManimValidatorFailure(
                passed_objects=passed,
                failed_objects=failed,
                results=results,
            )

    def validate_input(self, input_data: AgentInput) -> bool:
        return isinstance(input_data, ManimValidatorInput) and hasattr(input_data, "animation_code_objects")

    def get_retry_policy(self) -> RetryPolicy:
        return RetryPolicy(max_attempts=1)

    # ------------------------------------------------------------------
    # Validation pipeline
    # ------------------------------------------------------------------

    def _validate_scene(self, code_obj: AnimationCodeObject, dry_run: bool) -> ValidationResult:
        scene_id = code_obj.scene_id
        errors: List[str] = []
        warnings: List[str] = []

        # 0. Deterministic LaTeX gate — a text-mode Tex() carrying math tokens
        # dies in the LaTeX subprocess ("Missing $ inserted") and costs the
        # whole scene at render time (14% of a real run's video). Auto-fix the
        # fixable class in place; reject the unfixable class into the
        # regeneration loop. Zero LLM cost.
        fixed_code, tex_notes, tex_errors = self._fix_tex_math_mode(code_obj.code)
        if fixed_code != code_obj.code:
            code_obj.code = fixed_code  # propagates to persistence/render
        warnings.extend(tex_notes)
        errors.extend(tex_errors)
        latex_ok = not tex_errors

        # Reconstruct full source
        full_source = "\n".join(code_obj.imports) + "\n\n" + code_obj.code

        # 1. AST Parse
        syntax_ok = self._check_syntax(full_source, errors)

        # 2. Security gate (only if syntax passed) — errors, never warnings
        security_ok = True
        if syntax_ok:
            security_ok = self._check_security(full_source, errors)

        # 3. Import Check (only if syntax passed)
        imports_ok = True
        if syntax_ok:
            imports_ok = self._check_imports(full_source, errors, warnings)

        # 3b. Geometry-anchoring lint (warnings only): connectors whose BOTH
        # endpoints are hardcoded literals drift off-target — they should be
        # anchored via connect()/get_center()/on_axes()/drop_perpendicular().
        if syntax_ok:
            self._check_geometry_anchoring(full_source, warnings)

        # 4. Dry Run (optional, only if previous checks passed)
        dry_run_ok = None
        if dry_run and syntax_ok and security_ok and imports_ok:
            dry_run_ok = self._check_dry_run(full_source, scene_id, errors)

        passed = (
            syntax_ok and security_ok and imports_ok and latex_ok
            and (dry_run_ok is None or dry_run_ok)
        )
        return ValidationResult(
            scene_id=scene_id,
            passed=passed,
            syntax_ok=syntax_ok,
            imports_ok=imports_ok,
            dry_run_ok=dry_run_ok,
            security_ok=security_ok,
            errors=errors,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Deterministic LaTeX gate
    # ------------------------------------------------------------------

    # Math-only macros that make text-mode LaTeX abort with "Missing $
    # inserted". `_` and `^` are matched separately below.
    _MATH_MACROS = re.compile(
        r"\\(frac|sum|int|prod|lim|cdot|times|sqrt|infty|pm|mp|leq?|geq?|neq"
        r"|approx|equiv|sim|propto|vec|hat|bar|tilde|alpha|beta|gamma|delta"
        r"|epsilon|varepsilon|theta|lambda|mu|nu|rho|sigma|tau|phi|varphi"
        r"|omega|Gamma|Delta|Theta|Lambda|Sigma|Phi|Omega|partial|nabla"
        r"|rightarrow|Rightarrow|mapsto|in|subset|subseteq|cup|cap|forall"
        r"|exists|langle|rangle|mathbb|mathbf|mathrm|operatorname)\b"
    )

    @staticmethod
    def _outside_math_spans(s: str) -> str:
        """The parts of a Tex string OUTSIDE its $...$ math spans."""
        return re.sub(r"(?<!\\)\$.*?(?<!\\)\$", " ", s)

    def _fix_tex_math_mode(self, code: str):
        """Auto-fix / reject Tex() strings that would abort LaTeX at render.

        Returns ``(possibly_fixed_code, notes, errors)``:
          - Tex("...") with math tokens and NO $ spans → whole string wrapped
            in $...$ (rewritten in place as a raw literal) — compiles instead
            of killing the scene.
          - Any Tex/MathTex string with an unbalanced $ → error (rejected into
            the regeneration loop; no safe mechanical fix).
          - Tex mixing $-spans WITH math tokens outside them → error (the safe
            span boundaries are ambiguous; the regenerator gets told exactly
            what to do).
        Only static string literals are checked; f-strings are left alone.
        """
        notes: List[str] = []
        errors: List[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, notes, errors  # the syntax check will report it

        replacements = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            name = node.func.id
            if name not in ("Tex", "MathTex"):
                continue
            if (
                not node.args
                or not isinstance(node.args[0], ast.Constant)
                or not isinstance(node.args[0].value, str)
            ):
                continue
            snode = node.args[0]
            value = snode.value

            unescaped_dollars = len(re.findall(r"(?<!\\)\$", value))
            if unescaped_dollars % 2 == 1:
                errors.append(
                    f"LaTeX gate: {name} at line {node.lineno} has an unbalanced "
                    f"'$' in {value!r} — LaTeX will abort. Balance the $-span or "
                    "escape the dollar sign as \\$."
                )
                continue

            if name != "Tex":
                continue  # MathTex is already math mode

            outside = self._outside_math_spans(value)
            has_math = bool(re.search(r"(?<!\\)[_^]", outside)) or bool(
                self._MATH_MACROS.search(outside)
            )
            if not has_math:
                continue

            if unescaped_dollars > 0:
                errors.append(
                    f"LaTeX gate: Tex at line {node.lineno} mixes $-spans with "
                    f"math tokens OUTSIDE them ({value!r}). Move all math inside "
                    "$...$ or convert the call to MathTex."
                )
                continue

            new_value = f"${value}$"
            if "\n" in new_value:
                errors.append(
                    f"LaTeX gate: Tex at line {node.lineno} contains math tokens "
                    "in a multi-line string — wrap the math in $...$ or use MathTex."
                )
                continue
            if '"' not in new_value:
                new_literal = 'r"' + new_value + '"'
            elif "'" not in new_value:
                new_literal = "r'" + new_value + "'"
            else:
                errors.append(
                    f"LaTeX gate: Tex at line {node.lineno} needs math mode but "
                    "mixes both quote characters — rewrite it with MathTex."
                )
                continue
            replacements.append((snode, new_literal))

        if not replacements:
            return code, notes, errors

        # Splice by absolute offsets, last-to-first so positions stay valid.
        lines = code.splitlines(keepends=True)
        line_start = [0]
        for ln in lines:
            line_start.append(line_start[-1] + len(ln))
        text = "".join(lines)
        for snode, new_literal in sorted(
            replacements, key=lambda t: (t[0].lineno, t[0].col_offset), reverse=True
        ):
            a = line_start[snode.lineno - 1] + snode.col_offset
            b = line_start[snode.end_lineno - 1] + snode.end_col_offset
            notes.append(
                f"LaTeX gate: auto-wrapped math-mode Tex at line {snode.lineno}: "
                f"{text[a:b]} -> {new_literal}"
            )
            text = text[:a] + new_literal + text[b:]
        return text, notes, errors

    # ------------------------------------------------------------------

    def _check_syntax(self, source: str, errors: List[str]) -> bool:
        """AST-parse the source to catch syntax errors."""
        try:
            ast.parse(source)
            return True
        except SyntaxError as e:
            errors.append(f"SyntaxError at line {e.lineno}: {e.msg}")
            return False

    _CONNECTOR_CLASSES = {"Arrow", "Line", "DashedLine", "DoubleArrow"}

    def _check_geometry_anchoring(self, source: str, warnings: List[str]) -> None:
        """Warn on connector mobjects whose both endpoints are numeric literals.

        ``Arrow(start=[2.5, -1.7, 0], end=[3.0, -1.58, 0])`` is the signature
        of the LLM doing endpoint arithmetic in its head — a real run showed
        such arrows ending midway or in the wrong direction. Anchored forms
        (``connect(a, b)``, ``obj.get_center()``, ``on_axes(...)``) are exempt
        because their endpoints are expressions, not literals.
        """
        def is_literal_point(node: ast.AST) -> bool:
            if isinstance(node, (ast.List, ast.Tuple)):
                return all(
                    isinstance(el, ast.Constant)
                    or (isinstance(el, ast.UnaryOp) and isinstance(el.operand, ast.Constant))
                    for el in node.elts
                )
            return False

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return

        flagged: List[str] = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id not in self._CONNECTOR_CLASSES:
                continue
            endpoints = list(node.args[:2])
            for kw in node.keywords:
                if kw.arg in ("start", "end"):
                    endpoints.append(kw.value)
            if len(endpoints) >= 2 and all(is_literal_point(e) for e in endpoints[:2]):
                flagged.append(f"{node.func.id} at line {node.lineno}")

        if flagged:
            warnings.append(
                "Hardcoded connector endpoints (drift off-target): "
                + ", ".join(flagged[:5])
                + ". Anchor endpoints to objects via connect(a, b), "
                "obj.get_center(), on_axes(axes, x, y), or drop_perpendicular()."
            )

    def _check_security(self, source: str, errors: List[str]) -> bool:
        """Reject imports and constructs a Manim scene has no business using.

        Defends against prompt injection from malicious PDFs reaching local
        code execution: imports outside ALLOWED_IMPORT_ROOTS, calls to
        eval/exec/open/…, and dunder-reflection escapes all fail validation
        with an actionable error message.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False

        ok = True
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in ALLOWED_IMPORT_ROOTS:
                        errors.append(
                            f"Security: import of '{alias.name}' is not allowed in "
                            f"scene code (line {node.lineno}). Allowed roots: "
                            f"{sorted(ALLOWED_IMPORT_ROOTS)}."
                        )
                        ok = False
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    errors.append(
                        f"Security: relative import at line {node.lineno} is not "
                        "allowed in scene code."
                    )
                    ok = False
                    continue
                root = (node.module or "").split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    errors.append(
                        f"Security: import from '{node.module}' is not allowed in "
                        f"scene code (line {node.lineno}). Allowed roots: "
                        f"{sorted(ALLOWED_IMPORT_ROOTS)}."
                    )
                    ok = False
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALL_NAMES:
                    errors.append(
                        f"Security: call to '{func.id}(...)' at line {node.lineno} "
                        "is not allowed in scene code."
                    )
                    ok = False
            elif isinstance(node, ast.Attribute):
                if node.attr in FORBIDDEN_ATTRIBUTES:
                    errors.append(
                        f"Security: access to '{node.attr}' at line {node.lineno} "
                        "is not allowed in scene code."
                    )
                    ok = False
            elif isinstance(node, ast.Name) and node.id == "__builtins__":
                errors.append(
                    f"Security: access to '__builtins__' at line {node.lineno} "
                    "is not allowed in scene code."
                )
                ok = False

        return ok

    def _check_imports(self, source: str, errors: List[str], warnings: List[str]) -> bool:
        """Walk the AST looking for names that aren't known Manim classes."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False

        unknown_names: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id[0].isupper():
                # Only flag uppercase names (likely class references)
                if node.id not in KNOWN_MANIM_CLASSES and not node.id.startswith("_"):
                    # Skip common Python builtins
                    if node.id not in {"True", "False", "None", "Exception", "ValueError",
                                       "TypeError", "KeyError", "List", "Dict", "Any",
                                       "Optional", "Tuple", "Set", "Union"}:
                        unknown_names.append(node.id)

        if unknown_names:
            unique = list(set(unknown_names))
            warnings.append(f"Unknown Manim references: {', '.join(unique[:10])}")
            # This is a warning, not an error — LLM might use valid classes we don't know
        return True

    def _check_dry_run(self, source: str, scene_id: str, errors: List[str]) -> bool:
        """Write source to temp file and run `manim --dry_run`."""
        class_name = None
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == "Scene":
                            class_name = node.name
                            break
        except SyntaxError:
            errors.append("Cannot run dry_run: syntax error.")
            return False

        if not class_name:
            errors.append("No Scene subclass found for dry_run.")
            return False

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                # Prepend manim import if not present
                if "from manim import" not in source and "import manim" not in source:
                    f.write("from manim import *\n\n")
                f.write(source)
                temp_path = f.name

            result = subprocess.run(
                ["manim", "-ql", "--dry_run", temp_path, class_name],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                stderr_snippet = result.stderr[:500] if result.stderr else "No stderr."
                errors.append(f"manim dry_run failed (exit {result.returncode}): {stderr_snippet}")
                return False

            return True

        except FileNotFoundError:
            errors.append("manim CLI not found — skipping dry_run.")
            return True  # Don't block on missing manim
        except subprocess.TimeoutExpired:
            errors.append("manim dry_run timed out (60s).")
            return False
        except Exception as e:
            errors.append(f"Unexpected dry_run error: {e}")
            return False
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

        return False  # Unreachable, but satisfies type checker
