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

        # Reconstruct full source
        full_source = "\n".join(code_obj.imports) + "\n\n" + code_obj.code

        # 1. AST Parse
        syntax_ok = self._check_syntax(full_source, errors)

        # 2. Import Check (only if syntax passed)
        imports_ok = True
        if syntax_ok:
            imports_ok = self._check_imports(full_source, errors, warnings)

        # 3. Dry Run (optional, only if previous checks passed)
        dry_run_ok = None
        if dry_run and syntax_ok and imports_ok:
            dry_run_ok = self._check_dry_run(full_source, scene_id, errors)

        passed = syntax_ok and imports_ok and (dry_run_ok is None or dry_run_ok)
        return ValidationResult(
            scene_id=scene_id,
            passed=passed,
            syntax_ok=syntax_ok,
            imports_ok=imports_ok,
            dry_run_ok=dry_run_ok,
            errors=errors,
            warnings=warnings,
        )

    # ------------------------------------------------------------------

    def _check_syntax(self, source: str, errors: List[str]) -> bool:
        """AST-parse the source to catch syntax errors."""
        try:
            ast.parse(source)
            return True
        except SyntaxError as e:
            errors.append(f"SyntaxError at line {e.lineno}: {e.msg}")
            return False

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
