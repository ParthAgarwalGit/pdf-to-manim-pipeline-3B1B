"""V2 AST and contract verification for tracked raw Manim."""

from __future__ import annotations

import ast
import re

from src.manim_verifier.models import VerificationIssue, VerificationReport
from src.manim_verifier.static_verifier import StaticManimVerifier


class V2StaticVerifier(StaticManimVerifier):
    SCENE_BASES = StaticManimVerifier.SCENE_BASES | {"TrackedScene"}
    UNSUPPORTED_LATEX = {
        r"\centernot": r"Use \nRightarrow or a supported equivalent.",
        r"\cancel": "The default Manim TeX template does not load cancel.sty.",
        r"\begin{align}": r"Use aligned inside MathTex instead of a full align environment.",
    }

    def __init__(
        self,
        expected_artifact_ids: set[str],
        expected_beat_ids: set[str],
        *,
        expected_layout: str | None = None,
        expected_regions: dict[str, str] | None = None,
    ):
        super().__init__(
            trusted_import_roots={"src.pipeline_v2.tracked_scene"},
            require_purpose_comments=False,
        )
        self.expected_artifact_ids = expected_artifact_ids
        self.expected_beat_ids = expected_beat_ids
        self.expected_layout = expected_layout
        self.expected_regions = expected_regions or {}

    def verify(self, source: str) -> VerificationReport:
        report = super().verify(source)
        if not report.syntax_valid:
            return report
        tree = ast.parse(source)
        registered = self._literal_first_args(tree, "register_artifact")
        beats = self._literal_first_args(tree, "set_beat")
        layouts = self._literal_first_args(tree, "set_layout")
        missing_artifacts = self.expected_artifact_ids - registered
        unknown_artifacts = registered - self.expected_artifact_ids
        missing_beats = self.expected_beat_ids - beats
        if missing_artifacts:
            report.issues.append(VerificationIssue(
                code="MISSING_ARTIFACT_REGISTRATION", severity="error",
                message=f"Missing artifact registrations: {sorted(missing_artifacts)}",
                object_ids=sorted(missing_artifacts),
            ))
        if unknown_artifacts:
            report.issues.append(VerificationIssue(
                code="UNPLANNED_ARTIFACT_REGISTRATION", severity="warning",
                message=f"Registered artifacts absent from the contract: {sorted(unknown_artifacts)}",
                object_ids=sorted(unknown_artifacts),
            ))
        if missing_beats:
            report.issues.append(VerificationIssue(
                code="MISSING_BEAT_CHECKPOINT", severity="error",
                message=f"Missing set_beat calls: {sorted(missing_beats)}",
            ))
        if self.expected_layout and self.expected_layout not in layouts:
            report.issues.append(VerificationIssue(
                code="MISSING_LAYOUT_TEMPLATE", severity="error",
                message=f"Expected self.set_layout({self.expected_layout!r}).",
            ))
        registered_variables = self._registered_variables(tree)
        placed_variables = self._placed_variables(tree)
        for artifact_id, expected_region in self.expected_regions.items():
            variable = registered_variables.get(artifact_id)
            if not variable or variable not in placed_variables:
                report.issues.append(VerificationIssue(
                    code="MISSING_REGION_PLACEMENT", severity="error",
                    message=f"Artifact {artifact_id!r} is not placed with self.place_in_region().",
                    object_ids=[artifact_id],
                ))
                continue
            actual_region = placed_variables[variable]
            if actual_region != expected_region:
                report.issues.append(VerificationIssue(
                    code="REGION_PLACEMENT_MISMATCH", severity="error",
                    message=(
                        f"Artifact {artifact_id!r} is placed in {actual_region!r}; "
                        f"the contract requires {expected_region!r}."
                    ),
                    object_ids=[artifact_id],
                ))
        for macro, guidance in self.UNSUPPORTED_LATEX.items():
            for match in re.finditer(re.escape(macro), source):
                line = source.count("\n", 0, match.start()) + 1
                report.issues.append(VerificationIssue(
                    code="UNSUPPORTED_LATEX", severity="error",
                    message=f"Unsupported LaTeX {macro}. {guidance}", code_lines=[line],
                ))
        report.safe = not any(issue.severity == "error" for issue in report.issues)
        return report

    @staticmethod
    def _literal_first_args(tree: ast.AST, method: str) -> set[str]:
        values = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = StaticManimVerifier._call_name(node.func).rsplit(".", 1)[-1]
            if name == method and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                values.add(node.args[0].value)
        return values

    @staticmethod
    def _registered_variables(tree: ast.AST) -> dict[str, str]:
        values = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            name = StaticManimVerifier._call_name(node.func).rsplit(".", 1)[-1]
            artifact = node.args[0]
            variable = node.args[1]
            if (
                name == "register_artifact"
                and isinstance(artifact, ast.Constant)
                and isinstance(artifact.value, str)
                and isinstance(variable, ast.Name)
            ):
                values[artifact.value] = variable.id
        return values

    @staticmethod
    def _placed_variables(tree: ast.AST) -> dict[str, str]:
        values = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            name = StaticManimVerifier._call_name(node.func).rsplit(".", 1)[-1]
            variable = node.args[0]
            region = node.args[1]
            if (
                name == "place_in_region"
                and isinstance(variable, ast.Name)
                and isinstance(region, ast.Constant)
                and isinstance(region.value, str)
            ):
                values[variable.id] = region.value
        return values
