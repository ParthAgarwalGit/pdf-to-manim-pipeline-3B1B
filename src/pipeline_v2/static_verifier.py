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
    # TeX primitives that read or write files, run shell commands, load
    # packages, define macros or rebuild a control sequence from characters
    # (\csname, \scantokens, ^^5c). A superset of the legacy validator's
    # _TEX_FORBIDDEN. The lookahead keeps \infty, \int, \in and \deg legal.
    FORBIDDEN_TEX = re.compile(
        r"\\@{0,2}(?:input|include|includegraphics|InputIfFileExists|lstinputlisting"
        r"|verbatiminput|openout|openin|newread|newwrite|read|readline|write"
        r"|immediate|special|catcode|csname|expandafter|expanded|scantokens"
        r"|directlua|luaexec|ShellEscape|pdffiledump|pdfmdfivesum|usepackage"
        r"|RequirePackage|def|gdef|edef|xdef)(?![A-Za-z])"
        r"|\^\^"
    )
    # Callables that read a file named by their argument. Generated V2 scenes
    # ship no asset files, so a path is either broken or an exfiltration try.
    FILE_ONLY_READERS = {"SVGMobject", "TexTemplateFromFile", "from_file"}
    FILE_OR_ARRAY_READERS = {"ImageMobject"}
    PATH_CONSTRUCTORS = {"Path", "PurePath", "PosixPath", "WindowsPath", "str"}

    def __init__(
        self,
        expected_artifact_ids: set[str],
        expected_beat_ids: set[str],
        *,
        expected_layout: str | None = None,
        expected_regions: dict[str, str] | None = None,
        require_contract_tracking: bool = True,
    ):
        super().__init__(
            trusted_import_roots={"src.pipeline_v2.tracked_scene"},
            require_purpose_comments=False,
        )
        self.expected_artifact_ids = expected_artifact_ids
        self.expected_beat_ids = expected_beat_ids
        self.expected_layout = expected_layout
        self.expected_regions = expected_regions or {}
        self.require_contract_tracking = require_contract_tracking

    def verify(self, source: str) -> VerificationReport:
        report = super().verify(source)
        if not report.syntax_valid:
            return report
        tree = ast.parse(source)
        registered = self._literal_first_args(tree, "register_artifact")
        beats = self._literal_first_args(tree, "set_beat")
        layouts = self._literal_first_args(tree, "set_layout")
        if self.require_contract_tracking:
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
        self._check_forbidden_tex(tree, source, report.issues)
        self._check_file_read_mobjects(tree, report.issues)
        report.safe = not any(issue.severity == "error" for issue in report.issues)
        return report

    def _check_forbidden_tex(
        self, tree: ast.AST, source: str, issues: list[VerificationIssue],
    ) -> None:
        # The raw source catches every spelling written out literally, on its
        # exact line. The parsed strings also catch "\\inp" "ut",
        # "\\inp" + "ut" and "\x5cinput", which only exist after evaluation.
        # "^^" is only looked for inside strings: in a comment it is an arrow.
        hits = {
            (match.group(0), source.count("\n", 0, match.start()) + 1)
            for match in self.FORBIDDEN_TEX.finditer(source)
            if match.group(0) != "^^"
        }
        assembled = set()
        for node in ast.walk(tree):
            text = self._static_string(node)
            if text is None:
                continue
            last_line = getattr(node, "end_lineno", None) or node.lineno
            for match in self.FORBIDDEN_TEX.finditer(text):
                token = match.group(0)
                already_located = any(
                    found == token and node.lineno <= line <= last_line
                    for found, line in hits
                )
                if not already_located:
                    assembled.add((token, node.lineno))
        for token, line in sorted(hits | assembled, key=lambda item: (item[1], item[0])):
            issues.append(VerificationIssue(
                code="FORBIDDEN_TEX", severity="error",
                message=(
                    f"Forbidden TeX {token}: file, shell, package, macro-definition "
                    "and character-escape commands are not allowed in scene text."
                ),
                code_lines=[line],
            ))

    def _check_file_read_mobjects(
        self, tree: ast.AST, issues: list[VerificationIssue],
    ) -> None:
        readers = self.FILE_ONLY_READERS | self.FILE_OR_ARRAY_READERS
        path_names = {
            target.id
            for node in ast.walk(tree) if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and self._is_path_expression(node.value)
        }
        call_targets = {
            id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = self._call_name(node.func).rsplit(".", 1)[-1]
                if name in self.FILE_ONLY_READERS:
                    reads_file = True
                elif name in self.FILE_OR_ARRAY_READERS:
                    argument = node.args[0] if node.args else next(
                        (kw.value for kw in node.keywords if kw.arg == "filename_or_array"),
                        None,
                    )
                    reads_file = argument is not None and (
                        self._is_path_expression(argument)
                        or (isinstance(argument, ast.Name) and argument.id in path_names)
                    )
                else:
                    continue
                message = (
                    f"{name}(...) would read a file; generated scenes have no "
                    "asset files. Build visuals from Manim primitives (an "
                    "ImageMobject may take a pixel array)."
                )
            elif id(node) not in call_targets and (
                (isinstance(node, ast.Name) and node.id in readers)
                or (isinstance(node, ast.Attribute) and node.attr in readers)
            ):
                # Reader = SVGMobject; Reader("keys.env") hides the call above.
                name = node.id if isinstance(node, ast.Name) else node.attr
                reads_file = True
                message = (
                    f"{name} reads files and may only be called directly, "
                    "never aliased or passed around."
                )
            else:
                continue
            if reads_file:
                issues.append(VerificationIssue(
                    code="FILE_READ_MOBJECT", severity="error",
                    message=message, code_lines=[node.lineno],
                ))

    @classmethod
    def _static_string(cls, node: ast.AST) -> str | None:
        """Text of a string literal, f-string or ``+`` chain of them, else None."""
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, str) else None
        if isinstance(node, ast.JoinedStr):
            return "".join(cls._static_string(part) or "" for part in node.values)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = cls._static_string(node.left)
            right = cls._static_string(node.right)
            return left + right if left is not None and right is not None else None
        return None

    @classmethod
    def _is_path_expression(cls, node: ast.AST) -> bool:
        """True for expressions that are obviously a file path, not an array."""
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (str, bytes))
        if isinstance(node, ast.JoinedStr):
            return True
        if isinstance(node, ast.BinOp):  # "dir/" + name, Path(...) / "x.png"
            return cls._is_path_expression(node.left) or cls._is_path_expression(node.right)
        if isinstance(node, ast.Call):
            callee = cls._call_name(node.func)
            return (
                callee.rsplit(".", 1)[-1] in cls.PATH_CONSTRUCTORS
                or callee.endswith((".format", ".join"))
            )
        return False

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
