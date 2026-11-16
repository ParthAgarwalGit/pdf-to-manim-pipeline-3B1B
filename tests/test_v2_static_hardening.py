"""The V2 AST gate rejects file, shell and reflection tricks in tracked scenes."""

import pytest

from src.pipeline_v2.static_verifier import V2StaticVerifier


HEADER = '''from manim import *
import numpy as np
from src.pipeline_v2.tracked_scene import TrackedScene

'''

SCENE = HEADER + '''class Demo(TrackedScene):
    def construct(self):
        self.set_layout("full_visual")
        dot = Dot()
        self.place_in_region(dot, "content_full")
        self.register_artifact("dot", dot, category="primary_visual", purpose="Show point", beat_ids=["b1"])
        self.set_beat("b1")
        self.play(FadeIn(dot))
'''


def _verify(body: str, *, header: str = HEADER):
    """Verify a tracked scene whose construct() starts with ``body``."""
    indented = "\n".join(f"        {line}" for line in body.splitlines())
    source = SCENE.replace(HEADER, header).replace(
        "        dot = Dot()\n", f"{indented}\n        dot = Dot()\n",
    )
    return V2StaticVerifier(
        {"dot"}, {"b1"}, expected_layout="full_visual",
        expected_regions={"dot": "content_full"},
    ).verify(source)


def _codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def test_plain_tracked_scene_is_still_safe():
    report = V2StaticVerifier(
        {"dot"}, {"b1"}, expected_layout="full_visual",
        expected_regions={"dot": "content_full"},
    ).verify(SCENE)
    assert report.safe
    assert report.scene_class == "Demo"
    assert not [issue for issue in report.issues if issue.severity == "error"]


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ("e = exec\ne('print(1)')", "DANGEROUS_NAME"),
        ("exec('print(1)')", "DANGEROUS_CALL"),
        ("secret = getattr(self, 'renderer')", "DANGEROUS_CALL"),
        ("g = getattr", "DANGEROUS_NAME"),
        ("rows = np.loadtxt('keys.env', dtype=str)", "DANGEROUS_ATTRIBUTE"),
        ("rows = np.load('weights.npy')", "DANGEROUS_ATTRIBUTE"),
        ("loader = np.loadtxt", "DANGEROUS_ATTRIBUTE"),
        ("listing = Code('keys.env')", "DANGEROUS_CALL"),
        ("listing = Code(code_file='keys.env')", "DANGEROUS_CALL"),
        ("text = open('keys.env').read()", "DANGEROUS_CALL"),
        ("module = __import__('os')", "DANGEROUS_CALL"),
        ("home = os.environ['HOME']", "DANGEROUS_NAME"),
        ("subprocess.run(['whoami'])", "DANGEROUS_CALL"),
    ],
)
def test_dangerous_python_is_rejected(body, code):
    report = _verify(body)
    assert code in _codes(report)
    assert not report.safe


def test_inline_code_string_is_allowed():
    report = _verify("listing = Code(code_string='print(1)', language='python')")
    assert report.safe, report.issues


@pytest.mark.parametrize("statement", ["import os", "import subprocess", "from os import environ"])
def test_os_and_subprocess_imports_are_rejected(statement):
    report = _verify("pass", header=HEADER.replace("import numpy as np", statement))
    assert "IMPORT_NOT_ALLOWED" in _codes(report)
    assert not report.safe


@pytest.mark.parametrize(
    "body",
    [
        r'label = Tex(r"\input{keys.env}")',
        r'label = Tex(r"\include{secrets}")',
        r'label = Tex(r"\immediate\write18{curl example.invalid}")',
        r'label = Tex(r"\write18{id}")',
        r'label = Tex(r"\openout3=out.txt")',
        r'label = Tex(r"\openin3=keys.env \read3 to \line")',
        r'label = Tex(r"\catcode`\@=11")',
        r'label = Tex(r"\directlua{os.execute()}")',
        r'label = Tex(r"\usepackage{shellesc}")',
        r'label = Tex(r"\includegraphics{secret.png}")',
        r'label = Tex(r"\csname input\endcsname keys.env")',
        r'label = Tex(r"\@@input keys.env")',
        # Spellings that only become \input after Python evaluates the string.
        'label = Tex("\\\\input{keys.env}")',
        'label = Tex("\\\\inp" "ut{keys.env}")',
        'label = Tex("\\\\inp" + "ut{keys.env}")',
        'label = Tex("\\x5cinput{keys.env}")',
        # TeX reads ^^5c as a backslash before it looks for control sequences.
        r'label = Tex(r"^^5cinput{keys.env}")',
    ],
)
def test_forbidden_tex_is_rejected(body):
    report = _verify(body)
    findings = [issue for issue in report.issues if issue.code == "FORBIDDEN_TEX"]
    assert findings
    assert all(issue.severity == "error" for issue in findings)
    assert all(issue.code_lines for issue in findings)
    assert not report.safe


def test_forbidden_tex_reports_the_offending_line():
    source = SCENE.replace(
        "        dot = Dot()\n",
        '        label = Tex(r"""first line\n'
        '            \\input{keys.env}""")\n'
        "        dot = Dot()\n",
    )
    report = V2StaticVerifier({"dot"}, {"b1"}).verify(source)
    lines = [issue.code_lines for issue in report.issues if issue.code == "FORBIDDEN_TEX"]
    expected = source.splitlines().index("            \\input{keys.env}\"\"\")") + 1
    assert lines == [[expected]]


@pytest.mark.parametrize(
    "body",
    [
        r'formula = MathTex(r"\int_0^\infty e^{-x}\,dx = 1")',
        r'formula = MathTex(r"x \in \mathbb{R}, \quad \delta \leq \infty")',
        r'formula = MathTex(r"\lim_{n \to \infty} \inf_k a_k \in A")',
        r'formula = MathTex(r"\deg(p) \le 3 \implies \int p \, dx")',
        r'formula = MathTex(r"\begin{aligned} a &= b \\ c &= d \end{aligned}")',
        'formula = MathTex("x^2 + y^2")  # ^^ the Pythagorean form',
    ],
)
def test_ordinary_math_is_not_flagged(body):
    report = _verify(body)
    assert "FORBIDDEN_TEX" not in _codes(report)
    assert report.safe, report.issues


@pytest.mark.parametrize(
    "body",
    [
        "icon = SVGMobject('x.svg')",
        "icon = SVGMobject(file_name='x.svg')",
        "picture = ImageMobject('x.png')",
        "picture = ImageMobject(filename_or_array='x.png')",
        "index = 1\npicture = ImageMobject(f'frame{index}.png')",
        "picture = ImageMobject(Path('x.png'))",
        "name = 'keys.env'\npicture = ImageMobject(name)",
        "Reader = SVGMobject\nicon = Reader('x.svg')",
        "Reader = ImageMobject\npicture = Reader('x.png')",
        "template = TexTemplate.from_file('keys.env')",
    ],
)
def test_file_reading_mobjects_are_rejected(body):
    report = _verify(body)
    findings = [issue for issue in report.issues if issue.code == "FILE_READ_MOBJECT"]
    assert findings
    assert all(issue.severity == "error" and issue.code_lines for issue in findings)
    assert not report.safe


@pytest.mark.parametrize(
    "body",
    [
        "picture = ImageMobject(np.zeros((2, 2, 3)))",
        "pixels = np.uint8(np.zeros((2, 2, 3)))\npicture = ImageMobject(pixels)",
    ],
)
def test_image_mobject_from_a_pixel_array_is_allowed(body):
    report = _verify(body)
    assert "FILE_READ_MOBJECT" not in _codes(report)
    assert report.safe, report.issues


def test_contract_checks_still_fire_alongside_the_new_gates():
    report = V2StaticVerifier(
        {"dot", "label"}, {"b1", "b2"}, require_contract_tracking=True,
    ).verify(SCENE.replace(
        "        self.play(FadeIn(dot))",
        '        icon = SVGMobject("x.svg")\n'
        '        bad = MathTex(r"\\centernot\\implies")\n'
        "        self.play(FadeIn(dot))",
    ))
    codes = _codes(report)
    assert {
        "MISSING_ARTIFACT_REGISTRATION", "MISSING_BEAT_CHECKPOINT",
        "UNSUPPORTED_LATEX", "FILE_READ_MOBJECT",
    } <= codes
    missing = next(i for i in report.issues if i.code == "MISSING_ARTIFACT_REGISTRATION")
    assert missing.object_ids == ["label"]
    assert not report.safe


def test_contract_checks_stay_off_without_contract_tracking():
    report = V2StaticVerifier(
        {"dot", "label"}, {"b1", "b2"}, require_contract_tracking=False,
    ).verify(SCENE)
    assert report.safe
    assert not {"MISSING_ARTIFACT_REGISTRATION", "MISSING_BEAT_CHECKPOINT"} & _codes(report)


def test_layout_and_region_checks_are_unchanged():
    report = V2StaticVerifier(
        {"dot"}, {"b1"}, expected_layout="split",
        expected_regions={"dot": "content_left"},
    ).verify(SCENE)
    assert {"MISSING_LAYOUT_TEMPLATE", "REGION_PLACEMENT_MISMATCH"} <= _codes(report)
    assert not report.safe


def test_safe_is_recomputed_after_the_new_checks():
    # The base verifier alone accepts this source; only the V2 TeX gate objects.
    report = _verify(r'label = Tex(r"\input{keys.env}")')
    assert _codes(report) == {"FORBIDDEN_TEX"}
    assert report.syntax_valid and not report.safe
