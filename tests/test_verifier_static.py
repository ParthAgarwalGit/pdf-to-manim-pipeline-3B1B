from src.manim_verifier.static_verifier import StaticManimVerifier


SAFE_SCRIPT = """from manim import *

class Demo(Scene):
    def construct(self):
        self.play(Create(Circle()))
"""


def test_safe_manim_script_is_accepted():
    report = StaticManimVerifier().verify(SAFE_SCRIPT)
    assert report.safe
    assert report.syntax_valid
    assert report.scene_class == "Demo"


def test_syntax_error_is_rejected():
    report = StaticManimVerifier().verify("class Broken(Scene)\n    pass")
    assert not report.safe
    assert not report.syntax_valid
    assert report.issues[0].code == "SYNTAX_ERROR"


def test_multiple_scene_classes_are_rejected():
    source = SAFE_SCRIPT + "\nclass Other(Scene):\n    def construct(self):\n        pass\n"
    report = StaticManimVerifier().verify(source)
    assert not report.safe
    assert any(issue.code == "SCENE_CLASS_COUNT" for issue in report.issues)


def test_dangerous_import_and_calls_are_rejected():
    source = """from manim import *
import subprocess
class Demo(Scene):
    def construct(self):
        open('secret.txt')
        exec('x = 1')
        subprocess.run(['whoami'])
"""
    report = StaticManimVerifier().verify(source)
    codes = {issue.code for issue in report.issues}
    assert not report.safe
    assert "IMPORT_NOT_ALLOWED" in codes
    assert "DANGEROUS_CALL" in codes


def test_obvious_infinite_loop_is_rejected():
    source = SAFE_SCRIPT.replace(
        "self.play(Create(Circle()))",
        "while True:\n            self.wait()",
    )
    report = StaticManimVerifier().verify(source)
    assert any(issue.code == "OBVIOUS_INFINITE_LOOP" for issue in report.issues)


def test_numeric_infinite_loop_and_reflective_file_access_are_rejected():
    source = SAFE_SCRIPT.replace(
        "self.play(Create(Circle()))",
        "while 1:\n            getattr(__builtins__, 'open')('secret.txt')",
    )
    report = StaticManimVerifier().verify(source)
    codes = {issue.code for issue in report.issues}
    assert "OBVIOUS_INFINITE_LOOP" in codes
    assert "DANGEROUS_CALL" in codes
    assert "DANGEROUS_NAME" in codes


def test_repo_helpers_require_explicit_trust():
    source = SAFE_SCRIPT.replace(
        "from manim import *",
        "from manim import *\nfrom src.utils.layout_tracker import LayoutTracker",
    )
    assert not StaticManimVerifier().verify(source).safe
    assert StaticManimVerifier(trusted_import_roots={"src.utils"}).verify(source).safe


def test_purpose_comments_can_be_required():
    verifier = StaticManimVerifier(require_purpose_comments=True)
    missing = verifier.verify(SAFE_SCRIPT)
    assert any(issue.code == "PURPOSE_COMMENT_MISSING" for issue in missing.issues)

    documented = SAFE_SCRIPT.replace(
        "self.play(Create(Circle()))",
        "# Purpose [beat_001]: show the geometric object\n"
        "        circle = Circle()\n"
        "        self.play(Create(circle))",
    )
    assert verifier.verify(documented).safe

    empty_reason = documented.replace(
        "# Purpose [beat_001]: show the geometric object",
        "# Purpose []: ",
    )
    assert not verifier.verify(empty_reason).safe


def test_literal_offscreen_and_axes_coordinates_are_reported_without_rewriting():
    source = """from manim import *
class Demo(Scene):
    def construct(self):
        axes = Axes()
        point = Dot([9, 1, 0])
        self.add(axes, point)
"""
    report = StaticManimVerifier().verify(source)
    codes = {issue.code for issue in report.issues}
    assert report.safe
    assert "OFFSCREEN_LITERAL_COORDINATE" in codes
    assert "RAW_COORDINATE_WITH_AXES" in codes
    assert "[9, 1, 0]" in source


def _verify_body(body):
    source = (
        "from manim import *\nimport numpy as np\n\n"
        "class Demo(Scene):\n    def construct(self):\n" + body
    )
    return StaticManimVerifier().verify(source)


def test_aliased_exec_is_rejected():
    report = _verify_body("        e = exec\n        e('imp' + 'ort os')\n")
    assert not report.safe
    assert any(issue.code == "DANGEROUS_NAME" for issue in report.issues)


def test_numpy_file_read_is_rejected():
    report = _verify_body("        x = np.loadtxt('keys.env', dtype=str)\n")
    assert not report.safe


def test_code_mobject_file_read_is_rejected():
    report = _verify_body("        c = Code('keys.env')\n")
    assert not report.safe
    inline = _verify_body("        c = Code(code_string='x = 1')\n        self.add(c)\n")
    assert inline.safe


def test_bare_connect_helper_is_not_a_dangerous_attribute():
    report = _verify_body(
        "        a = Dot()\n        b = Dot()\n        line = connect(a, b)\n"
        "        self.add(a, b, line)\n"
    )
    assert not any(issue.code == "DANGEROUS_ATTRIBUTE" for issue in report.issues)
    sock = _verify_body("        s = Dot()\n        s.connect(('example.com', 80))\n")
    assert any(issue.code == "DANGEROUS_ATTRIBUTE" for issue in sock.issues)


def test_purpose_comment_covers_nested_multiline_constructor():
    source = (
        "from manim import *\n\n"
        "class Demo(Scene):\n    def construct(self):\n"
        "        # Purpose [beat_001]: label the two forces together\n"
        "        labels = VGroup(\n"
        "            Text('push'),\n"
        "            Text('pull'),\n"
        "        )\n"
        "        self.add(labels)\n"
    )
    report = StaticManimVerifier(require_purpose_comments=True).verify(source)
    assert not any(issue.code == "PURPOSE_COMMENT_MISSING" for issue in report.issues)
