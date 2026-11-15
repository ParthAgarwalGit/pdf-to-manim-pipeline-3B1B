"""Tests for the Manim Code Validator's security gate.

Generated scene code is executed locally by `manim`, and its content is
influenced by text from arbitrary user PDFs (prompt-injection surface).
The security gate must reject filesystem/network/process/reflection access
as ERRORS so the orchestrator regenerates the scene.
"""

from datetime import datetime

from src.agents.manim_code_validator import (
    ALLOWED_IMPORT_ROOTS,
    ManimCodeValidatorAgent,
    ManimValidatorFailure,
    ManimValidatorInput,
    ManimValidatorOutput,
)
from src.schemas.animation_code import AnimationCodeObject


def make_code_obj(code: str, scene_id: str = "scene_001", imports=None) -> AnimationCodeObject:
    return AnimationCodeObject(
        scene_id=scene_id,
        code=code,
        imports=imports if imports is not None else ["from manim import *"],
        class_name="TestScene",
        narration="Test narration.",
        duration_estimate=10.0,
        generation_timestamp=datetime.now(),
    )


CLEAN_SCENE = """
class TestScene(Scene):
    def construct(self):
        c = Circle(radius=2.0, color=BLUE)
        self.play(Create(c), run_time=1.5)
        self.wait(1.0)
"""


def run_single(code: str, imports=None):
    agent = ManimCodeValidatorAgent()
    result = agent.execute(ManimValidatorInput(
        animation_code_objects=[make_code_obj(code, imports=imports)],
    ))
    return result


class TestSecurityGate:
    def test_clean_scene_passes(self):
        result = run_single(CLEAN_SCENE)
        assert isinstance(result, ManimValidatorOutput)
        assert result.results[0].security_ok is True

    def test_repo_helper_imports_are_allowed(self):
        result = run_single(CLEAN_SCENE, imports=[
            "from manim import *",
            "from src.utils.layout_tracker import LayoutTracker",
            "import numpy as np",
            "import math",
        ])
        assert isinstance(result, ManimValidatorOutput)

    def test_os_import_is_rejected(self):
        bad = CLEAN_SCENE.replace(
            "        self.wait(1.0)",
            "        import os\n        os.system('echo pwned')",
        )
        result = run_single(bad)
        assert isinstance(result, ManimValidatorFailure)
        r = result.results[0]
        assert r.security_ok is False
        assert r.passed is False
        assert any("Security" in e and "os" in e for e in r.errors)

    def test_subprocess_from_import_is_rejected(self):
        bad = "from subprocess import run\n" + CLEAN_SCENE
        result = run_single(bad)
        assert isinstance(result, ManimValidatorFailure)
        assert result.results[0].security_ok is False

    def test_eval_call_is_rejected(self):
        bad = CLEAN_SCENE.replace(
            "        self.wait(1.0)",
            "        eval('1+1')",
        )
        result = run_single(bad)
        assert isinstance(result, ManimValidatorFailure)
        assert any("eval" in e for e in result.results[0].errors)

    def test_open_call_is_rejected(self):
        bad = CLEAN_SCENE.replace(
            "        self.wait(1.0)",
            "        open('secrets.txt').read()",
        )
        result = run_single(bad)
        assert isinstance(result, ManimValidatorFailure)

    def test_dunder_reflection_escape_is_rejected(self):
        bad = CLEAN_SCENE.replace(
            "        self.wait(1.0)",
            "        x = ().__class__.__mro__",
        )
        result = run_single(bad)
        assert isinstance(result, ManimValidatorFailure)
        assert any("__mro__" in e for e in result.results[0].errors)

    def test_failed_scene_is_routed_to_failed_objects(self):
        agent = ManimCodeValidatorAgent()
        good = make_code_obj(CLEAN_SCENE, scene_id="scene_001")
        bad = make_code_obj(
            "import socket\n" + CLEAN_SCENE, scene_id="scene_002",
        )
        result = agent.execute(ManimValidatorInput(
            animation_code_objects=[good, bad],
        ))
        assert isinstance(result, ManimValidatorFailure)
        assert [o.scene_id for o in result.passed_objects] == ["scene_001"]
        assert [o.scene_id for o in result.failed_objects] == ["scene_002"]

    def test_allowlist_covers_generated_import_roots(self):
        # Every import the animation generator emits must be allowed, or the
        # gate would reject all real output.
        from src.agents.animation_generator import AnimationCodeGeneratorAgent
        agent = AnimationCodeGeneratorAgent.__new__(AnimationCodeGeneratorAgent)
        for line in agent._get_imports():
            root = line.replace("from ", "").replace("import ", "").split(".")[0].split()[0]
            assert root in ALLOWED_IMPORT_ROOTS, f"generated import root {root!r} not allowed"


class TestLatexGate:
    """Deterministic LaTeX gate: text-mode Tex() carrying math tokens dies in
    the LaTeX subprocess at render time (14% of a real run's video). Fixable
    cases are rewritten in place; ambiguous ones reject into regeneration."""

    def _validate(self, body: str):
        agent = ManimCodeValidatorAgent()
        obj = make_code_obj(
            "class TestScene(Scene):\n    def construct(self):\n" + body
        )
        result = agent.execute(ManimValidatorInput(animation_code_objects=[obj]))
        return result, obj

    def test_mathy_tex_is_auto_wrapped_and_passes(self):
        result, obj = self._validate(
            '        t = Tex("|a_j|^2 = \\sum_i (a_j \\cdot v_i)^2")\n'
        )
        assert isinstance(result, ManimValidatorOutput)
        assert 'r"$' in obj.code and '$"' in obj.code  # wrapped in math mode
        assert any("LaTeX gate: auto-wrapped" in w for w in result.results[0].warnings)

    def test_plain_prose_tex_untouched(self):
        body = '        t = Tex("Greedy build-up of best-fit subspaces")\n'
        result, obj = self._validate(body)
        assert isinstance(result, ManimValidatorOutput)
        assert "$" not in obj.code

    def test_balanced_span_tex_untouched(self):
        body = '        t = Tex(r"the slope $m$ of the line")\n'
        result, obj = self._validate(body)
        assert isinstance(result, ManimValidatorOutput)
        assert obj.code.count("$") == 2

    def test_unbalanced_dollar_rejected(self):
        result, _ = self._validate('        t = Tex(r"cost is $5 per unit")\n')
        assert isinstance(result, ManimValidatorFailure)
        assert any("unbalanced" in e for e in result.results[0].errors)

    def test_mixed_spans_and_outside_math_rejected(self):
        result, _ = self._validate(
            '        t = Tex(r"$x$ squared is x^2 always")\n'
        )
        assert isinstance(result, ManimValidatorFailure)
        assert any("OUTSIDE" in e for e in result.results[0].errors)

    def test_mathtex_left_alone(self):
        body = '        t = MathTex(r"\\sum_i a_i^2")\n'
        result, obj = self._validate(body)
        assert isinstance(result, ManimValidatorOutput)
        assert "$" not in obj.code

    def test_fixed_code_still_syntactically_valid(self):
        import ast as _ast
        _, obj = self._validate(
            '        a = Tex("x_1 + x_2")\n        b = Tex("y^3")\n'
        )
        _ast.parse(obj.code)  # must not raise

    def test_subscript_macro_detection(self):
        result, obj = self._validate('        t = Tex(r"\\frac{a}{b} of the total")\n')
        assert isinstance(result, ManimValidatorOutput)
        assert obj.code.count("$") == 2  # wrapped


class TestPlacementNormalize:
    """Deterministic off-screen placement clamp (P1.5): move_to literals and
    oversized shifts are pulled back into the safe area — fixes the empty-frame
    class without touching the LLM. Auto-fix only, never rejects."""

    def _fix(self, body: str):
        agent = ManimCodeValidatorAgent()
        obj = make_code_obj(
            "class TestScene(Scene):\n    def construct(self):\n" + body
        )
        result = agent.execute(ManimValidatorInput(animation_code_objects=[obj]))
        return result, obj

    def test_offscreen_move_to_clamped(self):
        result, obj = self._fix('        c.move_to([9.0, -6.0, 0])\n')
        assert isinstance(result, ManimValidatorOutput)
        assert "[6.5, -3.5, 0" in obj.code
        assert any("clamped move_to" in w for w in result.results[0].warnings)

    def test_in_bounds_move_to_untouched(self):
        result, obj = self._fix('        c.move_to([2.0, -1.5, 0])\n')
        assert "[2.0, -1.5, 0]" in obj.code

    def test_expression_move_to_untouched(self):
        # get_center() must not be clamped (it's a computed anchor).
        result, obj = self._fix('        arrow.move_to(dot.get_center())\n')
        assert "dot.get_center()" in obj.code

    def test_oversized_vertical_shift_capped(self):
        result, obj = self._fix('        g.shift(UP * 8)\n')
        assert isinstance(result, ManimValidatorOutput)
        assert "UP * 3.0" in obj.code

    def test_oversized_horizontal_shift_capped(self):
        result, obj = self._fix('        g.shift(RIGHT * 9.5)\n')
        assert "RIGHT * 6.0" in obj.code

    def test_reasonable_shift_untouched(self):
        result, obj = self._fix('        g.shift(LEFT * 2)\n')
        assert "LEFT * 2" in obj.code

    def test_scalar_first_shift_form_capped(self):
        result, obj = self._fix('        g.shift(7 * DOWN)\n')
        assert "3.0 * DOWN" in obj.code or "DOWN" in obj.code and "7" not in obj.code

    def test_fixed_code_stays_valid_python(self):
        import ast as _ast
        _, obj = self._fix('        a.move_to([10, 10, 0])\n        b.shift(UP * 12)\n')
        _ast.parse(obj.code)

    def test_non_literal_z_left_untouched(self):
        # A clamp must never silently replace an expression element with 0.
        _, obj = self._fix('        a.move_to([9.0, 0, z_var])\n')
        assert "z_var" in obj.code
        assert "9.0" in obj.code  # x not clamped either — whole point skipped

    def test_multiline_balanced_spans_untouched(self):
        # $-spans that cross a newline are still spans (re.S), not "math
        # outside spans" — must not be rejected.
        body = '        t = Tex(r"""the sum $a_1 +\n a_2$ of terms""")\n'
        result, obj = self._fix(body)
        assert isinstance(result, ManimValidatorOutput)


class TestLatexMacroBlocklist:
    r"""File/expansion LaTeX macros reach a local latex subprocess; \input can
    read arbitrary files INTO the rendered video. Reject, never auto-fix."""

    def _validate(self, body: str):
        agent = ManimCodeValidatorAgent()
        obj = make_code_obj(
            "class TestScene(Scene):\n    def construct(self):\n" + body
        )
        return agent.execute(ManimValidatorInput(animation_code_objects=[obj]))

    def test_input_macro_rejected(self):
        result = self._validate('        t = Tex(r"\\input{secrets.txt}")\n')
        assert isinstance(result, ManimValidatorFailure)
        assert any("Security" in e and "input" in e for e in result.results[0].errors)

    def test_write18_rejected(self):
        result = self._validate('        t = MathTex(r"\\write18{whoami}")\n')
        assert isinstance(result, ManimValidatorFailure)

    def test_openout_rejected(self):
        result = self._validate('        t = Tex(r"\\openout1=x.txt")\n')
        assert isinstance(result, ManimValidatorFailure)

    def test_plain_math_macros_still_fine(self):
        # \int (integral) must not trip the \input blocklist prefix.
        result = self._validate('        t = MathTex(r"\\int_0^1 x\\,dx")\n')
        assert isinstance(result, ManimValidatorOutput)


class TestSecurityGateBypasses:
    """Constructs that slipped past the call-name check before."""

    def _validate(self, body: str):
        agent = ManimCodeValidatorAgent()
        obj = make_code_obj(
            "import numpy as np\n"
            "class TestScene(Scene):\n    def construct(self):\n" + body
        )
        return agent.execute(ManimValidatorInput(animation_code_objects=[obj]))

    def _errors(self, result):
        return result.results[0].errors

    def test_aliased_exec_rejected(self):
        result = self._validate("        e = exec\n        e('imp' + 'ort os')\n")
        assert isinstance(result, ManimValidatorFailure)
        assert any("aliasing" in e for e in self._errors(result))

    def test_getattr_reflection_rejected(self):
        result = self._validate(
            "        g = getattr(Scene.__init__, '__glob' + 'als__')\n"
        )
        assert isinstance(result, ManimValidatorFailure)

    def test_numpy_file_read_rejected(self):
        result = self._validate("        x = np.loadtxt('keys.env', dtype=str)\n")
        assert isinstance(result, ManimValidatorFailure)

    def test_code_mobject_file_read_rejected(self):
        result = self._validate("        c = Code('keys.env')\n        self.add(c)\n")
        assert isinstance(result, ManimValidatorFailure)
        assert any("code_string" in e for e in self._errors(result))

    def test_inline_code_string_still_allowed(self):
        result = self._validate(
            "        c = Code(code_string='x = 1', language='python')\n        self.add(c)\n"
        )
        assert isinstance(result, ManimValidatorOutput)
