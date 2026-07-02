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
