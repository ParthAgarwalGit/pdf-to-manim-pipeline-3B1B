# Reusable Manim Primitives And Helper Layer Implementation Plan

> **Status:** ✅ COMPLETED — shipped across commits `a0365a4` (coordinate
> system helpers), `d7b6829` (notation helpers), `1fda037` (pi_creature role
> helpers), `6717159` (wired helper layer into animation generator). The
> `src/utils/coordinate_system_helpers.py`, `notation_helpers.py`, and
> `pi_characters.py` modules are the artifacts. Retained as historical
> context for the helper-layer architecture.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a reusable Manim helper layer for coordinate systems, notation, and character roles so the pipeline compiles recurring visual patterns through deterministic helper functions instead of ad hoc prompt text or anonymous object code.

**Architecture:** This stage adds a small runtime helper surface inspired by the 3Blue1Brown `manimlib` model: reusable coordinate systems, helper-backed notation objects, and role-aware character utilities. The helper layer then becomes the target for `animation_generator.py`, shared continuity state in `scene_controller.py`, and semantic checks in `video_quality_reviewer.py`, so the pipeline can reuse the same visual grammar across scenes and retries.

**Tech Stack:** Python 3.12, Manim Community, Pydantic v2, pytest, ffprobe, reference semantics from installed `manimgl` / `manimlib`.

---

## File Structure

- Create: `C:/Users/Parth/Documents/3B1B/src/utils/coordinate_system_helpers.py`
  Responsibility: runtime helpers for reusable axes, planes, axis labels, and graph attachment.
- Create: `C:/Users/Parth/Documents/3B1B/src/utils/notation_helpers.py`
  Responsibility: runtime helpers for stable MathTex creation, isolated symbols, notation recoloring, and transform targets.
- Modify: `C:/Users/Parth/Documents/3B1B/src/utils/pi_characters.py`
  Responsibility: deterministic role-aware Pi-creature helpers for teacher and student roles.
- Modify: `C:/Users/Parth/Documents/3B1B/src/utils/manim_engine_contract.py`
  Responsibility: tell planner and code generator when to prefer the new helper functions.
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/animation_generator.py`
  Responsibility: compile visual intent through helper imports, stable object registries, and helper-backed transforms.
- Modify: `C:/Users/Parth/Documents/3B1B/src/schemas/scene_processing.py`
  Responsibility: carry reusable coordinate-system, notation, and character bindings in shared context.
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/scene_controller.py`
  Responsibility: persist helper-friendly continuity state between scenes.
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/video_quality_reviewer.py`
  Responsibility: detect when generated code violates helper-backed visual semantics.
- Create: `C:/Users/Parth/Documents/3B1B/tests/test_coordinate_system_helpers.py`
- Create: `C:/Users/Parth/Documents/3B1B/tests/test_notation_helpers.py`
- Create: `C:/Users/Parth/Documents/3B1B/tests/test_pi_characters.py`
- Modify: `C:/Users/Parth/Documents/3B1B/tests/test_animation_generator.py`
- Modify: `C:/Users/Parth/Documents/3B1B/tests/test_scene_controller.py`
- Create or Modify: `C:/Users/Parth/Documents/3B1B/tests/test_video_quality_reviewer.py`

### Task 1: Add Coordinate-System Helpers

**Files:**
- Create: `C:/Users/Parth/Documents/3B1B/src/utils/coordinate_system_helpers.py`
- Modify: `C:/Users/Parth/Documents/3B1B/src/utils/manim_engine_contract.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_coordinate_system_helpers.py`

- [ ] **Step 1: Write failing tests for reusable coordinate systems**

```python
from src.utils.coordinate_system_helpers import (
    attach_axis_labels,
    create_coordinate_system,
    plot_function_on_system,
)


def test_create_coordinate_system_supports_axes_and_number_plane():
    axes = create_coordinate_system(
        kind="axes",
        x_range=(-4, 4, 1),
        y_range=(-2, 2, 1),
        x_length=5,
        y_length=3,
    )
    plane = create_coordinate_system(
        kind="number_plane",
        x_range=(-4, 4, 1),
        y_range=(-2, 2, 1),
        x_length=5,
        y_length=3,
    )
    assert axes.__class__.__name__ == "Axes"
    assert plane.__class__.__name__ == "NumberPlane"


def test_attach_axis_labels_returns_group_with_labels():
    axes = create_coordinate_system(kind="axes")
    labelled = attach_axis_labels(axes, x_label="x", y_label="y")
    assert labelled.__class__.__name__ == "VGroup"
    assert len(labelled.submobjects) == 3


def test_plot_function_on_system_returns_curve():
    axes = create_coordinate_system(kind="axes")
    curve = plot_function_on_system(axes, lambda x: x**2, color="YELLOW")
    assert curve is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_coordinate_system_helpers.py -q`

Expected: FAIL because the helper module does not exist yet.

- [ ] **Step 3: Implement the coordinate-system helper module**

```python
from manim import Axes, ComplexPlane, MathTex, NumberLine, NumberPlane, ThreeDAxes, VGroup


def create_coordinate_system(
    kind: str = "axes",
    x_range=(-6, 6, 1),
    y_range=(-4, 4, 1),
    x_length: float = 5.5,
    y_length: float = 3.2,
):
    if kind == "number_plane":
        return NumberPlane(
            x_range=x_range,
            y_range=y_range,
            x_length=x_length,
            y_length=y_length,
        )
    if kind == "complex_plane":
        return ComplexPlane(
            x_range=x_range,
            y_range=y_range,
            x_length=x_length,
            y_length=y_length,
        )
    if kind == "number_line":
        return NumberLine(x_range=x_range, length=x_length)
    if kind == "three_d_axes":
        return ThreeDAxes(x_range=x_range, y_range=y_range)
    return Axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length)


def attach_axis_labels(system, x_label: str = "x", y_label: str = "y"):
    labels = VGroup()
    if hasattr(system, "get_x_axis_label"):
        labels.add(system.get_x_axis_label(MathTex(x_label)))
    if hasattr(system, "get_y_axis_label"):
        labels.add(system.get_y_axis_label(MathTex(y_label)))
    return VGroup(system, *labels)


def plot_function_on_system(system, fn, color="YELLOW"):
    if hasattr(system, "plot"):
        return system.plot(fn, color=color)
    raise TypeError(f"{system.__class__.__name__} does not support plot(...)")
```

- [ ] **Step 4: Update the engine contract to prefer the helper**

```python
PREFERRED_HELPER_PATTERNS = (
    "create_coordinate_system(kind=...) for reusable axes, planes, and number lines",
    "attach_axis_labels(system, x_label=..., y_label=...) for deterministic axis labelling",
    "plot_function_on_system(system, fn, color=...) when graphing on a reusable coordinate frame",
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_coordinate_system_helpers.py -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/utils/coordinate_system_helpers.py src/utils/manim_engine_contract.py tests/test_coordinate_system_helpers.py
git commit -m "feat: add reusable coordinate system helpers"
```

### Task 2: Add Notation Helpers For Stable MathTex Reuse

**Files:**
- Create: `C:/Users/Parth/Documents/3B1B/src/utils/notation_helpers.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_notation_helpers.py`

- [ ] **Step 1: Write failing tests for notation helper behavior**

```python
from src.utils.notation_helpers import (
    create_notation,
    create_notation_transform_target,
)


def test_create_notation_supports_isolated_substrings_and_color_map():
    expr = create_notation(
        tex=r"x' = \\Sigma x",
        notation_key="projected_vector",
        substrings_to_isolate=[r"x'", r"\\Sigma"],
        tex_to_color_map={r"\\Sigma": "YELLOW"},
    )
    assert expr.tex_string == r"x' = \\Sigma x"


def test_create_notation_transform_target_preserves_center():
    expr = create_notation(tex=r"x = Uy", notation_key="basis_equation")
    target = create_notation_transform_target(
        expr,
        tex=r"x = U\\Sigma v",
        notation_key="basis_equation",
    )
    assert target.get_center().tolist() == expr.get_center().tolist()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_notation_helpers.py -q`

Expected: FAIL because the module does not exist yet.

- [ ] **Step 3: Implement reusable notation helpers**

```python
from manim import MathTex, Text


def create_notation(
    tex: str,
    notation_key: str | None = None,
    color: str = "WHITE",
    substrings_to_isolate: list[str] | None = None,
    tex_to_color_map: dict[str, str] | None = None,
):
    mob = MathTex(
        tex,
        color=color,
        substrings_to_isolate=substrings_to_isolate or [],
        tex_to_color_map=tex_to_color_map or {},
    )
    mob.notation_key = notation_key
    return mob


def create_notation_transform_target(
    source,
    tex: str,
    notation_key: str | None = None,
    color: str = "WHITE",
    tex_to_color_map: dict[str, str] | None = None,
):
    target = create_notation(
        tex=tex,
        notation_key=notation_key or getattr(source, "notation_key", None),
        color=color,
        tex_to_color_map=tex_to_color_map,
    )
    target.move_to(source.get_center())
    return target


def create_text_label(text: str, color: str = "WHITE", font_size: int = 28):
    return Text(text, color=color, font_size=font_size)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_notation_helpers.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/notation_helpers.py tests/test_notation_helpers.py
git commit -m "feat: add reusable notation helpers"
```

### Task 3: Make Pi-Character Helpers Role-Aware And Deterministic

**Files:**
- Modify: `C:/Users/Parth/Documents/3B1B/src/utils/pi_characters.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_pi_characters.py`

- [ ] **Step 1: Write failing tests for role-aware character helpers**

```python
from src.utils.pi_characters import (
    create_character_for_role,
    create_character_lineup,
)


def test_create_character_for_role_supports_teacher_and_students():
    teacher = create_character_for_role("teacher")
    left_student = create_character_for_role("student_left")
    assert teacher is not None
    assert left_student is not None


def test_create_character_lineup_orders_students_left_to_right():
    lineup = create_character_lineup(
        ["student_left", "student_center", "student_right"],
        spacing=1.2,
    )
    xs = [mob.get_center()[0] for mob in lineup.submobjects]
    assert xs == sorted(xs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pi_characters.py -q`

Expected: FAIL because the new helper functions do not exist yet.

- [ ] **Step 3: Extend `pi_characters.py` with role-aware helpers**

```python
ROLE_ORDER = {
    "teacher": 0,
    "student_left": 1,
    "student_center": 2,
    "student_right": 3,
    "narrator_proxy": 4,
}


def create_character_for_role(role: str, height: float | None = None):
    if role == "teacher":
        return create_pi_teacher(height=height or 1.4)
    if role in {"student_left", "student_center", "student_right"}:
        return create_pi_students(count=1, height=height or 0.8)[0]
    return create_pi_teacher(height=height or 1.0)


def create_character_lineup(roles: list[str], spacing: float = 1.1):
    ordered_roles = sorted(roles, key=lambda role: ROLE_ORDER.get(role, 99))
    lineup = VGroup(*[create_character_for_role(role) for role in ordered_roles])
    lineup.arrange(RIGHT, buff=spacing)
    return lineup
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pi_characters.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/pi_characters.py tests/test_pi_characters.py
git commit -m "feat: add deterministic pi character role helpers"
```

### Task 4: Teach The Generator To Compile Through Helpers

**Files:**
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/animation_generator.py`
- Modify: `C:/Users/Parth/Documents/3B1B/src/utils/manim_engine_contract.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_animation_generator.py`

- [ ] **Step 1: Write failing tests for helper-backed generation**

```python
def test_animation_generator_uses_coordinate_helper_for_axes_objects(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec(
        math_objects=[
            {
                "object_id": "axes_main",
                "object_type": "axes",
                "render_hint": "NumberPlane",
                "content": "main frame",
                "style": {},
            }
        ]
    )
    result = agent.execute(AnimationGeneratorInput(SceneProcessingInput(
        scene_spec=scene_spec,
        context=build_test_context(),
        scene_index=0,
    )))
    assert "create_coordinate_system" in result.animation_code.code


def test_animation_generator_uses_notation_helper_when_notation_key_exists(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec(
        math_objects=[
            {
                "object_id": "eq_main",
                "object_type": "equation",
                "notation_key": "svd_core",
                "content": "X = U\\Sigma V^T",
                "style": {"color": "blue"},
            }
        ]
    )
    result = agent.execute(AnimationGeneratorInput(SceneProcessingInput(
        scene_spec=scene_spec,
        context=build_test_context(),
        scene_index=0,
    )))
    assert "create_notation" in result.animation_code.code
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_animation_generator.py -k "coordinate_helper or notation_helper" -q`

Expected: FAIL because the mock path still emits anonymous `obj_0` construction and direct `MathTex(...)` / `Axes(...)` calls.

- [ ] **Step 3: Refactor object creation to use stable ids and helper imports**

```python
def _object_var_name(self, obj_data: Dict[str, Any], index: int) -> str:
    object_id = obj_data.get("object_id") or f"obj_{index}"
    return re.sub(r"[^a-zA-Z0-9_]", "_", object_id)


def _get_imports(self) -> List[str]:
    return [
        "from manim import *",
        "from src.utils.coordinate_system_helpers import attach_axis_labels, create_coordinate_system, plot_function_on_system",
        "from src.utils.notation_helpers import create_notation, create_notation_transform_target, create_text_label",
        "from src.utils.pi_characters import create_character_for_role, create_character_lineup, create_speech_bubble",
        "from src.utils.layout_tracker import LayoutTracker",
    ]
```

- [ ] **Step 4: Replace anonymous object-generation branches with helper-backed branches**

```python
if obj_type == "equation":
    code_lines.append(
        f'        {var_name} = create_notation(tex=r"{latex}", notation_key={notation_key!r}, color={color.upper()})'
    )
elif obj_type == "text":
    code_lines.append(
        f'        {var_name} = create_text_label({content!r}, color={color.upper()}, font_size={context.visual_style.font_size})'
    )
elif obj_type == "axes":
    helper_kind = obj_data.get("coordinate_system_id") or (render_hint.lower() if render_hint else "axes")
    code_lines.append(
        f'        {var_name} = create_coordinate_system(kind="{helper_kind}", x_length=5.0, y_length=3.0)'
    )
elif obj_type == "pi_creature":
    role = obj_data.get("character_role") or "teacher"
    code_lines.append(f'        {var_name} = create_character_for_role("{role}")')
```

- [ ] **Step 5: Update transformation generation to use explicit ids and notation targets**

```python
if trans_type == "morph" and target_object_id:
    code_lines.append(
        f"        self.play(ReplacementTransform({source_var}, {target_var}), run_time={1.5/animation_speed})"
    )
elif trans_type == "highlight":
    code_lines.append(f"        self.play(Indicate({source_var}), run_time={0.7/animation_speed})")
```

- [ ] **Step 6: Update the codegen prompt so helper use is explicit**

```python
- Prefer `create_coordinate_system(...)` for reusable frames.
- Prefer `create_notation(...)` and `create_notation_transform_target(...)` when a notation_key exists.
- Prefer `create_character_for_role(...)` when a character_role exists.
- Reuse helper-backed objects by object_id rather than recreating them.
```

- [ ] **Step 7: Run the generator tests**

Run: `python -m pytest tests/test_animation_generator.py -q`

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/agents/animation_generator.py src/utils/manim_engine_contract.py tests/test_animation_generator.py
git commit -m "feat: compile storyboard objects through reusable helpers"
```

### Task 5: Persist Helper-Friendly Shared State Across Scenes

**Files:**
- Modify: `C:/Users/Parth/Documents/3B1B/src/schemas/scene_processing.py`
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/scene_controller.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_scene_controller.py`

- [ ] **Step 1: Write failing tests for shared helper state**

```python
def test_scene_controller_exports_coordinate_system_and_notation_state():
    storyboard = build_test_storyboard(num_scenes=1)
    storyboard.scenes[0].visual_intent.scene_state_out.coordinate_systems = {"axes_main": "number_plane"}
    storyboard.scenes[0].visual_intent.scene_state_out.notation_bindings = {"sigma": "eq_sigma"}
    result = SceneControllerAgent().execute(SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=None,
    ))
    assert result.shared_context.coordinate_system_bindings["axes_main"] == "number_plane"
    assert result.shared_context.notation_registry["sigma"] == "eq_sigma"


def test_scene_controller_persists_character_roles():
    storyboard = build_test_storyboard(num_scenes=1)
    storyboard.scenes[0].visual_intent.scene_state_out.character_roles = {"teacher": "teacher_obj"}
    result = SceneControllerAgent().execute(SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=None,
    ))
    assert result.shared_context.character_registry["teacher"] == "teacher_obj"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scene_controller.py -k "coordinate_system_and_notation_state or persists_character_roles" -q`

Expected: FAIL because `SharedContext` does not yet have the registry fields.

- [ ] **Step 3: Extend `SharedContext` with reusable helper registries**

```python
class SharedContext(BaseModel):
    concept_definitions: Dict[str, str] = Field(default_factory=dict)
    visual_style: VisualStyle = Field(default_factory=VisualStyle)
    variable_bindings: Dict[str, str] = Field(default_factory=dict)
    coordinate_system_bindings: Dict[str, str] = Field(default_factory=dict)
    notation_registry: Dict[str, str] = Field(default_factory=dict)
    character_registry: Dict[str, str] = Field(default_factory=dict)
```

- [ ] **Step 4: Update the scene controller to persist `scene_state_out` into shared context**

```python
scene_state_out = scene.visual_intent.scene_state_out
context.coordinate_system_bindings.update(scene_state_out.coordinate_systems)
context.notation_registry.update(scene_state_out.notation_bindings)
context.character_registry.update(scene_state_out.character_roles)
```

- [ ] **Step 5: Run the scene-controller tests**

Run: `python -m pytest tests/test_scene_controller.py -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/schemas/scene_processing.py src/agents/scene_controller.py tests/test_scene_controller.py
git commit -m "feat: persist helper-friendly continuity state"
```

### Task 6: Make Final Video Review Semantic About Helper Usage

**Files:**
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/video_quality_reviewer.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_video_quality_reviewer.py`

- [ ] **Step 1: Write failing tests for helper-semantic review findings**

```python
def test_video_quality_reviewer_flags_axes_redraw_when_helper_should_be_reused():
    code_obj = AnimationCodeObject(
        scene_id="scene_001",
        code="axes_main = Axes()\\nself.play(Create(axes_main), run_time=1.0)",
        imports=["from manim import *"],
        class_name="Scene001",
        narration="Reuse the same axes from before.",
        duration_estimate=10.0,
        generation_timestamp=datetime.now(),
    )
    scene = storyboard.scenes[0]
    scene.visual_intent.scene_state_in.coordinate_systems = {"axes_main": "axes"}
    result = VideoQualityReviewerAgent().execute(VideoQualityReviewerInput(
        storyboard=storyboard,
        manifest=manifest,
        animation_code_objects=[code_obj],
        final_video_expected=False,
    ))
    assert any("coordinate system" in finding.issue.lower() for finding in result.findings)


def test_video_quality_reviewer_flags_missing_notation_helper_when_notation_key_exists():
    code_obj = AnimationCodeObject(
        scene_id="scene_002",
        code="eq_main = MathTex(r'X = U\\\\Sigma V^T')\\nself.play(Write(eq_main), run_time=1.0)",
        imports=["from manim import *"],
        class_name="Scene002",
        narration="We keep the same notation while refining the factorization.",
        duration_estimate=12.0,
        generation_timestamp=datetime.now(),
    )
    scene = storyboard.scenes[0]
    scene.scene_id = "scene_002"
    scene.visual_intent.scene_state_out.notation_bindings = {"svd_core": "eq_main"}
    result = VideoQualityReviewerAgent().execute(VideoQualityReviewerInput(
        storyboard=storyboard,
        manifest=manifest,
        animation_code_objects=[code_obj],
        final_video_expected=False,
    ))
    assert any("notation" in finding.issue.lower() for finding in result.findings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_video_quality_reviewer.py -k "axes_redraw or missing_notation_helper" -q`

Expected: FAIL because the reviewer does not yet inspect helper fidelity.

- [ ] **Step 3: Add semantic helper checks**

```python
def _review_helper_semantics(self, storyboard, code_by_scene):
    findings = []
    for scene in storyboard.scenes:
        code = code_by_scene.get(scene.scene_id)
        if not code:
            continue
        source = code.code
        if scene.visual_intent.scene_state_in.coordinate_systems and "create_coordinate_system" not in source:
            findings.append(
                VideoQualityFinding(
                    issue="A reusable coordinate system was redrawn without the helper-backed factory.",
                    suspected_stage="animation_generator",
                    severity="medium",
                    scene_id=scene.scene_id,
                    evidence="scene_state_in declares reusable coordinate systems, but generated code does not call create_coordinate_system(...)",
                    suggested_fix="Reuse the prior coordinate system binding and build it through create_coordinate_system(...) only when first introduced.",
                    llm_direction=f"Regenerate scene {scene.scene_id} so recurring axes or planes are created or reused through create_coordinate_system(...).",
                )
            )
        if scene.visual_intent.scene_state_out.notation_bindings and "create_notation" not in source:
            findings.append(
                VideoQualityFinding(
                    issue="Persistent notation was emitted without the notation helper.",
                    suspected_stage="animation_generator",
                    severity="medium",
                    scene_id=scene.scene_id,
                    evidence="scene_state_out exports notation bindings, but generated code does not call create_notation(...)",
                    suggested_fix="Build persistent equations and symbolic labels through create_notation(...) so notation keys stay stable across retries.",
                    llm_direction=f"Regenerate scene {scene.scene_id} so notation-bearing MathTex objects are created through create_notation(...).",
                )
            )
        if scene.visual_intent.scene_state_out.character_roles and "create_character_for_role" not in source:
            findings.append(
                VideoQualityFinding(
                    issue="Character-role continuity was emitted without the role-aware helper.",
                    suspected_stage="animation_generator",
                    severity="medium",
                    scene_id=scene.scene_id,
                    evidence="scene_state_out exports character roles, but generated code does not call create_character_for_role(...)",
                    suggested_fix="Use create_character_for_role(...) so teacher and student roles stay deterministic across scenes.",
                    llm_direction=f"Regenerate scene {scene.scene_id} so persistent character roles use create_character_for_role(...).",
                )
            )
    return findings
```

- [ ] **Step 4: Call the new semantic review pass from `execute(...)`**

```python
findings.extend(self._review_helper_semantics(input_data.storyboard, code_by_scene))
```

- [ ] **Step 5: Run the reviewer tests**

Run: `python -m pytest tests/test_video_quality_reviewer.py -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/agents/video_quality_reviewer.py tests/test_video_quality_reviewer.py
git commit -m "feat: review helper fidelity in final video diagnostics"
```

### Task 7: Verification And Regression Checkpoint

**Files:**
- Modify: none
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_coordinate_system_helpers.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_notation_helpers.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_pi_characters.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_animation_generator.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_scene_controller.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_video_quality_reviewer.py`

- [ ] **Step 1: Run the full stage-two test slice**

Run: `python -m pytest tests/test_coordinate_system_helpers.py tests/test_notation_helpers.py tests/test_pi_characters.py tests/test_animation_generator.py tests/test_scene_controller.py tests/test_video_quality_reviewer.py -q`

Expected: PASS

- [ ] **Step 2: Run syntax verification on all touched files**

Run: `python -m py_compile src/utils/coordinate_system_helpers.py src/utils/notation_helpers.py src/utils/pi_characters.py src/utils/manim_engine_contract.py src/agents/animation_generator.py src/schemas/scene_processing.py src/agents/scene_controller.py src/agents/video_quality_reviewer.py`

Expected: no output

- [ ] **Step 3: Run a focused smoke test for helper imports inside generated code**

Run: `python -m pytest tests/test_animation_generator.py -k "coordinate_helper or notation_helper or pi_creature" -q`

Expected: PASS

- [ ] **Step 4: Commit the verification checkpoint**

```bash
git add docs/superpowers/plans/2026-04-09-reusable-manim-primitives-helpers.md
git commit -m "docs: add reusable manim primitives implementation plan"
```

## Self-Review

### Spec Coverage

- reusable coordinate systems: covered by Task 1, Task 4, Task 5, Task 6
- reusable notation helpers: covered by Task 2, Task 4, Task 5, Task 6
- persistent character roles through helper functions: covered by Task 3, Task 4, Task 5, Task 6
- generator compilation through helper-backed imports: covered by Task 4
- scene-to-scene continuity storage: covered by Task 5
- reviewer semantic enforcement: covered by Task 6

### Placeholder Scan

- no `TODO`, `TBD`, or "implement later" placeholders remain
- every task has named files, test commands, expected outcomes, and commit commands

### Type Consistency

- helper module names are consistent across tasks:
  - `create_coordinate_system`
  - `attach_axis_labels`
  - `plot_function_on_system`
  - `create_notation`
  - `create_notation_transform_target`
  - `create_character_for_role`
  - `create_character_lineup`
- shared-context registry names stay consistent across tasks:
  - `coordinate_system_bindings`
  - `notation_registry`
  - `character_registry`
