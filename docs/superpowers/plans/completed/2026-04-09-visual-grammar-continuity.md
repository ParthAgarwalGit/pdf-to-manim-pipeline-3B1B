# Visual Grammar Continuity Implementation Plan

> **Status:** ✅ COMPLETED — `coordinate_system_bindings`, `notation_registry`,
> `character_registry`, and `color_bindings` fields on `SharedContext` ship
> in `src/schemas/scene_processing.py`; the visual planner emits and
> propagates them via `scene_state_in`/`scene_state_out`. Wiring landed in
> commits `7156271` ("propagate continuity registries via SharedContext")
> and `a6c933d` ("reviewer checks helper-layer semantics"). Retained as
> historical context for the continuity contract.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a deterministic visual continuity contract for storyboard planning so recurring axes, reused notation, and persistent character roles survive retries and compile cleanly into later Manim stages.

**Architecture:** The first implementation wave changes the contract before changing the compiler. `manim_engine_contract.py` becomes the single source of truth for semantic and engine-backed enums, `storyboard.py` validates a richer continuity schema, and `script_architect.py` emits explicit continuity-aware visual plans. Later waves can then teach the scene controller and code generator to preserve and compile that state without guessing.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, OpenAI chat completions, Manim Community, reference semantics from installed `manimgl` / `manimlib`.

---

### Task 1: Expand Engine Contract Vocabulary

**Files:**
- Modify: `C:/Users/Parth/Documents/3B1B/src/utils/manim_engine_contract.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_schemas.py`

- [ ] **Step 1: Write failing tests for broadened continuity enums and normalized hints**

```python
def test_engine_contract_normalizes_broader_render_hints():
    obj = MathematicalObject(
        object_id="axes_main",
        object_role="shared_axes",
        object_type="axes",
        render_hint="complex_plane",
        content="main coordinate frame",
        style=ObjectStyle(),
    )
    assert obj.render_hint == "ComplexPlane"


def test_transformation_accepts_engine_animation_aliases_but_normalizes_family():
    t = Transformation(
        transformation_type="ReplacementTransform",
        source_object_id="eq_old",
        target_object_id="eq_new",
        continuity_action="transform",
        beat_id="beat_001",
        timing=1.0,
    )
    assert t.transformation_type == "morph"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_schemas.py -k "engine_contract_normalizes_broader_render_hints or transformation_accepts_engine_animation_aliases_but_normalizes_family" -q`

Expected: FAIL because the schema does not yet support the richer continuity fields and broader enum set.

- [ ] **Step 3: Implement the minimal contract expansion**

```python
CANONICAL_COORDINATE_SYSTEM_KINDS = (
    "axes",
    "number_plane",
    "complex_plane",
    "number_line",
    "three_d_axes",
)

CANONICAL_CAMERA_ACTIONS = (
    "static",
    "focus",
    "zoom_to_region",
    "pan_to_object",
    "follow_object",
    "restore",
)

RENDER_HINT_ALIASES.update({
    "complexplane": "ComplexPlane",
    "complex_plane": "ComplexPlane",
    "numberline": "NumberLine",
    "number_line": "NumberLine",
    "threedaxes": "ThreeDAxes",
    "three_d_axes": "ThreeDAxes",
    "decimalnumber": "DecimalNumber",
    "integer": "Integer",
    "vector": "Vector",
    "doublearrow": "DoubleArrow",
    "roundedrectangle": "RoundedRectangle",
    "brace": "Brace",
})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_schemas.py -k "engine_contract_normalizes_broader_render_hints or transformation_accepts_engine_animation_aliases_but_normalizes_family" -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/manim_engine_contract.py tests/test_schemas.py
git commit -m "feat: expand visual grammar engine contract"
```

### Task 2: Add Continuity Schema To Storyboard Models

**Files:**
- Modify: `C:/Users/Parth/Documents/3B1B/src/schemas/storyboard.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_schemas.py`

- [ ] **Step 1: Write failing tests for explicit continuity fields**

```python
def test_mathematical_object_requires_stable_identity_fields():
    obj = MathematicalObject(
        object_id="sigma_label",
        object_role="main_equation",
        object_type="equation",
        notation_key="sigma",
        content=r"\sigma",
        style=ObjectStyle(),
    )
    assert obj.object_id == "sigma_label"
    assert obj.object_role == "main_equation"
    assert obj.notation_key == "sigma"


def test_visual_intent_supports_scene_state_boundaries():
    visual_intent = VisualIntent(
        mathematical_objects=[],
        transformations=[],
        beats=[],
        scene_state_in=SceneState(),
        scene_state_out=SceneState(),
    )
    assert visual_intent.scene_state_in.active_object_ids == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_schemas.py -k "stable_identity_fields or scene_state_boundaries" -q`

Expected: FAIL because the models do not yet expose the continuity fields.

- [ ] **Step 3: Implement the continuity schema**

```python
class SceneState(BaseModel):
    active_object_ids: List[str] = Field(default_factory=list)
    coordinate_systems: Dict[str, str] = Field(default_factory=dict)
    notation_bindings: Dict[str, str] = Field(default_factory=dict)
    character_roles: Dict[str, str] = Field(default_factory=dict)


class VisualBeat(BaseModel):
    beat_id: str
    narration_excerpt: str
    start_time: float = Field(..., ge=0.0)
    end_time: float = Field(..., ge=0.0)
    active_object_ids: List[str] = Field(default_factory=list)
    camera_action: str = Field(default="static")
    beat_goal: str = Field(default="")
```

- [ ] **Step 4: Update `MathematicalObject`, `Transformation`, and `VisualIntent`**

```python
class MathematicalObject(BaseModel):
    object_id: str
    object_role: str
    object_type: str
    content: str
    coordinate_system_id: Optional[str] = None
    notation_key: Optional[str] = None
    character_role: Optional[str] = None
    persist_across_scenes: bool = False
    render_hint: Optional[str] = None
    style: ObjectStyle = Field(default_factory=ObjectStyle)


class Transformation(BaseModel):
    transformation_type: str
    source_object_id: str
    target_object_id: Optional[str] = None
    continuity_action: str = "transform"
    beat_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timing: float = Field(..., ge=0.0)
```

- [ ] **Step 5: Run schema tests**

Run: `python -m pytest tests/test_schemas.py -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/schemas/storyboard.py tests/test_schemas.py
git commit -m "feat: add storyboard continuity schema"
```

### Task 3: Teach ScriptArchitect To Emit Continuity-Aware Plans

**Files:**
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/script_architect.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_script_architect.py`

- [ ] **Step 1: Write failing tests for continuity-aware planner output**

```python
def test_visual_intent_includes_object_ids_and_scene_state(...):
    visual_intent = agent._generate_visual_intent(
        "concept_001",
        default_config,
        sample_ocr_output,
        narration="We reuse the same axes while updating the notation.",
    )
    assert visual_intent.beats
    assert visual_intent.scene_state_in is not None
    assert visual_intent.scene_state_out is not None
    assert all(obj.object_id for obj in visual_intent.mathematical_objects)


def test_visual_intent_tracks_coordinate_system_and_notation_keys(...):
    ...
    assert any(obj.coordinate_system_id for obj in visual_intent.mathematical_objects)
    assert any(obj.notation_key for obj in visual_intent.mathematical_objects)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_script_architect.py -k "object_ids_and_scene_state or coordinate_system_and_notation_keys" -q`

Expected: FAIL because planner output does not yet contain continuity-aware structure.

- [ ] **Step 3: Update the visual-intent prompt**

```python
engine_contract = build_visual_intent_contract_prompt()

prompt = f"""
...
Every object MUST include:
- object_id
- object_role
- object_family
- optional coordinate_system_id
- optional notation_key
- optional character_role

Every beat MUST include:
- beat_id
- narration_excerpt
- active_object_ids
- camera_action
- beat_goal

Every scene MUST include:
- scene_state_in
- scene_state_out
...
"""
```

- [ ] **Step 4: Parse continuity-aware output into the new schema**

```python
beat_models = [
    VisualBeat(
        beat_id=beat["beat_id"],
        narration_excerpt=beat.get("narration_excerpt", ""),
        start_time=float(beat.get("start_time", 0.0)),
        end_time=float(beat.get("end_time", 0.0)),
        active_object_ids=beat.get("active_object_ids", []),
        camera_action=beat.get("camera_action", "static"),
        beat_goal=beat.get("beat_goal", ""),
    )
    for beat in data.get("beats", [])
]
```

- [ ] **Step 5: Keep fallback generation compatible with the richer schema**

```python
return VisualIntent(
    mathematical_objects=objs,
    transformations=trans,
    beats=[VisualBeat(beat_id="beat_001", narration_excerpt=first_sentence, start_time=0.0, end_time=5.0)],
    scene_state_in=SceneState(),
    scene_state_out=SceneState(active_object_ids=[obj.object_id for obj in objs]),
    emphasis_points=[],
)
```

- [ ] **Step 6: Run focused architect tests**

Run: `python -m pytest tests/test_script_architect.py -q`

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/agents/script_architect.py tests/test_script_architect.py
git commit -m "feat: emit continuity-aware storyboard plans"
```

### Task 4: Regression Verification For Wave 1

**Files:**
- Modify: none
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_schemas.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_script_architect.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_animation_generator.py`

- [ ] **Step 1: Run the full Wave 1 test slice**

Run: `python -m pytest tests/test_schemas.py tests/test_script_architect.py tests/test_animation_generator.py -q`

Expected: PASS

- [ ] **Step 2: Run syntax verification on touched files**

Run: `python -m py_compile src/utils/manim_engine_contract.py src/schemas/storyboard.py src/agents/script_architect.py tests/test_schemas.py tests/test_script_architect.py`

Expected: no output

- [ ] **Step 3: Commit the verification checkpoint**

```bash
git add docs/superpowers/plans/2026-04-09-visual-grammar-continuity.md
git commit -m "docs: add wave 1 implementation plan"
```
