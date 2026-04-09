# Visual Grammar Continuity Design

**Date:** 2026-04-09

## Goal

Introduce a hard explicit visual continuity contract for the PDF-to-Manim pipeline so recurring axes, reused notation, persistent character roles, and beat-to-beat object lifecycle remain deterministic across retries and compile into stronger Manim scenes.

## Why This Exists

The current pipeline has become more engine-aware, but it still breaks continuity in the places that matter most for high-quality educational animation:

- storyboard objects do not have stable identity
- transformations target weak string references
- scene generation is mostly scene-local instead of stateful across the storyboard
- the grader optimizes for count-based richness rather than visual clarity
- the code generator collapses rich scene intent into generic placeholder animations

This design addresses those failures directly by making continuity a first-class contract, not an inference.

## Design Principles

1. Determinism over clever inference
   Continuity must be declared explicitly in the storyboard and shared context. Later agents should not guess whether two symbols or coordinate systems are “the same thing.”

2. Narrow semantic control, broad engine-backed rendering
   Semantic enums stay small and stable. Rendering enums can be broader and reflect real Manim primitives and animation classes.

3. Prompt and schema must match exactly
   The prompts shown to LLM agents must enumerate the exact allowed semantic enums and the exact broader engine-backed enums.

4. Reuse over redraw
   When a concept persists visually across beats or scenes, the contract should prefer reuse and transformation rather than fade-out-plus-recreate patterns.

5. 3Blue1Brown-style continuity as a system property
   Reused axes, persistent notation, evolving equations, and stable character roles should be supported by the architecture, not left to prompt luck.

## In-Scope Files

### Core Contract

- `src/schemas/storyboard.py`
- `src/utils/manim_engine_contract.py`

### Planning

- `src/agents/script_architect.py`

### Continuity State and Compilation

- `src/agents/scene_controller.py`
- `src/agents/animation_generator.py`

### Enforcement

- `src/agents/storyboard_grader.py`

## Out of Scope For This Phase

- `src/agents/video_quality_reviewer.py`
- `src/agents/manim_code_validator.py`
- `src/agents/vision_ocr.py`
- `src/agents/video_renderer.py`
- `src/agents/audio_video_composer.py`
- `src/agents/tts_agent.py`
- `src/agents/persistence.py`

These will eventually need to understand the richer contract, but they should not define it in this phase.

## Primary 3b1b / Manim Inspirations

This phase should draw concrete structure from the installed `manimlib` reference package and the 3b1b repository model:

- `Scene.construct()` object lifecycle and deterministic scene state
- reusable coordinate-system abstractions such as `Axes`, `NumberPlane`, `ComplexPlane`, `NumberLine`, and helper methods like `c2p`, axis labels, and graph helpers
- animation families organized around creation, transform, indication, movement, and fading
- persistent visual roles such as teacher/student characters

The runtime target remains Manim Community. These references shape the contract and prompts, not the runtime import surface.

## Contract Model

The pipeline should use a two-layer vocabulary.

### Layer 1: Stable Semantic Enums

These are small, deterministic, and required in planner output.

- `object_family`
- `object_role`
- `continuity_action`
- `coordinate_system_kind`
- `character_role`
- `camera_action`

These enums exist to keep planning stable and auditable.

### Layer 2: Broad Engine-Backed Enums

These are larger and map to real Manim primitives or animation classes.

- `render_hint`
- `animation_hint`
- `layout_hint`
- `label_attachment_kind`

These enums should be broad enough to represent the useful library surface without making the semantic contract noisy.

## Required Schema Changes

### `MathematicalObject`

Add:

- `object_id: str`
- `object_role: str`
- `coordinate_system_id: Optional[str]`
- `notation_key: Optional[str]`
- `character_role: Optional[str]`
- `persist_across_scenes: bool = False`
- existing `render_hint`

Behavior:

- `object_id` is mandatory and unique within a scene
- `notation_key` is used when the same notation should persist or evolve later
- `character_role` is only valid for character-family objects
- `coordinate_system_id` can point to a shared axes/plane object when relevant

### `Transformation`

Change and extend:

- keep `transformation_type`
- replace weak targeting with:
  - `source_object_id: str`
  - `target_object_id: Optional[str]`
- add:
  - `continuity_action: str`
  - `beat_id: str`
  - `animation_hint: Optional[str]`

Behavior:

- every transformation references explicit object ids
- `continuity_action` expresses semantic lifecycle:
  - `create`
  - `reuse`
  - `transform`
  - `remove`
  - `highlight`

### `VisualBeat`

Introduce a beat structure:

- `beat_id: str`
- `narration_excerpt: str`
- `start_time: float`
- `end_time: float`
- `active_object_ids: List[str]`
- `camera_action: str`
- `beat_goal: str`

### `VisualIntent`

Extend with:

- `beats: List[VisualBeat]`
- `scene_state_in: SceneState`
- `scene_state_out: SceneState`
- `shared_coordinate_system_ids: List[str]`
- `shared_notation_keys: List[str]`
- `persistent_character_roles: List[str]`

### `SceneState`

Introduce a reusable schema for cross-scene continuity:

- `active_object_ids: List[str]`
- `coordinate_systems: Dict[str, str]`
- `notation_bindings: Dict[str, str]`
- `character_roles: Dict[str, str]`

Interpretation:

- `coordinate_systems`: maps ids like `axes_main` to a semantic kind like `number_plane`
- `notation_bindings`: maps keys like `sigma` to object ids
- `character_roles`: maps `teacher` to the active character object id

## Enum Strategy

### Narrow Semantic Enums

These should stay tightly controlled.

#### `continuity_action`

- `create`
- `reuse`
- `transform`
- `remove`
- `highlight`

#### `character_role`

- `teacher`
- `student_left`
- `student_center`
- `student_right`
- `narrator_proxy`

#### `coordinate_system_kind`

- `axes`
- `number_plane`
- `complex_plane`
- `number_line`
- `three_d_axes`

#### `camera_action`

- `static`
- `focus`
- `zoom_to_region`
- `pan_to_object`
- `follow_object`
- `restore`

### Broad Engine-Backed Enums

These should be explicitly listed in prompts and normalized in the contract helper.

#### `render_hint`

Text and math:

- `Text`
- `Tex`
- `MathTex`
- `DecimalNumber`
- `Integer`

Coordinate systems:

- `Axes`
- `ThreeDAxes`
- `NumberPlane`
- `ComplexPlane`
- `NumberLine`

Geometry:

- `Line`
- `DashedLine`
- `Arrow`
- `Vector`
- `DoubleArrow`
- `Circle`
- `Ellipse`
- `Arc`
- `Polygon`
- `Rectangle`
- `RoundedRectangle`
- `Square`
- `Triangle`
- `Dot`
- `Brace`

Graphs and structured visuals:

- `ParametricCurve`
- `Graph`
- `DiGraph`
- `VGroup`
- `Group`

3Blue1Brown helpers:

- `PiCreatureTeacher`
- `PiCreatureStudents`
- `SpeechBubble`

#### `animation_hint`

Creation:

- `Create`
- `Write`
- `ShowCreation`
- `DrawBorderThenFill`
- `GrowFromCenter`
- `GrowArrow`
- `FadeIn`

Transform:

- `Transform`
- `ReplacementTransform`
- `TransformFromCopy`
- `MoveToTarget`

Indication:

- `Indicate`
- `Flash`
- `Circumscribe`
- `FocusOn`
- `Wiggle`

Movement:

- `MoveAlongPath`
- `Rotate`
- `ApplyWave`

Fade and removal:

- `FadeOut`
- `FadeTransform`
- `FadeTransformPieces`

## Prompt Changes

### `ScriptArchitectAgent`

The planner prompt must:

- enumerate every semantic enum explicitly
- enumerate every broad engine-backed enum explicitly
- require `object_id` on every planned object
- require `beat_id` on every visual beat
- require `scene_state_in` and `scene_state_out`
- require reuse of prior `coordinate_system_id`, `notation_key`, and `character_role` when present
- prefer transformation of persistent objects over fade-out-plus-redraw
- treat narration as the source of beat sequencing and timing

The prompt should reason using this structure:

- what coordinate frame persists
- what notation persists
- what character roles persist
- what objects are created, reused, transformed, highlighted, or removed in each beat

### `AnimationGeneratorAgent`

The compiler prompt must:

- include the exact same semantic and engine-backed enum lists
- explain that `object_id` maps to a persistent Python variable registry
- require reuse of prior objects when continuity says `reuse` or `transform`
- prefer coordinate-system reuse and graph attachment to those systems
- prefer `ReplacementTransform` and related transforms when notation evolves
- require generated code to preserve ids and roles logically across beats

## Agent Responsibilities

### `script_architect.py`

Responsibilities after redesign:

- emit the richer continuity contract
- plan explicit reuse of axes and notation
- represent persistent character roles by name
- output beat-level transitions, not just flat object lists

### `scene_controller.py`

Responsibilities after redesign:

- maintain shared continuity state across scenes
- validate that scene N only reuses what earlier scenes exported
- preserve deterministic continuity across retries

Shared state should include:

- active object ids
- active coordinate-system ids
- notation bindings
- character role bindings

### `animation_generator.py`

Responsibilities after redesign:

- create a registry keyed by `object_id`
- compile beat transitions rather than anonymous object lists
- reuse or transform existing objects explicitly
- compile coordinate systems and graph objects as reusable anchors
- compile persistent character roles consistently

This file should stop guessing target mappings from string patterns.

### `storyboard_grader.py`

Responsibilities after redesign:

Add grading dimensions for:

- continuity fidelity
- animatability
- Manim primitive quality
- notation consistency
- coordinate-system reuse quality
- character-role consistency

The heuristic path should also move beyond raw counts and narration length.

## Scene Controller Continuity Model

The shared context initialized by the scene controller should expand beyond visual style.

Add explicit shared state:

- `active_objects`
- `active_coordinate_systems`
- `notation_registry`
- `character_registry`
- `continuity_ledger`

The continuity ledger should exist for debugging and retry transparency. It should allow later enforcement and review layers to explain why a scene failed continuity validation.

## Compilation Model

The animation generator should move from this pattern:

- create `obj_0`, `obj_1`
- map target strings heuristically
- emit generic transforms

to this pattern:

- create variables from `object_id`
- maintain an object registry dictionary
- compile each beat in order
- resolve continuity actions explicitly
- reuse shared coordinate systems and characters
- compile notation evolution through transforms, not redraws

## Validation Rules

The schema and controller should hard-fail when:

- a transformation references an unknown `object_id`
- a scene declares reuse of a coordinate system that was never created
- a notation key changes identity without an explicit transform or rebind
- a character role is reused inconsistently
- a planner emits enums outside the allowed lists

## Testing Strategy

### Schema Tests

Add tests for:

- required `object_id`
- duplicate object ids rejected
- invalid continuity actions rejected
- invalid coordinate-system reuse rejected
- invalid character-role reuse rejected

### Planner Tests

Add tests for:

- architect emits explicit continuity fields
- architect uses valid semantic enums
- architect uses valid broad engine-backed enums
- architect reuses prior notation and coordinate systems when asked

### Controller and Compiler Tests

Add tests for:

- scene 2 reuses axes from scene 1 deterministically
- notation such as `u`, `v`, `Σ`, or `x'` keeps stable identity across scenes
- teacher/student roles persist across scenes
- transforms operate on explicit object ids
- generator avoids redraw when continuity says reuse or transform

## Success Criteria

This phase is complete when:

- the storyboard contract can explicitly represent recurring axes, reused notation, and persistent character roles
- retries preserve continuity deterministically
- the generator compiles object reuse instead of relying on guessed target mappings
- the grader can penalize broken continuity directly
- the pipeline can represent 3Blue1Brown-style progressive visual development as a system behavior

## Implementation Recommendation

Implement in two internal waves:

### Wave 1

- expand `storyboard.py`
- expand `manim_engine_contract.py`
- update `script_architect.py`

Goal: planner emits the correct explicit contract.

### Wave 2

- update `scene_controller.py`
- update `animation_generator.py`
- update `storyboard_grader.py`

Goal: continuity is enforced and compiled deterministically.

This keeps the contract definition ahead of the enforcement and code-generation rewrite.
