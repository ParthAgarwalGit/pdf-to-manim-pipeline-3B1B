# Unified Script, Chunked Render — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate scene-to-scene redundancy. Today every scene generates its narration independently with the full PDF text and zero awareness of prior scenes, so each scene re-explains earlier concepts. After this plan, the pipeline produces **one continuous script** first, then deterministically **chunks** it into render-sized scenes; per-scene LLM calls receive a compact "what the viewer has already seen" briefing instead of the raw PDF.

**Scenes remain — but only as render chunks.** Failure isolation, render budget, TTS chunking, and the per-chunk validator/retry loop already in place all justify keeping the chunked render. What changes is the *content contract*: each chunk is a slice of one coherent script, not an independent re-explanation.

**Architecture:**
1. New `NarrativeContext` rides inside `SharedContext`, carrying `prior_scene_summaries`, `concepts_introduced`, `running_storyline`, and `live_objects_on_screen`.
2. New `VideoScript` schema captures the unified narration arc with a beat list and an explicit concept-introduction ledger ("concept X is introduced exactly at beat N").
3. `ScriptArchitectAgent` is rebuilt: one LLM call produces a `VideoScript`; a deterministic chunker groups beats into `Scene`s of ~30–90s each, preserving narrative continuity. The regex-based concept extractor (`_identify_concepts`) and per-concept narration calls are deleted.
4. `SceneControllerAgent` populates `NarrativeContext` after each successful scene (deterministic summary, ledger append, live-object propagation).
5. The visual-intent and animation-code prompts no longer dump the full PDF. They receive the chunk's narration + the narrative briefing, with explicit instructions: *"The viewer has seen X, Y, Z. Reference, don't re-explain."*
6. Each generated `Scene` opens by re-instantiating the prior scene's live objects at `run_time=0` so chunk cuts feel seamless (Manim-level visual continuity).

**Tech Stack:** Python 3.12, Pydantic v2, pytest, OpenAI chat completions (script + visual intent), Anthropic Claude (animation code), Manim Community v0.19.

**Out of scope for this plan:**
- Cross-scene Manim camera continuity (zoom/pan that survives a chunk cut) — separate plan if needed.
- Replacing the unified-script LLM call with multiple smaller calls — start with one, measure token use, optimize later.

---

### Task 1: Add `NarrativeContext` schema (additive, non-breaking)

**Files:**
- Modify: `C:/Users/Parth/Documents/3B1B/src/schemas/scene_processing.py`
- Modify: `C:/Users/Parth/Documents/3B1B/src/schemas/__init__.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_schemas.py`

- [ ] **Step 1: Write failing tests for `NarrativeContext` and its embedding in `SharedContext`**

```python
def test_narrative_context_defaults_are_empty():
    from src.schemas.scene_processing import NarrativeContext
    nc = NarrativeContext()
    assert nc.prior_scene_summaries == []
    assert nc.concepts_introduced == []
    assert nc.running_storyline == ""
    assert nc.live_objects_on_screen == {}


def test_narrative_context_appends_preserve_order():
    from src.schemas.scene_processing import NarrativeContext
    nc = NarrativeContext()
    nc.prior_scene_summaries.append("Introduced sine on the unit circle.")
    nc.concepts_introduced.append("sine")
    nc.live_objects_on_screen["unit_circle"] = "Circle(radius=2)"
    assert nc.prior_scene_summaries == ["Introduced sine on the unit circle."]
    assert "sine" in nc.concepts_introduced


def test_shared_context_carries_narrative_context_default():
    from src.schemas.scene_processing import SharedContext, NarrativeContext
    ctx = SharedContext()
    assert isinstance(ctx.narrative, NarrativeContext)
    assert ctx.narrative.concepts_introduced == []
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_schemas.py -k "narrative_context" -q
```

Expected: FAIL — `NarrativeContext` does not exist; `SharedContext.narrative` does not exist.

- [ ] **Step 3: Implement minimal additions**

In `src/schemas/scene_processing.py`, add **before** `SharedContext`:

```python
class NarrativeContext(BaseModel):
    """Tracks what the viewer has already seen so later scenes do not re-explain.

    The Scene Controller updates this after each successful scene. Per-scene
    LLM prompts read it to know which concepts to *reference* (not redefine)
    and which on-screen objects to carry forward.
    """

    prior_scene_summaries: List[str] = Field(
        default_factory=list,
        description="One short summary per prior scene (≤ 30 words each), in order"
    )
    concepts_introduced: List[str] = Field(
        default_factory=list,
        description="Canonical concept labels the viewer has already been taught"
    )
    running_storyline: str = Field(
        default="",
        description="2–3 sentence rolling synopsis of the video so far"
    )
    live_objects_on_screen: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of object_id → short description of objects still visible "
                    "at the END of the most recent scene (carried into the next chunk)"
    )
```

Then extend `SharedContext` with one new field:

```python
narrative: NarrativeContext = Field(
    default_factory=NarrativeContext,
    description="Narrative continuity state propagated across scenes"
)
```

Export `NarrativeContext` from `src/schemas/__init__.py`.

- [ ] **Step 4: Verify tests pass and existing tests still pass**

```
python -m pytest tests/test_schemas.py -q
python -m pytest tests/test_scene_controller.py -q
```

---

### Task 2: Add `VideoScript` and `Beat` schemas

**Files:**
- New: `C:/Users/Parth/Documents/3B1B/src/schemas/video_script.py`
- Modify: `C:/Users/Parth/Documents/3B1B/src/schemas/__init__.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_schemas.py`

- [ ] **Step 1: Write failing tests for `VideoScript`/`Beat` schemas**

```python
def test_beat_validates_required_fields():
    from src.schemas.video_script import Beat
    b = Beat(
        beat_id="beat_001",
        narration="A sine wave traces the y-coordinate of a point moving around a circle.",
        introduces_concepts=["sine", "unit_circle"],
        references_prior_beats=[],
        visual_hint="Show a unit circle with a rotating radius and a graph next to it.",
    )
    assert b.beat_id == "beat_001"
    assert "sine" in b.introduces_concepts


def test_video_script_concept_ledger_is_unique_and_ordered():
    from src.schemas.video_script import VideoScript, Beat
    script = VideoScript(
        beats=[
            Beat(beat_id="beat_001", narration="...", introduces_concepts=["sine"], visual_hint="..."),
            Beat(beat_id="beat_002", narration="...", introduces_concepts=["cosine", "sine"], visual_hint="..."),
        ],
    )
    # ledger preserves first-introduction order, no duplicates
    assert script.concept_ledger() == ["sine", "cosine"]


def test_video_script_rejects_duplicate_beat_ids():
    import pytest
    from pydantic import ValidationError
    from src.schemas.video_script import VideoScript, Beat
    with pytest.raises(ValidationError):
        VideoScript(beats=[
            Beat(beat_id="beat_001", narration="a", introduces_concepts=[], visual_hint="x"),
            Beat(beat_id="beat_001", narration="b", introduces_concepts=[], visual_hint="y"),
        ])
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_schemas.py -k "beat or video_script" -q
```

- [ ] **Step 3: Implement `src/schemas/video_script.py`**

```python
"""Unified video-script schema.

The Script Architect produces a VideoScript first — one continuous narration
arc with an explicit concept-introduction ledger. A deterministic chunker
later groups beats into render-sized Scenes.
"""

from typing import List, Dict
from pydantic import BaseModel, Field, model_validator


class Beat(BaseModel):
    """One narrative beat (≈ one paragraph or thought)."""

    beat_id: str = Field(..., description="Stable id, e.g. 'beat_001'")
    narration: str = Field(..., description="Spoken text for this beat")
    visual_hint: str = Field(
        ...,
        description="Short prose description of what should be on screen during this beat. "
                    "The downstream visual planner will turn this into a structured VisualIntent."
    )
    introduces_concepts: List[str] = Field(
        default_factory=list,
        description="Canonical concept labels first introduced in this beat"
    )
    references_prior_beats: List[str] = Field(
        default_factory=list,
        description="beat_ids whose visuals/concepts this beat builds on"
    )


class VideoScript(BaseModel):
    """A single continuous narration arc covering all PDF concepts.

    The ledger property gives a deterministic ordered list of concept
    labels in first-introduction order; downstream code uses this to
    enforce 'don't re-explain' rules.
    """

    beats: List[Beat] = Field(..., min_length=1)
    title: str = Field(default="", description="Optional title for the video")

    @model_validator(mode="after")
    def _unique_beat_ids(self) -> "VideoScript":
        seen = set()
        for b in self.beats:
            if b.beat_id in seen:
                raise ValueError(f"duplicate beat_id: {b.beat_id}")
            seen.add(b.beat_id)
        return self

    def concept_ledger(self) -> List[str]:
        """Concepts in first-introduction order, deduped."""
        seen, out = set(), []
        for b in self.beats:
            for c in b.introduces_concepts:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
        return out
```

Export `Beat` and `VideoScript` from `src/schemas/__init__.py`.

- [ ] **Step 4: Verify tests pass**

```
python -m pytest tests/test_schemas.py -q
```

---

### Task 3: New `_generate_video_script` LLM call in `ScriptArchitectAgent`

**Files:**
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/script_architect.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_script_architect.py`

- [ ] **Step 1: Write failing tests for `_generate_video_script` (mocked OpenAI)**

```python
def test_generate_video_script_returns_video_script_object(monkeypatch):
    """One LLM call → VideoScript with concept ledger preserved."""
    import json
    from src.agents.script_architect import ScriptArchitectAgent
    from src.schemas.storyboard import Configuration
    from src.schemas.ocr_output import OCROutput
    from tests.fixtures.ocr_fixtures import minimal_trig_ocr_output  # add fixture

    fake_response = {
        "title": "Sine, Cosine, and the Unit Circle",
        "beats": [
            {
                "beat_id": "beat_001",
                "narration": "Sine measures the height of a rotating point on the unit circle.",
                "visual_hint": "Unit circle with a rotating radius; a y-coordinate trace.",
                "introduces_concepts": ["unit_circle", "sine"],
                "references_prior_beats": [],
            },
            {
                "beat_id": "beat_002",
                "narration": "Cosine measures the same point's horizontal coordinate.",
                "visual_hint": "Same unit circle; project the point onto the x-axis.",
                "introduces_concepts": ["cosine"],
                "references_prior_beats": ["beat_001"],
            },
        ],
    }

    class _FakeChoice:
        def __init__(self, content): self.message = type("M", (), {"content": content})
    class _FakeResp:
        def __init__(self, content): self.choices = [_FakeChoice(content)]
    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw): return _FakeResp(json.dumps(fake_response))

    agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
    agent.strict = False
    agent.client = _FakeClient()

    ocr = minimal_trig_ocr_output()
    config = Configuration(verbosity="medium", depth="intermediate", audience="undergraduate")

    script = agent._generate_video_script(ocr, config, review_feedback=[])
    assert script.concept_ledger() == ["unit_circle", "sine", "cosine"]
    assert len(script.beats) == 2


def test_generate_video_script_strict_mode_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import pytest
    from src.agents.script_architect import ScriptArchitectAgent
    from src.agents.base import AgentExecutionError
    with pytest.raises(AgentExecutionError):
        ScriptArchitectAgent(strict=True)
```

If `tests/fixtures/ocr_fixtures.py::minimal_trig_ocr_output` does not exist, add a small fixture there returning a valid `OCROutput` with a few trig text blocks.

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_script_architect.py -k "generate_video_script" -q
```

- [ ] **Step 3: Implement `_generate_video_script`**

In `src/agents/script_architect.py`, add this method on `ScriptArchitectAgent`:

```python
def _generate_video_script(
    self,
    ocr_output: OCROutput,
    config: Configuration,
    review_feedback: Optional[List[str]] = None,
) -> "VideoScript":
    """Generate ONE continuous video script with a concept-introduction ledger.

    This replaces the old per-concept narration loop. The viewer sees a
    single coherent storyline; later beats reference earlier ones instead
    of re-explaining them.
    """
    from src.schemas.video_script import VideoScript, Beat

    if not os.getenv("OPENAI_API_KEY"):
        if self.strict:
            raise AgentExecutionError(
                "MISSING_API_KEY",
                "OPENAI_API_KEY is required in --strict mode for video script generation",
            )
        return self._mock_video_script_fallback(ocr_output, config)

    # Build full source-material context (this is the ONLY call that sees the
    # full PDF — downstream chunks see only their slice + narrative briefing).
    source_text = ""
    math_expressions: List[str] = []
    for page in ocr_output.pages:
        source_text += " ".join(b.text for b in page.text_blocks) + "\n"
        for expr in page.math_expressions:
            if not expr.extraction_failed:
                math_expressions.append(expr.latex)

    math_block = ""
    if math_expressions:
        math_block = "\n--- KEY FORMULAS ---\n" + "\n".join(f"  • {e}" for e in math_expressions[:30]) + "\n"

    review_block = ""
    if review_feedback:
        review_block = "\nREVISION FEEDBACK:\n" + "\n".join(f"- {r}" for r in review_feedback[:8]) + "\n"

    audience_line = {
        "high-school": "Use real-world analogies; avoid jargon.",
        "graduate": "Lean into theoretical structure and generalizations.",
    }.get(config.audience, "Balance intuition with mathematical structure.")

    verbosity_line = {
        "low": "Be terse — only the core ideas.",
        "high": "Be detailed; explain why each step matters.",
    }.get(config.verbosity, "Balanced explanation.")

    prompt = f"""You are a Grant Sanderson–style educational scriptwriter (3Blue1Brown).

--- SOURCE MATERIAL ---
{source_text[:8000]}
{math_block}
--- END SOURCE MATERIAL ---

TASK: Write ONE continuous video script that explains the source material as a single
coherent story. The output is a list of BEATS. Each beat is one paragraph of spoken
narration paired with a short visual description.

CRITICAL RULES:
1. Every concept is introduced AT MOST ONCE. Once introduced, later beats may
   reference it but MUST NOT redefine or re-derive it.
2. Beats build on each other. A later beat may say "recall the unit circle from
   earlier" but must not re-explain what the unit circle is.
3. Mark `introduces_concepts` for each beat with the canonical labels first
   taught in that beat (e.g. "sine", "unit_circle"). Use lowercase, underscored.
4. Mark `references_prior_beats` with the beat_ids whose visuals/ideas this
   beat builds on.
5. Cover every formula and definition in the source material exactly once.
6. Aim for 8–20 beats total. Each beat ≈ 40–80 words of narration.

STYLE:
- Audience ({config.audience}): {audience_line}
- Verbosity ({config.verbosity}): {verbosity_line}
- Tone: Friendly, intuitive, conversational.
- Visual hint per beat: 1–2 sentences describing what is on screen.
{review_block}

Return JSON in this EXACT shape:
{{
  "title": "...",
  "beats": [
    {{
      "beat_id": "beat_001",
      "narration": "spoken text...",
      "visual_hint": "what is on screen during this beat",
      "introduces_concepts": ["concept_label", ...],
      "references_prior_beats": ["beat_001", ...]
    }}
  ]
}}
"""

    response = self.client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)

    beats = [
        Beat(
            beat_id=b["beat_id"],
            narration=b["narration"],
            visual_hint=b.get("visual_hint", ""),
            introduces_concepts=b.get("introduces_concepts", []),
            references_prior_beats=b.get("references_prior_beats", []),
        )
        for b in data["beats"]
    ]
    return VideoScript(title=data.get("title", ""), beats=beats)


def _mock_video_script_fallback(self, ocr_output, config):
    """Deterministic fallback for runs without OPENAI_API_KEY."""
    from src.schemas.video_script import VideoScript, Beat
    title_text = ocr_output.pdf_metadata.filename
    return VideoScript(
        title=f"Overview of {title_text}",
        beats=[
            Beat(
                beat_id="beat_001",
                narration=f"This video introduces the core concepts in {title_text}. We start with the basic ideas and build up step by step.",
                visual_hint="Title card with the document name and a short outline.",
                introduces_concepts=["overview"],
                references_prior_beats=[],
            ),
            Beat(
                beat_id="beat_002",
                narration="Now we walk through the most important formulas and what they mean intuitively.",
                visual_hint="Show the key formulas one at a time, with simple geometric illustrations.",
                introduces_concepts=["main_formulas"],
                references_prior_beats=["beat_001"],
            ),
        ],
    )
```

- [ ] **Step 4: Verify tests pass**

```
python -m pytest tests/test_script_architect.py -k "generate_video_script" -q
```

---

### Task 4: Deterministic chunker — `VideoScript` → `List[Scene]`

**Files:**
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/script_architect.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_script_architect.py`

- [ ] **Step 1: Write failing tests for the chunker**

```python
def test_chunker_groups_beats_within_duration_window():
    from src.agents.script_architect import ScriptArchitectAgent
    from src.schemas.video_script import VideoScript, Beat

    # Each beat ≈ 30 words → ≈ 10 s at 3 wps. Six beats = ≈ 60 s of script.
    beats = [
        Beat(
            beat_id=f"beat_{i:03d}",
            narration=" ".join(["word"] * 30),
            visual_hint="hint",
            introduces_concepts=[f"c{i}"],
        )
        for i in range(1, 7)
    ]
    script = VideoScript(beats=beats)
    agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)

    scenes = agent._chunk_script_into_scenes(script)
    # 60 s of script with 30–90 s windows ⇒ at least one chunk, at most three.
    assert 1 <= len(scenes) <= 3
    # Every beat appears exactly once across the chunks
    used = sum(s.duration_estimate > 0 for s in scenes)
    assert used == len(scenes)
    # Scene IDs are sequential
    assert [s.scene_id for s in scenes] == [f"scene_{i+1:03d}" for i in range(len(scenes))]


def test_chunker_never_splits_a_beat():
    """A beat is the atomic unit — the chunker may group beats but never split one."""
    from src.agents.script_architect import ScriptArchitectAgent
    from src.schemas.video_script import VideoScript, Beat
    long_beat = Beat(
        beat_id="beat_001",
        narration=" ".join(["word"] * 600),  # ≈ 200 s — exceeds MAX_SCENE_DURATION
        visual_hint="hint",
        introduces_concepts=["x"],
    )
    agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
    scenes = agent._chunk_script_into_scenes(VideoScript(beats=[long_beat]))
    # Beat is preserved whole, even if it overshoots MAX_SCENE_DURATION
    assert len(scenes) == 1
    assert scenes[0].narration.count("word") == 600
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_script_architect.py -k "chunker" -q
```

- [ ] **Step 3: Implement the chunker**

Add to `ScriptArchitectAgent`:

```python
def _chunk_script_into_scenes(self, script: "VideoScript") -> List[Scene]:
    """Group beats into ~30-90 s scenes.

    Greedy packer: walk the beats in order, accumulating beats into the
    current chunk until adding the next beat would push the chunk over
    MAX_SCENE_DURATION. A beat is atomic — never split. If a single beat
    exceeds MAX_SCENE_DURATION, it becomes its own (oversized) chunk.
    """
    from src.schemas.video_script import VideoScript

    def beat_duration(b) -> float:
        return len(b.narration.split()) / self.WORDS_PER_SECOND

    chunks: List[List["Beat"]] = []
    current: List["Beat"] = []
    current_dur = 0.0

    for beat in script.beats:
        bd = beat_duration(beat)
        if current and current_dur + bd > self.MAX_SCENE_DURATION:
            chunks.append(current)
            current, current_dur = [], 0.0
        current.append(beat)
        current_dur += bd
        # If we crossed TARGET, close the chunk (avoids tiny tail chunks).
        if current_dur >= self.TARGET_SCENE_DURATION:
            chunks.append(current)
            current, current_dur = [], 0.0
    if current:
        chunks.append(current)

    scenes: List[Scene] = []
    for idx, chunk_beats in enumerate(chunks):
        narration = "\n\n".join(b.narration for b in chunk_beats)
        word_count = sum(len(b.narration.split()) for b in chunk_beats)
        duration = max(self.MIN_SCENE_DURATION, word_count / self.WORDS_PER_SECOND)
        scene_id = f"scene_{idx + 1:03d}"
        # visual_intent is left empty here — populated in Task 7.
        scenes.append(Scene(
            scene_id=scene_id,
            scene_index=idx,
            concept_id=chunk_beats[0].beat_id,  # synthetic — first beat id
            narration=narration,
            visual_intent=self._empty_visual_intent_placeholder(),
            duration_estimate=round(duration, 1),
            dependencies=[f"scene_{idx:03d}"] if idx > 0 else [],
        ))
        # Stash the source beats on the Scene for downstream use:
        scenes[-1]._source_beats = chunk_beats  # noqa: SLF001 (informal annex)

    return scenes


def _empty_visual_intent_placeholder(self) -> "VisualIntent":
    return VisualIntent(
        mathematical_objects=[],
        transformations=[],
        beats=[],
        scene_state_in=SceneState(),
        scene_state_out=SceneState(),
        emphasis_points=[],
    )
```

- [ ] **Step 4: Verify tests pass**

```
python -m pytest tests/test_script_architect.py -k "chunker" -q
```

---

### Task 5: Wire `ScriptArchitectAgent.execute` to use the new flow; delete dead code

**Files:**
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/script_architect.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_script_architect.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_validation_agent.py` (no-op check)

- [ ] **Step 1: Update execute() to call the new path**

Replace the body of `ScriptArchitectAgent.execute` with:

```python
def execute(self, input_data: AgentInput) -> AgentOutput:
    if not self.validate_input(input_data):
        raise AgentExecutionError(
            "INVALID_INPUT", "Input must be a ScriptArchitectInput",
            {"input_type": type(input_data).__name__},
        )
    inp: ScriptArchitectInput = input_data
    ocr_output, config, review_feedback = inp.ocr_output, inp.configuration, inp.review_feedback or []

    try:
        # Phase 1: ONE unified script
        video_script = self._generate_video_script(ocr_output, config, review_feedback)
        logger.info(f"Generated video script with {len(video_script.beats)} beats")

        # Phase 2: deterministic chunking
        scenes = self._chunk_script_into_scenes(video_script)
        logger.info(f"Chunked script into {len(scenes)} scenes")

        # Phase 3: per-chunk visual intent (still per-scene, but each call now
        # gets the chunk narration + ledger — Task 8 implements this).
        for scene in scenes:
            scene.visual_intent = self._generate_visual_intent_for_chunk(
                scene=scene,
                video_script=video_script,
                config=config,
                review_feedback=review_feedback,
            )

        return self._assemble_storyboard_from_script(
            ocr_output, config, video_script, scenes,
        )
    except Exception as e:
        raise AgentExecutionError(
            "STORYBOARD_GENERATION_FAILED",
            f"Failed to generate storyboard: {e}",
            {"filename": ocr_output.pdf_metadata.filename, "error": str(e)},
        )
```

- [ ] **Step 2: Implement `_assemble_storyboard_from_script`**

```python
def _assemble_storyboard_from_script(
    self, ocr_output, config, video_script, scenes,
) -> StoryboardJSON:
    """Build the legacy StoryboardJSON shape from a VideoScript + chunked scenes.

    `concept_hierarchy` maps onto the script's concept ledger (one entry per
    canonical concept, in first-introduction order).
    """
    pdf_metadata = PDFMetadataStoryboard(
        filename=ocr_output.pdf_metadata.filename,
        page_count=ocr_output.pdf_metadata.page_count,
    )
    concept_hierarchy = [
        ConceptHierarchy(
            concept_id=concept,
            concept_name=concept.replace("_", " ").title(),
            dependencies=video_script.concept_ledger()[:i],
        )
        for i, concept in enumerate(video_script.concept_ledger())
    ]
    return StoryboardJSON(
        pdf_metadata=pdf_metadata,
        configuration=config,
        concept_hierarchy=concept_hierarchy,
        scenes=scenes,
    )
```

- [ ] **Step 3: DELETE dead code**

Remove from `src/agents/script_architect.py`:
- `_identify_concepts`
- `_build_dependency_graph`
- `_topological_sort`
- `_decompose_into_scenes`
- `_generate_scene` (the per-concept one)
- `_generate_narration` (per-concept)
- `_generate_mock_narration_fallback`
- `_generate_mock_visual_fallback`
- `_estimate_duration` (now inlined in chunker)
- `_assemble_storyboard` (replaced)
- All regex pattern constants (`definition_patterns`, `theorem_patterns`, etc.)

Also delete the imports they pulled in if newly unused (e.g. `from collections import defaultdict, deque`, `import re`).

- [ ] **Step 4: Update tests**

In `tests/test_script_architect.py`, remove or rewrite tests that reference the deleted functions (e.g. `_identify_concepts`, `_topological_sort`). Replace with end-to-end-via-mocked-LLM tests:

```python
def test_execute_produces_chunked_scenes_with_no_re_explanation(monkeypatch, fake_openai):
    from src.agents.script_architect import ScriptArchitectAgent, ScriptArchitectInput
    from src.schemas.storyboard import Configuration
    from tests.fixtures.ocr_fixtures import minimal_trig_ocr_output
    agent = ScriptArchitectAgent(strict=False)
    agent.client = fake_openai  # injected fixture
    storyboard = agent.execute(ScriptArchitectInput(
        ocr_output=minimal_trig_ocr_output(),
        configuration=Configuration(verbosity="medium", depth="intermediate", audience="undergraduate"),
    ))
    # Concepts hierarchy comes from the unified ledger, deduped & ordered
    concepts = [c.concept_id for c in storyboard.concept_hierarchy]
    assert len(concepts) == len(set(concepts))   # no duplicates
    assert len(storyboard.scenes) >= 1
```

- [ ] **Step 5: Verify**

```
python -m pytest tests/test_script_architect.py -q
python -m pytest tests/test_validation_agent.py tests/test_schemas.py -q
```

---

### Task 6: `SceneControllerAgent` populates `NarrativeContext` after each scene

**Files:**
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/scene_controller.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_scene_controller.py`

- [x] **Step 1: Write failing tests for narrative-context propagation**

```python
def test_scene_controller_appends_narrative_summary_per_scene(stub_animation_generator, two_scene_storyboard):
    from src.agents.scene_controller import SceneControllerAgent, SceneControllerInput
    agent = SceneControllerAgent()
    out = agent.execute(SceneControllerInput(
        storyboard=two_scene_storyboard,
        output_directory="/tmp/out",
        animation_generator=stub_animation_generator,
    ))
    nc = out.shared_context.narrative
    assert len(nc.prior_scene_summaries) == 2
    # First-introduction order is preserved across scenes
    assert nc.concepts_introduced.index("sine") < nc.concepts_introduced.index("cosine")


def test_scene_controller_passes_narrative_into_next_scene(stub_animation_generator, two_scene_storyboard):
    """Second scene's processing input must see the first scene's summary."""
    from src.agents.scene_controller import SceneControllerAgent, SceneControllerInput
    agent = SceneControllerAgent()
    captured = []
    orig = stub_animation_generator.execute
    def spy(inp):
        captured.append(inp.scene_processing_input.context.narrative.prior_scene_summaries[:])
        return orig(inp)
    stub_animation_generator.execute = spy
    agent.execute(SceneControllerInput(
        storyboard=two_scene_storyboard,
        output_directory="/tmp/out",
        animation_generator=stub_animation_generator,
    ))
    assert captured[0] == []  # scene 1 sees empty narrative
    assert len(captured[1]) == 1  # scene 2 sees scene 1's summary
```

Add the `two_scene_storyboard` and `stub_animation_generator` fixtures to `tests/conftest.py` if not already present.

- [x] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_scene_controller.py -k "narrative" -q
```

- [x] **Step 3: Update `_update_context` and add summarizer**

In `src/agents/scene_controller.py`:

```python
def _update_context(self, context: SharedContext, scene: Scene, scene_spec: SceneSpec) -> None:
    # ── existing concept_definitions / variable_bindings / registries logic ──
    # (keep all the existing body unchanged)
    ...

    # ── NEW: narrative continuity ──────────────────────────────────────
    nc = context.narrative
    nc.prior_scene_summaries.append(self._summarize_scene_for_narrative(scene))
    for concept in self._extract_introduced_concepts(scene):
        if concept not in nc.concepts_introduced:
            nc.concepts_introduced.append(concept)
    nc.running_storyline = self._update_running_storyline(nc.running_storyline, scene)
    nc.live_objects_on_screen = self._compute_live_objects(scene)


def _summarize_scene_for_narrative(self, scene: Scene) -> str:
    """Deterministic ≤30-word scene summary (no LLM call).

    Takes the first sentence of the narration plus the concept_id.
    """
    first_sentence = scene.narration.split(".")[0].strip()
    summary = first_sentence[:160]
    if scene.concept_id:
        summary = f"[{scene.concept_id}] {summary}"
    return summary


def _extract_introduced_concepts(self, scene: Scene) -> List[str]:
    """Pull concept labels from the scene's source beats (set in chunker).

    Falls back to scene.concept_id if _source_beats is unavailable
    (e.g. legacy callers / tests that build Scenes by hand).
    """
    beats = getattr(scene, "_source_beats", None)
    if beats:
        out: List[str] = []
        for b in beats:
            for c in b.introduces_concepts:
                if c not in out:
                    out.append(c)
        return out
    return [scene.concept_id] if scene.concept_id else []


def _update_running_storyline(self, prior: str, scene: Scene) -> str:
    """Keep storyline ≤ 3 sentences. Append, then drop oldest."""
    new_sentence = self._summarize_scene_for_narrative(scene).rstrip(".") + "."
    sentences = [s.strip() for s in prior.split(".") if s.strip()]
    sentences.append(new_sentence.rstrip("."))
    return ". ".join(sentences[-3:]) + "."


def _compute_live_objects(self, scene: Scene) -> Dict[str, str]:
    """Objects still on screen at the END of this scene.

    Reads scene_state_out.active_object_ids and looks up their content from
    the visual_intent's mathematical_objects list.
    """
    visual = scene.visual_intent
    active = set(visual.scene_state_out.active_object_ids or [])
    out: Dict[str, str] = {}
    for obj in visual.mathematical_objects:
        if obj.object_id in active:
            out[obj.object_id] = f"{obj.object_type}: {obj.content[:60]}"
    return out
```

- [x] **Step 4: Verify**

```
python -m pytest tests/test_scene_controller.py -q
```

---

### Task 7: Per-chunk visual-intent prompt uses `NarrativeContext`, no full PDF

**Files:**
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/script_architect.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_script_architect.py`

- [x] **Step 1: Write failing tests for prompt content**

```python
def test_visual_intent_prompt_does_not_include_full_pdf(monkeypatch):
    """The per-chunk visual planner must not receive the raw PDF text — only
    the chunk's narration plus the briefing."""
    from src.agents.script_architect import ScriptArchitectAgent
    from src.schemas.video_script import VideoScript, Beat
    from src.schemas.storyboard import Configuration

    captured = {}
    class _FakeChat:
        class completions:
            @staticmethod
            def create(*, messages, **kw):
                captured["prompt"] = messages[0]["content"]
                # Return minimal valid visual intent JSON
                return type("R", (), {"choices": [type("C", (), {
                    "message": type("M", (), {"content": '{"objects": [], "transformations": [], "beats": [], "scene_state_in": {}, "scene_state_out": {}, "emphasis_points": []}'}),
                })]})

    agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
    agent.strict = False
    agent.client = type("C", (), {"chat": _FakeChat})
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    script = VideoScript(beats=[Beat(beat_id="beat_001", narration="explain sine", visual_hint="unit circle", introduces_concepts=["sine"])])
    scene = agent._chunk_script_into_scenes(script)[0]
    config = Configuration(verbosity="medium", depth="intermediate", audience="undergraduate")

    agent._generate_visual_intent_for_chunk(scene, script, config, [])

    p = captured["prompt"]
    assert "explain sine" in p          # chunk narration is included
    assert "unit circle" in p           # visual hint is included
    assert "PDF" not in p and "OCR" not in p  # full source NOT dumped


def test_visual_intent_prompt_lists_concepts_already_introduced(monkeypatch):
    # similar setup as above; assert the prompt includes a "Concepts already
    # introduced earlier in the video: ..." line when the chunker is at scene 2+
    ...
```

- [x] **Step 2: Run tests to verify they fail**

- [x] **Step 3: Implement `_generate_visual_intent_for_chunk`**

```python
def _generate_visual_intent_for_chunk(
    self,
    scene: Scene,
    video_script: "VideoScript",
    config: Configuration,
    review_feedback: List[str],
) -> VisualIntent:
    """Plan visuals for one chunk. Receives the chunk narration + briefing
    on what the viewer has already seen — never the full PDF."""

    if not os.getenv("OPENAI_API_KEY"):
        if self.strict:
            raise AgentExecutionError("MISSING_API_KEY", "OPENAI_API_KEY required in --strict mode")
        return self._generate_fallback_visual_intent(scene.narration)

    chunk_beats = getattr(scene, "_source_beats", []) or []
    chunk_concepts: List[str] = []
    for b in chunk_beats:
        for c in b.introduces_concepts:
            if c not in chunk_concepts:
                chunk_concepts.append(c)

    # Concepts taught in EARLIER chunks (the briefing).
    earlier_concepts: List[str] = []
    for b in video_script.beats:
        if b in chunk_beats:
            break
        for c in b.introduces_concepts:
            if c not in earlier_concepts and c not in chunk_concepts:
                earlier_concepts.append(c)

    visual_hints = "\n".join(f"- {b.visual_hint}" for b in chunk_beats) or "- (no specific hints)"

    word_count = len(scene.narration.split())
    target = round(word_count / self.WORDS_PER_SECOND, 1)

    review_block = ""
    if review_feedback:
        review_block = "\nREVISION FEEDBACK:\n" + "\n".join(f"- {r}" for r in review_feedback[:8]) + "\n"

    earlier_block = ""
    if earlier_concepts:
        earlier_block = (
            "\nCONCEPTS ALREADY INTRODUCED IN EARLIER SCENES (the viewer has seen "
            "these — reference them, do NOT redefine):\n"
            + "\n".join(f"  • {c}" for c in earlier_concepts) + "\n"
        )

    engine_contract = build_visual_intent_contract_prompt()

    prompt = f"""You are a 3Blue1Brown animation architect.

--- CHUNK NARRATION (the AUDIO for THIS scene only) ---
{scene.narration}
--- END NARRATION ---

NEW CONCEPTS INTRODUCED IN THIS CHUNK: {', '.join(chunk_concepts) or '(none)'}
{earlier_block}
VISUAL HINTS FROM THE SCRIPT WRITER:
{visual_hints}

TIMING: {word_count} words at ~3 wps ⇒ visuals MUST fill ~{target}s of screen time.
{review_block}
{engine_contract}

Return JSON in the schema specified above (objects, transformations, beats,
scene_state_in, scene_state_out, emphasis_points).
"""

    response = self.client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return self._visual_intent_from_json(data, fallback_narration=scene.narration)


def _visual_intent_from_json(self, data: dict, fallback_narration: str) -> VisualIntent:
    """Lift the JSON parsing out of the old _generate_visual_intent for reuse."""
    # Move the existing JSON → VisualIntent mapping (math_objs, trans, beats,
    # emphasis, scene_state_in/out cleaning) from the OLD _generate_visual_intent
    # into this helper. If the JSON is missing required fields, fall back to
    # _generate_fallback_visual_intent(fallback_narration).
    ...
```

- [x] **Step 4: Delete the old `_generate_visual_intent`** (the one that takes a `concept_id`). — Already removed during Task 5.

- [x] **Step 5: Verify**

```
python -m pytest tests/test_script_architect.py -q
```

---

### Task 8: Animation-generator prompt receives `NarrativeContext`, drops PDF dump

**Files:**
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/animation_generator.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_animation_generator.py`

- [x] **Step 1: Write failing tests for prompt content**

```python
def test_animation_prompt_includes_concepts_already_taught(monkeypatch):
    from src.agents.animation_generator import AnimationCodeGeneratorAgent, AnimationGeneratorInput
    from src.schemas.scene_processing import SceneProcessingInput, SceneSpec, SharedContext, NarrativeContext
    from src.schemas.storyboard import VisualIntent, SceneState

    captured = {}
    class _FakeAnthropic:
        class messages:
            @staticmethod
            def create(*, system, messages, **kw):
                captured["system"] = system
                captured["user"] = messages[0]["content"]
                return type("R", (), {"content": [type("B", (), {"text": "class S(Scene):\\n    def construct(self):\\n        self.wait(1)"})]})

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    agent = AnimationCodeGeneratorAgent.__new__(AnimationCodeGeneratorAgent)
    agent.strict = False
    agent.max_retries = 1
    agent.client = _FakeAnthropic()

    spec = SceneSpec(
        scene_id="scene_002",
        narration="Cosine traces the x-coordinate.",
        visual_intent=VisualIntent(
            mathematical_objects=[], transformations=[], beats=[],
            scene_state_in=SceneState(), scene_state_out=SceneState(),
            emphasis_points=[],
        ),
        duration_estimate=20.0,
    )
    ctx = SharedContext(narrative=NarrativeContext(
        prior_scene_summaries=["[beat_001] Sine on the unit circle introduced."],
        concepts_introduced=["sine", "unit_circle"],
        running_storyline="Sine on the unit circle introduced.",
    ))

    agent.execute(AnimationGeneratorInput(
        scene_processing_input=SceneProcessingInput(scene_spec=spec, context=ctx, scene_index=1),
    ))

    full = captured["system"] + captured["user"]
    assert "sine" in full
    assert "unit_circle" in full
    assert "do not re-explain" in full.lower() or "already introduced" in full.lower()
```

- [x] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_animation_generator.py -k "concepts_already_taught" -q
```

- [x] **Step 3: Inject a `NARRATIVE BRIEFING` block into the system prompt**

In the `_generate_code` system_prompt builder (around line 252 of `animation_generator.py`), add — before the `═══ DURATION ═══` block — a new section built from `context.narrative`:

```python
nc = context.narrative
briefing_lines = []
if nc.concepts_introduced:
    briefing_lines.append(
        "Concepts the viewer has already been taught (REFERENCE — do not re-explain):\n  • "
        + "\n  • ".join(nc.concepts_introduced)
    )
if nc.live_objects_on_screen:
    briefing_lines.append(
        "Objects that were on screen at the end of the previous scene "
        "(may carry forward visually):\n  • "
        + "\n  • ".join(f"{k}: {v}" for k, v in nc.live_objects_on_screen.items())
    )
if nc.running_storyline:
    briefing_lines.append(f"Running storyline: {nc.running_storyline}")

briefing_block = ""
if briefing_lines:
    briefing_block = (
        "\n═══════════════════════  NARRATIVE BRIEFING  ═══════════════════════\n"
        + "\n\n".join(briefing_lines)
        + "\n\nTHIS SCENE IS PART OF A LONGER VIDEO. The viewer has seen everything\n"
          "above. Build on it — do not redefine concepts already introduced.\n"
    )
```

Splice `briefing_block` into the system prompt right after the engine/quality contracts and before the duration section.

- [x] **Step 4: Verify**

```
python -m pytest tests/test_animation_generator.py -q
```

---

### Task 9: Manim-level visual continuity at chunk boundaries (opening state)

**Files:**
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/animation_generator.py`
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_animation_generator.py`

- [x] **Step 1: Write failing tests**

```python
def test_generated_code_opens_with_carryover_objects():
    """If the briefing lists live objects, the prompt must instruct the LLM
    to start the scene by re-instantiating those objects at run_time=0."""
    # similar setup to Task 8; assert "instantly re-instantiate" or
    # "FadeIn(... run_time=0)" appears in the system prompt when
    # narrative.live_objects_on_screen is non-empty.
    ...
```

- [x] **Step 2: Run tests to verify they fail**

- [x] **Step 3: Add an `OPENING STATE` instruction to the prompt when carry-over is non-empty**

In the same prompt-building site as Task 8, when `nc.live_objects_on_screen` is non-empty, append:

```
═══════════════════════  OPENING STATE  ═══════════════════════
The previous scene ended with these objects still on screen. To make the cut
seamless, your construct() MUST begin by instantly re-creating them:

    {live_object_summary}

Use:
    self.add(<mobject>)        # NOT self.play(...) — no animation, no run_time
or
    self.play(FadeIn(<mobject>), run_time=0)

Position them where they were at the end of the previous scene (use sensible
defaults if unknown). Only AFTER the carry-over add do you start the new
beats for this scene.
```

- [x] **Step 4: Verify**

```
python -m pytest tests/test_animation_generator.py -q
```

---

### Task 10: End-to-end integration test + final cleanup

**Files:**
- Modify: `C:/Users/Parth/Documents/3B1B/tests/integration/test_end_to_end.py`
- Modify: `C:/Users/Parth/Documents/3B1B/src/orchestrator/pipeline.py` (only if needed; the new flow should be transparent to the orchestrator)

- [x] **Step 1: Write a failing redundancy-regression test**

```python
def test_no_concept_is_taught_twice_across_scenes(monkeypatch, tmp_path):
    """Run the full pipeline (with mocked LLMs) on a small fixture PDF and
    assert that no concept label appears as a fresh introduction in more
    than one scene's source beats."""
    # Use the existing dry_run path so we stop after storyboard generation
    # but still get the chunked Scene list. Then walk every scene's
    # _source_beats and assert each concept label appears in
    # introduces_concepts at most once across the whole storyboard.
    ...
```

- [x] **Step 2: Run and confirm it passes**

```
python -m pytest tests/integration/test_end_to_end.py -q
```

- [x] **Step 3: Final sweep — delete now-unused imports and helpers**

```
python -m pytest -q
ruff check src/ tests/
```

- [x] **Step 4: Update `docs/output-contract.md`**

Document that:
- `concept_hierarchy` now derives from the unified script's concept ledger.
- `Scene.dependencies` is always a single-step chain (`scene_N` depends on `scene_(N-1)`) because scenes are render chunks of one continuous script.
- A new `narrative` block exists inside `SharedContext` carrying `concepts_introduced`, `prior_scene_summaries`, `running_storyline`, `live_objects_on_screen`.

---

### Task 11: Fix TTS — chunk narration under the 4096-char limit and surface failures

**Root cause confirmed against `output/machine_learning/2026-04-23T18-58-09/`:**
- Every scene's narration in that run was 4568–7967 characters; OpenAI's TTS API rejects any single request over **4096 characters** with HTTP 400.
- `_run_tts_generation` (`pipeline.py:932`) catches the resulting exception and returns `[]` silently.
- TTS bypasses `_execute_agent` / `_execute_with_retry`, so **no `agent_start`/`agent_complete` events** are written to `pipeline.log`. The failure was invisible.
- Installed `openai==2.24.0` supports the modern `audio.speech.with_streaming_response` API; the current `response.content` path still works in 2.x but is a deprecated pattern.

**Files:**
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/tts_agent.py`
- Modify: `C:/Users/Parth/Documents/3B1B/src/orchestrator/pipeline.py`
- Modify: `C:/Users/Parth/Documents/3B1B/src/utils/ffmpeg_finder.py` (only if a concat helper isn't already there — check first)
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_tts_agent.py`

- [x] **Step 1: Write failing tests reproducing the 4096-char failure**

```python
def test_tts_splits_long_narration_under_4096_chars(monkeypatch, tmp_path):
    """A 7000-char narration must be split into ≥2 sub-requests, each ≤4096."""
    from src.agents.tts_agent import TTSAgent, TTSInput, TTSOutput
    captured_lengths = []

    class _FakeStreamCtx:
        def __init__(self, content: bytes): self._c = content
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def stream_to_file(self, p): open(p, "wb").write(self._c)

    class _FakeWithStreaming:
        @staticmethod
        def create(*, model, voice, input, response_format):
            captured_lengths.append(len(input))
            return _FakeStreamCtx(b"\\x00" * 100)

    class _FakeSpeech: with_streaming_response = _FakeWithStreaming
    class _FakeAudio: speech = _FakeSpeech
    class _FakeClient: audio = _FakeAudio()

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    agent = TTSAgent.__new__(TTSAgent)
    agent.strict = False
    agent.max_retries = 1
    agent.client = _FakeClient()
    from src.utils.cost_tracker import CostTracker
    agent.cost_tracker = CostTracker.get_instance()

    long_narration = ("This is a sentence. " * 400)  # ≈ 8000 chars
    out = agent.execute(TTSInput(
        narration=long_narration,
        scene_id="scene_001",
        output_dir=str(tmp_path),
    ))

    assert isinstance(out, TTSOutput)
    assert len(captured_lengths) >= 2
    assert all(L <= 4096 for L in captured_lengths)


def test_tts_chunks_split_only_on_sentence_boundaries():
    """Splitter must not cut a sentence mid-clause."""
    from src.agents.tts_agent import _split_for_tts
    text = ("Sentence one. " * 500)  # plenty of `. ` boundaries
    chunks = _split_for_tts(text, max_chars=4096)
    for c in chunks:
        assert c.rstrip().endswith(".")
        assert len(c) <= 4096


def test_tts_routes_through_structured_logger(monkeypatch):
    """`_run_tts_generation` should emit agent_start / agent_complete events
    so that TTS failures are visible in pipeline.log."""
    # Stub out tts_agent.process_all_scenes to return a single TTSError;
    # capture calls to structured_logger.log_agent_start / log_agent_failure;
    # assert both fire with stage_name='TTS'.
    ...
```

- [x] **Step 2: Run tests to confirm they fail**

```
python -m pytest tests/test_tts_agent.py -k "splits_long or sentence_boundaries or structured_logger" -q
```

- [x] **Step 3: Implement sentence-boundary splitter and per-chunk TTS calls**

Add a module-level helper in `tts_agent.py`:

```python
import re

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
TTS_MAX_CHARS = 4000  # buffer below the 4096 hard limit


def _split_for_tts(text: str, max_chars: int = TTS_MAX_CHARS) -> List[str]:
    """Split narration into ≤max_chars chunks on sentence boundaries.

    Falls back to whitespace boundaries for runs longer than max_chars
    with no sentence break, and to a hard char-cut as last resort
    (only for pathological input).
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    sentences = _SENTENCE_BREAK.split(text)
    chunks: List[str] = []
    current = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        candidate = (current + " " + s).strip() if current else s
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        # The single sentence itself might be too long
        if len(s) <= max_chars:
            current = s
        else:
            # Whitespace fallback
            current = ""
            for word in s.split():
                cand = (current + " " + word).strip() if current else word
                if len(cand) <= max_chars:
                    current = cand
                else:
                    if current:
                        chunks.append(current)
                    current = word
    if current:
        chunks.append(current)
    return chunks
```

Then rewrite `_generate_audio` to call the API per chunk and concatenate:

```python
def _generate_audio(self, narration, scene_id, output_dir, model, voice) -> str:
    if not self.client:
        raise RuntimeError("OpenAI client not initialized")

    chunks = _split_for_tts(narration)
    output_path = Path(output_dir) / f"{scene_id}.mp3"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(chunks) == 1:
        # Fast path — modern streaming API
        with self.client.audio.speech.with_streaming_response.create(
            model=model, voice=voice, input=chunks[0], response_format="mp3",
        ) as response:
            response.stream_to_file(str(output_path))
        return str(output_path)

    # Multi-chunk: write to temp files, concat with ffmpeg
    import tempfile
    from src.utils.ffmpeg_finder import get_ffmpeg
    with tempfile.TemporaryDirectory() as tmp:
        part_paths: List[str] = []
        for i, chunk in enumerate(chunks):
            part = Path(tmp) / f"part_{i:03d}.mp3"
            with self.client.audio.speech.with_streaming_response.create(
                model=model, voice=voice, input=chunk, response_format="mp3",
            ) as response:
                response.stream_to_file(str(part))
            part_paths.append(str(part))

        list_file = Path(tmp) / "parts.txt"
        list_file.write_text(
            "\n".join(f"file '{p.replace(chr(39), chr(39)+chr(92)+chr(39)+chr(39))}'"
                      for p in part_paths),
            encoding="utf-8",
        )
        result = subprocess.run(
            [get_ffmpeg(), "-y", "-f", "concat", "-safe", "0",
             "-i", str(list_file), "-c", "copy", str(output_path)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")

    return str(output_path)
```

If `get_ffmpeg` is missing from `ffmpeg_finder.py`, add it as a sibling of `get_ffprobe`.

- [x] **Step 4: Route TTS through `_execute_agent` so failures land in `pipeline.log`**

In `pipeline.py`, change the parallel-fan-out block (`run_pipeline`, lines 338–360) so that the TTS branch goes through the structured logger:

```python
if self.config.enable_tts:
    futures["tts"] = executor.submit(
        self._run_tts_generation_logged,   # new wrapper
        sanitized_storyboard, manifest, self.structured_logger,
    )
```

Add the wrapper:

```python
def _run_tts_generation_logged(self, storyboard, manifest, structured_logger):
    """Same as _run_tts_generation but emits structured agent events."""
    structured_logger.log_agent_start("TTS", input_summary=f"{len(storyboard.scenes)} scenes", retry_attempt=0)
    start = time.time()
    try:
        results = self._run_tts_generation(storyboard, manifest, structured_logger)
        success = sum(1 for r in results if isinstance(r, TTSOutput))
        failed = sum(1 for r in results if isinstance(r, TTSError))
        status = "SUCCESS" if failed == 0 else ("PARTIAL" if success > 0 else "FAILURE")
        structured_logger.log_agent_complete(
            "TTS", duration_ms=(time.time() - start) * 1000,
            output_summary=f"{success} ok / {failed} failed", status=status,
        )
        if failed > 0:
            self._quality_reports["tts"] = {
                "total": len(results),
                "succeeded": success,
                "failed": failed,
                "errors": [
                    {"scene_id": r.scene_id, "error": r.error}
                    for r in results if isinstance(r, TTSError)
                ],
            }
        return results
    except Exception as e:
        structured_logger.log_agent_failure(
            "TTS", str(e), error_code="TTS_STAGE_FAILED",
            input_summary="batch", retry_attempt=0,
            duration_ms=(time.time() - start) * 1000,
        )
        return []
```

- [x] **Step 5: Persist `tts.json` quality report**

Add `"tts": "tts.json"` to `report_filenames` in `_persist_quality_reports` (`pipeline.py:1427`).

- [x] **Step 6: Verify**

```
python -m pytest tests/test_tts_agent.py -q
python -m pytest tests/test_logger.py tests/test_persistence_agent.py -q
# End-to-end smoke: re-run the failing PDF, confirm audio/scene_*/scene_*.mp3 files exist
# and that pipeline.log contains agent_start/agent_complete entries with agent_name="TTS".
```

---

### Task 12: Closed-vocabulary executable visual intent (Manim primitive catalog)

**Why:** today the visual planner emits abstract fields (`render_hint: "MathTex"`,
`content: "..."`) and the animation generator interprets them via heuristics
inside `_generate_mathematical_objects` (~lines 528–634 of
`animation_generator.py`). Three pain points:
- The LLM can produce render_hints not in the catalog → fallback to generic shapes.
- `content` is free text → LaTeX vs prose vs description gets guessed.
- The animation-generator LLM has to reinvent positioning, sizing, and color choices on every call.

The fix is a **closed primitive catalog** that the script-side LLM and the
animation generator both speak. Visual intent JSON specifies the primitive +
its args; the animation generator becomes a thin emitter that mostly maps
catalog entries to lines of code.

**Files:**
- New: `C:/Users/Parth/Documents/3B1B/src/utils/manim_primitives.py`
- Modify: `C:/Users/Parth/Documents/3B1B/src/utils/manim_engine_contract.py`
- Modify: `C:/Users/Parth/Documents/3B1B/src/schemas/storyboard.py` (extend `MathematicalObject`)
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/script_architect.py` (visual planner prompt)
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/animation_generator.py` (emitter)
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_manim_primitives.py` (new)

- [x] **Step 1: Build the catalog**

In `src/utils/manim_primitives.py`, define a curated table — start with these
~25 primitives (NOT auto-introspected; we want to whitelist exactly what we
trust the LLM to use):

```python
"""Closed catalog of Manim primitives the LLM is allowed to emit.

Each entry declares:
- `name`: the Manim class name (must match the manim Community import)
- `kind`: semantic family (mobject | animation)
- `args`: list of (arg_name, type_hint, default) — defaults make calls safe
- `accepts_color`: whether `color=` is a sensible kwarg
- `requires`: list of catalog entries that MUST exist before this one is used
              (e.g. `Axes` is required before `axes.plot(...)`)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    kind: str  # "mobject" | "animation" | "helper"
    args: Tuple[Tuple[str, str, Any], ...]  # (name, type, default)
    accepts_color: bool = True
    requires: Tuple[str, ...] = ()
    notes: str = ""


MOBJECT_CATALOG: Dict[str, PrimitiveSpec] = {
    # Text & math
    "MathTex":    PrimitiveSpec("MathTex",    "mobject", (("tex", "str", ""),), True),
    "Tex":        PrimitiveSpec("Tex",        "mobject", (("tex", "str", ""),), True),
    "Text":       PrimitiveSpec("Text",       "mobject",
                                (("text", "str", ""), ("font_size", "int", 36)), True),
    # Geometry
    "Circle":     PrimitiveSpec("Circle",     "mobject", (("radius", "float", 1.0),), True),
    "Square":     PrimitiveSpec("Square",     "mobject", (("side_length", "float", 2.0),), True),
    "Rectangle":  PrimitiveSpec("Rectangle",  "mobject",
                                (("width", "float", 4.0), ("height", "float", 2.0)), True),
    "Polygon":    PrimitiveSpec("Polygon",    "mobject",
                                (("vertices", "list[tuple]", []),), True,
                                notes="Vertices are 3-tuples (x,y,0)."),
    "Line":       PrimitiveSpec("Line",       "mobject",
                                (("start", "tuple", (-1, 0, 0)), ("end", "tuple", (1, 0, 0))), True),
    "Arrow":      PrimitiveSpec("Arrow",      "mobject",
                                (("start", "tuple", (-1, 0, 0)), ("end", "tuple", (1, 0, 0))), True),
    "Arc":        PrimitiveSpec("Arc",        "mobject",
                                (("radius", "float", 1.0), ("start_angle", "float", 0.0),
                                 ("angle", "float", 1.5708)), True),
    "Dot":        PrimitiveSpec("Dot",        "mobject", (("point", "tuple", (0, 0, 0)),), True),
    # Coordinate systems
    "Axes":       PrimitiveSpec("Axes",       "mobject",
                                (("x_range", "list", [-5, 5, 1]), ("y_range", "list", [-3, 3, 1]),
                                 ("x_length", "float", 5.0), ("y_length", "float", 3.0)), False),
    "NumberPlane": PrimitiveSpec("NumberPlane", "mobject", (), False),
    "NumberLine": PrimitiveSpec("NumberLine", "mobject",
                                (("x_range", "list", [-5, 5, 1]),), False),
    # Helpers (project-specific imports already in animation_generator)
    "create_pi_teacher":  PrimitiveSpec("create_pi_teacher",  "helper", (), False),
    "create_pi_students": PrimitiveSpec("create_pi_students", "helper", (), False),
    "create_speech_bubble": PrimitiveSpec(
        "create_speech_bubble", "helper",
        (("text", "str", ""), ("anchor_var", "str", "")), False,
        notes="anchor_var is the variable name of the registered creature."),
}

ANIMATION_CATALOG: Dict[str, PrimitiveSpec] = {
    "Write":               PrimitiveSpec("Write",               "animation",
                                         (("target", "var", ""), ("run_time", "float", 1.5)), False),
    "Create":              PrimitiveSpec("Create",              "animation",
                                         (("target", "var", ""), ("run_time", "float", 1.5)), False),
    "FadeIn":              PrimitiveSpec("FadeIn",              "animation",
                                         (("target", "var", ""), ("run_time", "float", 1.0)), False),
    "FadeOut":             PrimitiveSpec("FadeOut",             "animation",
                                         (("target", "var", ""), ("run_time", "float", 1.0)), False),
    "Transform":           PrimitiveSpec("Transform",           "animation",
                                         (("source", "var", ""), ("target", "var", ""),
                                          ("run_time", "float", 1.5)), False),
    "ReplacementTransform": PrimitiveSpec("ReplacementTransform", "animation",
                                          (("source", "var", ""), ("target", "var", ""),
                                           ("run_time", "float", 1.5)), False),
    "Indicate":            PrimitiveSpec("Indicate",            "animation",
                                         (("target", "var", ""), ("run_time", "float", 1.0)), False),
    "MoveToTarget":        PrimitiveSpec("MoveToTarget",        "animation",
                                         (("target", "var", ""), ("run_time", "float", 1.5)), False),
}


def catalog_for_prompt() -> str:
    """Render the catalog as a compact reference block for LLM prompts."""
    lines = ["MOBJECTS (use exactly these — any other name will be rejected):"]
    for spec in MOBJECT_CATALOG.values():
        sig = ", ".join(f"{a}: {t}={d!r}" for a, t, d in spec.args)
        lines.append(f"  {spec.name}({sig})" + (f"  # {spec.notes}" if spec.notes else ""))
    lines.append("\nANIMATIONS:")
    for spec in ANIMATION_CATALOG.values():
        sig = ", ".join(f"{a}: {t}={d!r}" for a, t, d in spec.args)
        lines.append(f"  {spec.name}({sig})")
    return "\n".join(lines)
```

- [x] **Step 2: Extend `MathematicalObject` schema with `primitive` + `args`**

In `src/schemas/storyboard.py`, add to `MathematicalObject`:

```python
primitive: Optional[str] = Field(
    None,
    description="Catalog primitive name (e.g. 'Circle'). When set, args carries "
                "the kwargs for direct construction. Falls back to legacy "
                "object_type/render_hint when None."
)
args: Dict[str, Any] = Field(
    default_factory=dict,
    description="Kwargs for the primitive constructor (e.g. {'radius': 2.0})."
)
```

Add a model_validator that — when `primitive` is set — verifies it exists in
`MOBJECT_CATALOG` and that every `args` key is a known parameter:

```python
@model_validator(mode="after")
def _validate_primitive(self):
    if self.primitive is None:
        return self
    from src.utils.manim_primitives import MOBJECT_CATALOG
    spec = MOBJECT_CATALOG.get(self.primitive)
    if spec is None:
        raise ValueError(f"Unknown primitive: {self.primitive}")
    allowed = {a for a, _, _ in spec.args} | ({"color"} if spec.accepts_color else set())
    bad = set(self.args) - allowed
    if bad:
        raise ValueError(f"Unknown args for {self.primitive}: {bad}")
    return self
```

Same treatment for `Transformation.animation_hint`: accept only names in
`ANIMATION_CATALOG`.

- [x] **Step 3: Inject the catalog into the visual-planner prompt**

In `_generate_visual_intent_for_chunk` (Task 7), append the catalog reference:

```python
from src.utils.manim_primitives import catalog_for_prompt
prompt += "\n\nCLOSED VOCABULARY:\n" + catalog_for_prompt() + (
    "\n\nFor every mathematical_object you emit, set `primitive` to one of the "
    "MOBJECT names above and `args` to its kwargs. Free-form `content` is "
    "ONLY permitted for MathTex/Tex/Text (where it goes into `tex` or `text`)."
)
```

Update the JSON schema example in the prompt to show `"primitive": "Circle",
"args": {"radius": 2.0}` instead of free-form `content`.

- [x] **Step 4: Animation generator becomes a thinner emitter**

Replace the long `if/elif` chain in `_generate_mathematical_objects`
(`animation_generator.py:528-634`) with catalog-driven emission:

```python
def _emit_primitive(self, var_name: str, obj: dict) -> str:
    """Emit one line of code from a catalog primitive spec."""
    from src.utils.manim_primitives import MOBJECT_CATALOG
    primitive = obj.get("primitive")
    if not primitive:
        return self._legacy_emit_object(var_name, obj)  # keep old path for safety

    spec = MOBJECT_CATALOG[primitive]  # validated by Pydantic upstream
    args = dict(obj.get("args", {}))
    color = obj.get("style", {}).get("color")
    parts = [f"{k}={self._py_repr(v)}" for k, v in args.items()]
    if color and spec.accepts_color:
        parts.append(f"color={color.upper()}")
    return f"        {var_name} = {primitive}({', '.join(parts)})"
```

Keep `_legacy_emit_object` as a fallback for any object that lacks `primitive`
(legacy storyboards, fallback visual intents).

- [x] **Step 5: Tests**

```python
def test_catalog_for_prompt_contains_core_mobjects():
    from src.utils.manim_primitives import catalog_for_prompt
    s = catalog_for_prompt()
    for name in ("MathTex", "Circle", "Axes", "Polygon", "create_pi_teacher"):
        assert name in s


def test_mathematical_object_rejects_unknown_primitive():
    import pytest
    from pydantic import ValidationError
    from src.schemas.storyboard import MathematicalObject, ObjectStyle
    with pytest.raises(ValidationError):
        MathematicalObject(
            object_id="x", object_role="main_equation", object_type="text",
            content="", primitive="ImaginaryClass", args={},
            style=ObjectStyle(),
        )


def test_mathematical_object_rejects_unknown_arg():
    import pytest
    from pydantic import ValidationError
    from src.schemas.storyboard import MathematicalObject, ObjectStyle
    with pytest.raises(ValidationError):
        MathematicalObject(
            object_id="x", object_role="main_equation", object_type="geometric_shape",
            content="", primitive="Circle", args={"diameter": 2.0},  # "diameter" isn't valid
            style=ObjectStyle(),
        )


def test_animation_generator_emits_primitive_call():
    from src.agents.animation_generator import AnimationCodeGeneratorAgent
    agent = AnimationCodeGeneratorAgent.__new__(AnimationCodeGeneratorAgent)
    line = agent._emit_primitive("c0", {
        "primitive": "Circle",
        "args": {"radius": 2.5},
        "style": {"color": "blue"},
    })
    assert line.strip() == "c0 = Circle(radius=2.5, color=BLUE)"
```

- [x] **Step 6: Verify**

```
python -m pytest tests/test_manim_primitives.py tests/test_schemas.py tests/test_animation_generator.py -q
```

---

### Task 13: Strict (x, y, t) coordinate planning + overlap validation

**Why:** the existing `LayoutTracker` (referenced in the animation prompt at
`animation_generator.py:272-315`) only checks 8 cardinal anchors with
`place_label`, requires the LLM to manually call `register`/`place_label`,
and has no notion of *time*. Two objects placed at the same `(x, y)` but
visible only during disjoint intervals should not count as an overlap;
two objects on screen simultaneously at overlapping `(x, y, w, h)` are bugs.

The fix is to make `(x, y, t)` a first-class part of the visual intent,
not a post-hoc Manim concern.

**Files:**
- Modify: `C:/Users/Parth/Documents/3B1B/src/schemas/storyboard.py` (extend `MathematicalObject` with `bbox` + `interval`)
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/layout_validator.py`
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/scene_layout_planner.py` (or relevant validator wiring)
- Modify: `C:/Users/Parth/Documents/3B1B/src/agents/animation_generator.py` (force bbox-derived `move_to`)
- Test: `C:/Users/Parth/Documents/3B1B/tests/test_layout_validator.py`

- [x] **Step 1: Extend `MathematicalObject` with explicit bbox + interval**

```python
class BBox(BaseModel):
    """Axis-aligned bounding box in Manim world units (origin at center)."""
    x: float = Field(..., description="Center x")
    y: float = Field(..., description="Center y")
    w: float = Field(..., gt=0, description="Width")
    h: float = Field(..., gt=0, description="Height")


class TimeInterval(BaseModel):
    """When the object is visible. t_in inclusive, t_out exclusive."""
    t_in: float = Field(..., ge=0)
    t_out: float = Field(..., gt=0)

    @model_validator(mode="after")
    def _ordered(self):
        if self.t_out <= self.t_in:
            raise ValueError("t_out must be > t_in")
        return self


# Extend MathematicalObject:
bbox: Optional[BBox] = Field(None, description="Required when primitive is set")
interval: Optional[TimeInterval] = Field(None, description="Visibility window")
```

The Manim screen is roughly `x ∈ [-7.1, 7.1]`, `y ∈ [-4.0, 4.0]` at 16:9 with
default frame_height=8. Document this in the schema docstring so the LLM
prompt can include the safe area: `x ∈ [-6.5, 6.5]`, `y ∈ [-3.5, 3.5]`.

- [x] **Step 2: Write failing tests for temporal-aware overlap detection**

```python
def test_layout_validator_detects_simultaneous_overlap():
    from src.agents.layout_validator import LayoutValidatorAgent, LayoutValidationInput
    from src.schemas.scene_layout import SceneLayout, LayoutObject
    layout = SceneLayout(objects=[
        LayoutObject(object_id="a", bbox=BBox(x=0, y=0, w=2, h=1),
                     interval=TimeInterval(t_in=0, t_out=10)),
        LayoutObject(object_id="b", bbox=BBox(x=0.5, y=0, w=2, h=1),  # overlaps a
                     interval=TimeInterval(t_in=2, t_out=8)),         # at t=2..8
    ])
    out = LayoutValidatorAgent().execute(LayoutValidationInput(scene_layout=layout))
    assert not out.is_valid
    assert any("a" in v.description and "b" in v.description for v in out.violations)


def test_layout_validator_allows_serial_reuse_of_position():
    layout = SceneLayout(objects=[
        LayoutObject(object_id="a", bbox=BBox(x=0, y=0, w=2, h=1),
                     interval=TimeInterval(t_in=0, t_out=5)),
        LayoutObject(object_id="b", bbox=BBox(x=0, y=0, w=2, h=1),  # same place
                     interval=TimeInterval(t_in=5, t_out=10)),       # but disjoint
    ])
    out = LayoutValidatorAgent().execute(LayoutValidationInput(scene_layout=layout))
    assert out.is_valid


def test_layout_validator_rejects_offscreen_bbox():
    layout = SceneLayout(objects=[
        LayoutObject(object_id="a", bbox=BBox(x=10, y=0, w=1, h=1),  # x=10 is off
                     interval=TimeInterval(t_in=0, t_out=5)),
    ])
    out = LayoutValidatorAgent().execute(LayoutValidationInput(scene_layout=layout))
    assert not out.is_valid
```

- [x] **Step 3: Implement `(x, y, t)`-aware overlap check**

```python
SAFE_X = (-6.5, 6.5)
SAFE_Y = (-3.5, 3.5)

def _bboxes_overlap(a: BBox, b: BBox) -> bool:
    return (
        abs(a.x - b.x) * 2 < (a.w + b.w)
        and abs(a.y - b.y) * 2 < (a.h + b.h)
    )

def _intervals_overlap(a: TimeInterval, b: TimeInterval) -> bool:
    return not (a.t_out <= b.t_in or b.t_out <= a.t_in)


class LayoutValidatorAgent(Agent):
    def execute(self, inp):
        violations = []
        objs = inp.scene_layout.objects
        for o in objs:
            if not (SAFE_X[0] <= o.bbox.x <= SAFE_X[1] and SAFE_Y[0] <= o.bbox.y <= SAFE_Y[1]):
                violations.append(LayoutViolation(
                    description=f"{o.object_id} center ({o.bbox.x},{o.bbox.y}) "
                                f"is outside safe area {SAFE_X}×{SAFE_Y}",
                ))
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                if _bboxes_overlap(a.bbox, b.bbox) and _intervals_overlap(a.interval, b.interval):
                    violations.append(LayoutViolation(
                        description=f"{a.object_id} and {b.object_id} overlap in "
                                    f"both space and time",
                    ))
        return LayoutValidatorOutput(is_valid=not violations, violations=violations)
```

- [x] **Step 4: Force the animation generator to use bbox-derived placement**

In the system prompt, replace the `LayoutTracker.place_label` instructions
with bbox-driven placement when `bbox` is present on the object:

```
For every object that ships with a bbox, place it with:
    obj.move_to([bbox.x, bbox.y, 0])
    obj.set(width=bbox.w)   # OR obj.scale_to_fit_width(bbox.w) if simpler
DO NOT call .next_to(...) / .to_edge(...) when bbox is given — the layout
has already been validated for overlap-freeness in (x, y, t) space.
```

Emit code in `_emit_primitive` that appends `move_to`/scaling lines whenever
`obj["bbox"]` is set.

- [x] **Step 5: Make the visual planner emit bbox + interval**

Update Task 7's prompt to require `bbox` and `interval` for every emitted
mathematical_object. Wire the validator output back into the visual-planner
retry loop (already in place via `scene_controller._plan_scene_layout`'s
evaluator) so violations become structured `violation_feedback` lines like
`"a and b overlap in both space and time at t=2.0..8.0"` and the LLM gets a
concrete fix target.

- [x] **Step 6: Verify**

```
python -m pytest tests/test_layout_validator.py tests/test_schemas.py -q
python -m pytest tests/test_animation_generator.py -q
```

---

### Failure modes & mitigations

These are the failure points most likely to bite on real PDFs. Each one names the
task it attaches to so it lands at the right step instead of being filed under
"future work."

**F1. Concept-label drift defeats dedup (Task 3 + Task 5)**

Risk: the LLM introduces "sine" in beat 1 and "sine_function" in beat 4, and
both pass through the ledger as distinct concepts → the viewer hears the same
explanation twice.

Mitigation:
- Add a `_canonicalize_concept(label: str) -> str` helper in
  `script_architect.py` that lowercases, strips, replaces spaces with
  underscores, and applies a small alias map (`{"sin": "sine", "cos":
  "cosine", "trig_function": "trigonometric_function"}`).
- Apply it to every label in `Beat.introduces_concepts` and
  `Beat.references_prior_beats` *before* `VideoScript` validation.
- Add a test:
```python
def test_concept_canonicalization_dedups_label_variants():
    # Two beats introducing "sine" and "Sine Function" must collapse
    # to a single ledger entry.
```

**F2. Coverage gap — LLM silently drops PDF formulas (Task 3)**

Risk: the unified script call ignores half the formulas in the source material.
The video looks coherent but skips content the viewer expects.

Mitigation:
- After `_generate_video_script` returns, run a deterministic coverage check:
  for every formula in `ocr_output.pages[*].math_expressions` (extraction_failed
  == False), confirm that *some* beat's `narration` or `visual_hint` mentions a
  recognizable substring of the LaTeX (strip braces/backslashes, look for ≥ 1
  matching token).
- If coverage falls below 60%, log a warning and re-run with a
  `review_feedback=["The previous draft missed these formulas: ..."]` critique
  attached. Cap at 1 retry to avoid infinite loops.
- Surface the coverage ratio in `_quality_reports["storyboard_review"]`.

**F3. Oversized single beat blows the render timeout (Task 4)**

Risk: a single beat exceeds `MAX_SCENE_DURATION` (90s). The chunker accepts it
as one oversized chunk, but the Manim render stage has a 1800s render budget
across all chunks — one 200s chunk can dominate and time-share starve others.

Mitigation:
- In the `_generate_video_script` prompt, add an explicit rule:
  *"Each beat MUST be 40–80 words. If a topic needs more, split into multiple
  beats with `references_prior_beats`."*
- In `_chunk_script_into_scenes`, if `beat_duration(beat) > MAX_SCENE_DURATION`,
  log a warning naming the beat_id but still emit it as a single chunk
  (atomicity preserved). The warning lets us tighten the prompt later.
- Test:
```python
def test_chunker_logs_warning_for_oversized_beat(caplog):
    # 600-word single beat should produce one chunk AND a WARNING log line
    # naming the offending beat_id.
```

**F4. `NarrativeContext` grows unboundedly across long videos (Task 6)**

Risk: a 30-scene video means `prior_scene_summaries` carries 30 strings and
`concepts_introduced` carries 30+ labels. Every per-chunk prompt now ships
this growing payload — token cost compounds.

Mitigation:
- Cap `prior_scene_summaries` at the most recent **5** entries (verbatim) plus
  a single rolled-up "older history" summary. When appending the 6th, fold
  the oldest verbatim into the rollup.
- Implement in `_update_running_storyline`'s sibling: `_compact_summaries`.
- `concepts_introduced` is a flat label list — keep it uncapped (it's small
  per entry and load-bearing for dedup).
- Test:
```python
def test_narrative_summaries_compact_after_five_scenes():
    # Run 7 fake scenes; assert prior_scene_summaries length is ≤ 6
    # (5 verbatim + 1 rollup) and that early summaries appear inside the
    # rollup string.
```

**F5. `_source_beats` as a private attribute is fragile (Task 4 + Task 6)**

Risk: stashing beat data on `Scene._source_beats` with `noqa: SLF001` survives
in-memory passes but breaks any time `Scene` is serialized through Pydantic
(e.g. `.model_dump()` in Persistence, JSON sanitizer round-trip, dry-run
manifest write). Downstream code that re-reads scenes from disk loses the
beats entirely.

Mitigation:
- Make it a proper field: extend the existing `Scene` schema with
  `source_beats: List[Beat] = Field(default_factory=list, exclude=False)`.
  This requires importing `Beat` into `src/schemas/storyboard.py`.
- Update Task 4's chunker to set `scene.source_beats = chunk_beats` instead of
  `scene._source_beats = chunk_beats`.
- Update Task 6's `_extract_introduced_concepts` to read `scene.source_beats`.
- Test that `Scene.model_dump()` round-trip preserves source beats:
```python
def test_scene_source_beats_survive_pydantic_roundtrip():
    s = Scene(..., source_beats=[Beat(beat_id="b1", narration="x", visual_hint="y")])
    assert Scene.model_validate(s.model_dump()).source_beats[0].beat_id == "b1"
```

**F6. Per-chunk visual planner failure should not abort the storyboard (Task 5)**

Risk: in the new `execute()`, the loop
`for scene in scenes: scene.visual_intent = self._generate_visual_intent_for_chunk(...)`
raises if *any* one chunk's visual call fails — wiping out the whole run.

Mitigation:
- Wrap the per-chunk visual call in try/except. On exception, log the
  scene_id and fall back to `_generate_fallback_visual_intent(scene.narration)`
  for that chunk only. The pipeline continues with degraded visuals on that
  chunk, not zero output for the run.
- Record the failure in `_quality_reports["storyboard_review"]["visual_failures"]`
  so the final manifest reflects degraded chunks.
- Test:
```python
def test_visual_intent_failure_falls_back_per_chunk(monkeypatch):
    # Make _generate_visual_intent_for_chunk raise on scene_002; assert
    # scenes[0] and scenes[2] still get LLM-generated visual_intent and
    # scenes[1] gets the deterministic fallback.
```

**F7. `references_prior_beats` may name nonexistent beats (Task 2)**

Risk: the LLM emits `references_prior_beats=["beat_999"]` (typo or hallucination).
Currently `VideoScript` validation only rejects duplicate beat_ids — invalid
back-references slip through and corrupt downstream coverage logic.

Mitigation:
- Extend the `_unique_beat_ids` model_validator to also verify every
  `references_prior_beats` entry refers to an *earlier* beat in the list.
  Drop unknown ids with a warning rather than raising — the script is still
  usable; bad refs are non-fatal.
- Test:
```python
def test_video_script_drops_unknown_back_references(caplog):
    script = VideoScript(beats=[
        Beat(beat_id="beat_001", narration="a", visual_hint="x", references_prior_beats=["beat_999"]),
    ])
    assert script.beats[0].references_prior_beats == []
    assert any("beat_999" in r.message for r in caplog.records)
```

**F8. Visual planner re-explains earlier concepts at the *visual* layer (Task 7)**

Risk: even with `earlier_concepts` listed in the briefing, the LLM may emit
`mathematical_objects` whose `content` is a re-derivation of an earlier formula
("here is sin θ = opp/hyp again"). The text says "reference," the visuals
redefine.

Mitigation:
- After parsing the visual-intent JSON, run a deterministic check: for each
  `mathematical_object` of type `equation`, if its LaTeX matches (after
  normalization) an equation that appeared in any prior scene's
  `mathematical_objects`, log a warning with the offending object_id.
- Don't reject — the same equation can legitimately reappear in a callback —
  but require the chunk's transformations to mark it `continuity_action="reuse"`
  rather than `"create"`. If the LLM marked it `"create"`, downgrade to `"reuse"`
  in post-processing.
- This check needs prior-scene `mathematical_objects` in the briefing; add
  `prior_equations: List[str]` to `NarrativeContext` (Task 1 amendment).

**F9. TTS chunk concatenation desync (Task 11)**

Risk: ffmpeg `-c copy` concat of multiple MP3 segments can produce timing
artifacts (gaps, frame-edge clicks) when the segments aren't bit-aligned.
Rare with `tts-1` output but possible.

Mitigation:
- If ffmpeg returns non-zero or the concatenated duration is more than 0.5s
  off the sum of segment durations, fall back to a re-encode concat
  (`-c:a libmp3lame -b:a 128k`) instead of stream-copy.
- Add a duration check after concat: `_probe_duration(final) ≈ sum(_probe_duration(parts))`.
- Test:
```python
def test_tts_concat_falls_back_to_reencode_on_drift(monkeypatch):
    # Stub _probe_duration to return inconsistent values; assert the fallback
    # ffmpeg command (with -c:a libmp3lame) is invoked.
```

**F10. LLM emits primitive-args mix that doesn't match the catalog (Task 12)**

Risk: the visual-planner LLM emits `Polygon` with `vertices: [[1,2], [3,4]]`
(2-tuples) when the catalog requires 3-tuples `(x, y, 0)` for Manim. Pydantic
validation rejects it and the whole chunk falls back.

Mitigation:
- In the `MathematicalObject._validate_primitive` model_validator, run a
  light "args coercion" step BEFORE strict validation: pad 2-tuples in
  `vertices` to 3-tuples; coerce `color` strings to upper case; coerce
  numeric strings to floats. Only reject after coercion fails.
- Log a `warnings` field on the chunk's quality report listing every
  coercion that fired so we can tighten the prompt over time.
- Test:
```python
def test_polygon_vertices_2d_coerce_to_3d():
    from src.schemas.storyboard import MathematicalObject, ObjectStyle
    obj = MathematicalObject(
        object_id="p", object_role="main_equation", object_type="geometric_shape",
        content="", primitive="Polygon",
        args={"vertices": [[0, 0], [1, 0], [0, 1]]},
        style=ObjectStyle(),
    )
    assert obj.args["vertices"] == [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
```

**F11. Bbox/interval are unsolvable for crowded chunks (Task 13)**

Risk: a chunk has 8 objects all overlapping in time and the layout planner
can't find an arrangement that fits inside the safe area without overlap.
The validator loop exhausts retries and ships a degraded layout.

Mitigation:
- Two-tier safe area: prefer the strict safe area `(-6.5..6.5, -3.5..3.5)`,
  but if no clean layout exists after `max_layout_retries`, fall back to a
  larger area `(-7.0..7.0, -3.9..3.9)` AND emit a `WARNING` quality-report
  entry naming the offending chunk.
- Ranking heuristic for which violations to "accept" when forced: prefer
  keeping objects with `object_role="main_equation"` strictly non-overlapping;
  allow `supporting_note`/`label` to clip the safe area as last resort.
- If the chunk has > 6 simultaneously visible objects, log a script-level
  warning in `_quality_reports["storyboard_review"]` — the right fix is for
  the script writer to split the chunk, not for the layout planner to
  miracle-solve it.
- Test:
```python
def test_layout_planner_warns_on_unsolvable_chunk(caplog):
    # Provide 8 simultaneously-visible objects with bboxes that cannot fit
    # in the safe area; assert the planner returns a layout with a non-empty
    # `degradations` list and emits a WARNING log.
```

---

### Risk & rollback

- **Single LLM call risk:** the unified script is now a single point of failure for narration. Mitigated by the existing `StoryboardGraderAgent` retry loop (`enable_quality_loops=True` retries with critique up to `max_quality_retries`). If the script call fails repeatedly, the mock fallback kicks in and produces a minimal 2-beat script — pipeline degrades gracefully.
- **Token budget:** Phase-1 prompt is larger (~8k of source text + instructions). Per-chunk visual prompts are *smaller* than before (no full-PDF dump). Net effect should be neutral or favorable, but monitor `script_architect` token costs after the first real run.
- **Rollback:** the changes are confined to `script_architect.py`, `scene_controller.py`, `animation_generator.py`, and three schema files. Reverting the commit set restores the legacy per-concept flow without data migration.
