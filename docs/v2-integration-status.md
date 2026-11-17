# V2 Integration Status

Working branch: `vignesh/v2.1-integration`.
Protected checkpoint: `vignesh/v2.0` at `e54a7ca`.
Main merged at `c4b3a38` (parent `origin/main` at `e4c779a`).

## Connected To V2

- V2 remains the default CLI; legacy operations use `src.legacy_cli`.
- Generated Python passes the hardened AST gate before execution.
- Native execution scrubs credentials and uses scratch directories.
- Optional Docker execution adds network isolation and resource limits.
- Beat synchronization waits for narration, caps retiming, and reports lateness.
- Gemini TTS records each beat as its own request. Only the words to be spoken are
  sent as text; delivery goes in the Interactions API's `speech_metadata` style
  (the model reads any text it is given aloud). Each take is checked against its
  word count and recorded again when it holds extra or missing speech, and each
  beat's measured speech span sets its window, so visuals land on their words.
- The hook is checked (word limit, one closing question, no dashes, at least three
  sentences) and rewritten whole when it fails. The script review scores
  `logical_flow` and lists every sentence that does not follow from the one
  before; a break in the narration fails the review, a break in the hook triggers
  a hook rewrite. In `revise` mode only the beats holding flagged sentences are
  rewritten (the whole narration only when the review fails without one); the
  fixed script is reviewed once more and kept if it passes or scores higher.
- `convert` accepts professor briefs/directives, concept plans, and duration targets.
- Scope, notation and narration compliance are persisted and checked.
- `plan` creates an editable, source-derived outline locally without API calls.
- `revise` estimates edits before approval, then rebuilds affected scenes only.
- Visual-only edits retain audio; pronunciation/pacing edits can reuse visuals.
- Revision artifacts preserve before/plan/result records and resume state.
- Composition preserves narration instead of cutting to the shorter stream.
- Final delivery checks measure streams, loudness, silence and duration drift.
- Standard policy retains reviewed output; strict withholds official delivery on failure.
- Evaluation CLI and frozen benchmark remain available.
- Presenter bridges (`src/pipeline_v2/presenter.py`) are orchestrated by `PipelineV2`.
  Gemini Omni is the default presenter provider (`convert --presenter gemini`): a
  fictional lecturer described in words who speaks in the narration's prebuilt
  Gemini voice. The consent-gated D-ID avatar remains available (`--presenter d-id`
  with `--profile` and `--avatar-image-url`).
  - Opt-in, off by default; `--approve-bridges`, `--max-presenter-seconds`,
    `--presenter-persona`.
  - Provider configuration (key, consent, image) is checked before the first paid call.
  - Proposals are persisted to `presenter/bridges.json` (lesson-spine text for
    planned lessons) and only approved bridges are rendered; `pdf2manim bridges
    RUN_DIR --approve ...` then `pdf2manim resume RUN_DIR`.
  - Presenter costs go through atomic V2 reservations (settled on measured clip
    length) and are logged as events; a bridge over the presenter-seconds cap or
    the dollar budget is skipped and the run continues.
  - The hook is never lost: without an approved opening it opens the narration; an
    approved opening that cannot be rendered becomes a title card over TTS.
  - Approved clips are interleaved as `TimelineSegment(mode="presenter")` without
    replacing Manim scenes, with a fade through black at each presenter/Manim cut.
  - Presenter clips are in the delivery checks and `report.html` lists every bridge.
  - Clip tasks are keyed by the bridge fingerprint: resume and revision reuse
    unchanged clips and re-render only a bridge whose text changed.

## Not Yet Connected

- D-ID talk ids are not persisted before polling, so an ambiguous D-ID failure
  could create a second paid talk on resume (Gemini Omni is synchronous and
  caches clips by text and voice, so it has no such window).

`src/web/professor_app.py` still uses the legacy CLI. The V2 studio must call
V2 services directly, read `run.json`/`storyboard.json`/delivery reports, and
support plan/read-back, generation, review and approved revisions. Avoid claiming
the existing studio drives V2 until those handlers and tests have been replaced.

## Validation Policy

All integration tests are offline (provider mocks or keyless fallbacks).
Windows sandbox restrictions previously hung asyncio socket-pair creation;
run offline tests with appropriate local socket permissions and a project-local
`--basetemp`. Live API and property suites are separate opt-in lanes.
No paid presenter, LLM or TTS evaluation is authorized by this integration task.
