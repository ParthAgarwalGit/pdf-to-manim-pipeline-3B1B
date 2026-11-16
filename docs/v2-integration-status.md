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

## Not Yet Connected

`src/pipeline_v2/presenter.py` has independently tested bridge proposals,
approval, title-card rendering, a consent-gated D-ID provider, presenter duration
limits and timeline interleaving. These are not yet orchestrated by `PipelineV2`.

Before exposing paid presenter rendering:

1. Add explicit opt-in CLI/config settings, off by default.
2. Validate consent/profile/provider configuration before core paid calls.
3. Persist proposals and require approval before any presenter call.
4. Route presenter costs through atomic V2 reservations and event logging.
5. Persist provider job identifiers before polling so ambiguous failures never
   create a duplicate paid job on resume. Do not silently retry creation.
6. Insert approved clips into `TimelineSegment` without replacing Manim scenes.
7. Include presenter clips in final delivery checks and report provenance.
8. Test interruption, budget exhaustion and selective scene revision with bridges.

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
