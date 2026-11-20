# WPC — Gemini Omni presenter bridges, orchestrated in V2, in the narration's voice

You are implementing one work package in this repository. Decisions are made; implement them.

## Ground rules (binding)

- Work only in this checkout on its current branch. **No git commands.**
- **Frozen, read-only:** `src/pipeline_v2/prompt_text.py`, `src/pipeline_v2/model_config.py`,
  everything under `src/prompts/`.
- No paid API calls; all tests keyless with fakes (no Omni, no TTS, no network).
- Only shell command: `python -m pytest ... -p no:cacheprovider --basetemp=.pytest_tmp/wpc...`.
- Never weaken, skip or delete a test; update an expectation only for behaviour this spec changes,
  and list it. Known pre-existing failure to leave alone:
  `tests/test_v2_delivery.py::TestRealMedia::test_mixed_clips_join_into_one_uniform_file`.
- New config/state fields default so old runs still load and resume.

## Background

V2 already has `src/pipeline_v2/presenter.py`: bridge proposals (`propose_bridges`), approval
(`approve_bridges`, `save_bridges`, `load_bridges`), a provider protocol (`PresenterProvider`:
`name`, `synthetic`, `speaks`, `estimated_cost_usd`, `render`), title-card and D-ID providers, a
`PresenterBudget`, `render_bridge`, `bridge_fingerprint` and `interleave_segments`. None of it is
called by `PipelineV2` yet; `docs/v2-integration-status.md` lists what orchestration must
guarantee. The legacy pipeline has a working Gemini Omni presenter
(`src/agents/gemini_presenter.py`, class `GeminiOmniPresenter`: fictional lecturer, clips of at
most 15 s, trimmed to speech, cached by text, Whisper-free word-error check optional).

The user approved, after listening to a test, that the presenter speaks with the same prebuilt
Gemini voice as the narration (`speech_config` voice = `PipelineV2Config.voice`, default
`Sulafat`). A live test confirmed `client.interactions.create(..., generation_config={"speech_config":
[{"voice": "Sulafat", "language": "en-US"}]}, response_format={...})` is accepted by
`gemini-omni-1.2-flash`.

## C1. Omni speaks in the configured voice — `src/agents/gemini_presenter.py`

- `GeminiOmniPresenter.__init__` gains `voice: Optional[str] = None` (stored).
- `_request` passes `generation_config={"speech_config": [{"voice": self.voice, "language":
  "en-US"}]}` to `interactions.create` when `self.voice` is set (omit the key otherwise).
- The clip cache key includes the voice: `f"{OMNI_MODEL}|{self.persona}|{self.voice or ''}|{text}"`.
- Change the constant `BACKGROUND` to
  `"a plain, near-black studio background (hex #0B0B0B), evenly lit, with no visible props"`
  so cuts to Manim (black background) are not jarring.
- Legacy `src/orchestrator/pipeline.py` constructs `GeminiOmniPresenter`; pass
  `voice=None` there (legacy behaviour unchanged).

## C2. Provider — `src/pipeline_v2/presenter.py`

- `PresenterSettings.provider` accepts `"gemini"`; add `voice: str = DEFAULT_VOICE` and
  `persona: str = ""` fields.
- New class `GeminiPresenter` implementing the protocol: `name = "gemini"`, `synthetic = True`,
  `speaks = True`. It wraps `GeminiOmniPresenter(client=..., persona=settings.persona or default,
  voice=settings.voice, cache_dir=<run_dir>/presenter/cache)` (read its constructor for the exact
  arguments). `render(request, output_path)` calls `speak(request.text, output_path)` and returns
  a `PresenterClip(path, duration, synthetic=True, provider="gemini", detail={"voice": ...,
  "clips": n})`. Brand the clip with the same disclosure label the D-ID path draws
  (`DISCLOSURE_LABEL`, reuse `_brand` or `_draw_label`), and re-encode to the V2 encode args so
  the compositor gets uniform media (1280x720, 30 fps, the audio format the compositor uses).
- `estimated_cost_usd(request)`: `clip_seconds * 5792 * 12.00 / 1_000_000` where
  `clip_seconds` is the sum over `split_for_clips(text)` of each chunk's rounded-up clip length
  (the legacy module computes chunk seconds; use the same rule), plus one retake allowance
  (multiply by 1.5). Name the per-second constant `OMNI_VIDEO_TOKENS_PER_SECOND = 5792`.
- `build_presenter`: provider `"gemini"` requires `GEMINI_API_KEY` (from `environ` or
  `os.environ`) and raises `PresenterConfigError("GEMINI_API_KEY is not set; the gemini presenter
  cannot be used.")` otherwise. No consent profile is required: the persona is fictional and
  described in words, never a real person's likeness (keep the D-ID consent path unchanged).

## C3. Bridge text from the lesson spine

`propose_bridges(lesson, narration, *, words_per_minute=None, planned=False)`:
- `planned=True` when `lesson.central_question` is non-empty. Then: no `"recap"` slot (the
  script already ends on its payoff), and the opening text is `lesson.hook` verbatim when
  non-empty.
- New async function `write_bridges(router, proposals, lesson, narration) -> list[BridgeProposal]`
  for planned lessons: for every proposal except an opening whose text is `lesson.hook`, build one
  slot line with `src.prompts.narrative.bridge_slot_line(slot, kind, before, after,
  next_question)` where `slot` is the proposal's position, `before` is the last beat text of the
  preceding scene ("" for the first), `after` is the first beat text of `before_scene_id`'s scene,
  and `next_question` is that scene's `viewer_question`. Call the model with
  `src.prompts.narrative.bridge_prompt(lines, lesson)` (it reads `central_question`, `hook`,
  `takeaway` attributes), via the router's JSON path, `stage="bridges"`, `model=TEXT_MODEL`,
  `max_output_tokens=1500`. Accept a returned text only if it has 6-45 words and
  `src.utils.narration_flow.overlaps(text, neighbour) < 0.3` for both neighbours; pass accepted
  text through `spoken_plain`; otherwise keep the template text. Keyless or failure: keep
  templates, emit `bridges_written_by_template`. Recompute `estimated_seconds`.

## C4. Orchestration — `src/pipeline_v2/pipeline.py` and `src/cli.py`

Config (`PipelineV2Config`; `presenter` already exists from the previous package):
`approve_bridges: bool = False`, `max_presenter_seconds: float = Field(default=60.0, ge=0)`,
`presenter_persona: str = ""`, `presenter_profile: Optional[Path] = None`,
`avatar_image_url: Optional[str] = None`.

CLI `convert`: `--presenter [off|gemini|d-id]` (default `off`), `--approve-bridges`,
`--max-presenter-seconds` (default 60), `--presenter-persona`, `--profile` (d-id),
`--avatar-image-url` (d-id). New command `pdf2manim bridges RUN_DIR [--approve all|BRIDGE_ID ...]
[--reject BRIDGE_ID ...]` that lists proposals (id, kind, text, estimated seconds, approved) and
updates approvals in `presenter/bridges.json`; after approving, the user runs
`pdf2manim resume RUN_DIR`.

Pipeline behaviour, in order:
1. **Fail fast:** when `presenter != "off"`, call `build_presenter(...)` before the first paid
   call of the run (before OCR/lesson). A `PresenterConfigError` aborts the run with its message.
2. **Propose after narration** (and after the script review, if any): `propose_bridges(...,
   planned=bool(lesson.central_question))`, then `write_bridges(...)` for planned lessons, then
   `merge_proposals` with any existing `presenter/bridges.json` (resume keeps approvals and text),
   then `approve_bridges(all)` when `config.approve_bridges`, then `save_bridges`. Checkpoint this
   as a `_task` so resume does not pay for bridge writing twice.
3. **The hook is never lost.** The narration was written assuming the presenter says the hook
   (`presenter_opens=True`). If the opening bridge is not approved at TTS time, prepend
   `lesson.hook` as a new first sentence of the first scene's first beat before TTS (and record
   event `hook_moved_to_narration`). If the opening bridge is approved but its presenter render
   later fails or is refused by the budget, render it with `TitleCardPresenter` over Gemini TTS
   audio of the hook (same voice), so it is still spoken.
4. **Render approved bridges** after the Manim scenes are rendered, concurrently limited to one
   at a time: for each approved bridge, check `PresenterBudget(max_presenter_seconds)`; reserve
   `estimated_cost_usd` with the run store (`reserve_call`, provider `"google"`, model
   `PRESENTER_MODEL`, stage `"presenter"`) *before* the call; settle with the measured clip
   seconds × the per-second rate after. A bridge that does not fit the presenter-seconds cap or
   the dollar budget is skipped with event `presenter_skipped` (the run continues; the hook rule
   in 3 still applies to the opening). Omni is synchronous and clips are cached by text+voice, so
   a resumed run reuses rendered clips and never re-requests one. Use `bridge_fingerprint` in
   the task payload so a changed text re-renders and an unchanged one is reused.
5. **Compose** with `interleave_segments` so each approved clip sits before its
   `before_scene_id` (or at the end for `None`), as `TimelineSegment(mode="presenter")`.
   Presenter audio is the clip's own audio (same voice as narration). Loudness-normalise
   presenter clips the same way narration clips are normalised before concatenation, and add a
   0.25 s fade through black at every cut between a presenter segment and a Manim segment.
6. **Delivery checks and report** include presenter segments: duration, stream format,
   loudness; `report.html` lists each bridge (kind, text, approved, rendered/skipped, seconds,
   cost) and the voice used.
7. **Revision:** `revise` of a run with bridges keeps approved clips whose fingerprint is
   unchanged; a changed bridge text re-renders only that bridge.

Update `docs/v2-integration-status.md`: move the presenter items from "Not Yet Connected" to
"Connected", noting Gemini is the default presenter provider and D-ID remains available.

## Tests — new `tests/test_v2_gemini_presenter.py`

Use fakes: a fake `GeminiOmniPresenter.speak` that writes a short real MP4 with a tone (ffmpeg is
installed; build it with `ffmpeg -f lavfi`), records calls, and counts requests.

1. `test_omni_request_carries_voice` — `GeminiOmniPresenter(voice="Sulafat")._request` with a
   fake client: `generation_config.speech_config[0].voice == "Sulafat"`; cache key changes with
   the voice; without voice no `generation_config` key is sent.
2. `test_build_presenter_gemini_requires_key` — no key → `PresenterConfigError`; with key → a
   `GeminiPresenter` with `speaks is True`; no profile needed.
3. `test_cost_estimate_matches_clip_rule` — a 30-word text gives the expected dollars.
4. `test_planned_proposals_use_hook_and_drop_recap` — opening text equals `lesson.hook`, no recap.
5. `test_write_bridges_validates_and_falls_back` — fake router returns one good text, one that
   repeats the next beat (overlap ≥ 0.3) and one of 60 words: only the good one is used.
6. `test_unapproved_opening_moves_hook_into_narration` — pipeline-level with fakes and
   `presenter="gemini"`, `approve_bridges=False`: first beat text starts with the hook; no
   presenter render calls; event `hook_moved_to_narration`.
7. `test_approved_bridges_render_and_interleave` — `approve_bridges=True`: clips rendered once
   each, timeline has presenter segments before the right scenes, costs reserved and settled with
   provider `google` and model `PRESENTER_MODEL`.
8. `test_resume_does_not_rerender_cached_bridge` — run, then resume: fake speak call count
   unchanged.
9. `test_budget_skip_keeps_hook_spoken` — `max_presenter_seconds=1`: opening bridge skipped by
   the cap and rendered as a title card with TTS audio (fake TTS); event `presenter_skipped`.
10. `test_cli_presenter_options_and_bridges_command` — CliRunner: options reach the config;
    `bridges RUN --approve all` flips approvals in `presenter/bridges.json`.
11. `test_presenter_off_unchanged` — default config: no presenter calls, timeline identical to
    today's (no presenter segments).

## Gates

- G1 — CHECK: `python -m pytest tests/test_v2_gemini_presenter.py tests/test_v2_presenter.py -q -p no:cacheprovider --basetemp=.pytest_tmp/wpc1` — EXPECT: 0 failed
- G2 — CHECK: `python -m pytest tests -q -p no:cacheprovider --basetemp=.pytest_tmp/wpc2` — EXPECT: only the one known failure.

## Report (final message, under 300 words)

Files changed; gate lines; expectations changed and why; where the code differed from this spec
and what you did; anything you left unconnected.
