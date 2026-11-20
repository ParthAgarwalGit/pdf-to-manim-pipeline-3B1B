# WPA — One Gemini key: Gemini 4 Argon for text stages, Gemini TTS for narration

You are implementing one work package in this repository. The decisions are made; implement
exactly what is written. Where this spec says "read X to find Y", the wiring is yours to find,
but the behaviour is fixed.

## Ground rules (binding)

- Work only in this checkout, on its current branch (`claude/unified-v2`). **No git commands of
  any kind** (no commit, add, stash, checkout, reset, push).
- **Frozen, read-only:** `src/pipeline_v2/prompt_text.py`, everything under `src/prompts/`.
- No paid API calls. Tests are keyless (`tests/conftest.py` strips keys; mock providers).
- The only shell command you run is `python -m pytest ... -p no:cacheprovider --basetemp=.pytest_tmp/wpa`.
- Never weaken, skip or delete a test to get green. If a test fails because this spec changes
  behaviour on purpose, update its expectation and list it in your report; otherwise stop and report.
- Keep `run.json` / `run.db` of older runs loadable: new config fields have defaults.
- Known pre-existing failure you must leave alone:
  `tests/test_v2_delivery.py::TestRealMedia::test_mixed_clips_join_into_one_uniform_file`.

## Background

The user wants the V2 pipeline (`src/pipeline_v2/`, CLI `src/cli.py`) to need only a
`GEMINI_API_KEY`, and to use the strongest Gemini model for OCR, lesson/narration writing and
Manim code generation. They listened to a test and approved Gemini TTS with the prebuilt voice
`Sulafat` (constant `DEFAULT_VOICE` in `src/pipeline_v2/prompt_text.py`) for narration; the
on-camera presenter (a later package) will use the same voice, so narration and presenter sound
like one person. OpenAI `tts-1` with a 0.94x time-stretch sounded monotonic and is no longer the
default.

## A1. One place for model ids — new file `src/pipeline_v2/model_config.py`

```python
"""Model ids for every paid V2 stage. One place to change them."""

# Gemini 4 Argon: Google's documented latest and most capable developer model (Nov 2026).
TEXT_MODEL = "gemini-4-argon"      # lesson, narration, narration review, brief interpretation
OCR_MODEL = "gemini-4-argon"       # low-confidence page transcription (vision)
CODE_MODEL = "gemini-4-argon"      # Manim generation, revision and screenshot-guided repair
TTS_MODEL = "gemini-3.8-flash-tts"
PRESENTER_MODEL = "gemini-omni-1.2-flash"
# Seconds of audio per output token unit for Gemini TTS pricing: 25 audio tokens per second.
GEMINI_TTS_TOKENS_PER_SECOND = 25
```

Replace every hard-coded model id in `src/pipeline_v2/` with these constants:
- `planning.py`: `PLANNING_MODEL` becomes `TEXT_MODEL` (keep the name `PLANNING_MODEL` as an alias
  assignment `PLANNING_MODEL = TEXT_MODEL`, other modules import it); the OCR call's
  `model="gemini-3.7-flash"` becomes `OCR_MODEL`; the two `visual_contract` calls that use
  `provider="openai", model="gpt-5.4-mini"` become `provider="google", model=TEXT_MODEL`.
- `generation.py`: all three `model="gemini-3.7-flash"` become `CODE_MODEL`.
- `directives.py`: `BRIEF_PROVIDER = "google"`, `BRIEF_MODEL = TEXT_MODEL`.
- Any other `"gemini-3.7-flash"`, `"gpt-5.4-mini"` or `"gpt-5.4"` in `src/pipeline_v2/` or
  `src/evaluation/` that drives a pipeline stage (grep for them). In `src/evaluation/`, keep the
  `gemini`/`gemma` model aliases working, and change the `gemini` alias to `CODE_MODEL`.
- Bump `planning.PROMPT_VERSION`, `planning.NARRATION_PROMPT_VERSION` and
  `generation.CODE_PROMPT_VERSION` (append `-argon`) so cached responses from the old model
  are not reused.

## A2. Prices — `src/pipeline_v2/providers.py`

Add to `PRICES` (USD per 1M units, input then output) and set `PRICING_SNAPSHOT_DATE = "2026-11-17"`:

```python
("google", "gemini-4-argon"): (2.00, 10.00),        # $4.00 / $20.00 from 2026-12-01
("google", "gemini-3.8-flash"): (0.75, 3.75),
("google", "gemini-3.8-flash-tts"): (0.50, 9.00),   # text in / audio tokens out (25 per s)
("google", "gemini-omni-1.2-flash"): (1.50, 12.00), # text in / video tokens out (720p ~$0.07/s)
```

Keep the existing entries. Add a module comment that the Argon price doubles on 2026-12-01 and
the reservation logic should be re-checked then.

If the Gemini call path in `ProviderRouter._invoke` sets
`thinking_config=types.ThinkingConfig(thinking_budget=GOOGLE_STRUCTURED_THINKING_BUDGET)`, keep
it for every model. (A live smoke test after this package will confirm Argon accepts it; do not
guess an alternative.)

## A3. Gemini TTS — `src/pipeline_v2/tts.py`

1. `PipelineV2Config` (`models.py`): change `voice: str = "alloy"` to
   `voice: str = DEFAULT_VOICE` (import from `prompt_text`), and add
   `tts_provider: Literal["gemini", "openai"] = "gemini"`.
2. `NarrationAudioService.__init__` gains `provider: str = "gemini"` and
   `audience: Optional[str] = None`; the pipeline passes `config.tts_provider` and
   `config.audience`.
3. In `generate()`:
   - The keyless fallback checks the key of the configured provider (`GEMINI_API_KEY` for
     gemini, `OPENAI_API_KEY` for openai) and its messages name that key. Behaviour otherwise
     unchanged.
   - provider `"openai"`: today's code path exactly as it is.
   - provider `"gemini"`: new private method `_gemini_tts(narration_text, out_wav) -> float`
     (returns audio seconds) run via `asyncio.to_thread` under `self.api_limit`:
     ```python
     from google import genai
     from google.genai import types
     client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
     response = client.models.generate_content(
         model=TTS_MODEL,
         contents=f"{tts_direction(self.audience)}\n\n{narration_text}",
         config=types.GenerateContentConfig(
             response_modalities=["AUDIO"],
             speech_config=types.SpeechConfig(
                 voice_config=types.VoiceConfig(
                     prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice))),
         ),
     )
     pcm = response.candidates[0].content.parts[0].inline_data.data
     ```
     Write `pcm` as 24 kHz, 16-bit, mono WAV to `audio_dir / f"{scene.scene_id}_raw.wav"`
     (`wave` module), then convert to `audio_dir / f"{scene.scene_id}.mp3"` with ffmpeg
     (`get_ffmpeg()`, `-codec:a libmp3lame -q:a 2`). Duration = `len(pcm) / 48000`.
   - Cost for gemini: reserve before the call with
     `price("google", TTS_MODEL, estimate_tokens(prompt_text), estimated_seconds * 25)` where
     `estimated_seconds = len(narration_text.split()) * 60 / scene.target_words_per_minute + 5`;
     settle with the measured seconds (`seconds * GEMINI_TTS_TOKENS_PER_SECOND`) and
     `exact=True`. Record events and provider calls exactly as the openai branch does, with
     `provider="google"`, `model=TTS_MODEL`, `voice=self.voice`. A failure settles
     `FAILED_UNCERTAIN` like the openai branch.
   - Pacing for gemini: call `_pace_audio(original, paced, speech_rate=1.0,
     SCENE_END_PAUSE_SECONDS)` — the voice is directed to speak at teaching pace, so no
     time-stretch; only the trailing pause is added. The `RELAXED_SPEECH_RATE` path applies to
     openai only; for gemini under `--pacing relaxed`, use rate 1.0 and rely on the prompt
     (narration is already written for 115 wpm). Keep `TTS_TIMING_VERSION` semantics: bump it to
     `"tts-v2-5-gemini"` so old cached timing is not reused for gemini audio.
   - `repace()` with provider gemini: same rule (rate 1.0).
4. `revision.py`: `has_openai = bool(os.getenv("OPENAI_API_KEY")) and tts_enabled` becomes a
   check of the configured TTS provider's key (rename the variable `has_tts_key`; update the
   estimate function's parameter name accordingly).

## A4. CLI — `src/cli.py`

- `--voice`: free text (no `click.Choice`), default `DEFAULT_VOICE`, help: "Prebuilt voice name:
  a Gemini voice such as Sulafat, Kore or Puck for --tts-provider gemini; alloy, nova, ... for
  openai."
- New `--tts-provider` option, `click.Choice(["gemini", "openai"])`, default `"gemini"`, passed
  into `PipelineV2Config.tts_provider`. If `--tts-provider openai` is given with the default
  voice, use `"alloy"` (an OpenAI voice) instead.
- `--max-cost` default from `1.0` to `3.0` (Argon is pricier than 3.7 Flash; a 10-scene video
  with repairs measured just under $1 on Flash). Update the README sentence about the default.

## A5. Keys and docs

- `keys.example.env`: `GEMINI_API_KEY` is the only required key; `OPENAI_API_KEY` is optional and
  only needed for `--tts-provider openai` or the legacy pipeline; keep the D-ID and Anthropic
  notes as optional/legacy. Update the header comment to list what Gemini drives (OCR, lesson,
  narration, narration review, Manim code, repair, TTS, presenter).
- `SETUP.md`: replace the "Required APIs" / "API Configuration" material that asks for OpenAI,
  Anthropic and Mathpix with: one `GEMINI_API_KEY` (from Google AI Studio), the models table
  from `model_config.py`, and the optional keys. Remove the stale "mock agents" framing in the
  first paragraph of SETUP.md (V2 uses real providers when the key is present and deterministic
  fallbacks without it). Keep everything else in SETUP.md (install, ffmpeg, LaTeX, Docker).
- `README.md`: in "Installation", the keys block shows only `GEMINI_API_KEY` (+ the optional
  keys line). Replace "OPENAI_API_KEY=... # TTS". Mention Gemini TTS voice `Sulafat` default.

## Tests — add `tests/test_v2_gemini_models.py`

1. `test_stage_models_are_argon` — planning, OCR, code generation, revise, repair and brief
   interpretation read their model id from `model_config` and it equals `"gemini-4-argon"`
   (assert on the constants imported by each module, e.g. `planning.PLANNING_MODEL`).
2. `test_no_openai_model_in_v2_stages` — grep-style: no `.py` file under `src/pipeline_v2/`
   contains `"gpt-5.4` or `provider="openai"` except `tts.py` (openai branch) and
   `providers.py` (price table / generic invoke).
3. `test_prices_cover_v2_models` — `price("google", m, 1_000_000, 1_000_000)` is positive for
   every model in `model_config`.
4. `test_gemini_tts_writes_audio_and_settles_cost` — monkeypatch `GEMINI_API_KEY`, patch
   `google.genai.Client` so `models.generate_content` returns an object whose
   `candidates[0].content.parts[0].inline_data.data` is 2 s of silence PCM (96000 bytes), record
   the call kwargs; patch ffmpeg-dependent helpers only if needed (ffmpeg is installed on this
   machine, prefer the real one). Assert: the request used `TTS_MODEL`, voice `Sulafat`, and
   contents starting with `tts_direction(...)`; the returned scene has `audio_path` set and an
   `audio_duration_seconds` of 2 s plus the trailing pause; the store recorded a settled call
   with provider `google`.
5. `test_gemini_tts_keyless_fallback_names_gemini_key` — no key: silent placeholder audio and a
   `tts_fallback` event whose message mentions `GEMINI_API_KEY`.
6. `test_openai_tts_path_unchanged` — provider `"openai"` still goes through `TTSAgent` (patch it).
7. `test_cli_voice_and_tts_provider_defaults` — via `CliRunner` with `PipelineV2` patched (copy
   the pattern in `tests/test_v2_integration_wiring.py`): default config has
   `voice == "Sulafat"`, `tts_provider == "gemini"`, `max_cost_usd == 3.0`;
   `--tts-provider openai` gives voice `"alloy"`.

## Gates (run, report each in one line)

- G1 — CHECK: `python -m pytest tests/test_v2_gemini_models.py -q -p no:cacheprovider --basetemp=.pytest_tmp/wpa1` — EXPECT: `7 passed`
- G2 — CHECK: `python -m pytest tests -q -p no:cacheprovider --basetemp=.pytest_tmp/wpa2` — EXPECT: only the one known failure listed above.
- G3 — CHECK (grep): no remaining `gemini-3.7-flash` string in `src/pipeline_v2/` outside `model_config.py` and the price table.

## Report (final message, under 300 words)

Files changed (one line each); each gate's result line; tests whose expectations you changed and
why; anything in this spec that did not match the code and what you did.
