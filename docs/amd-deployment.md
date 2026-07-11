# Running on AMD Developer Cloud

This pipeline is containerized so the whole video-generation workload runs on an
AMD Developer Cloud instance.

## What runs on AMD

The pipeline is a **heterogeneous** system, and the split is deliberate:

| Work | Where it runs | Why |
|---|---|---|
| **Manim rendering + FFmpeg encoding** (the compute bulk — dozens of scenes per document, parallelized) | **AMD Developer Cloud** (AMD EPYC CPUs; optional AMD GPU, see below) | This is the majority of wall-clock compute. It's deterministic and quality-neutral, so it belongs on AMD. |
| Frontier reasoning: script, storyboard, vision/OCR, Manim code generation | Frontier models via API (gpt-5, claude) | These decide output quality. Swapping them for weaker open models would degrade the videos, so they are intentionally left on the frontier API. |

The honest claim this supports: **the video-generation pipeline runs on AMD
Developer Cloud; frontier models are used only for the reasoning/codegen step
over the API.** Rendering genuinely is the bulk of the compute, so this is not
a token gesture.

## Quick start

```bash
# 1. On the AMD Developer Cloud instance, clone the repo and cd in.
# 2. Provide keys (git-ignored, never baked into the image):
cp keys.example.env keys.env && $EDITOR keys.env

# 3. Prove the container is sound WITHOUT spending credits:
scripts/run_on_amd.sh smoke        # builds + keyless --help and --dry-run

# 4. Real conversion:
scripts/run_on_amd.sh convert /path/to/paper.pdf -q high

# 5. Or launch the web demo (the submission's Application URL):
scripts/run_on_amd.sh demo         # http://<instance-ip>:7860
```

`docker compose` works too:

```bash
docker compose run --rm convert   # uses ./input/document.pdf -> ./output
docker compose up demo            # web demo on :7860
```

## Capturing evidence for the submission

Judges reward a visible AMD footprint. While a render runs on the instance:

- `htop` / `lscpu` — shows the render/encode saturating **AMD EPYC** cores.
- Instance detail page — screenshot the **AMD Developer Cloud** instance type.
- If using the optional GPU path below: `rocm-smi` during a render — shows the
  **AMD Instinct/Radeon** GPU active. This single screenshot is strong material
  for the slides and demo video.

## Optional: use the AMD GPU for rendering/encoding (ROCm)

The default path renders on AMD CPUs — rock-solid and dependency-free. If your
instance has an AMD GPU and you want a stronger ROCm/Instinct story (still
quality-neutral), two levers:

1. **GPU OpenGL rendering.** Manim can render with the OpenGL renderer, which
   uses the GPU. Set `MANIM_RENDERER=opengl` and ensure the container has EGL +
   Mesa/RADV; on headless Instinct nodes this needs a headless EGL setup, so it
   is documented rather than enabled by default.
2. **Hardware video encode.** FFmpeg VAAPI (`-hwaccel vaapi -vaapi_device
   /dev/dri/renderD128`) offloads H.264/HEVC encoding to the AMD GPU. Pass the
   render node into the container with `--device /dev/dri`.

Both are opt-in so the default build stays reliable for the deadline.

## Notes

- `keys.env` is excluded by both `.gitignore` and `.dockerignore`; keys are
  passed at run time via `--env-file`, never copied into the image.
- The base image (`manimcommunity/manim`) already contains LaTeX, Cairo/Pango,
  and FFmpeg, so builds are self-contained.
- TTS narration uses OpenAI TTS by default. It is not required for a video —
  pass `--skip-tts` for a faster, audio-free run.
