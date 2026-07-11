# Submission pack — AMD Developer Hackathon: ACT II

Copy-paste starting points for the lablab.ai form. Edit to taste; keep the AMD
framing truthful (see [amd-deployment.md](amd-deployment.md)).

**Track:** Unicorn Track (judged on creativity, originality, completeness, use
of AMD platforms, and product/market potential).

---

## Project Title
**Grasp — turn any paper into a 3Blue1Brown-style explainer series, rendered on AMD**

## Short Description (1 line)
Upload a PDF; get a provably-complete series of short, animated explainer
vignettes — with rendering and encoding running on AMD Developer Cloud.

## Long Description

**Problem.** Dense academic PDFs are how knowledge is distributed and the worst
way to actually learn it. A great explainer video (think 3Blue1Brown) *draws*
each idea with motion — but producing one is hours of manual Manim work per
concept, so it doesn't scale to the millions of papers people need to
understand.

**Solution.** Grasp is a multi-agent pipeline that reads a PDF and emits a
*series* of short animated vignettes — one focused visual idea each — that
together cover the whole document. It is visualization-first: every scene must
draw the concept, never narrate over a wall of text.

**How it works (10-stage pipeline with quality loops):** ingestion → PDF
validation → vision/OCR extraction → a storyboard architect that segments the
source for guaranteed full coverage, scripts one beat per segment, and emits a
schema-validated *visual plan* (closed catalog of Manim primitives, each with an
(x, y) position and a visibility window) → a storyboard grader → scene code
generation that turns each plan into Manim code → parallel rendering + narration
→ audio/video composition with a frame-level visual-QA reviewer that samples
rendered pixels and flags blank or static scenes for regeneration.

**What makes it novel:**
- **Coverage guarantee** — the storyboard provably covers every source segment
  exactly once; gaps and redundancy are detected and reported.
- **Visual-archetype library** — 11 curated concept→visual patterns steer each
  scene toward a proven design (tangent-sweep for derivatives, tip-to-tail for
  vectors, grid-morph for matrices, …).
- **Closed primitive catalog** — the model can only emit schema-validated Manim
  primitives, so hallucinated classes are rejected at the boundary, not at
  render time.
- **AST security gate** — generated scene code is screened (import allowlist,
  no eval/exec/reflection) before execution, defending against prompt injection
  from a malicious PDF.
- **Frame-level visual QA** — rendered frames are judged directly, not just the
  code text.

**Use of AMD.** The pipeline is containerized and runs on **AMD Developer
Cloud**. Rendering and encoding — the bulk of the compute, dozens of scenes per
document — run on AMD (EPYC CPUs, with an optional ROCm GPU render/encode path).
Frontier models are used only for the reasoning/codegen step over the API, so
the visual quality that defines the product is uncompromised while the heavy,
parallelizable workload runs on AMD.

**Result.** [Link your hero video here.] A N-page paper becomes M short
vignettes in one run; cost is tracked per run in the manifest.

## Technology & Category Tags
`AMD Developer Cloud`, `AMD EPYC`, `ROCm`, `Manim`, `Multi-agent`, `LLM`,
`Generative Video`, `Computer Vision`, `EdTech`, `Python`, `Docker`, `Pydantic`,
`FFmpeg`

---

## Cover Image
A 2x3 contact sheet of striking frames from the hero render (a clean tangent
sweep, a vector tip-to-tail, a matrix grid-morph), with the title and a small
"Runs on AMD Developer Cloud" badge. Alternative: the 10-stage pipeline diagram
with the render/encode stages marked as running on AMD.

## Video Presentation (~3–4 min, clarity over polish)
1. **0:00–0:20 — Problem.** Scroll a dense PDF; "this is unwatchable."
2. **0:20–1:50 — Live demo.** Upload the PDF in the web app; show the vignette
   series playing. This is the most important 90 seconds — let the output speak.
3. **1:50–2:50 — How it works + AMD.** The 10-stage pipeline; coverage
   guarantee, archetype library, security gate, frame-QA. Show the render
   running on the AMD Developer Cloud instance (`htop`/`rocm-smi` screenshot).
4. **2:50–3:30 — Vision.** Any paper → an explainer series; where this goes.

## Slide Presentation (outline)
1. Title + one-line pitch + "Runs on AMD Developer Cloud".
2. Problem: papers don't teach; hand-made explainers don't scale.
3. Solution + a hero frame.
4. Architecture diagram (10 stages), AMD render/encode stages highlighted.
5. What's novel (coverage guarantee, archetype library, security gate, frame-QA).
6. AMD Developer Cloud: what runs where + the evidence screenshot.
7. Results (hero video) + product vision + team.

## Submission checklist (map to the form)
- [ ] Public GitHub repo with README setup/usage — **done** (this repo)
- [ ] Containerized + runnable — **done** (`Dockerfile`, `scripts/run_on_amd.sh smoke`)
- [ ] Application URL — host `app.py` on the AMD instance (`docker compose up demo`)
- [ ] Hero video produced on the AMD instance (long pole — start early)
- [ ] Cover image, video presentation, slides
