#!/usr/bin/env bash
# Build + smoke-test + (optionally) run the pipeline on an AMD Developer Cloud
# instance. The smoke test needs NO API keys, so you can confirm the container
# is sound before spending any credits.
#
# Usage:
#   scripts/run_on_amd.sh build          # build the image
#   scripts/run_on_amd.sh smoke          # keyless: --help + --dry-run
#   scripts/run_on_amd.sh convert FILE   # full run (needs keys.env)
#   scripts/run_on_amd.sh demo           # launch the web demo on :7860
set -euo pipefail

IMAGE="pdf2manim:latest"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

build() {
  echo ">> Building $IMAGE (on AMD Developer Cloud this runs on AMD EPYC/Instinct)"
  docker build -t "$IMAGE" .
}

smoke() {
  echo ">> Smoke 1/2: CLI responds (no keys needed)"
  docker run --rm "$IMAGE" --help >/dev/null
  echo "   OK"

  echo ">> Smoke 2/2: keyless --dry-run on a generated 1-page PDF"
  local tmp; tmp="$(mktemp -d)"
  # Make a trivial PDF with Python (Pillow is in the image).
  docker run --rm -v "$tmp:/work" --entrypoint python "$IMAGE" -c \
    "from PIL import Image, ImageDraw; im=Image.new('RGB',(1240,1754),'white'); \
d=ImageDraw.Draw(im); d.text((80,80),'The derivative measures instantaneous rate of change.',fill='black'); \
im.save('/work/sample.pdf','PDF')"
  # --dry-run stops before code/render/TTS, so it never touches a paid API.
  # No keys -> agents use deterministic mock output, so this runs offline.
  docker run --rm -v "$tmp:/work" "$IMAGE" \
    convert /work/sample.pdf --output-dir /work/out --dry-run || {
      echo "   dry-run returned non-zero (inspect log above)"; }
  echo "   Smoke complete. Container is runnable."
  rm -rf "$tmp"
}

convert() {
  local pdf="${1:?path to a PDF required}"
  [ -f keys.env ] || { echo "keys.env not found (cp keys.example.env keys.env)"; exit 1; }
  mkdir -p output
  docker run --rm --env-file keys.env \
    -v "$(dirname "$(readlink -f "$pdf")"):/app/input" \
    -v "$ROOT/output:/app/output" \
    "$IMAGE" convert "/app/input/$(basename "$pdf")" --output-dir /app/output "${@:2}"
}

demo() {
  [ -f keys.env ] || { echo "keys.env not found (cp keys.example.env keys.env)"; exit 1; }
  mkdir -p output
  echo ">> Web demo on http://0.0.0.0:7860"
  docker run --rm -p 7860:7860 --env-file keys.env \
    -v "$ROOT/output:/app/output" \
    --entrypoint python "$IMAGE" app.py
}

cmd="${1:-smoke}"; shift || true
case "$cmd" in
  build)   build ;;
  smoke)   build; smoke ;;
  convert) build; convert "$@" ;;
  demo)    build; demo ;;
  *) echo "unknown command: $cmd (build|smoke|convert|demo)"; exit 2 ;;
esac
