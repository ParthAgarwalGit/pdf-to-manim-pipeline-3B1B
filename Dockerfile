# PDF -> Manim pipeline, containerized for AMD Developer Cloud.
#
# The manim community image already ships the heavy system deps this pipeline
# needs: LaTeX (math rendering), Cairo/Pango (Manim), FFmpeg (encode + frame QA)
# and Python. On AMD Developer Cloud this container runs on AMD EPYC CPUs for
# rendering/encoding; frontier-model reasoning happens over the API (keys passed
# at run time), so no model weights are baked in.
FROM manimcommunity/manim:latest

# Install as root so pip can write to the image's environment, and so the
# mounted output directory is writable at run time.
USER root
WORKDIR /app

# Copy packaging metadata FIRST for better layer caching.
# NOTE: pyproject.toml sets `readme = "README.md"`, so README.md MUST be present
# at build time or `pip install` fails — this was the original build break.
COPY pyproject.toml README.md LICENSE NOTICE requirements.txt ./

# Application code + web demo.
COPY src/ src/
COPY app.py app.py

# Install the package plus the optional web-demo extra (Gradio).
RUN pip install --no-cache-dir -e ".[demo]"

# CLI is the default entrypoint:
#   docker run --rm --env-file keys.env -v "$PWD/output:/app/output" \
#     pdf2manim convert /app/input/doc.pdf --output-dir /app/output
# The web demo is opt-in by overriding the entrypoint:
#   docker run --rm -p 7860:7860 --env-file keys.env --entrypoint python IMAGE app.py
ENTRYPOINT ["pdf2manim"]
CMD ["--help"]
