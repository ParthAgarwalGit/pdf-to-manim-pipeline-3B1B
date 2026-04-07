FROM manimcommunity/manim:latest

WORKDIR /app

# Copy project files
COPY pyproject.toml pyproject.toml
COPY src/ src/

# Install Python dependencies from pyproject.toml
RUN pip install --no-cache-dir -e .

# Set entrypoint to CLI
ENTRYPOINT ["pdf2manim"]
CMD ["--help"]
