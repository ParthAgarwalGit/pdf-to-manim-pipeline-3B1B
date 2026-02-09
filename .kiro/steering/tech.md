# Technology Stack

## Core Technologies

### Language & Runtime
- **Python 3.8+**: Primary implementation language
- **Manim Community Edition (v0.17+)**: Mathematical animation engine for video generation

### Key Dependencies
- **pydantic**: Schema validation and data contracts
- **pytest**: Testing framework
- **hypothesis**: Property-based testing library
- **boto3**: AWS S3 integration for cloud storage triggers
- **jsonschema**: JSON schema validation
- **LaTeX**: Required for mathematical notation rendering (MathTex)
- **FFmpeg**: Required for video rendering from Manim

### LLM Integration
- **Multimodal Vision Models**: GPT-4 Vision, Claude 3 Opus (OCR, diagram interpretation)
- **Code Generation Models**: GPT-4, Claude 3.5 Sonnet, Codex (Manim code generation)
- **Large Context Models**: 128K+ token context for full PDF processing
- **Temperature=0**: Deterministic sampling for reproducibility

### Cloud Storage
- **AWS S3** / Google Cloud Storage / Azure Blob Storage
- Event-driven triggers for PDF upload detection

## Build & Development Commands

### Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install LaTeX (required for Manim)
# Ubuntu/Debian: sudo apt-get install texlive-full
# macOS: brew install --cask mactex
# Windows: Install MiKTeX or TeX Live

# Install FFmpeg (required for Manim)
# Ubuntu/Debian: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
# Windows: Download from ffmpeg.org
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run property-based tests only
pytest -k "property"

# Run specific test file
pytest tests/test_validation_agent.py
```

### Running the Pipeline
```bash
# Process a single PDF
python -m src.orchestrator.pipeline --input path/to/document.pdf --config config.json

# Process with custom configuration
python -m src.orchestrator.pipeline --input document.pdf --verbosity high --depth advanced --audience graduate
```

### Rendering Animations
```bash
# Render a single scene
manim -pql output/document/timestamp/scene_001.py Scene_001

# Render all scenes in directory
for file in output/document/timestamp/scene_*.py; do
    manim -pql "$file"
done
```

## Testing Strategy

### Property-Based Tests (Hypothesis)
- Validate universal correctness properties across all inputs
- Used for: file validation, JSON sanitization, code generation, determinism, idempotency

### Unit Tests (pytest)
- Validate specific examples, edge cases, error conditions
- Used for: agent logic, schema validation, error handling, transformations

### Integration Tests
- End-to-end pipeline execution with sample PDFs
- Concurrent processing validation
- Performance benchmarks (latency constraints)
