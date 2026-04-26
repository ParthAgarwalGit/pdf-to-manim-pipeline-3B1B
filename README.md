# PDF to Manim Animation Pipeline

An agent-orchestrated system that transforms academic PDF documents into 3Blue1Brown-style animated educational videos using the Manim animation engine.

## Overview

This pipeline converts static academic content (text, equations, diagrams, proofs) into executable Manim animation code with accompanying narration scripts. The system emphasizes visual intuition and conceptual continuity over textual density, following 3Blue1Brown's pedagogical principles.

## Architecture

The pipeline follows an agent-oriented architecture with the following stages:

1. **Ingestion Agent** - Detects cloud storage uploads and routes files
2. **Validation Agent** - Verifies PDF file type and structural integrity
3. **Vision + OCR Agent** - Extracts text, mathematical notation, and diagrams
4. **Script/Storyboard Architect** - Generates pedagogically sound scene sequences
5. **JSON Sanitizer** - Enforces schema compliance on storyboard output
6. **Scene Controller** - Manages deterministic iteration over scenes
7. **Persistence Agent** - Writes generated artifacts to file system
8. **Video Renderer Agent** - Renders generated scene files with Manim
9. **TTS Agent** - Generates narration audio
10. **Audio-Video Composer Agent** - Produces the final narrated video

Each agent has a single responsibility, explicit input/output contracts, and isolated failure modes.

## Installation

### Prerequisites

- Python 3.10+
- LaTeX (for mathematical notation rendering)
- FFmpeg (for video rendering)

#### Install LaTeX

**Ubuntu/Debian:**
```bash
sudo apt-get install texlive-full
```

**macOS:**
```bash
brew install --cask mactex
```

**Windows:**
Install MiKTeX or TeX Live from their respective websites.

#### Install FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org)

### Python Dependencies

```bash
pip install -e ".[dev]"
```

If you only want runtime dependencies, `pip install -r requirements.txt` still works.

## Usage

### Process a Single PDF

```bash
pdf2manim convert path/to/document.pdf --output-dir ./output
```

### Process with Custom Configuration

```bash
pdf2manim convert document.pdf \
    --verbosity high \
    --depth advanced \
    --audience graduate \
    --voice nova
```

### Execution Modes

```bash
# Planning only — no rendering or TTS (fast iteration)
pdf2manim convert document.pdf --dry-run

# Skip TTS (render video without narration audio)
pdf2manim convert document.pdf --skip-tts

# Skip rendering entirely (generate scene code only)
pdf2manim convert document.pdf --skip-render

# Set render quality: low / medium / high / 4k
pdf2manim convert document.pdf --quality high
```

### Estimate Cost Before Running

```bash
pdf2manim cost document.pdf
```

## Testing

### Run All Tests

```bash
python -m pytest
```

### Run with Coverage

```bash
python -m pytest --cov-report=html
```

### Run Property-Based Tests Only

```bash
python -m pytest -k "property"
```

### Run Specific Test File

```bash
python -m pytest tests/test_orchestrator.py
```

## Project Structure

```
.
├── src/
│   ├── agents/          # Agent implementations
│   ├── schemas/         # Pydantic data models
│   ├── orchestrator/    # Pipeline orchestration
│   └── utils/           # Shared utilities
├── tests/               # Test suite
├── output/              # Generated artifacts (runtime)
├── pyproject.toml      # Packaging, CLI, pytest, and lint configuration
├── requirements.txt    # Runtime dependency fallback
└── README.md           # This file
```

## Output Structure

Generated artifacts are organized in a deterministic directory structure:

```
output/{pdf_filename}/{timestamp}/
├── scene_*.py              # Generated Manim code
├── narration_script.txt    # Narration with scene markers
├── storyboard.json         # Scene specifications
├── processing_report.json  # Success/failure summary
├── storyboard_review.json  # Storyboard quality review
├── scene_code_review.json  # Scene code validation review
├── video_review.json       # Final media quality review
├── manifest.json           # File listing
└── pipeline.log            # Structured execution logs
```

## Key Features

- **Deterministic Output**: Same input produces structurally equivalent output
- **Failure Isolation**: Scene-level failures don't abort entire pipeline
- **Incremental Progress**: Intermediate artifacts can be inspected and reused
- **Property-Based Testing**: Universal correctness properties validated across all inputs
- **Structured Logging**: Detailed execution traces for debugging

## Configuration

Configuration parameters control pedagogical style:

- **verbosity**: `low` | `medium` | `high` - Controls narration detail level
- **depth**: `introductory` | `intermediate` | `advanced` - Controls content depth
- **audience**: `high-school` | `undergraduate` | `graduate` - Adjusts terminology

## Development

### Agent Interface

All agents implement a common interface:

```python
class Agent(ABC):
    def execute(self, input_data: AgentInput) -> AgentOutput:
        """Execute agent logic"""
        pass
    
    def validate_input(self, input_data: AgentInput) -> bool:
        """Validate input conforms to expected schema"""
        pass
    
    def get_retry_policy(self) -> RetryPolicy:
        """Return retry policy for this agent"""
        pass
```

### Adding a New Agent

1. Create agent class in `src/agents/`
2. Implement `Agent` interface
3. Define input/output schemas in `src/schemas/`
4. Write unit tests in `tests/`
5. Add agent to orchestrator sequence

## Documentation

- [Output Contract](docs/output-contract.md) — manifest schema, file types, and inspection commands
- [Troubleshooting](docs/troubleshooting.md) — common failures and fixes
- [Determinism](docs/determinism.md) — reproducibility guarantees

## License

[To be determined]

## Contributing

[To be determined]
