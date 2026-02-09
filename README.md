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
7. **Animation Code Generator** - Converts scene specifications to Manim code
8. **Persistence Agent** - Writes generated artifacts to file system

Each agent has a single responsibility, explicit input/output contracts, and isolated failure modes.

## Installation

### Prerequisites

- Python 3.8+
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
pip install -r requirements.txt
```

## Usage

### Process a Single PDF

```bash
python -m src.orchestrator.pipeline --input path/to/document.pdf --config config.json
```

### Process with Custom Configuration

```bash
python -m src.orchestrator.pipeline \
    --input document.pdf \
    --verbosity high \
    --depth advanced \
    --audience graduate
```

### Render Generated Animations

```bash
# Render a single scene
manim -pql output/document/timestamp/scene_001.py Scene_001

# Render all scenes in directory
for file in output/document/timestamp/scene_*.py; do
    manim -pql "$file"
done
```

## Testing

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html
```

### Run Property-Based Tests Only

```bash
pytest -k "property"
```

### Run Specific Test File

```bash
pytest tests/test_validation_agent.py
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
├── requirements.txt     # Python dependencies
├── pytest.ini          # Pytest configuration
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

## License

[To be determined]

## Contributing

Parth Agarwal
Vignesh KS
Divyansh Bisht
Sumit Galgalkar
