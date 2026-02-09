# PDF to Manim Pipeline - Setup Guide

## Overview

This guide will help you set up and run the PDF-to-Manim pipeline. The current implementation uses **mock agents** for development and testing. To enable full end-to-end functionality with real LLM integration, you'll need to configure API keys.

## Current Status

✅ **Fully Functional (Mock Mode)**
- All 8 agents implemented with mock/deterministic behavior
- Complete pipeline orchestration
- File validation and processing
- Scene generation and persistence
- 358 passing tests with 93% code coverage

⚠️ **Requires API Setup (Production Mode)**
- Vision+OCR Agent: Needs multimodal vision API (GPT-4 Vision, Claude 3 Opus, or Mathpix)
- Script Architect: Needs LLM API for storyboard generation
- Animation Generator: Needs code generation API

## Quick Start (Mock Mode)

### 1. Prerequisites

- **Python 3.8+** installed
- **Git** installed (for version control)
- **pip** package manager

### 2. Installation

```bash
# Clone or navigate to the project directory
cd path/to/pdf-to-manim-pipeline

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
pytest tests/ -v
```

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test suite
pytest tests/test_orchestrator.py -v
```

### 4. Process a PDF (Mock Mode)

```python
# example_usage.py
from src.orchestrator.pipeline import Orchestrator, PipelineConfig

# Create orchestrator with configuration
config = PipelineConfig(
    verbosity="medium",
    depth="intermediate",
    audience="undergraduate",
    base_output_dir="output"
)

orchestrator = Orchestrator(config=config)

# Process a PDF file
manifest = orchestrator.run_pipeline(
    pdf_file_path="path/to/your/document.pdf",
    event_source="local"
)

print(f"Pipeline completed with status: {manifest.status}")
print(f"Output directory: {manifest.output_directory}")
print(f"Generated {manifest.successful_scenes} scenes")
```

## Production Setup (Real LLM Integration)

### Required APIs

#### 1. **Vision+OCR Agent** (Choose one)

**Option A: OpenAI GPT-4 Vision**
```bash
# Set environment variable
export OPENAI_API_KEY="your-api-key-here"
```

**Option B: Anthropic Claude 3 Opus**
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

**Option C: Mathpix OCR** (for math notation)
```bash
export MATHPIX_APP_ID="your-app-id"
export MATHPIX_APP_KEY="your-app-key"
```

#### 2. **Script Architect Agent**

Uses the same LLM API as Vision+OCR (OpenAI or Anthropic).

#### 3. **Animation Generator Agent**

Uses the same LLM API for code generation.

### API Configuration

Create a `.env` file in the project root:

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MATHPIX_APP_ID=...
MATHPIX_APP_KEY=...

# Optional: AWS S3 for cloud storage triggers
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET_NAME=pdf-uploads
```

### Update Agent Implementations

To enable real LLM calls, you'll need to update these files:

1. **`src/agents/vision_ocr.py`** - Replace mock OCR with real API calls
2. **`src/agents/script_architect.py`** - Replace mock storyboard generation with LLM
3. **`src/agents/animation_generator.py`** - Replace mock code generation with LLM

Example for Vision+OCR Agent:

```python
# In src/agents/vision_ocr.py
import openai
import os

class VisionOCRAgent(Agent):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
    
    def _extract_diagram_description(self, image_data):
        # Real API call instead of mock
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this diagram in detail:"},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
            }],
            temperature=0  # Deterministic
        )
        return response.choices[0].message.content
```

## Optional Dependencies

### For Full Manim Rendering

If you want to render the generated animations:

```bash
# Install Manim Community Edition
pip install manim

# Install LaTeX (required for math rendering)
# Ubuntu/Debian:
sudo apt-get install texlive-full

# macOS:
brew install --cask mactex

# Windows:
# Download and install MiKTeX from https://miktex.org/

# Install FFmpeg (required for video rendering)
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Windows:
# Download from https://ffmpeg.org/
```

### For Cloud Storage Integration

```bash
# AWS S3
pip install boto3

# Google Cloud Storage
pip install google-cloud-storage

# Azure Blob Storage
pip install azure-storage-blob
```

## Project Structure

```
pdf-to-manim-pipeline/
├── src/                    # Source code
│   ├── agents/            # 8 agent implementations
│   ├── schemas/           # Pydantic data models
│   ├── orchestrator/      # Pipeline orchestration
│   └── utils/             # Shared utilities
├── tests/                 # Test suite (358 tests)
├── docs/                  # Documentation
├── output/                # Generated animations (runtime)
├── requirements.txt       # Python dependencies
├── pytest.ini            # Test configuration
├── README.md             # Project overview
└── SETUP.md              # This file
```

## Running the Pipeline

### Command Line Interface (TODO)

```bash
# Process a single PDF
python -m src.orchestrator.pipeline \
    --input document.pdf \
    --verbosity high \
    --depth advanced \
    --audience graduate \
    --output-dir ./output

# Process with cloud storage trigger
python -m src.orchestrator.pipeline \
    --s3-bucket my-bucket \
    --s3-key path/to/document.pdf
```

### Python API

```python
from src.orchestrator.pipeline import Orchestrator, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    verbosity="high",          # low/medium/high
    depth="advanced",          # introductory/intermediate/advanced
    audience="graduate",       # high-school/undergraduate/graduate
    base_output_dir="output",
    enable_ocr_debug=True      # Save OCR output for debugging
)

# Create orchestrator
orchestrator = Orchestrator(config=config)

# Process PDF
try:
    manifest = orchestrator.run_pipeline(
        pdf_file_path="document.pdf",
        event_source="local"
    )
    
    print(f"✅ Success! Generated {manifest.successful_scenes} scenes")
    print(f"📁 Output: {manifest.output_directory}")
    
    # Render animations with Manim
    for file in manifest.files:
        if file.type == "animation_code":
            print(f"🎬 Render: manim -pql {file.path}")
            
except Exception as e:
    print(f"❌ Pipeline failed: {e}")
```

## Rendering Generated Animations

After the pipeline generates scene files:

```bash
# Navigate to output directory
cd output/document_name/2024-01-15T10-30-00/

# Render a single scene (preview quality, low resolution)
manim -pql scene_001.py Scene_001

# Render high quality
manim -pqh scene_001.py Scene_001

# Render all scenes
for file in scene_*.py; do
    scene_class=$(basename "$file" .py | sed 's/scene_/Scene_/')
    manim -pql "$file" "$scene_class"
done
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure you're in the project root
cd path/to/pdf-to-manim-pipeline

# Install in development mode
pip install -e .
```

**2. API Rate Limits**
- The pipeline uses exponential backoff for retries
- Adjust `max_attempts` in retry policies if needed
- Consider caching LLM responses

**3. PDF Validation Failures**
- Ensure PDF is valid (not corrupted)
- Check file size (100KB - 50MB)
- Verify PDF has proper header/trailer

**4. Test Failures**
```bash
# Run tests with verbose output
pytest -vv

# Run specific failing test
pytest tests/test_determinism.py::test_name -vv

# Skip property-based tests (they can be slow)
pytest -k "not property"
```

## Development Workflow

### Running Tests During Development

```bash
# Watch mode (requires pytest-watch)
pip install pytest-watch
ptw

# Run tests on file change
pytest --looponfail

# Generate coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Adding New Features

1. Update requirements in `.kiro/specs/pdf-to-manim-pipeline/requirements.md`
2. Update design in `.kiro/specs/pdf-to-manim-pipeline/design.md`
3. Add tasks to `.kiro/specs/pdf-to-manim-pipeline/tasks.md`
4. Implement feature with tests
5. Run full test suite

## Performance Considerations

### Current Performance (Mock Mode)
- **Ingestion**: < 1 second
- **Validation**: < 1 second
- **Vision+OCR**: < 1 second (mock)
- **Script Architect**: < 1 second (mock)
- **Scene Controller**: < 1 second per scene
- **Persistence**: < 1 second

### Expected Performance (Production Mode)
- **Vision+OCR**: 30 seconds per page (API calls)
- **Script Architect**: 2 minutes for 20-page PDF
- **Animation Generator**: 10 seconds per scene
- **Total**: ~10 minutes for 10-page PDF with 20 scenes

### Optimization Tips
- Enable caching for LLM responses
- Process pages in parallel (Vision+OCR)
- Use faster models for simple content
- Batch API requests where possible

## Cost Estimates (Production Mode)

### OpenAI GPT-4 Vision
- **Vision+OCR**: ~$0.01 per page
- **Script Architect**: ~$0.05 per PDF
- **Animation Generator**: ~$0.02 per scene
- **Total**: ~$0.50 for 10-page PDF with 20 scenes

### Anthropic Claude 3 Opus
- **Similar pricing structure**
- **Slightly lower costs for text-only operations**

## Next Steps

1. ✅ **Test the mock pipeline** - Run tests to verify everything works
2. 🔑 **Get API keys** - Sign up for OpenAI or Anthropic
3. 🔧 **Update agents** - Replace mock implementations with real API calls
4. 📄 **Test with real PDFs** - Process actual academic documents
5. 🎬 **Render animations** - Use Manim to create videos
6. 🚀 **Deploy** - Set up cloud storage triggers for production

## Support

For issues or questions:
- Check the test suite for examples: `tests/`
- Review agent implementations: `src/agents/`
- Read the design document: `.kiro/specs/pdf-to-manim-pipeline/design.md`
- Check determinism documentation: `docs/determinism.md`

## License

[Add your license here]

## Contributors

[Add contributors here]
