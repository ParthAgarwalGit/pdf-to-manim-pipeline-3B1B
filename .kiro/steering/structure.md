# Project Structure

## Directory Organization

```
.
├── .kiro/
│   ├── specs/
│   │   └── pdf-to-manim-pipeline/    # Feature specification
│   │       ├── requirements.md        # User stories & acceptance criteria
│   │       ├── design.md              # Architecture & agent specifications
│   │       └── tasks.md               # Implementation task breakdown
│   └── steering/                      # AI assistant guidance documents
│       ├── product.md
│       ├── tech.md
│       └── structure.md
│
├── src/                               # Source code (to be created)
│   ├── agents/                        # Agent implementations
│   │   ├── __init__.py
│   │   ├── base.py                    # Base Agent interface
│   │   ├── ingestion.py               # Ingestion Agent
│   │   ├── validation.py              # Validation Agent
│   │   ├── vision_ocr.py              # Vision + OCR Agent
│   │   ├── script_architect.py        # Script/Storyboard Architect
│   │   ├── json_sanitizer.py          # JSON Sanitizer
│   │   ├── scene_controller.py        # Scene Controller
│   │   ├── animation_generator.py     # Animation Code Generator
│   │   └── persistence.py             # Persistence Agent
│   │
│   ├── schemas/                       # Pydantic data models
│   │   ├── __init__.py
│   │   ├── file_reference.py
│   │   ├── ocr_output.py
│   │   ├── storyboard.py
│   │   ├── animation_code.py
│   │   ├── scene_processing.py
│   │   └── output_manifest.py
│   │
│   ├── orchestrator/                  # Pipeline orchestration
│   │   ├── __init__.py
│   │   ├── pipeline.py                # Main orchestrator
│   │   ├── retry_policy.py            # Retry logic
│   │   └── logger.py                  # Structured logging
│   │
│   └── utils/                         # Shared utilities
│       ├── __init__.py
│       └── helpers.py
│
├── tests/                             # Test suite (to be created)
│   ├── __init__.py
│   ├── test_ingestion_agent.py
│   ├── test_validation_agent.py
│   ├── test_vision_ocr_agent.py
│   ├── test_script_architect.py
│   ├── test_json_sanitizer.py
│   ├── test_scene_controller.py
│   ├── test_animation_generator.py
│   ├── test_persistence_agent.py
│   ├── test_orchestrator.py
│   ├── test_determinism.py            # Property tests for determinism
│   ├── test_idempotency.py            # Property tests for idempotency
│   └── integration/
│       ├── __init__.py
│       └── test_end_to_end.py
│
├── output/                            # Generated artifacts (runtime)
│   └── {pdf_filename}/
│       └── {timestamp}/
│           ├── scene_*.py             # Generated Manim code
│           ├── narration_script.txt   # Narration with scene markers
│           ├── storyboard.json        # Scene specifications
│           ├── processing_report.json # Success/failure summary
│           ├── manifest.json          # File listing
│           └── pipeline.log           # Structured execution logs
│
├── requirements.txt                   # Python dependencies (to be created)
├── pytest.ini                         # Pytest configuration (to be created)
└── README.md                          # Project documentation (to be created)
```

## Architecture Patterns

### Agent-Oriented Design
- Each agent is a self-contained module with single responsibility
- Agents implement common interface: `execute()`, `validate_input()`, `get_retry_policy()`
- Linear pipeline flow: Ingestion → Validation → Vision+OCR → Script Architect → JSON Sanitizer → Scene Controller → Animation Generator → Persistence

### Data Flow
- Explicit input/output contracts using Pydantic schemas
- Immutable data passing between agents (no shared mutable state)
- Structured JSON for intermediate artifacts (OCR output, storyboard)

### Error Handling
- **Abort**: Validation/sanitization failures terminate pipeline immediately
- **Retry**: Transient failures (API timeouts) use exponential backoff
- **Continue**: Scene-level failures are isolated, pipeline continues

### Determinism & Idempotency
- Temperature=0 for all LLM calls
- Deterministic ordering for all array operations
- Atomic file writes with cleanup on retry
- Same input → structurally equivalent output

## File Naming Conventions

### Source Code
- Snake_case for Python files: `vision_ocr.py`, `scene_controller.py`
- PascalCase for classes: `VisionOCRAgent`, `SceneController`
- Descriptive names matching agent responsibilities

### Generated Outputs
- Scene files: `scene_{scene_id}.py` (e.g., `scene_001.py`, `scene_intro.py`)
- Fixed names: `narration_script.txt`, `storyboard.json`, `manifest.json`, `pipeline.log`
- Output directory: `output/{pdf_filename}/{ISO8601_timestamp}/`

### Tests
- Test files: `test_{module_name}.py` (e.g., `test_validation_agent.py`)
- Property tests: Include "property" in test name for filtering
- Integration tests: Separate `integration/` subdirectory

## Key Design Principles

1. **Separation of Concerns**: Each agent has one well-defined responsibility
2. **Failure Isolation**: Agent failures don't cascade to entire pipeline
3. **Incremental Progress**: Intermediate artifacts can be inspected and reused
4. **Testability**: Property-based tests for universal properties, unit tests for specifics
5. **Observability**: Structured logging at every stage with timestamps and context
