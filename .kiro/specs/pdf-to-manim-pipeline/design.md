# Design Document: PDF to Manim Pipeline

## Overview

The PDF-to-Manim pipeline is an agent-orchestrated system that transforms academic PDF documents into executable Manim animation code with accompanying narration scripts. The system follows a linear multi-stage architecture where each stage is implemented as an autonomous agent with explicit inputs, outputs, and failure modes.

### Why Agent-Oriented Architecture

An agent-based approach is chosen over a monolithic LLM call for the following reasons:

1. **Separation of Concerns**: Each agent has a single, well-defined responsibility (OCR extraction, storyboard generation, code generation), making the system easier to test, debug, and maintain.
2. **Failure Isolation**: If one agent fails (e.g., animation code generation for a single scene), the failure is contained and does not abort the entire pipeline.
3. **Specialized Models**: Different agents can use different LLM models optimized for their task (multimodal models for vision, code-specialized models for Manim generation).
4. **Incremental Progress**: The pipeline produces intermediate artifacts (OCR output, storyboard JSON) that can be inspected, validated, and reused.
5. **Retry Granularity**: Individual agents can be retried without re-executing the entire pipeline.

### Architecture Diagram

```
┌─────────────────┐
│ Cloud Storage   │
│ Upload Trigger  │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Ingestion Agent │
└────────┬────────┘
         │
         v
┌─────────────────┐
│Validation Agent │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Vision+OCR      │
│ Agent           │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Script/         │
│ Storyboard      │
│ Architect       │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ JSON Sanitizer  │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Scene Controller│◄──┐
└────────┬────────┘   │
         │            │
         v            │
┌─────────────────┐   │
│ Animation Code  │   │
│ Generator       │───┘
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Persistence     │
│ Agent           │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ File Output     │
└─────────────────┘
```


## Agent Specifications

### 1. Ingestion Agent

**Responsibility**: Detect cloud storage upload events and route files to validation.

**Inputs**:
- Cloud storage event notification (file path, upload timestamp, metadata)

**Outputs**:
- File reference object: `{file_path: string, filename: string, size_bytes: number, upload_timestamp: ISO8601, metadata: object}`

**Logic**:
1. Subscribe to cloud storage upload events for designated bucket/prefix
2. On event trigger, extract file metadata from event payload
3. Construct file reference object with all required fields
4. Pass file reference to Validation Agent
5. Log ingestion event with timestamp and file identifier

**Failure Modes**:
- Event notification system unavailable → Log error, retry with exponential backoff (max 3 attempts)
- Malformed event payload → Log error with payload sample, skip file
- File deleted before processing → Log warning, skip file

**Retry Strategy**: Exponential backoff for transient failures (network issues, temporary service unavailability). No retry for malformed payloads or missing files.

---

### 2. Validation Agent

**Responsibility**: Verify file type, size, and structural integrity before processing.

**Inputs**:
- File reference object from Ingestion Agent

**Outputs**:
- Validated file reference (same structure as input) OR validation error object: `{error_code: string, reason: string, file_path: string}`

**Logic**:
1. Read file MIME type from file header (magic bytes: `%PDF-`)
2. Verify MIME type is `application/pdf`
3. Check file size is within bounds [100KB, 50MB]
4. Validate PDF structure by checking for `%PDF-` header and `%%EOF` trailer
5. If all checks pass, return validated file reference
6. If any check fails, return validation error object with specific failure reason

**Failure Modes**:
- File not readable (permissions, corruption) → Return error code `FILE_UNREADABLE`
- Invalid MIME type → Return error code `INVALID_MIME_TYPE`
- File size out of bounds → Return error code `INVALID_FILE_SIZE`
- Corrupted PDF structure → Return error code `CORRUPTED_PDF`

**Retry Strategy**: No retry. Validation failures are deterministic and indicate invalid input.

---

### 3. Vision + OCR Agent

**Responsibility**: Extract text, mathematical notation, and diagrams from PDF using multimodal OCR.

**Inputs**:
- Validated file reference

**Outputs**:
- OCR output object (see Data Contracts section for schema)

**Logic**:
1. Load PDF and render each page as high-resolution image (300 DPI)
2. For each page:
   a. Run OCR to extract text blocks with bounding boxes and reading order
   b. Identify mathematical notation regions using heuristics (LaTeX-like patterns, equation environments)
   c. Extract math notation as LaTeX strings using specialized math OCR model
   d. Detect diagram regions using image segmentation (contiguous non-text areas > 1000px²)
   e. Generate textual descriptions of diagrams using multimodal vision model
3. Aggregate all extracted content into structured OCR output object
4. Preserve logical reading order across multi-column layouts using spatial sorting

**Failure Modes**:
- OCR model unavailable → Retry with exponential backoff (max 3 attempts), then abort
- Math notation extraction fails for specific region → Log warning, mark region as `math_extraction_failed`, continue
- Diagram description generation fails → Log warning, use fallback description `[Diagram: visual content]`, continue
- Page rendering fails → Log error, skip page, continue with remaining pages

**Retry Strategy**: Retry transient failures (API timeouts, rate limits). For partial failures (single region), log and continue. Abort only if entire PDF fails to render.

---

### 4. Script/Storyboard Architect

**Responsibility**: Generate pedagogically sound scene sequences with narration and visual intent.

**Inputs**:
- OCR output object
- Configuration parameters: `{verbosity: string, depth: string, audience: string}`

**Outputs**:
- Storyboard JSON (see Data Contracts section for schema)

**Logic**:
1. Analyze OCR output to identify core concepts (definitions, theorems, proofs, examples)
2. Build concept dependency graph (which concepts depend on which)
3. Perform topological sort on dependency graph to determine pedagogical ordering
4. Decompose content into atomic scenes (one concept per scene, target duration 30-90 seconds)
5. For each scene:
   a. Generate narration text emphasizing intuition and visual explanation
   b. Minimize formal notation in narration (move to visual layer)
   c. Specify visual intent: mathematical objects to create, transformations to apply, emphasis points
   d. Estimate scene duration based on narration word count and transformation complexity
6. Ensure conceptual continuity: each scene builds on previous scenes
7. Apply configuration parameters:
   - Verbosity: low → shorter narration, high → more detailed explanations
   - Depth: introductory → skip proofs, advanced → include proof details
   - Audience: adjust terminology and assumed background knowledge
8. Output Storyboard JSON with ordered scene array

**Failure Modes**:
- Concept extraction fails (no clear structure) → Generate single-scene storyboard with full content, log warning
- Dependency cycle detected → Break cycle by removing weakest dependency, log warning
- LLM generates invalid JSON → Retry with explicit schema instructions (max 3 attempts), then abort
- Scene duration estimates exceed reasonable bounds → Clamp to [10s, 180s], log warning

**Retry Strategy**: Retry LLM calls with schema enforcement for JSON formatting issues. No retry for content-level issues (use fallback strategies).


---

### 5. JSON Sanitizer

**Responsibility**: Enforce schema compliance on storyboard output.

**Inputs**:
- Storyboard JSON (potentially non-compliant)

**Outputs**:
- Sanitized Storyboard JSON (schema-compliant) OR validation error report: `{errors: array, attempted_corrections: array}`

**Logic**:
1. Load JSON schema definition for Storyboard JSON
2. Validate input JSON against schema using JSON schema validator
3. If validation passes, return input unchanged
4. If validation fails:
   a. Collect all validation errors with field paths and violation types
   b. Attempt automatic corrections:
      - Missing optional fields → Add with default values
      - Type mismatches for coercible types → Coerce (e.g., number string to number)
      - Duplicate scene_ids → Append numeric suffix to duplicates
   c. Re-validate after corrections
   d. If validation now passes, return corrected JSON with correction log
   e. If validation still fails, return validation error report
5. Verify scene_id uniqueness across scene array (additional check beyond schema)

**Failure Modes**:
- Input is not valid JSON → Return error `INVALID_JSON`, no retry
- Schema validation fails after corrections → Return error `SCHEMA_VIOLATION` with details, no retry
- Required fields missing with no default → Return error `MISSING_REQUIRED_FIELD`, no retry

**Retry Strategy**: No retry. Sanitization is deterministic. Failures indicate upstream agent issues.

---

### 6. Scene Controller

**Responsibility**: Manage deterministic iteration over scene array with failure isolation.

**Inputs**:
- Sanitized Storyboard JSON

**Outputs**:
- Scene processing report: `{total_scenes: number, successful_scenes: number, failed_scenes: array, output_directory: string}`

**Logic**:
1. Extract scene array from Storyboard JSON
2. Initialize shared context object:
   ```
   {
     concept_definitions: {},
     visual_style: {colors: {}, font_size: 24, animation_speed: 1.0},
     variable_bindings: {}
   }
   ```
3. For each scene in array (deterministic order by index):
   a. Construct scene processing input: `{scene_spec: scene, context: shared_context, scene_index: number}`
   b. Pass to Animation Generator
   c. If generation succeeds:
      - Update shared context with new concept definitions from scene
      - Mark scene as successful
   d. If generation fails:
      - Log failure with scene_id and error message
      - Mark scene as failed
      - Continue to next scene (do not abort)
4. After all scenes processed, generate summary report
5. If failure rate > 50%, mark overall status as CRITICAL_FAILURE
6. Return scene processing report

**Failure Modes**:
- Animation Generator unavailable → Retry scene with exponential backoff (max 3 attempts per scene)
- Single scene fails after retries → Log failure, continue to next scene
- Majority of scenes fail (>50%) → Complete processing, return CRITICAL_FAILURE status

**Retry Strategy**: Retry individual scene generation on transient failures. Never abort entire loop due to single scene failure.

---

### 7. Animation Code Generator

**Responsibility**: Convert scene specifications into executable Manim code.

**Inputs**:
- Scene processing input: `{scene_spec: object, context: object, scene_index: number}`

**Outputs**:
- Animation code object: `{scene_id: string, code: string, imports: array, class_name: string}` OR generation error: `{scene_id: string, error: string}`

**Logic**:
1. Parse scene specification:
   - Extract narration text
   - Extract visual intent (mathematical_objects, transformations, emphasis_points)
   - Extract duration estimate
2. Parse shared context for relevant definitions and visual style
3. Generate Manim code:
   a. Create class definition inheriting from `manim.Scene`
   b. Generate `construct` method
   c. For each mathematical object in visual_intent:
      - Generate Manim object creation code (MathTex, Axes, Circle, etc.)
      - Apply visual style from context (colors, font size)
   d. For each transformation in visual_intent:
      - Generate Manim animation code (Transform, FadeIn, FadeOut, etc.)
      - Sequence transformations with appropriate timing
   e. Add narration as comments or separate audio timing markers
4. Validate generated code:
   - Check Python syntax using AST parser
   - Verify Manim imports are correct
   - Verify class structure (inherits Scene, has construct method)
5. Format code with consistent indentation (4 spaces)
6. Add header comment with scene_id, narration, timestamp
7. Return animation code object

**Failure Modes**:
- LLM generates syntactically invalid Python → Retry with syntax error feedback (max 3 attempts)
- LLM generates code with undefined Manim objects → Retry with API reference (max 3 attempts)
- Visual intent is ambiguous or contradictory → Log warning, generate simplified version
- LaTeX rendering syntax errors → Retry with corrected LaTeX (max 2 attempts)
- All retries exhausted → Return generation error object

**Retry Strategy**: Retry with error feedback for syntax/API issues. Use simplified fallback for ambiguous intent. Return error after max retries.

---

### 8. Persistence Agent

**Responsibility**: Write generated artifacts to deterministic file structure.

**Inputs**:
- Scene processing report
- Array of animation code objects
- Storyboard JSON
- OCR output object (optional, for debugging)

**Outputs**:
- Output manifest: `{output_directory: string, files: array, timestamp: ISO8601, status: string}`

**Logic**:
1. Determine output directory path: `output/{pdf_filename}/{timestamp}/`
2. Create directory structure if not exists
3. For each animation code object:
   a. Write code to file: `scene_{scene_id}.py`
   b. Use atomic write (write to temp file, then rename)
4. Extract narration from all scenes and write to `narration_script.txt` with scene markers
5. Write Storyboard JSON to `storyboard.json`
6. Write scene processing report to `processing_report.json`
7. Optionally write OCR output to `ocr_output.json` (for debugging)
8. Generate manifest file listing all created files with metadata:
   ```
   {
     output_directory: string,
     files: [{path: string, size_bytes: number, type: string}],
     timestamp: ISO8601,
     status: "SUCCESS" | "PARTIAL_SUCCESS" | "FAILURE",
     scene_success_rate: number
   }
   ```
9. Write manifest to `manifest.json`
10. Return output manifest

**Failure Modes**:
- Directory creation fails (permissions) → Abort, return error `DIRECTORY_CREATION_FAILED`
- File write fails (disk full, permissions) → Abort, return error `FILE_WRITE_FAILED`
- Partial write due to crash → On retry, clean up incomplete files before writing

**Retry Strategy**: Retry transient failures (temporary disk issues). On retry, clean up output directory first to ensure idempotency.


## Data Contracts

### OCR Output Schema

```json
{
  "pdf_metadata": {
    "filename": "string",
    "page_count": "number",
    "file_size_bytes": "number"
  },
  "pages": [
    {
      "page_number": "number",
      "text_blocks": [
        {
          "text": "string",
          "bounding_box": {"x": "number", "y": "number", "width": "number", "height": "number"},
          "reading_order": "number",
          "confidence": "number (0-1)"
        }
      ],
      "math_expressions": [
        {
          "latex": "string",
          "bounding_box": {"x": "number", "y": "number", "width": "number", "height": "number"},
          "context": "string (surrounding text)",
          "confidence": "number (0-1)",
          "extraction_failed": "boolean (optional)"
        }
      ],
      "diagrams": [
        {
          "diagram_id": "string",
          "bounding_box": {"x": "number", "y": "number", "width": "number", "height": "number"},
          "diagram_type": "string (graph|geometric|flowchart|plot|unknown)",
          "description": "string",
          "visual_elements": ["string (array of detected elements)"],
          "confidence": "number (0-1)"
        }
      ]
    }
  ]
}
```

**Required Fields**: `pdf_metadata`, `pages`, `page_number`, `text_blocks`, `text`, `bounding_box`

**Optional Fields**: `math_expressions`, `diagrams`, `confidence`, `extraction_failed`, `visual_elements`

---

### Storyboard JSON Schema

```json
{
  "pdf_metadata": {
    "filename": "string",
    "page_count": "number"
  },
  "configuration": {
    "verbosity": "string (low|medium|high)",
    "depth": "string (introductory|intermediate|advanced)",
    "audience": "string (high-school|undergraduate|graduate)"
  },
  "concept_hierarchy": [
    {
      "concept_id": "string",
      "concept_name": "string",
      "dependencies": ["string (array of concept_ids)"]
    }
  ],
  "scenes": [
    {
      "scene_id": "string (unique)",
      "scene_index": "number",
      "concept_id": "string (reference to concept_hierarchy)",
      "narration": "string",
      "visual_intent": {
        "mathematical_objects": [
          {
            "object_type": "string (equation|graph|geometric_shape|axes|text)",
            "content": "string (LaTeX for equations, description for others)",
            "style": {
              "color": "string (optional)",
              "position": "string (optional, e.g., 'center', 'top-left')"
            }
          }
        ],
        "transformations": [
          {
            "transformation_type": "string (morph|highlight|move|fade_in|fade_out|scale)",
            "target_object": "string (reference to mathematical_objects)",
            "parameters": "object (transformation-specific params)",
            "timing": "number (seconds from scene start)"
          }
        ],
        "emphasis_points": [
          {
            "timestamp": "number (seconds from scene start)",
            "description": "string (what to emphasize)"
          }
        ]
      },
      "duration_estimate": "number (seconds)",
      "dependencies": ["string (array of scene_ids, optional)"],
      "difficulty_level": "string (optional)"
    }
  ]
}
```

**Required Fields**: `pdf_metadata`, `configuration`, `scenes`, `scene_id`, `scene_index`, `narration`, `visual_intent`, `mathematical_objects`, `transformations`, `duration_estimate`

**Optional Fields**: `concept_hierarchy`, `dependencies`, `difficulty_level`, `emphasis_points`, `style`, `position`

---

### Animation Code Object Schema

```json
{
  "scene_id": "string",
  "code": "string (Python code)",
  "imports": ["string (array of import statements)"],
  "class_name": "string (e.g., 'Scene_001')",
  "narration": "string (copied from scene spec)",
  "duration_estimate": "number (seconds)",
  "generation_timestamp": "ISO8601"
}
```

**Required Fields**: All fields are required.

---

### Scene Processing Input Schema

```json
{
  "scene_spec": {
    "scene_id": "string",
    "narration": "string",
    "visual_intent": "object (see Storyboard JSON schema)",
    "duration_estimate": "number"
  },
  "context": {
    "concept_definitions": {
      "concept_id": "string (definition text)"
    },
    "visual_style": {
      "colors": {
        "primary": "string (hex color)",
        "secondary": "string (hex color)",
        "background": "string (hex color)"
      },
      "font_size": "number",
      "animation_speed": "number (multiplier, default 1.0)"
    },
    "variable_bindings": {
      "variable_name": "string (value or reference)"
    }
  },
  "scene_index": "number"
}
```

**Required Fields**: `scene_spec`, `context`, `scene_index`, `scene_id`, `narration`, `visual_intent`

**Optional Fields**: All fields in `context` are optional (may be empty objects).

---

### Output Manifest Schema

```json
{
  "output_directory": "string (absolute path)",
  "files": [
    {
      "path": "string (relative to output_directory)",
      "size_bytes": "number",
      "type": "string (animation_code|narration_script|storyboard|manifest|report)"
    }
  ],
  "timestamp": "ISO8601",
  "status": "string (SUCCESS|PARTIAL_SUCCESS|FAILURE)",
  "scene_success_rate": "number (0-1)",
  "total_scenes": "number",
  "successful_scenes": "number",
  "failed_scenes": [
    {
      "scene_id": "string",
      "error": "string"
    }
  ]
}
```

**Required Fields**: All fields are required.


## Control Flow and Orchestration

### Execution Sequence

The pipeline executes as a linear sequence of agent invocations with explicit handoffs:

1. **Ingestion Agent** → produces file reference
2. **Validation Agent** → consumes file reference, produces validated file reference OR error
3. **IF validation error**: Abort pipeline, log error, return failure status
4. **Vision+OCR Agent** → consumes validated file reference, produces OCR output
5. **Script/Storyboard Architect** → consumes OCR output + configuration, produces Storyboard JSON
6. **JSON Sanitizer** → consumes Storyboard JSON, produces sanitized Storyboard JSON OR error
7. **IF sanitization error**: Abort pipeline, log error, return failure status
8. **Scene Controller** → consumes sanitized Storyboard JSON, orchestrates scene loop
9. **Scene Loop** (within Scene Controller):
   - FOR each scene in scenes array (index 0 to N-1):
     - Construct scene processing input (scene spec + shared context)
     - Invoke **Animation Code Generator**
     - IF generation succeeds: update shared context, collect animation code object
     - IF generation fails: log failure, mark scene as failed, continue to next scene
   - After loop completes: generate scene processing report
10. **Persistence Agent** → consumes scene processing report + animation code objects + Storyboard JSON, writes files
11. **Return output manifest** to caller

### Loop Semantics

The scene loop (step 9) has the following semantics:

- **Deterministic iteration**: Scenes are processed in array order (index 0, 1, 2, ..., N-1)
- **Independent scenes**: Each scene generation is independent (no shared mutable state between scenes)
- **Shared context**: Read-only context is passed to each scene, updated after successful generation
- **Failure isolation**: Scene generation failure does not abort the loop
- **No early termination**: Loop always processes all scenes, even if some fail

### Context Propagation

Shared context is propagated as follows:

1. **Initialization**: Scene Controller initializes empty context before loop
2. **Read-only during generation**: Animation Generator receives context as read-only input
3. **Update after success**: Scene Controller updates context with new definitions from successful scene
4. **No update on failure**: Failed scenes do not modify context
5. **Cumulative**: Context accumulates definitions across all successful scenes

Context update logic:
```python
# After successful scene generation
if scene_spec.concept_id in concept_hierarchy:
    context.concept_definitions[scene_spec.concept_id] = extract_definition(scene_spec)

# Extract variable bindings from visual_intent
for obj in scene_spec.visual_intent.mathematical_objects:
    if obj.object_type == "equation" and is_definition(obj.content):
        var_name = extract_variable_name(obj.content)
        context.variable_bindings[var_name] = obj.content
```

### Abort vs Retry Logic

**Abort conditions** (pipeline terminates immediately):
- Validation Agent returns error (invalid input)
- JSON Sanitizer returns error after correction attempts (malformed storyboard)
- Persistence Agent fails to create output directory (infrastructure failure)

**Retry conditions** (agent retries operation):
- Vision+OCR Agent: transient API failures (network timeout, rate limit)
- Script/Storyboard Architect: LLM returns invalid JSON (retry with schema enforcement)
- Animation Code Generator: generated code has syntax errors (retry with error feedback)
- Scene Controller: Animation Generator unavailable (retry individual scene)

**Continue conditions** (pipeline continues despite failure):
- Vision+OCR Agent: single math expression extraction fails (log warning, continue)
- Scene Controller: single scene generation fails after retries (log failure, continue to next scene)

### Idempotency Guarantees

The pipeline is idempotent with respect to file output:

1. **Deterministic output directory**: Same PDF + timestamp → same output directory path
2. **Atomic writes**: Persistence Agent uses atomic file operations (write to temp, then rename)
3. **Cleanup on retry**: If output directory exists, Persistence Agent removes incomplete files before writing
4. **No append operations**: All file writes are full overwrites, not appends

Idempotency does NOT extend to LLM-generated content (storyboard, animation code) due to inherent LLM variability. However, using temperature=0 minimizes variability.


## Components and Interfaces

### Agent Interface Contract

All agents implement a common interface pattern:

```python
class Agent:
    def execute(self, input: AgentInput) -> AgentOutput:
        """
        Execute agent logic.
        
        Args:
            input: Agent-specific input object
            
        Returns:
            Agent-specific output object OR error object
            
        Raises:
            AgentExecutionError: For unrecoverable failures
        """
        pass
    
    def validate_input(self, input: AgentInput) -> bool:
        """Validate input conforms to expected schema."""
        pass
    
    def get_retry_policy(self) -> RetryPolicy:
        """Return retry policy for this agent."""
        pass
```

### Orchestrator Component

The orchestrator is responsible for:
- Sequencing agent invocations
- Passing outputs from one agent as inputs to the next
- Handling abort conditions
- Implementing retry logic based on agent retry policies
- Logging execution trace

Orchestrator pseudocode:
```python
def run_pipeline(pdf_file_path: str, config: Config) -> OutputManifest:
    # Step 1: Ingestion
    file_ref = ingestion_agent.execute(pdf_file_path)
    
    # Step 2: Validation
    validated_ref = validation_agent.execute(file_ref)
    if isinstance(validated_ref, ValidationError):
        abort_pipeline(validated_ref)
    
    # Step 3: Vision+OCR
    ocr_output = retry_with_policy(
        lambda: vision_ocr_agent.execute(validated_ref),
        vision_ocr_agent.get_retry_policy()
    )
    
    # Step 4: Script/Storyboard
    storyboard = retry_with_policy(
        lambda: script_architect.execute(ocr_output, config),
        script_architect.get_retry_policy()
    )
    
    # Step 5: JSON Sanitization
    sanitized_storyboard = json_sanitizer.execute(storyboard)
    if isinstance(sanitized_storyboard, SanitizationError):
        abort_pipeline(sanitized_storyboard)
    
    # Step 6: Scene Processing
    scene_report = scene_controller.execute(sanitized_storyboard)
    
    # Step 7: Persistence
    manifest = persistence_agent.execute(scene_report)
    
    return manifest
```

### Logging and Observability

Each agent logs the following events:

**Execution Start**:
```json
{
  "event": "agent_start",
  "agent_name": "string",
  "timestamp": "ISO8601",
  "input_summary": "string (truncated input description)"
}
```

**Execution Complete**:
```json
{
  "event": "agent_complete",
  "agent_name": "string",
  "timestamp": "ISO8601",
  "duration_ms": "number",
  "output_summary": "string (truncated output description)",
  "status": "SUCCESS|FAILURE"
}
```

**Execution Failure**:
```json
{
  "event": "agent_failure",
  "agent_name": "string",
  "timestamp": "ISO8601",
  "error_message": "string",
  "error_code": "string",
  "input_context": "string (relevant input excerpt)",
  "retry_attempt": "number (0 for first attempt)"
}
```

All logs are written to `pipeline.log` in the output directory using structured JSON format (one JSON object per line).


## Technology Assumptions

### LLM Role Separation

Different agents use different LLM models optimized for their tasks:

- **Vision+OCR Agent**: Multimodal model with vision capabilities (e.g., GPT-4 Vision, Claude 3 Opus)
  - Required capabilities: image understanding, OCR, diagram interpretation
  - Context window: 128K+ tokens (to handle full PDF content)

- **Script/Storyboard Architect**: Large context model with strong reasoning (e.g., GPT-4, Claude 3 Opus)
  - Required capabilities: long-form reasoning, structured output, pedagogical understanding
  - Context window: 128K+ tokens
  - Temperature: 0 for determinism

- **Animation Code Generator**: Code-specialized model (e.g., GPT-4, Claude 3.5 Sonnet, Codex)
  - Required capabilities: Python code generation, API knowledge (Manim), syntax validation
  - Context window: 32K+ tokens (sufficient for single scene)
  - Temperature: 0 for determinism

- **JSON Sanitizer**: No LLM required (rule-based validation and correction)

### Animation Engine

- **Manim Community Edition (v0.17+)**: Open-source mathematical animation engine
  - Python 3.8+ required
  - LaTeX installation required for MathTex rendering
  - FFmpeg required for video rendering
  - Key objects used: Scene, MathTex, Axes, Circle, Line, Arrow, Graph
  - Key animations used: Transform, FadeIn, FadeOut, Write, Create, MoveToTarget

### Storage

- **Cloud Storage**: AWS S3, Google Cloud Storage, or Azure Blob Storage
  - Upload event notifications (S3 Event Notifications, Cloud Storage Pub/Sub, Event Grid)
  - File access via SDK (boto3, google-cloud-storage, azure-storage-blob)

- **Output Storage**: Local filesystem or cloud storage
  - Deterministic directory structure: `output/{pdf_filename}/{timestamp}/`
  - File types: `.py` (animation code), `.txt` (narration), `.json` (metadata)

### OCR and Vision

- **Multimodal OCR**: Cloud-based vision APIs or self-hosted models
  - Text extraction: Tesseract OCR, Google Cloud Vision, AWS Textract
  - Math notation: Mathpix, specialized math OCR models
  - Diagram understanding: Multimodal LLMs (GPT-4 Vision, Claude 3)

### Execution Environment

- **Containerized**: Docker container with all dependencies (Python, Manim, LaTeX, FFmpeg)
- **Orchestration**: Kubernetes, AWS ECS, or serverless functions (AWS Lambda, Google Cloud Functions)
- **Concurrency**: Support for 10 concurrent pipeline executions
- **Resource limits**: 4 CPU cores, 8GB RAM per pipeline execution

