# Requirements Document

## Introduction

This specification defines an agent-orchestrated pipeline that converts academic PDF documents into 3Blue1Brown-style animated educational videos. The system transforms static academic content (text, equations, diagrams, proofs) into executable Manim animation code with accompanying narration scripts, prioritizing visual intuition and conceptual continuity over textual density.

The target output style emphasizes animated mathematical objects, minimal on-screen text, and pedagogical clarity through visual transformation rather than static explanation.

## Glossary

- **Pipeline**: The complete agent-orchestrated workflow from PDF ingestion to animation code output
- **Vision_OCR_Agent**: Agent responsible for extracting text, mathematical notation, and diagrams from PDF using multimodal OCR
- **Script_Architect**: Agent that generates scene-based storyboards with narration and visual intent
- **Scene_Controller**: Agent that manages deterministic iteration over scene array
- **Animation_Generator**: Agent that converts scene specifications into executable Manim code
- **Manim**: Mathematical Animation Engine (Python library for programmatic animations)
- **Scene**: Atomic unit of animation with single conceptual focus, narration, and visual elements
- **Storyboard_JSON**: Structured JSON output containing ordered scene specifications
- **Ingestion_Agent**: Agent that handles cloud storage upload triggers and file routing
- **Validation_Agent**: Agent that verifies PDF file type and structural integrity
- **JSON_Sanitizer**: Agent that enforces schema compliance on storyboard output
- **Persistence_Agent**: Agent that writes scripts and animation code to deterministic file structure

## Requirements

### Requirement 1: PDF Ingestion

**User Story:** As a system operator, I want the pipeline to accept PDF uploads via cloud storage triggers, so that processing begins automatically without manual intervention.

#### Acceptance Criteria

1. WHEN a PDF file is uploaded to the designated cloud storage location, THE Ingestion_Agent SHALL detect the upload event within 5 seconds
2. WHEN the upload event is detected, THE Ingestion_Agent SHALL extract file metadata (filename, size, upload timestamp, path)
3. WHEN metadata extraction completes, THE Ingestion_Agent SHALL pass the file reference and metadata to the Validation_Agent
4. IF the upload event fails to trigger, THEN THE Ingestion_Agent SHALL log the failure with timestamp and file identifier

### Requirement 2: Workflow Configuration

**User Story:** As a content creator, I want to specify pedagogical parameters (verbosity, depth, audience level), so that the output matches my target audience and teaching style.

#### Acceptance Criteria

1. THE Pipeline SHALL accept configuration parameters for verbosity (low/medium/high), depth (introductory/intermediate/advanced), and audience (high-school/undergraduate/graduate)
2. WHEN configuration is provided via metadata, THE Pipeline SHALL parse and validate all parameters before processing begins
3. WHEN configuration is missing, THE Pipeline SHALL use default values (verbosity: medium, depth: intermediate, audience: undergraduate)
4. THE Pipeline SHALL propagate configuration context to Script_Architect and Animation_Generator agents

### Requirement 3: File Validation

**User Story:** As a system operator, I want invalid files rejected early, so that processing resources are not wasted on incompatible inputs.

#### Acceptance Criteria

1. WHEN a file is received, THE Validation_Agent SHALL verify the file has PDF MIME type (application/pdf)
2. WHEN a file is received, THE Validation_Agent SHALL verify the file size is between 100KB and 50MB
3. WHEN a file is received, THE Validation_Agent SHALL verify the PDF structure is not corrupted using header validation
4. IF validation fails, THEN THE Validation_Agent SHALL abort the pipeline and return an error code with failure reason
5. IF validation succeeds, THEN THE Validation_Agent SHALL pass the validated file reference to Vision_OCR_Agent

### Requirement 4: Vision and OCR Extraction

**User Story:** As a pipeline developer, I want accurate extraction of text, mathematical notation, and diagrams from PDFs, so that downstream agents have complete source material.

#### Acceptance Criteria

1. WHEN a validated PDF is received, THE Vision_OCR_Agent SHALL extract all text content with page number annotations
2. WHEN a validated PDF is received, THE Vision_OCR_Agent SHALL extract mathematical notation in LaTeX format with positional context
3. WHEN a validated PDF is received, THE Vision_OCR_Agent SHALL identify and extract diagram regions with bounding box coordinates
4. WHEN a validated PDF is received, THE Vision_OCR_Agent SHALL generate textual descriptions of diagrams using multimodal vision models
5. THE Vision_OCR_Agent SHALL output structured JSON containing text blocks, math expressions, and diagram metadata
6. THE Vision_OCR_Agent SHALL preserve logical reading order across multi-column layouts

### Requirement 5: Script and Storyboard Generation

**User Story:** As a content creator, I want the system to generate pedagogically sound scene sequences with narration and visual intent, so that the output follows 3Blue1Brown teaching principles.

#### Acceptance Criteria

1. WHEN OCR output is received, THE Script_Architect SHALL analyze content to identify core concepts and their dependencies
2. WHEN concepts are identified, THE Script_Architect SHALL generate a concept hierarchy ordered by pedagogical progression
3. WHEN the concept hierarchy is complete, THE Script_Architect SHALL decompose content into atomic scenes with single conceptual focus
4. FOR EACH scene, THE Script_Architect SHALL generate narration text that prioritizes intuition over formalism
5. FOR EACH scene, THE Script_Architect SHALL specify visual intent (what mathematical objects to animate, transformation sequences, emphasis points)
6. THE Script_Architect SHALL output Storyboard_JSON conforming to the defined schema with ordered scene array
7. THE Script_Architect SHALL minimize on-screen text density in visual specifications
8. THE Script_Architect SHALL ensure conceptual continuity between consecutive scenes

### Requirement 6: JSON Schema Enforcement

**User Story:** As a pipeline developer, I want storyboard output to conform strictly to the defined schema, so that downstream agents can parse it reliably.

#### Acceptance Criteria

1. WHEN Storyboard_JSON is generated, THE JSON_Sanitizer SHALL validate the output against the defined schema
2. IF schema validation fails, THEN THE JSON_Sanitizer SHALL identify all violations with field paths and expected types
3. IF schema validation fails, THEN THE JSON_Sanitizer SHALL attempt automatic correction for common violations (missing optional fields, type coercion)
4. IF automatic correction succeeds, THEN THE JSON_Sanitizer SHALL log corrections and pass sanitized JSON to Scene_Controller
5. IF automatic correction fails, THEN THE JSON_Sanitizer SHALL abort the pipeline and return detailed error report
6. THE JSON_Sanitizer SHALL verify all required fields are present (scene_id, narration, visual_intent, duration_estimate)
7. THE JSON_Sanitizer SHALL verify scene_id uniqueness across the scene array

### Requirement 7: Scene Iteration Control

**User Story:** As a pipeline developer, I want deterministic iteration over scenes with failure isolation, so that one broken scene does not abort the entire pipeline.

#### Acceptance Criteria

1. WHEN sanitized Storyboard_JSON is received, THE Scene_Controller SHALL extract the scene array
2. THE Scene_Controller SHALL iterate over scenes in array order with deterministic indexing
3. FOR EACH scene, THE Scene_Controller SHALL pass scene specification and shared context to Animation_Generator
4. THE Scene_Controller SHALL maintain shared context (concept definitions, variable bindings, visual style) across scene iterations
5. IF a scene fails to generate valid animation code, THEN THE Scene_Controller SHALL log the failure and continue to the next scene
6. THE Scene_Controller SHALL track success/failure status for each scene with error messages
7. WHEN all scenes are processed, THE Scene_Controller SHALL generate a summary report with success rate and failure details

### Requirement 8: Animation Code Generation

**User Story:** As a content creator, I want executable Manim code generated for each scene, so that I can render animations without manual coding.

#### Acceptance Criteria

1. WHEN a scene specification is received, THE Animation_Generator SHALL parse narration text and visual intent
2. WHEN visual intent is parsed, THE Animation_Generator SHALL generate Manim code that creates specified mathematical objects (equations, graphs, geometric shapes)
3. WHEN visual intent is parsed, THE Animation_Generator SHALL generate Manim code for transformation sequences (morphing, highlighting, movement)
4. THE Animation_Generator SHALL generate Manim code that synchronizes visual transformations with narration timing
5. THE Animation_Generator SHALL output syntactically valid Python code using Manim library API
6. THE Animation_Generator SHALL include scene class definition with proper inheritance from Manim Scene class
7. THE Animation_Generator SHALL handle LaTeX rendering for mathematical notation using Manim's MathTex objects
8. IF code generation fails, THEN THE Animation_Generator SHALL return an error with the scene_id and failure reason

### Requirement 9: Persistence and File Output

**User Story:** As a system operator, I want all generated artifacts written to a deterministic file structure, so that outputs are easily retrievable and organized.

#### Acceptance Criteria

1. WHEN animation code is generated for a scene, THE Persistence_Agent SHALL write the code to a file named `scene_{scene_id}.py`
2. WHEN all scenes are processed, THE Persistence_Agent SHALL write the complete narration script to `narration_script.txt` with scene markers
3. WHEN all scenes are processed, THE Persistence_Agent SHALL write the Storyboard_JSON to `storyboard.json`
4. THE Persistence_Agent SHALL create a directory structure: `output/{pdf_filename}/{timestamp}/`
5. THE Persistence_Agent SHALL write a manifest file `manifest.json` listing all generated files with metadata
6. THE Persistence_Agent SHALL ensure file writes are atomic (no partial writes on failure)
7. THE Persistence_Agent SHALL return the output directory path upon successful completion

### Requirement 10: Determinism and Reproducibility

**User Story:** As a pipeline developer, I want identical inputs to produce identical outputs, so that the system behavior is predictable and debuggable.

#### Acceptance Criteria

1. WHEN the same PDF and configuration are processed multiple times, THE Pipeline SHALL produce structurally equivalent Storyboard_JSON (same scene count, same scene order, equivalent narration content)
2. THE Pipeline SHALL use deterministic ordering for all array operations (scene iteration, concept hierarchy)
3. THE Pipeline SHALL avoid non-deterministic operations (random sampling, timestamp-based ordering) in core logic
4. WHERE LLM-based agents introduce variability, THE Pipeline SHALL use temperature=0 or equivalent deterministic sampling

### Requirement 11: Idempotency

**User Story:** As a system operator, I want to safely retry failed pipeline runs, so that transient failures do not require manual cleanup.

#### Acceptance Criteria

1. WHEN a pipeline run is retried with the same input, THE Pipeline SHALL overwrite previous output in the same directory
2. THE Pipeline SHALL not create duplicate files or append to existing files on retry
3. IF a partial run exists in the output directory, THEN THE Pipeline SHALL clean up incomplete artifacts before starting
4. THE Persistence_Agent SHALL use atomic file operations to prevent corruption during retries

### Requirement 12: Observability

**User Story:** As a system operator, I want detailed logging at each pipeline stage, so that I can diagnose failures and monitor performance.

#### Acceptance Criteria

1. WHEN each agent begins execution, THE Pipeline SHALL log the agent name, input summary, and timestamp
2. WHEN each agent completes execution, THE Pipeline SHALL log the agent name, output summary, duration, and timestamp
3. IF an agent fails, THEN THE Pipeline SHALL log the agent name, error message, input context, and timestamp
4. THE Pipeline SHALL log all schema validation failures with field paths and violation details
5. THE Pipeline SHALL log scene-level success/failure status during iteration
6. THE Pipeline SHALL write all logs to a structured log file `pipeline.log` in the output directory

### Requirement 13: Latency Constraints

**User Story:** As a content creator, I want reasonable processing times per pipeline stage, so that I can iterate on content efficiently.

#### Acceptance Criteria

1. THE Vision_OCR_Agent SHALL complete extraction within 30 seconds per page for PDFs up to 20 pages
2. THE Script_Architect SHALL complete storyboard generation within 2 minutes for PDFs up to 20 pages
3. THE Animation_Generator SHALL complete code generation within 10 seconds per scene
4. THE Pipeline SHALL complete end-to-end processing within 10 minutes for a 10-page academic PDF with 20 scenes

### Requirement 14: Error Handling and Failure Modes

**User Story:** As a pipeline developer, I want explicit handling of known failure modes, so that the system degrades gracefully.

#### Acceptance Criteria

1. IF Vision_OCR_Agent fails to extract mathematical notation, THEN THE Pipeline SHALL log the failure and continue with text-only extraction
2. IF Script_Architect generates invalid JSON, THEN THE JSON_Sanitizer SHALL attempt correction up to 3 times before aborting
3. IF Animation_Generator produces syntactically invalid Manim code, THEN THE Scene_Controller SHALL log the failure and mark the scene as failed
4. IF more than 50% of scenes fail code generation, THEN THE Pipeline SHALL abort and return a critical failure status
5. THE Pipeline SHALL distinguish between retryable errors (transient API failures) and non-retryable errors (invalid input format)

### Requirement 15: Mathematical Fidelity

**User Story:** As a content creator, I want mathematical notation preserved accurately from PDF to animation code, so that the output is mathematically correct.

#### Acceptance Criteria

1. WHEN LaTeX expressions are extracted, THE Vision_OCR_Agent SHALL preserve operator precedence and grouping
2. WHEN LaTeX expressions are extracted, THE Vision_OCR_Agent SHALL distinguish between similar symbols (e.g., × vs x, ∈ vs ε)
3. WHEN animation code is generated, THE Animation_Generator SHALL render LaTeX expressions using Manim's MathTex with correct syntax
4. IF mathematical notation is ambiguous, THEN THE Vision_OCR_Agent SHALL flag the ambiguity in the output metadata

### Requirement 16: Diagram Interpretation

**User Story:** As a content creator, I want diagrams from PDFs translated into animation instructions, so that visual explanations are preserved.

#### Acceptance Criteria

1. WHEN a diagram is detected, THE Vision_OCR_Agent SHALL classify the diagram type (graph, geometric figure, flowchart, plot)
2. WHEN a diagram is classified, THE Vision_OCR_Agent SHALL extract key visual elements (axes, labels, shapes, arrows)
3. WHEN diagram metadata is passed to Script_Architect, THE Script_Architect SHALL generate visual intent that recreates the diagram using Manim primitives
4. THE Animation_Generator SHALL translate diagram visual intent into Manim code using appropriate objects (Axes, Line, Circle, Arrow, etc.)

### Requirement 17: Scalability

**User Story:** As a system operator, I want the pipeline to handle concurrent PDF processing, so that multiple users can submit jobs simultaneously.

#### Acceptance Criteria

1. THE Pipeline SHALL support processing up to 10 concurrent PDF jobs without resource contention
2. THE Pipeline SHALL isolate execution contexts for concurrent jobs (separate output directories, no shared mutable state)
3. THE Pipeline SHALL queue jobs beyond concurrency limit with FIFO ordering
4. THE Pipeline SHALL provide job status API for querying progress of concurrent jobs

### Requirement 18: Storyboard JSON Schema Definition

**User Story:** As a pipeline developer, I want an explicit JSON schema for storyboard output, so that all agents agree on data contracts.

#### Acceptance Criteria

1. THE Pipeline SHALL define a JSON schema with the following required top-level fields: `pdf_metadata`, `configuration`, `scenes`
2. THE `scenes` field SHALL be an array of scene objects
3. EACH scene object SHALL have required fields: `scene_id` (string), `narration` (string), `visual_intent` (object), `duration_estimate` (number in seconds)
4. THE `visual_intent` object SHALL have required fields: `mathematical_objects` (array), `transformations` (array), `emphasis_points` (array)
5. THE schema SHALL define optional fields: `dependencies` (array of scene_ids), `difficulty_level` (string)
6. THE JSON_Sanitizer SHALL validate against this schema using a JSON schema validation library

### Requirement 19: Animation Code Output Format

**User Story:** As a content creator, I want animation code files to follow a consistent format, so that I can easily review and modify them.

#### Acceptance Criteria

1. EACH scene file SHALL contain a single Python class inheriting from `manim.Scene`
2. EACH scene class SHALL have a `construct` method containing the animation logic
3. EACH scene file SHALL include a header comment with scene_id, narration text, and generation timestamp
4. EACH scene file SHALL be independently executable (no cross-file dependencies except Manim library)
5. THE Persistence_Agent SHALL format code using consistent indentation (4 spaces) and line length (max 100 characters)

### Requirement 20: Context Propagation

**User Story:** As a pipeline developer, I want shared context (definitions, visual style) propagated across scenes, so that animations maintain consistency.

#### Acceptance Criteria

1. THE Scene_Controller SHALL maintain a context object containing concept definitions introduced in previous scenes
2. THE Scene_Controller SHALL maintain a context object containing visual style parameters (colors, font sizes, animation speeds)
3. WHEN processing a scene, THE Animation_Generator SHALL receive both the scene specification and the shared context
4. THE Animation_Generator SHALL reuse visual style parameters from context for consistent appearance
5. IF a scene references a concept defined in a previous scene, THEN THE Animation_Generator SHALL use the definition from context
