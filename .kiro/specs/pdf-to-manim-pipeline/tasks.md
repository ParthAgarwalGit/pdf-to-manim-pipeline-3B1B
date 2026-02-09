# Implementation Plan: PDF to Manim Pipeline

## Overview

This implementation plan breaks down the PDF-to-Manim pipeline into discrete coding tasks following the agent-oriented architecture. The pipeline transforms academic PDFs into executable Manim animation code through a sequence of specialized agents: Ingestion, Validation, Vision+OCR, Script/Storyboard Architect, JSON Sanitizer, Scene Controller, Animation Code Generator, and Persistence.

Each task builds incrementally, with early validation through tests. The plan follows the linear agent sequence defined in the design document.

## Tasks

- [x] 1. Set up project structure and core interfaces
  - Create directory structure: `src/agents/`, `src/schemas/`, `src/orchestrator/`, `tests/`
  - Define base `Agent` interface with `execute()`, `validate_input()`, and `get_retry_policy()` methods
  - Set up Python package structure with `__init__.py` files
  - Configure testing framework (pytest) and property-based testing library (Hypothesis)
  - Create requirements.txt with dependencies: manim, pytest, hypothesis, pydantic, boto3
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. Implement data schemas and validation
  - [x] 2.1 Create Pydantic models for all data contracts
    - Implement `FileReference` schema with file_path, filename, size_bytes, upload_timestamp, metadata
    - Implement `OCROutput` schema with pdf_metadata, pages, text_blocks, math_expressions, diagrams
    - Implement `StoryboardJSON` schema with pdf_metadata, configuration, concept_hierarchy, scenes
    - Implement `AnimationCodeObject` schema with scene_id, code, imports, class_name, narration
    - Implement `SceneProcessingInput` schema with scene_spec, context, scene_index
    - Implement `OutputManifest` schema with output_directory, files, timestamp, status
    - _Requirements: 4.5, 5.6, 6.6, 6.7, 18.1, 18.2, 18.3, 18.4, 18.5, 18.6_

  - [x] 2.2 Write unit tests for schema validation
    - Test required field validation for each schema
    - Test optional field defaults
    - Test type coercion and validation errors
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.6_

- [x] 3. Implement Ingestion Agent
  - [x] 3.1 Create IngestAgent class implementing Agent interface
    - Implement cloud storage event listener (S3 event notifications)
    - Extract file metadata from event payload (filename, size, timestamp, path)
    - Construct FileReference object from metadata
    - Implement logging for ingestion events with timestamps
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [x] 3.2 Write unit tests for Ingestion Agent
    - Test event payload parsing with valid S3 event
    - Test metadata extraction accuracy
    - Test FileReference object construction
    - Test error handling for malformed payloads
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 4. Implement Validation Agent
  - [x] 4.1 Create ValidationAgent class implementing Agent interface
    - Implement PDF MIME type verification (check for `%PDF-` magic bytes)
    - Implement file size validation (100KB to 50MB bounds)
    - Implement PDF structure validation (check for `%PDF-` header and `%%EOF` trailer)
    - Return validated FileReference on success or ValidationError with error_code and reason
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 4.2 Write property test for file validation
    - **Property 1: Valid PDFs pass validation**
    - *For any* file with valid PDF structure (correct header, trailer, MIME type, size in bounds), validation should succeed and return the same FileReference
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.5**

  - [x] 4.3 Write unit tests for validation edge cases
    - Test rejection of non-PDF files (wrong MIME type)
    - Test rejection of files outside size bounds
    - Test rejection of corrupted PDFs (missing trailer)
    - Test specific error codes for each failure type
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 5. Implement Vision + OCR Agent
  - [x] 5.1 Create VisionOCRAgent class implementing Agent interface
    - Implement PDF page rendering to high-resolution images (300 DPI)
    - Integrate OCR library (Tesseract or cloud API) for text extraction
    - Extract text blocks with bounding boxes and reading order
    - Implement spatial sorting for multi-column layout preservation
    - _Requirements: 4.1, 4.6_

  - [x] 5.2 Implement mathematical notation extraction
    - Identify math notation regions using LaTeX-like pattern detection
    - Integrate specialized math OCR (Mathpix API or equivalent)
    - Extract LaTeX strings with positional context
    - Handle extraction failures gracefully (mark as `math_extraction_failed`)
    - _Requirements: 4.2, 14.1, 15.1, 15.2_

  - [x] 5.3 Implement diagram detection and description
    - Implement image segmentation to detect diagram regions (contiguous non-text areas > 1000px²)
    - Extract bounding box coordinates for each diagram
    - Classify diagram type (graph, geometric, flowchart, plot, unknown)
    - Generate textual descriptions using multimodal vision model (GPT-4 Vision or Claude 3)
    - Extract visual elements from diagrams
    - _Requirements: 4.3, 4.4, 16.1, 16.2_

  - [x] 5.4 Assemble OCR output and implement error handling
    - Aggregate all extracted content into OCROutput schema
    - Implement retry logic for transient API failures (exponential backoff, max 3 attempts)
    - Handle partial failures (log warnings, continue processing)
    - Output structured JSON conforming to OCR Output Schema
    - _Requirements: 4.5, 14.1_

  - [x] 5.5 Write unit tests for Vision+OCR Agent
    - Test text extraction accuracy with sample PDF pages
    - Test math notation extraction with LaTeX expressions
    - Test diagram detection and classification
    - Test multi-column layout reading order preservation
    - Test error handling for failed extractions
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 6. Checkpoint - Ensure extraction pipeline works
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement Script/Storyboard Architect Agent
  - [x] 7.1 Create ScriptArchitectAgent class implementing Agent interface
    - Parse OCR output to identify core concepts (definitions, theorems, proofs, examples)
    - Build concept dependency graph from identified concepts
    - Perform topological sort on dependency graph for pedagogical ordering
    - Handle dependency cycles by breaking weakest dependencies
    - _Requirements: 5.1, 5.2_

  - [x] 7.2 Implement scene decomposition and generation
    - Decompose content into atomic scenes (one concept per scene, 30-90 second target)
    - Generate narration text emphasizing intuition over formalism
    - Minimize on-screen text density in narration
    - Specify visual intent for each scene (mathematical_objects, transformations, emphasis_points)
    - Estimate scene duration based on narration word count and transformation complexity
    - Ensure conceptual continuity between consecutive scenes
    - _Requirements: 5.3, 5.4, 5.5, 5.8_

  - [x] 7.3 Implement configuration parameter handling
    - Parse and validate configuration parameters (verbosity, depth, audience)
    - Apply default values when configuration is missing (medium, intermediate, undergraduate)
    - Adjust narration based on verbosity level (low → shorter, high → detailed)
    - Adjust content depth (introductory → skip proofs, advanced → include proofs)
    - Adjust terminology and background assumptions based on audience level
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 7.4 Generate and output Storyboard JSON
    - Construct StoryboardJSON object with ordered scene array
    - Include concept_hierarchy with dependencies
    - Ensure all required fields are present for each scene
    - Use temperature=0 for LLM calls to maximize determinism
    - Implement retry logic for invalid JSON generation (max 3 attempts with schema instructions)
    - _Requirements: 5.6, 5.7, 10.4_

  - [x] 7.5 Write unit tests for Script Architect
    - Test concept extraction from sample OCR output
    - Test dependency graph construction and topological sort
    - Test scene decomposition logic
    - Test configuration parameter application
    - Test narration generation quality (intuition vs formalism)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 2.1, 2.2, 2.3_

- [x] 8. Implement JSON Sanitizer Agent
  - [x] 8.1 Create JSONSanitizerAgent class implementing Agent interface
    - Load JSON schema definition for Storyboard JSON
    - Validate input JSON against schema using jsonschema library
    - Collect all validation errors with field paths and violation types
    - Implement automatic corrections for common violations (missing optional fields, type coercion, duplicate scene_ids)
    - Re-validate after corrections and return sanitized JSON or validation error report
    - Verify scene_id uniqueness across scene array
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [x] 8.2 Write property test for JSON sanitization
    - **Property 2: Valid storyboards pass sanitization unchanged**
    - *For any* Storyboard JSON that conforms to the schema, sanitization should return the input unchanged
    - **Validates: Requirements 6.1**

  - [x] 8.3 Write unit tests for sanitization corrections
    - Test automatic correction of missing optional fields
    - Test type coercion for coercible types
    - Test duplicate scene_id correction (append numeric suffix)
    - Test validation error reporting for non-correctable violations
    - _Requirements: 6.2, 6.3, 6.4, 6.5, 6.7_

- [x] 9. Implement Scene Controller Agent
  - [x] 9.1 Create SceneControllerAgent class implementing Agent interface
    - Extract scene array from sanitized Storyboard JSON
    - Initialize shared context object (concept_definitions, visual_style, variable_bindings)
    - Implement deterministic scene iteration loop (index 0 to N-1)
    - Construct SceneProcessingInput for each scene (scene_spec + shared context + scene_index)
    - _Requirements: 7.1, 7.2, 7.3, 20.1, 20.2_

  - [x] 9.2 Implement context propagation and failure isolation
    - Pass read-only context to Animation Generator for each scene
    - Update shared context after successful scene generation (extract concept definitions, variable bindings)
    - Log failures for individual scenes without aborting loop
    - Track success/failure status for each scene with error messages
    - Continue to next scene on failure (failure isolation)
    - _Requirements: 7.3, 7.4, 7.5, 20.3, 20.4, 20.5_

  - [x] 9.3 Generate scene processing report
    - Calculate total_scenes, successful_scenes counts
    - Collect failed_scenes array with scene_id and error messages
    - Determine overall status (SUCCESS, PARTIAL_SUCCESS, CRITICAL_FAILURE)
    - Mark status as CRITICAL_FAILURE if failure rate > 50%
    - Return scene processing report
    - _Requirements: 7.6, 7.7, 14.4_

  - [x] 9.4 Write property test for scene iteration
    - **Property 3: Scene iteration is deterministic**
    - *For any* scene array, iterating over scenes should process them in array order (index 0, 1, 2, ..., N-1) regardless of individual scene failures
    - **Validates: Requirements 7.2, 10.2**

  - [x] 9.5 Write unit tests for Scene Controller
    - Test context initialization
    - Test context update after successful scene
    - Test failure isolation (one scene fails, others continue)
    - Test CRITICAL_FAILURE status when >50% scenes fail
    - Test scene processing report generation
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

- [x] 10. Implement Animation Code Generator Agent
  - [x] 10.1 Create AnimationCodeGeneratorAgent class implementing Agent interface
    - Parse scene specification (narration, visual_intent, duration_estimate)
    - Parse shared context for relevant concept definitions and visual style
    - Generate Python class definition inheriting from `manim.Scene`
    - Generate `construct` method containing animation logic
    - _Requirements: 8.1, 8.6, 20.3_

  - [x] 10.2 Implement mathematical object code generation
    - Generate Manim object creation code for each mathematical_object in visual_intent
    - Support object types: equation (MathTex), graph (Axes + Graph), geometric_shape (Circle, Line, etc.), axes (Axes), text (Text)
    - Apply visual style from context (colors, font_size)
    - Handle LaTeX rendering for mathematical notation using MathTex
    - Reuse concept definitions from context when referenced
    - _Requirements: 8.2, 8.7, 15.3, 20.4, 20.5_

  - [x] 10.3 Implement transformation code generation
    - Generate Manim animation code for each transformation in visual_intent
    - Support transformation types: morph (Transform), highlight (Indicate), move (MoveToTarget), fade_in (FadeIn), fade_out (FadeOut), scale (ScaleInPlace)
    - Sequence transformations with appropriate timing based on timestamp parameters
    - Synchronize visual transformations with narration timing
    - _Requirements: 8.3, 8.4_

  - [x] 10.4 Implement code validation and formatting
    - Validate generated Python code syntax using AST parser
    - Verify Manim imports are correct
    - Verify class structure (inherits Scene, has construct method)
    - Format code with consistent indentation (4 spaces, max 100 char line length)
    - Add header comment with scene_id, narration text, generation timestamp
    - Return AnimationCodeObject or generation error
    - _Requirements: 8.5, 8.8, 19.1, 19.2, 19.3, 19.4, 19.5_

  - [x] 10.5 Implement retry logic for code generation failures
    - Retry with syntax error feedback for invalid Python (max 3 attempts)
    - Retry with API reference for undefined Manim objects (max 3 attempts)
    - Retry with corrected LaTeX for rendering errors (max 2 attempts)
    - Use simplified fallback for ambiguous visual intent
    - Return generation error after max retries exhausted
    - Use temperature=0 for LLM calls to maximize determinism
    - _Requirements: 14.3, 10.4_

  - [x] 10.6 Write property test for code generation
    - **Property 4: Generated code is syntactically valid Python**
    - *For any* scene specification with valid visual_intent, the generated code should parse successfully with Python's AST parser
    - **Validates: Requirements 8.5**

  - [x] 10.7 Write unit tests for Animation Code Generator
    - Test MathTex generation for LaTeX equations
    - Test Axes and Graph generation for plots
    - Test geometric shape generation (Circle, Line, Arrow)
    - Test transformation sequencing and timing
    - Test code formatting (indentation, line length)
    - Test header comment generation
    - Test retry logic for syntax errors
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

- [x] 11. Checkpoint - Ensure scene processing works end-to-end
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Implement Persistence Agent
  - [x] 12.1 Create PersistenceAgent class implementing Agent interface
    - Determine output directory path: `output/{pdf_filename}/{timestamp}/`
    - Create directory structure if not exists
    - Implement atomic file write operations (write to temp file, then rename)
    - Handle directory creation failures (permissions, disk space)
    - _Requirements: 9.4, 9.6, 11.3_

  - [x] 12.2 Write animation code files
    - Write each AnimationCodeObject to file: `scene_{scene_id}.py`
    - Use atomic writes to prevent partial file corruption
    - Ensure files are independently executable
    - _Requirements: 9.1, 19.4_

  - [x] 12.3 Write narration script and metadata files
    - Extract narration from all scenes and write to `narration_script.txt` with scene markers
    - Write Storyboard JSON to `storyboard.json`
    - Write scene processing report to `processing_report.json`
    - Optionally write OCR output to `ocr_output.json` for debugging
    - _Requirements: 9.2, 9.3_

  - [x] 12.4 Generate and write output manifest
    - List all created files with metadata (path, size_bytes, type)
    - Include timestamp, status, scene_success_rate, total_scenes, successful_scenes, failed_scenes
    - Write manifest to `manifest.json`
    - Return OutputManifest object
    - _Requirements: 9.5, 9.7_

  - [x] 12.5 Implement idempotency and cleanup
    - On retry, clean up incomplete files in output directory before writing
    - Ensure no append operations (all writes are full overwrites)
    - Implement atomic file operations to prevent corruption
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [x] 12.6 Write property test for persistence idempotency
    - **Property 5: Retrying persistence with same input produces identical output**
    - *For any* scene processing report and animation code objects, writing twice to the same output directory should produce identical files (same content, same structure)
    - **Validates: Requirements 11.1, 11.2**

  - [x] 12.7 Write unit tests for Persistence Agent
    - Test directory creation
    - Test atomic file writes
    - Test narration script generation with scene markers
    - Test manifest generation with correct metadata
    - Test cleanup on retry (idempotency)
    - Test error handling for disk full, permissions
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 11.1, 11.2, 11.3, 11.4_

- [~] 13. Implement Orchestrator
  - [x] 13.1 Create Orchestrator class
    - Implement linear agent sequencing (Ingestion → Validation → Vision+OCR → Script Architect → JSON Sanitizer → Scene Controller → Persistence)
    - Pass outputs from one agent as inputs to the next
    - Implement abort logic for validation and sanitization errors
    - Implement retry logic based on agent retry policies
    - _Requirements: 1.1, 1.2, 1.3, 3.4, 3.5, 6.5_

  - [x] 13.2 Implement retry policies and error handling
    - Define RetryPolicy class with max_attempts, backoff_strategy, retryable_errors
    - Implement exponential backoff for transient failures
    - Distinguish between retryable errors (API timeouts) and non-retryable errors (invalid input)
    - Abort pipeline on validation or sanitization errors
    - Continue pipeline on partial scene failures
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

  - [x] 13.3 Implement logging and observability
    - Log agent_start events with agent_name, timestamp, input_summary
    - Log agent_complete events with agent_name, timestamp, duration_ms, output_summary, status
    - Log agent_failure events with agent_name, timestamp, error_message, error_code, input_context, retry_attempt
    - Write all logs to `pipeline.log` in output directory using structured JSON format
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

  - [x] 13.4 Write integration tests for orchestrator
    - Test end-to-end pipeline execution with sample PDF
    - Test abort on validation failure
    - Test abort on sanitization failure
    - Test retry logic for transient failures
    - Test partial success (some scenes fail, pipeline continues)
    - Test CRITICAL_FAILURE status when >50% scenes fail
    - _Requirements: 3.4, 6.5, 7.5, 14.4_

- [x] 14. Implement determinism and reproducibility features
  - [x] 14.1 Ensure deterministic operations across pipeline
    - Use deterministic ordering for all array operations (scene iteration, concept hierarchy)
    - Avoid non-deterministic operations (random sampling, timestamp-based ordering) in core logic
    - Use temperature=0 for all LLM calls
    - Implement deterministic scene_id generation
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [x] 14.2 Write property test for pipeline determinism
    - **Property 6: Identical inputs produce structurally equivalent outputs**
    - *For any* PDF and configuration, processing the same input twice should produce structurally equivalent Storyboard JSON (same scene count, same scene order, equivalent narration content)
    - **Validates: Requirements 10.1, 10.2**

- [x] 15. Implement mathematical fidelity features
  - [x] 15.1 Enhance LaTeX extraction accuracy
    - Preserve operator precedence and grouping in extracted LaTeX
    - Distinguish between similar symbols (× vs x, ∈ vs ε)
    - Flag ambiguous mathematical notation in output metadata
    - _Requirements: 15.1, 15.2, 15.4_

  - [x] 15.2 Ensure correct LaTeX rendering in Manim code
    - Validate LaTeX syntax before generating MathTex code
    - Use correct Manim MathTex syntax for rendering
    - Handle special characters and escaping properly
    - _Requirements: 15.3_

  - [x] 15.3 Write unit tests for mathematical fidelity
    - Test operator precedence preservation in LaTeX extraction
    - Test symbol disambiguation (× vs x, ∈ vs ε)
    - Test LaTeX rendering in generated Manim code
    - Test ambiguity flagging
    - _Requirements: 15.1, 15.2, 15.3, 15.4_

- [x] 16. Implement diagram interpretation features
  - [x] 16.1 Enhance diagram-to-animation translation
    - Translate diagram visual intent into Manim primitives (Axes, Line, Circle, Arrow, etc.)
    - Map diagram types to appropriate Manim objects (graph → Axes + Graph, geometric → shapes, flowchart → arrows + text)
    - Preserve spatial relationships from original diagram
    - _Requirements: 16.3, 16.4_

  - [x] 16.2 Write unit tests for diagram interpretation
    - Test diagram type classification accuracy
    - Test visual element extraction
    - Test Manim code generation for different diagram types
    - Test spatial relationship preservation
    - _Requirements: 16.1, 16.2, 16.3, 16.4_

- [x] 17. Implement concurrency and scalability features
  - [x] 17.1 Add support for concurrent pipeline executions
    - Implement job queue with FIFO ordering
    - Support up to 10 concurrent PDF processing jobs
    - Isolate execution contexts (separate output directories, no shared mutable state)
    - Implement job status API for querying progress
    - _Requirements: 17.1, 17.2, 17.3, 17.4_

  - [x] 17.2 Write integration tests for concurrency
    - Test concurrent processing of multiple PDFs
    - Test execution context isolation
    - Test job queue FIFO ordering
    - Test job status API
    - _Requirements: 17.1, 17.2, 17.3, 17.4_

- [x] 18. Implement latency optimizations
  - [x] 18.1 Optimize Vision+OCR Agent performance
    - Implement parallel page processing for multi-page PDFs
    - Cache OCR results to avoid reprocessing
    - Optimize image rendering resolution based on content density
    - Target: 30 seconds per page for PDFs up to 20 pages
    - _Requirements: 13.1_

  - [x] 18.2 Optimize Script Architect performance
    - Implement efficient concept extraction algorithms
    - Cache dependency graph computations
    - Optimize LLM prompt size to reduce latency
    - Target: 2 minutes for PDFs up to 20 pages
    - _Requirements: 13.2_

  - [x] 18.3 Optimize Animation Code Generator performance
    - Implement code generation template caching
    - Optimize LLM prompt construction
    - Target: 10 seconds per scene
    - _Requirements: 13.3_

  - [x] 18.4 Write performance tests
    - Test Vision+OCR latency with 20-page PDF
    - Test Script Architect latency with 20-page PDF
    - Test Animation Code Generator latency per scene
    - Test end-to-end pipeline latency for 10-page PDF with 20 scenes
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

- [x] 19. Final integration and end-to-end testing
  - [x] 19.1 Create end-to-end test suite
    - Test complete pipeline with sample academic PDFs
    - Verify output file structure matches specification
    - Verify generated Manim code is executable
    - Test error handling for various failure scenarios
    - _Requirements: All requirements_

  - [x] 19.2 Create example PDFs and expected outputs
    - Create sample PDFs with text, equations, and diagrams
    - Generate expected outputs for regression testing
    - Document test cases and expected behaviors
    - _Requirements: All requirements_

- [x] 20. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All tasks are required for comprehensive implementation with full testing coverage
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate universal correctness properties across all inputs
- Unit tests validate specific examples, edge cases, and error conditions
- The implementation follows the linear agent sequence: Ingestion → Validation → Vision+OCR → Script Architect → JSON Sanitizer → Scene Controller → Animation Code Generator → Persistence
- Orchestrator coordinates all agents and implements retry/abort logic
- Determinism is achieved through temperature=0 LLM calls and deterministic ordering
- Concurrency support allows processing multiple PDFs simultaneously with isolated contexts
