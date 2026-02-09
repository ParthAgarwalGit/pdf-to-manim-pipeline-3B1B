# Determinism in the PDF-to-Manim Pipeline

This document describes how determinism is ensured across the pipeline to satisfy Requirements 10.1-10.4.

## Overview

The pipeline is designed to produce **structurally equivalent outputs** for identical inputs. This means:
- Same PDF + same configuration → same scene count, same scene order, equivalent content
- Deterministic operations throughout the pipeline
- No random sampling or non-deterministic ordering
- Temperature=0 for all LLM calls (when implemented)

## Deterministic Operations by Agent

### 1. Ingestion Agent (`src/agents/ingestion.py`)
- **File metadata extraction**: Deterministic (reads file properties)
- **Timestamp**: Used only for output directory naming, not for core logic
- **Event processing**: Deterministic parsing of event payloads

### 2. Validation Agent (`src/agents/validation.py`)
- **File validation**: Deterministic checks (MIME type, size, structure)
- **No randomness**: All validation rules are deterministic
- **Consistent error codes**: Same input → same validation result

### 3. Vision+OCR Agent (`src/agents/vision_ocr.py`)
- **Page rendering**: Deterministic (fixed DPI=300)
- **Text block extraction**: Uses PyMuPDF's deterministic text extraction
- **Reading order sorting**: Deterministic spatial sorting (top-to-bottom, left-to-right)
  - Blocks grouped into rows by vertical overlap
  - Rows sorted by minimum Y coordinate
  - Within rows, blocks sorted by X coordinate
- **Math expression extraction**: Pattern-based (deterministic regex matching)
- **Diagram detection**: Deterministic image block detection
- **LLM calls** (when implemented): Will use temperature=0 for diagram descriptions

**Key Code:**
```python
# Deterministic reading order sort (vision_ocr.py:273-278)
rows.sort(key=lambda row: min(b.bounding_box.y for b in row))  # Top to bottom
for row in rows:
    row.sort(key=lambda b: b.bounding_box.x)  # Left to right
```

### 4. Script/Storyboard Architect (`src/agents/script_architect.py`)
- **Concept extraction**: Deterministic regex pattern matching
- **Dependency graph**: Deterministic based on concept types and content analysis
- **Topological sort**: Kahn's algorithm (deterministic)
- **Cycle breaking**: Alphabetically sorted remaining nodes (deterministic)
- **Scene ID generation**: Sequential numbering `scene_{index+1:03d}`
- **Scene ordering**: Follows topological sort order (deterministic)
- **LLM calls** (when implemented): Will use temperature=0

**Key Code:**
```python
# Deterministic scene ID generation (script_architect.py:455)
scene_id = f"scene_{scene_index + 1:03d}"

# Deterministic cycle breaking (script_architect.py:394)
sorted_order.extend(sorted(remaining_nodes))  # Alphabetical sort
```

### 5. JSON Sanitizer (`src/agents/json_sanitizer.py`)
- **Schema validation**: Deterministic (jsonschema library)
- **Automatic corrections**: Deterministic rules
- **Scene ID uniqueness check**: Deterministic set comparison
- **No randomness**: All sanitization operations are rule-based

### 6. Scene Controller (`src/agents/scene_controller.py`)
- **Scene iteration**: Deterministic array order (index 0 to N-1)
- **Context initialization**: Deterministic default values
- **Context updates**: Deterministic extraction from scene content
- **No shuffling or reordering**: Scenes processed in exact array order

**Key Code:**
```python
# Deterministic scene iteration (scene_controller.py:88-89)
for scene_index, scene in enumerate(scenes):
    # Process scenes in exact array order
```

### 7. Animation Code Generator (`src/agents/animation_generator.py`)
- **Code generation**: Deterministic template-based (mock implementation)
- **Object naming**: Sequential numbering `obj_{i}`
- **Class name generation**: Deterministic regex-based sanitization
- **LLM calls** (when implemented): Will use temperature=0

**Key Code:**
```python
# Deterministic object naming (animation_generator.py:195)
var_name = f"obj_{i}"

# Deterministic class name generation (animation_generator.py:327-337)
class_name = re.sub(r'[^a-zA-Z0-9_]', '_', scene_id)
```

### 8. Persistence Agent (`src/agents/persistence.py`)
- **File naming**: Deterministic based on scene IDs
- **Directory structure**: Deterministic (timestamp used only for isolation)
- **File writing**: Deterministic content, atomic operations
- **Manifest generation**: Deterministic file listing (sorted)

## Non-Deterministic Elements (Controlled)

### Timestamps
- **Usage**: Output directory naming only (`output/{pdf_filename}/{timestamp}/`)
- **Not used for**: Core logic, ordering, or content generation
- **Purpose**: Isolate multiple runs, prevent overwrites
- **Impact**: Different runs create different directories, but content is equivalent

### LLM Calls (Future Implementation)
- **Current**: Mock implementations (fully deterministic)
- **Future**: Will use temperature=0 for all LLM API calls
- **Agents affected**: Vision+OCR (diagram descriptions), Script Architect (narration), Animation Generator (code)
- **Note**: Even with temperature=0, LLMs may have minor variations due to model updates or API changes

## Testing Determinism

### Property-Based Test (Task 14.2)
```python
# Test: Identical inputs produce structurally equivalent outputs
# For any PDF and configuration, processing twice should yield:
# - Same scene count
# - Same scene order (scene IDs match)
# - Equivalent narration content (structurally similar)
```

### Unit Tests
- Test scene ID generation is sequential
- Test reading order sorting is consistent
- Test topological sort produces same order
- Test cycle breaking is deterministic

## Verification Checklist

✅ **Array Operations**
- Scene iteration: Deterministic (index-based loop)
- Concept ordering: Deterministic (topological sort + alphabetical cycle breaking)
- Text block sorting: Deterministic (spatial coordinates)
- File listing: Deterministic (sorted in manifest)

✅ **Scene ID Generation**
- Format: `scene_{index+1:03d}` (e.g., scene_001, scene_002)
- Sequential numbering based on array index
- No random components

✅ **Temperature Settings**
- Mock implementations: Fully deterministic
- Future LLM calls: Will use temperature=0
- Documented in agent docstrings

✅ **No Random Operations**
- No `random.choice()`, `random.shuffle()`, etc.
- No timestamp-based ordering in core logic
- No hash-based ordering (dict iteration is deterministic in Python 3.7+)

## Limitations

### Acceptable Non-Determinism
1. **Output directory timestamps**: Different runs create different directories
2. **Generation timestamps**: Metadata includes current time (not used for logic)
3. **LLM variability**: Even with temperature=0, minor variations possible

### Structural Equivalence
The pipeline guarantees **structural equivalence**, not **byte-for-byte identity**:
- Scene count and order: Identical
- Scene IDs: Identical
- Narration content: Semantically equivalent (may have minor wording differences with LLMs)
- Code structure: Equivalent (formatting may vary slightly)

## Future Enhancements

1. **Seed-based randomness**: If randomness is needed, use seeded RNGs
2. **Deterministic LLM caching**: Cache LLM responses by input hash
3. **Reproducibility metadata**: Include pipeline version, model versions in manifest
4. **Determinism validation**: Automated tests comparing multiple runs

## References

- **Requirements**: 10.1, 10.2, 10.3, 10.4
- **Design Document**: Section on "Determinism & Idempotency"
- **Task**: 14.1 - Ensure deterministic operations across pipeline
