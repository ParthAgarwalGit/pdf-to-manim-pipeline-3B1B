"""Script/Storyboard Architect Agent for scene generation

This module implements the Script/Storyboard Architect Agent which generates
pedagogically sound scene sequences with narration and visual intent from OCR output.

Note: This implementation uses mock/stub implementations for LLM calls
since we don't have actual API keys. The focus is on agent structure,
concept extraction, dependency graph building, and data flow.
"""

import json
import logging
import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple
import openai
import os


from src.agents.base import (
    Agent,
    AgentExecutionError,
    AgentInput,
    AgentOutput,
    BackoffStrategy,
    RetryPolicy,
)
from src.schemas.ocr_output import OCROutput
from src.schemas.storyboard import (
    Configuration,
    ConceptHierarchy,
    EmphasisPoint,
    MathematicalObject,
    ObjectStyle,
    PDFMetadataStoryboard,
    Scene,
    StoryboardJSON,
    Transformation,
    VisualIntent,
)

logger = logging.getLogger(__name__)


class ScriptArchitectInput(AgentInput):
    """Input for Script/Storyboard Architect Agent"""
    
    def __init__(
        self,
        ocr_output: OCROutput,
        configuration: Configuration,
        review_feedback: Optional[List[str]] = None,
    ):
        """Initialize input
        
        Args:
            ocr_output: OCR output from Vision+OCR Agent
            configuration: Configuration parameters for storyboard generation
            review_feedback: Optional critique to apply on a regeneration pass
        """
        
        self.ocr_output = ocr_output
        self.configuration = configuration
        self.review_feedback = review_feedback or []


class ScriptArchitectAgent(Agent):
    """Agent responsible for generating scene-based storyboards with narration
    
    The ScriptArchitectAgent performs the following operations:
    1. Analyze OCR output to identify core concepts and dependencies
    2. Build concept dependency graph and perform topological sort
    3. Decompose content into atomic scenes (30-90 second target)
    4. Generate narration emphasizing intuition over formalism
    5. Specify visual intent for each scene
    6. Apply configuration parameters (verbosity, depth, audience)
    7. Output structured Storyboard JSON
    
    The agent uses temperature=0 for LLM calls to maximize determinism and
    implements retry logic for invalid JSON generation.
    
    Determinism (Requirements 10.2, 10.3, 10.4):
    - Concept extraction: Deterministic regex patterns
    - Dependency graph: Deterministic based on concept types
    - Topological sort: Kahn's algorithm (deterministic)
    - Scene ID generation: Sequential numbering (scene_001, scene_002, ...)
    - LLM calls: temperature=0 (when implemented)
    """
    
    # Configuration constants
    MIN_SCENE_DURATION = 30.0  # Minimum scene duration in seconds
    MAX_SCENE_DURATION = 90.0  # Maximum scene duration in seconds
    TARGET_SCENE_DURATION = 60.0  # Target scene duration in seconds
    WORDS_PER_SECOND = 2.5  # Average narration speed
    MAX_JSON_RETRIES = 3  # Maximum retries for invalid JSON
    
    # Default configuration values
    DEFAULT_VERBOSITY = "medium"
    DEFAULT_DEPTH = "intermediate"
    DEFAULT_AUDIENCE = "undergraduate"
    
    def __init__(self, strict: bool = False):
        """Initialize the Script/Storyboard Architect Agent.

        Args:
            strict: If True, raise immediately when OPENAI_API_KEY is missing
                    instead of falling back to mock narration/visual generation.
        """
        self.retry_count = 0
        self.strict = strict
        super().__init__()
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            if self.strict:
                raise AgentExecutionError(
                    "MISSING_API_KEY",
                    "OPENAI_API_KEY is required in --strict mode for ScriptArchitect"
                )
            self.client = None
            logger.warning("OPENAI_API_KEY not found. ScriptArchitect will use mock fallbacks.")
    
    def execute(self, input_data: AgentInput) -> AgentOutput:
        """Execute storyboard generation
        
        Args:
            input_data: ScriptArchitectInput with OCR output and configuration
            
        Returns:
            StoryboardJSON object with ordered scene specifications
            
        Raises:
            AgentExecutionError: For unrecoverable failures
        """
        if not self.validate_input(input_data):
            raise AgentExecutionError(
                error_code="INVALID_INPUT",
                message="Input must be a ScriptArchitectInput object",
                context={"input_type": type(input_data).__name__}
            )
        
        script_input: ScriptArchitectInput = input_data
        ocr_output = script_input.ocr_output
        config = script_input.configuration
        review_feedback = script_input.review_feedback or []
        
        logger.info(f"Starting storyboard generation for: {ocr_output.pdf_metadata.filename}")
        logger.info(f"Configuration: verbosity={config.verbosity}, depth={config.depth}, audience={config.audience}")
        if review_feedback:
            logger.info(f"Applying {len(review_feedback)} storyboard revision hints")
        
        try:
            # Step 1: Identify core concepts from OCR output
            concepts = self._identify_concepts(ocr_output, config)
            logger.info(f"Identified {len(concepts)} core concepts")
            
            # Step 2: Build concept dependency graph
            dependency_graph = self._build_dependency_graph(concepts, ocr_output)
            logger.info(f"Built dependency graph with {len(dependency_graph)} nodes")
            
            # Step 3: Perform topological sort for pedagogical ordering
            ordered_concepts = self._topological_sort(dependency_graph)
            logger.info(f"Ordered {len(ordered_concepts)} concepts pedagogically")
            
            # Step 4: Decompose into atomic scenes
            scenes = self._decompose_into_scenes(
                ordered_concepts,
                ocr_output,
                config,
                review_feedback=review_feedback,
            )
            logger.info(f"Decomposed content into {len(scenes)} scenes")
            
            # Step 5: Assemble storyboard JSON
            storyboard = self._assemble_storyboard(
                ocr_output,
                config,
                concepts,
                scenes
            )
            
            logger.info(f"Storyboard generation completed: {len(scenes)} scenes")
            return storyboard
            
        except Exception as e:
            raise AgentExecutionError(
                error_code="STORYBOARD_GENERATION_FAILED",
                message=f"Failed to generate storyboard: {str(e)}",
                context={
                    "filename": ocr_output.pdf_metadata.filename,
                    "error": str(e)
                }
            )
    
    def _identify_concepts(
        self,
        ocr_output: OCROutput,
        config: Configuration
    ) -> List[Dict[str, Any]]:
        """Identify core concepts from OCR output
        
        Analyzes text and math expressions to identify definitions, theorems,
        proofs, examples, and other pedagogical concepts.
        
        Args:
            ocr_output: OCR output with extracted content
            config: Configuration parameters
            
        Returns:
            List of concept dictionaries with id, name, type, content, dependencies
        """
        concepts: List[Dict[str, Any]] = []
        concept_id_counter = 1
        
        # Patterns for identifying different concept types
        definition_patterns = [
            r'(?i)definition\s*\d*[:\.]?\s*(.+?)(?=\n|$)',
            r'(?i)we define\s+(.+?)\s+as',
            r'(?i)let\s+(.+?)\s+be\s+(?:a|an|the)',
        ]
        
        theorem_patterns = [
            r'(?i)theorem\s*\d*[:\.]?\s*(.+?)(?=\n|$)',
            r'(?i)proposition\s*\d*[:\.]?\s*(.+?)(?=\n|$)',
            r'(?i)lemma\s*\d*[:\.]?\s*(.+?)(?=\n|$)',
            r'(?i)corollary\s*\d*[:\.]?\s*(.+?)(?=\n|$)',
        ]
        
        proof_patterns = [
            r'(?i)proof[:\.]?\s*(.+?)(?=\n|$)',
            r'(?i)we prove\s+(.+?)(?=\n|$)',
        ]
        
        example_patterns = [
            r'(?i)example\s*\d*[:\.]?\s*(.+?)(?=\n|$)',
            r'(?i)for example[,\s]+(.+?)(?=\n|$)',
        ]
        
        # Extract concepts from each page
        for page in ocr_output.pages:
            # Combine all text blocks for analysis
            page_text = " ".join(block.text for block in page.text_blocks)
            
            # Identify definitions
            for pattern in definition_patterns:
                matches = re.finditer(pattern, page_text)
                for match in matches:
                    concept_name = match.group(1).strip()[:100]  # Limit length
                    if concept_name:
                        concepts.append({
                            "concept_id": f"concept_{concept_id_counter:03d}",
                            "concept_name": concept_name,
                            "concept_type": "definition",
                            "content": match.group(0),
                            "page_number": page.page_number,
                            "dependencies": []
                        })
                        concept_id_counter += 1
            
            # Identify theorems
            for pattern in theorem_patterns:
                matches = re.finditer(pattern, page_text)
                for match in matches:
                    concept_name = match.group(1).strip()[:100]
                    if concept_name:
                        concepts.append({
                            "concept_id": f"concept_{concept_id_counter:03d}",
                            "concept_name": concept_name,
                            "concept_type": "theorem",
                            "content": match.group(0),
                            "page_number": page.page_number,
                            "dependencies": []
                        })
                        concept_id_counter += 1
            
            # Identify proofs (skip if depth is introductory)
            if config.depth != "introductory":
                for pattern in proof_patterns:
                    matches = re.finditer(pattern, page_text)
                    for match in matches:
                        concept_name = match.group(1).strip()[:100]
                        if concept_name:
                            concepts.append({
                                "concept_id": f"concept_{concept_id_counter:03d}",
                                "concept_name": concept_name,
                                "concept_type": "proof",
                                "content": match.group(0),
                                "page_number": page.page_number,
                                "dependencies": []
                            })
                            concept_id_counter += 1
            
            # Identify examples
            for pattern in example_patterns:
                matches = re.finditer(pattern, page_text)
                for match in matches:
                    concept_name = match.group(1).strip()[:100]
                    if concept_name:
                        concepts.append({
                            "concept_id": f"concept_{concept_id_counter:03d}",
                            "concept_name": concept_name,
                            "concept_type": "example",
                            "content": match.group(0),
                            "page_number": page.page_number,
                            "dependencies": []
                        })
                        concept_id_counter += 1
        
        # If no concepts found, create a general introduction concept
        if not concepts:
            concepts.append({
                "concept_id": "concept_001",
                "concept_name": "Introduction",
                "concept_type": "introduction",
                "content": "General introduction to the document content",
                "page_number": 1,
                "dependencies": []
            })
        
        return concepts
    
    def _build_dependency_graph(
        self,
        concepts: List[Dict[str, Any]],
        ocr_output: OCROutput
    ) -> Dict[str, List[str]]:
        """Build concept dependency graph
        
        Analyzes concept relationships to determine which concepts depend on others.
        Uses heuristics based on concept types and page ordering.
        
        Args:
            concepts: List of identified concepts
            ocr_output: OCR output for additional context
            
        Returns:
            Adjacency list representation of dependency graph
        """
        graph: Dict[str, List[str]] = defaultdict(list)
        
        # Initialize graph with all concept IDs
        for concept in concepts:
            graph[concept["concept_id"]] = []
        
        # Build dependencies based on concept types and ordering
        for i, concept in enumerate(concepts):
            concept_id = concept["concept_id"]
            concept_type = concept["concept_type"]
            
            # Definitions typically have no dependencies (foundational)
            if concept_type == "definition":
                # Check if this definition references earlier definitions
                for j in range(i):
                    earlier_concept = concepts[j]
                    if earlier_concept["concept_type"] == "definition":
                        # Simple heuristic: check if earlier concept name appears in current content
                        if earlier_concept["concept_name"].lower() in concept["content"].lower():
                            concept["dependencies"].append(earlier_concept["concept_id"])
            
            # Theorems depend on relevant definitions
            elif concept_type == "theorem":
                for j in range(i):
                    earlier_concept = concepts[j]
                    if earlier_concept["concept_type"] in ["definition", "theorem"]:
                        # Check for name references
                        if earlier_concept["concept_name"].lower() in concept["content"].lower():
                            concept["dependencies"].append(earlier_concept["concept_id"])
            
            # Proofs depend on the theorem they prove
            elif concept_type == "proof":
                # Find the most recent theorem
                for j in range(i - 1, -1, -1):
                    earlier_concept = concepts[j]
                    if earlier_concept["concept_type"] == "theorem":
                        concept["dependencies"].append(earlier_concept["concept_id"])
                        break
            
            # Examples depend on relevant definitions and theorems
            elif concept_type == "example":
                for j in range(i):
                    earlier_concept = concepts[j]
                    if earlier_concept["concept_type"] in ["definition", "theorem"]:
                        if earlier_concept["concept_name"].lower() in concept["content"].lower():
                            concept["dependencies"].append(earlier_concept["concept_id"])
            
            # Add dependencies to graph
            for dep_id in concept["dependencies"]:
                if dep_id in graph:
                    graph[dep_id].append(concept_id)
        
        return graph
    
    def _topological_sort(
        self,
        graph: Dict[str, List[str]]
    ) -> List[str]:
        """Perform topological sort on dependency graph
        
        Uses Kahn's algorithm for topological sorting. Handles cycles by
        breaking the weakest dependency (most recent edge).
        
        This operation is deterministic (Requirements 10.2, 10.3):
        - Kahn's algorithm processes nodes in a deterministic order
        - Cycle breaking uses alphabetical sorting for consistency
        
        Args:
            graph: Adjacency list representation of dependency graph
            
        Returns:
            List of concept IDs in pedagogical order
        """
        # Calculate in-degrees
        in_degree: Dict[str, int] = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1
        
        # Initialize queue with nodes having no dependencies
        queue = deque([node for node in graph if in_degree[node] == 0])
        sorted_order: List[str] = []
        
        # Process nodes in topological order
        while queue:
            node = queue.popleft()
            sorted_order.append(node)
            
            # Reduce in-degree for neighbors
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Handle cycles: if not all nodes processed, there's a cycle
        if len(sorted_order) < len(graph):
            logger.warning("Dependency cycle detected, breaking weakest dependencies")
            
            # Find nodes still with dependencies (part of cycle)
            remaining_nodes = [node for node in graph if node not in sorted_order]
            
            # Add remaining nodes in deterministic sorted order (break cycle)
            # Sort alphabetically to ensure deterministic ordering
            sorted_order.extend(sorted(remaining_nodes))
        
        return sorted_order
    
    def _decompose_into_scenes(
        self,
        ordered_concepts: List[str],
        ocr_output: OCROutput,
        config: Configuration,
        review_feedback: Optional[List[str]] = None,
    ) -> List[Scene]:
        """Decompose content into atomic scenes
        
        Creates one scene per concept with narration, visual intent, and duration.
        Applies configuration parameters to adjust content.
        
        Args:
            ordered_concepts: List of concept IDs in pedagogical order
            ocr_output: OCR output for content extraction
            config: Configuration parameters
            
        Returns:
            List of Scene objects
        """
        scenes: List[Scene] = []
        
        # Build concept lookup
        concept_map = {}
        for page in ocr_output.pages:
            page_text = " ".join(block.text for block in page.text_blocks)
            # Store page text for concept lookup
            concept_map[page.page_number] = page_text
        
        for scene_index, concept_id in enumerate(ordered_concepts):
            # Generate scene for this concept
            scene = self._generate_scene(
                concept_id=concept_id,
                scene_index=scene_index,
                ocr_output=ocr_output,
                config=config,
                review_feedback=review_feedback,
            )
            scenes.append(scene)
        
        return scenes
    
    def _generate_scene(
        self,
        concept_id: str,
        scene_index: int,
        ocr_output: OCROutput,
        config: Configuration,
        review_feedback: Optional[List[str]] = None,
    ) -> Scene:
        """Generate a single scene for a concept
        
        Args:
            concept_id: ID of the concept for this scene
            scene_index: Sequential index of the scene
            ocr_output: OCR output for content
            config: Configuration parameters
            
        Returns:
            Scene object with narration and visual intent
        """
        # Generate scene ID deterministically from scene index
        # Format: scene_001, scene_002, etc. (Requirements 10.2, 10.4)
        scene_id = f"scene_{scene_index + 1:03d}"
        
        # Generate narration FIRST — this is the creative backbone
        narration = self._generate_narration(
            concept_id,
            config,
            ocr_output,
            review_feedback=review_feedback,
        )

        # Generate visual intent SECOND — pass narration so visuals match what's spoken
        visual_intent = self._generate_visual_intent(
            concept_id,
            config,
            ocr_output,
            narration=narration,
            review_feedback=review_feedback,
        )

        # Duration is driven by narration word count (the source of truth)
        duration = self._estimate_duration(narration, visual_intent)
        
        # Determine dependencies (previous scene if not first)
        dependencies = []
        if scene_index > 0:
            dependencies.append(f"scene_{scene_index:03d}")
        
        return Scene(
            scene_id=scene_id,
            scene_index=scene_index,
            concept_id=concept_id,
            narration=narration,
            visual_intent=visual_intent,
            duration_estimate=duration,
            dependencies=dependencies
        )
    
    def _generate_narration(
        self,
        concept_id: str,
        config: Configuration,
        ocr_output: OCROutput,
        review_feedback: Optional[List[str]] = None,
    ) -> str:
        """Generate narration text for a concept using LLM
        
        Args:
            concept_id: ID of the concept
            config: Configuration parameters
            ocr_output: OCR output for content
            
        Returns:
            Narration text string
        """
        # 1. Safety Check: If no API key, fall back to the old mock logic
        if not os.getenv("OPENAI_API_KEY"):
            if self.strict:
                raise AgentExecutionError(
                    "MISSING_API_KEY",
                    "OPENAI_API_KEY is required in --strict mode for narration generation"
                )
            logger.warning("No OpenAI API key found. Using mock narration.")
            return self._generate_mock_narration_fallback(concept_id, config)

        try:
            # 2. Extract Context — full OCR text + extracted math
            context_text = ""
            math_expressions = []
            for page in ocr_output.pages:
                context_text += " ".join(block.text for block in page.text_blocks) + "\n"
                for expr in page.math_expressions:
                    if not expr.extraction_failed:
                        math_expressions.append(expr.latex)

            math_block = ""
            if math_expressions:
                math_block = "\n--- KEY FORMULAS FROM THE PDF ---\n"
                math_block += "\n".join(f"  • {e}" for e in math_expressions[:20])
                math_block += "\n---------------------------------\n"

            # 3. Style nuances from config
            verbosity_instruction = "Provide a balanced explanation."
            if config.verbosity == "low":
                verbosity_instruction = "Keep it concise. Focus ONLY on the fundamental idea."
            elif config.verbosity == "high":
                verbosity_instruction = "Be detailed. Connect this to previous concepts, explain why it's important, and explicitly mention how the visual elements help build intuition."

            audience_instruction = "Build understanding through clear, standard academic examples."
            if config.audience == "high-school":
                audience_instruction = "Use relatable, real-world examples from everyday life. Avoid dense jargon."
            elif config.audience == "graduate":
                audience_instruction = "Focus on theoretical implications, generalizations, and formal rigor."

            review_block = ""
            if review_feedback:
                review_block = "\nREVISION FEEDBACK FROM THE LAST QUALITY PASS:\n"
                review_block += "\n".join(f"- {item}" for item in review_feedback[:8])
                review_block += "\nAddress these issues directly in the revised narration.\n"

            # 4. Construct the Prompt — anchored to actual PDF content
            prompt = f"""You are a world-class educational scriptwriter like Grant Sanderson (3Blue1Brown).

--- SOURCE MATERIAL (extracted from the PDF) ---
{context_text[:4000]}
{math_block}
--- END SOURCE MATERIAL ---

TASK: Write a narration script that EXPLAINS the content above.

CRITICAL CONTENT RULES:
1. You MUST reference the specific definitions, formulas, and examples from
   the source material.  If the PDF defines foundational formulae to the concepts like sin θ = opposite / hypotenuse in trigonometry, then
   your narration must state that exact relationship and explain it.
2. Every formula listed above should be mentioned and explained intuitively.
3. Build deeper understanding by connecting the formulas to visual intuition
   (e.g. "sin θ equals the y-coordinate on the unit circle" if the PDF
   covers unit circles).  This deeper insight is what makes it 3Blue1Brown.
4. Do NOT invent examples that contradict or ignore the source material.
   Extend and illustrate, but stay faithful. If a visually explainable example is present in the pdf, you MUST use it.
5. EVERY concrete example you mention (like a specific triangle with
   sides 6 and 10) MUST be something that can be animated. If you mention
   it in words, the viewer will expect to see it on screen.

STYLE GUIDELINES:
1. TONE: Friendly, conversational, and intuitive.
2. VERBOSITY ({config.verbosity}): {verbosity_instruction}
3. AUDIENCE ({config.audience}): {audience_instruction}
4. FORMAT: Return ONLY the spoken text. No markdown, no "Scene 1:", no
   stage directions, no headers.
{review_block}
"""

            # 5. Call the API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, # Slight creativity allowed, but mostly deterministic
            )
            
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Failed to generate narration via LLM: {e}")
            return f"Error generating narration for {concept_id}. Please check logs."

    def _generate_mock_narration_fallback(self, concept_id, config):
        """Helper to keep the old mock logic accessible if API fails"""
        base = f"Let's explore the concept of {concept_id}. "
        if config.verbosity == "low": return base + "This is a fundamental idea."
        if config.audience == "high-school": return base + "Imagine this in real life."
        return base + "This concept is crucial."
    
    def _generate_visual_intent(
        self,
        concept_id: str,
        config: Configuration,
        ocr_output: OCROutput,
        narration: str = "",
        review_feedback: Optional[List[str]] = None,
    ) -> VisualIntent:
        """Generate visual intent for a concept, synchronized with narration.

        The narration is the single source of truth. Visual intent must depict
        exactly what the narration describes, in the same order and at matching
        timestamps so that the final video and TTS audio stay in lock-step.

        Args:
            concept_id: ID of the concept
            config: Configuration parameters
            ocr_output: OCR output for math expressions
            narration: The already-generated narration text for this scene.
                       Visuals are planned to match this narration beat-by-beat.

        Returns:
            VisualIntent object with structured animation plan
        """
        # 1. Safety Check: Fallback if no API key
        if not os.getenv("OPENAI_API_KEY"):
            if self.strict:
                raise AgentExecutionError(
                    "MISSING_API_KEY",
                    "OPENAI_API_KEY is required in --strict mode for visual intent generation"
                )
            logger.warning("No OpenAI API key found. Using mock visual intent.")
            return self._generate_mock_visual_fallback(concept_id)

        try:
            # 2. Extract Math/Text Context from OCR
            math_context = []
            for page in ocr_output.pages:
                for expr in page.math_expressions:
                    if not expr.extraction_failed:
                        math_context.append(expr.latex)

            context_str = f"Available Math: {', '.join(math_context[:10])}" if math_context else "No math expressions extracted."

            # 3. Compute timing budget from narration
            word_count = len(narration.split())
            tts_duration = round(word_count / 3.0, 1)  # ~3 words/sec for TTS

            review_block = ""
            if review_feedback:
                review_block = "\nREVISION FEEDBACK TO INCORPORATE:\n"
                review_block += "\n".join(f"- {item}" for item in review_feedback[:8])
                review_block += "\n"

            # 4. Construct the narration-aware prompt
            prompt = f"""
You are a 3Blue1Brown animation architect.

CONCEPT: {concept_id}
MATH CONTEXT: {context_str}

--- NARRATION (this is the AUDIO the viewer will hear) ---
{narration[:4000]}
--- END NARRATION ---

TIMING: The narration is ~{word_count} words. At ~3 words/sec TTS this
will produce ~{tts_duration}s of audio. Your visual plan MUST fill
exactly {tts_duration}s of screen time.
{review_block}

TASK — Plan Manim visuals that stay perfectly in sync with the narration:

1. Split the narration into BEATS (logical paragraphs / sentences).
   For each beat, specify the objects on screen and the animations.
2. Every example or metaphor mentioned in the narration MUST have a
   matching visual (equation, graph, geometric shape, or diagram).
   If the narration contains a discussion-only aside with no natural
   visual, use a "pi_character" object (a blue Pi character talking
   to 3 smaller dark-brown Pi characters — 3Blue1Brown style).
3. Objects must NOT overlap. Use these screen regions:
   - "top": title / section headers
   - "center": main equation or primary visual
   - "bottom_left": axes / graphs
   - "bottom_right": supplementary labels
   - "right": side notes / explanations
   When a region is reused, emit a FadeOut for the previous occupant first.
4. Transformation timings must sum to ≈{tts_duration}s total, matching
   the narration pace. Include self.wait() gaps for the narrator to speak
   between animations.

Return JSON in this EXACT schema:
{{
  "beats": [
    {{
      "beat_index": 0,
      "narration_excerpt": "first ~15 words of this beat...",
      "start_time": 0.0,
      "end_time": 12.0
    }}
  ],
  "objects": [
    {{
      "id": "obj_0",
      "type": "equation|text|graph|geometric_shape|pi_character",
      "content": "LaTeX or description",
      "color": "BLUE",
      "position": "center|top|bottom_left|bottom_right|right"
    }}
  ],
  "transformations": [
    {{
      "type": "Write|FadeIn|FadeOut|Transform|ReplacementTransform|Create|Indicate",
      "target": "obj_0",
      "timing": 1.0,
      "beat_index": 0
    }}
  ],
  "emphasis_points": [
    {{
      "timestamp": 2.0,
      "description": "Focus on the exponent"
    }}
  ]
}}
"""

            # 5. Call LLM
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )

            data = json.loads(response.choices[0].message.content)

            # 6. Map JSON back to project classes

            # A. Mathematical Objects
            math_objs = []
            for obj in data.get('objects', []):
                math_objs.append(MathematicalObject(
                    object_type=obj.get('type', 'text'),
                    content=obj.get('content', ''),
                    style=ObjectStyle(
                        color=obj.get('color', 'WHITE'),
                        position=obj.get('position', 'center')
                    )
                ))

            # B. Transformations
            trans = []
            for t in data.get('transformations', []):
                trans.append(Transformation(
                    transformation_type=t.get('type', 'FadeIn'),
                    target_object=t.get('target', 'obj_0'),
                    parameters={
                        "beat_index": t.get('beat_index', 0)
                    },
                    timing=float(t.get('timing', 1.0))
                ))

            # C. Emphasis Points
            emphasis = []
            for e in data.get('emphasis_points', []):
                emphasis.append(EmphasisPoint(
                    timestamp=float(e.get('timestamp', 0.0)),
                    description=e.get('description', 'Key moment')
                ))

            return VisualIntent(
                mathematical_objects=math_objs,
                transformations=trans,
                emphasis_points=emphasis
            )

        except Exception as e:
            logger.error(f"Failed to generate visual intent via LLM: {e}", exc_info=True)
            logger.warning("Falling back to narration-derived visual intent")
            return self._generate_fallback_visual_intent(narration)

    def _generate_fallback_visual_intent(self, narration: str) -> VisualIntent:
        """Build a minimal visual intent from the narration text.

        Instead of the old generic `f(x)=x^2` mock, this scans the narration
        for keywords and creates placeholder objects the animation generator
        can use to at least show relevant text on screen.
        """
        objs = []
        # Show the key narration summary as a text object
        # Take the first sentence as a title-like object
        first_sentence = narration.split(".")[0].strip() if narration else "Introduction"
        objs.append(MathematicalObject(
            object_type="text",
            content=first_sentence[:80],
            style=ObjectStyle(color="BLUE", position="top")
        ))

        # If narration mentions equations, add a placeholder
        import re as _re
        latex_hints = _re.findall(r'(sin|cos|tan|theta|pi|equation|formula)', narration.lower())
        if latex_hints:
            objs.append(MathematicalObject(
                object_type="equation",
                content=r"\sin(\theta) = \frac{\text{opposite}}{\text{hypotenuse}}",
                style=ObjectStyle(color="YELLOW", position="center")
            ))

        trans = [
            Transformation(
                transformation_type="Write",
                target_object="obj_0",
                parameters={},
                timing=2.0,
            )
        ]

        return VisualIntent(
            mathematical_objects=objs,
            transformations=trans,
            emphasis_points=[],
        )
    
    def _estimate_duration(
        self,
        narration: str,
        visual_intent: VisualIntent
    ) -> float:
        """Estimate scene duration from narration word count.

        The narration is the single source of truth for duration.
        TTS produces audio at ~3 words/second, so duration = words / 3.
        We floor at 15s to avoid degenerate scenes.

        Args:
            narration: Narration text
            visual_intent: Visual intent specification (unused but kept for API)

        Returns:
            Estimated duration in seconds
        """
        word_count = len(narration.split())
        tts_duration = word_count / 3.0  # TTS rate: ~3 words/sec

        # Floor at 15s, no upper cap — long narrations get long videos
        return round(max(15.0, tts_duration), 1)
    
    def _assemble_storyboard(
        self,
        ocr_output: OCROutput,
        config: Configuration,
        concepts: List[Dict[str, Any]],
        scenes: List[Scene]
    ) -> StoryboardJSON:
        """Assemble complete storyboard JSON
        
        Args:
            ocr_output: OCR output
            config: Configuration parameters
            concepts: List of identified concepts
            scenes: List of generated scenes
            
        Returns:
            StoryboardJSON object
        """
        # Create PDF metadata for storyboard
        pdf_metadata = PDFMetadataStoryboard(
            filename=ocr_output.pdf_metadata.filename,
            page_count=ocr_output.pdf_metadata.page_count
        )
        
        # Create concept hierarchy
        concept_hierarchy = [
            ConceptHierarchy(
                concept_id=concept["concept_id"],
                concept_name=concept["concept_name"],
                dependencies=concept["dependencies"]
            )
            for concept in concepts
        ]
        
        # Assemble storyboard
        storyboard = StoryboardJSON(
            pdf_metadata=pdf_metadata,
            configuration=config,
            concept_hierarchy=concept_hierarchy,
            scenes=scenes
        )
        
        return storyboard
    
    def validate_input(self, input_data: AgentInput) -> bool:
        """Validate input is a ScriptArchitectInput object
        
        Args:
            input_data: Input object to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        if not isinstance(input_data, ScriptArchitectInput):
            return False
        
        # Validate OCR output and configuration are present
        if not hasattr(input_data, 'ocr_output') or not hasattr(input_data, 'configuration'):
            return False
        
        return isinstance(input_data.ocr_output, OCROutput) and isinstance(input_data.configuration, Configuration)
    
    def get_retry_policy(self) -> RetryPolicy:
        """Return retry policy for Script/Storyboard Architect Agent
        
        LLM calls may fail due to invalid JSON generation or API issues.
        Use retry with schema enforcement for JSON issues.
        
        Returns:
            RetryPolicy with limited retries
        """
        return RetryPolicy(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.CONSTANT,
            base_delay_seconds=1.0,
            max_delay_seconds=5.0,
            retryable_errors=[
                "INVALID_JSON",
                "SCHEMA_VIOLATION",
                "API_TIMEOUT",
                "RATE_LIMIT_EXCEEDED"
            ]
        )
