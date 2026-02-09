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
    
    def __init__(self, ocr_output: OCROutput, configuration: Configuration):
        """Initialize input
        
        Args:
            ocr_output: OCR output from Vision+OCR Agent
            configuration: Configuration parameters for storyboard generation
        """
        
        self.ocr_output = ocr_output
        self.configuration = configuration


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
    
    def __init__(self):
        """Initialize the Script/Storyboard Architect Agent"""
        self.retry_count = 0
        super().__init__()
        # FIX: Initialize the client HERE, inside __init__
        # This checks for the key safely when the agent is actually used
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = None
            print("⚠️ Warning: OPENAI_API_KEY not found. ScriptArchitect will fail if called.")
    
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
        
        logger.info(f"Starting storyboard generation for: {ocr_output.pdf_metadata.filename}")
        logger.info(f"Configuration: verbosity={config.verbosity}, depth={config.depth}, audience={config.audience}")
        
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
            scenes = self._decompose_into_scenes(ordered_concepts, ocr_output, config)
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
        config: Configuration
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
                config=config
            )
            scenes.append(scene)
        
        return scenes
    
    def _generate_scene(
        self,
        concept_id: str,
        scene_index: int,
        ocr_output: OCROutput,
        config: Configuration
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
        
        # Generate narration based on configuration
        narration = self._generate_narration(concept_id, config, ocr_output)
        
        # Generate visual intent
        visual_intent = self._generate_visual_intent(concept_id, config, ocr_output)
        
        # Estimate duration based on narration length and transformations
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
        ocr_output: OCROutput
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
        # This prevents the pipeline from crashing if keys are missing
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("No OpenAI API key found. Using mock narration.")
            return self._generate_mock_narration_fallback(concept_id, config)

        try:
            # 2. Extract Context (From your snippet)
            # We gather text to give the LLM source material
            context_text = ""
            for page in ocr_output.pages:
                context_text += " ".join(block.text for block in page.text_blocks)

            # 3. Define Nuances (Preserving your original logic)
            # We translate the config enums into specific style instructions
            
            # Verbosity nuances
            verbosity_instruction = "Provide a balanced explanation."
            if config.verbosity == "low":
                verbosity_instruction = "Keep it concise. Focus ONLY on the fundamental idea."
            elif config.verbosity == "high":
                verbosity_instruction = "Be detailed. Connect this to previous concepts, explain why it's important, and explicitly mention how the visual elements help build intuition."
            
            # Audience nuances
            audience_instruction = "Build understanding through clear, standard academic examples."
            if config.audience == "high-school":
                audience_instruction = "Use relatable, real-world examples from everyday life. Avoid dense jargon."
            elif config.audience == "graduate":
                audience_instruction = "Focus on theoretical implications, generalizations, and formal rigor."

            # 4. Construct the Prompt
            prompt = f"""
            You are a world-class educational scriptwriter like Grant Sanderson (3Blue1Brown).
            
            --- SOURCE MATERIAL ---
            {context_text[:3000]} # Context limited to ~3000 chars to stay within token limits
            -----------------------

            TASK: Write a narration script for the concept: "{concept_id}"
            
            STRICT STYLE GUIDELINES:
            1. TONE: Friendly, conversational, and intuitive.
            2. VERBOSITY ({config.verbosity}): {verbosity_instruction}
            3. AUDIENCE ({config.audience}): {audience_instruction}
            4. FORMAT: Return ONLY the spoken text. No markdown, no "Scene 1:", no stage directions.
            """

            # 5. Call the API
            response = client.chat.completions.create(
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
        ocr_output: OCROutput
    ) -> VisualIntent:
        """Generate visual intent for a concept using LLM analysis
        
        Args:
            concept_id: ID of the concept
            config: Configuration parameters
            ocr_output: OCR output for math expressions
            
        Returns:
            VisualIntent object with structured animation plan
        """
        # 1. Safety Check: Fallback if no API key
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("No OpenAI API key found. Using mock visual intent.")
            return self._generate_mock_visual_fallback(concept_id)

        try:
            # 2. Extract Math/Text Context from OCR
            # We prioritize math expressions for the visual intent
            math_context = []
            for page in ocr_output.pages:
                for expr in page.math_expressions:
                    if not expr.extraction_failed:
                        math_context.append(expr.latex)
            
            # Limit context
            context_str = f"Available Math: {', '.join(math_context[:10])}..."
            
            # 3. Construct the Prompt with strict Schema
            prompt = f"""
            Act as a 3Blue1Brown animation architect.
            CONCEPT: {concept_id}
            CONTEXT: {context_str}
            
            Plan the Manim animation scene.
            1. Identify key mathematical objects (Equations, text, graphs).
            2. Define smooth transitions (Write, FadeIn, Transform).
            3. Highlight key moments (Emphasis Points).
            
            Return JSON in this EXACT schema:
            {{
                "objects": [
                    {{ "id": "obj_0", "type": "equation", "content": "LATEX_CODE", "color": "BLUE", "position": "center" }}
                ],
                "transformations": [
                    {{ "type": "Write", "target": "obj_0", "timing": 1.0 }}
                ],
                "emphasis_points": [
                    {{ "timestamp": 2.0, "description": "Focus on the exponent" }}
                ]
            }}
            """

            # 4. Call LLM (Using Mini for speed/structure)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            
            data = json.loads(response.choices[0].message.content)
            
            # 5. Map JSON back to Project Classes (Preserving Nuances)
            
            # A. Mathematical Objects
            math_objs = []
            for obj in data.get('objects', []):
                math_objs.append(MathematicalObject(
                    object_type=obj.get('type', 'text'),
                    content=obj.get('content', ''),
                    # Preserving the nested ObjectStyle nuance
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
                    parameters={}, # LLM can populate this if needed later
                    timing=float(t.get('timing', 1.0))
                ))

            # C. Emphasis Points (Preserving the feature from your mock)
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
            logger.error(f"Failed to generate visual intent: {e}")
            return self._generate_mock_visual_fallback(concept_id)

    def _generate_mock_visual_fallback(self, concept_id):
        """Helper to return the original mock data if API fails"""
        # ... (Your original mock logic goes here as a safety net) ...
        return VisualIntent(
            mathematical_objects=[MathematicalObject(
                object_type="equation", 
                content="f(x)=x^2", 
                style=ObjectStyle(color="blue", position="center")
            )],
            transformations=[Transformation(
                transformation_type="FadeIn", 
                target_object="obj_0", 
                timing=1.0
            )],
            emphasis_points=[]
        )
    
    def _estimate_duration(
        self,
        narration: str,
        visual_intent: VisualIntent
    ) -> float:
        """Estimate scene duration based on narration and transformations
        
        Args:
            narration: Narration text
            visual_intent: Visual intent specification
            
        Returns:
            Estimated duration in seconds
        """
        # Calculate narration duration
        word_count = len(narration.split())
        narration_duration = word_count / self.WORDS_PER_SECOND
        
        # Add time for transformations
        transformation_duration = len(visual_intent.transformations) * 2.0  # 2 seconds per transformation
        
        # Total duration
        total_duration = narration_duration + transformation_duration
        
        # Clamp to reasonable bounds
        total_duration = max(self.MIN_SCENE_DURATION, min(total_duration, self.MAX_SCENE_DURATION))
        
        return round(total_duration, 1)
    
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
