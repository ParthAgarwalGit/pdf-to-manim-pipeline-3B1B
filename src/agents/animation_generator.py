"""Animation Code Generator Agent for converting scene specifications to Manim code.

This agent generates executable Python/Manim code from scene specifications,
including mathematical objects, transformations, and timing synchronization.
"""

import ast
import re
import anthropic 
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.agents.base import Agent, AgentExecutionError, AgentInput, AgentOutput, BackoffStrategy, RetryPolicy
from src.schemas.animation_code import AnimationCodeObject
from src.schemas.scene_processing import SceneProcessingInput, SharedContext


class AnimationGeneratorInput(AgentInput):
    """Input for Animation Code Generator Agent.
    
    This is an alias for SceneProcessingInput for clarity.
    """
    
    def __init__(self, scene_processing_input: SceneProcessingInput):
        self.scene_processing_input = scene_processing_input


class AnimationGeneratorOutput(AgentOutput):
    """Output from Animation Code Generator Agent.
    
    Attributes:
        animation_code: Generated AnimationCodeObject
    """
    
    def __init__(self, animation_code: AnimationCodeObject):
        self.animation_code = animation_code


class AnimationGeneratorError(AgentOutput):
    """Error output when code generation fails.
    
    Attributes:
        scene_id: Scene identifier
        error: Error message
    """
    
    def __init__(self, scene_id: str, error: str):
        self.scene_id = scene_id
        self.error = error


class AnimationCodeGeneratorAgent(Agent):
    """Agent responsible for generating executable Manim animation code.
    
    The Animation Code Generator:
    - Parses scene specification (narration, visual_intent, duration)
    - Parses shared context for concept definitions and visual style
    - Generates Manim code for mathematical objects (MathTex, Axes, shapes)
    - Generates Manim code for transformations (Transform, FadeIn, etc.)
    - Validates generated code syntax using AST parser
    - Formats code with consistent indentation
    - Adds header comment with scene metadata
    
    Note: This is a mock implementation. In production, this would use
    an LLM (GPT-4, Claude 3.5 Sonnet) with temperature=0 for code generation.
    """
    
def __init__(self, max_retries: int = 3):
        super().__init__()
        self.max_retries = max_retries
        
        # Initialize Anthropic Client
        # This looks for os.environ["ANTHROPIC_API_KEY"]
        if os.getenv("ANTHROPIC_API_KEY"):
            self.client = anthropic.Anthropic()
        else:
            self.client = None
    
    def execute(self, input_data: AnimationGeneratorInput) -> AgentOutput:
        """Execute animation code generation.
        
        Args:
            input_data: AnimationGeneratorInput with scene processing input
            
        Returns:
            AnimationGeneratorOutput with generated code OR AnimationGeneratorError
            
        Raises:
            AgentExecutionError: For unrecoverable failures
        """
        if not self.validate_input(input_data):
            raise AgentExecutionError(
                "INVALID_INPUT",
                "Input must be AnimationGeneratorInput",
                {"input_type": type(input_data).__name__}
            )
        
        scene_input = input_data.scene_processing_input
        scene_spec = scene_input.scene_spec
        context = scene_input.context
        
        try:
            # Generate code with retry logic
            for attempt in range(self.max_retries):
                try:
                    code = self._generate_code(scene_spec, context)
                    
                    # Validate syntax
                    self._validate_syntax(code)
                    
                    # Format code
                    formatted_code = self._format_code(code, scene_spec, context)
                    
                    # Create animation code object
                    animation_code = AnimationCodeObject(
                        scene_id=scene_spec.scene_id,
                        code=formatted_code,
                        imports=self._get_imports(),
                        class_name=self._get_class_name(scene_spec.scene_id),
                        narration=scene_spec.narration,
                        duration_estimate=scene_spec.duration_estimate,
                        generation_timestamp=datetime.now()
                    )
                    
                    return AnimationGeneratorOutput(animation_code=animation_code)
                    
                except SyntaxError as e:
                    if attempt < self.max_retries - 1:
                        # Retry with syntax error feedback
                        continue
                    else:
                        raise
        
        except Exception as e:
            return AnimationGeneratorError(
                scene_id=scene_spec.scene_id,
                error=str(e)
            )
    
    def validate_input(self, input_data: AgentInput) -> bool:
        """Validate input is AnimationGeneratorInput.
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(input_data, AnimationGeneratorInput):
            return False
        if not isinstance(input_data.scene_processing_input, SceneProcessingInput):
            return False
        return True
    
    def get_retry_policy(self) -> RetryPolicy:
        """Return retry policy for Animation Code Generator.
        
        Returns:
            RetryPolicy with exponential backoff for transient failures
        """
        return RetryPolicy(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay_seconds=1.0,
            max_delay_seconds=30.0,
            retryable_errors=["SYNTAX_ERROR", "API_TIMEOUT", "TRANSIENT_FAILURE"]
        )
    
    def _generate_code(self, scene_spec: Any, context: SharedContext) -> str:
        """Generate Manim code using GPT-4o with 3B1B style enforcement"""
        
        # Safety: Fallback if no key
        if not os.getenv("OPENAI_API_KEY"):
            return self._mock_generate_code(scene_spec, context)

        class_name = self._get_class_name(scene_spec.scene_id)
        
        # Prepare data for the LLM
        scene_data = {
            "narration": scene_spec.narration,
            "duration": scene_spec.duration_estimate,
            "objects": [
                {"type": obj.object_type, "content": obj.content, "style": obj.style.__dict__} 
                for obj in scene_spec.visual_intent.mathematical_objects
            ],
            "transformations": [
                {"type": t.transformation_type, "target": t.target_object, "timing": t.timing} 
                for t in scene_spec.visual_intent.transformations
            ]
        }

        # THE 3B1B SYSTEM PROMPT
        system_prompt = f"""
        You are an expert Manim developer specializing in the 3Blue1Brown style.
        
        TASK:
        Write a Python class named '{class_name}' that inherits from 'Scene'.
        
        STYLE GUIDELINES:
        1. Animations: Use 'Write' and 'Create' for text. Use 'ReplacementTransform' for equations.
        2. Colors: Use the default black background. Use accents like BLUE, YELLOW, and GREEN for emphasis.
        3. Pacing: Ensure animations match the duration: {scene_spec.duration_estimate}s.
        4. Math: ALWAYS use MathTex(r"...") for equations.
        
        OUTPUT RULES:
        - Return ONLY valid Python code.
        - Do NOT include markdown formatting (no ```python blocks).
        - Do NOT include imports (I will handle them).
        """

        user_prompt = f"""
        Generate the 'construct(self)' method for this scene plan:
        {json.dumps(scene_data, indent=2)}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1 # Keep strict syntax
            )
            
            raw_code = response.choices[0].message.content
            return self._clean_llm_code(raw_code, class_name)

        except Exception as e:
            print(f"LLM generation failed: {e}")
            return self._mock_generate_code(scene_spec, context)

    def _clean_llm_code(self, raw_code: str, class_name: str) -> str:
        """Strip markdown and ensure class structure"""
        # Remove markdown fences if present
        clean = raw_code.replace("```python", "").replace("```", "").strip()
        
        # If LLM only returned the method body, wrap it in the class
        if f"class {class_name}" not in clean:
            # Indent the body
            body = "\n".join(["    " + line for line in clean.split("\n")])
            return f"class {class_name}(Scene):\n    def construct(self):\n{body}"
            
        return clean
    
    def _generate_mathematical_objects(
        self,
        math_objects: List[Dict[str, Any]],
        context: SharedContext
    ) -> str:
        """Generate code for mathematical objects.
        
        Args:
            math_objects: List of mathematical object specifications
            context: Shared context with visual style
            
        Returns:
            Generated code string
        """
        if not math_objects:
            return "        # No mathematical objects"
        
        code_lines = []
        colors = context.visual_style.colors
        
        for i, obj in enumerate(math_objects):
            obj_type = obj.get("object_type", "text")
            content = obj.get("content", "")
            style = obj.get("style", {})
            color = style.get("color", colors.get("primary", "BLUE"))
            
            var_name = f"obj_{i}"
            
            if obj_type == "equation":
                # Generate MathTex for equations
                latex = self._escape_latex(content)
                code_lines.append(f'        {var_name} = MathTex(r"{latex}", color={color.upper()})')
            
            elif obj_type == "text":
                code_lines.append(f'        {var_name} = Text("{content}", color={color.upper()})')
            
            elif obj_type == "axes":
                code_lines.append(f'        {var_name} = Axes()')
            
            elif obj_type == "graph":
                code_lines.append(f'        {var_name} = Axes()')
                code_lines.append(f'        graph_{i} = {var_name}.plot(lambda x: x**2, color={color.upper()})')
            
            elif obj_type == "geometric_shape":
                # Parse shape from content
                if "circle" in content.lower():
                    code_lines.append(f'        {var_name} = Circle(color={color.upper()})')
                elif "square" in content.lower() or "rectangle" in content.lower():
                    code_lines.append(f'        {var_name} = Square(color={color.upper()})')
                elif "line" in content.lower():
                    code_lines.append(f'        {var_name} = Line(color={color.upper()})')
                else:
                    code_lines.append(f'        {var_name} = Circle(color={color.upper()})')
        
        return "\n".join(code_lines)
    
    def _generate_transformations(
        self,
        transformations: List[Dict[str, Any]],
        context: SharedContext
    ) -> str:
        """Generate code for transformations.
        
        Args:
            transformations: List of transformation specifications
            context: Shared context with animation speed
            
        Returns:
            Generated code string
        """
        if not transformations:
            return "        # No transformations"
        
        code_lines = []
        animation_speed = context.visual_style.animation_speed
        
        for trans in transformations:
            trans_type = trans.get("transformation_type", "fade_in")
            target = trans.get("target_object", "obj_0")
            timing = trans.get("timing", 0.0)
            
            # Map target_object reference to variable name
            # Simplified: assume target_object is like "equation_0" -> "obj_0"
            if "_" in target:
                parts = target.split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    var_name = f"obj_{parts[1]}"
                else:
                    var_name = "obj_0"
            else:
                var_name = "obj_0"
            
            # Generate transformation code
            if trans_type == "fade_in":
                code_lines.append(f'        self.play(FadeIn({var_name}), run_time={1.0/animation_speed})')
            
            elif trans_type == "fade_out":
                code_lines.append(f'        self.play(FadeOut({var_name}), run_time={1.0/animation_speed})')
            
            elif trans_type == "write":
                code_lines.append(f'        self.play(Write({var_name}), run_time={2.0/animation_speed})')
            
            elif trans_type == "morph":
                # Simplified morph (would need target object)
                code_lines.append(f'        self.play(Transform({var_name}, {var_name}.copy()), run_time={1.5/animation_speed})')
            
            elif trans_type == "highlight":
                code_lines.append(f'        self.play(Indicate({var_name}), run_time={0.5/animation_speed})')
            
            elif trans_type == "move":
                code_lines.append(f'        self.play({var_name}.animate.shift(UP), run_time={1.0/animation_speed})')
            
            elif trans_type == "scale":
                params = trans.get("parameters", {})
                scale_factor = params.get("scale_factor", 1.5)
                code_lines.append(f'        self.play({var_name}.animate.scale({scale_factor}), run_time={1.0/animation_speed})')
            
            # Add wait for timing
            if timing > 0:
                code_lines.append(f'        self.wait({timing})')
        
        return "\n".join(code_lines)
    
    def _validate_syntax(self, code: str) -> None:
        """Validate Python syntax using AST parser.
        
        Args:
            code: Python code string to validate
            
        Raises:
            SyntaxError: If code has syntax errors
        """
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise SyntaxError(f"Generated code has syntax error: {e}")
    
    def _format_code(
        self,
        code: str,
        scene_spec: Any,
        context: SharedContext
    ) -> str:
        """Format code with header comment and consistent style.
        
        Args:
            code: Generated code string
            scene_spec: Scene specification
            context: Shared context
            
        Returns:
            Formatted code string
        """
        # Add header comment
        header = f'''"""
Scene: {scene_spec.scene_id}
Narration: {scene_spec.narration[:100]}{"..." if len(scene_spec.narration) > 100 else ""}
Duration: {scene_spec.duration_estimate}s
Generated: {datetime.now().isoformat()}
"""

'''
        
        # Add imports
        imports = "\n".join(self._get_imports())
        
        # Combine
        formatted = f"{header}{imports}\n\n{code}"
        
        return formatted
    
    def _get_imports(self) -> List[str]:
        """Get required import statements.
        
        Returns:
            List of import statements
        """
        return [
            "from manim import *"
        ]
    
    def _get_class_name(self, scene_id: str) -> str:
        """Generate valid Python class name from scene_id.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Valid Python class name
        """
        # Replace non-alphanumeric characters with underscores
        class_name = re.sub(r'[^a-zA-Z0-9_]', '_', scene_id)
        
        # Ensure it starts with a letter or underscore
        if class_name and class_name[0].isdigit():
            class_name = f"Scene_{class_name}"
        elif not class_name:
            class_name = "Scene_default"
        
        # Capitalize first letter
        class_name = class_name[0].upper() + class_name[1:] if len(class_name) > 1 else class_name.upper()
        
        return class_name
    
    def _escape_latex(self, latex: str) -> str:
        """Escape LaTeX string for Python raw string literal.
        
        Args:
            latex: LaTeX string
            
        Returns:
            Escaped LaTeX string safe for r"..." literals
        """
        # For raw strings (r"..."), we need to:
        # 1. Replace double quotes with single quotes to avoid breaking the literal
        # 2. Handle trailing backslashes (can't end raw string with \)
        result = latex.replace('"', "'")
        
        # If string ends with odd number of backslashes, add a space
        # to avoid syntax error in raw string
        if result.endswith('\\') and not result.endswith('\\\\'):
            result += ' '
        
        return result
