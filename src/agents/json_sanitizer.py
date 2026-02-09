"""JSON Sanitizer Agent for enforcing schema compliance on storyboard output.

This agent validates Storyboard JSON against the defined schema and attempts
automatic corrections for common violations before passing to Scene Controller.
"""

import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from src.agents.base import Agent, AgentExecutionError, AgentInput, AgentOutput, RetryPolicy
from src.schemas.storyboard import StoryboardJSON


@dataclass
class SanitizationCorrection:
    """Record of an automatic correction made during sanitization."""
    
    field_path: str
    violation_type: str
    original_value: Any
    corrected_value: Any
    description: str


@dataclass
class ValidationErrorDetail:
    """Detailed information about a validation error."""
    
    field_path: str
    violation_type: str
    expected: str
    actual: str
    message: str


class JSONSanitizerInput(AgentInput):
    """Input for JSON Sanitizer Agent.
    
    Attributes:
        storyboard_dict: Dictionary representation of Storyboard JSON (potentially non-compliant)
    """
    
    def __init__(self, storyboard_dict: Dict[str, Any]):
        self.storyboard_dict = storyboard_dict


class JSONSanitizerOutput(AgentOutput):
    """Output from JSON Sanitizer Agent.
    
    Attributes:
        storyboard: Validated and sanitized StoryboardJSON object
        corrections: List of automatic corrections applied
        was_corrected: Whether any corrections were made
    """
    
    def __init__(
        self,
        storyboard: StoryboardJSON,
        corrections: List[SanitizationCorrection],
        was_corrected: bool
    ):
        self.storyboard = storyboard
        self.corrections = corrections
        self.was_corrected = was_corrected


class SanitizationError(AgentOutput):
    """Error output when sanitization fails.
    
    Attributes:
        errors: List of validation errors that could not be corrected
        attempted_corrections: List of corrections that were attempted
    """
    
    def __init__(
        self,
        errors: List[ValidationErrorDetail],
        attempted_corrections: List[SanitizationCorrection]
    ):
        self.errors = errors
        self.attempted_corrections = attempted_corrections


class JSONSanitizerAgent(Agent):
    """Agent responsible for enforcing schema compliance on storyboard output.
    
    The JSON Sanitizer validates Storyboard JSON against the defined Pydantic schema
    and attempts automatic corrections for common violations:
    - Missing optional fields → Add with default values
    - Type mismatches for coercible types → Coerce (e.g., number string to number)
    - Duplicate scene_ids → Append numeric suffix to duplicates
    
    If validation passes after corrections, returns sanitized JSON with correction log.
    If validation still fails, returns validation error report.
    """
    
    def __init__(self):
        """Initialize JSON Sanitizer Agent."""
        pass
    
    def execute(self, input_data: JSONSanitizerInput) -> AgentOutput:
        """Execute JSON sanitization and validation.
        
        Args:
            input_data: JSONSanitizerInput containing storyboard dictionary
            
        Returns:
            JSONSanitizerOutput with sanitized storyboard OR SanitizationError
            
        Raises:
            AgentExecutionError: For unrecoverable failures (invalid JSON structure)
        """
        if not self.validate_input(input_data):
            raise AgentExecutionError(
                "INVALID_INPUT",
                "Input must be JSONSanitizerInput with storyboard_dict",
                {"input_type": type(input_data).__name__}
            )
        
        storyboard_dict = copy.deepcopy(input_data.storyboard_dict)
        corrections: List[SanitizationCorrection] = []
        
        # Apply automatic corrections BEFORE validation
        # This allows us to track what was corrected
        storyboard_dict, corrections = self._apply_corrections(storyboard_dict)
        
        # Attempt validation with corrections applied
        try:
            storyboard = StoryboardJSON(**storyboard_dict)
            return JSONSanitizerOutput(
                storyboard=storyboard,
                corrections=corrections,
                was_corrected=len(corrections) > 0
            )
        except ValidationError as e:
            # Validation still fails, return error report
            errors = self._extract_validation_errors(e)
            return SanitizationError(
                errors=errors,
                attempted_corrections=corrections
            )
    
    def validate_input(self, input_data: AgentInput) -> bool:
        """Validate input is JSONSanitizerInput with storyboard_dict.
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(input_data, JSONSanitizerInput):
            return False
        if not isinstance(input_data.storyboard_dict, dict):
            return False
        return True
    
    def get_retry_policy(self) -> RetryPolicy:
        """Return retry policy for JSON Sanitizer.
        
        Returns:
            RetryPolicy with max_attempts=1 (no retry, deterministic validation)
        """
        return RetryPolicy(max_attempts=1)
    
    def _apply_corrections(
        self,
        storyboard_dict: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[SanitizationCorrection]]:
        """Apply automatic corrections to storyboard dictionary.
        
        Args:
            storyboard_dict: Storyboard dictionary to correct
            
        Returns:
            Tuple of (corrected_dict, list_of_corrections)
        """
        corrections: List[SanitizationCorrection] = []
        
        # Correction 1: Type coercion for common mismatches (do this first to detect before Pydantic)
        corrections.extend(self._apply_type_coercion(storyboard_dict))
        
        # Correction 2: Add missing optional fields with defaults
        if "concept_hierarchy" not in storyboard_dict:
            storyboard_dict["concept_hierarchy"] = []
            corrections.append(SanitizationCorrection(
                field_path="concept_hierarchy",
                violation_type="missing_optional_field",
                original_value=None,
                corrected_value=[],
                description="Added missing optional field 'concept_hierarchy' with default empty list"
            ))
        
        # Correction 3: Fix duplicate scene_ids
        if "scenes" in storyboard_dict and isinstance(storyboard_dict["scenes"], list):
            scene_ids = []
            for i, scene in enumerate(storyboard_dict["scenes"]):
                if isinstance(scene, dict) and "scene_id" in scene:
                    scene_id = scene["scene_id"]
                    if scene_id in scene_ids:
                        # Duplicate found, append numeric suffix
                        original_id = scene_id
                        suffix = 1
                        while f"{scene_id}_{suffix}" in scene_ids:
                            suffix += 1
                        new_id = f"{scene_id}_{suffix}"
                        storyboard_dict["scenes"][i]["scene_id"] = new_id
                        scene_ids.append(new_id)
                        corrections.append(SanitizationCorrection(
                            field_path=f"scenes[{i}].scene_id",
                            violation_type="duplicate_scene_id",
                            original_value=original_id,
                            corrected_value=new_id,
                            description=f"Renamed duplicate scene_id '{original_id}' to '{new_id}'"
                        ))
                    else:
                        scene_ids.append(scene_id)
        
        # Correction 4: Add missing optional fields in scenes
        if "scenes" in storyboard_dict and isinstance(storyboard_dict["scenes"], list):
            for i, scene in enumerate(storyboard_dict["scenes"]):
                if isinstance(scene, dict):
                    # Add missing dependencies field
                    if "dependencies" not in scene:
                        scene["dependencies"] = []
                        corrections.append(SanitizationCorrection(
                            field_path=f"scenes[{i}].dependencies",
                            violation_type="missing_optional_field",
                            original_value=None,
                            corrected_value=[],
                            description=f"Added missing optional field 'dependencies' to scene {i}"
                        ))
                    
                    # Add missing emphasis_points in visual_intent
                    if "visual_intent" in scene and isinstance(scene["visual_intent"], dict):
                        if "emphasis_points" not in scene["visual_intent"]:
                            scene["visual_intent"]["emphasis_points"] = []
                            corrections.append(SanitizationCorrection(
                                field_path=f"scenes[{i}].visual_intent.emphasis_points",
                                violation_type="missing_optional_field",
                                original_value=None,
                                corrected_value=[],
                                description=f"Added missing optional field 'emphasis_points' to scene {i}"
                            ))
        
        return storyboard_dict, corrections
    
    def _apply_type_coercion(
        self,
        storyboard_dict: Dict[str, Any]
    ) -> List[SanitizationCorrection]:
        """Apply type coercion for common type mismatches.
        
        Args:
            storyboard_dict: Storyboard dictionary to correct (modified in place)
            
        Returns:
            List of corrections applied
        """
        corrections: List[SanitizationCorrection] = []
        
        # Coerce page_count to int if it's a string
        if "pdf_metadata" in storyboard_dict:
            metadata = storyboard_dict["pdf_metadata"]
            if isinstance(metadata, dict) and "page_count" in metadata:
                page_count = metadata["page_count"]
                if isinstance(page_count, str) and page_count.isdigit():
                    original = page_count
                    metadata["page_count"] = int(page_count)
                    corrections.append(SanitizationCorrection(
                        field_path="pdf_metadata.page_count",
                        violation_type="type_mismatch",
                        original_value=original,
                        corrected_value=int(page_count),
                        description=f"Coerced page_count from string '{original}' to int"
                    ))
        
        # Coerce numeric fields in scenes
        if "scenes" in storyboard_dict and isinstance(storyboard_dict["scenes"], list):
            for i, scene in enumerate(storyboard_dict["scenes"]):
                if isinstance(scene, dict):
                    # Coerce scene_index
                    if "scene_index" in scene:
                        scene_index = scene["scene_index"]
                        if isinstance(scene_index, str) and scene_index.isdigit():
                            original = scene_index
                            scene["scene_index"] = int(scene_index)
                            corrections.append(SanitizationCorrection(
                                field_path=f"scenes[{i}].scene_index",
                                violation_type="type_mismatch",
                                original_value=original,
                                corrected_value=int(scene_index),
                                description=f"Coerced scene_index from string to int"
                            ))
                    
                    # Coerce duration_estimate
                    if "duration_estimate" in scene:
                        duration = scene["duration_estimate"]
                        if isinstance(duration, str):
                            try:
                                original = duration
                                scene["duration_estimate"] = float(duration)
                                corrections.append(SanitizationCorrection(
                                    field_path=f"scenes[{i}].duration_estimate",
                                    violation_type="type_mismatch",
                                    original_value=original,
                                    corrected_value=float(duration),
                                    description=f"Coerced duration_estimate from string to float"
                                ))
                            except ValueError:
                                pass  # Cannot coerce, will fail validation
        
        return corrections
    
    def _extract_validation_errors(
        self,
        validation_error: ValidationError
    ) -> List[ValidationErrorDetail]:
        """Extract detailed error information from Pydantic ValidationError.
        
        Args:
            validation_error: Pydantic ValidationError
            
        Returns:
            List of ValidationErrorDetail objects
        """
        errors: List[ValidationErrorDetail] = []
        
        for error in validation_error.errors():
            field_path = ".".join(str(loc) for loc in error["loc"])
            error_type = error["type"]
            message = error["msg"]
            
            # Extract expected and actual values from context
            ctx = error.get("ctx", {})
            expected = str(ctx.get("expected", "unknown"))
            actual = str(ctx.get("actual", "unknown"))
            
            errors.append(ValidationErrorDetail(
                field_path=field_path,
                violation_type=error_type,
                expected=expected,
                actual=actual,
                message=message
            ))
        
        return errors
