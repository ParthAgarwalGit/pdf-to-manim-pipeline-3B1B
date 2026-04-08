"""Text-to-Speech Agent for generating narration audio using OpenAI's TTS API.

This agent converts narration text to MP3 audio files using OpenAI's TTS models,
with support for different voices and quality levels. It includes cost tracking,
retry logic with exponential backoff, and graceful handling of missing API keys.
"""

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.agents.base import Agent, AgentExecutionError, AgentInput, AgentOutput, BackoffStrategy, RetryPolicy
from src.utils.cost_tracker import CostTracker
from src.utils.ffmpeg_finder import get_ffprobe


logger = logging.getLogger(__name__)


@dataclass
class TTSInput(AgentInput):
    """Input for Text-to-Speech Agent.

    Attributes:
        narration: Text content to convert to speech
        scene_id: Identifier for the scene
        output_dir: Directory where audio file should be saved
        quality: Quality level ("standard" for tts-1, "high" for tts-1-hd)
        voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
    """

    def __init__(
        self,
        narration: str,
        scene_id: str,
        output_dir: str,
        quality: str = "standard",
        voice: str = "alloy"
    ):
        self.narration = narration
        self.scene_id = scene_id
        self.output_dir = output_dir
        self.quality = quality
        self.voice = voice


@dataclass
class TTSOutput(AgentOutput):
    """Output from Text-to-Speech Agent.

    Attributes:
        audio_file_path: Path to generated MP3 audio file
        duration_seconds: Duration of generated audio in seconds
        scene_id: Scene identifier
        cost_usd: Cost of the TTS API call in USD
    """

    def __init__(
        self,
        audio_file_path: str,
        duration_seconds: float,
        scene_id: str,
        cost_usd: float = 0.0
    ):
        self.audio_file_path = audio_file_path
        self.duration_seconds = duration_seconds
        self.scene_id = scene_id
        self.cost_usd = cost_usd


class TTSError(AgentOutput):
    """Error output when TTS generation fails.

    Attributes:
        scene_id: Scene identifier
        error: Error message
    """

    def __init__(self, scene_id: str, error: str):
        self.scene_id = scene_id
        self.error = error


class TTSAgent(Agent):
    """Agent for generating speech audio from text using OpenAI's TTS API.

    The TTS Agent:
    - Accepts narration text and scene metadata
    - Calls OpenAI's TTS API (tts-1 or tts-1-hd)
    - Saves audio as MP3 files to specified output directory
    - Estimates audio duration using character count heuristic
    - Tracks API costs via CostTracker
    - Gracefully handles missing API keys with logging
    - Implements retry policy with exponential backoff for transient failures

    Supported voices: alloy, echo, fable, onyx, nova, shimmer
    """

    # OpenAI TTS API pricing (per 1M characters as of 2026)
    TTS_COST_PER_1M_CHARS = 15.00

    # Approximate speaking rate for duration estimation (chars per second)
    # This is a rough estimate; actual duration will be provided by OpenAI
    CHARS_PER_SECOND = 12.5

    # Valid configuration options
    VALID_MODELS = {"tts-1", "tts-1-hd"}
    VALID_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
    VALID_QUALITIES = {"standard", "high"}

    def __init__(self, max_retries: int = 3, strict: bool = False):
        """Initialize TTS Agent.

        Args:
            max_retries: Maximum number of retry attempts for API calls
            strict: If True, raise immediately when API key is missing
                    instead of returning TTSError and skipping generation.
        """
        super().__init__()
        self.max_retries = max_retries
        self.strict = strict

        # Initialize OpenAI client if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
            except ImportError:
                if self.strict:
                    raise AgentExecutionError(
                        "MISSING_DEPENDENCY",
                        "openai package is required in --strict mode for TTS"
                    )
                logger.warning("OpenAI package not installed. TTS will be skipped.")
                self.client = None
        else:
            if self.strict:
                raise AgentExecutionError(
                    "MISSING_API_KEY",
                    "OPENAI_API_KEY is required in --strict mode for TTS"
                )
            logger.warning("OPENAI_API_KEY not found. TTS will be skipped.")
            self.client = None

        # Get cost tracker for tracking API calls
        self.cost_tracker = CostTracker.get_instance()

    def execute(self, input_data: TTSInput) -> AgentOutput:
        """Execute TTS audio generation.

        Args:
            input_data: TTSInput with narration text and metadata

        Returns:
            TTSOutput with audio file path and duration OR TTSError on failure

        Raises:
            AgentExecutionError: For unrecoverable failures
        """
        if not self.validate_input(input_data):
            raise AgentExecutionError(
                "INVALID_INPUT",
                "Input must be TTSInput with valid narration, scene_id, and output_dir",
                {"input_type": type(input_data).__name__}
            )

        # Check if client is available
        if not self.client:
            logger.warning(f"OpenAI client not available. Skipping TTS for scene {input_data.scene_id}")
            return TTSError(
                scene_id=input_data.scene_id,
                error="OpenAI API key not configured or OpenAI package not installed"
            )

        try:
            # Validate configuration
            self._validate_config(input_data)

            # Create output directory if it doesn't exist
            output_path = Path(input_data.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Determine model based on quality setting
            model = "tts-1-hd" if input_data.quality == "high" else "tts-1"

            # Generate audio file with retry logic
            audio_file_path = None
            for attempt in range(self.max_retries):
                try:
                    audio_file_path = self._generate_audio(
                        narration=input_data.narration,
                        scene_id=input_data.scene_id,
                        output_dir=input_data.output_dir,
                        model=model,
                        voice=input_data.voice
                    )
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        # Calculate backoff delay
                        delay = self._calculate_backoff(attempt)
                        logger.info(f"TTS attempt {attempt + 1} failed. Retrying in {delay}s. Error: {e}")
                        time.sleep(delay)
                    else:
                        raise

            if not audio_file_path:
                raise AgentExecutionError(
                    "TTS_GENERATION_FAILED",
                    "Failed to generate audio file after retries",
                    {"scene_id": input_data.scene_id}
                )

            # Measure actual duration from the generated file; fall back to
            # heuristic estimate when ffprobe is unavailable.
            duration_seconds = self._probe_duration(audio_file_path)
            if duration_seconds <= 0:
                duration_seconds = self._estimate_duration(input_data.narration)

            # Track cost
            char_count = len(input_data.narration)
            cost = self._calculate_cost(char_count)
            self.cost_tracker.track_call(
                provider="openai",
                model=model,
                input_tokens=char_count,  # Using character count as proxy for tokens
                output_tokens=0,  # TTS doesn't have output tokens
                call_type="tts"
            )

            logger.info(f"Generated TTS audio for scene {input_data.scene_id}: {audio_file_path} ({duration_seconds:.2f}s, ${cost:.4f})")

            return TTSOutput(
                audio_file_path=audio_file_path,
                duration_seconds=duration_seconds,
                scene_id=input_data.scene_id,
                cost_usd=cost
            )

        except AgentExecutionError:
            raise
        except Exception as e:
            logger.error(f"Unrecoverable error during TTS generation for scene {input_data.scene_id}: {e}")
            return TTSError(
                scene_id=input_data.scene_id,
                error=str(e)
            )

    def validate_input(self, input_data: AgentInput) -> bool:
        """Validate input is TTSInput with required fields.

        Args:
            input_data: Input to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(input_data, TTSInput):
            return False

        if not input_data.narration or not isinstance(input_data.narration, str):
            return False

        if not input_data.scene_id or not isinstance(input_data.scene_id, str):
            return False

        if not input_data.output_dir or not isinstance(input_data.output_dir, str):
            return False

        return True

    def get_retry_policy(self) -> RetryPolicy:
        """Return retry policy for TTS Agent.

        Returns:
            RetryPolicy with exponential backoff for transient failures
        """
        return RetryPolicy(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay_seconds=1.0,
            max_delay_seconds=30.0,
            retryable_errors=["RATE_LIMIT_EXCEEDED", "API_TIMEOUT", "SERVICE_UNAVAILABLE"]
        )

    def process_all_scenes(
        self,
        scenes: List[dict],
        output_dir: str,
        quality: str = "standard",
        voice: str = "alloy"
    ) -> List[AgentOutput]:
        """Process TTS generation for multiple scenes.

        This batch method processes all scene narrations in sequence,
        accumulating results and errors.

        Args:
            scenes: List of scene dictionaries with keys:
                - scene_id (str): Unique scene identifier
                - narration (str): Text narration to convert to speech
            output_dir: Base directory for all audio output files
            quality: Quality level ("standard" or "high")
            voice: Voice to use for all scenes

        Returns:
            List of TTSOutput and TTSError objects (one per scene)
        """
        results = []

        logger.info(f"Processing TTS for {len(scenes)} scenes")

        for scene in scenes:
            scene_id = scene.get("scene_id")
            narration = scene.get("narration")

            if not scene_id or not narration:
                logger.warning(f"Skipping scene with missing scene_id or narration: {scene}")
                results.append(TTSError(
                    scene_id=scene_id or "unknown",
                    error="Missing scene_id or narration"
                ))
                continue

            # Create scene-specific output directory
            scene_output_dir = os.path.join(output_dir, str(scene_id))

            # Generate TTS for this scene
            tts_input = TTSInput(
                narration=narration,
                scene_id=scene_id,
                output_dir=scene_output_dir,
                quality=quality,
                voice=voice
            )

            output = self.execute(tts_input)
            results.append(output)

        # Log summary
        success_count = sum(1 for r in results if isinstance(r, TTSOutput))
        error_count = sum(1 for r in results if isinstance(r, TTSError))
        total_cost = sum(r.cost_usd for r in results if isinstance(r, TTSOutput))

        logger.info(f"TTS batch complete: {success_count} succeeded, {error_count} failed. Total cost: ${total_cost:.4f}")

        return results

    def _generate_audio(
        self,
        narration: str,
        scene_id: str,
        output_dir: str,
        model: str,
        voice: str
    ) -> str:
        """Call OpenAI TTS API and save audio file.

        Args:
            narration: Text to convert to speech
            scene_id: Scene identifier for logging
            output_dir: Directory to save audio file
            model: TTS model to use (tts-1 or tts-1-hd)
            voice: Voice to use

        Returns:
            Path to generated audio file

        Raises:
            Exception: If API call fails
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        # Generate audio using OpenAI TTS API
        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=narration,
            response_format="mp3"
        )

        # Save audio to file
        output_path = Path(output_dir) / f"{scene_id}.mp3"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(response.content)

        return str(output_path)

    def _validate_config(self, input_data: TTSInput) -> None:
        """Validate configuration parameters.

        Args:
            input_data: Input to validate

        Raises:
            ValueError: If configuration is invalid
        """
        if input_data.quality not in self.VALID_QUALITIES:
            raise ValueError(
                f"Invalid quality: {input_data.quality}. "
                f"Must be one of {self.VALID_QUALITIES}"
            )

        if input_data.voice not in self.VALID_VOICES:
            raise ValueError(
                f"Invalid voice: {input_data.voice}. "
                f"Must be one of {self.VALID_VOICES}"
            )

    @staticmethod
    def _probe_duration(file_path: str) -> float:
        """Return actual audio duration via ffprobe, or -1.0 on failure."""
        try:
            result = subprocess.run(
                [
                    get_ffprobe(), "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    file_path,
                ],
                capture_output=True, text=True, timeout=10,
            )
            return float(result.stdout.strip())
        except Exception:
            return -1.0

    def _estimate_duration(self, narration: str) -> float:
        """Estimate audio duration from narration text.

        Uses a simple character-per-second heuristic. Actual duration
        will be determined by the audio file.

        Args:
            narration: Text to estimate duration for

        Returns:
            Estimated duration in seconds
        """
        char_count = len(narration)
        estimated_seconds = char_count / self.CHARS_PER_SECOND
        return max(0.1, estimated_seconds)  # Minimum 0.1 seconds

    def _calculate_cost(self, char_count: int) -> float:
        """Calculate API cost based on character count.

        Args:
            char_count: Number of characters in narration

        Returns:
            Estimated cost in USD
        """
        cost = (char_count / 1_000_000) * self.TTS_COST_PER_1M_CHARS
        return round(cost, 4)

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay.

        Uses formula: min(base_delay * 2^attempt, max_delay)

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        policy = self.get_retry_policy()
        delay = policy.base_delay_seconds * (2 ** attempt)
        return min(delay, policy.max_delay_seconds)
