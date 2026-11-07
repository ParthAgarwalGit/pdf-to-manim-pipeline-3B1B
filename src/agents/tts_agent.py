"""Text-to-Speech Agent for generating narration audio using OpenAI's TTS API.

This agent converts narration text to MP3 audio files using OpenAI's TTS models,
with support for different voices and quality levels. It includes cost tracking,
retry logic with exponential backoff, and graceful handling of missing API keys.
"""

import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]

from src.agents.base import Agent, AgentExecutionError, AgentInput, AgentOutput, BackoffStrategy, RetryPolicy
from src.utils.cost_tracker import CostTracker
from src.utils.ffmpeg_finder import get_ffmpeg, get_ffprobe


logger = logging.getLogger(__name__)


# OpenAI TTS rejects any single request over 4096 characters with HTTP 400.
# We pick 4000 as a safety buffer and split longer narration on sentence
# boundaries, generating one MP3 per chunk and concatenating with ffmpeg.
TTS_MAX_CHARS = 4000

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def _split_for_tts(text: str, max_chars: int = TTS_MAX_CHARS) -> List[str]:
    """Split narration into chunks ≤ max_chars on sentence boundaries.

    Falls back to whitespace boundaries when a single sentence exceeds the
    limit, and to a hard char-cut as last resort (only for pathological
    input with no whitespace at all).
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = [s for s in _SENTENCE_BREAK.split(text) if s.strip()]
    chunks: List[str] = []
    current = ""

    def _flush() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    def _pack_oversized(piece: str) -> None:
        """Split a single oversized piece on whitespace, then on hard char cuts."""
        nonlocal current
        words = piece.split()
        for w in words:
            if len(w) > max_chars:
                # Pathological: a single token longer than max_chars.
                # Hard-cut every max_chars characters.
                _flush()
                for i in range(0, len(w), max_chars):
                    chunks.append(w[i:i + max_chars])
                continue
            cand = (current + " " + w).strip() if current else w
            if len(cand) <= max_chars:
                current = cand
            else:
                _flush()
                current = w

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        candidate = (current + " " + s).strip() if current else s
        if len(candidate) <= max_chars:
            current = candidate
            continue
        # Adding this sentence would overflow → flush current chunk.
        _flush()
        if len(s) <= max_chars:
            current = s
        else:
            # Sentence itself is too long — fall back to word packing.
            _pack_oversized(s)

    _flush()
    return chunks


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
        voice: str = "alloy",
        beats: Optional[List[Tuple[str, str]]] = None,
        pauses: Optional[Dict[str, float]] = None,
    ):
        self.narration = narration
        self.scene_id = scene_id
        self.output_dir = output_dir
        self.quality = quality
        self.voice = voice
        # (beat_id, text) in order. When given, each beat is synthesized
        # separately so the output carries a beat-level timing track.
        self.beats = beats or None
        self.pauses = pauses or {}  # beat_id -> seconds of silence after it


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
        cost_usd: float = 0.0,
        beat_timings: Optional[Dict[str, List[float]]] = None,
    ):
        self.audio_file_path = audio_file_path
        self.duration_seconds = duration_seconds
        self.scene_id = scene_id
        self.cost_usd = cost_usd
        # beat_id -> [start, end] seconds within this audio file.
        self.beat_timings = beat_timings or {}


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
            if openai is None:
                if self.strict:
                    raise AgentExecutionError(
                        "MISSING_DEPENDENCY",
                        "openai package is required in --strict mode for TTS"
                    )
                logger.warning("OpenAI package not installed. TTS will be skipped.")
                self.client = None
            else:
                self.client = openai.OpenAI(api_key=api_key)
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
            beat_timings: Dict[str, List[float]] = {}
            for attempt in range(self.max_retries):
                try:
                    if input_data.beats:
                        audio_file_path, beat_timings = self._generate_beat_audio(
                            beats=input_data.beats,
                            scene_id=input_data.scene_id,
                            output_dir=input_data.output_dir,
                            model=model,
                            voice=input_data.voice,
                            pauses=input_data.pauses,
                        )
                    else:
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
            # NOTE: CostTracker.track_call expects token counts. For TTS the
            # billable unit is characters, not tokens; we pass char_count as
            # input_tokens so the tracker records *something* proportional to
            # cost.  A future CostTracker revision should add a dedicated
            # `characters` field.
            self.cost_tracker.track_call(
                provider="openai",
                model=model,
                input_tokens=char_count,
                output_tokens=0,
                call_type="tts",
            )

            logger.info(f"Generated TTS audio for scene {input_data.scene_id}: {audio_file_path} ({duration_seconds:.2f}s, ${cost:.4f})")

            return TTSOutput(
                audio_file_path=audio_file_path,
                duration_seconds=duration_seconds,
                scene_id=input_data.scene_id,
                cost_usd=cost,
                beat_timings=beat_timings,
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
                voice=voice,
                beats=scene.get("beats"),
                pauses=scene.get("pauses"),
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

        For narrations > TTS_MAX_CHARS the text is split on sentence
        boundaries, each chunk is generated as its own MP3, and the
        chunks are concatenated with ffmpeg into the final file.

        Args:
            narration: Text to convert to speech
            scene_id: Scene identifier for logging
            output_dir: Directory to save audio file
            model: TTS model to use (tts-1 or tts-1-hd)
            voice: Voice to use

        Returns:
            Path to generated audio file

        Raises:
            Exception: If any API call or ffmpeg concat fails
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        output_path = Path(output_dir) / f"{scene_id}.mp3"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        chunks = _split_for_tts(narration)
        if not chunks:
            raise RuntimeError(f"Empty narration for scene {scene_id}")

        if len(chunks) == 1:
            # Fast path — single API call.
            response = self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=chunks[0],
                response_format="mp3",
            )
            with open(output_path, "wb") as f:
                f.write(response.content)
            return str(output_path)

        logger.info(
            f"Scene {scene_id}: narration {len(narration)} chars > {TTS_MAX_CHARS} "
            f"limit, splitting into {len(chunks)} TTS calls and concatenating."
        )

        with tempfile.TemporaryDirectory() as tmp:
            part_paths: List[str] = []
            for i, chunk in enumerate(chunks):
                response = self.client.audio.speech.create(
                    model=model,
                    voice=voice,
                    input=chunk,
                    response_format="mp3",
                )
                part = Path(tmp) / f"part_{i:03d}.mp3"
                with open(part, "wb") as f:
                    f.write(response.content)
                part_paths.append(str(part))

            # ffmpeg concat demuxer wants a list file with single-quoted paths.
            list_file = Path(tmp) / "parts.txt"
            list_lines = [
                "file '" + p.replace("'", "'\\''") + "'"
                for p in part_paths
            ]
            list_file.write_text("\n".join(list_lines), encoding="utf-8")

            cmd = [
                get_ffmpeg(), "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg concat failed for scene {scene_id}: {result.stderr.strip()[:500]}"
                )

        return str(output_path)

    def _generate_beat_audio(
        self,
        beats: List[Tuple[str, str]],
        scene_id: str,
        output_dir: str,
        model: str,
        voice: str,
        pauses: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, Dict[str, List[float]]]:
        """Synthesize each beat separately and join them into the scene audio.

        Returns the audio path and the beat-level timing track
        (``beat_id -> [start, end]`` seconds). Per-part durations are measured
        and scaled to the joined file's measured length, so the track lines up
        with what is actually played. ``pauses`` adds silence after a beat
        (professor pacing edits).
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        output_path = Path(output_dir) / f"{scene_id}.mp3"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pauses = pauses or {}

        with tempfile.TemporaryDirectory() as tmp:
            parts: List[str] = []
            spans: List[Tuple[str, int, int]] = []  # beat_id, first part, end part
            for beat_id, text in beats:
                first = len(parts)
                for chunk in _split_for_tts(text):
                    response = self.client.audio.speech.create(
                        model=model, voice=voice, input=chunk, response_format="mp3",
                    )
                    part = Path(tmp) / f"part_{len(parts):03d}.mp3"
                    part.write_bytes(response.content)
                    parts.append(str(part))
                end = len(parts)
                pause = float(pauses.get(beat_id, 0.0) or 0.0)
                if pause > 0 and parts:
                    silence = Path(tmp) / f"part_{len(parts):03d}.mp3"
                    self._write_silence(silence, pause, like=parts[-1])
                    parts.append(str(silence))
                spans.append((beat_id, first, end))
            if not parts:
                raise RuntimeError(f"Empty narration for scene {scene_id}")

            durations = [self._probe_duration(p) for p in parts]
            if len(parts) == 1:
                output_path.write_bytes(Path(parts[0]).read_bytes())
            else:
                self._concat_mp3(parts, output_path, tmp, scene_id)

        total = self._probe_duration(str(output_path))
        measured = sum(d for d in durations if d > 0)
        if any(d <= 0 for d in durations) or measured <= 0:
            # No ffprobe: fall back to the character-rate estimate per beat.
            durations = [self._estimate_duration(t) for _, t in beats]
            spans = [(b, i, i + 1) for i, (b, _) in enumerate(beats)]
            measured = sum(durations)
        scale = total / measured if total > 0 and measured > 0 else 1.0
        starts = [0.0]
        for duration in durations:
            starts.append(starts[-1] + duration * scale)
        timings = {
            beat_id: [round(starts[first], 3), round(starts[end], 3)]
            for beat_id, first, end in spans
        }
        return str(output_path), timings

    @staticmethod
    def _write_silence(path: Path, seconds: float, like: str) -> None:
        """MP3 silence matching ``like``'s sample rate, so stream-copy joins work."""
        sample_rate = 24000
        try:
            from src.utils.media_probe import probe
            info = probe(like)
            if info and info.audio and info.audio.sample_rate:
                sample_rate = info.audio.sample_rate
        except Exception:
            pass
        result = subprocess.run(
            [
                get_ffmpeg(), "-y", "-f", "lavfi",
                "-i", f"anullsrc=r={sample_rate}:cl=mono", "-t", f"{seconds:.3f}",
                "-c:a", "libmp3lame", "-b:a", "64k", str(path),
            ],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg silence failed: {result.stderr.strip()[:300]}")

    @staticmethod
    def _concat_mp3(parts: List[str], output_path: Path, tmp: str, scene_id: str) -> None:
        list_file = Path(tmp) / "parts.txt"
        # ffmpeg concat demuxer wants a list file with single-quoted paths.
        list_lines = ["file '" + p.replace("'", "'\\''") + "'" for p in parts]
        list_file.write_text("\n".join(list_lines), encoding="utf-8")
        # Re-encode at one constant bitrate: stream-copying parts of different
        # bitrates (speech vs. generated silence) leaves an MP3 whose duration
        # ffprobe estimates wrongly, which would skew the timing track.
        result = subprocess.run(
            [
                get_ffmpeg(), "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file), "-c:a", "libmp3lame", "-b:a", "128k",
                str(output_path),
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg concat failed for scene {scene_id}: {result.stderr.strip()[:500]}"
            )

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
