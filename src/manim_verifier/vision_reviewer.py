"""Provider-neutral visual repair interface and implementations."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from pathlib import Path
from src.manim_verifier.models import RepairResult, VerificationReport


class VisionReviewer(ABC):
    @abstractmethod
    async def repair(
        self,
        *,
        brief: str,
        storyboard: str,
        code: str,
        report: VerificationReport,
        frame_paths: list[Path],
    ) -> RepairResult:
        """Return one complete executable Manim script, never a patch."""


class MockVisionReviewer(VisionReviewer):
    """Deterministic reviewer used by offline tests."""

    def __init__(self, replacement_code: str):
        self.replacement_code = replacement_code
        self.calls: list[dict] = []

    async def repair(self, **kwargs) -> RepairResult:
        self.calls.append(kwargs)
        return RepairResult(code=self.replacement_code, provider="mock", model="mock")


class GeminiVisionReviewer(VisionReviewer):
    """Gemini multimodal repair provider. Construction performs no API call."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.7-flash",
        timeout_seconds: int = 180,
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini visual repair.")
        from google import genai
        from google.genai import types

        # Without a timeout one stalled request hangs the whole pipeline run.
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_seconds * 1000),
        )
        self.model = model

    @staticmethod
    def resolve_api_key(env_path: Path = Path(".env"), allow_environ: bool = False) -> str:
        """Find GEMINI_API_KEY in ``env_path``, then (if allowed) os.environ.

        The standalone CLI reads only the file it is given. The pipeline sets
        ``allow_environ`` because src/cli.py loads keys.env into the process
        environment, which is also where code generation reads the key.
        """
        import os

        from dotenv import dotenv_values

        values = dotenv_values(env_path) if Path(env_path).is_file() else {}
        key = values.get("GEMINI_API_KEY")
        if not key and allow_environ:
            key = os.environ.get("GEMINI_API_KEY")
        if not key:
            where = f"{env_path} or the environment" if allow_environ else str(env_path)
            raise ValueError(f"GEMINI_API_KEY was not found in {where}.")
        return str(key)

    @classmethod
    def from_env(
        cls,
        env_path: Path = Path(".env"),
        allow_environ: bool = False,
    ) -> "GeminiVisionReviewer":
        return cls(api_key=cls.resolve_api_key(env_path, allow_environ=allow_environ))

    async def repair(
        self,
        *,
        brief: str,
        storyboard: str,
        code: str,
        report: VerificationReport,
        frame_paths: list[Path],
    ) -> RepairResult:
        return await asyncio.to_thread(
            self._repair_sync,
            brief,
            storyboard,
            code,
            report,
            frame_paths,
        )

    def _repair_sync(
        self,
        brief: str,
        storyboard: str,
        code: str,
        report: VerificationReport,
        frame_paths: list[Path],
    ) -> RepairResult:
        from google.genai import types

        issue_text = "\n".join(
            f"- [{issue.severity}] {issue.code}: {issue.message}"
            for issue in report.issues
        ) or "- No structured issue was emitted. Inspect the screenshots carefully."
        prompt = f"""You repair raw Manim Community Python after a preview render.

BRIEF:
{brief}

STORYBOARD:
{storyboard}

RENDER ERROR:
{report.render_error or '(none)'}

VERIFIER ISSUES:
{issue_text}

CURRENT SCRIPT:
{code}

Preserve correct mathematics and the intended visual story. Make the smallest
changes needed to fix the reported problems. Return one complete executable
Manim Python script only. Keep exactly one Scene subclass and only the approved
Manim/numpy imports already present. Do not add file, process, environment, or
network access. Every created visual object must retain or gain an immediately
preceding '# Purpose [beat_ids]: ...' comment. Do not return a patch,
explanation, or markdown fence.
"""
        contents: list = [prompt]
        for frame_path in frame_paths:
            contents.append(types.Part.from_bytes(
                data=frame_path.read_bytes(),
                mime_type="image/png",
            ))

        start = time.monotonic()
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(max_output_tokens=12000),
        )
        elapsed = time.monotonic() - start
        usage = getattr(response, "usage_metadata", None)
        return RepairResult(
            code=(response.text or "").strip(),
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            latency_seconds=elapsed,
            provider="google",
            model=self.model,
        )
