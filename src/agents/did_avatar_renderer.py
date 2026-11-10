"""D-ID avatar renderer: the ``did-avatar`` render backend.

When a run selects ``render_backend="did-avatar"``, this agent replaces the
Manim render stage. Every narrated scene becomes a talking-professor clip:

    scene narration --POST /talks--> talk id --GET /talks/{id} (poll)--> done
        --> result_url --download--> <run>/avatar/<scene_id>.mp4  (video + voice)

D-ID speaks the narration itself with a stock voice, so the pipeline skips
TTS in this mode; the composer concatenates the clips keeping their audio.

Safety rules this module enforces:
- The portrait is an explicit public HTTPS image URL of a fictional or
  licensed avatar. Nothing here generates portraits or clones a voice.
- The API key comes from the environment (``DID_API_KEY``, or the older
  ``D_ID_API_KEY``) and is never logged, stored or put in an exception
  message: every string that leaves this module passes through ``redact``.
- The signed ``result_url`` is used for the download only and never stored.

HTTP goes through a small injectable transport (standard library by
default), so the hermetic test suite never reaches the network.
"""

import base64
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from src.agents.base import (
    Agent,
    AgentExecutionError,
    AgentInput,
    AgentOutput,
    BackoffStrategy,
    RetryPolicy,
)
from src.orchestrator.retry_policy import execute_with_retry

logger = logging.getLogger(__name__)

# ── configuration constants ─────────────────────────────────────────────
DID_API_BASE = "https://api.d-id.com"
TALKS_PATH = "/talks"
CREDITS_PATH = "/credits"
DID_KEY_ENV_NAMES = ("DID_API_KEY", "D_ID_API_KEY")  # documented name, then legacy alias
DEFAULT_VOICE_ID = "en-US-JennyNeural"  # Microsoft stock voice; no cloning
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_MAX_POLL_ATTEMPTS = 120          # 10 minutes per scene at the default interval
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0   # one API call
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 300.0  # one clip download
DEFAULT_MAX_ESTIMATED_SECONDS = 600.0    # narration budget per run (spend cap)
WORDS_PER_SECOND = 2.5                   # ~150 wpm, for the budget estimate
AVATAR_DIRNAME = "avatar"
REPORT_FILENAME = "did_avatar.json"

# Error codes. Transient ones are retried with the agent's backoff policy.
RATE_LIMITED = "DID_RATE_LIMITED"
SERVER_ERROR = "DID_SERVER_ERROR"
NETWORK_ERROR = "DID_NETWORK_ERROR"
UNAUTHORIZED = "DID_UNAUTHORIZED"
INSUFFICIENT_CREDITS = "DID_INSUFFICIENT_CREDITS"
BAD_REQUEST = "DID_BAD_REQUEST"
BAD_RESPONSE = "DID_BAD_RESPONSE"
TALK_FAILED = "DID_TALK_FAILED"
TIMEOUT = "DID_TIMEOUT"
MISSING_API_KEY = "MISSING_API_KEY"
INVALID_IMAGE_URL = "INVALID_AVATAR_IMAGE_URL"
BUDGET_EXCEEDED = "AVATAR_BUDGET_EXCEEDED"
TRANSIENT_ERRORS = [RATE_LIMITED, SERVER_ERROR, NETWORK_ERROR]
# Failures that would repeat for every scene: stop calling D-ID for the run.
RUN_FATAL_ERRORS = {UNAUTHORIZED, INSUFFICIENT_CREDITS}

PENDING_STATUSES = {"created", "started", "queued", "processing"}
FAILED_STATUSES = {"error", "rejected"}


# ── secrets and inputs ──────────────────────────────────────────────────

def resolve_did_api_key(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The D-ID key from ``DID_API_KEY``, falling back to ``D_ID_API_KEY``."""
    env = os.environ if environ is None else environ
    for name in DID_KEY_ENV_NAMES:
        value = (env.get(name) or "").strip()
        if value:
            return value
    return None


_AUTH_HEADER = re.compile(r"(?i)(authorization['\"]?\s*[:=]\s*['\"]?)(?:basic|bearer)?\s*[^\s,;'\"}]+")
_AUTH_SCHEME = re.compile(r"(?i)\b(basic|bearer)\s+[A-Za-z0-9+/=._:\-]{8,}")
_URL_QUERY = re.compile(r"(https?://[^\s?#'\"]+)\?[^\s'\"]*")


def redact(text: Any, secrets: Iterable[Optional[str]] = ()) -> str:
    """``text`` with keys, auth headers and signed URL query strings removed.

    Each secret is removed verbatim, base64-encoded, URL-quoted and per
    ``:``-separated part (a D-ID key is ``<base64 user>:<secret>``).
    """
    out = str(text)
    forms = set()
    for secret in secrets:
        if not secret:
            continue
        forms.update({
            secret,
            base64.b64encode(secret.encode("utf-8")).decode("ascii"),
            urllib.parse.quote(secret, safe=""),
        })
        forms.update(part for part in secret.split(":") if len(part) >= 8)
    for form in sorted(forms, key=len, reverse=True):
        out = out.replace(form, "[REDACTED]")
    out = _AUTH_HEADER.sub(r"\1[REDACTED]", out)
    out = _AUTH_SCHEME.sub(r"\1 [REDACTED]", out)
    return _URL_QUERY.sub(r"\1?[REDACTED]", out)


def validate_avatar_image_url(url: Optional[str]) -> str:
    """The portrait URL, or ValueError saying why it cannot be used.

    D-ID fetches the image itself, so it must be publicly reachable over
    HTTPS. Credentials in the URL are refused (they would be stored).
    """
    if not url or not str(url).strip():
        raise ValueError(
            "--avatar-image-url is required for the did-avatar backend: a public HTTPS URL "
            "of a fictional or licensed portrait."
        )
    value = str(url).strip()
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme.lower() != "https":
        raise ValueError(f"The avatar image URL must use https:// (got {parsed.scheme or 'no scheme'}).")
    if not parsed.hostname:
        raise ValueError("The avatar image URL has no host.")
    if parsed.username or parsed.password:
        raise ValueError("The avatar image URL must not contain credentials.")
    if parsed.hostname in {"localhost", "127.0.0.1", "::1"} or parsed.hostname.endswith(".local"):
        raise ValueError("The avatar image URL must be publicly reachable, not a local address.")
    return value


def display_url(url: Optional[str]) -> str:
    """A URL safe to store: no query string or fragment (they may be signed)."""
    if not url:
        return ""
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse(parsed._replace(query="", fragment=""))


def estimate_seconds(text: str) -> float:
    return len((text or "").split()) / WORDS_PER_SECOND


# ── errors and transport ────────────────────────────────────────────────

class DIDAPIError(AgentExecutionError):
    """A D-ID call failed. Messages are already redacted."""


Transport = Callable[[str, str, Dict[str, str], Optional[bytes], float], Tuple[int, bytes]]


def urllib_transport(method: str, url: str, headers: Dict[str, str], body: Optional[bytes], timeout: float) -> Tuple[int, bytes]:
    """Standard-library HTTP: returns (status, body); HTTP errors are statuses."""
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read() or b""
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise DIDAPIError(NETWORK_ERROR, f"Network error calling D-ID: {type(exc).__name__}: {reason}")


def _error_summary(body: bytes) -> str:
    """D-ID's ``{kind, description}`` error body, shortened."""
    try:
        data = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return body.decode("utf-8", "replace").strip()[:200]
    if isinstance(data, dict):
        kind = data.get("kind") or data.get("error") or ""
        description = data.get("description") or data.get("message") or ""
        if isinstance(kind, dict):
            kind, description = kind.get("kind", ""), kind.get("description", description)
        text = ": ".join(str(p) for p in (kind, description) if p)
        return (text or json.dumps(data))[:300]
    return str(data)[:300]


class DIDClient:
    """Minimal D-ID REST client. Every request goes through ``transport``."""

    def __init__(
        self,
        api_key: str,
        transport: Optional[Transport] = None,
        retry_policy: Optional[RetryPolicy] = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        download_timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
        base_url: str = DID_API_BASE,
    ):
        if not api_key:
            raise ValueError("A D-ID API key is required.")
        self._api_key = api_key
        self._transport = transport or urllib_transport
        self.retry_policy = retry_policy or DIDAvatarRendererAgent.default_retry_policy()
        self.request_timeout = request_timeout
        self.download_timeout = download_timeout
        self.base_url = base_url.rstrip("/")

    def __repr__(self) -> str:  # never show the key
        return f"DIDClient(base_url={self.base_url!r})"

    def _redact(self, text: Any) -> str:
        return redact(text, [self._api_key])

    def _headers(self) -> Dict[str, str]:
        return {
            # D-ID Basic auth takes the key exactly as issued.
            "Authorization": f"Basic {self._api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _once(self, method: str, path: str, payload: Optional[dict]) -> Any:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        try:
            status, raw = self._transport(method, self.base_url + path, self._headers(), body, self.request_timeout)
        except DIDAPIError as exc:
            raise DIDAPIError(exc.error_code, self._redact(exc.message))
        if 200 <= status < 300:
            try:
                return json.loads(raw.decode("utf-8")) if raw else {}
            except (ValueError, UnicodeDecodeError):
                raise DIDAPIError(BAD_RESPONSE, f"D-ID {method} {path} returned non-JSON (HTTP {status}).")
        summary = self._redact(_error_summary(raw))
        if status == 429:
            code = RATE_LIMITED
        elif status >= 500:
            code = SERVER_ERROR
        elif status in (401, 403):
            code = UNAUTHORIZED
        elif status == 402:
            code = INSUFFICIENT_CREDITS
        else:
            code = BAD_REQUEST
        raise DIDAPIError(code, f"D-ID {method} {path} failed with HTTP {status}: {summary}")

    def _call(self, method: str, path: str, payload: Optional[dict] = None) -> Any:
        return execute_with_retry(
            lambda: self._once(method, path, payload), self.retry_policy, f"D-ID {method} {path.split('/tlk_')[0]}",
        )

    def create_talk(self, image_url: str, text: str, voice_id: str = DEFAULT_VOICE_ID) -> str:
        data = self._call("POST", TALKS_PATH, {
            "source_url": image_url,
            "script": {
                "type": "text",
                "input": text,
                "provider": {"type": "microsoft", "voice_id": voice_id},
            },
            # Paste the animated face back onto the full portrait.
            "config": {"stitch": True},
        })
        talk_id = (data or {}).get("id") if isinstance(data, dict) else None
        if not talk_id:
            raise DIDAPIError(BAD_RESPONSE, "D-ID accepted the talk but returned no id.")
        return str(talk_id)

    def get_talk(self, talk_id: str) -> Dict[str, Any]:
        data = self._call("GET", f"{TALKS_PATH}/{urllib.parse.quote(talk_id, safe='')}")
        if not isinstance(data, dict):
            raise DIDAPIError(BAD_RESPONSE, "D-ID returned an unexpected talk status.")
        return data

    def remaining_credits(self) -> Optional[int]:
        data = self._call("GET", CREDITS_PATH)
        value = data.get("remaining") if isinstance(data, dict) else None
        return int(value) if isinstance(value, (int, float)) else None

    def download(self, url: str, dest: Path) -> Path:
        """Fetch the finished clip. The signed URL gets no Authorization header."""

        def once() -> bytes:
            try:
                status, raw = self._transport("GET", url, {"Accept": "*/*"}, None, self.download_timeout)
            except DIDAPIError as exc:
                raise DIDAPIError(exc.error_code, self._redact(exc.message))
            if status == 429:
                raise DIDAPIError(RATE_LIMITED, f"Clip download rate-limited (HTTP {status}).")
            if status >= 500:
                raise DIDAPIError(SERVER_ERROR, f"Clip download failed (HTTP {status}).")
            if not 200 <= status < 300 or not raw:
                raise DIDAPIError(BAD_RESPONSE, f"Clip download failed (HTTP {status}, {len(raw)} bytes).")
            return raw

        raw = execute_with_retry(once, self.retry_policy, "D-ID clip download")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(raw)
        tmp.replace(dest)
        return dest


# ── agent contract ──────────────────────────────────────────────────────

@dataclass
class AvatarScene:
    scene_id: str
    text: str


@dataclass
class AvatarSceneResult:
    scene_id: str
    status: str  # done | failed | skipped
    video_path: Optional[str] = None
    duration_seconds: Optional[float] = None
    talk_id: Optional[str] = None
    polls: int = 0
    error_code: Optional[str] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status == "done" and bool(self.video_path)


class DIDAvatarRendererInput(AgentInput):
    def __init__(
        self,
        scenes: List[AvatarScene],
        output_dir: str,
        image_url: str,
        voice_id: str = DEFAULT_VOICE_ID,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        max_poll_attempts: int = DEFAULT_MAX_POLL_ATTEMPTS,
        max_estimated_seconds: Optional[float] = DEFAULT_MAX_ESTIMATED_SECONDS,
    ):
        self.scenes = scenes
        self.output_dir = output_dir
        self.image_url = image_url
        self.voice_id = voice_id
        self.poll_interval_seconds = poll_interval_seconds
        self.max_poll_attempts = max_poll_attempts
        self.max_estimated_seconds = max_estimated_seconds


@dataclass
class DIDAvatarRendererOutput(AgentOutput):
    scene_results: List[AvatarSceneResult] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)

    @property
    def rendered(self) -> int:
        return sum(1 for r in self.scene_results if r.success)


class DIDAvatarRendererAgent(Agent):
    """Renders one D-ID talking-professor clip per narrated scene.

    Scene-level failures are isolated: a rejected or timed-out talk fails
    that scene only. Authentication and credit errors stop further calls
    for the run (they would fail every scene) and mark the rest skipped.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        strict: bool = False,
        client: Optional[DIDClient] = None,
        transport: Optional[Transport] = None,
        retry_policy: Optional[RetryPolicy] = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        download_timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.api_key = resolve_did_api_key() if api_key is None else api_key
        self.strict = strict
        self._retry_policy = retry_policy or self.default_retry_policy()
        self._client = client
        self._transport = transport
        self.request_timeout = request_timeout
        self.download_timeout = download_timeout
        self._sleep = sleep

    def __repr__(self) -> str:  # never show the key
        return f"DIDAvatarRendererAgent(strict={self.strict}, key_configured={bool(self.api_key)})"

    @staticmethod
    def default_retry_policy() -> RetryPolicy:
        return RetryPolicy(
            max_attempts=4,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay_seconds=2.0,
            max_delay_seconds=30.0,
            retryable_errors=list(TRANSIENT_ERRORS),
        )

    def get_retry_policy(self) -> RetryPolicy:
        # Transient errors are retried per request inside the client; the
        # stage itself is not re-run (that would re-render finished scenes).
        return RetryPolicy(max_attempts=1, retryable_errors=[])

    def validate_input(self, input_data: AgentInput) -> bool:
        return (
            isinstance(input_data, DIDAvatarRendererInput)
            and isinstance(input_data.scenes, list)
            and bool(input_data.output_dir)
        )

    @property
    def client(self) -> DIDClient:
        if self._client is None:
            self._client = DIDClient(
                self.api_key,
                transport=self._transport,
                retry_policy=self._retry_policy,
                request_timeout=self.request_timeout,
                download_timeout=self.download_timeout,
            )
        return self._client

    def _redact(self, text: Any) -> str:
        return redact(text, [self.api_key])

    def execute(self, input_data: AgentInput) -> DIDAvatarRendererOutput:
        if not self.validate_input(input_data):
            raise AgentExecutionError("INVALID_INPUT", "Expected DIDAvatarRendererInput with scenes and output_dir.")
        try:
            image_url = validate_avatar_image_url(input_data.image_url)
        except ValueError as exc:
            raise AgentExecutionError(INVALID_IMAGE_URL, str(exc))

        results: List[AvatarSceneResult] = []
        credits_before = credits_after = None
        if not self.api_key:
            message = "DID_API_KEY (or D_ID_API_KEY) is not set; the did-avatar backend cannot render."
            if self.strict:
                raise AgentExecutionError(MISSING_API_KEY, message + " Required in --strict mode.")
            logger.warning(message + " Every avatar scene is marked failed.")
            results = [
                AvatarSceneResult(scene.scene_id, "failed", error_code=MISSING_API_KEY, error=message)
                for scene in input_data.scenes
            ]
        else:
            credits_before = self._credits()
            results = self._render_all(input_data, image_url)
            if any(r.talk_id for r in results):
                credits_after = self._credits()

        output = DIDAvatarRendererOutput(scene_results=results)
        output.report = self._report(input_data, image_url, results, credits_before, credits_after)
        logger.info(
            "D-ID avatar: %d/%d scene clip(s) rendered (%d failed, %d skipped)",
            output.rendered, len(results),
            sum(1 for r in results if r.status == "failed"),
            sum(1 for r in results if r.status == "skipped"),
        )
        return output

    def _credits(self) -> Optional[int]:
        try:
            return self.client.remaining_credits()
        except Exception as exc:  # informational only
            logger.info("D-ID credit balance unavailable: %s", self._redact(exc))
            return None

    def _render_all(self, input_data: DIDAvatarRendererInput, image_url: str) -> List[AvatarSceneResult]:
        avatar_dir = Path(input_data.output_dir) / AVATAR_DIRNAME
        results: List[AvatarSceneResult] = []
        fatal: Optional[DIDAPIError] = None
        budget = input_data.max_estimated_seconds
        spent = 0.0
        for scene in input_data.scenes:
            if fatal is not None:
                results.append(AvatarSceneResult(
                    scene.scene_id, "skipped", error_code=fatal.error_code,
                    error="Not attempted after an earlier run-wide D-ID failure.",
                ))
                continue
            if not (scene.text or "").strip():
                results.append(AvatarSceneResult(scene.scene_id, "skipped", error="Scene has no narration."))
                continue
            estimate = estimate_seconds(scene.text)
            if budget is not None and spent + estimate > budget:
                results.append(AvatarSceneResult(
                    scene.scene_id, "skipped", error_code=BUDGET_EXCEEDED,
                    error=f"Over the {budget:g}s avatar narration budget (max_avatar_seconds).",
                ))
                continue
            spent += estimate
            try:
                results.append(self._render_scene(scene, input_data, image_url, avatar_dir))
            except DIDAPIError as exc:
                logger.warning("D-ID scene %s failed: [%s] %s", scene.scene_id, exc.error_code, self._redact(exc.message))
                results.append(AvatarSceneResult(
                    scene.scene_id, "failed", talk_id=getattr(exc, "talk_id", None),
                    polls=getattr(exc, "polls", 0), error_code=exc.error_code, error=self._redact(exc.message),
                ))
                if exc.error_code in RUN_FATAL_ERRORS:
                    fatal = exc
            except Exception as exc:  # isolation: never let one scene sink the run
                logger.warning("D-ID scene %s failed unexpectedly: %s", scene.scene_id, self._redact(exc))
                results.append(AvatarSceneResult(
                    scene.scene_id, "failed", error_code="DID_UNEXPECTED_ERROR",
                    error=self._redact(f"{type(exc).__name__}: {exc}"),
                ))
        return results

    def _render_scene(
        self,
        scene: AvatarScene,
        input_data: DIDAvatarRendererInput,
        image_url: str,
        avatar_dir: Path,
    ) -> AvatarSceneResult:
        talk_id = self.client.create_talk(image_url, scene.text, input_data.voice_id)
        logger.info("D-ID talk %s created for scene %s", talk_id, scene.scene_id)
        polls = 0
        status: Dict[str, Any] = {}
        for polls in range(1, max(1, int(input_data.max_poll_attempts)) + 1):
            try:
                status = self.client.get_talk(talk_id)
            except DIDAPIError as exc:
                exc.talk_id, exc.polls = talk_id, polls
                raise
            state = str(status.get("status", "")).lower()
            if state == "done":
                result_url = status.get("result_url")
                if not result_url:
                    error = DIDAPIError(BAD_RESPONSE, "D-ID reported the talk done without a result URL.")
                    error.talk_id, error.polls = talk_id, polls
                    raise error
                dest = self.client.download(str(result_url), avatar_dir / f"{scene.scene_id}.mp4")
                return AvatarSceneResult(
                    scene.scene_id, "done", video_path=str(dest),
                    duration_seconds=_clip_duration(dest, status.get("duration")),
                    talk_id=talk_id, polls=polls,
                )
            if state in FAILED_STATUSES:
                detail = status.get("error") or status.get("result") or {}
                summary = _error_summary(json.dumps(detail).encode("utf-8")) if detail else state
                error = DIDAPIError(TALK_FAILED, f"D-ID talk {state}: {self._redact(summary)}")
                error.talk_id, error.polls = talk_id, polls
                raise error
            if polls < input_data.max_poll_attempts:
                self._sleep(input_data.poll_interval_seconds)
        error = DIDAPIError(
            TIMEOUT,
            f"D-ID talk not done after {input_data.max_poll_attempts} polls "
            f"({input_data.poll_interval_seconds:g}s apart); last status: {status.get('status', 'unknown')}.",
        )
        error.talk_id, error.polls = talk_id, polls
        raise error

    def _report(self, input_data, image_url, results, credits_before, credits_after) -> Dict[str, Any]:
        scenes = []
        for result in results:
            entry = asdict(result)
            if entry.get("video_path"):
                try:
                    entry["video_path"] = Path(entry["video_path"]).relative_to(input_data.output_dir).as_posix()
                except ValueError:
                    entry["video_path"] = Path(entry["video_path"]).name
            entry["error"] = self._redact(entry["error"]) if entry.get("error") else None
            scenes.append(entry)
        report = {
            "backend": "did-avatar",
            "provider": "d-id",
            "endpoint": f"POST {DID_API_BASE}{TALKS_PATH}",
            "image_url": display_url(image_url),
            "voice_id": input_data.voice_id,
            "poll_interval_seconds": input_data.poll_interval_seconds,
            "max_poll_attempts": input_data.max_poll_attempts,
            "max_estimated_seconds": input_data.max_estimated_seconds,
            "rendered": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if r.status == "failed"),
            "skipped": sum(1 for r in results if r.status == "skipped"),
            "credits_before": credits_before,
            "credits_after": credits_after,
            "scenes": scenes,
        }
        # Belt and braces: nothing in the report may carry the key.
        return json.loads(self._redact(json.dumps(report)))


def _clip_duration(path: Path, reported: Any) -> Optional[float]:
    try:
        from src.utils.media_probe import probe

        info = probe(str(path))
        if info and info.duration:
            return round(float(info.duration), 3)
    except Exception:  # ffprobe missing: fall back to D-ID's figure
        pass
    try:
        return round(float(reported), 3) if reported is not None else None
    except (TypeError, ValueError):
        return None
