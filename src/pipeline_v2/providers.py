"""Budgeted, cached, provider-neutral model calls for V2."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

from .observability import RunArtifacts
from .state import BudgetExceeded, RunStore


T = TypeVar("T", bound=BaseModel)
STRUCTURED_OUTPUT_VERSION = "v2-structured-2"

PRICES: dict[tuple[str, str], tuple[float, float]] = {
    ("anthropic", "claude-sonnet-5"): (2.00, 10.00),
    ("openai", "gpt-5.4"): (2.50, 15.00),
    ("openai", "gpt-5.4-mini"): (0.75, 4.50),
    ("google", "gemini-3.7-flash"): (0.75, 3.75),
    ("openai", "tts-1"): (15.00, 0.00),
    ("openai", "tts-1-hd"): (30.00, 0.00),
}
PRICING_SNAPSHOT_DATE = "2026-08-29"


@dataclass
class ProviderResponse:
    text: str
    provider: str
    model: str
    input_units: int
    output_units: int
    exact_usage: bool
    raw: dict[str, Any]


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def price(provider: str, model: str, input_units: int, output_units: int) -> float:
    input_rate, output_rate = PRICES.get((provider, model), (5.0, 25.0))
    return input_units * input_rate / 1_000_000 + output_units * output_rate / 1_000_000


def clean_json_text(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    start = value.find("{")
    end = value.rfind("}")
    return value[start:end + 1] if start >= 0 and end >= start else value


class ProviderRouter:
    """One direct model gateway; provider-specific details do not leak into stages."""

    def __init__(
        self,
        store: RunStore,
        artifacts: RunArtifacts,
        *,
        max_api_workers: int = 4,
        strict_keys: bool = False,
        retry_delays: tuple[float, float] = (5.0, 15.0),
    ):
        self.store = store
        self.artifacts = artifacts
        self.strict_keys = strict_keys
        self._api_limit = asyncio.Semaphore(max_api_workers)
        self._retry_delays = retry_delays

    @property
    def api_limit(self) -> asyncio.Semaphore:
        return self._api_limit

    async def structured(
        self,
        *,
        stage: str,
        task_id: str,
        provider: str,
        model: str,
        system: str,
        prompt: str,
        schema: type[T],
        fallback: Callable[[], T],
        scene_id: Optional[str] = None,
        max_output_tokens: int = 6000,
        semantic_retries: int = 1,
    ) -> T:
        schema_json = schema.model_json_schema()
        schema_context = ""
        if provider != "google":
            schema_context = (
                "\n\nThe required JSON Schema is:\n"
                + json.dumps(schema_json, ensure_ascii=False, separators=(",", ":"))
            )
        last_error: Optional[Exception] = None
        for attempt in range(semantic_retries + 1):
            suffix = (
                "\n\nReturn exactly one valid JSON object matching the supplied schema. "
                "Do not include Markdown fences or explanatory text."
                + schema_context
            )
            if attempt > 0:
                suffix += " The previous response did not validate; return the complete object."
            try:
                response = await self.text(
                    stage=stage,
                    task_id=task_id,
                    provider=provider,
                    model=model,
                    system=system,
                    prompt=prompt + suffix,
                    scene_id=scene_id,
                    max_output_tokens=max_output_tokens,
                    schema_json=schema_json,
                )
                return schema.model_validate_json(clean_json_text(response.text))
            except MissingProviderKey as exc:
                if self.strict_keys:
                    raise
                fallback_value = fallback()
                call_id = f"{stage}-local-{uuid.uuid4().hex[:10]}"
                self.artifacts.record_provider_call(
                    stage=stage, call_id=call_id, scene_id=scene_id,
                    request={
                        "provider": provider, "model": model, "system": system,
                        "prompt": prompt + suffix, "schema": schema_json,
                        "max_output_tokens": max_output_tokens,
                    },
                    response={
                        "provider": "local", "model": "deterministic-fallback",
                        "output": fallback_value.model_dump(mode="json"),
                    },
                )
                self.artifacts.event(
                    "provider_fallback", stage=stage, scene_id=scene_id,
                    task_id=task_id, message=str(exc), call_id=call_id,
                )
                return fallback_value
            except (
                BudgetExceeded,
                ProviderCallUncertain,
                ProviderRequestRejected,
                ProviderUnavailable,
            ):
                raise
            except Exception as exc:
                last_error = exc
                self.artifacts.event(
                    "schema_attempt_failed", stage=stage, scene_id=scene_id,
                    task_id=task_id, attempt=attempt + 1, message=str(exc),
                )
        if self.strict_keys:
            raise RuntimeError(f"{stage} failed schema validation: {last_error}")
        self.artifacts.event(
            "schema_fallback", stage=stage, scene_id=scene_id,
            task_id=task_id, message=str(last_error),
        )
        fallback_value = fallback()
        call_id = f"{stage}-schema-fallback-{uuid.uuid4().hex[:10]}"
        self.artifacts.record_provider_call(
            stage=stage, call_id=call_id, scene_id=scene_id,
            request={
                "provider": provider, "model": model, "system": system,
                "prompt": prompt, "schema": schema_json,
                "max_output_tokens": max_output_tokens,
            },
            response={
                "provider": "local", "model": "deterministic-schema-fallback",
                "output": fallback_value.model_dump(mode="json"),
            },
        )
        return fallback_value

    async def text(
        self,
        *,
        stage: str,
        task_id: str,
        provider: str,
        model: str,
        system: str,
        prompt: str,
        scene_id: Optional[str] = None,
        max_output_tokens: int = 8000,
        schema_json: Optional[dict[str, Any]] = None,
    ) -> ProviderResponse:
        request = {
            "provider": provider,
            "model": model,
            "system": system,
            "prompt": prompt,
            "schema": schema_json,
            "max_output_tokens": max_output_tokens,
        }
        canonical = json.dumps(request, sort_keys=True, ensure_ascii=False)
        cache_key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        cached = self.store.cache_get(cache_key)
        if cached:
            self.artifacts.event(
                "cache_hit", stage=stage, scene_id=scene_id, task_id=task_id,
                message=f"Reused {provider}/{model} response", cache_key=cache_key,
            )
            return ProviderResponse(**cached)

        try:
            self._require_key(provider)
        except MissingProviderKey:
            call_id = f"{stage}-missing-key-{uuid.uuid4().hex[:10]}"
            self.artifacts.record_provider_call(
                stage=stage, call_id=call_id, scene_id=scene_id, request=request,
            )
            raise
        call_id = f"{stage}-{uuid.uuid4().hex[:12]}"
        native_schema = json.dumps(schema_json or {}) if provider == "google" else ""
        input_units = estimate_tokens(system + prompt + native_schema)
        reserved = price(provider, model, input_units, max_output_tokens)
        self.store.reserve_call(
            call_id, task_id=task_id, scene_id=scene_id, stage=stage,
            provider=provider, model=model, amount_usd=reserved,
        )
        self.artifacts.record_provider_call(
            stage=stage, call_id=call_id, scene_id=scene_id, request=request,
        )
        self.artifacts.event(
            "api_call_started", stage=stage, scene_id=scene_id, task_id=task_id,
            provider=provider, model=model, call_id=call_id, reserved_usd=reserved,
        )
        try:
            response = await self._invoke_with_retries(
                lambda: asyncio.to_thread(
                    self._invoke, provider, model, system, prompt, max_output_tokens,
                    schema_json,
                ),
                stage=stage,
                scene_id=scene_id,
                task_id=task_id,
                provider=provider,
                model=model,
                call_id=call_id,
            )
        except ProviderUnavailable as exc:
            self.store.release_call(call_id, status="FAILED_RETRYABLE")
            self.artifacts.event(
                "api_call_failed", stage=stage, scene_id=scene_id, task_id=task_id,
                provider=provider, model=model, call_id=call_id, message=str(exc),
                billed=False,
            )
            raise
        except ProviderRequestRejected as exc:
            self.store.release_call(call_id, status="FAILED_REQUEST")
            self.artifacts.event(
                "api_call_failed", stage=stage, scene_id=scene_id, task_id=task_id,
                provider=provider, model=model, call_id=call_id, message=str(exc),
                billed=False,
            )
            raise
        except ProviderCallUncertain as exc:
            # An ambiguous provider timeout may already be billable; retain the
            # reservation and never retry it invisibly.
            self.store.settle_call(
                call_id, actual_usd=reserved, input_units=input_units,
                output_units=0, exact=False, status="FAILED_UNCERTAIN",
            )
            self.artifacts.event(
                "api_call_failed", stage=stage, scene_id=scene_id, task_id=task_id,
                provider=provider, model=model, call_id=call_id, message=str(exc),
            )
            raise

        actual = price(
            provider, model, response.input_units or input_units, response.output_units,
        )
        self.store.settle_call(
            call_id, actual_usd=actual, input_units=response.input_units,
            output_units=response.output_units, exact=response.exact_usage,
        )
        payload = {
            "text": response.text,
            "provider": response.provider,
            "model": response.model,
            "input_units": response.input_units,
            "output_units": response.output_units,
            "exact_usage": response.exact_usage,
            "raw": response.raw,
        }
        self.store.cache_put(cache_key, payload)
        self.artifacts.record_provider_call(
            stage=stage, call_id=call_id, scene_id=scene_id,
            request=request, response=payload,
        )
        self.artifacts.event(
            "api_call_finished", stage=stage, scene_id=scene_id, task_id=task_id,
            provider=provider, model=model, call_id=call_id, actual_usd=actual,
            input_units=response.input_units, output_units=response.output_units,
            exact_usage=response.exact_usage,
        )
        return response

    async def vision(
        self,
        *,
        stage: str,
        task_id: str,
        scene_id: str,
        model: str,
        system: str,
        prompt: str,
        image_paths: list,
        max_output_tokens: int = 8000,
    ) -> ProviderResponse:
        valid_paths = [path.resolve() for path in image_paths if path.exists()]
        image_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in valid_paths]
        request = {
            "provider": "google", "model": model, "system": system, "prompt": prompt,
            "image_paths": [str(path) for path in valid_paths], "image_hashes": image_hashes,
            "max_output_tokens": max_output_tokens,
        }
        canonical = json.dumps(request, sort_keys=True, ensure_ascii=False)
        cache_key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        cached = self.store.cache_get(cache_key)
        if cached:
            self.artifacts.event(
                "cache_hit", stage=stage, scene_id=scene_id, task_id=task_id,
                message=f"Reused google/{model} vision response", cache_key=cache_key,
            )
            return ProviderResponse(**cached)
        try:
            self._require_key("google")
        except MissingProviderKey:
            call_id = f"{stage}-missing-key-{uuid.uuid4().hex[:10]}"
            self.artifacts.record_provider_call(
                stage=stage, call_id=call_id, scene_id=scene_id, request=request,
            )
            raise
        call_id = f"{stage}-{uuid.uuid4().hex[:12]}"
        # A conservative image-token reserve avoids concurrency overcommitting
        # the run even when the provider reports image usage only afterward.
        input_units = estimate_tokens(system + prompt) + 1500 * len(valid_paths)
        reserved = price("google", model, input_units, max_output_tokens)
        self.store.reserve_call(
            call_id, task_id=task_id, scene_id=scene_id, stage=stage,
            provider="google", model=model, amount_usd=reserved,
        )
        self.artifacts.record_provider_call(
            stage=stage, call_id=call_id, scene_id=scene_id, request=request,
        )
        self.artifacts.event(
            "api_call_started", stage=stage, scene_id=scene_id, task_id=task_id,
            provider="google", model=model, call_id=call_id, reserved_usd=reserved,
            image_count=len(valid_paths),
        )
        try:
            response = await self._invoke_with_retries(
                lambda: asyncio.to_thread(
                    self._invoke_google_vision, model, system, prompt,
                    valid_paths, max_output_tokens,
                ),
                stage=stage,
                scene_id=scene_id,
                task_id=task_id,
                provider="google",
                model=model,
                call_id=call_id,
            )
        except ProviderUnavailable as exc:
            self.store.release_call(call_id, status="FAILED_RETRYABLE")
            self.artifacts.event(
                "api_call_failed", stage=stage, scene_id=scene_id, task_id=task_id,
                provider="google", model=model, call_id=call_id, message=str(exc),
                billed=False,
            )
            raise
        except ProviderRequestRejected as exc:
            self.store.release_call(call_id, status="FAILED_REQUEST")
            self.artifacts.event(
                "api_call_failed", stage=stage, scene_id=scene_id, task_id=task_id,
                provider="google", model=model, call_id=call_id, message=str(exc),
                billed=False,
            )
            raise
        except ProviderCallUncertain as exc:
            self.store.settle_call(
                call_id, actual_usd=reserved, input_units=input_units,
                output_units=0, exact=False, status="FAILED_UNCERTAIN",
            )
            self.artifacts.event(
                "api_call_failed", stage=stage, scene_id=scene_id, task_id=task_id,
                provider="google", model=model, call_id=call_id, message=str(exc),
            )
            raise
        actual = price("google", model, response.input_units, response.output_units)
        self.store.settle_call(
            call_id, actual_usd=actual, input_units=response.input_units,
            output_units=response.output_units, exact=response.exact_usage,
        )
        payload = {
            "text": response.text, "provider": response.provider, "model": response.model,
            "input_units": response.input_units, "output_units": response.output_units,
            "exact_usage": response.exact_usage, "raw": response.raw,
        }
        self.store.cache_put(cache_key, payload)
        self.artifacts.record_provider_call(
            stage=stage, call_id=call_id, scene_id=scene_id,
            request=request, response=payload,
        )
        self.artifacts.event(
            "api_call_finished", stage=stage, scene_id=scene_id, task_id=task_id,
            provider="google", model=model, call_id=call_id, actual_usd=actual,
            input_units=response.input_units, output_units=response.output_units,
        )
        return response

    async def _invoke_with_retries(
        self,
        invoke: Callable[[], Any],
        *,
        stage: str,
        scene_id: Optional[str],
        task_id: str,
        provider: str,
        model: str,
        call_id: str,
    ) -> ProviderResponse:
        attempts = len(self._retry_delays) + 1
        for attempt in range(1, attempts + 1):
            try:
                async with self._api_limit:
                    return await invoke()
            except Exception as exc:
                if _deterministic_request_rejection(exc):
                    raise ProviderRequestRejected(str(exc)) from exc
                if not _confirmed_retryable(exc):
                    raise ProviderCallUncertain(str(exc)) from exc
                if attempt >= attempts:
                    raise ProviderUnavailable(
                        f"{provider}/{model} remained unavailable after {attempts} attempts: {exc}"
                    ) from exc
                delay = self._retry_delays[attempt - 1]
                self.artifacts.event(
                    "api_call_retry", stage=stage, scene_id=scene_id, task_id=task_id,
                    provider=provider, model=model, call_id=call_id,
                    attempt=attempt, retry_in_seconds=delay, message=str(exc),
                )
                if delay:
                    await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def _require_key(provider: str) -> None:
        names = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GEMINI_API_KEY",
        }
        name = names.get(provider)
        if not name or not os.getenv(name):
            raise MissingProviderKey(f"{name or provider + ' key'} is not configured")

    @staticmethod
    def _invoke(
        provider: str,
        model: str,
        system: str,
        prompt: str,
        max_output_tokens: int,
        schema_json: Optional[dict[str, Any]],
    ) -> ProviderResponse:
        if provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            result = client.messages.create(
                model=model,
                max_tokens=max_output_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(getattr(block, "text", "") for block in result.content)
            usage = getattr(result, "usage", None)
            return ProviderResponse(
                text=text, provider=provider, model=model,
                input_units=int(getattr(usage, "input_tokens", estimate_tokens(system + prompt))),
                output_units=int(getattr(usage, "output_tokens", estimate_tokens(text))),
                exact_usage=usage is not None,
                raw={"id": getattr(result, "id", None), "stop_reason": getattr(result, "stop_reason", None)},
            )
        if provider == "openai":
            from openai import OpenAI

            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            kwargs: dict[str, Any] = {
                "model": model,
                "max_completion_tokens": max_output_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            }
            if schema_json:
                kwargs["response_format"] = {"type": "json_object"}
            result = client.chat.completions.create(**kwargs)
            text = result.choices[0].message.content or ""
            usage = result.usage
            return ProviderResponse(
                text=text, provider=provider, model=model,
                input_units=int(getattr(usage, "prompt_tokens", estimate_tokens(system + prompt))),
                output_units=int(getattr(usage, "completion_tokens", estimate_tokens(text))),
                exact_usage=usage is not None,
                raw={"id": result.id, "finish_reason": result.choices[0].finish_reason},
            )
        if provider == "google":
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            config_kwargs: dict[str, Any] = {
                "system_instruction": system,
                "max_output_tokens": max_output_tokens,
            }
            if schema_json:
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_json_schema"] = schema_json
            result = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            text = result.text or ""
            usage = getattr(result, "usage_metadata", None)
            return ProviderResponse(
                text=text, provider=provider, model=model,
                input_units=int(getattr(usage, "prompt_token_count", estimate_tokens(system + prompt))),
                output_units=int(getattr(usage, "candidates_token_count", estimate_tokens(text))),
                exact_usage=usage is not None,
                raw={"finish_reason": str(getattr(result, "finish_reason", ""))},
            )
        raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def _invoke_google_vision(
        model: str,
        system: str,
        prompt: str,
        image_paths: list,
        max_output_tokens: int,
    ) -> ProviderResponse:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        contents: list[Any] = [prompt]
        for path in image_paths:
            mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            contents.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type))
        result = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_output_tokens,
            ),
        )
        text = result.text or ""
        usage = getattr(result, "usage_metadata", None)
        return ProviderResponse(
            text=text, provider="google", model=model,
            input_units=int(getattr(usage, "prompt_token_count", estimate_tokens(system + prompt))),
            output_units=int(getattr(usage, "candidates_token_count", estimate_tokens(text))),
            exact_usage=usage is not None,
            raw={"finish_reason": str(getattr(result, "finish_reason", ""))},
        )


class MissingProviderKey(RuntimeError):
    pass


class ProviderCallUncertain(RuntimeError):
    """A provider attempt may have been billed and must not be retried blindly."""


class ProviderRequestRejected(RuntimeError):
    """A deterministic client error was rejected before billable model work."""


class ProviderUnavailable(RuntimeError):
    """Confirmed unbilled provider failures exhausted the bounded retry policy."""


def _confirmed_retryable(exc: Exception) -> bool:
    status = _provider_status_code(exc)
    if status in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return False
    message = str(exc).upper()
    return (
        isinstance(exc, ConnectionError)
        or "APICONNECTIONERROR" in name.upper()
        or "RESOURCE_EXHAUSTED" in message
        or "SERVICE_UNAVAILABLE" in message
    )


def _deterministic_request_rejection(exc: Exception) -> bool:
    status = _provider_status_code(exc)
    return isinstance(status, int) and 400 <= status < 500 and status not in {
        408, 409, 425, 429,
    }


def _provider_status_code(exc: Exception) -> Optional[int]:
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if isinstance(value, int):
            return value
        nested = getattr(value, "value", None)
        if isinstance(nested, int):
            return nested
        if isinstance(value, str) and value.isdigit():
            return int(value)
    match = re.match(r"^\s*(\d{3})\b", str(exc))
    return int(match.group(1)) if match else None
