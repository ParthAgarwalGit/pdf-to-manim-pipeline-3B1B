"""Provider gateway concurrency, cost, and cache behavior."""

import asyncio
import json
import threading
import time

import pytest
from pydantic import BaseModel

from src.pipeline_v2.observability import RunArtifacts
from src.pipeline_v2.providers import (
    ProviderCallUncertain,
    ProviderRequestRejected,
    ProviderResponse,
    ProviderRouter,
)
from src.pipeline_v2.state import RunStore


class _StructuredPayload(BaseModel):
    value: int


def test_structured_openai_request_mentions_json_on_first_attempt(monkeypatch, tmp_path):
    store = RunStore(tmp_path / "run.db", 10.0)
    artifacts = RunArtifacts(tmp_path, "run")
    router = ProviderRouter(store, artifacts)
    monkeypatch.setattr(router, "_require_key", lambda _provider: None)
    messages = []

    def invoke(provider, model, system, prompt, max_output_tokens, schema):
        messages.append(system + "\n" + prompt)
        assert schema is not None
        return ProviderResponse(
            text='{"value": 7}', provider=provider, model=model,
            input_units=10, output_units=5, exact_usage=True, raw={},
        )

    monkeypatch.setattr(router, "_invoke", invoke)
    result = asyncio.run(router.structured(
        stage="structured", task_id="task", provider="openai", model="gpt-5.4-mini",
        system="Return the requested object.", prompt="Set value to seven.",
        schema=_StructuredPayload, fallback=lambda: _StructuredPayload(value=0),
        semantic_retries=0,
    ))

    assert result.value == 7
    assert len(messages) == 1
    assert "json" in messages[0].lower()
    assert '"properties":{"value"' in messages[0]
    store.close()


def test_global_api_limit_and_cache(monkeypatch, tmp_path):
    store = RunStore(tmp_path / "run.db", 10.0)
    artifacts = RunArtifacts(tmp_path, "run")
    router = ProviderRouter(store, artifacts, max_api_workers=2)
    monkeypatch.setattr(router, "_require_key", lambda _provider: None)
    lock = threading.Lock()
    active = 0
    maximum = 0
    calls = 0

    def fake_invoke(provider, model, system, prompt, max_output_tokens, schema):
        nonlocal active, maximum, calls
        with lock:
            active += 1
            calls += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return ProviderResponse(
            text=prompt, provider=provider, model=model,
            input_units=10, output_units=5, exact_usage=True, raw={},
        )

    monkeypatch.setattr(router, "_invoke", fake_invoke)

    async def exercise():
        await asyncio.gather(*[
            router.text(
                stage="test", task_id=f"task-{index}", provider="google", model="gemini-3.7-flash",
                system="system", prompt=f"prompt-{index}", max_output_tokens=20,
            ) for index in range(6)
        ])
        first = await router.text(
            stage="test", task_id="cached-1", provider="google", model="gemini-3.7-flash",
            system="system", prompt="same", max_output_tokens=20,
        )
        second = await router.text(
            stage="test", task_id="cached-2", provider="google", model="gemini-3.7-flash",
            system="system", prompt="same", max_output_tokens=20,
        )
        assert first.text == second.text

    asyncio.run(exercise())
    assert maximum == 2
    assert calls == 7
    assert any(json.loads(line)["event"] == "cache_hit"
               for line in (tmp_path / "events.jsonl").read_text().splitlines())
    store.close()


def test_confirmed_rate_limit_retries_twice_then_succeeds(monkeypatch, tmp_path):
    store = RunStore(tmp_path / "run.db", 10.0)
    artifacts = RunArtifacts(tmp_path, "run")
    router = ProviderRouter(store, artifacts, retry_delays=(0.0, 0.0))
    monkeypatch.setattr(router, "_require_key", lambda _provider: None)
    calls = 0

    class RateLimited(RuntimeError):
        status_code = 429

    def invoke(*_args):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RateLimited("try later")
        return ProviderResponse(
            text="ok", provider="google", model="gemini-3.7-flash",
            input_units=10, output_units=2, exact_usage=True, raw={},
        )

    monkeypatch.setattr(router, "_invoke", invoke)
    response = asyncio.run(router.text(
        stage="test", task_id="task", provider="google", model="gemini-3.7-flash",
        system="system", prompt="prompt", max_output_tokens=10,
    ))
    assert response.text == "ok"
    assert calls == 3
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert sum(event["event"] == "api_call_retry" for event in events) == 2
    store.close()


def test_ambiguous_timeout_is_not_retried(monkeypatch, tmp_path):
    store = RunStore(tmp_path / "run.db", 10.0)
    artifacts = RunArtifacts(tmp_path, "run")
    router = ProviderRouter(store, artifacts, retry_delays=(0.0, 0.0))
    monkeypatch.setattr(router, "_require_key", lambda _provider: None)
    calls = 0

    def invoke(*_args):
        nonlocal calls
        calls += 1
        raise TimeoutError("ambiguous provider timeout")

    monkeypatch.setattr(router, "_invoke", invoke)
    with pytest.raises(ProviderCallUncertain):
        asyncio.run(router.text(
            stage="test", task_id="task", provider="google", model="gemini-3.7-flash",
            system="system", prompt="prompt", max_output_tokens=10,
        ))
    assert calls == 1
    assert store.cost_rows()[0]["status"] == "FAILED_UNCERTAIN"
    store.close()


def test_deterministic_bad_request_is_unbilled_and_not_retried(monkeypatch, tmp_path):
    store = RunStore(tmp_path / "run.db", 10.0)
    artifacts = RunArtifacts(tmp_path, "run")
    router = ProviderRouter(store, artifacts, retry_delays=(0.0, 0.0))
    monkeypatch.setattr(router, "_require_key", lambda _provider: None)
    calls = 0

    class BadRequest(RuntimeError):
        status_code = 400

    def invoke(*_args):
        nonlocal calls
        calls += 1
        raise BadRequest("invalid response_format request")

    monkeypatch.setattr(router, "_invoke", invoke)
    with pytest.raises(ProviderRequestRejected):
        asyncio.run(router.text(
            stage="test", task_id="task", provider="openai", model="gpt-5.4-mini",
            system="system", prompt="prompt", max_output_tokens=10,
        ))

    assert calls == 1
    row = store.cost_rows()[0]
    assert row["status"] == "FAILED_REQUEST"
    assert row["actual_usd"] == 0
    assert store.total_cost() == 0
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    failure = next(event for event in events if event["event"] == "api_call_failed")
    assert failure["billed"] is False
    store.close()
