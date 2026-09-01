"""Durable state, budget, cache, and redaction tests."""

import threading

import pytest

from src.pipeline_v2.observability import RunArtifacts
from src.pipeline_v2.state import BudgetExceeded, RunStore


def test_atomic_budget_reservations_never_overcommit(tmp_path):
    store = RunStore(tmp_path / "run.db", 1.0)
    successes = []
    barrier = threading.Barrier(10)

    def reserve(index):
        barrier.wait()
        try:
            store.reserve_call(
                f"call-{index}", task_id=f"task-{index}", scene_id=None,
                stage="test", provider="google", model="gemini", amount_usd=0.2,
            )
            successes.append(index)
        except BudgetExceeded:
            pass

    threads = [threading.Thread(target=reserve, args=(index,)) for index in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(successes) == 5
    assert store.total_cost() == pytest.approx(1.0)
    store.close()


def test_task_and_response_cache_round_trip(tmp_path):
    store = RunStore(tmp_path / "run.db", 1.0)
    store.upsert_task("task", "stage", None, "hash", ["upstream"])
    assert store.task_output("task", "hash") is None
    store.start_task("task")
    store.finish_task("task", {"answer": 42}, 10.0)
    assert store.task_output("task", "hash") == {"answer": 42}
    assert store.task_rows()[0]["dependencies_json"] == '["upstream"]'
    store.cache_put("key", {"text": "cached"})
    assert store.cache_get("key") == {"text": "cached"}
    store.close()


def test_running_tasks_become_interrupted_and_remain_resumable(tmp_path):
    store = RunStore(tmp_path / "run.db", 1.0)
    store.upsert_task("render", "preview", "scene_001", "hash", ["code"])
    store.start_task("render")
    store.interrupt_running()
    row = store.task_rows()[0]
    assert row["status"] == "INTERRUPTED"
    assert store.task_output("render", "hash") is None
    assert store.start_task("render") == 2
    store.close()


def test_scene_retry_invalidates_generated_work_but_preserves_tts(tmp_path):
    store = RunStore(tmp_path / "run.db", 1.0)
    for task_id, stage in (
        ("tts-scene_001", "tts"),
        ("code-scene_001", "code_generation"),
        ("preview-scene_001-00", "preview"),
    ):
        store.upsert_task(task_id, stage, "scene_001", "hash")
        store.start_task(task_id)
        store.finish_task(task_id, {"task": task_id}, 1.0)
    store.upsert_task("final-compose", "composition", None, "hash")
    store.start_task("final-compose")
    store.finish_task("final-compose", "final.mp4", 1.0)

    attempts = store.prepare_scene_retry(["scene_001"])
    rows = {row["task_id"]: row for row in store.task_rows()}

    assert attempts == {"scene_001": 1}
    assert rows["tts-scene_001"]["status"] == "SUCCEEDED"
    assert store.successful_task_output("tts-scene_001") == {"task": "tts-scene_001"}
    assert rows["code-scene_001"]["status"] == "PENDING"
    assert rows["preview-scene_001-00"]["status"] == "PENDING"
    assert rows["final-compose"]["status"] == "PENDING"
    assert store.prepare_scene_retry(["scene_001"]) == {"scene_001": 2}
    store.close()


def test_full_artifacts_redact_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value")
    artifacts = RunArtifacts(tmp_path, "run")
    artifacts.event("test", message="Authorization: Bearer sk-super-secret-value")
    artifacts.record_provider_call(
        stage="lesson", call_id="call", scene_id=None,
        request={"api_key": "sk-super-secret-value", "prompt": "hello"},
        response={"text": "token=sk-super-secret-value"},
    )
    all_text = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.rglob("*") if path.is_file())
    assert "sk-super-secret-value" not in all_text
    assert "[REDACTED]" in all_text
