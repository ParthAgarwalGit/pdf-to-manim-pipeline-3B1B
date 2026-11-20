"""The run's task cache: a task is reused only when its inputs are unchanged."""

import asyncio

from src.pipeline_v2.observability import RunArtifacts
from src.pipeline_v2.pipeline import PipelineV2
from src.pipeline_v2.state import RunStore


def _pipeline(tmp_path):
    pipeline = PipelineV2.__new__(PipelineV2)
    pipeline.store = RunStore(tmp_path / "run.db", 10.0)
    pipeline.artifacts = RunArtifacts(tmp_path / "run", "run")
    return pipeline


def test_changed_inputs_rerun_the_task_and_unchanged_inputs_reuse_it(tmp_path):
    pipeline = _pipeline(tmp_path)
    calls = []

    async def operation(value):
        calls.append(value)
        return {"value": value}

    async def run(value):
        return await pipeline._task("lesson", "lesson", None, {"input": value},
                                    lambda: operation(value))

    assert asyncio.run(run("a")) == {"value": "a"}
    assert asyncio.run(run("a")) == {"value": "a"}
    assert calls == ["a"]  # same inputs: reused
    assert asyncio.run(run("b")) == {"value": "b"}
    assert calls == ["a", "b"]  # changed inputs: run again, never the stale output
    assert asyncio.run(run("b")) == {"value": "b"}
    assert calls == ["a", "b"]
