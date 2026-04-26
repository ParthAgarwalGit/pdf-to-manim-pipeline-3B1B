# ADR-001: Synchronous Orchestrator Is the Canonical Pipeline Path

**Status:** Accepted  
**Date:** 2026-04-23  
**Deciders:** Parth Agarwal

---

## Context

The repository contains two orchestrator implementations:

- `src/orchestrator/pipeline.py` — synchronous `Orchestrator` / `PipelineConfig`
- `src/orchestrator/async_pipeline.py` — async wrapper providing `AsyncPipelineRunner` and `TaskQueue`

Both exist under `src/orchestrator/`. `async_pipeline.py` was built to support WebSocket progress callbacks and concurrent multi-pipeline web deployments. However, as of this writing:

- `async_pipeline.py` is **not imported** by any other module (`cli.py`, `__init__.py`, tests).
- The sync orchestrator is the only path exercised by the CLI and all test suites.
- The pipeline processes **one PDF at a time** per invocation — there is no concurrent-request workload yet.
- `async_pipeline.py` wraps the sync `Orchestrator` via `asyncio.to_thread()`, so it is an adapter, not a parallel implementation.

---

## Options Considered

### Option A — Adopt async as the primary path

Migrate all agent execution and orchestration to `async`/`await`. This would require async-all-the-way-down changes across every agent, making debugging harder and adding complexity for no immediate benefit given the one-PDF-at-a-time workload.

**TCO (3 years):** High — every new agent must be written async; stack traces become coroutine chains.  
**Risk:** High — partial async is worse than either extreme; introduces subtle race conditions.

### Option B — Keep sync as canonical; treat async as an optional adapter (chosen)

The synchronous orchestrator remains the single source of truth for pipeline logic. `async_pipeline.py` is retained as a documented async adapter layer for future web deployment scenarios. New pipeline stages are added to the sync path first.

**TCO (3 years):** Low — one orchestrator to maintain; all agents are plain Python.  
**Risk:** Low — `ThreadPoolExecutor` already provides the one place parallelism is needed (render ‖ TTS).

### Option C — Delete async_pipeline.py

Remove the file since nothing imports it. Deferred: the file is not dead weight yet (it is a valid future integration point) and deletion would need re-implementation if a web deployment need arises.

---

## Decision

**The synchronous orchestrator in `src/orchestrator/pipeline.py` is the canonical pipeline execution path.**

`src/orchestrator/async_pipeline.py` is retained as an async adapter/task-queue wrapper around the sync pipeline. It is **not** a second orchestration implementation and should not diverge from the sync path's feature set.

---

## Rationale

1. The product currently processes one PDF pipeline at a time. Async offers no throughput benefit here.
2. `asyncio.to_thread()` already wraps the sync pipeline in `async_pipeline.py` — the sync core is shared, not duplicated.
3. `ThreadPoolExecutor` (added in the same sprint) provides the only parallelism that matters today: render and TTS running concurrently.
4. Full async all the way down would require every agent to be rewritten and would make exception handling and logging significantly harder to follow.

---

## Consequences

### What becomes easier
- New pipeline stages can be added to `pipeline.py` without any async considerations.
- Test coverage and debugging remain straightforward (no coroutine stack traces).
- Contributors do not need to reason about event-loop safety when writing agents.

### What becomes harder
- If a multi-tenant web API is ever needed, `async_pipeline.py` will need to be brought into sync with the current sync pipeline's feature set before it can be trusted.
- Any future move to full-async orchestration requires a new ADR and careful migration.

### Standing constraints
- `async_pipeline.py` carries a module docstring explicitly marking it as an adapter. It must not acquire orchestration logic that diverges from `pipeline.py`.
- Any future async migration requires its own ADR superseding this one.
