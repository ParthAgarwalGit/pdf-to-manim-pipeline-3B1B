"""SQLite WAL state, cache, and atomic cost reservations."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .models import TaskStatus


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BudgetExceeded(RuntimeError):
    pass


class RunStore:
    def __init__(self, path: Path, max_cost_usd: float):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_cost_usd = max_cost_usd
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._initialize()

    def _initialize(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    scene_id TEXT,
                    stage TEXT NOT NULL,
                    dependencies_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    input_hash TEXT,
                    output_json TEXT,
                    error TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    duration_ms REAL
                );
                CREATE TABLE IF NOT EXISTS api_calls (
                    call_id TEXT PRIMARY KEY,
                    task_id TEXT,
                    scene_id TEXT,
                    stage TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reserved_usd REAL NOT NULL,
                    actual_usd REAL,
                    input_units INTEGER,
                    output_units INTEGER,
                    exact INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            columns = {
                row["name"]
                for row in self._connection.execute("PRAGMA table_info(tasks)").fetchall()
            }
            if "dependencies_json" not in columns:
                self._connection.execute(
                    "ALTER TABLE tasks ADD COLUMN dependencies_json TEXT NOT NULL DEFAULT '[]'"
                )
        self.set_meta("max_cost_usd", self.max_cost_usd)

    def close(self) -> None:
        self._connection.close()

    def set_meta(self, key: str, value: Any) -> None:
        payload = json.dumps(value, default=str)
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, payload),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self._connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def upsert_task(
        self,
        task_id: str,
        stage: str,
        scene_id: Optional[str],
        input_hash: str,
        dependencies: Optional[list[str]] = None,
    ) -> None:
        dependencies_json = json.dumps(dependencies or [])
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO tasks(task_id, scene_id, stage, dependencies_json, status, input_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    input_hash=excluded.input_hash,
                    dependencies_json=excluded.dependencies_json
                """,
                (
                    task_id,
                    scene_id,
                    stage,
                    dependencies_json,
                    TaskStatus.PENDING.value,
                    input_hash,
                ),
            )

    def start_task(self, task_id: str) -> int:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE tasks SET status=?, attempt=attempt+1, started_at=?, error=NULL WHERE task_id=?",
                (TaskStatus.RUNNING.value, utc_now(), task_id),
            )
            row = self._connection.execute("SELECT attempt FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            return int(row["attempt"])

    def finish_task(self, task_id: str, output: Any, duration_ms: float) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE tasks SET status=?, output_json=?, finished_at=?, duration_ms=? WHERE task_id=?",
                (TaskStatus.SUCCEEDED.value, json.dumps(output, default=str), utc_now(), duration_ms, task_id),
            )

    def fail_task(self, task_id: str, error: str, duration_ms: float, status: TaskStatus = TaskStatus.FAILED) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE tasks SET status=?, error=?, finished_at=?, duration_ms=? WHERE task_id=?",
                (status.value, error, utc_now(), duration_ms, task_id),
            )

    def task_output(self, task_id: str, input_hash: str) -> Optional[Any]:
        row = self._connection.execute(
            "SELECT status, input_hash, output_json FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if not row or row["status"] != TaskStatus.SUCCEEDED.value or row["input_hash"] != input_hash:
            return None
        return json.loads(row["output_json"]) if row["output_json"] else None

    def interrupt_running(self) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE tasks SET status=?, finished_at=? WHERE status=?",
                (TaskStatus.INTERRUPTED.value, utc_now(), TaskStatus.RUNNING.value),
            )

    def reserve_call(
        self,
        call_id: str,
        *,
        task_id: str,
        scene_id: Optional[str],
        stage: str,
        provider: str,
        model: str,
        amount_usd: float,
    ) -> None:
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT status FROM api_calls WHERE call_id=?", (call_id,)
            ).fetchone()
            if existing:
                return
            row = self._connection.execute(
                "SELECT COALESCE(SUM(CASE WHEN actual_usd IS NULL THEN reserved_usd ELSE actual_usd END), 0) total FROM api_calls"
            ).fetchone()
            committed = float(row["total"])
            if committed + amount_usd > self.max_cost_usd + 1e-9:
                raise BudgetExceeded(
                    f"Call would commit ${committed + amount_usd:.4f} against the ${self.max_cost_usd:.2f} run limit."
                )
            self._connection.execute(
                """
                INSERT INTO api_calls(call_id, task_id, scene_id, stage, provider, model, status,
                                      reserved_usd, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'RESERVED', ?, ?)
                """,
                (call_id, task_id, scene_id, stage, provider, model, amount_usd, utc_now()),
            )

    def settle_call(
        self,
        call_id: str,
        *,
        actual_usd: float,
        input_units: int,
        output_units: int,
        exact: bool,
        status: str = "SUCCEEDED",
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE api_calls SET status=?, actual_usd=?, input_units=?, output_units=?,
                                     exact=?, finished_at=? WHERE call_id=?
                """,
                (status, actual_usd, input_units, output_units, int(exact), utc_now(), call_id),
            )

    def release_call(self, call_id: str, status: str = "SKIPPED") -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE api_calls SET status=?, actual_usd=0, finished_at=? WHERE call_id=?",
                (status, utc_now(), call_id),
            )

    def cost_rows(self) -> list[dict[str, Any]]:
        rows = self._connection.execute("SELECT * FROM api_calls ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]

    def total_cost(self) -> float:
        row = self._connection.execute(
            "SELECT COALESCE(SUM(COALESCE(actual_usd, reserved_usd)), 0) total FROM api_calls"
        ).fetchone()
        return float(row["total"])

    def cache_get(self, cache_key: str) -> Optional[dict[str, Any]]:
        row = self._connection.execute(
            "SELECT response_json FROM response_cache WHERE cache_key=?", (cache_key,)
        ).fetchone()
        return json.loads(row["response_json"]) if row else None

    def cache_put(self, cache_key: str, response: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO response_cache(cache_key, response_json, created_at) VALUES (?, ?, ?)",
                (cache_key, json.dumps(response, default=str), utc_now()),
            )

    def task_rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._connection.execute("SELECT * FROM tasks ORDER BY rowid")]
