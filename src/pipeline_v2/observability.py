"""Human and machine-readable V2 run artifacts."""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .redaction import redact, redact_text


class RunArtifacts:
    def __init__(self, run_dir: Path, run_id: str):
        self.run_dir = run_dir.resolve()
        self.run_id = run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.log_path = self.run_dir / "pipeline.log"
        self._lock = threading.RLock()
        self.logger = logging.getLogger(f"pipeline_v2.{run_id}")
        for old_handler in list(self.logger.handlers):
            self.logger.removeHandler(old_handler)
            old_handler.close()
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(handler)
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        self.logger.addHandler(console)

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def event(
        self,
        event: str,
        *,
        stage: Optional[str] = None,
        scene_id: Optional[str] = None,
        task_id: Optional[str] = None,
        message: str = "",
        **fields: Any,
    ) -> None:
        record = redact({
            "event": event,
            "timestamp": self.now(),
            "run_id": self.run_id,
            "stage": stage,
            "scene_id": scene_id,
            "task_id": task_id,
            "message": message,
            **fields,
        })
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            details = []
            for key in ("worker", "attempt", "status", "provider", "model"):
                if record.get(key) is not None:
                    details.append(f"{key}={record[key]}")
            if record.get("duration_ms") is not None:
                details.append(f"duration={float(record['duration_ms']) / 1000:.2f}s")
            for cost_key in ("reserved_usd", "actual_usd", "cost_usd"):
                if record.get(cost_key) is not None:
                    details.append(f"{cost_key}=${float(record[cost_key]):.6f}")
            human = " | ".join(
                str(part)
                for part in [event, stage, scene_id, task_id, message, *details]
                if part not in (None, "")
            )
            if event.endswith("failed") or event == "error":
                self.logger.error(redact_text(human))
            else:
                self.logger.info(redact_text(human))

    def write_json(self, relative: str | Path, value: Any) -> Path:
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(redact(value), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return path.resolve()

    def write_text(self, relative: str | Path, value: str) -> Path:
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(redact_text(value), encoding="utf-8")
        return path.resolve()

    def record_provider_call(
        self,
        *,
        stage: str,
        call_id: str,
        scene_id: Optional[str],
        request: dict[str, Any],
        response: Optional[dict[str, Any]] = None,
    ) -> Path:
        base = Path("prompts") / stage / call_id
        self.write_json(base / "request.json", request)
        if response is not None:
            self.write_json(base / "response.json", response)
        return (self.run_dir / base).resolve()
