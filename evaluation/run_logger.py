"""Structured JSONL run logging (prompt hash, retrieval, outputs, latency)."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class RunLogger:
    def __init__(self, log_dir: str | Path, run_id: str | None = None) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.jsonl_path = self.log_dir / f"run_{self.run_id}.jsonl"
        self._fh = self.jsonl_path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "type": event_type,
            **payload,
        }
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def log_generation(
        self,
        *,
        example_id: str,
        prompt: str,
        response: str,
        retrieved_doc_ids: list[str],
        ranks: list[int],
        latency_s: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "example_id": example_id,
            "prompt_sha256": _sha256_text(prompt),
            "prompt_len": len(prompt),
            "response_len": len(response),
            "retrieved_doc_ids": retrieved_doc_ids,
            "ranks": ranks,
            "latency_s": latency_s,
            "response_preview": response[:500],
        }
        if extra:
            payload.update(extra)
        self.log_event("generation", payload)


class StageTimer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def split(self) -> float:
        now = time.perf_counter()
        dt = now - self.t0
        self.t0 = now
        return dt
