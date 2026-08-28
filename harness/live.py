from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.config import RESULTS_DIR

_LOCK = threading.Lock()


def live_log_path(results_dir: Path | None = None) -> Path:
    return (results_dir or RESULTS_DIR) / "live.jsonl"


def emit_live(event: dict[str, Any], *, path: Path | None = None) -> None:
    """Append one JSONL event. Line writes are atomic under PIPE_BUF."""
    payload = dict(event)
    payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
    line = json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"
    log_path = path or live_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
    except OSError:
        return
