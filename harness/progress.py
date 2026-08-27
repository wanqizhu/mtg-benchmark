from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from harness.core import Transcript


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


class RolloutProgress:
    """Stdout status lines + incremental result file writes during a rollout."""

    def __init__(
        self,
        *,
        model_name: str,
        sample_id: str,
        out_path: Path,
        payload_for: Callable[[Transcript, str], dict[str, Any]],
        heartbeat_seconds: float = 30.0,
        log_stream: Callable[[str], None] | None = None,
    ) -> None:
        self.model_name = model_name
        self.sample_id = sample_id
        self.out_path = out_path
        self.payload_for = payload_for
        self.heartbeat_seconds = heartbeat_seconds
        self._log = log_stream or (lambda msg: print(msg, file=sys.stderr, flush=True))

        self._api_turn = 0
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._last_heartbeat_state: dict[str, Any] | None = None
        self._stream_state = {
            "block_type": "starting",
            "output_chars": 0,
            "output_tokens": 0,
            "input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        self._stream_lock = threading.Lock()

    def _prefix(self) -> str:
        return f"[{self.model_name} {self.sample_id}]"

    def log(self, message: str) -> None:
        self._log(f"{self._prefix()} {message}")

    def write(self, transcript: Transcript, *, status: str) -> None:
        payload = self.payload_for(transcript, status)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(self.out_path, payload)

    def on_rollout_start(self, transcript: Transcript) -> None:
        self.write(transcript, status="in_progress")
        self.log("rollout started")

    def on_api_turn_start(self, api_turn: int) -> None:
        self._api_turn = api_turn
        with self._stream_lock:
            self._stream_state = {
                "block_type": "starting",
                "output_chars": 0,
                "output_tokens": 0,
                "input_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            }
            self._last_heartbeat_state = None
        self.log(f"api turn {api_turn} started")
        self._start_heartbeat()

    def on_message_start(self, usage: dict[str, int]) -> None:
        with self._stream_lock:
            self._stream_state.update(usage)
        self.log(
            f"api turn {self._api_turn} message started "
            f"(input={usage.get('input_tokens', 0):,}, "
            f"cache_read={usage.get('cache_read_input_tokens', 0):,}, "
            f"cache_create={usage.get('cache_creation_input_tokens', 0):,})"
        )

    def update_stream(
        self,
        *,
        block_type: str | None = None,
        output_chars: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        with self._stream_lock:
            if block_type is not None:
                self._stream_state["block_type"] = block_type
            if output_chars is not None:
                self._stream_state["output_chars"] = output_chars
            if output_tokens is not None:
                self._stream_state["output_tokens"] = output_tokens

    def on_turn_finished(self, transcript: Transcript) -> None:
        self._stop_heartbeat()
        assistant_turns = sum(1 for turn in transcript.turns if turn.get("role") == "assistant")
        self.write(transcript, status="in_progress")
        self.log(
            f"turn {assistant_turns} written "
            f"({transcript.tool_call_count} tool calls, "
            f"{transcript.usage.get('output_tokens', 0):,} output tokens so far)"
        )

    def on_rollout_complete(self, transcript: Transcript) -> None:
        self._stop_heartbeat()
        self.write(transcript, status="complete")
        self.log(
            f"rollout complete ({transcript.tool_call_count} tool calls, "
            f"{transcript.usage.get('output_tokens', 0):,} output tokens)"
        )

    def on_rollout_failed(self, transcript: Transcript) -> None:
        self._stop_heartbeat()
        self.write(transcript, status="failed")
        self.log("rollout failed")

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self.heartbeat_seconds):
            with self._stream_lock:
                state = dict(self._stream_state)
            tokens = state["output_tokens"]
            chars = state["output_chars"]
            block = state["block_type"]
            cache_read = state.get("cache_read_input_tokens", 0)
            cache_create = state.get("cache_creation_input_tokens", 0)
            size_parts: list[str] = []
            if chars:
                size_parts.append(f"{chars:,} output chars")
            if tokens:
                size_parts.append(f"{tokens:,} output tokens")
            size = ", ".join(size_parts) if size_parts else "starting"
            stale = (
                self._last_heartbeat_state is not None
                and state.get("output_chars") == self._last_heartbeat_state.get("output_chars")
                and state.get("output_tokens") == self._last_heartbeat_state.get("output_tokens")
            )
            self._last_heartbeat_state = state
            suffix = " (no stream updates since last heartbeat)" if stale else ""
            self.log(
                f"api turn {self._api_turn} in progress "
                f"({block}, {size}, cache_read={cache_read:,}, cache_create={cache_create:,}){suffix}"
            )

    def _start_heartbeat(self) -> None:
        self._stop_heartbeat()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"rollout-heartbeat-{self.sample_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        if self._heartbeat_thread is None:
            return
        self._heartbeat_stop.set()
        self._heartbeat_thread.join(timeout=1)
        self._heartbeat_thread = None


def snapshot_output_size(message: Any) -> tuple[int, str | None]:
    """Estimate streamed output size from the SDK message snapshot."""
    chars = 0
    block_type: str | None = None
    for block in message.content:
        if block.type == "thinking":
            chars += len(block.thinking or "")
            block_type = "thinking"
        elif block.type == "text":
            chars += len(block.text or "")
            block_type = "text"
        elif block.type == "tool_use":
            block_type = "tool_use"
    usage = getattr(message, "usage", None)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    return max(chars, output_tokens), block_type
