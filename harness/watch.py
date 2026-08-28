from __future__ import annotations

import json
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from harness.live import live_log_path
from harness.pricing import estimate_cost

SKIP_RUN_DIRS = frozenset({"site", "logs", "_archive", "_live"})
PREFIX_BYTES = 12_288
DEFAULT_WATCH_WINDOW_S = 300.0
# Thinking streams often report 0 output_tokens until the turn ends.
CHARS_PER_OUTPUT_TOKEN = 4


def _parse_ts(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _peek_prefix(path: Path, nbytes: int = PREFIX_BYTES) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read(nbytes)


def _field(prefix: str, key: str) -> str | None:
    match = re.search(rf'"{key}":\s*"((?:\\.|[^"\\])*)"', prefix)
    return match.group(1) if match else None


def _int_field(prefix: str, key: str) -> int | None:
    match = re.search(rf'"{key}":\s*(-?\d+)', prefix)
    return int(match.group(1)) if match else None


def _bool_field(prefix: str, key: str) -> bool | None:
    match = re.search(rf'"{key}":\s*(true|false)', prefix)
    if not match:
        return None
    return match.group(1) == "true"


def iter_rollout_paths(results_dir: Path) -> Iterator[Path]:
    if not results_dir.exists():
        return
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name in SKIP_RUN_DIRS or run_dir.name.startswith("."):
            continue
        for path in run_dir.rglob("*.json"):
            if ".judge." in path.name or path.name.startswith("prewarm") or path.name == "live.json":
                continue
            if path.name.endswith(".tmp"):
                continue
            yield path


def hit_max_turns(
    *,
    assistant_turns: int,
    continues: int,
    last_stop: str | None,
    max_turns: int | None,
    max_token_continues: int | None,
) -> bool:
    """True only when the cell has no remaining turns or max-token continues."""
    if last_stop in {None, "end_turn"}:
        return False
    max_continues = max_token_continues or 0
    if last_stop == "max_tokens" and continues < max_continues:
        return False
    if max_turns is None:
        return last_stop == "max_tokens" and max_continues > 0 and continues >= max_continues
    return assistant_turns >= max_turns


def collect_cells(results_dir: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for path in iter_rollout_paths(results_dir):
        try:
            prefix = _peek_prefix(path)
        except OSError:
            continue
        status = _field(prefix, "status") or "unknown"
        model = _field(prefix, "model_name") or path.parent.name
        sample_id = _field(prefix, "sample_id") or path.stem
        run_id = _field(prefix, "run_id") or path.parents[1].name
        assistant_turns = _int_field(prefix, "assistant_turns") or 0
        continues = _int_field(prefix, "continues") or 0
        last_stop = _field(prefix, "last_stop_reason")
        max_turns = _int_field(prefix, "max_turns")
        max_token_continues = _int_field(prefix, "max_token_continues")
        judge_path = path.with_name(f"{path.stem}.judge.json")
        judged = False
        passed: bool | None = None
        if judge_path.exists():
            judged = True
            try:
                passed = _bool_field(_peek_prefix(judge_path, 4000), "passed")
            except OSError:
                passed = None
        cells.append(
            {
                "path": str(path),
                "run_id": run_id,
                "model_name": model,
                "sample_id": sample_id,
                "status": status,
                "assistant_turns": assistant_turns,
                "continues": continues,
                "last_stop_reason": last_stop,
                "judged": judged,
                "passed": passed,
                "hit_max_turns": hit_max_turns(
                    assistant_turns=assistant_turns,
                    continues=continues,
                    last_stop=last_stop,
                    max_turns=max_turns,
                    max_token_continues=max_token_continues,
                ),
                "errored": status == "failed",
            }
        )
    return cells


def summarize_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    counts = defaultdict(int)
    judged = 0
    passed = 0
    failed_grade = 0
    errored = 0
    max_turns = 0
    max_tokens = 0
    api_turns = 0
    for cell in cells:
        counts[cell["status"]] += 1
        api_turns += int(cell.get("assistant_turns") or 0)
        if cell["judged"]:
            judged += 1
            if cell["passed"] is True:
                passed += 1
            elif cell["passed"] is False:
                failed_grade += 1
        if cell["errored"]:
            errored += 1
        if cell["hit_max_turns"]:
            max_turns += 1
        if cell.get("last_stop_reason") == "max_tokens":
            max_tokens += 1
    complete = counts["complete"]
    return {
        "cells": len(cells),
        "in_progress": counts["in_progress"],
        "complete": complete,
        "failed": counts["failed"],
        "other": len(cells) - complete - counts["in_progress"] - counts["failed"],
        "api_turns": api_turns,
        "judged": judged,
        "unjudged_complete": max(0, complete - judged),
        "passed": passed,
        "grade_failed": failed_grade,
        "errored": errored,
        "hit_max_turns": max_turns,
        "last_stop_max_tokens": max_tokens,
    }


class LiveTail:
    def __init__(self, path: Path, *, window_s: float = DEFAULT_WATCH_WINDOW_S) -> None:
        self.path = path
        self.window_s = window_s
        self._offset = 0
        self._last: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._deltas: deque[tuple[float, int, int, int, float]] = deque()

    def poll(self) -> None:
        if not self.path.exists():
            return
        size = self.path.stat().st_size
        if size < self._offset:
            self._offset = 0
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
            self._offset = handle.tell()
        now = time.time()
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._ingest(event, now=now)
        cutoff = now - self.window_s
        while self._deltas and self._deltas[0][0] < cutoff:
            self._deltas.popleft()

    def _ingest(self, event: dict[str, Any], *, now: float) -> None:
        ts = _parse_ts(event.get("ts")) or now
        key = (
            str(event.get("run_id") or ""),
            str(event.get("model_name") or ""),
            str(event.get("sample_id") or ""),
        )
        output = int(event.get("output_tokens") or 0)
        output_chars = int(event.get("output_chars") or 0)
        input_tokens = int(event.get("input_tokens") or 0)
        cache_read = int(event.get("cache_read_input_tokens") or 0)
        cache_create = int(event.get("cache_creation_input_tokens") or 0)
        model_id = str(event.get("model_id") or "")
        cache_ttl = str(event.get("cache_ttl") or "1h")
        prev = self._last.get(key)
        d_out = d_in = d_chars = 0
        d_read = d_create = 0
        if prev is not None and prev.get("api_turn") == event.get("api_turn"):
            d_out = max(0, output - int(prev.get("output_tokens") or 0))
            d_chars = max(0, output_chars - int(prev.get("output_chars") or 0))
            d_in = max(0, input_tokens - int(prev.get("input_tokens") or 0))
            d_read = max(0, cache_read - int(prev.get("cache_read") or 0))
            d_create = max(0, cache_create - int(prev.get("cache_create") or 0))
        elif prev is None or prev.get("api_turn") != event.get("api_turn"):
            d_in = input_tokens
            d_read = cache_read
            d_create = cache_create
            d_chars = output_chars
            d_out = output if event.get("event") == "message_start" else 0
        billed_out = d_out or (d_chars // CHARS_PER_OUTPUT_TOKEN)
        d_cost = 0.0
        if model_id and (billed_out or d_in or d_read or d_create):
            try:
                d_cost = float(
                    estimate_cost(
                        {
                            "output_tokens": billed_out,
                            "input_tokens": d_in,
                            "cache_read_input_tokens": d_read,
                            "cache_creation_input_tokens": d_create,
                        },
                        model_id=model_id,
                        cache_ttl=cache_ttl,
                    )["usd"]
                )
            except KeyError:
                d_cost = 0.0
        if d_out or d_in or d_chars or d_cost:
            self._deltas.append((ts, d_out, d_in, d_chars, d_cost))
        self._last[key] = {
            "ts": ts,
            "api_turn": event.get("api_turn"),
            "output_tokens": output,
            "output_chars": output_chars,
            "input_tokens": input_tokens,
            "cache_read": cache_read,
            "cache_create": cache_create,
            "event": event.get("event"),
        }

    def window_totals(self) -> dict[str, float]:
        out = inp = chars = 0
        usd = 0.0
        for _, d_out, d_in, d_chars, d_cost in self._deltas:
            out += d_out
            inp += d_in
            chars += d_chars
            usd += d_cost
        return {
            "output_tokens": out,
            "input_tokens": inp,
            "output_chars": chars,
            "usd": usd,
            "events": len(self._deltas),
        }

    def stalled(self, *, stall_after_s: float, now: float | None = None) -> list[str]:
        now = now or time.time()
        names: list[str] = []
        for (run_id, model, sample), snap in sorted(self._last.items()):
            event = snap.get("event")
            if event in {"complete", "failed"}:
                continue
            age = now - float(snap.get("ts") or 0)
            if age >= stall_after_s:
                names.append(f"{run_id}/{model}/{sample} ({age:.0f}s)")
        return names


def format_window(window_s: float) -> str:
    if window_s >= 60 and abs(window_s / 60 - round(window_s / 60)) < 1e-6:
        return f"{int(round(window_s / 60))}m"
    return f"{window_s:.0f}s"


def _est_tok_suffix(window: dict[str, float]) -> str:
    out = int(window.get("output_tokens") or 0)
    chars = int(window.get("output_chars") or 0)
    if out or not chars:
        return ""
    return f" (~{chars // CHARS_PER_OUTPUT_TOKEN:,} est)"


def render_dashboard(
    *,
    cells: list[dict[str, Any]],
    window: dict[str, float],
    stalled: list[str],
    window_s: float,
    now: datetime | None = None,
) -> str:
    summary = summarize_cells(cells)
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"=== eval watch  {stamp}  window={format_window(window_s)} ===",
        (
            f"cells {summary['cells']}   "
            f"in_progress {summary['in_progress']}   "
            f"complete {summary['complete']}   "
            f"failed {summary['failed']}   "
            f"api_turns {summary['api_turns']}"
        ),
        (
            f"graded {summary['judged']} "
            f"(pass {summary['passed']}, fail {summary['grade_failed']})   "
            f"unjudged_complete {summary['unjudged_complete']}   "
            f"errored {summary['errored']}   "
            f"hit_max_turns {summary['hit_max_turns']}   "
            f"last_stop=max_tokens {summary['last_stop_max_tokens']}"
        ),
        (
            f"last {format_window(window_s)}:  "
            f"+{int(window['output_tokens']):,} out tok"
            f"{_est_tok_suffix(window)}   "
            f"+{int(window.get('output_chars') or 0):,} out chars   "
            f"+{int(window['input_tokens']):,} in tok   "
            f"+${window['usd']:.4f}"
        ),
    ]
    if stalled:
        lines.append("stalled: " + ", ".join(stalled[:8]))
        if len(stalled) > 8:
            lines.append(f"  … {len(stalled) - 8} more")
    else:
        lines.append("stalled: none")
    return "\n".join(lines)


def watch_loop(
    results_dir: Path,
    *,
    interval_s: float = 5.0,
    window_s: float = DEFAULT_WATCH_WINDOW_S,
    stall_after_s: float = 90.0,
    once: bool = False,
) -> None:
    tail = LiveTail(live_log_path(results_dir), window_s=window_s)
    while True:
        cells = collect_cells(results_dir)
        tail.poll()
        text = render_dashboard(
            cells=cells,
            window=tail.window_totals(),
            stalled=tail.stalled(stall_after_s=stall_after_s),
            window_s=window_s,
        )
        print(text, flush=True)
        if once:
            return
        time.sleep(interval_s)
        print(flush=True)
