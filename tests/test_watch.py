import json
from datetime import datetime, timezone
from pathlib import Path

from harness.live import emit_live
from harness.watch import LiveTail, collect_cells, hit_max_turns, render_dashboard, summarize_cells


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_collect_cells_counts_status_grades_and_max_turns(tmp_path):
    run = tmp_path / "tools-rules"
    _write(
        run / "model-a" / "001.json",
        {
            "status": "complete",
            "run_id": "tools-rules",
            "model_name": "model-a",
            "sample_id": "001",
            "model_config": {"max_turns": 20, "max_token_continues": 5},
            "progress": {
                "assistant_turns": 2,
                "continues": 0,
                "last_stop_reason": "end_turn",
            },
            "transcript": {},
        },
    )
    _write(
        run / "model-a" / "001.judge.json",
        {"verdict": {"passed": True}},
    )
    _write(
        run / "model-a" / "001.judge.haiku.json",
        {"verdict": {"passed": False}},
    )
    _write(
        run / "model-a" / "002.json",
        {
            "status": "failed",
            "run_id": "tools-rules",
            "model_name": "model-a",
            "sample_id": "002",
            "model_config": {"max_turns": 20, "max_token_continues": 5},
            "progress": {
                "assistant_turns": 20,
                "continues": 0,
                "last_stop_reason": "tool_use",
            },
            "transcript": {},
        },
    )
    _write(
        run / "model-a" / "003.json",
        {
            "status": "in_progress",
            "run_id": "tools-rules",
            "model_name": "model-a",
            "sample_id": "003",
            "model_config": {"max_turns": 1, "max_token_continues": 5},
            "progress": {
                "assistant_turns": 1,
                "continues": 1,
                "last_stop_reason": "max_tokens",
            },
            "transcript": {},
        },
    )

    cells = collect_cells(tmp_path)
    summary = summarize_cells(cells)
    assert summary["cells"] == 3
    assert summary["complete"] == 1
    assert summary["failed"] == 1
    assert summary["in_progress"] == 1
    assert summary["judged"] == 1
    assert summary["passed"] == 1
    assert summary["errored"] == 1
    assert summary["hit_max_turns"] == 1
    assert summary["api_turns"] == 23


def test_hit_max_turns_ignores_cells_that_can_still_continue():
    assert not hit_max_turns(
        assistant_turns=1,
        continues=1,
        last_stop="max_tokens",
        max_turns=1,
        max_token_continues=5,
    )
    assert hit_max_turns(
        assistant_turns=6,
        continues=5,
        last_stop="max_tokens",
        max_turns=1,
        max_token_continues=5,
    )
    assert hit_max_turns(
        assistant_turns=20,
        continues=0,
        last_stop="tool_use",
        max_turns=20,
        max_token_continues=5,
    )


def test_live_tail_window_rates(tmp_path):
    log = tmp_path / "live.jsonl"
    now = datetime.now(timezone.utc).isoformat()
    emit_live(
        {
            "ts": now,
            "event": "stream",
            "run_id": "r",
            "model_name": "m",
            "model_id": "claude-sonnet-5",
            "sample_id": "001",
            "api_turn": 1,
            "output_tokens": 100,
            "input_tokens": 50,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        path=log,
    )
    emit_live(
        {
            "ts": now,
            "event": "stream",
            "run_id": "r",
            "model_name": "m",
            "model_id": "claude-sonnet-5",
            "sample_id": "001",
            "api_turn": 1,
            "output_tokens": 250,
            "input_tokens": 50,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        path=log,
    )
    tail = LiveTail(log, window_s=60)
    tail.poll()
    totals = tail.window_totals()
    assert totals["output_tokens"] == 150
    assert totals["output_chars"] == 0
    assert totals["usd"] > 0


def test_live_tail_counts_end_of_turn_token_jumps(tmp_path):
    log = tmp_path / "live.jsonl"
    now = datetime.now(timezone.utc).isoformat()
    emit_live(
        {
            "ts": now,
            "event": "stream",
            "run_id": "r",
            "model_name": "m",
            "model_id": "claude-sonnet-5",
            "sample_id": "001",
            "api_turn": 1,
            "output_tokens": 8,
            "output_chars": 400_000,
            "input_tokens": 50,
        },
        path=log,
    )
    emit_live(
        {
            "ts": now,
            "event": "turn_done",
            "run_id": "r",
            "model_name": "m",
            "model_id": "claude-sonnet-5",
            "sample_id": "001",
            "api_turn": 1,
            "output_tokens": 115_000,
            "output_chars": 400_000,
            "input_tokens": 50,
        },
        path=log,
    )
    tail = LiveTail(log, window_s=300)
    tail.poll()
    totals = tail.window_totals()
    assert totals["output_tokens"] == 114_992
    assert totals["usd"] > 0


def test_live_tail_estimates_cost_from_output_chars(tmp_path):
    log = tmp_path / "live.jsonl"
    now = datetime.now(timezone.utc).isoformat()
    emit_live(
        {
            "ts": now,
            "event": "stream",
            "run_id": "r",
            "model_name": "m",
            "model_id": "claude-sonnet-5",
            "sample_id": "001",
            "api_turn": 1,
            "output_tokens": 0,
            "output_chars": 4000,
            "input_tokens": 0,
        },
        path=log,
    )
    emit_live(
        {
            "ts": now,
            "event": "stream",
            "run_id": "r",
            "model_name": "m",
            "model_id": "claude-sonnet-5",
            "sample_id": "001",
            "api_turn": 1,
            "output_tokens": 0,
            "output_chars": 8000,
            "input_tokens": 0,
        },
        path=log,
    )
    tail = LiveTail(log, window_s=60)
    tail.poll()
    totals = tail.window_totals()
    assert totals["output_tokens"] == 0
    assert totals["output_chars"] == 8000
    assert totals["usd"] > 0
    text = render_dashboard(
        cells=[],
        window=totals,
        stalled=[],
        window_s=60,
    )
    assert "~2,000 est" in text


def test_render_dashboard_mentions_stalls(tmp_path):
    text = render_dashboard(
        cells=[],
        window={"output_tokens": 10, "input_tokens": 2, "output_chars": 40, "usd": 0.01, "events": 1},
        stalled=["tools-rules/model/001 (120s)"],
        window_s=60,
    )
    assert "cells 0" in text
    assert "stalled: tools-rules/model/001 (120s)" in text
    assert "+$0.0100" in text
