#!/usr/bin/env python3
"""Backfill cost estimates into existing result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.config import DEFAULT_JUDGE_MODEL, RESULTS_DIR, get_model
from harness.pricing import estimate_cost, infer_cache_ttl


def backfill_rollout(path: Path) -> float | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    usage = payload.get("transcript", {}).get("usage", {})
    model_id = payload.get("model_id")
    if not usage or not model_id:
        return None

    cache_ttl = infer_cache_ttl(payload, result_path=str(path))
    payload["cost"] = estimate_cost(usage, model_id=model_id, cache_ttl=cache_ttl)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload["cost"]["usd"]


def backfill_judge(path: Path) -> float | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    usage = payload.get("usage") or payload.get("verdict", {}).get("usage", {})
    if not usage:
        return None

    model_id = payload.get("judge_model_id") or get_model(DEFAULT_JUDGE_MODEL).model_id
    payload.setdefault("judge_model_id", model_id)
    payload["cost"] = estimate_cost(usage, model_id=model_id)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload["cost"]["usd"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Results directory to update (default: results/)",
    )
    args = parser.parse_args()

    rollout_total = 0.0
    rollout_count = 0
    for path in sorted(args.results_dir.rglob("*.json")):
        if path.name.endswith(".judge.json"):
            continue
        cost = backfill_rollout(path)
        if cost is None:
            continue
        rollout_total += cost
        rollout_count += 1
        print(f"rollout {path}: ${cost:.4f}")

    judge_total = 0.0
    judge_count = 0
    for path in sorted(args.results_dir.rglob("*.judge.json")):
        cost = backfill_judge(path)
        if cost is None:
            continue
        judge_total += cost
        judge_count += 1
        print(f"judge  {path}: ${cost:.4f}")

    print(
        f"\nUpdated {rollout_count} rollout files (${rollout_total:.4f}) "
        f"and {judge_count} judge files (${judge_total:.4f}); "
        f"total ${rollout_total + judge_total:.4f}"
    )


if __name__ == "__main__":
    main()
