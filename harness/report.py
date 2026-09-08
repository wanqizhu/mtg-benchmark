from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _rate_row(name: str, results: list[bool]) -> dict[str, object]:
    passed = sum(results)
    total = len(results)
    return {
        "name": name,
        "passed": passed,
        "total": total,
        "rate": passed / total if total else 0.0,
    }


def _sum_costs(paths: list[Path]) -> tuple[float, float]:
    rollout_total = 0.0
    judge_total = 0.0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cost = (payload.get("cost") or {}).get("usd")
        if cost is None:
            continue
        if path.name.endswith(".judge.json"):
            judge_total += float(cost)
        else:
            rollout_total += float(cost)
    return rollout_total, judge_total


def build_report(run_dir: Path) -> dict[str, object]:
    judge_files = sorted(run_dir.rglob("*.judge.json"))
    if not judge_files:
        return {
            "run_dir": str(run_dir),
            "has_judges": False,
            "by_model": [],
            "by_difficulty": [],
            "cost": {"rollouts_usd": 0.0, "judges_usd": 0.0, "total_usd": 0.0},
        }

    by_model: dict[str, list[bool]] = defaultdict(list)
    by_difficulty: dict[str, list[bool]] = defaultdict(list)

    for path in judge_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        passed = bool(payload["verdict"]["passed"])
        model = payload["model_name"]
        by_model[model].append(passed)

        result_path = path.with_suffix("").with_suffix(".json")
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            difficulty = result.get("reference", {}).get("difficulty")
            if difficulty:
                by_difficulty[difficulty].append(passed)

    rollout_paths = [p for p in run_dir.rglob("*.json") if not p.name.endswith(".judge.json")]
    rollout_cost, judge_cost = _sum_costs(rollout_paths + judge_files)
    return {
        "run_dir": str(run_dir),
        "has_judges": True,
        "by_model": [_rate_row(model, results) for model, results in sorted(by_model.items())],
        "by_difficulty": [
            _rate_row(difficulty, results) for difficulty, results in sorted(by_difficulty.items())
        ],
        "cost": {
            "rollouts_usd": rollout_cost,
            "judges_usd": judge_cost,
            "total_usd": rollout_cost + judge_cost,
        },
    }


def print_report(run_dir: Path) -> None:
    report = build_report(run_dir)
    if not report["has_judges"]:
        print(f"No judge results found in {run_dir}")
        return

    print(f"Report for {run_dir}\n")
    print("By model:")
    for row in report["by_model"]:
        print(f"  {row['name']}: {row['passed']}/{row['total']} ({row['rate']:.1%})")

    if report["by_difficulty"]:
        print("\nBy difficulty:")
        for row in report["by_difficulty"]:
            print(f"  {row['name']}: {row['passed']}/{row['total']} ({row['rate']:.1%})")

    cost = report["cost"]
    if cost["rollouts_usd"] or cost["judges_usd"]:
        print("\nEstimated API cost:")
        print(f"  rollouts: ${cost['rollouts_usd']:.4f}")
        print(f"  judges:   ${cost['judges_usd']:.4f}")
        print(f"  total:    ${cost['total_usd']:.4f}")
