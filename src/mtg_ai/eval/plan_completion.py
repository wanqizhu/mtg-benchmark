from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlanCheck:
    check_id: str
    description: str
    passed: bool
    detail: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completion status for execution-plan acceptance checks.")
    parser.add_argument("--milestones-csv", default="artifacts/eval/milestones.csv")
    parser.add_argument("--generalization-csv", default="artifacts/eval/generalization_matrix_tuned.csv")
    parser.add_argument("--deck-vs-starter-csv", default="artifacts/eval/deck_vs_starter_tuned.csv")
    parser.add_argument("--convergence-csv", default="artifacts/eval/convergence.csv")
    parser.add_argument("--optimality-json", default="artifacts/eval/optimality_audit.json")
    parser.add_argument("--replay-index-csv", default="artifacts/replays/samples/index.csv")
    parser.add_argument("--report-md", default="artifacts/report.md")
    parser.add_argument("--output-csv", default="artifacts/eval/plan_completion.csv")
    args = parser.parse_args()

    milestones = _read_csv(Path(args.milestones_csv))
    milestone_map = {row["milestone"]: row for row in milestones}
    checks: list[PlanCheck] = []

    checks.append(
        PlanCheck(
            "rules_and_milestones",
            "Rules engine + baseline/trained milestone suite passes",
            sum(int(row["passed"]) for row in milestones) >= max(1, len(milestones) - 1),
            f"passed={sum(int(row['passed']) for row in milestones)}/{len(milestones)} (allows <=1 fail)",
        )
    )

    replay_rows = _read_csv(Path(args.replay_index_csv))
    checks.append(
        PlanCheck(
            "replay_interface",
            "Replay index contains representative sample games",
            len(replay_rows) >= 4,
            f"replay_rows={len(replay_rows)}",
        )
    )

    generalization_rows = _read_csv(Path(args.generalization_csv))
    mixed_rows = [row for row in generalization_rows if row["opponent"] == "mixed"]
    min_mixed = min(float(row["trained_win_rate"]) for row in mixed_rows) if mixed_rows else 0.0
    checks.append(
        PlanCheck(
            "generalization",
            "Champion policy generalizes across life/hand settings",
            min_mixed >= 0.45,
            f"min_winrate_vs_mixed={min_mixed:.4f} (threshold=0.45)",
        )
    )

    deck_rows = _read_csv(Path(args.deck_vs_starter_csv))
    min_deck = min(float(row["best_vs_starter_win_rate"]) for row in deck_rows) if deck_rows else 0.0
    checks.append(
        PlanCheck(
            "deck_better_than_starter",
            "Searched decks outperform starter deck across scenarios",
            min_deck >= 0.5,
            f"min_best_vs_starter={min_deck:.4f}",
        )
    )

    convergence_rows = _read_csv(Path(args.convergence_csv))
    deltas = [float(row["elo_delta_to"]) for row in convergence_rows]
    trailing = deltas[-3:] if len(deltas) >= 3 else deltas
    trailing_mean = sum(trailing) / len(trailing) if trailing else 0.0
    checks.append(
        PlanCheck(
            "convergence_signal",
            "League progression shows narrowing checkpoint gains",
            abs(trailing_mean) <= 5.0,
            f"trailing_mean_elo_delta={trailing_mean:+.4f}",
        )
    )

    optimality = json.loads(Path(args.optimality_json).read_text(encoding="utf-8"))
    checks.append(
        PlanCheck(
            "solver_alignment",
            "Solver alignment is strong in tractable regimes",
            float(optimality["solver_alignment"]) >= 0.70,
            f"solver_alignment={float(optimality['solver_alignment']):.4f}",
        )
    )

    checks.append(
        PlanCheck(
            "optimality_confidence",
            "Overall optimality audit confidence is acceptable",
            float(optimality["confidence"]) >= 0.35,
            f"confidence={float(optimality['confidence']):.4f}",
        )
    )

    report_text = Path(args.report_md).read_text(encoding="utf-8")
    required_sections = [
        "Milestone checklist",
        "Checkpoint convergence analysis",
        "Recommended champion policy",
        "Head-to-head significance among top candidates",
        "Artifact manifest",
        "Optimality audit",
    ]
    missing = [section for section in required_sections if section not in report_text]
    checks.append(
        PlanCheck(
            "report_completeness",
            "Report contains required comprehensive sections",
            not missing,
            "missing=" + ",".join(missing) if missing else "all present",
        )
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["check_id", "description", "passed", "detail"])
        writer.writeheader()
        for check in checks:
            writer.writerow(
                {
                    "check_id": check.check_id,
                    "description": check.description,
                    "passed": int(check.passed),
                    "detail": check.detail,
                }
            )

    total = len(checks)
    passed = sum(int(check.passed) for check in checks)
    print(f"Wrote plan completion checks to {output_path} ({passed}/{total} passed)")


if __name__ == "__main__":
    main()

