from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from mtg_ai.training.rl_policy import WeightedHeuristicPolicy
from mtg_ai.training.value_iteration_baseline import evaluate_solver_alignment


@dataclass(frozen=True)
class AuditResult:
    champion_name: str
    solver_alignment: float
    best_baseline_alignment: float
    solver_gap_to_best_baseline: float
    convergence_trailing_mean_elo_delta: float
    champion_objective_margin: float
    pass_solver_gap: int
    pass_convergence: int
    pass_margin: int
    confidence: float


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _trailing_mean(values: list[float], n: int) -> float:
    if not values:
        return 0.0
    trailing = values[-n:] if len(values) >= n else values
    return sum(trailing) / len(trailing)


def run_audit(
    champion_json: Path,
    convergence_csv: Path,
    policy_selection_csv: Path,
    seed: int,
    solver_depth: int,
    solver_samples: int,
    solver_gap_tolerance: float,
    convergence_tolerance: float,
    objective_margin_tolerance: float,
) -> AuditResult:
    champion_payload = json.loads(champion_json.read_text(encoding="utf-8"))
    champion_name = champion_payload["name"]
    policy_path = Path(champion_payload["path"])
    champion_policy = WeightedHeuristicPolicy.load(policy_path)

    alignment = evaluate_solver_alignment(
        trained_policy=champion_policy,
        seed=seed,
        life=10,
        opening_hand=4,
        depth=solver_depth,
        sample_count=solver_samples,
    )
    best_baseline = max(alignment.mixed_agreement, alignment.bolt_agreement, alignment.goblin_agreement)
    solver_gap = best_baseline - alignment.trained_agreement

    convergence_rows = _read_csv(convergence_csv)
    convergence_deltas = [float(row["elo_delta_to"]) for row in convergence_rows]
    trailing_mean = _trailing_mean(convergence_deltas, 3)

    selection_rows = _read_csv(policy_selection_csv)
    objective_margin = 0.0
    if len(selection_rows) >= 2:
        key = "objective"
        if key not in selection_rows[0]:
            key = "robust_objective"
        objective_margin = float(selection_rows[0][key]) - float(selection_rows[1][key])

    pass_solver_gap = int(solver_gap <= solver_gap_tolerance)
    pass_convergence = int(abs(trailing_mean) <= convergence_tolerance)
    pass_margin = int(objective_margin >= objective_margin_tolerance)
    confidence = (
        0.40 * max(0.0, 1.0 - max(0.0, solver_gap))
        + 0.35 * max(0.0, 1.0 - abs(trailing_mean) / max(1.0, convergence_tolerance))
        + 0.25 * min(1.0, max(0.0, objective_margin / max(0.001, objective_margin_tolerance if objective_margin_tolerance > 0 else 0.01)))
    )
    confidence = max(0.0, min(1.0, confidence))

    return AuditResult(
        champion_name=champion_name,
        solver_alignment=alignment.trained_agreement,
        best_baseline_alignment=best_baseline,
        solver_gap_to_best_baseline=solver_gap,
        convergence_trailing_mean_elo_delta=trailing_mean,
        champion_objective_margin=objective_margin,
        pass_solver_gap=pass_solver_gap,
        pass_convergence=pass_convergence,
        pass_margin=pass_margin,
        confidence=confidence,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit optimality/progress signals for selected champion policy.")
    parser.add_argument("--champion-json", default="artifacts/eval/champion.json")
    parser.add_argument("--convergence-csv", default="artifacts/eval/convergence.csv")
    parser.add_argument("--policy-selection-csv", default="artifacts/eval/champion_selection.csv")
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--solver-depth", type=int, default=7)
    parser.add_argument("--solver-samples", type=int, default=12)
    parser.add_argument("--solver-gap-tolerance", type=float, default=0.12)
    parser.add_argument("--convergence-tolerance", type=float, default=3.0)
    parser.add_argument("--objective-margin-tolerance", type=float, default=0.01)
    parser.add_argument("--output-json", default="artifacts/eval/optimality_audit.json")
    parser.add_argument("--output-csv", default="artifacts/eval/optimality_audit.csv")
    args = parser.parse_args()

    result = run_audit(
        champion_json=Path(args.champion_json),
        convergence_csv=Path(args.convergence_csv),
        policy_selection_csv=Path(args.policy_selection_csv),
        seed=args.seed,
        solver_depth=args.solver_depth,
        solver_samples=args.solver_samples,
        solver_gap_tolerance=args.solver_gap_tolerance,
        convergence_tolerance=args.convergence_tolerance,
        objective_margin_tolerance=args.objective_margin_tolerance,
    )

    payload = {
        "champion_name": result.champion_name,
        "solver_alignment": result.solver_alignment,
        "best_baseline_alignment": result.best_baseline_alignment,
        "solver_gap_to_best_baseline": result.solver_gap_to_best_baseline,
        "convergence_trailing_mean_elo_delta": result.convergence_trailing_mean_elo_delta,
        "champion_objective_margin": result.champion_objective_margin,
        "pass_solver_gap": result.pass_solver_gap,
        "pass_convergence": result.pass_convergence,
        "pass_margin": result.pass_margin,
        "confidence": result.confidence,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)

    print(f"Wrote optimality audit JSON to {output_json}")
    print(f"Wrote optimality audit CSV to {output_csv}")


if __name__ == "__main__":
    main()

