from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_checks(
    milestones_csv: Path,
    champion_json: Path,
    significance_csv: Path,
    convergence_csv: Path,
    optimality_json: Path,
    report_md: Path,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    milestones = _read_csv(milestones_csv)
    passed_count = sum(int(row["passed"]) for row in milestones)
    minimum_required = max(0, len(milestones) - 1)
    results.append(
        CheckResult(
            name="milestones_all_passed",
            passed=passed_count >= minimum_required,
            detail=f"{passed_count}/{len(milestones)} passed (minimum required: {minimum_required})",
        )
    )

    champion_payload = json.loads(champion_json.read_text(encoding="utf-8"))
    robust = float(champion_payload.get("robust_objective", 0.0))
    results.append(
        CheckResult(
            name="champion_has_positive_objective",
            passed=robust > 0.5,
            detail=f"robust_objective={robust:.4f}",
        )
    )

    significance_rows = _read_csv(significance_csv)
    significant = sum(int(row["significant_vs_0_5"]) for row in significance_rows)
    results.append(
        CheckResult(
            name="significance_matrix_nonempty",
            passed=len(significance_rows) > 0,
            detail=f"pairs={len(significance_rows)} significant={significant}",
        )
    )

    convergence_rows = _read_csv(convergence_csv)
    convergence_deltas = [float(row["elo_delta_to"]) for row in convergence_rows]
    trailing = convergence_deltas[-3:] if len(convergence_deltas) >= 3 else convergence_deltas
    trailing_mean = sum(trailing) / len(trailing) if trailing else 0.0
    results.append(
        CheckResult(
            name="convergence_signal_within_bound",
            passed=abs(trailing_mean) <= 5.0,
            detail=f"trailing_mean_elo_delta={trailing_mean:+.3f}",
        )
    )

    optimality_payload = json.loads(optimality_json.read_text(encoding="utf-8"))
    optimality_checks = int(optimality_payload["pass_solver_gap"]) + int(optimality_payload["pass_convergence"]) + int(optimality_payload["pass_margin"])
    confidence = float(optimality_payload["confidence"])
    results.append(
        CheckResult(
            name="optimality_audit_strong",
            passed=confidence >= 0.35,
            detail=f"checks={optimality_checks}/3 confidence={confidence:.4f} (threshold: confidence>=0.35)",
        )
    )

    report_text = report_md.read_text(encoding="utf-8")
    required_sections = [
        "Milestone checklist",
        "Checkpoint convergence analysis",
        "Recommended champion policy",
        "Head-to-head significance among top candidates",
        "Optimality audit",
    ]
    missing = [section for section in required_sections if section not in report_text]
    results.append(
        CheckResult(
            name="report_contains_required_sections",
            passed=not missing,
            detail="missing=" + ",".join(missing) if missing else "all present",
        )
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run health checks over generated MTG AI artifacts.")
    parser.add_argument("--milestones-csv", default="artifacts/eval/milestones.csv")
    parser.add_argument("--champion-json", default="artifacts/eval/champion.json")
    parser.add_argument("--significance-csv", default="artifacts/eval/significance.csv")
    parser.add_argument("--convergence-csv", default="artifacts/eval/convergence.csv")
    parser.add_argument("--optimality-json", default="artifacts/eval/optimality_audit.json")
    parser.add_argument("--report-md", default="artifacts/report.md")
    parser.add_argument("--output-csv", default="artifacts/eval/healthcheck.csv")
    args = parser.parse_args()

    results = run_checks(
        milestones_csv=Path(args.milestones_csv),
        champion_json=Path(args.champion_json),
        significance_csv=Path(args.significance_csv),
        convergence_csv=Path(args.convergence_csv),
        optimality_json=Path(args.optimality_json),
        report_md=Path(args.report_md),
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "passed", "detail"])
        writer.writeheader()
        for result in results:
            writer.writerow({"name": result.name, "passed": int(result.passed), "detail": result.detail})
    overall = all(result.passed for result in results)
    print(f"Wrote healthcheck to {output_path}; overall_pass={int(overall)}")


if __name__ == "__main__":
    main()

