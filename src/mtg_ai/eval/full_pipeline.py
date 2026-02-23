from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_step(label: str, module: str, args: list[str]) -> None:
    print(f"[pipeline] {label}")
    command = [sys.executable, "-m", module, *args]
    completed = subprocess.run(command, check=False, cwd=Path.cwd())
    if completed.returncode != 0:
        raise RuntimeError(f"Step failed: {label} ({module})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete evaluation/report pipeline for constrained MTG AI.")
    parser.add_argument("--seed", type=int, default=103)
    parser.add_argument("--champion-games", type=int, default=60)
    parser.add_argument("--significance-games", type=int, default=240)
    parser.add_argument("--milestone-games", type=int, default=120)
    parser.add_argument("--convergence-games", type=int, default=80)
    parser.add_argument("--output-report", default="artifacts/report.md")
    args = parser.parse_args()

    run_step(
        "champion selection",
        "mtg_ai.eval.champion",
        [
            "--games",
            str(args.champion_games),
            "--seed",
            str(args.seed + 5000),
            "--output-csv",
            "artifacts/eval/champion_selection.csv",
            "--output-json",
            "artifacts/eval/champion.json",
        ],
    )
    champion_payload = json.loads(Path("artifacts/eval/champion.json").read_text(encoding="utf-8"))
    champion_policy_path = champion_payload["path"]
    run_step(
        "generalization matrix",
        "mtg_ai.eval.generalization",
        [
            "--policy-path",
            champion_policy_path,
            "--life-values",
            "5,10,20",
            "--hand-values",
            "5,7",
            "--games",
            "120",
            "--seed",
            str(args.seed + 5200),
            "--output-csv",
            "artifacts/eval/generalization_matrix_tuned.csv",
        ],
    )
    run_step(
        "deck search",
        "mtg_ai.deck.search",
        [
            "--policy",
            "trained",
            "--trained-path",
            champion_policy_path,
            "--deck-size",
            "12",
            "--mode",
            "evolutionary",
            "--population-size",
            "8",
            "--generations",
            "6",
            "--games-per-opponent",
            "8",
            "--life-values",
            "5,10,20",
            "--hand-sizes",
            "5,7",
            "--seed",
            str(args.seed + 5300),
            "--output-csv",
            "artifacts/deck_search/results_tuned.csv",
        ],
    )
    run_step(
        "deck validation",
        "mtg_ai.eval.deck_validation",
        [
            "--policy-path",
            champion_policy_path,
            "--deck-search-csv",
            "artifacts/deck_search/results_tuned.csv",
            "--games",
            "100",
            "--seed",
            str(args.seed + 5400),
            "--output-csv",
            "artifacts/eval/deck_vs_starter_tuned.csv",
        ],
    )
    run_step(
        "policy frontier selection",
        "mtg_ai.eval.policy_selection",
        [
            "--policy-globs",
            "artifacts/training/self_play*/policy*.json,artifacts/training/solver_tune/*.json,artifacts/training/sweep_seed_*/policy_final.json,artifacts/training/sweep_runs/*/policy_final.json",
            "--games",
            "16",
            "--seed",
            str(args.seed + 5500),
            "--top-k",
            "6",
            "--alignment-depth",
            "7",
            "--alignment-samples",
            "6",
            "--output-csv",
            "artifacts/eval/policy_selection.csv",
            "--matrix-csv",
            "artifacts/eval/policy_selection_matrix.csv",
        ],
    )
    run_step(
        "pairwise significance",
        "mtg_ai.eval.significance",
        [
            "--selection-csv",
            "artifacts/eval/champion_selection.csv",
            "--top-k",
            "5",
            "--games",
            str(args.significance_games),
            "--seed",
            str(args.seed + 5600),
            "--life",
            "10",
            "--opening-hand",
            "7",
            "--output-csv",
            "artifacts/eval/significance.csv",
        ],
    )
    run_step(
        "milestone suite",
        "mtg_ai.eval.milestones",
        [
            "--policy-path",
            champion_policy_path,
            "--games",
            str(args.milestone_games),
            "--seed",
            str(args.seed + 5700),
            "--output-csv",
            "artifacts/eval/milestones.csv",
        ],
    )
    run_step(
        "convergence analysis",
        "mtg_ai.eval.convergence",
        [
            "--checkpoint-dir",
            "artifacts/training/self_play_multi",
            "--games",
            str(args.convergence_games),
            "--life",
            "10",
            "--opening-hand",
            "7",
            "--seed",
            str(args.seed + 5800),
            "--output-csv",
            "artifacts/eval/convergence.csv",
        ],
    )
    run_step(
        "replay samples",
        "mtg_ai.eval.replay_samples",
        [
            "--champion-json",
            "artifacts/eval/champion.json",
            "--output-dir",
            "artifacts/replays/samples",
            "--index-csv",
            "artifacts/replays/samples/index.csv",
            "--seed",
            str(args.seed + 5900),
        ],
    )
    run_step(
        "artifact manifest",
        "mtg_ai.eval.manifest",
        [
            "--paths",
            "artifacts/eval/champion.json",
            "artifacts/eval/champion_selection.csv",
            "artifacts/eval/generalization_matrix_tuned.csv",
            "artifacts/deck_search/results_tuned.csv",
            "artifacts/eval/deck_vs_starter_tuned.csv",
            "artifacts/eval/policy_selection.csv",
            "artifacts/eval/policy_selection_matrix.csv",
            "artifacts/eval/significance.csv",
            "artifacts/eval/milestones.csv",
            "artifacts/eval/convergence.csv",
            "artifacts/replays/samples/index.csv",
            "--output-json",
            "artifacts/eval/manifest.json",
        ],
    )
    run_step(
        "optimality audit",
        "mtg_ai.eval.optimality_audit",
        [
            "--champion-json",
            "artifacts/eval/champion.json",
            "--convergence-csv",
            "artifacts/eval/convergence.csv",
            "--policy-selection-csv",
            "artifacts/eval/champion_selection.csv",
            "--seed",
            str(args.seed + 6100),
            "--solver-depth",
            "7",
            "--solver-samples",
            "12",
            "--solver-gap-tolerance",
            "0.12",
            "--convergence-tolerance",
            "3.0",
            "--objective-margin-tolerance",
            "0.01",
            "--output-json",
            "artifacts/eval/optimality_audit.json",
            "--output-csv",
            "artifacts/eval/optimality_audit.csv",
        ],
    )
    run_step(
        "final report",
        "mtg_ai.analysis.report",
        [
            "--training-dir",
            "artifacts/training/self_play_multi",
            "--deck-search-csv",
            "artifacts/deck_search/results_tuned.csv",
            "--generalization-csv",
            "artifacts/eval/generalization_matrix_tuned.csv",
            "--sweep-csv",
            "artifacts/training/sweep_results.csv",
            "--solver-tune-history-csv",
            "artifacts/training/solver_tune/history.csv",
            "--deck-vs-starter-csv",
            "artifacts/eval/deck_vs_starter_tuned.csv",
            "--policy-selection-csv",
            "artifacts/eval/policy_selection.csv",
            "--policy-matrix-csv",
            "artifacts/eval/policy_selection_matrix.csv",
            "--milestones-csv",
            "artifacts/eval/milestones.csv",
            "--convergence-csv",
            "artifacts/eval/convergence.csv",
            "--champion-json",
            "artifacts/eval/champion.json",
            "--champion-csv",
            "artifacts/eval/champion_selection.csv",
            "--significance-csv",
            "artifacts/eval/significance.csv",
            "--replay-index-csv",
            "artifacts/replays/samples/index.csv",
            "--manifest-json",
            "artifacts/eval/manifest.json",
            "--optimality-audit-json",
            "artifacts/eval/optimality_audit.json",
            "--output",
            args.output_report,
            "--seed",
            str(args.seed),
        ],
    )
    print(f"[pipeline] complete: {args.output_report}")


if __name__ == "__main__":
    main()

