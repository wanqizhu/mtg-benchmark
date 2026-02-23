from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from mtg_ai.bots.fixed_bots import BoltFaceBot, GoblinAggroBot, MixedBoltFaceBot
from mtg_ai.bots.random_bot import RandomBot
from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.eval.tournament import Entrant, play_match, round_robin
from mtg_ai.training.rl_policy import WeightedHeuristicPolicy
from mtg_ai.training.value_iteration_baseline import evaluate_solver_alignment


@dataclass(frozen=True)
class PolicyCandidate:
    name: str
    path: Path
    policy: WeightedHeuristicPolicy


def _evaluate_candidate(
    candidate: PolicyCandidate,
    deck: list[str],
    scenarios: list[tuple[int, int]],
    games: int,
    seed: int,
    alignment_depth: int,
    alignment_samples: int,
) -> dict[str, float]:
    baseline_values: list[float] = []
    mixed_values: list[float] = []
    for idx, (life, hand) in enumerate(scenarios):
        cfg = GameConfig(starting_life=life, opening_hand_size=hand, random_seed=seed + idx * 10)
        summaries = [
            play_match(
                Entrant("candidate", candidate.policy, deck),
                Entrant("random", RandomBot(), deck),
                games=games,
                config=cfg,
                seed=seed + idx * 100 + 11,
            ),
            play_match(
                Entrant("candidate", candidate.policy, deck),
                Entrant("bolt", BoltFaceBot(), deck),
                games=games,
                config=cfg,
                seed=seed + idx * 100 + 29,
            ),
            play_match(
                Entrant("candidate", candidate.policy, deck),
                Entrant("goblin", GoblinAggroBot(), deck),
                games=games,
                config=cfg,
                seed=seed + idx * 100 + 47,
            ),
            play_match(
                Entrant("candidate", candidate.policy, deck),
                Entrant("mixed", MixedBoltFaceBot(), deck),
                games=games,
                config=cfg,
                seed=seed + idx * 100 + 71,
            ),
        ]
        baseline_values.extend(summary.win_rate_a for summary in summaries)
        mixed_values.append(summaries[-1].win_rate_a)

    baseline_avg = sum(baseline_values) / len(baseline_values)
    mixed_avg = sum(mixed_values) / len(mixed_values)
    mixed_floor = min(mixed_values)
    alignment = evaluate_solver_alignment(
        trained_policy=candidate.policy,
        seed=seed + 500,
        life=10,
        opening_hand=4,
        depth=alignment_depth,
        sample_count=alignment_samples,
    )
    solver_alignment = alignment.trained_agreement
    objective = 0.45 * baseline_avg + 0.35 * mixed_floor + 0.20 * solver_alignment
    return {
        "baseline_avg": baseline_avg,
        "mixed_avg": mixed_avg,
        "mixed_floor": mixed_floor,
        "solver_alignment": solver_alignment,
        "objective": objective,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best policy from artifact candidates.")
    parser.add_argument("--policy-globs", default="artifacts/training/*/policy*.json")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2501)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--alignment-depth", type=int, default=7)
    parser.add_argument("--alignment-samples", type=int, default=8)
    parser.add_argument("--output-csv", default="artifacts/eval/policy_selection.csv")
    parser.add_argument("--matrix-csv", default="artifacts/eval/policy_selection_matrix.csv")
    args = parser.parse_args()

    patterns = [pattern.strip() for pattern in args.policy_globs.split(",") if pattern.strip()]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(".").glob(pattern))
    unique_paths = sorted({path.resolve() for path in paths})
    if not unique_paths:
        raise ValueError("No policy files found for provided glob patterns.")

    candidates: list[PolicyCandidate] = []
    for path in unique_paths:
        try:
            policy = WeightedHeuristicPolicy.load(path)
        except (KeyError, FileNotFoundError):
            continue
        name = path.parent.name + "/" + path.name
        candidates.append(PolicyCandidate(name=name, path=path, policy=policy))
    if not candidates:
        raise ValueError("No compatible weighted policies found.")

    deck = ["Mountain"] * 12 + ["Lightning Bolt"] * 9 + ["Raging Goblin"] * 9
    scenarios = [(5, 5), (5, 7), (10, 5), (10, 7), (20, 5), (20, 7)]
    rows: list[dict[str, object]] = []
    for idx, candidate in enumerate(candidates):
        metrics = _evaluate_candidate(
            candidate=candidate,
            deck=deck,
            scenarios=scenarios,
            games=args.games,
            seed=args.seed + idx * 101,
            alignment_depth=args.alignment_depth,
            alignment_samples=args.alignment_samples,
        )
        rows.append(
            {
                "name": candidate.name,
                "path": str(candidate.path),
                "objective": f"{metrics['objective']:.4f}",
                "baseline_avg": f"{metrics['baseline_avg']:.4f}",
                "mixed_avg": f"{metrics['mixed_avg']:.4f}",
                "mixed_floor": f"{metrics['mixed_floor']:.4f}",
                "solver_alignment": f"{metrics['solver_alignment']:.4f}",
            }
        )
        print(rows[-1])

    rows.sort(key=lambda row: float(row["objective"]), reverse=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote policy ranking to {output_csv}")

    top = rows[: max(2, min(args.top_k, len(rows)))]
    entrants = []
    for row in top:
        policy = WeightedHeuristicPolicy.load(Path(row["path"]))
        entrants.append(Entrant(row["name"], policy, deck))
    summaries, ratings = round_robin(
        entrants=entrants,
        games_per_pair=max(8, args.games // 2),
        config=GameConfig(starting_life=10, opening_hand_size=7, random_seed=args.seed + 9000),
        seed=args.seed + 9000,
    )
    matrix_rows: list[dict[str, object]] = []
    for summary in summaries:
        matrix_rows.append(
            {
                "a": summary.entrant_a,
                "b": summary.entrant_b,
                "games": summary.games,
                "wins_a": summary.wins_a,
                "wins_b": summary.wins_b,
                "draws": summary.draws,
                "win_rate_a": f"{summary.win_rate_a:.4f}",
            }
        )
    for name, rating in ratings.items():
        matrix_rows.append(
            {
                "a": name,
                "b": "__elo__",
                "games": "",
                "wins_a": "",
                "wins_b": "",
                "draws": "",
                "win_rate_a": f"{rating:.2f}",
            }
        )

    matrix_csv = Path(args.matrix_csv)
    matrix_csv.parent.mkdir(parents=True, exist_ok=True)
    with matrix_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix_rows[0].keys()))
        writer.writeheader()
        writer.writerows(matrix_rows)
    print(f"Wrote policy matrix to {matrix_csv}")


if __name__ == "__main__":
    main()

