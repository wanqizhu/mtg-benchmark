from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

from mtg_ai.bots.fixed_bots import BoltFaceBot, GoblinAggroBot, MixedBoltFaceBot
from mtg_ai.bots.random_bot import RandomBot
from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.eval.tournament import Entrant, play_match
from mtg_ai.training.rl_policy import WeightedHeuristicPolicy
from mtg_ai.training.value_iteration_baseline import evaluate_solver_alignment


@dataclass(frozen=True)
class CandidateMetrics:
    objective: float
    baseline_avg: float
    mixed_floor: float
    solver_alignment: float
    mixed_avg: float


def _evaluate_candidate(
    policy: WeightedHeuristicPolicy,
    deck: list[str],
    seed: int,
    games: int,
    scenarios: list[tuple[int, int]],
    alignment_depth: int,
    alignment_samples: int,
) -> CandidateMetrics:
    baseline_values: list[float] = []
    mixed_values: list[float] = []

    for idx, (life, hand) in enumerate(scenarios):
        cfg = GameConfig(starting_life=life, opening_hand_size=hand, random_seed=seed + idx * 100)
        summaries = [
            play_match(
                Entrant("trained", policy, deck),
                Entrant("random", RandomBot(), deck),
                games=games,
                config=cfg,
                seed=seed + idx * 1000 + 11,
            ),
            play_match(
                Entrant("trained", policy, deck),
                Entrant("bolt", BoltFaceBot(), deck),
                games=games,
                config=cfg,
                seed=seed + idx * 1000 + 29,
            ),
            play_match(
                Entrant("trained", policy, deck),
                Entrant("goblin", GoblinAggroBot(), deck),
                games=games,
                config=cfg,
                seed=seed + idx * 1000 + 47,
            ),
            play_match(
                Entrant("trained", policy, deck),
                Entrant("mixed", MixedBoltFaceBot(), deck),
                games=games,
                config=cfg,
                seed=seed + idx * 1000 + 71,
            ),
        ]
        baseline_values.extend(summary.win_rate_a for summary in summaries)
        mixed_values.append(summaries[-1].win_rate_a)

    baseline_avg = sum(baseline_values) / len(baseline_values)
    mixed_avg = sum(mixed_values) / len(mixed_values)
    mixed_floor = min(mixed_values)
    alignment = evaluate_solver_alignment(
        trained_policy=policy,
        seed=seed + 333,
        life=10,
        opening_hand=4,
        depth=alignment_depth,
        sample_count=alignment_samples,
    )
    objective = 0.45 * baseline_avg + 0.35 * mixed_floor + 0.20 * alignment.trained_agreement
    return CandidateMetrics(
        objective=objective,
        baseline_avg=baseline_avg,
        mixed_floor=mixed_floor,
        solver_alignment=alignment.trained_agreement,
        mixed_avg=mixed_avg,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune weighted policy parameters with solver-aware objective.")
    parser.add_argument("--base-policy-path", default="artifacts/training/self_play_multi/policy_final.json")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--candidates-per-iter", type=int, default=6)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=313)
    parser.add_argument("--alignment-depth", type=int, default=7)
    parser.add_argument("--alignment-samples", type=int, default=10)
    parser.add_argument("--output-dir", default="artifacts/training/solver_tune")
    args = parser.parse_args()

    deck = ["Mountain"] * 12 + ["Lightning Bolt"] * 9 + ["Raging Goblin"] * 9
    scenarios = [(5, 5), (5, 7), (10, 5), (10, 7), (20, 5), (20, 7)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    best_policy_path = output_dir / "policy_best.json"

    rng = random.Random(args.seed)
    best_policy = WeightedHeuristicPolicy.load(args.base_policy_path)
    best_metrics = _evaluate_candidate(
        best_policy,
        deck=deck,
        seed=args.seed,
        games=args.games,
        scenarios=scenarios,
        alignment_depth=args.alignment_depth,
        alignment_samples=args.alignment_samples,
    )
    sigma = args.sigma

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "accepted",
                "sigma",
                "objective",
                "baseline_avg",
                "mixed_avg",
                "mixed_floor",
                "solver_alignment",
            ],
        )
        writer.writeheader()

        for iteration in range(1, args.iterations + 1):
            candidates = [best_policy] + [best_policy.mutate(rng, sigma) for _ in range(args.candidates_per_iter - 1)]
            scored: list[tuple[CandidateMetrics, WeightedHeuristicPolicy]] = []
            for idx, candidate in enumerate(candidates):
                metrics = _evaluate_candidate(
                    candidate,
                    deck=deck,
                    seed=args.seed + iteration * 100 + idx * 17,
                    games=args.games,
                    scenarios=scenarios,
                    alignment_depth=args.alignment_depth,
                    alignment_samples=args.alignment_samples,
                )
                scored.append((metrics, candidate))
            scored.sort(key=lambda item: item[0].objective, reverse=True)
            top_metrics, top_policy = scored[0]

            accepted = False
            if top_metrics.objective >= best_metrics.objective:
                best_policy = top_policy
                best_metrics = top_metrics
                sigma = max(0.05, sigma * 0.97)
                accepted = True
            else:
                sigma = min(1.5, sigma * 1.05)

            writer.writerow(
                {
                    "iteration": iteration,
                    "accepted": int(accepted),
                    "sigma": f"{sigma:.4f}",
                    "objective": f"{best_metrics.objective:.4f}",
                    "baseline_avg": f"{best_metrics.baseline_avg:.4f}",
                    "mixed_avg": f"{best_metrics.mixed_avg:.4f}",
                    "mixed_floor": f"{best_metrics.mixed_floor:.4f}",
                    "solver_alignment": f"{best_metrics.solver_alignment:.4f}",
                }
            )

    best_policy.save(best_policy_path)
    print(
        "Best policy metrics:",
        {
            "objective": round(best_metrics.objective, 4),
            "baseline_avg": round(best_metrics.baseline_avg, 4),
            "mixed_avg": round(best_metrics.mixed_avg, 4),
            "mixed_floor": round(best_metrics.mixed_floor, 4),
            "solver_alignment": round(best_metrics.solver_alignment, 4),
        },
    )
    print(f"Saved tuned policy to {best_policy_path}")
    print(f"Saved history to {history_path}")


if __name__ == "__main__":
    main()

