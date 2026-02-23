from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from mtg_ai.bots.fixed_bots import BoltFaceBot, GoblinAggroBot, MixedBoltFaceBot
from mtg_ai.bots.random_bot import RandomBot
from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.eval.tournament import Entrant, play_match
from mtg_ai.training.self_play import train_self_play
from mtg_ai.training.value_iteration_baseline import evaluate_solver_alignment


@dataclass(frozen=True)
class SweepConfig:
    epsilon: float
    solver_weight: float
    history_weight: float
    seed: int


def parse_float_list(spec: str) -> list[float]:
    return [float(token.strip()) for token in spec.split(",") if token.strip()]


def parse_int_list(spec: str) -> list[int]:
    return [int(token.strip()) for token in spec.split(",") if token.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep for MTG training objectives.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-games", type=int, default=6)
    parser.add_argument("--life-values", default="5,10,20")
    parser.add_argument("--hand-values", default="5,7")
    parser.add_argument("--epsilons", default="0.18,0.22")
    parser.add_argument("--solver-weights", default="0.0,0.1,0.2")
    parser.add_argument("--history-weights", default="0.0,0.1")
    parser.add_argument("--seeds", default="71,73")
    parser.add_argument("--benchmark-games", type=int, default=50)
    parser.add_argument("--output-dir", default="artifacts/training/sweep_runs")
    parser.add_argument("--output-csv", default="artifacts/training/sweep_results.csv")
    args = parser.parse_args()

    life_values = parse_int_list(args.life_values)
    hand_values = parse_int_list(args.hand_values)
    epsilons = parse_float_list(args.epsilons)
    solver_weights = parse_float_list(args.solver_weights)
    history_weights = parse_float_list(args.history_weights)
    seeds = parse_int_list(args.seeds)

    configurations: list[SweepConfig] = []
    for epsilon in epsilons:
        for solver_weight in solver_weights:
            for history_weight in history_weights:
                for seed in seeds:
                    configurations.append(
                        SweepConfig(
                            epsilon=epsilon,
                            solver_weight=solver_weight,
                            history_weight=history_weight,
                            seed=seed,
                        )
                    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    deck = ["Mountain"] * 12 + ["Lightning Bolt"] * 9 + ["Raging Goblin"] * 9
    rows: list[dict[str, object]] = []
    for idx, config in enumerate(configurations, start=1):
        run_dir = output_dir / f"run_{idx:03d}_seed_{config.seed}_sw_{config.solver_weight:.2f}_hw_{config.history_weight:.2f}_eps_{config.epsilon:.2f}"
        policy = train_self_play(
            episodes=args.episodes,
            alpha=0.08,
            epsilon=config.epsilon,
            seed=config.seed,
            output_dir=run_dir,
            eval_interval=max(2, args.episodes // 4),
            eval_games=args.eval_games,
            starting_life=10,
            opening_hand_size=7,
            solver_weight=config.solver_weight,
            solver_seed=211 + config.seed,
            solver_depth=7,
            solver_sample_count=6,
            history_weight=config.history_weight,
            history_games=8,
            train_life_values=life_values,
            train_hand_values=hand_values,
        )

        mixed_rates: list[float] = []
        baseline_avg = 0.0
        for life in life_values:
            for hand in hand_values:
                cfg = GameConfig(starting_life=life, opening_hand_size=hand, random_seed=3000 + life * 10 + hand + config.seed)
                summary_random = play_match(
                    Entrant("trained", policy, deck),
                    Entrant("random", RandomBot(), deck),
                    games=args.benchmark_games,
                    config=cfg,
                    seed=4000 + config.seed + life * 10 + hand,
                )
                summary_bolt = play_match(
                    Entrant("trained", policy, deck),
                    Entrant("bolt", BoltFaceBot(), deck),
                    games=args.benchmark_games,
                    config=cfg,
                    seed=5000 + config.seed + life * 10 + hand,
                )
                summary_goblin = play_match(
                    Entrant("trained", policy, deck),
                    Entrant("goblin", GoblinAggroBot(), deck),
                    games=args.benchmark_games,
                    config=cfg,
                    seed=6000 + config.seed + life * 10 + hand,
                )
                summary_mixed = play_match(
                    Entrant("trained", policy, deck),
                    Entrant("mixed", MixedBoltFaceBot(), deck),
                    games=args.benchmark_games,
                    config=cfg,
                    seed=7000 + config.seed + life * 10 + hand,
                )
                mixed_rates.append(summary_mixed.win_rate_a)
                baseline_avg += (
                    summary_random.win_rate_a
                    + summary_bolt.win_rate_a
                    + summary_goblin.win_rate_a
                    + summary_mixed.win_rate_a
                ) / 4.0

        baseline_avg /= max(1, len(life_values) * len(hand_values))
        mixed_floor = min(mixed_rates) if mixed_rates else 0.0
        alignment = evaluate_solver_alignment(
            trained_policy=policy,
            seed=11 + config.seed,
            life=10,
            opening_hand=4,
            depth=7,
            sample_count=12,
        )
        objective = 0.6 * baseline_avg + 0.25 * mixed_floor + 0.15 * alignment.trained_agreement
        row = {
            "run_id": idx,
            "seed": config.seed,
            "epsilon": config.epsilon,
            "solver_weight": config.solver_weight,
            "history_weight": config.history_weight,
            "baseline_avg_winrate": f"{baseline_avg:.4f}",
            "mixed_floor_winrate": f"{mixed_floor:.4f}",
            "solver_alignment": f"{alignment.trained_agreement:.4f}",
            "objective": f"{objective:.4f}",
            "run_dir": str(run_dir),
        }
        rows.append(row)
        print(row)

    rows.sort(key=lambda row: float(row["objective"]), reverse=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote sweep results to {output_csv}")


if __name__ == "__main__":
    main()

