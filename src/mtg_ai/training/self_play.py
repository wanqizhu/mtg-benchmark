from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
import random

from mtg_ai.bots.fixed_bots import BoltFaceBot, GoblinAggroBot, MixedBoltFaceBot
from mtg_ai.bots.random_bot import RandomBot
from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.eval.tournament import Entrant, play_match
from mtg_ai.training.rl_policy import WeightedHeuristicPolicy
from mtg_ai.training.value_iteration_baseline import evaluate_solver_alignment


def build_deck(mountains: int, bolts: int, goblins: int) -> list[str]:
    return ["Mountain"] * mountains + ["Lightning Bolt"] * bolts + ["Raging Goblin"] * goblins


def evaluate_policy(
    policy: WeightedHeuristicPolicy,
    eval_games: int,
    config: GameConfig,
    seed: int,
    deck: list[str],
) -> dict[str, float]:
    trained = Entrant("trained", policy, deck)
    baselines = [
        Entrant("random", RandomBot(), deck),
        Entrant("bolt", BoltFaceBot(), deck),
        Entrant("goblin", GoblinAggroBot(), deck),
        Entrant("mixed", MixedBoltFaceBot(), deck),
    ]
    results: dict[str, float] = {}
    for idx, baseline in enumerate(baselines):
        summary = play_match(trained, baseline, games=eval_games, config=config, seed=seed + idx * 1000)
        results[f"winrate_vs_{baseline.name}"] = summary.win_rate_a
    return results


def _aggregate_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {
            "winrate_vs_random": 0.0,
            "winrate_vs_bolt": 0.0,
            "winrate_vs_goblin": 0.0,
            "winrate_vs_mixed": 0.0,
        }
    keys = list(metric_rows[0].keys())
    aggregated: dict[str, float] = {}
    for key in keys:
        aggregated[key] = sum(row[key] for row in metric_rows) / len(metric_rows)
    return aggregated


def _historical_win_rate(
    candidate: WeightedHeuristicPolicy,
    historical_policies: list[WeightedHeuristicPolicy],
    deck: list[str],
    eval_games: int,
    seed: int,
    life: int,
    opening_hand: int,
) -> float:
    if not historical_policies:
        return 0.0
    rates: list[float] = []
    for idx, historical in enumerate(historical_policies):
        summary = play_match(
            Entrant("candidate", candidate, deck),
            Entrant("historical", historical, deck),
            games=eval_games,
            config=GameConfig(starting_life=life, opening_hand_size=opening_hand, random_seed=seed + idx),
            seed=seed + idx * 97,
        )
        rates.append(summary.win_rate_a)
    return sum(rates) / len(rates)


def _fitness(win_rates: dict[str, float]) -> float:
    return (
        0.15 * win_rates["winrate_vs_random"]
        + 0.25 * win_rates["winrate_vs_bolt"]
        + 0.25 * win_rates["winrate_vs_goblin"]
        + 0.35 * win_rates["winrate_vs_mixed"]
    )


def train_self_play(
    episodes: int,
    alpha: float,
    epsilon: float,
    seed: int,
    output_dir: Path,
    eval_interval: int,
    eval_games: int,
    starting_life: int,
    opening_hand_size: int,
    solver_weight: float = 0.0,
    solver_seed: int = 101,
    solver_depth: int = 6,
    solver_sample_count: int = 6,
    history_weight: float = 0.0,
    history_games: int = 12,
    train_life_values: list[int] | None = None,
    train_hand_values: list[int] | None = None,
) -> WeightedHeuristicPolicy:
    _ = alpha
    policy = WeightedHeuristicPolicy(name="weighted_heuristic")
    train_deck = build_deck(12, 9, 9)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "training_metrics.csv"
    rng = random.Random(seed)
    sigma = max(0.2, epsilon * 8.0)
    life_values = train_life_values or [starting_life]
    hand_values = train_hand_values or [opening_hand_size]

    best_metric_rows: list[dict[str, float]] = []
    eval_seed_cursor = seed
    for life in life_values:
        for hand in hand_values:
            best_metric_rows.append(
                evaluate_policy(
                    policy=policy,
                    eval_games=max(20, eval_games),
                    config=GameConfig(starting_life=life, opening_hand_size=hand, random_seed=eval_seed_cursor),
                    seed=eval_seed_cursor,
                    deck=train_deck,
                )
            )
            eval_seed_cursor += 100
    best_metrics = _aggregate_metrics(best_metric_rows)
    best_fitness = _fitness(best_metrics)
    best_solver_alignment = 0.0
    hall_of_fame: list[WeightedHeuristicPolicy] = []
    best_history_win_rate = 0.0
    if solver_weight > 0:
        alignment = evaluate_solver_alignment(
            trained_policy=policy,
            seed=solver_seed,
            life=max(5, min(10, starting_life)),
            opening_hand=max(3, min(4, opening_hand_size)),
            depth=solver_depth,
            sample_count=solver_sample_count,
        )
        best_solver_alignment = alignment.trained_agreement
        best_fitness += solver_weight * best_solver_alignment
    if history_weight > 0:
        best_history_win_rate = _historical_win_rate(
            candidate=policy,
            historical_policies=hall_of_fame,
            deck=train_deck,
            eval_games=history_games,
            seed=seed,
            life=starting_life,
            opening_hand=opening_hand_size,
        )
        best_fitness += history_weight * best_history_win_rate

    with metrics_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "episode_generation",
            "fitness",
            "winrate_vs_random",
            "winrate_vs_bolt",
            "winrate_vs_goblin",
            "winrate_vs_mixed",
            "solver_alignment",
            "history_win_rate",
            "sigma",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for episode in range(1, episodes + 1):
            candidates = [policy] + [policy.mutate(rng, sigma) for _ in range(5)]
            scored: list[tuple[float, WeightedHeuristicPolicy, dict[str, float]]] = []
            for idx, candidate in enumerate(candidates):
                metric_rows: list[dict[str, float]] = []
                scenario_seed_cursor = seed + episode * 1000 + idx * 100
                for life in life_values:
                    for hand in hand_values:
                        metric_rows.append(
                            evaluate_policy(
                                policy=candidate,
                                eval_games=eval_games,
                                config=GameConfig(
                                    starting_life=life,
                                    opening_hand_size=hand,
                                    random_seed=scenario_seed_cursor,
                                ),
                                seed=scenario_seed_cursor,
                                deck=train_deck,
                            )
                        )
                        scenario_seed_cursor += 17
                metrics = _aggregate_metrics(metric_rows)
                scored.append((_fitness(metrics), candidate, metrics))
            scored.sort(key=lambda item: item[0], reverse=True)
            candidate_fitness, candidate_policy, candidate_metrics = scored[0]
            candidate_solver_alignment = 0.0
            if solver_weight > 0:
                alignment = evaluate_solver_alignment(
                    trained_policy=candidate_policy,
                    seed=solver_seed + episode,
                    life=max(5, min(10, starting_life)),
                    opening_hand=max(3, min(4, opening_hand_size)),
                    depth=solver_depth,
                    sample_count=solver_sample_count,
                )
                candidate_solver_alignment = alignment.trained_agreement
                candidate_fitness += solver_weight * candidate_solver_alignment
            candidate_history_win_rate = 0.0
            if history_weight > 0:
                candidate_history_win_rate = _historical_win_rate(
                    candidate=candidate_policy,
                    historical_policies=hall_of_fame[-5:],
                    deck=train_deck,
                    eval_games=history_games,
                    seed=seed + episode * 37,
                    life=starting_life,
                    opening_hand=opening_hand_size,
                )
                candidate_fitness += history_weight * candidate_history_win_rate
            if candidate_fitness >= best_fitness:
                policy = candidate_policy
                best_fitness = candidate_fitness
                best_metrics = candidate_metrics
                best_solver_alignment = candidate_solver_alignment
                best_history_win_rate = candidate_history_win_rate
                sigma = max(0.05, sigma * 0.98)
            else:
                sigma = min(2.5, sigma * 1.03)

            row = {
                "episode_generation": episode,
                "fitness": f"{best_fitness:.4f}",
                "winrate_vs_random": f"{best_metrics['winrate_vs_random']:.4f}",
                "winrate_vs_bolt": f"{best_metrics['winrate_vs_bolt']:.4f}",
                "winrate_vs_goblin": f"{best_metrics['winrate_vs_goblin']:.4f}",
                "winrate_vs_mixed": f"{best_metrics['winrate_vs_mixed']:.4f}",
                "solver_alignment": f"{best_solver_alignment:.4f}",
                "history_win_rate": f"{best_history_win_rate:.4f}",
                "sigma": f"{sigma:.4f}",
            }
            writer.writerow(row)
            if episode % eval_interval == 0 or episode == episodes:
                checkpoint_path = output_dir / f"checkpoint_ep{episode}.json"
                policy.save(checkpoint_path)
                hall_of_fame.append(copy.deepcopy(policy))

    policy.save(output_dir / "policy_final.json")
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train weighted MTG bot with evolutionary self-play.")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=24)
    parser.add_argument("--life", type=int, default=20)
    parser.add_argument("--opening-hand", type=int, default=7)
    parser.add_argument("--life-values")
    parser.add_argument("--hand-values")
    parser.add_argument("--solver-weight", type=float, default=0.0)
    parser.add_argument("--solver-seed", type=int, default=101)
    parser.add_argument("--solver-depth", type=int, default=6)
    parser.add_argument("--solver-sample-count", type=int, default=6)
    parser.add_argument("--history-weight", type=float, default=0.0)
    parser.add_argument("--history-games", type=int, default=12)
    parser.add_argument("--output-dir", default="artifacts/training/self_play")
    args = parser.parse_args()

    life_values = [int(token.strip()) for token in args.life_values.split(",")] if args.life_values else None
    hand_values = [int(token.strip()) for token in args.hand_values.split(",")] if args.hand_values else None
    train_self_play(
        episodes=args.episodes,
        alpha=args.alpha,
        epsilon=args.epsilon,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        starting_life=args.life,
        opening_hand_size=args.opening_hand,
        solver_weight=args.solver_weight,
        solver_seed=args.solver_seed,
        solver_depth=args.solver_depth,
        solver_sample_count=args.solver_sample_count,
        history_weight=args.history_weight,
        history_games=args.history_games,
        train_life_values=life_values,
        train_hand_values=hand_values,
    )
    print(f"Training complete: {args.output_dir}")


if __name__ == "__main__":
    main()

