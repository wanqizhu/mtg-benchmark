from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random

from mtg_ai.bots.fixed_bots import BoltFaceBot, GoblinAggroBot, MixedBoltFaceBot
from mtg_ai.bots.random_bot import RandomBot
from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.eval.tournament import Entrant, play_match
from mtg_ai.training.rl_policy import WeightedHeuristicPolicy


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
) -> WeightedHeuristicPolicy:
    _ = alpha
    policy = WeightedHeuristicPolicy(name="weighted_heuristic")
    train_deck = build_deck(12, 9, 9)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "training_metrics.csv"
    rng = random.Random(seed)
    sigma = max(0.2, epsilon * 8.0)
    best_metrics = evaluate_policy(
        policy=policy,
        eval_games=max(20, eval_games),
        config=GameConfig(starting_life=starting_life, opening_hand_size=opening_hand_size, random_seed=seed),
        seed=seed,
        deck=train_deck,
    )
    best_fitness = _fitness(best_metrics)

    with metrics_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "episode_generation",
            "fitness",
            "winrate_vs_random",
            "winrate_vs_bolt",
            "winrate_vs_goblin",
            "winrate_vs_mixed",
            "sigma",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for episode in range(1, episodes + 1):
            candidates = [policy] + [policy.mutate(rng, sigma) for _ in range(5)]
            scored: list[tuple[float, WeightedHeuristicPolicy, dict[str, float]]] = []
            for idx, candidate in enumerate(candidates):
                metrics = evaluate_policy(
                    policy=candidate,
                    eval_games=eval_games,
                    config=GameConfig(
                        starting_life=starting_life,
                        opening_hand_size=opening_hand_size,
                        random_seed=seed + episode * 17 + idx,
                    ),
                    seed=seed + episode * 19 + idx,
                    deck=train_deck,
                )
                scored.append((_fitness(metrics), candidate, metrics))
            scored.sort(key=lambda item: item[0], reverse=True)
            candidate_fitness, candidate_policy, candidate_metrics = scored[0]
            if candidate_fitness >= best_fitness:
                policy = candidate_policy
                best_fitness = candidate_fitness
                best_metrics = candidate_metrics
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
                "sigma": f"{sigma:.4f}",
            }
            writer.writerow(row)
            if episode % eval_interval == 0 or episode == episodes:
                checkpoint_path = output_dir / f"checkpoint_ep{episode}.json"
                policy.save(checkpoint_path)

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
    parser.add_argument("--output-dir", default="artifacts/training/self_play")
    args = parser.parse_args()
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
    )
    print(f"Training complete: {args.output_dir}")


if __name__ == "__main__":
    main()

