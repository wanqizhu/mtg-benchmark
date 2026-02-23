from __future__ import annotations

import argparse
from pathlib import Path

from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.eval.tournament import Entrant, round_robin
from mtg_ai.training.rl_policy import WeightedHeuristicPolicy


def build_deck(mountains: int, bolts: int, goblins: int) -> list[str]:
    return ["Mountain"] * mountains + ["Lightning Bolt"] * bolts + ["Raging Goblin"] * goblins


def load_checkpoints(path: Path) -> list[Entrant]:
    entrants: list[Entrant] = []
    deck = build_deck(12, 9, 9)
    for checkpoint in sorted(path.glob("checkpoint_ep*.json")):
        try:
            policy = WeightedHeuristicPolicy.load(checkpoint)
        except KeyError:
            continue
        policy.name = checkpoint.stem
        entrants.append(Entrant(checkpoint.stem, policy, deck))
    if not entrants:
        raise ValueError(f"No checkpoints found under {path}")
    return entrants


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate training checkpoints in a league.")
    parser.add_argument("--checkpoint-dir", default="artifacts/training/self_play")
    parser.add_argument("--games-per-pair", type=int, default=60)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--life", type=int, default=20)
    parser.add_argument("--opening-hand", type=int, default=7)
    args = parser.parse_args()

    entrants = load_checkpoints(Path(args.checkpoint_dir))
    summaries, ratings = round_robin(
        entrants=entrants,
        games_per_pair=args.games_per_pair,
        config=GameConfig(starting_life=args.life, opening_hand_size=args.opening_hand, random_seed=args.seed),
        seed=args.seed,
    )
    print("=== League Elo ===")
    for name, rating in sorted(ratings.items(), key=lambda item: item[1], reverse=True):
        print(f"{name}: {rating:.2f}")
    print("\n=== Matchups ===")
    for summary in summaries:
        print(
            f"{summary.entrant_a} vs {summary.entrant_b}: "
            f"{summary.wins_a}-{summary.wins_b}-{summary.draws} "
            f"wr={summary.win_rate_a:.3f}"
        )


if __name__ == "__main__":
    main()

