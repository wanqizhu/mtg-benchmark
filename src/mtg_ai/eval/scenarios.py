from __future__ import annotations

import argparse
from dataclasses import asdict

from mtg_ai.bots.fixed_bots import BoltFaceBot, GoblinAggroBot, MixedBoltFaceBot
from mtg_ai.bots.random_bot import RandomBot
from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.eval.tournament import Entrant, round_robin


def build_deck(mountains: int, bolts: int, goblins: int) -> list[str]:
    return ["Mountain"] * mountains + ["Lightning Bolt"] * bolts + ["Raging Goblin"] * goblins


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MTG AI benchmark scenarios.")
    parser.add_argument("--games-per-pair", type=int, default=100)
    parser.add_argument("--life", type=int, default=20)
    parser.add_argument("--opening-hand", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    starter = build_deck(10, 10, 10)
    bolt_only = build_deck(12, 18, 0)
    goblin_only = build_deck(12, 0, 18)
    mixed = build_deck(12, 9, 9)
    entrants = [
        Entrant("random_starter", RandomBot(), starter),
        Entrant("bolt_face", BoltFaceBot(), bolt_only),
        Entrant("goblin_aggro", GoblinAggroBot(), goblin_only),
        Entrant("mixed_bolt_face", MixedBoltFaceBot(), mixed),
    ]

    summaries, ratings = round_robin(
        entrants=entrants,
        games_per_pair=args.games_per_pair,
        config=GameConfig(starting_life=args.life, opening_hand_size=args.opening_hand, random_seed=args.seed),
        seed=args.seed,
    )

    print("=== Match summaries ===")
    for summary in summaries:
        print(asdict(summary))
    print("\n=== Elo ratings ===")
    for name, rating in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {rating:.2f}")


if __name__ == "__main__":
    main()

