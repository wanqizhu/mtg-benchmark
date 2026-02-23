from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mtg_ai.bots.fixed_bots import BoltFaceBot, GoblinAggroBot, MixedBoltFaceBot
from mtg_ai.bots.random_bot import RandomBot
from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.eval.tournament import Entrant, play_match
from mtg_ai.training.rl_policy import WeightedHeuristicPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained policy generalization across life and hand sizes.")
    parser.add_argument("--policy-path", default="artifacts/training/self_play_multi/policy_final.json")
    parser.add_argument("--life-values", default="5,10,20")
    parser.add_argument("--hand-values", default="5,7")
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1300)
    parser.add_argument("--output-csv", default="artifacts/eval/generalization_matrix.csv")
    args = parser.parse_args()

    life_values = [int(token.strip()) for token in args.life_values.split(",") if token.strip()]
    hand_values = [int(token.strip()) for token in args.hand_values.split(",") if token.strip()]

    policy = WeightedHeuristicPolicy.load(args.policy_path)
    deck = ["Mountain"] * 12 + ["Lightning Bolt"] * 9 + ["Raging Goblin"] * 9
    rows: list[dict[str, object]] = []
    for life in life_values:
        for hand in hand_values:
            cfg = GameConfig(starting_life=life, opening_hand_size=hand, random_seed=args.seed + life * 10 + hand)
            for name, bot in [
                ("random", RandomBot()),
                ("bolt", BoltFaceBot()),
                ("goblin", GoblinAggroBot()),
                ("mixed", MixedBoltFaceBot()),
            ]:
                summary = play_match(
                    Entrant("trained", policy, deck),
                    Entrant(name, bot, deck),
                    games=args.games,
                    config=cfg,
                    seed=args.seed + life * 100 + hand * 10 + len(name),
                )
                rows.append(
                    {
                        "life": life,
                        "opening_hand": hand,
                        "opponent": name,
                        "games": args.games,
                        "trained_win_rate": f"{summary.win_rate_a:.4f}",
                        "wins": summary.wins_a,
                        "losses": summary.wins_b,
                        "draws": summary.draws,
                    }
                )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote generalization matrix to {output_path}")


if __name__ == "__main__":
    main()

