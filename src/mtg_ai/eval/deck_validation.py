from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.eval.tournament import Entrant, play_match
from mtg_ai.training.rl_policy import WeightedHeuristicPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate searched best decks against a starter deck.")
    parser.add_argument("--policy-path", default="artifacts/training/solver_tune/policy_best.json")
    parser.add_argument("--deck-search-csv", default="artifacts/deck_search/results_tuned.csv")
    parser.add_argument("--starter-mountains", type=int, default=10)
    parser.add_argument("--starter-bolts", type=int, default=10)
    parser.add_argument("--starter-goblins", type=int, default=10)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2200)
    parser.add_argument("--output-csv", default="artifacts/eval/deck_vs_starter_tuned.csv")
    args = parser.parse_args()

    policy = WeightedHeuristicPolicy.load(args.policy_path)
    starter = (
        ["Mountain"] * args.starter_mountains
        + ["Lightning Bolt"] * args.starter_bolts
        + ["Raging Goblin"] * args.starter_goblins
    )

    with Path(args.deck_search_csv).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Deck search CSV has no rows.")

    output_rows: list[dict[str, object]] = []
    for row in rows:
        life = int(row["life"])
        hand = int(row["opening_hand"])
        best = (
            ["Mountain"] * int(row["mountains"])
            + ["Lightning Bolt"] * int(row["bolts"])
            + ["Raging Goblin"] * int(row["goblins"])
        )
        summary = play_match(
            Entrant("best", policy, best),
            Entrant("starter", policy, starter),
            games=args.games,
            config=GameConfig(starting_life=life, opening_hand_size=hand, random_seed=args.seed + life * 10 + hand),
            seed=args.seed + life * 10 + hand,
        )
        output_rows.append(
            {
                "life": life,
                "opening_hand": hand,
                "best_deck": f"M{row['mountains']}/B{row['bolts']}/G{row['goblins']}",
                "best_vs_starter_win_rate": f"{summary.win_rate_a:.4f}",
                "wins": summary.wins_a,
                "losses": summary.wins_b,
                "draws": summary.draws,
            }
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote deck-vs-starter results to {output_path}")


if __name__ == "__main__":
    main()

