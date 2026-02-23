from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.eval.elo import update_elo
from mtg_ai.eval.tournament import Entrant, play_match
from mtg_ai.training.rl_policy import WeightedHeuristicPolicy


def build_deck() -> list[str]:
    return ["Mountain"] * 12 + ["Lightning Bolt"] * 9 + ["Raging Goblin"] * 9


def checkpoint_episode(path: Path) -> int:
    stem = path.stem
    if "ep" not in stem:
        return 0
    suffix = stem.split("ep")[-1]
    try:
        return int(suffix)
    except ValueError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze checkpoint-to-checkpoint convergence.")
    parser.add_argument("--checkpoint-dir", default="artifacts/training/self_play_multi")
    parser.add_argument("--games", type=int, default=80)
    parser.add_argument("--life", type=int, default=10)
    parser.add_argument("--opening-hand", type=int, default=7)
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--output-csv", default="artifacts/eval/convergence.csv")
    args = parser.parse_args()

    checkpoint_paths = sorted(Path(args.checkpoint_dir).glob("checkpoint_ep*.json"), key=checkpoint_episode)
    if len(checkpoint_paths) < 2:
        raise ValueError("Need at least two checkpoints for convergence analysis.")

    deck = build_deck()
    ratings: dict[str, float] = {}
    for checkpoint in checkpoint_paths:
        ratings[checkpoint.stem] = 1200.0

    rows: list[dict[str, object]] = []
    for idx in range(len(checkpoint_paths) - 1):
        a_path = checkpoint_paths[idx]
        b_path = checkpoint_paths[idx + 1]
        a_name = a_path.stem
        b_name = b_path.stem
        a_policy = WeightedHeuristicPolicy.load(a_path)
        b_policy = WeightedHeuristicPolicy.load(b_path)
        summary = play_match(
            Entrant(a_name, a_policy, deck),
            Entrant(b_name, b_policy, deck),
            games=args.games,
            config=GameConfig(starting_life=args.life, opening_hand_size=args.opening_hand, random_seed=args.seed + idx),
            seed=args.seed + idx * 103,
        )
        score_a = summary.win_rate_a
        old_a = ratings[a_name]
        old_b = ratings[b_name]
        new_a, new_b = update_elo(old_a, old_b, score_a)
        ratings[a_name] = new_a
        ratings[b_name] = new_b

        rows.append(
            {
                "from": a_name,
                "to": b_name,
                "games": args.games,
                "win_rate_from": f"{summary.win_rate_a:.4f}",
                "wins_from": summary.wins_a,
                "wins_to": summary.wins_b,
                "draws": summary.draws,
                "elo_from_before": f"{old_a:.2f}",
                "elo_to_before": f"{old_b:.2f}",
                "elo_from_after": f"{new_a:.2f}",
                "elo_to_after": f"{new_b:.2f}",
                "elo_delta_to": f"{new_b - old_b:.2f}",
            }
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote convergence analysis to {output_path}")


if __name__ == "__main__":
    main()

