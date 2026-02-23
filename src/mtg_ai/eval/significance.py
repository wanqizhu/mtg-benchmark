from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.eval.tournament import Entrant, play_match
from mtg_ai.training.rl_policy import WeightedHeuristicPolicy


@dataclass(frozen=True)
class Candidate:
    name: str
    path: Path
    policy: WeightedHeuristicPolicy


def _load_candidates(paths: list[Path]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for path in paths:
        try:
            policy = WeightedHeuristicPolicy.load(path)
        except (KeyError, FileNotFoundError):
            continue
        candidates.append(Candidate(name=f"{path.parent.name}/{path.name}", path=path, policy=policy))
    return candidates


def _from_csv(csv_path: Path, top_k: int) -> list[Path]:
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Selection CSV is empty.")
    return [Path(row["path"]) for row in rows[:top_k]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute statistical significance matrix across top policy candidates.")
    parser.add_argument("--selection-csv", default="artifacts/eval/champion_selection.csv")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--games", type=int, default=240)
    parser.add_argument("--seed", type=int, default=6201)
    parser.add_argument("--life", type=int, default=10)
    parser.add_argument("--opening-hand", type=int, default=7)
    parser.add_argument("--output-csv", default="artifacts/eval/significance.csv")
    args = parser.parse_args()

    candidate_paths = _from_csv(Path(args.selection_csv), args.top_k)
    candidates = _load_candidates(candidate_paths)
    if len(candidates) < 2:
        raise ValueError("Need at least two valid candidates for significance matrix.")

    deck = ["Mountain"] * 12 + ["Lightning Bolt"] * 9 + ["Raging Goblin"] * 9
    rows: list[dict[str, object]] = []
    pair_seed = args.seed
    for idx in range(len(candidates)):
        for jdx in range(idx + 1, len(candidates)):
            a = candidates[idx]
            b = candidates[jdx]
            summary = play_match(
                Entrant(a.name, a.policy, deck),
                Entrant(b.name, b.policy, deck),
                games=args.games,
                config=GameConfig(starting_life=args.life, opening_hand_size=args.opening_hand, random_seed=pair_seed),
                seed=pair_seed,
            )
            # Treat win_rate_a CI around 0.5 as significance indicator.
            significant = int(summary.ci95_low > 0.5 or summary.ci95_high < 0.5)
            favored = a.name if summary.win_rate_a > 0.5 else b.name if summary.win_rate_a < 0.5 else "none"
            rows.append(
                {
                    "a": a.name,
                    "b": b.name,
                    "games": summary.games,
                    "wins_a": summary.wins_a,
                    "wins_b": summary.wins_b,
                    "draws": summary.draws,
                    "win_rate_a": f"{summary.win_rate_a:.4f}",
                    "ci95_low": f"{summary.ci95_low:.4f}",
                    "ci95_high": f"{summary.ci95_high:.4f}",
                    "significant_vs_0_5": significant,
                    "favored": favored,
                }
            )
            pair_seed += args.games + 19

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote significance matrix to {output_path}")


if __name__ == "__main__":
    main()

