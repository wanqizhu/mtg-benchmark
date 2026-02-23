from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mtg_ai.bots.fixed_bots import BoltFaceBot, GoblinAggroBot, MixedBoltFaceBot
from mtg_ai.bots.random_bot import RandomBot
from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.eval.tournament import Entrant, play_match, round_robin
from mtg_ai.training.rl_policy import WeightedHeuristicPolicy
from mtg_ai.training.value_iteration_baseline import evaluate_solver_alignment


def _deck(mountains: int, bolts: int, goblins: int) -> list[str]:
    return ["Mountain"] * mountains + ["Lightning Bolt"] * bolts + ["Raging Goblin"] * goblins


def _record(rows: list[dict[str, object]], name: str, metric: float, threshold: float, details: str) -> None:
    rows.append(
        {
            "milestone": name,
            "metric": f"{metric:.4f}",
            "threshold": f"{threshold:.4f}",
            "passed": int(metric >= threshold),
            "details": details,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run milestone validation suite for constrained MTG AI project.")
    parser.add_argument("--policy-path", default="artifacts/training/solver_tune/policy_best.json")
    parser.add_argument("--champion-json")
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--seed", type=int, default=3301)
    parser.add_argument("--output-csv", default="artifacts/eval/milestones.csv")
    args = parser.parse_args()

    policy_path = args.policy_path
    if args.champion_json:
        payload = json.loads(Path(args.champion_json).read_text(encoding="utf-8"))
        policy_path = payload["path"]
    policy = WeightedHeuristicPolicy.load(policy_path)
    rows: list[dict[str, object]] = []

    starter_deck = _deck(10, 10, 10)
    constrained_deck = _deck(12, 9, 9)
    bolt_only = _deck(12, 18, 0)
    goblin_only = _deck(12, 0, 18)

    summary_random = play_match(
        Entrant("random_a", RandomBot(), starter_deck),
        Entrant("random_b", RandomBot(), starter_deck),
        games=30,
        config=GameConfig(starting_life=10, opening_hand_size=7, random_seed=args.seed),
        seed=args.seed,
    )
    _record(rows, "random_bot_runs", 1.0, 1.0, f"wins={summary_random.wins_a}/{summary_random.wins_b} draws={summary_random.draws}")

    summary_low_life = play_match(
        Entrant("mixed", MixedBoltFaceBot(), constrained_deck),
        Entrant("random", RandomBot(), constrained_deck),
        games=60,
        config=GameConfig(starting_life=5, opening_hand_size=7, random_seed=args.seed + 11),
        seed=args.seed + 11,
    )
    _record(rows, "lower_life_totals_behavior", summary_low_life.win_rate_a, 0.55, "mixed_vs_random life=5")

    summary_bolt = play_match(
        Entrant("bolt_face", BoltFaceBot(), bolt_only),
        Entrant("random", RandomBot(), starter_deck),
        games=args.games,
        config=GameConfig(starting_life=10, opening_hand_size=7, random_seed=args.seed + 101),
        seed=args.seed + 101,
    )
    _record(rows, "fixed_bolt_bot", summary_bolt.win_rate_a, 0.55, "bolt-only deck vs random")

    summary_goblin = play_match(
        Entrant("goblin_aggro", GoblinAggroBot(), goblin_only),
        Entrant("random", RandomBot(), starter_deck),
        games=args.games,
        config=GameConfig(starting_life=10, opening_hand_size=7, random_seed=args.seed + 131),
        seed=args.seed + 131,
    )
    _record(rows, "fixed_goblin_bot", summary_goblin.win_rate_a, 0.52, "goblin-only deck vs random")

    summary_mixed = play_match(
        Entrant("mixed_bolt_face", MixedBoltFaceBot(), constrained_deck),
        Entrant("random", RandomBot(), constrained_deck),
        games=args.games,
        config=GameConfig(starting_life=10, opening_hand_size=7, random_seed=args.seed + 151),
        seed=args.seed + 151,
    )
    _record(rows, "fixed_mixed_bot", summary_mixed.win_rate_a, 0.55, "mixed baseline vs random")

    baseline_bots = [
        ("random", RandomBot()),
        ("bolt", BoltFaceBot()),
        ("goblin", GoblinAggroBot()),
        ("mixed", MixedBoltFaceBot()),
    ]
    trained_win_rates: list[float] = []
    for idx, (name, bot) in enumerate(baseline_bots):
        summary = play_match(
            Entrant("trained", policy, constrained_deck),
            Entrant(name, bot, constrained_deck),
            games=args.games,
            config=GameConfig(starting_life=10, opening_hand_size=7, random_seed=args.seed + 200 + idx),
            seed=args.seed + 200 + idx,
        )
        trained_win_rates.append(summary.win_rate_a)
        _record(rows, f"trained_beats_{name}", summary.win_rate_a, 0.5, "life=10 hand=7")

    generalization_rates: list[float] = []
    for life in (5, 10, 20):
        for hand in (5, 7):
            summary = play_match(
                Entrant("trained", policy, constrained_deck),
                Entrant("mixed", MixedBoltFaceBot(), constrained_deck),
                games=80,
                config=GameConfig(starting_life=life, opening_hand_size=hand, random_seed=args.seed + life * 10 + hand),
                seed=args.seed + life * 10 + hand,
            )
            generalization_rates.append(summary.win_rate_a)
    _record(
        rows,
        "trained_generalizes_life_hand",
        min(generalization_rates),
        0.5,
        "min winrate vs mixed over life∈{5,10,20}, hand∈{5,7}",
    )

    deck_rows_path = Path("artifacts/eval/deck_vs_starter_tuned.csv")
    deck_metric = 0.0
    if deck_rows_path.exists():
        with deck_rows_path.open("r", encoding="utf-8") as f:
            deck_rows = list(csv.DictReader(f))
        if deck_rows:
            deck_metric = min(float(row["best_vs_starter_win_rate"]) for row in deck_rows)
    _record(rows, "deck_better_than_starter", deck_metric, 0.5, "min best-deck winrate vs starter")

    checkpoint_paths = sorted(Path("artifacts/training/self_play_multi").glob("checkpoint_ep*.json"))
    checkpoint_entries: list[Entrant] = []
    for path in checkpoint_paths:
        checkpoint_entries.append(Entrant(path.stem, WeightedHeuristicPolicy.load(path), constrained_deck))
    elo_gain_metric = 0.0
    if len(checkpoint_entries) >= 2:
        _, ratings = round_robin(
            entrants=checkpoint_entries,
            games_per_pair=16,
            config=GameConfig(starting_life=10, opening_hand_size=7, random_seed=args.seed + 500),
            seed=args.seed + 500,
        )
        first = checkpoint_entries[0].name
        last = checkpoint_entries[-1].name
        elo_gain_metric = ratings[last] - ratings[first]
    _record(rows, "elo_improves_over_training", elo_gain_metric, 0.0, "last checkpoint Elo minus first checkpoint Elo")

    alignment = evaluate_solver_alignment(
        trained_policy=policy,
        seed=args.seed + 700,
        life=10,
        opening_hand=4,
        depth=7,
        sample_count=12,
    )
    _record(rows, "solver_alignment", alignment.trained_agreement, 0.8, "agreement on sampled minimax-optimal states")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote milestones to {output_path}")


if __name__ == "__main__":
    main()

