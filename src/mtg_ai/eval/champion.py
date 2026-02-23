from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from mtg_ai.bots.fixed_bots import BoltFaceBot, GoblinAggroBot, MixedBoltFaceBot
from mtg_ai.bots.random_bot import RandomBot
from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.eval.tournament import Entrant, play_match
from mtg_ai.training.rl_policy import WeightedHeuristicPolicy
from mtg_ai.training.value_iteration_baseline import evaluate_solver_alignment


@dataclass(frozen=True)
class Candidate:
    name: str
    path: Path
    policy: WeightedHeuristicPolicy


def _evaluate(candidate: Candidate, deck: list[str], scenarios: list[tuple[int, int]], games: int, seed: int) -> dict[str, float]:
    baseline_values: list[float] = []
    mixed_values: list[float] = []
    for idx, (life, hand) in enumerate(scenarios):
        cfg = GameConfig(starting_life=life, opening_hand_size=hand, random_seed=seed + idx)
        results = [
            play_match(Entrant("cand", candidate.policy, deck), Entrant("random", RandomBot(), deck), games=games, config=cfg, seed=seed + idx * 100 + 3),
            play_match(Entrant("cand", candidate.policy, deck), Entrant("bolt", BoltFaceBot(), deck), games=games, config=cfg, seed=seed + idx * 100 + 11),
            play_match(Entrant("cand", candidate.policy, deck), Entrant("goblin", GoblinAggroBot(), deck), games=games, config=cfg, seed=seed + idx * 100 + 19),
            play_match(Entrant("cand", candidate.policy, deck), Entrant("mixed", MixedBoltFaceBot(), deck), games=games, config=cfg, seed=seed + idx * 100 + 27),
        ]
        baseline_values.extend(summary.win_rate_a for summary in results)
        mixed_values.append(results[-1].win_rate_a)
    baseline_avg = sum(baseline_values) / len(baseline_values)
    mixed_avg = sum(mixed_values) / len(mixed_values)
    mixed_floor = min(mixed_values)
    alignment = evaluate_solver_alignment(
        trained_policy=candidate.policy,
        seed=seed + 777,
        life=10,
        opening_hand=4,
        depth=7,
        sample_count=12,
    ).trained_agreement
    robust_objective = 0.40 * baseline_avg + 0.40 * mixed_floor + 0.20 * alignment
    return {
        "baseline_avg": baseline_avg,
        "mixed_avg": mixed_avg,
        "mixed_floor": mixed_floor,
        "solver_alignment": alignment,
        "robust_objective": robust_objective,
    }


def _load_candidates(paths: list[Path]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for path in paths:
        try:
            policy = WeightedHeuristicPolicy.load(path)
        except (FileNotFoundError, KeyError):
            continue
        candidates.append(Candidate(name=f"{path.parent.name}/{path.name}", path=path, policy=policy))
    return candidates


def _paths_from_globs(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(Path(".").glob(pattern))
    return sorted(path.resolve() for path in paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick robust champion policy from available candidates.")
    parser.add_argument(
        "--policy-globs",
        default="artifacts/training/self_play*/policy*.json,artifacts/training/solver_tune/*.json,artifacts/training/sweep_seed_*/policy_final.json,artifacts/training/sweep_runs/*/policy_final.json",
    )
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--output-csv", default="artifacts/eval/champion_selection.csv")
    parser.add_argument("--output-json", default="artifacts/eval/champion.json")
    args = parser.parse_args()

    patterns = [token.strip() for token in args.policy_globs.split(",") if token.strip()]
    candidates = _load_candidates(_paths_from_globs(patterns))
    if not candidates:
        raise ValueError("No valid weighted policy candidates found.")

    deck = ["Mountain"] * 12 + ["Lightning Bolt"] * 9 + ["Raging Goblin"] * 9
    scenarios = [(5, 5), (5, 7), (10, 5), (10, 7), (20, 5), (20, 7)]
    rows: list[dict[str, object]] = []
    for idx, candidate in enumerate(candidates):
        metrics = _evaluate(candidate, deck=deck, scenarios=scenarios, games=args.games, seed=args.seed + idx * 101)
        row = {
            "name": candidate.name,
            "path": str(candidate.path),
            "baseline_avg": f"{metrics['baseline_avg']:.4f}",
            "mixed_avg": f"{metrics['mixed_avg']:.4f}",
            "mixed_floor": f"{metrics['mixed_floor']:.4f}",
            "solver_alignment": f"{metrics['solver_alignment']:.4f}",
            "robust_objective": f"{metrics['robust_objective']:.4f}",
        }
        rows.append(row)
        print(row)
    rows.sort(key=lambda row: float(row["robust_objective"]), reverse=True)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote champion ranking to {output_csv}")

    winner = rows[0]
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(winner, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote champion selection to {output_json}")


if __name__ == "__main__":
    main()

