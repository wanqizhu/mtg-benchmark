from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

from mtg_ai.bots.base import BotPolicy
from mtg_ai.bots.fixed_bots import BoltFaceBot, GoblinAggroBot, MixedBoltFaceBot
from mtg_ai.bots.random_bot import RandomBot
from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.deck.space import DeckComposition, enumerate_compositions
from mtg_ai.eval.tournament import Entrant, play_match
from mtg_ai.training.rl_policy import WeightedHeuristicPolicy


@dataclass
class DeckScore:
    composition: DeckComposition
    score: float


def starter_deck() -> list[str]:
    return ["Mountain"] * 10 + ["Lightning Bolt"] * 10 + ["Raging Goblin"] * 10


def resolve_policy(policy_name: str, trained_path: str | None) -> BotPolicy:
    if policy_name == "random":
        return RandomBot()
    if policy_name == "bolt":
        return BoltFaceBot()
    if policy_name == "goblin":
        return GoblinAggroBot()
    if policy_name == "mixed":
        return MixedBoltFaceBot()
    if policy_name == "trained":
        if trained_path is None:
            raise ValueError("trained policy requested without --trained-path")
        return WeightedHeuristicPolicy.load(trained_path)
    raise ValueError(f"Unknown policy {policy_name}")


def evaluate_deck(
    composition: DeckComposition,
    candidate_policy: BotPolicy,
    config: GameConfig,
    seed: int,
    games_per_opponent: int,
) -> float:
    candidate = Entrant("candidate", candidate_policy, composition.to_deck())
    opponents = [
        Entrant("starter_random", RandomBot(), starter_deck()),
        Entrant("starter_bolt", BoltFaceBot(), starter_deck()),
        Entrant("starter_goblin", GoblinAggroBot(), starter_deck()),
        Entrant("starter_mixed", MixedBoltFaceBot(), starter_deck()),
    ]
    score = 0.0
    for idx, opponent in enumerate(opponents):
        summary = play_match(candidate, opponent, games=games_per_opponent, config=config, seed=seed + idx * 1000)
        score += summary.win_rate_a
    return score / len(opponents)


def exhaustive_search(
    total_cards: int,
    candidate_policy: BotPolicy,
    config: GameConfig,
    seed: int,
    games_per_opponent: int,
) -> DeckScore:
    best = DeckScore(composition=DeckComposition(0, 0, total_cards), score=float("-inf"))
    for idx, composition in enumerate(enumerate_compositions(total_cards)):
        score = evaluate_deck(composition, candidate_policy, config, seed + idx * 17, games_per_opponent)
        if score > best.score:
            best = DeckScore(composition=composition, score=score)
    return best


def mutate(comp: DeckComposition, rng: random.Random) -> DeckComposition:
    values = [comp.mountains, comp.bolts, comp.goblins]
    src = rng.randrange(3)
    dst = rng.randrange(3)
    if src == dst or values[src] == 0:
        return comp
    values[src] -= 1
    values[dst] += 1
    return DeckComposition(*values)


def evolutionary_search(
    total_cards: int,
    candidate_policy: BotPolicy,
    config: GameConfig,
    seed: int,
    games_per_opponent: int,
    population_size: int = 24,
    generations: int = 20,
) -> DeckScore:
    rng = random.Random(seed)
    population = rng.sample(enumerate_compositions(total_cards), k=min(population_size, len(enumerate_compositions(total_cards))))
    scored: list[DeckScore] = []
    for generation in range(generations):
        scored = []
        for idx, comp in enumerate(population):
            score = evaluate_deck(comp, candidate_policy, config, seed + generation * 1000 + idx * 13, games_per_opponent)
            scored.append(DeckScore(composition=comp, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        elites = [item.composition for item in scored[: max(2, population_size // 4)]]
        next_population = list(elites)
        while len(next_population) < population_size:
            parent = rng.choice(elites)
            next_population.append(mutate(parent, rng))
        population = next_population
    return scored[0]


def parse_int_list(spec: str) -> list[int]:
    return [int(token.strip()) for token in spec.split(",") if token.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search for strong decks in constrained MTG.")
    parser.add_argument("--policy", default="mixed", choices=["random", "bolt", "goblin", "mixed", "trained"])
    parser.add_argument("--trained-path")
    parser.add_argument("--deck-size", type=int, default=30)
    parser.add_argument("--mode", default="exhaustive", choices=["exhaustive", "evolutionary"])
    parser.add_argument("--games-per-opponent", type=int, default=40)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--life-values", default="20,10,5")
    parser.add_argument("--hand-sizes", default="7,5")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--output-csv", default="artifacts/deck_search/results.csv")
    args = parser.parse_args()

    policy = resolve_policy(args.policy, args.trained_path)
    life_values = parse_int_list(args.life_values)
    hand_sizes = parse_int_list(args.hand_sizes)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["life", "opening_hand", "mode", "deck_size", "mountains", "bolts", "goblins", "score"],
        )
        writer.writeheader()
        for life in life_values:
            for opening_hand in hand_sizes:
                config = GameConfig(starting_life=life, opening_hand_size=opening_hand, random_seed=args.seed)
                if args.mode == "exhaustive":
                    best = exhaustive_search(args.deck_size, policy, config, args.seed + life * 10 + opening_hand, args.games_per_opponent)
                else:
                    best = evolutionary_search(
                        args.deck_size,
                        policy,
                        config,
                        args.seed + life * 10 + opening_hand,
                        args.games_per_opponent,
                        population_size=args.population_size,
                        generations=args.generations,
                    )
                writer.writerow(
                    {
                        "life": life,
                        "opening_hand": opening_hand,
                        "mode": args.mode,
                        "deck_size": args.deck_size,
                        "mountains": best.composition.mountains,
                        "bolts": best.composition.bolts,
                        "goblins": best.composition.goblins,
                        "score": f"{best.score:.4f}",
                    }
                )
                print(
                    f"life={life} hand={opening_hand}: "
                    f"M={best.composition.mountains} B={best.composition.bolts} G={best.composition.goblins} score={best.score:.4f}"
                )
    print(f"Wrote deck search results to {output_path}")


if __name__ == "__main__":
    main()

