from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mtg_ai.bots.fixed_bots import BoltFaceBot, GoblinAggroBot, MixedBoltFaceBot
from mtg_ai.bots.random_bot import RandomBot
from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.game.session import GameSession
from mtg_ai.training.rl_policy import WeightedHeuristicPolicy


def build_deck() -> list[str]:
    return ["Mountain"] * 12 + ["Lightning Bolt"] * 9 + ["Raging Goblin"] * 9


def load_champion_policy(champion_json_path: Path) -> tuple[str, WeightedHeuristicPolicy]:
    payload = json.loads(champion_json_path.read_text(encoding="utf-8"))
    policy_path = Path(payload["path"])
    policy = WeightedHeuristicPolicy.load(policy_path)
    return payload["name"], policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample replay logs for key bot-vs-bot matchups.")
    parser.add_argument("--champion-json", default="artifacts/eval/champion.json")
    parser.add_argument("--output-dir", default="artifacts/replays/samples")
    parser.add_argument("--index-csv", default="artifacts/replays/samples/index.csv")
    parser.add_argument("--seed", type=int, default=7101)
    args = parser.parse_args()

    champion_name, champion_policy = load_champion_policy(Path(args.champion_json))
    deck = build_deck()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matchups = [
        ("champion_vs_mixed_l10_h7", champion_policy, MixedBoltFaceBot(), 10, 7),
        ("champion_vs_bolt_l10_h7", champion_policy, BoltFaceBot(), 10, 7),
        ("champion_vs_random_l10_h7", champion_policy, RandomBot(), 10, 7),
        ("champion_vs_goblin_l5_h5", champion_policy, GoblinAggroBot(), 5, 5),
        ("mixed_vs_bolt_l10_h7", MixedBoltFaceBot(), BoltFaceBot(), 10, 7),
        ("mixed_vs_goblin_l10_h7", MixedBoltFaceBot(), GoblinAggroBot(), 10, 7),
    ]

    rows: list[dict[str, object]] = []
    for idx, (name, bot_a, bot_b, life, hand) in enumerate(matchups):
        seed = args.seed + idx * 31
        session = GameSession(
            decks={0: list(deck), 1: list(deck)},
            controllers={0: bot_a, 1: bot_b},
            config=GameConfig(starting_life=life, opening_hand_size=hand, random_seed=seed),
            seed=seed,
        )
        result = session.run(max_actions=260)
        replay_path = output_dir / f"{name}.jsonl"
        session.replay.save_jsonl(replay_path)
        rows.append(
            {
                "matchup": name,
                "player0": getattr(bot_a, "name", "policy_a"),
                "player1": getattr(bot_b, "name", "policy_b"),
                "life": life,
                "opening_hand": hand,
                "winner": result.winner if result.winner is not None else "draw",
                "result": result.result,
                "turns": result.turns,
                "actions": result.actions,
                "replay_path": str(replay_path),
                "champion_source": champion_name,
            }
        )

    index_csv = Path(args.index_csv)
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    with index_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote replay sample index to {index_csv}")


if __name__ == "__main__":
    main()

