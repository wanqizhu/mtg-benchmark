from __future__ import annotations

import argparse
from typing import Callable

from mtg_ai.bots.base import BotPolicy
from mtg_ai.bots.fixed_bots import BoltFaceBot, GoblinAggroBot, MixedBoltFaceBot
from mtg_ai.bots.random_bot import RandomBot
from mtg_ai.core.actions import GameAction
from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.game.session import DecisionFn, GameSession
from mtg_ai.interfaces.render import render_state


def parse_deck_spec(spec: str) -> list[str]:
    cards: list[str] = []
    if not spec.strip():
        return cards
    for chunk in spec.split(","):
        name, count = chunk.split("*")
        cards.extend([name.strip()] * int(count))
    return cards


def action_label(action: GameAction) -> str:
    fields = action.__dict__.copy()
    action_type = fields.pop("action_type")
    actor_id = fields.pop("actor_id")
    return f"{action_type}(P{actor_id}, {fields})"


def human_controller_factory(player_id: int) -> DecisionFn:
    def choose_action(state, legal_actions):
        print(render_state(state, registry=session.registry, perspective_player_id=player_id))
        print(f"\nPlayer {player_id} legal actions:")
        for idx, action in enumerate(legal_actions):
            print(f"  [{idx}] {action_label(action)}")
        while True:
            raw = input(f"P{player_id}> ").strip()
            if raw.isdigit():
                index = int(raw)
                if 0 <= index < len(legal_actions):
                    return legal_actions[index]
            print("Invalid action index.")

    return choose_action


def build_bot(name: str) -> BotPolicy:
    if name == "random":
        return RandomBot()
    if name == "bolt":
        return BoltFaceBot()
    if name == "goblin":
        return GoblinAggroBot()
    if name == "mixed":
        return MixedBoltFaceBot()
    raise ValueError(f"Unknown bot type: {name}")


def resolve_controller(spec: str, player_id: int) -> BotPolicy | DecisionFn:
    if spec == "human":
        return human_controller_factory(player_id)
    return build_bot(spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play constrained MTG games.")
    parser.add_argument("--player0", default="human", choices=["human", "random", "bolt", "goblin", "mixed"])
    parser.add_argument("--player1", default="random", choices=["human", "random", "bolt", "goblin", "mixed"])
    parser.add_argument("--deck0", default="Mountain*20,Lightning Bolt*20,Raging Goblin*20")
    parser.add_argument("--deck1", default="Mountain*20,Lightning Bolt*20,Raging Goblin*20")
    parser.add_argument("--life", type=int, default=20)
    parser.add_argument("--opening-hand", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--replay-path", default="artifacts/replays/last_game.jsonl")
    args = parser.parse_args()

    decks = {0: parse_deck_spec(args.deck0), 1: parse_deck_spec(args.deck1)}
    config = GameConfig(starting_life=args.life, opening_hand_size=args.opening_hand, random_seed=args.seed)
    controllers: dict[int, BotPolicy | DecisionFn] = {
        0: resolve_controller(args.player0, 0),
        1: resolve_controller(args.player1, 1),
    }
    global session
    session = GameSession(decks=decks, controllers=controllers, config=config, seed=args.seed)
    result = session.run()
    print(render_state(session.state, registry=session.registry, perspective_player_id=None))
    print(f"\nResult: {result.result}, winner={result.winner}, turns={result.turns}, actions={result.actions}")
    session.replay.save_jsonl(args.replay_path)
    print(f"Replay saved to {args.replay_path}")


if __name__ == "__main__":
    main()

