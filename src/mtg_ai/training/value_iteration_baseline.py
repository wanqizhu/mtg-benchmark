from __future__ import annotations

import argparse
import copy
from functools import lru_cache

from mtg_ai.bots.fixed_bots import BoltFaceBot, MixedBoltFaceBot
from mtg_ai.core.rules_engine import GameConfig, RulesEngine
from mtg_ai.core.serialization import action_to_dict, serialize_state
from mtg_ai.training.obs import encode_action


def build_deck(mountains: int, bolts: int, goblins: int) -> list[str]:
    return ["Mountain"] * mountains + ["Lightning Bolt"] * bolts + ["Raging Goblin"] * goblins


class DeterministicMinimaxSolver:
    def __init__(self, engine: RulesEngine, max_depth: int = 8) -> None:
        self.engine = engine
        self.max_depth = max_depth

    def best_action(self, state, player_id: int):
        legal_actions = self.engine.legal_actions(state, player_id)
        if not legal_actions:
            return None, 0.0
        best = legal_actions[0]
        best_value = float("-inf")
        for action in legal_actions:
            engine_copy = copy.deepcopy(self.engine)
            state_copy = copy.deepcopy(state)
            engine_copy.apply_action(state_copy, action)
            value = self._value(engine_copy, state_copy, maximizing_player=player_id, depth=self.max_depth - 1)
            if value > best_value:
                best = action
                best_value = value
        return best, best_value

    @lru_cache(maxsize=200000)
    def _cached_value(self, state_key: str, maximizing_player: int, depth: int) -> float:
        raise RuntimeError("This method should never be called directly.")

    def _value(self, engine: RulesEngine, state, maximizing_player: int, depth: int) -> float:
        if state.game_over:
            if state.winner is None:
                return 0.0
            return 1.0 if state.winner == maximizing_player else -1.0
        if depth <= 0:
            opponent = 1 - maximizing_player
            return (state.players[maximizing_player].life - state.players[opponent].life) / 20.0

        state_key = repr(serialize_state(state))
        cache_key = (state_key, maximizing_player, depth)
        if cache_key in self._memo:
            return self._memo[cache_key]

        actor = state.turn.priority_player_id
        legal_actions = engine.legal_actions(state, actor)
        if not legal_actions:
            return 0.0
        values: list[float] = []
        for action in legal_actions:
            engine_copy = copy.deepcopy(engine)
            state_copy = copy.deepcopy(state)
            engine_copy.apply_action(state_copy, action)
            values.append(self._value(engine_copy, state_copy, maximizing_player, depth - 1))
        if actor == maximizing_player:
            value = max(values)
        else:
            value = min(values)
        self._memo[cache_key] = value
        return value

    _memo: dict[tuple[str, int, int], float] = {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic minimax on low-life settings for validation.")
    parser.add_argument("--life", type=int, default=5)
    parser.add_argument("--opening-hand", type=int, default=3)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    engine = RulesEngine(config=GameConfig(starting_life=args.life, opening_hand_size=args.opening_hand, random_seed=args.seed))
    deck_a = build_deck(3, 3, 3)
    deck_b = build_deck(3, 3, 3)
    state, _ = engine.create_game(decks={0: deck_a, 1: deck_b}, preserve_deck_order=True, starting_player_id=0)
    solver = DeterministicMinimaxSolver(engine, max_depth=args.depth)
    action, value = solver.best_action(state, player_id=state.turn.priority_player_id)
    print(f"Best opening action value={value:.3f}: {action_to_dict(action) if action else None}")

    bolt_bot = BoltFaceBot()
    mixed_bot = MixedBoltFaceBot()
    legal = engine.legal_actions(state, state.turn.priority_player_id)
    bolt_choice = bolt_bot.choose_action(state, legal, __import__("random").Random(args.seed))
    mixed_choice = mixed_bot.choose_action(state, legal, __import__("random").Random(args.seed + 1))
    print(f"BoltFace opening action: {encode_action(bolt_choice, state.turn.priority_player_id)}")
    print(f"Mixed opening action: {encode_action(mixed_choice, state.turn.priority_player_id)}")


if __name__ == "__main__":
    main()

