from __future__ import annotations

from typing import Callable

from mtg_ai.core.actions import GameAction, PassPriorityAction
from mtg_ai.core.rules_engine import RulesEngine
from mtg_ai.core.state import GameState


def choose_action(legal_actions: list[GameAction], predicate: Callable[[GameAction], bool]) -> GameAction:
    for action in legal_actions:
        if predicate(action):
            return action
    raise ValueError("Expected action not found.")


def pass_priority_cycle(engine: RulesEngine, state: GameState) -> None:
    actor = state.turn.priority_player_id
    legal = engine.legal_actions(state, actor)
    action = choose_action(legal, lambda a: isinstance(a, PassPriorityAction))
    engine.apply_action(state, action)
    actor = state.turn.priority_player_id
    legal = engine.legal_actions(state, actor)
    action = choose_action(legal, lambda a: isinstance(a, PassPriorityAction))
    engine.apply_action(state, action)

