from __future__ import annotations

from mtg_ai.core.actions import CastSpellAction, PassPriorityAction, PlayLandAction
from mtg_ai.core.enums import Step
from mtg_ai.core.rules_engine import GameConfig, RulesEngine
from tests.helpers import choose_action, pass_priority_cycle


def test_stack_resolves_after_both_players_pass() -> None:
    engine = RulesEngine(config=GameConfig(starting_life=20, opening_hand_size=7, random_seed=8))
    deck = ["Mountain"] * 30 + ["Lightning Bolt"] * 30
    state, _ = engine.create_game(decks={0: deck, 1: deck}, starting_player_id=0, preserve_deck_order=True)
    pass_priority_cycle(engine, state)
    pass_priority_cycle(engine, state)
    assert state.turn.step == Step.PRECOMBAT_MAIN

    legal = engine.legal_actions(state, 0)
    play = choose_action(legal, lambda action: isinstance(action, PlayLandAction))
    engine.apply_action(state, play)
    legal = engine.legal_actions(state, 0)
    cast = choose_action(
        legal,
        lambda action: isinstance(action, CastSpellAction)
        and len(action.targets) == 1
        and action.targets[0].kind == "player"
        and action.targets[0].id == 1,
    )
    engine.apply_action(state, cast)
    assert len(state.stack) == 1

    legal = engine.legal_actions(state, 1)
    engine.apply_action(state, choose_action(legal, lambda action: isinstance(action, PassPriorityAction)))
    legal = engine.legal_actions(state, 0)
    engine.apply_action(state, choose_action(legal, lambda action: isinstance(action, PassPriorityAction)))
    assert len(state.stack) == 0
    assert state.players[1].life == 17

