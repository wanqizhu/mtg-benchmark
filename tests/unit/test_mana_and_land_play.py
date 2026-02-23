from __future__ import annotations

from mtg_ai.core.actions import ActivateManaAbilityAction, PlayLandAction
from mtg_ai.core.enums import Step
from mtg_ai.core.rules_engine import GameConfig, RulesEngine
from tests.helpers import choose_action, pass_priority_cycle


def test_land_play_limited_to_one_per_turn() -> None:
    engine = RulesEngine(config=GameConfig(starting_life=20, opening_hand_size=7, random_seed=2, expose_mana_actions=True))
    deck = ["Mountain"] * 30
    state, _ = engine.create_game(decks={0: deck, 1: deck}, starting_player_id=0, preserve_deck_order=True)
    pass_priority_cycle(engine, state)
    pass_priority_cycle(engine, state)
    assert state.turn.step == Step.PRECOMBAT_MAIN

    legal = engine.legal_actions(state, 0)
    play = choose_action(legal, lambda a: isinstance(a, PlayLandAction))
    engine.apply_action(state, play)
    legal_after = engine.legal_actions(state, 0)
    assert all(not isinstance(action, PlayLandAction) for action in legal_after)


def test_mana_pool_empties_between_steps() -> None:
    engine = RulesEngine(config=GameConfig(starting_life=20, opening_hand_size=7, random_seed=3, expose_mana_actions=True))
    deck = ["Mountain"] * 30
    state, _ = engine.create_game(decks={0: deck, 1: deck}, starting_player_id=0, preserve_deck_order=True)
    pass_priority_cycle(engine, state)
    pass_priority_cycle(engine, state)

    legal = engine.legal_actions(state, 0)
    play = choose_action(legal, lambda a: isinstance(a, PlayLandAction))
    engine.apply_action(state, play)
    legal = engine.legal_actions(state, 0)
    mana = choose_action(legal, lambda a: isinstance(a, ActivateManaAbilityAction))
    engine.apply_action(state, mana)
    assert state.players[0].mana_pool_red == 1

    pass_priority_cycle(engine, state)
    assert state.turn.step == Step.BEGIN_COMBAT
    assert state.players[0].mana_pool_red == 0

