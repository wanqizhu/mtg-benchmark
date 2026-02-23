from __future__ import annotations

from mtg_ai.core.enums import Step
from mtg_ai.core.rules_engine import GameConfig, RulesEngine
from tests.helpers import pass_priority_cycle


def test_starting_player_skips_first_draw_step() -> None:
    engine = RulesEngine(config=GameConfig(starting_life=20, opening_hand_size=7, random_seed=1))
    deck = ["Mountain"] * 20
    state, _ = engine.create_game(decks={0: deck, 1: deck}, starting_player_id=0, preserve_deck_order=True)
    assert state.turn.step == Step.UPKEEP
    assert len(state.players[0].hand) == 7

    pass_priority_cycle(engine, state)
    assert state.turn.step == Step.DRAW
    assert len(state.players[0].hand) == 7

    pass_priority_cycle(engine, state)
    assert state.turn.step == Step.PRECOMBAT_MAIN
    assert len(state.players[0].hand) == 7


def test_second_player_draws_on_first_turn() -> None:
    engine = RulesEngine(config=GameConfig(starting_life=20, opening_hand_size=7, random_seed=1))
    deck = ["Mountain"] * 20
    state, _ = engine.create_game(decks={0: deck, 1: deck}, starting_player_id=0, preserve_deck_order=True)
    target_turn = 1
    target_active_player = 1
    target_step = Step.DRAW

    for _ in range(40):
        if state.turn.active_player_id == target_active_player and state.turn.step == target_step:
            break
        pass_priority_cycle(engine, state)
    assert state.turn.active_player_id == target_active_player
    assert state.turn.step == target_step
    assert len(state.players[1].hand) == 8

