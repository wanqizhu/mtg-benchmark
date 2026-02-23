from __future__ import annotations

from mtg_ai.core.enums import Step
from mtg_ai.core.rules_engine import GameConfig, RulesEngine


def test_cleanup_discards_to_maximum_hand_size() -> None:
    engine = RulesEngine(config=GameConfig(starting_life=20, opening_hand_size=7, random_seed=11))
    deck = ["Mountain"] * 30
    state, _ = engine.create_game(decks={0: deck, 1: deck}, starting_player_id=0, preserve_deck_order=True)
    player = state.players[0]
    player.hand.extend([player.library.pop(), player.library.pop()])
    assert len(player.hand) == 9
    state.turn.active_player_id = 0
    state.turn.step = Step.CLEANUP

    events = []
    engine._begin_step(state, events)

    assert len(player.hand) == 7
    assert len(player.graveyard) == 2
    assert state.turn.active_player_id == 1
    assert state.turn.turn_number == 2

