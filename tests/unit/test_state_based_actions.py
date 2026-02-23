from __future__ import annotations

from mtg_ai.core.actions import CastSpellAction, PassPriorityAction
from mtg_ai.core.enums import Step
from mtg_ai.core.rules_engine import GameConfig, RulesEngine
from mtg_ai.core.state import PermanentState
from tests.helpers import choose_action, pass_priority_cycle


def _pass_until(engine: RulesEngine, state, predicate, max_cycles: int = 200) -> None:
    for _ in range(max_cycles):
        if predicate():
            return
        pass_priority_cycle(engine, state)
    raise AssertionError("Condition not reached.")


def test_player_loses_when_life_reaches_zero() -> None:
    engine = RulesEngine(config=GameConfig(starting_life=20, opening_hand_size=7, random_seed=6))
    deck0 = ["Mountain"] * 10 + ["Lightning Bolt"] * 10
    deck1 = ["Mountain"] * 20
    state, _ = engine.create_game(decks={0: deck0, 1: deck1}, starting_player_id=0, preserve_deck_order=True)
    for player in state.players.values():
        player.hand.clear()
        player.library.clear()
        player.graveyard.clear()
        player.battlefield.clear()
        player.mana_pool_red = 0
    bolt = next(card_id for card_id, card in state.card_instances.items() if card.owner_id == 0 and card.definition_name == "Lightning Bolt")
    mountain = next(card_id for card_id, card in state.card_instances.items() if card.owner_id == 0 and card.definition_name == "Mountain")
    state.players[0].hand.append(bolt)
    state.players[1].life = 3
    perm = "perm100"
    state.permanents = {perm: PermanentState(permanent_id=perm, card_id=mountain, controller_id=0, entered_turn_number=1)}
    state.players[0].battlefield.append(perm)
    state.turn.step = Step.PRECOMBAT_MAIN
    state.turn.active_player_id = 0
    state.turn.priority_player_id = 0

    legal = engine.legal_actions(state, 0)
    cast_face = choose_action(
        legal,
        lambda action: isinstance(action, CastSpellAction)
        and len(action.targets) == 1
        and action.targets[0].kind == "player"
        and action.targets[0].id == 1,
    )
    engine.apply_action(state, cast_face)
    legal = engine.legal_actions(state, 1)
    engine.apply_action(state, choose_action(legal, lambda action: isinstance(action, PassPriorityAction)))
    legal = engine.legal_actions(state, 0)
    engine.apply_action(state, choose_action(legal, lambda action: isinstance(action, PassPriorityAction)))

    assert state.game_over
    assert state.winner == 0


def test_draw_from_empty_library_causes_loss_on_next_draw() -> None:
    engine = RulesEngine(config=GameConfig(starting_life=20, opening_hand_size=7, random_seed=7))
    deck = ["Mountain"] * 7
    state, _ = engine.create_game(decks={0: deck, 1: ["Mountain"] * 30}, starting_player_id=0, preserve_deck_order=True)
    assert len(state.players[0].library) == 0
    assert not state.game_over

    _pass_until(
        engine,
        state,
        predicate=lambda: state.turn.turn_number == 2 and state.turn.active_player_id == 0 and state.turn.step == Step.DRAW,
    )
    assert state.game_over
    assert state.winner == 1

