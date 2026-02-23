from __future__ import annotations

from mtg_ai.core.actions import CastSpellAction, PassPriorityAction
from mtg_ai.core.enums import Step
from mtg_ai.core.rules_engine import GameConfig, RulesEngine
from mtg_ai.core.state import PermanentState
from tests.helpers import choose_action


def _pick_cards(state, owner_id: int, name: str, count: int) -> list[str]:
    picked: list[str] = []
    for card_id, card in state.card_instances.items():
        if card.owner_id != owner_id:
            continue
        if card.definition_name != name:
            continue
        picked.append(card_id)
        if len(picked) == count:
            return picked
    raise ValueError(f"Could not find {count} cards for {name}")


def test_lightning_bolt_fizzles_when_target_illegal_at_resolution() -> None:
    engine = RulesEngine(config=GameConfig(starting_life=20, opening_hand_size=7, random_seed=4))
    deck0 = ["Mountain"] * 10 + ["Lightning Bolt"] * 10
    deck1 = ["Mountain"] * 10 + ["Raging Goblin"] * 10
    state, _ = engine.create_game(decks={0: deck0, 1: deck1}, starting_player_id=0, preserve_deck_order=True)

    for player in state.players.values():
        player.hand.clear()
        player.library.clear()
        player.graveyard.clear()
        player.battlefield.clear()
        player.mana_pool_red = 0

    bolt_a, bolt_b = _pick_cards(state, 0, "Lightning Bolt", 2)
    mountain_a, mountain_b = _pick_cards(state, 0, "Mountain", 2)
    goblin_card = _pick_cards(state, 1, "Raging Goblin", 1)[0]
    state.players[0].hand.extend([bolt_a, bolt_b])

    perm_m1 = "perm10"
    perm_m2 = "perm11"
    perm_g1 = "perm12"
    state.permanents = {
        perm_m1: PermanentState(permanent_id=perm_m1, card_id=mountain_a, controller_id=0, entered_turn_number=1),
        perm_m2: PermanentState(permanent_id=perm_m2, card_id=mountain_b, controller_id=0, entered_turn_number=1),
        perm_g1: PermanentState(permanent_id=perm_g1, card_id=goblin_card, controller_id=1, entered_turn_number=1),
    }
    state.players[0].battlefield.extend([perm_m1, perm_m2])
    state.players[1].battlefield.append(perm_g1)
    engine._next_permanent_id = 20
    state.turn.step = Step.PRECOMBAT_MAIN
    state.turn.active_player_id = 0
    state.turn.priority_player_id = 0

    legal = engine.legal_actions(state, 0)
    cast_first = choose_action(
        legal,
        lambda action: isinstance(action, CastSpellAction)
        and len(action.targets) == 1
        and action.targets[0].kind == "permanent"
        and action.targets[0].id == perm_g1,
    )
    engine.apply_action(state, cast_first)

    legal = engine.legal_actions(state, 1)
    engine.apply_action(state, choose_action(legal, lambda action: isinstance(action, PassPriorityAction)))

    legal = engine.legal_actions(state, 0)
    cast_second = choose_action(
        legal,
        lambda action: isinstance(action, CastSpellAction)
        and len(action.targets) == 1
        and action.targets[0].kind == "permanent"
        and action.targets[0].id == perm_g1,
    )
    engine.apply_action(state, cast_second)

    legal = engine.legal_actions(state, 1)
    engine.apply_action(state, choose_action(legal, lambda action: isinstance(action, PassPriorityAction)))
    legal = engine.legal_actions(state, 0)
    engine.apply_action(state, choose_action(legal, lambda action: isinstance(action, PassPriorityAction)))

    assert perm_g1 not in state.permanents
    assert len(state.stack) == 1

    legal = engine.legal_actions(state, 1)
    engine.apply_action(state, choose_action(legal, lambda action: isinstance(action, PassPriorityAction)))
    legal = engine.legal_actions(state, 0)
    engine.apply_action(state, choose_action(legal, lambda action: isinstance(action, PassPriorityAction)))

    assert len(state.stack) == 0
    assert len(state.players[0].graveyard) == 2
    assert len(state.players[1].graveyard) == 1

