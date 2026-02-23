from __future__ import annotations

from mtg_ai.core.actions import CastSpellAction, DeclareAttackersAction, DeclareBlockersAction, PlayLandAction
from mtg_ai.core.enums import Step
from mtg_ai.core.rules_engine import GameConfig, RulesEngine
from tests.helpers import choose_action, pass_priority_cycle


def test_raging_goblin_can_attack_on_entry_due_to_haste() -> None:
    engine = RulesEngine(config=GameConfig(starting_life=20, opening_hand_size=7, random_seed=5))
    deck = ["Mountain"] * 30 + ["Raging Goblin"] * 30
    state, _ = engine.create_game(decks={0: deck, 1: deck}, starting_player_id=0, preserve_deck_order=True)

    pass_priority_cycle(engine, state)
    pass_priority_cycle(engine, state)
    assert state.turn.step == Step.PRECOMBAT_MAIN

    legal = engine.legal_actions(state, 0)
    play = choose_action(legal, lambda action: isinstance(action, PlayLandAction))
    engine.apply_action(state, play)
    legal = engine.legal_actions(state, 0)
    cast = choose_action(legal, lambda action: isinstance(action, CastSpellAction) and len(action.targets) == 0)
    engine.apply_action(state, cast)

    pass_priority_cycle(engine, state)
    pass_priority_cycle(engine, state)
    assert state.turn.step == Step.DECLARE_ATTACKERS

    legal = engine.legal_actions(state, 0)
    attack = choose_action(legal, lambda action: isinstance(action, DeclareAttackersAction) and len(action.attacker_ids) == 1)
    attacker_id = attack.attacker_ids[0]
    engine.apply_action(state, attack)

    legal = engine.legal_actions(state, 1)
    no_blocks = choose_action(legal, lambda action: isinstance(action, DeclareBlockersAction) and len(action.blocks) == 0)
    engine.apply_action(state, no_blocks)

    assert state.turn.step == Step.END_COMBAT
    assert state.players[1].life == 19
    assert state.permanents[attacker_id].tapped

