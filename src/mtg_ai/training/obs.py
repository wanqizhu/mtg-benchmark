from __future__ import annotations

from collections import Counter

from mtg_ai.core.actions import CastSpellAction, DeclareAttackersAction, DeclareBlockersAction, GameAction
from mtg_ai.core.state import GameState


CARD_ORDER = ("Mountain", "Lightning Bolt", "Raging Goblin")


def hand_feature(state: GameState, player_id: int) -> tuple[int, int, int]:
    names = [state.card_instances[cid].definition_name for cid in state.players[player_id].hand]
    counts = Counter(names)
    return tuple(counts.get(card, 0) for card in CARD_ORDER)


def battlefield_feature(state: GameState, player_id: int) -> tuple[int, int, int, int]:
    mountains_untapped = 0
    mountains_tapped = 0
    goblins_untapped = 0
    goblins_tapped = 0
    for permanent_id in state.players[player_id].battlefield:
        permanent = state.permanents.get(permanent_id)
        if permanent is None:
            continue
        name = state.card_instances[permanent.card_id].definition_name
        if name == "Mountain":
            if permanent.tapped:
                mountains_tapped += 1
            else:
                mountains_untapped += 1
        if name == "Raging Goblin":
            if permanent.tapped:
                goblins_tapped += 1
            else:
                goblins_untapped += 1
    return mountains_untapped, mountains_tapped, goblins_untapped, goblins_tapped


def encode_state(state: GameState, player_id: int) -> tuple:
    opponent_id = 1 - player_id
    return (
        state.turn.step.value,
        state.turn.active_player_id == player_id,
        state.turn.priority_player_id == player_id,
        state.players[player_id].life,
        state.players[opponent_id].life,
        len(state.players[player_id].library),
        len(state.players[opponent_id].library),
        state.players[player_id].mana_pool_red,
        state.players[opponent_id].mana_pool_red,
        state.players[player_id].lands_played_this_turn,
        hand_feature(state, player_id),
        hand_feature(state, opponent_id),
        battlefield_feature(state, player_id),
        battlefield_feature(state, opponent_id),
        len(state.stack),
        len(state.combat.attackers),
    )


def encode_action(action: GameAction, perspective_player_id: int) -> str:
    action_type = action.action_type
    if action_type == "cast_spell":
        cast_action = action if isinstance(action, CastSpellAction) else None
        if cast_action is not None:
            if len(cast_action.targets) == 0:
                return "cast_goblin"
            target = cast_action.targets[0]
            if target.kind == "player":
                if int(target.id) == perspective_player_id:
                    return "cast_bolt_self"
                return "cast_bolt_face"
            return "cast_bolt_creature"
    if action_type == "declare_attackers" and isinstance(action, DeclareAttackersAction):
        if len(action.attacker_ids) == 0:
            return "declare_attackers_none"
        return f"declare_attackers_{len(action.attacker_ids)}"
    if action_type == "declare_blockers" and isinstance(action, DeclareBlockersAction):
        if len(action.blocks) == 0:
            return "declare_blockers_none"
        return f"declare_blockers_{len(action.blocks)}"
    if action_type == "activate_mana_ability":
        return "activate_mana"
    if action_type == "play_land":
        return "play_land"
    return action_type

