from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mtg_ai.core.actions import (
    ActivateManaAbilityAction,
    CastSpellAction,
    ConcedeAction,
    DeclareAttackersAction,
    DeclareBlockersAction,
    GameAction,
    PassPriorityAction,
    PlayLandAction,
    TargetRef,
)
from mtg_ai.core.state import GameState


def action_to_dict(action: GameAction) -> dict[str, Any]:
    if isinstance(action, CastSpellAction):
        return {
            "action_type": action.action_type,
            "actor_id": action.actor_id,
            "card_id": action.card_id,
            "targets": [asdict(t) for t in action.targets],
            "tap_mana_sources": list(action.tap_mana_sources),
        }
    if isinstance(action, DeclareAttackersAction):
        return {
            "action_type": action.action_type,
            "actor_id": action.actor_id,
            "attacker_ids": list(action.attacker_ids),
        }
    if isinstance(action, DeclareBlockersAction):
        return {
            "action_type": action.action_type,
            "actor_id": action.actor_id,
            "blocks": [list(pair) for pair in action.blocks],
        }
    if isinstance(action, PlayLandAction):
        return {"action_type": action.action_type, "actor_id": action.actor_id, "card_id": action.card_id}
    if isinstance(action, ActivateManaAbilityAction):
        return {
            "action_type": action.action_type,
            "actor_id": action.actor_id,
            "permanent_id": action.permanent_id,
        }
    if isinstance(action, PassPriorityAction | ConcedeAction):
        return {"action_type": action.action_type, "actor_id": action.actor_id}
    raise ValueError(f"Unsupported action type: {type(action)}")


def action_from_dict(payload: dict[str, Any]) -> GameAction:
    action_type = payload["action_type"]
    actor_id = int(payload["actor_id"])
    if action_type == "cast_spell":
        targets = tuple(TargetRef(kind=t["kind"], id=t["id"]) for t in payload["targets"])
        return CastSpellAction(
            actor_id=actor_id,
            card_id=payload["card_id"],
            targets=targets,
            tap_mana_sources=tuple(payload["tap_mana_sources"]),
        )
    if action_type == "declare_attackers":
        return DeclareAttackersAction(actor_id=actor_id, attacker_ids=tuple(payload["attacker_ids"]))
    if action_type == "declare_blockers":
        return DeclareBlockersAction(actor_id=actor_id, blocks=tuple(tuple(pair) for pair in payload["blocks"]))
    if action_type == "play_land":
        return PlayLandAction(actor_id=actor_id, card_id=payload["card_id"])
    if action_type == "activate_mana_ability":
        return ActivateManaAbilityAction(actor_id=actor_id, permanent_id=payload["permanent_id"])
    if action_type == "pass_priority":
        return PassPriorityAction(actor_id=actor_id)
    if action_type == "concede":
        return ConcedeAction(actor_id=actor_id)
    raise ValueError(f"Unsupported action_type {action_type}")


def serialize_state(state: GameState) -> dict[str, Any]:
    players: dict[str, Any] = {}
    for player_id, player in sorted(state.players.items()):
        players[str(player_id)] = {
            "life": player.life,
            "library": list(player.library),
            "hand": list(player.hand),
            "graveyard": list(player.graveyard),
            "exile": list(player.exile),
            "battlefield": list(player.battlefield),
            "mana_pool_red": player.mana_pool_red,
            "lands_played_this_turn": player.lands_played_this_turn,
            "mulligans_taken": player.mulligans_taken,
        }
    permanents: dict[str, Any] = {}
    for permanent_id, permanent in sorted(state.permanents.items()):
        permanents[permanent_id] = {
            "card_id": permanent.card_id,
            "controller_id": permanent.controller_id,
            "tapped": permanent.tapped,
            "damage_marked": permanent.damage_marked,
            "entered_turn_number": permanent.entered_turn_number,
        }
    stack: list[dict[str, Any]] = []
    for item in state.stack:
        stack.append(
            {
                "stack_id": item.stack_id,
                "card_id": item.card_id,
                "controller_id": item.controller_id,
                "targets": [asdict(target) for target in item.targets],
            }
        )
    return {
        "turn": {
            "turn_number": state.turn.turn_number,
            "active_player_id": state.turn.active_player_id,
            "priority_player_id": state.turn.priority_player_id,
            "step": state.turn.step.value,
            "starting_player_id": state.turn.starting_player_id,
            "passed_priority_in_row": state.turn.passed_priority_in_row,
        },
        "players": players,
        "permanents": permanents,
        "stack": stack,
        "combat": {
            "attackers": list(state.combat.attackers),
            "blocks": {k: list(v) for k, v in sorted(state.combat.blocks.items())},
            "defending_player_id": state.combat.defending_player_id,
        },
        "game_over": state.game_over,
        "winner": state.winner,
        "result": state.result,
        "pending_draw_loss": sorted(state.pending_draw_loss),
        "event_index": state.event_index,
    }

