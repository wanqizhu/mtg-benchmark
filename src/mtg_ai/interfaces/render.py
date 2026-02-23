from __future__ import annotations

from mtg_ai.cards.registry import CardRegistry
from mtg_ai.core.state import GameState


def render_state(state: GameState, registry: CardRegistry, perspective_player_id: int | None = None) -> str:
    lines: list[str] = []
    lines.append(
        f"Turn {state.turn.turn_number} | Active P{state.turn.active_player_id} | Priority P{state.turn.priority_player_id} | Step {state.turn.step.value}"
    )
    lines.append(f"Stack size: {len(state.stack)}")
    if state.stack:
        for item in reversed(state.stack):
            card_name = state.card_instances[item.card_id].definition_name
            target_text = ", ".join(f"{target.kind}:{target.id}" for target in item.targets)
            lines.append(f"  - {item.stack_id} {card_name} by P{item.controller_id} targets [{target_text}]")
    for player_id in sorted(state.players):
        player = state.players[player_id]
        lines.append(
            f"P{player_id} life={player.life} hand={len(player.hand)} library={len(player.library)} graveyard={len(player.graveyard)} mana(R)={player.mana_pool_red}"
        )
        if perspective_player_id == player_id:
            hand_names = [state.card_instances[cid].definition_name for cid in player.hand]
            lines.append(f"  hand: {hand_names}")
        battlefield_items: list[str] = []
        for permanent_id in player.battlefield:
            permanent = state.permanents.get(permanent_id)
            if permanent is None:
                continue
            card_name = state.card_instances[permanent.card_id].definition_name
            definition = registry.get(card_name)
            stats = ""
            if definition.is_creature:
                stats = f" {definition.power}/{definition.toughness} dmg={permanent.damage_marked}"
            tapped = " tapped" if permanent.tapped else ""
            battlefield_items.append(f"{permanent_id}:{card_name}{stats}{tapped}")
        lines.append("  battlefield: " + (", ".join(battlefield_items) if battlefield_items else "-"))
    if state.combat.attackers:
        lines.append(f"Combat attackers: {state.combat.attackers}")
    if state.combat.blocks:
        lines.append(f"Combat blocks: {state.combat.blocks}")
    if state.game_over:
        lines.append(f"Game over: {state.result}")
    return "\n".join(lines)

