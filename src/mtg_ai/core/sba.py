from __future__ import annotations

from mtg_ai.cards.registry import CardRegistry
from mtg_ai.core.state import GameState


def apply_state_based_actions(state: GameState, registry: CardRegistry) -> list[str]:
    applied: list[str] = []
    while True:
        changed = False

        dead_permanents: list[str] = []
        for permanent_id, permanent in state.permanents.items():
            definition = registry.get(state.card_instances[permanent.card_id].definition_name)
            if definition.is_creature and definition.toughness is not None and permanent.damage_marked >= definition.toughness:
                dead_permanents.append(permanent_id)
        for permanent_id in dead_permanents:
            _move_permanent_to_graveyard(state, permanent_id)
            applied.append(f"destroy:{permanent_id}")
            changed = True

        losing_players: set[int] = set()
        for player_id, player in state.players.items():
            if player.life <= 0:
                losing_players.add(player_id)
            if player_id in state.pending_draw_loss:
                losing_players.add(player_id)

        if losing_players:
            _apply_losses(state, losing_players)
            applied.append(f"lose:{','.join(str(pid) for pid in sorted(losing_players))}")
            changed = True
            if state.game_over:
                return applied

        if not changed:
            break

    return applied


def _move_permanent_to_graveyard(state: GameState, permanent_id: str) -> None:
    permanent = state.permanents.pop(permanent_id)
    player = state.players[permanent.controller_id]
    if permanent_id in player.battlefield:
        player.battlefield.remove(permanent_id)
    player.graveyard.append(permanent.card_id)
    if permanent_id in state.combat.attackers:
        state.combat.attackers.remove(permanent_id)
    if permanent_id in state.combat.blocks:
        del state.combat.blocks[permanent_id]
    for attacker, blockers in list(state.combat.blocks.items()):
        if permanent_id in blockers:
            blockers.remove(permanent_id)
        if not blockers:
            del state.combat.blocks[attacker]


def _apply_losses(state: GameState, losers: set[int]) -> None:
    if len(losers) == len(state.players):
        state.game_over = True
        state.winner = None
        state.result = "draw"
        return
    if len(losers) > 1:
        state.game_over = True
        state.winner = None
        state.result = "draw"
        return
    loser = next(iter(losers))
    winner = next(player_id for player_id in state.players if player_id != loser)
    state.game_over = True
    state.winner = winner
    state.result = f"player_{winner}_wins"

