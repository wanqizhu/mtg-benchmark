from __future__ import annotations

import random

from mtg_ai.bots.base import BotPolicy
from mtg_ai.bots.rule_helpers import any_block_action, find_face_bolt, no_blocks_action, pass_action, play_land_action, prefer_attack_all
from mtg_ai.core.actions import CastSpellAction, GameAction
from mtg_ai.core.state import GameState


def _first_non_target_spell(actions: list[GameAction]) -> CastSpellAction | None:
    for action in actions:
        if isinstance(action, CastSpellAction) and len(action.targets) == 0:
            return action
    return None


def _fallback_pass(actions: list[GameAction]) -> GameAction:
    action = pass_action(actions)
    if action is None:
        return actions[0]
    return action


class BoltFaceBot(BotPolicy):
    name = "bolt_face"

    def choose_action(self, state: GameState, legal_actions: list[GameAction], rng: random.Random) -> GameAction:
        actor_id = legal_actions[0].actor_id
        opponent_id = 1 - actor_id
        action = play_land_action(legal_actions)
        if action is not None:
            return action
        action = find_face_bolt(legal_actions, opponent_id)
        if action is not None:
            return action
        action = prefer_attack_all(legal_actions)
        if action is not None:
            return action
        action = no_blocks_action(legal_actions)
        if action is not None:
            return action
        return _fallback_pass(legal_actions)


class GoblinAggroBot(BotPolicy):
    name = "goblin_aggro"

    def choose_action(self, state: GameState, legal_actions: list[GameAction], rng: random.Random) -> GameAction:
        action = play_land_action(legal_actions)
        if action is not None:
            return action
        action = _first_non_target_spell(legal_actions)
        if action is not None:
            return action
        action = prefer_attack_all(legal_actions)
        if action is not None:
            return action
        action = any_block_action(legal_actions)
        if action is not None:
            return action
        return _fallback_pass(legal_actions)


class MixedBoltFaceBot(BotPolicy):
    name = "mixed_bolt_face"

    def choose_action(self, state: GameState, legal_actions: list[GameAction], rng: random.Random) -> GameAction:
        actor_id = legal_actions[0].actor_id
        opponent_id = 1 - actor_id
        action = play_land_action(legal_actions)
        if action is not None:
            return action
        action = find_face_bolt(legal_actions, opponent_id)
        if action is not None:
            return action
        action = _first_non_target_spell(legal_actions)
        if action is not None:
            return action
        action = prefer_attack_all(legal_actions)
        if action is not None:
            return action
        action = any_block_action(legal_actions)
        if action is not None:
            return action
        return _fallback_pass(legal_actions)

