from __future__ import annotations

from mtg_ai.core.actions import (
    ActivateManaAbilityAction,
    CastSpellAction,
    DeclareAttackersAction,
    DeclareBlockersAction,
    GameAction,
    PassPriorityAction,
    PlayLandAction,
)


def first_action_of_type(actions: list[GameAction], action_type: type[GameAction]) -> GameAction | None:
    for action in actions:
        if isinstance(action, action_type):
            return action
    return None


def actions_of_type(actions: list[GameAction], action_type: type[GameAction]) -> list[GameAction]:
    return [action for action in actions if isinstance(action, action_type)]


def find_face_bolt(actions: list[GameAction], opponent_id: int) -> CastSpellAction | None:
    for action in actions:
        if isinstance(action, CastSpellAction):
            if len(action.targets) == 1 and action.targets[0].kind == "player" and int(action.targets[0].id) == opponent_id:
                return action
    return None


def prefer_attack_all(actions: list[GameAction]) -> GameAction | None:
    declare_actions = [action for action in actions if isinstance(action, DeclareAttackersAction)]
    if not declare_actions:
        return None
    return max(declare_actions, key=lambda action: len(action.attacker_ids))


def no_blocks_action(actions: list[GameAction]) -> GameAction | None:
    for action in actions:
        if isinstance(action, DeclareBlockersAction) and len(action.blocks) == 0:
            return action
    return None


def any_block_action(actions: list[GameAction]) -> GameAction | None:
    blockers = [action for action in actions if isinstance(action, DeclareBlockersAction)]
    if not blockers:
        return None
    return max(blockers, key=lambda action: len(action.blocks))


def pass_action(actions: list[GameAction]) -> PassPriorityAction | None:
    for action in actions:
        if isinstance(action, PassPriorityAction):
            return action
    return None


def mana_action(actions: list[GameAction]) -> ActivateManaAbilityAction | None:
    for action in actions:
        if isinstance(action, ActivateManaAbilityAction):
            return action
    return None


def play_land_action(actions: list[GameAction]) -> PlayLandAction | None:
    for action in actions:
        if isinstance(action, PlayLandAction):
            return action
    return None

