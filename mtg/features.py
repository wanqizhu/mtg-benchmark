from __future__ import annotations
import numpy as np
from .enums import ActionType, TargetType, Step
from .game import Action, GameView

NUM_STEPS = 12
MAX_CREATURES = 8
STATE_DIM = 33

# Compact action layout:
# 0: PASS_PRIORITY
# 1: PLAY_MOUNTAIN
# 2: CAST_BOLT → opponent
# 3: CAST_BOLT → self
# 4..11: CAST_BOLT → my creature [0..7]
# 12..19: CAST_BOLT → opp creature [0..7]
# 20: CAST_GOBLIN
# 21..276: DECLARE_ATTACKERS bitmask [0..255] (up to 8 creatures)
# 277: NO_BLOCK
# 278..285: BLOCK_WITH_CREATURE [0..7]
# 286: KEEP_HAND
# 287: MULLIGAN
# 288..294: PUT_CARD_ON_BOTTOM [hand idx 0..6]
ACTION_DIM = 295

STEP_INDEX = {s: i for i, s in enumerate([
    Step.UNTAP, Step.UPKEEP, Step.DRAW,
    Step.PRECOMBAT_MAIN,
    Step.BEGIN_COMBAT, Step.DECLARE_ATTACKERS, Step.DECLARE_BLOCKERS,
    Step.COMBAT_DAMAGE, Step.END_COMBAT,
    Step.POSTCOMBAT_MAIN,
    Step.END_STEP, Step.CLEANUP,
])}


def state_to_features(view: GameView) -> np.ndarray:
    f = np.zeros(STATE_DIM, dtype=np.float32)
    sl = max(view.starting_life, 1)

    f[0] = view.my_life / sl
    f[1] = view.opp_life / sl
    f[2] = sum(1 for c in view.my_hand if c.name == "Mountain")
    f[3] = sum(1 for c in view.my_hand if c.name == "Lightning Bolt")
    f[4] = sum(1 for c in view.my_hand if c.name == "Raging Goblin")

    my_bf = view.my_battlefield
    f[5] = sum(1 for c in my_bf if c.name == "Mountain" and not c.tapped)
    f[6] = sum(1 for c in my_bf if c.name == "Mountain" and c.tapped)
    f[7] = sum(1 for c in my_bf if c.is_creature and not c.tapped)
    f[8] = sum(1 for c in my_bf if c.is_creature and c.tapped)

    opp_bf = view.opp_battlefield
    f[9] = sum(1 for c in opp_bf if c.name == "Mountain" and not c.tapped)
    f[10] = sum(1 for c in opp_bf if c.name == "Mountain" and c.tapped)
    f[11] = sum(1 for c in opp_bf if c.is_creature and not c.tapped)
    f[12] = sum(1 for c in opp_bf if c.is_creature and c.tapped)

    f[13] = view.opp_hand_size
    f[14] = view.my_library_size / 60.0
    f[15] = view.opp_library_size / 60.0

    si = STEP_INDEX.get(view.step, 0)
    f[16 + si] = 1.0

    f[28] = 1.0 if view.active_player == view.my_player else 0.0
    f[29] = 1.0 if view.land_played_this_turn else 0.0
    f[30] = len(view.stack)
    f[31] = view.turn_number / 20.0
    f[32] = sum(1 for c in my_bf if c.is_creature)

    return f


def _sorted_creatures(battlefield: list) -> list:
    return sorted([c for c in battlefield if c.is_creature], key=lambda c: c.instance_id)


def _creature_slot(creature_id: int, battlefield: list) -> int:
    for i, c in enumerate(_sorted_creatures(battlefield)):
        if c.instance_id == creature_id:
            return i
    return -1


def action_to_index(action: Action, view: GameView) -> int:
    at = action.action_type

    if at == ActionType.PASS_PRIORITY:
        return 0
    if at == ActionType.PLAY_LAND:
        return 1

    if at == ActionType.CAST_SPELL and action.card:
        if action.card.name == "Lightning Bolt":
            if action.targets:
                t = action.targets[0]
                if t.target_type == TargetType.PLAYER:
                    return 2 if t.target_id != view.my_player else 3
                elif t.target_type == TargetType.CREATURE:
                    slot = _creature_slot(t.target_id, view.my_battlefield)
                    if 0 <= slot < MAX_CREATURES:
                        return 4 + slot
                    slot = _creature_slot(t.target_id, view.opp_battlefield)
                    if 0 <= slot < MAX_CREATURES:
                        return 12 + slot
            return 2
        if action.card.name == "Raging Goblin":
            return 20

    if at == ActionType.DECLARE_ATTACKERS:
        my_creatures = _sorted_creatures(view.my_battlefield)
        mask = 0
        for i, c in enumerate(my_creatures[:MAX_CREATURES]):
            if c.instance_id in action.attackers:
                mask |= (1 << i)
        return 21 + mask

    if at == ActionType.DECLARE_BLOCKERS:
        if action.blocker_id is None:
            return 277
        slot = _creature_slot(action.blocker_id, view.my_battlefield)
        if 0 <= slot < MAX_CREATURES:
            return 278 + slot
        return 277

    if at == ActionType.MULLIGAN_KEEP:
        return 286
    if at == ActionType.MULLIGAN_TAKE:
        return 287

    if at == ActionType.CHOOSE_BOTTOM_CARD:
        if action.hand_index is not None:
            return 288 + min(action.hand_index, 6)
        return 288

    return 0


def get_legal_action_mask(legal_actions: list[Action], view: GameView) -> np.ndarray:
    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    for a in legal_actions:
        idx = action_to_index(a, view)
        if idx < ACTION_DIM:
            mask[idx] = True
    return mask


def index_to_action(idx: int, legal_actions: list[Action], view: GameView) -> Action:
    for a in legal_actions:
        if action_to_index(a, view) == idx:
            return a
    return legal_actions[0]
