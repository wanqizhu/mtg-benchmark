from __future__ import annotations

import random

from mtg_ai.bots.base import BotPolicy
from mtg_ai.core.actions import ActivateManaAbilityAction, GameAction
from mtg_ai.core.state import GameState


class RandomBot(BotPolicy):
    name = "random"

    def choose_action(self, state: GameState, legal_actions: list[GameAction], rng: random.Random) -> GameAction:
        non_mana = [action for action in legal_actions if not isinstance(action, ActivateManaAbilityAction)]
        if non_mana:
            return rng.choice(non_mana)
        return rng.choice(legal_actions)

