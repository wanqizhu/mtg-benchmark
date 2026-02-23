from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from mtg_ai.core.actions import GameAction
from mtg_ai.core.state import GameState


@runtime_checkable
class BotPolicy(Protocol):
    name: str

    def choose_action(self, state: GameState, legal_actions: list[GameAction], rng: random.Random) -> GameAction:
        ...


@dataclass(frozen=True)
class HumanDecisionError(Exception):
    message: str

