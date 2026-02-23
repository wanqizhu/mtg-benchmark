from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class TargetRef:
    kind: Literal["player", "permanent"]
    id: int | str

    def to_key(self) -> str:
        return f"{self.kind}:{self.id}"


@dataclass(frozen=True)
class GameAction:
    actor_id: int
    action_type: str


@dataclass(frozen=True)
class PassPriorityAction(GameAction):
    action_type: str = "pass_priority"


@dataclass(frozen=True)
class ConcedeAction(GameAction):
    action_type: str = "concede"


@dataclass(frozen=True)
class PlayLandAction(GameAction):
    card_id: str = ""
    action_type: str = "play_land"


@dataclass(frozen=True)
class ActivateManaAbilityAction(GameAction):
    permanent_id: str = ""
    action_type: str = "activate_mana_ability"


@dataclass(frozen=True)
class CastSpellAction(GameAction):
    card_id: str = ""
    targets: tuple[TargetRef, ...] = field(default_factory=tuple)
    tap_mana_sources: tuple[str, ...] = field(default_factory=tuple)
    action_type: str = "cast_spell"


@dataclass(frozen=True)
class DeclareAttackersAction(GameAction):
    attacker_ids: tuple[str, ...] = field(default_factory=tuple)
    action_type: str = "declare_attackers"


@dataclass(frozen=True)
class DeclareBlockersAction(GameAction):
    blocks: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    action_type: str = "declare_blockers"

