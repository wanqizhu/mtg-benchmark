from __future__ import annotations

from dataclasses import dataclass, field

from mtg_ai.core.enums import CardType


@dataclass(frozen=True)
class ManaCost:
    red: int = 0

    @property
    def total(self) -> int:
        return self.red


@dataclass(frozen=True)
class CardDefinition:
    name: str
    types: tuple[CardType, ...]
    mana_cost: ManaCost = field(default_factory=ManaCost)
    power: int | None = None
    toughness: int | None = None
    keywords: frozenset[str] = field(default_factory=frozenset)
    has_mana_ability: bool = False

    @property
    def is_land(self) -> bool:
        return CardType.LAND in self.types

    @property
    def is_creature(self) -> bool:
        return CardType.CREATURE in self.types

    @property
    def is_instant(self) -> bool:
        return CardType.INSTANT in self.types


@dataclass(frozen=True)
class CardInstance:
    card_id: str
    definition_name: str
    owner_id: int

