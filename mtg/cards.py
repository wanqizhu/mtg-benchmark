from __future__ import annotations
from dataclasses import dataclass, field
from .enums import CardType, AbilityKeyword


@dataclass(frozen=True)
class ManaCost:
    generic: int = 0
    white: int = 0
    blue: int = 0
    black: int = 0
    red: int = 0
    green: int = 0

    @property
    def cmc(self) -> int:
        return self.generic + self.white + self.blue + self.black + self.red + self.green

    def can_pay(self, mana_available: dict[str, int]) -> bool:
        colored = {
            "white": self.white, "blue": self.blue, "black": self.black,
            "red": self.red, "green": self.green,
        }
        remaining = 0
        for color, needed in colored.items():
            avail = mana_available.get(color, 0)
            if avail < needed:
                return False
            remaining += avail - needed
        remaining += mana_available.get("generic", 0)
        return remaining >= self.generic


@dataclass(frozen=True)
class CardDef:
    name: str
    card_types: frozenset[CardType]
    mana_cost: ManaCost = field(default_factory=ManaCost)
    power: int | None = None
    toughness: int | None = None
    keywords: frozenset[AbilityKeyword] = frozenset()
    supertype_basic: bool = False

    @property
    def is_land(self) -> bool:
        return CardType.LAND in self.card_types

    @property
    def is_creature(self) -> bool:
        return CardType.CREATURE in self.card_types

    @property
    def is_instant(self) -> bool:
        return CardType.INSTANT in self.card_types

    @property
    def is_sorcery(self) -> bool:
        return CardType.SORCERY in self.card_types

    @property
    def has_haste(self) -> bool:
        return AbilityKeyword.HASTE in self.keywords


MOUNTAIN = CardDef(
    name="Mountain",
    card_types=frozenset({CardType.LAND}),
    supertype_basic=True,
)

LIGHTNING_BOLT = CardDef(
    name="Lightning Bolt",
    card_types=frozenset({CardType.INSTANT}),
    mana_cost=ManaCost(red=1),
)

RAGING_GOBLIN = CardDef(
    name="Raging Goblin",
    card_types=frozenset({CardType.CREATURE}),
    mana_cost=ManaCost(red=1),
    power=1,
    toughness=1,
    keywords=frozenset({AbilityKeyword.HASTE}),
)

CARD_REGISTRY: dict[str, CardDef] = {
    "Mountain": MOUNTAIN,
    "Lightning Bolt": LIGHTNING_BOLT,
    "Raging Goblin": RAGING_GOBLIN,
}


@dataclass
class CardInstance:
    card_def: CardDef
    instance_id: int
    owner: int
    tapped: bool = False
    summoning_sick: bool = False
    damage_marked: int = 0

    @property
    def name(self) -> str:
        return self.card_def.name

    @property
    def is_land(self) -> bool:
        return self.card_def.is_land

    @property
    def is_creature(self) -> bool:
        return self.card_def.is_creature

    @property
    def is_instant(self) -> bool:
        return self.card_def.is_instant

    @property
    def has_haste(self) -> bool:
        return self.card_def.has_haste

    @property
    def power(self) -> int:
        return self.card_def.power

    @property
    def toughness(self) -> int:
        return self.card_def.toughness

    def can_attack(self) -> bool:
        if not self.is_creature or self.tapped:
            return False
        if self.summoning_sick and not self.has_haste:
            return False
        return True

    def can_block(self) -> bool:
        return self.is_creature and not self.tapped

    def __repr__(self) -> str:
        parts = [self.name]
        if self.tapped:
            parts.append("(T)")
        if self.is_creature:
            parts.append(f"{self.power}/{self.toughness}")
            if self.damage_marked > 0:
                parts.append(f"[{self.damage_marked} dmg]")
        return " ".join(parts)
