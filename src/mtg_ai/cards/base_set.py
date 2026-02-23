from __future__ import annotations

from mtg_ai.core.cards import CardDefinition, ManaCost
from mtg_ai.core.enums import CardType

MOUNTAIN = CardDefinition(
    name="Mountain",
    types=(CardType.LAND,),
    mana_cost=ManaCost(red=0),
    has_mana_ability=True,
)

LIGHTNING_BOLT = CardDefinition(
    name="Lightning Bolt",
    types=(CardType.INSTANT,),
    mana_cost=ManaCost(red=1),
)

RAGING_GOBLIN = CardDefinition(
    name="Raging Goblin",
    types=(CardType.CREATURE,),
    mana_cost=ManaCost(red=1),
    power=1,
    toughness=1,
    keywords=frozenset({"haste"}),
)

BASE_SET_DEFINITIONS = {
    MOUNTAIN.name: MOUNTAIN,
    LIGHTNING_BOLT.name: LIGHTNING_BOLT,
    RAGING_GOBLIN.name: RAGING_GOBLIN,
}

