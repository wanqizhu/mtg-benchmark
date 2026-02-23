from __future__ import annotations

from mtg_ai.cards.base_set import BASE_SET_DEFINITIONS
from mtg_ai.core.cards import CardDefinition


class CardRegistry:
    def __init__(self, definitions: dict[str, CardDefinition] | None = None) -> None:
        self._definitions = dict(definitions or BASE_SET_DEFINITIONS)

    def get(self, card_name: str) -> CardDefinition:
        return self._definitions[card_name]

    def has(self, card_name: str) -> bool:
        return card_name in self._definitions

    def all_cards(self) -> dict[str, CardDefinition]:
        return dict(self._definitions)

