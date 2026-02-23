from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeckComposition:
    mountains: int
    bolts: int
    goblins: int

    @property
    def size(self) -> int:
        return self.mountains + self.bolts + self.goblins

    def to_deck(self) -> list[str]:
        return ["Mountain"] * self.mountains + ["Lightning Bolt"] * self.bolts + ["Raging Goblin"] * self.goblins


def enumerate_compositions(total_cards: int, min_each: int = 0) -> list[DeckComposition]:
    compositions: list[DeckComposition] = []
    for mountains in range(min_each, total_cards + 1):
        for bolts in range(min_each, total_cards - mountains + 1):
            goblins = total_cards - mountains - bolts
            if goblins < min_each:
                continue
            compositions.append(DeckComposition(mountains=mountains, bolts=bolts, goblins=goblins))
    return compositions

