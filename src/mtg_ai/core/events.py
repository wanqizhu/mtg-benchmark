from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GameEvent:
    index: int
    kind: str
    payload: dict[str, object] = field(default_factory=dict)

