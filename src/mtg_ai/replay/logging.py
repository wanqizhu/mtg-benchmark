from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mtg_ai.core.events import GameEvent
from mtg_ai.core.serialization import action_to_dict, serialize_state
from mtg_ai.core.state import GameState


class ReplayLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record_state(self, state: GameState, note: str = "") -> None:
        self.records.append(
            {
                "kind": "state",
                "note": note,
                "state": serialize_state(state),
            }
        )

    def record_action(self, action_dict: dict[str, Any]) -> None:
        self.records.append({"kind": "action", "action": action_dict})

    def record_events(self, events: list[GameEvent]) -> None:
        for event in events:
            self.records.append({"kind": "event", "event": {"index": event.index, "kind": event.kind, "payload": event.payload}})

    def record_action_obj(self, action: object) -> None:
        self.record_action(action_to_dict(action))  # type: ignore[arg-type]

    def save_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record, sort_keys=True))
                f.write("\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

