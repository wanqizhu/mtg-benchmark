from __future__ import annotations

import argparse
import json

from mtg_ai.replay.logging import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay logged MTG games.")
    parser.add_argument("path")
    parser.add_argument("--show-state", action="store_true")
    args = parser.parse_args()

    records = load_jsonl(args.path)
    for idx, record in enumerate(records):
        kind = record["kind"]
        if kind == "action":
            print(f"[{idx}] ACTION {record['action']}")
        elif kind == "event":
            print(f"[{idx}] EVENT {record['event']['kind']} {json.dumps(record['event']['payload'], sort_keys=True)}")
        elif kind == "state":
            if args.show_state:
                print(f"[{idx}] STATE {json.dumps(record['state'], sort_keys=True)}")
            else:
                step = record["state"]["turn"]["step"]
                turn_number = record["state"]["turn"]["turn_number"]
                print(f"[{idx}] STATE turn={turn_number} step={step}")


if __name__ == "__main__":
    main()

