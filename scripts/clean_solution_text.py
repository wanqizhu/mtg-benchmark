#!/usr/bin/env python3
"""Copy solution_text to solution_text_raw, then replace solution_text with parsed steps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from benchmarks.mtg.dataset import MANUAL_SOLUTIONS_PATH, dataset_dir, load_manual_overrides
from benchmarks.mtg.solution_parse import parse_official_solution


def dataset_root() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return dataset_dir()


def clean_metadata(
    metadata: dict[str, object],
    *,
    puzzle_id: str,
    overrides: dict[str, str] | None = None,
) -> tuple[dict[str, object], bool]:
    if overrides is None:
        overrides = load_manual_overrides()

    changed = False
    raw = metadata.get("solution_text_raw")
    if raw is None:
        raw = metadata.get("solution_text", "")
        metadata["solution_text_raw"] = raw
        changed = True

    parsed = overrides.get(puzzle_id) or parse_official_solution(str(raw))
    if metadata.get("solution_text") != parsed:
        metadata["solution_text"] = parsed
        changed = True

    return metadata, changed


def main() -> None:
    root = dataset_root()
    overrides = load_manual_overrides()
    if overrides:
        print(f"Manual overrides: {len(overrides)} from {MANUAL_SOLUTIONS_PATH}")
    else:
        print(f"Manual overrides: none ({MANUAL_SOLUTIONS_PATH} not found)")

    updated = 0
    empty_after_parse = 0
    total_with_raw = 0

    for entry in sorted(root.iterdir()):
        metadata_path = entry / "metadata.json"
        if not metadata_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        puzzle_id = str(metadata.get("puzzle_id", entry.name))
        raw = str(metadata.get("solution_text_raw") or metadata.get("solution_text", ""))
        if raw.strip():
            total_with_raw += 1

        metadata, changed = clean_metadata(metadata, puzzle_id=puzzle_id, overrides=overrides)
        if not str(metadata.get("solution_text", "")).strip() and raw.strip():
            empty_after_parse += 1

        if changed:
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            updated += 1

    print(f"Updated {updated} metadata file(s)")
    print(f"Non-empty raw: {total_with_raw}")
    print(f"Non-empty raw but empty parsed: {empty_after_parse}")


if __name__ == "__main__":
    main()
