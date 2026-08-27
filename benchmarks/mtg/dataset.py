from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATASET_DIR = REPO_ROOT / "datasets" / "mtg"
MANUAL_SOLUTIONS_PATH = REPO_ROOT / "datasets" / "manual_solutions.json"


@dataclass(frozen=True)
class Puzzle:
    id: str
    gold_md: str
    image_path: Path
    difficulty: str
    solution_text: str


def dataset_dir() -> Path:
    env = os.getenv("MTG_DATASET_DIR", "").strip()
    if env:
        path = Path(env).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path
    return LOCAL_DATASET_DIR


def load_manual_overrides(path: Path | None = None) -> dict[str, str]:
    overrides_path = MANUAL_SOLUTIONS_PATH if path is None else path
    if not overrides_path.exists():
        return {}
    data = json.loads(overrides_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manual solutions file must be a JSON object: {overrides_path}")
    return {str(key): str(value) for key, value in data.items() if str(value).strip()}


def rules_path() -> Path:
    return dataset_dir() / "common" / "comp_rules.txt"


def clarifications_path() -> Path:
    return dataset_dir() / "common" / "clarifications.txt"


def load_rules_text() -> str:
    return rules_path().read_text(encoding="utf-8")


def load_puzzles(*, puzzle_ids: list[str] | None = None, include_excluded: bool = False) -> list[Puzzle]:
    root = dataset_dir()
    puzzles: list[Puzzle] = []

    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not entry.name.isdigit():
            continue
        gold_path = entry / "problem_gold.md"
        if not gold_path.exists():
            continue
        puzzle_id = entry.name
        if puzzle_ids and puzzle_id not in puzzle_ids:
            continue

        metadata_path = entry / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if metadata.get("excluded") and not include_excluded:
            continue
        image_path = next(
            (entry / name for name in ("puzzle.png", "puzzle.jpg") if (entry / name).exists()),
            None,
        )
        if image_path is None:
            # iCloud placeholders (.puzzle.png.icloud) are not real images.
            continue

        puzzles.append(
            Puzzle(
                id=puzzle_id,
                gold_md=gold_path.read_text(encoding="utf-8"),
                image_path=image_path,
                difficulty=metadata.get("difficulty", "Unknown"),
                solution_text=metadata.get("solution_text", ""),
            )
        )

    return puzzles
