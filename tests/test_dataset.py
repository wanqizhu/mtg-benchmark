import json
from pathlib import Path

from benchmarks.mtg import dataset as dataset_mod


def test_env_absolute_path(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_mod, "LOCAL_DATASET_DIR", tmp_path / "missing")
    monkeypatch.setenv("MTG_DATASET_DIR", str(tmp_path / "custom"))
    assert dataset_mod.dataset_dir() == tmp_path / "custom"


def test_defaults_to_local_repo_dataset(monkeypatch, tmp_path):
    local = tmp_path / "datasets" / "mtg"
    local.mkdir(parents=True)
    monkeypatch.setattr(dataset_mod, "LOCAL_DATASET_DIR", local)
    monkeypatch.delenv("MTG_DATASET_DIR", raising=False)
    assert dataset_mod.dataset_dir() == local


def _write_puzzle(root: Path, puzzle_id: str, **metadata) -> None:
    entry = root / puzzle_id
    entry.mkdir(parents=True)
    (entry / "problem_gold.md").write_text(f"# Puzzle {puzzle_id}", encoding="utf-8")
    (entry / "puzzle.jpg").write_bytes(b"")
    (entry / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_excluded_puzzles_are_skipped(monkeypatch, tmp_path):
    _write_puzzle(tmp_path, "001")
    _write_puzzle(tmp_path, "018", excluded=True, excluded_reason="unreliable reference solution")
    monkeypatch.setenv("MTG_DATASET_DIR", str(tmp_path))

    assert [p.id for p in dataset_mod.load_puzzles()] == ["001"]
    assert [p.id for p in dataset_mod.load_puzzles(include_excluded=True)] == ["001", "018"]
    # An explicit id request must not smuggle an excluded puzzle back into a run.
    assert dataset_mod.load_puzzles(puzzle_ids=["018"]) == []


def test_manual_overrides_missing_file_are_empty(tmp_path):
    assert dataset_mod.load_manual_overrides(tmp_path / "missing.json") == {}


def test_manual_overrides_load_json(tmp_path):
    path = tmp_path / "manual_solutions.json"
    path.write_text(json.dumps({"159": "1. Win."}), encoding="utf-8")
    assert dataset_mod.load_manual_overrides(path) == {"159": "1. Win."}
