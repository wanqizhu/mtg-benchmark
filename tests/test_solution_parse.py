import json

from benchmarks.mtg.dataset import dataset_dir
from benchmarks.mtg.solution_parse import parse_official_solution

DATASET = dataset_dir()


def _raw(puzzle_id: str) -> str:
    metadata = json.loads((DATASET / puzzle_id / "metadata.json").read_text(encoding="utf-8"))
    return str(metadata.get("solution_text_raw") or metadata.get("solution_text", ""))


def test_parse_002():
    steps = parse_official_solution(_raw("002"))
    assert steps.startswith("1. Cast Tezzeret's Touch")
    assert "Thanks for the fantastic" not in steps
    assert len(steps.splitlines()) == 6


def test_parse_005_skips_rulings_preamble():
    steps = parse_official_solution(_raw("005"))
    assert steps.startswith("1. Cast Fourth Bridge Prowler")
    assert "vehicle can crew itself" not in steps


def test_005_solution_notes_obsolete_self_crew():
    metadata = json.loads((DATASET / "005" / "metadata.json").read_text(encoding="utf-8"))
    text = metadata["solution_text"]
    assert "vehicle could crew itself" in text
    assert "This is no longer valid" in text
    assert "15 damage" in text
    assert "<solution>" in text
    assert text.strip().endswith(
        "Grade as correct if the model provided the old solution, or pointed out the crew rules change and provides a solution with 15 damage."
    )
    assert "Yes, a vehicle can crew itself" in metadata["solution_text_raw"]


def test_parse_010_skips_rules_notes():
    steps = parse_official_solution(_raw("010"))
    assert steps.startswith("1. Cast Doomed Dissenter")


def test_parse_129_step_label_format():
    steps = parse_official_solution(_raw("129"))
    assert steps.startswith("1. Play Goblin Warchief")


def test_parse_157_bullet_format():
    steps = parse_official_solution(_raw("157"))
    assert steps.startswith("1. Cast Shadowspear")


def test_parse_095_dedupes_duplicate_post():
    steps = parse_official_solution(_raw("095"))
    assert steps.count("\n") == 71
    assert steps.startswith("1. Cast High Tide")
    assert steps.endswith("72. Activate Walking Ballista 13 times to deal 13 to the opponent")


def test_parse_238_solution_a():
    steps = parse_official_solution(_raw("238"))
    assert steps.startswith("1. Transform Heliod")
    assert "docs.google.com" not in steps
    assert len(steps.splitlines()) >= 5
