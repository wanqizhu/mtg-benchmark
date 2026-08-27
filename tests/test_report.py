import json

from harness.report import build_report


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_report_summarizes_models_difficulties_and_costs(tmp_path):
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "model-a" / "001.json",
        {
            "model_name": "model-a",
            "reference": {"difficulty": "Easy"},
            "cost": {"usd": 0.25},
        },
    )
    _write_json(
        run_dir / "model-a" / "001.judge.json",
        {
            "model_name": "model-a",
            "verdict": {"passed": True},
            "cost": {"usd": 0.05},
        },
    )
    _write_json(
        run_dir / "model-b" / "001.json",
        {
            "model_name": "model-b",
            "reference": {"difficulty": "Easy"},
            "cost": {"usd": 0.5},
        },
    )
    _write_json(
        run_dir / "model-b" / "001.judge.json",
        {
            "model_name": "model-b",
            "verdict": {"passed": False},
            "cost": {"usd": 0.1},
        },
    )

    report = build_report(run_dir)

    assert report["has_judges"] is True
    assert report["by_model"] == [
        {"name": "model-a", "passed": 1, "total": 1, "rate": 1.0},
        {"name": "model-b", "passed": 0, "total": 1, "rate": 0.0},
    ]
    assert report["by_difficulty"] == [{"name": "Easy", "passed": 1, "total": 2, "rate": 0.5}]
    assert round(report["cost"]["rollouts_usd"], 2) == 0.75
    assert round(report["cost"]["judges_usd"], 2) == 0.15
    assert round(report["cost"]["total_usd"], 2) == 0.9


def test_build_report_handles_runs_without_judges(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    report = build_report(run_dir)

    assert report["has_judges"] is False
    assert report["by_model"] == []
