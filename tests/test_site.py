import json

from harness.site import collect_multi_run_site_data, collect_site_data, write_multi_run_site, write_site


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_problem(
    dataset_dir,
    sample_id,
    *,
    difficulty="Common",
    solution_text="1. Win the game.",
    excluded=False,
):
    sample_dir = dataset_dir / sample_id
    sample_dir.mkdir(parents=True)
    (sample_dir / "problem_gold.md").write_text(f"# Puzzle {sample_id}", encoding="utf-8")
    metadata = {"difficulty": difficulty, "solution_text": solution_text}
    if excluded:
        metadata["excluded"] = True
        metadata["excluded_reason"] = "unreliable reference solution"
    (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _result(model_name, sample_id, *, status="complete"):
    return {
        "benchmark": "mtg",
        "run_id": "run",
        "status": status,
        "model_name": model_name,
        "model_id": f"{model_name}-id",
        "model_config": {"max_tokens": 4096},
        "sample_id": sample_id,
        "reference": {
            "difficulty": "Common",
            "problem_gold_md": f"# Puzzle {sample_id}",
            "solution_text": "1. Win the game.",
        },
        "transcript": {
            "turns": [
                {"role": "user", "text": "Solve it."},
                {"role": "assistant", "text": "<solution>\n1. Win.\n</solution>"},
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2},
        },
        "cost": {"usd": 0.25},
    }


def _judge(model_name, sample_id, passed):
    return {
        "benchmark": "mtg",
        "run_id": "run",
        "sample_id": sample_id,
        "model_name": model_name,
        "verdict": {
            "passed": passed,
            "reasoning": "Looks good." if passed else "Not enough.",
            "prompt": {"system": "Judge.", "user": "Grade."},
        },
        "cost": {"usd": 0.05},
    }


def test_collect_site_data_normalizes_models_problems_and_details(tmp_path):
    run_dir = tmp_path / "results" / "run"
    dataset_dir = tmp_path / "dataset"
    _write_problem(dataset_dir, "001")
    _write_problem(dataset_dir, "002", difficulty="Rare", solution_text="1. No rollout yet.")
    (dataset_dir / "001" / "puzzle.png").write_bytes(b"png")
    (dataset_dir / "002" / "puzzle.jpg").write_bytes(b"jpg")
    _write_json(run_dir / "model-a" / "001.json", _result("model-a", "001"))
    _write_json(run_dir / "model-a" / "001.judge.json", _judge("model-a", "001", True))
    _write_json(run_dir / "model-b" / "001.json", _result("model-b", "001", status="failed"))
    _write_json(run_dir / "model-a" / "001.judge.haiku.json", {"model_name": "model-a", "sample_id": "001"})

    data = collect_site_data(run_dir, dataset_root=dataset_dir)

    assert data["run_id"] == "run"
    assert [row["name"] for row in data["leaderboard"]] == ["model-a", "model-b"]
    assert data["leaderboard"][0]["passed"] == 1
    assert data["leaderboard"][1]["unjudged"] == 1
    assert data["leaderboard"][0]["output_tokens"] == 2
    assert data["leaderboard"][0]["tool_call_count"] == 0
    assert data["leaderboard"][1]["output_tokens"] == 2
    assert data["problems"][0]["image"].endswith("001/puzzle.png")
    assert data["problems"][0]["attempted"] == 2
    assert len(data["problems"][0]["models"]) == 2
    assert data["problems"][1]["id"] == "002"
    assert data["problems"][1]["attempted"] == 0
    assert data["problems"][1]["difficulty"] == "Rare"
    assert [row["model"] for row in data["problems"][0]["models"]] == ["model-a", "model-b"]
    assert [row["model"] for row in data["problems"][1]["models"]] == ["model-a", "model-b"]
    assert data["problems"][1]["models"][0]["status"] == "not_attempted"
    assert data["problems"][1]["models"][1]["status"] == "not_attempted"
    assert data["details"]["model-a"]["001"]["model_solution"] == "1. Win."
    assert data["details"]["model-b"]["001"]["passed"] is None
    assert data["problems"][0]["models"][0]["output_tokens"] == 2
    assert data["problems"][0]["models"][0]["tool_call_count"] == 0


def test_collect_site_data_skips_excluded_problems(tmp_path):
    run_dir = tmp_path / "results" / "run"
    dataset_dir = tmp_path / "dataset"
    _write_problem(dataset_dir, "001")
    _write_problem(dataset_dir, "018", difficulty="Special", excluded=True)
    (dataset_dir / "001" / "puzzle.png").write_bytes(b"png")
    (dataset_dir / "018" / "puzzle.jpg").write_bytes(b"jpg")
    _write_json(run_dir / "model-a" / "001.json", _result("model-a", "001"))

    data = collect_site_data(run_dir, dataset_root=dataset_dir)

    assert [problem["id"] for problem in data["problems"]] == ["001"]


def test_collect_site_data_uses_output_tokens_and_tool_calls(tmp_path):
    run_dir = tmp_path / "results" / "tools-rules"
    dataset_dir = tmp_path / "dataset"
    _write_problem(dataset_dir, "001")
    (dataset_dir / "001" / "puzzle.png").write_bytes(b"png")
    payload = _result("model-a", "001")
    payload["transcript"]["tool_call_count"] = 4
    payload["transcript"]["usage"]["output_tokens"] = 12345
    _write_json(run_dir / "model-a" / "001.json", payload)

    data = collect_site_data(run_dir, dataset_root=dataset_dir)

    row = data["problems"][0]["models"][0]
    assert row["output_tokens"] == 12345
    assert row["tool_call_count"] == 4
    assert data["leaderboard"][0]["output_tokens"] == 12345
    assert data["leaderboard"][0]["tool_call_count"] == 4


def test_write_site_emits_static_assets_and_copies_images(tmp_path):
    run_dir = tmp_path / "results" / "run"
    dataset_dir = tmp_path / "dataset"
    _write_problem(dataset_dir, "001")
    (dataset_dir / "001" / "puzzle.jpg").write_bytes(b"jpg")
    _write_json(run_dir / "model-a" / "001.json", _result("model-a", "001"))
    _write_json(run_dir / "model-a" / "001.judge.json", _judge("model-a", "001", True))

    site_dir = write_site(run_dir, dataset_root=dataset_dir)

    assert (site_dir / "index.html").exists()
    assert (site_dir / "app.js").exists()
    assert (site_dir / "styles.css").exists()
    assert (site_dir / "data.json").exists()
    assert (site_dir / "assets" / "problems" / "001.jpg").exists()
    data = json.loads((site_dir / "data.json").read_text(encoding="utf-8"))
    assert data["problems"][0]["image"] == "assets/problems/001.jpg"
    js = (site_dir / "app.js").read_text(encoding="utf-8")
    css = (site_dir / "styles.css").read_text(encoding="utf-8")
    assert "location.hash" in js
    assert "split-layout" in js
    assert "split-pane" in css
    assert "overflow: hidden" in css
    assert "output tokens" in js
    assert "tool calls" in js
    assert "<th>Tokens</th>" in js
    assert "<th>Attempted</th>" not in js
    assert "<th>Unjudged</th>" not in js


def test_collect_multi_run_site_data_includes_all_result_runs(tmp_path):
    results_dir = tmp_path / "results"
    dataset_dir = tmp_path / "dataset"
    _write_problem(dataset_dir, "001")
    (dataset_dir / "001" / "puzzle.png").write_bytes(b"png")
    _write_json(results_dir / "inline-rules" / "model-a" / "001.json", _result("model-a", "001"))
    _write_json(
        results_dir / "inline-rules" / "model-a" / "001.judge.json",
        _judge("model-a", "001", True),
    )
    _write_json(results_dir / "tools-rules" / "model-b" / "001.json", _result("model-b", "001"))
    (results_dir / "site").mkdir()

    data = collect_multi_run_site_data(results_dir, dataset_root=dataset_dir)

    assert [run["run_id"] for run in data["runs"]] == ["all-versions", "inline-rules", "tools-rules"]
    assert data["default_run_id"] == "all-versions"
    assert data["runs"][0]["label"] == "All versions"
    assert data["runs"][0]["leaderboard"][0]["version"] == "inline-rules"
    assert data["runs"][0]["leaderboard"][0]["original_model"] == "model-a"
    assert data["runs"][2]["leaderboard"][0]["unjudged"] == 1
    combined = data["runs"][0]
    order = [row["name"] for row in combined["leaderboard"]]
    assert order == [row["model"] for row in combined["problems"][0]["models"]]


def test_write_multi_run_site_emits_dropdown_data_shape(tmp_path):
    results_dir = tmp_path / "results"
    dataset_dir = tmp_path / "dataset"
    _write_problem(dataset_dir, "001")
    (dataset_dir / "001" / "puzzle.png").write_bytes(b"png")
    _write_json(results_dir / "run-a" / "model-a" / "001.json", _result("model-a", "001"))

    site_dir = write_multi_run_site(results_dir, dataset_root=dataset_dir)

    assert site_dir == results_dir / "site"
    data = json.loads((site_dir / "data.json").read_text(encoding="utf-8"))
    assert data["runs"][0]["run_id"] == "run-a"
    assert data["runs"][0]["problems"][0]["image"] == "assets/problems/001.png"
