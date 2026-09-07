import json

from harness.site import (
    collect_multi_run_site_data,
    collect_site_data,
    combine_run_summaries,
    expand_problem_ids,
    run_label,
    write_multi_run_site,
    write_site,
)


def test_run_label_uses_friendly_version_names():
    assert run_label("grep-rules") == "grep rules"
    assert run_label("full-rules-in-context") == "full rules in context"
    assert run_label("tools-rules") == "grep rules"
    assert run_label("inline-rules") == "full rules in context"
    assert run_label("smoke-001") == "smoke-001"


def test_expand_problem_ids_supports_ranges_and_padding():
    assert expand_problem_ids("1-3") == ["001", "002", "003"]
    assert expand_problem_ids("1-39,050")[-1] == "050"
    assert expand_problem_ids("1-39")[0] == "001"
    assert expand_problem_ids("1-39")[-1] == "039"
    assert len(expand_problem_ids("1-39")) == 39
    assert expand_problem_ids("10-12,015,1") == ["010", "011", "012", "015", "001"]
    assert expand_problem_ids(["001-003", "5"]) == ["001", "002", "003", "005"]
    assert expand_problem_ids(None) is None


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
    source_url=None,
):
    sample_dir = dataset_dir / sample_id
    sample_dir.mkdir(parents=True)
    (sample_dir / "problem_gold.md").write_text(f"# Puzzle {sample_id}", encoding="utf-8")
    metadata = {"difficulty": difficulty, "solution_text": solution_text}
    if excluded:
        metadata["excluded"] = True
        metadata["excluded_reason"] = "unreliable reference solution"
    if source_url:
        metadata["source_url"] = source_url
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
    assert data["problems"][0]["image"] == "001/puzzle.png"
    assert data["problems"][0]["has_detail"] is True
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
    assert data["details"]["model-a"]["001"]["judge_reasoning"] == "Looks good."
    assert data["details"]["model-a"]["001"]["summary"]["detail_path"] == "data/runs/run/model-a/001.json"
    assert "transcript" not in data["details"]["model-a"]["001"]["summary"]
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
    run_dir = tmp_path / "results" / "grep-rules"
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


def test_collect_site_data_can_hide_problem_details(tmp_path):
    run_dir = tmp_path / "results" / "run"
    dataset_dir = tmp_path / "dataset"
    _write_problem(dataset_dir, "001")
    _write_problem(dataset_dir, "002", difficulty="Rare")
    (dataset_dir / "001" / "puzzle.png").write_bytes(b"png")
    (dataset_dir / "002" / "puzzle.jpg").write_bytes(b"jpg")
    _write_json(run_dir / "model-a" / "001.json", _result("model-a", "001"))
    _write_json(run_dir / "model-a" / "002.json", _result("model-a", "002"))

    data = collect_site_data(run_dir, dataset_root=dataset_dir, detail_ids=["001"])

    by_id = {problem["id"]: problem for problem in data["problems"]}
    assert by_id["001"]["has_detail"] is True
    assert by_id["001"]["problem_gold_md"].startswith("# Puzzle 001")
    assert by_id["001"]["source_url"] is None
    assert by_id["002"]["has_detail"] is False
    assert by_id["002"]["problem_gold_md"] == ""
    assert by_id["002"]["image"] is None
    assert "002" not in data["details"]["model-a"]
    assert data["details"]["model-a"]["001"]["summary"]["detail_path"] == "data/runs/run/model-a/001.json"


def test_write_site_emits_split_data_without_transcripts(tmp_path):
    run_dir = tmp_path / "results" / "run"
    dataset_dir = tmp_path / "dataset"
    _write_problem(dataset_dir, "001", source_url="https://www.possibilitystorm.com/aer1/")
    (dataset_dir / "001" / "puzzle.jpg").write_bytes(b"jpg")
    _write_json(run_dir / "model-a" / "001.json", _result("model-a", "001"))
    _write_json(run_dir / "model-a" / "001.judge.json", _judge("model-a", "001", True))

    site_dir = write_site(run_dir, dataset_root=dataset_dir)

    assert (site_dir / "index.html").exists()
    assert (site_dir / "app.js").exists()
    assert (site_dir / "styles.css").exists()
    assert (site_dir / ".nojekyll").exists()
    assert not (site_dir / "data.json").exists()
    assert (site_dir / "assets" / "problems" / "001.jpg").exists()
    manifest = json.loads((site_dir / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["default_run_id"] == "run"
    assert [run["run_id"] for run in manifest["runs"]] == ["run"]
    problem = json.loads((site_dir / "data" / "problems" / "001.json").read_text(encoding="utf-8"))
    assert problem["image"] == "assets/problems/001.jpg"
    assert problem["source_url"] == "https://www.possibilitystorm.com/aer1/"
    summary = json.loads((site_dir / "data" / "runs" / "run" / "summary.json").read_text(encoding="utf-8"))
    assert "details" not in summary
    assert "problem_gold_md" not in summary["problems"][0]
    detail = json.loads((site_dir / "data" / "runs" / "run" / "model-a" / "001.json").read_text(encoding="utf-8"))
    assert detail["model_solution"] == "1. Win."
    assert "transcript" not in detail
    assert "result" not in detail
    js = (site_dir / "app.js").read_text(encoding="utf-8")
    css = (site_dir / "styles.css").read_text(encoding="utf-8")
    assert "data/manifest.json" in js
    assert "cache: \"no-store\"" in js
    assert "Original problem page" in js
    assert "split-layout" in js
    assert "split-pane" in css
    assert "overflow: hidden" in css
    assert "output tokens" in js
    assert "tool calls" in js
    assert "<th>Tokens</th>" in js
    assert "showVersionColumn" in js
    assert "grep-rules" in js
    assert "runLabel" in js
    assert "<th>Attempted</th>" not in js
    assert "<th>Unjudged</th>" not in js
    assert "unjudged" not in js


def test_collect_multi_run_site_data_includes_all_result_runs(tmp_path):
    results_dir = tmp_path / "results"
    dataset_dir = tmp_path / "dataset"
    _write_problem(dataset_dir, "001")
    (dataset_dir / "001" / "puzzle.png").write_bytes(b"png")
    _write_json(results_dir / "full-rules-in-context" / "model-a" / "001.json", _result("model-a", "001"))
    _write_json(
        results_dir / "full-rules-in-context" / "model-a" / "001.judge.json",
        _judge("model-a", "001", True),
    )
    _write_json(results_dir / "grep-rules" / "model-b" / "001.json", _result("model-b", "001"))
    (results_dir / "site").mkdir()

    data = collect_multi_run_site_data(results_dir, dataset_root=dataset_dir)

    assert [run["run_id"] for run in data["runs"]] == ["full-rules-in-context", "grep-rules"]
    assert [run["label"] for run in data["runs"]] == ["full rules in context", "grep rules"]
    assert data["default_run_id"] == "all-versions"
    assert data["problems"] == ["001"]
    combined = combine_run_summaries(data["runs"])
    assert combined["run_id"] == "all-versions"
    assert combined["leaderboard"][0]["version"] == "full-rules-in-context"
    assert combined["leaderboard"][0]["original_model"] == "model-a"
    assert combined["leaderboard"][0]["name"] == "model-a @ full rules in context"
    assert data["runs"][1]["leaderboard"][0]["unjudged"] == 1
    order = [row["name"] for row in combined["leaderboard"]]
    assert order == [row["model"] for row in combined["problems"][0]["models"]]


def test_write_multi_run_site_emits_split_layout(tmp_path):
    results_dir = tmp_path / "results"
    dataset_dir = tmp_path / "dataset"
    _write_problem(dataset_dir, "001")
    (dataset_dir / "001" / "puzzle.png").write_bytes(b"png")
    _write_json(results_dir / "run-a" / "model-a" / "001.json", _result("model-a", "001"))

    site_dir = write_multi_run_site(results_dir, dataset_root=dataset_dir)

    assert site_dir == results_dir / "site"
    manifest = json.loads((site_dir / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["runs"][0]["run_id"] == "run-a"
    assert not (site_dir / "data" / "runs" / "all-versions").exists()
    problem = json.loads((site_dir / "data" / "problems" / "001.json").read_text(encoding="utf-8"))
    assert problem["image"] == "assets/problems/001.png"
