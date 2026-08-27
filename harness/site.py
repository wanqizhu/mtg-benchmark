from __future__ import annotations

import json
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.mtg.dataset import dataset_dir as default_dataset_dir
from benchmarks.mtg.solution import parse_solution

SITE_ASSET_DIR = "assets/problems"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _result_paths(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        return []
    paths: list[Path] = []
    for model_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name != "site"):
        paths.extend(
            sorted(
                p
                for p in model_dir.glob("*.json")
                if p.stem.isdigit()
            )
        )
    return paths


def _judge_path(result_path: Path) -> Path:
    return result_path.with_name(f"{result_path.stem}.judge.json")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _cost_usd(payload: dict[str, Any] | None) -> float:
    if not payload:
        return 0.0
    return _safe_float(payload.get("cost", {}).get("usd"))


def _verdict(judge: dict[str, Any] | None) -> dict[str, Any] | None:
    if not judge:
        return None
    verdict = judge.get("verdict")
    return verdict if isinstance(verdict, dict) else None


def _passed(judge: dict[str, Any] | None) -> bool | None:
    verdict = _verdict(judge)
    if verdict is None or "passed" not in verdict:
        return None
    return bool(verdict["passed"])


def _assistant_text(turn: dict[str, Any]) -> str:
    text = turn.get("text")
    if isinstance(text, str) and text.strip():
        return text
    content = turn.get("content")
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
    ).strip()


def _model_solution(transcript: dict[str, Any]) -> str | None:
    turns = transcript.get("turns", [])
    if not isinstance(turns, list):
        return None
    for turn in reversed(turns):
        if isinstance(turn, dict) and turn.get("role") == "assistant":
            text = _assistant_text(turn)
            parsed = parse_solution(text)
            if parsed:
                return parsed
            if text:
                return text
    return None


def _usage_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _output_tokens(transcript: dict[str, Any]) -> int:
    usage = _usage_dict(transcript.get("usage"))
    if usage.get("output_tokens") is not None:
        return int(usage["output_tokens"])
    total = 0
    for turn in transcript.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        turn_usage = _usage_dict(turn.get("usage"))
        if turn_usage.get("output_tokens") is not None:
            total += int(turn_usage["output_tokens"])
    return total


def _tool_call_count(transcript: dict[str, Any]) -> int:
    raw = transcript.get("tool_call_count")
    if isinstance(raw, int):
        return raw
    count = 0
    for turn in transcript.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        calls = turn.get("tool_calls")
        if isinstance(calls, list):
            count += len(calls)
        content = turn.get("content")
        if isinstance(content, list):
            count += sum(1 for block in content if isinstance(block, dict) and block.get("type") == "tool_use")
    return count


def _image_source(dataset_root: Path, sample_id: str) -> Path | None:
    sample_dir = dataset_root / sample_id
    for name in ("puzzle.png", "puzzle.jpg", "puzzle.jpeg", "puzzle.webp"):
        path = sample_dir / name
        if path.exists():
            return path
    return None


def _copy_image(dataset_root: Path, out_dir: Path, sample_id: str) -> str | None:
    source = _image_source(dataset_root, sample_id)
    if source is None:
        return None
    target_dir = out_dir / SITE_ASSET_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{sample_id}{source.suffix.lower()}"
    shutil.copy2(source, target)
    return f"{SITE_ASSET_DIR}/{target.name}"


def _problem_image(dataset_root: Path, site_dir: Path, sample_id: str, *, copy_images: bool) -> str | None:
    if copy_images:
        return _copy_image(dataset_root, site_dir, sample_id)
    image_source = _image_source(dataset_root, sample_id)
    return str(image_source) if image_source else None


def _dataset_problems(dataset_root: Path, site_dir: Path, *, copy_images: bool) -> dict[str, dict[str, Any]]:
    if not dataset_root.exists():
        return {}

    problems: dict[str, dict[str, Any]] = {}
    for entry in sorted(dataset_root.iterdir()):
        if not entry.is_dir() or not entry.name.isdigit():
            continue
        gold_path = entry / "problem_gold.md"
        if not gold_path.exists():
            continue

        metadata_path = entry / "metadata.json"
        metadata = _read_json(metadata_path) if metadata_path.exists() else {}
        sample_id = entry.name
        problems[sample_id] = {
            "id": sample_id,
            "difficulty": metadata.get("difficulty", "Unknown"),
            "problem_gold_md": gold_path.read_text(encoding="utf-8"),
            "solution_text": metadata.get("solution_text", ""),
            "image": _problem_image(dataset_root, site_dir, sample_id, copy_images=copy_images),
            "models": [],
        }
    return problems


def _has_results(run_dir: Path) -> bool:
    return bool(_result_paths(run_dir))


def _available_run_dirs(results_dir: Path) -> list[Path]:
    return sorted(
        (path for path in results_dir.iterdir() if path.is_dir() and path.name != "site" and _has_results(path)),
        key=lambda path: path.name,
    )


def _model_version_name(model_name: str, run_id: str) -> str:
    return f"{model_name} @ {run_id}"


def _sample_sort_key(sample_id: str) -> tuple[int, str]:
    return (int(sample_id), sample_id) if sample_id.isdigit() else (10**9, sample_id)


def _model_row(name: str, entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    judged = [entry for entry in entries.values() if entry["passed"] is not None]
    passed = sum(1 for entry in judged if entry["passed"])
    rollout_cost = sum(_cost_usd(entry["result"]) for entry in entries.values())
    judge_cost = sum(_cost_usd(entry["judge"]) for entry in entries.values())
    first_result = next((entry["result"] for entry in entries.values() if entry["result"]), {})
    total = len(judged)
    return {
        "name": name,
        "model_id": first_result.get("model_id", ""),
        "model_config": first_result.get("model_config", {}),
        "attempted": len(entries),
        "judged": total,
        "passed": passed,
        "failed": total - passed,
        "unjudged": len(entries) - total,
        "pass_rate": passed / total if total else 0.0,
        "rollout_cost_usd": rollout_cost,
        "judge_cost_usd": judge_cost,
        "total_cost_usd": rollout_cost + judge_cost,
    }


def collect_site_data(
    run_dir: Path,
    *,
    dataset_root: Path | None = None,
    site_dir: Path | None = None,
    copy_images: bool = False,
) -> dict[str, Any]:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    dataset_root = dataset_root or default_dataset_dir()
    site_dir = site_dir or run_dir / "site"

    entries_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    problems = _dataset_problems(dataset_root, site_dir, copy_images=copy_images)
    details: dict[str, dict[str, Any]] = {}

    for result_path in _result_paths(run_dir):
        result = _read_json(result_path)
        judge_path = _judge_path(result_path)
        judge = _read_json(judge_path) if judge_path.exists() else None
        sample_id = str(result.get("sample_id") or result_path.stem)
        model_name = str(result.get("model_name") or result_path.parent.name)
        reference = result.get("reference", {})
        if not isinstance(reference, dict):
            reference = {}

        passed = _passed(judge)
        image = _problem_image(dataset_root, site_dir, sample_id, copy_images=copy_images)

        problem = problems.setdefault(
            sample_id,
            {
                "id": sample_id,
                "difficulty": reference.get("difficulty", "Unknown"),
                "problem_gold_md": reference.get("problem_gold_md", ""),
                "solution_text": reference.get("solution_text", ""),
                "image": image,
                "models": [],
            },
        )
        if not problem.get("image") and image:
            problem["image"] = image
        if not problem.get("problem_gold_md") and reference.get("problem_gold_md"):
            problem["problem_gold_md"] = reference["problem_gold_md"]
        if not problem.get("solution_text") and reference.get("solution_text"):
            problem["solution_text"] = reference["solution_text"]
        if problem.get("difficulty") == "Unknown" and reference.get("difficulty"):
            problem["difficulty"] = reference["difficulty"]

        summary = {
            "model": model_name,
            "sample_id": sample_id,
            "status": result.get("status", "unknown"),
            "passed": passed,
            "judged": judge is not None,
            "output_tokens": _output_tokens(result.get("transcript") or {}),
            "tool_call_count": _tool_call_count(result.get("transcript") or {}),
            "rollout_cost_usd": _cost_usd(result),
            "judge_cost_usd": _cost_usd(judge),
            "updated_at": result.get("updated_at"),
        }
        problem["models"].append(summary)

        entry = {
            "result_path": str(result_path),
            "judge_path": str(judge_path) if judge else None,
            "result": result,
            "judge": judge,
            "passed": passed,
            "model_solution": _model_solution(result.get("transcript", {})),
            "summary": summary,
        }
        entries_by_model.setdefault(model_name, {})[sample_id] = entry
        details.setdefault(model_name, {})[sample_id] = entry

    leaderboard = sorted(
        (_model_row(model_name, entries) for model_name, entries in entries_by_model.items()),
        key=lambda row: (-row["pass_rate"], -row["passed"], row["name"]),
    )
    for rank, row in enumerate(leaderboard, start=1):
        row["rank"] = rank

    problem_rows = []
    for problem in problems.values():
        problem["models"] = sorted(problem["models"], key=lambda row: row["model"])
        judged = [row for row in problem["models"] if row["passed"] is not None]
        problem["judged"] = len(judged)
        problem["passed"] = sum(1 for row in judged if row["passed"])
        problem["attempted"] = len(problem["models"])
        problem_rows.append(problem)
    problem_rows.sort(key=lambda row: _sample_sort_key(row["id"]))

    model_rows = []
    for model in leaderboard:
        entries = entries_by_model[model["name"]]
        model_rows.append(
            {
                **model,
                "problems": [
                    entries[sample_id]["summary"] for sample_id in sorted(entries, key=_sample_sort_key)
                ],
            }
        )

    return {
        "run_id": run_dir.name,
        "label": run_dir.name,
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "leaderboard": leaderboard,
        "problems": problem_rows,
        "models": model_rows,
        "details": details,
    }


def _combined_site_data(runs: list[dict[str, Any]], *, results_dir: Path, dataset_root: Path) -> dict[str, Any] | None:
    if len(runs) < 2:
        return None

    leaderboard: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    problems_by_id: dict[str, dict[str, Any]] = {}

    for run in runs:
        run_id = str(run["run_id"])
        for row in run["leaderboard"]:
            display_name = _model_version_name(str(row["name"]), run_id)
            versioned_row = deepcopy(row)
            versioned_row["name"] = display_name
            versioned_row["original_model"] = row["name"]
            versioned_row["version"] = run_id
            leaderboard.append(versioned_row)

        for model in run["models"]:
            display_name = _model_version_name(str(model["name"]), run_id)
            versioned_model = deepcopy(model)
            versioned_model["name"] = display_name
            versioned_model["original_model"] = model["name"]
            versioned_model["version"] = run_id
            for summary in versioned_model.get("problems", []):
                summary["model"] = display_name
                summary["original_model"] = model["name"]
                summary["version"] = run_id
            models.append(versioned_model)

        for problem in run["problems"]:
            sample_id = str(problem["id"])
            combined_problem = problems_by_id.setdefault(
                sample_id,
                {
                    "id": sample_id,
                    "difficulty": problem.get("difficulty", "Unknown"),
                    "problem_gold_md": problem.get("problem_gold_md", ""),
                    "solution_text": problem.get("solution_text", ""),
                    "image": problem.get("image"),
                    "models": [],
                },
            )
            if not combined_problem.get("image") and problem.get("image"):
                combined_problem["image"] = problem["image"]
            for summary in problem.get("models", []):
                versioned_summary = deepcopy(summary)
                versioned_summary["model"] = _model_version_name(str(summary["model"]), run_id)
                versioned_summary["original_model"] = summary["model"]
                versioned_summary["version"] = run_id
                combined_problem["models"].append(versioned_summary)

        for model_name, samples in run["details"].items():
            display_name = _model_version_name(str(model_name), run_id)
            details[display_name] = {}
            for sample_id, entry in samples.items():
                versioned_entry = deepcopy(entry)
                versioned_entry["version"] = run_id
                versioned_entry["original_model"] = model_name
                if isinstance(versioned_entry.get("summary"), dict):
                    versioned_entry["summary"]["model"] = display_name
                    versioned_entry["summary"]["original_model"] = model_name
                    versioned_entry["summary"]["version"] = run_id
                details[display_name][sample_id] = versioned_entry

    leaderboard.sort(key=lambda row: (-row["pass_rate"], -row["passed"], row["version"], row["name"]))
    for rank, row in enumerate(leaderboard, start=1):
        row["rank"] = rank

    models_by_name = {model["name"]: model for model in models}
    models = [models_by_name[row["name"]] for row in leaderboard]

    problem_rows = []
    for problem in problems_by_id.values():
        problem["models"] = sorted(problem["models"], key=lambda row: (row["version"], row["model"]))
        judged = [row for row in problem["models"] if row["passed"] is not None]
        problem["judged"] = len(judged)
        problem["passed"] = sum(1 for row in judged if row["passed"])
        problem["attempted"] = len(problem["models"])
        problem_rows.append(problem)
    problem_rows.sort(key=lambda row: _sample_sort_key(row["id"]))

    return {
        "run_id": "all-versions",
        "label": "All versions",
        "run_dir": str(results_dir),
        "dataset_dir": str(dataset_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "leaderboard": leaderboard,
        "problems": problem_rows,
        "models": models,
        "details": details,
    }


def write_site(run_dir: Path, *, output_dir: Path | None = None, dataset_root: Path | None = None) -> Path:
    site_dir = output_dir or run_dir / "site"
    site_dir.mkdir(parents=True, exist_ok=True)
    data = collect_site_data(run_dir, dataset_root=dataset_root, site_dir=site_dir, copy_images=True)
    (site_dir / "data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (site_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (site_dir / "styles.css").write_text(STYLES_CSS, encoding="utf-8")
    (site_dir / "app.js").write_text(APP_JS, encoding="utf-8")
    return site_dir


def collect_multi_run_site_data(
    results_dir: Path,
    *,
    dataset_root: Path | None = None,
    site_dir: Path | None = None,
    copy_images: bool = False,
) -> dict[str, Any]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    dataset_root = dataset_root or default_dataset_dir()
    site_dir = site_dir or results_dir / "site"
    runs = [
        collect_site_data(run_dir, dataset_root=dataset_root, site_dir=site_dir, copy_images=copy_images)
        for run_dir in _available_run_dirs(results_dir)
    ]
    combined = _combined_site_data(runs, results_dir=results_dir, dataset_root=dataset_root)
    if combined is not None:
        runs = [combined, *runs]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "dataset_dir": str(dataset_root),
        "default_run_id": runs[0]["run_id"] if runs else "",
        "runs": runs,
    }


def write_multi_run_site(
    results_dir: Path,
    *,
    output_dir: Path | None = None,
    dataset_root: Path | None = None,
) -> Path:
    site_dir = output_dir or results_dir / "site"
    site_dir.mkdir(parents=True, exist_ok=True)
    data = collect_multi_run_site_data(
        results_dir,
        dataset_root=dataset_root,
        site_dir=site_dir,
        copy_images=True,
    )
    (site_dir / "data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (site_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (site_dir / "styles.css").write_text(STYLES_CSS, encoding="utf-8")
    (site_dir / "app.js").write_text(APP_JS, encoding="utf-8")
    return site_dir


INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>MTG Eval Viewer</title>
    <link rel="stylesheet" href="styles.css">
  </head>
  <body>
    <header class="topbar">
      <div>
        <p class="eyebrow">MTG Eval Viewer</p>
        <h1 id="run-title">Loading...</h1>
      </div>
      <div class="topbar-controls">
        <label class="run-picker">
          <span>Version</span>
          <select id="run-select"></select>
        </label>
        <nav>
          <button class="nav-button active" data-view="leaderboard">Leaderboard</button>
          <button class="nav-button" data-view="problems">Problems</button>
          <button class="nav-button" data-view="models">Models</button>
        </nav>
      </div>
    </header>
    <main id="app" class="shell"></main>
    <div id="lightbox" class="lightbox" hidden>
      <button type="button" class="lightbox-close" aria-label="Close">Close</button>
      <img id="lightbox-image" alt="">
    </div>
    <script src="app.js"></script>
  </body>
</html>
"""


STYLES_CSS = """
:root {
  color-scheme: light dark;
  --bg: #f7f4ef;
  --panel: #fffaf2;
  --panel-strong: #ffffff;
  --text: #241f1a;
  --muted: #71685d;
  --line: #ded4c5;
  --accent: #8d4f20;
  --pass: #177245;
  --fail: #a52a2a;
  --warn: #946200;
  --code: #1f2937;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #171412;
    --panel: #211d19;
    --panel-strong: #2a251f;
    --text: #f2e8dc;
    --muted: #b5a89a;
    --line: #40382f;
    --accent: #e09a5f;
    --pass: #69d194;
    --fail: #ff8a80;
    --warn: #e3bd63;
    --code: #f2e8dc;
  }
}

* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.topbar {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 24px;
  padding: 28px clamp(18px, 4vw, 48px);
  border-bottom: 1px solid var(--line);
  background: var(--panel);
  flex: 0 0 auto;
  z-index: 2;
}

.eyebrow {
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .12em;
  margin: 0 0 6px;
  text-transform: uppercase;
}

h1, h2, h3 { margin: 0; }
h1 { font-size: clamp(24px, 4vw, 40px); }
h2 { font-size: 24px; margin-bottom: 14px; }
h3 { font-size: 17px; margin-bottom: 10px; }

nav, .topbar-controls { display: flex; gap: 8px; flex-wrap: wrap; align-items: end; }
.run-picker {
  display: grid;
  gap: 4px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
}
.run-picker select {
  border: 1px solid var(--line);
  border-radius: 999px;
  background: var(--panel-strong);
  color: var(--text);
  font: inherit;
  letter-spacing: normal;
  min-width: 190px;
  padding: 8px 32px 8px 13px;
  text-transform: none;
}
button, .link-button {
  border: 1px solid var(--line);
  border-radius: 999px;
  background: var(--panel-strong);
  color: var(--text);
  cursor: pointer;
  font: inherit;
  padding: 8px 13px;
}
button:hover, .link-button:hover, button.active { border-color: var(--accent); color: var(--accent); }

.shell {
  width: min(1500px, 100%);
  margin: 0 auto;
  padding: 28px clamp(18px, 4vw, 48px) 56px;
  flex: 1 1 auto;
  min-height: 0;
  overflow: auto;
}
.shell.split {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding-bottom: 24px;
}

.grid { display: grid; gap: 18px; }
.split-layout {
  flex: 1 1 auto;
  min-height: 0;
  display: grid;
  gap: 18px;
  align-items: stretch;
}
.split-layout.models-layout { grid-template-columns: minmax(280px, 420px) minmax(0, 1fr); }
.split-layout.problem-layout { grid-template-columns: minmax(180px, 280px) minmax(0, 1fr); }
.split-pane {
  min-height: 0;
  overflow: auto;
}
.split-pane.detail-pane {
  display: grid;
  gap: 18px;
  align-content: start;
}
@media (max-width: 900px) {
  .split-layout.models-layout, .split-layout.problem-layout {
    grid-template-columns: 1fr;
    grid-template-rows: minmax(160px, 32vh) minmax(0, 1fr);
  }
  .problem-source-grid { grid-template-columns: 1fr; }
  .topbar, .topbar-controls { align-items: start; flex-direction: column; }
}

.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 1px 0 rgba(0,0,0,.03);
}

.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 18px; }
.stat { background: var(--panel-strong); border: 1px solid var(--line); border-radius: 14px; padding: 14px; }
.stat .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
.stat .value { font-size: 24px; font-weight: 750; margin-top: 4px; }

table { width: 100%; border-collapse: collapse; }
th, td { border-bottom: 1px solid var(--line); padding: 10px 8px; text-align: left; vertical-align: top; }
th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
tr.clickable { cursor: pointer; }
tr.clickable:hover { background: color-mix(in srgb, var(--accent) 8%, transparent); }

.pill {
  border: 1px solid var(--line);
  border-radius: 999px;
  display: inline-flex;
  font-size: 12px;
  font-weight: 700;
  gap: 5px;
  padding: 3px 8px;
}
.pass { color: var(--pass); border-color: color-mix(in srgb, var(--pass) 55%, var(--line)); }
.fail { color: var(--fail); border-color: color-mix(in srgb, var(--fail) 55%, var(--line)); }
.warn { color: var(--warn); border-color: color-mix(in srgb, var(--warn) 55%, var(--line)); }
.muted { color: var(--muted); }

.list { display: grid; gap: 10px; }
.list button { width: 100%; text-align: left; border-radius: 14px; }
.list button.active { background: color-mix(in srgb, var(--accent) 12%, var(--panel-strong)); }

.problem-image {
  width: 100%;
  max-height: 720px;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--panel-strong);
  object-fit: contain;
  cursor: zoom-in;
}
.problem-source-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 18px;
  align-items: start;
}
.problem-source-panel {
  min-width: 0;
}
.scroll-md {
  max-height: 720px;
  overflow: auto;
  color: var(--code);
  background: var(--panel-strong);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 14px 16px;
  line-height: 1.5;
}
.scroll-md > :first-child { margin-top: 0; }
.scroll-md > :last-child { margin-bottom: 0; }
.scroll-md h1, .scroll-md h2, .scroll-md h3, .scroll-md h4 {
  margin: 1em 0 .45em;
  line-height: 1.25;
}
.scroll-md h1 { font-size: 1.35rem; }
.scroll-md h2 { font-size: 1.15rem; }
.scroll-md h3 { font-size: 1.02rem; }
.scroll-md p { margin: .55em 0; }
.scroll-md ul { margin: .45em 0; padding-left: 1.25em; }
.scroll-md li { margin: .2em 0; }
.scroll-md code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .92em;
}
.lightbox {
  position: fixed;
  inset: 0;
  z-index: 20;
  display: grid;
  place-items: center;
  padding: 24px;
  background: rgba(20, 16, 12, .82);
}
.lightbox[hidden] { display: none; }
.lightbox img {
  max-width: min(96vw, 1400px);
  max-height: 90vh;
  border-radius: 12px;
  box-shadow: 0 12px 48px rgba(0, 0, 0, .45);
  background: var(--panel-strong);
  object-fit: contain;
}
.lightbox-close {
  position: absolute;
  top: 18px;
  right: 18px;
  border-radius: 999px;
}

pre {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  color: var(--code);
  background: var(--panel-strong);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 14px;
  margin: 0;
  line-height: 1.45;
}

details {
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--panel-strong);
  padding: 10px 12px;
}
details + details { margin-top: 10px; }
summary { cursor: pointer; font-weight: 700; }
details pre { border: 0; padding: 10px 0 0; background: transparent; }

.turn {
  border-left: 4px solid var(--line);
  padding: 12px 0 12px 14px;
}
.turn.assistant { border-color: var(--accent); }
.turn.user { border-color: #5f7ead; }
.turn.system { border-color: var(--muted); }
.turn-meta { color: var(--muted); font-size: 12px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: .08em; }
.block { margin-top: 10px; }
.block-title { color: var(--muted); font-size: 12px; font-weight: 800; letter-spacing: .08em; margin-bottom: 6px; text-transform: uppercase; }
"""


APP_JS = """
let DATASET = null;
let DATA = null;
let currentView = "leaderboard";
let lastListView = "leaderboard";

const app = document.querySelector("#app");
const VALID_VIEWS = new Set(["leaderboard", "problems", "models", "detail"]);

function defaultRoute() {
  return {
    runId: DATASET?.default_run_id || DATASET?.runs?.[0]?.run_id || "",
    view: "leaderboard",
    problemId: null,
    modelName: null,
    sampleId: null,
  };
}

function parseHash() {
  const route = defaultRoute();
  const encoded = (location.hash || "").replace(/^#\\/?/, "");
  const parts = encoded.split("/").filter(Boolean).map((part) => {
    try { return decodeURIComponent(part); } catch { return part; }
  });
  if (parts[0]) route.runId = parts[0];
  if (parts[1] === "detail") {
    route.view = "detail";
    route.modelName = parts[2] || null;
    route.sampleId = parts[3] || null;
    return route;
  }
  if (parts[1] && VALID_VIEWS.has(parts[1])) route.view = parts[1];
  if (route.view === "problems") route.problemId = parts[2] || null;
  if (route.view === "models") route.modelName = parts[2] || null;
  return route;
}

function routeHash(route) {
  const runId = encodeURIComponent(route.runId || defaultRoute().runId);
  if (route.view === "detail" && route.modelName && route.sampleId) {
    return `#/${runId}/detail/${encodeURIComponent(route.modelName)}/${encodeURIComponent(route.sampleId)}`;
  }
  if (route.view === "problems") {
    return route.problemId
      ? `#/${runId}/problems/${encodeURIComponent(route.problemId)}`
      : `#/${runId}/problems`;
  }
  if (route.view === "models") {
    return route.modelName
      ? `#/${runId}/models/${encodeURIComponent(route.modelName)}`
      : `#/${runId}/models`;
  }
  return `#/${runId}/leaderboard`;
}

function navigate(patch, { replace = false } = {}) {
  const next = { ...parseHash(), ...patch };
  const hash = routeHash(next);
  if (hash === location.hash) {
    applyRoute(next);
    return;
  }
  if (replace) {
    history.replaceState(null, "", hash);
    applyRoute(next);
    return;
  }
  location.hash = hash;
}

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function inlineMarkdown(text) {
  return esc(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>")
    .replace(/\\*([^*]+)\\*/g, "<em>$1</em>");
}

function renderMarkdown(source) {
  const lines = String(source || "").replaceAll("\\r\\n", "\\n").split("\\n");
  const html = [];
  let listStack = [];

  function closeLists(toLevel = 0) {
    while (listStack.length > toLevel) {
      html.push("</ul>");
      listStack.pop();
    }
  }

  for (const rawLine of lines) {
    const line = rawLine.replace(/\\s+$/, "");
    if (!line.trim()) {
      closeLists(0);
      continue;
    }

    const heading = line.match(/^(#{1,4})\\s+(.*)$/);
    if (heading) {
      closeLists(0);
      const level = heading[1].length;
      html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const listMatch = line.match(/^(\\s*)-\\s+(.*)$/);
    if (listMatch) {
      const indent = listMatch[1].replaceAll("\\t", "  ").length;
      const level = Math.floor(indent / 2) + 1;
      while (listStack.length < level) {
        html.push("<ul>");
        listStack.push(true);
      }
      closeLists(level);
      html.push(`<li>${inlineMarkdown(listMatch[2])}</li>`);
      continue;
    }

    closeLists(0);
    html.push(`<p>${inlineMarkdown(line)}</p>`);
  }
  closeLists(0);
  return html.join("");
}

function openLightbox(src, alt) {
  const lightbox = document.querySelector("#lightbox");
  const image = document.querySelector("#lightbox-image");
  if (!lightbox || !image) return;
  image.src = src;
  image.alt = alt || "";
  lightbox.hidden = false;
  document.body.style.overflow = "hidden";
}

function closeLightbox() {
  const lightbox = document.querySelector("#lightbox");
  const image = document.querySelector("#lightbox-image");
  if (!lightbox || !image) return;
  lightbox.hidden = true;
  image.removeAttribute("src");
  document.body.style.overflow = "";
}

function bindLightbox(root = document) {
  root.querySelectorAll("img.problem-image[data-zoom]").forEach((image) => {
    image.addEventListener("click", () => openLightbox(image.src, image.alt));
  });
}

function pct(value) {
  return `${Math.round((value || 0) * 1000) / 10}%`;
}

function money(value) {
  return value ? `$${Number(value).toFixed(4)}` : "-";
}

function statusPill(passed, judged) {
  if (!judged || passed === null || passed === undefined) return `<span class="pill warn">Unjudged</span>`;
  return passed ? `<span class="pill pass">Pass</span>` : `<span class="pill fail">Fail</span>`;
}

function formatCount(value) {
  return Number(value || 0).toLocaleString();
}

function showToolCalls(row) {
  return (row.version || DATA.run_id || "") === "tools-rules";
}

function rolloutUsage(row) {
  const parts = [`${formatCount(row.output_tokens)} output tokens`];
  if (showToolCalls(row)) parts.push(`${formatCount(row.tool_call_count)} tool calls`);
  return parts.join(" · ");
}

function normalizeDataset(data) {
  if (Array.isArray(data.runs)) return data;
  return {
    generated_at: data.generated_at,
    results_dir: data.run_dir,
    dataset_dir: data.dataset_dir,
    default_run_id: data.run_id,
    runs: [{...data, label: data.label || data.run_id}],
  };
}

function setRun(runId) {
  navigate({ runId });
}

function initRunSelector() {
  const selector = document.querySelector("#run-select");
  selector.innerHTML = DATASET.runs
    .map((run) => `<option value="${esc(run.run_id)}">${esc(run.label || run.run_id)}</option>`)
    .join("");
  selector.disabled = DATASET.runs.length <= 1;
  selector.addEventListener("change", () => navigate({ runId: selector.value }));
}

function applyRoute(route = parseHash()) {
  const run = DATASET.runs.find((item) => item.run_id === route.runId) || DATASET.runs[0] || null;
  DATA = run;
  document.querySelector("#run-title").textContent = DATA ? (DATA.label || `Run ${DATA.run_id}`) : "No runs found";
  const selector = document.querySelector("#run-select");
  if (selector && DATA) selector.value = DATA.run_id;

  const view = VALID_VIEWS.has(route.view) ? route.view : "leaderboard";
  currentView = view === "detail" ? lastListView : view;
  if (view === "detail" && lastListView === "leaderboard") lastListView = "problems";
  if (view !== "detail") lastListView = view;
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === (view === "detail" ? lastListView : view));
  });
  app.classList.toggle("split", view === "problems" || view === "models");

  if (!DATA) {
    app.innerHTML = `<section class="card"><h2>No Runs</h2><p class="muted">No result runs were found.</p></section>`;
    return;
  }
  if (view === "leaderboard") renderLeaderboard();
  else if (view === "problems") renderProblems(route.problemId);
  else if (view === "models") renderModels(route.modelName);
  else renderRunDetail(route.modelName, route.sampleId);
}

function setView(view) {
  const route = parseHash();
  if (view === "problems") {
    navigate({
      view,
      modelName: null,
      sampleId: null,
      problemId: route.problemId || (route.view === "detail" ? route.sampleId : null),
    });
  } else if (view === "models") {
    navigate({
      view,
      problemId: null,
      sampleId: null,
      modelName: route.modelName,
    });
  } else {
    navigate({ view: "leaderboard", problemId: null, modelName: null, sampleId: null });
  }
}

function renderStats() {
  const judged = DATA.leaderboard.reduce((sum, row) => sum + row.judged, 0);
  const passed = DATA.leaderboard.reduce((sum, row) => sum + row.passed, 0);
  const cost = DATA.leaderboard.reduce((sum, row) => sum + row.total_cost_usd, 0);
  return `<section class="stat-grid">
    <div class="stat"><div class="label">Models</div><div class="value">${DATA.leaderboard.length}</div></div>
    <div class="stat"><div class="label">Problems</div><div class="value">${DATA.problems.length}</div></div>
    <div class="stat"><div class="label">Judged</div><div class="value">${judged}</div></div>
    <div class="stat"><div class="label">Pass Rate</div><div class="value">${pct(judged ? passed / judged : 0)}</div></div>
    <div class="stat"><div class="label">Cost</div><div class="value">${money(cost)}</div></div>
  </section>`;
}

function renderLeaderboard() {
  app.innerHTML = `${renderStats()}
    <section class="card">
      <h2>Leaderboard</h2>
      <table>
        <thead><tr><th>Rank</th><th>Model</th><th>Version</th><th>Pass Rate</th><th>Passed</th><th>Attempted</th><th>Unjudged</th><th>Cost</th></tr></thead>
        <tbody>
          ${DATA.leaderboard.map((row) => `<tr class="clickable" data-model="${esc(row.name)}">
            <td>${row.rank}</td>
            <td><strong>${esc(row.original_model || row.name)}</strong><br><span class="muted">${esc(row.model_id)}</span></td>
            <td>${esc(row.version || DATA.run_id)}</td>
            <td>${pct(row.pass_rate)}</td>
            <td>${row.passed}/${row.judged}</td>
            <td>${row.attempted}</td>
            <td>${row.unjudged}</td>
            <td>${money(row.total_cost_usd)}</td>
          </tr>`).join("")}
        </tbody>
      </table>
    </section>`;
  app.querySelectorAll("[data-model]").forEach((row) => {
    row.addEventListener("click", () => navigate({
      view: "models",
      modelName: row.dataset.model,
      problemId: null,
      sampleId: null,
    }));
  });
}

function bindPairButtons(root) {
  root.querySelectorAll("[data-pair]").forEach((button) => {
    const [model, sample] = button.dataset.pair.split("|||");
    button.addEventListener("click", () => navigate({ view: "detail", modelName: model, sampleId: sample }));
  });
}

function renderProblems(selectedId = DATA.problems[0]?.id) {
  const problem = DATA.problems.find((item) => item.id === selectedId) || DATA.problems[0];
  if (!problem) {
    app.classList.remove("split");
    app.innerHTML = `<section class="card"><h2>No Problems</h2></section>`;
    return;
  }
  const existing = app.querySelector(".split-layout.problem-layout");
  if (existing) {
    app.querySelectorAll("[data-problem]").forEach((button) => {
      button.classList.toggle("active", button.dataset.problem === problem.id);
    });
    const detail = app.querySelector(".detail-pane");
    detail.innerHTML = renderProblemDetail(problem);
    bindPairButtons(detail);
    bindLightbox(detail);
    return;
  }
  app.innerHTML = `<div class="split-layout problem-layout">
    <aside class="card split-pane">
      <h2>Problems</h2>
      <div class="list">
        ${DATA.problems.map((item) => `<button data-problem="${esc(item.id)}" class="${item.id === problem.id ? "active" : ""}">
          <strong>Problem ${esc(item.id)}</strong><br>
          <span class="muted">${esc(item.difficulty)} · ${item.passed}/${item.judged} passed · ${item.attempted} attempts</span>
        </button>`).join("")}
      </div>
    </aside>
    <section class="split-pane detail-pane">
      ${renderProblemDetail(problem)}
    </section>
  </div>`;
  app.querySelectorAll("[data-problem]").forEach((button) => {
    button.addEventListener("click", () => navigate({
      view: "problems",
      problemId: button.dataset.problem,
      modelName: null,
      sampleId: null,
    }));
  });
  bindPairButtons(app);
  bindLightbox(app);
}

function renderProblemDetail(problem) {
  return `<article class="card">
    <h2>Problem ${esc(problem.id)}</h2>
    <p class="muted">${esc(problem.difficulty)}</p>
    <div class="problem-source-grid">
      <div class="problem-source-panel">
        <h3>Original Image</h3>
        ${problem.image ? `<img class="problem-image" data-zoom src="${esc(problem.image)}" alt="Problem ${esc(problem.id)} image">` : `<p class="muted">No image found for this problem.</p>`}
      </div>
      <div class="problem-source-panel">
        <h3>Parsed Problem</h3>
        <div class="scroll-md">${renderMarkdown(problem.problem_gold_md)}</div>
      </div>
    </div>
  </article>
  <article class="card">
    <h3>Official Solution</h3>
    <div class="scroll-md">${renderMarkdown(problem.solution_text || "No official solution text found.")}</div>
  </article>
  <article class="card">
    <h3>Model Outcomes</h3>
    <table><thead><tr><th>Model</th><th>Status</th><th>Usage</th><th>Cost</th></tr></thead><tbody>
      ${problem.models.map((row) => `<tr>
        <td><button class="link-button" data-pair="${esc(row.model)}|||${esc(row.sample_id)}">${esc(row.original_model || row.model)}${row.version ? ` · ${esc(row.version)}` : ""}</button></td>
        <td>${statusPill(row.passed, row.judged)}</td>
        <td>${esc(rolloutUsage(row))}</td>
        <td>${money((row.rollout_cost_usd || 0) + (row.judge_cost_usd || 0))}</td>
      </tr>`).join("")}
    </tbody></table>
  </article>`;
}

function renderModels(selectedName = DATA.models[0]?.name) {
  const model = DATA.models.find((item) => item.name === selectedName) || DATA.models[0];
  if (!model) {
    app.classList.remove("split");
    app.innerHTML = `<section class="card"><h2>No Models</h2></section>`;
    return;
  }
  const existing = app.querySelector(".split-layout.models-layout");
  if (existing) {
    app.querySelectorAll("[data-model-list]").forEach((button) => {
      button.classList.toggle("active", button.dataset.modelList === model.name);
    });
    const detail = app.querySelector(".detail-pane");
    detail.innerHTML = `<section class="card">${renderModelTable(model)}</section>`;
    bindPairButtons(detail);
    return;
  }
  app.innerHTML = `<div class="split-layout models-layout">
    <aside class="card split-pane">
      <h2>Models</h2>
      <div class="list">
        ${DATA.models.map((item) => `<button data-model-list="${esc(item.name)}" class="${item.name === model.name ? "active" : ""}">
          <strong>${esc(item.original_model || item.name)}</strong><br>
          <span class="muted">${item.version ? `${esc(item.version)} · ` : ""}${pct(item.pass_rate)} · ${item.passed}/${item.judged}</span>
        </button>`).join("")}
      </div>
    </aside>
    <section class="split-pane detail-pane">
      <section class="card">
        ${renderModelTable(model)}
      </section>
    </section>
  </div>`;
  app.querySelectorAll("[data-model-list]").forEach((button) => {
    button.addEventListener("click", () => navigate({
      view: "models",
      modelName: button.dataset.modelList,
      problemId: null,
      sampleId: null,
    }));
  });
  bindPairButtons(app);
}

function renderModelDetail(name) {
  navigate({ view: "models", modelName: name, problemId: null, sampleId: null });
}

function renderModelTable(model) {
  return `<h2>${esc(model.original_model || model.name)}</h2>
    <p class="muted">${model.version ? `${esc(model.version)} · ` : ""}${esc(model.model_id)} · ${pct(model.pass_rate)} · ${model.passed}/${model.judged} passed · ${model.unjudged} unjudged</p>
    <details><summary>Model config</summary><pre>${esc(JSON.stringify(model.model_config || {}, null, 2))}</pre></details>
    <table>
      <thead><tr><th>Problem</th><th>Status</th><th>Usage</th><th>Cost</th></tr></thead>
      <tbody>${model.problems.map((row) => `<tr>
        <td><button class="link-button" data-pair="${esc(model.name)}|||${esc(row.sample_id)}">Problem ${esc(row.sample_id)}</button></td>
        <td>${statusPill(row.passed, row.judged)}</td>
        <td>${esc(rolloutUsage({...row, version: row.version || model.version}))}</td>
        <td>${money((row.rollout_cost_usd || 0) + (row.judge_cost_usd || 0))}</td>
      </tr>`).join("")}</tbody>
    </table>`;
}

function renderRunDetail(modelName, sampleId) {
  const detail = DATA.details?.[modelName]?.[sampleId];
  if (!detail) {
    navigate({ view: lastListView === "models" ? "models" : "problems", modelName, sampleId: null, problemId: sampleId }, { replace: true });
    return;
  }
  const result = detail.result || {};
  const judge = detail.judge || null;
  app.classList.remove("split");
  app.innerHTML = `<button class="link-button" id="back-button">Back to ${esc(lastListView)}</button>
    <section class="grid">
      <article class="card">
        <h2>${esc(modelName)} · Problem ${esc(sampleId)}</h2>
        <p>${statusPill(detail.passed, Boolean(judge))} <span class="muted">${esc(rolloutUsage({
          ...(detail.summary || {}),
          version: detail.version || detail.summary?.version,
        }))}</span></p>
      </article>
      <article class="card">
        <h3>Model Proposed Solution</h3>
        <pre>${esc(detail.model_solution || "No final solution found.")}</pre>
      </article>
      ${renderJudge(judge)}
      <article class="card">
        <h3>Rollout Transcript</h3>
        ${renderTranscript(result.transcript)}
      </article>
    </section>`;
  document.querySelector("#back-button").addEventListener("click", () => {
    if (lastListView === "models") {
      navigate({ view: "models", modelName, problemId: null, sampleId: null });
    } else {
      navigate({ view: "problems", problemId: sampleId, modelName: null, sampleId: null });
    }
  });
}

function renderJudge(judge) {
  if (!judge) return `<article class="card"><h3>Judging Transcript</h3><p class="muted">No judge file found yet.</p></article>`;
  const verdict = judge.verdict || {};
  return `<article class="card">
    <h3>Judge Verdict</h3>
    <p>${statusPill(verdict.passed, true)}</p>
    <pre>${esc(verdict.reasoning || "")}</pre>
    <details><summary>Judge system prompt</summary><pre>${esc(verdict.prompt?.system || "")}</pre></details>
    <details><summary>Judge user prompt</summary><pre>${esc(verdict.prompt?.user || "")}</pre></details>
    <details><summary>Raw judge payload</summary><pre>${esc(JSON.stringify(judge, null, 2))}</pre></details>
  </article>`;
}

function renderTranscript(transcript) {
  const turns = transcript?.turns || [];
  if (!turns.length) return `<p class="muted">No transcript turns found.</p>`;
  return turns.map((turn, index) => renderTurn(turn, index)).join("");
}

function renderTurn(turn, index) {
  const role = turn.role || "unknown";
  const parts = [`<div class="turn ${esc(role)}"><div class="turn-meta">${index + 1}. ${esc(role)}</div>`];
  if (turn.text) parts.push(collapsibleText("Text", turn.text, role === "system" || turn.text.length > 2500));
  if (turn.thinking) parts.push(collapsibleText("Thinking", turn.thinking, true));
  if (turn.content) parts.push(renderContent(turn.content));
  if (turn.tools) parts.push(collapsibleText("Tools", JSON.stringify(turn.tools, null, 2), true));
  if (turn.tool_calls) parts.push(collapsibleText("Tool calls", JSON.stringify(turn.tool_calls, null, 2), false));
  if (turn.stop_reason || turn.usage || turn.duration_ms) {
    parts.push(`<div class="block"><div class="block-title">Metadata</div><pre>${esc(JSON.stringify({
      stop_reason: turn.stop_reason,
      usage: turn.usage,
      duration_ms: turn.duration_ms,
      started_at: turn.started_at,
      finished_at: turn.finished_at,
    }, null, 2))}</pre></div>`);
  }
  parts.push(`</div>`);
  return parts.join("");
}

function renderContent(content) {
  if (!Array.isArray(content)) return collapsibleText("Content", JSON.stringify(content, null, 2), false);
  return content.map((block) => {
    if (block.type === "text") return collapsibleText("Output", block.text || "", (block.text || "").length > 2500);
    if (block.type === "thinking") return collapsibleText("Thinking", block.thinking || "", true);
    if (block.type === "tool_result") return collapsibleText("Tool result", block.content || "", (block.content || "").length > 1000);
    return collapsibleText(block.type || "Block", JSON.stringify(block, null, 2), true);
  }).join("");
}

function collapsibleText(title, text, collapsed) {
  const body = `<pre>${esc(text || "")}</pre>`;
  if (collapsed) return `<details><summary>${esc(title)}</summary>${body}</details>`;
  return `<div class="block"><div class="block-title">${esc(title)}</div>${body}</div>`;
}

const lightbox = document.querySelector("#lightbox");
if (lightbox) {
  lightbox.addEventListener("click", (event) => {
    if (event.target === lightbox || event.target.classList.contains("lightbox-close")) closeLightbox();
  });
}
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeLightbox();
});

fetch("data.json")
  .then((response) => response.json())
  .then((data) => {
    DATASET = normalizeDataset(data);
    initRunSelector();
    document.querySelectorAll(".nav-button").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
    window.addEventListener("hashchange", () => applyRoute());
    if (!location.hash) {
      history.replaceState(null, "", routeHash(defaultRoute()));
    }
    applyRoute();
  })
  .catch((error) => {
    app.innerHTML = `<section class="card"><h2>Could not load data.json</h2><pre>${esc(error.stack || error.message || error)}</pre></section>`;
  });
"""
