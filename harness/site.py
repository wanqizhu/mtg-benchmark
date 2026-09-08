from __future__ import annotations

import json
import re
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.mtg.dataset import REPO_ROOT, dataset_dir as default_dataset_dir
from benchmarks.mtg.solution import parse_solution

SITE_ASSET_DIR = "assets/problems"
SITE_STATIC_DIR = Path(__file__).resolve().parent / "site_static"
ALL_VERSIONS_ID = "all-versions"
DEFAULT_SITE_RUNS = ("grep-rules", "tools-rules")
DEFAULT_EXCLUDED_MODELS = ("claude-sonnet-5-thinking-low",)
SKIP_RUN_DIRS = frozenset({"site", "logs", "_archive", "_live"})
RUN_LABELS = {
    "grep-rules": "grep rules",
    "full-rules-in-context": "full rules in context",
    "tools-rules": "grep rules",
    "inline-rules": "full rules in context",
}
_ABS_PATH_RE = re.compile(r"(?:[A-Za-z]:)?/(?:Users|home|private|opt|var|tmp)/[^\s\"'<>]+")


def run_label(run_id: str) -> str:
    return RUN_LABELS.get(run_id, run_id)


def _relative_path(path: Path | None, *bases: Path) -> str | None:
    """POSIX-relative path for site JSON; never emit a machine-absolute path."""
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    seen: list[Path] = []
    for base in (*bases, REPO_ROOT, Path.cwd()):
        try:
            base_resolved = Path(base).expanduser().resolve()
        except OSError:
            continue
        if base_resolved in seen:
            continue
        seen.append(base_resolved)
        try:
            return resolved.relative_to(base_resolved).as_posix()
        except ValueError:
            continue
    return Path(path).name


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalize_problem_id(token: str) -> str:
    token = token.strip()
    return token.zfill(3) if token.isdigit() else token


def expand_problem_ids(ids: list[str] | str | None) -> list[str] | None:
    """Expand '001,5,10-12' or ['1-3', '010'] into zero-padded ids."""
    if ids is None:
        return None
    tokens = [ids] if isinstance(ids, str) else ids
    expanded: list[str] = []
    seen: set[str] = set()
    for raw in tokens:
        for part in str(raw).split(","):
            token = part.strip()
            if not token:
                continue
            if "-" in token:
                left, right = token.split("-", 1)
                if left.strip().isdigit() and right.strip().isdigit():
                    start = int(left)
                    end = int(right)
                    if start > end:
                        start, end = end, start
                    width = max(3, len(left.strip()), len(right.strip()))
                    values = [str(number).zfill(width) for number in range(start, end + 1)]
                else:
                    values = [_normalize_problem_id(token)]
            else:
                values = [_normalize_problem_id(token)]
            for value in values:
                if value not in seen:
                    seen.add(value)
                    expanded.append(value)
    return expanded


def _excluded_models(model_names: list[str] | None) -> set[str]:
    if model_names is None:
        return set(DEFAULT_EXCLUDED_MODELS)
    return {name for name in model_names if name and name.lower() != "none"}


def _id_set(ids: list[str] | None) -> set[str] | None:
    expanded = expand_problem_ids(ids)
    if expanded is None:
        return None
    return set(expanded)


def _result_paths(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        return []
    paths: list[Path] = []
    for model_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name != "site"):
        paths.extend(sorted(p for p in model_dir.glob("*.json") if p.stem.isdigit()))
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


def _judge_reasoning(judge: dict[str, Any] | None) -> str:
    verdict = _verdict(judge)
    if not verdict:
        return ""
    return str(verdict.get("reasoning") or "")


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


def _problem_image(
    dataset_root: Path,
    site_dir: Path,
    sample_id: str,
    *,
    copy_images: bool,
    publish_detail: bool,
) -> str | None:
    if not publish_detail:
        return None
    if copy_images:
        return _copy_image(dataset_root, site_dir, sample_id)
    image_source = _image_source(dataset_root, sample_id)
    return _relative_path(image_source, dataset_root) if image_source else None


def _dataset_problems(
    dataset_root: Path,
    site_dir: Path,
    *,
    copy_images: bool,
    problem_ids: set[str] | None = None,
    detail_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
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
        # Mirrors load_puzzles(): puzzles with an unreliable reference solution are never run,
        # so they must not show up as empty rows on the site either.
        if metadata.get("excluded"):
            continue
        sample_id = entry.name
        if problem_ids is not None and sample_id not in problem_ids:
            continue
        has_detail = detail_ids is None or sample_id in detail_ids
        source_url = str(metadata.get("source_url") or metadata.get("page_url") or "").strip()
        solution_url = str(metadata.get("solution_url") or "").strip()
        problems[sample_id] = {
            "id": sample_id,
            "difficulty": metadata.get("difficulty", "Unknown"),
            "problem_gold_md": gold_path.read_text(encoding="utf-8") if has_detail else "",
            "solution_text": metadata.get("solution_text", "") if has_detail else "",
            "image": _problem_image(
                dataset_root,
                site_dir,
                sample_id,
                copy_images=copy_images,
                publish_detail=has_detail,
            ),
            "source_url": source_url or None,
            "solution_url": solution_url or None,
            "has_detail": has_detail,
            "models": [],
        }
    return problems


def _has_results(run_dir: Path) -> bool:
    return bool(_result_paths(run_dir))


def _available_run_dirs(results_dir: Path, run_ids: list[str] | None = None) -> list[Path]:
    available = sorted(
        (
            path
            for path in results_dir.iterdir()
            if path.is_dir() and path.name not in SKIP_RUN_DIRS and _has_results(path)
        ),
        key=lambda path: path.name,
    )
    if run_ids:
        if any(token.lower() == "all" for token in run_ids):
            return available
        wanted = set(run_ids)
        return [path for path in available if path.name in wanted]
    preferred = [path for path in available if path.name in DEFAULT_SITE_RUNS]
    return preferred or available


def _model_version_name(model_name: str, run_id: str) -> str:
    return f"{model_name} @ {run_label(run_id)}"


def _sample_sort_key(sample_id: str) -> tuple[int, str]:
    return (int(sample_id), sample_id) if sample_id.isdigit() else (10**9, sample_id)


def _detail_path(run_id: str, model_name: str, sample_id: str) -> str:
    return f"data/runs/{run_id}/{model_name}/{sample_id}.json"


def _model_row(name: str, entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    judged = [entry for entry in entries.values() if entry["passed"] is not None]
    passed = sum(1 for entry in judged if entry["passed"])
    rollout_cost = sum(_cost_usd(entry["result"]) for entry in entries.values())
    judge_cost = sum(_cost_usd(entry["judge"]) for entry in entries.values())
    first_result = next((entry["result"] for entry in entries.values() if entry["result"]), {})
    total = len(judged)
    transcripts = [(entry.get("result") or {}).get("transcript") or {} for entry in entries.values()]
    output_tokens = sum(_output_tokens(transcript) for transcript in transcripts)
    tool_call_count = sum(_tool_call_count(transcript) for transcript in transcripts)
    return {
        "name": name,
        "model_id": first_result.get("model_id", ""),
        "model_config": first_result.get("model_config", {}),
        "attempted": len(entries),
        "judged": total,
        "passed": passed,
        "failed": total - passed,
        "unjudged": len(entries) - total,
        "output_tokens": output_tokens,
        "tool_call_count": tool_call_count,
        "pass_rate": passed / total if total else 0.0,
        "rollout_cost_usd": rollout_cost,
        "judge_cost_usd": judge_cost,
        "total_cost_usd": rollout_cost + judge_cost,
    }


def _empty_attempt(
    model_name: str,
    sample_id: str,
    *,
    original_model: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    row = {
        "model": model_name,
        "sample_id": sample_id,
        "status": "not_attempted",
        "passed": None,
        "judged": False,
        "output_tokens": 0,
        "tool_call_count": 0,
        "rollout_cost_usd": 0.0,
        "judge_cost_usd": 0.0,
    }
    if original_model is not None:
        row["original_model"] = original_model
    if version is not None:
        row["version"] = version
    return row


def _align_problem_models(problems: list[dict[str, Any]], leaderboard: list[dict[str, Any]]) -> None:
    """Keep every problem's model table in leaderboard order, including missing attempts."""
    for problem in problems:
        by_model = {row["model"]: row for row in problem.get("models", [])}
        aligned = []
        for model in leaderboard:
            name = str(model["name"])
            existing = by_model.get(name)
            if existing is not None:
                aligned.append(existing)
            else:
                aligned.append(
                    _empty_attempt(
                        name,
                        str(problem["id"]),
                        original_model=model.get("original_model"),
                        version=model.get("version"),
                    )
                )
        problem["models"] = aligned
        judged = [row for row in aligned if row["passed"] is not None]
        problem["judged"] = len(judged)
        problem["passed"] = sum(1 for row in judged if row["passed"])
        problem["attempted"] = sum(1 for row in aligned if row.get("status") != "not_attempted")


def _sanitize_text(value: Any) -> str:
    text = str(value or "")
    return _ABS_PATH_RE.sub(lambda match: Path(match.group(0)).name, text)


def _public_tool_input(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    payload = dict(value)
    path = payload.get("path")
    if isinstance(path, str) and path.strip():
        payload["path"] = Path(path).name
    return payload


def _public_block(block: Any) -> dict[str, Any] | None:
    if not isinstance(block, dict):
        return None
    kind = str(block.get("type") or "")
    if kind == "thinking":
        return {"type": "thinking", "thinking": _sanitize_text(block.get("thinking") or block.get("text") or "")}
    if kind == "text":
        return {"type": "text", "text": _sanitize_text(block.get("text") or "")}
    if kind == "tool_use":
        return {
            "type": "tool_use",
            "id": block.get("id"),
            "name": block.get("name"),
            "input": _public_tool_input(block.get("input")),
        }
    if kind == "tool_result":
        content = block.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, indent=2) if content is not None else ""
        return {
            "type": "tool_result",
            "tool_use_id": block.get("tool_use_id"),
            "content": _sanitize_text(content),
        }
    return None


def _public_turn(turn: dict[str, Any]) -> dict[str, Any] | None:
    role = turn.get("role")
    if role == "config":
        return {
            key: turn[key]
            for key in ("role", "model", "model_id", "thinking", "output_config", "max_tokens")
            if key in turn
        }
    if role == "tools":
        tools = []
        for tool in turn.get("tools") or []:
            if isinstance(tool, dict):
                public_tool = {key: tool[key] for key in ("name", "description") if key in tool}
                if "description" in public_tool:
                    public_tool["description"] = _sanitize_text(public_tool["description"])
                tools.append(public_tool)
        return {"role": "tools", "tools": tools}
    if role not in {"system", "user", "assistant"}:
        return None
    payload: dict[str, Any] = {"role": role}
    if turn.get("text"):
        payload["text"] = _sanitize_text(turn.get("text"))
    content = turn.get("content")
    if isinstance(content, list):
        blocks = [block for block in (_public_block(item) for item in content) if block]
        if blocks:
            payload["content"] = blocks
    return payload


def _public_transcript(transcript: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(transcript, dict) or not transcript:
        return None
    turns = [turn for turn in (_public_turn(item) for item in transcript.get("turns") or []) if turn]
    if not turns:
        return None
    usage = transcript.get("usage")
    return {
        "turns": turns,
        "tool_call_count": transcript.get("tool_call_count"),
        "usage": usage if isinstance(usage, dict) else {},
        "model": transcript.get("model"),
        "provider": transcript.get("provider"),
    }


def _public_detail(entry: dict[str, Any]) -> dict[str, Any]:
    summary = entry.get("summary") or {}
    result = entry.get("result") or {}
    return {
        "model": summary.get("model"),
        "original_model": summary.get("original_model"),
        "version": summary.get("version"),
        "sample_id": summary.get("sample_id"),
        "status": summary.get("status"),
        "passed": summary.get("passed"),
        "judged": summary.get("judged"),
        "output_tokens": summary.get("output_tokens", 0),
        "tool_call_count": summary.get("tool_call_count", 0),
        "rollout_cost_usd": summary.get("rollout_cost_usd", 0.0),
        "judge_cost_usd": summary.get("judge_cost_usd", 0.0),
        "model_solution": entry.get("model_solution"),
        "judge_reasoning": entry.get("judge_reasoning") or "",
        "transcript": _public_transcript(result.get("transcript") or {}),
        "model_id": result.get("model_id", ""),
        "model_config": result.get("model_config") or {},
        "updated_at": result.get("updated_at"),
        "detail_path": summary.get("detail_path"),
    }


def _summary_problem(problem: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": problem["id"],
        "difficulty": problem.get("difficulty", "Unknown"),
        "has_detail": bool(problem.get("has_detail", True)),
        "image": problem.get("image"),
        "attempted": problem.get("attempted", 0),
        "judged": problem.get("judged", 0),
        "passed": problem.get("passed", 0),
        "models": problem.get("models") or [],
    }


def _summary_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run["run_id"],
        "label": run["label"],
        "leaderboard": run["leaderboard"],
        "problems": [_summary_problem(problem) for problem in run["problems"]],
        "models": run["models"],
    }


def collect_site_data(
    run_dir: Path,
    *,
    dataset_root: Path | None = None,
    site_dir: Path | None = None,
    copy_images: bool = False,
    problem_ids: list[str] | None = None,
    detail_ids: list[str] | None = None,
    exclude_models: list[str] | None = None,
) -> dict[str, Any]:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    dataset_root = dataset_root or default_dataset_dir()
    site_dir = site_dir or run_dir / "site"
    wanted = _id_set(problem_ids)
    details_wanted = _id_set(detail_ids)
    skipped_models = _excluded_models(exclude_models)

    entries_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    problems = _dataset_problems(
        dataset_root,
        site_dir,
        copy_images=copy_images,
        problem_ids=wanted,
        detail_ids=details_wanted,
    )
    details: dict[str, dict[str, Any]] = {}

    for result_path in _result_paths(run_dir):
        result = _read_json(result_path)
        judge_path = _judge_path(result_path)
        judge = _read_json(judge_path) if judge_path.exists() else None
        sample_id = str(result.get("sample_id") or result_path.stem)
        model_name = str(result.get("model_name") or result_path.parent.name)
        if model_name in skipped_models:
            continue
        if wanted is not None and sample_id not in wanted:
            continue
        reference = result.get("reference", {})
        if not isinstance(reference, dict):
            reference = {}

        passed = _passed(judge)
        has_detail = details_wanted is None or sample_id in details_wanted
        image = _problem_image(
            dataset_root,
            site_dir,
            sample_id,
            copy_images=copy_images,
            publish_detail=has_detail,
        )

        problem = problems.setdefault(
            sample_id,
            {
                "id": sample_id,
                "difficulty": reference.get("difficulty", "Unknown"),
                "problem_gold_md": reference.get("problem_gold_md", "") if has_detail else "",
                "solution_text": reference.get("solution_text", "") if has_detail else "",
                "image": image,
                "has_detail": has_detail,
                "models": [],
            },
        )
        if has_detail and not problem.get("image") and image:
            problem["image"] = image
        if has_detail and not problem.get("problem_gold_md") and reference.get("problem_gold_md"):
            problem["problem_gold_md"] = reference["problem_gold_md"]
        if has_detail and not problem.get("solution_text") and reference.get("solution_text"):
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
            "detail_path": _detail_path(run_dir.name, model_name, sample_id) if has_detail else None,
        }
        problem["models"].append(summary)

        entry = {
            "result": result,
            "judge": judge,
            "passed": passed,
            "model_solution": _model_solution(result.get("transcript") or {}),
            "judge_reasoning": _judge_reasoning(judge),
            "summary": summary,
        }
        entries_by_model.setdefault(model_name, {})[sample_id] = entry
        if has_detail:
            details.setdefault(model_name, {})[sample_id] = entry

    leaderboard = sorted(
        (_model_row(model_name, entries) for model_name, entries in entries_by_model.items()),
        key=lambda row: (-row["pass_rate"], -row["passed"], row["name"]),
    )
    for rank, row in enumerate(leaderboard, start=1):
        row["rank"] = rank

    problem_rows = list(problems.values())
    problem_rows.sort(key=lambda row: _sample_sort_key(row["id"]))
    _align_problem_models(problem_rows, leaderboard)

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
        "label": run_label(run_dir.name),
        "leaderboard": leaderboard,
        "problems": problem_rows,
        "models": model_rows,
        "details": details,
    }


def combine_run_summaries(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Client-equivalent merge of per-run summaries into an All versions view."""
    if len(runs) < 2:
        raise ValueError("combine_run_summaries requires at least two runs")

    leaderboard: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
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
                    "has_detail": bool(problem.get("has_detail", True)),
                    "image": problem.get("image"),
                    "models": [],
                },
            )
            if not combined_problem.get("image") and problem.get("image"):
                combined_problem["image"] = problem["image"]
            if problem.get("has_detail"):
                combined_problem["has_detail"] = True
            for summary in problem.get("models", []):
                versioned_summary = deepcopy(summary)
                versioned_summary["model"] = _model_version_name(str(summary["model"]), run_id)
                versioned_summary["original_model"] = summary["model"]
                versioned_summary["version"] = run_id
                combined_problem["models"].append(versioned_summary)

    leaderboard.sort(key=lambda row: (-row["pass_rate"], -row["passed"], row["version"], row["name"]))
    for rank, row in enumerate(leaderboard, start=1):
        row["rank"] = rank

    models_by_name = {model["name"]: model for model in models}
    models = [models_by_name[row["name"]] for row in leaderboard]

    problem_rows = list(problems_by_id.values())
    problem_rows.sort(key=lambda row: _sample_sort_key(row["id"]))
    _align_problem_models(problem_rows, leaderboard)

    return {
        "run_id": ALL_VERSIONS_ID,
        "label": "All versions",
        "leaderboard": leaderboard,
        "problems": problem_rows,
        "models": models,
    }


def collect_multi_run_site_data(
    results_dir: Path,
    *,
    dataset_root: Path | None = None,
    site_dir: Path | None = None,
    copy_images: bool = False,
    problem_ids: list[str] | None = None,
    detail_ids: list[str] | None = None,
    run_ids: list[str] | None = None,
    exclude_models: list[str] | None = None,
) -> dict[str, Any]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    dataset_root = dataset_root or default_dataset_dir()
    site_dir = site_dir or results_dir / "site"
    runs = [
        collect_site_data(
            run_dir,
            dataset_root=dataset_root,
            site_dir=site_dir,
            copy_images=copy_images,
            problem_ids=problem_ids,
            detail_ids=detail_ids,
            exclude_models=exclude_models,
        )
        for run_dir in _available_run_dirs(results_dir, run_ids)
    ]
    default_run_id = ALL_VERSIONS_ID if len(runs) >= 2 else (runs[0]["run_id"] if runs else "")
    problem_list = sorted({problem["id"] for run in runs for problem in run["problems"]}, key=_sample_sort_key)
    return {
        "default_run_id": default_run_id,
        "runs": runs,
        "problems": problem_list,
    }


def _copy_static(site_dir: Path) -> None:
    if not SITE_STATIC_DIR.exists():
        raise FileNotFoundError(f"Site static directory not found: {SITE_STATIC_DIR}")
    for name in ("index.html", "app.js", "styles.css", "methodology.md", "robots.txt"):
        shutil.copy2(SITE_STATIC_DIR / name, site_dir / name)
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")


def _replace_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _write_problem_files(site_dir: Path, problems: list[dict[str, Any]]) -> None:
    problem_dir = site_dir / "data" / "problems"
    _replace_dir(problem_dir)
    for problem in problems:
        if not problem.get("has_detail", True):
            continue
        _write_json(
            problem_dir / f"{problem['id']}.json",
            {
                "id": problem["id"],
                "difficulty": problem.get("difficulty", "Unknown"),
                "problem_gold_md": problem.get("problem_gold_md", ""),
                "solution_text": problem.get("solution_text", ""),
                "image": problem.get("image"),
                "source_url": problem.get("source_url"),
                "solution_url": problem.get("solution_url"),
            },
        )


def _write_run_files(site_dir: Path, run: dict[str, Any]) -> None:
    run_id = str(run["run_id"])
    run_dir = site_dir / "data" / "runs" / run_id
    _replace_dir(run_dir)
    _write_json(run_dir / "summary.json", _summary_run(run))
    for model_name, samples in (run.get("details") or {}).items():
        for sample_id, entry in samples.items():
            _write_json(run_dir / model_name / f"{sample_id}.json", _public_detail(entry))


def _write_manifest(site_dir: Path, *, default_run_id: str, runs: list[dict[str, Any]], problems: list[str]) -> None:
    manifest_runs: list[dict[str, Any]] = []
    if default_run_id == ALL_VERSIONS_ID:
        manifest_runs.append({"run_id": ALL_VERSIONS_ID, "label": "All versions", "virtual": True})
    for run in runs:
        manifest_runs.append({"run_id": run["run_id"], "label": run.get("label") or run["run_id"]})
    _write_json(
        site_dir / "data" / "manifest.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "default_run_id": default_run_id,
            "runs": manifest_runs,
            "problems": problems,
        },
    )


def write_site(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    dataset_root: Path | None = None,
    problem_ids: list[str] | None = None,
    detail_ids: list[str] | None = None,
    exclude_models: list[str] | None = None,
) -> Path:
    site_dir = output_dir or run_dir / "site"
    site_dir.mkdir(parents=True, exist_ok=True)
    assets = site_dir / SITE_ASSET_DIR
    if assets.exists():
        shutil.rmtree(assets)
    data = collect_site_data(
        run_dir,
        dataset_root=dataset_root,
        site_dir=site_dir,
        copy_images=True,
        problem_ids=problem_ids,
        detail_ids=detail_ids,
        exclude_models=exclude_models,
    )
    _replace_dir(site_dir / "data" / "runs")
    _write_run_files(site_dir, data)
    _write_problem_files(site_dir, data["problems"])
    _write_manifest(
        site_dir,
        default_run_id=data["run_id"],
        runs=[data],
        problems=[problem["id"] for problem in data["problems"]],
    )
    _copy_static(site_dir)
    stale = site_dir / "data.json"
    if stale.exists():
        stale.unlink()
    return site_dir


def write_multi_run_site(
    results_dir: Path,
    *,
    output_dir: Path | None = None,
    dataset_root: Path | None = None,
    problem_ids: list[str] | None = None,
    detail_ids: list[str] | None = None,
    run_ids: list[str] | None = None,
    exclude_models: list[str] | None = None,
) -> Path:
    site_dir = output_dir or results_dir / "site"
    site_dir.mkdir(parents=True, exist_ok=True)
    assets = site_dir / SITE_ASSET_DIR
    if assets.exists():
        shutil.rmtree(assets)
    data = collect_multi_run_site_data(
        results_dir,
        dataset_root=dataset_root,
        site_dir=site_dir,
        copy_images=True,
        problem_ids=problem_ids,
        detail_ids=detail_ids,
        run_ids=run_ids,
        exclude_models=exclude_models,
    )
    _replace_dir(site_dir / "data" / "runs")
    for run in data["runs"]:
        _write_run_files(site_dir, run)
    problems_by_id: dict[str, dict[str, Any]] = {}
    for run in data["runs"]:
        for problem in run["problems"]:
            problems_by_id.setdefault(problem["id"], problem)
    _write_problem_files(site_dir, list(problems_by_id.values()))
    _write_manifest(
        site_dir,
        default_run_id=data["default_run_id"],
        runs=data["runs"],
        problems=data["problems"],
    )
    _copy_static(site_dir)
    stale = site_dir / "data.json"
    if stale.exists():
        stale.unlink()
    return site_dir
