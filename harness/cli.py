from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from harness.config import DEFAULT_JUDGE_MODEL, DEFAULT_MAX_TURNS, RESULTS_DIR
from harness.report import print_report
from harness.site import write_multi_run_site, write_site
from harness.watch import DEFAULT_WATCH_WINDOW_S


def _load_benchmark(name: str, *, rules_mode: str = "tools", judge_model: str | None = None):
    if name == "mtg":
        from benchmarks.mtg.bench import MTGBenchmark

        return MTGBenchmark(rules_mode=rules_mode, judge_model=judge_model)
    raise KeyError(f"Unknown benchmark: {name}")


def _parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


async def _async_main(args: argparse.Namespace) -> None:
    if args.command == "watch":
        from harness.watch import watch_loop

        watch_loop(
            Path(args.results_dir),
            interval_s=args.interval,
            window_s=args.window,
            stall_after_s=args.stall_after,
            once=args.once,
        )
        return
    if args.command == "report":
        run_dir = Path(args.results_dir) / (args.run_id or "")
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        print_report(run_dir)
        return
    if args.command == "site":
        output_dir = Path(args.output_dir) if args.output_dir else None
        dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
        if args.run_id:
            run_dir = Path(args.results_dir) / args.run_id
            site_dir = write_site(run_dir, output_dir=output_dir, dataset_root=dataset_dir)
        else:
            site_dir = write_multi_run_site(
                Path(args.results_dir),
                output_dir=output_dir,
                dataset_root=dataset_dir,
            )
        print(f"Wrote eval site to {site_dir}")
        return

    rules_mode = getattr(args, "rules_mode", "tools")
    judge_model = getattr(args, "judge_model", None)
    benchmark = _load_benchmark(args.benchmark, rules_mode=rules_mode, judge_model=judge_model)
    from harness.runner import Runner

    runner = Runner(
        benchmark,
        run_id=args.run_id,
        results_dir=Path(args.results_dir),
        max_turns=getattr(args, "max_turns", DEFAULT_MAX_TURNS),
        concurrency=getattr(args, "concurrency", 4),
        rules_mode=rules_mode,
        max_tokens=getattr(args, "max_tokens", None),
        max_token_continues=getattr(args, "max_token_continues", 0),
        judge_model=judge_model,
    )

    if args.command == "run":
        paths = await runner.run_rollouts(
            _parse_csv(args.models) or [],
            sample_ids=_parse_csv(args.samples),
            force=args.force,
            resume=args.resume,
            judge=args.judge,
        )
        print(f"Wrote {len(paths)} rollout result(s) to {runner.run_dir}")
        if args.judge:
            judge_paths = getattr(runner, "last_judge_paths", [])
            print(f"Wrote {len(judge_paths)} judge result(s) to {runner.run_dir}")
            print_report(runner.run_dir)
    elif args.command == "judge":
        paths = await runner.run_judges(
            _parse_csv(args.models) or [],
            sample_ids=_parse_csv(args.samples),
            force=args.force,
        )
        print(f"Wrote {len(paths)} judge result(s) to {runner.run_dir}")
        print_report(runner.run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM benchmark harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--benchmark", default="mtg")
        p.add_argument("--run-id", required=True)
        p.add_argument("--results-dir", default=str(RESULTS_DIR))
        p.add_argument(
            "--models",
            help=(
                "Comma-separated model names, e.g. "
                "claude-sonnet-4-6-thinking-high, gpt-5.6-sol-thinking-high, "
                "or grok-4.6-high"
            ),
        )
        p.add_argument("--samples", help="Comma-separated sample ids, e.g. 001,002")
        p.add_argument("--force", action="store_true")
        p.add_argument(
            "--judge-model",
            default=DEFAULT_JUDGE_MODEL,
            help=f"Judge model name (default: {DEFAULT_JUDGE_MODEL})",
        )

    run_parser = subparsers.add_parser("run", help="Run model rollouts")
    add_common(run_parser)
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing rollout JSONs from their saved signed assistant content",
    )
    run_parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    run_parser.add_argument("--max-tokens", type=int, help="Override model max output tokens")
    run_parser.add_argument(
        "--max-token-continues",
        type=int,
        default=0,
        help="Append a single 'continue' user message up to N times after stop_reason=max_tokens",
    )
    run_parser.add_argument("--concurrency", type=int, default=4)
    run_parser.add_argument(
        "--rules-mode",
        choices=("tools", "inline"),
        default="tools",
        help="tools: model retrieves rules via grep/read; inline: full rules in prompt, one-shot",
    )
    run_parser.add_argument(
        "--judge",
        action="store_true",
        help="Judge each complete rollout as soon as it finishes (and any already-complete results)",
    )

    judge_parser = subparsers.add_parser("judge", help="Judge existing rollouts")
    add_common(judge_parser)
    judge_parser.add_argument("--concurrency", type=int, default=4)

    report_parser = subparsers.add_parser("report", help="Print report for a run")
    report_parser.add_argument("--run-id", required=True)
    report_parser.add_argument("--results-dir", default=str(RESULTS_DIR))

    site_parser = subparsers.add_parser("site", help="Generate a static eval website")
    site_parser.add_argument("--run-id", help="Generate a single-run site instead of the default all-runs site")
    site_parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    site_parser.add_argument("--output-dir", help="Directory to write the generated site")
    site_parser.add_argument(
        "--dataset-dir",
        help="Dataset directory for problem images; defaults to datasets/mtg or MTG_DATASET_DIR",
    )

    watch_parser = subparsers.add_parser("watch", help="Live dashboard for all eval runs")
    watch_parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    watch_parser.add_argument("--interval", type=float, default=5.0, help="Refresh seconds")
    watch_parser.add_argument(
        "--window",
        type=float,
        default=DEFAULT_WATCH_WINDOW_S,
        help="Rate window seconds (default: 300, 5 minutes)",
    )
    watch_parser.add_argument(
        "--stall-after",
        type=float,
        default=90.0,
        help="Flag in-progress cells with no live events for this many seconds",
    )
    watch_parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")

    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
