# mtg-benchmark

LLM benchmark harness for MTG puzzles.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
# set ANTHROPIC_API_KEY (and optionally OPENAI_API_KEY / XAI_API_KEY) in .env

Put the puzzles dataset at `datasets/mtg/` (gitignored; never commit it).
`results/` is also gitignored, so rollout transcripts stay local.
```

## Smoke test (puzzle 001, Haiku)

```bash
bench run --run-id smoke-001 --models claude-haiku-4-5 --samples 001 --judge
```

Claude names use `claude-{haiku|sonnet|opus|fable}-{version}[-thinking][-{low|medium|high|xhigh|max}]`. OpenAI names use `gpt-{version}[-{tier}][-thinking][-{none|low|medium|high|xhigh|max}]`. Grok names use `grok-{version}[-{low|medium|high|xhigh}]` (always reasons; default effort is high):

- `claude-haiku-4-5`
- `claude-sonnet-4-6-thinking-medium`
- `claude-opus-4-8-thinking-xhigh`
- `claude-fable-5-1-thinking-high`
- `gpt-5.6-sol-thinking-high`
- `grok-4.6-high`

Results are written under `results/<run-id>/`. The two published versions use `--run-id grep-rules` (`--rules-mode tools`) and `--run-id full-rules-in-context` (`--rules-mode inline`).

`--models` is comma-separated and runs those models in one process (same `--rules-mode` / `--run-id`). Tools vs inline are different modes, so they are separate commands. To run several versions at once, start one `bench run` per mode/run-id (or per model if you want isolated logs) and they share `--concurrency` within that process.

```bash
bench run --run-id grep-rules --rules-mode tools \
  --models claude-sonnet-5-thinking-high,claude-sonnet-5-thinking-low,claude-haiku-4-5-thinking \
  --samples 001,002,003 --concurrency 8 --judge
```

## Live monitor

While evals are running (one or many processes):

```bash
bench watch
```

This scans every run under `results/` and tails `results/live.jsonl` for last-minute token/cost rates. Ctrl-C to stop. `bench watch --once` prints a single snapshot.

## Eval website

The published site lives in a sibling repo (`../mtg-benchmark-site` by default). Generate split JSON (no transcripts) plus copied frontend files:

```bash
bench site --output-dir ../mtg-benchmark-site
```

Local preview without the public repo:

```bash
bench site
python -m http.server 8000 --directory results/site
```

Then visit `http://localhost:8000`. The generator writes `data/manifest.json`, per-run `summary.json` files, per-attempt detail JSON, and `data/problems/{id}.json`. Re-run `bench site` after adding rollouts, judges, or problems.

Include a range of transcribed problems, with full pages only for a subset:

```bash
bench site --output-dir ../mtg-benchmark-site --problems 1-39 --detail-problems 1-20
```

To generate a standalone site for one run instead:

```bash
bench site --run-id smoke-001
```
