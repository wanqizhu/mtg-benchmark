# mtg-ai

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

Claude names use `claude-{haiku|sonnet|opus|fable}-{version}[-thinking][-{low|medium|high|xhigh|max}]`. Grok names use `grok-{version}[-{low|medium|high|xhigh}]` (always reasons; default effort is high):

- `claude-haiku-4-5`
- `claude-sonnet-4-6-thinking-medium`
- `claude-opus-4-8-thinking-xhigh`
- `grok-4.6-high`

Results are written under `results/<run-id>/`.

`--models` is comma-separated and runs those models in one process (same `--rules-mode` / `--run-id`). Tools vs inline are different modes, so they are separate commands. To run several versions at once, start one `bench run` per mode/run-id (or per model if you want isolated logs) and they share `--concurrency` within that process.

```bash
bench run --run-id tools-rules --rules-mode tools \
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

Generate a static website for all runs:

```bash
bench site
```

The site is written to `results/site/` and includes a version dropdown for each run directory. Open it from a local web server:

```bash
python -m http.server 8000 --directory results/site
```

Then visit `http://localhost:8000`. Re-run `bench site` after adding more model rollouts, judge files, or problems to refresh the leaderboard and detail pages.

To generate a standalone site for one run instead:

```bash
bench site --run-id smoke-001
```
