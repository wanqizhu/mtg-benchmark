# mtg-ai

LLM benchmark harness for MTG puzzles.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
# set ANTHROPIC_API_KEY in .env

Put the puzzles dataset at `datasets/mtg/` (gitignored; never commit it).
`results/` is also gitignored, so rollout transcripts stay local.
```

## Smoke test (puzzle 001, Haiku)

```bash
bench run --run-id smoke-001 --models claude-haiku-4-5 --samples 001 --judge
```

Model names use the format `claude-{haiku|sonnet|opus|fable}-{version}[-thinking][-{low|medium|high|xhigh|max}]`, for example:

- `claude-haiku-4-5`
- `claude-sonnet-4-6-thinking-medium`
- `claude-opus-4-8-thinking-xhigh`

Results are written under `results/<run-id>/`.

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
