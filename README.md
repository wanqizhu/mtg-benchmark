# mtg-benchmark

LLM benchmark harness for MTG puzzles, sourced from [Possibility Storm](https://www.patreon.com/mtgpuzzles). Web leaderboard: [mtg-benchmark-site](https://github.com/wanqizhu/mtg-benchmark-site) ([live](https://wanqizhu.github.io/mtg-benchmark-site/)).

![Possibility Storm puzzle 001](docs/example-puzzle.png)

Example puzzle above. We give the model the puzzle transcribed in text and ability to grep official rules and just let the model reason.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
# set ANTHROPIC_API_KEY / OPENAI_API_KEY / XAI_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY in .env

Put the puzzles dataset at `datasets/mtg/` (gitignored).
`results/` is also gitignored, so rollout transcripts stay local.
```

## Run Eval

```bash
bench run --run-id smoke-001 --models claude-haiku-4-5 --samples 001 --judge
```

Results are written under `results/<run-id>/`.

Model names: `claude-{family}-{version}[-thinking][-{effort}]`, `gpt-{version}[-{tier}]-thinking-{effort}`, `grok-{version}-{effort}`, `gemini-{version}[-{tier}]-thinking-{level}` (e.g. `gemini-3.1-pro-thinking-high`, `gemini-3.8-flash-thinking-high`), and open-weight models via OpenRouter: an alias such as `glm-5.3-thinking-high`, `deepseek-v4-pro-thinking-high`, `kimi-k3-thinking-high`, `gpt-oss-120b-thinking-high`, or the raw `openrouter/{vendor}/{model}[-thinking][-{effort}]`. OpenRouter cost comes from the billed `usage.cost`; other providers use the rate table in `harness/pricing.py`. Aliased OpenRouter models are pinned to a fixed host at native precision (`OPENROUTER_PROVIDER_PINS` in `harness/providers/openrouter.py`) so results are reproducible and prompt caching works across turns; each transcript records the pin (`provider_pin` on the config turn) and the host that actually served each turn (`served_by`). Raw `openrouter/...` names use OpenRouter's default routing.

`--models` is comma-separated and runs those models in one process (same `--rules-mode` / `--run-id`). To run several versions at once, start one `bench run` per mode/run-id (or per model if you want isolated logs) and they share `--concurrency` within that process.

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

```bash
bench site --output-dir ../mtg-benchmark-site \
  --problems 1-20,298-307 \
  --detail-problems 1-15 \
  --exclude-models claude-sonnet-5-thinking-low,claude-opus-4-8-thinking-high
```

`--problems` is the leaderboard/models set. `--detail-problems` is the published puzzle pages. Preview with `python -m http.server 8000 --directory ../mtg-benchmark-site`. The site repo `.gitignore` allowlists problem files for 001-020 so later ids cannot be committed.

