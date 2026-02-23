# mtg-ai

Python implementation of a constrained Magic: The Gathering rules engine plus bot-training/evaluation tooling.

## Scope of current environment

Allowed cards:
- `Mountain`
- `Lightning Bolt`
- `Raging Goblin`

Deck construction override:
- Any deck size is allowed.

Core rules implemented for this environment:
- full turn/step progression
- priority passing
- stack and spell resolution
- target legality re-check at resolution
- combat (attack/block/damage)
- haste + summoning sickness interaction
- state-based actions (creature lethal damage, life-based loss, draw-from-empty loss)
- cleanup discard to max hand size
- deterministic replay logging

## Project layout

- `src/mtg_ai/core`: rules engine, game state, actions, serialization, SBAs
- `src/mtg_ai/cards`: card definitions and registry
- `src/mtg_ai/interfaces`: CLI renderer + manual play interface
- `src/mtg_ai/replay`: replay logging + replay CLI
- `src/mtg_ai/bots`: random and fixed baseline bots
- `src/mtg_ai/training`: evolutionary training + low-life minimax baseline
- `src/mtg_ai/eval`: tournament + Elo tooling
- `src/mtg_ai/deck`: deck composition search
- `src/mtg_ai/analysis`: report + ASCII charts

## Run commands

Use `PYTHONPATH=src` for all commands.

### 1) Human / bot play

Human vs human:
```bash
PYTHONPATH=src python3 -m mtg_ai.interfaces.cli \
  --player0 human --player1 human \
  --deck0 "Mountain*6,Lightning Bolt*6,Raging Goblin*6" \
  --deck1 "Mountain*6,Lightning Bolt*6,Raging Goblin*6" \
  --life 5 --opening-hand 5 --seed 3 \
  --replay-path artifacts/replays/human_vs_human_demo.jsonl
```

Human vs bot:
```bash
PYTHONPATH=src python3 -m mtg_ai.interfaces.cli \
  --player0 human --player1 mixed \
  --deck0 "Mountain*8,Lightning Bolt*6,Raging Goblin*4" \
  --deck1 "Mountain*8,Lightning Bolt*6,Raging Goblin*4" \
  --life 8 --opening-hand 5 --seed 9 \
  --replay-path artifacts/replays/human_vs_bot_demo.jsonl
```

Replay a game:
```bash
PYTHONPATH=src python3 -m mtg_ai.replay.replay_cli artifacts/replays/human_vs_bot_demo.jsonl
```

### 2) Baseline milestone gauntlet

```bash
PYTHONPATH=src python3 -m mtg_ai.eval.scenarios \
  --games-per-pair 60 --life 10 --opening-hand 7 --seed 42
```

### 3) Train a stronger bot

```bash
PYTHONPATH=src python3 -m mtg_ai.training.self_play \
  --episodes 80 \
  --epsilon 0.25 \
  --eval-interval 10 \
  --eval-games 20 \
  --life 10 \
  --opening-hand 7 \
  --seed 13 \
  --output-dir artifacts/training/self_play
```

### 4) Elo progression over training checkpoints

```bash
PYTHONPATH=src python3 -m mtg_ai.training.league \
  --checkpoint-dir artifacts/training/self_play \
  --games-per-pair 30 \
  --life 10 --opening-hand 7 --seed 111
```

### 5) Deck search (trained policy)

```bash
PYTHONPATH=src python3 -m mtg_ai.deck.search \
  --policy trained \
  --trained-path artifacts/training/self_play/policy_final.json \
  --deck-size 12 \
  --mode evolutionary \
  --population-size 8 \
  --generations 6 \
  --games-per-opponent 8 \
  --life-values 5,10,20 \
  --hand-sizes 5,7 \
  --seed 77 \
  --output-csv artifacts/deck_search/results.csv
```

### 6) Final aggregate report

```bash
PYTHONPATH=src python3 -m mtg_ai.analysis.report \
  --training-dir artifacts/training/self_play \
  --deck-search-csv artifacts/deck_search/results.csv \
  --output artifacts/report.md \
  --seed 97
```

## Development checks

Syntax validation:
```bash
python3 -m compileall src
```

Pytest tests are included in `tests/` (if `pytest` is available in your environment).
