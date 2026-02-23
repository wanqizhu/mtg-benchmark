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
  --episodes 35 \
  --epsilon 0.22 \
  --eval-interval 5 \
  --eval-games 8 \
  --life 10 \
  --opening-hand 7 \
  --life-values 5,10,20 \
  --hand-values 5,7 \
  --solver-weight 0.15 \
  --solver-seed 211 \
  --solver-depth 7 \
  --solver-sample-count 6 \
  --history-weight 0.1 \
  --history-games 10 \
  --seed 41 \
  --output-dir artifacts/training/self_play_multi
```

### 4) Elo progression over training checkpoints

```bash
PYTHONPATH=src python3 -m mtg_ai.training.league \
  --checkpoint-dir artifacts/training/self_play_multi \
  --games-per-pair 30 \
  --life 10 --opening-hand 7 --seed 111
```

### 5) Solver-backed low-life validation

```bash
PYTHONPATH=src python3 -m mtg_ai.training.value_iteration_baseline \
  --life 10 \
  --opening-hand 4 \
  --depth 7 \
  --sample-count 12 \
  --seed 5 \
  --policy-path artifacts/training/self_play_multi/policy_final.json
```

### 6) Deck search (trained policy)

```bash
PYTHONPATH=src python3 -m mtg_ai.deck.search \
  --policy trained \
  --trained-path artifacts/training/self_play_multi/policy_final.json \
  --deck-size 12 \
  --mode evolutionary \
  --population-size 8 \
  --generations 6 \
  --games-per-opponent 8 \
  --life-values 5,10,20 \
  --hand-sizes 5,7 \
  --seed 91 \
  --output-csv artifacts/deck_search/results_multi.csv
```

### 7) Generalization matrix export

```bash
PYTHONPATH=src python3 -m mtg_ai.eval.generalization \
  --policy-path artifacts/training/self_play_multi/policy_final.json \
  --life-values 5,10,20 \
  --hand-values 5,7 \
  --games 120 \
  --seed 1300 \
  --output-csv artifacts/eval/generalization_matrix.csv
```

### 8) Hyperparameter sweep (optional deeper search)

```bash
PYTHONPATH=src python3 -m mtg_ai.training.hyperparam_sweep \
  --episodes 20 \
  --eval-games 6 \
  --life-values 5,10,20 \
  --hand-values 5,7 \
  --epsilons 0.18,0.22 \
  --solver-weights 0.0,0.1,0.2 \
  --history-weights 0.0,0.1 \
  --seeds 71,73 \
  --output-dir artifacts/training/sweep_runs \
  --output-csv artifacts/training/sweep_results.csv
```

### 9) Final aggregate report

```bash
PYTHONPATH=src python3 -m mtg_ai.analysis.report \
  --training-dir artifacts/training/self_play_multi \
  --deck-search-csv artifacts/deck_search/results_multi.csv \
  --generalization-csv artifacts/eval/generalization_matrix.csv \
  --sweep-csv artifacts/training/sweep_results.csv \
  --output artifacts/report.md \
  --seed 103
```

## Development checks

Syntax validation:
```bash
python3 -m compileall src
```

Pytest tests are included in `tests/` (if `pytest` is available in your environment).
