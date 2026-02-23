# MTG Rules Engine & AI Training System

A Python-based Magic: The Gathering rules engine and reinforcement learning system for the card pool: **Mountain**, **Lightning Bolt**, and **Raging Goblin**.

## Quick Start

```bash
pip install -r requirements.txt

# Play human vs bot
python play.py --p1 human --p2 mixed --life 20 --deck1 "20M,10B,10G"

# Play trained AI vs bot
python play.py --p1 trained --p2 mixed --games 100 --life 20 --model1 models/life20/best.pt

# Bot vs bot bulk games
python play.py --p1 bolt_face --p2 aggro_goblin --games 1000 --life 20

# Train a new model
python train.py --life 20 --deck "20M,10B,10G" --episodes 50000 --save-dir models/life20

# Curriculum training (low life → high life)
python train.py --curriculum

# Run tournament
python tournament.py --life 20 --games 200

# Full analysis (tournament + deck optimization + charts)
python run_all.py
```

## System Architecture

### Game Engine (`mtg/`)

The rules engine faithfully implements MTG rules for the 3-card environment:

- **Turn structure**: Untap → Upkeep → Draw → Main 1 → Combat (Begin → Attackers → Blockers → Damage → End) → Main 2 → End → Cleanup
- **Priority system**: Correct priority cycling with stack — both players must pass in succession for the stack to resolve or the phase to advance
- **The Stack**: Spells go on the stack and can be responded to (e.g., bolt in response to bolt)
- **Combat**: Full declare attackers/blockers system with damage assignment
- **State-based actions**: Life ≤ 0 loses, creatures with lethal damage die, empty library loses
- **London mulligan**: Draw 7, decide keep/mulligan, put N cards on bottom after N mulligans
- **Mana system**: Auto-tapping Mountains (unambiguous in this card pool, but the mana pool abstraction supports future multi-color extensions)

Key files:
- `mtg/enums.py` — Phases, steps, card types, action types
- `mtg/cards.py` — Card definitions (`CardDef`, `CardInstance`), card registry
- `mtg/game.py` — `Game` class with full game loop, priority system, combat, mulligan, action generation

### Agents (`mtg/agents.py`)

| Agent | Strategy |
|-------|----------|
| `RandomAgent` | Uniform random from legal actions |
| `HumanAgent` | CLI interface with game state display |
| `BoltFaceAgent` | Play mountain, bolt opponent's face, attack all, never block |
| `AggroGoblinAgent` | Play mountain, cast goblins, attack all, never block |
| `MixedBoltFaceAgent` | Play mountain, cast goblins first then bolt face, attack all |
| `NNAgent` | DQN-trained neural network agent |

### Training (`mtg/training.py`, `train.py`)

DQN (Deep Q-Network) with:
- **Reward shaping**: Small intermediate rewards for life advantage changes + terminal ±1
- **Mixed opponent pool**: 65% fixed bots, 20% past checkpoints, 15% self-play
- **Experience replay**: 200K buffer, batch size 256
- **Target network**: Updated every 1000 steps
- **Epsilon-greedy**: 1.0 → 0.05 over 75% of training

Network: `33 → 128 → 128 → 295` (state features → Q-values with action masking)

State features (33-dim): life totals, hand composition, battlefield counts (untapped/tapped lands and creatures for both players), opponent hand size, library sizes, phase one-hot, turn info.

Action space (295 indices): pass, play land, cast bolt (18 target slots), cast goblin, declare attackers (256 subsets for up to 8 creatures), declare blockers (9 options per attacker), mulligan keep/take, choose bottom card.

### ELO System (`mtg/elo.py`, `tournament.py`)

Standard ELO (K=32) with round-robin tournaments. Supports arbitrary agent pools.

### Deck Optimizer (`mtg/deck.py`)

Grid search over deck compositions (mountains, bolts, goblins) with two-phase evaluation: quick screen then full evaluation of top candidates.

## Results

### Tournament Rankings (Life=20, deck=20M/10B/10G, 300 games/pair)

| Agent | ELO | vs Random | vs BoltFace | vs AggroGoblin | vs Mixed |
|-------|-----|-----------|-------------|----------------|----------|
| trained_v3 | 1459 | 98.7% | 96.3% | 92.0% | 54.7% |
| mixed | 1378 | 97.0% | 100.0% | 83.7% | — |
| trained_v1 | 1336 | 96.3% | 94.3% | 78.0% | 72.7% |
| aggro_goblin | 823 | 72.3% | 98.3% | — | 16.3% |
| bolt_face | 580 | 56.3% | — | 1.7% | 0.0% |
| random | 424 | — | 43.7% | 27.7% | 3.0% |

trained_v3 beats trained_v1 head-to-head 63.3% to 36.7%.

### Key Findings

1. **At life=3**: Bolt is king — BoltFace wins because one bolt is lethal. Going first with Mountain+Bolt is a guaranteed win (P≈100% of having both in opening hand).

2. **At life=20**: The trained agent slightly edges the MixedBoltFace strategy. The mixed strategy (play goblins for sustained damage, bolt face for burst) is near-optimal for this card pool.

3. **Optimal deck size**: Smaller decks (15 cards) outperform larger ones due to consistency — you draw your action cards faster.

4. **Optimal composition**: At life=20, approximately 11M/3B/1G (15 total) — very lean, bolt-heavy relative to deck size.

5. **Going first advantage**: Significant, especially at low life totals where tempo matters more.

### Charts

Generated in `results/`:
- `comprehensive_analysis.png` — ELO by life total, win rates, deck composition, key insights
- `tournament_matrix_life20.png` — Full win rate matrix
- `elo_history_life20.png` — ELO progression during training
- `training_loss_life20.png` — DQN training loss curve
- `optimal_decks.png` — Optimal deck composition by life total
- `game_lengths_life20.png` — Game length distributions

## Design for Extensibility

The system is designed to scale to more cards:

1. **Card Registry**: Add new `CardDef` entries to `CARD_REGISTRY` in `cards.py`
2. **Spell Effects**: Add resolution logic in `game.py` (dispatch by card name)
3. **Keywords**: `AbilityKeyword` enum supports haste, flying, first strike, etc.
4. **Mana System**: `ManaCost` supports all 5 colors + generic; `can_pay()` checks availability
5. **Card Types**: All standard types (land, creature, instant, sorcery, enchantment, artifact)
6. **Features**: `state_to_features()` can be extended for new card types
7. **Action Space**: Action encoding handles arbitrary spells, targets, and combat decisions

To add a new card (e.g., Shock — {R} instant, deal 2 damage):
```python
# In cards.py:
SHOCK = CardDef(name="Shock", card_types=frozenset({CardType.INSTANT}), mana_cost=ManaCost(red=1))
CARD_REGISTRY["Shock"] = SHOCK

# In game.py, add resolution:
elif card.name == "Shock":
    # deal 2 damage to target (same pattern as Lightning Bolt)
```

## File Structure

```
mtg/
├── __init__.py
├── enums.py          # Phase, Step, Zone, CardType, ActionType, etc.
├── cards.py          # CardDef, CardInstance, ManaCost, CARD_REGISTRY
├── game.py           # Game loop, priority, combat, stack, mulligan
├── agents.py         # Agent ABC, Human, Random, Fixed strategy, NN
├── features.py       # State featurization, action encoding
├── network.py        # QNetwork (PyTorch)
├── training.py       # DQN trainer, NNAgent, replay buffer
├── elo.py            # ELO ratings, tournament runner
├── deck.py           # Deck construction, grid search optimizer
└── analyze.py        # Chart generation (matplotlib)

play.py               # CLI: human/bot games
train.py              # Training entry point
tournament.py         # ELO tournament entry point
run_all.py            # Full analysis pipeline
tests/test_engine.py  # 18 rule correctness tests
models/               # Saved model checkpoints
results/              # Generated charts and reports
```
