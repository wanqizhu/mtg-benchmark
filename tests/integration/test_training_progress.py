from __future__ import annotations

from pathlib import Path

from mtg_ai.bots.random_bot import RandomBot
from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.eval.tournament import Entrant, play_match
from mtg_ai.training.self_play import build_deck, train_self_play


def test_self_play_training_produces_bot_that_beats_random(tmp_path: Path) -> None:
    output_dir = tmp_path / "training"
    policy = train_self_play(
        episodes=30,
        alpha=0.08,
        epsilon=0.2,
        seed=21,
        output_dir=output_dir,
        eval_interval=10,
        eval_games=10,
        starting_life=12,
        opening_hand_size=7,
    )
    trained = Entrant("trained", policy, build_deck(12, 9, 9))
    random_baseline = Entrant("random", RandomBot(), build_deck(12, 9, 9))
    summary = play_match(
        trained,
        random_baseline,
        games=80,
        config=GameConfig(starting_life=12, opening_hand_size=7, random_seed=123),
        seed=123,
    )
    assert summary.win_rate_a >= 0.55
    assert (output_dir / "policy_final.json").exists()
    assert (output_dir / "training_metrics.csv").exists()

