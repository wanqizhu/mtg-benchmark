from __future__ import annotations

from mtg_ai.bots.fixed_bots import BoltFaceBot, GoblinAggroBot, MixedBoltFaceBot
from mtg_ai.bots.random_bot import RandomBot
from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.eval.tournament import Entrant, play_match


def _deck(mountains: int, bolts: int, goblins: int) -> list[str]:
    return ["Mountain"] * mountains + ["Lightning Bolt"] * bolts + ["Raging Goblin"] * goblins


def test_fixed_bots_run_and_produce_expected_relative_strengths() -> None:
    config = GameConfig(starting_life=10, opening_hand_size=7, random_seed=12)
    random_entrant = Entrant("random", RandomBot(), _deck(10, 10, 10))
    bolt_entrant = Entrant("bolt", BoltFaceBot(), _deck(12, 18, 0))
    goblin_entrant = Entrant("goblin", GoblinAggroBot(), _deck(12, 0, 18))
    mixed_entrant = Entrant("mixed", MixedBoltFaceBot(), _deck(12, 9, 9))

    summary_bolt_vs_random = play_match(bolt_entrant, random_entrant, games=80, config=config, seed=100)
    summary_mixed_vs_random = play_match(mixed_entrant, random_entrant, games=80, config=config, seed=200)
    summary_mixed_vs_goblin = play_match(mixed_entrant, goblin_entrant, games=80, config=config, seed=300)

    assert summary_bolt_vs_random.win_rate_a > 0.5
    assert summary_mixed_vs_random.win_rate_a > 0.5
    assert summary_mixed_vs_goblin.win_rate_a > 0.45

