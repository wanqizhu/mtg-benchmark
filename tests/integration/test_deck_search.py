from __future__ import annotations

from mtg_ai.bots.fixed_bots import MixedBoltFaceBot
from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.deck.search import exhaustive_search, starter_deck
from mtg_ai.deck.space import DeckComposition
from mtg_ai.eval.tournament import Entrant, play_match


def test_deck_search_finds_deck_better_than_starter_in_low_life_setting() -> None:
    config = GameConfig(starting_life=8, opening_hand_size=7, random_seed=88)
    policy = MixedBoltFaceBot()
    best = exhaustive_search(
        total_cards=12,
        candidate_policy=policy,
        config=config,
        seed=88,
        games_per_opponent=20,
    )
    assert isinstance(best.composition, DeckComposition)

    candidate = Entrant("candidate", policy, best.composition.to_deck())
    starter = Entrant("starter", policy, starter_deck())
    summary = play_match(candidate, starter, games=60, config=config, seed=99)
    assert summary.win_rate_a >= 0.5

