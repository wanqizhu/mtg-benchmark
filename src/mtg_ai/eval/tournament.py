from __future__ import annotations

import math
from dataclasses import dataclass

from mtg_ai.bots.base import BotPolicy
from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.eval.elo import update_elo
from mtg_ai.game.session import GameSession


@dataclass(frozen=True)
class Entrant:
    name: str
    policy: BotPolicy
    deck: list[str]


@dataclass
class MatchSummary:
    entrant_a: str
    entrant_b: str
    games: int
    wins_a: int
    wins_b: int
    draws: int
    win_rate_a: float
    ci95_low: float
    ci95_high: float


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def play_match(
    entrant_a: Entrant,
    entrant_b: Entrant,
    games: int,
    config: GameConfig,
    seed: int,
) -> MatchSummary:
    wins_a = 0
    wins_b = 0
    draws = 0
    for game_idx in range(games):
        session = GameSession(
            decks={0: list(entrant_a.deck), 1: list(entrant_b.deck)},
            controllers={0: entrant_a.policy, 1: entrant_b.policy},
            config=GameConfig(
                starting_life=config.starting_life,
                opening_hand_size=config.opening_hand_size,
                max_mulligans=config.max_mulligans,
                random_seed=seed + game_idx,
            ),
            seed=seed + game_idx,
        )
        result = session.run(max_actions=250)
        if result.winner == 0:
            wins_a += 1
        elif result.winner == 1:
            wins_b += 1
        else:
            draws += 1
    effective_total = max(games, 1)
    win_rate_a = (wins_a + 0.5 * draws) / effective_total
    ci_low, ci_high = _wilson_interval(wins_a, effective_total)
    return MatchSummary(
        entrant_a=entrant_a.name,
        entrant_b=entrant_b.name,
        games=games,
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
        win_rate_a=win_rate_a,
        ci95_low=ci_low,
        ci95_high=ci_high,
    )


def round_robin(
    entrants: list[Entrant],
    games_per_pair: int,
    config: GameConfig,
    seed: int = 0,
) -> tuple[list[MatchSummary], dict[str, float]]:
    summaries: list[MatchSummary] = []
    ratings = {entrant.name: 1200.0 for entrant in entrants}
    match_seed = seed
    for idx in range(len(entrants)):
        for jdx in range(idx + 1, len(entrants)):
            summary = play_match(entrants[idx], entrants[jdx], games_per_pair, config=config, seed=match_seed)
            summaries.append(summary)
            score_a = (summary.wins_a + 0.5 * summary.draws) / summary.games
            new_a, new_b = update_elo(ratings[entrants[idx].name], ratings[entrants[jdx].name], score_a)
            ratings[entrants[idx].name] = new_a
            ratings[entrants[jdx].name] = new_b
            match_seed += games_per_pair + 37
    return summaries, ratings

