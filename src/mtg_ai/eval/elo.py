from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EloRating:
    value: float = 1200.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float, k_factor: float = 24.0) -> tuple[float, float]:
    expected_a = expected_score(rating_a, rating_b)
    expected_b = expected_score(rating_b, rating_a)
    new_a = rating_a + k_factor * (score_a - expected_a)
    score_b = 1.0 - score_a
    new_b = rating_b + k_factor * (score_b - expected_b)
    return new_a, new_b

