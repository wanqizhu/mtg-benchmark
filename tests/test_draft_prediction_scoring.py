import importlib.util
import math
from pathlib import Path

import pytest


def test_scores_use_each_sources_coverage_including_unshared_cards():
    path = Path(__file__).resolve().parents[1] / 'evaluations/historical-creators-2026-10-09/refresh_benchmark_outcomes.py'
    spec = importlib.util.spec_from_file_location('draft_outcomes', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def card(actual, model, creator, *, games=500, basic=False):
        return dict(actual=actual, games=games, type='Basic Land' if basic else 'Creature',
                    shared=False, values={'model': dict(prediction=model), 'creator': dict(prediction=creator)})

    data = dict(cards=[card(50, 52, 50), card(60, 58, None), card(70, 68, 72),
                       card(None, 90, 90), card(95, 95, 95, games=499),
                       card(30, 30, 30, basic=True)],
                sources=[dict(id='model', kind='ai', name='Model'),
                         dict(id='creator', kind='creator', name='Creator')])
    module.rescore_sources(data)
    scores = {s['id']: s['score'] for s in data['sources']}
    assert scores['model']['n'] == 3
    assert scores['model']['rmse'] == pytest.approx(2)
    assert scores['creator']['n'] == 2
    assert scores['creator']['rmse'] == pytest.approx(math.sqrt(2))
    assert scores['creator']['pearson'] == pytest.approx(1)
    assert data['eligible_outcomes'] == 3
