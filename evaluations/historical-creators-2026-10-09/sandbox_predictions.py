"""Add frozen sandbox forecasts to the existing comparison; never fit predictions."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from refresh_benchmark_outcomes import rescore_sources

ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT / 'inputs/sandbox_predictions.json'
GROUPS = [
    dict(id='internet', marker='*', description='With internet, so it could see some community tierlists before set release.'),
    dict(id='sandbox', marker='†', description='Without internet; just cards, rules, and historical data.'),
]


def norm(name):
    return name.split(' // ')[0].replace('’', "'").replace('−', '-').strip().casefold()


def add_sandbox_sources(data, bundle):
    """Preserve every existing forecast and creator fit; score each source's coverage."""
    assert bundle['schema_version'] == 1 and bundle['expansion'] == data['expansion'] == 'FRA'
    result = copy.deepcopy(data)
    result['prediction_groups'] = copy.deepcopy(GROUPS)
    result['sandbox_audit'] = copy.deepcopy(bundle['audit'])
    result['sandbox_prompt'] = bundle['prompt']
    catalog = {norm(c['name']): c for c in result['cards']}
    assert len(catalog) == len(result['cards']), 'Ambiguous front-face card names'
    existing = {s['id']: s for s in result['sources']}
    for source in result['sources']:
        if source['kind'] == 'ai' and source.get('prediction_group') != 'sandbox':
            source.update(prediction_group='internet', prediction_marker='*')
    for model in bundle['models']:
        sid = model['id']
        assert sid not in existing or existing[sid].get('prediction_group') == 'sandbox'
        rates = {norm(k): v for k, v in model['predictions'].items()}
        assert len(rates) == len(model['predictions']) and set(rates) <= set(catalog)
        assert all(isinstance(v, (int, float)) and np.isfinite(v) and 0 <= v <= 100 for v in rates.values())
        for key, card in catalog.items():
            card['values'][sid] = dict(prediction=rates.get(key), grade=None)
            assert not card['shared'] or rates.get(key) is not None, f'Missing shared card: {sid}, {key}'
        source = {k: copy.deepcopy(v) for k, v in model.items() if k != 'predictions'}
        source.update(kind='ai', prediction_group='sandbox', prediction_marker='†')
        if sid in existing:
            result['sources'] = [source if s['id'] == sid else s for s in result['sources']]
        else:
            result['sources'].append(source)
        existing[sid] = source
    assert len({s['id'] for s in result['sources']}) == len(result['sources'])
    assert result['shared_cards'] == sum(c['shared'] for c in result['cards'])
    return rescore_sources(result)


if __name__ == '__main__':
    from refresh_benchmark_outcomes import write_export
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    data = add_sandbox_sources(json.loads(args.input.read_text()), json.loads(args.bundle.read_text()))
    write_export(data, args.output)
    print(f"Exported {len(data['sources'])} sources / {data['catalog_cards']} cards / per-source scores")
