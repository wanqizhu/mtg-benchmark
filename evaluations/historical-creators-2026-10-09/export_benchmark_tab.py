"""Export only original AI predictions and four historical creator mappings.

No fit or network request. Current evaluation outcomes come from the committed snapshot.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from refresh_benchmark_outcomes import apply_outcomes, write_export

ROOT = Path(__file__).resolve().parent
SOURCE_NAMES = {
    # Confirmed from the original September 24 prediction run's model metadata.
    'gpt-6': 'GPT-6 Sol',
    'alexnikolic': 'Alex Nikolic (Limited Level-Ups)',
}


def export(model_data, destination, outcomes=None):
    ai = json.loads(model_data.read_text())
    history = json.loads((ROOT / 'results/analysis.json').read_text())
    assert ai['manifest']['sha256'] == history['fra_truth_sha256']
    models = history['models']['models']
    expected = {'casualspike', 'alexnikolic', 'limitedresources', 'draftsim'}
    assert set(models) == expected
    ai_ids = [key for key, value in ai['sources'].items() if value['kind'] == 'model']
    assert len(ai_ids) == 6
    images = {c['name']: c for c in json.loads((ROOT / 'inputs/card_images.json').read_text())['cards']}
    lookup = {(r['creator'], r['card']): r for r in history['records']}
    shared = set.intersection(*[{r['card'] for r in history['records']
                               if r['creator'] == key and r['primary'] and r['prediction'] is not None}
                              for key in models])
    assert len(shared) == 266
    sources = [dict(id=key, name=SOURCE_NAMES.get(key, ai['sources'][key]['name']), kind='ai',
                    url=ai['sources'][key]['url'], commit=ai['sources'][key]['commit']) for key in ai_ids]
    audits = json.loads((ROOT / 'inputs/source_audit.json').read_text())['sources']
    fra_sources = {s['id']: s for s in json.loads((ROOT / 'inputs/fra_ratings.json').read_text())['sources']}
    current_labels = {'casualspike': 'Ratings sheet', 'alexnikolic': 'Tier list',
                      'limitedresources': 'Commons and uncommons review', 'draftsim': 'Bot pick order'}
    for key, model in models.items():
        fields = ['set', 'release', 'published_or_frozen', 'basis', 'url', 'rated_main', 'complete', 'notes']
        collection = [{k: row.get(k) for k in fields} for row in audits if row['creator'] == key and row['admitted']]
        current_sources = [dict(label=current_labels[key], url=fra_sources[key]['url'])]
        if fra_sources[key].get('rare_episode_url'):
            current_sources.append(dict(label='Rares and mythics review', url=fra_sources[key]['rare_episode_url']))
        sources.append(dict(id=key, name=SOURCE_NAMES.get(key, model['name']), kind='creator', model=model,
                            collection=collection, url=fra_sources[key]['url'],
                            current_sources=current_sources,
                            provisional=key == 'draftsim', fra_provenance=fra_sources[key].get('caveat', '')))
    cards = []
    for card in ai['cards']:
        assert images[card['name']]['scryfall_uri'] == card['url']
        values = {}
        for key in ai_ids:
            raw = card['values'].get(key, {}).get('raw')
            values[key] = dict(prediction=raw, grade=None)
            if card['name'] in shared:
                assert raw is not None
        for key in models:
            row = lookup[key, card['name']]
            if card['actual'] is not None:
                assert abs(row['actual'] - card['actual']) < 1e-8
            values[key] = dict(prediction=row['prediction'], grade=row['native_grade'],
                               grade_used=row['grade_used'], contexts=row['contexts'],
                               applied_contexts=row['applied_contexts'], rating_structure=row['rating_structure'],
                               support=row['support_notes'], source_url=row['source_url'])
        cards.append(dict(name=card['name'], rarity=card['rarity'], color=card['color'],
                          set=card['set'], cost=card['cost'], type=card['type'], oracle=card['oracle'],
                          url=card['url'], actual=card['actual'], games=card['n'],
                          shared=card['name'] in shared, values=values,
                          images=images[card['name']]['images']))
    assert len(cards) == 295 and len({c['name'] for c in cards}) == 295
    for source in sources:
        pool = [c for c in cards if c['shared']]
        predicted = np.array([c['values'][source['id']]['prediction'] for c in pool])
        actual = np.array([c['actual'] for c in pool])
        source['score'] = dict(n=len(pool), pearson=float(pearsonr(predicted, actual).statistic),
                               rmse=float(np.sqrt(np.mean((predicted - actual) ** 2))))
        if source['kind'] == 'creator':
            prior = next(r for r in history['scores'] if r['creator'] == source['id'] and r['pool'] == 'common cards across all active creators')
            assert abs(prior['pearson'] - source['score']['pearson']) < 1e-10
            assert abs(prior['rmse_pp'] - source['score']['rmse']) < 1e-10
    # Keep table/CSV columns grouped, with each group ranked by Pearson r.
    # Filtering this ordered list also ranks the creator detail sections.
    sources.sort(key=lambda s: (s['kind'] != 'ai', -s['score']['pearson'], s['name']))
    data = dict(schema_version=1, expansion='FRA', name='Reality Fracture',
                shared_cards=len(shared), catalog_cards=len(cards), sources=sources, cards=cards,
                outcome=dict(url=ai['manifest']['page_url'], retrieved_at=ai['manifest']['retrieved_at'],
                             sha256=history['fra_truth_sha256'], event='PremierDraft',
                             scope=ai['manifest']['scope']),
                fit=dict(training_observations=sum(m['train_n'] for m in models.values()),
                         historical_target='FIRST_WEEK', minimum_gih_games=500,
                         publication_cutoff='Within three calendar days of Arena launch',
                         retrospective=True, outcome_access_guard_verified=True))
    if outcomes is not None:
        data = apply_outcomes(data, json.loads(outcomes.read_text()))
    write_export(data, destination)
    print('Exported', len(cards), 'cards,', len(sources), 'sources,', len(shared), 'shared cards;', destination.stat().st_size, 'bytes')
    for s in sorted(data['sources'], key=lambda s: -s['score']['pearson']):
        print(s['name'], f"r={s['score']['pearson']:.6f}", f"RMSE={s['score']['rmse']:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--outcomes', type=Path, default=ROOT / 'inputs/fra_outcomes.json')
    args = parser.parse_args()
    export(args.model_data, args.output, args.outcomes)
