"""Refresh evaluation outcomes while retaining every prediction and fitted model."""
import argparse
import copy
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent
API_URL = 'https://www.17lands.com/api/card_data?expansion=FRA&event_type=PremierDraft&time_period=ALL_TIME'
PAGE_URL = 'https://www.17lands.com/card_data?expansion=FRA&format=PremierDraft&time_period=ALL_TIME'


def norm(name):
    return name.split(' // ')[0].replace('’', "'").replace('−', '-').strip().casefold()


def stats_hash(cards):
    return hashlib.sha256(json.dumps(cards, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def extract_snapshot(data, raw, manifest):
    assert hashlib.sha256(raw).hexdigest() == manifest['sha256']
    payload = json.loads(raw)
    rows = payload['data']
    lookup = {norm(row['name']): row for row in rows}
    catalog = {norm(card['name']): card for card in data['cards']}
    assert len(lookup) == len(rows) == 290
    assert len(catalog) == len(data['cards']) == 295
    assert set(lookup) <= set(catalog), 'Unmatched outcome cards'
    cards = []
    for card in data['cards']:
        row = lookup.get(norm(card['name']))
        assert row is not None or card['type'].startswith('Basic Land')
        games = 0 if row is None else row['ever_drawn_game_count']
        rate = None if row is None else row['ever_drawn_win_rate']
        assert isinstance(games, int) and games >= 0
        assert rate is None or 0 <= rate <= 1
        actual = None if rate is None or games < 500 else 100 * rate
        cards.append(dict(name=card['name'], actual=actual, games=games))
    cards.sort(key=lambda c: c['name'])
    return dict(schema_version=1, expansion='FRA',
                outcome=dict(url=PAGE_URL, api_url=API_URL, event='PremierDraft',
                             time_period='ALL_TIME', retrieved_at=manifest['retrieved_at'],
                             sha256=manifest['sha256'], card_stats_sha256=stats_hash(cards),
                             scope='Cumulative ALL_TIME PremierDraft GIH, all tracked users and all deck colors; not a first-week-only window.'),
                minimum_gih_games=500,
                shared_pool=sorted(c['name'] for c in data['cards'] if c['shared']),
                cards=cards)


def apply_outcomes(data, snapshot):
    assert snapshot['schema_version'] == 1 and snapshot['expansion'] == data['expansion'] == 'FRA'
    assert snapshot['outcome']['event'] == 'PremierDraft'
    assert snapshot['outcome']['time_period'] == 'ALL_TIME'
    assert snapshot['minimum_gih_games'] == 500
    assert stats_hash(snapshot['cards']) == snapshot['outcome']['card_stats_sha256']
    assert sorted(c['name'] for c in data['cards'] if c['shared']) == snapshot['shared_pool']
    assert len(snapshot['shared_pool']) == data['shared_cards'] == 266
    lookup = {c['name']: c for c in snapshot['cards']}
    assert len(lookup) == len(snapshot['cards']) == data['catalog_cards'] == 295
    assert set(lookup) == {c['name'] for c in data['cards']}
    refreshed = copy.deepcopy(data)
    for card in refreshed['cards']:
        row = lookup[card['name']]
        actual, games = row['actual'], row['games']
        assert isinstance(games, int) and games >= 0
        assert actual is None or (np.isfinite(actual) and 0 <= actual <= 100 and games >= 500)
        assert not card['shared'] or actual is not None, 'Previously scored card unavailable'
        card.update(actual=actual, games=games)
    pool = [c for c in refreshed['cards'] if c['shared']]
    actual = np.array([c['actual'] for c in pool], dtype=float)
    for source in refreshed['sources']:
        predictions = np.array([c['values'][source['id']]['prediction'] for c in pool], dtype=float)
        assert np.isfinite(predictions).all()
        source['score'] = dict(n=len(pool), pearson=float(pearsonr(predictions, actual).statistic),
                               rmse=float(np.sqrt(np.mean((predictions - actual) ** 2))))
    refreshed['sources'].sort(key=lambda s: (s['kind'] != 'ai', -s['score']['pearson'], s['name']))
    refreshed['outcome'] = copy.deepcopy(snapshot['outcome'])
    return refreshed


def write_export(data, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False,
                                     separators=(',', ':')) + '\n')
    headers = ['Card', 'Set', 'Rarity', 'Observed GIH (%)', 'GIH games', 'Shared comparison']
    for source in data['sources']:
        headers.append(source['name'] + ' prediction (%)')
        if source['kind'] == 'creator':
            headers.append(source['name'] + ' raw grade')
    with destination.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(headers)
        for card in data['cards']:
            row = [card['name'], card['set'], card['rarity'], card['actual'], card['games'], card['shared']]
            for source in data['sources']:
                row.append(card['values'][source['id']]['prediction'])
                if source['kind'] == 'creator':
                    row.append(card['values'][source['id']]['grade'])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Existing benchmark JSON with frozen predictions')
    parser.add_argument('--output', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot', type=Path, help='Committed normalized outcome snapshot; defaults to inputs/fra_outcomes.json')
    group.add_argument('--truth', type=Path, help='Local raw 17Lands response; requires --manifest')
    group.add_argument('--fetch', action='store_true', help='Pull current outcomes without fitting any model')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--save-snapshot', type=Path, default=ROOT / 'inputs/fra_outcomes.json')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if args.fetch or args.truth:
        if args.fetch:
            with urlopen(Request(API_URL, headers={'Cache-Control': 'no-cache'}), timeout=60) as response:
                raw = response.read()
            manifest = dict(retrieved_at=datetime.now(timezone.utc).isoformat(),
                            sha256=hashlib.sha256(raw).hexdigest())
        else:
            if args.manifest is None:
                parser.error('--truth requires --manifest with the actual pull timestamp and response hash')
            raw = args.truth.read_bytes()
            manifest = json.loads(args.manifest.read_text())
        snapshot = extract_snapshot(data, raw, manifest)
    else:
        snapshot = json.loads((args.snapshot or ROOT / 'inputs/fra_outcomes.json').read_text())
    refreshed = apply_outcomes(data, snapshot)
    if args.fetch or args.truth:
        args.save_snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.save_snapshot.write_text(json.dumps(snapshot, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    write_export(refreshed, args.output)
    print('Outcome pull:', refreshed['outcome']['retrieved_at'])
    print('Fixed comparison pool:', refreshed['shared_cards'], 'cards')
    for source in sorted(refreshed['sources'], key=lambda s: -s['score']['pearson']):
        print(source['name'], f"r={source['score']['pearson']:.6f}", f"RMSE={source['score']['rmse']:.6f}")


if __name__ == '__main__':
    main()
