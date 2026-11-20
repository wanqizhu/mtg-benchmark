"""Freeze TRK forecasts separately from the 17Lands snapshot used to score them."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from historical_creators import norm, prepare, restore_model
from refresh_benchmark_outcomes import apply_outcomes, extract_snapshot, rescore_sources, write_export


MIN_GIH = 500
JOINT_EDITION = '2026-11-08T22:18:32'
TIER_ROOT = 'https://www.17lands.com/tier_list/'
ALEX = TIER_ROOT + '5d052764d08f49238b8232f591b46507'
JOINT = TIER_ROOT + '03a686ab050b42d5a62bce8902e7f0df'
LR_CU = 'https://lrcast.com/limited-resources-878-star-trek-set-review-commons-and-uncommons/'
LR_RM = 'https://lrcast.com/limited-resources-879-star-trek-set-review-rare-and-mythic-rare/'
CHUNK = 'https://chunk.science/mtga-star-trek.html'
CHUNK_ARCHIVE = 'https://web.archive.org/web/20261105085610/https://chunk.science/mtga-star-trek.html'
DRAFTSIM = 'https://draftsim.com/TRK-pick-order/'
DS_ARCHIVE = 'https://web.archive.org/web/20261107181328/https://draftsim.com/TRK-pick-order/'
REPO = 'https://github.com/wanqizhu/mtg-win-rate-eval'
MODELS = {
    4: ('trk-fable', 'Claude Fable 5.1', 'fd8a1df07313447bfe754d3696447c8e41db2568', 'trk_prediction/trk_gih_predictions.csv', 'name', 'predicted_gih_wr'),
    5: ('trk-muse', 'Muse Wonder 1.0', '9914c5aee9bb33fc68a9730d3bc977ac472fe219', 'predictions.csv', 'card_name', 'predicted_gih_win_rate'),
    6: ('trk-opus', 'Claude Opus 5.5', 'e37911f24f21bde9323522b160cb2a5e05ba6b29', 'predictions/trk_gih_predictions.csv', 'name', 'gih_wr_pred'),
    7: ('trk-grok', 'Grok 4.7', 'dceec53436e6a6d22bf31c4fa598c5bbc49194a8', 'predictions/trk_premier_draft_gih.csv', 'name', 'gih_win_rate'),
    9: ('trk-sol', 'GPT-6.1 Sol', '1eefe40ee6de02a85aed0edd9df2afafc8de4031', 'predictions.csv', 'card_name', 'predicted_gih_win_rate'),
    10: ('trk-astra', 'GPT-6 Astra', '49870f6dce2466652895a75a84a1f23861dcfe24', 'predictions.csv', 'name', 'predicted_gih_win_rate'),
    11: ('trk-muse-spark', 'Muse Spark 1.3', 'fb9395049d79634fb7bd66b83b2fb42429f1e0a1', 'trk_gih_predictions.csv', 'card', 'predicted_gih_wr'),
    12: ('trk-gemini-argon', 'Gemini 4 Argon', '134c0f49a82adb06af0ae5466a3819ccb859085c', 'artifacts/trk_gih_predictions.csv', 'card', 'predicted_gih_wr'),
}
POST_RELEASE_PRS = {11, 12}


def read_json(p):
    return json.loads(p.read_text())


def document(path):
    raw = path.read_bytes()
    return (gzip.decompress(raw) if raw.startswith(b'\x1f\x8b') else raw).decode()


def collector_data(path):
    text = document(path)
    return json.JSONDecoder().raw_decode(text[text.index('{'):])[0]


def stamp(path):
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec='seconds')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def draftsim_ratings(path):
    text = document(path)
    lookup = re.search(r'"\.\./data/TRK\.txt":([\w$]+)', text)
    if not lookup:
        raise ValueError(f'No TRK ratings asset in {path.name}')
    match = re.search(r'(?<![\w$])' + re.escape(lookup[1]) + r'=`([^`]+)`', text)
    table = match[1].replace('\\r', '').replace('\\t', '\t').replace('\\n', '\n')
    rows = list(csv.DictReader(io.StringIO(table), delimiter='\t'))
    result = {norm(row['Name'].replace('_', ' ')): row['Rating'] for row in rows}
    assert len(rows) == len(result) == 341
    assert {k for k,v in result.items() if v == '-1'} == {'plains','island','swamp','mountain','forest'}
    assert all(0 <= float(v) <= 5 for v in result.values() if v != '-1')
    return result


def tier_rows(data, id_, url):
    assert data['expansion'] == 'TRK'
    result = []
    for card in data['ratings']:
        native = card['tier']
        if native in (None, 'TBD'):
            continue
        flags = card['flags']
        contexts = [{'kind': kind, 'native': None} for flag, kind in
                    [('buildaround', 'build_around'), ('synergy', 'synergy')] if flags.get(flag)]
        result.append(dict(name=card['name'], creator=id_, native=native, ordinary_native=native,
                           contexts=contexts, source_url=url))
    return result


def catalog_row(card):
    faces = card.get('card_faces', [card])
    images = [dict(name=f['name'], url=f['image_uris']['normal']) for f in faces if f.get('image_uris')]
    if not images and card.get('image_uris'):
        images = [dict(name=card['name'], url=card['image_uris']['normal'])]
    return dict(name=card['name'], set='TRK', rarity=card['rarity'],
                color=''.join(card.get('colors', card['color_identity'])) or 'C',
                cost=' // '.join(f.get('mana_cost', '') for f in faces), type=card['type_line'],
                oracle='\n//\n'.join(f.get('oracle_text', '') for f in faces),
                url=card['scryfall_uri'].split('?')[0], images=images,
                collector_number=int(card['collector_number']))


def read_model_source(pr, blob, main):
    id_, name, commit, path, card_key, rate_key = MODELS[pr]
    rows = list(csv.DictReader(io.StringIO(blob.decode('utf-8-sig'))))
    values = {norm(row[card_key]): float(100 * Decimal(row[rate_key])) for row in rows}
    assert len(rows) == len(values) and main.keys() <= values.keys(), f'Invalid coverage in PR #{pr}'
    assert len(values) == (341 if pr in {5, 6, 12} else 311)
    assert all(0 <= rate <= 100 for rate in values.values())
    source = dict(id=id_, kind='ai', name=name, url=f'{REPO}/pull/{pr}',
                  commit=commit, source_file=path, sha256=digest(blob),
                  prediction_group='sandbox', grade_rows=len(values),
                  predictions={card_name: values[key] for key, card_name in main.items()})
    if pr in POST_RELEASE_PRS:
        source['post_release'] = True
    return source


def freeze(raw, git_dir, output):
    raw_cards = read_json(raw/'trk-cards.json')['data'] + read_json(raw/'trk-cards2.json')['data']
    catalog = sorted(map(catalog_row, raw_cards), key=lambda c:c['collector_number'])
    main = {norm(c['name']): c['name'] for c in catalog}
    assert [c['collector_number'] for c in catalog] == list(range(1, 312)) and len(main) == 311
    assert all('arena' in c['games'] for c in raw_cards)
    sources = []
    for pr, (_, _, commit, path, _, _) in MODELS.items():
        blob = subprocess.check_output(['git', '-C', str(git_dir), 'show', f'{commit}:{path}'])
        sources.append(read_model_source(pr, blob, main))

    alex = read_json(raw/'alex.json')
    joint = read_json(raw/'llu-joint.json')
    assert alex['last_updated'] < '2026-11-10'
    assert not any(r['date'] >= '2026-11-10' for r in alex['history'])
    joint_edits = [r['date'] for r in joint['history'] if r['date'] >= '2026-11-10']
    assert joint_edits
    alex_rows = tier_rows(alex, 'alexnikolic', ALEX)
    archived = collector_data(raw/'chunk-nov05.js')
    alex_archived = {norm(c['name']): c['ratings']['alexnikolic']['native'] for c in archived['cards']
                     if 'alexnikolic' in c['ratings']}
    assert alex_archived == {norm(r['name']): r['native'] for r in alex_rows}
    creators = [dict(id='alexnikolic', url=ALEX,
                     current_sources=[dict(label='Alex’s individual list', url=ALEX),
                                      dict(label='November 5 archive', url=CHUNK_ARCHIVE),
                                      dict(label='Joint list (excluded)', url=JOINT)],
                     provenance='All 201 individual grades match the November 5 archive, and 17Lands records November 2 as the last edit. The separate joint Alex/Marc list has postlaunch edits and is excluded.',
                     evidence=dict(last_updated=alex['last_updated'], archive_matching_grades=len(alex_archived),
                                   joint_postlaunch_edits=joint_edits),
                     rows=alex_rows)]
    collector = collector_data(raw/'chunk-data.js')
    lr_archived = {norm(c['name']): c['ratings']['limitedresources']['native'] for c in archived['cards']
                   if 'limitedresources' in c['ratings']}
    lr_evidence = next(s for s in collector['videoRatings'] if s['sourceKey']=='limitedresources')
    timestamps = {r['cardId']: r for r in lr_evidence['ratings']}
    lr_rows = []
    for card in collector['cards']:
        grade = card['ratings'].get('limitedresources')
        if not grade or norm(card['name']) not in main:
            continue
        timing = timestamps[card['id']]
        url = f"https://www.youtube.com/watch?v={timing['videoId']}&t={int(timing['timestampSeconds'])}s"
        lr_rows.append(dict(name=card['name'], creator='limitedresources', native=grade['native'],
                            ordinary_native=grade['ordinaryNative'], contexts=grade['contexts'], source_url=url))
    lr_grades = {norm(r['name']): r['native'] for r in lr_rows}
    assert lr_archived.keys() <= lr_grades.keys()
    corrections = [name for name, native in lr_archived.items() if lr_grades[name]!=native]
    assert len(corrections)==19 and all(lr_archived[k].replace('++','+') in (lr_grades[k], f'{lr_grades[k]} / {lr_grades[k]}') for k in corrections)
    episodes = {str(n): {k: read_json(raw/f'lr{n}-wp.json')[0][k] for k in ('date_gmt', 'modified_gmt')}
                for n in (878, 879)}
    creators.append(dict(id='limitedresources', url=LR_CU,
                         current_sources=[dict(label='Commons and uncommons', url=LR_CU),
                                          dict(label='Rares and mythics', url=LR_RM),
                                          dict(label='Transcribed grades', url=CHUNK)],
                         provenance='Original episodes, posted before launch; no later page edit in the publisher metadata. The commons/uncommons archive differs on 19 documented caption corrections (not changes by the hosts). Rare grades are a transcription of the dated episode.',
                         evidence=dict(episodes=episodes, transcription_captured=collector['generatedAt'],
                                       archived_common_uncommon_grades=len(lr_archived), caption_corrections=len(corrections)), rows=lr_rows))

    current = draftsim_ratings(raw/'draftsim.js')
    old_path = next((raw/name for name in ('draftsim-nov07.js','draftsim-old-direct.js')
                     if (raw/name).exists() and (raw/name).stat().st_size > 100000), None)
    if old_path:
        old = draftsim_ratings(old_path)
        changed = sum(Decimal(old[k]) != Decimal(current[k]) for k in old.keys() & current.keys())
        assert old.keys() == current.keys()
        ratings = old
        provenance = f'Used the November 7 archived rating asset. Of 336 rated cards, {changed} differ in the current asset. The five basic lands were unrated in both.'
    else:
        changed = None
        ratings = current
        provenance = 'Frozen current bot ratings. A November 7 archive of the landing page exists, but its rating asset was not available to verify whether the current grades changed.'
    assert main.keys() <= ratings.keys(), sorted(main.keys()-ratings.keys())
    ds_rows = [dict(name=name, creator='draftsim', native=ratings[k], ordinary_native=ratings[k],
                    contexts=[], source_url=DS_ARCHIVE if old_path else DRAFTSIM)
               for k,name in main.items() if ratings[k] != '-1']
    creators.append(dict(id='draftsim', url=DRAFTSIM,
                         current_sources=[dict(label='Bot pick order', url=DRAFTSIM),
                                          dict(label='November 7 archive', url=DS_ARCHIVE)],
                         provenance=provenance, evidence=dict(archive_used=bool(old_path), changes=changed,
                         selected_asset_sha256=digest((old_path or raw/'draftsim.js').read_bytes())), rows=ds_rows))
    missing = [dict(id='casualspike', name='Casual Spike', url='https://www.reddit.com/user/Casual_Spike/',
                    provenance='No Star Trek ratings by Casual Spike could be verified. The FRA sheet covers Reality Fracture only; the Star Trek list by Roberto Castiello is a different reviewer.')]
    optional_casual = raw/'casual-trk-verified.json'
    if optional_casual.exists():
        verified = read_json(optional_casual)
        assert verified['id'] == 'casualspike' and verified['rows'] and verified['provenance']
        creators.append(verified)
        missing = []
    for creator in creators:
        counts = Counter(norm(r['name']) for r in creator['rows'])
        assert all(x == 1 for x in counts.values())
    evidence = {name: digest((raw/name).read_bytes()) for name in ('alex.json','llu-joint.json','chunk-data.js','chunk-nov05.js','lr878-wp.json','lr879-wp.json','draftsim.js','trk-cards.json','trk-cards2.json')}
    write_json(output, dict(schema_version=1, expansion='TRK', cards=catalog,
                            models=sources, creators=creators, missing=missing, evidence=evidence))


def prelaunch_joint(path):
    snapshot = read_json(path)
    expected = {}
    for revision in sorted(snapshot['history'], key=lambda r: r['date']):
        if revision['date'] <= JOINT_EDITION:
            expected.update((change['card_id'], change['to_tier']) for change in revision['changes'])
    assert len(expected) == 336
    assert all(expected[r['card_id']] == r['tier'] for r in snapshot['ratings'])
    url = 'https://www.17lands.com/data/tier_list/' + JOINT.rsplit('/', 1)[1] + '?date=' + JOINT_EDITION
    main = {r['name'] for r in snapshot['ratings'] if r['expansion'] == 'TRK'}
    rows = [r for r in tier_rows(snapshot, 'alexnikolic', url) if r['name'] in main]
    assert len(rows) == 306
    return dict(id='alexnikolic', url=JOINT,
                current_sources=[dict(label='Tier list', url=JOINT),
                                 dict(label='November 8 snapshot JSON', url=url)],
                provenance='November 8, 2026 tier list snapshot at 22:18:32 UTC, before the November 10 Arena launch. All 336 grades match the change history through that timestamp. The 306 main-set grades use Alex’s existing historical mapping.',
                evidence=dict(edition_timestamp=JOINT_EDITION, main_set_grades=306,
                              snapshot_sha256=digest(path.read_bytes())), rows=rows)


def build(frozen, fra_original, trk_raw, fra_raw, site_dir, alex_ratings=None):
    data = read_json(frozen)
    if alex_ratings is not None:
        data['creators'] = [prelaunch_joint(alex_ratings) if c['id'] == 'alexnikolic' else c
                            for c in data['creators']]
    baseline = read_json(fra_original)
    fits = {s['id']: s for s in baseline['sources'] if s['kind'] == 'creator'}
    if fra_raw is not None:
        snapshot = extract_snapshot(baseline, fra_raw.read_bytes(), dict(sha256=digest(fra_raw.read_bytes()), retrieved_at=stamp(fra_raw)))
        write_export(apply_outcomes(baseline, snapshot), site_dir/'data/draft-predictions-fra-current.json')

    predictions = {s['id']: {k: dict(prediction=v, grade=None) for k,v in s['predictions'].items()}
                   for s in data['models']}
    sources = [{k:v for k,v in s.items() if k != 'predictions'} for s in data['models']]
    for source in sources:
        source['prediction_marker'] = '†'
    for creator in data['creators']:
        id_ = creator['id']
        source = {**fits[id_], **{k:v for k,v in creator.items() if k!='rows'}}
        rows = [prepare(r) for r in creator['rows']]
        vals, notes = restore_model(source['model']).predict(rows)
        output = {}
        for r, val, support in zip(rows, vals, notes):
            assert val is not None, (id_, r['name'], r['native'])
            output[norm(r['name'])] = dict(prediction=val, grade=r['native'], grade_used=r['grade_used'],
                                          contexts=r['raw_buckets'], applied_contexts=r['buckets'],
                                          rating_structure=r['rating_structure'], support=support, source_url=r['source_url'])
        source['grade_rows'] = sum(norm(c['name']) in output for c in data['cards'])
        predictions[id_] = {c['name']: output[norm(c['name'])] for c in data['cards'] if norm(c['name']) in output}
        sources.append(source)

    raw = trk_raw.read_bytes()
    rows = json.loads(raw)['data']
    results = {norm(r['name']): r for r in rows}
    assert len(rows) == len(results)
    cards = data['cards']
    for card in cards:
        name = card['name']
        row = results.get(norm(name), {})
        games, rate = row.get('ever_drawn_game_count') or 0, row.get('ever_drawn_win_rate')
        assert rate is None or 0 <= rate <= 1
        actual = 100 * rate if rate is not None and games >= MIN_GIH else None
        values = {s['id']: predictions[s['id']].get(name, dict(prediction=None, grade=None)) for s in sources}
        shared = actual is not None and all(v['prediction'] is not None for v in values.values())
        card.update(actual=actual, games=games, shared=shared, values=values)
    pool = [c for c in cards if c['shared']]
    payload = dict(schema_version=1, expansion='TRK', name='Star Trek',
                   shared_cards=len(pool), catalog_cards=len(cards), sources=sources, cards=cards,
                   missing_creators=data['missing'], eligible_outcomes=sum(c['actual'] is not None for c in cards),
                   prediction_groups=baseline['prediction_groups'],
                   sandbox_prompt=(Path(__file__).parent/'preview-inputs/trk-sandbox-prompt.txt').read_text().strip(),
                   sandbox_prompt_source='Original Codex Cloud prompt supplied by the user on November 19, 2026',
                   outcome=dict(url='https://www.17lands.com/card_data?expansion=TRK&format=PremierDraft&time_period=ALL_TIME',
                                retrieved_at=stamp(trk_raw), sha256=digest(raw), minimum_gih=MIN_GIH))
    rescore_sources(payload)
    write_export(payload, site_dir/'data/draft-predictions-trk.json')
    for s in sorted(sources, key=lambda s:-s['score']['pearson']):
        print(f"{s['name']}: {s['score']['n']} scored cards; r={s['score']['pearson']:.3f}, RMSE={s['score']['rmse']:.2f}")
    print(f"Eligible observed cards: {payload['eligible_outcomes']}; each source uses its own rated cards")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest='command', required=True)
    freeze_cmd = cmd.add_parser('freeze')
    for key in ('raw', 'git-dir', 'output'):
        freeze_cmd.add_argument('--'+key, type=Path, required=True)
    build_cmd = cmd.add_parser('build')
    for key in ('frozen', 'fra-original', 'trk-raw', 'site-dir'):
        build_cmd.add_argument('--'+key, type=Path, required=True)
    build_cmd.add_argument('--fra-raw', type=Path)
    build_cmd.add_argument('--alex-ratings', type=Path,
                           default=Path(__file__).parent/'preview-inputs/alex-joint-2026-11-08.json')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.raw, args.git_dir, args.output)
    else:
        build(args.frozen, args.fra_original, args.trk_raw, args.fra_raw, args.site_dir, args.alex_ratings)
