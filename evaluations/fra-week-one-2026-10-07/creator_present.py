"""Readable all-card table and workbook-ready data for fitted creator models."""
import html
import json
import math

import markdown
import pandas as pd

STYLE = '''
:root{color-scheme:light dark;--bg:#f9fafc;--fg:#202837;--muted:#566379;--line:#d7dce4;--head:#edf0f5;--link:#3659a9}
@media(prefers-color-scheme:dark){:root{--bg:#171c25;--fg:#e3e8f1;--muted:#a6b2c6;--line:#414958;--head:#222936;--link:#9cb9ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 system-ui,sans-serif}main{padding:28px;max-width:1400px;margin:auto}
h1{font-size:25px;margin:0 0 12px}h2{font-size:19px;margin-top:28px}h3{font-size:16px}p{max-width:1000px}a{color:var(--link)}nav{display:flex;gap:20px;flex-wrap:wrap;margin:14px 0 20px}.muted{color:var(--muted)}
.table-wrap{overflow-x:auto;width:100%;margin:14px 0 24px}table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;width:100%}th,td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}th{background:var(--head);font-weight:500}th:first-child,td:first-child{text-align:left}
#all-cards{width:max-content;min-width:100%}#all-cards th{max-width:190px;min-width:100px}#all-cards td:nth-child(even){text-align:left;max-width:190px;min-width:115px}#all-cards th:first-child,#all-cards td:first-child{position:sticky;left:0;background:var(--bg);min-width:240px;max-width:280px;border-right:1px solid var(--line)}
pre{overflow-x:auto}code{font-size:13px}.rate{white-space:nowrap}
@media(max-width:700px){main{padding:18px 12px}h1{font-size:22px}}
'''


def page(title, body):
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+STYLE+'</style></head><body><main>'+body+'</main></body></html>'


def nullable(value):
    if value is None or isinstance(value,float) and math.isnan(value): return None
    if hasattr(value,'item'): return value.item()
    return value


def present(frame, metadata, wide, long, spec, scores, buckets, knots, output):
    report = markdown.markdown((output/'creator_models.md').read_text(),extensions=['tables','fenced_code'])
    report = report.replace('<table>','<div class="table-wrap"><table>').replace('</table>','</table></div>')
    (output/'creator_models.html').write_text(page('Creator models', '<nav><a href="creator_table.html">Complete card table</a><a href="creator_models.json">Exact model specifications</a></nav>'+report))
    body='''<h1>Every card and creator rating</h1>
<p>295 canonical cards: 280 main-set nonbasic cards, ten Special Guests and five basic lands. Each creator has its published grade and the percentage from its final fitted model. Missing grades and labels such as bare SB remain blank. The final mappings use the frozen October 7 outcomes; honest accuracy scores use separate held-out values.</p>
<nav><a href="creator_predictions_wide.csv">Download full card table</a><a href="creator_predictions.xlsx">Workbook</a><a href="creator_models.html">Every creator's final model</a><a href="context_buckets.csv">Context bucket counts</a><a href="creator_predictions_long.csv">Detailed grades, contexts and held-out values</a></nav>
<p class="muted">LR combines LSV + Marshall's commons/uncommons and rare review. It includes the September 29 release-day audio edition. Every percentage column is a fitted conversion, rather than a percentage originally published by a creator. Scroll horizontally to see all ten sources.</p>
<div class="table-wrap"><table id="all-cards"><thead><tr><th>Card</th>'''
    for source in metadata['sources']:
        name=html.escape(source['name']);body+=f'<th>{name}<br>Raw grade</th><th>{name}<br>Mapped GIH</th>'
    body+='</tr></thead><tbody>'
    card_rows=[]
    for name,row in wide.sort_index().iterrows():
        body+='<tr><td>'+html.escape(name)+'</td>';vals=[name]
        for source in metadata['sources']:
            key=source['id'];raw=nullable(row[key+'__raw_grade']);mapped=nullable(row[key+'__mapped_wr'])
            body+='<td>'+html.escape(str(raw)) if raw is not None else '<td>—'
            body+='</td><td class="rate">'+(f'{mapped:.2f}%' if mapped is not None else '—')+'</td>'
            # Typed native numerics are maintained in the workbook.
            if raw is not None:
                try: raw=float(raw)
                except ValueError: pass
            vals.extend([raw,None if mapped is None else mapped/100])
        body+='</tr>';card_rows.append(vals)
    body+='</tbody></table></div><p class="muted">Source grades: <a href="https://chunk.science/mtga-reality-fracture.html">community collector</a> and its linked original publishers. Observed outcomes: <a href="https://www.17lands.com/card_data?expansion=FRA&amp;format=PremierDraft&amp;time_period=ALL_TIME">17Lands</a>. Independent local analysis, without endorsement. No numerical model is available for Lords of Limited or Jim Davis because the retained sources contain no complete raw grade table for them.</p>'
    (output/'creator_table.html').write_text(page('All creator card predictions',body))
    source_rows=[];summaries=[]
    for source in metadata['sources']:
        key=source['id'];model=spec['models'][key];score=scores[scores.source.eq(key)].iloc[0]
        summaries.append([model['name'],model['raw_scale'],model['collected_cards'],model['mapped_cards'],model['training_cards'],
                          nullable(score['raw_grade_spearman_rho']),float(score['held_out_mae_pp']),
                          model['context_offsets_pp'].get('build_around'),'; '.join(model['context_offsets_pp']) or 'none'])
        source_rows.append([source['name'],source['publication_or_recorded_date'],source['url'],source.get('caveat','')])
    mapping_rows=[[r['creator'],r['grade_labels'],r['native_grade_value'],int(r['training_cards']),r['baseline_mapped_wr']/100,
                   spec['models'][r['source']]['context_offsets_pp'].get('build_around',0)] for r in knots.to_dict('records')]
    bucket_table=[[r['creator'],r['bucket_label'],int(r['distinct_rated_cards']),int(r['main_rated_cards']),int(r['training_cards']),bool(r['include_term'])] for r in buckets.to_dict('records')]
    bucket_summary=[]
    for source in metadata['sources']:
        m=spec['models'][source['id']]
        selected=buckets[buckets.source.eq(source['id']) & buckets.include_term]
        bucket_summary.append([source['name'],*[m['context_bucket_counts'][b] for b in ['build_around','sideboard','scenario','alternate']],
                               '; '.join(m['context_offsets_pp']) or 'none',int(selected.training_cards.sum()) if len(selected) else None])
    workbook={'cards_header':['Card']+[label for s in metadata['sources'] for label in [s['name']+' raw grade',s['name']+' mapped GIH']],
              'cards':card_rows,'model_summary':summaries,'mapping_rows':mapping_rows,'buckets':bucket_table,'bucket_summary':bucket_summary,'sources':source_rows,
              'method':'Final mapped GIH = monotone native-grade curve + eligible context offsets. More than 20 distinct cards per creator is required for a context term. Outcomes are the frozen October 7 snapshot.',
              'grade_rule':'Ordinary grade first. If absent, use the first explicit contextual letter or number. Bare SB and missing grades remain blank. Numeric grades retain their native scale; letter modifiers use ordinal steps.',
              'fit_rule':'Final mappings are fitted snapshots. Accuracy uses 5 shared folds repeated 20 times; every scored card is excluded from each model that predicts it.'}
    (output/'workbook_data.json').write_text(json.dumps(workbook,ensure_ascii=False,allow_nan=False))
