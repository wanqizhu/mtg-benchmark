"""Offline dashboard and downloadable analysis tables; no remote JavaScript."""
import argparse,html,json,math,shutil,textwrap,zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs
import markdown
from historical_creators import ROOT,PREVIOUS,CREATORS,CONTEXTS,prepare,norm,value

STYLE='''
:root{color-scheme:light dark;--bg:#f6f8fc;--paper:#fff;--fg:#182638;--muted:#5b6c80;--line:#dae2ed;--blue:#235caa;--header:#eaf0f8}
@media(prefers-color-scheme:dark){:root{--bg:#131b27;--paper:#1c2736;--fg:#e8eef8;--muted:#aab9cc;--line:#39485d;--blue:#91bfff;--header:#263449}}
*{box-sizing:border-box}body{margin:0;font:14px/1.55 system-ui,sans-serif;background:var(--bg);color:var(--fg)}main{max-width:1500px;margin:auto;padding:28px}h1{font-size:29px;line-height:1.2;margin:0 0 10px}h2{margin:30px 0 14px;font-size:23px}h3{font-size:18px}p{max-width:1100px}a{color:var(--blue)}nav{display:flex;gap:20px;flex-wrap:wrap;margin:18px 0 24px}.tag{border:1px solid var(--line);border-radius:8px;padding:5px 9px;font-size:12px}.notice{background:var(--paper);border-left:4px solid #da9b37;padding:13px 18px;margin:18px 0}.muted{color:var(--muted)}.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:center;padding:14px;background:var(--paper);border:1px solid var(--line);border-radius:10px}select,input{font:inherit;color:var(--fg);background:var(--paper);padding:7px 10px;border:1px solid var(--line);border-radius:6px}.grid{display:grid;grid-template-columns:minmax(450px,1fr) minmax(260px,340px);gap:20px}.panel{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:18px}.chart{min-height:470px}.widechart{min-height:330px}.table-wrap{overflow:auto;margin:14px 0 22px}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:8px 12px;text-align:right;border-bottom:1px solid var(--line);vertical-align:top}th{background:var(--header)}th:first-child,td:first-child{text-align:left}td:first-child{min-width:170px}#cards{width:max-content;min-width:100%}#cards th:first-child,#cards td:first-child{position:sticky;left:0;background:var(--paper);border-right:1px solid var(--line);min-width:235px}#cards td:nth-child(odd){max-width:200px}#detail h3{margin-top:0}#detail pre{white-space:pre-wrap;font:13px/1.5 system-ui,sans-serif}.downloads{display:flex;gap:18px;flex-wrap:wrap;margin:16px 0}details{padding:12px 0}summary{cursor:pointer;font-weight:600}.source-table td{text-align:left;max-width:420px}.sparse{color:#be7a1f}code{font-size:12px}pre{overflow:auto}.formula{background:var(--paper);border:1px solid var(--line);padding:16px;border-radius:8px;font-family:ui-monospace,monospace;white-space:pre-wrap}.model-section{padding:20px;background:var(--paper);border:1px solid var(--line);border-radius:10px;margin:18px 0}.metric{font-size:16px;font-weight:600}.pillrow{display:flex;gap:9px;flex-wrap:wrap}.tiny{font-size:12px}
@media(max-width:850px){main{padding:18px 12px}.grid{grid-template-columns:1fr}.chart{min-height:420px}h1{font-size:25px}.controls{gap:8px}}
'''


def esc(x):return html.escape(str(x))
def fmt(x,d=2):return '—' if x is None or pd.isna(x) else f'{x:.{d}f}'
def table(rows,headers,cls=''):
    return f'<div class="table-wrap"><table class="{cls}"><thead><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{v}</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'


def run(output,destination):
    d=json.loads((output/'analysis.json').read_text());models=d['models']['models'];records=d['records'];destination.mkdir(parents=True,exist_ok=True)
    doc=json.loads((ROOT/'inputs/historical_ratings.json').read_text());train=pd.read_csv(output/'historical_training_observations.csv');train_keys={(r.creator,r.set,norm(r['name'])) for _,r in train.iterrows()}
    conditional=[];groups={}
    for raw in doc['rows']:
        for ctx in raw.get('contexts',[]):
            key=(raw['creator'],'build_around' if ctx['kind']=='build_around_flag' else ctx['kind'],ctx.get('native') or '[icon only; no conditional grade]')
            groups.setdefault(key,[]).append(raw)
    for (creator,kind,native),rows in groups.items():
        tr=[r for r in rows if (creator,r['set'],norm(r['name'])) in train_keys]
        conditional.append({'creator':creator,'context_kind':kind,'raw_context_grade':native,'captured_card_set_observations':len(rows),'training_card_set_observations':len(tr),'training_distinct_card_names':len({norm(r['name']) for r in tr})})
    pd.DataFrame(conditional).to_csv(output/'conditional_grade_bucket_counts.csv',index=False)
    mappings=[];contexts=[]
    for key,m in models.items():
        for r in m['grade_mapping']:mappings.append({'creator':key,**r})
        for r in m['context_mapping']:contexts.append({'creator':key,'term':'flag offset',**r})
        for r in m.get('context_gap_mapping',[]):contexts.append({'creator':key,'term':'conditional native grade gap',**r})
    pd.DataFrame(mappings).to_csv(output/'grade_mappings.csv',index=False);pd.DataFrame(contexts).to_csv(output/'context_coefficients.csv',index=False)
    audit=d['source_audit']['sources'];pd.DataFrame([{k:v for k,v in r.items() if k not in ['archive_files','archive_sha256']} for r in audit]).to_csv(output/'source_audit.csv',index=False)
    leaderboard=[r for r in d['scores'] if r['pool']=='common cards across all five creators'];leaderboard.sort(key=lambda r:r['mae_pp'])
    score_table=table([[esc(r['name']),r['n'],fmt(r['pearson'],3),fmt(r['spearman'],3),fmt(r['mae_pp']),fmt(r['rmse_pp']),fmt(r['bias_pp'])] for r in leaderboard],['Creator','Cards','Pearson r','Spearman ρ','MAE, pp','RMSE, pp','Bias, pp'])
    own=[r for r in d['scores'] if r['pool']=='own reported main-set coverage']
    own_table=table([[esc(r['name']),r['historical_sets'],r['historical_training_n'],r['n'],fmt(r['pearson'],3),fmt(r['mae_pp'])] for r in own],['Creator','Historical sets','Training observations','FRA scored cards','Pearson r','MAE, pp'])
    sources_table=table([[esc(CREATORS.get(r['creator'],r['creator'])),esc(r['set']),esc(r.get('published_or_frozen','')),str(r.get('rated_main','—')),('Complete graded list' if r.get('complete') else 'Partial / ungraded omissions'),f'<a href="{esc(r.get("url","https://lrcast.com/episodes/"))}" target="_blank" rel="noreferrer">Source</a><br><span class="tiny">{esc(r.get("basis",""))}<br>{esc(r.get("notes",""))}</span>'] for r in audit if r['admitted']],['Creator','Set','Edition / original publication','Captured main ratings','Coverage','Evidence'],cls='source-table')
    excluded_table=table([[esc(r['creator']),esc(r['set']),esc(r.get('published_or_frozen','')),esc(r.get('basis',''))] for r in audit if not r['admitted']],['Source','Set','Date, if known','Reason excluded'],cls='source-table')
    model_sections=''
    for key,m in models.items():
        rows=[]
        for r in m['grade_mapping']:
            count=r['card_set_observations'];label=f'<span class="sparse">{count} · sparse</span>' if count<10 else str(count)
            rows.append([esc(r['grade']),label,r['distinct_card_names'],fmt(r['effective_weight'],1),fmt(r['mapped_wr'])+'%',fmt(r['observed_mean_wr'])+'%',fmt(r['loso_min'])+'–'+fmt(r['loso_max'])+'%'])
        model_sections+=f'<section class="model-section"><h3>{esc(m["name"])}</h3><p>{m["historical_sets"]} sets · {m["train_n"]} reported first-week card-set observations. Monotone curve; grade penalty {m["grade_lambda"]:g}; context penalty {m["context_lambda"]:g} (unused when there are no context terms). Historical selection RMSE {m["historical_validation"]["rmse_pp"]:.3f} pp.</p>'
        model_sections+=table(rows,['Native grade','Training card-set count','Distinct card names','Effective fractional weight','Fitted GIH','Unadjusted observed bucket mean','Omit-one-set mapping range'])
        for r in m['context_mapping']:
            model_sections+=f'<p><b>{esc(r["bucket"])}</b>: {r["offset_pp"]:+.3f} pp · {r["card_set_observations"]} training observations.</p>'
        for r in m.get('context_gap_mapping',[]):
            model_sections+=f'<p><b>{esc(r["bucket"])} conditional-grade gap</b>: {r["pp_per_native_grade_step"]:+.3f} pp per native grade step · {r["nonzero_gap_card_set_observations"]} nonzero historical examples.</p>'
        if not m['context_mapping']:model_sections+='<p>No published contextual buckets were supplied by this source.</p>'
        model_sections+='</section>'
    sparse=[r for r in mappings if r['card_set_observations']<10]
    sparse_table=table([[esc(CREATORS[r['creator']]),esc(r['grade']),r['card_set_observations'],r['distinct_card_names'],fmt(r['mapped_wr'])+'%'] for r in sparse],['Creator','Native bucket','Training card-set observations','Distinct names','Fitted GIH'])
    conditional_table=table([[esc(CREATORS[r['creator']]),esc(r['context_kind']),esc(r['raw_context_grade']),r['captured_card_set_observations'],r['training_card_set_observations']] for r in sorted(conditional,key=lambda x:(x['creator'],x['context_kind'],x['raw_context_grade']))],['Creator','Context','Raw conditional label','Captured observations','Training observations'])
    sensitivity=table([[esc(CREATORS[r['creator']]),esc(r['variant']),r['training_n'],r['n'],fmt(r['pearson'],3),fmt(r['mae_pp']),fmt(r['bias_pp'])] for r in d['sensitivity_scores']],['Creator','Sensitivity','Historical rows','FRA cards','Pearson r','MAE, pp','Bias, pp'],cls='source-table')
    initial=json.loads((output/'initial_flag_only_analysis.json').read_text())
    comparison=[]
    for key in CREATORS:
        before=next(r for r in initial['scores'] if r['creator']==key and r['pool']=='common cards across all five creators');after=next(r for r in leaderboard if r['creator']==key)
        comparison.append([esc(CREATORS[key]),fmt(before['mae_pp']),fmt(after['mae_pp']),fmt(before['pearson'],3),fmt(after['pearson'],3),'Yes' if models[key]['use_gaps'] else 'No'])
    initial_table=table(comparison,['Creator','Initial flag-only MAE','Final historical MAE','Initial r','Final r','Conditional gap selected'])
    miss_sections=''
    for key in CREATORS:
        rr=[r for r in d['largest_mistakes'] if r['creator']==key]
        miss_sections+=f'<details><summary>{esc(CREATORS[key])}: biggest misses</summary>'+table([[esc(r['card']),esc(r['native_grade']),fmt(r['prediction'])+'%',fmt(r['actual'])+'%',f'{r["error_pp"]:+.2f}',r['gih_n'],esc(r['direction'])] for r in rr],['Card','Published grade','Predicted','Observed','Error, pp','GIH games','Direction'])+'</details>'
    colors=[]
    for r in d['group_errors']:
        if r['group_type']=='color' and r['group'] in ['W','U','B','R','G']:
            colors.append([esc(CREATORS[r['creator']]),r['group'],r['n'],fmt(r['bias_pp']),fmt(r['mae_pp'])])
    color_table=table(colors,['Creator','Color','Cards','Mean error, pp','MAE, pp'])
    miss_md=''
    for key in CREATORS:
        rr=[r for r in d['largest_mistakes'] if r['creator']==key]
        miss_md+='\n### '+CREATORS[key]+'\n\n'
        for direction in ['overestimated','underestimated']:
            selected=[r for r in rr if r['direction']==direction][:4]
            miss_md+=direction.capitalize()+': '+ '; '.join(f"{r['card']} ({r['native_grade']}, {r['prediction']:.2f}% vs {r['actual']:.2f}%, {r['error_pp']:+.2f} pp)" for r in selected)+'.\n\n'
    qualitative=[r for r in records if r['creator']=='limitedresources' and r['primary'] and r['prediction'] is None]
    no_grade=table([[esc(r['card']),esc(r['native_grade']),'<a href="'+esc(r['source_url'])+'" target="_blank">Original captions / video timestamp</a>'] for r in qualitative],['FRA card','Verified qualitative assessment','Evidence'],cls='source-table')
    methods='''
## How the historical fit works

The target is Premier Draft GIH win rate (`ever_drawn_win_rate`), all colors and all tracked users, from 17Lands' historical `FIRST_WEEK` window. Only reported outcomes with at least 500 GIH games and nonbasic main-set cards enter fitting. Every card-set observation has equal weight; a commonly drawn card does not dominate the fit. Bonus-sheet cards and basic lands are excluded from the primary training and leaderboard.

Each creator has a separate model. Every observed native letter or decimal grade has its own fitted win-rate parameter. Numeric grades remain on the creator's original scale; equivalent strings such as `2` and `2.0` share one numerical bucket. C++ is retained between C+ and B−. Split grades use the mean of their fitted bucket values; explicitly recorded nested host ranges use fractional weights. No convenience 0–100 community normalization enters the model.

The model adds an offset for every context type with observed historical training data: build-around, synergy icon, graded sideboard, scenario, or alternate grade. Bare SB is its own unordered native bucket, so it does not receive a duplicate sideboard offset. If the ordinary grade is absent, an explicitly supplied conditional letter can supply the grade input. An assessment with no verified letter or integer remains unmapped.

We compare a flag-only model with a model that also uses the difference between an explicit conditional grade and the ordinary grade. This gap is measured on the native grade order, and its coefficient is fitted from history. It is zero when no separate conditional grade is supplied. The monotone family constrains these gap coefficients to be nonnegative. The raw conditional labels and their counts are included separately in the downloadable artifacts.

Every observed ordinary-grade bucket and historical context type enters fitting. Individual conditional combinations such as “build-around B” share the grade curve and context effects; they do not each receive an unrelated free parameter. Almost all such combinations still have fewer than ten observations, so their evidence is pooled through these terms. A future larger archive could support separate grade-by-context interactions.

Fit objective:

```
Σcards (observed GIH − predicted GIH)²
+ λgrade × Σadjacent ((βnext − βprevious) / native grade spacing)²
+ λgrade × (βSB − βlowest supported ordinary grade)²  [if SB exists]
+ λcontext × Σcontext coefficients²

prediction = weighted ordinary-grade bucket value
           + context flag offsets
           + selected conditional-grade-gap terms
```

The grade difference penalty borrows strength across adjacent buckets. SB borrows strength from the lowest ordinary-grade bucket without imposing an order on SB. Context penalties shrink sparse context estimates toward zero. There is no rule that invents observations, drops every bucket below ten, or treats four cards as precise evidence. The data do **not** supply ten observations in every bucket. Mapping ranges show sensitivity to omitting one set, not statistical confidence intervals.

We compare monotone and unconstrained curves; grade penalties 0, 2, 10, and 50; context penalties 2, 10, and 50; and conditional gaps on/off. Model choice minimizes historical leave-one-set-out RMSE. Each fold also removes all training printings with a name appearing in the held-out scoring fold. These historical CV scores are tuning scores, not an untouched estimate of historical generalization.

After choosing the family and penalties, the model fits all admitted history and is saved with a checksum. Only then does evaluation read the original October 7 FRA outcome file. Every FRA outcome is excluded from fitting and hyperparameter choice. Historical printings of a FRA card are legitimate past information in the primary fit; a sensitivity run removes all of them.

This is a retrospective transfer analysis, not a forecast registered before FRA results were known. The conditional-gap family was added after inspecting the initial flag-only FRA evaluation; its coefficients and selection nevertheless use historical data only, and both results are retained. The recovered Casual Spike TMNT source was added because its original publication became available, not because of its FRA performance.

## Source policy and coverage

The declared window is original prediction publication no later than **three calendar days after Arena launch**. Where a 17Lands list was later changed, we request the original historical version. We freeze the earliest version covering the main cards that received a grade in that source and verify its grades against the history replay. Current sheet values, later grade edits, and later context icons are excluded. A list with an intentionally ungraded card remains marked incomplete.

LR's rare reviews were available: Hobbit and Marvel audio were published two days after Arena launch, Lorwyn also two days after, and Strixhaven on release day. The original audio pages establish publication timing; linked review-video captions supply card-level evidence. Avatar's rare audio on November 22 is four days after November 18, so it is excluded from the three-day primary policy; its prelaunch CU review supplies the fifth historical LR set. TMNT's located LR primer is not a full card-grade list.

LR is the jointly authored LSV + Marshall source, not an isolated LSV-only grade stream. Hobbit supplies a full dated secondary transcription. The other historical LR sets use a **partial, manually checked reconstruction of explicit grades from autogenerated captions**. No videos were opened or played. Design praise, provisional remarks, negative comparisons, front-only DFC grades, and grades attached to the next card are not silently substituted for final whole-card grades. Remaining transcription omissions can cause sample-selection bias; those LR fits are provisional.

Draftsim here means the bot's 0–5 decimal pick-order table, not Andrew Quinn's separate 0–10 written reviews. Hobbit comes from the early dated collector files, whose bot grades are identical on August 11 and August 14. Strixhaven's public bot table labels its latest update April 15, before launch; no later edit evidence was found. These timing guarantees are weaker than 17Lands' version history. Later revised MSH, TMT, ECL, TLA, EOE, TDM, DFT, FDN, and DSK bot tables are excluded. No original archive was verified for those bot versions.

Alex and Marc use separate individual lists; their joint Hobbit list is excluded. Casual Spike's historical versions come from dated authored publications and linked 17Lands history. Five recent sets are admitted where obtainable; older linked Casual posts are not needed after the five-set cap is reached.

## Interpretation and limitations

The historical target is an exact site-defined first-week window. FRA uses the original October 7 early ALL_TIME snapshot, roughly 7.6 days after Arena launch, so the horizons are close but not identical. Suppressed outcomes are never imputed. The common leaderboard uses the same 265 FRA main-set cards for all five creators. Four sources cover 270 reported main-set cards; LR has five qualitative assessments without a recoverable explicit letter and scores 265. The full table retains all 295 canonical cards, including missing grades, basics and Special Guests.

A high grade maps to the historical *average* of that bucket. It cannot assign a different number to two otherwise identical grade/context inputs. Ajani's miss is largely an unusually extreme card compressed into an A/A+ average; it is not evidence that the creators ignored its strength. Color residuals suggest a format-specific transfer error: blue is generally underestimated and red generally overestimated. Sideboard grades are also an uncertain guide to BO1 main-deck GIH, particularly with very few historical sideboard observations.

The tiny Marc/Alex error difference is a point estimate. The sources share many cards and game populations, and only a few historical formats underpin the mappings. Incomplete LR extraction and the bot's two-set history limit strong ranking claims. The common pool excludes the five qualitative LR assessments; the full-coverage results show how that restriction affects each source. First-week rates are noisy estimates, even above 500 games. Negative context offsets are conditional associations given the grade, not causal claims that a build-around makes a card worse.

Intermediate artifacts include the source audit, original-version IDs and hashes, curated native source rows, manual caption-attribution corrections, excluded statements, historical training observations, every historical CV candidate and fold prediction, selected model coefficients, omit-one-set mapping ranges, sparse and conditional bucket counts, complete FRA raw/mapped tables, group residuals, and largest-mistake tables. Captions and live response envelopes remain local scratch files; generated outcomes are not committed to the public repository.
'''
    # Dynamic markdown report mirrors the dashboard's exact tables and formulas.
    md='# Historical creator calibration against early FRA\n\n'
    md+='## Shared-card results\n\n| Creator | Historical sets | Training rows | FRA cards | Pearson r | Spearman rho | MAE pp | RMSE pp | Bias pp |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in leaderboard:
        m=models[r['creator']];md+=f"| {r['name']} | {m['historical_sets']} | {m['train_n']} | {r['n']} | {r['pearson']:.3f} | {r['spearman']:.3f} | {r['mae_pp']:.2f} | {r['rmse_pp']:.2f} | {r['bias_pp']:+.2f} |\n"
    md+='\n'+methods+'\n## Largest mistakes\n'+miss_md
    (output/'historical_report.md').write_text(md)
    body=f'''<h1>Historical creator ratings → early FRA GIH</h1><p class="muted">Native grades fitted on earlier sets. FRA outcomes are excluded from every fit and model choice. Original October 7 outcome snapshot.</p>
<div class="pillrow"><span class="tag">{sum(m['train_n'] for m in models.values()):,} training observations</span><span class="tag">{len([r for r in audit if r['admitted']])} admitted source/set editions</span><span class="tag">265 shared scored FRA cards</span><span class="tag">295-card complete table</span></div>
<nav><a href="#results">Results</a><a href="#plots">Card hover plots</a><a href="#cards-section">All card predictions</a><a href="#models">Exact models & counts</a><a href="#sources">Source audit</a><a href="#method">Method & limitations</a></nav>
<div class="notice"><b>Coverage matters.</b> Casual Spike, Alex and Marc now each have five admitted sets. LR has five sets with partial caption reconstruction outside Hobbit. Only two unrevised bot editions were recovered. Several native and contextual buckets remain below ten cards.</div>
<section id="results"><h2>FRA evaluation</h2><p>The same 265 cards are used in this leaderboard. MAE and RMSE are in percentage points; bias is predicted minus observed. Correlations measure the mapped historical predictions, not the creators' original percentage forecasts.</p>{score_table}<details><summary>Each creator's full coverage and historical training size</summary>{own_table}</details><details><summary>Initial flag-only fit versus final historical selection</summary><p>The conditional-gap check is exploratory after the first evaluation. Selection and fitting use history only. The initial Casual result has four sets; the final result adds the subsequently recovered original TMNT list.</p>{initial_table}</details><details><summary>First-week/mature targets, prior reprints, context terms and source sensitivity</summary><p>Each sensitivity uses its creator's own supported main-set coverage. Compare with the full-coverage scores above, not a different card count. Primary historical hyperparameters are held fixed.</p>{sensitivity}</details></section>
<section id="plots"><h2>Predicted versus observed, with card details</h2><div class="controls"><label>Creator <select id="creator">'''+''.join(f'<option value="{k}">{esc(v)}</option>' for k,v in CREATORS.items())+'''</select></label><label>Rarity <select id="rarity"><option value="all">All</option><option>common</option><option>uncommon</option><option>rare</option><option>mythic</option></select></label><label>Card <input id="plot-search" placeholder="Filter by name" size="23"></label><label><input type="checkbox" id="shared-only"> Shared 265-card pool</label></div><p class="muted">The plot uses each creator's own coverage unless the shared-pool filter is selected. Hover for the grade, rates and sample size. Click a point for the complete card text and model inputs. The dashed diagonal is a perfect prediction.</p><p class="pillrow tiny" aria-label="Card colors"><span style="color:#bd9b30">● White</span><span style="color:#2979d0">● Blue</span><span style="color:#55536c">● Black</span><span style="color:#d56748">● Red</span><span style="color:#369474">● Green</span><span style="color:#8896a8">● Colorless</span><span style="color:#ae7bd0">● Multicolor</span></p><div class="grid"><div id="scatter" class="chart panel"></div><aside id="detail" class="panel"><h3>Card details</h3><p>Click a card in the plot for its full rules and model inputs.</p></aside></div><h3>Historical native-grade mapping</h3><p class="muted">Whiskers show omit-one-set sensitivity plus the full-fit value, not confidence intervals. SB is an unordered bucket shown separately.</p><div id="curve" class="widechart panel"></div><h3>Errors by color</h3><p class="muted">Positive values mean overestimation; negative values mean underestimation. Monocolor main-set cards only, with counts shown in hover.</p><div id="colors" class="widechart panel"></div></section>
<section id="mistakes"><h2>Where the mappings missed</h2><p>Blue generally outperforms its historical grade mapping; red tends to underperform it. Ajani is an extreme positive outlier in every source. The conversion also struggles with contextual and sideboard cards.</p>'''+miss_sections+f'''<details><summary>Color error table</summary>{color_table}</details></section>
<section id="cards-section"><h2>Every card: published grade, historical prediction and outcome</h2><div class="controls"><input id="table-search" placeholder="Search 295 cards" size="30"><span id="table-count"></span></div><div class="downloads"><a href="fra_predictions_wide.csv">Complete wide CSV</a><a href="fra_predictions_long.csv">Detailed card/context CSV</a><a href="historical_models.json">Exact model JSON</a><a href="analysis_source.zip">Source and method bundle</a></div><div class="table-wrap"><table id="cards"><thead><tr><th>Card</th><th>Observed GIH</th><th>GIH games</th>'''+''.join(f'<th>{esc(v)}<br>Raw grade</th><th>{esc(v)}<br>Predicted GIH</th>' for v in CREATORS.values())+'''</tr></thead><tbody id="card-body"></tbody></table></div><details><summary>Five LR qualitative assessments left without a fabricated percentage</summary>'''+no_grade+f'''</details></section>
<section id="models"><h2>Each creator's final model and sparse buckets</h2><p>prediction = grade-bucket mapping + observed context offsets + selected conditional-grade-gap effects. Ordinary grades are used first; conditional-only explicit letters are used when an ordinary label is unavailable. Raw labels and source conditions remain visible in the card tables.</p><div class="downloads"><a href="grade_mappings.csv">All grade mappings</a><a href="context_coefficients.csv">Context coefficients</a><a href="historical_bucket_counts.csv">Raw/flag bucket counts</a><a href="conditional_grade_bucket_counts.csv">Every conditional label count</a><a href="historical_cv_candidates.json">Every model-choice candidate</a><a href="historical_cv_predictions.csv">Historical selection fold predictions</a></div><details><summary>Buckets with fewer than ten historical training observations</summary>{sparse_table}</details><details><summary>All explicit conditional grade buckets, including sparse labels</summary><p>Counts are for the captured source rows; LR's incomplete transcription means these are not whole-review universe counts.</p>{conditional_table}</details>{model_sections}</section>
<section id="sources"><h2>Timestamp and edition audit</h2><p>Original editions no later than three calendar days after Arena launch are admitted. Historical versions supply original grades and icons; current edited tables are excluded. <a href="source_audit.csv">Download audit</a></p>{sources_table}<details><summary>Excluded or unselected source editions</summary>{excluded_table}</details></section>
<section id="method"><h2>Method and interpretation</h2>'''+markdown.markdown(methods,extensions=['fenced_code','tables'])+'''<div class="downloads"><a href="historical_report.md">Full written report</a><a href="fit_lock.json">Fit lock/checksum</a><a href="validation.json">Verification results</a><a href="historical_training_observations.csv">Historical training observations</a><a href="largest_mistakes.csv">Full mistake table</a><a href="sensitivity_scores.csv">Sensitivity scores</a></div></section>'''
    payload={'models':models,'records':records,'creators':CREATORS,'group_errors':d['group_errors']}
    script=r'''
const D=__DATA__;
const keys=Object.keys(D.creators),byCard=new Map();D.records.forEach(r=>{if(!byCard.has(r.card))byCard.set(r.card,{});byCard.get(r.card)[r.creator]=r});
const common=new Set([...byCard].filter(([n,rs])=>keys.every(k=>rs[k].primary&&rs[k].prediction!==null)).map(x=>x[0]));
const E=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const F=(x,n=2)=>x===null||x===undefined?'—':Number(x).toFixed(n);
const palette={W:'#bd9b30',U:'#2979d0',B:'#55536c',R:'#d56748',G:'#369474',C:'#8896a8',M:'#ae7bd0'};
function colorKey(r){return ['W','U','B','R','G','C'].includes(r.color)?r.color:'M'}
function wrapHover(s){let limit=innerWidth<650?28:65,out=[],line='';for(let word of String(s??'').split(/\s+/)){if(line.length+word.length+1>limit&&line){out.push(E(line));line=word}else{line+=(line?' ':'')+word}}if(line)out.push(E(line));return out.join('<br>')}
function theme(){let dark=matchMedia('(prefers-color-scheme:dark)').matches;return {bg:dark?'#1c2736':'#ffffff',fg:dark?'#e8eef8':'#182638',grid:dark?'#39485d':'#dde5ef'}}
function layout(extra={}){const t=theme();return {paper_bgcolor:t.bg,plot_bgcolor:t.bg,font:{color:t.fg,family:'system-ui, sans-serif'},margin:{l:65,r:22,t:35,b:60},hoverlabel:{font:{size:12},align:'left'},...extra}}
function details(r){let m=D.models[r.creator];let atoms=Object.entries(r.grade_atoms||{}).map(([g,w])=>{let row=m.grade_mapping.find(x=>x.grade===g);return E(g)+' × '+F(w,2)+(row?' · '+row.card_set_observations+' historical observations':' · interpolated/unseen bucket')}).join('<br>');let gaps=Object.entries(r.context_gaps||{}).map(([k,v])=>E(k)+': '+F(v,2)+' native steps').join('<br>');document.querySelector('#detail').innerHTML='<h3>'+E(r.card)+'</h3><p>'+E(r.cost)+' · '+E(r.type)+'<br>'+E(r.rarity)+' · '+E(r.color)+'</p><p class="metric">Predicted '+F(r.prediction)+'%<br>Observed '+F(r.actual)+'%<br>Error '+(r.error_pp>=0?'+':'')+F(r.error_pp)+' pp</p><p>Published: <b>'+E(r.native_grade)+'</b><br>Grade used: '+E(r.grade_used)+'<br>Context: '+E(r.contexts.join(', ')||'none')+'<br>GIH games: '+r.gih_n.toLocaleString()+'</p><p>'+atoms+'</p>'+(gaps?'<p>Conditional gaps<br>'+gaps+'</p>':'')+'<pre>'+E(r.oracle)+'</pre>'+(r.support_notes.length?'<p class="muted">'+E(r.support_notes.join('; '))+'</p>':'')+'<p><a target="_blank" rel="noreferrer" href="'+E(r.scryfall_uri)+'">Card reference</a> · <a target="_blank" rel="noreferrer" href="'+E(r.source_url)+'">Original grade source</a></p>'}
function render(){document.querySelector('#detail').innerHTML='<h3>Card details</h3><p>Click a card in the plot for its full rules and model inputs.</p>';let k=document.querySelector('#creator').value,rar=document.querySelector('#rarity').value,q=document.querySelector('#plot-search').value.toLowerCase(),share=document.querySelector('#shared-only').checked;let rows=D.records.filter(r=>r.creator===k&&r.primary&&r.prediction!==null&&(!share||common.has(r.card))&&(rar==='all'||r.rarity===rar)&&r.card.toLowerCase().includes(q));let traces=[];
for(let c of ['W','U','B','R','G','C','M']){let rr=rows.filter(r=>colorKey(r)===c);if(!rr.length)continue;traces.push({type:'scatter',mode:'markers',name:c==='M'?'Multicolor':c,x:rr.map(r=>r.actual),y:rr.map(r=>r.prediction),marker:{color:palette[c],size:8,opacity:.8,line:{width:.4,color:theme().fg}},customdata:rr.map(r=>[wrapHover(r.card),wrapHover(r.native_grade),F(r.error_pp),r.gih_n.toLocaleString(),wrapHover(r.cost+' · '+r.type)]),_cards:rr,hovertemplate:'<b>%{customdata[0]}</b><br>%{customdata[4]}<br>Grade: %{customdata[1]}<br>Observed %{x:.2f}%<br>Predicted %{y:.2f}%<br>Error %{customdata[2]} pp<br>%{customdata[3]} GIH games<br>Click for full rules and inputs<extra></extra>'})}
traces.push({type:'scatter',mode:'lines',name:'Perfect prediction',x:[38,79],y:[38,79],line:{color:theme().fg,dash:'dash',width:1},hoverinfo:'skip'});
Plotly.react('scatter',traces,layout({title:{text:wrapHover(D.creators[k]+' · '+rows.length+' cards'),font:{size:16}},margin:{l:55,r:18,t:60,b:60},xaxis:{title:{text:'Observed early FRA GIH (%)'},range:[38,79],gridcolor:theme().grid},yaxis:{title:{text:'Historical prediction (%)'},range:[38,79],gridcolor:theme().grid},showlegend:false,dragmode:'zoom'}),{responsive:true,displaylogo:false,displayModeBar:innerWidth>=650}).then(()=>{let g=document.querySelector('#scatter');g.removeAllListeners('plotly_click');g.on('plotly_click',ev=>{let p=ev.points[0];if(p.data._cards)details(p.data._cards[p.pointIndex])})});
let m=D.models[k],mm=m.grade_mapping.filter(r=>r.grade!=='SB'),sb=m.grade_mapping.filter(r=>r.grade==='SB');let cs=[];for(let r of mm){let lo=Math.min(r.mapped_wr,r.loso_min??r.mapped_wr),hi=Math.max(r.mapped_wr,r.loso_max??r.mapped_wr);cs.push({type:'scatter',mode:'lines',x:[r.grade,r.grade],y:[lo,hi],line:{color:'#8a9aaf',width:2},hoverinfo:'skip',showlegend:false})}
cs.push({type:'scatter',mode:'lines+markers',name:'Native-grade fit',x:mm.map(r=>r.grade),y:mm.map(r=>r.mapped_wr),line:{color:'#4386cd',width:2},marker:{size:mm.map(r=>8+Math.sqrt(r.card_set_observations)*.5)},customdata:mm.map(r=>[r.card_set_observations,r.distinct_card_names,F(r.observed_mean_wr),F(r.loso_min),F(r.loso_max)]),hovertemplate:'Grade %{x}<br>Fitted %{y:.2f}%<br>%{customdata[0]} card-set observations · %{customdata[1]} distinct names<br>Unadjusted mean %{customdata[2]}%<br>Omit-one-set range %{customdata[3]}–%{customdata[4]}%<extra></extra>'});if(sb.length)cs.push({type:'scatter',mode:'markers',name:'SB (unordered)',x:['SB'],y:[sb[0].mapped_wr],marker:{size:10,color:'#b7853c'},hovertemplate:'SB<br>%{y:.2f}% · '+sb[0].card_set_observations+' historical observations<extra></extra>'});
Plotly.react('curve',cs,layout({xaxis:{title:{text:'Native bucket (no semantic 0–100 conversion)'},type:'category',categoryorder:'array',categoryarray:[...mm.map(r=>r.grade),'SB'],gridcolor:theme().grid},yaxis:{title:{text:'Mapped GIH (%)'},gridcolor:theme().grid},showlegend:false}),{responsive:true,displaylogo:false});
let gr=D.group_errors.filter(r=>r.creator===k&&r.group_type==='color'&&['W','U','B','R','G'].includes(r.group));Plotly.react('colors',[{type:'bar',x:gr.map(r=>r.group),y:gr.map(r=>r.bias_pp),marker:{color:gr.map(r=>palette[r.group])},customdata:gr.map(r=>[r.n,F(r.mae_pp)]),hovertemplate:'%{x}<br>Mean error %{y:+.2f} pp<br>%{customdata[0]} cards · MAE %{customdata[1]} pp<extra></extra>'}],layout({xaxis:{title:{text:'Monocolor cards'}},yaxis:{title:{text:'Prediction − outcome, pp'},zeroline:true,gridcolor:theme().grid}}),{responsive:true,displaylogo:false});}
function cardTable(){let q=document.querySelector('#table-search').value.toLowerCase(),rr=[...byCard].filter(([n])=>n.toLowerCase().includes(q)).sort((a,b)=>a[0].localeCompare(b[0]));document.querySelector('#table-count').textContent=rr.length+' cards';document.querySelector('#card-body').innerHTML=rr.map(([n,rs])=>{let first=rs[keys[0]];return '<tr><td>'+E(n)+'<br><span class="tiny muted">'+E(first.set.toUpperCase())+' · '+E(first.rarity)+'</span></td><td>'+(first.actual===null?'—':F(first.actual)+'%')+'</td><td>'+first.gih_n.toLocaleString()+'</td>'+keys.map(k=>'<td>'+E(rs[k].native_grade??'—')+'</td><td title="'+E(rs[k].support_notes.join('; '))+'">'+(rs[k].prediction===null?'—':F(rs[k].prediction)+'%')+'</td>').join('')+'</tr>'}).join('')}
['creator','rarity','shared-only'].forEach(id=>document.querySelector('#'+id).addEventListener('change',render));document.querySelector('#plot-search').addEventListener('input',render);document.querySelector('#table-search').addEventListener('input',cardTable);matchMedia('(prefers-color-scheme:dark)').addEventListener('change',render);render();cardTable();
'''.replace('__DATA__',json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c'))
    js=get_plotlyjs()
    page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Historical creator calibration · FRA</title><style>'+STYLE+'</style><script>'+js+'</script></head><body><main>'+body+'</main><script>'+script+'</script></body></html>'
    (output/'historical_creators.html').write_text(page)
    # Analytical comparison, conditional on the observed common card pool.
    rng=np.random.default_rng(10309);aligned={k:[r for r in records if r['creator']==k and r['card'] in {r['card'] for r in records if r['primary']} and r['primary'] and r['prediction'] is not None] for k in CREATORS}
    shared=set.intersection(*[{r['card'] for r in rr} for rr in aligned.values()]);names=sorted(shared)
    errors={k:np.array([abs(next(r['error_pp'] for r in aligned[k] if r['card']==n)) for n in names]) for k in CREATORS};a=errors['marcanderson']-errors['alexnikolic'];samples=rng.integers(0,len(a),size=(5000,len(a)));diffs=a[samples].mean(axis=1)
    comparison={'comparison':'Marc MAE minus Alex MAE on identical FRA cards','n':len(a),'mean_difference_pp':float(a.mean()),'card_bootstrap_95_percent_interval_pp':np.quantile(diffs,[.025,.975]).tolist(),'qualification':'Descriptive card resampling conditional on frozen outcomes; does not account for shared game/cohort dependence or historical mapping uncertainty. Difference is small, not a decisive superiority claim.'}
    (output/'paired_comparison.json').write_text(json.dumps(comparison,indent=2))
    for file in output.iterdir():
        if file.is_file() and file.suffix in ['.csv','.json','.md','.html'] and not file.name.startswith('initial_flag_only_'):shutil.copyfile(file,destination/file.name)
    for f in ['historical_models.json','fit_lock.json','analysis.json']:
        src=output/('initial_flag_only_'+f)
        if src.exists():shutil.copyfile(src,destination/src.name)
    (destination/'historical_ratings_source.json').write_text((ROOT/'inputs/historical_ratings.json').read_text())
    (destination/'fra_ratings_source.json').write_text((ROOT/'inputs/fra_ratings.json').read_text())
    with zipfile.ZipFile(destination/'analysis_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for file in list(ROOT.glob('*.py'))+list(ROOT.glob('README.md')):
            z.write(file,f'{ROOT.name}/{file.name}')
        for subdir in ['inputs','collection']:
            for file in (ROOT/subdir).iterdir():
                if file.is_file():z.write(file,f'{ROOT.name}/{subdir}/{file.name}')
        # Keep the sibling paths used by evaluate_stage; live response envelopes
        # are deliberately excluded. The README explains the missing outcomes.
        for f in ['cards.json','ground_truth_manifest.json']:
            z.write(PREVIOUS/'inputs'/f,f'{PREVIOUS.name}/inputs/{f}')
        for file in output.iterdir():
            if file.is_file() and file.suffix in ['.csv','.json','.md']:
                z.write(file,f'{ROOT.name}/results/{file.name}')
    print('dashboard',destination/'historical_creators.html','bytes',len(page.encode()),'paired comparison',comparison,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'results');ap.add_argument('--destination',type=Path,required=True);a=ap.parse_args();run(a.output,a.destination)
