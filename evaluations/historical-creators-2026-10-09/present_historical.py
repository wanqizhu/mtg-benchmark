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
        if raw['creator'] not in CREATORS:continue
        for ctx in raw.get('contexts',[]):
            key=(raw['creator'],'build_around' if ctx['kind']=='build_around_flag' else ctx['kind'],ctx.get('native') or '[icon only; no conditional grade]')
            groups.setdefault(key,[]).append(raw)
    for (creator,kind,native),rows in groups.items():
        tr=[r for r in rows if (creator,r['set'],norm(r['name'])) in train_keys]
        conditional.append({'creator':creator,'context_kind':kind,'raw_context_grade':native,'captured_card_set_observations':len(rows),'training_card_set_observations':len(tr),'training_distinct_card_names':len({norm(r['name']) for r in tr})})
    pd.DataFrame(conditional).to_csv(output/'conditional_grade_bucket_counts.csv',index=False)
    mappings=[];contexts=[];coefficients=[];borrowed=[]
    for key,m in models.items():
        for r in m['grade_mapping']:mappings.append({'creator':key,**r})
        for r in m['context_mapping']:contexts.append({'creator':key,'term':'flag offset',**r})
        for r in m.get('context_gap_mapping',[]):contexts.append({'creator':key,'term':'conditional native grade gap',**r})
        if m['model_type']=='bot_regression':
            coefficients.extend({'creator':key,'term_type':'regression basis','term':r['term'],'coefficient':r['coefficient'],'unit':'GIH percentage points','weighted_basis_support':r['weighted_historical_support']} for r in m['regression_coefficients'])
        else:
            for r in m['grade_mapping']:
                coefficients.append({'creator':key,'term_type':'letter grade','term':r['grade'],'coefficient':r['mapped_wr'],'unit':'GIH percent','real_observations':r['observed_n'],'borrowed_weight':r['borrowed_weight'],'base_coefficient':r['base_coefficient']})
                for donor in r['donors']:borrowed.append({'creator':key,'term_type':'letter grade','term':r['grade'],'real_observations':r['observed_n'],**donor})
        for r in m['context_mapping']+m.get('context_gap_mapping',[]):
            gap='pp_per_native_grade_step' in r;coefficient=r['pp_per_native_grade_step'] if gap else r['offset_pp']
            coefficients.append({'creator':key,'term_type':'conditional grade gap' if gap else 'context offset','term':r['bucket'],'coefficient':coefficient,'unit':'percentage points per native step' if gap else 'percentage points','real_observations':r['observed_n'],'borrowed_weight':r['borrowed_weight'],'base_coefficient':r['base_coefficient']})
            if r['borrowed_weight']:borrowed.append({'creator':key,'term_type':'conditional grade gap' if gap else 'context offset','term':r['bucket'],'real_observations':r['observed_n'],'grade':'ordinary-card zero-effect baseline','borrowed_weight':r['borrowed_weight'],'base_coefficient':0.})
    pd.DataFrame(coefficients).to_csv(output/'model_coefficients.csv',index=False);pd.DataFrame(borrowed).to_csv(output/'borrowed_sample_weights.csv',index=False)
    pd.DataFrame(mappings).to_csv(output/'grade_mappings.csv',index=False);pd.DataFrame(contexts).to_csv(output/'context_coefficients.csv',index=False)
    audit=d['source_audit']['sources'];pd.DataFrame([{k:v for k,v in r.items() if k not in ['archive_files','archive_sha256']} for r in audit]).to_csv(output/'source_audit.csv',index=False)
    leaderboard=[r for r in d['scores'] if r['pool']=='common cards across all active creators'];leaderboard.sort(key=lambda r:r['mae_pp'])
    score_table=table([[esc(r['name']),r['n'],fmt(r['pearson'],3),fmt(r['spearman'],3),fmt(r['mae_pp']),fmt(r['rmse_pp']),fmt(r['bias_pp'])] for r in leaderboard],['Creator','Cards','Pearson r','Spearman ρ','MAE, pp','RMSE, pp','Bias, pp'])
    own=[r for r in d['scores'] if r['pool']=='own reported main-set coverage']
    own_table=table([[esc(r['name']),r['historical_sets'],r['historical_training_n'],r['n'],fmt(r['pearson'],3),fmt(r['mae_pp'])] for r in own],['Creator','Historical sets','Training observations','FRA scored cards','Pearson r','MAE, pp'])
    sources_table=table([[esc(CREATORS.get(r['creator'],r['creator'])),esc(r['set']),esc(r.get('published_or_frozen','')),str(r.get('rated_main','—')),('Complete graded list' if r.get('complete') else 'Partial / ungraded omissions'),f'<a href="{esc(r.get("url","https://lrcast.com/episodes/"))}" target="_blank" rel="noreferrer">Source</a><br><span class="tiny">{esc(r.get("basis",""))}<br>{esc(r.get("notes",""))}</span>'] for r in audit if r['admitted']],['Creator','Set','Edition / original publication','Captured main ratings','Coverage','Evidence'],cls='source-table')
    excluded_table=table([[esc(r['creator']),esc(r['set']),esc(r.get('published_or_frozen','')),esc(r.get('basis',''))] for r in audit if not r['admitted']],['Source','Set','Date, if known','Reason excluded'],cls='source-table')
    model_sections=''
    for key,m in models.items():
        if m['model_type']=='bot_regression':
            model_sections+=f'<section class="model-section"><h3>{esc(m["name"])}</h3><p>{m["historical_sets"]} sets · {m["train_n"]} historical observations. {esc(m["family"])} · <b>{m["nominal_degrees_of_freedom"]} total fitted coefficients</b> · curvature penalty {m["curvature_lambda"]:g}. Historical selection RMSE {m["historical_validation"]["rmse_pp"]:.3f} pp.</p><p>Native-rating domain {m["domain"][0]:g}–{m["domain"][1]:g}; constant predictions beyond those endpoints. The curve uses all historical card observations. Numeric labels are inputs to this regression, with no separate bucket parameters.</p>'
            model_sections+=table([[esc(r['term']),fmt(r['coefficient'],5),fmt(r['weighted_historical_support'],2)] for r in m['regression_coefficients']],['Regression coefficient','Coefficient, GIH pp','Weighted basis support'])
            if m['knots']:model_sections+=f'<p><b>Cubic B-spline knot vector:</b> <code>{esc(json.dumps(m["knots"]))}</code></p><p>Prediction = Σ coefficient[j] × B-spline basis[j](native rating). Fixed knots and boundaries add no fitted coefficients. Unconstrained smoother trace: {m["unconstrained_smoother_degrees_of_freedom"]:.3f}; monotonic constraints can reduce flexibility further.</p>'
            else:
                slope=(m['coefficients'][1]-m['coefficients'][0])/(m['domain'][1]-m['domain'][0]);intercept=m['coefficients'][0]-slope*m['domain'][0]
                model_sections+=f'<p>Prediction = {intercept:.5f} + {slope:.5f} × clipped native rating.</p>'
            model_sections+='</section>';continue
        rows=[]
        for r in m['grade_mapping']:
            count=r['card_set_observations'];label=f'<span class="sparse">{count} · sparse</span>' if count<10 else str(count)
            donors='; '.join(f'{x["grade"]}: {x["borrowed_weight"]:g} × {x["base_coefficient"]:.2f}%' for x in r['donors']) or '—'
            rows.append([esc(r['grade']),label,fmt(r['base_coefficient']),fmt(r['borrowed_weight'],1),esc(donors),fmt(r['mapped_wr'])+'%',fmt(r['loso_min'])+'–'+fmt(r['loso_max'])+'%'])
        model_sections+=f'<section class="model-section"><h3>{esc(m["name"])}</h3><p>{m["historical_sets"]} sets · {m["train_n"]} real historical observations. Base curve: {"monotone" if m["monotone"] else "free"}; grade penalty {m["grade_lambda"]:g}; context penalty {m["context_lambda"]:g}. The fixed ten-sample adjustment follows this base fit and is applied in historical validation too. Historical selection RMSE {m["historical_validation"]["rmse_pp"]:.3f} pp.</p>'
        model_sections+=table(rows,['Native grade; SB = F','Real card-set count','Base coefficient, %','Borrowed sample weight','Donor means and weights','Final coefficient','Omit-one-set range'])
        for r in m['context_mapping']:
            model_sections+=f'<p><b>{esc(r["bucket"])}</b>: {r["offset_pp"]:+.5f} pp · {r["observed_n"]} real observations + {r["borrowed_weight"]:g} zero-effect sample weight; base offset {r["base_coefficient"]:+.5f} pp.</p>'
        for r in m.get('context_gap_mapping',[]):
            model_sections+=f'<p><b>{esc(r["bucket"])} conditional-grade gap</b>: {r["pp_per_native_grade_step"]:+.5f} pp per native grade step · {r["observed_n"]} real examples + {r["borrowed_weight"]:g} zero-effect weight; base slope {r["base_coefficient"]:+.5f}.</p>'
        if not m['context_mapping']:model_sections+='<p>No published contextual buckets were supplied by this source.</p>'
        model_sections+='</section>'
    sparse=[r for r in mappings if r['creator']!='draftsim' and r['card_set_observations']<10]
    sparse_table=table([[esc(CREATORS[r['creator']]),esc(r['grade']),r['card_set_observations'],r['distinct_card_names'],fmt(r['mapped_wr'])+'%'] for r in sparse],['Creator','Native bucket','Training card-set observations','Distinct names','Fitted GIH'])
    conditional_table=table([[esc(CREATORS[r['creator']]),esc(r['context_kind']),esc(r['raw_context_grade']),r['captured_card_set_observations'],r['training_card_set_observations']] for r in sorted(conditional,key=lambda x:(x['creator'],x['context_kind'],x['raw_context_grade']))],['Creator','Context','Raw conditional label','Captured observations','Training observations'])
    sensitivity=table([[esc(CREATORS[r['creator']]),esc(r['variant']),r['training_n'],r['n'],fmt(r['pearson'],3),fmt(r['mae_pp']),fmt(r['bias_pp'])] for r in d['sensitivity_scores']],['Creator','Sensitivity','Historical rows','FRA cards','Pearson r','MAE, pp','Bias, pp'],cls='source-table')
    comparison=[]
    if (output/'comparison_prior_scores.json').exists():
        initial=json.loads((output/'comparison_prior_scores.json').read_text())
        current_shared=sorted(set.intersection(*[{r['card'] for r in records if r['creator']==k and r['primary'] and r['prediction'] is not None} for k in CREATORS]))
        assert set(initial['common_card_names'])<=set(current_shared)
        for key in CREATORS:
            before=next(r for r in initial['scores'] if r['creator']==key and r['pool']=='common cards across all five creators')
            aligned=[r for r in records if r['creator']==key and r['card'] in set(initial['common_card_names'])];errors=np.array([r['error_pp'] for r in aligned]);after={'mae_pp':float(np.mean(abs(errors))),'pearson':float(np.corrcoef([r['prediction'] for r in aligned],[r['actual'] for r in aligned])[0,1])}
            comparison.append([esc(CREATORS[key]),fmt(before['mae_pp']),fmt(after['mae_pp']),fmt(before['pearson'],3),fmt(after['pearson'],3),'Yes' if models[key]['use_gaps'] else 'No'])
    initial_table=table(comparison,['Creator','Prior policy MAE','Updated MAE','Prior r','Updated r','Conditional gap selected'])
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
## Historical targets and validation

The target is Premier Draft GIH win rate (`ever_drawn_win_rate`), all colors and tracked users, from the official historical `FIRST_WEEK` window. Training uses reported outcomes with at least 500 GIH games and nonbasic main-set cards. Each card-set observation has equal loss weight. Bonus-sheet cards and basics remain in the complete FRA catalog but do not enter the primary fit or leaderboard.

Each source has a separate model. Historical leave-one-set-out selection removes every training printing whose name appears in the held-out scoring fold. Model selection uses historical RMSE only. These fold scores are tuning scores, not nested unbiased estimates. Selected models are refitted on all admitted history and locked by checksum before evaluation reads FRA outcomes. The October 7 FRA snapshot remains frozen. A complete repeat fit is tested with every FRA file inaccessible.

## Letter grades: fixed ten-sample rule

Bare SB and explicitly sideboard-marked ratings are grouped with F for the letter sources. Original published labels remain in the all-card table and source audit. There is no separately fitted SB or sideboard-effect coefficient.

The base model fits native letter coefficients plus historically observed context offsets. C++ is placed halfway between C+ and B−. Split grades average the component fitted values, with original nested host weights retained. Monotone/free base curves, grade penalties 0/2/10/50, context penalties 2/10/50 and conditional-grade-gap effects on/off are compared using history only.

The requested rule is applied to every sparse letter coefficient after each base fit, including fits inside historical validation:

```
For a grade with n < 10 real card-set memberships:
    final coefficient = (n × base coefficient + Σ donor weight × donor base coefficient) / 10
    Σ donor weight = 10 − n

For n ≥ 10: final coefficient = base coefficient.
```

Donors are the closest observed native grades by ordinal distance. Equally close grades share weight equally, capped at each donor's actual card-set count; only then does borrowing expand to the next distance. All donor means are taken from the same unsmoothed base fit. Adjustments are simultaneous, so borrowed values are not recursively borrowed again. A class with no observations borrows all ten weights and is labelled n=0. The base fit's grade/context penalties remain; the ten-sample operation is a separate explicit adjustment. Large buckets receive no ten-sample adjustment, although refitting and historical model selection can change their base coefficients.

For a context offset or conditional-grade-gap effect with fewer than ten identifying examples, missing weight comes from the ordinary-card baseline, whose context effect is zero: `final effect = n/10 × base effect`. This is the declared extension for unordered effects; they have no ordinal nearby grade class. Sideboard effects disappear under the F merge. Raw conditional-label counts remain available for provenance, and most conditional combinations still do not have ten real observations.

Borrowed weights are fractional pseudo-sample weights, not extra observed cards. The tables show real n, base coefficient, borrowed weight, exact donor means/weights and final coefficient. Split-grade memberships and repeated printings can share information; ten units of coefficient weight do not imply ten independent cards. Omit-one-set ranges show mapping sensitivity, not confidence intervals.

## Draftsim bot: fewer than ten fitted coefficients

Draftsim's native 0–5 decimal rating is a continuous input to a monotone regression. We compare a linear curve with two coefficients against cubic B-splines with four through nine coefficients, using curvature penalties 0/2/10/50 for the splines. Every candidate has fewer than ten total fitted coefficients. Fixed knot locations add no fitted parameters. All historical card observations enter fitting; there is no independent parameter for each decimal bucket and no per-bucket ten-sample adjustment for this source.

Spline knots are uniformly spaced inside the historical observed native-rating range, with repeated endpoint knots. Coefficients are constrained in increasing order, giving a monotone curve. Outside the historical input range, predictions stay at the nearest endpoint. The model panel shows the selected family, nominal parameter count, curvature penalty, full coefficient vector and knot vector. Its unconstrained smoother trace is also reported; shape constraints can reduce flexibility further. The native-grade curve evaluations are reference predictions, not additional coefficients.

## Source policy and coverage

Original editions must be published no later than three calendar days after Arena launch. 17Lands history supplies original grades and icons, excluding later edits. Casual Spike and Alex each contribute five individual editions. LR contributes Hobbit, Marvel, Strixhaven, Lorwyn and Avatar; the latter is CU only because its rare episode was four days after Arena launch. Hobbit's LR transcription is complete; the other LR sets use partial, manually checked explicit grades from caption text. No media was played. Unresolved attribution, design-only praise and unverified letters are excluded. The isolated captured LR D− is S.H.I.E.L.D. Helicarrier in Marvel; this does not count every grade spoken across the complete reviews.

Draftsim means the bot's 0–5 decimal pick-order ratings. Hobbit uses a secondary collector copy self-dated August 11, corroborated by the same 188 values in its self-dated August 14 copy. Those embedded dates are not an independently verified archive timestamp. Strixhaven uses the current public rating asset with a publisher-declared April 15 update date, six days before Arena launch; no independent prelaunch copy or edit history was recovered. Absence of later edit evidence does not prove the asset was never revised. Both are admitted provisionally under these dating assumptions, not verified immutable prerelease editions. Their one-set source-strength sensitivities remain available. Bot lists with an explicitly late update date are excluded. Source omissions and partial LR transcription limit strong ranking claims.

The FRA bot inputs are also provisional: the collector records September 25 source activity, but the saved collector bundle was generated October 6, after the September 29 Arena launch. That ledger date does not verify that the retained decimal ratings are an unchanged prelaunch edition. No independent prelaunch bot copy or edit history was recovered. Historical fitting excludes FRA outcomes, but that alone does not establish prerelease provenance of the FRA bot inputs.

## FRA interpretation and limits

Four active sources are evaluated on the same 266 main-set FRA cards. Casual, Alex and the bot each score 270 reported main-set cards; LR scores 266. Its qualitative sideboard assessment of Refute Destiny maps to F under the requested grouping, leaving four other assessments without a recoverable native grade. The before/after comparison separately uses the previous 265-card pool. The full table retains all 295 canonical cards. Historical targets use the official first-week window; FRA uses the original early ALL_TIME snapshot from October 7, about 7.6 days after launch. Suppressed outcomes are not imputed.

This is retrospective transfer analysis. The earlier conditional-gap family was added after inspecting the original flag-only FRA results; its coefficients and selection use history only. The new sparse-support rule, SB merge and bot degree-of-freedom cap were requested after those results were available. Current validation still uses history only. Historical printings of FRA cards are allowed as past information, with a sensitivity excluding them. Other sensitivities use mature targets, grade-only inputs, no postlaunch LR rare episodes and each bot set separately; primary model choices remain fixed.

High letter grades map to historical bucket averages and can compress exceptional bombs. GIH also reflects deck, player and cohort composition. Negative context offsets are associations conditional on grade, not causal claims. Sources have different historical coverage and outcome noise remains even above 500 games.

Downloads include actual model coefficients, donor weights, raw source labels, every historical selection candidate, fold predictions, sensitivity results, all-card predictions and largest errors. Source code and curated prediction facts are committed; live API envelopes and generated outcome tables remain local.
'''
    # Dynamic markdown report mirrors the dashboard's exact tables and formulas.
    md='# Historical creator calibration against early FRA\n\n'
    md+='## Shared-card results\n\n| Creator | Historical sets | Training rows | FRA cards | Pearson r | Spearman rho | MAE pp | RMSE pp | Bias pp |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in leaderboard:
        m=models[r['creator']];md+=f"| {r['name']} | {m['historical_sets']} | {m['train_n']} | {r['n']} | {r['pearson']:.3f} | {r['spearman']:.3f} | {r['mae_pp']:.2f} | {r['rmse_pp']:.2f} | {r['bias_pp']:+.2f} |\n"
    letter_keys=[k for k in CREATORS if models[k]['model_type']=='native_bucket'];grades=list(models[letter_keys[0]]['grade_mapping'])
    md+='\n## Historical letter-grade coefficients\n\nValues are final predicted GIH percentages; parentheses show real card-set memberships. SB/sideboard is included in F.\n\n| Grade | '+' | '.join(CREATORS[k] for k in letter_keys)+' |\n|---|'+ '|'.join(['---:']*len(letter_keys))+'|\n'
    for row in grades:
        g=row['grade'];entries=[next(r for r in models[k]['grade_mapping'] if r['grade']==g) for k in letter_keys]
        md+='| '+g+' | '+' | '.join(f'{r["mapped_wr"]:.5f}% ({r["observed_n"]})' for r in entries)+' |\n'
    bot=models['draftsim'];md+=f'\n## Draftsim regression coefficients\n\n{bot["family"]}; {bot["nominal_degrees_of_freedom"]} fitted coefficients; historical domain {bot["domain"]}; curvature penalty {bot["curvature_lambda"]:g}.\n\n| Term | Coefficient |\n|---|---:|\n'
    for r in bot['regression_coefficients']:md+=f'| {r["term"]} | {r["coefficient"]:.8f} |\n'
    md+='\nKnot vector: `'+json.dumps(bot['knots'])+'`.\n'
    md+='\n## Historical context coefficients\n\n| Creator | Term | Coefficient | Real examples | Borrowed zero-effect weight |\n|---|---|---:|---:|---:|\n'
    for r in coefficients:
        if r['term_type'] in ['context offset','conditional grade gap']:md+=f'| {CREATORS[r["creator"]]} | {r["term"]} {r["term_type"]} | {r["coefficient"]:+.8f} | {r["real_observations"]} | {r["borrowed_weight"]} |\n'
    md+='\n'+methods+'\n## Largest mistakes\n'+miss_md
    (output/'historical_report.md').write_text(md)
    body=f'''<h1>Historical creator ratings → early FRA GIH</h1><p class="muted">Native grades fitted on earlier sets. FRA outcomes are excluded from every fit and model choice. Original October 7 outcome snapshot.</p>
<div class="pillrow"><span class="tag">{sum(m['train_n'] for m in models.values()):,} training observations</span><span class="tag">{len([r for r in audit if r['admitted']])} admitted source/set editions</span><span class="tag">{d["common_card_count"]} shared scored FRA cards</span><span class="tag">295-card complete table</span></div>
<nav><a href="#results">Results</a><a href="#plots">Card hover plots</a><a href="#cards-section">All card predictions</a><a href="#models">Exact models & counts</a><a href="#sources">Source audit</a><a href="#method">Method & limitations</a></nav>
<div class="notice"><b>Coverage and dating matter.</b> Casual Spike and Alex each have five admitted sets. LR has five sets with partial caption reconstruction outside Hobbit. Draftsim's two historical editions are provisional: Hobbit has self-dated secondary snapshots; Strixhaven has a publisher-declared prelaunch update date on a current asset. Its FRA values likewise come from a postlaunch collector with prelaunch source-activity metadata. No bot edition has independently verified immutable prerelease provenance. Several native and contextual buckets remain below ten cards.</div>
<section id="results"><h2>FRA evaluation</h2><p>The same {d["common_card_count"]} cards are used in this leaderboard. MAE and RMSE are in percentage points; bias is predicted minus observed. Correlations measure the mapped historical predictions, not the creators' original percentage forecasts.</p>{score_table}<details><summary>Each creator's full coverage and historical training size</summary>{own_table}</details><details><summary>Previous policy versus requested changes</summary><p>The same 265-card pool is used in both columns. The update merges sideboard ratings with F, adds the fixed sparse-letter rule and replaces bot bucket parameters with a regression of fewer than ten coefficients. All selection and fitting use history only.</p>{initial_table}</details><details><summary>First-week/mature targets, prior reprints, context terms and source sensitivity</summary><p>Each sensitivity uses its creator's own supported main-set coverage. Compare with the full-coverage scores above, not a different card count. Primary historical hyperparameters are held fixed.</p>{sensitivity}</details></section>
<section id="plots"><h2>Predicted versus observed, with card details</h2><div class="controls"><label>Creator <select id="creator">'''+''.join(f'<option value="{k}">{esc(v)}</option>' for k,v in CREATORS.items())+'''</select></label><label>Rarity <select id="rarity"><option value="all">All</option><option>common</option><option>uncommon</option><option>rare</option><option>mythic</option></select></label><label>Card <input id="plot-search" placeholder="Filter by name" size="23"></label><label><input type="checkbox" id="shared-only"> Shared {d["common_card_count"]}-card pool</label></div><p class="muted">The plot uses each creator's own coverage unless the shared-pool filter is selected. Hover for the grade, rates and sample size. Click a point for the complete card text and model inputs. The dashed diagonal is a perfect prediction.</p><p class="pillrow tiny" aria-label="Card colors"><span style="color:#bd9b30">● White</span><span style="color:#2979d0">● Blue</span><span style="color:#55536c">● Black</span><span style="color:#d56748">● Red</span><span style="color:#369474">● Green</span><span style="color:#8896a8">● Colorless</span><span style="color:#ae7bd0">● Multicolor</span></p><div class="grid"><div id="scatter" class="chart panel"></div><aside id="detail" class="panel"><h3>Card details</h3><p>Click a card in the plot for its full rules and model inputs.</p></aside></div><h3>Historical native-grade mapping</h3><p class="muted">Whiskers show omit-one-set sensitivity plus the full-fit value, not confidence intervals. SB and sideboard ratings are grouped with F. Draftsim uses a continuous regression curve.</p><div id="curve" class="widechart panel"></div><h3>Errors by color</h3><p class="muted">Positive values mean overestimation; negative values mean underestimation. Monocolor main-set cards only, with counts shown in hover.</p><div id="colors" class="widechart panel"></div></section>
<section id="mistakes"><h2>Where the mappings missed</h2><p>Blue generally outperforms its historical grade mapping; red tends to underperform it. Ajani is an extreme positive outlier in every source. The conversion also struggles with contextual and sideboard cards.</p>'''+miss_sections+f'''<details><summary>Color error table</summary>{color_table}</details></section>
<section id="cards-section"><h2>Every card: published grade, historical prediction and outcome</h2><div class="controls"><input id="table-search" placeholder="Search 295 cards" size="30"><span id="table-count"></span></div><div class="downloads"><a href="fra_predictions_wide.csv">Complete wide CSV</a><a href="fra_predictions_long.csv">Detailed card/context CSV</a><a href="historical_models.json">Exact model JSON</a><a href="analysis_source.zip">Source and method bundle</a></div><div class="table-wrap"><table id="cards"><thead><tr><th>Card</th><th>Observed GIH</th><th>GIH games</th>'''+''.join(f'<th>{esc(v)}<br>Raw grade</th><th>{esc(v)}<br>Predicted GIH</th>' for v in CREATORS.values())+'''</tr></thead><tbody id="card-body"></tbody></table></div><details><summary>Four LR qualitative assessments without an explicit grade or applicable SB alias</summary>'''+no_grade+f'''</details></section>
<section id="models"><h2>Historical coefficients and sparse support</h2><p>Letter prediction = final grade coefficient + context offsets + selected conditional-grade-gap effects. Bot prediction = its continuous regression curve. SB/sideboard inputs are grouped with F. Ordinary grades are used first; other conditional-only explicit letters supply the grade input when available. Raw labels and source conditions remain visible in the card tables.</p><div class="downloads"><a href="model_coefficients.csv">Actual fitted coefficients</a><a href="borrowed_sample_weights.csv">Exact borrowed sample weights</a><a href="grade_mappings.csv">Grade mappings and bot curve evaluations</a><a href="context_coefficients.csv">Context coefficients</a><a href="historical_bucket_counts.csv">Raw/flag bucket counts</a><a href="conditional_grade_bucket_counts.csv">Every conditional label count</a><a href="historical_cv_candidates.json">Every model-choice candidate</a><a href="historical_cv_predictions.csv">Historical selection fold predictions</a></div><details><summary>Buckets with fewer than ten historical training observations</summary>{sparse_table}</details><details><summary>All explicit conditional grade buckets, including sparse labels</summary><p>Counts are for the captured source rows; LR's incomplete transcription means these are not whole-review universe counts.</p>{conditional_table}</details>{model_sections}</section>
<section id="sources"><h2>Timestamp and edition audit</h2><p>Original editions no later than three calendar days after Arena launch are admitted. Historical versions supply original grades and icons; current edited tables are excluded. <a href="source_audit.csv">Download audit</a></p>{sources_table}<details><summary>Excluded or unselected source editions</summary>{excluded_table}</details></section>
<section id="method"><h2>Method and interpretation</h2>'''+markdown.markdown(methods,extensions=['fenced_code','tables'])+'''<div class="downloads"><a href="historical_report.md">Full written report</a><a href="fit_lock.json">Fit lock/checksum</a><a href="validation.json">Verification results</a><a href="historical_training_observations.csv">Historical training observations</a><a href="largest_mistakes.csv">Full mistake table</a><a href="sensitivity_scores.csv">Sensitivity scores</a></div></section>'''
    body=body.replace('{d["common_card_count"]}',str(d['common_card_count']))
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
let m=D.models[k],bot=m.model_type==='bot_regression',mm=m.grade_mapping;let xx=r=>bot?r.native_value:r.grade,cs=[];for(let r of mm){let lo=Math.min(r.mapped_wr,r.loso_min??r.mapped_wr),hi=Math.max(r.mapped_wr,r.loso_max??r.mapped_wr);cs.push({type:'scatter',mode:'lines',x:[xx(r),xx(r)],y:[lo,hi],line:{color:'#8a9aaf',width:1},hoverinfo:'skip',showlegend:false})}
cs.push({type:'scatter',mode:bot?'lines':'lines+markers',name:bot?'Continuous regression':'Letter coefficient',x:mm.map(xx),y:mm.map(r=>r.mapped_wr),line:{color:'#4386cd',width:2},marker:{size:mm.map(r=>8+Math.sqrt(r.card_set_observations)*.5)},customdata:mm.map(r=>[r.card_set_observations,r.distinct_card_names,F(r.loso_min),F(r.loso_max),r.borrowed_weight??0]),hovertemplate:bot?('Native rating %{x}<br>Regression GIH %{y:.2f}%<br>'+m.nominal_degrees_of_freedom+' total fitted coefficients<br>%{customdata[0]} historical cards at this rating<br>Omit-one-set range %{customdata[2]}–%{customdata[3]}%<extra></extra>'):'Grade %{x}<br>Final coefficient %{y:.2f}%<br>%{customdata[0]} real card-set memberships<br>%{customdata[4]} borrowed sample weight<br>Omit-one-set range %{customdata[2]}–%{customdata[3]}%<extra></extra>'});
Plotly.react('curve',cs,layout({xaxis:{title:{text:bot?'Native bot rating (continuous input)':'Native grade; SB = F'},type:bot?'linear':'category',categoryorder:'array',categoryarray:mm.map(r=>r.grade),gridcolor:theme().grid},yaxis:{title:{text:'Mapped GIH (%)'},gridcolor:theme().grid},showlegend:false}),{responsive:true,displaylogo:false});
let gr=D.group_errors.filter(r=>r.creator===k&&r.group_type==='color'&&['W','U','B','R','G'].includes(r.group));Plotly.react('colors',[{type:'bar',x:gr.map(r=>r.group),y:gr.map(r=>r.bias_pp),marker:{color:gr.map(r=>palette[r.group])},customdata:gr.map(r=>[r.n,F(r.mae_pp)]),hovertemplate:'%{x}<br>Mean error %{y:+.2f} pp<br>%{customdata[0]} cards · MAE %{customdata[1]} pp<extra></extra>'}],layout({xaxis:{title:{text:'Monocolor cards'}},yaxis:{title:{text:'Prediction − outcome, pp'},zeroline:true,gridcolor:theme().grid}}),{responsive:true,displaylogo:false});}
function cardTable(){let q=document.querySelector('#table-search').value.toLowerCase(),rr=[...byCard].filter(([n])=>n.toLowerCase().includes(q)).sort((a,b)=>a[0].localeCompare(b[0]));document.querySelector('#table-count').textContent=rr.length+' cards';document.querySelector('#card-body').innerHTML=rr.map(([n,rs])=>{let first=rs[keys[0]];return '<tr><td>'+E(n)+'<br><span class="tiny muted">'+E(first.set.toUpperCase())+' · '+E(first.rarity)+'</span></td><td>'+(first.actual===null?'—':F(first.actual)+'%')+'</td><td>'+first.gih_n.toLocaleString()+'</td>'+keys.map(k=>'<td>'+E(rs[k].native_grade??'—')+'</td><td title="'+E(rs[k].support_notes.join('; '))+'">'+(rs[k].prediction===null?'—':F(rs[k].prediction)+'%')+'</td>').join('')+'</tr>'}).join('')}
['creator','rarity','shared-only'].forEach(id=>document.querySelector('#'+id).addEventListener('change',render));document.querySelector('#plot-search').addEventListener('input',render);document.querySelector('#table-search').addEventListener('input',cardTable);matchMedia('(prefers-color-scheme:dark)').addEventListener('change',render);render();cardTable();
'''.replace('__DATA__',json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c'))
    js=get_plotlyjs()
    page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Historical creator calibration · FRA</title><style>'+STYLE+'</style><script>'+js+'</script></head><body><main>'+body+'</main><script>'+script+'</script></body></html>'
    (output/'historical_creators.html').write_text(page)
    # Copy current four-source artifacts; preserve prior snapshots only in scratch.
    for file in output.iterdir():
        if file.is_file() and file.suffix in ['.csv','.json','.md','.html'] and not file.name.startswith(('initial_flag_only_','prior_policy_')) and file.name!='paired_comparison.json':shutil.copyfile(file,destination/file.name)
    for folder in [output,destination]:
        legacy=folder/'paired_comparison.json'
        if legacy.exists():legacy.unlink()
    for file in destination.glob('initial_flag_only_*.json'):file.unlink()
    for file in destination.glob('prior_policy_*.json'):file.unlink()
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
            if file.is_file() and file.suffix in ['.csv','.json','.md'] and not file.name.startswith(('initial_flag_only_','prior_policy_')):
                z.write(file,f'{ROOT.name}/results/{file.name}')
    print('dashboard',destination/'historical_creators.html','bytes',len(page.encode()),'active creators',list(CREATORS),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'results');ap.add_argument('--destination',type=Path,required=True);a=ap.parse_args();run(a.output,a.destination)
