"""Native-grade creator models, including LR rares and explicit context counts.

python creator_native.py --truth inputs/local/17lands_fra_all_time.json
The original October 7 evaluation remains independently reproducible.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize, LinearConstraint
from scipy.stats import pearsonr, spearmanr

from evaluate import ROOT, fold_assignments, load_inputs, norm, markdown_table, SEED, REPEATS

LETTER_VALUES = {'F': 0, 'D-': 1, 'D': 2, 'D+': 3, 'C-': 4, 'C': 5,
                 'C+': 6, 'C++': 6.5, 'B-': 7, 'B': 8, 'B+': 9,
                 'A-': 10, 'A': 11, 'A+': 12}
BUCKETS = ['build_around', 'sideboard', 'scenario', 'alternate']
BUCKET_LABELS = {'build_around': 'Build-around', 'sideboard': 'Sideboard',
                 'scenario': 'Archetype / matchup condition', 'alternate': 'Alternative grade / uncertainty'}
THRESHOLD = 20


def grade_value(label):
    """Numeric scores stay on their native scale; letters use ordered modifiers."""
    if label is None:
        return None
    label = str(label).replace('−', '-').replace('–', '-').strip()
    if re.fullmatch(r'\d+(?:\.\d+)?', label):
        return float(label)
    if label in LETTER_VALUES:
        return float(LETTER_VALUES[label])
    parts = [p.strip() for p in re.split(r'\s*/\s*|\s+to\s+', label)]
    if len(parts) > 1 and all(p in LETTER_VALUES for p in parts):
        return float(np.mean([LETTER_VALUES[p] for p in parts]))
    return None  # SB alone and verbal discussion do not contain a raw ordered grade.


def prepare_row(row):
    flags = set()
    for context in row.get('contexts', []):
        kind = context['kind']
        if kind in ['build_around', 'build_around_flag']:
            flags.add('build_around')
        elif kind in BUCKETS:
            flags.add(kind)
        else:
            raise ValueError(f'Unmapped context kind: {kind}')
    label = row.get('ordinary_native')
    basis = 'ordinary published grade'
    if not label:
        label = row['native']
        basis = 'native published grade'
    score = grade_value(label)
    if score is None:
        candidates = [(c['native'], grade_value(c['native'])) for c in row.get('contexts', [])]
        candidates = [(label, value) for label, value in candidates if value is not None]
        if candidates:
            label, score = candidates[0]
            basis = 'explicit conditional grade; no ordinary grade supplied'
    return {**row, 'grade_used': label, 'grade_value': score, 'grade_basis': basis,
            'buckets': sorted(flags), 'missing_reason': None if score is not None else 'No raw A-F or numeric grade (e.g. SB alone)'}


class GradeContextModel:
    """Convex additive least squares: monotone grade curve + context offsets."""
    def fit(self, x, y, z):
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        self.beta = np.zeros(z.shape[1])
        previous = np.inf
        for iteration in range(10000):
            self.curve = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(x, y-z@self.beta)
            fitted = self.curve.predict(x)
            new_beta = np.linalg.lstsq(z, y-fitted, rcond=None)[0] if z.shape[1] else self.beta
            loss = np.sum((fitted + z@new_beta-y)**2)
            assert loss <= previous + 1e-8, 'Alternating least squares must decrease loss'
            if np.max(np.abs(new_beta-self.beta), initial=0) < 1e-9:
                self.beta = new_beta
                break
            self.beta, previous = new_beta, loss
        else:
            raise RuntimeError('Context fit did not converge')
        self.curve = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(x, y-z@self.beta)
        self.iterations = iteration+1
        assert np.all(np.diff(self.curve.y_thresholds_) >= -1e-10)
        return self

    def predict(self, x, z):
        return np.clip(self.curve.predict(np.asarray(x)) + np.asarray(z)@self.beta, 0, 100)


def verify_quadratic_fit(model, x, y, z):
    """Independent constrained least-squares check of the only context model."""
    levels, inverse = np.unique(x, return_inverse=True)
    design = np.column_stack([np.eye(len(levels))[inverse], z])
    start = np.r_[model.curve.predict(levels), model.beta]
    constraints = np.column_stack([np.diff(np.eye(len(levels)), axis=0), np.zeros((len(levels)-1, z.shape[1]))])
    objective = lambda b: np.sum((design@b-y)**2)
    result = minimize(objective, start, jac=lambda b: 2*design.T@(design@b-y),
                      constraints=LinearConstraint(constraints, 0, np.inf),
                      method='SLSQP', options={'ftol':1e-9, 'maxiter':1000})
    assert result.success, result.message
    assert abs(result.fun-objective(start)) < 1e-5
    return float(abs(result.fun-objective(start)))


def run(truth):
    manifest = json.loads((ROOT/'inputs/ground_truth_manifest.json').read_text())
    assert hashlib.sha256(truth.read_bytes()).hexdigest() == manifest['sha256']
    frame, _, _, _, _ = load_inputs(truth)
    metadata = json.loads((ROOT/'inputs/creator_ratings_native.json').read_text())
    aliases = {norm(name):name for name in frame.index}
    primary = frame['is_main'] & frame['gih_wr'].notna() & frame['gih_n'].ge(500)
    assignments = fold_assignments(frame.index[primary].tolist())
    models, all_rows, memberships, bucket_rows, knot_rows, score_rows, cv_ledger = {}, [], [], [], [], [], []
    output = ROOT/'results/native-creators'
    output.mkdir(parents=True, exist_ok=True)
    wide = frame[['set','rarity','gih_wr','gih_n']].copy()
    wide.index.name = 'card'
    independent_checks = {}
    for source in metadata['sources']:
        key = source['id']
        rows = [prepare_row(row) for row in source['rows']]
        names = [aliases[norm(row['name'])] for row in rows]
        assert len(set(names)) == len(names)
        for row, name in zip(rows, names): row['name'] = name
        indexed = {row['name']:row for row in rows}
        counts = {b:sum(b in row['buckets'] for row in rows) for b in BUCKETS}
        eligible = [b for b in BUCKETS if counts[b] > THRESHOLD]
        # Eligibility uses published metadata only, not test outcomes, and stays fixed across folds.
        training = [name for name in names if primary[name] and indexed[name]['grade_value'] is not None]
        for bucket in BUCKETS:
            bucket_rows.append({'source':key,'creator':source['name'],'bucket':bucket,'bucket_label':BUCKET_LABELS[bucket],
                                'distinct_rated_cards':counts[bucket],
                                'main_rated_cards':sum(bucket in indexed[n]['buckets'] for n in names if frame.loc[n,'is_main']),
                                'training_cards':sum(bucket in indexed[n]['buckets'] for n in training),
                                'include_term':bucket in eligible,'threshold_rule':'more than 20 distinct rated cards per creator'})
        def arrays(selected):
            x = np.array([indexed[n]['grade_value'] for n in selected], dtype=float)
            z = np.array([[int(b in indexed[n]['buckets']) for b in eligible] for n in selected], dtype=float).reshape(len(selected),len(eligible))
            return x, frame.loc[selected,'gih_wr'].to_numpy(), z
        x, y, z = arrays(training)
        final = GradeContextModel().fit(x,y,z)
        if eligible:
            independent_checks[key] = verify_quadratic_fit(final,x,y,z)
        oof, appearances = {n:0.0 for n in training}, {n:0 for n in training}
        for repeat, assignment in enumerate(assignments):
            for fold in range(5):
                test = [n for n in training if assignment[n]==fold]
                train = [n for n in training if assignment[n]!=fold]
                assert set(train).isdisjoint(test)
                xx,yy,zz = arrays(train)
                fit = GradeContextModel().fit(xx,yy,zz)
                tx,_,tz = arrays(test)
                predicted = fit.predict(tx,tz)
                for name, value in zip(test,predicted):
                    oof[name] += float(value);appearances[name] += 1
                if repeat==0:
                    cv_ledger.append({'source':key,'fold':fold,'train':len(train),'test':len(test),'overlap':0,'context_terms':eligible})
        assert all(n==REPEATS for n in appearances.values())
        oof = {n:value/REPEATS for n,value in oof.items()}
        mapped = {}
        for row in rows:
            name = row['name'];value=row['grade_value']
            full = None if value is None else float(final.predict([value],np.array([[int(b in row['buckets']) for b in eligible]],dtype=float).reshape(1,len(eligible)))[0])
            mapped[name] = full
            actual = None if pd.isna(frame.loc[name,'gih_wr']) else float(frame.loc[name,'gih_wr'])
            all_rows.append({'card':name,'set':frame.loc[name,'set'],'source':key,'creator':source['name'],
                             'raw_published_grade':row['native'],'grade_used':row['grade_used'],'native_grade_value':value,
                             'grade_basis':row['grade_basis'],'context_buckets':'; '.join(row['buckets']),
                             'final_model_mapped_wr':full,'held_out_mapped_wr':oof.get(name),'observed_gih_wr':actual,
                             'gih_n':int(frame.loc[name,'gih_n']),'used_to_fit_final_model':name in training,
                             'source_url':row['source_url'],'missing_reason':row['missing_reason']})
            for bucket in row['buckets']:
                memberships.append({'creator':source['name'],'source':key,'card':name,'bucket':bucket,
                                    'raw_grade':row['native'],'grade_used':row['grade_used'],'term_included':bucket in eligible,
                                    'used_in_training':name in training})
        wide[key+'__raw_grade'] = pd.Series({n:indexed[n]['native'] for n in names})
        wide[key+'__mapped_wr'] = pd.Series(mapped)
        for level in sorted(set(x)):
            selected=[n for n in training if indexed[n]['grade_value']==level]
            labels=sorted(set(indexed[n]['grade_used'] for n in selected))
            knot_rows.append({'source':key,'creator':source['name'],'grade_labels':' / '.join(labels),
                              'native_grade_value':float(level),'training_cards':len(selected),
                              'observed_mean_wr':float(frame.loc[selected,'gih_wr'].mean()),
                              'baseline_mapped_wr':float(final.curve.predict([level])[0])})
        pred=np.array([oof[n] for n in training]);raw=x
        score_rows.append({'source':key,'creator':source['name'],'training_cards':len(training),
                           'raw_grade_pearson_r':float(pearsonr(raw,y).statistic),
                           'raw_grade_spearman_rho':float(spearmanr(raw,y).statistic),
                           'held_out_mae_pp':float(np.mean(np.abs(pred-y))),
                           'held_out_pearson_r':float(pearsonr(pred,y).statistic),
                           'final_fit_mae_pp':float(np.mean(np.abs(final.predict(x,z)-y))),
                           'context_terms':'; '.join(eligible) or 'none'})
        models[key] = {'name':source['name'],'source_url':source['url'],'edition':source['edition'],
                       'date':source['publication_or_recorded_date'],'caveat':source.get('caveat',''),
                       'raw_scale':source['scale'],'collected_cards':len(rows),'mapped_cards':sum(v is not None for v in mapped.values()),
                       'training_cards':len(training),'context_bucket_counts':counts,
                       'context_offsets_pp':{b:float(beta) for b,beta in zip(eligible,final.beta)},
                       'grade_knots':[{'native_value':float(a),'baseline_wr':float(b)} for a,b in zip(final.curve.X_thresholds_,final.curve.y_thresholds_)],
                       'formula':'clip(grade_curve(native_grade) + sum(context_offset * context_indicator), 0, 100)',
                       'interpolation':'linear between native-grade knots; clamp outside training grade range',
                       'fit_iterations':final.iterations}
    # One row for every catalog card/source, including missing grades and all five basics.
    existing={(r['card'],r['source']) for r in all_rows}
    for name in frame.index:
        for source in metadata['sources']:
            if (name,source['id']) not in existing:
                all_rows.append({'card':name,'set':frame.loc[name,'set'],'source':source['id'],'creator':source['name'],
                                 'raw_published_grade':None,'grade_used':None,'native_grade_value':None,'grade_basis':None,
                                 'context_buckets':'','final_model_mapped_wr':None,'held_out_mapped_wr':None,
                                 'observed_gih_wr':None if pd.isna(frame.loc[name,'gih_wr']) else frame.loc[name,'gih_wr'],
                                 'gih_n':int(frame.loc[name,'gih_n']),'used_to_fit_final_model':False,
                                 'source_url':source['url'],'missing_reason':'No published grade in retained source'})
    long=pd.DataFrame(all_rows).sort_values(['card','source'])
    assert len(long)==295*10 and not long.duplicated(['card','source']).any()
    buckets=pd.DataFrame(bucket_rows);scores=pd.DataFrame(score_rows);knots=pd.DataFrame(knot_rows)
    wide.to_csv(output/'creator_predictions_wide.csv')
    long.to_csv(output/'creator_predictions_long.csv',index=False)
    buckets.to_csv(output/'context_buckets.csv',index=False)
    pd.DataFrame(memberships).to_csv(output/'context_memberships.csv',index=False)
    knots.to_csv(output/'creator_grade_mappings.csv',index=False)
    scores.to_csv(output/'creator_scores.csv',index=False)
    spec={'truth_manifest':manifest,'method':'Equal-card monotone least squares with additive explicit context indicators',
          'letter_encoding':LETTER_VALUES,'context_threshold':THRESHOLD,
          'threshold_population':'Distinct published rated cards per creator, independent of outcomes; fixed across CV',
          'grade_selection':'Use explicit ordinary grade; otherwise the first explicit conditional letter/number. SB alone stays unmapped.',
          'cross_validation':{'seed':SEED,'folds':5,'repeats':REPEATS},'models':models}
    (output/'creator_models.json').write_text(json.dumps(spec,indent=2,ensure_ascii=False))
    validation={'catalog_cards':295,'main_nonbasic':280,'special_guests':10,'basic_lands':5,'creator_card_rows':len(long),
                'snapshot_sha256':manifest['sha256'],'ratings_sha256':hashlib.sha256((ROOT/'inputs/creator_ratings_native.json').read_bytes()).hexdigest(),
                'selected_context_terms':{k:list(v['context_offsets_pp']) for k,v in models.items()},
                'training_test_overlap':0,'predictions_per_scored_card':REPEATS,'folds_first_repeat':cv_ledger,
                'independent_context_fit_objective_differences':independent_checks,
                'checks':['Raw numeric values preserved','Letters and modifiers ordered explicitly','No collector semantic index used',
                          'No contextual grades invented','Strict >20 threshold applied per creator without outcomes',
                          'Final grade curves monotonic','Context optimizer independently checked','Every scored card held out',
                          'All 295 cards by all 10 creators represented','Missing rates and grades stay missing']}
    (output/'validation.json').write_text(json.dumps(validation,indent=2))
    write_models_report(spec,scores,buckets,knots,output)
    from creator_present import present
    present(frame,metadata,wide,long,spec,scores,buckets,knots,output)
    print(scores.to_string(index=False));print(buckets[buckets.distinct_rated_cards.gt(0)].to_string(index=False))
    return spec


def write_models_report(spec,scores,buckets,knots,output):
    text='''# Creator models from native grades

This update includes the user-authorized [LR rare review, episode 873](https://lrcast.com/limited-resources-873-reality-fracture-set-review-rare-and-mythic-rare/) (September 29 audio, September 30 video), together with [episode 872](https://lrcast.com/limited-resources-872-reality-fracture-set-review-commons-and-uncommons/) as one LSV + Marshall source. Other retained creator editions are unchanged. The ground truth remains the frozen October 7 ALL_TIME snapshot; it was not refreshed on October 8.

Each creator has a separate model: **mapped GIH = monotone native-grade curve + included contextual offsets**, clipped to 0–100%. Numeric scores stay on their original 0–10 or 0–5 scale. Letter grades use ordered modifier steps: F, D−, D, D+, C−, C, C+, B−, B, B+, A−, A, A+; C++ sits halfway between C+ and B−. Split letter grades use the midpoint of their native ordinal values. The collector's 0–100 semantic index is not used.

An explicit ordinary grade is the baseline when present. If only an explicit conditional letter is given, that letter enters with its context flag. A bare SB label has no ordered grade and remains unmapped. All conditional alternatives and original wording remain in the long table. Buckets count explicit structured labels, not every contextual remark in the underlying reviews. A card can belong to more than one bucket.

Only a bucket with **more than 20 distinct rated cards for that creator** receives an additive term. Eligibility uses ratings metadata only and stays fixed across folds. Training then uses main-set cards with reported GIH and at least 500 observations. Each card has equal weight. Alternate-grade uncertainty is counted separately from deck conditions. No pooling of creator-specific bucket counts is used to make a sparse term eligible.

The final model for the complete table is fitted to all eligible reported main-set cards. The table includes rates for rated cards with suppressed outcomes and Special Guests through that same final model; their actual rates stay missing when suppressed. Final fitted values are retrospective mappings. Accuracy scores use separate five-fold held-out mappings, repeated twenty times. Final fitted MAE is shown only as a fit diagnostic, not an honest forecast score.

## Contextual bucket counts

'''
    text+=markdown_table(buckets[buckets.distinct_rated_cards.gt(0)],['creator','bucket_label','distinct_rated_cards','main_rated_cards','training_cards','include_term'])+'\n\n'
    text+='Only LR build-around qualifies. Its coefficient is an association conditional on the selected published grade, not a causal penalty for playing build-arounds. Ordinary grades with conditional ceilings and conditional-only grades have different meanings; the single additive term is a simple approximation under the requested bucket rule.\n\n'
    text+='## Held-out comparison\n\n'+markdown_table(scores,['creator','training_cards','raw_grade_pearson_r','raw_grade_spearman_rho','held_out_mae_pp','final_fit_mae_pp','context_terms'])+'\n\nCoverage differs across creators. Rank correlations use the unadjusted native grade input; context-adjusted predictions have a separate held-out correlation in the CSV.\n'
    for key,model in spec['models'].items():
        text+='\n## '+model['name']+'\n\n'
        text+=f"Native scale: {model['raw_scale']}. Collected {model['collected_cards']} ratings; mapped {model['mapped_cards']}; fitted on {model['training_cards']} main-set cards.\n\n"
        text+='Final model: **grade curve'
        for bucket,beta in model['context_offsets_pp'].items():text+=f" {beta:+.4f} pp × {BUCKET_LABELS[bucket]} indicator"
        text+='**.\n\n'
        text+=markdown_table(knots[knots.source.eq(key)],['grade_labels','native_grade_value','training_cards','baseline_mapped_wr'])+'\n\n'
        text+=f"Edition: {model['date']}. [Original source]({model['source_url']}). {model['caveat']}\n"
    text+='''
## Files and reproduction

`creator_predictions_wide.csv` is the complete card table with each creator's original grade and final mapped percentage. `creator_predictions_long.csv` additionally retains selected grades, context flags, source links, missing reasons and held-out values. `creator_models.json` is the exact machine-readable model, with grade knots and context coefficients. `creator_grade_mappings.csv` and `context_memberships.csv` allow tier-by-tier and card-by-card audits. The workbook contains the full table, model mappings, contextual counts and sources.

Run `python creator_native.py --truth inputs/local/17lands_fra_all_time.json`. The snapshot checksum is enforced. LR's 283 captured grades include 97 from the rare review; three rare-review caption grades remain ambiguous and are missing. Lords of Limited and Jim Davis have no complete raw grade table in the retained sources, so no numerical model is invented for them. Creator transcript grades remain collector transcriptions; dated live editions have the provenance limitations documented in the original analysis. Live 17Lands data and derived tables remain local, with attribution to 17Lands and no endorsement implied.
'''
    (output/'creator_models.md').write_text(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--truth',type=Path,required=True)
    run(parser.parse_args().truth)
