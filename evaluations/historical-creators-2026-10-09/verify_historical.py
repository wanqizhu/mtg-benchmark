"""Independent data-flow and export checks for the historical transfer experiment."""
import argparse,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from historical_creators import ROOT,PREVIOUS,CREATORS,fit_stage,sha,norm


def run(truth_dir,source_scratch,output):
    analysis=json.loads((output/'analysis.json').read_text());package=analysis['models'];checks={}
    assert sha(output/'historical_models.json')==analysis['fit_lock']['models_sha256']
    assert sha(ROOT/'inputs/historical_ratings.json')==package['sources_file_sha256']
    assert analysis['fra_truth_sha256']==json.loads((PREVIOUS/'inputs/ground_truth_manifest.json').read_text())['sha256']
    checks['source_and_model_and_frozen_FRA_hashes_match']=True
    plans=json.loads((source_scratch/'freeze_plan.json').read_text());mismatch=[]
    for p in plans:
        ratings=json.loads((source_scratch/'tiers'/f"{p['creator']}_{p['set']}_frozen.json").read_text())['ratings'];byid={str(r['card_id']):r['tier'] for r in ratings}
        mismatch.extend((p['creator'],p['set'],cid) for cid,g in p['grades'].items() if byid[cid]!=g)
    assert not mismatch
    checks['original_tier_history_replay_matches_frozen_grades']=True;checks['frozen_individual_tier_versions']=len(plans)
    # A repeat fit is forbidden from touching any previous/FRA inputs, including
    # raw card outcomes, source grades, and catalog. This checks the real fit path.
    forbidden=[PREVIOUS.resolve(),(ROOT/'inputs/fra_ratings.json').resolve(),(ROOT/'inputs/fra_qualitative_reviews.json').resolve()]
    old_text,old_bytes=Path.read_text,Path.read_bytes;accesses=[]
    def guard(path):
        absolute=path.resolve();assert not any(absolute==f or f in absolute.parents for f in forbidden),f'FRA information read during historical fit: {absolute}'
        accesses.append(str(absolute))
    def read_text(path,*a,**kw):guard(path);return old_text(path,*a,**kw)
    def read_bytes(path,*a,**kw):guard(path);return old_bytes(path,*a,**kw)
    with tempfile.TemporaryDirectory(prefix='mtg-history-audit-') as temp:
        with patch.object(Path,'read_text',read_text),patch.object(Path,'read_bytes',read_bytes):repeat=fit_stage(truth_dir,Path(temp))
    for key,m in package['models'].items():
        r=repeat['models'][key];assert m['train_sets']==r['train_sets'];assert 'FRA' not in m['train_sets'];assert np.allclose(m['coefficients'],r['coefficients'],rtol=0,atol=1e-9)
        assert m['grade_lambda']==r['grade_lambda'] and m['context_lambda']==r['context_lambda'] and m['use_gaps']==r['use_gaps']
        assert np.all(np.diff(m['coefficients'][:len(m['levels'])])>=-1e-8)
        assert all(0<=x['mapped_wr']<=100 for x in m['grade_mapping'])
    checks['full_refit_with_all_FRA_files_inaccessible_succeeds']=True
    checks['repeat_fit_coefficients_and_historical_model_choices_identical']=True
    checks['fitted_native_curves_monotone_and_percentage_scale_valid']=True
    checks['fit_file_accesses']=sorted(set(accesses))
    wide=pd.read_csv(output/'fra_predictions_wide.csv');long=pd.read_csv(output/'fra_predictions_long.csv');scores=pd.read_csv(output/'fra_scores.csv')
    assert len(wide)==295 and wide.card.nunique()==295
    assert len(long)==295*5 and not long.duplicated(['creator','card']).any()
    assert not long.prediction.dropna().lt(0).any() and not long.prediction.dropna().gt(100).any()
    shared=set.intersection(*[set(long.loc[(long.creator==k)&long.primary&long.prediction.notna(),'card']) for k in CREATORS]);assert len(shared)==265
    for key in CREATORS:
        rr=long[(long.creator==key)&long.card.isin(shared)];x=rr.actual.to_numpy();p=rr.prediction.to_numpy();s=scores[(scores.creator==key)&(scores.pool=='common cards across all five creators')].iloc[0]
        assert abs(np.mean(abs(p-x))-s.mae_pp)<1e-10
        assert abs(np.sqrt(np.mean((p-x)**2))-s.rmse_pp)<1e-10
        assert abs(np.corrcoef(x,p)[0,1]-s.pearson)<1e-10
    checks['295_unique_cards_and_1475_creator_card_rows']=True
    checks['265_card_shared_scores_recomputed_independently']=True
    # Official historical FIRST_WEEK responses must differ from mature data.
    for s,info in package['truth_manifests'].items():
        first=json.loads((truth_dir/f'{s}_FIRST_WEEK.json').read_text())['data'];mature=json.loads((truth_dir/f'{s}.json').read_text())['data']
        assert sha(truth_dir/f'{s}_FIRST_WEEK.json')!=sha(truth_dir/f'{s}.json')
        assert sum(r['ever_drawn_game_count'] for r in first)<=sum(r['ever_drawn_game_count'] for r in mature)
    checks['FIRST_WEEK_responses_differ_from_mature_and_have_fewer_games']=True
    checks['historical_target_period']='FIRST_WEEK';checks['training_observations']=sum(m['train_n'] for m in package['models'].values());checks['source_limitations']=['LR partial reconstructed historical coverage','Bot only two qualified editions','Sparse buckets remain below ten','Conditional-grade-gap family added after initial FRA inspection; both versions retained']
    checks['script_sha256']=sha(ROOT/'historical_creators.py')
    (output/'validation.json').write_text(json.dumps(checks,indent=2))
    print('VALIDATED',checks['training_observations'],'historical observations; 295 cards; 265 common scored cards; guarded repeat fit had no FRA file access',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--truth-dir',type=Path,required=True);a.add_argument('--source-scratch',type=Path,required=True);a.add_argument('--output',type=Path,default=ROOT/'results');x=a.parse_args();run(x.truth_dir,x.source_scratch,x.output)
