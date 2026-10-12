"""Independent data-flow and export checks for the historical transfer experiment."""
import argparse,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from historical_creators import ROOT,PREVIOUS,CREATORS,fit_stage,sha,norm,nearest_topup,prepare


def run(truth_dir,source_scratch,output):
    analysis=json.loads((output/'analysis.json').read_text());package=analysis['models'];checks={}
    # Independent hand-calculated cases exercise equal-distance borrowing,
    # donor capacity, unchanged dense classes and an empty target class.
    example=[{'grade':'F','n':5,'coefficient':40.},{'grade':'D-','n':1,'coefficient':70.},{'grade':'D','n':20,'coefficient':60.}]
    prediction,donors=nearest_topup('D-',1,70.,example);assert abs(prediction-52.)<1e-10
    example[0]['n']=2
    prediction,donors=nearest_topup('D-',1,70.,example);assert abs(prediction-57.)<1e-10
    assert nearest_topup('D',20,60.,example)==(60.,[])
    prediction,donors=nearest_topup('D+',0,None,example);assert prediction==60. and sum(r['borrowed_weight'] for r in donors)==10.
    assert prepare({'creator':'alexnikolic','native':'SB','contexts':[]})['grade_atoms']=={'F':1.}
    assert prepare({'creator':'limitedresources','native':'Sideboard B','contexts':[{'kind':'sideboard','native':'B'}]})['grade_atoms']=={'F':1.}
    checks['hand_calculated_nearest_class_examples_and_SB_alias_pass']=True
    # Independently specified rating cases check the user-requested semantics.
    pair=prepare({'creator':'alexnikolic','native':'C+ / build-around B+',
                  'ordinary_native':'C+','contexts':[{'kind':'build_around','native':'B+'},
                                                    {'kind':'synergy','native':None}]})
    assert pair['grade_atoms']=={'C+':1.} and pair['buckets']==[]
    only=prepare({'creator':'limitedresources','native':'Build-around A',
                  'ordinary_native':'A','contexts':[{'kind':'build_around','native':'A'}]})
    assert only['rating_structure']=='conditional_only' and only['grade_atoms']=={'A':1.}
    assert only['buckets']==['context_flag']
    single=prepare({'creator':'alexnikolic','native':'B','ordinary_native':'B',
                    'contexts':[{'kind':'build_around','native':None},{'kind':'synergy','native':None}]})
    assert single['buckets']==['context_flag'] and single['grade_atoms']=={'B':1.}
    copied=prepare({'creator':'limitedresources','native':'B','ordinary_native':'B',
                    'contexts':[{'kind':'build_around','native':'B'}]})
    assert copied['rating_structure']=='single_grade_with_context' and copied['buckets']==['context_flag']
    narrow=prepare({'creator':'limitedresources','native':'B (heavy-green only; B+ in monogreen)',
                    'ordinary_native':None,'contexts':[{'kind':'scenario','native':'B+'},
                                                       {'kind':'build_around','native':'B','relation':'applicability'}]})
    assert narrow['grade_atoms']=={'B':1.} and narrow['buckets']==['context_flag']
    assert all(r['context_gaps']=={} for r in [pair,only,single,copied,narrow])
    checks['explicit_pairs_base_only_and_other_context_cases_once_pass']=True
    assert sha(output/'historical_models.json')==analysis['fit_lock']['models_sha256']
    assert sha(ROOT/'inputs/historical_ratings.json')==package['sources_file_sha256']
    assert analysis['fra_truth_sha256']==json.loads((PREVIOUS/'inputs/ground_truth_manifest.json').read_text())['sha256']
    checks['source_and_model_and_frozen_FRA_hashes_match']=True
    plans=json.loads((source_scratch/'freeze_plan.json').read_text());mismatch=[]
    for p in plans:
        if p['creator'] not in CREATORS:continue
        ratings=json.loads((source_scratch/'tiers'/f"{p['creator']}_{p['set']}_frozen.json").read_text())['ratings'];byid={str(r['card_id']):r['tier'] for r in ratings}
        mismatch.extend((p['creator'],p['set'],cid) for cid,g in p['grades'].items() if byid[cid]!=g)
    assert not mismatch
    checks['original_tier_history_replay_matches_frozen_grades']=True;checks['frozen_individual_tier_versions']=sum(p['creator'] in CREATORS for p in plans)
    # A repeat fit is forbidden from touching any previous/FRA inputs, including
    # raw card outcomes, source grades, and catalog. This checks the real fit path.
    forbidden=[PREVIOUS.resolve(),(ROOT/'inputs/fra_ratings.json').resolve(),(ROOT/'inputs/fra_qualitative_reviews.json').resolve(),output.resolve()]
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
        if m['model_type']=='bot_regression':
            assert m['nominal_degrees_of_freedom']<10
            assert len(m['regression_coefficients'])==m['nominal_degrees_of_freedom']
            assert m['unconstrained_smoother_degrees_of_freedom']<=m['nominal_degrees_of_freedom']+1e-8
            assert np.all(np.diff([x['mapped_wr'] for x in m['grade_mapping']])>=-1e-8)
        else:
            assert m['contexts']==['context_flag'] and len(m['context_mapping'])==1
            assert not m['use_gaps'] and not m['context_gap_mapping']
            assert len(m['coefficients'])==len(m['levels'])+1
            assert not m['has_sb'] and all(g['grade']!='SB' for g in m['grade_mapping'])
            assert 'sideboard' not in m['contexts'] and 'sideboard' not in m['gap_contexts']
            if m['monotone']:assert np.all(np.diff(m['base_coefficients'][:len(m['levels'])])>=-1e-8)
            for grade in m['grade_mapping']:
                n=grade['observed_n'];borrowed=grade['borrowed_weight']
                assert abs(borrowed-max(0,10-n))<1e-8
                assert all(d['borrowed_weight']<=d['available_observations']+1e-8 for d in grade['donors'])
                raw=0 if n==0 else n*grade['base_coefficient']
                target=(raw+sum(d['borrowed_weight']*d['base_coefficient'] for d in grade['donors']))/(n+borrowed)
                assert abs(target-grade['mapped_wr'])<1e-8
            for term in m['context_mapping']+m['context_gap_mapping']:
                final=term.get('offset_pp',term.get('pp_per_native_grade_step'))
                assert abs(final-min(1,term['observed_n']/10)*term['base_coefficient'])<1e-8
        assert all(0<=x['mapped_wr']<=100 for x in m['grade_mapping'])
    checks['full_refit_with_all_FRA_files_inaccessible_succeeds']=True
    checks['repeat_fit_coefficients_and_historical_model_choices_identical']=True
    checks['native_base_curve_constraints_and_percentage_scale_valid']=True
    checks['ten_sample_formula_and_donor_capacities_verified']=True
    checks['bot_has_fewer_than_ten_total_fitted_coefficients']=True
    checks['SB_grouped_with_F_without_separate_sideboard_parameters']=True
    checks['three_letter_models_have_exactly_one_shared_context_coefficient']=True
    checks['final_curve_monotonicity']={k:m['final_mapping_monotone'] for k,m in package['models'].items()}
    checks['fit_file_accesses']=sorted(set(accesses))
    wide=pd.read_csv(output/'fra_predictions_wide.csv');long=pd.read_csv(output/'fra_predictions_long.csv');scores=pd.read_csv(output/'fra_scores.csv')
    assert len(wide)==295 and wide.card.nunique()==295
    assert len(long)==295*len(CREATORS) and not long.duplicated(['creator','card']).any()
    assert set(long.creator)==set(CREATORS) and 'marcanderson' not in set(long.creator)
    assert not long.prediction.dropna().lt(0).any() and not long.prediction.dropna().gt(100).any()
    shared=set.intersection(*[set(long.loc[(long.creator==k)&long.primary&long.prediction.notna(),'card']) for k in CREATORS]);assert len(shared)==266
    for key in CREATORS:
        rr=long[(long.creator==key)&long.card.isin(shared)];x=rr.actual.to_numpy();p=rr.prediction.to_numpy();s=scores[(scores.creator==key)&(scores.pool=='common cards across all active creators')].iloc[0]
        assert abs(np.mean(abs(p-x))-s.mae_pp)<1e-10
        assert abs(np.sqrt(np.mean((p-x)**2))-s.rmse_pp)<1e-10
        assert abs(np.corrcoef(x,p)[0,1]-s.pearson)<1e-10
    checks['295_unique_cards_and_1180_creator_card_rows']=True
    checks['four_active_creators_no_Marc']=True
    checks['266_card_shared_scores_recomputed_independently']=True
    # Official historical FIRST_WEEK responses must differ from mature data.
    for s,info in package['truth_manifests'].items():
        first=json.loads((truth_dir/f'{s}_FIRST_WEEK.json').read_text())['data'];mature=json.loads((truth_dir/f'{s}.json').read_text())['data']
        assert sha(truth_dir/f'{s}_FIRST_WEEK.json')!=sha(truth_dir/f'{s}.json')
        assert sum(r['ever_drawn_game_count'] for r in first)<=sum(r['ever_drawn_game_count'] for r in mature)
    checks['FIRST_WEEK_responses_differ_from_mature_and_have_fewer_games']=True
    checks['historical_target_period']='FIRST_WEEK';checks['training_observations']=sum(m['train_n'] for m in package['models'].values());checks['source_limitations']=['LR partial reconstructed historical coverage','Bot only two qualified editions','Sparse buckets remain below ten','Shared context policy specified after FRA inspection; retrospective comparison']
    checks['script_sha256']=sha(ROOT/'historical_creators.py')
    (output/'validation.json').write_text(json.dumps(checks,indent=2))
    print('VALIDATED',checks['training_observations'],'historical observations; 295 cards;',len(shared),'common scored cards; guarded repeat fit had no FRA file access',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--truth-dir',type=Path,required=True);a.add_argument('--source-scratch',type=Path,required=True);a.add_argument('--output',type=Path,default=ROOT/'results');x=a.parse_args();run(x.truth_dir,x.source_scratch,x.output)
