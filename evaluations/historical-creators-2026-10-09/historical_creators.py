"""Historical-only native-grade calibration, then frozen FRA evaluation.

Two stages are deliberately separate: fit never loads FRA outcomes; evaluate
loads already locked model specifications and the October 7 truth snapshot.
All response envelopes remain local, outside Git. No creator semantic 0–100
normalization is used. Percentages are fitted from raw native grade buckets.
"""
import argparse, hashlib, json, re, unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import pearsonr, spearmanr

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'fra-week-one-2026-10-07'
CREATORS={'casualspike':'Casual Spike','alexnikolic':'Alex Nikolic','marcanderson':'Marc Anderson','limitedresources':'LSV + Marshall / Limited Resources','draftsim':'Draftsim bot'}
LETTER={'F':0.,'D-':1.,'D':2.,'D+':3.,'C-':4.,'C':5.,'C+':6.,'C++':6.5,'B-':7.,'B':8.,'B+':9.,'A-':10.,'A':11.,'A+':12.}
CONTEXTS=['build_around','synergy','sideboard','scenario','alternate']
GRADE_LAMBDAS=[0.,2.,10.,50.]
CONTEXT_LAMBDAS=[2.,10.,50.]


def norm(s):
    return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',s.split(' // ')[0]).lower())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def value(label):return LETTER[label] if label in LETTER else float(label)


def atoms(label):
    if label is None:return {}
    label=str(label).strip().replace('−','-').replace('–','-')
    if label=='SB':return {'SB':1.}
    if label in LETTER:return {label:1.}
    if re.fullmatch(r'\d+(?:\.\d+)?',label):return {format(float(label),'g'):1.}
    parts=[x.strip() for x in re.split(r'\s*/\s*|\s+to\s+',label)]
    if len(parts)>1 and all(p in LETTER or re.fullmatch(r'\d+(?:\.\d+)?',p) for p in parts):
        c=Counter(parts);return {k:n/len(parts) for k,n in c.items()}
    return {}


def prepare(row):
    r=dict(row);flags=set()
    for c in r.get('contexts',[]):
        k=c['kind'];k='build_around' if k in ['buildaround','build_around_flag'] else k
        assert k in CONTEXTS,(r,k);flags.add(k)
    label=r.get('ordinary_native');basis='ordinary raw grade'
    a=atoms(label)
    if not a:a=atoms(r.get('native'));label=r.get('native');basis='published raw grade'
    if not a:
        for c in r.get('contexts',[]):
            if atoms(c.get('native')):
                a=atoms(c['native']);label=c['native'];basis='explicit conditional grade; no ordinary grade provided';break
    if r.get('grade_weights'):
        a=r['grade_weights'];assert abs(sum(a.values())-1)<1e-8
    if str(r.get('native','')).lower().startswith(('build-around','sideboard')):basis='explicit conditional grade; no ordinary grade provided'
    if 'SB' in a:flags.discard('sideboard') # SB has its own unordered bucket, no collinear extra flag.
    gaps={}
    if a and 'SB' not in a:
        base=sum(value(g)*w for g,w in a.items())
        for c in r.get('contexts',[]):
            k='build_around' if c['kind']=='build_around_flag' else c['kind'];ca=atoms(c.get('native'))
            if ca and 'SB' not in ca:
                gap=sum(value(g)*w for g,w in ca.items())-base
                if abs(gap)>1e-9:gaps.setdefault(k,[]).append(gap)
    gaps={k:float(np.mean(v)) for k,v in gaps.items()}
    if not a:basis='No verified letter or numeric grade'
    r.update(grade_used=label,grade_atoms=a,grade_basis=basis,buckets=sorted(flags),context_gaps=gaps)
    return r


def training_inputs(truth_dir,period):
    doc=json.loads((ROOT/'inputs/historical_ratings.json').read_text());rows=[];join_exclusions=[];files={}
    for s in sorted({r['set'] for r in doc['rows']}):
        path=truth_dir/(f'{s}_FIRST_WEEK.json' if period=='FIRST_WEEK' else f'{s}.json')
        data=json.loads(path.read_text())['data'];lookup={norm(x['name']):x for x in data}
        assert len(data)==len(lookup)
        files[s]={'file':str(path.resolve()),'sha256':sha(path),'period':period,'url':f'https://www.17lands.com/api/card_data?expansion={s}&event_type=PremierDraft&time_period={period}'}
        for raw in [r for r in doc['rows'] if r['set']==s]:
            r=prepare(raw);t=lookup.get(norm(r['name']))
            if not t:
                join_exclusions.append({'creator':r['creator'],'set':s,'name':r['name'],'reason':'Card has no historical outcome row; no alias invented'});continue
            rate=t['ever_drawn_win_rate'];n=t['ever_drawn_game_count']
            if rate is None or n<500 or not r['grade_atoms']:
                join_exclusions.append({'creator':r['creator'],'set':s,'name':r['name'],'native':r['native'],'n':n,'reason':'Suppressed GIH / fewer than 500 games / no explicit grade'});continue
            assert 0<=rate<=1 and s!='FRA'
            r.update(name=t['name'],actual=100*rate,gih_n=n);rows.append(r)
    return doc,rows,join_exclusions,files


class BucketModel:
    def __init__(self,monotone,grade_lambda,context_lambda,use_gaps=False):
        self.monotone=monotone;self.grade_lambda=grade_lambda;self.context_lambda=context_lambda;self.use_gaps=use_gaps

    def basis(self,row):
        x=np.zeros(len(self.levels)+int(self.has_sb)+len(self.contexts)+len(self.gap_contexts));unseen=[]
        for label,w in row['grade_atoms'].items():
            if label=='SB':
                if not self.has_sb:return None,['SB has no reported historical training outcome']
                x[len(self.levels)]+=w;continue
            v=value(label)
            if v in self.levels:x[self.levels.index(v)]+=w
            elif v<self.levels[0]:x[0]+=w;unseen.append(label+' below supported range; nearest grade used')
            elif v>self.levels[-1]:x[len(self.levels)-1]+=w;unseen.append(label+' above supported range; nearest grade used')
            else:
                j=int(np.searchsorted(self.levels,v));frac=(v-self.levels[j-1])/(self.levels[j]-self.levels[j-1]);x[j-1]+=w*(1-frac);x[j]+=w*frac
                unseen.append(label+' absent from training; interpolated between adjacent native grades')
        for k in row['buckets']:
            if k in self.contexts:x[len(self.levels)+int(self.has_sb)+self.contexts.index(k)]=1
            else:unseen.append(k+' has no historical fitted coefficient; zero offset used')
        for i,k in enumerate(self.gap_contexts):x[len(self.levels)+int(self.has_sb)+len(self.contexts)+i]=row.get('context_gaps',{}).get(k,0)
        return x,unseen

    def fit(self,rows):
        self.levels=sorted({value(g) for r in rows for g in r['grade_atoms'] if g!='SB'})
        self.labels={v:sorted({g for r in rows for g in r['grade_atoms'] if g!='SB' and value(g)==v}) for v in self.levels}
        self.has_sb=any('SB' in r['grade_atoms'] for r in rows)
        self.contexts=[c for c in CONTEXTS if any(c in r['buckets'] for r in rows)]
        self.gap_contexts=[c for c in CONTEXTS if self.use_gaps and any(abs(r.get('context_gaps',{}).get(c,0))>1e-9 for r in rows)]
        assert self.levels
        x=np.stack([self.basis(r)[0] for r in rows]);y=np.array([r['actual'] for r in rows]);nbase=len(self.levels)+int(self.has_sb);p=x.shape[1]
        regularizer=[]
        for j in range(1,len(self.levels)):
            r=np.zeros(p);gap=self.levels[j]-self.levels[j-1];r[j]=np.sqrt(self.grade_lambda)/gap;r[j-1]=-r[j];regularizer.append(r)
        if self.has_sb:
            r=np.zeros(p);r[len(self.levels)]=np.sqrt(self.grade_lambda);r[0]=-r[len(self.levels)];regularizer.append(r)
        for j in range(nbase,p):
            r=np.zeros(p);r[j]=np.sqrt(self.context_lambda);regularizer.append(r)
        penalty=np.array(regularizer).reshape(-1,p)
        design=np.vstack([x,penalty]);target=np.r_[y,np.zeros(len(penalty))]
        if self.monotone:
            transform=np.eye(p);transform[:len(self.levels),:len(self.levels)]=np.tril(np.ones((len(self.levels),len(self.levels))))
            low=np.full(p,-np.inf);high=np.full(p,np.inf);low[0]=0;high[0]=100
            low[1:len(self.levels)]=0
            if self.gap_contexts:low[nbase+len(self.contexts):]=0
            if self.has_sb:low[len(self.levels)]=0;high[len(self.levels)]=100
            solution=lsq_linear(design@transform,target,bounds=(low,high),tol=1e-11,max_iter=300)
            assert solution.success,solution.message;self.coef=transform@solution.x
        else:self.coef=np.linalg.lstsq(design,target,rcond=None)[0]
        self.train_n=len(rows);self.train_sets=sorted({r['set'] for r in rows});self.training_rmse=float(np.sqrt(np.mean((x@self.coef-y)**2)))
        return self

    def predict(self,rows):
        result=[];support=[]
        for r in rows:
            if not r['grade_atoms']:result.append(None);support.append(['No raw native grade']);continue
            x,notes=self.basis(r);result.append(None if x is None else float(np.clip(x@self.coef,0,100)));support.append(notes)
        return result,support

    def specification(self,rows):
        nbase=len(self.levels)+int(self.has_sb);counts=[]
        labels=sorted({g for r in rows for g in r['grade_atoms']},key=lambda x:(x=='SB',0 if x=='SB' else value(x)))
        for g in labels:
            members=[r for r in rows if g in r['grade_atoms']]
            pred=self.predict([{'grade_atoms':{g:1.},'buckets':[]}])[0][0]
            counts.append({'grade':g,'native_value':None if g=='SB' else value(g),'card_set_observations':len(members),'distinct_card_names':len({norm(r['name']) for r in members}),'effective_weight':sum(r['grade_atoms'][g] for r in members),'mapped_wr':pred,'observed_mean_wr':float(np.mean([r['actual'] for r in members]))})
        contexts=[{'bucket':k,'card_set_observations':sum(k in r['buckets'] for r in rows),'distinct_card_names':len({norm(r['name']) for r in rows if k in r['buckets']}),'offset_pp':float(self.coef[nbase+i])} for i,k in enumerate(self.contexts)]
        gap_mapping=[{'bucket':k,'nonzero_gap_card_set_observations':sum(abs(r.get('context_gaps',{}).get(k,0))>1e-9 for r in rows),'pp_per_native_grade_step':float(self.coef[nbase+len(self.contexts)+i])} for i,k in enumerate(self.gap_contexts)]
        return {'use_gaps':self.use_gaps,'gap_contexts':self.gap_contexts,'context_gap_mapping':gap_mapping,'family':'monotone native bucket curve + all observed context offsets' if self.monotone else 'unordered native bucket means with adjacent-bucket smoothing + all observed context offsets','monotone':self.monotone,'grade_lambda':self.grade_lambda,'context_lambda':self.context_lambda,'levels':self.levels,'has_sb':self.has_sb,'contexts':self.contexts,'coefficients':self.coef.tolist(),'train_n':self.train_n,'train_sets':self.train_sets,'training_rmse_pp':self.training_rmse,'grade_mapping':counts,'context_mapping':contexts}

    @classmethod
    def restore(cls,spec):
        m=cls(spec['monotone'],spec['grade_lambda'],spec['context_lambda'],spec.get('use_gaps',False));m.gap_contexts=spec.get('gap_contexts',[]);m.levels=spec['levels'];m.has_sb=spec['has_sb'];m.contexts=spec['contexts'];m.coef=np.array(spec['coefficients']);return m


def metrics(actual,pred):
    a=np.array(actual,dtype=float);p=np.array([np.nan if x is None else x for x in pred],dtype=float);good=np.isfinite(a)&np.isfinite(p);a=a[good];p=p[good];err=p-a
    return {'n':len(a),'pearson':float(pearsonr(a,p).statistic) if len(a)>2 and np.std(p)>1e-10 else None,'spearman':float(spearmanr(a,p).statistic) if len(a)>2 and np.std(p)>1e-10 else None,'mae_pp':float(np.mean(abs(err))) if len(a) else None,'rmse_pp':float(np.sqrt(np.mean(err**2))) if len(a) else None,'bias_pp':float(np.mean(err)) if len(a) else None}


def folds(rows):
    for s in sorted({r['set'] for r in rows}):
        test=[r for r in rows if r['set']==s];names={norm(r['name']) for r in test}
        train=[r for r in rows if r['set']!=s and norm(r['name']) not in names]
        assert not names & {norm(r['name']) for r in train}
        yield s,train,test


def choose_model(rows):
    candidates=[]
    for use_gaps in [False,True]:
        for monotone in [False,True]:
            for gl in GRADE_LAMBDAS:
                for cl in CONTEXT_LAMBDAS:
                    actual=[];pred=[];fold_rows=[]
                    for s,train,test in folds(rows):
                        model=BucketModel(monotone,gl,cl,use_gaps).fit(train);pr,_=model.predict(test);actual.extend(r['actual'] for r in test);pred.extend(pr)
                        fold_rows.append({'set':s,'train_n':len(train),'purged_reprint_rows':sum(r['set']!=s for r in rows)-len(train),'test_n':len(test),**metrics([r['actual'] for r in test],pr)})
                    result=metrics(actual,pred);candidates.append({'use_gaps':use_gaps,'monotone':monotone,'grade_lambda':gl,'context_lambda':cl,**result,'folds':fold_rows})
    # Only historical purged-LOSO RMSE is used for model choice; fixed complexity tie break.
    winner=min(candidates,key=lambda c:(round(c['rmse_pp'],10),not c['monotone'],c['use_gaps'],-c['grade_lambda'],-c['context_lambda']))
    final=BucketModel(winner['monotone'],winner['grade_lambda'],winner['context_lambda'],winner['use_gaps']).fit(rows)
    return final,winner,candidates


def fit_stage(truth_dir,output):
    doc,rows,join_exclusions,files=training_inputs(truth_dir,'FIRST_WEEK');output.mkdir(parents=True,exist_ok=True)
    all_rows=doc['rows'];models={};validation={};audit=[];bucket_counts=[];cv_ledger=[]
    for creator,name in CREATORS.items():
        train=[r for r in rows if r['creator']==creator];model,winner,candidates=choose_model(train);spec=model.specification(train)
        spec.update(name=name,source_rows=sum(r['creator']==creator for r in all_rows),historical_sets=len(model.train_sets),chosen_by='Minimum RMSE in historical leave-one-set-out validation; shared card names purged from each training fold',historical_validation=winner)
        # Mapping sensitivity holds chosen hyperparameters fixed and omits one whole set.
        sensitivities=[]
        for s,tr,te in folds(train):
            m=BucketModel(winner['monotone'],winner['grade_lambda'],winner['context_lambda'],winner['use_gaps']).fit(tr)
            sensitivities.append({'omitted_set':s,'specification':m.specification(tr)})
            pr,notes=m.predict(te)
            for r,pred,note in zip(te,pr,notes):cv_ledger.append({'creator':creator,'held_out_set':s,'card':r['name'],'native':r['native'],'actual':r['actual'],'prediction':pred,'support_notes':'; '.join(note),'training_sets':'; '.join(m.train_sets)})
        spec['leave_one_set_out_mapping_sensitivity']=sensitivities
        for mapping in spec['grade_mapping']:
            trial={'grade_atoms':{mapping['grade']:1.},'buckets':[]};values=[BucketModel.restore(s['specification']).predict([trial])[0][0] for s in sensitivities];values=[x for x in values if x is not None]
            mapping.update(loso_min=min(values) if values else None,loso_max=max(values) if values else None)
        models[creator]=spec;validation[creator]=candidates
        for raw_grade,n in Counter(r['native'] for r in all_rows if r['creator']==creator).items():
            members=[r for r in train if r['native']==raw_grade];bucket_counts.append({'creator':creator,'kind':'raw published label','bucket':raw_grade,'published_card_set_observations':n,'training_card_set_observations':len(members),'training_distinct_card_names':len({norm(r['name']) for r in members})})
        for ctx in CONTEXTS:
            raw_prepared=[prepare(r) for r in all_rows if r['creator']==creator];members=[r for r in train if ctx in r['buckets']]
            bucket_counts.append({'creator':creator,'kind':'context','bucket':ctx,'published_card_set_observations':sum(ctx in r['buckets'] for r in raw_prepared),'training_card_set_observations':len(members),'training_distinct_card_names':len({norm(r['name']) for r in members})})
        print('FIT',creator,len(train),winner['monotone'],winner['grade_lambda'],winner['context_lambda'],'conditional gaps',winner['use_gaps'],round(winner['rmse_pp'],3),flush=True)
    package={'created_utc':datetime.now(timezone.utc).isoformat(),'fit_scope':'Historical nonbasic main-set PremierDraft FIRST_WEEK GIH, all colors and users, reported rates and n >= 500; FRA outcomes never loaded by fit stage','equal_weight':'One card-set observation has one unit of loss; no weighting by GIH count or creator semantic grade normalization','raw_grade_policy':'Each native letter/decimal bucket is fitted separately. Split host grades average fitted bucket values, with explicit nested host weights where recorded. SB is an unordered native bucket. Conditional-only grades use their explicit letter and retain the context flag. Unobserved grades interpolate on the native order or use the nearest supported endpoint. Unidentified SB stays unmapped.','penalty_policy':'Adjacent bucket differences divided by native-grade spacing receive grade_lambda penalty; unordered SB borrows strength from the lowest supported ordinary grade with the same penalty. Every observed context coefficient receives context_lambda penalty. An optional nonnegative conditional-grade-gap effect (conditional raw grade minus ordinary raw grade, on the native order) is compared with the flag-only model in historical CV. These gap coefficients share context_lambda shrinkage. No minimum-count bucket dropping or arbitrary 10-card claim.','sources_file_sha256':sha(ROOT/'inputs/historical_ratings.json'),'truth_manifests':files,'models':models,'join_exclusions':join_exclusions}
    modelpath=output/'historical_models.json';modelpath.write_text(json.dumps(package,ensure_ascii=False,indent=2,allow_nan=False))
    (output/'fit_lock.json').write_text(json.dumps({'models_sha256':sha(modelpath),'locked_before_FRA_outcome_read':True,'utc':datetime.now(timezone.utc).isoformat(),'historical_training_sets':sorted(files),'historical_ratings_sha256':sha(ROOT/'inputs/historical_ratings.json')},indent=2))
    (output/'historical_cv_candidates.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2,allow_nan=False))
    pd.DataFrame(cv_ledger).to_csv(output/'historical_cv_predictions.csv',index=False)
    pd.DataFrame(bucket_counts).to_csv(output/'historical_bucket_counts.csv',index=False)
    pd.DataFrame([{**{k:r.get(k) for k in ['creator','set','name','native','grade_used','actual','gih_n','grade_basis','source_url']},'contexts':'; '.join(r['buckets'])} for r in rows]).to_csv(output/'historical_training_observations.csv',index=False)
    # This is a retrospective transfer experiment. The compute lock prevents label leakage;
    # it does not claim an analyst preregistration before FRA results were published.
    return package


def evaluate_stage(truth_dir,output,fra_truth):
    lock=json.loads((output/'fit_lock.json').read_text());assert sha(output/'historical_models.json')==lock['models_sha256']
    pack=json.loads((output/'historical_models.json').read_text());assert 'FRA' not in lock['historical_training_sets']
    manifest=json.loads((PREVIOUS/'inputs/ground_truth_manifest.json').read_text());assert sha(fra_truth)==manifest['sha256']
    cards=json.loads((PREVIOUS/'inputs/cards.json').read_text());outcomes=json.loads(fra_truth.read_text())['data'];truth={norm(r['name']):r for r in outcomes};cardmap={norm(c['name']):c for c in cards}
    fra=json.loads((ROOT/'inputs/fra_ratings.json').read_text());sources={s['id']:s for s in fra['sources']};records=[];long=[];score_rows=[];diagnostics=[];sensitivities=[]
    historical_doc,history,_,_=training_inputs(truth_dir,'FIRST_WEEK')
    _,mature,_,_=training_inputs(truth_dir,'ALL_TIME')
    for creator,name in CREATORS.items():
        model=BucketModel.restore(pack['models'][creator]);raw=[prepare({**r,'creator':creator,'set':'FRA'}) for r in sources[creator]['rows']]
        byname={norm(r['name']):r for r in raw};assert len(byname)==len(raw);assert set(byname)<=set(cardmap)
        predictions,notes=model.predict(raw);preds={norm(r['name']):(r,p,n) for r,p,n in zip(raw,predictions,notes)}
        actual=[];predicted=[];scored=[]
        for c in cards:
            k=norm(c['name']);t=truth.get(k);item=preds.get(k)
            wr=None if not t or t['ever_drawn_win_rate'] is None else 100*t['ever_drawn_win_rate'];n=0 if not t else t['ever_drawn_game_count']
            if item:r,pred,support=item
            else:r=None;pred=None;support=['Creator supplied no grade']
            primary=c['set']=='fra' and not c['type'].startswith('Basic Land') and wr is not None and n>=500
            rec={'creator':creator,'card':c['name'],'set':c['set'],'rarity':c['rarity'],'color':''.join(c['colors']) or 'C','native_grade':None if r is None else r['native'],'grade_used':None if r is None else r['grade_used'],'contexts':[] if r is None else r['buckets'],'grade_atoms':{} if r is None else r['grade_atoms'],'context_gaps':{} if r is None else r['context_gaps'],'context_details':[] if r is None else r['contexts'],'prediction':pred,'actual':wr,'gih_n':n,'error_pp':None if pred is None or wr is None else pred-wr,'primary':primary,'support_notes':support,'source_url':None if r is None else r.get('source_url'),'oracle':c['oracle'],'cost':c['cost'],'type':c['type'],'scryfall_uri':c['scryfall_uri']}
            long.append(rec)
            if primary:
                actual.append(wr);predicted.append(pred)
                if pred is not None:scored.append(rec)
        score_rows.append({'creator':creator,'name':name,'pool':'own reported main-set coverage','historical_sets':pack['models'][creator]['historical_sets'],'historical_training_n':pack['models'][creator]['train_n'],**metrics(actual,predicted)})
        baseline=float(np.mean([r['actual'] for r in history if r['creator']==creator]));score_rows.append({'creator':creator,'name':name+' historical constant','pool':'historical mean baseline',**metrics(actual,[baseline]*len(actual))})
        for groupcol in ['rarity','color']:
            for group in sorted({r[groupcol] for r in scored}):
                subset=[r for r in scored if r[groupcol]==group]
                diagnostics.append({'creator':creator,'group_type':groupcol,'group':group,**metrics([r['actual'] for r in subset],[r['prediction'] for r in subset])})
        # Prespecified sensitivity: same hyperparameters, historical ALL_TIME target.
        chosen=pack['models'][creator];past=[r for r in mature if r['creator']==creator]
        mt=BucketModel(chosen['monotone'],chosen['grade_lambda'],chosen['context_lambda'],chosen['use_gaps']).fit(past)
        mpr,_=mt.predict(raw);actual_raw=[None if norm(r['name']) not in truth or truth[norm(r['name'])]['ever_drawn_win_rate'] is None else 100*truth[norm(r['name'])]['ever_drawn_win_rate'] for r in raw]
        valid=[i for i,r in enumerate(raw) if cardmap[norm(r['name'])]['set']=='fra' and actual_raw[i] is not None and not cardmap[norm(r['name'])]['type'].startswith('Basic Land')]
        sensitivities.append({'creator':creator,'variant':'Mature ALL_TIME targets; primary hyperparameters locked','training_n':len(past),**metrics([actual_raw[i] for i in valid],[mpr[i] for i in valid])})
        # Exclude historical reprints of FRA cards, while keeping original hyperparameters.
        fra_names=set(cardmap);past=[r for r in history if r['creator']==creator and norm(r['name']) not in fra_names]
        rp=BucketModel(chosen['monotone'],chosen['grade_lambda'],chosen['context_lambda'],chosen['use_gaps']).fit(past);rpr,_=rp.predict(raw)
        sensitivities.append({'creator':creator,'variant':'Remove every historical printing of FRA card names','training_n':len(past),**metrics([actual_raw[i] for i in valid],[rpr[i] for i in valid])})
        # Grade-only conversion demonstrates what the added context terms contribute.
        past=[{**r,'buckets':[],'context_gaps':{}} for r in history if r['creator']==creator];gm=BucketModel(chosen['monotone'],chosen['grade_lambda'],chosen['context_lambda'],chosen['use_gaps']).fit(past);gpr,_=gm.predict([{**r,'buckets':[],'context_gaps':{}} for r in raw])
        sensitivities.append({'creator':creator,'variant':'Grade buckets only; same smoothing','training_n':len(past),**metrics([actual_raw[i] for i in valid],[gpr[i] for i in valid])})
        if creator=='limitedresources':
            # Episodes issued after launch are allowed by user but separated for transparency.
            rare_ids=['b-mJNcpSo-w','y11-rxC1rQk','mxysznkjMB4']
            past=[r for r in history if r['creator']==creator and not any(v in r.get('source_url','') for v in rare_ids)]
            early=BucketModel(chosen['monotone'],chosen['grade_lambda'],chosen['context_lambda'],chosen['use_gaps']).fit(past);epr,_=early.predict(raw)
            sensitivities.append({'creator':creator,'variant':'Exclude after-launch rare episodes (SOS launch-day review retained)','training_n':len(past),**metrics([actual_raw[i] for i in valid],[epr[i] for i in valid])})
        if creator=='draftsim':
            for s in ['HOB','SOS']:
                past=[r for r in history if r['creator']==creator and r['set']==s];single=BucketModel(chosen['monotone'],chosen['grade_lambda'],chosen['context_lambda'],chosen['use_gaps']).fit(past);spr,_=single.predict(raw)
                sensitivities.append({'creator':creator,'variant':f'{s} only; other edition excluded','training_n':len(past),**metrics([actual_raw[i] for i in valid],[spr[i] for i in valid])})
    shared=sorted({r['card'] for r in long if r['primary']} & set.intersection(*[{r['card'] for r in long if r['creator']==c and r['primary'] and r['prediction'] is not None} for c in CREATORS]))
    for creator,name in CREATORS.items():
        subset=[r for r in long if r['creator']==creator and r['card'] in shared]
        score_rows.append({'creator':creator,'name':name,'pool':'common cards across all five creators',**metrics([r['actual'] for r in subset],[r['prediction'] for r in subset])})
    score_df=pd.DataFrame(score_rows);score_df.to_csv(output/'fra_scores.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(output/'fra_group_errors.csv',index=False);pd.DataFrame(sensitivities).to_csv(output/'sensitivity_scores.csv',index=False)
    flat=[{**r,'contexts':'; '.join(r['contexts']),'support_notes':'; '.join(r['support_notes'])} for r in long]
    pd.DataFrame(flat).to_csv(output/'fra_predictions_long.csv',index=False)
    wide=[]
    for c in cards:
        rr={'card':c['name'],'set':c['set'],'rarity':c['rarity']}
        for creator in CREATORS:
            r=next(r for r in long if r['creator']==creator and r['card']==c['name']);rr[creator+'__native']=r['native_grade'];rr[creator+'__predicted_gih']=r['prediction'];rr['actual_gih']=r['actual'];rr['gih_n']=r['gih_n']
        wide.append(rr)
    pd.DataFrame(wide).to_csv(output/'fra_predictions_wide.csv',index=False)
    misses=[]
    for creator in CREATORS:
        scored=[r for r in long if r['creator']==creator and r['primary'] and r['prediction'] is not None]
        for direction in ['overestimated','underestimated']:
            selected=sorted(scored,key=lambda r:r['error_pp'],reverse=direction=='overestimated')[:12]
            for rank,r in enumerate(selected,1):misses.append({**r,'direction':direction,'rank':rank})
    pd.DataFrame([{**r,'contexts':'; '.join(r['contexts']),'support_notes':'; '.join(r['support_notes'])} for r in misses]).to_csv(output/'largest_mistakes.csv',index=False)
    result={'models':pack,'fit_lock':lock,'fra_truth_sha256':sha(fra_truth),'fra_truth_fetched_at':manifest.get('retrieved_at'),'scores':score_rows,'sensitivity_scores':sensitivities,'group_errors':diagnostics,'records':long,'largest_mistakes':misses,'common_card_count':len(shared),'source_audit':json.loads((ROOT/'inputs/source_audit.json').read_text()),'validation':{'no_FRA_outcomes_in_fit':True,'models_file_still_matches_lock':sha(output/'historical_models.json')==lock['models_sha256'],'primary_outcome_snapshot':'Original October 7 ALL_TIME FRA response, not a newly fetched snapshot; early-format horizon is approximately first week','historical_target_period':'FIRST_WEEK','main_nonbasic_catalog_count':sum(c['set']=='fra' and not c['type'].startswith('Basic Land') for c in cards),'all_card_catalog_count':len(cards),'historical_LR_partial_coverage':True,'source_scope_is_retrospective':True}}
    (output/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(score_df[score_df.pool=='common cards across all five creators'][['name','n','pearson','spearman','mae_pp','rmse_pp','bias_pp']].to_string(index=False),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fit','evaluate','all']);ap.add_argument('--truth-dir',type=Path,required=True);ap.add_argument('--fra-truth',type=Path,default=PREVIOUS/'inputs/local/17lands_fra_all_time.json');ap.add_argument('--output',type=Path,default=ROOT/'results');args=ap.parse_args()
    if args.stage in ['fit','all']:fit_stage(args.truth_dir,args.output)
    if args.stage in ['evaluate','all']:evaluate_stage(args.truth_dir,args.output,args.fra_truth)

if __name__=='__main__':main()
