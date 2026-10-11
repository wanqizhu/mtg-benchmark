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
from scipy.interpolate import BSpline
from scipy.stats import pearsonr, spearmanr

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'fra-week-one-2026-10-07'
CREATORS={'casualspike':'Casual Spike','alexnikolic':'Alex Nikolic','limitedresources':'LSV + Marshall / Limited Resources','draftsim':'Draftsim bot'}
LETTER={'F':0.,'D-':1.,'D':2.,'D+':3.,'C-':4.,'C':5.,'C+':6.,'C++':6.5,'B-':7.,'B':8.,'B+':9.,'A-':10.,'A':11.,'A+':12.}
CONTEXTS=['build_around','synergy','sideboard','scenario','alternate','context_flag']
CONTEXT_FLAG_KINDS={'build_around','synergy','scenario'}
GRADE_LAMBDAS=[0.,2.,10.,50.]
CONTEXT_LAMBDAS=[2.,10.,50.]
MINIMUM_BUCKET_SAMPLES=10
BOT_MAX_COEFFICIENTS=9


def norm(s):
    return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',s.split(' // ')[0]).lower())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def value(label):return LETTER[label] if label in LETTER else float(label)


def nearest_topup(label,n,base,available,minimum=MINIMUM_BUCKET_SAMPLES):
    """Fixed minimum-support coefficient rule, using unsmoothed donor means.

    Equal-distance classes share weight equally, capped at their actual support.
    Expand to the next distance only after exhausting closer classes. SB uses
    the bottom of the ordinary grade scale as its explicit unordered reference.
    Borrowed weights are fractional pseudo-samples, never additional real cards.
    """
    required=max(0.,float(minimum-n));donors=[]
    if not required:return float(base),donors
    ordinary=[r for r in available if r['grade']!='SB' and r['grade']!=label and r['n']>0]
    if not ordinary:return float(base) if base is not None else None,donors
    target=min(value(r['grade']) for r in ordinary) if label=='SB' else value(label)
    shells={}
    for r in ordinary:shells.setdefault(round(abs(value(r['grade'])-target),10),[]).append(r)
    remaining=required
    for distance in sorted(shells):
        active=[dict(r,capacity=float(r['n'])) for r in shells[distance]]
        while active and remaining>1e-10:
            amount=min(remaining/len(active),min(r['capacity'] for r in active))
            for r in active:
                hit=next((d for d in donors if d['grade']==r['grade']),None)
                if hit is None:
                    hit={'grade':r['grade'],'distance':distance,'available_observations':r['n'],'borrowed_weight':0.,'base_coefficient':float(r['coefficient'])};donors.append(hit)
                hit['borrowed_weight']+=amount;r['capacity']-=amount
            remaining-=amount*len(active);active=[r for r in active if r['capacity']>1e-10]
        if remaining<=1e-10:break
    borrowed=sum(r['borrowed_weight'] for r in donors)
    denominator=float(n)+borrowed
    assert denominator>0
    # n=0 classes have no base evidence; their coefficient is wholly borrowed.
    observed=0. if n==0 else n*float(base)
    return (observed+sum(r['borrowed_weight']*r['base_coefficient'] for r in donors))/denominator,donors


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
        # The applicability grade is the primary grade when a conditional-only
        # review also gives a narrower scenario grade (e.g. heavy-green/monogreen).
        conditions=sorted(r.get('contexts',[]),key=lambda c:c.get('relation')!='applicability')
        for c in conditions:
            if atoms(c.get('native')):
                a=atoms(c['native']);label=c['native'];basis='explicit conditional grade; no ordinary grade provided';break
    if r.get('grade_weights'):
        a=r['grade_weights'];assert abs(sum(a.values())-1)<1e-8
    if str(r.get('native','')).lower().startswith(('build-around','sideboard')):basis='explicit conditional grade; no ordinary grade provided'
    sideboard='SB' in a or 'sideboard' in flags or str(r.get('native','')).lower().startswith('sideboard')
    if sideboard:
        label='1' if r.get('creator')=='draftsim' else 'F';a={label:1.};flags.discard('sideboard');basis='SB / sideboard grouped with lowest model rating'
    # Source assembly sometimes copied a lone contextual grade into both fields.
    # That is not evidence of two published grades. Prefix labels identify the
    # historical conditional-only records despite their duplicated ordinary field.
    explicit_only=str(r.get('native','')).lower().startswith('build-around')
    ordinary=atoms(r.get('ordinary_native'))
    conditional=[atoms(c.get('native')) for c in r.get('contexts',[])
                 if c['kind'] in CONTEXT_FLAG_KINDS and atoms(c.get('native'))]
    relevant=bool(flags & CONTEXT_FLAG_KINDS)
    if sideboard:structure='sideboard'
    elif relevant and (explicit_only or (not ordinary and conditional)):structure='conditional_only'
    elif relevant and ordinary and any(c!=ordinary for c in conditional):structure='base_and_conditional'
    elif relevant:structure='single_grade_with_context'
    else:structure='ordinary'
    applied=['context_flag'] if a and structure in ['conditional_only','single_grade_with_context'] else []
    if not a:basis='No verified letter or numeric grade'
    r.update(grade_used=label,grade_atoms=a,grade_basis=basis,
             raw_buckets=sorted(flags),buckets=applied,context_gaps={},rating_structure=structure)
    return r


def training_inputs(truth_dir,period):
    doc=json.loads((ROOT/'inputs/historical_ratings.json').read_text());rows=[];join_exclusions=[];files={}
    for s in sorted({r['set'] for r in doc['rows']}):
        path=truth_dir/(f'{s}_FIRST_WEEK.json' if period=='FIRST_WEEK' else f'{s}.json')
        data=json.loads(path.read_text())['data'];lookup={norm(x['name']):x for x in data}
        assert len(data)==len(lookup)
        files[s]={'file':str(path.resolve()),'sha256':sha(path),'period':period,'url':f'https://www.17lands.com/api/card_data?expansion={s}&event_type=PremierDraft&time_period={period}'}
        for raw in [r for r in doc['rows'] if r['set']==s]:
            if raw['creator'] not in CREATORS:continue
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
        assert not use_gaps,'The shared context-flag model has no conditional-gap terms'
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
        self.gap_contexts=[]
        assert self.contexts in [[],['context_flag']],self.contexts
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
        self.base_coef=self.coef.copy();self.base_training_rmse=float(np.sqrt(np.mean((x@self.coef-y)**2)))
        self.apply_ten_sample_rule(rows)
        self.train_n=len(rows);self.train_sets=sorted({r['set'] for r in rows});self.training_rmse=float(np.sqrt(np.mean((np.array(self.predict(rows)[0])-y)**2)))
        return self

    def apply_ten_sample_rule(self,rows):
        observed=sorted({g for r in rows for g in r['grade_atoms']},key=lambda g:(g=='SB',0 if g=='SB' else value(g)))
        self.bucket_support={g:sum(g in r['grade_atoms'] for r in rows) for g in observed}
        base={g:float(self.base_coef[len(self.levels)] if g=='SB' else self.base_coef[self.levels.index(value(g))]) for g in observed}
        available=[{'grade':g,'n':self.bucket_support[g],'coefficient':base[g]} for g in observed]
        numeric=all(g=='SB' or g not in LETTER for g in observed)
        grades=sorted(set([format(i/10,'g') for i in range(51)]+observed),key=lambda g:value(g)) if numeric else list(LETTER)
        if self.has_sb:grades.append('SB')
        self.grade_lookup={};self.topup_audit={}
        for g in grades:
            n=self.bucket_support.get(g,0);coefficient,donors=nearest_topup(g,n,base.get(g),available)
            self.grade_lookup[g]=coefficient
            self.topup_audit[g]={'observed_n':n,'borrowed_weight':sum(r['borrowed_weight'] for r in donors),'base_coefficient':base.get(g),'donors':donors,'reference':'nearest ordinary native grades' if g!='SB' else 'lowest supported ordinary native grade; SB has no intrinsic order'}
            if g in base:
                index=len(self.levels) if g=='SB' else self.levels.index(value(g));self.coef[index]=coefficient
        nbase=len(self.levels)+int(self.has_sb);self.context_topups={};self.gap_topups={}
        for i,k in enumerate(self.contexts):
            n=sum(k in r['buckets'] for r in rows);factor=min(1.,n/MINIMUM_BUCKET_SAMPLES);index=nbase+i
            self.coef[index]=self.base_coef[index]*factor
            self.context_topups[k]={'observed_n':n,'borrowed_weight':max(0,MINIMUM_BUCKET_SAMPLES-n),'base_coefficient':float(self.base_coef[index]),'donor_coefficient':0.,'reference':'ordinary cards of the same grade; zero context offset'}
        for i,k in enumerate(self.gap_contexts):
            n=sum(abs(r.get('context_gaps',{}).get(k,0))>1e-9 for r in rows);factor=min(1.,n/MINIMUM_BUCKET_SAMPLES);index=nbase+len(self.contexts)+i
            self.coef[index]=self.base_coef[index]*factor
            self.gap_topups[k]={'observed_n':n,'borrowed_weight':max(0,MINIMUM_BUCKET_SAMPLES-n),'base_coefficient':float(self.base_coef[index]),'donor_coefficient':0.,'reference':'ordinary cards; zero conditional-grade-gap effect'}

    def predict(self,rows):
        result=[];support=[]
        for r in rows:
            if not r['grade_atoms']:result.append(None);support.append(['No raw native grade']);continue
            if hasattr(self,'grade_lookup'):
                if not set(r['grade_atoms'])<=set(self.grade_lookup):
                    x,notes=self.basis(r);result.append(None if x is None else float(np.clip(x@self.coef,0,100)));support.append(notes);continue
                grade=sum(self.grade_lookup[g]*w for g,w in r['grade_atoms'].items());notes=[]
                for g in r['grade_atoms']:
                    audit=self.topup_audit[g]
                    if audit['observed_n']<MINIMUM_BUCKET_SAMPLES:notes.append(f"{g}: {audit['observed_n']} real historical observations + {audit['borrowed_weight']:g} borrowed sample weight")
                nbase=len(self.levels)+int(self.has_sb);adjustment=0.
                for k in r['buckets']:
                    if k in self.contexts:adjustment+=self.coef[nbase+self.contexts.index(k)]
                    else:notes.append(k+' has no historical fitted coefficient; zero offset used')
                for i,k in enumerate(self.gap_contexts):adjustment+=self.coef[nbase+len(self.contexts)+i]*r.get('context_gaps',{}).get(k,0)
                result.append(float(np.clip(grade+adjustment,0,100)));support.append(notes)
            else:
                x,notes=self.basis(r);result.append(None if x is None else float(np.clip(x@self.coef,0,100)));support.append(notes)
        return result,support

    def specification(self,rows):
        nbase=len(self.levels)+int(self.has_sb);counts=[]
        labels=sorted(self.grade_lookup,key=lambda x:(x=='SB',0 if x=='SB' else value(x)))
        for g in labels:
            members=[r for r in rows if g in r['grade_atoms']]
            pred=self.predict([{'grade_atoms':{g:1.},'buckets':[]}])[0][0]
            counts.append({'grade':g,'native_value':None if g=='SB' else value(g),'card_set_observations':len(members),'distinct_card_names':len({norm(r['name']) for r in members}),'effective_weight':sum(r['grade_atoms'][g] for r in members),'mapped_wr':pred,'observed_mean_wr':float(np.mean([r['actual'] for r in members])) if members else None,**self.topup_audit[g]})
        contexts=[{'bucket':k,'card_set_observations':sum(k in r['buckets'] for r in rows),'distinct_card_names':len({norm(r['name']) for r in rows if k in r['buckets']}),'offset_pp':float(self.coef[nbase+i]),**self.context_topups[k]} for i,k in enumerate(self.contexts)]
        gap_mapping=[{'bucket':k,'nonzero_gap_card_set_observations':sum(abs(r.get('context_gaps',{}).get(k,0))>1e-9 for r in rows),'pp_per_native_grade_step':float(self.coef[nbase+len(self.contexts)+i]),**self.gap_topups[k]} for i,k in enumerate(self.gap_contexts)]
        return {'model_type':'native_bucket','context_policy':'single-context-flag-v1','rating_structure_counts':dict(Counter(r.get('rating_structure','ordinary') for r in rows)),'sideboard_policy':'SB and sideboard-marked inputs grouped with F; no separate SB/context-sideboard coefficient','sparse_topup_policy':'nearest-class-ten-sample-v1','minimum_bucket_samples':MINIMUM_BUCKET_SAMPLES,'use_gaps':False,'gap_contexts':[],'context_gap_mapping':[],'family':('monotone' if self.monotone else 'free')+' historical native curve + one shared context flag + fixed ten-sample top-up','monotone':self.monotone,'final_mapping_monotone':bool(np.all(np.diff([r['mapped_wr'] for r in counts if r['grade']!='SB'])>=-1e-8)),'grade_lambda':self.grade_lambda,'context_lambda':self.context_lambda,'levels':self.levels,'has_sb':self.has_sb,'contexts':self.contexts,'coefficients':self.coef.tolist(),'base_coefficients':self.base_coef.tolist(),'train_n':self.train_n,'train_sets':self.train_sets,'base_training_rmse_pp':self.base_training_rmse,'training_rmse_pp':self.training_rmse,'grade_mapping':counts,'context_mapping':contexts}

    @classmethod
    def restore(cls,spec):
        m=cls(spec['monotone'],spec['grade_lambda'],spec['context_lambda'],spec.get('use_gaps',False));m.gap_contexts=spec.get('gap_contexts',[]);m.levels=spec['levels'];m.has_sb=spec['has_sb'];m.contexts=spec['contexts'];m.coef=np.array(spec['coefficients'])
        if spec.get('sparse_topup_policy'):
            m.grade_lookup={r['grade']:r['mapped_wr'] for r in spec['grade_mapping']};m.topup_audit={r['grade']:r for r in spec['grade_mapping']}
        return m


class BotRegression:
    """Monotone linear or cubic B-spline regression with at most nine parameters."""
    def __init__(self,n_coefficients,curvature_lambda=0.):
        assert 2<=n_coefficients<=BOT_MAX_COEFFICIENTS
        assert n_coefficients==2 or n_coefficients>=4
        self.n_coefficients=n_coefficients;self.curvature_lambda=curvature_lambda

    @staticmethod
    def native(row):return sum(value(g)*w for g,w in row['grade_atoms'].items())

    def design(self,x):
        x=np.clip(np.asarray(x,dtype=float),self.domain[0],self.domain[1])
        if self.n_coefficients==2:
            fraction=(x-self.domain[0])/(self.domain[1]-self.domain[0]);return np.c_[1-fraction,fraction]
        return BSpline.design_matrix(x,self.knots,3).toarray()

    def fit(self,rows):
        xx=np.array([self.native(r) for r in rows]);self.domain=[float(xx.min()),float(xx.max())];assert self.domain[0]<self.domain[1]
        interior=np.linspace(*self.domain,self.n_coefficients-2)[1:-1] if self.n_coefficients>2 else []
        self.knots=[self.domain[0]]*4+list(interior)+[self.domain[1]]*4 if self.n_coefficients>2 else []
        x=self.design(xx);y=np.array([r['actual'] for r in rows]);p=self.n_coefficients
        difference=np.diff(np.eye(p),n=2,axis=0) if p>2 else np.zeros((0,p))
        design=np.vstack([x,np.sqrt(self.curvature_lambda)*difference]);target=np.r_[y,np.zeros(len(difference))]
        transform=np.tril(np.ones((p,p)));low=np.zeros(p);high=np.full(p,np.inf);high[0]=100
        solution=lsq_linear(design@transform,target,bounds=(low,high),tol=1e-11,max_iter=300);assert solution.success,solution.message
        self.coef=transform@solution.x
        self.df_unconstrained=float(np.trace(np.linalg.pinv(design.T@design)@(x.T@x)))
        assert self.df_unconstrained<=p+1e-8 and p<10
        self.train_n=len(rows);self.train_sets=sorted({r['set'] for r in rows});self.training_rmse=float(np.sqrt(np.mean((x@self.coef-y)**2)))
        self.basis_support=x.sum(axis=0).tolist()
        return self

    def predict(self,rows):
        predictions=[];notes=[]
        for r in rows:
            if not r['grade_atoms']:predictions.append(None);notes.append(['No raw native grade']);continue
            v=self.native(r);predictions.append(float(np.clip((self.design([v])@self.coef)[0],0,100)))
            support=[f'{self.n_coefficients}-coefficient historical regression; no independent numeric bucket parameter']
            if not self.domain[0]<=v<=self.domain[1]:support.append('Native rating outside historical support; constant endpoint prediction used')
            notes.append(support)
        return predictions,notes

    def specification(self,rows):
        mappings=[]
        for i in range(51):
            g=format(i/10,'g');members=[r for r in rows if abs(self.native(r)-i/10)<1e-9];prediction=self.predict([{'grade_atoms':{g:1.}}])[0][0]
            mappings.append({'grade':g,'native_value':i/10,'card_set_observations':len(members),'distinct_card_names':len({norm(r['name']) for r in members}),'effective_weight':len(members),'mapped_wr':prediction,'observed_mean_wr':float(np.mean([r['actual'] for r in members])) if members else None,'entry_type':'regression curve evaluation; not an independently fitted coefficient'})
        parameters=[{'term':('low_endpoint' if j==0 else 'high_endpoint') if self.n_coefficients==2 else f'cubic_basis_{j+1}','coefficient':float(c),'weighted_historical_support':self.basis_support[j]} for j,c in enumerate(self.coef)]
        return {'model_type':'bot_regression','family':'monotone linear regression' if self.n_coefficients==2 else 'monotone cubic B-spline regression','nominal_degrees_of_freedom':self.n_coefficients,'unconstrained_smoother_degrees_of_freedom':self.df_unconstrained,'monotone':True,'final_mapping_monotone':True,'n_coefficients':self.n_coefficients,'curvature_lambda':self.curvature_lambda,'degree':1 if self.n_coefficients==2 else 3,'domain':self.domain,'knots':self.knots,'coefficients':self.coef.tolist(),'regression_coefficients':parameters,'boundary_policy':'Constant endpoint extrapolation outside historical native-rating range','use_gaps':False,'gap_contexts':[],'context_gap_mapping':[],'contexts':[],'context_mapping':[],'grade_lambda':0.,'context_lambda':0.,'train_n':self.train_n,'train_sets':self.train_sets,'training_rmse_pp':self.training_rmse,'grade_mapping':mappings}

    @classmethod
    def restore(cls,spec):
        model=cls(spec['n_coefficients'],spec['curvature_lambda']);model.domain=spec['domain'];model.knots=spec['knots'];model.coef=np.array(spec['coefficients']);return model


def restore_model(spec):return BotRegression.restore(spec) if spec.get('model_type')=='bot_regression' else BucketModel.restore(spec)


def refit_model(rows,spec):
    model=BotRegression(spec['n_coefficients'],spec['curvature_lambda']) if spec.get('model_type')=='bot_regression' else BucketModel(spec['monotone'],spec['grade_lambda'],spec['context_lambda'],spec['use_gaps'])
    return model.fit(rows)


def choose_bot_model(rows):
    candidates=[]
    for p,penalty in [(2,0.)]+[(p,penalty) for p in range(4,BOT_MAX_COEFFICIENTS+1) for penalty in [0.,2.,10.,50.]]:
        actual=[];predictions=[];fold_rows=[]
        for s,train,test in folds(rows):
            model=BotRegression(p,penalty).fit(train);pred,_=model.predict(test);actual.extend(r['actual'] for r in test);predictions.extend(pred)
            fold_rows.append({'set':s,'train_n':len(train),'test_n':len(test),**metrics([r['actual'] for r in test],pred)})
        candidates.append({'model_type':'bot_regression','n_coefficients':p,'curvature_lambda':penalty,'monotone':True,'grade_lambda':0.,'context_lambda':0.,'use_gaps':False,**metrics(actual,predictions),'folds':fold_rows})
    winner=min(candidates,key=lambda c:(round(c['rmse_pp'],10),c['n_coefficients'],-c['curvature_lambda']))
    return BotRegression(winner['n_coefficients'],winner['curvature_lambda']).fit(rows),winner,candidates


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
    if rows[0]['creator']=='draftsim':return choose_bot_model(rows)
    candidates=[]
    for use_gaps in [False]:
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
    all_rows=[r for r in doc['rows'] if r['creator'] in CREATORS];models={};validation={};audit=[];bucket_counts=[];cv_ledger=[]
    for creator,name in CREATORS.items():
        train=[r for r in rows if r['creator']==creator];model,winner,candidates=choose_model(train);spec=model.specification(train)
        spec.update(name=name,source_rows=sum(r['creator']==creator for r in all_rows),historical_sets=len(model.train_sets),chosen_by='Minimum RMSE in historical leave-one-set-out validation; shared card names purged from each training fold',historical_validation=winner)
        # Mapping sensitivity holds chosen hyperparameters fixed and omits one whole set.
        sensitivities=[]
        for s,tr,te in folds(train):
            m=refit_model(tr,spec)
            sensitivities.append({'omitted_set':s,'specification':m.specification(tr)})
            pr,notes=m.predict(te)
            for r,pred,note in zip(te,pr,notes):cv_ledger.append({'creator':creator,'held_out_set':s,'card':r['name'],'native':r['native'],'actual':r['actual'],'prediction':pred,'support_notes':'; '.join(note),'training_sets':'; '.join(m.train_sets)})
        spec['leave_one_set_out_mapping_sensitivity']=sensitivities
        for mapping in spec['grade_mapping']:
            trial={'grade_atoms':{mapping['grade']:1.},'buckets':[]};values=[restore_model(s['specification']).predict([trial])[0][0] for s in sensitivities];values=[x for x in values if x is not None]
            mapping.update(loso_min=min(values) if values else None,loso_max=max(values) if values else None)
        models[creator]=spec;validation[creator]=candidates
        for raw_grade,n in Counter(r['native'] for r in all_rows if r['creator']==creator).items():
            members=[r for r in train if r['native']==raw_grade];bucket_counts.append({'creator':creator,'kind':'raw published label','bucket':raw_grade,'published_card_set_observations':n,'training_card_set_observations':len(members),'training_distinct_card_names':len({norm(r['name']) for r in members})})
        for ctx in CONTEXTS:
            raw_prepared=[prepare(r) for r in all_rows if r['creator']==creator]
            bucket_field='buckets' if ctx=='context_flag' else 'raw_buckets'
            members=[r for r in train if ctx in r[bucket_field]]
            bucket_counts.append({'creator':creator,'kind':'applied context flag' if ctx=='context_flag' else 'published context','bucket':ctx,'published_card_set_observations':sum(ctx in r[bucket_field] for r in raw_prepared),'training_card_set_observations':len(members),'training_distinct_card_names':len({norm(r['name']) for r in members})})
        print('FIT',creator,len(train),spec['family'],'grade penalty',winner['grade_lambda'],'context penalty',winner['context_lambda'],'conditional gaps',winner['use_gaps'],'bot parameters',winner.get('n_coefficients'),'validation RMSE',round(winner['rmse_pp'],3),flush=True)
    package={'created_utc':datetime.now(timezone.utc).isoformat(),'fit_scope':'Historical nonbasic main-set PremierDraft FIRST_WEEK GIH, all colors and users, reported rates and n >= 500; FRA outcomes never loaded by fit stage','equal_weight':'One card-set observation has one unit of loss; no weighting by GIH count or creator semantic grade normalization','raw_grade_policy':'Each native letter/decimal bucket is fitted separately. Split host grades average fitted bucket values, with explicit nested host weights where recorded. SB is an unordered native bucket. Conditional-only grades use their explicit letter and retain the context flag. Unobserved grades interpolate on the native order or use the nearest supported endpoint. Unidentified SB stays unmapped.','penalty_policy':'Adjacent bucket differences divided by native-grade spacing receive grade_lambda penalty; unordered SB borrows strength from the lowest supported ordinary grade with the same penalty. Every observed context coefficient receives context_lambda penalty. An optional nonnegative conditional-grade-gap effect (conditional raw grade minus ordinary raw grade, on the native order) is compared with the flag-only model in historical CV. These gap coefficients share context_lambda shrinkage. No minimum-count bucket dropping or arbitrary 10-card claim.','sources_file_sha256':sha(ROOT/'inputs/historical_ratings.json'),'truth_manifests':files,'models':models,'join_exclusions':join_exclusions}
    package['active_creators']=CREATORS
    package['raw_grade_policy']='Original labels are retained. Explicit base/conditional pairs use only the base grade and no context offset. Conditional-only grades and single grades with build-around, synergy or scenario labels use the supplied grade plus one shared context-flag coefficient per creator. Multiple flags activate the coefficient only once. Copied same-grade fields are single ratings, not verified pairs. Tentative alternative ratings do not activate the context flag. SB and sideboard-marked ratings use F (or native 1 for the bot) with no context offset. Split grades keep their published weights. Missing qualitative grades stay unmapped.'
    package['penalty_policy']='Letter models validate grade and shared context penalties on historical sets; conditional-gap terms are disabled. The user fixed ten-sample top-up is applied inside every fit and validation fold. For n<10 real grade memberships, fill 10-n fractional sample weight from nearest observed native classes using unsmoothed fitted coefficients, with equal-distance allocation balanced and capped at actual donor counts. A sparse context flag shrinks toward zero. All adjustments are simultaneous. Draftsim uses a monotone linear/cubic B-spline with 2 or 4-9 total coefficients selected by historical purged leave-one-set-out RMSE.'
    package['empty_class_policy']='Letter classes with n=0 borrow all ten sample weights. These remain labelled as zero real observations. The bot curve clamps to its historical endpoint outside the observed native-rating range.'
    modelpath=output/'historical_models.json';modelpath.write_text(json.dumps(package,ensure_ascii=False,indent=2,allow_nan=False))
    (output/'fit_lock.json').write_text(json.dumps({'models_sha256':sha(modelpath),'locked_before_FRA_outcome_read':True,'utc':datetime.now(timezone.utc).isoformat(),'historical_training_sets':sorted(files),'historical_ratings_sha256':sha(ROOT/'inputs/historical_ratings.json')},indent=2))
    (output/'historical_cv_candidates.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2,allow_nan=False))
    pd.DataFrame(cv_ledger).to_csv(output/'historical_cv_predictions.csv',index=False)
    pd.DataFrame(bucket_counts).to_csv(output/'historical_bucket_counts.csv',index=False)
    pd.DataFrame([{**{k:r.get(k) for k in ['creator','set','name','native','grade_used','actual','gih_n','grade_basis','rating_structure','source_url']},'contexts':'; '.join(r['buckets']),'published_contexts':'; '.join(r['raw_buckets'])} for r in rows]).to_csv(output/'historical_training_observations.csv',index=False)
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
        model=restore_model(pack['models'][creator]);raw=[prepare({**r,'creator':creator,'set':'FRA'}) for r in sources[creator]['rows']]
        byname={norm(r['name']):r for r in raw};assert len(byname)==len(raw);assert set(byname)<=set(cardmap)
        predictions,notes=model.predict(raw);preds={norm(r['name']):(r,p,n) for r,p,n in zip(raw,predictions,notes)}
        actual=[];predicted=[];scored=[]
        for c in cards:
            k=norm(c['name']);t=truth.get(k);item=preds.get(k)
            wr=None if not t or t['ever_drawn_win_rate'] is None else 100*t['ever_drawn_win_rate'];n=0 if not t else t['ever_drawn_game_count']
            if item:r,pred,support=item
            else:r=None;pred=None;support=['Creator supplied no grade']
            primary=c['set']=='fra' and not c['type'].startswith('Basic Land') and wr is not None and n>=500
            rec={'creator':creator,'card':c['name'],'set':c['set'],'rarity':c['rarity'],'color':''.join(c['colors']) or 'C','native_grade':None if r is None else r['native'],'grade_used':None if r is None else r['grade_used'],'contexts':[] if r is None else r['raw_buckets'],'applied_contexts':[] if r is None else r['buckets'],'rating_structure':None if r is None else r['rating_structure'],'grade_atoms':{} if r is None else r['grade_atoms'],'context_gaps':{} if r is None else r['context_gaps'],'context_details':[] if r is None else r['contexts'],'prediction':pred,'actual':wr,'gih_n':n,'error_pp':None if pred is None or wr is None else pred-wr,'primary':primary,'support_notes':support,'source_url':None if r is None else r.get('source_url'),'oracle':c['oracle'],'cost':c['cost'],'type':c['type'],'scryfall_uri':c['scryfall_uri']}
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
        mt=refit_model(past,chosen)
        mpr,_=mt.predict(raw);actual_raw=[None if norm(r['name']) not in truth or truth[norm(r['name'])]['ever_drawn_win_rate'] is None else 100*truth[norm(r['name'])]['ever_drawn_win_rate'] for r in raw]
        valid=[i for i,r in enumerate(raw) if cardmap[norm(r['name'])]['set']=='fra' and actual_raw[i] is not None and not cardmap[norm(r['name'])]['type'].startswith('Basic Land')]
        sensitivities.append({'creator':creator,'variant':'Mature ALL_TIME targets; primary hyperparameters locked','training_n':len(past),**metrics([actual_raw[i] for i in valid],[mpr[i] for i in valid])})
        # Exclude historical reprints of FRA cards, while keeping original hyperparameters.
        fra_names=set(cardmap);past=[r for r in history if r['creator']==creator and norm(r['name']) not in fra_names]
        rp=refit_model(past,chosen);rpr,_=rp.predict(raw)
        sensitivities.append({'creator':creator,'variant':'Remove every historical printing of FRA card names','training_n':len(past),**metrics([actual_raw[i] for i in valid],[rpr[i] for i in valid])})
        # Grade-only conversion demonstrates what the added context terms contribute.
        past=[{**r,'buckets':[],'context_gaps':{}} for r in history if r['creator']==creator];gm=refit_model(past,chosen);gpr,_=gm.predict([{**r,'buckets':[],'context_gaps':{}} for r in raw])
        sensitivities.append({'creator':creator,'variant':'Grade buckets only; same smoothing','training_n':len(past),**metrics([actual_raw[i] for i in valid],[gpr[i] for i in valid])})
        if creator=='limitedresources':
            # Episodes issued after launch are allowed by user but separated for transparency.
            rare_ids=['b-mJNcpSo-w','y11-rxC1rQk','mxysznkjMB4']
            past=[r for r in history if r['creator']==creator and not any(v in r.get('source_url','') for v in rare_ids)]
            early=refit_model(past,chosen);epr,_=early.predict(raw)
            sensitivities.append({'creator':creator,'variant':'Exclude after-launch rare episodes (SOS launch-day review retained)','training_n':len(past),**metrics([actual_raw[i] for i in valid],[epr[i] for i in valid])})
        if creator=='draftsim':
            for s in ['HOB','SOS']:
                past=[r for r in history if r['creator']==creator and r['set']==s];single=refit_model(past,chosen);spr,_=single.predict(raw)
                sensitivities.append({'creator':creator,'variant':f'{s} only; other edition excluded','training_n':len(past),**metrics([actual_raw[i] for i in valid],[spr[i] for i in valid])})
    shared=sorted({r['card'] for r in long if r['primary']} & set.intersection(*[{r['card'] for r in long if r['creator']==c and r['primary'] and r['prediction'] is not None} for c in CREATORS]))
    for creator,name in CREATORS.items():
        subset=[r for r in long if r['creator']==creator and r['card'] in shared]
        score_rows.append({'creator':creator,'name':name,'pool':'common cards across all active creators',**metrics([r['actual'] for r in subset],[r['prediction'] for r in subset])})
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
    source_audit=json.loads((ROOT/'inputs/source_audit.json').read_text());source_audit['sources']=[r for r in source_audit['sources'] if not r['admitted'] or r['creator'] in CREATORS]
    result={'models':pack,'fit_lock':lock,'fra_truth_sha256':sha(fra_truth),'fra_truth_fetched_at':manifest.get('retrieved_at'),'scores':score_rows,'sensitivity_scores':sensitivities,'group_errors':diagnostics,'records':long,'largest_mistakes':misses,'common_card_count':len(shared),'source_audit':source_audit,'validation':{'no_FRA_outcomes_in_fit':True,'models_file_still_matches_lock':sha(output/'historical_models.json')==lock['models_sha256'],'primary_outcome_snapshot':'Original October 7 ALL_TIME FRA response, not a newly fetched snapshot; early-format horizon is approximately first week','historical_target_period':'FIRST_WEEK','main_nonbasic_catalog_count':sum(c['set']=='fra' and not c['type'].startswith('Basic Land') for c in cards),'all_card_catalog_count':len(cards),'historical_LR_partial_coverage':True,'source_scope_is_retrospective':True}}
    (output/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(score_df[score_df.pool=='common cards across all active creators'][['name','n','pearson','spearman','mae_pp','rmse_pp','bias_pp']].to_string(index=False),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fit','evaluate','all']);ap.add_argument('--truth-dir',type=Path,required=True);ap.add_argument('--fra-truth',type=Path,default=PREVIOUS/'inputs/local/17lands_fra_all_time.json');ap.add_argument('--output',type=Path,default=ROOT/'results');args=ap.parse_args()
    if args.stage in ['fit','all']:fit_stage(args.truth_dir,args.output)
    if args.stage in ['evaluate','all']:evaluate_stage(args.truth_dir,args.output,args.fra_truth)

if __name__=='__main__':main()
