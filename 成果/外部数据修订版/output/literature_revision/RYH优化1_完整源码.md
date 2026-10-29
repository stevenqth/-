# RYH优化1 完整源码

按模块汇总当前代码，供浏览。执行时使用独立.py文件。正文各级标题按用户要求左对齐。

## 模块索引

1. literature_revision.py

2. literature_models.py

3. literature_nsga2.py

4. literature_generation.py

5. literature_inference.py

6. validate_literature_revision.py

7. build_literature_paper.py

8. format_ryh1_template.py

9. build_paper.py

10. common.py

11. external_analysis.py

12. package_delivery.py

13. package_literature_revision.py

14. package_ryh1_source.py

15. question1.py

16. question2.py

17. question3.py

18. validate_outputs.py

## literature_revision.py

```python

"""Run the literature revision without overwriting baseline experiments.

python code/literature_revision.py --stage all --subsamples 100
Requires numpy pandas scipy scikit-learn matplotlib networkx openpyxl joblib.
The saved cVAE is evaluated with a restricted NumPy inference implementation.
"""
from common import *
import argparse,time,joblib,hashlib
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from scipy.stats import wasserstein_distance
from literature_models import FoldFeatures,candidate,conditional_donors
from literature_inference import NumpyCVAE

REV=OUT/'literature_revision'; REV.mkdir(exist_ok=True)
RFIG=FIG/'literature_revision'; RFIG.mkdir(exist_ok=True)
CANDIDATES=['Mean','RankPLS3','RankPLS8','RankRidge','RandomForest','ExtraTrees']
NON_MOLECULAR=['kcalperc','esters_sum','aroma_hops_sum','acids_sum','sulfur_sum',
               'pH','color.EBC.','betaglucan.mg.L.','protein.g.L.','bitterness',
               'GFS','unfermentables','CO2.PSI.']


def save(obj,name):
    (REV/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist() if isinstance(x,np.ndarray) else str(x)),encoding='utf-8')


def evidence_audit():
    X,Y,s,ids=load()
    calibrated=pd.read_excel(DATA/'8.xlsx',sheet_name='Calibrated chemical dataset').set_index('beer').loc[ids.beer]
    cols=[c for c in X if c in calibrated]
    rows=[]
    for c in cols:
        observed=X[c].to_numpy();ref=calibrated[c].to_numpy(float)
        ok=np.isfinite(observed)&np.isfinite(ref)&(ref>0)
        raw=np.abs(observed[ok]-ref[ok]); lg=np.abs(observed[ok]-np.log10(ref[ok]))
        ln=np.abs(observed[ok]-np.log(ref[ok]))
        errors={'raw':raw,'log10':lg,'ln':ln}
        choice=min(errors,key=lambda k:np.median(errors[k]) if len(errors[k]) else np.inf)
        err=errors[choice]; exact=len(err)>=5 and np.all(err<=1e-6)
        rows.append({'compound':c,'n_compared':len(err),'suggested_transform':choice,
                     'n_discrepancy':int(np.sum(err>1e-6)),
                     'max_abs_error':float(err.max()) if len(err) else None,
                     'median_abs_error':float(np.median(err)) if len(err) else None,
                     'status':'verified_positive_rows' if exact else 'mixed_or_unresolved'})
    table=pd.DataFrame(rows);table.to_csv(REV/'scale_evidence.csv',index=False)
    floor=pd.DataFrame({'compound':list(X),'minimum':X.min().to_numpy(),
                        'minimum_count':(X==X.min()).sum().to_numpy(),
                        'explicit_missing':X.isna().sum().to_numpy()})
    floor['interpretation']='Repeated minima are recorded states, not demonstrated detection limits'
    floor.to_csv(REV/'floor_evidence.csv',index=False)
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in [DATA/'1.csv',DATA/'4.csv',DATA/'8.xlsx',OUT/'q3_cvae.pt']}
    save({'scale_status_counts':table.status.value_counts().to_dict(),
          'verified_transforms':table.query("status=='verified_positive_rows'").suggested_transform.value_counts().to_dict(),
          'complete_cases':int(X.notna().all(axis=1).sum()),'input_sha256':files,
          'baseline_commit':'05a3216b568d1c3665710b5801711898259d606a',
          'source_ids':['U1','U2']},'audit_summary.json')
    print('Scale audit',table.status.value_counts().to_dict(),flush=True)


def inner_choice(X,Y,seed):
    splits=list(KFold(3,shuffle=True,random_state=seed).split(X))
    rows=[]
    for name in CANDIDATES:
        losses=[]
        for tr,va in splits:
            m=candidate(name,seed).fit(X.iloc[tr],Y.iloc[tr])
            losses.append(mean_squared_error(Y.iloc[va],m.predict(X.iloc[va])))
        rows.append({'model':name,'inner_mse':float(np.mean(losses))})
    return min(rows,key=lambda r:r['inner_mse'])['model'],rows


def q1():
    X,Y,s,ids=load(); records=[];choices=[];predictions=[];per_target=[]
    for seed in [42,1024,2024]:
        oof=np.zeros(Y.shape); baseline=np.zeros(Y.shape)
        for k,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(X,s)):
            assert not set(tr)&set(va)
            name,inner=inner_choice(X.iloc[tr],Y.iloc[tr],seed+k)
            m=candidate(name,seed).fit(X.iloc[tr],Y.iloc[tr]);p=m.predict(X.iloc[va]);oof[va]=p
            b=candidate('RandomForest',seed).fit(X.iloc[tr],Y.iloc[tr]);bp=b.predict(X.iloc[va]);baseline[va]=bp
            records += [{'seed':seed,'fold':k,'model':'NestedSelected',**metrics(Y.iloc[va],p)},
                        {'seed':seed,'fold':k,'model':'FixedRandomForest',**metrics(Y.iloc[va],bp)}]
            choices.append({'seed':seed,'outer_fold':k,'selected':name,'outer_train_n':len(tr),'outer_test_n':len(va),'inner_scores':inner})
            print('Nested Q1',seed,k,name,flush=True)
        predictions.append(oof)
        for label,pred in [('NestedSelected',oof),('FixedRandomForest',baseline)]:
            records.append({'seed':seed,'fold':'pooled_oof','model':label,**metrics(Y,pred)})
        for j,col in enumerate(Y):per_target.append({'seed':seed,'target':col,**metrics(Y.iloc[:,j],oof[:,j])})
    pd.DataFrame(records).to_csv(REV/'q1_nested_metrics.csv',index=False)
    pd.DataFrame(per_target).to_csv(REV/'q1_nested_per_target.csv',index=False)
    save(choices,'q1_nested_choices.json')
    name,inner=inner_choice(X,Y,42);final=candidate(name,42).fit(X,Y)
    joblib.dump(final,REV/'q1_model.joblib')
    pred=np.mean(predictions,axis=0)
    pd.concat([ids,pd.DataFrame(pred,columns=Y.columns)],axis=1).to_csv(REV/'q1_oof_predictions.csv',index=False)
    # Descriptive fitted quantile grades, never interpreted as observed 0-6 labels.
    fitted=final.predict(X)
    grades=np.column_stack([np.clip(np.searchsorted(np.sort(Y.iloc[:,j]),fitted[:,j],side='right')*7//len(Y),0,6) for j in range(Y.shape[1])])
    pd.concat([ids,pd.DataFrame(grades,columns=Y.columns)],axis=1).to_csv(REV/'q1_relative_grades.csv',index=False)
    df=pd.DataFrame(records);pooled=df[df.fold=='pooled_oof']
    save({'outer':'3 repeats x 3 style-stratified folds','inner':'3 shuffled folds on each outer training set',
          'candidates':CANDIDATES,'final_selected':name,'full_fit_inner_scores':inner,
          'pooled_oof_scores':pooled.groupby('model')[['r2','mae','spearman']].mean().to_dict('index'),
          'std_across_repeats':pooled.groupby('model').r2.std().to_dict(),
          'selected_counts':pd.Series([v['selected'] for v in choices]).value_counts().to_dict(),
          'source_ids':['U1','U2','L1'],'grade_scope':'relative grades only; no absolute grade accuracy',
          'caveat':'outer estimates evaluate a predeclared selection procedure, not independent brewery/batch extrapolation'},'q1_summary.json')
    plt=plotting();fig,ax=plt.subplots(figsize=(7,4))
    g=pooled.groupby('model').r2;g.mean().plot.bar(yerr=g.std(),ax=ax,color=['#357a86','#b46e38'])
    ax.set_ylabel('完整外层折外预测平均 R²');ax.set_xlabel('');ax.tick_params(axis='x',rotation=0)
    fig.tight_layout();fig.savefig(RFIG/'q1_nested.png');plt.close(fig)


def q2(B):
    from question2 import graph,pick,jac
    X,Y,s,ids=load();cols=list(X);g,mods=graph(cols)
    excluded=np.array([c in NON_MOLECULAR for c in cols])
    assert (~excluded).sum()==218
    # No proxy-of-proxy target: every fit uses measured sensory Y.
    # Target-specific signed ridge associations, adjusted for style.
    prep=FoldFeatures(rank=True,indicators=False);a=prep.fit_transform(X)
    d=pd.get_dummies(s).to_numpy(float);rx=a-d@np.linalg.lstsq(d,a,rcond=None)[0]
    ry=Y.to_numpy()-d@np.linalg.lstsq(d,Y,rcond=None)[0]
    from scipy.stats import rankdata
    corr=rx.T@ry/len(a);score=rankdata(abs(corr),axis=0)/len(cols)
    score[excluded]=-np.inf
    tops=pick(score,cols,mods);freq=np.zeros_like(score);rng=np.random.default_rng(42)
    for b in range(B):
        # Genuine half-sampling without replacement, approximately style-stratified.
        idx=np.concatenate([rng.choice(np.flatnonzero(s.to_numpy()==st),max(1,int((s==st).sum()//2)),replace=False) for st in s.unique()])
        assert len(np.unique(idx))==len(idx)
        aa=FoldFeatures(rank=True,indicators=False).fit_transform(X.iloc[idx]);dd=pd.get_dummies(s.iloc[idx]).to_numpy(float)
        xx=aa-dd@np.linalg.lstsq(dd,aa,rcond=None)[0];yy=Y.iloc[idx].to_numpy()-dd@np.linalg.lstsq(dd,Y.iloc[idx],rcond=None)[0]
        sc=abs(xx.T@yy/len(idx))
        sc[excluded]=-np.inf
        for j,ls in enumerate(pick(sc,cols,mods)):freq[ls,j]+=1
    freq/=B
    npd=[]
    # Fold-specific explanatory top-3s avoid selecting on held-out responses.
    for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(X,s)):
        fit=candidate('ExtraTrees',42+fold).fit(X.iloc[tr],Y.iloc[tr]);base=fit.predict(X.iloc[va])
        train_imp=X.iloc[tr].fillna(X.iloc[tr].median()).to_numpy();test_imp=X.iloc[va].fillna(X.iloc[tr].median()).to_numpy()
        p=FoldFeatures(rank=True,indicators=False).fit_transform(X.iloc[tr]);dd=pd.get_dummies(s.iloc[tr]).to_numpy(float)
        tx=p-dd@np.linalg.lstsq(dd,p,rcond=None)[0];ty=Y.iloc[tr].to_numpy()-dd@np.linalg.lstsq(dd,Y.iloc[tr],rcond=None)[0]
        foldscore=abs(tx.T@ty/len(tr));foldscore[excluded]=-np.inf
        foldtop=pick(foldscore,cols,mods)
        jobs={i for ls in foldtop for i in ls}
        for i in sorted(jobs):
            for repeat in range(3):
                donors,ncond=conditional_donors(train_imp,test_imp,i,seed=42+100*fold+repeat)
                conditional=X.iloc[va].copy();conditional.iloc[:,i]=X.iloc[va].iloc[donors,i].to_numpy()
                marginal=X.iloc[va].copy();marginal.iloc[:,i]=np.random.default_rng(142+100*fold+repeat).permutation(marginal.iloc[:,i].to_numpy())
                pc=fit.predict(conditional);pm=fit.predict(marginal)
                for j,ls in enumerate(foldtop):
                    if i not in ls:continue
                    baseline=np.mean((Y.iloc[va,j].to_numpy()-base[:,j])**2)
                    npd.append({'fold':fold,'repeat':repeat,'target':Y.columns[j],'compound':cols[i],'conditioning_features':ncond,
                                'conditional_delta_mse':np.mean((Y.iloc[va,j].to_numpy()-pc[:,j])**2)-baseline,
                                'marginal_delta_mse':np.mean((Y.iloc[va,j].to_numpy()-pm[:,j])**2)-baseline})
        print('Conditional attribution fold',fold,flush=True)
    pd.DataFrame(npd).to_csv(REV/'q2_conditional_importance.csv',index=False)
    rows=[]
    for j,t in enumerate(Y):
        for rank,i in enumerate(tops[j],1):
            p=freq[i,j];se=np.sqrt(p*(1-p)/B)
            rows.append({'target':t,'rank':rank,'compound':cols[i],'module':mods[cols[i]],'score':score[i,j],
                         'half_sample_frequency':p,'mc_se':se,'stable_at_0_8':p>=.8,'sign':int(np.sign(corr[i,j]))})
    top=pd.DataFrame(rows);top.to_csv(REV/'q2_top3.csv',index=False)
    old=pd.read_csv(OUT/'q2_top3.csv')
    overlap=[]
    for target in Y:
        aa=set(old.loc[old.target==target,'compound']);bb=set(top.loc[top.target==target,'compound'])
        overlap.append({'target':target,'jaccard_vs_original_three_method':len(aa&bb)/len(aa|bb)})
    pd.DataFrame(overlap).to_csv(REV/'q2_selector_comparison.csv',index=False)
    pd.DataFrame(freq,index=cols,columns=Y.columns).to_csv(REV/'q2_all_half_frequencies.csv')
    save({'subsamples':B,'sampling':'style-stratified approximate half-sampling without replacement',
          'fixed_graph':True,'stable_entries_at_0_8':int(top.stable_at_0_8.sum()),'top3_mean_frequency':top.half_sample_frequency.mean(),
          'nodes':len(g),'edges':g.number_of_edges(),'maximum_mc_se':.5/np.sqrt(B),
          'single_chemical_candidates':int((~excluded).sum()),'excluded_macro_or_derived':NON_MOLECULAR,
          'conditional_importance':'training-selected correlated coordinates + held-out kNN donors; an approximation, not tree-cutpoint conditional importance',
          'selector':'style-residualized rank association + graph-community diversity; no claim to reproduce U1 SGL-SHAP',
          'error_control':'no PFER or FDR guarantee; exchangeability assumptions and original selector not established',
          'source_ids':['U1','L2','L3']},'q2_summary.json')
    plt=plotting();fig,ax=plt.subplots(figsize=(7,4));top.boxplot(column='half_sample_frequency',by='rank',ax=ax)
    ax.set_ylabel('无放回半样本入选频率');ax.set_title('Top-3 稳定性');fig.suptitle('');fig.tight_layout();fig.savefig(RFIG/'q2_stability.png');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['audit','q1','q2','q3','all'],default='all');p.add_argument('--subsamples',type=int,default=100);a=p.parse_args()
    if a.stage in ['audit','all']:evidence_audit()
    if a.stage in ['q1','all']:q1()
    if a.stage in ['q2','all']:q2(a.subsamples)
    if a.stage in ['q3','all']:
        from literature_generation import run
        run()

```

## literature_models.py

```python

"""Literature-backed fold-local models. U1, U2, L1, L2, L3 in sources.json.

The empirical-rank mapping is a modelling transform, not a chemical unit claim.
"""
import numpy as np
from scipy.special import ndtri
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge


class FoldFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, rank=False, indicators=True):
        self.rank=rank; self.indicators=indicators

    def fit(self,X,y=None):
        a=np.asarray(X,float)
        self.median_=np.nanmedian(a,axis=0)
        if not np.isfinite(self.median_).all(): raise ValueError('All-missing training feature')
        self.floor_=np.nanmin(a,axis=0)
        self.floor_active_=np.sum(a==self.floor_,axis=0)>=3
        self.sorted_=[np.sort(v[np.isfinite(v)]) for v in a.T]
        self.n_=len(a)
        return self

    def transform(self,X):
        a=np.asarray(X,float); missing=~np.isfinite(a)
        floor=(a==self.floor_)&self.floor_active_
        v=np.where(missing,self.median_,a)
        if self.rank:
            z=np.empty_like(v)
            for j,s in enumerate(self.sorted_):
                left=np.searchsorted(s,v[:,j],'left'); right=np.searchsorted(s,v[:,j],'right')
                u=(left+right)/(2*len(s))
                z[:,j]=ndtri(np.clip(u,.5/len(s),1-.5/len(s)))
            v=z
        return np.hstack([v,missing.astype(float),floor.astype(float)]) if self.indicators else v


def candidate(name,seed=42):
    if name.startswith('RankPLS'):
        return make_pipeline(FoldFeatures(rank=True),StandardScaler(),
                             PLSRegression(n_components=int(name[7:]),scale=False,max_iter=500))
    if name=='RankRidge':
        return make_pipeline(FoldFeatures(rank=True),StandardScaler(),Ridge(alpha=1000.))
    if name=='ExtraTrees':
        estimator=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=2,max_features='sqrt',n_jobs=4,random_state=seed)
    elif name=='RandomForest':
        estimator=RandomForestRegressor(n_estimators=160,min_samples_leaf=3,max_features='sqrt',n_jobs=4,random_state=seed)
    elif name=='Mean': estimator=DummyRegressor(strategy='mean')
    else: raise ValueError(name)
    return make_pipeline(FoldFeatures(),estimator)


def conditional_donors(Xtrain,Xtest,j,seed,k=8):
    """Near-neighbour conditional permutation approximation, NOT Strobl's tree grid.

    Conditioning features are selected using training correlations only. Donors
    come exclusively from held-out X; no held-out labels are consulted.
    """
    from sklearn.neighbors import NearestNeighbors
    train=np.asarray(Xtrain,float); test=np.asarray(Xtest,float)
    corr=np.corrcoef(train,rowvar=False)[j]
    eligible=np.flatnonzero((np.abs(np.nan_to_num(corr))>=.3)&(np.arange(train.shape[1])!=j))
    eligible=eligible[np.argsort(-np.abs(corr[eligible]))[:5]]
    rng=np.random.default_rng(seed)
    if not len(eligible): return rng.permutation(len(test)),0
    sd=train[:,eligible].std(0); sd=np.where(sd>1e-12,sd,1)
    z=(test[:,eligible]-train[:,eligible].mean(0))/sd
    neigh=NearestNeighbors(n_neighbors=min(k+1,len(test))).fit(z).kneighbors(z,return_distance=False)
    donors=[]
    for i,row in enumerate(neigh):
        row=row[row!=i]
        if not len(row): raise ValueError('Conditional permutation needs >1 test row')
        donors.append(rng.choice(row))
    return np.array(donors),len(eligible)

```

## literature_nsga2.py

```python

"""L5: NSGA-II with constrained domination, SBX, polynomial mutation.

Vectorized evaluations; feasibility always precedes objective ranking.
"""
import numpy as np


def ranks_and_crowding(F,violation):
    F=np.asarray(F,float);v=np.maximum(np.asarray(violation,float),0)
    feasible=v<=1e-8
    pareto=np.all(F[:,None,:]<=F[None,:,:],2)&np.any(F[:,None,:]<F[None,:,:],2)
    dominates=(feasible[:,None]&~feasible[None,:]) | (feasible[:,None]&feasible[None,:]&pareto) | (
        ~feasible[:,None]&~feasible[None,:]&(v[:,None]<v[None,:]))
    count=dominates.sum(0);rank=np.full(len(F),-1,int);front=np.flatnonzero(count==0);k=0
    crowd=np.zeros(len(F))
    while len(front):
        rank[front]=k
        if len(front)<=2:crowd[front]=np.inf
        else:
            for j in range(F.shape[1]):
                order=front[np.argsort(F[front,j],kind='stable')]
                span=F[order[-1],j]-F[order[0],j]
                if span<=1e-14:continue
                crowd[order[[0,-1]]]=np.inf
                crowd[order[1:-1]]+=(F[order[2:],j]-F[order[:-2],j])/span
        count-=dominates[front].sum(0);count[front]=-1
        front=np.flatnonzero((count==0)&(rank<0));k+=1
    assert np.all(rank>=0)
    return rank,crowd


def offspring(U,rank,crowd,rng,lo,hi):
    n,d=U.shape;children=[]
    def tournament():
        a,b=rng.integers(n,size=2)
        return a if rank[a]<rank[b] or (rank[a]==rank[b] and crowd[a]>=crowd[b]) else b
    for _ in range(n//2):
        p,q=U[tournament()],U[tournament()]
        u=rng.random(d);beta=np.where(u<=.5,(2*u)**(1/16),(1/(2*(1-u)))**(1/16))
        mask=rng.random(d)<.5;beta=np.where(mask,beta,1)
        for child in [.5*((1+beta)*p+(1-beta)*q),.5*((1-beta)*p+(1+beta)*q)]:
            child=np.clip(child,lo,hi)
            # Bounded polynomial mutation in normalized decision coordinates.
            z=(child-lo)/(hi-lo);sel=rng.random(d)<1/d;r=rng.random(d)
            delta=np.where(r<.5,(2*r+(1-2*r)*(1-z)**21)**(1/21)-1,
                            1-(2*(1-r)+2*(r-.5)*z**21)**(1/21))
            children.append(np.clip(child+sel*delta*(hi-lo),lo,hi))
    return np.array(children)


def solve(evaluate,lo,hi,seed=42,population=80,generations=100,initial=None):
    assert population%2==0
    rng=np.random.default_rng(seed);lo=np.asarray(lo);hi=np.asarray(hi)
    U=rng.uniform(lo,hi,(population,len(lo)))
    if initial is not None:U[:min(len(initial),population)]=np.clip(initial[:population],lo,hi)
    F,c=evaluate(U);trace=[]
    for generation in range(generations):
        rank,crowd=ranks_and_crowding(F,c)
        kids=offspring(U,rank,crowd,rng,lo,hi);kf,kc=evaluate(kids)
        both=np.vstack([U,kids]);bf=np.vstack([F,kf]);bc=np.r_[c,kc]
        br,bd=ranks_and_crowding(bf,bc)
        idx=np.lexsort((-bd,br))[:population]
        U,F,c=both[idx],bf[idx],bc[idx]
        trace.append({'generation':generation+1,'feasible':int(np.sum(c<=1e-8)),
                      'best_f1_feasible':float(F[c<=1e-8,0].min()) if np.any(c<=1e-8) else None})
    rank,_=ranks_and_crowding(F,c);ok=(rank==0)&(c<=1e-8)
    return U[ok],F[ok],c[ok],trace


def self_check():
    F=np.array([[1,2],[2,1],[3,3],[0,0.]])
    r,_=ranks_and_crowding(F,[0,0,0,1])
    assert r[0]==r[1]==0 and r[2]>0 and r[3]>r[2]
    def zdt1(U):
        g=1+9*U[:,1:].mean(1)
        return np.column_stack([U[:,0],g*(1-np.sqrt(U[:,0]/g))]),np.zeros(len(U))
    u,f,c,_=solve(zdt1,np.zeros(3),np.ones(3),seed=7,population=60,generations=100)
    residual=np.abs(f[:,1]-(1-np.sqrt(f[:,0])))
    assert len(f)>15 and np.median(residual)<.03 and np.ptp(f[:,0])>.8
    return {'zdt1_n':len(f),'zdt1_median_front_residual':float(np.median(residual)),
            'feasibility_priority_fixture':True}

```

## literature_generation.py

```python

"""U1/U2/L4/L5: evaluate the saved cVAE, compare posterior sampling,
conditional shrinkage Gaussian and empirical baselines, then latent NSGA-II.
No full-data generator is misreported as held-out validation.
"""
from common import *
import joblib,time
from sklearn.model_selection import StratifiedKFold
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.special import ndtr
from scipy.stats import wasserstein_distance
from literature_models import FoldFeatures
from literature_inference import NumpyCVAE
from literature_nsga2 import solve,ranks_and_crowding,self_check

REV=OUT/'literature_revision';RFIG=FIG/'literature_revision'


def save(obj,name):
    (REV/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist() if isinstance(v,np.ndarray) else str(v)),encoding='utf-8')


def empirical_inverse(z,sorted_cols):
    return np.column_stack([np.interp(ndtr(z[:,j]),(np.arange(len(v))+.5)/len(v),v) for j,v in enumerate(sorted_cols)])


def rank_gaussian_validation(X,s):
    # Shared within-style covariance prevents between-style mean differences
    # from masquerading as within-style dependence (U1, PDF pp.21-23).
    rows=[];rng=np.random.default_rng(42)
    for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(X,s)):
        pre=FoldFeatures(rank=True,indicators=False).fit(X.iloc[tr]);z=pre.transform(X.iloc[tr]);zv=pre.transform(X.iloc[va]);st=s.iloc[tr].to_numpy()
        residual=z.copy()
        for style in np.unique(st):residual[st==style]-=z[st==style].mean(0)
        covariance=LedoitWolf().fit(residual).covariance_
        L=np.linalg.cholesky(covariance+1e-7*np.eye(z.shape[1]));globalmean=z.mean(0)
        for style in ['Tripel','Kriek','Lambic']:
            train=z[st==style];test=zv[s.iloc[va].to_numpy()==style]
            if not len(train) or not len(test):continue
            center=(len(train)*train.mean(0)+3*globalmean)/(len(train)+3)
            for name,generated in [('ConditionalRankGaussian',center+rng.normal(size=(128,z.shape[1]))@L.T),
                                  ('UnconditionalRankGaussian',globalmean+rng.normal(size=(128,z.shape[1]))@L.T),
                                  ('StyleResampling',train[rng.integers(len(train),size=128)])]:
                w=np.mean([wasserstein_distance(test[:,j],generated[:,j]) for j in range(z.shape[1])])
                rows.append({'fold':fold,'style':style,'method':name,'heldout_n':len(test),'mean_marginal_w1':w})
    pd.DataFrame(rows).to_csv(REV/'q3_heldout_distribution.csv',index=False)
    return pd.DataFrame(rows).groupby('method').mean_marginal_w1.mean().to_dict()


def run():
    started=time.time();X,Y,s,ids=load();cols=list(X)
    rank_summary=rank_gaussian_validation(X,s)
    generator=NumpyCVAE(OUT/'q3_cvae.pt');assert generator.meta['feature_columns']==cols
    prep=joblib.load(OUT/'q3_preprocessing.joblib');imp,scale=prep['imputer'],prep['scaler']
    filled=imp.transform(X);z=scale.transform(filled)
    le=LabelEncoder().fit(s);lab=le.transform(s);cond=np.eye(len(le.classes_))[lab]
    assert generator.meta['style_classes']==list(le.classes_)
    mu,lv=generator.encode(z,cond)
    assert mu.shape==(len(X),8) and np.isfinite(lv).all()
    reconstruction=generator.decode(mu,cond)
    assert np.isfinite(reconstruction).all()
    clf=RandomForestClassifier(n_estimators=160,max_features='sqrt',class_weight='balanced',random_state=42,n_jobs=4).fit(z,lab)
    proxy=joblib.load(REV/'q1_model.joblib')
    audit=pd.read_csv(REV/'scale_evidence.csv').set_index('compound')
    rules={}
    for c,threshold in [('DMS',17.25),('X2.3.butanedione',.29925),('X4VG',.206333333),('X.E..2.nonenal',None)]:
        evidence=audit.loc[c]; verified=evidence.status=='verified_positive_rows'
        rules[c]={'index':cols.index(c),'log10':evidence.suggested_transform=='log10',
                  'physical_verified':verified,'threshold':threshold if verified else None,
                  'scope':'verified positive concentration rows' if verified else 'non-worsening on supplied analysis scale only'}
    lo,hi=np.nanmin(filled,0),np.nanmax(filled,0)
    def project(v,source=None):
        v=np.clip(np.asarray(v),lo,hi).copy()
        for c,r in rules.items():
            i=r['index'];cap=np.inf
            if r['threshold'] is not None:cap=.95*r['threshold'];cap=np.log10(cap) if r['log10'] else cap
            if source is not None:
                baseline=source[i] if r['log10'] else .95*source[i]
                if r['log10']:baseline+=np.log10(.95)
                cap=min(cap,baseline)
            v[:,i]=np.minimum(v[:,i],cap)
        return v
    # L4: explicit prior vs empirical-posterior diagnostic (not conditional prior network).
    rng=np.random.default_rng(42);sampler=[];generated=[];stylecol=[]
    for k,style in enumerate(le.classes_):
        index=rng.choice(np.flatnonzero(lab==k),60,replace=True);c=np.tile(np.eye(len(le.classes_))[k],(60,1))
        eps=rng.normal(size=(60,8))
        for mode,latent in [('posterior_mean_noise',mu[index]+.2*eps),
                            ('posterior_reparameterized',mu[index]+np.exp(.5*lv[index])*eps),
                            ('standard_normal_prior',eps)]:
            v=project(scale.inverse_transform(generator.decode(latent,c)))
            rankpre=FoldFeatures(rank=True,indicators=False).fit(X)
            zv=rankpre.transform(v);zr=rankpre.transform(X.iloc[np.flatnonzero(lab==k)])
            from sklearn.neighbors import NearestNeighbors
            dist=NearestNeighbors(n_neighbors=1).fit(zr).kneighbors(zv)[0][:,0]/np.sqrt(X.shape[1])
            sampler.append({'style':style,'sampler':mode,'classifier_agreement':np.mean(clf.predict(scale.transform(v))==k),
                            'mean_rank_variance':np.var(zv,axis=0).mean(),'nearest_real_rms':dist.mean(),
                            'near_duplicate_fraction':np.mean(dist<.01)})
            if mode=='posterior_reparameterized':generated.append(v);stylecol += [style]*60
    pd.DataFrame(sampler).to_csv(REV/'q3_sampler_diagnostics.csv',index=False)
    generated=np.vstack(generated);gpred=proxy.predict(pd.DataFrame(generated,columns=cols))
    pd.concat([pd.Series(stylecol,name='generated_style'),pd.DataFrame(generated,columns=cols)],axis=1).to_csv(REV/'q3_generated_chemistry.csv',index=False)
    pd.concat([pd.Series(stylecol,name='generated_style'),pd.DataFrame(gpred,columns=Y.columns)],axis=1).to_csv(REV/'q3_generated_sensory.csv',index=False)
    # Re-evaluate new generated profiles; do not reuse old generator's correlations.
    from external_analysis import coarse,normalized_counts,STYLE
    from scipy.stats import spearmanr
    ext=pd.read_csv(EXT/'beer_data_set.csv')
    realcoarse=coarse(Y).rank(pct=True)
    rawcoarse=coarse(Y);newcoarse=coarse(pd.DataFrame(gpred,columns=Y.columns))
    extcoarse=normalized_counts(ext).rank(pct=True)
    agreement=[]
    for dim in rawcoarse:
        rr=[];gg=[];ee=[]
        reference=np.sort(rawcoarse[dim])
        genpercentile=np.searchsorted(reference,newcoarse[dim],side='right')/len(reference)
        for style,external_styles in STYLE.items():
            mask=ext.Style.isin(external_styles)
            if not mask.any():continue
            rr.append(realcoarse.loc[s==style,dim].mean())
            gg.append(genpercentile[np.asarray(stylecol)==style].mean())
            ee.append(extcoarse.loc[mask,dim].mean())
        agreement.append({'dimension':dim,'styles':len(rr),
                          'official_rho':float(spearmanr(rr,ee).statistic),
                          'generated_rho':float(spearmanr(gg,ee).statistic)})
    pd.DataFrame(agreement).to_csv(REV/'external_agreement.csv',index=False)
    # L5 actual NSGA-II search: 8 latent variables; one prespecified source beer.
    src_idx=np.flatnonzero(s.to_numpy()=='Pils/Lager');src=src_idx[np.argmin(np.mean((z[src_idx]-z[src_idx].mean(0))**2,axis=1))]
    source=filled[src];target=le.transform(['Tripel'])[0];targety=Y.loc[s=='Tripel'].mean().to_numpy()
    calibrated_source=pd.read_excel(DATA/'8.xlsx',sheet_name='Calibrated chemical dataset').set_index('beer').loc[int(ids.iloc[src].beer)]
    source_physical=[name for name,r in rules.items() if r['physical_verified'] and pd.notna(calibrated_source[name]) and calibrated_source[name]>0]
    targetc=np.eye(len(le.classes_))[target];yscale=Y.std().to_numpy();stale=Y.columns.get_loc('stale_hops')
    source_y=proxy.predict(pd.DataFrame(source[None,:],columns=cols))[0]
    targetmu=mu[lab==target].mean(0);sigma=np.maximum(mu.std(0),.25)
    lower=np.minimum(mu[src]-2*sigma,targetmu-2*sigma);upper=np.maximum(mu[src]+2*sigma,targetmu+2*sigma)
    sd=np.maximum(np.nanstd(filled,axis=0,ddof=1),1e-8)
    def evaluate(U,details=False):
        conditions=np.tile(targetc,(len(U),1));v=project(scale.inverse_transform(generator.decode(U,conditions)),source)
        yp=proxy.predict(pd.DataFrame(v,columns=cols))
        burden=[]
        for c,r in rules.items():
            i=r['index']
            if r['log10']:burden.append(10**(v[:,i]-source[i]))
            else:burden.append(v[:,i]/max(source[i],1e-12))
        F=np.column_stack([np.mean(((yp-targety)/yscale)**2,axis=1),
                           np.sqrt(np.mean(((v-source)/sd)**2,axis=1)),np.mean(burden,axis=0)])
        violation=np.maximum((yp[:,stale]-source_y[stale])/yscale[stale],0)
        if not np.isfinite(F).all():raise ValueError('Nonfinite objective')
        return (F,violation,v,yp) if details else (F,violation)
    Ulist=[];Flist=[];runs=[]
    for seed in [42,43,44]:
        u,f,c,trace=solve(evaluate,lower,upper,seed=seed,population=80,generations=80,
                         initial=np.vstack([mu[src],targetmu]))
        Ulist.append(u);Flist.append(f);runs.append({'seed':seed,'n_feasible_front':len(u),'stop':'fixed 80-generation budget; no global optimality claim'})
        pd.DataFrame(trace).to_csv(REV/f'q3_nsga2_trace_{seed}.csv',index=False)
        print('NSGA-II seed',seed,'front',len(u),flush=True)
    U=np.vstack(Ulist);F=np.vstack(Flist)
    if not len(U):
        save({'status':'no feasible front','runs':runs,'source_beer':int(ids.iloc[src].beer)},'q3_nsga2_summary.json')
        raise RuntimeError('No feasible latent Pareto points; constraints not relaxed')
    U,indices=np.unique(U,axis=0,return_index=True);F=F[indices]
    rank,_=ranks_and_crowding(F,np.zeros(len(F)));U,F=U[rank==0],F[rank==0]
    F,c,V,YP=evaluate(U,True);assert np.all(c<=1e-8)
    for r in rules.values():assert np.all(V[:,r['index']]<=source[r['index']]+1e-10)
    span=np.ptp(F,axis=0);active=span>1e-12;Z=np.zeros_like(F);Z[:,active]=(F[:,active]-F[:,active].min(0))/span[active]
    choices=[]
    for label,w in [('balanced',[1,1,1]),('sensory_first',[.6,.2,.2]),('small_change',[.2,.6,.2]),('low_defect',[.2,.2,.6])]:
        w=np.asarray(w)*active;w=w/w.sum() if w.sum() else w
        plus=np.sqrt(np.sum(Z**2*w,1));minus=np.sqrt(np.sum((Z-active)**2*w,1))
        closeness=np.divide(minus,plus+minus,out=np.full(len(F),.5),where=plus+minus>0);i=int(closeness.argmax())
        choices.append({'scenario':label,'front_id':i+1,'f1':F[i,0],'f2':F[i,1],'f3':F[i,2],'topsis':closeness[i]})
    pd.DataFrame(choices).to_csv(REV/'q3_decisions.csv',index=False)
    pd.DataFrame(np.column_stack([U,F,c]),columns=[f'z{i}' for i in range(8)]+['f1','f2','f3','constraint_violation']).to_csv(REV/'q3_pareto.csv',index=False)
    pd.DataFrame(V,columns=cols).to_csv(REV/'q3_pareto_chemistry.csv',index=False)
    pd.DataFrame(YP,columns=Y.columns).to_csv(REV/'q3_pareto_sensory.csv',index=False)
    save({'source_beer':int(ids.iloc[src].beer),'target_style':'Tripel','n_pareto':len(F),
          'runs':runs,'verified_physical_caps':source_physical,
          'positive_row_transform_fields':[c for c,r in rules.items() if r['physical_verified']],
          'analysis_scale_only':[c for c,r in rules.items() if c not in source_physical],
          'source_calibrated_4vg':float(calibrated_source['X4VG']),
          'source_low_value_caveat':'Source 4VG calibrated value is zero; its supplied floor does not establish a positive physical source concentration. 4VG non-worsening is analysis-scale only.',
          'max_stale_violation':float(c.max()),'engine':'Python NSGA-II implementation; NOT MATLAB',
          'source_ids':['U1','U2','L4','L5'],'tests':self_check()},'q3_nsga2_summary.json')
    save({'heldout_rank_gaussian_w1':rank_summary,
          'sampler_averages':pd.DataFrame(sampler).groupby('sampler')[['classifier_agreement','mean_rank_variance','nearest_real_rms','near_duplicate_fraction']].mean().to_dict('index'),
          'saved_cvae_full_data_reconstruction_mse':float(np.mean((reconstruction-z)**2)),
          'cvae_scope':'fixed pretrained full-data cVAE, no new heldout cVAE claim; NumPy dense float32 inference',
          'rank_gaussian_scope':'fixed shrinkage controls kappa=3, covariance=shared LedoitWolf; not U1 full hyperparameter selection',
          'seconds':time.time()-started},'q3_summary.json')
    plt=plotting();fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection='3d');sc=ax.scatter(F[:,0],F[:,1],F[:,2],c=F[:,0],s=16,cmap='viridis')
    ax.set_xlabel('目标感官偏差');ax.set_ylabel('化学谱改动');ax.set_zlabel('相对缺陷指标');fig.colorbar(sc,ax=ax,shrink=.6)
    fig.tight_layout();fig.savefig(RFIG/'q3_pareto.png');plt.close(fig)

```

## literature_inference.py

```python

"""Restricted NumPy reader for the project's own saved dense float32 cVAE.

Only explicit numeric tensor rebuilds and OrderedDict are allowed; arbitrary
pickle globals are rejected. This avoids installing PyTorch for inference.
"""
import collections,io,pickle,zipfile
import numpy as np


def load_checkpoint(path):
    with zipfile.ZipFile(path) as archive:
        key=next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        root=key.rsplit('/',1)[0]
        class Reader(pickle.Unpickler):
            def find_class(self,module,name):
                if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
                if module=='torch' and name=='FloatStorage':return np.dtype('<f4')
                if module=='torch._utils' and name=='_rebuild_tensor_v2':
                    def rebuild(storage,offset,size,stride,*rest):
                        return np.ndarray(tuple(size),dtype=storage.dtype,buffer=storage,
                            offset=offset*storage.itemsize,strides=tuple(s*storage.itemsize for s in stride)).copy()
                    return rebuild
                raise pickle.UnpicklingError('Unsupported checkpoint global '+module+'.'+name)
            def persistent_load(self,pid):
                kind,dtype,storage,location,count=pid
                if kind!='storage' or str(location)!='cpu':raise ValueError('Expected CPU numeric checkpoint')
                result=np.frombuffer(archive.read(root+'/data/'+str(storage)),dtype=dtype)
                if len(result)!=count:raise ValueError('Storage size mismatch')
                return result
        result=Reader(io.BytesIO(archive.read(key))).load()
    return result


class NumpyCVAE:
    def __init__(self,path):
        saved=load_checkpoint(path); self.w=saved['state_dict']; self.meta={k:v for k,v in saved.items() if k!='state_dict'}
    def linear(self,a,key):return a@self.w[key+'.weight'].T+self.w[key+'.bias']
    def encode(self,z,c):
        a=np.concatenate([z,c],axis=1).astype(np.float32)
        a=np.maximum(self.linear(a,'enc.0'),0);a=np.maximum(self.linear(a,'enc.2'),0)
        return self.linear(a,'mu'),np.clip(self.linear(a,'lv'),-10,10)
    def decode(self,z,c):
        a=np.concatenate([z,c],axis=1).astype(np.float32)
        a=np.maximum(self.linear(a,'dec.0'),0);a=np.maximum(self.linear(a,'dec.2'),0)
        return self.linear(a,'dec.4')

```

## validate_literature_revision.py

```python

"""Independent invariants for the revised model outputs and final paper."""
from common import *
import ast,joblib
from literature_models import FoldFeatures
from literature_nsga2 import ranks_and_crowding

REV=OUT/'literature_revision'


def main():
    checks=[]
    def check(label,condition):
        checks.append({'check':label,'pass':bool(condition)})
        if not condition:raise AssertionError(label)
    X,Y,s,ids=load()
    # Fit rules must remain frozen even when validation values shift markedly.
    train=np.array([[0.,np.nan],[0.,2.],[0.,4.],[8.,6.]])
    p=FoldFeatures(rank=True).fit(train);floor=p.floor_.copy();med=p.median_.copy()
    transformed=p.transform(np.array([[100.,np.nan],[-50.,3.]]))
    check('fold rules frozen on unseen extreme validation values',np.array_equal(floor,p.floor_) and np.array_equal(med,p.median_))
    check('unseen extreme is not a newly invented repeated floor',transformed[1,4]==0)
    check('missing state separate from repeated floor state',transformed[0,3]==1 and transformed[0,5]==0)
    check('fold transforms finite',np.isfinite(transformed).all())
    choices=json.loads((REV/'q1_nested_choices.json').read_text(encoding='utf-8'))
    check('nine independent outer fit selections',len(choices)==9 and len({(r['seed'],r['outer_fold']) for r in choices})==9)
    check('all inner choices contain all six declared candidates',all(len(r['inner_scores'])==6 for r in choices))
    oof=pd.read_csv(REV/'q1_oof_predictions.csv');check('OOF rows aligned',oof.beer.equals(ids.beer))
    grade=pd.read_csv(REV/'q1_relative_grades.csv')[list(Y)].to_numpy()
    check('250 by 50 relative grades, finite integers 0 to 6',grade.shape==Y.shape and np.isfinite(grade).all() and np.equal(grade,np.rint(grade)).all() and ((grade>=0)&(grade<=6)).all())
    source=json.loads((REV/'audit_summary.json').read_text(encoding='utf-8'))
    check('input and checkpoint hashes unchanged',all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h for path,h in source['input_sha256'].items()))
    scale=pd.read_csv(REV/'scale_evidence.csv').set_index('compound')
    check('no blanket raw concentration verification',scale.loc['X2.3.butanedione','status']=='mixed_or_unresolved' and scale.loc['X.E..2.nonenal','status']=='mixed_or_unresolved')
    top=pd.read_csv(REV/'q2_top3.csv')
    check('Top3 chemically grouped and unique',len(top)==150 and top.groupby('target').compound.nunique().eq(3).all())
    check('fixed graph diversity constraints retained',top.groupby('target').module.nunique().eq(3).all())
    from literature_revision import NON_MOLECULAR
    check('single-chemical Top3 excludes all 13 macro and derived fields',not top.compound.isin(NON_MOLECULAR).any())
    check('half-sample probabilities valid',top.half_sample_frequency.between(0,1).all())
    conditional=pd.read_csv(REV/'q2_conditional_importance.csv')
    check('all three heldout explanation folds and three repeats present',set(conditional.fold)=={0,1,2} and set(conditional.repeat)=={0,1,2})
    check('negative conditional effects not silently clipped',(conditional.conditional_delta_mse<0).any())
    ns=json.loads((REV/'q3_nsga2_summary.json').read_text(encoding='utf-8'))
    a=pd.read_csv(REV/'q3_pareto.csv');chem=pd.read_csv(REV/'q3_pareto_chemistry.csv');pred=pd.read_csv(REV/'q3_pareto_sensory.csv')
    check('all Pareto files have matching rows',len(a)==len(chem)==len(pred)==ns['n_pareto'])
    check('numeric Pareto outputs finite',np.isfinite(a.to_numpy()).all() and np.isfinite(chem.to_numpy()).all())
    m=joblib.load(REV/'q1_model.joblib');fresh=m.predict(chem[list(X)])
    check('stored Pareto sensory predictions match independent proxy evaluation',np.allclose(fresh,pred[list(Y)],atol=1e-9))
    src=X.loc[ids.beer==ns['source_beer']].fillna(X.median());srcpred=m.predict(src)[0]
    stale=Y.columns.get_loc('stale_hops')
    check('independent final stale proxy non-worsening',np.all(fresh[:,stale]<=srcpred[stale]+1e-8))
    for col in ['DMS','X2.3.butanedione','X4VG','X.E..2.nonenal']:
        check('independent analysis scale non-worsening '+col,np.all(chem[col]<=src.iloc[0][col]+1e-10))
    check('DMS assumed concentration threshold',np.all(10**chem.DMS<=17.25+1e-9))
    check('zero-calibrated source 4VG explicitly not physical non-worsening',ns['source_calibrated_4vg']==0 and 'X4VG' in ns['analysis_scale_only'] and 'X4VG' not in ns['verified_physical_caps'])
    ranks,_=ranks_and_crowding(a[['f1','f2','f3']].to_numpy(),np.zeros(len(a)))
    check('independent combined Pareto nondominance',np.all(ranks==0))
    target=Y.loc[s=='Tripel'].mean().to_numpy();f1=np.mean(((fresh-target)/Y.std().to_numpy())**2,axis=1)
    check('independent objective1 recomputation',np.allclose(f1,a.f1,atol=1e-9))
    d=pd.read_csv(REV/'q3_decisions.csv');check('all four decision preferences refer to real Pareto rows',len(d)==4 and d.front_id.between(1,len(a)).all())
    check('known analytical NSGA-II test performed',ns['tests']['zdt1_median_front_residual']<.03 and ns['tests']['feasibility_priority_fixture'])
    refs=json.loads((ROOT/'literature/sources.json').read_text(encoding='utf-8'))
    check('two provided papers and exactly five additional papers',sum(r['provided'] for r in refs)==2 and sum(not r['provided'] for r in refs)==5)
    for file in (ROOT/'code').glob('literature*.py'):ast.parse(file.read_text(encoding='utf-8'))
    check('all new experimental modules parse',True)
    pdf=ROOT/'output/literature_revision/啤酒风味论文_七篇文献优化版.pdf'
    if pdf.exists():
        from pypdf import PdfReader
        reader=PdfReader(pdf);text='\n'.join(p.extract_text() or '' for p in reader.pages)
        check('revised paper includes all model chapters and references',all(word in text for word in ['问题一','问题二','问题三','Pareto','参考文献']))
        check('paper preserves important limitations',all(word in text for word in ['不是精确','未安装MATLAB','零','原始绝对评分']))
        check('paper pages contain no replacement glyph', '\ufffd' not in text)
        source['pdf_pages']=len(reader.pages)
    pd.DataFrame(checks).to_csv(REV/'validation_checks.csv',index=False)
    (REV/'validation_summary.json').write_text(json.dumps({'checks':len(checks),'all_passed':True,'pdf_pages':source.get('pdf_pages'),'visual_review':'separate PNG review required'},indent=2),encoding='utf-8')
    print('PASS',len(checks))


if __name__=='__main__':main()

```

## build_literature_paper.py

```python

"""Build a complete revised paper and per-source change ledger from actual runs.
PDF rendering and visual inspection are a separate required delivery step.
"""
from common import *
import html
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

REV=OUT/'literature_revision';RFIG=FIG/'literature_revision'
OUTDOC=ROOT/'output/literature_revision'


def main():
    OUTDOC.mkdir(parents=True,exist_ok=True)
    def js(n):return json.loads((REV/n).read_text(encoding='utf-8'))
    audit=js('audit_summary.json');q1=js('q1_summary.json');q2=js('q2_summary.json');q3=js('q3_summary.json');nsga=js('q3_nsga2_summary.json')
    refs=json.loads((ROOT/'literature/sources.json').read_text(encoding='utf-8'))
    old1=json.loads((OUT/'q1_summary.json').read_text(encoding='utf-8'))
    external=json.loads((OUT/'external_audit.json').read_text(encoding='utf-8'))
    ratings=json.loads((OUT/'external_rating_summary.json').read_text(encoding='utf-8'))
    v1=q1['pooled_oof_scores']['NestedSelected'];vbase=q1['pooled_oof_scores']['FixedRandomForest']
    top=pd.read_csv(REV/'q2_top3.csv');ci=pd.read_csv(REV/'q2_conditional_importance.csv')
    decisions=pd.read_csv(REV/'q3_decisions.csv');scale=pd.read_csv(REV/'scale_evidence.csv')
    compare=pd.read_csv(REV/'q2_selector_comparison.csv')
    pdfmetrics.registerFont(TTFont('Song',r'C:\Windows\Fonts\simsun.ttc',subfontIndex=0))
    pdfmetrics.registerFont(TTFont('Hei',r'C:\Windows\Fonts\simhei.ttf'))
    sty={
      'body':ParagraphStyle('body',fontName='Song',fontSize=10.5,leading=18,spaceAfter=7,firstLineIndent=21,wordWrap='CJK',alignment=TA_JUSTIFY),
      'h1':ParagraphStyle('h1',fontName='Hei',fontSize=15,leading=23,spaceBefore=15,spaceAfter=9,keepWithNext=True),
      'h2':ParagraphStyle('h2',fontName='Hei',fontSize=12,leading=20,spaceBefore=10,spaceAfter=6,keepWithNext=True),
      'title':ParagraphStyle('title',fontName='Hei',fontSize=22,leading=33,spaceAfter=20,alignment=TA_CENTER),
      'cell':ParagraphStyle('cell',fontName='Song',fontSize=8.5,leading=13,wordWrap='CJK'),
      'caption':ParagraphStyle('caption',fontName='Song',fontSize=9,leading=14,alignment=TA_CENTER,spaceAfter=8,keepWithNext=True)}
    story=[];md=[]
    def p(text,style='body'):
        story.append(Paragraph(html.escape(str(text)),sty[style]));md.append(str(text))
    def h(text):p(text,'h1');md[-1]='## '+md[-1]
    def sub(text):p(text,'h2');md[-1]='### '+md[-1]
    def table(title,headers,rows,widths=None):
        p(title,'caption');data=[[Paragraph(html.escape(str(c)),sty['cell']) for c in r] for r in [headers]+rows]
        t=Table(data,colWidths=widths or [475/len(headers)]*len(headers),repeatRows=1,hAlign='CENTER')
        t.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.7,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),.7,colors.black),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf2f4')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([t,Spacer(1,8)]);md.extend(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
    def fig(name,caption):
        path=RFIG/name;w,hh=PILImage.open(path).size
        story.append(KeepTogether([Image(str(path),width=460,height=460*hh/w),Paragraph(html.escape(caption),sty['caption'])]));md.append(f'![{caption}](../../figures/literature_revision/{name})')
    def f(x,n=4):return f'{float(x):.{n}f}'
    p('啤酒感官预测与受约束风格迁移','title')
    p('结合外部数据与文献证据的多目标决策研究','caption')
    h('摘要')
    p('针对化学测量、专业感官评分与消费者评论具有不同数据尺度和采样机制的问题，本文在250款啤酒、231个化学与理化特征及50个感官响应上，构建带有证据审计的预测、归因、生成和多目标决策流程。外部啤酒评论数据只作风格趋势对照，不充当化学测量或专业品评标签。')
    p(f'针对问题一，采用折内缺失及重复底值状态处理，引入秩正态偏最小二乘与树模型候选。外层三折重复三次评价完整选型过程，内层三折选择模型。嵌套程序的完整折外预测平均R²为{f(v1["r2"])}、MAE为{f(v1["mae"])}；同划分固定随机森林的R²为{f(vbase["r2"])}。输出0至6相对等级，但不宣称恢复原始绝对评分。')
    p(f'针对问题二，保留固定共表达网络的原三方法结果作为对照，新增风格残差秩关联选择器及{q2["subsamples"]}次无放回近似分层半样本诊断。150个候选中有{q2["stable_entries_at_0_8"]}项频率达到0.8；通过未见标签的验证特征近邻供体估计条件置换重要性，明确相关变量、候选选择规则及固定图的局限。')
    p(f'针对问题三，复用已训练条件VAE比较三种采样方式，另对条件收缩高斯及两个经验基准进行分层留出分布检验。尺度审计发现70列有标定对照，其中{audit["scale_status_counts"]["verified_positive_rows"]}列在可比正值行中一致、32列存在局部不一致。迁移将已核实正浓度约束和分析尺度非恶化分开，实际运行三次8维潜空间NSGA-II搜索，获得{nsga["n_pareto"]}个近似Pareto候选。')
    p('不同偏好下的决策体现目标感官接近度、化学画像改动与相对缺陷指标之间的取舍。数值约束及代理模型结果仅用于计算机筛选，不替代加标品评或酿造验证。本文的改进首先在于验证隔离、证据标注与可追溯决策，不以增加算法数量代替性能和实验依据。')
    p('关键词  嵌套验证  条件重要性  稳定性选择  条件生成  Pareto决策','caption');story.append(PageBreak())
    h('一 问题与研究范围')
    p('赛题要求由化学谱预测感官评分、识别各感官属性Top-3因素，以及在风格条件下生成化学谱并完成缺陷不恶化的迁移。本文保留外部数据修订版的数据对齐和风格语义映射，并在其基础上重新实现可被文献支持的验证与优化模块。三问之间传递的是预测模型及候选证据，不是已经验证的因果关系。')
    p('用户提供方法稿[1]强调折内尺度处理、重复底值状态与约束证据分层；Schreurs等[2]在真实商业啤酒上比较多类模型并开展加标品评。本文进一步参考模型选择偏差[3]、条件重要性[4]、稳定性选择[5]、条件生成[6]和NSGA-II[7]，分别改善统计验证、解释可靠性及决策透明度。')
    p('原稿[2]的175/75分层训练测试及五折选参，与本文重复嵌套验证不同；其盲评加标结果是该研究自身的实验，不能作为本文候选产品已经改善的证据。[1]为用户提供的方法稿，未核实其作者和公开发表状态，本研究不把它等同于已同行评审成果，也不搬用其数值。')
    h('二 数据尺度与证据')
    sub('2.1 样本连接与来源')
    table('表1 数据与分析角色',['来源','规模','角色'],[['附件1与4','250款、231特征、50响应','按唯一beer连接并核对风格'],['附件2','231维相关结构摘要','固定网络先验，非独立新样本'],['附件3','空白评分卡及汇总','定义评分维度，不能从缓存零造标签'],['附件8','70列有可比标定','逐元素尺度与单位证据'],['外部数据',f'{external["rows"]}款、{external["breweries"]}个酒厂','评论词频和风格背景，不与专业标签拼接']],[105,155,215])
    p(f'主数据有{audit["complete_cases"]}个完整案例，其他样本保留并在训练折内插补。输入以SHA256固定。未把同一啤酒的50个感官维度视为50个独立样本，也未以发生精度损失的长商品编号强行匹配外部产品。')
    sub('2.2 尺度账本与局部不一致')
    p('对每个有标定的字段，在参考浓度为正且双方非缺失的行上，逐一比较恒等、常用对数及自然对数关系。只有至少五个可比行且全部误差不超过1e-6，才记为正值行已核实；中位数误差接近零不能掩盖少量不一致。该证据不覆盖零值替代机制，也不证明不同版本数据完全一致。')
    defects=['DMS','X2.3.butanedione','X4VG','X.E..2.nonenal'];r=scale.set_index('compound')
    table('表2 四类缺陷的尺度证据',['字段','候选变换','不一致条数','本次约束口径'],[[c,r.loc[c,'suggested_transform'],int(r.loc[c,'n_discrepancy']),'正值域浓度及阈值' if r.loc[c,'status']=='verified_positive_rows' else '仅附件分析值非恶化'] for c in defects],[155,75,75,170])
    p('DMS和4VG的可比正值支持log10转换；双乙酰和反式2壬烯醛虽多数支持原值口径，却存在局部不一致，本次不以“多数一致”宣称全部物理浓度已验证。后两项继续采用附件值非增加约束，不对其未知或未完全核实尺度作完整化学阈值保证。DMS的17.25 µg/L和4VG的0.206333333 mg/L来自附件参考表，属于风味参考值，不是法定安全限值。')
    sub('2.3 缺失与重复底值')
    p('参考[1]第5页，显式NA以训练中位数插补并附缺失标记；当某列训练最小值至少重复三次时，另构造底值状态标记。重复最低值不是已知仪器检出限，不随机抖动也不直接当作真实浓度。验证样本只按训练规则变换，不能重新估计中位数、底值或经验分布。')
    p('秩正态候选使用训练经验中秩u=(小于当前值的个数+等于值的个数/2)/训练非缺失数，并以1/(2n)截断概率后映射到标准正态分位。它是一种稳健预测尺度，不是对全部字段重复施加化学对数。')
    h('三 问题一 感官预测与验证隔离')
    sub('3.1 候选模型与完整选型程序')
    p('候选预先固定为均值、3或8成分的秩正态PLS、秩正态Ridge、随机森林和极端随机树。PLS在相关响应间共享低维结构；两个树模型各160棵，分别使用最小叶样本3和2、平方根特征抽样。Ridge强度固定1000。所有候选均保留NA和重复底值状态，PLS与Ridge另执行训练内标准化。')
    p('根据[3]，选择模型本身属于训练。外层按风格分三折、使用42、1024和2024三个种子重复，内层在外层训练集上打乱三折比较多响应均方误差。内层采用普通K折，是因为稀有风格进一步划分后可能不足三例。每次内层拟合都独立重建预处理；外层响应不参与选型。最终评价对象是这套预声明选型程序。')
    p('每个重复拼接完整的250行折外预测后，按各感官目标计算R²、MAE和Spearman，再跨目标与重复汇总。折间结果、逐目标结果及各外层内层选择明细均保存。跨重复标准差只描述划分波动，不是九个独立试验的置信区间。')
    table('表3 同划分外层折外预测结果',['程序','R²','MAE','Spearman'],[[k,f(v['r2']),f(v['mae']),f(v['spearman'])] for k,v in q1['pooled_oof_scores'].items()],[180,95,95,105]);fig('q1_nested.png','图1 选型程序与固定随机森林的重复外层折外表现')
    gain=v1['r2']-vbase['r2']
    p(f'本次嵌套程序相对固定随机森林的平均R²差为{f(gain)}。该差异来自同划分的两套程序，但本研究没有据此进行未经设计的显著性宣称。原修订版固定模型比较中随机森林折均R²为{f(old1["scores"]["RandomForest"]["r2"])}，其汇总尺度和本次完整OOF汇总不同，不能直接把两者之差全归因于算法提高。')
    table('表4 外层训练内部选择次数',['候选','次数'],[[k,v] for k,v in q1['selected_counts'].items()],[300,175])
    sub('3.2 相对等级与误差传播')
    p('附件4是标准化连续响应，评分卡不提供原始逐人0至6校准锚点。因此论文首先报告连续响应的可检验误差，再输出训练响应经验分位划成七档的相对等级。全样本拟合表只用于描述已知样品，不用拟合等级一致率证明绝对评分恢复。后续真实品评应记录原始逐人评分及标准化参数。')
    per=pd.read_csv(REV/'q1_nested_per_target.csv').groupby('target').r2.mean()
    p(f'50个属性中有{int((per>0).sum())}个的重复外层平均R²大于零，stale_hops为{f(per.loc["stale_hops"])}。尤其第三问使用感官代理作为约束时，必须同时公开该属性的泛化能力；数值不恶化不等于真实品评不恶化。')
    h('四 问题二 相关化合物与稳定性')
    sub('4.1 保留原结果并引入可核查对照')
    p('原修订版融合风格控制残差相关、Ridge系数与逐属性树重要性，且Top-3来自不同图社区。本次保留该表及原Bootstrap结果；参考[1]排除13项宏观与派生汇总量，在218项单项化学候选内新增秩正态风格残差关联选择器。第一问预测仍可使用全部231项信息，但不能把糖类总量或派生酯类汇总冒充可直接干预的单一分子。新选择器用于诊断对照，不宣称自动优于原三证据融合。')
    p(f'新选择器与原三方法Top-3的逐属性平均Jaccard为{f(compare.jaccard_vs_original_three_method.mean())}。名单变化提示选择规则敏感，不能仅因更换选择器后频率提高便称化学机制更可靠。固定附件网络仍来自全样本摘要，所以所有结果均条件于该图，未构成独立网络验证。')
    sub('4.2 条件重要性')
    p('Strobl等[4]指出，在相关预测变量上，普通随机置换可能破坏协变量依赖并扭曲重要性。本次只用每个训练折选择Top-3及至多五个绝对相关大于等于0.3的上下文变量，再在验证特征的上下文空间寻找八个近邻供体，整体替换该分子的原值，由管线重算其NA和底值状态。')
    p('分别进行三次局部供体置换及普通置换，计算逐目标MSE相对未置换预测的增量，并保留负值。供体选择不读取验证响应。该近邻方法是本地条件分布的近似，不是[4]按树切点分组的精确实现，不能当作条件随机化检验或因果效应。')
    group=ci.groupby(['target','compound'])[['conditional_delta_mse','marginal_delta_mse']].mean()
    table('表5 条件与普通置换的描述性结果',['指标','结果'],[['目标化合物组合数',len(group)],['条件置换平均MSE增量',f(group.conditional_delta_mse.mean())],['普通置换平均MSE增量',f(group.marginal_delta_mse.mean())],['条件增量为负的组合数',int((group.conditional_delta_mse<0).sum())]],[320,155])
    sub('4.3 无放回半样本稳定性')
    p(f'参考[5]，重复{q2["subsamples"]}次在各风格内无放回抽取约一半样本，稀有风格至少保留一个，每次重估插补、经验秩、风格残差与Top-3。它与原有有放回Bootstrap不同。由于逐风格取整，样本数并非严格总样本的一半；故这里只借鉴子样本选择概率思想，不套用其原理论的误选数界。')
    p(f'新诊断Top-3平均频率为{f(q2["top3_mean_frequency"])}，达到0.8的条目为{q2["stable_entries_at_0_8"]}/150；有限重复的最坏Monte Carlo标准误约{f(q2["maximum_mc_se"])}。频率只反映当前样本、固定图和选型规则的稳定性，不是化合物为真因子的概率。')
    selected=['A_esters_all','F_hops_all','dms','stale_hops','overall'];rows=[]
    for target in selected:
        a=top[top.target==target];rows.append([target,'；'.join(f'{r.compound} ({r.half_sample_frequency:.2f})' for r in a.itertuples())])
    table('表6 代表属性的诊断Top-3及半样本频率',['属性','候选及频率'],rows,[105,370]);fig('q2_stability.png','图2 无放回半样本诊断的入选频率')
    h('五 问题三 条件生成与分布检验')
    sub('5.1 条件变分模型的继承范围')
    p('复用外部数据修订版已保存的8维条件VAE和其插补、标准化参数。编码器以化学谱和风格输出mu及logvar，解码器以潜变量和风格输出化学画像。与[6]一般的条件先验网络不同，原模型KL项使用标准正态先验；拼接风格本身不能被写成已学习条件先验。')
    p('本次不重新训练cVAE，使用限制可加载类型的NumPy推理器读取项目自身的float32权重，计算确定性线性层和ReLU。原模型完整数据重构仅为训练内诊断；没有把它称为留出验证。原修订版三折cVAE留出误差较大，仍保留作为当前生成器泛化不足的证据。')
    sub('5.2 后验不确定性与三种采样')
    p('根据[6]的重参数化形式，新增z=mu+exp(logvar/2)*epsilon，与原有后验均值加固定0.2噪声及标准正态采样并列。前者反映已学习编码后验的不确定性，后两者分别是经验扰动及先验对照；任一方式都不能仅凭高风格识别率证明分布真实。')
    table('表7 固定cVAE的内部采样诊断',['采样','风格一致率','秩空间方差','最近真实谱RMS'],[[k,f(v['classifier_agreement']),f(v['mean_rank_variance']),f(v['nearest_real_rms'])] for k,v in q3['sampler_averages'].items()],[185,95,95,100])
    p('每种风格各生成60条并记录最近真实样品距离与近重复率。方差为组内诊断平均，不能与包含风格间差异的总体方差混称。生成谱明确带generated_style标识；它们是模型输出，不是新增观测，未被回灌为第一问监督标签。')
    sub('5.3 分层留出分布基准')
    p('借鉴[1]第21至23页，在每个外层训练集用共同经验中秩构造正态得分，先减去各风格均值再用LedoitWolf估计共享组内协方差。风格中心按n/(n+3)向全局中心收缩，参数3与共享协方差事先固定，本次没有复现[1]完整超参数网格。')
    p('将Tripel、Kriek和Lambic的留出样本与条件高斯生成、无条件高斯及风格内训练谱重采样比较。评价为231个正态得分边缘的平均Wasserstein距离；它衡量边缘而非联合分布，不能证明消费者喜好或物理可酿造性。所有边缘和协方差均由对应训练折拟合。')
    table('表8 三折留出平均边缘距离',['模型','W1 越小越好'],[[k,f(v)] for k,v in q3['heldout_rank_gaussian_w1'].items()],[330,145])
    p('无论哪一种方法数值最优，均完整保留经验重采样基准。复制训练样品可能在边缘距离上表现较好，但新颖性有限；模型需要同时报告分布接近程度和与真实样品的距离，不能删除强基准突出新方法。')
    h('六 多目标迁移与Pareto决策')
    sub('6.1 决策变量与约束证据')
    p(f'以真实源样品beer={nsga["source_beer"]}从Pils/Lager迁移到Tripel为演示，在固定解码器的8维潜空间搜索。潜变量边界覆盖源后验与目标风格中心附近两个经验标准差；解码输出限制到训练各列经验范围，再施加缺陷修正。该候选族是化学分析谱，不是配方，经验边界也不证明联合物理可行。')
    p('DMS采用正浓度口径，数值上限制为不超过参考阈值和源值的0.95倍。4VG虽在正标定行支持log10变换，但本次源样品的参考标定为零，附件记录的是底值，不能把其指数值当作已核实源浓度；因此4VG源非恶化与双乙酰、反式2壬烯醛一样只解释为分析尺度限制。4VG阈值比较是变换假设下的数值检查，不是源浓度或真实风味保证。stale_hops由第一问新代理预测，候选不得高于同一代理对源样品的预测。')
    sub('6.2 三个目标与求解')
    table('表9 最小化目标',['目标','定义','限定'],[['f1','50维标准化感官偏差平方的均值','与目标风格实测均值比较，包含代理误差'],['f2','相对源画像的标准化RMS改变量','改动代理，非财务或工艺成本'],['f3','四项缺陷指标相对源值比的均值','DMS浓度比加三项分析指标比，不是毒理风险']],[50,230,195])
    p('依据[7]实现非支配排序、目标空间拥挤距离、父子代精英保留、模拟二进制交叉和多项式变异。约束支配使可行解先于不可行解，不可行者按违例大小比较。种群80，每种种子42、43、44固定运行80代，合并三次可行解后重新计算非支配关系。固定代数结束不等于证明收敛或全球最优。')
    p('本次实际执行的是Python实现，未安装MATLAB，不能把这些数值称作MATLAB gamultiobj结果。算法先通过具有已知解析前沿的ZDT1收敛检查及可行性优先的构造样例，再逐项复核本题候选的原始缺陷指标与感官代理非恶化。')
    table('表10 三次求解记录',['随机种子','可行前沿数','停止原因'],[[r['seed'],r['n_feasible_front'],'80代预算结束'] for r in nsga['runs']],[110,110,255]);fig('q3_pareto.png','图3 三次潜空间NSGA-II合并后的近似Pareto集合')
    sub('6.3 偏好后置的决策分析')
    p('先保留Pareto集合，再按其各目标范围归一化，恒定目标不参与距离。采用加权TOPSIS比较到理想点与最差点的距离；均衡、感官优先、少改动和低缺陷场景的权重分别为等权、0.6/0.2/0.2、0.2/0.6/0.2和0.2/0.2/0.6。权重为演示偏好，尚不是团队正式决策。')
    table('表11 四种偏好下的候选选择',['偏好','前沿编号','f1','f2','f3'],[[r.scenario,r.front_id,f(r.f1),f(r.f2),f(r.f3)] for r in decisions.itertuples()],[135,70,90,90,90])
    p('不同偏好可以指向同一候选，也可以产生不同取舍；单一推荐不是客观唯一最优。推荐表需要连同化学画像、感官代理、源编号和证据等级一起审阅。严禁把没有满足代理约束的候选通过放宽条件伪装成合格；若没有可行前沿，程序停止并报告失败。')
    h('七 外部数据对照与验证边界')
    p(f'保留外部修订版的{external["rows"]}款评论汇总与三张词典工作表。外部key不是题目beer编号，未建立产品一一对应；消费者词频和专业感官z分数通过显式风格及九个粗粒度属性对齐，只能用于趋势一致性。本文不以外部数据调节第一问选型或迁移权重。')
    p('原外部评分关联实验按酒厂分五折验证，以避免同酒厂产品同时进入训练和测试。该部分未因本次文献修订重新训练，保留原结果与原版本标识，不把旧生成谱的外部一致性冒充新后验采样表现。评论词频与消费者评分来自同一评论环境，仍可能存在同源与高分样本选择偏差。')
    rate_rows=[]
    for name in ['Style_ABV','Style_ABV_Flavors']:
        if name in ratings:rate_rows.append([name,f(ratings[name]['r2']),f(ratings[name]['mae'])])
    table('表12 沿用原版的分酒厂消费者评分关联',['模型','折外R²','MAE'],rate_rows,[285,95,95])
    if (REV/'external_agreement.csv').exists():
        ea=pd.read_csv(REV/'external_agreement.csv')
        table('表13 本次生成代理与外部画像的风格级相关',['维度','实测对外部','新生成代理对外部'],[[r.dimension,f(r.official_rho),f(r.generated_rho)] for r in ea.itertuples()],[155,160,160])
        p('此处新生成画像由本次后验重参数化样品和新感官代理重新计算。相关系数是按显式对应风格聚合后的生态层面描述，不是产品级化学验证；没有对这九个维度进行选择性显著性报告。')
    h('八 局限与验证方案')
    sub('8.1 哪些改动已经验证')
    p('已验证内容包括键对齐、折内预处理、外层标签与内层选型隔离、半样本不重复、构造支配关系、解析测试前沿以及本题推荐约束。七篇来源逐条对应代码函数、数据表及论文章节，完整对照另附。程序运行成功不等于预测准确或真实风味改善。')
    sub('8.2 尚不能推出的结论')
    p('样本仍小且缺少新酒厂或新批次独立测试；重复嵌套验证只评价当前附件总体。半样本选择不满足原稳定性选择理论全部假设，不能给出FDR/PFER保证。近邻置换不是精确条件检验。固定VAE来自完整数据，其内部采样质量不能当成新的留出泛化。单一源样品和一个目标风格的Pareto结果不能外推到所有迁移对。')
    p('双乙酰和反式2壬烯醛的局部标定差异、重复底值机制、未知尺度及缺少工艺变量均限制物理解释。派生总量和非线性基质作用没有完全纳入约束；陈旧酒花约束依赖预测较弱的代理。当前结果适合筛选实验候选，不支持可酿造、可销售或盲评偏好已改善的声明。')
    sub('8.3 真实品评与工艺验证设计')
    p('参考[2]的加标对照思路，后续应先核实化合物单位、批次和完整浓度，选择有限的Pareto候选，设置同批次未加标对照；随机化顺序、盲法编码、暗色容器及一致开瓶操作，记录逐位品评员原始0至6评分、方向差异与偏好。该段是待执行实验方案，不列为本研究已完成实验。')
    p('对候选的工艺可实现性，应补充原料、发酵条件、组分耦合、总量关系与干预成本，先完成小规模加标或试酿，再对真实感官和缺陷响应作独立检验。若实测与代理冲突，应更新代理与约束，而不是只保留符合模型预期的样品。')
    h('九 结论')
    p(f'本研究以七篇来源为依据，对外部数据修订版的验证、解释与决策流程进行了可复现修改。第一问完整嵌套选型的折外R²为{f(v1["r2"])}；第二问以条件置换和半样本频率揭示名单敏感性；第三问分别报告固定cVAE采样、条件高斯留出分布以及潜空间Pareto搜索，不混同三者的验证层次。')
    p('论文的可交付价值是可追溯的模型比较、证据分级和候选选择，而不是未经品评证实的产品改良。后续应优先取得独立化学感官配对样本、原始评分与标定数据，再扩大迁移对象和工艺变量范围。')
    h('参考文献')
    for index,r in enumerate(refs,1):
        bibli=f'[{index}] {r["authors"]}. {r["title"]}. {r.get("publication",r.get("location",""))}, {r["year"]}.'
        if r.get('doi'):bibli+=' DOI: '+r['doi']+'.'
        elif r.get('url'):bibli+=' '+r['url']+'.'
        p(bibli)
    p('[8] 2026年MathorCup大数据竞赛赛道A题面及附件1至8. 用户提供的本地材料, 2026.')
    p('[9] sp1222 / stephenpolozoff. Beer Information Tasting Profiles. Kaggle. https://www.kaggle.com/datasets/stephenpolozoff/top-beer-information . 外部文件由用户提供，平台资料访问日期2026-10-28.')
    h('附录 复现与逐篇修改记录')
    p('执行code/literature_revision.py --stage all --subsamples 100生成独立结果；执行code/build_literature_paper.py生成本论文及Markdown；执行code/validate_literature_revision.py复核数据、模型和推荐。不覆盖原results/下的实验表，新增结果位于results/literature_revision/，输入及cVAE快照哈希保存于audit_summary.json。')
    table('附表A 七篇来源对应的实际改动',['文献','论文章节','代码与验证'],[[r['id'],'、'.join(r['paper_sections']),r['change']] for r in refs],[45,130,300])
    def footer(c,doc):c.setFont('Helvetica',8);c.drawCentredString(A4[0]/2,25,str(doc.page))
    pdf=OUTDOC/'啤酒风味论文_七篇文献优化版.pdf'
    SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=60,rightMargin=60,topMargin=45,bottomMargin=45,
                      title='啤酒感官预测与受约束风格迁移',author='铱豪').build(story,onFirstPage=footer,onLaterPages=footer)
    (OUTDOC/'论文正文_七篇文献优化版.md').write_text('\n\n'.join(md),encoding='utf-8')
    ledger=['# 七篇文献的具体内容与修改对照','日期 2026-10-29；操作者 铱豪；本表区分直接采用、近似实现和待验证内容。']
    for r in refs:
        ledger += ['## '+r['id']+' '+r['title'], '**来源**：'+r['authors']+'；'+str(r['year'])+'；'+r.get('url',r.get('location','用户本地稿件')),
                   '**具体文献内容与位置**：'+r['evidence'], '**修改了什么**：'+r['change'],
                   '**论文章节**：'+'；'.join(r['paper_sections']), '**对应代码**：'+'；'.join(r['code'])]
        if r['id'] in ['U1','U2','L1']:ledger+=['**验证证据**：results/literature_revision/scale_evidence.csv、q1_nested_choices.json、q1_nested_metrics.csv。']
        if r['id'] in ['U1','L2','L3']:ledger+=['**验证证据**：q2_conditional_importance.csv、q2_top3.csv、q2_selector_comparison.csv；频率不能冒充因果概率或FDR保证。']
        if r['id'] in ['U1','U2','L4','L5']:ledger+=['**验证证据**：q3_sampler_diagnostics.csv、q3_heldout_distribution.csv、q3_nsga2_summary.json、q3_decisions.csv；均在results/literature_revision/。']
    ledger += ['## 与原版比较','原版输出保留，新结果另存。第一问使用同划分固定随机森林对照；第二问保留原三证据名单并报告新诊断选择器的Jaccard；第三问复用原cVAE，只把重新运行的采样与NSGA-II标为新结果。',
               'MATLAB未安装，本轮NSGA-II实际运行于Python。未完成真实盲评或试酿，不作生产保证。']
    (OUTDOC/'逐篇文献内容与修改对照.md').write_text('\n\n'.join(ledger),encoding='utf-8')
    print(pdf)


if __name__=='__main__':main()

```

## format_ryh1_template.py

```python

"""Format existing scientific content by cloning the retained MathorCup template.
No model/data changes. Process appendix is deliberately excluded.
"""
from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile,ZIP_DEFLATED
import re,json,hashlib
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Mm,Pt

BASE=Path(__file__).resolve().parents[1]
TEMPLATE=BASE.parents[1]/'资料/Mathorcup论文标准模板(1).docx'
SOURCE=BASE/'output/literature_revision/论文正文_七篇文献优化版.md'
DEST=BASE/'output/literature_revision/RYH优化1.docx'


def parse():
    parts=[p.strip() for p in SOURCE.read_text(encoding='utf-8').split('\n\n') if p.strip()]
    sections={};name='preamble';sections[name]=[]
    for block in parts:
        if block.startswith('## '):
            name=block[3:];sections[name]=[]
        elif block.startswith('|') and sections[name] and sections[name][-1].startswith('|'):
            sections[name][-1]+='\n'+block
        else:sections[name].append(block)
    return sections


def main():
    original_hash=hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
    ref=Document(TEMPLATE);doc=Document(TEMPLATE)
    samples={k:deepcopy(ref.paragraphs[i]._p.pPr) for k,i in [('title',2),('abstract',3),('body',5),('keyword',26),('h1',29),('h2',30),('h3',54),('reference',109)]}
    sections=parse();body=doc.element.body
    table_sequence=[1]
    for child in list(body):body.remove(child)
    def no_number(p):
        n=p._p.get_or_add_pPr().get_or_add_numPr();i=n.get_or_add_numId();i.val=0
    def para(text,role='body',style=None):
        p=doc.add_paragraph(style=style or ('Normal' if role in ['title','abstract','body','keyword'] else {'h1':'Heading 1','h2':'Heading 2','h3':'Heading 3','reference':'参考文献'}.get(role,'Normal')))
        if samples.get(role) is not None:
            if p._p.pPr is not None:p._p.remove(p._p.pPr)
            p._p.insert(0,deepcopy(samples[role]))
        p.add_run(text)
        if role in ['h1','h2','h3']:
            p.paragraph_format.keep_with_next=True
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT
        if role=='title':p.runs[0].font.size=Pt(14)
        if role=='keyword':p.alignment=WD_ALIGN_PARAGRAPH.LEFT
        return p
    def heading(text,level=1):
        p=para(text,'h'+str(level))
        if level==1:p.paragraph_format.first_line_indent=Pt(0)
        return p
    def table(block):
        lines=[ln for ln in block.splitlines() if ln.strip().startswith('|')]
        rows=[[v.strip() for v in line.strip().strip('|').split('|')] for line in lines]
        rows=[r for r in rows if not all(re.fullmatch(r':?-+:?',c.strip()) for c in r)]
        n=len(rows[0]);t=doc.add_table(rows=len(rows),cols=n);t.style='三线表'
        # Template's 80% symbol-table width expands to usable page width for longer result tables.
        t.autofit=False; widths=[160/n]*n
        if n==2:widths=[34,126] if rows[0][0] in ['属性','维度'] else [105,55]
        if n==3:widths=[22,72,66] if rows[0][0] in ['目标','符号'] else ([25,55,80] if rows[0][0]=='来源' else [65,45,50])
        if n==4:widths=[55,25,25,55] if rows[0][0]=='字段' else [60,33,33,34]
        if n==5:widths=[50,25,28,28,29]
        for i,row in enumerate(rows):
            for j,value in enumerate(row):
                cell=t.cell(i,j);cell.width=Mm(widths[j]);p=cell.paragraphs[0]
                p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(0)
                indent=p._p.get_or_add_pPr().get_or_add_ind();indent.set(qn('w:firstLineChars'),'0');indent.set(qn('w:firstLine'),'0')
                p.alignment=WD_ALIGN_PARAGRAPH.CENTER if i==0 or len(value)<15 else WD_ALIGN_PARAGRAPH.LEFT
                p.add_run(value)
                suppress=OxmlElement('w:suppressAutoHyphens');p._p.get_or_add_pPr().append(suppress)
            if i==0:
                repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
            avoid=OxmlElement('w:cantSplit');t.rows[i]._tr.get_or_add_trPr().append(avoid)
        for c,w in zip(t.columns,widths):c.width=Mm(w)
    def blocks(items,sublevel=3,strip_subheads=False):
        for b in items:
            if b.startswith('### '):
                if not strip_subheads:heading(re.sub(r'^\d+\.\d+\s*','',b[4:]),sublevel)
            elif b.startswith('|'):table(b)
            elif b.startswith('!['):
                match=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',b)
                if not match:raise ValueError(b)
                caption,path=match.groups();path=(SOURCE.parent/path).resolve()
                p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Pt(0)
                p.add_run().add_picture(str(path),width=Mm(145))
                cp=doc.add_paragraph(caption,style='图注');no_number(cp);cp.paragraph_format.keep_with_next=False
            elif re.match(r'^表\d+',b):
                table_sequence[0]+=1
                b=re.sub(r'^表\d+',f'表{table_sequence[0]}',b)
                p=doc.add_paragraph(b,style='表注');no_number(p);p.paragraph_format.keep_with_next=True
            else:para(b)
    def subsection(items,label):
        out=[];inside=False
        for b in items:
            if b.startswith('### '):
                if inside:break
                inside=label in b
            elif inside:out.append(b)
        return out
    def before_results(items):
        idx=next((i for i,b in enumerate(items) if b.startswith('|') or b.startswith('表') or b.startswith('![')),len(items))
        # Include a table's immediately preceding caption in the results group.
        return items[:idx],items[idx:]
    # Source-derived cover: keep table and two spacer paragraphs.
    body.append(deepcopy(ref.tables[0]._tbl))
    doc.tables[0].cell(1,1).text='A'
    body.append(deepcopy(ref.paragraphs[0]._p));body.append(deepcopy(ref.paragraphs[1]._p))
    para('啤酒感官预测与受约束风格迁移','title');para('摘 要','abstract')
    summary=sections['摘要']
    for b in summary:
        if b.startswith('关键词'):
            p=para('关键词：嵌套验证；条件重要性；稳定性选择；条件生成；Pareto决策','keyword')
        else:
            p=para(b)
            if b.startswith(('针对问题一','针对问题二','针对问题三')):
                # Template emphasizes each problem lead-in in the abstract.
                text=p.text;p.clear();p.add_run(text[:6]).bold=True;p.add_run(text[6:])
    body.append(deepcopy(ref.paragraphs[27]._p))
    # Keep original two-level/three-level TOC style via an updatable TOC field.
    p=doc.add_paragraph('目录');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Pt(0)
    p.runs[0].font.name='黑体';p.runs[0].font.size=Pt(16)
    p=doc.add_paragraph();p.paragraph_format.first_line_indent=Pt(0)
    for kind in ['begin','instr','separate','text','end']:
        run=OxmlElement('w:r')
        if kind=='instr':
            node=OxmlElement('w:instrText');node.set(qn('xml:space'),'preserve');node.text=' TOC \\o "1-3" \\h \\z \\u '
        elif kind=='text':node=OxmlElement('w:t');node.text='目录将在Word中更新'
        else:node=OxmlElement('w:fldChar');node.set(qn('w:fldCharType'),kind)
        run.append(node);p._p.append(run)
    body.append(deepcopy(ref.paragraphs[28]._p))
    heading('问题重述');heading('问题背景',2)
    blocks(sections['一 问题与研究范围'])
    heading('问题提出',2)
    for text in ['问题一：利用化学与理化测量预测50项感官响应，并在评分锚点不足时提供明确标注的相对等级。',
                 '问题二：结合共表达网络筛选各感官属性的Top-3单项化学候选，并检验相关变量归因及名单稳定性。',
                 '问题三：在风格条件下生成化学画像，实现受约束跨风格迁移，并通过Pareto集合分析多目标取舍。']:para(text)
    heading('问题分析')
    for text in ['问题一的主要困难是样本量小、输入相关且量纲不统一。模型选择应隔离于最终误差评价，输出相对等级时不能把空白评分卡当作绝对评分标签。',
                 '问题二需要区分预测关联、名单稳定性和因果作用。相关化合物可能互相替代，因此除固定Top-3外，还需要条件置换与子抽样诊断。',
                 '问题三同时涉及生成分布、源样品改变和缺陷约束。第一问的代理误差会传递到迁移评价，故将尺度证据、约束可行性与真实品评效果分开讨论。']:para(text)
    heading('模型假设与符号说明');heading('模型基本假设',2)
    for text in ['同一beer编号代表同一款测量样品，化学与感官数据按该键对齐；风格标签一致性由输入数据核对。',
                 '训练与验证划分评价当前附件总体内的插值能力，不预设其能够外推到新酒厂、新批次或未见风格。',
                 '秩正态变换、近邻条件置换及风格收缩分布是统计建模假设，不赋予未知化学列确定的物理含义。',
                 '感官代理及Pareto候选用于筛选后续实验对象，不等同于已经完成真实加标、盲评或酿造验证。']:para(text)
    heading('符号说明',2)
    no_number(doc.add_paragraph('表1 符号说明',style='表注'))
    table('| 符号 | 含义 | 单位或范围 |\n| --- | --- | --- |\n| X | 化学与理化输入矩阵 | 250×231，附件分析尺度 |\n| Y | 感官响应矩阵 | 250×50，标准化值 |\n| s | 风格标签 | 22类 |\n| z | 条件VAE潜变量 | 8维，无量纲 |\n| pi | 子样本入选频率 | 0至1 |\n| f1 f2 f3 | 感官偏差、改动与缺陷指标 | 均取最小值 |')
    heading('数据预处理');heading('指标选取',2)
    data=sections['二 数据尺度与证据']
    blocks(subsection(data,'2.1'))
    heading('数据清洗',2);blocks(subsection(data,'2.2'));blocks(subsection(data,'2.3'))
    heading('模型建立与求解')
    heading('问题一模型建立与求解',2);heading('问题一求解思路',3)
    para('先在训练折内建立缺失与底值状态，再比较稳健线性和树模型候选；用内层选型、外层评价控制选择偏差，最后给出连续感官预测及相对等级。')
    one=sections['三 问题一 感官预测与验证隔离'];models,results=before_results(subsection(one,'3.1'))
    heading('问题一模型建立',3);blocks(models)
    heading('问题一模型求解与分析',3);blocks(results);blocks(subsection(one,'3.2'))
    heading('问题二模型建立与求解',2);heading('问题二求解思路',3)
    two=sections['四 问题二 相关化合物与稳定性'];blocks(subsection(two,'4.1'))
    model2,res2=before_results(subsection(two,'4.2'));stab=subsection(two,'4.3')
    heading('问题二模型建立',3);blocks(model2);blocks(stab[:1])
    heading('问题二模型求解与分析',3);blocks(res2);blocks(stab[1:])
    heading('问题三模型建立与求解',2);heading('问题三求解思路',3)
    para('对固定条件VAE的采样方式进行诊断，并用独立分层留出检验比较条件收缩高斯和经验基准；在潜空间中用NSGA-II搜索可行候选，再进行偏好后置决策。')
    three=sections['五 问题三 条件生成与分布检验'];multi=sections['六 多目标迁移与Pareto决策']
    heading('问题三模型建立',3);blocks(subsection(three,'5.1'))
    m3,r3=before_results(subsection(three,'5.2'));blocks(m3)
    m4,r4=before_results(subsection(three,'5.3'));blocks(m4)
    blocks(subsection(multi,'6.1'))
    heading('问题三模型求解与分析',3);blocks(r3);blocks(r4);blocks(subsection(multi,'6.2'))
    heading('模型分析检验');heading('灵敏度分析',2);blocks(subsection(multi,'6.3'))
    heading('误差分析',2);blocks(sections['七 外部数据对照与验证边界'])
    heading('模型评价与推广');heading('模型的优点',2)
    para('模型将专业评分、评论词频及化学尺度分别处理，使用重复嵌套验证评价完整选型程序，公开逐属性误差、条件重要性和子样本稳定性。约束迁移保留多目标取舍，使不同偏好可以对应不同候选，而不以单一总分掩盖差异。')
    heading('模型的不足',2)
    blocks(subsection(sections['八 局限与验证方案'],'8.2'))
    heading('模型的推广',2);blocks(subsection(sections['八 局限与验证方案'],'8.3'));blocks(sections['九 结论'])
    body.append(deepcopy(ref.paragraphs[96]._p))
    p=heading('参考文献');no_number(p)
    for b in sections['参考文献']:
        p=para(b,'reference');no_number(p);p.paragraph_format.first_line_indent=Pt(0)
    body.append(deepcopy(ref.sections[3]._sectPr))
    # Keep the first three section breaks and references section; no process-only appendix.
    doc.core_properties.title='RYH优化1'
    doc.core_properties.author=''
    settings=doc.settings.element
    update=settings.find(qn('w:updateFields'))
    if update is None:update=OxmlElement('w:updateFields');settings.append(update)
    update.set(qn('w:val'),'true')
    doc.save(DEST)
    # Restore opaque template parts byte-for-byte rather than normalizing their XML.
    with ZipFile(TEMPLATE) as z:original={n:z.read(n) for n in z.namelist()}
    with ZipFile(DEST) as z:new={n:z.read(n) for n in z.namelist()}
    preserve=[n for n in original if n in ['word/styles.xml','word/numbering.xml','word/theme/theme1.xml'] or n.startswith('word/header') or n.startswith('word/footer')]
    for n in preserve:new[n]=original[n]
    with ZipFile(DEST,'w',ZIP_DEFLATED) as z:
        for n,data in new.items():z.writestr(n,data)
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()==original_hash
    checks={'template_sha256':original_hash,'source':str(SOURCE),'output':str(DEST),
            'sections':len(Document(DEST).sections),'keyword_alignment':'left',
            'appendix_process_removed':True,'preserved_template_parts':preserve}
    (BASE/'results/literature_revision/RYH优化1_格式核对.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(DEST)


if __name__=='__main__':main()

```

## build_paper.py

```python

"""Build the Chinese paper directly from actual experiment outputs (PDF + text).
Uses ReportLab; no manually entered experimental results. Windows CJK fonts are
preferred, otherwise supply BEER_CJK_FONT pointing to a TrueType Chinese font.
"""
from common import *
import html,platform,importlib.metadata
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def main():
    out=ROOT/'output/pdf';out.mkdir(parents=True,exist_ok=True)
    font=Path(os.environ.get('BEER_CJK_FONT',r'C:\Windows\Fonts\simsun.ttc'))
    if not font.exists():raise FileNotFoundError('Set BEER_CJK_FONT to a Chinese TrueType font')
    pdfmetrics.registerFont(TTFont('CJK',str(font),subfontIndex=0))
    bold=Path(r'C:\Windows\Fonts\simhei.ttf');pdfmetrics.registerFont(TTFont('CJKbold',str(bold if bold.exists() else font)))
    styles={
      'body':ParagraphStyle('body',fontName='CJK',fontSize=10.5,leading=18,spaceAfter=7,firstLineIndent=21,wordWrap='CJK',alignment=TA_JUSTIFY),
      'h1':ParagraphStyle('h1',fontName='CJKbold',fontSize=15,leading=22,spaceBefore=16,spaceAfter=10,keepWithNext=True),
      'h2':ParagraphStyle('h2',fontName='CJKbold',fontSize=12,leading=19,spaceBefore=12,spaceAfter=7,keepWithNext=True),
      'title':ParagraphStyle('title',fontName='CJKbold',fontSize=22,leading=34,alignment=TA_CENTER,spaceAfter=18),
      'caption':ParagraphStyle('caption',fontName='CJK',fontSize=9,leading=14,alignment=TA_CENTER,spaceAfter=8),
      'tablecaption':ParagraphStyle('tablecaption',fontName='CJK',fontSize=9,leading=14,alignment=TA_CENTER,spaceAfter=8,keepWithNext=True),
      'cell':ParagraphStyle('cell',fontName='CJK',fontSize=8.5,leading=12,wordWrap='CJK'),
      'eq':ParagraphStyle('eq',fontName='CJK',fontSize=10,leading=17,alignment=TA_CENTER,spaceBefore=6,spaceAfter=9),
    }
    story=[];text=[]
    def p(t,style='body'):
        text.append(t);story.append(Paragraph(html.escape(str(t)),styles[style]))
    def h(t):p(t,'h1')
    def sub(t):p(t,'h2')
    def table(title,headers,rows,widths=None):
        p(title,'tablecaption');data=[[Paragraph(html.escape(str(v)),styles['cell']) for v in row] for row in [headers]+rows]
        tab=Table(data,colWidths=widths or [475/len(headers)]*len(headers),repeatRows=1,hAlign='CENTER')
        tab.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.8,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),.8,colors.black),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf2f4')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(tab);story.append(Spacer(1,9));text.extend([' | '.join(map(str,r)) for r in [headers]+rows])
    def fig(name,caption):
        from PIL import Image as PI
        path=FIG/name;im=PI.open(path);w,h0=im.size;story.append(KeepTogether([Image(str(path),width=475,height=475*h0/w),Paragraph(html.escape(caption),styles['caption'])]));text.append(caption)
    def f(x,n=3):return '未定义' if x is None or pd.isna(x) else f'{float(x):.{n}f}'
    def js(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
    q1=js('q1_summary.json');q2=js('q2_summary.json');q3=js('q3_summary.json');ex=js('external_audit.json');rating=js('external_rating_summary.json');audit=js('data_audit.json')
    folds=pd.read_csv(OUT/'q1_cv_folds.csv');grades=pd.read_csv(OUT/'q1_grade_sensitivity.csv');top=pd.read_csv(OUT/'q2_top3.csv');agreement=pd.read_csv(OUT/'external_agreement.csv');transfer=pd.read_csv(OUT/'q3_transfer_checks.csv');best=q1['selected_model'];bestscore=q1['scores'][best]
    # Cover contains no invented team number or person name.
    story.append(Spacer(1,70));p('基于化学测量与外部风味描述的\n啤酒感官预测及风格生成研究'.replace('\n',' '),'title');p('2026 年 MathorCup 数学应用挑战赛 大数据竞赛赛道 A','caption');story.append(Spacer(1,24));p('摘要','h1')
    p('啤酒化学组成、专业品评与消费者语言描述之间存在尺度和采样机制差异。本文以赛题提供的250款啤酒为主数据，构建感官预测、关键因素识别及风格条件生成的递进框架，并引入5,558款外部啤酒的评论风味特征进行独立的风格趋势对照。外部数据不具备完整化合物浓度，因此不与主数据直接拼接训练。')
    p(f'针对问题一，采用训练折内缺失插补与三次三折风格分层验证，比较均值、Ridge、随机森林和极端随机树。{best}的折均R²为{f(bestscore["r2"])}，MAE为{f(bestscore["mae"])}。通过训练折内极差和分位数映射输出0至6级结果，并明确区分假设分级与真实原始评分。')
    p(f'针对问题二，以附件相关矩阵构建{q2["nodes"]}节点、{q2["edges"]}边的共表达图，融合风格控制偏相关、Ridge系数和极端随机树重要性，在社区多样性约束下提取Top-3。{q2["bootstrap_repeats"]}次分层有放回重采样所得Top-1与Top-3平均入选频率分别为{f(q2["top1_frequency"])}、{f(q2["top3_frequency"])}，反映候选因素的稳定性差异。')
    p(f'针对问题三，在给定数据尺度标准化后训练条件变分自编码器，生成{q3["generated_n"]}条化合物谱；通过浓度阈值投影和陈旧酒花代理约束完成{q3["transfer_n"]}次迁移尝试，出现{q3["identity_fallback_n"]}次原样回退，非恶化违例计数为{q3["nonworsening_violations"]}。外部分析建立{q3["external"]["mapped_style_count"]}类风格与9个粗粒度感官维度的对应关系，检验风格趋势的一致与偏离。消费者评分模型采用酒厂分组验证，避免同酒厂产品同时进入训练和测试。')
    p('模型将化合物实测、品评标签和评论词频的证据边界分别处理。结果可用于候选因素筛选和计算机辅助探索；原始评分校准、独立化学感官配对数据及酿造盲评仍是提高预测可信度和实际应用能力的关键。')
    p('关键词  感官预测  共表达图  条件变分自编码器  风格迁移  外部验证','caption');story.append(PageBreak())
    h('一 问题重述')
    p('赛题围绕啤酒酿造化合物与风味形成提出三个任务：由存在缺失的化学数据预测Aroma和Flavour评分；依据化合物相关网络识别每个感官维度的三个关键因素并检验稳定性；以啤酒风格为条件生成化合物谱、给出感官画像并实现缺陷不恶化的跨风格迁移[1]。')
    p('参考研究通过化学测量、训练品评组及消费者评论建立啤酒风味预测模型，为本题提供方法与数据背景[2]。本文继承该研究及已有项目中的树模型、网络结构和条件生成思路，对验证划分、重采样、约束结果及外部证据的表达进行重新实现。')
    h('二 问题分析')
    sub('2.1 三个问题的递进关系')
    p('第一问建立化学向量到感官画像的预测函数，其输出是第三问的代理评价；第二问从相关结构和监督证据中给出可解释候选因素；第三问在风格条件下生成向量，并检查化学约束和代理画像。第一问误差会传播到第三问，因此代理评价不能代替独立品评。第二问的关联不直接成为可操作的因果配方。')
    sub('2.2 外部数据在研究中的位置')
    p('Kaggle数据来自BeerAdvocate评论汇总，主要由高评分啤酒构成[3]。其风味列是描述词频，既不是化学浓度，也不是0至6专业评分。本文将它用于风格级画像、描述词词典对照和消费者评分关联分析，并保持其独立于前三问主模型训练和参数选择。')
    p('若外部画像与主数据不一致，可能来源于产品构成、风格分类、语言表达和采样偏差，不能直接判断某一方错误。跨域对照只检验粗粒度方向与相对排序，不主张数值尺度相等。')
    h('三 模型假设与符号说明')
    table('表1 主要符号',['符号','含义','维度或尺度'],[['X','化学及理化特征','250 × 231，沿用题给尺度'],['Y','专业品评结果','250 × 50，标准化连续值'],['s','啤酒风格标签','22类'],['g(Y)','假设的有序等级映射','0至6整数'],['G','化合物共表达图','无向加权图'],['z','条件VAE潜变量','8维'],['p外部','11维风味词频份额','每行归一化后和为1'],['T','附件参考阈值','保留各化合物单位']],[60,205,210])
    p('假设一：附件中相同beer编号指向同一款测量样品，按该标识对齐化学与感官数据。假设二：在缺少原始品评员标准化参数时，0至6级只能采用单调映射近似，不能唯一还原。假设三：共表达社区可帮助分散高度共线候选，但不保证社区内变量可相互替代。')
    p('假设四：风格标签能够提供条件生成的统计信息，但小样本风格的分布估计不稳定。假设五：缺陷参考阈值适用于本题的约束演示，阈值属于风味参考而非法定安全限值。假设六：显式匹配的风格与粗粒度词义可作趋势比较，其差异需要通过敏感性分析体现。')
    h('四 数据预处理')
    sub('4.1 数据规模与来源审计')
    table('表2 输入数据与用途',['数据','规模','用途'],[['附件1.csv',f'{audit["n"]}行，{audit["chemical_features"]}个特征','三问化学与理化输入'],['附件2.csv','231 × 231','化合物网络结构先验'],['附件3.xlsx','品评卡及汇总表','维度定义与分级刻度'],['附件4.csv','250 × 50个感官目标','第一问标签与第三问代理'],['附件5和6.csv','感官及化学感官相关摘要','同源背景核查，不作独立验证'],['附件7.csv','250个样品，186个描述符','消费者描述词语义对照'],['附件8.xlsx','校准浓度、阈值及参考模型表','单位核验、来源与阈值注释'],['外部CSV',f'{ex["rows"]}行，{ex["styles"]}类风格，21列','外部画像及评分关联'],['外部Excel','3个词典工作表','描述词归类']],[105,150,220])
    p(f'化学矩阵共有{audit["missing_chemical_cells"]}个缺失单元，感官矩阵有{audit["missing_sensory_cells"]}个缺失单元。化学表beer_id唯一，但感官表存在精度损失造成的重复；本版以唯一beer编号重排并核对风格标签。输入文件的字节数和SHA256写入input_manifest.csv，避免因文件版本不同导致结果无法复现。')
    p(f'外部数据有{ex["breweries"]}个酒厂，key无重复；缺失名称有{ex["missing"].get("Name",0)}条，11项风味词频总和为零的样本有{ex["zero_flavor_totals"]}条。名称缺失不影响按风格汇总，但不能用于产品身份匹配。外部key与题目beer_id含义不同，本文未假定样本一一对应。')
    sub('4.2 缺失与数值尺度')
    p('回归验证中，插补器仅在训练折拟合中位数，并添加缺失指示列；Ridge额外执行训练折标准化。树模型沿用输入尺度，不对未经确认的化学列强行反变换。生成模型同样在题给尺度逐列标准化，既避免将负数直接当作负浓度，也不宣称全部231列已经获得统一的物理量纲。')
    p('缺陷约束只对经校准表核验的列转换浓度：DMS和4VG采用log10还原，二乙酰与反式2壬烯醛保留原始浓度。约束后按原列尺度写回。其他列的经验上下界仅作为防止极端外推的统计限制，不代替质量守恒或工艺可行性。')
    calibration=pd.read_csv(OUT/'q3_calibration_checks.csv')
    p('按beer编号逐元素核对校准浓度，DMS和4VG的正浓度条目与log10转换一致；二乙酰及反式2壬烯醛的大多数条目支持原始浓度口径，但仍有少量数值差异。模型沿用题目1.csv，不静默覆盖这些观测；完整差异统计见q3_calibration_checks.csv。')
    table('表2a 缺陷列校准一致性',['化学字段','可比条目','差异超过1e-6','最大绝对差异'],[[r.compound,r.n_compared,r.n_discrepancy_above_1e_6,f(r.max_abs_difference,6)] for r in calibration.itertuples()],[180,85,105,105])
    sub('4.3 外部词频标准化')
    p('对每款外部啤酒，将11项词频除以该行词频总和，得到组成份额，以减少总描述量差异。该变换仍不能完全校正评论数、文本长度、语言或词典重叠；增加某项份额也会机械改变其他份额，因此不作独立因果解释。')
    p('132条风味词频总和为零的记录无法计算份额，风格画像统计将其排除；消费者评分模型则用全零向量表示未获得风味信息并保留样本。该向量不能解释为完全没有风味，未来取得原始评论数后应进一步区分无评论与无词典命中。')
    p('p(i,k) = count(i,k) / sum_j count(i,j)                                      （1）','eq')
    p('跨域比较时，分别在专业感官数据和外部份额数据中按维度计算经验百分位，再按风格汇总。生成样品的百分位以真实专业感官样本为参考，不在生成样本自身重新排序。比较指标包括跨风格Spearman相关及平均绝对百分位差。')
    h('五 问题一的模型建立与求解')
    sub('5.1 求解思路与模型')
    p('以50维标准化感官结果为联合回归目标，比较均值预测、Ridge、随机森林与Extra Trees[4,5]。均值预测检验模型是否优于不使用化学信息的基准。两个树模型均为300棵树，随机森林叶节点最小样本数为3，Extra Trees为2，特征子采样采用平方根；Ridge正则强度固定为100。参数预先固定，避免本轮对外部数据调参。')
    p('最小风格只有3个样品，因此采用3折分层而非5折。重复种子42、1024和2024，共9个验证折。插补及线性标准化在每个训练折独立拟合。报告各折平均与标准差，标准差仅反映划分波动，不是独立样本的置信区间。')
    p('R² = 1 - sum_i (y_i - yhat_i)² / sum_i (y_i - mean(y))²                     （2）','eq')
    p('综合R²为各目标R²的均值，MAE为标准化目标的平均绝对误差。候选模型按本轮平均R²选择并在全部样品上重训；同一交叉验证用于选择和报告，存在选择乐观性，不能等同于独立外部测试。')
    sub('5.2 有序等级的假设映射')
    p('主口径将训练折每个目标的最小值、最大值映射到0和6，验证预测值先截断到区间再取整；敏感性口径使用训练折经验分位数切成7级。所有边界仅用训练折估计，避免验证标签参与边界确定。')
    p('g(y) = round[6 × clip((y - min_train)/(max_train - min_train), 0, 1)]      （3）','eq')
    p('分级精确率、相差不超过一级的比例和加权Kappa均相对于同一映射得到的标签计算，不能证明恢复真实0至6评分。导出全部50目标预测与品评卡维度映射；没有对应标签的乙醛维度标为不可评分，不以零值填充。')
    sub('5.3 求解结果与误差分析')
    rows=[]
    for name,g in folds.groupby('model'):
        rows.append([name,f'{f(g.r2.mean())} ± {f(g.r2.std())}',f(g.mae.mean()),f(g.spearman.mean())])
    table('表3 问题一三次三折验证结果',['模型','R² 均值±标准差','MAE','Spearman'],rows,[115,145,100,115]);fig('q1_models.png','图1 不同回归模型的验证R²及折间波动')
    per=pd.read_csv(OUT/'q1_per_target.csv');p(f'选择的模型为{best}。逐目标结果中有{int((per.r2>0).sum())}/{len(per)}个维度的平均重复OOF预测R²大于零。不同目标难度不一致，因此总平均不能代表每个香气维度均可可靠预测；完整逐目标结果随代码交付。')
    gg=grades[grades.model==best].groupby('mapping')[['exact','within1','kappa']].mean()
    table('表4 选定模型的假设分级敏感性',['映射','精确一致率','相差至多一级','加权Kappa'],[[i,f(r.exact),f(r.within1),f(r.kappa)] for i,r in gg.iterrows()])
    p('发布的0至6答案表由全数据重训模型生成，是已知样本的拟合答案；另提供折外答案表用于评估。两者用途不同，不用全数据拟合误差替代泛化误差。')
    h('六 问题二的模型建立与求解')
    sub('6.1 共表达图与候选证据')
    p('附件2.csv中的标签经已有字段映射表对齐化学列，建立无向加权图。若两化合物相关系数绝对值不低于0.5则连边，边权为绝对相关系数。采用贪心模块度社区划分，各感官目标的三个候选尽量来自不同社区。图来自全样本摘要，作为固定结构先验使用。')
    p('监督证据包含三类：先对化学和感官变量分别回归风格哑变量，再计算残差相关；标准化化学特征上的Ridge系数绝对值；逐感官目标拟合的Extra Trees特征重要性。每类证据转为231个特征内的百分位后等权平均。树重要性用于候选排序，不能解释为因果效应大小。')
    p('Score(c,t) = [rank(|partial(c,t)|) + rank(|ridge(c,t)|) + rank(tree(c,t))] / 3  （4）','eq')
    p('附件6.csv由同源样本产生，不作为新增独立证据参与评分，避免将已有相关性重复计权。按综合分数降序依次选取不同社区候选，每个目标输出三个因素，并保留原始偏相关方向、来源类别、阈值与单位。')
    sub('6.2 稳定性估计')
    p(f'本版在每个风格内部有放回抽取与原风格同数量的样本，重复{q2["bootstrap_repeats"]}次；每次重新插补并计算三种监督证据，然后重新选择Top-3。此过程是真正的分层Bootstrap，而不是将重复交叉验证折计为Bootstrap。固定相关图不随抽样变化，因此频率表示条件于当前图的选择稳定性。')
    p('f(c,t) = 被选入目标t的Top-3的次数 / Bootstrap次数                         （5）','eq')
    p(f'在当前重复次数下，二项近似的最大Monte Carlo标准误为{f(q2["maximum_mc_se"])}。该误差仅衡量有限重采样次数造成的数值波动；若要精确区分0.5与0.6的入选频率，应增加到200次或更多。')
    sub('6.3 网络与关键因素结果')
    table('表5 网络与稳定性统计',['指标','结果'],[['节点 / 边 / 社区',f'{q2["nodes"]} / {q2["edges"]} / {q2["modules"]}'],['Top-1平均入选频率',f(q2['top1_frequency'])],['Top-3平均入选频率',f(q2['top3_frequency'])],['三个候选均达到0.8的目标数',str(q2['all3_frequency_at_least_0_8'])],['三种方法平均Top-3 Jaccard',f(q2['mean_method_jaccard'])],['具有来源类别注释的候选数',str(q2['source_annotated'])+' / 150']],[285,190])
    selected=['acidity','bitternes','sweetness','alcohol','X4vg','dms','overall'];r=[]
    for target in selected:
        t=top[top.target==target].sort_values('rank');r.append([target,'；'.join(f'{a.compound} ({a.bootstrap_frequency:.2f})' for a in t.itertuples())])
    table('表6 代表性感官维度的Top-3及入选频率',['维度','候选化合物与频率'],r,[85,390]);fig('q2_stability.png','图2 Top-3入选频率分布及网络阈值和证据消融敏感性')
    p('化学合理性讨论分成两个层次：来源注释说明候选与酵母、麦芽或酒花的已知类别关系；相关方向与阈值帮助检查明显不合理的解释。来源匹配只能提供解释线索，不是机制验证。同名缺陷化合物若未进入Top-3，可能与浓度范围、共线性或测量噪声有关，不能据此否认其感官作用。')
    p('当第二、第三名频率较低或方法间Jaccard较低时，论文保留不稳定标记，推荐把候选看作实验筛选集合。若需要判断单一化合物的真实贡献，应进行控制其他条件的加标或干预实验。')
    h('七 问题三的模型建立与求解')
    sub('7.1 条件生成模型')
    p('以22类风格的独热向量作为条件，使用条件VAE[6]。编码器输入231维标准化化学向量与风格条件，隐藏层为128和64维，输出8维潜变量均值和对数方差；解码器将潜变量与风格拼接，经64、128维隐藏层重建231维特征。激活函数为ReLU，优化器Adam，学习率0.001，KL权重0.005，训练1200轮。')
    p('L = mean[(x - decoder(z,s))²] + 0.005 × KL(q(z|x,s) || N(0,I))             （6）','eq')
    p('生成时从目标风格真实样本的后验均值中有放回抽样，再加入标准差0.2的高斯扰动。相较只围绕一个风格中心采样，这种后验混合保留更多组内差异，但仍可能记忆训练样本。每种风格生成60条谱，反标准化后截断到训练特征经验范围，再实施缺陷约束。')
    sub('7.2 留出重构与基准')
    p('三折验证中，插补和标准化均在训练折拟合，VAE仅使用训练折样本学习。将验证样本编码后重建，比较标准化MSE及逐特征平均R²，并与8维PCA重构及训练集风格均值基准比较。该测试衡量重构泛化，不等同于无条件生成质量。全数据训练的风格分类器仅用于内部一致性检查，其真实样本折外准确率另行报告。')
    table('表7 三折留出重构结果',['模型','标准化MSE','逐特征平均R²'],[[k,f(v['standardized_mse']),f(v['r2'])] for k,v in q3['heldout_reconstruction'].items()],[175,150,150])
    p('本次留出检验暴露了生成模型的泛化不足：训练折中接近常量的特征可能在验证折出现极端取值，放大标准化误差；验证目标方差很小也会使逐特征R²极负。因此不能沿用旧版训练集高重构R²来证明泛化。应结合PCA基准、绝对误差与逐列尺度核查解读，当前生成结果只作受约束探索，不应称为高保真化学谱。')
    p(f'真实样本的分类器折外准确率为{f(q3["real_classifier_oof_accuracy"])}，生成谱在全数据分类器上的风格一致率为{f(q3["generated_classifier_agreement"])}。两者测试对象和训练条件不同，不能以生成一致率高于真实折外准确率证明生成更真实。')
    sub('7.3 缺陷约束与跨风格迁移')
    rules=js('q3_defect_rules.json');table('表8 缺陷尺度与参考阈值',['化学字段','输入尺度','阈值','单位'],[[k,v['transform'],f(v['threshold'],5),str(v['unit'])] for k,v in rules.items()],[180,85,100,110])
    p('对有参考阈值的生成缺陷浓度采用0.95倍阈值上限。迁移时进一步要求不超过源样品浓度的0.95倍；反式2壬烯醛没有可用绝对阈值时，只采用相对非恶化约束，不编造阈值。陈旧酒花没有单一对应化学列，用第一问模型预测的stale_hops作为代理约束。')
    p('C_new = min(C_candidate, 0.95 × T, 0.95 × C_source)                       （7）','eq')
    p('对每个源样本生成24个目标条件候选，先进行化学投影，再筛选陈旧酒花代理不高于源样品的候选，最后选目标分类概率最高者。若无可行候选，原样返回源样品并明确记为迁移失败。身份回退只保证非恶化，不保证低于绝对阈值，也不作为成功生成。')
    original_x,_,_,original_ids=load();original_x.index=original_ids.beer
    missing_source=int(original_x.loc[transfer.beer,list(rules)].isna().any(axis=1).sum())
    p(f'迁移源样品中有{missing_source}次尝试涉及至少一个缺失缺陷浓度，程序采用训练数据中位数作为约束基准。这些尝试的非恶化仅相对于插补值成立，不能证明相对于未知真实浓度也不恶化；其余源浓度可由输出独立核查。')
    t=transfer.groupby(['source_style','target_style']).agg(n=('beer','size'),fallback=('fallback_identity','sum'),success=('target_class_success','mean'),violations=('nonworsening_violations','sum')).reset_index()
    table('表9 跨风格迁移结果',['方向','尝试数','原样回退','目标识别率','非恶化违例'],[[r.source_style+' → '+r.target_style,r.n,r.fallback,f(r.success),r.violations] for r in t.itertuples()],[205,60,65,75,70]);fig('q3_validation.png','图3 留出重构误差和跨风格迁移的内部识别结果')
    checks=pd.read_csv(OUT/'q3_defect_checks.csv');table('表10 按风格生成的阈值检查',['字段','投影前超阈值','投影后超阈值','被投影比例'],[[r.compound,f(r.raw_exceedance),f(r.projected_exceedance),f(r.projection_fraction)] for r in checks.itertuples()],[175,100,100,100])
    p('投影后零超限由约束算子保证，并不意味着VAE已自主学会缺陷控制；若投影比例较高，生成分布会被集中到边界附近。实际产品验证还需结合基质效应、风格适用性及测量误差，尤其4VG在部分风格中具有特征风味而非普遍缺陷。')
    h('八 外部数据的补充分析')
    sub('8.1 风格和描述词映射')
    mapping=pd.read_csv(OUT/'external_style_mapping.csv');table('表11 外部风格对应与样本数',['题目风格','题目样本','外部样本','对应程度'],[[r.official_style,r.official_n,r.external_n,{'closer':'较接近','approximate':'近似','unmapped':'未对应'}[r.confidence]] for r in mapping.itertuples()],[190,85,85,115])
    p('对应程度为较接近仍不代表类别定义完全一致。Tripel、Dubbel、Faro等有较明确名称对应；Pils/Lager、Wheat和Stout/Porter需要合并外部子类，标为近似。Kriek与Lambic - Fruit的关系并非一一对应，本版不强行合并。全部16类及8类较接近风格分别计算结果，避免只展示一致性最好的类别。')
    p(f'词典共提取{ex["vocabulary"]["vocabulary_entries"]}个类别词条，按去掉aroma、taste、palate前缀后的精确词匹配及少量显式类别名别名，匹配原附件{ex["vocabulary"]["mapped_original_columns"]}/186个消费者描述列。一个词可能属于多个词典类别，保留重叠标记；未匹配字段不做模糊补全。附件7已标准化，聚合结果只作相对强弱对照，不能解释为原始词频。')
    p('专业品评与外部数据对齐酒体、酒精感、苦、甜、酸、果香、酒花、香料和麦芽9个维度；涩感与咸味缺少直接专业标签，留作未对应。果香等合并多个专业感官列，是显式语义假设，完整映射写入external_sensory_mapping.csv。')
    sub('8.2 风格画像的外部一致性')
    a=agreement[(agreement.subset=='all_mapped')&(agreement.source=='official')].set_index('dimension');b=agreement[(agreement.subset=='all_mapped')&(agreement.source=='generated')].set_index('dimension')
    table('表12 16类风格的跨域排序相关',['维度','实测与外部rho','95%重采样区间','生成代理与外部rho'],[[k,f(r.spearman),f'[{f(r.ci_low)}, {f(r.ci_high)}]',f(b.loc[k,'spearman'])] for k,r in a.iterrows()],[90,105,165,115]);fig('external_profiles.png','图4 Tripel与Lambic的专业实测 外部评论和生成代理百分位画像')
    p('95%区间对匹配风格进行500次有放回重采样，反映有限风格数量带来的不确定性；并非每个产品的预测区间。不同风格共享来源平台和词典，重采样独立性假设仍有限。9维结果为探索性描述，不以多重显著性筛选作为结论。')
    p(f'跨维度描述性平均相关为：专业实测对外部{f(q3["external"]["mean_official_rho"])}，生成代理对外部{f(q3["external"]["mean_generated_rho"])}。平均相关只用于概览，具体判断应结合表12的逐维方向、区间与图4的偏差。生成画像经第一问模型转换，仍携带其预测误差，不能作为独立化学验证。')
    closer=agreement.query("subset=='closer_only' and source=='official'");p(f'仅保留8类较接近风格时，专业实测对外部的跨维度平均rho为{f(closer.spearman.mean())}。子集减少使估计更不稳定；若与全部风格结果差异明显，应优先核查类别映射而非声称模型效果改变。')
    cons=pd.read_csv(OUT/'external_consumer_agreement.csv');p(f'对原附件消费者描述符按词典聚合后，{len(cons)}个可匹配维度与外部风格排序的平均rho为{f(cons.spearman.mean())}。这项比较补充了专业品评对照，但两个评论来源的语言、评分人群及聚合方式仍不同。')
    sub('8.3 消费者评分的探索性预测')
    p('以Ave Rating为目标，比较“风格与ABV”基线和“风格与ABV加11项词频份额”模型。采用按酒厂分组的5折验证，同一酒厂不会同时进入训练和测试；风格独热编码器只在训练折拟合，未知类别忽略。两个模型均使用160棵树、叶节点至少5个样本的随机森林。')
    table('表13 酒厂分组验证的消费者评分结果',['模型','OOF R²','OOF MAE','OOF Spearman'],[[k,f(rating[k]['r2']),f(rating[k]['mae']),f(rating[k]['spearman'])] for k in ['Style_ABV','Style_ABV_Flavors']],[190,95,95,95]);fig('external_rating.png','图5 按酒厂验证的风味特征置换重要性')
    gain=rating['equal_brewery_mae_gain'];p(f'先计算每个酒厂的平均绝对误差改善，再对酒厂进行1000次重采样，得到等酒厂加权MAE改善{f(gain["estimate"],4)}，95%区间[{f(gain["ci_low"],4)}, {f(gain["ci_high"],4)}]。该估计与按所有产品等权的总体MAE差值不同，应避免混用。')
    p('风味词频和平均评分来自同一评论体系，可能共享评价语境；外部样本又偏向高评分产品。因此该模型是描述性关联分析，不代表增加某种风味就能提升评分，也不用于直接优化题目中的化合物浓度。置换单个份额会暂时破坏组成约束，重要性仅作敏感度参考。')
    h('九 模型分析与检验')
    sub('9.1 缺失和等级映射敏感性')
    missing=pd.read_csv(OUT/'q1_missing_sensitivity.csv').groupby('extra_missing_rate')[['r2','mae']].mean();table('表14 只对验证特征增加随机缺失',['新增缺失比例','平均R²','平均MAE'],[[f(i,2),f(r.r2),f(r.mae)] for i,r in missing.iterrows()])
    p('扰动实验保持训练集不变，仅在验证集随机遮蔽0%、5%和10%的单元。该实验评估随机缺失鲁棒性，不代表检测限或特定仪器造成的非随机缺失。等级映射的差异已由表4给出；若结果随映射明显改变，分级结论必须保留假设限定。')
    sub('9.2 网络与生成敏感性')
    noise=pd.read_csv(OUT/'q3_noise_sensitivity.csv');table('表15 固定生成器下潜变量噪声敏感性',['噪声标准差','风格分类一致率','标准化总体方差'],[[f(r.latent_noise,1),f(r.classifier_agreement),f(r.mean_standardized_variance)] for r in noise.itertuples()])
    p('第二问分别使用0.4、0.5和0.6相关阈值，并逐一移除三种证据，计算相对主方案的Jaccard。第三问改变潜变量扰动强度，只检验已拟合模型的采样敏感性，不将其表述为重新训练的超参数稳健性。总体方差包括风格间差异，不能单独证明组内多样性。')
    norm=pd.read_csv(OUT/'external_normalization_sensitivity.csv');p(f'外部词频原值与份额归一化得到的风格排序相关，在9个对应维度中的范围为{f(norm.raw_vs_share_style_spearman.min())}至{f(norm.raw_vs_share_style_spearman.max())}。该差异提示文本量处理会影响外部对照，归一化选择需随论文公开。')
    sub('9.3 验证层次与误差传播')
    p('第一问的折外预测评价泛化误差；第二问的Bootstrap评价固定图条件下的候选稳定性；第三问的留出重构评价编码解码能力，生成分类评价内部风格一致性；外部评论对照评价粗粒度语义趋势。四类指标不能互相替代。特别是使用第一问代理验证第三问，存在共享模型误差，须由真实品评补充。')
    h('十 模型评价与推广')
    sub('10.1 优点')
    p('本研究将各类数据来源与量纲分开处理，使用折内预处理、适合小风格样本数的分层划分和酒厂分组验证；关键因素选择同时提供频率、方法差异和来源注释；生成过程显式记录投影比例及迁移失败；全部表格由实际实验文件自动生成，减少人工抄录错误。')
    sub('10.2 不足')
    p('主数据样本量小而特征维度高，模型选择尚无独立测试集；0至6原始评分无法唯一还原；固定全样本网络使稳定性结果条件于已有结构；Bootstrap次数有限。外部样本具有高评分选择偏差，粗粒度映射并不精确。生成谱缺少物理守恒与工艺变量约束，不能证明可酿造或可销售。')
    p('条件VAE的经验后验采样依赖已观测风格，不支持对未见风格作可靠外推。阈值投影可能造成边界堆积，陈旧酒花仅靠代理模型，反式2壬烯醛缺少绝对阈值；原样回退满足非恶化却没有实现目标风格，这类情况已如实计入失败。')
    sub('10.3 提高内容质量所需的证据')
    table('表16 后续补充材料及用途',['优先级','所需材料','可解决的问题'],[['高','原始逐品评员0至6评分及标准化参数','校准真实等级、分解评委差异'],['高','完整化学单位与变换、检出限数据字典','核实所有列及缺失机制'],['高','独立批次化学与感官配对样本','真正检验泛化与候选因素'],['高','盲评及小规模酿造或加标实验','验证生成风味及因果机制'],['中','原附件商品身份、酒厂、年份与外部key对照','建立产品级外部验证'],['中','逐条评论、有效评论数和文本长度','校正词频曝光及同源偏差'],['中','普通及低评分产品的外部样本','降低高分筛选偏差'],['中','缺陷阈值文献与风格适用条件','完善非恶化与可接受性约束'],['中','配方、温度、发酵时间和质量守恒信息','将谱空间迁移转化为可实施工艺']],[45,220,210])
    p('推广应先在同风格、同测量口径下开展外部批次验证，再引入原料和工艺条件形成受约束的生成模型。消费者喜好目标应在独立评分样本上检验，并以多个感官维度及缺陷风险共同评价，避免将单一总体评分作为唯一优化目标。')
    h('十一 参考文献')
    refs=[
    '[1] 2026年MathorCup数学应用挑战赛大数据竞赛赛道A初赛题面及附件1至8. 用户提供的本地赛题材料, 2026.',
    '[2] Schreurs M, Piampongsant S, Roncoroni M, et al. Predicting and improving complex beer flavor through machine learning. Nature Communications, 15:2368, 2024. doi:10.1038/s41467-024-46346-0.',
    '[3] sp1222 / stephenpolozoff. Beer Information - Tasting Profiles. Kaggle. https://www.kaggle.com/datasets/stephenpolozoff/top-beer-information . 用户提供文件及页面说明, 访问日期2026-10-28.',
    '[4] Breiman L. Random Forests. Machine Learning, 45:5-32, 2001.',
    '[5] Geurts P, Ernst D, Wehenkel L. Extremely randomized trees. Machine Learning, 63:3-42, 2006.',
    '[6] Kingma D P, Welling M. Auto-Encoding Variational Bayes. ICLR, 2014. 条件形式在本文中通过拼接风格向量实现.',
    '[7] Schreurs M, et al. Data from Predicting and improving complex beer flavor through machine learning. Zenodo, 2024. doi:10.5281/zenodo.10653704.',
    ]
    for r in refs:p(r)
    h('附录 代码与复现说明')
    p('本论文配套三问独立入口：question1.py、question2.py、question3.py；共享common.py，第三问调用external_analysis.py。先运行问题一生成感官代理，再运行问题二和问题三。所有输入按脚本位置定位，不依赖旧版D盘路径。完整代码与结果随交付包提供，正文未复制数百行程序以免掩盖建模内容。')
    table('附表A 三问输入输出',['入口','主要输入','主要输出'],[['question1.py','1.csv、4.csv、评分卡映射','折外预测、0至6等级、模型比较'],['question2.py','1.csv、2.csv、4.csv、8.xlsx、字段映射','150条Top-3、Bootstrap频率、图边及社区'],['question3.py','主数据、问题一模型、两份外部文件','生成谱、迁移谱、约束检查、外部画像与评分实验']],[110,180,185])
    p('复现命令：python code/question1.py；python code/question2.py --bootstrap 30；python code/question3.py --epochs 1200；python code/build_paper.py。Python及依赖版本见requirements.txt和results/environment.json。文件哈希见input_manifest.csv及交付清单。')
    p('已验证结论对应本次脚本的实际运行输出；风格语义对应、单调分级和代理阈值适用性属于明确假设；真实品评校准、产品级身份匹配和工艺可行性属于待验证问题。')
    def footer(c,doc):
        c.setFont('CJK',8);c.setFillColor(colors.HexColor('#555555'));c.drawCentredString(A4[0]/2,25,str(doc.page))
    pdf=out/'啤酒风味建模与外部数据验证论文.pdf'
    doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=60,leftMargin=60,topMargin=48,bottomMargin=48,title='基于化学测量与外部风味描述的啤酒感官预测及风格生成研究',author='')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (ROOT/'output/论文正文.txt').write_text('\n\n'.join(text),encoding='utf-8')
    versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','matplotlib','networkx','torch','openpyxl','reportlab']}
    dump({'python':platform.python_version(),'packages':versions},'environment.json')
    print(pdf)

if __name__=='__main__': main()

```

## common.py

```python

"""Shared paths, aligned data, numerical and reproducibility helpers."""
from pathlib import Path
import os, sys, json, hashlib
ROOT = Path(__file__).resolve().parents[1]
# Optional task-local dependency folder; normal pip installations work without it.
local = ROOT.parent / '.beer_runtime'
if local.exists(): sys.path.insert(0, str(local))
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
DATA = ROOT/'data'/'official'
EXT = ROOT/'data'/'external'
OUT = ROOT/'results'
FIG = ROOT/'figures'
for p in [OUT, FIG]: p.mkdir(parents=True, exist_ok=True)
META = ['beer','beer_id','tasting_category_fine']
SEED = 42

def load():
    x = pd.read_csv(DATA/'1.csv'); y = pd.read_csv(DATA/'4.csv')
    assert x.beer.is_unique and y.beer.is_unique
    y = y.set_index('beer').loc[x.beer].reset_index()
    assert np.array_equal(x.tasting_category_fine, y.tasting_category_fine)
    return x.drop(columns=META), y.drop(columns=META), x.tasting_category_fine, x[META]

def dump(obj, name):
    def conv(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        return str(x)
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=conv), encoding='utf-8')

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.ndim == 1: y,p=y[:,None],p[:,None]
    rho=[spearmanr(y[:,j],p[:,j]).statistic for j in range(y.shape[1]) if np.std(p[:,j])>1e-12 and np.std(y[:,j])>1e-12]
    return dict(r2=float(r2_score(y,p)),mae=float(mean_absolute_error(y,p)),spearman=float(np.mean(rho)) if rho else None)

def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.dpi':140,'savefig.dpi':180})
    return plt

def audit():
    files=[]
    for p in sorted((ROOT/'data').rglob('*')):
        if p.is_file(): files.append({'file':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    pd.DataFrame(files).to_csv(OUT/'input_manifest.csv',index=False,encoding='utf-8-sig')
    x,y,s,_=load()
    dump({'n':len(x),'chemical_features':len(x.columns),'sensory_targets':len(y.columns),'missing_chemical_cells':int(x.isna().sum().sum()),'missing_sensory_cells':int(y.isna().sum().sum()),'style_counts':s.value_counts().to_dict()},'data_audit.json')

```

## external_analysis.py

```python

"""Kaggle audit, vocabulary bridge, brewery-grouped rating study, external profiles.
All cross-domain correspondences are explicit hypotheses, not label equivalences.
"""
from common import *
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

FLAVORS=['Astringency','Body','Alcohol','Bitter','Sweet','Sour','Salty','Fruits','Hoppy','Spices','Malty']
SENSORY={'Body':['body'],'Alcohol':['alcohol'],'Bitter':['bitternes'],'Sweet':['sweetness'],'Sour':['acidity'],
         'Fruits':['A_esters_fruity','F_esters_fruity','A_hops_citrus','F_hops_citrus','A_hops_tropical','F_hops_tropical'],
         'Hoppy':['A_hops_all','F_hops_all'],'Spices':['clove','coriander'],'Malty':['A_malt_all','F_malt_all']}
# Conservative explicit selection; categories omitted here remain unaligned.
STYLE={
 'Tripel':['Tripel'], 'Dubbel':['Dubbel'], 'Faro':['Lambic - Faro'],
 'Lambic':['Lambic - Traditional','Lambic - Gueuze'],
 'Saison':['Farmhouse Ale - Saison'], 'Blond':['Blonde Ale - Belgian'],
 'Flanders old brown':['Sour - Flanders Oud Bruin'],
 'Brut':['Bière de Champagne / Bière Brut'],
 'Low/No alcohol':['Low Alcohol Beer'],
 'Scotch':['Scotch Ale / Wee Heavy','Scottish Ale'],
 'Pils/Lager':['Pilsner - Bohemian / Czech','Pilsner - German','Lager - European Pale','Lager - Helles','Lager - American'],
 'Stout/Porter':['Stout - American','Stout - English','Stout - Irish Dry','Stout - Sweet / Milk','Porter - American','Porter - English','Porter - Robust'],
 'Wheat':['Wheat Beer - Witbier','Wheat Beer - Hefeweizen'],
 'Brown':['Brown Ale - Belgian Dark','Brown Ale - English','Brown Ale - American'],
 'Fruitbeer':['Fruit and Field Beer'],
 'Strong ale':['Strong Ale - Belgian Dark','Strong Ale - Belgian Pale'],
}
HIGH={'Tripel','Dubbel','Faro','Saison','Blond','Flanders old brown','Brut','Lambic'}

def normalized_counts(e):
    a=e[FLAVORS].astype(float);total=a.sum(axis=1)
    return a.div(total.replace(0,np.nan),axis=0)

def coarse(y):
    return pd.DataFrame({k:y[v].mean(axis=1) for k,v in SENSORY.items()})

def ecdf(reference,values):
    return pd.DataFrame({c:np.searchsorted(np.sort(reference[c].dropna()),values[c],side='right')/reference[c].notna().sum() for c in values},index=values.index)

def bootstrap_mean(a,rng,B=500):
    a=np.asarray(a);bs=a[rng.integers(0,len(a),(B,len(a)))].mean(axis=1)
    return float(a.mean()),*np.quantile(bs,[.025,.975]).tolist()

def vocabulary():
    book=pd.read_excel(EXT/'Beer Descriptors Simplified.xlsx',sheet_name=None,header=None)
    vocab=[]
    alias={'Astringent':'Astringency','Alcoholic':'Alcohol','Fruity':'Fruits'}
    for sheet,d in book.items():
        for j in range(0,d.shape[1],2):
            category=alias.get(d.iloc[0,j],d.iloc[0,j])
            for i in range(1,len(d)):
                word=d.iloc[i,j]
                if pd.notna(word):vocab.append({'sheet':sheet,'category':category,'word':str(word).lower().strip(),'impact':d.iloc[i,j+1]})
    v=pd.DataFrame(vocab);v.to_csv(OUT/'external_vocabulary.csv',index=False,encoding='utf-8-sig')
    d7=pd.read_csv(DATA/'7.csv');pairs=[]
    direct={'fruity':'Fruits','hoppy':'Hoppy','malt':'Malty','sweet':'Sweet','sour':'Sour','bitter':'Bitter','alcohol':'Alcohol'}
    for c in d7.columns[3:]:
        word=c.split('_',1)[-1];cats=sorted(set(v.loc[v.word==word,'category']))
        if word in direct:cats=sorted(set(cats+[direct[word]]))
        for cat in cats:pairs.append({'source_column':c,'word':word,'external_dimension':cat,'rule':'exact word or explicit category-name alias','multiple_categories':len(cats)>1})
    pd.DataFrame(pairs).to_csv(OUT/'external_descriptor_mapping.csv',index=False,encoding='utf-8-sig')
    # Existing standardized consumer descriptors are averaged, never treated as raw word counts.
    agg=pd.DataFrame({cat:d7[sorted(set(p['source_column'] for p in pairs if p['external_dimension']==cat))].mean(axis=1) for cat in FLAVORS if any(p['external_dimension']==cat for p in pairs)})
    pd.concat([d7[META],agg],axis=1).to_csv(OUT/'external_official_consumer_coarse.csv',index=False)
    return {'vocabulary_entries':len(v),'mapped_original_columns':len(set(p['source_column'] for p in pairs)),'original_descriptor_columns':len(d7.columns)-3,'mapping_pairs':len(pairs),'overlapping_dictionary_categories':True}

def audit_external():
    e=pd.read_csv(EXT/'beer_data_set.csv');assert e.key.is_unique
    p=normalized_counts(e);rng=np.random.default_rng(42);profiles=[]
    for style,idx in e.groupby('Style').groups.items():
        for c in FLAVORS:
            a=p.loc[idx,c].dropna()
            if len(a):
                mean,lo,hi=bootstrap_mean(a,rng)
                profiles.append({'style':style,'dimension':c,'n':len(a),'mean_share':mean,'ci_low':lo,'ci_high':hi})
    pd.DataFrame(profiles).to_csv(OUT/'external_style_profiles.csv',index=False,encoding='utf-8-sig')
    x,y,s,_=load();maps=[]
    for st in sorted(s.unique()):
        names=STYLE.get(st,[])
        maps.append({'official_style':st,'external_styles':' | '.join(names),'official_n':int((s==st).sum()),'external_n':int(e.Style.isin(names).sum()),'confidence':'closer' if st in HIGH else ('approximate' if names else 'unmapped'),'reason':'explicit stylistic grouping; not sample identity' if names else 'no sufficiently specific correspondence'})
    pd.DataFrame(maps).to_csv(OUT/'external_style_mapping.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'external_dimension':k,'official_sensory_columns':' | '.join(v),'status':'coarse semantic correspondence, scales differ'} for k,v in SENSORY.items()]+[{'external_dimension':k,'official_sensory_columns':'','status':'not aligned'} for k in ['Astringency','Salty']]).to_csv(OUT/'external_sensory_mapping.csv',index=False)
    info={'rows':len(e),'columns':len(e.columns),'styles':e.Style.nunique(),'breweries':e.Brewery.nunique(),'missing':e.isna().sum().to_dict(),'zero_flavor_totals':int(e[FLAVORS].sum(axis=1).eq(0).sum()),'duplicate_keys':int(e.key.duplicated().sum()),'mapped_styles':len(STYLE),'closer_styles':len(HIGH),'source':'https://www.kaggle.com/datasets/stephenpolozoff/top-beer-information','vocabulary':vocabulary()}
    dump(info,'external_audit.json');return e

def rating_study(e):
    p=normalized_counts(e).fillna(0)
    a=pd.concat([e[['Style','ABV']].reset_index(drop=True),p],axis=1)
    y=e['Ave Rating'].to_numpy();valid=np.isfinite(y)&np.isfinite(e.ABV)
    a=a.loc[valid].reset_index(drop=True); y=y[valid]; groups=e.loc[valid,'Brewery'].to_numpy()
    records=[];oof={n:np.zeros(len(y)) for n in ['Style_ABV','Style_ABV_Flavors']};imps=[]
    for fold,(tr,va) in enumerate(GroupKFold(5).split(a,y,groups)):
        assert not set(groups[tr])&set(groups[va])
        for name in oof:
            num=['ABV']+(FLAVORS if name.endswith('Flavors') else [])
            pre=ColumnTransformer([('style',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['Style']),('num',SimpleImputer(),num)])
            m=make_pipeline(pre,RandomForestRegressor(n_estimators=160,min_samples_leaf=5,max_features=.7,n_jobs=4,random_state=42+fold))
            m.fit(a.iloc[tr],y[tr]); pred=m.predict(a.iloc[va]);oof[name][va]=pred
            records.append({'fold':fold,'model':name,'n_test':len(va),**metrics(y[va],pred)})
            if name.endswith('Flavors'):
                rng=np.random.default_rng(42+fold)
                for col in FLAVORS:
                    v=a.iloc[va].copy();v[col]=rng.permutation(v[col].to_numpy())
                    imps.append({'fold':fold,'feature':col,'increase_mae':float(np.mean(abs(y[va]-m.predict(v)))-np.mean(abs(y[va]-pred)))})
        print('External rating group fold',fold,flush=True)
    pd.DataFrame(records).to_csv(OUT/'external_rating_cv.csv',index=False);pd.DataFrame(imps).to_csv(OUT/'external_rating_importance.csv',index=False)
    pd.DataFrame({'observed':y,**oof}).to_csv(OUT/'external_rating_oof.csv',index=False)
    delta=abs(y-oof['Style_ABV'])-abs(y-oof['Style_ABV_Flavors'])
    group_delta=pd.Series(delta).groupby(groups).mean().to_numpy();rng=np.random.default_rng(42)
    mean,lo,hi=bootstrap_mean(group_delta,rng,1000)
    info={n:metrics(y,pred) for n,pred in oof.items()};info['equal_brewery_mae_gain']={'estimate':mean,'ci_low':lo,'ci_high':hi};info['caveat']='same review corpus generates rating and text; associations are not causal or prospective predictions'
    dump(info,'external_rating_summary.json')
    plt=plotting();fig,ax=plt.subplots(figsize=(7.5,4));v=pd.DataFrame(imps).groupby('feature').increase_mae.mean().sort_values();v.plot.barh(ax=ax,color='#337a96');ax.set_xlabel('打乱验证集该指标后的 MAE 增量');ax.set_title('消费者评分 分酒厂验证的风味指标重要性');fig.tight_layout();fig.savefig(FIG/'external_rating.png');plt.close(fig)

def compare_external(generated_y,generated_style):
    e=pd.read_csv(EXT/'beer_data_set.csv');x,y,s,_=load();real=coarse(y);gen=coarse(generated_y)
    realp=real.rank(pct=True);genp=ecdf(real,gen)
    counts=normalized_counts(e);extp=counts.rank(pct=True)
    rows=[]
    for st,names in STYLE.items():
        mask=e.Style.isin(names)
        if not mask.any():continue
        for c in real:
            rows.append({'style':st,'dimension':c,'confidence':'closer' if st in HIGH else 'approximate','official_n':int((s==st).sum()),'external_n':int(mask.sum()),'official_percentile':realp.loc[s==st,c].mean(),'external_percentile':extp.loc[mask,c].mean(),'generated_percentile':genp.loc[np.asarray(generated_style)==st,c].mean()})
    d=pd.DataFrame(rows);d.to_csv(OUT/'external_profile_comparison.csv',index=False)
    results=[];rng=np.random.default_rng(42)
    for subset in ['all_mapped','closer_only']:
        a=d if subset=='all_mapped' else d[d.confidence=='closer']
        for c,t in a.groupby('dimension'):
            for source in ['official','generated']:
                v=t[source+'_percentile'].to_numpy();u=t.external_percentile.to_numpy();rho=spearmanr(v,u).statistic
                boot=[]
                for _ in range(500):
                    idx=rng.integers(0,len(v),len(v));r=spearmanr(v[idx],u[idx]).statistic
                    if np.isfinite(r):boot.append(r)
                lo,hi=np.quantile(boot,[.025,.975]) if boot else [np.nan,np.nan]
                results.append({'subset':subset,'dimension':c,'source':source,'n_styles':len(v),'spearman':rho,'ci_low':lo,'ci_high':hi,'mean_absolute_percentile_gap':np.mean(abs(v-u))})
    pd.DataFrame(results).to_csv(OUT/'external_agreement.csv',index=False)
    consumer=pd.read_csv(OUT/'external_official_consumer_coarse.csv')
    # Use unique beer index: beer_id is rounded and duplicated in sensory tables.
    _,_,_,official_ids=load()
    consumer=consumer.set_index('beer').loc[official_ids.beer].reset_index()
    consumer_ranks=consumer[[c for c in FLAVORS if c in consumer]].rank(pct=True)
    consumer_results=[]
    for c in consumer_ranks:
        a=[];b=[]
        for st,names in STYLE.items():
            a.append(consumer_ranks.loc[s==st,c].mean());b.append(extp.loc[e.Style.isin(names),c].mean())
        consumer_results.append({'dimension':c,'n_styles':len(a),'spearman':spearmanr(a,b).statistic})
    pd.DataFrame(consumer_results).to_csv(OUT/'external_consumer_agreement.csv',index=False)
    # Sensitivity to review-length confounding: raw counts vs normalized shares.
    raw=e[FLAVORS].rank(pct=True);sens=[]
    for c in SENSORY:
        rr=[raw.loc[e.Style.isin(names),c].mean() for names in STYLE.values()]
        pp=[extp.loc[e.Style.isin(names),c].mean() for names in STYLE.values()]
        sens.append({'dimension':c,'raw_vs_share_style_spearman':spearmanr(rr,pp).statistic})
    pd.DataFrame(sens).to_csv(OUT/'external_normalization_sensitivity.csv',index=False)
    plt=plotting();fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for ax,st in zip(axes,['Tripel','Lambic']):
        t=d[d['style']==st].set_index('dimension');t[['official_percentile','external_percentile','generated_percentile']].plot.bar(ax=ax,width=.8);ax.set_title(st);ax.set_ylim(0,1);ax.set_xlabel('');ax.tick_params(axis='x',rotation=55);ax.legend(['题目实测','外部评论','生成代理'],fontsize=7)
    fig.tight_layout();fig.savefig(FIG/'external_profiles.png');plt.close(fig)
    return {'mapped_style_count':d['style'].nunique(),'dimensions':len(SENSORY),'mean_official_rho':float(pd.DataFrame(results).query("subset=='all_mapped' and source=='official'").spearman.mean()),'mean_generated_rho':float(pd.DataFrame(results).query("subset=='all_mapped' and source=='generated'").spearman.mean()),'note':'exploratory ecological trend comparison, not chemical validation; external data unused in generation or tuning'}

if __name__=='__main__':
    e=audit_external();rating_study(e)

```

## package_delivery.py

```python

"""Create a portable delivery bundle, excluding caches and legacy experiments."""
from pathlib import Path
import hashlib,csv,zipfile
ROOT=Path(__file__).resolve().parents[1]

def main():
    allowed={'code','data','results','figures','output'}
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if any(part in {'__pycache__','.cache','tmp','legacy'} for part in rel.parts):continue
        if p.suffix in {'.pyc','.log','.zip'} or p.name=='交付文件清单.csv':continue
        if rel.parts[0] in allowed or p.parent==ROOT:files.append(p)
    manifest=ROOT/'交付文件清单.csv'
    with manifest.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['path','bytes','sha256'])
        for p in files:writer.writerow([p.relative_to(ROOT).as_posix(),p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest()])
    bundle=ROOT.parent/'啤酒建模修订版_论文与三问代码.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[manifest]:z.write(p,'beer_revision/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None
        for name in ['code/question1.py','code/question2.py','code/question3.py','code/common.py','code/external_analysis.py','data/official/1.csv','data/external/beer_data_set.csv','output/pdf/啤酒风味建模与外部数据验证论文.pdf']:
            assert 'beer_revision/'+name in z.namelist()
    print(bundle, 'files',len(files)+1,'bytes',bundle.stat().st_size)

if __name__=='__main__':main()

```

## package_literature_revision.py

```python

"""Package the complete runnable revision; omit caches and self-hash recursion."""
from pathlib import Path
import json,hashlib,zipfile,importlib.metadata,platform


def main():
    base=Path(__file__).resolve().parents[1]
    rev=base/'results/literature_revision'
    names=['numpy','pandas','scipy','scikit-learn','matplotlib','networkx','joblib','openpyxl','reportlab','pypdf','pypdfium2']
    (rev/'runtime_environment.json').write_text(json.dumps({'python':platform.python_version(),
        'packages':{name:importlib.metadata.version(name) for name in names}},indent=2),encoding='utf-8')
    qa=json.loads((rev/'delivery_verification.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((base/qa['pdf']).read_bytes()).hexdigest()==qa['sha256'],'PDF changed after visual review'
    rows=[];manifest=base/'交付文件清单_文献优化版.json'
    for p in sorted(base.rglob('*')):
        if p.is_file() and '.cache' not in p.parts and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.log'] and p!=manifest:
            rows.append({'file':p.relative_to(base).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    output=base.parent/'文献优化版_论文代码与证据.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for row in rows:archive.write(base/row['file'],arcname='外部数据修订版/'+row['file'])
        archive.write(manifest,arcname='外部数据修订版/'+manifest.name)
    with zipfile.ZipFile(output) as archive:assert archive.testzip() is None
    print('Verified ZIP:',output,'bytes:',output.stat().st_size,'files:',len(rows)+1)


if __name__=='__main__':main()

```

## package_ryh1_source.py

```python

"""Refresh the readable source and runnable source/data bundle for RYH优化1."""
from pathlib import Path
import ast,hashlib,json,zipfile


def main():
    base=Path(__file__).resolve().parents[1];dest=base/'output/literature_revision';repo=base.parents[1]
    first=['literature_revision.py','literature_models.py','literature_nsga2.py','literature_generation.py','literature_inference.py',
           'validate_literature_revision.py','build_literature_paper.py','format_ryh1_template.py']
    names=first+sorted(p.name for p in (base/'code').glob('*.py') if p.name not in first)
    overview=['# RYH优化1 完整源码','按模块汇总当前代码，供浏览。执行时使用独立.py文件。正文各级标题按用户要求左对齐。','## 模块索引']
    overview += [f'{i}. {name}' for i,name in enumerate(names,1)]
    for name in names:
        source=(base/'code'/name).read_text(encoding='utf-8');ast.parse(source)
        overview += ['## '+name,'```python',source.rstrip(),'```']
    (dest/'RYH优化1_完整源码.md').write_text('\n\n'.join(overview),encoding='utf-8')
    readme='''# RYH优化1 完整代码与运行材料

保留仓库相对目录结构。进入“成果/外部数据修订版”，依次运行：

    python -m pip install -r requirements_literature.txt
    python code/literature_revision.py --stage all --subsamples 100
    python code/build_literature_paper.py
    python code/validate_literature_revision.py
    python code/format_ryh1_template.py

新实验结果位于results/literature_revision，原版结果保留。
本次复用已有cVAE权重做NumPy推理，无需PyTorch；重训原版生成器时另安装原requirements.txt。
模板版论文是RYH优化1.docx，正文各级标题及关键词左对齐，过程性附录已移除。
使用Microsoft Word更新目录和页码后导出PDF；包内已有最新Word与PDF成稿。
完整源码.md用于通读，运行时使用各独立.py模块。主实验、数据、现有权重和元数据已随包提供。
'''
    (dest/'RYH优化1_代码运行说明.md').write_text(readme,encoding='utf-8')
    files=list((base/'code').glob('*.py'))+[p for p in (base/'data').rglob('*') if p.is_file()]
    files += [base/n for n in ['requirements.txt','requirements_literature.txt','README_文献优化版.md','literature/sources.json']]
    files += [base/'results'/n for n in ['q3_cvae.pt','q3_preprocessing.joblib','q1_summary.json','q2_top3.csv','external_audit.json','external_rating_summary.json']]
    files += [dest/n for n in ['RYH优化1_完整源码.md','RYH优化1_代码运行说明.md','RYH优化1.docx','RYH优化1.pdf']]
    files += list((base/'results/literature_revision').glob('*.json'))+list((base/'results/literature_revision').glob('*.csv'))
    files += [repo/'资料/Mathorcup论文标准模板(1).docx']
    unique={p.resolve():p for p in files if p.is_file()};manifest=[]
    output=dest/'RYH优化1_完整代码与运行材料.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in unique.values():
            name=p.relative_to(repo).as_posix();z.write(p,arcname=name)
            manifest.append({'file':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        z.writestr('源码包文件校验.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(output) as z:assert z.testzip() is None
    print('Source modules:',len(names),'bundle bytes:',output.stat().st_size)


if __name__=='__main__':main()

```

## question1.py

```python

"""Question 1: fold-local missingness handling, 3x3 CV, ordinal proxy scores.
Run: python code/question1.py. No external data are used as sensory labels.
"""
from common import *
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

def model(name, seed=42):
    if name=='ExtraTrees': m=ExtraTreesRegressor(n_estimators=300,max_features='sqrt',min_samples_leaf=2,n_jobs=4,random_state=seed)
    elif name=='RandomForest': m=RandomForestRegressor(n_estimators=300,max_features='sqrt',min_samples_leaf=3,n_jobs=4,random_state=seed)
    elif name=='Ridge': m=make_pipeline(StandardScaler(),Ridge(alpha=100.))
    else: m=DummyRegressor(strategy='mean')
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),m)

def grade(y, train, method='minmax'):
    a=np.asarray(train); y=np.asarray(y)
    if method=='minmax':
        lo=a.min(0); span=np.maximum(a.max(0)-lo,1e-12)
        return np.rint(np.clip((y-lo)/span,0,1)*6).astype(int)
    return np.column_stack([np.minimum(6,np.searchsorted(np.sort(a[:,j]),y[:,j],side='right')*7//len(a)) for j in range(a.shape[1])])

def plot_results(df):
    plt=plotting();fig,axes=plt.subplots(1,2,figsize=(9,4),gridspec_kw={'width_ratios':[3,1]})
    for ax,names in zip(axes,[['Mean','RandomForest','ExtraTrees'],['Ridge']]):
        g=df[df.model.isin(names)].groupby('model').r2
        g.mean().plot.bar(yerr=g.std(),ax=ax,color='#337a96');ax.set_ylabel('验证集平均 R²');ax.tick_params(axis='x',rotation=0);ax.set_xlabel('');ax.axhline(0,color='gray',lw=.6)
    axes[0].set_title('树模型与均值基线');axes[1].set_title('Ridge 单独尺度')
    fig.tight_layout();fig.savefig(FIG/'q1_models.png');plt.close(fig)

def main():
    audit(); X,Y,s,ids=load(); assert not Y.isna().any().any()
    names=['Mean','Ridge','RandomForest','ExtraTrees']; records=[]; preds={n:[] for n in names}; grade_rows=[]
    for seed in [42,1024,2024]:
        cv=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(X,s))
        for name in names:
            oof=np.zeros(Y.shape)
            for fold,(tr,va) in enumerate(cv):
                m=model(name,seed).fit(X.iloc[tr],Y.iloc[tr]); p=m.predict(X.iloc[va]);oof[va]=p
                rec={'seed':seed,'fold':fold,'model':name,**metrics(Y.iloc[va],p)}; records.append(rec)
                for method in ['minmax','quantile']:
                    a=grade(Y.iloc[va],Y.iloc[tr],method); b=grade(p,Y.iloc[tr],method)
                    grade_rows.append({'seed':seed,'fold':fold,'model':name,'mapping':method,'exact':np.mean(a==b),'within1':np.mean(abs(a-b)<=1),'kappa':cohen_kappa_score(a.ravel(),b.ravel(),weights='quadratic')})
            preds[name].append(oof)
            print('Q1',seed,name,metrics(Y,oof),flush=True)
    df=pd.DataFrame(records); df.to_csv(OUT/'q1_cv_folds.csv',index=False)
    gr=pd.DataFrame(grade_rows);gr.to_csv(OUT/'q1_grade_sensitivity.csv',index=False)
    summary=df.groupby('model')[['r2','mae','spearman']].agg(['mean','std']);summary.to_csv(OUT/'q1_model_summary.csv')
    best=df.groupby('model').r2.mean().idxmax(); final=model(best).fit(X,Y);joblib.dump(final,OUT/'q1_model.joblib')
    oof=np.mean(preds[best],axis=0)
    pd.concat([ids,pd.DataFrame(oof,columns=Y.columns)],axis=1).to_csv(OUT/'q1_oof_predictions.csv',index=False)
    pred=final.predict(X)
    pd.concat([ids,pd.DataFrame(grade(pred,Y),columns=Y.columns)],axis=1).to_csv(OUT/'q1_grades_0_6.csv',index=False)
    mapping=pd.read_csv(ROOT/'data/metadata/00_感官维度映射.csv')
    mapping['available_in_model']=mapping.target.isin(Y.columns)
    mapping.to_csv(OUT/'q1_card_dimension_mapping.csv',index=False,encoding='utf-8-sig')
    # OOF grades use repeat-42 fold-local calibration. Main grades are a descriptive fitted answer.
    goof=np.zeros(Y.shape,dtype=int)
    for tr,va in StratifiedKFold(3,shuffle=True,random_state=42).split(X,s): goof[va]=grade(preds[best][0][va],Y.iloc[tr])
    pd.concat([ids,pd.DataFrame(goof,columns=Y.columns)],axis=1).to_csv(OUT/'q1_oof_grades_0_6.csv',index=False)
    per=[]
    for j,t in enumerate(Y): per.append({'target':t,**metrics(Y.iloc[:,j],oof[:,j])})
    pd.DataFrame(per).to_csv(OUT/'q1_per_target.csv',index=False)
    # Missingness perturbation only on validation features; fit remains fold-local.
    sens=[]
    for tr,va in StratifiedKFold(3,shuffle=True,random_state=42).split(X,s):
        m=model(best).fit(X.iloc[tr],Y.iloc[tr])
        for rate in [0,.05,.10]:
            a=X.iloc[va].copy();rng=np.random.default_rng(42+int(100*rate));a=a.mask(rng.random(a.shape)<rate)
            sens.append({'extra_missing_rate':rate,**metrics(Y.iloc[va],m.predict(a))})
    pd.DataFrame(sens).to_csv(OUT/'q1_missing_sensitivity.csv',index=False)
    dump({'selected_model':best,'selection_note':'fixed candidate comparison; CV is not nested and selected score may be optimistic','cv':'3 repeats x 3 stratified folds','scores':df.groupby('model')[['r2','mae']].mean().to_dict('index'),'grade_note':'0-6 grades are assumed monotone mappings, not recovered original ratings','missing_sensitivity':pd.DataFrame(sens).groupby('extra_missing_rate').r2.mean().to_dict()},'q1_summary.json')
    plot_results(df)

if __name__=='__main__': main()

```

## question2.py

```python

"""Question 2: supplied coexpression graph, three-method evidence, true bootstrap.
The fixed graph is a full-data structural prior. Bootstrap stability is conditional
on this graph, not independent predictive validation. 6.csv is a same-source check.
"""
from common import *
import argparse
import networkx as nx
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor

def graph(cols,threshold=.5):
    c=pd.read_csv(DATA/'2.csv',index_col=0)
    mp=pd.read_csv(ROOT/'data/metadata/00_数据字典_字段映射.csv')
    m=mp[mp.source=='2.csv'].drop_duplicates('label').drop_duplicates('target')
    mapping={r.label:r.target for r in m.itertuples() if r.label in c.index and r.target in cols}
    g=nx.Graph();g.add_nodes_from(cols)
    labels=list(mapping)
    for i,a in enumerate(labels):
        for b in labels[i+1:]:
            v=c.loc[a,b]
            if pd.notna(v) and abs(v)>=threshold:g.add_edge(mapping[a],mapping[b],weight=abs(float(v)))
    comm=list(nx.community.greedy_modularity_communities(g,weight='weight'))
    modules={c:i for i,com in enumerate(comm) for c in com}
    return g,modules

def evidence(X,Y,s,seed=42,trees=60):
    a=SimpleImputer(strategy='median').fit_transform(X)
    a=StandardScaler().fit_transform(a); b=np.asarray(Y)
    d=pd.get_dummies(s).to_numpy(dtype=float)
    rx=a-d@np.linalg.lstsq(d,a,rcond=None)[0]; ry=b-d@np.linalg.lstsq(d,b,rcond=None)[0]
    rx=rx/(rx.std(0)+1e-12);ry=ry/(ry.std(0)+1e-12)
    partial=rx.T@ry/len(a)
    ridge=Ridge(alpha=100).fit(a,b).coef_.T
    tree=np.empty((a.shape[1],b.shape[1]))
    for j in range(b.shape[1]):
        tree[:,j]=ExtraTreesRegressor(n_estimators=trees,max_features='sqrt',min_samples_leaf=3,n_jobs=1,random_state=seed).fit(a,b[:,j]).feature_importances_
    ev=[(rankdata(abs(v),axis=0)-1)/(a.shape[1]-1) for v in [partial,ridge,tree]]
    return np.stack(ev),partial

def pick(score,cols,modules):
    out=[]
    for j in range(score.shape[1]):
        chosen=[]; used=set()
        for i in np.argsort(-score[:,j],kind='stable'):
            c=cols[i]
            if modules[c] in used:continue
            chosen.append(i);used.add(modules[c])
            if len(chosen)==3:break
        out.append(chosen)
    return out

def jac(a,b):return len(set(a)&set(b))/len(set(a)|set(b))

def main(B=30):
    X,Y,s,ids=load();cols=list(X);g,mods=graph(cols);ev,partial=evidence(X,Y,s)
    score=ev.mean(0);tops=pick(score,cols,mods);freq=np.zeros(score.shape)
    rng=np.random.default_rng(42)
    groups=[np.flatnonzero(s.to_numpy()==v) for v in s.unique()]
    for k in range(B):
        idx=np.concatenate([rng.choice(a,len(a),replace=True) for a in groups])
        e,_=evidence(X.iloc[idx],Y.iloc[idx],s.iloc[idx],42+k)
        for j,ls in enumerate(pick(e.mean(0),cols,mods)):freq[ls,j]+=1
        print('Q2 bootstrap',k+1,'/',B,flush=True)
    freq/=B
    thr=pd.read_excel(DATA/'8.xlsx',sheet_name='Sensory thresholds').drop_duplicates('field_name_R').set_index('field_name_R')
    rows=[]
    for j,t in enumerate(Y):
        for rank,i in enumerate(tops[j],1):
            c=cols[i]; meta=thr.loc[c].to_dict() if c in thr.index else {}
            rows.append({'target':t,'rank':rank,'compound':c,'score':score[i,j],'partial_corr':partial[i,j],'module':mods[c],'bootstrap_frequency':freq[i,j],'bootstrap_mc_se':np.sqrt(freq[i,j]*(1-freq[i,j])/B),'origin':meta.get('origin_paper'),'threshold':meta.get('threshold_mean'),'unit':meta.get('unit')})
    table=pd.DataFrame(rows);table.to_csv(OUT/'q2_top3.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(freq,index=cols,columns=Y.columns).to_csv(OUT/'q2_all_selection_frequencies.csv')
    nx.to_pandas_edgelist(g).to_csv(OUT/'q2_graph_edges.csv',index=False)
    pd.DataFrame({'compound':cols,'module':[mods[c] for c in cols]}).to_csv(OUT/'q2_modules.csv',index=False)
    ab=[]
    for threshold in [.4,.5,.6]:
        _,m=graph(cols,threshold); p=pick(score,cols,m)
        ab.append({'variant':'graph_'+str(threshold),'jaccard_vs_main':np.mean([jac(a,b) for a,b in zip(tops,p)])})
    for k,name in enumerate(['partial','ridge','tree']):
        p=pick(np.delete(ev,k,axis=0).mean(0),cols,mods)
        ab.append({'variant':'without_'+name,'jaccard_vs_main':np.mean([jac(a,b) for a,b in zip(tops,p)])})
    pd.DataFrame(ab).to_csv(OUT/'q2_sensitivity.csv',index=False)
    method=[pick(v,cols,mods) for v in ev]
    jacc=[jac(method[a][j],method[b][j]) for a in range(3) for b in range(a+1,3) for j in range(Y.shape[1])]
    dump({'bootstrap_repeats':B,'nodes':len(g),'edges':g.number_of_edges(),'modules':len(set(mods.values())),'top1_frequency':table[table['rank']==1].bootstrap_frequency.mean(),'top3_frequency':table.bootstrap_frequency.mean(),'all3_frequency_at_least_0_8':int(table.groupby('target').bootstrap_frequency.min().ge(.8).sum()),'mean_method_jaccard':np.mean(jacc),'source_annotated':int(table.origin.notna().sum()),'maximum_mc_se':.5/np.sqrt(B),'note':'fixed full-data graph; bootstrap conditional stability; no causal identification'},'q2_summary.json')
    plt=plotting();fig,axes=plt.subplots(1,2,figsize=(9,4));table.boxplot(column='bootstrap_frequency',by='rank',ax=axes[0]);axes[0].set_title('Top-3 重采样入选频率');axes[0].set_xlabel('排名');axes[0].set_ylabel('频率');axes[1].barh([a['variant'] for a in ab],[a['jaccard_vs_main'] for a in ab],color='#337a96');axes[1].set_xlabel('与主方案的平均 Jaccard');fig.suptitle('');fig.tight_layout();fig.savefig(FIG/'q2_stability.png');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bootstrap',type=int,default=30);args=p.parse_args();main(args.bootstrap)

```

## question3.py

```python

"""Question 3: cVAE generation, held-out reconstruction, constrained transfer,
and independent external style profiles. Run after question1.py.
"""
from common import *
import argparse,joblib
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from external_analysis import audit_external,rating_study,compare_external
torch.set_num_threads(4)

class CVAE(nn.Module):
    def __init__(self,p,k,latent=8):
        super().__init__();self.enc=nn.Sequential(nn.Linear(p+k,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU());self.mu=nn.Linear(64,latent);self.lv=nn.Linear(64,latent)
        self.dec=nn.Sequential(nn.Linear(latent+k,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU(),nn.Linear(128,p))
    def encode(self,x,c):
        h=self.enc(torch.cat([x,c],1));return self.mu(h),self.lv(h).clamp(-10,10)
    def decode(self,z,c):return self.dec(torch.cat([z,c],1))

def train(z,c,epochs,beta=.005,latent=8,seed=42):
    torch.manual_seed(seed);m=CVAE(z.shape[1],c.shape[1],latent);opt=torch.optim.Adam(m.parameters(),lr=.001)
    x=torch.tensor(z,dtype=torch.float32);cond=torch.tensor(c,dtype=torch.float32);trace=[]
    for epoch in range(epochs):
        opt.zero_grad();mu,lv=m.encode(x,cond);pred=m.decode(mu+torch.randn_like(mu)*torch.exp(lv*.5),cond)
        rec=(pred-x).square().mean();kl=-.5*(1+lv-mu.square()-lv.exp()).mean();loss=rec+beta*kl;loss.backward();opt.step()
        if epoch%100==0:trace.append({'epoch':epoch,'reconstruction':rec.item(),'kl':kl.item()})
    m.eval();return m,trace

def encode(m,z,c):
    with torch.no_grad(): return m.encode(torch.tensor(z,dtype=torch.float32),torch.tensor(c,dtype=torch.float32))[0].numpy()

def decode(m,z,c):
    with torch.no_grad(): return m.decode(torch.tensor(z,dtype=torch.float32),torch.tensor(c,dtype=torch.float32)).numpy()

def defect_rules(cols):
    t=pd.read_excel(DATA/'8.xlsx',sheet_name='Sensory thresholds').set_index('field_name_R')
    verified=pd.read_csv(ROOT/'data/metadata/00_数值变换核验.csv').set_index('compound')
    supplied=pd.read_csv(DATA/'1.csv').set_index('beer')
    calibrated=pd.read_excel(DATA/'8.xlsx',sheet_name='Calibrated chemical dataset').set_index('beer').loc[supplied.index]
    out={};calibration=[]
    for c in ['DMS','X2.3.butanedione','X4VG','X.E..2.nonenal']:
        assert c in cols
        assert c in verified.index and verified.loc[c,'transform'] in ['log10','raw'], 'Defect scale must be verified'
        positive=calibrated[c]>0
        expected=np.log10(calibrated.loc[positive,c]) if verified.loc[c,'transform']=='log10' else calibrated.loc[positive,c]
        differences=abs(supplied.loc[positive,c]-expected).dropna()
        assert (differences<1e-6).mean()>.95, 'Calibration does not support the assumed scale'
        calibration.append({'compound':c,'n_compared':len(differences),'n_discrepancy_above_1e_6':int((differences>=1e-6).sum()),'max_abs_difference':float(differences.max()),'median_abs_difference':float(differences.median())})
        cap=float(t.loc[c,'threshold_mean']) if c in t.index and pd.notna(t.loc[c,'threshold_mean']) else None
        out[c]={'index':cols.index(c),'threshold':cap,'transform':verified.loc[c,'transform'],'unit':t.loc[c,'unit'] if c in t.index else None}
    pd.DataFrame(calibration).to_csv(OUT/'q3_calibration_checks.csv',index=False)
    return out

def project(a,rules,source=None):
    a=a.copy()
    for c,d in rules.items():
        cap=.95*d['threshold'] if d['threshold'] is not None else np.inf
        if source is not None:
            source_concentration=10**source[d['index']] if d['transform']=='log10' else source[d['index']]
            cap=np.minimum(cap,.95*source_concentration)
        if d['transform']=='log10': cap=np.log10(cap)
        a[:,d['index']]=np.minimum(a[:,d['index']],cap)
    return a

def main(epochs=1200):
    if not (OUT/'q1_model.joblib').exists():raise RuntimeError('Run question1.py first')
    X,Y,s,ids=load();cols=list(X);le=LabelEncoder().fit(s);labels=le.transform(s);cond=np.eye(len(le.classes_))[labels]
    validation=[];classification=[];classifier_oof=np.zeros(len(s),dtype=int)
    for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(X,s)):
        imp=SimpleImputer(strategy='median').fit(X.iloc[tr]);sc=StandardScaler().fit(imp.transform(X.iloc[tr]));z=sc.transform(imp.transform(X.iloc[tr]));v=sc.transform(imp.transform(X.iloc[va]))
        model,_=train(z,cond[tr],epochs,seed=42+fold);rec=decode(model,encode(model,v,cond[va]),cond[va])
        means=np.array([z[labels[tr]==i].mean(0) for i in range(len(le.classes_))]);pca=PCA(8).fit(z)
        for name,p in [('cVAE',rec),('PCA8',pca.inverse_transform(pca.transform(v))),('StyleMean',means[labels[va]])]:
            validation.append({'fold':fold,'model':name,'standardized_mse':np.mean((v-p)**2),**metrics(v,p)})
        clf=RandomForestClassifier(n_estimators=250,max_features='sqrt',class_weight='balanced',random_state=42,n_jobs=4).fit(z,labels[tr]);classifier_oof[va]=clf.predict(v)
        print('Q3 held-out fold',fold,flush=True)
    pd.DataFrame(validation).to_csv(OUT/'q3_reconstruction_validation.csv',index=False)
    imp=SimpleImputer(strategy='median').fit(X);sc=StandardScaler().fit(imp.transform(X));z=sc.transform(imp.transform(X));m,trace=train(z,cond,epochs)
    pd.DataFrame(trace).to_csv(OUT/'q3_training_trace.csv',index=False)
    mus=encode(m,z,cond);clf=RandomForestClassifier(n_estimators=300,max_features='sqrt',class_weight='balanced',random_state=42,n_jobs=4).fit(z,labels)
    proxy=joblib.load(OUT/'q1_model.joblib');rules=defect_rules(cols);rng=np.random.default_rng(42)
    lo=np.nanmin(X,axis=0);hi=np.nanmax(X,axis=0);generated=[];genstyles=[];raw_all=[];genlabels=[]
    for k,st in enumerate(le.classes_):
        # Empirical posterior mixture retains more within-style spread than a single centroid.
        idx=rng.choice(np.flatnonzero(labels==k),60,replace=True);latent=mus[idx]+rng.normal(0,.2,(60,mus.shape[1]))
        a=sc.inverse_transform(decode(m,latent,np.tile(np.eye(len(le.classes_))[k],(60,1))))
        a=np.clip(a,lo,hi);raw_all.append(a);generated.append(project(a,rules));genstyles += [st]*60;genlabels += [k]*60
    raw=np.vstack(raw_all);g=np.vstack(generated);gp=pd.DataFrame(proxy.predict(pd.DataFrame(g,columns=cols)),columns=Y.columns)
    pd.concat([pd.Series(genstyles,name='style'),pd.DataFrame(g,columns=cols)],axis=1).to_csv(OUT/'q3_generated_chemistry.csv',index=False)
    pd.concat([pd.Series(genstyles,name='style'),gp],axis=1).to_csv(OUT/'q3_generated_sensory.csv',index=False)
    pd.concat([pd.Series(genstyles,name='style'),pd.DataFrame(raw,columns=cols)],axis=1).to_csv(OUT/'q3_generated_before_projection.csv',index=False)
    defectstats=[]
    for c,d in rules.items():
        i=d['index'];t=d['threshold'];before=10**raw[:,i] if d['transform']=='log10' else raw[:,i];after=10**g[:,i] if d['transform']=='log10' else g[:,i]
        defectstats.append({'compound':c,'threshold':t,'unit':d['unit'],'raw_exceedance':float(np.mean(before>t)) if t else None,'projected_exceedance':float(np.mean(after>t)) if t else None,'projection_fraction':np.mean(raw[:,i]!=g[:,i])})
    pd.DataFrame(defectstats).to_csv(OUT/'q3_defect_checks.csv',index=False)
    # Transfer selection uses chemistry-trained classifier plus sensory proxy constraint.
    pairs=[('Pils/Lager','Tripel'),('Tripel','Lambic'),('Blond','Kriek'),('Wheat','Saison'),('Stout/Porter','Dubbel')]
    transfer=[];spectra=[];stale=Y.columns.get_loc('stale_hops')
    filled=imp.transform(X)
    for source,target in pairs:
        k=le.transform([target])[0]
        for idx in np.flatnonzero(s.to_numpy()==source):
            candidates=decode(m,mus[idx][None,:]+rng.normal(0,.25,(24,mus.shape[1])),np.tile(np.eye(len(le.classes_))[k],(24,1)))
            a=project(np.clip(sc.inverse_transform(candidates),lo,hi),rules,filled[idx])
            source_stale=proxy.predict(pd.DataFrame(filled[idx][None,:],columns=cols))[0,stale]
            candidate_stale=proxy.predict(pd.DataFrame(a,columns=cols))[:,stale]
            feasible=candidate_stale<=source_stale+1e-12
            prob=clf.predict_proba(sc.transform(a))[:,k]
            fallback=not feasible.any()
            if fallback:
                # Identity fallback satisfies non-worsening; it is recorded as failed transfer.
                best=filled[idx].copy();chosen_prob=float(clf.predict_proba(sc.transform(best[None,:]))[0,k])
            else:
                j=np.argmax(np.where(feasible,prob,-np.inf));best=a[j];chosen_prob=float(prob[j])
            final_stale=float(proxy.predict(pd.DataFrame(best[None,:],columns=cols))[0,stale])
            success=int(clf.predict(sc.transform(best[None,:]))[0]==k and not fallback)
            violations=sum(best[d['index']]>filled[idx,d['index']]+1e-12 for d in rules.values())+int(final_stale>source_stale+1e-12)
            row={'beer':int(ids.iloc[idx].beer),'source_style':source,'target_style':target,'fallback_identity':fallback,'classifier_target_probability':chosen_prob,'target_class_success':success,'nonworsening_violations':violations,'source_stale_proxy':source_stale,'target_stale_proxy':final_stale}
            transfer.append(row);spectra.append({**row,**dict(zip(cols,best))})
    pd.DataFrame(transfer).to_csv(OUT/'q3_transfer_checks.csv',index=False);pd.DataFrame(spectra).to_csv(OUT/'q3_transfer_chemistry.csv',index=False)
    # Three noise levels, fixed fitted generator, explicit quality/diversity tradeoff.
    sensitivity=[]
    for noise in [0,.2,.5]:
        a=[];lab=[]
        for k in range(len(le.classes_)):
            idx=rng.choice(np.flatnonzero(labels==k),30,replace=True);c=np.tile(np.eye(len(le.classes_))[k],(30,1))
            aa=project(np.clip(sc.inverse_transform(decode(m,mus[idx]+rng.normal(0,noise,(30,mus.shape[1])),c)),lo,hi),rules);a.append(aa);lab += [k]*30
        aa=np.vstack(a);sensitivity.append({'latent_noise':noise,'classifier_agreement':accuracy_score(lab,clf.predict(sc.transform(aa))),'mean_standardized_variance':np.var(sc.transform(aa),axis=0).mean()})
    pd.DataFrame(sensitivity).to_csv(OUT/'q3_noise_sensitivity.csv',index=False)
    torch.save({'state_dict':m.state_dict(),'feature_columns':cols,'style_classes':le.classes_.tolist(),'latent':8,'epochs':epochs},OUT/'q3_cvae.pt');joblib.dump({'imputer':imp,'scaler':sc},OUT/'q3_preprocessing.joblib')
    ext=audit_external();rating_study(ext);external=compare_external(gp,genstyles)
    info={'epochs':epochs,'generated_n':len(g),'generated_classifier_agreement':accuracy_score(genlabels,clf.predict(sc.transform(g))),'real_classifier_oof_accuracy':accuracy_score(labels,classifier_oof),'heldout_reconstruction':pd.DataFrame(validation).groupby('model')[['standardized_mse','r2']].mean().to_dict('index'),'transfer_n':len(transfer),'identity_fallback_n':int(pd.DataFrame(transfer).fallback_identity.sum()),'transfer_classifier_success_rate':pd.DataFrame(transfer).target_class_success.mean(),'nonworsening_violations':int(pd.DataFrame(transfer).nonworsening_violations.sum()),'external':external,'limitations':['unknown transformations retained in supplied units then standardized','classification is internal consistency, not independent chemical validation','identity fallbacks are failures of style transfer','stale_hops constraint is only proxy-based','unknown trans-2-nonenal absolute threshold not invented']}
    dump(info,'q3_summary.json');dump(rules,'q3_defect_rules.json')
    plt=plotting();fig,axes=plt.subplots(1,2,figsize=(9,4));pd.DataFrame(validation).groupby('model').standardized_mse.mean().plot.bar(ax=axes[0],color='#337a96');axes[0].set_title('三折留出重构误差');axes[0].set_ylabel('标准化 MSE');axes[0].tick_params(axis='x',rotation=0);pd.DataFrame(transfer).groupby(['source_style','target_style']).target_class_success.mean().plot.barh(ax=axes[1],color='#be8345');axes[1].set_title('迁移后的目标风格识别率');axes[1].set_ylabel('');fig.tight_layout();fig.savefig(FIG/'q3_validation.png');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=1200);a=p.parse_args();main(a.epochs)

```

## validate_outputs.py

```python

"""Meaningful data and delivery checks; fails on a violated invariant."""
from common import *
import joblib
from pypdf import PdfReader

def main():
    checks=[]
    def check(name,condition):
        checks.append({'check':name,'pass':bool(condition)})
        if not condition:raise AssertionError(name)
    X,Y,s,ids=load()
    grade=pd.read_csv(OUT/'q1_grades_0_6.csv');vals=grade[Y.columns].to_numpy()
    check('250 samples and 50 grade targets',vals.shape==(250,50))
    check('all grades finite integers in 0..6',np.isfinite(vals).all() and ((vals>=0)&(vals<=6)).all() and np.equal(vals,np.rint(vals)).all())
    check('fold-local OOF outputs aligned',pd.read_csv(OUT/'q1_oof_predictions.csv').beer.equals(ids.beer))
    top=pd.read_csv(OUT/'q2_top3.csv');check('exactly 3 distinct factors per target',len(top)==150 and top.groupby('target').compound.nunique().eq(3).all())
    check('distinct graph modules per target',top.groupby('target').module.nunique().eq(3).all())
    check('bootstrap frequencies within [0,1]',top.bootstrap_frequency.between(0,1).all())
    g=pd.read_csv(OUT/'q3_generated_chemistry.csv');check('22 styles with 60 generated samples',len(g)==1320 and g.groupby('style').size().eq(60).all())
    check('generated values finite',np.isfinite(g[list(X)].to_numpy()).all())
    t=pd.read_csv(OUT/'q3_transfer_checks.csv');check('no reported non-worsening violations',t.nonworsening_violations.sum()==0)
    rules=json.loads((OUT/'q3_defect_rules.json').read_text(encoding='utf-8'))
    actual=pd.read_csv(OUT/'q3_transfer_chemistry.csv');sources=X.copy();sources.index=ids.beer
    for col,r in rules.items():
        source_values=sources.loc[actual.beer,col].to_numpy();known=np.isfinite(source_values)
        check('observed-source transfer non-worsening independently: '+col,np.all(actual[col].to_numpy()[known]<=source_values[known]+1e-10))
        check('imputed-source transfer non-worsening independently: '+col,np.all(actual[col].to_numpy()<=np.where(known,source_values,X[col].median())+1e-10))
        if r['threshold'] is not None:
            c=10**g[col] if r['transform']=='log10' else g[col]
            check('generated threshold independently: '+col,np.all(c<=r['threshold']+1e-10))
    m=joblib.load(OUT/'q1_model.joblib');stale=Y.columns.get_loc('stale_hops')
    source_x=sources.loc[actual.beer].copy();source_x=source_x.fillna(X.median())
    src=m.predict(source_x)[:,stale];dst=m.predict(actual[list(X)])[:,stale]
    check('stale-hop proxy non-worsening independently',np.all(dst<=src+1e-10))
    check('identity fallback not counted successful',t.loc[t.fallback_identity,'target_class_success'].sum()==0)
    e=pd.read_csv(OUT/'external_style_mapping.csv');check('unmapped styles retained explicitly',e.confidence.eq('unmapped').any())
    v=pd.read_csv(OUT/'external_vocabulary.csv');check('all three vocabulary sheets parsed',v.sheet.nunique()==3)
    a=pd.read_csv(OUT/'external_agreement.csv');check('both broad and conservative external comparisons',set(a.subset)=={'all_mapped','closer_only'})
    pdf=ROOT/'output/pdf/啤酒风味建模与外部数据验证论文.pdf'
    if pdf.exists():
        reader=PdfReader(pdf);txt='\n'.join(p.extract_text() for p in reader.pages)
        check('paper has all three model chapters',all(h in txt for h in ['问题一的模型','问题二的模型','问题三的模型']))
        check('paper has external analysis and limitations','外部数据的补充分析' in txt and '不足' in txt)
        check('no obsolete drive paths in paper','D:\\github mama' not in txt)
        dump({'pages':len(reader.pages),'characters':len(txt)},'pdf_validation.json')
    pd.DataFrame(checks).to_csv(OUT/'validation_checks.csv',index=False,encoding='utf-8-sig');print('PASS',len(checks))

if __name__=='__main__':main()

```