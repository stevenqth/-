"""Four-objective, six-decision constrained optimization.
Main objectives: expert style error, chemical adjustment, defect burden, external
consumer-profile error. The external heldout partition never enters the solver.
"""
from common import *
import argparse,joblib,importlib.util,time
import train_surrogates
from prepare_external import ecdf
from nsga2 import nsga2,fronts,survive,self_test
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
threadpool_limits(4)
DEFECTS=['DMS','X2.3.butanedione','X4VG','X.E..2.nonenal']
ISLOG=np.array([True,False,True,False])
OBJECTIVES=['expert_error','adjustment','defect_burden','external_error']
VARIABLES=['alpha','beta_DMS','beta_diacetyl','beta_4VG','beta_t2n','gamma']

def concentration(x):
    c=x.copy();c[...,ISLOG]=10**c[...,ISLOG];return c

class Problem:
    def __init__(self,source_beer,target_style):
        self.X,self.Y,self.ids=load();self.model=joblib.load(MODELS/'surrogates.joblib')
        self.features=list(self.X);self.meta=json.loads((OUT/'surrogate_summary.json').read_text(encoding='utf-8'))
        self.dimensions=self.meta['consumer_dimensions'];self.profiles=pd.read_csv(OUT/'external_profiles.csv')
        self.source_beer=source_beer;self.target_style=target_style
        self.source_row=int(np.flatnonzero(self.ids.beer==source_beer)[0]);self.source_style=self.ids.iloc[self.source_row].tasting_category_fine
        self.imputer=self.model.models[0];self.filled=self.X.fillna(self.X.median()).to_numpy()
        self.source=self.filled[self.source_row];self.target=np.median(self.filled[self.ids.tasting_category_fine==target_style],axis=0)
        self.targetY=self.Y.loc[self.ids.tasting_category_fine==target_style].mean().to_numpy()
        self.yScale=self.Y.std(ddof=1).to_numpy(copy=True);self.yScale[self.yScale<1e-12]=1
        self.stale=list(self.Y).index('stale_hops');self.di=np.array([self.features.index(c) for c in DEFECTS])
        self.source_defect_imputed=bool(self.X.iloc[self.source_row,self.di].isna().any())
        self.sourceC=concentration(self.source[self.di]);assert np.all(self.sourceC>0)
        self.cap=.95*np.minimum(self.sourceC,[17.25,.29925,.206333333,np.inf])
        self.refConsumer=pd.read_csv(OUT/'official_consumer_targets.csv')[self.dimensions].to_numpy()
        self.external_target=self.profile('optimization');self.external_holdout=self.profile('heldout')
        per=pd.read_csv(OUT/'per_target_cv.csv').query("model=='optimized'").set_index('target')
        reliability=np.array([max(.05,float(per.loc['consumer_'+c,'r2'])) for c in self.dimensions]);self.weights=reliability/reliability.sum()
        self.sourcePred=self.model.all_predictions(self.source[None,:])[:,0,:]
        self.guard_models=np.argsort(self.model.inner_mse[:,self.stale])[:3]
        z=self.imputer.transform(self.filled);self.neighbors=NearestNeighbors(n_neighbors=2).fit(z)
        distances=self.neighbors.kneighbors(z)[0][:,1]/np.sqrt(z.shape[1]);self.support_limit=float(np.quantile(distances,.95))
        # Choose one nuisance-control variable by finite perturbations, NOT a causal intervention.
        usable=np.array([j for j,c in enumerate(self.features) if j not in self.di and not c.endswith('_sum')])
        endpoints=np.tile(self.source,(len(usable)*2,1));row=np.arange(len(endpoints));col=np.repeat(usable,2)
        endpoints[row,col]=np.column_stack([self.imputer.lo[usable],self.imputer.hi[usable]]).ravel()
        pp=self.model.all_predictions(endpoints)[self.guard_models,:,self.stale]
        delta=(pp-self.sourcePred[self.guard_models,self.stale,None])/self.yScale[self.stale]
        chosen=int(np.argmin(np.max(delta,axis=0)));self.control=int(col[chosen]);self.control_bound=float(endpoints[chosen,self.control])
        self.case=f'{self.source_style}_beer{source_beer}_to_{target_style}'.replace('/','_').replace(' ','_')

    def profile(self,split):
        t=self.profiles.query('style==@self.target_style and split==@split').set_index('dimension')
        return t.loc[self.dimensions,'mean_percentile'].to_numpy()

    def evaluate(self,u,details=False):
        u=np.atleast_2d(u);v=self.source+u[:,0,None]*(self.target-self.source)
        c=np.minimum(concentration(v[:,self.di]),self.cap)*2**(-u[:,1:5])
        v[:,self.di]=c;v[:,self.di[ISLOG]]=np.log10(c[:,ISLOG])
        v[:,self.control]+=u[:,5]*(self.control_bound-v[:,self.control])
        p=self.model.all_predictions(v);yh=np.column_stack([p[self.model.selected[j],:,j] for j in range(p.shape[2])])
        pct=ecdf(self.refConsumer,yh[:,50:]);external=np.sum((pct-self.external_target)**2*self.weights,axis=1)
        f=np.c_[np.mean(((yh[:,:50]-self.targetY)/self.yScale)**2,axis=1),np.sqrt(np.mean(((v-self.source)/self.imputer.sd)**2,axis=1)),np.mean(c/self.cap,axis=1),external]
        delta=(p[self.guard_models,:,self.stale]-self.sourcePred[self.guard_models,self.stale,None])/self.yScale[self.stale]
        guard=np.max(delta,axis=0)
        support=self.neighbors.kneighbors(self.imputer.transform(v),n_neighbors=1)[0][:,0]/np.sqrt(v.shape[1])
        cv=np.maximum.reduce([guard,support/self.support_limit-1,np.zeros(len(v))])
        if details:
            return f,cv,{'chemistry':v,'predictions':yh,'consumer_percentile':pct,'stale_worst_delta':guard,'support_distance':support,'holdout_external_error':np.sum((pct-self.external_holdout)**2*self.weights,axis=1),'concentrations':c}
        return f,cv

def topsis(f,weights):
    lo=f.min(0);span=np.ptp(f,axis=0);active=span>1e-12;z=np.zeros_like(f);z[:,active]=(f[:,active]-lo[active])/span[active]
    w=np.asarray(weights,dtype=float)*active
    if w.sum()==0:return np.zeros(len(f)),0
    w/=w.sum();dp=np.sqrt(np.sum(z*z*w,axis=1));dm=np.sqrt(np.sum((z-active)**2*w,axis=1));score=np.divide(dm,dp+dm,out=np.full_like(dm,.5),where=dp+dm>0)
    return score,int(np.argmax(score))

def metric_hv(front,lo,hi,points):
    # Shared fixed Monte Carlo integration box for comparisons within a case.
    if not len(front):return 0.
    z=(front-lo)/(hi-lo);dominated=np.zeros(len(points),bool)
    for start in range(0,len(z),40):dominated|=np.any(np.all(z[start:start+40,None,:]<=points[None,:,:],axis=2),axis=0)
    return float(dominated.mean())

def solve_case(problem,population,generations,seeds,ablation=False,only_variants=None):
    start=time.time();p=problem;root=OUT/p.case;root.mkdir(exist_ok=True)
    config={'source_beer':p.source_beer,'source_style':p.source_style,'target_style':p.target_style,'source_defect_imputed':p.source_defect_imputed,'control_feature':p.features[p.control],'control_bound':p.control_bound,'defect_caps':p.cap,'guard_models':[train_surrogates.NAMES[k] for k in p.guard_models],'support_limit':p.support_limit,'population':population,'generations':generations,'seeds':seeds,'dimensions':p.dimensions,'external_weights':p.weights}
    (root/'case_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
    variants=only_variants or (['four_objective','no_external','shared_defect_knob'] if ablation else ['four_objective','shared_defect_knob'])
    archives={};runrows=[]
    for variant in variants:
        us=[]
        def evaluate(u):
            if variant=='shared_defect_knob':u=np.c_[u[:,0],np.repeat(u[:,1,None],4,axis=1),u[:,2]]
            f,cv=p.evaluate(u)
            return f[:,:3] if variant=='no_external' else f,cv
        d=3 if variant=='shared_defect_knob' else 6
        for seed in seeds:
            rng=np.random.default_rng(seed);pool=rng.random((600,d));pf,pc=evaluate(pool);initial=pool[survive(pf,pc,population)]
            u,f,cv,h,nfe=nsga2(evaluate,d,seed,population,generations,initial)
            if d==3:u=np.c_[u[:,0],np.repeat(u[:,1,None],4,axis=1),u[:,2]]
            us.append(u[cv<=1e-8]);runrows.append({'variant':variant,'seed':seed,'evaluations':nfe+600,'feasible_final':int((cv<=1e-8).sum()),'min_violation':float(cv.min())})
            pd.DataFrame(h).to_json(root/f'{variant}_seed{seed}_history.json',orient='records',indent=2)
            print('Solved',p.case,variant,seed,'feasible',int((cv<=1e-8).sum()),flush=True)
        merged=np.vstack(us)
        if not len(merged):archives[variant]=(np.empty((0,6)),np.empty((0,4)));continue
        merged=np.unique(merged,axis=0);f,cv=p.evaluate(merged);ix=fronts(f if variant!='no_external' else f[:,:3])[0];archives[variant]=(merged[ix],f[ix])
    # Uniform random benchmark under the same six decisions and guards, at equal total budget.
    if ablation:
        total=len(seeds)*(600+population*(generations+1));u=np.random.default_rng(7).random((total,6));f,cv=p.evaluate(u);u=u[cv<=1e-8];f=f[cv<=1e-8]
        ix=fronts(f)[0] if len(f) else [];archives['random_equal_budget']=(u[ix],f[ix])
        runrows.append({'variant':'random_equal_budget','seed':7,'evaluations':total,'feasible_final':len(u),'min_violation':float(cv.min())})
    runframe=pd.DataFrame(runrows)
    if only_variants and (root/'runs.csv').exists():
        old=pd.read_csv(root/'runs.csv');runframe=pd.concat([old[~old.variant.isin(variants)],runframe],ignore_index=True)
    runframe.to_csv(root/'runs.csv',index=False)
    summaries=[];recommendations=[]
    for variant,(u,f) in archives.items():
        if not len(u):summaries.append({'case':p.case,'variant':variant,'pareto_points':0,'status':'no feasible candidate'});continue
        f,cv,det=p.evaluate(u,True)
        frame=pd.DataFrame(np.c_[u,f,cv,det['stale_worst_delta'],det['support_distance'],det['holdout_external_error']],columns=VARIABLES+OBJECTIVES+['constraint_violation','worst_model_stale_delta','support_distance','heldout_external_error'])
        frame.to_csv(root/f'{variant}_pareto.csv',index=False)
        for scenario,w in {'balanced':[1,1,1,1],'sensory_first':[.55,.15,.15,.15],'small_change':[.15,.55,.15,.15],'external_first':[.15,.15,.15,.55]}.items():
            if variant=='no_external':w=[1,1,1,0] if scenario=='balanced' else [w[0],w[1],w[2],0]
            score,i=topsis(f,w)
            recommendations.append({'case':p.case,'variant':variant,'scenario':scenario,**frame.iloc[i].to_dict(),'topsis':score[i]})
            pd.DataFrame([det['chemistry'][i]],columns=p.features).to_csv(root/f'{variant}_{scenario}_chemistry.csv',index=False)
            pd.DataFrame([det['predictions'][i]],columns=p.meta['target_columns']).to_csv(root/f'{variant}_{scenario}_predictions.csv',index=False)
        summaries.append({'case':p.case,'variant':variant,'pareto_points':len(u),'status':'feasible','best_expert_error':f[:,0].min(),'best_external_error':f[:,3].min(),'best_heldout_external_error':det['holdout_external_error'].min(),'max_stale_delta':det['stale_worst_delta'].max(),'max_support_ratio':(det['support_distance']/p.support_limit).max()})
    # Heldout best values are retrospective descriptors; recommendations above were fixed without them.
    if ablation and any(len(f) for _,f in archives.values()):
        ff=np.vstack([f for _,f in archives.values() if len(f)]);lo=ff.min(0);hi=ff.max(0);hi=lo+np.maximum(hi-lo,1e-8)*1.1
        points=np.random.default_rng(42).random((12000,4))
        for row in summaries:row['common_box_mc_hypervolume']=metric_hv(archives[row['variant']][1],lo,hi,points)
    recframe=pd.DataFrame(recommendations)
    if only_variants and (root/'recommendations.csv').exists():
        old=pd.read_csv(root/'recommendations.csv');recframe=pd.concat([old[~old.variant.isin(variants)],recframe],ignore_index=True)
    recframe.to_csv(root/'recommendations.csv',index=False)
    save({'case':p.case,'elapsed_seconds':time.time()-start,'variants':summaries},p.case+'_summary.json')
    return summaries,recommendations

def main(population=64,generations=50):
    checks=self_test();save(checks,'optimizer_self_checks.json')
    X,Y,ids=load();m=joblib.load(MODELS/'surrogates.joblib');filled=X.fillna(X.median()).to_numpy();scale=m.models[0].sd
    def medoid(style):
        ix=np.flatnonzero(ids.tasting_category_fine==style);a=filled[ix];return int(ids.iloc[ix[np.argmin(np.mean(((a-a.mean(0))/scale)**2,axis=1))]].beer)
    pils=ids.loc[ids.tasting_category_fine=='Pils/Lager','beer'].tolist();other=[b for b in pils if b!=218][:2]
    cases=[(218,'Tripel')]+[(int(b),'Tripel') for b in other]+[(medoid('Wheat'),'Saison'),(medoid('Stout/Porter'),'Dubbel')]
    summaries=[];rec=[]
    for i,(beer,target) in enumerate(cases):
        p=Problem(beer,target);a,b=solve_case(p,population,generations,[42,43],ablation=i==0);summaries+=a;rec+=b
    pd.DataFrame(summaries).to_csv(OUT/'optimization_summary.csv',index=False);pd.DataFrame(rec).to_csv(OUT/'all_recommendations.csv',index=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--population',type=int,default=64);ap.add_argument('--generations',type=int,default=50);args=ap.parse_args();main(args.population,args.generations)
