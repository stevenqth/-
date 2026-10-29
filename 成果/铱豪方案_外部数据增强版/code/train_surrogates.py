"""Nested 3x3 CV; fold-local winsorization and target-wise model selection.
Selection is repeated inside each outer training fold. No Kaggle holdout is used.
"""
from common import *
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
threadpool_limits(4)
NAMES=['Ridge1000','ET3','ET8','RF5']

class Regressor:
    def __init__(self,name,seed=42):self.name=name;self.seed=seed
    def transform(self,x):
        x=np.asarray(x);x=np.where(np.isfinite(x),x,self.median)
        return (np.clip(x,self.lo,self.hi)-self.mu)/self.sd
    def fit(self,x,y):
        x=np.asarray(x);y=np.asarray(y);self.median=np.nanmedian(x,axis=0)
        a=np.where(np.isfinite(x),x,self.median);self.lo,self.hi=np.quantile(a,[.01,.99],axis=0)
        a=np.clip(a,self.lo,self.hi);self.mu=a.mean(0);self.sd=a.std(0,ddof=1);self.sd[self.sd<1e-12]=1
        self.ym=y.mean(0);self.ys=y.std(0,ddof=1);self.ys[self.ys<1e-12]=1
        if self.name=='Ridge1000':self.model=Ridge(alpha=1000)
        elif self.name.startswith('ET'):self.model=ExtraTreesRegressor(n_estimators=160,max_features='sqrt',min_samples_leaf=int(self.name[2:]),n_jobs=1,random_state=self.seed)
        else:self.model=RandomForestRegressor(n_estimators=160,max_features='sqrt',min_samples_leaf=5,n_jobs=1,random_state=self.seed)
        self.model.fit(self.transform(x),(y-self.ym)/self.ys);return self
    def predict(self,x):return self.model.predict(self.transform(x))*self.ys+self.ym

class TargetSelector:
    def fit(self,x,y,styles,seed=42):
        x=np.asarray(x);y=np.asarray(y);self.inner=np.zeros((4,*y.shape))
        inner_folds=min(3,int(pd.Series(styles).value_counts().min()))
        assert inner_folds>=2
        for fold,(tr,va) in enumerate(StratifiedKFold(inner_folds,shuffle=True,random_state=seed).split(x,styles)):
            for k,name in enumerate(NAMES):self.inner[k,va]=Regressor(name,seed+fold).fit(x[tr],y[tr]).predict(x[va])
        mse=np.mean((self.inner-y[None,:,:])**2,axis=1);self.selected=np.argmin(mse,axis=0)
        self.models=[Regressor(name,seed).fit(x,y) for name in NAMES]
        self.inner_mse=mse;return self
    def all_predictions(self,x):return np.array([m.predict(x) for m in self.models])
    def predict(self,x):
        p=self.all_predictions(x);return np.column_stack([p[self.selected[j],:,j] for j in range(p.shape[2])])
    def predict_one(self,x,j):return self.models[self.selected[j]].predict(x)[:,j]

def main():
    X,Y,ids=load();consumer=pd.read_csv(OUT/'official_consumer_targets.csv').set_index('beer').loc[ids.beer].reset_index()
    cs=[c for c in consumer if c not in META];target=pd.concat([Y,consumer[cs].rename(columns={c:'consumer_'+c for c in cs})],axis=1)
    x=X.to_numpy();y=target.to_numpy();s=ids.tasting_category_fine.to_numpy();stale=list(Y).index('stale_hops')
    base=np.zeros_like(y);pred=np.zeros_like(y);outer=[];selection=[]
    for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=2026).split(x,s)):
        model=TargetSelector().fit(x[tr],y[tr],s[tr],42+fold)
        pred[va]=model.predict(x[va]);base[va]=model.models[0].predict(x[va])
        for name,p in [('yihao_clipped_ridge',base[va]),('nested_target_selector',pred[va])]:
            outer.append({'fold':fold,'model':name,'expert_r2':r2_score(y[va,:50],p[:,:50]),'expert_mae':mean_absolute_error(y[va,:50],p[:,:50]),'consumer_r2':r2_score(y[va,50:],p[:,50:]),'stale_r2':r2_score(y[va,stale],p[:,stale])})
        for j,t in enumerate(target):selection.append({'fold':fold,'target':t,'selected_model':NAMES[model.selected[j]]})
        print('Nested outer fold',fold,'finished',flush=True)
    pd.DataFrame(outer).to_csv(OUT/'nested_cv_folds.csv',index=False);pd.DataFrame(selection).to_csv(OUT/'nested_selections.csv',index=False)
    pd.concat([ids,pd.DataFrame(pred,columns=target.columns)],axis=1).to_csv(OUT/'nested_oof.csv',index=False)
    pd.concat([ids,pd.DataFrame(base,columns=target.columns)],axis=1).to_csv(OUT/'baseline_oof.csv',index=False)
    per=[]
    for j,t in enumerate(target):
        for name,p in [('baseline',base),('optimized',pred)]:per.append({'target':t,'model':name,'r2':r2_score(y[:,j],p[:,j]),'mae':mean_absolute_error(y[:,j],p[:,j])})
    pd.DataFrame(per).to_csv(OUT/'per_target_cv.csv',index=False)
    final=TargetSelector().fit(x,y,s);joblib.dump(final,MODELS/'surrogates.joblib')
    pd.DataFrame({'target':list(target),'selected_model':[NAMES[k] for k in final.selected]}).to_csv(OUT/'final_model_selection.csv',index=False)
    # Dependence-aware paired uncertainty: resample styles, preserving products within style.
    rng=np.random.default_rng(42);groups=[np.flatnonzero(s==v) for v in np.unique(s)];bs=[]
    for _ in range(800):
        idx=np.concatenate([groups[k] for k in rng.integers(0,len(groups),len(groups))]);bs.append(mean_absolute_error(y[idx,:50],base[idx,:50])-mean_absolute_error(y[idx,:50],pred[idx,:50]))
    summary={}
    for name,p in [('baseline',base),('optimized',pred)]:summary[name]={'expert_oof_r2':r2_score(y[:,:50],p[:,:50]),'expert_mae':mean_absolute_error(y[:,:50],p[:,:50]),'consumer_oof_r2':r2_score(y[:,50:],p[:,50:]),'stale_oof_r2':r2_score(y[:,stale],p[:,stale])}
    summary['expert_mae_gain_style_bootstrap_ci']=np.quantile(bs,[.025,.975]);summary['cv']='3 outer stratified folds; inner folds adapt to min style count (2 in outer training; 3 on full data); fixed candidate set'
    summary['feature_columns']=list(X);summary['target_columns']=list(target);summary['consumer_dimensions']=cs
    summary['limitations']=['small data; one nested partition','model agreement is not a calibrated statistical bound','consumer targets are overlapping dictionary aggregates, not raw counts']
    save(summary,'surrogate_summary.json');print(json.dumps(summary,default=str,ensure_ascii=True)[:900],flush=True)

if __name__=='__main__':
    # Stable importable classes in saved joblib files.
    from train_surrogates import main as run
    run()
