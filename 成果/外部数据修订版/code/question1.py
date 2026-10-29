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

