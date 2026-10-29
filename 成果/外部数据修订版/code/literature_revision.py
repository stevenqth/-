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
