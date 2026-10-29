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

