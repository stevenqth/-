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
