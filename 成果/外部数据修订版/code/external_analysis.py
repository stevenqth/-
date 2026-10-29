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
