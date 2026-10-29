"""External breweries are disjoint between optimization and held-out evaluation.
The vocabulary aggregates original standardized descriptors, not chemical labels.
"""
from common import *
from sklearn.model_selection import GroupShuffleSplit

FLAVORS=['Astringency','Body','Alcohol','Bitter','Sweet','Sour','Salty','Fruits','Hoppy','Spices','Malty']
STYLE={'Tripel':['Tripel'],'Dubbel':['Dubbel'],'Faro':['Lambic - Faro'],'Lambic':['Lambic - Traditional','Lambic - Gueuze'],'Saison':['Farmhouse Ale - Saison'],'Blond':['Blonde Ale - Belgian'],'Flanders old brown':['Sour - Flanders Oud Bruin'],'Brut':['Bière de Champagne / Bière Brut'],'Low/No alcohol':['Low Alcohol Beer'],'Scotch':['Scotch Ale / Wee Heavy','Scottish Ale'],'Pils/Lager':['Pilsner - Bohemian / Czech','Pilsner - German','Lager - European Pale','Lager - Helles','Lager - American'],'Stout/Porter':['Stout - American','Stout - English','Stout - Irish Dry','Stout - Sweet / Milk','Porter - American','Porter - English','Porter - Robust'],'Wheat':['Wheat Beer - Witbier','Wheat Beer - Hefeweizen'],'Brown':['Brown Ale - Belgian Dark','Brown Ale - English','Brown Ale - American'],'Fruitbeer':['Fruit and Field Beer'],'Strong ale':['Strong Ale - Belgian Dark','Strong Ale - Belgian Pale']}

def ecdf(ref,values):
    # Midrank CDF, interpolated so the optimization landscape is continuous.
    ref=np.asarray(ref);values=np.atleast_2d(values);out=np.empty(values.shape)
    for j in range(ref.shape[1]):
        col=ref[:,j];v=np.sort(col[np.isfinite(col)])
        unique,idx,count=np.unique(v,return_index=True,return_counts=True)
        quantiles=(idx+.5*count)/len(v)
        out[:,j]=np.interp(values[:,j],unique,quantiles,left=0.,right=1.)
    return out

def main():
    e=pd.read_csv(EXT/'beer_data_set.csv');assert e.key.is_unique
    train,test=next(GroupShuffleSplit(1,test_size=.35,random_state=42).split(e,groups=e.Brewery))
    assert not set(e.iloc[train].Brewery)&set(e.iloc[test].Brewery)
    split=np.repeat('optimization',len(e));split[test]='heldout'
    pd.DataFrame({'key':e.key,'brewery':e.Brewery,'split':split}).to_csv(OUT/'external_split.csv',index=False)
    vocabulary=[]
    aliases={'Astringent':'Astringency','Alcoholic':'Alcohol','Fruity':'Fruits'}
    for sheet,d in pd.read_excel(EXT/'Beer Descriptors Simplified.xlsx',sheet_name=None,header=None).items():
        for j in range(0,d.shape[1],2):
            category=aliases.get(d.iloc[0,j],d.iloc[0,j])
            for i in range(1,len(d)):
                if pd.notna(d.iloc[i,j]):vocabulary.append({'sheet':sheet,'category':category,'word':str(d.iloc[i,j]).strip().lower(),'impact':d.iloc[i,j+1]})
    v=pd.DataFrame(vocabulary);v.to_csv(OUT/'dictionary.csv',index=False)
    X,Y,ids=load();d7=pd.read_csv(DATA/'7.csv').set_index('beer').loc[ids.beer].reset_index()
    mapping=[];direct={'fruity':'Fruits','hoppy':'Hoppy','malt':'Malty','sweet':'Sweet','sour':'Sour','bitter':'Bitter','alcohol':'Alcohol'}
    for c in d7.columns:
        if c in META:continue
        word=c.split('_',1)[-1];cats=set(v.loc[v.word==word,'category'])
        if word in direct:cats.add(direct[word])
        for cat in sorted(cats):mapping.append({'column':c,'word':word,'category':cat,'overlap':len(cats)>1})
    pd.DataFrame(mapping).to_csv(OUT/'dictionary_mapping.csv',index=False)
    cats=[c for c in FLAVORS if any(m['category']==c for m in mapping)]
    consumer=pd.DataFrame({c:d7[sorted({m['column'] for m in mapping if m['category']==c})].mean(axis=1) for c in cats})
    assert np.isfinite(consumer).all().all()
    pd.concat([ids,consumer],axis=1).to_csv(OUT/'official_consumer_targets.csv',index=False)
    a=e[FLAVORS].to_numpy(float);sums=a.sum(1);shares=np.divide(a,sums[:,None],out=np.full_like(a,np.nan),where=sums[:,None]>0)
    ranks=ecdf(shares[train][:,[FLAVORS.index(c) for c in cats]],shares[:,[FLAVORS.index(c) for c in cats]])
    ranks[sums==0]=np.nan
    rng=np.random.default_rng(42);profiles=[]
    for st,names in STYLE.items():
        for side in ['optimization','heldout']:
            ix=np.flatnonzero(e.Style.isin(names).to_numpy()&(split==side)&(sums>0))
            assert len(ix)>0,(st,side)
            names_b=e.iloc[ix].Brewery.to_numpy();unique=np.unique(names_b)
            groups=[ranks[ix[names_b==b]] for b in unique]
            boot=[]
            for _ in range(400):
                sampled=rng.integers(0,len(groups),len(groups));boot.append(np.vstack([groups[k] for k in sampled]).mean(0))
            lo,hi=np.quantile(boot,[.025,.975],axis=0);mean=ranks[ix].mean(0)
            for j,c in enumerate(cats):profiles.append({'style':st,'split':side,'dimension':c,'mean_percentile':mean[j],'ci_low':lo[j],'ci_high':hi[j],'products':len(ix),'breweries':len(unique)})
    pd.DataFrame(profiles).to_csv(OUT/'external_profiles.csv',index=False)
    save({'rows':len(e),'styles':e.Style.nunique(),'optimization_rows':len(train),'heldout_rows':len(test),'zero_word_count_rows':int((sums==0).sum()),'disjoint_breweries':True,'dimensions':cats,'vocabulary_entries':len(v),'mapped_original_columns':len({m['column'] for m in mapping}),'style_mapping':STYLE,'scope':'external split is independent across breweries; cross-domain semantic matching remains an assumption; selection-biased high-rated beer sample'},'external_audit.json')
    rows=[]
    for p in sorted((ROOT/'data').rglob('*')):
        if p.is_file():rows.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    pd.DataFrame(rows).to_csv(OUT/'input_manifest.csv',index=False)
    print('Prepared external data and',len(cats),'consumer targets',flush=True)

if __name__=='__main__':main()
