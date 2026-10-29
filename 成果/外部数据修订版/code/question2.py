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

