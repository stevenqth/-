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
