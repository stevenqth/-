"""Python NSGA-II: constraint-domination, fast sorting, crowding, bounded
simulated binary crossover, polynomial mutation, elitist survival.
This is an actually executable Python optimizer, independent of MATLAB.
"""
import numpy as np

def fronts(f):
    dom=np.all(f[:,None,:]<=f[None,:,:],axis=2)&np.any(f[:,None,:]<f[None,:,:],axis=2)
    count=dom.sum(0);todo=np.ones(len(f),bool);result=[]
    while todo.any():
        current=np.flatnonzero(todo&(count==0));assert len(current)
        result.append(current);todo[current]=False;count-=dom[current].sum(0)
    return result

def crowding(f):
    n=len(f);d=np.zeros(n)
    if n<=2:return np.full(n,np.inf)
    for j in range(f.shape[1]):
        order=np.argsort(f[:,j],kind='stable');span=np.ptp(f[:,j])
        if span<=1e-14:continue
        d[order[[0,-1]]]=np.inf;d[order[1:-1]]+=(f[order[2:],j]-f[order[:-2],j])/span
    return d

def rank_crowd(f,cv):
    rank=np.zeros(len(f),int);crowd=np.zeros(len(f));feasible=np.flatnonzero(cv<=1e-8);r=0
    if len(feasible):
        for r,fr in enumerate(fronts(f[feasible])):
            ix=feasible[fr];rank[ix]=r;crowd[ix]=crowding(f[ix])
        r+=1
    bad=np.flatnonzero(cv>1e-8);order=bad[np.argsort(cv[bad],kind='stable')]
    rank[order]=r+np.arange(len(order));return rank,crowd

def survive(f,cv,n):
    r,c=rank_crowd(f,cv);return np.lexsort((-c,r))[:n]

def children(pop,rank,crowd,rng):
    n,d=pop.shape
    def tournament():
        a,b=rng.integers(0,n,(2,n));better=(rank[a]<rank[b])|((rank[a]==rank[b])&(crowd[a]>crowd[b]))
        return pop[np.where(better,a,b)].copy()
    p=tournament();q=np.roll(p,1,axis=0);lo=np.minimum(p,q);hi=np.maximum(p,q);delta=np.maximum(hi-lo,1e-14);u=rng.random((n,d));eta=15
    def beta(side):
        bb=1+2*side/delta;alpha=2-bb**(-(eta+1))
        return np.where(u<=1/alpha,(u*alpha)**(1/(eta+1)),(1/(2-u*alpha))**(1/(eta+1)))
    a=.5*(lo+hi-beta(lo)*delta);b=.5*(lo+hi+beta(1-hi)*delta)
    ch=np.where(rng.random((n,d))<.5,a,b)
    crossed=(rng.random((n,1))<.9)&(rng.random((n,d))<.5)&(hi-lo>1e-12)
    ch=np.clip(np.where(crossed,ch,p),0,1)
    eta=20;u=rng.random((n,d));power=1/(eta+1)
    dq=np.where(u<=.5,(2*u+(1-2*u)*(1-ch)**(eta+1))**power-1,1-(2*(1-u)+2*(u-.5)*ch**(eta+1))**power)
    ch=np.clip(ch+np.where(rng.random((n,d))<1/d,dq,0),0,1)
    return ch

def nsga2(evaluate,d,seed=42,population=80,generations=60,initial=None):
    rng=np.random.default_rng(seed);u=rng.random((population,d)) if initial is None else np.asarray(initial).copy()
    f,cv=evaluate(u);nfe=len(u);history=[]
    for generation in range(generations):
        rank,crowd=rank_crowd(f,cv);child=children(u,rank,crowd,rng);cf,cc=evaluate(child);nfe+=len(child)
        uu=np.vstack([u,child]);ff=np.vstack([f,cf]);vv=np.r_[cv,cc];keep=survive(ff,vv,population)
        u,f,cv=uu[keep],ff[keep],vv[keep]
        history.append({'generation':generation+1,'feasible':int((cv<=1e-8).sum()),'min_violation':float(cv.min()),'objective_minima':f.min(0).tolist()})
    return u,f,cv,history,nfe

def self_test():
    f=np.array([[1,2],[2,1],[3,3],[1,2]],float);assert set(fronts(f)[0])=={0,1,3}
    assert survive(np.array([[0,0],[1,1]]),np.array([.1,0]),1)[0]==1
    # Analytical ZDT1 front has f2=1-sqrt(f1), with all other decisions zero.
    def zdt(x):
        g=1+9*x[:,1:].mean(1);return np.c_[x[:,0],g*(1-np.sqrt(x[:,0]/g))],np.zeros(len(x))
    u,f,cv,h,n=nsga2(zdt,6,population=80,generations=120)
    error=np.mean(abs(f[:,1]-(1-np.sqrt(f[:,0]))));assert error<.06,error
    assert np.all((u>=0)&(u<=1));assert len(fronts(f)[0])>30
    return {'dominance_fixture':True,'feasibility_priority':True,'zdt1_mean_front_error':float(error),'zdt1_nondominated':len(fronts(f)[0]),'evaluations':n}

if __name__=='__main__':print(self_test())
