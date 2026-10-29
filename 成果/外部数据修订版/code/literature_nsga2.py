"""L5: NSGA-II with constrained domination, SBX, polynomial mutation.

Vectorized evaluations; feasibility always precedes objective ranking.
"""
import numpy as np


def ranks_and_crowding(F,violation):
    F=np.asarray(F,float);v=np.maximum(np.asarray(violation,float),0)
    feasible=v<=1e-8
    pareto=np.all(F[:,None,:]<=F[None,:,:],2)&np.any(F[:,None,:]<F[None,:,:],2)
    dominates=(feasible[:,None]&~feasible[None,:]) | (feasible[:,None]&feasible[None,:]&pareto) | (
        ~feasible[:,None]&~feasible[None,:]&(v[:,None]<v[None,:]))
    count=dominates.sum(0);rank=np.full(len(F),-1,int);front=np.flatnonzero(count==0);k=0
    crowd=np.zeros(len(F))
    while len(front):
        rank[front]=k
        if len(front)<=2:crowd[front]=np.inf
        else:
            for j in range(F.shape[1]):
                order=front[np.argsort(F[front,j],kind='stable')]
                span=F[order[-1],j]-F[order[0],j]
                if span<=1e-14:continue
                crowd[order[[0,-1]]]=np.inf
                crowd[order[1:-1]]+=(F[order[2:],j]-F[order[:-2],j])/span
        count-=dominates[front].sum(0);count[front]=-1
        front=np.flatnonzero((count==0)&(rank<0));k+=1
    assert np.all(rank>=0)
    return rank,crowd


def offspring(U,rank,crowd,rng,lo,hi):
    n,d=U.shape;children=[]
    def tournament():
        a,b=rng.integers(n,size=2)
        return a if rank[a]<rank[b] or (rank[a]==rank[b] and crowd[a]>=crowd[b]) else b
    for _ in range(n//2):
        p,q=U[tournament()],U[tournament()]
        u=rng.random(d);beta=np.where(u<=.5,(2*u)**(1/16),(1/(2*(1-u)))**(1/16))
        mask=rng.random(d)<.5;beta=np.where(mask,beta,1)
        for child in [.5*((1+beta)*p+(1-beta)*q),.5*((1-beta)*p+(1+beta)*q)]:
            child=np.clip(child,lo,hi)
            # Bounded polynomial mutation in normalized decision coordinates.
            z=(child-lo)/(hi-lo);sel=rng.random(d)<1/d;r=rng.random(d)
            delta=np.where(r<.5,(2*r+(1-2*r)*(1-z)**21)**(1/21)-1,
                            1-(2*(1-r)+2*(r-.5)*z**21)**(1/21))
            children.append(np.clip(child+sel*delta*(hi-lo),lo,hi))
    return np.array(children)


def solve(evaluate,lo,hi,seed=42,population=80,generations=100,initial=None):
    assert population%2==0
    rng=np.random.default_rng(seed);lo=np.asarray(lo);hi=np.asarray(hi)
    U=rng.uniform(lo,hi,(population,len(lo)))
    if initial is not None:U[:min(len(initial),population)]=np.clip(initial[:population],lo,hi)
    F,c=evaluate(U);trace=[]
    for generation in range(generations):
        rank,crowd=ranks_and_crowding(F,c)
        kids=offspring(U,rank,crowd,rng,lo,hi);kf,kc=evaluate(kids)
        both=np.vstack([U,kids]);bf=np.vstack([F,kf]);bc=np.r_[c,kc]
        br,bd=ranks_and_crowding(bf,bc)
        idx=np.lexsort((-bd,br))[:population]
        U,F,c=both[idx],bf[idx],bc[idx]
        trace.append({'generation':generation+1,'feasible':int(np.sum(c<=1e-8)),
                      'best_f1_feasible':float(F[c<=1e-8,0].min()) if np.any(c<=1e-8) else None})
    rank,_=ranks_and_crowding(F,c);ok=(rank==0)&(c<=1e-8)
    return U[ok],F[ok],c[ok],trace


def self_check():
    F=np.array([[1,2],[2,1],[3,3],[0,0.]])
    r,_=ranks_and_crowding(F,[0,0,0,1])
    assert r[0]==r[1]==0 and r[2]>0 and r[3]>r[2]
    def zdt1(U):
        g=1+9*U[:,1:].mean(1)
        return np.column_stack([U[:,0],g*(1-np.sqrt(U[:,0]/g))]),np.zeros(len(U))
    u,f,c,_=solve(zdt1,np.zeros(3),np.ones(3),seed=7,population=60,generations=100)
    residual=np.abs(f[:,1]-(1-np.sqrt(f[:,0])))
    assert len(f)>15 and np.median(residual)<.03 and np.ptp(f[:,0])>.8
    return {'zdt1_n':len(f),'zdt1_median_front_residual':float(np.median(residual)),
            'feasibility_priority_fixture':True}
