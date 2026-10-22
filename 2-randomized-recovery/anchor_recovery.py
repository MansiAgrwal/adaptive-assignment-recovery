"""Joint measurement/transition recovery for randomized diagnostic episodes.

Independent people, one episode/person. Pre/post: three independent items of
each requirement A/B/AB. General response probabilities allow partial mastery.
All measurement probabilities, initial state weights and joint transition rows
are estimated. This is an identifiable reference design, NOT validation of
the full adaptive longitudinal experiment. No empirical data are used.
"""
import argparse, json, time
from pathlib import Path
import numpy as np

STATES=np.array([[0,0],[1,0],[0,1],[1,1]])
# Eight response parameters: two for A, two for B, four for AB.
MAP=np.array([[0,1,0,1],[2,2,3,3],[4,5,6,7]])
TRUE_R=np.array([.25,.90,.22,.88,.12,.48,.44,.84])

def kernel(lam, loss):
    out=np.zeros((3,4,4))
    gains=np.array([[lam[0],0],[0,lam[1]],[lam[2],lam[3]]])
    for e in range(3):
        for s,x in enumerate(STATES):
            u=(1-x)*gains[e]+x*(1-np.array(loss))
            out[e,s]=np.prod(np.where(STATES,u,1-u),axis=1)
    return out

def scenario(heterogeneous):
    if not heterogeneous:
        return np.array([1.]),np.array([[.4,.2,.2,.2]]),np.array([kernel([.20,.13,.12,.21],[.03,.04])])
    # Gain and initial-state heterogeneity. Mixture of product kernels need
    # not itself factor by skill; the fitted joint rows remain unrestricted.
    return np.array([.55,.45]),np.array([[.58,.18,.16,.08],[.18,.24,.24,.34]]),np.array([
        kernel([.08,.05,.04,.085],[.03,.04]),
        kernel([.32,.22,.19,.34],[.03,.04])])

def generate(n, hetero, rng):
    weights,pclass,tclass=scenario(hetero)
    p=weights@pclass
    true_t=np.einsum('c,cs,cesr->esr',weights,pclass,tclass)/p[None,:,None]
    cls=rng.choice(len(weights),size=n,p=weights)
    s=(rng.random(n)[:,None]>np.cumsum(pclass[cls],axis=1)).sum(1)
    e=rng.integers(0,3,n)
    sp=(rng.random(n)[:,None]>np.cumsum(tclass[cls,e,s],axis=1)).sum(1)
    before=rng.binomial(3,TRUE_R[MAP[:,s]].T)
    after=rng.binomial(3,TRUE_R[MAP[:,sp]].T)
    return before,after,e,p,true_t

def emission(counts,r):
    theta=r[MAP]
    return np.exp(counts@np.log(theta)+(3-counts)@np.log1p(-theta))

def loglike(before,after,e,p,t,r):
    a=emission(before,r);b=emission(after,r)
    return np.log(np.einsum('is,isr,ir->i',a*p,t[e],b)).sum()

def fit(before,after,e,rng,starts=3,maxiter=1200,tol=1e-7):
    n=len(e);best=None
    masks=[e==j for j in range(3)]
    for start in range(starts):
        r=np.array([.2,.85,.2,.85,.15,.45,.45,.8])
        r=np.clip(r+rng.normal(0,.04,8),.025,.975)
        p=rng.dirichlet(np.ones(4)*5)
        t=np.array([np.eye(4)*.65+np.ones((4,4))*.35/4 for _ in range(3)])
        if start:
            t=.75*t+.25*rng.dirichlet(np.ones(4),size=(3,4))
        prev=-np.inf; converged=False;decreased=False
        for it in range(maxiter):
            a=emission(before,r);b=emission(after,r)
            w=a[:,:,None]*b[:,None,:]*p[None,:,None]*t[e]
            den=w.sum((1,2));ll=np.log(den).sum();w/=den[:,None,None]
            if ll+1e-7<prev:decreased=True
            if it>3 and abs(ll-prev)<tol*n:
                converged=True;break
            prev=ll
            ga=w.sum(2);gb=w.sum(1)
            p=ga.mean(0)
            for j,mask in enumerate(masks):
                cnt=w[mask].sum(0)
                t[j]=cnt/cnt.sum(1,keepdims=True)
            for v in range(8):
                ty,st=np.where(MAP==v)
                num=0.;denom=0.
                for typ,state in zip(ty,st):
                    num+=ga[:,state]@before[:,typ]+gb[:,state]@after[:,typ]
                    denom+=3*(ga[:,state].sum()+gb[:,state].sum())
                r[v]=np.clip(num/denom,1e-6,1-1e-6)
        if r[0]>=r[1] or r[2]>=r[3]:continue
        ans=dict(p=p,t=t,r=r,ll=ll,converged=converged,iterations=it+1,decreased=decreased)
        if best is None or ll>best['ll']:best=ans
    if best is None:raise RuntimeError('All starts lost the skill-label ordering')
    return best

def gain_contrasts(p,t):
    # Averaged over pre-episode states with the target skill absent, using
    # the SAME state distribution for both randomized activity assignments.
    out=[]
    for k,single in [(0,0),(1,1)]:
        base=STATES[:,k]==0
        dest=STATES[:,k]==1
        gain=t[:,:,dest].sum(2)
        out.append(np.sum(p[base]*(gain[2,base]-gain[single,base]))/p[base].sum())
    return np.array(out)

def exact_witness():
    # Three binary anchor items per skill: O has 64 observable patterns.
    patterns=np.array(list(__import__('itertools').product([0,1],repeat=6)))
    theta=TRUE_R[MAP[:2]]
    itemps=np.vstack([theta[0]]*3+[theta[1]]*3)
    o=np.prod(np.where(patterns[:,:,None],itemps[None,:,:],1-itemps[None,:,:]),axis=1)
    p=np.array([.4,.2,.2,.2]);t=kernel([.2,.13,.12,.21],[.03,.04])
    left=np.linalg.pinv(o)
    errors=[]
    for e in range(3):
        j=o@np.diag(p)@t[e]@o.T
        recovered=np.diag(1/p)@left@j@left.T
        errors.append(float(np.max(np.abs(recovered-t[e]))))
    return dict(emission_rank=int(np.linalg.matrix_rank(o)),states=4,
                smallest_singular_value=float(np.linalg.svd(o,compute_uv=False)[-1]),
                max_exact_reconstruction_error=max(errors),
                caveat='Known/invariant full-column-rank measurement and positive initial states; independent randomized activity. Not claimed as a new theorem.')

def main():
    a=argparse.ArgumentParser();a.add_argument('--reps',type=int,default=30);a.add_argument('--ns',type=int,nargs='+',default=[300,600,1200]);a.add_argument('--starts',type=int,default=3);a.add_argument('--output',default='anchor_recovery_results.json');args=a.parse_args()
    rng=np.random.default_rng(11012026);raw=[];start=time.time()
    result={'scope':__doc__,'seed':11012026,'exact_witness':exact_witness(),'replications':args.reps,'starts':args.starts,'results':[]}
    for hetero in [False,True]:
        for n in args.ns:
            cell=[]
            for rep in range(args.reps):
                pre,post,e,p,t=generate(n,hetero,rng)
                est=fit(pre,post,e,rng,starts=args.starts)
                record={'n':n,'heterogeneous':hetero,'rep':rep,'contrast_error':(gain_contrasts(est['p'],est['t'])-gain_contrasts(p,t)).tolist(),'transition_rmse':float(np.sqrt(np.mean((est['t']-t)**2))),'measurement_rmse':float(np.sqrt(np.mean((est['r']-TRUE_R)**2))),'converged':est['converged'],'iterations':est['iterations'],'ll_decreased':est['decreased']}
                cell.append(record);raw.append(record)
            err=np.array([x['contrast_error'] for x in cell])
            summary={'n':n,'heterogeneous':hetero,'contrast_bias':err.mean(0).tolist(),'contrast_rmse':np.sqrt(np.mean(err**2,axis=0)).tolist(),'contrast_bias_mcse':(err.std(0,ddof=1)/np.sqrt(args.reps)).tolist(),'mean_transition_rmse':float(np.mean([x['transition_rmse'] for x in cell])),'mean_measurement_rmse':float(np.mean([x['measurement_rmse'] for x in cell])),'convergence_fraction':float(np.mean([x['converged'] for x in cell])),'maximum_iterations':max(x['iterations'] for x in cell),'likelihood_decrease_count':sum(x['ll_decreased'] for x in cell)}
            result['results'].append(summary);result['elapsed_seconds']=time.time()-start;result['raw']=raw
            Path(args.output).write_text(json.dumps(result,indent=2))
            print(json.dumps(summary),flush=True)
    print('Seconds',time.time()-start,flush=True)

if __name__=='__main__':main()
