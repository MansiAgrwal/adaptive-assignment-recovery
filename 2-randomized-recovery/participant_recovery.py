"""Repeated randomized episodes with independent people as sampling units.

All episodes of a person share a latent learner type. Activity is independently
randomized each episode. The estimator is a marginal composite likelihood, NOT
an independent-episode likelihood for uncertainty. Empirical sampling SD is
computed across independently regenerated participant samples. Optional bootstrap
resamples people, preserving their entire episode bundles.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from anchor_recovery import scenario, STATES, MAP, TRUE_R, fit, gain_contrasts


def generate_people(n,repeats,hetero,rng,order_drift=False):
    w,pc,tc=scenario(hetero)
    cls=rng.choice(len(w),size=n,p=w)
    person=np.repeat(np.arange(n),repeats)
    types=np.repeat(cls,repeats)
    # Fresh task contexts: initial states vary by episode conditional on type.
    # The drift scenario changes initial knowledge and learning by episode order,
    # while current activity remains independently randomized.
    probs=np.tile(pc[None,:,:],(repeats,1,1))
    kernels=np.tile(tc[None,:,:,:,:],(repeats,1,1,1,1))
    if order_drift:
        for k in range(repeats):
            mix=.18*k/max(repeats-1,1)
            probs[k]=(1-mix)*pc+mix*np.array([.05,.15,.15,.65])[None,:]
            # Fatigue or transfer can alter net change over context order.
            kernels[k]=(1-.15*k/max(repeats-1,1))*tc+(.15*k/max(repeats-1,1))*np.eye(4)[None,None,:,:]
    episode=np.tile(np.arange(repeats),n)
    s=(rng.random(n*repeats)[:,None]>np.cumsum(probs[episode,types],axis=1)).sum(1)
    e=rng.integers(0,3,n*repeats)
    sp=(rng.random(n*repeats)[:,None]>np.cumsum(kernels[episode,types,e,s],axis=1)).sum(1)
    pre=rng.binomial(3,TRUE_R[MAP[:,s]].T)
    post=rng.binomial(3,TRUE_R[MAP[:,sp]].T)
    # Exact marginal target averages over scheduled episode positions and types.
    p=np.einsum('c,kcs->s',w,probs)/repeats
    joint=np.einsum('c,kcs,kcesr->esr',w,probs,kernels)/repeats
    t=joint/p[None,:,None]
    return pre,post,e,person,p,t


def summarize(records):
    err=np.array([x['contrast_error'] for x in records])
    out=dict(bias=err.mean(0).tolist(),sampling_sd=err.std(0,ddof=1).tolist(),
             rmse=np.sqrt(np.mean(err**2,axis=0)).tolist(),
             bias_mcse=(err.std(0,ddof=1)/np.sqrt(len(err))).tolist(),
             converged_fraction=float(np.mean([x['converged'] for x in records])))
    if records[0].get('covered') is not None:
        cover=np.array([x['covered'] for x in records]);rate=cover.mean(0)
        out.update(coverage=rate.tolist(),coverage_mcse=np.sqrt(rate*(1-rate)/len(records)).tolist(),
                   mean_interval_width=np.mean([x['interval_width'] for x in records],axis=0).tolist())
    return out


def main():
    a=argparse.ArgumentParser();a.add_argument('--people',type=int,nargs='+',default=[180,240,300]);a.add_argument('--episodes',type=int,nargs='+',default=[3,6]);a.add_argument('--reps',type=int,default=100);a.add_argument('--bootstrap',type=int,default=0);a.add_argument('--output',default='participant_recovery_results.json');a.add_argument('--scenarios',nargs='+',default=['homogeneous','heterogeneous','order_drift']);args=a.parse_args()
    rng=np.random.default_rng(12102026);brng=np.random.default_rng(12102027)
    result=dict(scope=__doc__,seed=12102026,bootstrap_seed=12102027,replications=args.reps,bootstrap_replicates=args.bootstrap,results=[],raw=[])
    start=time.time()
    for name in args.scenarios:
        for n in args.people:
            for repeats in args.episodes:
                records=[]
                for r in range(args.reps):
                    pre,post,e,person,p,t=generate_people(n,repeats,name!='homogeneous',rng,name=='order_drift')
                    target=gain_contrasts(p,t)
                    est=fit(pre,post,e,rng,starts=3)
                    estimate=gain_contrasts(est['p'],est['t'])
                    rec=dict(scenario=name,people=n,episodes_per_person=repeats,rep=r,
                             true_contrast=target.tolist(),estimate=estimate.tolist(),contrast_error=(estimate-target).tolist(),
                             converged=est['converged'])
                    if args.bootstrap:
                        boot=[]
                        for _ in range(args.bootstrap):
                            ids=brng.integers(0,n,n)
                            idx=(ids[:,None]*repeats+np.arange(repeats)[None,:]).ravel()
                            fitted=fit(pre[idx],post[idx],e[idx],brng,starts=2)
                            boot.append(gain_contrasts(fitted['p'],fitted['t']))
                        lo,hi=np.quantile(boot,[.025,.975],axis=0)
                        rec.update(covered=((lo<=target)&(target<=hi)).tolist(),interval_width=(hi-lo).tolist(),
                                   lower=lo.tolist(),upper=hi.tolist())
                    records.append(rec);result['raw'].append(rec)
                    if args.bootstrap and (r+1)%5==0:print(name,n,repeats,'rep',r+1,flush=True)
                summary=dict(scenario=name,people=n,episodes_per_person=repeats,**summarize(records))
                result['results'].append(summary);result['elapsed_seconds']=time.time()-start
                Path(args.output).write_text(json.dumps(result,indent=2))
                print(json.dumps(summary),flush=True)
    print('Seconds',time.time()-start,flush=True)


if __name__=='__main__':main()
