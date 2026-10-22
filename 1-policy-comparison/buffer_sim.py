"""Illustrative simulation of the planned design: recency-18 vs stratified (6 per requirement) buffer.
All learning parameters are assumptions chosen for illustration, not estimates."""
import numpy as np, sys
ACT = ['A','B','AB']; REQ = {'A':(1,0),'B':(0,1),'AB':(1,1)}

def run(n, version, arm, P, rng):
    out_acc = np.zeros(n); expo = np.zeros((n,3)); cov = np.zeros((n,3))
    for i in range(n):
        # person-level heterogeneity in learning rates
        mult = rng.lognormal(0, P['het'], size=2)
        lam = {('a','A'):P['la']*mult[0], ('b','B'):P['lb']*mult[1],
               ('a','AB'):P['lab']*mult[0], ('b','AB'):P['lab']*mult[1]}
        lam = {k:min(v,0.9) for k,v in lam.items()}
        x = [int(rng.random()<P['p0a']), int(rng.random()<P['p0b'])]
        hist = []  # (activity, correct)
        def trial(e):
            q = REQ[e]; eta = all(x[k] for k in range(2) if q[k])
            p = P['g'] + (1-P['s']-P['g'])*eta
            y = int(rng.random()<p)
            hist.append((e,y)); expo[i,ACT.index(e)] += 1
            if x[0]==0 and x[1]==1 and e=='AB': pass
            # coverage: count AB trials while in partial state, etc. (state-activity occupancy of non-mastered skill)
            if any(q[k] and not x[k] for k in range(2)): cov[i,ACT.index(e)] += 1
            # transition
            for k,sk in enumerate('ab'):
                if x[k]==0:
                    l = lam.get((sk,e),0.0)
                    if rng.random()<l: x[k]=1
                else:
                    if rng.random()<P['phi']: x[k]=0
        def buffer():
            if arm=='recency': return hist[-18:]
            keep_idx=set()
            for e in ACT:
                idx=[j for j,(ee,_) in enumerate(hist) if ee==e][-6:]
                keep_idx.update(idx)
            rest=[j for j in range(len(hist)) if j not in keep_idx]
            need=18-len(keep_idx)
            keep_idx.update(rest[-need:] if need>0 else [])
            return [hist[j] for j in sorted(keep_idx)]
        init = ['A']*6+['B']*6+['AB']*6 if version=='balanced' else ['A']*12+['B']*3+['AB']*3
        rng.shuffle(init)
        for e in init: trial(e)
        for blk in range(6):
            sched_pos = set(rng.choice(12,3,replace=False)); order=list(rng.permutation(ACT))
            for t in range(12):
                if t in sched_pos: trial(order.pop()); continue
                buf=buffer()
                w=[]
                for e in ACT:
                    ne=sum(1 for ee,_ in buf if ee==e); ce=sum(y for ee,y in buf if ee==e)
                    w.append(1-(1+ce)/(2+ne))
                w=np.array(w); pi=0.1/3+0.9*w/w.sum()
                trial(ACT[rng.choice(3,p=pi)])
        # 24h retention
        for k in range(2):
            if x[k]==1 and rng.random()<P['phi24']: x[k]=0
        # 24 novel items, 8 per requirement, same DINA params
        c=0
        for e in ACT:
            q=REQ[e]; eta=all(x[k] for k in range(2) if q[k]); p=P['g']+(1-P['s']-P['g'])*eta
            c+=rng.binomial(8,p)
        out_acc[i]=c/24
    return out_acc, expo, cov

scen = {
 'S1 symmetric, g=.25': dict(g=.25,s=.10,la=.08,lb=.08,lab=.04,phi=.005,phi24=.15,p0a=.2,p0b=.2,het=.5),
 'S2 b harder, g=.25':  dict(g=.25,s=.10,la=.10,lb=.03,lab=.04,phi=.005,phi24=.15,p0a=.2,p0b=.1,het=.5),
 'S3 b harder, g=.50':  dict(g=.50,s=.10,la=.10,lb=.03,lab=.04,phi=.005,phi24=.15,p0a=.2,p0b=.1,het=.5),
 'S4 slow learning, high het': dict(g=.25,s=.10,la=.03,lb=.02,lab=.02,phi=.005,phi24=.20,p0a=.1,p0b=.1,het=1.0),
}
N=int(sys.argv[1]) if len(sys.argv)>1 else 4000
rng=np.random.default_rng(2026)
for name,P in scen.items():
    print('\n'+name)
    for version in ['balanced','skewed']:
        res={}
        for arm in ['recency','stratified']:
            res[arm]=run(N,version,arm,P,rng)
        a,b=res['recency'][0],res['stratified'][0]
        sd=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2); d=(b.mean()-a.mean())/sd
        se=np.sqrt(a.var(ddof=1)/N+b.var(ddof=1)/N)/sd
        ea,eb=res['recency'][1].mean(0),res['stratified'][1].mean(0)
        print(f"  {version:9s} delayed acc rec={a.mean():.3f} strat={b.mean():.3f}  d={d:+.3f} (MC SE {se:.3f}) | "
              f"exposure A/B/AB rec={ea.round(1)} strat={eb.round(1)}")
