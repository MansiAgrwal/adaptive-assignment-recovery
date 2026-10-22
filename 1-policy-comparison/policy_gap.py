"""On identical histories, how different are the selection probabilities implied by the two buffers?"""
import numpy as np
ACT=['A','B','AB']; REQ={'A':(1,0),'B':(0,1),'AB':(1,1)}
def bufs(hist):
    rec=hist[-18:]
    keep=set()
    for e in ACT: keep.update([j for j,(ee,_) in enumerate(hist) if ee==e][-6:])
    rest=[j for j in range(len(hist)) if j not in keep]; need=18-len(keep)
    if need>0: keep.update(rest[-need:])
    return rec,[hist[j] for j in sorted(keep)]
def pi(buf):
    w=np.array([1-(1+sum(y for e,y in buf if e==a))/(2+sum(1 for e,_ in buf if e==a)) for a in ACT])
    return 0.1/3+0.9*w/w.sum()
rng=np.random.default_rng(7)
for version in ['balanced','skewed']:
  for P in [dict(g=.25,s=.1,la=.10,lb=.03,lab=.04,p0a=.2,p0b=.1)]:
    gaps=[];same=0;tot=0;first_gap=[]
    for i in range(3000):
        x=[int(rng.random()<P['p0a']),int(rng.random()<P['p0b'])]; hist=[]
        def trial(e):
            q=REQ[e]; eta=all(x[k] for k in range(2) if q[k]); y=int(rng.random()<P['g']+(1-P['s']-P['g'])*eta); hist.append((e,y))
            for k,sk in enumerate('ab'):
                l={('a','A'):P['la'],('b','B'):P['lb'],('a','AB'):P['lab'],('b','AB'):P['lab']}.get((sk,e),0)
                if x[k]==0 and rng.random()<l: x[k]=1
        init=['A']*6+['B']*6+['AB']*6 if version=='balanced' else ['A']*12+['B']*3+['AB']*3
        rng.shuffle(init); [trial(e) for e in init]
        r,s=bufs(hist); assert sorted(r)==sorted(s)  # identical at start of adaptive practice
        for blk in range(6):
            sp=set(rng.choice(12,3,replace=False)); order=list(rng.permutation(ACT))
            for t in range(12):
                if t in sp: trial(order.pop()); continue
                r,s=bufs(hist); pr,ps=pi(r),pi(s); g=np.abs(pr-ps).max(); gaps.append(g); tot+=1; same+= g<1e-12
                trial(ACT[rng.choice(3,p=pr)])
    gaps=np.array(gaps)
    print(f"{version}: adaptive decisions={tot}; identical policy {same/tot:.1%}; max|dpi| mean={gaps.mean():.3f}, median={np.median(gaps):.3f}, 90th pct={np.quantile(gaps,.9):.3f}")
