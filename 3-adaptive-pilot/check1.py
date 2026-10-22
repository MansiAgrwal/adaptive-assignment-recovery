import time, numpy as np
from ademp_sim import *
rng=np.random.default_rng(1)
print('true estimand (a,b):', estimand(TRUE['pi_fast'], TRUE['gains']))
for pol in ['random','adaptive']:
    d=generate(1000,pol,rng)
    cnt=np.bincount(d['acts'].ravel(),minlength=3)/d['acts'].size
    print(pol,'activity shares A/B/AB',cnt.round(3))
# independent slow likelihood for 3 persons vs vectorised
m=Model(2); prm=dict(pi=np.array([.4,.6]),p0=TRUE['p0'],gains=TRUE['gains'],g=TRUE['g'],s=TRUE['s'],loss=TRUE['loss'])
d=generate(3,'adaptive',rng)
vec=m.loglik_persons(d,prm)
def slow(i):
    from math import comb
    tot=0.
    for k in range(2):
        a=np.array([prm['pi'][k]*prm['p0'][s] for s in range(4)])
        def emis(j,s):
            p=1.
            if j<72:
                r=d['acts'][i,j]; pc=(1-prm['s'][r]) if ETA[r,s] else prm['g'][r]
                p*= pc if d['ys'][i,j] else 1-pc
            if j in PROBE_SLOTS or j==73:
                kk=PROBE_SLOTS.index(j) if j in PROBE_SLOTS else 7
                for r in range(3):
                    pc=(1-prm['s'][r]) if ETA[r,s] else prm['g'][r]; c=d['probes'][i,kk,r]
                    p*=pc**c*(1-pc)**(3-c)
            return p
        a=a*np.array([emis(0,s) for s in range(4)])
        for j in range(1,74):
            e=d['acts'][i,j-1] if j-1<72 else 3
            Tm=np.zeros((4,4))
            for s0 in range(4):
                for s1 in range(4):
                    if e<3:
                        la={0:prm['gains'][k,0],2:prm['gains'][k,1]}.get(e,0.); lb={1:prm['gains'][k,2],2:prm['gains'][k,3]}.get(e,0.)
                        pa=1. if XA[s0] else la; pb=1. if XB[s0] else lb
                    else:
                        pa=(1-prm['loss'][0]) if XA[s0] else 0.; pb=(1-prm['loss'][1]) if XB[s0] else 0.
                    Tm[s0,s1]=(pa if XA[s1] else 1-pa)*(pb if XB[s1] else 1-pb)
            a=(a@Tm)*np.array([emis(j,s) for s in range(4)])
        tot+=a.sum()
    return np.log(tot)
print('vectorised vs independent loglik:', vec, [slow(i) for i in range(3)])
# timing
d=generate(1000,'adaptive',rng)
t=time.time(); ll=m.loglik_persons(d,prm); print('one forward pass N=1000: %.3fs'%(time.time()-t))
t=time.time(); out=m.em(d,start_values(2,rng,d),maxit=50); print('50 EM iterations mixture: %.1fs'%(time.time()-t))
