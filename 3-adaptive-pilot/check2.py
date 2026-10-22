"""Large-N recovery: mixture from truth and from a random start; homogeneous pseudo-true values."""
import time, json, numpy as np
from ademp_sim import *
rng=np.random.default_rng(7); N=10000; out={}
truth=dict(pi=np.array([.4,.6]),p0=TRUE['p0'],gains=TRUE['gains'],g=TRUE['g'],s=TRUE['s'],loss=TRUE['loss'])
for pol in ['random','adaptive']:
    d=generate(N,pol,rng); t=time.time()
    m2=Model(2); prmT,llT,cT,itT,hT=m2.em(d,{k:v.copy() for k,v in truth.items()},maxit=300,tol=1e-8)
    prmR,llR,cR,itR,hR=m2.em(d,start_values(2,rng,d),maxit=800,tol=1e-8)
    m1=Model(1); prmH,llH,cH,itH,hH=m1.em(d,start_values(1,rng,d),maxit=800,tol=1e-8)
    res=dict(mix_from_truth=dict(contrast=m2.contrast(prmT).round(4).tolist(),gains=prmT['gains'].round(3).tolist(),pi=prmT['pi'].round(3).tolist(),ll=llT,iters=itT,conv=cT,monotone=bool(np.all(np.diff(hT)>-1e-6))),
             mix_random_start=dict(contrast=m2.contrast(prmR).round(4).tolist(),ll=llR,iters=itR,conv=cR,monotone=bool(np.all(np.diff(hR)>-1e-6))),
             homogeneous=dict(contrast=m1.contrast(prmH).round(4).tolist(),gains=prmH['gains'].round(4).tolist(),ll=llH,iters=itH,conv=cH,monotone=bool(np.all(np.diff(hH)>-1e-6))),
             measurement_mix=dict(g=prmT['g'].round(3).tolist(),s=prmT['s'].round(3).tolist(),loss=prmT['loss'].round(3).tolist(),p0=prmT['p0'].round(3).tolist()),
             seconds=round(time.time()-t))
    out[pol]=res; print(pol, json.dumps(res, default=float), flush=True)
json.dump(out,open('check2_largeN.json','w'),indent=1,default=float)
