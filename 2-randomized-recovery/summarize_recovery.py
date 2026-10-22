"""Separate gain-contrast bias and sampling SD; all N are independent people."""
from pathlib import Path
import csv
import json
import numpy as np
from anchor_recovery import scenario, gain_contrasts, TRUE_R, MAP

B=Path(__file__).parent
data=json.loads((B/'anchor_recovery_results.json').read_text())
rows=[]
for cell in data['results']:
    n=cell['n'];hetero=cell['heterogeneous']
    err=np.array([r['contrast_error'] for r in data['raw'] if r['n']==n and r['heterogeneous']==hetero])
    w,pc,tc=scenario(hetero);p=w@pc
    t=np.einsum('c,cs,cesr->esr',w,pc,tc)/p[None,:,None]
    target=gain_contrasts(p,t)
    for k in range(2):
        x=err[:,k];bias=x.mean();sd=x.std(ddof=1);rmse=np.sqrt(np.mean(x*x))
        row=dict(scenario='heterogeneous' if hetero else 'homogeneous',
                 independent_people=n,episodes_per_person=1,replications=len(x),
                 contrast='joint_minus_single_'+('a' if k==0 else 'b'),
                 true_contrast=float(target[k]),mean_estimate=float(target[k]+bias),
                 bias=float(bias),sampling_sd=float(sd),rmse=float(rmse),
                 bias_mcse=float(sd/np.sqrt(len(x))),
                 rmse_decomposition_error=float(abs(rmse**2-bias**2-(len(x)-1)/len(x)*sd**2)),
                 interval_coverage='not_evaluated',
                 converged_fraction=cell['convergence_fraction'])
        rows.append(row)
with (B/'Recovery_Contrast_Diagnostics.csv').open('w',newline='') as f:
    out=csv.DictWriter(f,fieldnames=rows[0].keys());out.writeheader();out.writerows(rows)

# Exact comparison of the observed endpoint law under the two-class
# generator and its unrestricted marginal transition representation.
# Retaining full six-pure-item response patterns avoids combinatorial factors.
import itertools
pat=np.array(list(itertools.product([0,1],repeat=6)))
theta=TRUE_R[MAP[:2]]
ps=np.vstack([theta[0]]*3+[theta[1]]*3)
o=np.prod(np.where(pat[:,:,None],ps[None,:,:],1-ps[None,:,:]),axis=1)
w,pc,tc=scenario(True);p=w@pc
t=np.einsum('c,cs,cesr->esr',w,pc,tc)/p[None,:,None]
diff=[]
for e in range(3):
    mixture=sum(w[c]*(o@np.diag(pc[c])@tc[c,e]@o.T) for c in range(len(w)))
    marginal=o@np.diag(p)@t[e]@o.T
    diff.append(float(np.max(np.abs(mixture-marginal))))
report={'diagnostics':rows,'heterogeneity_check':{
    'max_observable_probability_difference':max(diff),
    'scope':'One independently randomized activity; invariant measurement independent of latent type. Unrestricted joint transition rows exactly represent the endpoint mixture. This is not a misspecified homogeneous longitudinal HMM and does not validate transport across repeated adaptive episodes.'}}
(B/'Recovery_Contrast_Diagnostics.json').write_text(json.dumps(report,indent=2))
print('scenario N contrast true bias SD RMSE MCSE_bias')
for r in rows:
    print(r['scenario'],r['independent_people'],r['contrast'],*[f'{r[k]:.5f}' for k in ['true_contrast','bias','sampling_sd','rmse','bias_mcse']])
print(report['heterogeneity_check'])
