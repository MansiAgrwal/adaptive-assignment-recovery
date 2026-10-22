"""Exact small-model checks of the randomized reference design, not novel theorems."""
from pathlib import Path
import itertools
import json
import numpy as np
from anchor_recovery import exact_witness, generate, fit, loglike


def gain_kernel(g):
    return np.array([[1-g, g], [0., 1.]])


def endpoint_counterexample():
    # One skill, no loss; one use of each activity, order may vary.
    first=[gain_kernel(.2), gain_kernel(.4)]
    second=[gain_kernel(.4), gain_kernel(.2)]
    endpoint_errors=[]
    for seq in [(0,1),(1,0)]:
        endpoint_errors.append(float(np.max(np.abs(first[seq[0]]@first[seq[1]]-
                                                       second[seq[0]]@second[seq[1]]))))
    return dict(gains_first=[.2,.4],gains_second=[.4,.2],
                contrasts=[.2,-.2],common_two_activity_gain=.52,
                max_endpoint_difference=max(endpoint_errors),
                one_activity_kernel_difference=float(np.max(np.abs(first[0]-second[0]))),
                scope='Endpoint-only assessments with equal activity counts; both AB and BA orders fail here. Does not establish nonidentification when informative intermediate responses are retained.')


def policy_support_check():
    # Three observations and two chosen actions; choice uses only observed
    # previous correctness. Compare each word with the fixed-action law.
    t=[np.array([[.8,.2],[.05,.95]]),np.array([[.6,.4],[.08,.92]])]
    p=np.array([.65,.35]);r=np.array([.2,.85])
    errors=[];total=0.;mins=[]
    for ys in itertools.product([0,1],repeat=3):
        for es in itertools.product([0,1],repeat=2):
            law=p*np.where(ys[0],r,1-r)
            assignment=1.
            for j,e in enumerate(es):
                pi=np.array([.7,.3]) if ys[j]==0 else np.array([.25,.75])
                assignment*=pi[e];mins.append(float(pi[e]))
                law=(law@t[e])*np.where(ys[j+1],r,1-r)
            fixed=float(law.sum());adaptive=assignment*fixed
            total+=adaptive;errors.append(abs(adaptive/assignment-fixed))
    return dict(adaptive_probability_sum=total,
                minimum_activity_probability=min(mins),
                max_reweighting_error=max(errors),
                scope='Exact enumeration of a correctly specified model with sequential observed-history randomization. This illustrates support equivalence, not a general proof or finite-sample variance guarantee.')


def policy_extrema():
    # Exhaust all three-type count/correctness configurations of a full
    # 18-slot history under the supplied Laplace-smoothed policy.
    lo=(1.,None);hi=(0.,None)
    for n0 in range(19):
        for n1 in range(19-n0):
            n=np.array([n0,n1,18-n0-n1])
            for c in itertools.product(*(range(int(v)+1) for v in n)):
                phat=(1+np.array(c))/(2+n)
                prob=.1/3+.9*(1-phat)/(1-phat).sum()
                if prob[0]<lo[0]:lo=(float(prob[0]),[n.tolist(),list(c)])
                if prob[0]>hi[0]:hi=(float(prob[0]),[n.tolist(),list(c)])
    return dict(min_probability=lo[0],max_probability=hi[0],min_counts_correct=lo[1],
                max_counts_correct=hi[1],scope='All count configurations at exactly 18 retained responses; some may be unreachable under a particular scheduled design.')


def recovery_implementation_check():
    rng=np.random.default_rng(812026)
    pre,post,e,_,_=generate(1200,True,rng)
    est=fit(pre,post,e,rng,starts=5,tol=1e-10)
    calculated=loglike(pre,post,e,est['p'],est['t'],est['r'])
    # Direct latent-state summation for a subset must match the vectorized
    # expression used by the estimator (binomial coefficients are constants).
    from anchor_recovery import MAP
    direct=0.
    for i in range(10):
        q=0.
        for s in range(4):
            for sp in range(4):
                a=np.prod(est['r'][MAP[:,s]]**pre[i]*(1-est['r'][MAP[:,s]])**(3-pre[i]))
                b=np.prod(est['r'][MAP[:,sp]]**post[i]*(1-est['r'][MAP[:,sp]])**(3-post[i]))
                q+=est['p'][s]*est['t'][e[i],s,sp]*a*b
        direct+=np.log(q)
    vectorized=loglike(pre[:10],post[:10],e[:10],est['p'],est['t'],est['r'])
    return dict(stored_likelihood_error=float(abs(calculated-est['ll'])),
                latent_sum_error=float(abs(direct-vectorized)),
                max_row_sum_error=float(np.max(np.abs(est['t'].sum(2)-1))),
                converged=est['converged'],likelihood_decreased=est['decreased'],
                scope='Arithmetic/likelihood check; does not prove global optimization or interval coverage.')


if __name__=='__main__':
    result={'reference_identification':exact_witness(),
            'endpoint_counterexample':endpoint_counterexample(),
            'adaptive_support':policy_support_check(),
            'buffer_policy_extrema':policy_extrema(),
            'recovery_implementation':recovery_implementation_check()}
    assert result['reference_identification']['max_exact_reconstruction_error']<1e-12
    assert result['endpoint_counterexample']['max_endpoint_difference']<1e-12
    assert abs(result['adaptive_support']['adaptive_probability_sum']-1)<1e-12
    assert result['adaptive_support']['max_reweighting_error']<1e-12
    assert result['recovery_implementation']['latent_sum_error']<1e-10
    path=Path(__file__).with_name('design_checks.json')
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
