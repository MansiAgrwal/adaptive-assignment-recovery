"""Interval-method check: on regenerated adaptive datasets (same seeds as run_reps.py), refit the mixture
and compare OPG-based SEs with observed-information (numerical Hessian) SEs.
python se_check.py N first_rep last_rep out.jsonl"""
import sys, json, numpy as np
from ademp_sim import Model, generate, fit

N, r0, r1, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
m = Model(2)

def hessian_se(d, prm, h=2e-3):
    v = m.pack(prm); p = len(v)
    f = lambda x: m.loglik_persons(d, m.unpack(x)).sum()
    H = np.empty((p, p)); f0 = f(v)
    for i in range(p):
        for j in range(i, p):
            ei = np.zeros(p); ej = np.zeros(p); ei[i] = h; ej[j] = h
            H[i, j] = H[j, i] = (f(v + ei + ej) - f(v + ei - ej) - f(v - ei + ej) + f(v - ei - ej)) / (4 * h * h)
    cov = np.linalg.inv(-H)
    J = np.empty((2, p))
    for i in range(p):
        e = np.zeros(p); e[i] = 1e-4
        J[:, i] = (m.contrast(m.unpack(v + e)) - m.contrast(m.unpack(v - e))) / 2e-4
    var = np.einsum('ip,pq,iq->i', J, cov, J)
    return np.sqrt(var) if np.all(var > 0) else None

with open(out, 'a') as fo:
    for rep in range(r0, r1):
        rng = np.random.default_rng([20261012, N, rep, 1])          # adaptive policy seed, as in run_reps
        d = generate(N, 'adaptive', rng)
        runs = [fit(m, d, rng, 1) for _ in range(3)]
        prm = max(runs, key=lambda r: r[1])[0]
        se_opg, _ = m.wald(d, prm)
        se_h = hessian_se(d, prm)
        fo.write(json.dumps(dict(N=N, rep=rep, est=m.contrast(prm).tolist(),
                                 se_opg=None if se_opg is None else se_opg.tolist(),
                                 se_hess=None if se_h is None else se_h.tolist())) + '\n'); fo.flush()
