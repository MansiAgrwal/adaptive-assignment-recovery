"""Run replications for the 2 x 2 design: python run_reps.py N first_rep last_rep out.jsonl
Each replication generates one dataset per assignment policy (independent seeds) and fits both
estimators to it, so the estimator factor is paired within dataset."""
import sys, json, time, numpy as np
from ademp_sim import Model, generate, fit, estimand, TRUE

N, r0, r1, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
truth = estimand(TRUE['pi_fast'], TRUE['gains'])
with open(out, 'a') as f:
    for rep in range(r0, r1):
        for pi_, pol in enumerate(['random', 'adaptive']):
            rng = np.random.default_rng([20261012, N, rep, pi_])
            t0 = time.time()
            d = generate(N, pol, rng)
            for K, name, starts in [(1, 'homogeneous', 2), (2, 'mixture', 3)]:
                m = Model(K)
                runs = []
                for _ in range(starts):
                    runs.append(fit(m, d, rng, 1))
                lls = np.array([r[1] for r in runs]); best = runs[int(lls.argmax())]
                prm, ll, conv, it, mono = best
                est = m.contrast(prm)
                se, cond = m.wald(d, prm)
                rec = dict(N=N, rep=rep, policy=pol, estimator=name, est=est.tolist(),
                           se=None if se is None else se.tolist(), truth=truth.tolist(),
                           converged=bool(conv), iters=int(it), monotone=bool(mono), ll=float(ll),
                           starts_agree=int((np.abs(lls - lls.max()) < 1e-3 * N).sum()),
                           cond=cond, pi=prm['pi'].tolist(), gains=prm['gains'].tolist(),
                           seconds=round(time.time() - t0, 2))
                f.write(json.dumps(rec) + '\n'); f.flush()
