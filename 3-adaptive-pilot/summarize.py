"""Summarise the replication files into ADEMP performance measures (with Monte Carlo SEs)."""
import json, glob, sys, csv
import numpy as np

rows = [json.loads(l) for f in sorted(glob.glob('res_*.jsonl')) for l in open(f)]
out = []
for N in sorted({r['N'] for r in rows}, reverse=True):
    for pol in ['random', 'adaptive']:
        for est in ['homogeneous', 'mixture']:
            R = [r for r in rows if r['N'] == N and r['policy'] == pol and r['estimator'] == est]
            if not R: continue
            for k, skill in enumerate(['a', 'b']):
                e = np.array([r['est'][k] for r in R]); t = R[0]['truth'][k]
                ok = [r['se'] is not None for r in R]
                se = np.array([r['se'][k] if r['se'] is not None else np.nan for r in R])
                n = len(e); sd = e.std(ddof=1); bias = e.mean() - t
                cov = np.nanmean(np.abs(e - t) <= 1.96 * se)
                cov_own = np.nanmean(np.abs(e - e.mean()) <= 1.96 * se)   # coverage of its own limit
                rej0 = np.nanmean(np.abs(e) > 1.96 * se)
                out.append(dict(
                    N=N, policy=pol, estimator=est, skill=skill, reps=n, truth=round(t, 4),
                    mean_estimate=round(e.mean(), 4), bias=round(bias, 4), bias_mcse=round(sd / np.sqrt(n), 4),
                    empirical_sd=round(sd, 4), sd_mcse=round(sd / np.sqrt(2 * (n - 1)), 4),
                    mean_model_se=round(np.nanmean(se), 4), se_over_sd=round(np.nanmean(se) / sd, 3),
                    rmse=round(np.sqrt(np.mean((e - t) ** 2)), 4),
                    coverage_95=round(cov, 3), coverage_mcse=round(np.sqrt(cov * (1 - cov) / n), 3),
                    coverage_of_own_mean=round(cov_own, 3),
                    mean_interval_width=round(np.nanmean(2 * 1.96 * se), 4),
                    rejects_zero=round(rej0, 3),
                    converged=round(np.mean([r['converged'] for r in R]), 3),
                    wald_available=round(np.mean(ok), 3),
                    starts_agree_all=round(np.mean([r['starts_agree'] == (2 if est == 'homogeneous' else 3) for r in R]), 3)))
keys = list(out[0])
with open('ADEMP_results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(out)
for o in out:
    print(f"N={o['N']:5d} {o['policy']:8s} {o['estimator']:11s} {o['skill']} reps={o['reps']:3d} "
          f"mean={o['mean_estimate']:+.4f} bias={o['bias']:+.4f}({o['bias_mcse']:.4f}) sd={o['empirical_sd']:.4f} "
          f"se/sd={o['se_over_sd']:.2f} cov={o['coverage_95']:.3f}({o['coverage_mcse']:.3f}) "
          f"covOwn={o['coverage_of_own_mean']:.3f} width={o['mean_interval_width']:.4f} rej0={o['rejects_zero']:.3f} "
          f"conv={o['converged']:.2f} wald={o['wald_available']:.2f} agree={o['starts_agree_all']:.2f}")
