"""Exposure-weighted limit of the homogeneous contrast for the three heterogeneity conditions of the step 4 grid,
compared with the mean fitted homogeneous estimates stored in data/recovery_summary.csv.
Gains per condition: marginal gains (.092,.070,.078,.078) for (a|A, a|AB, b|B, b|AB); activity-by-type = the
two-type gains of step 3; common shift = logit shifts +/-.8 with weights .4/.6; continuous = normal logit effect
with sd 1.2 (60-node Gauss-Hermite types). Assignment and response model as in oracle.py."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from numpy.polynomial.hermite import hermgauss
MARG = np.array([.092, .070, .078, .078]); GUESS = np.array([.25, .20, .15]); SLIP = np.array([.10, .12, .15])
P0 = np.array([.60, .15, .15, .10]); T = 72; EPS = .15; CAP = 18
def family(name):
    if name == 'interaction': return np.array([.4, .6]), np.array([[.20, .10, .15, .15], [.02, .05, .03, .03]])
    if name == 'discrete': s, w = np.array([.8, -.8]), np.array([.4, .6])
    else:
        x, w = hermgauss(60); s = np.sqrt(2) * 1.2 * x; w = w / np.sqrt(np.pi)
    base = np.array([brentq(lambda b: w @ expit(b + s) - v, -15, 10) for v in MARG])
    return w, expit(base[None, :] + s[:, None])
def exposures(policy, g, n, rng):
    st = rng.choice(4, size=n, p=P0); a = np.isin(st, [1, 3]); b = np.isin(st, [2, 3])
    be = np.full((n, CAP), -1); by = np.zeros((n, CAP)); pos = 0; Na = np.zeros(3); Nb = np.zeros(3)
    for t in range(T):
        if policy == 'random': e = rng.integers(3, size=n)
        else:
            sc = np.stack([((by * (be == k)).sum(1) + 1) / ((be == k).sum(1) + 2) for k in range(3)], 1)
            low = np.isclose(sc, sc.min(1, keepdims=True)); p = EPS / 3 + (1 - EPS) * low / low.sum(1, keepdims=True)
            e = (rng.random((n, 1)) > p.cumsum(1)).sum(1)
        m = np.where(e == 0, a, np.where(e == 1, b, a & b))
        y = np.where(m, rng.random(n) < 1 - SLIP[e], rng.random(n) < GUESS[e]).astype(float)
        be[:, pos] = e; by[:, pos] = y; pos = (pos + 1) % CAP
        for k in range(3): Na[k] += np.sum(~a & (e == k)); Nb[k] += np.sum(~b & (e == k))
        ga = np.where(e == 0, g[0], np.where(e == 2, g[1], 0.)); gb = np.where(e == 1, g[2], np.where(e == 2, g[3], 0.))
        a = a | (~a & (rng.random(n) < ga)); b = b | (~b & (rng.random(n) < gb))
    return Na / n, Nb / n
import csv, pathlib
fitted = {}
for row in csv.DictReader(open(pathlib.Path(__file__).parent / 'data' / 'recovery_summary.csv')):
    if row['model'] == 'homogeneous' and row['skill'] == 'a' and row['interval'] == 'hessian':
        fitted[(row['family'], row['policy'])] = float(row['mean_estimate'])
rng = np.random.default_rng(29102026)
for fam in ['interaction', 'discrete', 'continuous']:
    w, G = family(fam); target = w @ (G[:, 1] - G[:, 0]); keep = w > 1e-6; w, G = w[keep] / w[keep].sum(), G[keep]
    n = 200000 if len(w) == 2 else 30000
    for pol in ['random', 'adaptive']:
        NA = np.array([exposures(pol, G[r], n, rng)[0] for r in range(len(w))])
        oA = w * NA[:, 0]; oA /= oA.sum(); oAB = w * NA[:, 2]; oAB /= oAB.sum()
        hom = oAB @ G[:, 1] - oA @ G[:, 0]; ob = (oA + oAB) / 2; d = oAB - oA
        comp = ob @ (G[:, 1] - G[:, 0]) - target; sort = (d / 2) @ (G[:, 1] + G[:, 0])
        print(f'{fam:11s} {pol:8s} target {target:+.4f}  limit {hom:+.4f} (comp {comp:+.4f}, sort {sort:+.4f})  fitted {fitted[(fam,pol)]:+.4f}  fitted-limit {fitted[(fam,pol)]-hom:+.4f}', flush=True)
