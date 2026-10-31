"""Exposure-weighted limit of the one-type (homogeneous) acquisition estimate, activity-by-type condition.
Generator as in the step 3 and step 4 simulations: two attributes, activities A, B, AB, no loss during practice,
72 practice trials, responses before transitions, guess (A,B,AB) = .25,.20,.15, slip = .10,.12,.15,
start probabilities (none, a only, b only, both) = .60,.15,.15,.10 for every type.
Types (weights .4, .6): gains (a|A, a|AB, b|B, b|AB) = (.20,.10,.15,.15) and (.02,.05,.03,.03).
With fully observed states, the homogeneous maximum-likelihood gain converges to
sum_r w_r N_re g_e(r) / sum_r w_r N_re, where N_re = E[# trials at risk for the attribute with activity e | type r].
Prints exposures, exposure weights, the limiting contrast split into composition and sorting terms,
and a check of E[acquisitions | r, e] = g_e(r) N_re."""
import numpy as np
W = np.array([.4, .6]); G = np.array([[.20, .10, .15, .15], [.02, .05, .03, .03]])
GUESS = np.array([.25, .20, .15]); SLIP = np.array([.10, .12, .15]); P0 = np.array([.60, .15, .15, .10])
T = 72; EPS = .15; CAP = 18
def simulate(policy, r, n, rng):
    st = rng.choice(4, size=n, p=P0); a = np.isin(st, [1, 3]); b = np.isin(st, [2, 3])
    buf_e = np.full((n, CAP), -1); buf_y = np.zeros((n, CAP)); pos = 0
    risk = {'a': np.zeros(3), 'b': np.zeros(3)}; acq = {'a': np.zeros(3), 'b': np.zeros(3)}
    for t in range(T):
        if policy == 'random':
            e = rng.integers(3, size=n)
        else:
            score = np.stack([((buf_y * (buf_e == k)).sum(1) + 1) / ((buf_e == k).sum(1) + 2) for k in range(3)], 1)
            low = np.isclose(score, score.min(1, keepdims=True))
            prob = EPS / 3 + (1 - EPS) * low / low.sum(1, keepdims=True)
            e = (rng.random((n, 1)) > prob.cumsum(1)).sum(1)
        mastered = np.where(e == 0, a, np.where(e == 1, b, a & b))
        y = np.where(mastered, rng.random(n) < 1 - SLIP[e], rng.random(n) < GUESS[e]).astype(float)
        buf_e[:, pos] = e; buf_y[:, pos] = y; pos = (pos + 1) % CAP
        ga = np.where(e == 0, G[r, 0], np.where(e == 2, G[r, 1], 0.)); gb = np.where(e == 1, G[r, 2], np.where(e == 2, G[r, 3], 0.))
        for k in range(3):
            risk['a'][k] += np.sum(~a & (e == k)); risk['b'][k] += np.sum(~b & (e == k))
        na = ~a & (rng.random(n) < ga); nb = ~b & (rng.random(n) < gb)
        for k in range(3):
            acq['a'][k] += np.sum(na & (e == k)); acq['b'][k] += np.sum(nb & (e == k))
        a = a | na; b = b | nb
    return {s: risk[s] / n for s in risk}, {s: acq[s] / n for s in acq}
rng = np.random.default_rng(20261029); n = 400000
truth_a = W @ (G[:, 1] - G[:, 0]); truth_b = W @ (G[:, 3] - G[:, 2])
print(f'target contrasts: a {truth_a:+.4f}, b {truth_b:+.4f}')
for policy in ['random', 'adaptive']:
    N = [simulate(policy, r, n, rng) for r in range(2)]
    Na = np.array([N[r][0]['a'] for r in range(2)]); Nb = np.array([N[r][0]['b'] for r in range(2)])
    # pseudo-true homogeneous gains (exposure-weighted) and the resulting contrast
    def pt(Nmat, col): return (W * Nmat[:, col] * G[:, col_map[col]]).sum() / (W * Nmat[:, col]).sum()
    wA = W * Na[:, 0] / (W * Na[:, 0]).sum(); wAB = W * Na[:, 2] / (W * Na[:, 2]).sum()
    gA = wA @ G[:, 0]; gAB = wAB @ G[:, 1]
    wB = W * Nb[:, 1] / (W * Nb[:, 1]).sum(); wABb = W * Nb[:, 2] / (W * Nb[:, 2]).sum()
    hb = wABb @ G[:, 3] - wB @ G[:, 2]
    wbar = (wA + wAB) / 2; d = wAB - wA
    comp = wbar @ (G[:, 1] - G[:, 0]) - truth_a; sort = (d / 2) @ (G[:, 1] + G[:, 0])
    # empirical check of the identity E[acquisitions | r, e] = g_e(r) N_re
    acq = np.array([N[r][1]['a'] for r in range(2)])
    print(f'{policy:8s}: at-risk exposures for a, A by type {Na[:,0].round(3)}, AB by type {Na[:,2].round(3)}')
    print(f'          weights on fast type: A {wA[0]:.3f}, AB {wAB[0]:.3f} (baseline weight {W[0]:.3f})')
    print(f'          oracle homogeneous contrast a {gAB-gA:+.4f} (composition {comp:+.4f} + sorting {sort:+.4f}); b {hb:+.4f}')
    print(f'          check acquisitions/(g*N): {(acq[:,0]/(G[:,0]*Na[:,0])).round(3)}, {(acq[:,2]/(G[:,1]*Na[:,2])).round(3)}')
# analytic random case: hazard per trial for a = (gA + gAB)/3, starting unmastered with prob .75
for r in range(2):
    h = (G[r, 0] + G[r, 1]) / 3; Nr = .75 * (1 - (1 - h) ** T) / h / 3
    print(f'analytic random N_rA = N_rAB for type {r}: {Nr:.3f}')
