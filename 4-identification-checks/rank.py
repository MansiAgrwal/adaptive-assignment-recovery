"""Rank of block-emission matrices for the product hidden state Z = (type r, competence s), M = 2 types.
Future block: F[y_1..y_L, z] = P(y_1..y_L | Z_1 = z) under a fixed activity pattern.
Past block:   J[y_-L..y_-1, z] = P(y_-L..y_-1, Z_0 = z), started from a type-specific initial law."""
import itertools, numpy as np
STATES = [(0, 0), (1, 0), (0, 1), (1, 1)]; REQ = {0: (1, 0), 1: (0, 1), 2: (1, 1)}
def emit(e, s, guess, slip):
    need = REQ[e]; m = all(s[k] >= need[k] for k in range(2)); return (1 - slip[e]) if m else guess[e]
def trans(e, s, g):  # g = (a|A, a|AB, b|B, b|AB) for one type; no loss
    pa = 1. if s[0] else (g[0] if e == 0 else g[1] if e == 2 else 0.); pb = 1. if s[1] else (g[2] if e == 1 else g[3] if e == 2 else 0.)
    out = np.zeros(4)
    for j, sp in enumerate(STATES):
        out[j] = (pa if sp[0] else 1 - pa) * (pb if sp[1] else 1 - pb)
    return out
def future(L, pattern, G, guess, slip):
    Z = [(r, i) for r in range(len(G)) for i in range(4)]; rows = []
    for ys in itertools.product([0, 1], repeat=L):
        row = []
        for r, i in Z:
            v = np.zeros(4); v[i] = 1.
            p = 1.
            alpha = v.copy()
            for t, y in enumerate(ys):
                e = pattern[t % len(pattern)]
                like = np.array([emit(e, s, guess, slip) if y else 1 - emit(e, s, guess, slip) for s in STATES])
                alpha = alpha * like
                alpha = np.array([sum(alpha[k] * trans(e, STATES[k], G[r])[j] for k in range(4)) for j in range(4)])
            row.append(alpha.sum())
        rows.append(row)
    return np.array(rows)
def past(L, pattern, G, guess, slip, nu, w):
    Z = [(r, i) for r in range(len(G)) for i in range(4)]; rows = []
    for ys in itertools.product([0, 1], repeat=L):
        row = []
        for r in range(len(G)):
            alpha = w[r] * nu[r].copy()
            for t, y in enumerate(ys):
                e = pattern[t % len(pattern)]
                like = np.array([emit(e, s, guess, slip) if y else 1 - emit(e, s, guess, slip) for s in STATES])
                alpha = alpha * like
                alpha = np.array([sum(alpha[k] * trans(e, STATES[k], G[r])[j] for k in range(4)) for j in range(4)])
            row += list(alpha)
        rows.append(row)
    return np.array(rows)
def rank(M): s = np.linalg.svd(M, compute_uv=False); return int((s > 1e-10 * s[0]).sum())
guess = [.25, .20, .15]; slip = [.10, .12, .15]
Gint = [(.20, .10, .15, .15), (.02, .05, .03, .03)]
rng = np.random.default_rng(5); Grand = [tuple(rng.uniform(.05, .6, 4)) for _ in range(2)]
nu = [np.array([.6, .15, .15, .1])] * 2; w = [.4, .6]
for name, G in [('simulation values', Gint), ('random generic gains', Grand)]:
    for pattern in [(0, 1, 2), (2,), (0, 2), (0, 1)]:
        fr = [rank(future(L, pattern, G, guess, slip)) for L in range(1, 9)]
        pr = [rank(past(L, pattern, G, guess, slip, nu, w)) for L in range(1, 9)]
        print(f'{name:22s} pattern {pattern}: future-block rank L=1..8 {fr} | past-block rank {pr}  (hidden states = 8)')
# columns of the future block for (type 1, both mastered) and (type 2, both mastered)
F = future(6, (0, 1, 2), Gint, guess, slip); print('max |difference| between the two both-mastered columns:', np.abs(F[:, 3] - F[:, 7]).max())
