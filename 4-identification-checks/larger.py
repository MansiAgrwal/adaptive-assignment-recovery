"""Larger competence structures (knowledge spaces over K=3 attributes, union-closed, contain the empty set and Q).
Part 1: ranks of future/past block-emission matrices for Z=(R,S), M=2, versus the bound M|C|-M+1.
Part 2: exposure-weighted type composition by state depth in a prerequisite chain, and its dependence on session length."""
import itertools, numpy as np
rng = np.random.default_rng(7)
K = 3; FULL = (1 << K) - 1
STRUCT = {'free (8 states)': list(range(8)),
          'chain a<b<c (4 states)': [0, 1, 3, 7],
          'c needs a and b (5 states)': [0, 1, 2, 3, 7]}
ACTS = [1, 2, 4, 3, 7]                     # requirement sets {a},{b},{c},{a,b},{a,b,c}
def kernel(C, e, g):                        # g[k] gain for attribute k; acquire fringe attributes in Q(e)
    idx = {s: i for i, s in enumerate(C)}; P = np.zeros((len(C), len(C)))
    for s in C:
        fr = [k for k in range(K) if (e >> k) & 1 and not (s >> k) & 1 and (s | 1 << k) in idx]
        for out in itertools.product([0, 1], repeat=len(fr)):
            p = np.prod([g[k] if o else 1 - g[k] for k, o in zip(fr, out)]) if fr else 1.
            s2 = s | sum(1 << k for k, o in zip(fr, out) if o); P[idx[s], idx[s2]] += p
    return P
def build(C, M):
    G = rng.uniform(.03, .3, (M, len(ACTS), K)); beta = rng.uniform(.05, .2, len(ACTS)); eta = rng.uniform(.1, .3, len(ACTS))
    T = [np.kron(np.eye(M), np.zeros((len(C), len(C)))) for _ in ACTS]
    for j, e in enumerate(ACTS):
        T[j] = np.zeros((M * len(C),) * 2)
        for r in range(M): T[j][r*len(C):(r+1)*len(C), r*len(C):(r+1)*len(C)] = kernel(C, e, G[r, j])
    p1 = np.array([[1 - beta[j] if (s & e) == e else eta[j] for s in C] * M for j, e in enumerate(ACTS)])  # P(Y=1|z,e)
    return T, p1
def future(T, p1, seq):                     # rows: response patterns, cols: z at block start
    F = np.ones((1, p1.shape[1]))
    for j in reversed(seq):                 # P(y_t..|z) = f(y_t|z,e) * sum_z' T(z,z') P(rest|z')
        nxt = F @ T[j].T if F.shape[0] else F
        F = np.vstack([nxt * p1[j], nxt * (1 - p1[j])])
    return F
def past(T, p1, seq, init):                 # rows: response patterns, cols: z at block end (joint)
    A = init[None, :]
    for j in seq:
        A = np.vstack([(A * p1[j]) @ T[j], (A * (1 - p1[j])) @ T[j]])
    return A
rank = lambda X: np.linalg.matrix_rank(X / np.abs(X).max(), tol=1e-10)
print('Part 1: block-matrix ranks, M=2 (bound M|C|-M+1)')
for name, C in STRUCT.items():
    M = 2; T, p1 = build(C, M); init = rng.dirichlet(np.ones(M * len(C)))
    seq_all = [j % len(ACTS) for j in range(9)]
    fr = [int(rank(future(T, p1, seq_all[:L]))) for L in range(1, 10)]
    pr = [int(rank(past(T, p1, seq_all[:L], init))) for L in range(1, 10)]
    F = future(T, p1, seq_all[:6]); top = [r * len(C) + C.index(FULL) for r in range(M)]
    print(f'  {name:28s} |Z|={M*len(C):2d} bound {M*len(C)-M+1:2d}  future L=1..9 {fr}  past {pr}  '
          f'max |diff| full-mastery columns {np.abs(F[:, top[0]] - F[:, top[1]]).max():.1e}')
print('Part 2: chain a<b<c, all users start empty, random assignment over the 5 activities, T=72, weights .4/.6')
C = [0, 1, 3, 7]; w = np.array([.4, .6])
for gf, gs in [(.20, .03), (.15, .05), (.10, .05)]:
    occ = []
    for g in (gf, gs):
        P = sum(kernel(C, e, [g] * K) for e in ACTS) / len(ACTS); x = np.array([1., 0, 0, 0]); N = np.zeros(4)
        for t in range(72): N += x; x = x @ P
        occ.append(N)
    occ = np.array(occ); om = (w[:, None] * occ) / (w[:, None] * occ).sum(0)
    print(f'  gains fast {gf:.2f} slow {gs:.2f}: expected occasions per state fast {occ[0,:3].round(1)}, slow {occ[1,:3].round(1)}; '
          f'fast-type share among exposures at empty, {{a}}, {{a,b}}: {om[0,:3].round(3)} (population share .400)')

print('Part 3: fast-type share in empty, {a}, {a,b} by session length (gains .20, .03)')
for T_ in [36, 72, 144, 288, 1000]:
    occ = []
    for g in (.20, .03):
        P = sum(kernel(C, e, [g] * K) for e in ACTS) / len(ACTS); x = np.array([1., 0, 0, 0]); N = np.zeros(4)
        for t in range(T_): N += x; x = x @ P
        occ.append(N)
    occ = np.array(occ); om = (w[:, None] * occ) / (w[:, None] * occ).sum(0)
    print(f'  {T_:5d} trials: {om[0,:3].round(3)}')
