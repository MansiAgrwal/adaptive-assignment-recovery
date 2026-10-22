"""ADEMP pilot: 2 x 2 factorial (assignment policy x estimator) on continuous learning trajectories.

Aims       Bias, precision and Wald-interval coverage for the joint-minus-single gain contrast when
           (a) assignment is uniform random vs history-dependent adaptive, and
           (b) the estimator is a homogeneous HMM vs a 2-type latent mixture HMM.
DGM        Two skills (a, b); states 00, a, b, ab. Activities A (needs a), B (needs b), AB (needs both).
           Persistent learner type u in {fast, slow}; type-specific per-trial gains; no within-session
           loss; gains only on practised skills. Probe blocks of 9 items (3 per requirement) at
           slots 0, 12, ..., 72; 72 practice trials with feedback; 24-hour block with loss; delayed probe.
Estimand   theta_k = sum_u pi_u [lambda_joint,k(u) - lambda_single,k(u)], for k = a, b.
Methods    EM (closed-form M-steps) with multistart; Wald intervals from the outer-product-of-scores
           information on the unconstrained scale, delta method for the contrast.
All parameter values are illustrative and uncalibrated.
"""
import numpy as np

# ---------------------------------------------------------------- structure
XA = np.array([0, 1, 0, 1]); XB = np.array([0, 0, 1, 1])          # state s = xa + 2*xb
ETA = np.stack([XA, XB, XA * XB])                                  # eta[r, s], r = A, B, AB
PROBE_SLOTS = [0, 12, 24, 36, 48, 60, 72]                          # + slot 73 (delayed probe)
NSLOT = 74
GAIN_KEYS = [('a', 0), ('a', 2), ('b', 1), ('b', 2)]               # (skill, activity): aA, aAB, bB, bAB

TRUE = dict(
    pi_fast=0.4,
    p0=np.array([.60, .15, .15, .10]),
    # per-trial gains [aA, aAB, bB, bAB] for (fast, slow)
    gains=np.array([[.20, .10, .15, .15],
                    [.02, .05, .03, .03]]),
    g=np.array([.25, .20, .15]), s=np.array([.10, .12, .15]),
    loss=np.array([.15, .15]))


def estimand(pi, gains):
    """theta_a, theta_b: baseline-type-weighted joint-minus-single contrast."""
    w = np.array([pi, 1 - pi]) if np.ndim(pi) == 0 else pi
    d = np.stack([gains[:, 1] - gains[:, 0], gains[:, 3] - gains[:, 2]], 1)  # (types, 2)
    return w @ d


# ---------------------------------------------------------------- data generation
def p_correct(state, r, g, s):
    return np.where(ETA[r, state] == 1, 1 - s[r], g[r])


def generate(n, policy, rng, P=TRUE, eps=0.15, window=18):
    u = (rng.random(n) >= P['pi_fast']).astype(int)               # 0 = fast, 1 = slow
    st = (rng.random(n)[:, None] > np.cumsum(P['p0'])).sum(1)
    probes = np.zeros((n, 8, 3), int)
    acts = np.zeros((n, 72), int); ys = np.zeros((n, 72), int)
    def probe(k):
        for r in range(3):
            probes[:, k, r] = rng.binomial(3, p_correct(st, r, P['g'], P['s']))
    probe(0)
    for t in range(72):
        if policy == 'random':
            e = rng.integers(0, 3, n)
        else:
            lo = max(0, t - window)
            ba, by = acts[:, lo:t], ys[:, lo:t]
            phat = np.stack([(1 + ((ba == a) * by).sum(1)) / (2 + (ba == a).sum(1)) for a in range(3)], 1)
            best = phat == phat.min(1, keepdims=True)
            pi = eps / 3 + (1 - eps) * best / best.sum(1, keepdims=True)
            e = (rng.random(n)[:, None] > np.cumsum(pi, 1)).sum(1)
            e = np.minimum(e, 2)
        y = (rng.random(n) < p_correct(st, e, P['g'], P['s'])).astype(int)
        acts[:, t] = e; ys[:, t] = y
        xa, xb = XA[st].copy(), XB[st].copy()
        gt = P['gains'][u]                                          # (n, 4)
        la = np.where(e == 0, gt[:, 0], np.where(e == 2, gt[:, 1], 0.))
        lb = np.where(e == 1, gt[:, 2], np.where(e == 2, gt[:, 3], 0.))
        xa = np.where((xa == 0) & (rng.random(n) < la), 1, xa)
        xb = np.where((xb == 0) & (rng.random(n) < lb), 1, xb)
        st = xa + 2 * xb
        if (t + 1) % 12 == 0:
            probe((t + 1) // 12)
    xa, xb = XA[st].copy(), XB[st].copy()
    xa = np.where((xa == 1) & (rng.random(n) < P['loss'][0]), 0, xa)
    xb = np.where((xb == 1) & (rng.random(n) < P['loss'][1]), 0, xb)
    st = xa + 2 * xb
    probe(7)
    return dict(acts=acts, ys=ys, probes=probes, u=u)


# ---------------------------------------------------------------- model
class Model:
    """K-type HMM on (type, state); K = 1 is the homogeneous model."""

    def __init__(self, K):
        self.K = K; self.Z = 4 * K

    # parameters: dict pi (K,), p0 (4,), gains (K,4), g (3,), s (3,), loss (2,)
    def kernels(self, prm):
        Z, K = self.Z, self.K
        T = np.zeros((4, Z, Z))
        for k in range(K):
            for e in range(3):
                for s0 in range(4):
                    xa, xb = XA[s0], XB[s0]
                    la = prm['gains'][k, 0] if e == 0 else (prm['gains'][k, 1] if e == 2 else 0.)
                    lb = prm['gains'][k, 2] if e == 1 else (prm['gains'][k, 3] if e == 2 else 0.)
                    pa = 1. if xa else la; pb = 1. if xb else lb       # P(mastered next)
                    for s1 in range(4):
                        T[e, 4 * k + s0, 4 * k + s1] = (pa if XA[s1] else 1 - pa) * (pb if XB[s1] else 1 - pb)
            for s0 in range(4):                                       # 24-hour block: loss only
                xa, xb = XA[s0], XB[s0]
                pa = 1 - prm['loss'][0] if xa else 0.; pb = 1 - prm['loss'][1] if xb else 0.
                for s1 in range(4):
                    T[3, 4 * k + s0, 4 * k + s1] = (pa if XA[s1] else 1 - pa) * (pb if XB[s1] else 1 - pb)
        return T

    def emissions(self, d, prm):
        n = d['acts'].shape[0]
        pc = np.where(ETA == 1, 1 - prm['s'][:, None], prm['g'][:, None])        # (3 r, 4 s)
        E = np.ones((n, NSLOT, 4))
        e, y = d['acts'], d['ys']
        E[:, :72, :] = np.where(y[:, :, None] == 1, pc[e], 1 - pc[e])
        for k, slot in enumerate(PROBE_SLOTS + [73]):
            c = d['probes'][:, k, :]                                              # (n, 3)
            E[:, slot, :] *= np.prod(pc[None] ** c[:, :, None] * (1 - pc[None]) ** (3 - c[:, :, None]), 1)
        return np.tile(E, (1, 1, self.K))                                         # (n, slot, Z)

    def kinds(self, d):
        return np.concatenate([d['acts'], np.full((d['acts'].shape[0], 1), 3)], 1)  # (n, 73)

    def init(self, prm):
        return (prm['pi'][:, None] * prm['p0'][None, :]).ravel()

    def forward(self, d, prm, T=None, E=None):
        T = self.kernels(prm) if T is None else T
        E = self.emissions(d, prm) if E is None else E
        kd = self.kinds(d); n = E.shape[0]
        masks = self.masks(kd)
        alpha = np.empty((n, NSLOT, self.Z)); c = np.empty((n, NSLOT))
        a = self.init(prm)[None] * E[:, 0]
        c[:, 0] = a.sum(1); alpha[:, 0] = a / c[:, :1]
        for j in range(1, NSLOT):
            a = np.empty((n, self.Z)); prev = alpha[:, j - 1]
            for kind, m in masks[j - 1]:
                a[m] = prev[m] @ T[kind]
            a *= E[:, j]
            c[:, j] = a.sum(1); alpha[:, j] = a / c[:, j:j + 1]
        return alpha, c, T, E, kd

    def masks(self, kd):
        key = id(kd)
        out = []
        for j in range(kd.shape[1]):
            col = kd[:, j]
            out.append([(k, np.flatnonzero(col == k)) for k in range(4) if (col == k).any()])
        return out

    def loglik_persons(self, d, prm):
        _, c, *_ = self.forward(d, prm)
        return np.log(c).sum(1)

    def em(self, d, prm, maxit=400, tol=1e-7):
        n = d['acts'].shape[0]; K = self.K
        prev = -np.inf; hist = []
        for it in range(maxit):
            alpha, c, T, E, kd = self.forward(d, prm)
            ll = np.log(c).sum(); hist.append(ll)
            if abs(ll - prev) < tol * n:
                return prm, ll, True, it, hist
            prev = ll
            beta = np.ones((n, NSLOT, self.Z))
            ecount = np.zeros((4, self.Z, self.Z))
            masks = self.masks(kd)
            for j in range(NSLOT - 1, 0, -1):
                b = E[:, j] * beta[:, j] / c[:, j:j + 1]
                ap = alpha[:, j - 1]; bnew = np.empty((n, self.Z))
                for kind, m in masks[j - 1]:
                    ecount[kind] += T[kind] * (ap[m].T @ b[m])
                    bnew[m] = b[m] @ T[kind].T
                beta[:, j - 1] = bnew
            gam = alpha * beta                                                       # (n, slot, Z)
            # M-step
            g0 = gam[:, 0].reshape(n, K, 4)
            new = dict(pi=g0.sum(2).mean(0), p0=g0.sum(1).mean(0))
            gains = np.zeros((K, 4))
            for k in range(K):
                blk = lambda kind: ecount[kind][4 * k:4 * k + 4, 4 * k:4 * k + 4]
                for gi, (sk, act) in enumerate(GAIN_KEYS):
                    M = blk(act); x = XA if sk == 'a' else XB
                    frm = x == 0
                    num = M[np.ix_(frm, x == 1)].sum(); den = M[frm].sum()
                    gains[k, gi] = num / den if den > 0 else prm['gains'][k, gi]
            new['gains'] = np.clip(gains, 1e-6, 1 - 1e-6)
            loss = np.zeros(2)
            for idx, x in enumerate([XA, XB]):
                M = sum(ecount[3][4 * k:4 * k + 4, 4 * k:4 * k + 4] for k in range(K))
                loss[idx] = M[np.ix_(x == 1, x == 0)].sum() / max(M[x == 1].sum(), 1e-12)
            new['loss'] = np.clip(loss, 1e-6, 1 - 1e-6)
            gs = gam.reshape(n, NSLOT, K, 4).sum(2)                                  # (n, slot, 4)
            corr = np.zeros((3, 2)); tri = np.zeros((3, 2))
            e, y = d['acts'], d['ys']
            for r in range(3):
                m = e == r                                                           # practice slots 0..71
                w = gs[:, :72][m]                                                    # (#, 4)
                for h in (0, 1):
                    sel = ETA[r] == h
                    tri[r, h] += w[:, sel].sum(); corr[r, h] += (w[:, sel].sum(1) * y[m]).sum()
                for k, slot in enumerate(PROBE_SLOTS + [73]):
                    w = gs[:, slot]; cnt = d['probes'][:, k, r]
                    for h in (0, 1):
                        sel = ETA[r] == h
                        tri[r, h] += 3 * w[:, sel].sum(); corr[r, h] += (w[:, sel].sum(1) * cnt).sum()
            new['g'] = np.clip(corr[:, 0] / tri[:, 0], 1e-4, 1 - 1e-4)
            new['s'] = np.clip(1 - corr[:, 1] / tri[:, 1], 1e-4, 1 - 1e-4)
            prm = new
        return prm, ll, False, maxit, hist

    # ---- unconstrained vector <-> parameters (for the information matrix)
    def pack(self, prm):
        lg = lambda p: np.log(p) - np.log1p(-p)
        v = [lg(prm['pi'][:-1] / prm['pi'].sum())] if self.K == 2 else []
        v += [np.log(prm['p0'][1:] / prm['p0'][0]), lg(prm['gains']).ravel(), lg(prm['g']), lg(prm['s']), lg(prm['loss'])]
        return np.concatenate(v)

    def unpack(self, v):
        ex = lambda x: 1 / (1 + np.exp(-x)); i = 0; prm = {}
        if self.K == 2:
            p = ex(v[0]); prm['pi'] = np.array([p, 1 - p]); i = 1
        else:
            prm['pi'] = np.array([1.])
        z = np.concatenate([[0.], v[i:i + 3]]); prm['p0'] = np.exp(z) / np.exp(z).sum(); i += 3
        prm['gains'] = ex(v[i:i + 4 * self.K]).reshape(self.K, 4); i += 4 * self.K
        prm['g'] = ex(v[i:i + 3]); i += 3
        prm['s'] = ex(v[i:i + 3]); i += 3
        prm['loss'] = ex(v[i:i + 2])
        return prm

    def contrast(self, prm):
        return estimand(prm['pi'], prm['gains'])

    def wald(self, d, prm, h=1e-4):
        """OPG information on the unconstrained scale; delta method for the two contrasts."""
        v = self.pack(prm); p = len(v)
        S = np.empty((d['acts'].shape[0], p))
        for i in range(p):
            dv = np.zeros(p); dv[i] = h
            S[:, i] = (self.loglik_persons(d, self.unpack(v + dv)) - self.loglik_persons(d, self.unpack(v - dv))) / (2 * h)
        info = S.T @ S
        try:
            cov = np.linalg.inv(info)
            if not np.all(np.isfinite(cov)) or np.any(np.diag(cov) <= 0): raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            return None, None
        J = np.empty((2, p))
        for i in range(p):
            dv = np.zeros(p); dv[i] = h
            J[:, i] = (self.contrast(self.unpack(v + dv)) - self.contrast(self.unpack(v - dv))) / (2 * h)
        se = np.sqrt(np.einsum('ip,pq,iq->i', J, cov, J))
        return se, float(np.linalg.cond(info))


def start_values(K, rng, d):
    prm = dict(pi=np.full(K, 1 / K) if K == 1 else np.array([.5, .5]),
               p0=rng.dirichlet(np.ones(4) * 8),
               g=np.clip(rng.uniform(.15, .3, 3), .01, .99), s=np.clip(rng.uniform(.05, .2, 3), .01, .99),
               loss=rng.uniform(.05, .3, 2))
    if K == 1:
        prm['gains'] = rng.uniform(.03, .15, (1, 4))
    else:  # one faster and one slower type, randomised magnitudes
        prm['gains'] = np.stack([rng.uniform(.10, .30, 4), rng.uniform(.01, .06, 4)])
        prm['pi'] = np.array([q := rng.uniform(.3, .6), 1 - q])
    return prm


def fit(model, d, rng, starts):
    best = None
    for _ in range(starts):
        prm, ll, conv, it, hist = model.em(d, start_values(model.K, rng, d))
        mono = bool(np.all(np.diff(hist) > -1e-6 * d['acts'].shape[0]))
        if best is None or ll > best[1]: best = (prm, ll, conv, it, mono)
    return best
