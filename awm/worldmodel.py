"""Learned world model of one business, fit only to receipted facts and the platform's action log.

Post: "The world model turns business signals into a learned representation. [...] Steve models how a
proposed action could change that state over time and what the downstream effects on KPIs might be."
Post: "Steve connects the metrics behind a goal in a KPI tree. For AlphaSocial, monthly recurring revenue
depends on subscriptions and average revenue per user. Subscriptions are affected by signups,
signup-to-paid conversion and churn."

What is FIXED (accounting the platform knows because it bills the customers):
    N[t+1] = N[t] + new[t] - cancels[t]
    MRR = P * (N - Nd) + 0.5 * P * Nd        (Nd = subscribers on the 50% save discount, a 90-day offer)
    P[t+1] = P[t] + (list_price - P[t]) / 30  (repricing at renewal)
What is LEARNED, per business: five Poisson rate heads on the KPI-tree edges,
    rate = exposure * exp(x . theta),   x = [1, lever traces at tau = 3, 14, 45 days]
    so  signups_org / visits         sa  signups_ad / paid visits     c   new_subs / EWMA14(signups)
    h   cancels / subs               sv  saves (discount inflow) / subs
and a latent demand filter (day-of-week + AR(1) on smoothed log visits).

Prior theta_lever ~ N(0, 1) and a Laplace posterior: a lever the founders never pulled keeps its prior
width, so the model KNOWS it does not know what a save discount or a bundle will do. Deliberate
misspecification (the sim has trial pools, pack holders returning, tenure buckets, gaming feedback; this
model has none of them) keeps it from being the true simulator in disguise.
"""
import numpy as np
from .schema import LEVERS, O, CPC, apply, levers

TAUS = (3., 14., 45.)
K = len(LEVERS)
NF = 1 + 3 * K
PUP_SCALE = 30.                      # price-increase impulses, scaled so their traces are O(0.1-1)
PRIOR_SD = 1.0
HEADS = {'so': ('signups_org', 'visits', ['promo', 'listing']),
         'sa': ('signups_ad', 'visits_ad', []),
         'c': ('new_subs', 'sig_ewma', ['logpm', 'promo', 'cred', 'bundle', 'fix']),
         'h': ('cancels', 'subs_prev', ['pup', 'promo', 'save_new', 'save_all', 'fix']),
         'sv': ('saves', 'subs_prev', ['save_new', 'save_all'])}


def design(L, prev_logpm, tr0):
    """Lever path (N, T, K) -> design (N, T, NF) of exponential traces, plus the final trace state."""
    L = L.copy()
    lp = np.concatenate([prev_logpm[:, None], L[:, :, 0]], 1)
    L[:, :, 1] = PUP_SCALE * np.maximum(0., np.diff(lp, axis=1))
    N, T, _ = L.shape
    X = np.empty((N, T, NF)); X[:, :, 0] = 1.
    tr = tr0.copy()
    for t in range(T):
        for j, tau in enumerate(TAUS):
            sl = slice(1 + j * K, 1 + (j + 1) * K)
            tr[:, sl] += (L[:, t] - tr[:, sl]) / tau
        X[:, t, 1:] = tr
    return X, tr


def ffill(a):
    out = a.copy()
    for t in range(1, a.shape[1]):
        m = np.isnan(out[:, t]); out[m, t] = out[m, t - 1]
    return out


def lastv(a):
    return ffill(a)[:, -1]


def ewma_prev(x, tau):
    """out[:, t] = EWMA of x over days < t (unknown days skipped); also returns the EWMA through the end."""
    B, D = x.shape
    out = np.full((B, D), np.nan); e = np.full(B, np.nan)
    for t in range(D):
        out[:, t] = e
        v = x[:, t]; ok = np.isfinite(v)
        e = np.where(ok, np.where(np.isfinite(e), e + (v - e) / tau, v), e)
    return out, e


def fit_poisson(X, y, logE, w, mu0, prec, iters=12):
    """Batched MAP Newton for Poisson regression with a Gaussian prior; returns theta and the Hessian."""
    F = X.shape[2]
    th = mu0.copy()
    I = np.eye(F)[None]
    for it in range(iters + 1):
        mu = np.exp(np.clip(logE + np.einsum('bdf,bf->bd', X, th), -30, 12)) * w
        Hs = np.matmul(X.transpose(0, 2, 1), X * mu[..., None]) + prec[:, :, None] * I
        if it == iters:
            break
        g = np.einsum('bdf,bd->bf', X, w * y - mu) - prec * (th - mu0)
        st = np.linalg.solve(Hs, g[..., None])[..., 0]
        st *= np.minimum(1., 3. / np.maximum(np.linalg.norm(st, axis=1, keepdims=True), 1e-12))
        th = th + st
    return th, Hs


class WorldModel:
    def __init__(self, E=16, seed=0):
        self.E, self.seed = E, seed

    def fit(self, V, LV, g, fit_hi=None):
        """V: (B, >=g, M) visible facts (NaN = unknown); LV: (B, >=g, K) action log.
        Parameters are fit on days < fit_hi (default g); the current state is read from days < g."""
        fh = g if fit_hi is None else fit_hi
        B = V.shape[0]
        Vf, Lf = V[:, :fh], LV[:, :fh]
        X, tr = design(Lf, np.zeros(B), np.zeros((B, NF - 1)))
        self.tr = tr if fh == g else design(LV[:, :g], np.zeros(B), np.zeros((B, NF - 1)))[1]
        self.prev_logpm = LV[:, g - 1, 0]
        f = lambda k: Vf[:, :, O[k]]
        sig = f('signups_org') + f('signups_ad')
        sig_e, _ = ewma_prev(sig, 14.)
        subs_prev = np.concatenate([np.full((B, 1), np.nan), ffill(f('subs'))[:, :-1]], 1)
        expo = dict(visits=f('visits'), visits_ad=f('visits_ad'), sig_ewma=sig_e, subs_prev=subs_prev)
        rng = np.random.default_rng(self.seed)
        self.th, self.draws, self.sd = {}, {}, {}
        for h, (yk, ek, rel) in HEADS.items():
            y, Ex = f(yk), expo[ek]
            w = np.isfinite(y) & np.isfinite(Ex) & (Ex > 1e-9)
            yy = np.where(w, y, 0.); lE = np.log(np.where(w, Ex, 1.)); w = w * 1.
            prec = np.full(NF, 1e6); prec[0] = 1 / 9.
            for l in rel:
                prec[1 + LEVERS.index(l)::K] = 1 / PRIOR_SD ** 2
            prec = np.broadcast_to(prec, (B, NF)).copy()
            mu0 = np.zeros((B, NF))
            mu0[:, 0] = np.log((yy.sum(1) + .5) / ((w * np.exp(lE)).sum(1) + 50.))
            th, Hs = fit_poisson(X, yy, lE, w, mu0, prec)
            cov = np.linalg.inv(Hs); cov = (cov + cov.transpose(0, 2, 1)) / 2
            Lc = np.linalg.cholesky(cov + 1e-10 * np.eye(NF))
            z = rng.standard_normal((B, self.E, NF))
            self.th[h] = th
            self.draws[h] = th[:, None] + np.einsum('bij,bej->bei', Lc, z)
            self.sd[h] = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
        # latent demand: day-of-week + AR(1) on smoothed, deseasonalised log visits
        lv = np.log(np.maximum(f('visits'), 1.))
        dw = np.arange(fh) % 7
        lo = max(0, fh - 182)
        base = np.nanmean(lv[:, lo:], 1)
        self.dow = np.stack([np.nanmean(lv[:, lo:][:, dw[lo:] == j], 1) - base for j in range(7)], 1)
        ds = lv - self.dow[:, dw]
        sm, _ = ewma_prev(ds, 3.)
        self.mu = np.nanmean(ds[:, max(0, fh - 365):], 1)
        d = sm - self.mu[:, None]
        a, b_ = d[:, 1:], d[:, :-1]
        ok = np.isfinite(a) & np.isfinite(b_)
        self.phi = np.clip(np.where(ok, a * b_, 0).sum(1) / (np.where(ok, b_ * b_, 0).sum(1) + 1e-9), .8, .995)
        self.dsd = 1.5 * np.nanstd(np.where(ok, a - self.phi[:, None] * b_, np.nan), 1)
        self.dev0 = np.nan_to_num(lastv(d[:, :g]))
        # current state of the KPI tree
        Vs = V[:, :g]
        self.N0 = lastv(Vs[:, :, O['subs']]); self.mrr0 = lastv(Vs[:, :, O['sub_mrr']])
        sv = np.nan_to_num(Vs[:, :, O['saves']]); Nd = np.zeros(B)
        for t in range(g):
            Nd = Nd * (1 - 1 / 90) + sv[:, t]
        self.Nd0 = np.minimum(Nd, .9 * self.N0)
        self.P0 = self.mrr0 / np.maximum(self.N0 - .5 * self.Nd0, 1.)
        self.sig0 = ewma_prev(Vs[:, :, O['signups_org']] + Vs[:, :, O['signups_ad']], 14.)[1]
        self.pbase = lastv(Vs[:, :, O['list_price']]) / np.exp(self.prev_logpm)
        return self

    def forecast(self, ctrl, g, cand, H=90):
        """Roll every candidate (B, M) forward H days under E posterior draws x demand paths.
        Returns goal score G (B, M, E) = mean sub-MRR over days 61-90 minus cost/3, and MRR at day +30."""
        B, M = cand.shape
        E = self.E
        C = {k: np.repeat(v, M) for k, v in ctrl.items() if k in ('pm', 'pm_prev', 'ads', 'ads_end', 'promo',
             'promo_end', 'cred', 'bundle', 'bundle_start', 'save', 'fix_on', 'fix_done', 'listing', 'due',
             'price_rev', 'cred_rev')}
        C['due'] = np.zeros(B * M)
        cost1 = apply(C, cand.ravel(), g).reshape(B, M, 1)
        LVf = np.stack([levers(C, g + t) for t in range(H)], 1)
        pend = (C['fix_on'] > 0) & (C['fix_done'] == 0)          # a fix in flight: assume it ships in ~12 days
        LVf[:, :, 8] = np.where(pend[:, None], (np.arange(H) >= 12)[None] * 1., LVf[:, :, 8])
        X, _ = design(LVf, np.repeat(self.prev_logpm, M), np.repeat(self.tr, M, 0))
        X = X.reshape(B, M, H, NF)
        r = {h: np.exp(np.clip(np.einsum('bmtf,bef->bmet', X, self.draws[h]), -25, 3)) for h in HEADS}
        xi = np.random.default_rng(self.seed + 1).standard_normal((B, E, H))
        dev = np.empty((B, E, H)); d = np.repeat(self.dev0[:, None], E, 1)
        for t in range(H):
            d = self.phi[:, None] * d + self.dsd[:, None] * xi[:, :, t]; dev[:, :, t] = d
        dw = (g + np.arange(H)) % 7
        Vis = np.exp(self.mu[:, None, None] + dev + self.dow[:, dw][:, None, :])[:, None]
        ads = LVf[:, :, 2].reshape(B, M, 1, H) * 100.
        cred = LVf[:, :, 4].reshape(B, M, 1, H)
        price = (np.repeat(self.pbase, M)[:, None] * np.exp(LVf[:, :, 0])).reshape(B, M, 1, H)
        rep = lambda v: np.broadcast_to(v[:, None, None], (B, M, E)).copy()
        N, Nd, P, sg = rep(self.N0), rep(self.Nd0), rep(self.P0), rep(np.nan_to_num(self.sig0))
        mrr = np.empty((B, M, E, H)); spend = np.zeros((B, M, E))
        for t in range(H):
            S = Vis[..., t] * r['so'][..., t] + ads[..., t] / CPC * r['sa'][..., t]
            new, canc, sav = sg * r['c'][..., t], N * r['h'][..., t], N * r['sv'][..., t]
            sg = sg + (S - sg) / 14
            N = np.maximum(N + new - canc, 0.)
            Nd = np.minimum(Nd * (1 - 1 / 90) + sav, N)
            P = P + (price[..., t] - P) / 30
            mrr[..., t] = P * (N - Nd) + .5 * P * Nd
            spend += ads[..., t] + 1.5 * cred[..., t] * S
        G = mrr[..., 60:90].mean(-1) - (spend + cost1) / 3
        return G, mrr[..., 29]
