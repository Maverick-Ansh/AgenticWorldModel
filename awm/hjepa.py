"""H-JEPA world model of a business, planned top-down (Zhang, Terver, Rabbat, LeCun, Balestriero, arXiv 2610.06805).

Paper, Eq. (1): "z(l)_t = E(l)(z(l-1)_{t*s : t*s+w}),  a(l)_t = A(l)(a(l-1)_{t*s+w-1 : (t+1)*s+w-1})"
Paper, Eq. (2)-(3): teacher-forced latent prediction ||F(l)(z_<tau, a_<tau) - z_tau+1||^2 + lambda * SIGReg(Z(l)),
  "All levels are trained end-to-end, with gradients from upper-level objectives propagating through
  lower-level encoders."
Paper, Eq. (4): the top level optimizes macro-actions toward the goal; Eq. (5)-(6): each lower level optimizes
  its actions so that E(l+1)(its predicted states) match the subgoals predicted above (closed loop: K = 1).
  "We optimize action sequences at every level by gradient descent."

Mapping to Steve's business (deviations, stated where they happen):
  * level 1 step = one week (weekly means of the 15 daily facts + the 10 platform lever values); level-1
    action = this week's move (16 options + 8 rollbacks, one-hot); strides s2 = s3 = 2, windows w = 1 as in
    the paper -> level 2 = two weeks, level 3 = four weeks. Steve's goal stack (weekly -> monthly -> quarter).
  * the goal is a KPI, not a goal image: the top-level cost is a frozen-feature probe of (sub-MRR at the
    horizon) minus (mean spend), the analogue of Steve's goal score G. Probes are trained on detached
    latents, so they never shape the representation.
  * discrete level-1 actions are optimized as softmax-relaxed one-hots, then argmax'd for execution.
  * the model is trained offline on 4,096 OTHER simulated businesses under a random weekly policy, and
    evaluated on the 32 held-out businesses: it must infer each business's hidden behaviour in context from
    its last 17 weeks of receipted facts. It never sees an evaluation world's parameters.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from .schema import OBS, O, LEVERS, ROLLBACK, REVERSIBLE, IDX, apply, levers, active
from .sim import sample_params, init_state, make_tape, step

NA, FD = 24, len(OBS) + len(LEVERS)
CNT = ['visits', 'visits_ad', 'signups_org', 'signups_ad', 'activations', 'new_subs', 'cancels', 'saves', 'packs',
       'sub_mrr', 'subs']
CTX = 17                                     # context weeks; index 16 sits on the level-3 grid (multiple of 4)


def featurize(wk, ref, lev):
    f = np.zeros(wk.shape[:-1] + (FD,), np.float32)
    for k in CNT:
        f[..., O[k]] = np.log((wk[..., O[k]] + 1) / (ref[..., O[k]] + 1))
    f[..., O['error_rate']] = wk[..., O['error_rate']] / 10
    f[..., O['list_price']] = np.log(wk[..., O['list_price']] / ref[..., O['list_price']])
    m = ref[..., O['sub_mrr']] / 30
    f[..., O['spend']] = wk[..., O['spend']] / m
    f[..., O['revenue']] = wk[..., O['revenue']] / m
    f[..., len(OBS):] = lev
    return np.nan_to_num(f, nan=0., posinf=0., neginf=0.)


def gen(n, seed, weeks=80, raw=False):
    """Offline dataset: n businesses with fresh hidden parameters, random weekly moves and rollbacks.
    raw=True also returns weekly raw facts and per-week control snapshots (for rebuilding text packs)."""
    from .schema import CONTROL_KEYS
    p = sample_params(n // 2, n - n // 2, seed)
    s = init_state(p, seed=seed)
    tape = make_tape(seed + 1, n, weeks * 7)
    rng = np.random.default_rng(seed + 2)
    Wk, LVw, A = np.zeros((n, weeks, len(OBS))), np.zeros((n, weeks, len(LEVERS))), np.zeros((n, weeks), int)
    Cs = {c: np.zeros((n, weeks)) for c in CONTROL_KEYS}
    for k in range(weeks):
        g = 7 * k
        a = np.where(rng.random(n) < .5, rng.integers(1, 16, n), 0)
        kk = rng.integers(0, 8, n)
        rb = (rng.random(n) < .12) & active(s, g)[np.arange(n), kk] & REVERSIBLE[kk]
        a = np.where(rb, 16 + kk, a)
        apply(s, np.where(a >= 16, ROLLBACK + a - 16, a), g)
        A[:, k] = a
        for c in Cs:
            Cs[c][:, k] = s[c]
        acc = 0
        for t in range(7):
            acc = acc + step(s, p, tape[:, g + t], g + t)
        Wk[:, k], LVw[:, k] = acc / 7, levers(s, g + 6)
    X = featurize(Wk, Wk[:, :4].mean(1, keepdims=True), LVw)
    return (X, A, Wk, Cs) if raw else (X, A)


def probe_targets(x):
    return x[..., [O['sub_mrr'], O['spend']]]


class MLP(nn.Sequential):
    def __init__(self, i, o, h=256):
        super().__init__(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


class Pred(nn.Module):
    """F(l): causal transformer over (state + action) tokens, predicting the next latent state."""
    def __init__(self, d, layers=2, ctx=40):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(ctx, d))
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.0, batch_first=True, norm_first=True), layers)
        self.out = nn.Linear(d, d)

    def forward(self, z, a):
        T = z.shape[1]
        m = torch.triu(torch.full((T, T), float('-inf'), device=z.device), 1)
        return self.out(self.tf(z + a + self.pos[:T], mask=m))


def sigreg(z, M=64):
    """SIGReg (sketched isotropic-Gaussian regularizer): Epps-Pulley distance between the empirical
    characteristic function of random 1-D projections and that of N(0, 1)."""
    Z = z.reshape(-1, z.shape[-1])
    u = torch.randn(Z.shape[1], M, device=Z.device); u = u / u.norm(dim=0, keepdim=True)
    t = torch.linspace(-3, 3, 17, device=Z.device)
    tp = (Z @ u)[:, :, None] * t
    w = torch.exp(-t ** 2 / 2)
    return (((torch.cos(tp).mean(0) - w) ** 2 + torch.sin(tp).mean(0) ** 2) * w).mean()


class HJEPA(nn.Module):
    def __init__(self, levels=3, d=128, lam=10.):
        super().__init__()
        self.L, self.d, self.lam, self.s = levels, d, lam, 2
        # encoders end in a non-affine LayerNorm: at a near-zero init SIGReg's gradient vanishes (cos(tP) is flat
        # at P = 0; the first smoke run sat at z-std 0.03), and unit-norm latents can never shrink to a point
        enc = lambda i: nn.Sequential(MLP(i, d), nn.LayerNorm(d, elementwise_affine=False))
        self.E = nn.ModuleList([enc(FD)] + [enc(d) for _ in range(levels - 1)])
        self.A = nn.ModuleList([nn.Linear(NA, d, bias=False)] + [MLP(2 * d, d) for _ in range(levels - 1)])
        self.F = nn.ModuleList([Pred(d) for _ in range(levels)])
        self.probe = nn.ModuleList([nn.Linear(d, 2) for _ in range(levels)])

    def up(self, l, z_prev, a_prev):
        """Eq. (1) with s = 2, w = 1: pointwise encoder on every other state; actions pooled in pairs."""
        T = (z_prev.shape[1] // 2) * 2
        za = self.E[l](z_prev[:, 0:T:2]) if z_prev.shape[1] % 2 == 0 else self.E[l](z_prev[:, 0::2])
        aa = None if a_prev is None else self.A[l](a_prev[:, :T].reshape(a_prev.shape[0], T // 2, 2 * self.d))
        return za, aa

    def encode(self, x, a1):
        zs, as_ = [self.E[0](x)], [self.A[0](a1)]
        for l in range(1, self.L):
            z, a = self.up(l, zs[-1], as_[-1])
            zs.append(z); as_.append(a)
        return zs, as_

    def loss(self, x, a1):
        zs, as_ = self.encode(x, a1)
        y = probe_targets(x)
        tot, logs = 0., []
        for l in range(self.L):
            z, a = zs[l], as_[l]
            T = min(z.shape[1], a.shape[1])
            pred = self.F[l](z[:, :T - 1], a[:, :T - 1])
            lp = ((pred - z[:, 1:T]) ** 2).mean()
            lr = sigreg(z)
            lpr = ((self.probe[l](z.detach()) - y[:, ::2 ** l][:, :z.shape[1]]) ** 2).mean()
            tot = tot + lp + self.lam * lr + lpr
            logs.append((lp.item(), lr.item(), lpr.item(), z.std(0).mean().item()))
        return tot, logs


def train(model, X, A, steps=2500, bs=256, T=32, lr=3e-4, dev='cuda', log=print):
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt, At = torch.tensor(X), torch.tensor(A)
    n, W = X.shape[:2]
    rng = np.random.default_rng(0)
    for it in range(steps):
        i = rng.integers(0, n, bs); j = rng.integers(0, W - T, bs)
        idx = j[:, None] + np.arange(T)[None]
        x = Xt[i[:, None], idx].to(dev)
        a = torch.nn.functional.one_hot(At[i[:, None], idx + 1], NA).float().to(dev)    # a_t: move made after week t
        loss, logs = model.loss(x, a)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        if it % 500 == 0 or it == steps - 1:
            log(f'  L={model.L} it {it} ' + ' | '.join(f'pred {a_:.3f} sig {b:.4f} probe {c:.3f} zstd {d_:.2f}' for a_, b, c, d_ in logs))
    return model.eval()


def roll(F, zc, ac, afut):
    """Autoregressive rollout: context states zc (N,C,d), past actions ac (N,C-1,d), actions from now afut (N,H,d)."""
    z, a, out = zc, torch.cat([ac, afut[:, :1]], 1), []
    for h in range(afut.shape[1]):
        nxt = F(z, a)[:, -1:]
        out.append(nxt); z = torch.cat([z, nxt], 1)
        if h + 1 < afut.shape[1]:
            a = torch.cat([a, afut[:, h + 1:h + 2]], 1)
    return torch.cat(out, 1)


def goal_cost(model, l, zh):
    y = model.probe[l](zh)                       # (N, H, 2): predicted sub-MRR feature and spend feature
    return -y[:, -1, 0] + y[:, :, 1].mean(1)


def plan(model, x, a1, mask, N=16, iters=40, lr=.1, flat_H=12, top_H=3):
    """x (B, CTX, FD) context features; a1 (B, CTX-1, NA) past moves; mask (B, NA) disallowed moves.
    Returns the level-1 move for this week (B,) by top-down gradient-descent planning (Eq. 4-6)."""
    B, dev = x.shape[0], x.device
    rep = lambda t: t.repeat_interleave(N, 0)
    with torch.no_grad():
        zs, as_ = model.encode(x, torch.cat([a1, torch.zeros(B, 1, NA, device=dev)], 1))
    zs = [rep(z) for z in zs]; as_ = [rep(a) for a in as_]
    maskN = rep(mask)

    def opt_level(l, H, cost_fn, discrete):
        shape = (B * N, H, NA if discrete else model.d)
        p = (torch.randn(shape, device=dev) * (1. if discrete else .5)).requires_grad_(True)
        o = torch.optim.Adam([p], lr=lr)
        for _ in range(iters):
            emb = model.A[0](torch.softmax(p - 1e4 * maskN[:, None], -1)) if discrete else p
            zh = roll(model.F[l], zs[l], as_[l][:, :zs[l].shape[1] - 1], emb)
            c = cost_fn(zh)
            o.zero_grad(); c.sum().backward(); o.step()
        with torch.no_grad():
            emb = model.A[0](torch.softmax(p - 1e4 * maskN[:, None], -1)) if discrete else p
            zh = roll(model.F[l], zs[l], as_[l][:, :zs[l].shape[1] - 1], emb)
            c = cost_fn(zh).view(B, N)
            best = c.argmin(1) + torch.arange(B, device=dev) * N
        return zh[best], p[best].detach(), best

    if model.L == 1:
        _, p, _ = opt_level(0, flat_H, lambda zh: goal_cost(model, 0, zh), True)
    else:
        L = model.L - 1
        zt, _, _ = opt_level(L, top_H, lambda zh: goal_cost(model, L, zh), False)
        sub = zt[:, 0]                                                 # K = 1: first subgoal
        for l in range(L - 1, -1, -1):
            target = sub.repeat_interleave(N, 0)
            enc = lambda zh, l=l, target=target: ((model.E[l + 1](zh[:, 1]) - target) ** 2).mean(-1)   # Eq. (5)-(6)
            zl, p, _ = opt_level(l, 2, enc, l == 0)
            sub = zl[:, 0]
    return torch.softmax(p[:, 0] - 1e4 * mask, -1).argmax(-1).cpu().numpy()


def evaluate(W, model, name, dev='cuda', log=print, planner=None):
    """Closed-loop: plan each week from the arm's own receipted facts, execute through the platform."""
    from .runner import Arm, execute, day_step, H, WEEKS
    from .sim import LIVE_DAYS, D_TOT
    a = Arm(name, W)
    B = W.B
    acts = np.zeros((B, D_TOT), int)                                   # move made at the start of each day
    for b in range(B):
        for g in np.where(W.founder_opt[b] > 0)[0]:
            acts[b, g] = W.founder_opt[b, g]
    ref = np.nanmean(W.facts.v[:, :28], 1)[:, None]
    tsum = 0.
    for w in range(WEEKS):
        g = H + 7 * w
        V, _ = a.facts.query(g, g - 7 * CTX, g, 'strict')
        wk = np.nanmean(V.reshape(B, CTX, 7, -1), 2)
        lev = a.LV[:, g - 7 * CTX + 6:g:7]
        x = torch.tensor(featurize(wk, ref, lev), device=dev)
        blk = acts[:, g - 7 * CTX:g].reshape(B, CTX, 7).max(2)          # any move inside a week block
        a1 = torch.nn.functional.one_hot(torch.tensor(blk[:, 1:], device=dev), NA).float()
        act = active(a.s, g)
        mask = np.zeros((B, NA), bool); mask[:, IDX['PRICE+20']] = True                 # owner guardrail
        mask[:, 16:] = ~(act & REVERSIBLE[None])                                          # only live moves roll back
        t0 = time.time()
        mt = torch.tensor(mask, device=dev)
        mv = plan(model, x, a1, mt) if planner is None else planner(x, a1, mt, a, g)
        tsum += time.time() - t0
        code = np.where(mv >= 16, ROLLBACK + mv - 16, mv)
        done = execute(a, code, g)
        acts[:, g] = np.where(done != 0, mv, 0)
        for t in range(7):
            day_step(a, W, g + t, launch=False)
    for g in range(H + LIVE_DAYS, D_TOT):
        day_step(a, W, g, launch=False)
    hist = np.bincount(acts[:, H:H + LIVE_DAYS].ravel(), minlength=NA)
    return dict(mrr=a.mrr, plan_sec_per_week=tsum / WEEKS, move_hist=hist.tolist())


def probe_nmse(model, X, A, dev='cuda'):
    """Selective abstraction (paper Fig. 4): ridge probe NMSE of each feature from each level's latent."""
    with torch.no_grad():
        x = torch.tensor(X[:, :32], device=dev)
        a = torch.nn.functional.one_hot(torch.tensor(A[:, 1:33], device=dev), NA).float()
        zs, _ = model.encode(x, a)
    out = {}
    for l, z in enumerate(zs):
        Z = z.reshape(-1, z.shape[-1]).cpu().double().numpy()
        Y = X[:, :32][:, ::2 ** l][:, :z.shape[1]].reshape(-1, FD).astype(float)
        n = len(Z) // 2
        Zb = np.c_[Z, np.ones(len(Z))]
        th = np.linalg.solve(Zb[:n].T @ Zb[:n] + 1e-2 * np.eye(Zb.shape[1]), Zb[:n].T @ Y[:n])
        err = ((Zb[n:] @ th - Y[n:]) ** 2).mean(0) / (Y[n:].var(0) + 1e-9)
        out[l + 1] = {k: round(float(err[O[k]]), 3) for k in ['visits', 'signups_org', 'new_subs', 'cancels', 'subs', 'sub_mrr', 'spend']}
    return out
