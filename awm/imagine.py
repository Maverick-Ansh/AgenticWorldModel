"""Policies trained purely in imagination, by BPTT through the frozen H-JEPA world model (SuperMimic-style).

SuperMimic (arXiv 2610.28480, Sec. 3.3): the policy "interacts with the learned world model instead of the
expensive physics simulator", imagines 32 steps ahead, and gets first-order gradients by backpropagation
through time through the frozen dynamics. Adam betas (0.9, 0.95), lr 2e-4, global grad clip 1.0, Gaussian
noise on policy inputs. Their Fig. 4b: without a critic, H = 32 still reaches the target and H <= 16 does not.

Here, plan by Fable 5.2:
  * world model = the level-1 H-JEPA predictor F[0] (frozen), on a 31-token sliding window.
  * actions = Gumbel-softmax straight-through one-hots into the WM's action encoder A[0], so the frozen
    model only ever sees one-hot moves, as in training.
  * loss = the GD planner's own goal cost on the 32-week imagined trajectory (-MRR probe + spend probe).
  * arm (b): pi(z) reads only the world-model latent. Arm (a): pi(z, h) also reads h = Qwen2.5-3B's
    mean-pooled hidden state of the SAME text context pack the CEO agent reads (PCA-32, computed once per
    pack, no LLM gradients): "does an LLM's reading of the business add anything a world model lacks?"
    Arm (a-shuf): (a) with h rolled across worlds at evaluation, so an (a) - (b) gain that survives
    shuffling is capacity, not information.
Deviations: no critic V(s_32), one seed, offline packs rebuilt from weekly means (np.repeat over days).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from .hjepa import CTX, NA, goal_cost
from .schema import IDX, REVERSIBLE, active
from .llm import context_pack

HZ, WIN, PCA = 32, 31, 32


class Policy(nn.Module):
    def __init__(self, d=128, dh=PCA, use_h=True):
        super().__init__()
        self.use_h = use_h
        self.net = nn.Sequential(nn.Linear(d + (dh if use_h else 0), 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                 nn.Linear(256, NA))

    def forward(self, z, h):
        return self.net(torch.cat([z, h], -1) if self.use_h else z)


class PackEncoder:
    """Qwen hidden state of a context pack: masked mean-pool of the last layer, then standardize + PCA-32."""
    def __init__(self, llm):
        self.tok, self.m, self.dev = llm.tok, llm.models[0], llm.devices[0]
        self.mu = self.sd = self.V = None

    @torch.no_grad()
    def raw(self, texts, bs=32):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok([t.split('GUARDRAILS')[0] for t in texts[i:i + bs]], return_tensors='pt', padding=True).to(self.dev)
            hs = self.m(**enc, output_hidden_states=True).hidden_states[-1].float()
            m = enc['attention_mask'][..., None].float()
            out.append(((hs * m).sum(1) / m.sum(1)).cpu())
        return torch.cat(out)

    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-4
        _, _, self.V = torch.pca_lowrank((X - self.mu) / self.sd, q=PCA, center=False)
        P = ((X - self.mu) / self.sd) @ self.V
        self.pm, self.ps = P.mean(0), P.std(0) + 1e-4

    def __call__(self, texts):
        return ((((self.raw(texts) - self.mu) / self.sd) @ self.V - self.pm) / self.ps)


def offline_starts(X, A, Wk, C, n, seed, weeks_hi=60):
    """Start points from offline businesses: 17 context weeks, the move history, a mask, and the text pack."""
    rng = np.random.default_rng(seed)
    b = rng.integers(0, X.shape[0], n); t0 = rng.integers(CTX, weeks_hi, n)
    x = np.stack([X[i, t - CTX:t] for i, t in zip(b, t0)])
    a1 = np.stack([A[i, t - CTX + 1:t] for i, t in zip(b, t0)])
    mask = np.zeros((n, NA), bool); mask[:, IDX['PRICE+20']] = True
    packs = []
    for j, (i, t) in enumerate(zip(b, t0)):
        c = {k: v[i, t - 1] for k, v in C.items()}
        act = active({k: np.atleast_1d(v) for k, v in c.items()}, 7 * t)[0]
        mask[j, 16:] = ~(act & REVERSIBLE)
        packs.append(context_pack(np.repeat(Wk[i, t - 4:t], 7, 0), c, 7 * t, 7 * CTX, [], []))
    return x, a1, mask, packs


def imagine(model, pol, x, a1, mask, h, noise=0.05, hard=True):
    """32-week imagined rollout through frozen F[0]; returns imagined latents (N, HZ, d) and actions."""
    zs, as_ = model.encode(x, torch.cat([a1, torch.zeros_like(a1[:, :1])], 1))
    Z, Aemb, acts, outs = zs[0], as_[0][:, :-1], [], []
    for t in range(HZ):
        logits = pol(Z[:, -1] + noise * torch.randn_like(Z[:, -1]), h) - 1e4 * mask
        a = Fn.gumbel_softmax(logits, tau=1., hard=hard) if pol.training else Fn.one_hot(logits.argmax(-1), NA).float()
        Aemb = torch.cat([Aemb, model.A[0](a)[:, None]], 1)
        zz, aa = Z[:, -WIN:], Aemb[:, -WIN:]
        nxt = model.F[0](zz, aa)[:, -1:]
        Z = torch.cat([Z, nxt], 1); acts.append(a.argmax(-1)); outs.append(nxt)
    return torch.cat(outs, 1), torch.stack(acts, 1)


def train_policy(model, pol, data, steps=600, bs=512, dev='cuda', log=print):
    x, a1, mask, h = data
    opt = torch.optim.Adam(pol.parameters(), lr=2e-4, betas=(.9, .95))
    rng = np.random.default_rng(0)
    pol.to(dev).train()
    for it in range(steps):
        i = torch.tensor(rng.integers(0, len(x), bs))
        zh, _ = imagine(model, pol, x[i], a1[i], mask[i], h[i])
        loss = goal_cost(model, 0, zh).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.); opt.step()
        if it % 100 == 0 or it == steps - 1:
            log(f'  policy(h={pol.use_h}) it {it} imagined cost {loss.item():.4f}')
    return pol.eval()


@torch.no_grad()
def open_loop_nmse(model, X, A, dev, n=256, t0=20):
    """Guard: how fast does the frozen WM drift when rolled 32 weeks on LOGGED actions? (MRR-probe NMSE)"""
    from .hjepa import FD
    from .schema import O
    x = torch.tensor(X[:n, t0 - CTX:t0], device=dev)
    a_ctx = Fn.one_hot(torch.tensor(A[:n, t0 - CTX + 1:t0], device=dev), NA).float()
    zs, as_ = model.encode(x, torch.cat([a_ctx, torch.zeros_like(a_ctx[:, :1])], 1))
    Z, Aemb = zs[0], as_[0][:, :-1]
    fut = Fn.one_hot(torch.tensor(A[:n, t0:t0 + HZ], device=dev), NA).float()
    pr = []
    for t in range(HZ):
        Aemb = torch.cat([Aemb, model.A[0](fut[:, t])[:, None]], 1)
        nxt = model.F[0](Z[:, -WIN:], Aemb[:, -WIN:])[:, -1:]
        Z = torch.cat([Z, nxt], 1); pr.append(model.probe[0](nxt[:, 0])[:, 0].cpu().numpy())
    true = X[:n, t0:t0 + HZ, O['sub_mrr']]
    pr = np.stack(pr, 1)
    return {h: round(float(((pr[:, h - 1] - true[:, h - 1]) ** 2).mean() / (true[:, h - 1].var() + 1e-9)), 3) for h in (1, 8, 16, 32)}


def make_planner(model, pol, penc, shuffle=False):
    def planner(x, a1, mask, arm, g):
        from .runner import pack_for
        B = x.shape[0]
        h = penc([pack_for(arm, b, g, lessons=False) for b in range(B)]).to(x.device) if pol.use_h else torch.zeros(B, PCA, device=x.device)
        if shuffle:
            h = torch.roll(h, B // 2, 0)
        with torch.no_grad():
            zs, _ = model.encode(x, torch.cat([a1, torch.zeros_like(a1[:, :1])], 1))
            return (pol(zs[0][:, -1], h) - 1e4 * mask).argmax(-1).cpu().numpy()
    return planner
