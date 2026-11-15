"""Sealed-day replays: Steve's plan without hindsight vs the founders' actual decision, on days that happened.

Post: "There is no staging environment for last March. [...] What you can do is keep that past well
enough to rerun it." / "A standing check rebuilds a chosen day twice from those records and fails unless
the two copies match byte for byte" / "a guard reads every answer on the way in, so anything from after the
cutoff is refused and recorded as a gap" / "Agreement with the founders proves nothing by itself: what the
fork exists to find is the decision where Steve diverged and hindsight sided with Steve. When the scored
suite runs, its grades land here."

Skyfall had not published replay scores. Here the simulator supplies what a real business cannot: the
exact counterfactual of BOTH decisions from the sealed state, on 16 shared noise tapes. Each trial gets a
fresh world model and its own view of the record (one trial per environment).
Leakage ablation: the same trial with the guard in 'event' mode (late-arriving facts leak in) and 'off'
(the future record leaks in). If leaking inflates the score, the guard is doing real work.
"""
import numpy as np
from .schema import OPTIONS, active, CONTROL_KEYS, O
from .sim import step, copy_state, H_DAYS as H
from .facts import _sha
from .worldmodel import WorldModel
from .runner import true_delta, admit, BAR_FRAC
from .llm import context_pack, parse

MODES = ('strict', 'event', 'off')


def sl(d, b):
    return {k: v[[b]].copy() for k, v in d.items()}


def rebuild_ok(W, b, day):
    """Re-run `day` twice from the sealed post-decision state; both copies must equal the stored record."""
    hs = []
    for _ in range(2):
        s = sl(W.post[day], b)
        o = step(s, sl(W.p, b), W.tape[[b], day], day)[0]
        hs.append(_sha(np.where(W.facts.gap[b, day], np.nan, o)))
    return hs[0] == hs[1] == W.facts.day_hash(b, day)


def pick(D, P, bar):
    ok, mean = admit(D, P, bar, P > 0)
    j = np.argmax(np.where(ok, mean, -np.inf), 1)
    return np.where(ok.any(1), P[np.arange(len(P)), j], 0), np.where(ok.any(1), mean[np.arange(len(P)), j], 0.)


def run_replays(W, llm, ntape=16, log=print):
    rng = np.random.default_rng(W.seed + 5)
    pairs = []
    for b in range(W.B):
        fd = np.where(W.founder_opt[b, 100:330] > 0)[0] + 100
        days = [int(d) for d in rng.choice(fd, min(3, len(fd)), replace=False)] if len(fd) else []
        quiet = [d for d in range(105, 330, 21) if W.founder_opt[b, d] == 0 and not active(W.snaps[d], d)[b].any()]
        if quiet:
            days.append(int(rng.choice(quiet)))
        pairs += [(b, d) for d in days]
    log(f'{len(pairs)} sealed days')
    prompts = []
    for b, c in pairs:
        v, _ = W.facts.query(c, c - 28, c, 'strict', worlds=b)
        prompts.append(context_pack(v, {k: W.snaps[c][k][b] for k in CONTROL_KEYS}, c, H, [], []))
    texts = llm.generate(prompts, seed=999)
    out = []
    allP = np.arange(1, 16)[None]
    for (b, c), tx in zip(pairs, texts):
        r = dict(b=b, c=c, founder=int(W.founder_opt[b, c]), rebuild=rebuild_ok(W, b, c - 1))
        sb, ctrl = sl(W.snaps[c], b), sl(W.snaps[c], b)
        codes = []
        for mode in MODES:
            V, rec = W.facts.query(c, 0, H, mode, worlds=[b])
            wm = WorldModel(16, seed=c).fit(V, W.LV[[b]], c, fit_hi=H if mode == 'off' else None)
            Gf, _ = wm.forecast(ctrl, c, np.concatenate([[[0]], allP], 1))
            code, fmean = pick(Gf[:, 1:] - Gf[:, :1], allP, BAR_FRAC * wm.mrr0)
            r[f'pick_{mode}'], r[f'fc_{mode}'], r[f'refused_{mode}'] = int(code[0]), float(fmean[0]), rec['refused']
            codes.append(int(code[0]))
            if mode == 'strict':
                P = np.array([parse(tx, 3, rng)[0]])
                G2, _ = wm.forecast(ctrl, c, np.concatenate([[[0]], P], 1))
                lc, lf = pick(G2[:, 1:] - G2[:, :1], P, BAR_FRAC * wm.mrr0)
                r['llm_props'], r['pick_llm'], r['fc_llm'] = P[0].tolist(), int(lc[0]), float(lf[0])
        cand = np.array([[0, r['founder']] + codes + [r['pick_llm']]])
        td, _ = true_delta(sb, sl(W.p, b), c, cand, ntape, W.seed + 4000)
        r['true'] = dict(zip(['founder'] + list(MODES) + ['llm'], td[0, 1:].round(2).tolist()))
        r['text'] = tx[:160]
        out.append(r)
    log('replays done')
    return out
