"""Lockstep runner: every arm runs the same 32 businesses on the same noise tape, decision by decision.

Post: "The high level flow compares possible plans over a time horizon, carries out the one with the best
predicted result and measures the effects throughout that horizon while updating its estimates."
Post: "A candidate move rolls forward against the model, its futures are compared, and only a move that
clears the bar becomes tasks, so the mistake, when there is one, is spent in the model instead of on
customers."

Timeline per business: 365 days of founder history (backfilled into the fact store), 26 weekly decisions
(182 live days), then a 90-day runoff with no decisions. Score = mean true subscription MRR over the
runoff. Every proposed or executed move is also scored by an EXACT counterfactual: fork the true
simulator, run the move and NOOP on 16 shared noise tapes, compare the goal score G (mean sub-MRR over
days 61-90 minus cost/3). Skyfall cannot do this for real businesses; a simulator can.
"""
import copy
import os
import pickle
import time
import zlib
import numpy as np
from .schema import (OPTIONS, IDX, KIND_OF, REVERSIBLE, ROLLBACK, MIN_INTENSITY, FOUNDER_OPTS, LEVERS, O,
                     CONTROL_KEYS, apply, active, levers)
from .sim import (sample_params, init_state, make_tape, step, fork, copy_state, H_DAYS as H, LIVE_DAYS,
                  RUNOFF_DAYS, D_TOT, NCH)
from .facts import FactStore, accepted
from .worldmodel import WorldModel
from .platform import Platform, worker, grade, owner_rule, NEEDS_APPROVAL, kind_of_code
from .llm import context_pack, parse

WEEKS = LIVE_DAYS // 7
GATE = {'rand_gate': ('rand', 3), 'rand_gate_b': ('rand', 3), 'wm_full': ('all', 15),
        'llm_gate_top1': ('llm', 1), 'llm_gate_k3': ('llm', 3)}
MYOPIC = {'rand_myopic': ('rand', 'sub_mrr'), 'llm_myopic_goal': ('llm', 'sub_mrr'), 'llm_myopic_rev': ('llm', 'revenue')}
STATIC = ['noop', 'founder', 'oracle', 'sc_price10', 'sc_ads100', 'sc_promo', 'sc_kitchen']
LLM_ARMS = ['llm_myopic_goal', 'llm_gate_top1', 'llm_gate_k3', 'llm_myopic_rev']
ALL_ARMS = STATIC + list(MYOPIC) + list(GATE)
LEVER_OF_KIND = [0, 2, 3, 4, 5, 6, 8, 9]
BAR_FRAC = .005


def founder_step(s, g, rng, launch):
    """Founders' habit: every 21 days, w.p. 0.6, try a price/ads/promo/credits move for 14-28 days."""
    m = g >= s['price_rev']; s['pm'][m] = s['pm_prev'][m]; s['price_rev'][m] = 1e9
    m = g >= s['cred_rev']; s['cred'][m] = 0; s['cred_rev'][m] = 1e9
    if not launch or g % 21:
        return None
    B = len(s['pm'])
    opt = np.where(rng.random(B) < .6, rng.choice(FOUNDER_OPTS, B), 0)
    dur = rng.integers(14, 29, B)
    apply(s, opt, g)
    k = KIND_OF[opt]
    s['ads_end'] = np.where(k == 1, g + dur, s['ads_end']); s['promo_end'] = np.where(k == 2, g + dur, s['promo_end'])
    s['price_rev'] = np.where(k == 0, g + dur, s['price_rev']); s['cred_rev'] = np.where(k == 3, g + dur, s['cred_rev'])
    return opt


class World:
    """32 businesses: hidden parameters, a shared noise tape, and 12 months of founder history."""
    def __init__(self, seed=0, n_alpha=16, n_gamma=16):
        self.seed = seed
        self.p = sample_params(n_alpha, n_gamma, seed)
        self.B = B = len(self.p['Dbar'])
        s = init_state(self.p, seed=seed)
        self.tape = make_tape(seed + 1, B, D_TOT)
        self.facts = FactStore(B, D_TOT, seed)
        self.LV = np.zeros((B, D_TOT, len(LEVERS)))
        self.founder_opt = np.zeros((B, H), int)
        self.snaps, self.post = {}, {}
        self.founder_rng = np.random.default_rng(seed + 7)
        for g in range(H):
            self.snaps[g] = copy_state(s)                       # sealed-day state BEFORE that day's decision
            o = founder_step(s, g, self.founder_rng, True)
            if o is not None:
                self.founder_opt[:, g] = o
            self.post[g] = copy_state(s)                        # after the decision, before the day runs
            self.LV[:, g] = levers(s, g)
            self.facts.write(g, step(s, self.p, self.tape[:, g], g))
        self.s_end = s


def true_delta(s, p, g, cand, ntape=16, seed=0, Hh=90):
    """Exact counterfactual. cand (B, M), column 0 must be NOOP. Returns (dG, d_rev7) of shape (B, M)."""
    B, M = cand.shape
    n = M * ntape
    sf, pf = fork(s, n), fork(p, n)
    apply(sf, np.repeat(cand, ntape, 1).ravel(), g)
    z = np.random.default_rng([seed, g, 77]).standard_normal((B, 1, ntape, Hh, NCH))
    z = np.broadcast_to(z, (B, M, ntape, Hh, NCH)).reshape(B * n, Hh, NCH)
    mrr, spend, rev7 = np.zeros(B * n), np.zeros(B * n), np.zeros(B * n)
    for t in range(Hh):
        o = step(sf, pf, z[:, t], g + t)
        if t >= 60:
            mrr += o[:, O['sub_mrr']] / 30
        spend += o[:, O['spend']]
        if t < 7:
            rev7 += o[:, O['revenue']] / 7
    G = (mrr - spend / 3).reshape(B, M, ntape)
    r7 = rev7.reshape(B, M, ntape)
    return (G - G[:, :1]).mean(2), (r7 - r7[:, :1]).mean(2)


class Arm:
    def __init__(self, name, W):
        B = W.B
        self.name = name
        self.s = copy_state(W.s_end)
        self.facts = W.facts.copy()
        self.LV = W.LV.copy()
        self.rng = np.random.default_rng(zlib.crc32(name.encode()))
        self.frng = copy.deepcopy(W.founder_rng)
        self.mrr = np.zeros((B, LIVE_DAYS + RUNOFF_DAYS))
        self.pend = []                                           # myopic readouts: [b, launch day, code]
        self.probe_end, self.probe_kind = np.full(B, -1), np.full(B, -1)
        self.nprobe = np.zeros((B, 2), int)
        self.log = [[] for _ in range(B)]
        self.lessons = [[] for _ in range(B)]
        self.checks = []                                         # frozen +30-day forecasts awaiting evidence
        self.banned = np.zeros((B, len(OPTIONS)), bool)
        self.declines = np.zeros((B, len(OPTIONS)), int)
        self.moves = []
        self.rec = []
        self.platform = Platform(W.seed)
        self.graded = [0, 0]                                     # [tasks, tasks the grader passed]


def ctrl_of(a, b):
    return {k: a.s[k][b] for k in CONTROL_KEYS}


def lesson_lines(a, b, k=3):
    acc = [l for l in a.lessons[b] if l['accepted']]
    acc.sort(key=lambda l: -(l['hits'] + 1) / (l['trials'] + 2))
    return [f"{l['text']} (track record {l['hits']}/{l['trials']})" for l in acc[:k]]


def pack_for(a, b, g, lessons=True):
    v, _ = a.facts.query(g, g - 28, g, 'strict', worlds=b)
    return context_pack(v, ctrl_of(a, b), g, H, a.log[b], lesson_lines(a, b) if lessons else [])


def owner_ok(a, b, c, g):
    if c not in NEEDS_APPROVAL:
        return True
    yes = a.platform.owner(a.name, b, c, g, owner_rule)
    if not yes:
        a.declines[b, c] += 1
        if a.declines[b, c] >= 2:                                # a repeated correction becomes standing policy
            a.banned[b, c] = True
    return yes


def execute(a, codes, g, check_owner=True):
    """Every move goes owner gate -> function-scoped worker -> platform evidence -> independent grader."""
    done = np.zeros(len(codes), int)
    for b in np.where(codes != 0)[0]:
        c = int(codes[b])
        if check_owner and c < ROLLBACK and not owner_ok(a, b, c, g):
            continue
        claim = worker(a.platform, f'{a.name}:{g}:{b}:{c}', a.s, b, c, g)
        ok = grade(a.platform, claim)
        a.graded[0] += 1; a.graded[1] += ok
        if ok:
            done[b] = c
    return done


def rand_props(a, B, K):
    return np.stack([a.rng.choice(np.arange(1, len(OPTIONS)), K, replace=False) for _ in range(B)])


def tried(a, P, g):
    cols = np.array(LEVER_OF_KIND)[np.maximum(KIND_OF[P], 0)]
    out = np.zeros(P.shape, bool)
    for b in range(P.shape[0]):
        for j in range(P.shape[1]):
            c = cols[b, j]
            x = a.LV[b, :g, c] != 0
            if KIND_OF[P[b, j]] == 5:
                x = (a.LV[b, :g, 6] != 0) | (a.LV[b, :g, 7] != 0)
            out[b, j] = x.any()
    return out


def myopic_rollback(a, g, metric):
    """Today's agents: keep a move if the 6 days after it read better than the 7 days before."""
    B = a.s['pm'].shape[0]
    roll = np.full(B, -1)
    act = active(a.s, g)
    keep = []
    for b, L, code in a.pend:
        v, _ = a.facts.query(g, L - 7, L + 6, 'strict', worlds=b)
        x = v[:, O[metric]][None]
        if not accepted(x)[0]:
            if g - L < 28:
                keep.append([b, L, code])                       # wait for a complete receipted window
            continue
        worse = np.nanmean(x[0, 7:]) < np.nanmean(x[0, :7])
        k = kind_of_code(code)
        if worse and REVERSIBLE[k] and act[b, k]:
            roll[b] = k
        a.log[b].append(f"day {L - H}: {OPTIONS[code]} -> 7-day {metric} {'down' if worse else 'up'}, "
                        f"{'rolled back' if roll[b] >= 0 else 'kept'}")
    a.pend = keep
    return roll


def admit(D, props, bar, valid):
    """The gate: admit candidates whose pessimistic (20th pct) forecast gain beats the bar."""
    mean, q20 = D.mean(2), np.quantile(D, .2, axis=2)
    ok = valid & (q20 > bar[:, None])
    return ok, mean


def gate_decide(a, W, g, P):
    B, K = P.shape
    V, _ = a.facts.query(g, 0, g, 'strict')
    wm = WorldModel(16, seed=g).fit(V, a.LV, g)
    act = active(a.s, g)
    cand = np.concatenate([np.zeros((B, 1), int), P, np.broadcast_to(ROLLBACK + np.arange(8), (B, 8))], 1)
    Gf, m30 = wm.forecast(a.s, g, cand)
    D = Gf - Gf[:, :1]
    bar = BAR_FRAC * wm.mrr0
    Drb = D[:, 1 + K:]
    q20r, q80r = np.quantile(Drb, .2, axis=2), np.quantile(Drb, .8, axis=2)
    want = act & REVERSIBLE[None] & (q20r > 0)                    # re-pricing: q80(continue) < 0
    for b in np.where((a.probe_end >= 0) & (g >= a.probe_end))[0]:
        k = a.probe_kind[b]
        if act[b, k] and REVERSIBLE[k] and not q80r[b, k] < -bar[b]:
            want[b, k] = True                                    # a probe stays only if continuing clearly pays
        a.probe_end[b] = -1
    roll = np.where(want.any(1), np.argmax(np.where(want, q20r, -np.inf), 1), -1)
    Dp = D[:, 1:1 + K]
    valid = (P > 0) & ~a.banned[np.arange(B)[:, None], P] & (KIND_OF[P] != roll[:, None])
    ok, mean = admit(Dp, P, bar, valid)
    q80 = np.quantile(Dp, .8, axis=2)
    new, probe = np.zeros(B, int), np.zeros(B, bool)
    qtr = min((g - H) // 91, 1)
    for b in range(B):
        for j in np.argsort(-np.where(ok[b], mean[b], -np.inf)):
            if not ok[b, j]:
                break
            if owner_ok(a, b, int(P[b, j]), g):                  # owner says no -> amend to the next admitted move
                new[b] = P[b, j]; break
        if new[b] == 0 and a.nprobe[b, qtr] < 3:
            cands = valid[b] & (q80[b] > bar[b])
            if cands.any():
                c = int(P[b, np.argmax(np.where(cands, q80[b], -np.inf))])
                c = MIN_INTENSITY.get(c, c)
                if owner_ok(a, b, c, g):
                    new[b], probe[b] = c, True
                    a.nprobe[b, qtr] += 1; a.probe_end[b] = g + 14; a.probe_kind[b] = KIND_OF[c]
    jsel = np.argmax(P == new[:, None], 1)
    q = lambda x, z: np.quantile(x, z, axis=-1)
    m30s = m30[np.arange(B), 1 + jsel]
    info = dict(fmean=Dp.mean(2), fq10=q(Dp, .1), fq90=q(Dp, .9), ok=ok, probe=probe, tried=tried(a, P, g),
                bar=bar, m30=np.stack([q(m30s, .1), q(m30s, .5), q(m30s, .9)], 1), _wm=wm, _D=D)
    return roll, new, info


def check_lessons(a, g):
    """Post: "It can propose a lesson that states what should change and by when. A separate judge reviews
    the proposal and records a reason for accepting or rejecting it. Accepted lessons enter probation and
    rank higher when their predictions hold up." """
    keep = []
    for b, g0, code, q10, q50, q90 in a.checks:
        day = g0 + 29
        v, rec = a.facts.query(g, day, day + 1, 'strict', worlds=b)
        real = v[0, O['sub_mrr']]
        if not np.isfinite(real):
            if g - g0 < 50:
                keep.append([b, g0, code, q10, q50, q90])
            continue
        k = KIND_OF[code]
        sgn = np.sign(real - q50)
        for l in a.lessons[b]:                                    # probation: score earlier predictions
            if l['kind'] == k:
                l['trials'] += 1; l['hits'] += int(sgn == l['dir'])
        if q10 <= real <= q90:
            continue
        z = (real - q50) / ((q90 - q10) / 2.563 + 1e-9)
        accept = bool(abs(z) > 2 and a.facts.verify(rec))
        a.lessons[b].append(dict(
            kind=k, dir=int(sgn), day=g, z=float(z), accepted=accept, hits=0, trials=0,
            reason=f"|z|={abs(z):.1f} {'>' if accept else '<='} 2, receipt {rec['sha']} verified",
            text=f"{OPTIONS[code]} on day {g0 - H}: model said MRR ${q50:,.0f} at +30d, measured ${real:,.0f}; "
                 f"treat {['price', 'ads', 'promo', 'credits', 'bundle', 'save', 'fix', 'listing'][k]} moves here as "
                 f"{'worse' if sgn < 0 else 'better'} than forecast"))
    a.checks = keep


def oracle_decide(a, W, g):
    B = W.B
    cand = np.concatenate([np.zeros((B, 1), int), np.broadcast_to(np.arange(1, 16), (B, 15)),
                           np.broadcast_to(ROLLBACK + np.arange(8), (B, 8))], 1)
    d, _ = true_delta(a.s, W.p, g, cand, 8, W.seed + 1000)      # own tapes: the oracle never sees the live future
    want = active(a.s, g) & REVERSIBLE[None] & (d[:, 16:] > 0)
    roll = np.where(want.any(1), np.argmax(np.where(want, d[:, 16:], -np.inf), 1), -1)
    dn = d[:, 1:16].copy()
    dn[:, IDX['PRICE+20'] - 1] = -np.inf                          # owner never approves +20%
    dn[KIND_OF[np.arange(1, 16)][None] == roll[:, None]] = -np.inf
    return roll, np.where(dn.max(1) > 0, np.argmax(dn, 1) + 1, 0)


KITCHEN = [IDX[c] for c in ['FIX_OAUTH', 'BUNDLE', 'CREDITS1', 'LISTING', 'SAVE_ALL', 'PRICE+10', 'PROMO20']]


def static_decide(a, g, w):
    B = a.s['pm'].shape[0]
    n = a.name
    if n == 'sc_price10':
        return np.full(B, IDX['PRICE+10'] if w == 0 else 0)
    if n == 'sc_promo':
        return np.where(a.s['promo_end'] <= g, IDX['PROMO20'], 0)
    if n == 'sc_kitchen' and w < len(KITCHEN):
        return np.full(B, KITCHEN[w])
    if n in ('sc_ads100', 'sc_kitchen'):
        return np.where(a.s['ads_end'] <= g, IDX['ADS100'], 0)
    return np.zeros(B, int)


def decide(a, W, g, w, props, fails, extra):
    B, n = W.B, a.name
    P, roll, new, info = None, np.full(B, -1), np.zeros(B, int), {}
    if n == 'oracle':
        roll, new = oracle_decide(a, W, g)
    elif n in MYOPIC:
        src, metric = MYOPIC[n]
        P = props[n] if src == 'llm' else rand_props(a, B, 3)
        roll = myopic_rollback(a, g, metric)
        new = P[:, 0].copy()
    elif n in GATE:
        src, Kp = GATE[n]
        P = props[n][:, :Kp] if src == 'llm' else (rand_props(a, B, 3) if src == 'rand'
                                                   else np.broadcast_to(np.arange(1, 16), (B, 15)).copy())
        check_lessons(a, g)
        roll, new, info = gate_decide(a, W, g, P)
    elif n not in ('noop', 'founder'):
        new = static_decide(a, g, w)
    for b in np.where(roll >= 0)[0]:                              # rollbacks first
        if execute(a, np.where(np.arange(B) == b, ROLLBACK + roll[b], 0), g)[b]:
            for m in a.moves[::-1]:
                if m['b'] == b and KIND_OF[m['code']] == roll[b] and m['end'] is None:
                    m['end'] = g; break
            if n in GATE:
                a.log[b].append(f"day {g - H}: rolled back {['price', 'ads', 'promo', 'credits', 'bundle', 'save', 'fix', 'listing'][roll[b]]} (re-forecast)")
    cols = [np.zeros((B, 1), int)] + ([P] if P is not None else []) + [new[:, None]]
    td, r7 = true_delta(a.s, W.p, g, np.concatenate(cols, 1), 16, W.seed + 2000)
    done = execute(a, new, g, check_owner=n not in GATE)
    for b in np.where(done > 0)[0]:
        a.moves.append(dict(b=int(b), code=int(done[b]), start=g, end=None, d=float(td[b, -1])))
        if n in MYOPIC:
            a.pend.append([int(b), g, int(done[b])])
        if n in GATE:
            m = info['m30'][b]
            if not info['probe'][b]:                             # probes run at a different intensity
                a.checks.append([int(b), g, int(done[b]), m[0], m[1], m[2]])
            a.log[b].append(f"day {g - H}: {OPTIONS[done[b]]} admitted ({'probe' if info['probe'][b] else 'forecast gain'})")
    if n in GATE and n == 'llm_gate_k3' and 'nolessons' in props:
        Pn = props['nolessons']
        wm, D = info['_wm'], info['_D']
        Gn, _ = wm.forecast(a.s, g, np.concatenate([np.zeros((B, 1), int), Pn], 1))
        Dn = Gn - Gn[:, :1]
        okw, mw = admit(D[:, 1:4], P, info['bar'], P > 0)
        okn, mn = admit(Dn[:, 1:], Pn, info['bar'], Pn > 0)
        pick = lambda ok, m, PP: np.where(ok.any(1), PP[np.arange(B), np.argmax(np.where(ok, m, -np.inf), 1)], 0)
        pw, pn = pick(okw, mw, P), pick(okn, mn, Pn)
        tdx, _ = true_delta(a.s, W.p, g, np.concatenate([np.zeros((B, 1), int), P, Pn, pw[:, None], pn[:, None]], 1),
                            16, W.seed + 2000)
        nles = np.array([sum(l['accepted'] for l in a.lessons[b]) for b in range(B)])
        extra['e4'].append(dict(w=w, P=P.copy(), Pn=Pn.copy(), td=tdx, nlessons=nles))
    info = {k: v for k, v in info.items() if not k.startswith('_')}
    a.rec.append(dict(w=w, g=g, props=None if P is None else P.copy(), fail=fails.get(n), td=td, r7=r7,
                      new=new.copy(), done=done, roll=roll.copy(), **info))


def day_step(a, W, g, launch):
    founder_step(a.s, g, a.frng, launch)
    a.LV[:, g] = levers(a.s, g)
    o = step(a.s, W.p, W.tape[:, g], g)
    a.facts.write(g, o)
    a.mrr[:, g - H] = o[:, O['sub_mrr']]


def run_live(W, llm, out, arms_names=ALL_ARMS, fork_weeks=(14, 19, 23), log=print):
    ck = os.path.join(out, 'live_ckpt.pkl')
    if os.path.exists(ck):
        arms, w0, extra = pickle.load(open(ck, 'rb'))
        log(f'resumed at week {w0}')
    else:
        arms, w0, extra = {n: Arm(n, W) for n in arms_names}, 0, dict(e4=[], raw=[])
    prng = np.random.default_rng(W.seed + 31)
    for w in range(w0, WEEKS):
        t0 = time.time()
        g = H + 7 * w
        prompts, keys = [], []
        for n in LLM_ARMS:
            if n in arms:
                for b in range(W.B):
                    prompts.append(pack_for(arms[n], b, g, lessons=n == 'llm_gate_k3')); keys.append((n, b))
        if w in fork_weeks and 'llm_gate_k3' in arms:
            for b in range(W.B):
                prompts.append(pack_for(arms['llm_gate_k3'], b, g, lessons=False)); keys.append(('nolessons', b))
        texts = llm.generate(prompts, seed=w) if prompts else []
        t1 = time.time()
        props, fails = {}, {}
        for (n, b), tx in zip(keys, texts):
            codes, fail = parse(tx, 3, prng)
            props.setdefault(n, np.zeros((W.B, 3), int))[b] = codes
            fails.setdefault(n, np.zeros(W.B, bool))[b] = fail
            if b < 2 and w % 5 == 0:
                extra['raw'].append(dict(w=w, arm=n, b=b, prompt=prompts[keys.index((n, b))], text=tx))
        for a in arms.values():
            decide(a, W, g, w, props, fails, extra)
        for t in range(7):
            for a in arms.values():
                day_step(a, W, g + t, launch=a.name == 'founder')
        pickle.dump((arms, w + 1, extra), open(ck + '.tmp', 'wb')); os.replace(ck + '.tmp', ck)
        log(f'week {w:2d} g={g} llm {t1 - t0:5.1f}s total {time.time() - t0:5.1f}s prompts {len(prompts)}')
    for g in range(H + LIVE_DAYS, D_TOT):
        for a in arms.values():
            day_step(a, W, g, launch=False)
    res = dict(arms={n: dict(mrr=a.mrr, rec=a.rec, moves=a.moves, graded=a.graded, lessons=a.lessons,
                             owner=a.platform.owner_facts, log=a.log) for n, a in arms.items()}, extra=extra)
    pickle.dump(res, open(os.path.join(out, 'live_results.pkl'), 'wb'))
    return res
