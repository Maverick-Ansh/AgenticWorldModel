"""Turn raw run records into the claim table: every number bracketed by noop (floor) and oracle (ceiling)."""
import json
import os
import pickle
import numpy as np
from .schema import OPTIONS, KIND_OF
from .sim import LIVE_DAYS
from .runner import GATE, MYOPIC, LLM_ARMS


def rank(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1]) if len(a) > 2 else float('nan')


def boot(f, n, reps=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array([f(rng.integers(0, n, n)) for _ in range(reps)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def summarize(W, out, log=print):
    R = pickle.load(open(os.path.join(out, 'live_results.pkl'), 'rb'))
    A = R['arms']
    sc = {n: a['mrr'][:, LIVE_DAYS:].mean(1) for n, a in A.items()}
    noop, orc = sc['noop'], sc['oracle']
    B = len(noop)
    norm = lambda x, i=slice(None): float((x[i] - noop[i]).sum() / (orc[i] - noop[i]).sum())
    S = dict(oracle_gap_pct=float(((orc - noop) / noop).mean() * 100), arms={})
    for n, x in sc.items():
        moves = A[n]['moves']
        harm = [m for m in moves if m['d'] < 0]
        end = lambda m: m['end'] if m['end'] is not None else (m['start'] + 14 if KIND_OF[m['code']] in (1, 2) else 365 + LIVE_DAYS)
        S['arms'][n] = dict(norm=norm(x), ci=boot(lambda i: norm(x, i), B), pct_vs_noop=float(((x - noop) / noop).mean() * 100),
                            moves=len(moves), harmful_moves=len(harm), harmful_days=int(sum(end(m) - m['start'] for m in harm)),
                            rollbacks=int(sum(m['end'] is not None for m in moves)),
                            grader_pass=A[n]['graded'][1] / max(A[n]['graded'][0], 1))
    d = lambda p, q: dict(diff=norm(sc[p]) - norm(sc[q]), ci=boot(lambda i: norm(sc[p], i) - norm(sc[q], i), B))
    S['C1_gate_top1_minus_myopic_goal'] = d('llm_gate_top1', 'llm_myopic_goal')
    S['C5_llm_k3_minus_rand_gate'] = d('llm_gate_k3', 'rand_gate')
    S['C5_wm_full_minus_llm_k3'] = d('wm_full', 'llm_gate_k3')
    S['AA_rand_gate_minus_rand_gate_b'] = d('rand_gate', 'rand_gate_b')
    S['gate_k3_minus_myopic_goal'] = d('llm_gate_k3', 'llm_myopic_goal')
    # C2: week-up / quarter-down trap instances among proposals
    c2 = {}
    for n in list(GATE) + list(MYOPIC):
        T, Rv, OK, Pc = [], [], [], []
        for r in A[n]['rec']:
            if r['props'] is None:
                continue
            K = r['props'].shape[1]
            T.append(r['td'][:, 1:1 + K]); Rv.append(r['r7'][:, 1:1 + K]); Pc.append(r['props'])
            OK.append(r['ok'] if 'ok' in r else np.zeros_like(r['props'], bool))
        T, Rv, OK, Pc = map(np.concatenate, (T, Rv, OK, Pc))
        trap = (T < 0) & (Rv > 0)
        good = T > 0
        promo = np.isin(Pc, [7, 8])
        c2[n] = dict(trap_share=float(trap.mean()), promo_trap_share=float(trap[promo].mean()) if promo.any() else None,
                     admit_trap=float(OK[trap].mean()) if n in GATE and trap.any() else None,
                     admit_good=float(OK[good].mean()) if n in GATE and good.any() else None,
                     admit_promo_trap=float(OK[trap & promo].mean()) if n in GATE and (trap & promo).any() else None,
                     rejected_bad_share=float((T[~OK] < 0).mean()) if n in GATE and (~OK).any() else None,
                     base_bad_share=float((T < 0).mean()))
    for n in MYOPIC:
        tm = [m for m in A[n]['moves'] if m['d'] < 0]
        c2[n]['kept_harmful'] = float(np.mean([m['end'] is None for m in tm])) if tm else None
        pm = [m for m in tm if m['code'] in (7, 8)]
        c2[n]['kept_harmful_promo'] = float(np.mean([m['end'] is None for m in pm])) if pm else None
    S['C2_C4'] = c2
    # C3: forecast calibration on every candidate the gate scored
    c3 = {}
    for n in GATE:
        F, Q1, Q9, T, TR, WK = [], [], [], [], [], []
        for r in A[n]['rec']:
            K = r['props'].shape[1]
            F.append(r['fmean']); Q1.append(r['fq10']); Q9.append(r['fq90']); T.append(r['td'][:, 1:1 + K])
            TR.append(r['tried']); WK.append(np.full(r['props'].shape, r['w']))
        F, Q1, Q9, T, TR, WK = [np.concatenate(x).ravel() for x in (F, Q1, Q9, T, TR, WK)]
        def cal(m):
            return dict(n=int(m.sum()), spearman=spearman(F[m], T[m]), cover80=float(((Q1[m] <= T[m]) & (T[m] <= Q9[m])).mean()),
                        sign_acc=float((np.sign(F[m]) == np.sign(T[m]))[np.abs(T[m]) > 20].mean()) if (m & (np.abs(T) > 20)).any() else None)
        al = np.ones(len(F), bool)
        c3[n] = dict(all=cal(al), tried=cal(TR), untried=cal(~TR), early=cal(WK < 9), late=cal(WK >= 17))
    S['C3'] = c3
    # LLM health
    S['llm'] = {}
    for n in LLM_ARMS:
        P = np.concatenate([r['props'] for r in A[n]['rec']])
        fails = np.concatenate([r['fail'] for r in A[n]['rec']])
        h = np.bincount(P.ravel(), minlength=len(OPTIONS)) / P.size
        S['llm'][n] = dict(parse_fail=float(fails.mean()), entropy_bits=float(-(h[h > 0] * np.log2(h[h > 0])).sum()),
                           top=[(OPTIONS[i], round(float(h[i]), 3)) for i in np.argsort(-h)[:5]])
    # C7 lessons + E4 forks
    L = [l for ls in A['llm_gate_k3']['lessons'] for l in ls]
    hr = lambda ls: (sum(l['hits'] for l in ls) / max(sum(l['trials'] for l in ls), 1), sum(l['trials'] for l in ls))
    acc, rej = [l for l in L if l['accepted']], [l for l in L if not l['accepted']]
    e4 = R['extra']['e4']
    c7 = dict(proposed=len(L), accepted=len(acc), hit_accepted=hr(acc), hit_rejected=hr(rej))
    if e4:
        td = np.concatenate([e['td'] for e in e4]); nl = np.concatenate([e['nlessons'] for e in e4]) > 0
        c7.update(fork_worlds_with_lessons=int(nl.sum()), pick_with=float(td[nl, 7].mean()) if nl.any() else None,
                  pick_without=float(td[nl, 8].mean()) if nl.any() else None,
                  harmful_props_with=float((td[nl, 1:4] < 0).mean()) if nl.any() else None,
                  harmful_props_without=float((td[nl, 4:7] < 0).mean()) if nl.any() else None,
                  ci_pick_diff=boot(lambda i: (td[nl][i, 7] - td[nl][i, 8]).mean(), int(nl.sum())) if nl.sum() > 2 else None)
    S['C7'] = c7
    S['owner_facts'] = {n: len(a['owner']) for n, a in A.items() if a['owner']}
    # C6 replays
    rp_path = os.path.join(out, 'replay.pkl')
    fig = dict(arms={n: [v['norm']] + v['ci'] for n, v in S['arms'].items()},
               traj={n: (A[n]['mrr'].mean(0)[::7]).round(0).tolist() for n in ['noop', 'oracle', 'llm_myopic_goal', 'llm_myopic_rev', 'llm_gate_top1', 'llm_gate_k3', 'wm_full']})
    if os.path.exists(rp_path):
        rp = pickle.load(open(rp_path, 'rb'))
        tv = {k: np.array([r['true'][k] for r in rp]) for k in rp[0]['true']}
        n = len(rp)
        S['C6'] = dict(n=n, rebuild_ok=float(np.mean([r['rebuild'] for r in rp])),
                       mean_true={k: float(v.mean()) for k, v in tv.items()},
                       steve_minus_founder=dict(diff=float((tv['strict'] - tv['founder']).mean()), ci=boot(lambda i: (tv['strict'] - tv['founder'])[i].mean(), n)),
                       llm_minus_founder=dict(diff=float((tv['llm'] - tv['founder']).mean()), ci=boot(lambda i: (tv['llm'] - tv['founder'])[i].mean(), n)),
                       leak_off_minus_strict=dict(diff=float((tv['off'] - tv['strict']).mean()), ci=boot(lambda i: (tv['off'] - tv['strict'])[i].mean(), n)),
                       leak_event_minus_strict=dict(diff=float((tv['event'] - tv['strict']).mean()), ci=boot(lambda i: (tv['event'] - tv['strict'])[i].mean(), n)),
                       refused_strict=float(np.mean([r['refused_strict'] for r in rp])), refused_off=float(np.mean([r['refused_off'] for r in rp])),
                       diverged=int(sum(r['pick_strict'] != r['founder'] for r in rp)),
                       diverged_and_steve_better=int(sum((r['pick_strict'] != r['founder']) and r['true']['strict'] > r['true']['founder'] for r in rp)),
                       fc_err={m: float(np.mean([abs(r[f'fc_{m}'] - r['true'][m]) for r in rp])) for m in ('strict', 'event', 'off')})
        fig['replay'] = {k: float(v.mean()) for k, v in tv.items()}
    # figure data: promo trap scatter + calibration scatter (subsampled)
    rng = np.random.default_rng(0)
    pts, cal = [], []
    for n in LLM_ARMS:
        for r in A[n]['rec']:
            K = r['props'].shape[1]
            for b, j in zip(*np.where(np.isin(r['props'], [7, 8]))):
                pts.append([round(float(r['r7'][b, 1 + j]), 1), round(float(r['td'][b, 1 + j]), 1)])
    for n in ('llm_gate_k3', 'wm_full'):
        for r in A[n]['rec']:
            K = r['props'].shape[1]
            for b, j in zip(*np.where(np.ones(r['props'].shape, bool))):
                cal.append([round(float(r['fmean'][b, j]), 1), round(float(r['td'][b, 1 + j]), 1), int(r['tried'][b, j])])
    fig['promo'] = [pts[i] for i in rng.permutation(len(pts))[:250]]
    fig['cal'] = [cal[i] for i in rng.permutation(len(cal))[:400]]
    json.dump(S, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    json.dump(fig, open(os.path.join(out, 'figdata.json'), 'w'))
    log('SUMMARY', json.dumps(S, separators=(',', ':')))
    return S
