"""Train flat JEPA and 3-level H-JEPA on 4,096 offline businesses, then plan closed-loop on the 32 held-out ones.

    python hjepa_run.py --out results --dev cuda:1
"""
import argparse
import json
import os
import pickle
import time
import numpy as np
import torch


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results'); ap.add_argument('--dev', default='cuda:0')
    ap.add_argument('--steps', type=int, default=2500); ap.add_argument('--n', type=int, default=4096)
    ap.add_argument('--seeds', type=int, default=2)
    a = ap.parse_args()
    from awm import hjepa as HJ
    from awm.runner import World, H
    from awm.sim import LIVE_DAYS
    t = time.time()
    X, A = HJ.gen(a.n, seed=100); Xv, Av = HJ.gen(512, seed=200)
    log(f'data {X.shape} in {time.time() - t:.0f}s')
    W = World(0)
    res = dict(probe={}, eval={})
    for seed in range(a.seeds):
        for L in (1, 3):
            torch.manual_seed(seed)
            m = HJ.train(HJ.HJEPA(levels=L), X, A, steps=a.steps, dev=a.dev, log=log)
            for q in m.parameters():
                q.requires_grad_(False)
            res['probe'][f'L{L}_s{seed}'] = HJ.probe_nmse(m, Xv, Av, a.dev)
            torch.manual_seed(seed)
            ev = HJ.evaluate(W, m, f'hjepa_L{L}_s{seed}', dev=a.dev, log=log)
            res['eval'][f'L{L}_s{seed}'] = ev
            log(f'L={L} seed={seed} plan {ev["plan_sec_per_week"]:.2f}s/week  runoff MRR {ev["mrr"][:, LIVE_DAYS:].mean():.0f}')
            pickle.dump(res, open(os.path.join(a.out, 'hjepa.pkl'), 'wb'))
    lr = os.path.join(a.out, 'live_results.pkl')
    if os.path.exists(lr):
        R = pickle.load(open(lr, 'rb'))['arms']
        sc = lambda m: m[:, LIVE_DAYS:].mean(1)
        noop, orc = sc(R['noop']['mrr']), sc(R['oracle']['mrr'])
        norm = lambda x, i=slice(None): float((x[i] - noop[i]).sum() / (orc[i] - noop[i]).sum())
        rng = np.random.default_rng(0)
        out = {}
        for k, ev in res['eval'].items():
            out[k] = dict(norm=norm(sc(ev['mrr'])), pct=float(((sc(ev['mrr']) - noop) / noop).mean() * 100),
                          plan_s=ev['plan_sec_per_week'], top_moves=np.argsort(ev['move_hist'])[::-1][:4].tolist())
        for s in range(a.seeds):
            h, f = sc(res['eval'][f'L3_s{s}']['mrr']), sc(res['eval'][f'L1_s{s}']['mrr'])
            bs = [norm(h, i) - norm(f, i) for i in (rng.integers(0, len(h), len(h)) for _ in range(5000))]
            out[f'hier_minus_flat_s{s}'] = dict(diff=norm(h) - norm(f), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        res['summary'] = out
        pickle.dump(res, open(os.path.join(a.out, 'hjepa.pkl'), 'wb'))
        json.dump(dict(summary=out, probe=res['probe']), open(os.path.join(a.out, 'hjepa.json'), 'w'), indent=1)
        log('HJEPA_SUMMARY', json.dumps(dict(summary=out, probe=res['probe']), separators=(',', ':')))
    log('HJEPA_DONE', f'{time.time() - t:.0f}s')


if __name__ == '__main__':
    main()
