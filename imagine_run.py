"""Imagination-trained policies (BPTT through frozen H-JEPA) with and without the LLM's reading of the pack.

    CUDA_VISIBLE_DEVICES=1 python imagine_run.py --out results_im --live results_hj/live_results.pkl
"""
import argparse
import glob
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
    ap.add_argument('--out', default='results_im'); ap.add_argument('--live', default='results_hj/live_results.pkl')
    ap.add_argument('--steps', type=int, default=600); ap.add_argument('--starts', type=int, default=1024)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    dev = 'cuda:0'
    from awm import hjepa as HJ, imagine as IM
    from awm.runner import World
    from awm.sim import LIVE_DAYS
    from awm.llm import LLM
    t = time.time()
    X, A, Wk, C = HJ.gen(4096, seed=100, raw=True); Xv, Av = HJ.gen(512, seed=200)
    torch.manual_seed(0)
    m = HJ.train(HJ.HJEPA(levels=3), X, A, steps=2500, dev=dev, log=log)
    for q in m.parameters():
        q.requires_grad_(False)
    res = dict(open_loop_nmse=IM.open_loop_nmse(m, Xv, Av, dev))
    log('open-loop MRR-probe NMSE on logged actions', res['open_loop_nmse'])
    llm = LLM(glob.glob('/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/*')[0], devices=[dev])
    pe = IM.PackEncoder(llm)
    x, a1, mask, packs = IM.offline_starts(X, A, Wk, C, a.starts, seed=5)
    hr = pe.raw(packs); pe.fit(hr)
    h = ((((hr - pe.mu) / pe.sd) @ pe.V - pe.pm) / pe.ps)
    data = lambda use_h: (torch.tensor(x, device=dev), torch.nn.functional.one_hot(torch.tensor(a1, device=dev), HJ.NA).float(),
                          torch.tensor(mask, device=dev), (h if use_h else torch.zeros_like(h)).to(dev))
    log(f'{len(packs)} packs encoded, {time.time() - t:.0f}s')
    W = World(0)
    pols = {}
    for use_h in (False, True):
        torch.manual_seed(1)
        pols[use_h] = IM.train_policy(m, IM.Policy(use_h=use_h), data(use_h), steps=a.steps, dev=dev, log=log)
        for q in pols[use_h].parameters():
            q.requires_grad_(False)
    arms = {'gd_planner': None, 'pi_wm': IM.make_planner(m, pols[False], pe),
            'pi_wm_llm': IM.make_planner(m, pols[True], pe), 'pi_wm_llm_shuf': IM.make_planner(m, pols[True], pe, shuffle=True)}
    ev = {}
    for name, pl in arms.items():
        torch.manual_seed(0)
        ev[name] = HJ.evaluate(W, m, name, dev=dev, planner=pl)
        log(name, f"{ev[name]['plan_sec_per_week']:.3f}s/week runoff MRR {ev[name]['mrr'][:, LIVE_DAYS:].mean():.0f}")
    R = pickle.load(open(a.live, 'rb'))['arms']
    sc = lambda mm: mm[:, LIVE_DAYS:].mean(1)
    noop, orc = sc(R['noop']['mrr']), sc(R['oracle']['mrr'])
    norm = lambda v, i=slice(None): float((v[i] - noop[i]).sum() / (orc[i] - noop[i]).sum())
    rng = np.random.default_rng(0)
    S = {k: dict(norm=norm(sc(v['mrr'])), plan_s=v['plan_sec_per_week'],
                 modal_move_share=float(np.max(v['move_hist'][1:]) / max(np.sum(v['move_hist'][1:]), 1)),
                 top_moves=np.argsort(v['move_hist'])[::-1][:4].tolist()) for k, v in ev.items()}
    for p, q in [('pi_wm_llm', 'pi_wm'), ('pi_wm_llm', 'pi_wm_llm_shuf'), ('pi_wm', 'gd_planner')]:
        u, v = sc(ev[p]['mrr']), sc(ev[q]['mrr'])
        bs = [norm(u, i) - norm(v, i) for i in (rng.integers(0, 32, 32) for _ in range(5000))]
        S[f'{p}_minus_{q}'] = dict(diff=norm(u) - norm(v), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    res.update(summary=S)
    json.dump(res, open(os.path.join(a.out, 'imagine.json'), 'w'), indent=1)
    log('IMAGINE_SUMMARY', json.dumps(res, separators=(',', ':')))
    log('IMAGINE_DONE', f'{time.time() - t:.0f}s')


if __name__ == '__main__':
    main()
