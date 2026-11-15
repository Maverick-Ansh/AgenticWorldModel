"""Entry point. Stages: e0 (environment calibration), live (all arms, lockstep), replay, analyze.

    python run.py --stage all              # on the GPU box
    python run.py --stage all --fake --weeks 3 --out results_smoke   # CPU smoke test, random 'LLM'
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time
import numpy as np


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def e0(W, out):
    """Calibrate the environment BEFORE any gate run: trap base rates, shortcut checks, invariants."""
    from awm.runner import true_delta, World, H
    from awm.sim import pools
    from awm.schema import OPTIONS
    B, s = W.B, W.s_end
    mrr0 = s['P'] * (s['Ny'] + s['Nm']) + .5 * s['P'] * s['Nd']
    cand = np.concatenate([np.zeros((B, 1), int), np.broadcast_to(np.arange(1, 16), (B, 15))], 1)
    d, r7 = true_delta(s, W.p, H, cand, 16, W.seed + 3000)
    tab = {}
    for j, o in enumerate(OPTIONS[1:], 1):
        tab[o] = dict(harmful=float((d[:, j] < 0).mean()), clears_bar=float((d[:, j] > .005 * mrr0).mean()),
                      mean_pct=float((d[:, j] / mrr0).mean() * 100), week_up_quarter_down=float(((r7[:, j] > 0) & (d[:, j] < 0)).mean()))
    cons = np.abs(pools(s) - s['pool0'] - (s['cum_in'] - s['cum_out'])) / pools(s)
    W2 = World(W.seed)
    res = dict(options=tab, mrr0_alpha=float(mrr0[W.p['alpha'] > 0].mean()), mrr0_gamma=float(mrr0[W.p['alpha'] == 0].mean()),
               subs0=float((s['Ny'] + s['Nm'] + s['Nd']).mean()), conservation_max_rel_err=float(cons.max()),
               history_rebuild_identical=bool(np.array_equal(W.facts.v, W2.facts.v, equal_nan=True)),
               best_static_option_by_world=[OPTIONS[i] for i in d[:, 1:].argmax(1) + 1],
               founder_launches_per_world=float((W.founder_opt > 0).sum(1).mean()),
               preregistered=dict(C1='gate_top1 - myopic_goal >= 0.10 normalized, CI excludes 0; gate > noop by >= 0.05',
                                  C3='spearman >= 0.4 and 80% coverage in [0.65, 0.92]', noise='A/A rand_gate vs rand_gate_b'))
    json.dump(res, open(os.path.join(out, 'E0.json'), 'w'), indent=1)
    log('E0', json.dumps({k: v for k, v in res.items() if k not in ('options', 'best_static_option_by_world')}))
    for o, v in tab.items():
        log(f"  {o:10s} harmful {v['harmful']:.2f} clears {v['clears_bar']:.2f} mean {v['mean_pct']:+6.2f}% weekUp/qtrDown {v['week_up_quarter_down']:.2f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all')
    ap.add_argument('--fake', action='store_true')
    ap.add_argument('--out', default='results')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--weeks', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import awm.runner as R
    if a.weeks:
        R.WEEKS = a.weeks
    t = time.time()
    W = R.World(a.seed)
    log(f'world built in {time.time() - t:.1f}s, B={W.B}')
    stages = ['e0', 'live', 'replay', 'analyze'] if a.stage == 'all' else a.stage.split(',')
    llm = None
    if 'live' in stages or 'replay' in stages:
        from awm.llm import LLM, FakeLLM
        llm = FakeLLM() if a.fake else LLM(glob.glob('/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/*')[0])
        log('llm ready')
    if 'e0' in stages:
        e0(W, a.out)
    if 'live' in stages:
        R.run_live(W, llm, a.out, log=log)
        log('LIVE_DONE')
    if 'replay' in stages:
        from awm.replay import run_replays
        rp = run_replays(W, llm, log=log)
        pickle.dump(rp, open(os.path.join(a.out, 'replay.pkl'), 'wb'))
        log('REPLAY_DONE')
    if 'analyze' in stages:
        from awm.analysis import summarize
        summarize(W, a.out, log=log)
    log('ALL_DONE', f'{time.time() - t:.0f}s')


if __name__ == '__main__':
    main()
