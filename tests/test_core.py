"""Property tests: the simulator's invariants and the harness rules from the post.

Post: "The four hold Steve to the record inside a single task." In a simulator they hold by construction,
so they are tested here rather than reported as findings.
"""
import ast
import pathlib
import numpy as np
from awm.runner import World, Arm, true_delta, pack_for, execute, owner_ok, H
from awm.sim import pools
from awm.schema import IDX, ROLLBACK
from awm.platform import Platform, worker, grade

ROOT = pathlib.Path(__file__).resolve().parents[1]
W = World(0, n_alpha=3, n_gamma=3)


def test_people_conservation():
    s = W.s_end
    err = np.abs(pools(s) - s['pool0'] - (s['cum_in'] - s['cum_out'])) / pools(s)
    assert err.max() < 1e-9
    for k in ('T', 'K', 'Ny', 'Nm', 'Nd'):
        assert (s[k] >= 0).all()


def test_mrr_identity_and_noop_fork_is_exact():
    cand = np.zeros((W.B, 3), int)
    d, r7 = true_delta(W.s_end, W.p, H, cand, 4, 1)
    assert np.all(d == 0) and np.all(r7 == 0), "fork(noop) must equal the trunk exactly"


def test_history_rebuilds_byte_identical():
    W2 = World(0, n_alpha=3, n_gamma=3)
    assert np.array_equal(W.facts.v, W2.facts.v, equal_nan=True)


def test_world_model_cannot_see_the_simulator():
    for f in ('worldmodel.py', 'llm.py'):
        tree = ast.parse((ROOT / 'awm' / f).read_text())
        mods = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + \
               [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(m and m.endswith('sim') for m in mods), f"{f} imports the simulator"


def test_receipts_rehash_and_detect_tampering():
    f = W.facts.copy()
    vals, rec = f.query(200, 100, 150, 'strict')
    assert f.verify(rec)
    f.v[0, 120, 0] += 1.0
    assert not f.verify(rec), "a changed record must fail its receipt"


def test_strict_guard_refuses_the_future():
    vals, rec = W.facts.query(200, 0, H, 'strict')
    assert np.isnan(vals[:, 200:]).all() and rec['refused'] > 0
    vals, rec = W.facts.query(200, 0, H, 'off')
    assert rec['refused'] == 0


def test_agent_never_sees_secrets():
    a = Arm('llm_gate_k3', W)
    secrets = list(a.platform._secrets.values())
    for b in range(W.B):
        p = pack_for(a, b, H)
        assert 'sk_live_' not in p and not any(s in p for s in secrets)


def test_grader_catches_false_claims_and_self_report_does_not():
    a = Arm('x', W)
    rng = np.random.default_rng(0)
    caught, faults = 0, 0
    for i in range(60):
        fault = rng.random() < .3
        claim = worker(a.platform, f't{i}', a.s, i % W.B, IDX['LISTING'] if i % 2 else IDX['ADS30'], H, fault=fault)
        assert claim['claim'] == 'done'                       # the agent's own report always says done
        passed = grade(a.platform, claim)
        assert passed == (not fault)
        faults += fault; caught += fault and not passed
    assert faults > 0 and caught == faults


def test_tokens_are_scoped_and_short_lived():
    p = Platform(0)
    a = Arm('y', W)
    tok = p.issue('growth', 't', H)
    assert not p.call(tok, a.s, 0, IDX['FIX_OAUTH'], H)       # growth cannot touch the repository
    assert p.call(tok, a.s, 0, IDX['ADS30'], H)
    assert not p.call(tok, a.s, 0, IDX['ADS30'], H + 1)       # expired next day


def test_owner_veto_is_a_dated_fact_and_repeats_become_policy():
    a = Arm('z', W)
    assert not owner_ok(a, 0, IDX['PRICE+20'], H)
    assert not owner_ok(a, 0, IDX['PRICE+20'], H + 7)
    assert a.banned[0, IDX['PRICE+20']]
    f = a.platform.owner_facts
    assert len(f) == 2 and f[0]['id'] != f[1]['id'] and f[0]['answer'] == 'no' and f[1]['day'] == H + 7
    done = execute(a, np.array([IDX['FIX_OAUTH']] + [0] * (W.B - 1)), H)
    assert done[0] == IDX['FIX_OAUTH'] and a.platform.owner_facts[-1]['answer'] == 'yes'
