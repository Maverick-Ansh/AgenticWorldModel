"""The harness around the agents: credentials, evidence, independent grading, owner approvals.

Post: "Credentials. A billing key gives whoever holds it all the permissions it carries. Steve keeps
short-lived tokens on the platform and uses them to run integrations. The agent never sees the secret."
"Evidence. An agent's progress report can claim work that never happened. Steve records what actually ran."
"Grading. Steve uses that evidence to determine whether a task meets its completion criteria. The agent is
never allowed to grade its own work."
"Owners intervene. [...] the owner's answer is stored with its own id and cited like evidence."

In a simulator these hold by construction, so they are shipped as property tests (tests/test_harness.py),
not as findings. Every arm's moves still go through this path, so the live runs exercise it.
"""
import hashlib
import numpy as np
from .schema import KINDS, KIND_OF, ROLLBACK, FUNCTION_OF_KIND, OPTIONS, IDX, CONTROL_KEYS, apply

TOOLS = {'growth': {'ads', 'promo', 'credits', 'listing'}, 'product': {'price', 'bundle'},
         'support': {'save'}, 'engineering': {'fix'}}
NEEDS_APPROVAL = {IDX['PRICE+20'], IDX['FIX_OAUTH']}       # big price moves and production releases


def kind_of_code(code):
    return code - ROLLBACK if code >= ROLLBACK else int(KIND_OF[code])


def ctrl_hash(c, b):
    return hashlib.sha256(np.array([c[k][b] for k in CONTROL_KEYS]).tobytes()).hexdigest()[:12]


class Platform:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed + 4242)
        self._secrets = {f: 'sk_live_' + rng.bytes(12).hex() for f in TOOLS}   # never leaves this object
        self.tokens, self.evidence, self.owner_facts, self.n = {}, [], [], 0

    def issue(self, function, task, day):
        tok = f'tok_{self.n:06d}'; self.n += 1
        self.tokens[tok] = (function, task, day + 1)               # short-lived: expires next day
        return tok

    def call(self, tok, ctrl, b, code, day):
        function, task, exp = self.tokens.get(tok, (None, None, -1))
        kind = KINDS[kind_of_code(code)]
        ok = function is not None and day < exp and kind in TOOLS[function]
        before = ctrl_hash(ctrl, b)
        if ok:
            one = np.zeros(len(ctrl['pm']), int); one[b] = code    # world b only; NOOP touches nothing
            apply(ctrl, one, day)
        self.evidence.append(dict(task=task, b=b, code=int(code), exit=0 if ok else 1, before=before,
                                  after=ctrl_hash(ctrl, b), day=day))
        return ok

    def owner(self, arm, b, code, day, rule):
        """Owner approval for gated actions. Returns True if approved; every answer is a dated fact."""
        yes = rule(b, code)
        self.owner_facts.append(dict(id=f'own_{len(self.owner_facts):05d}', arm=arm, b=b, option=OPTIONS[code],
                                     day=day, answer='yes' if yes else 'no'))
        return yes


def owner_rule(b, code):
    """The owners' standing preference in this replication: releases yes, +20% price no ('not yet')."""
    return code != IDX['PRICE+20']


def worker(platform, task, ctrl, b, code, day, fault=False):
    """A function-scoped agent. It gets a token, never the secret. Its claim says 'done' either way."""
    function = FUNCTION_OF_KIND[kind_of_code(code)]
    tok = platform.issue(function, task, day)
    if not fault:
        platform.call(tok, ctrl, b, code, day)
    return dict(task=task, b=b, code=int(code), claim='done')


def grade(platform, claim):
    """Independent check: the evidence, not the claim, decides whether the task met its criteria."""
    rows = [e for e in platform.evidence if e['task'] == claim['task'] and e['b'] == claim['b']]
    return any(r['exit'] == 0 and r['code'] == claim['code'] for r in rows)
