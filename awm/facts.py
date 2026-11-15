"""Bitemporal fact store with receipts: the platform's record of what happened and when it was known.

Post: "Collected data is stored as typed facts. A query over those facts returns a receipt recording what
was read and when and the receipt is re-hashed on later reads. A KPI reading is accepted only if its
received query covers the entire measurement window."
Post (replays): "Everything the platform saves gets two stamps: one when it is saved, and a second, written
separately, when it is confirmed saved. Evidence counts at a cutoff only when both stamps land strictly
before it." / "If a source cannot provide a timestamp, the missing information is recorded as a gap."

Each fact (world, event day, metric) has saved = event + 1 (daily reconcile), cancellations reconcile up
to 10 days late, 5% of facts are confirmed 1-2 days after saving, and 1% never arrive (gaps). Lags are a
property of the data pipeline, not of the business, so every arm shares them.

Guard modes (for the replay leakage ablation):
  strict  both stamps < as_of                      (what the platform knew that day)
  event   event day < as_of, however late it landed (the "receipt created in March for January" leak)
  off     everything in the record, future included
"""
import hashlib
import numpy as np
from .schema import OBS, O


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


class FactStore:
    def __init__(self, B, D, seed):
        rng = np.random.default_rng(seed + 991)
        M = len(OBS)
        saved = np.broadcast_to(np.arange(D)[None, :, None] + 1., (B, D, M)).copy()
        saved[:, :, O['cancels']] += rng.integers(0, 10, (B, D))
        conf = saved + (rng.random((B, D, M)) < .05) * rng.integers(1, 3, (B, D, M))
        gap = rng.random((B, D, M)) < .01
        gap[:, :, O['list_price']] = False                       # the platform sets the price itself
        self.saved, self.conf, self.gap = saved, conf, gap
        self.v = np.full((B, D, M), np.nan)

    def copy(self):
        f = FactStore.__new__(FactStore)
        f.saved, f.conf, f.gap, f.v = self.saved, self.conf, self.gap, self.v.copy()
        return f

    def write(self, day, obs):
        self.v[:, day] = np.where(self.gap[:, day], np.nan, obs)

    def visible(self, as_of, mode, lo, hi):
        if mode == 'strict':
            return (self.saved[:, lo:hi] < as_of) & (self.conf[:, lo:hi] < as_of)
        if mode == 'event':
            return np.broadcast_to((np.arange(lo, hi) < as_of)[None, :, None], self.v[:, lo:hi].shape)
        return np.ones(self.v[:, lo:hi].shape, bool)

    def query(self, as_of, lo, hi, mode='strict', worlds=None):
        """Values for event days [lo, hi) as known at `as_of`; NaN = not known. Returns (values, receipt)."""
        v = self.v[:, lo:hi] if worlds is None else self.v[worlds, lo:hi]
        ok = self.visible(as_of, mode, lo, hi)
        ok = ok if worlds is None else ok[worlds]
        vals = np.where(ok, v, np.nan)
        refused = int((~ok & np.isfinite(v)).sum())               # facts that exist but the guard refused
        rec = dict(as_of=int(as_of), lo=int(lo), hi=int(hi), mode=mode, refused=refused, sha=_sha(vals),
                   worlds=None if worlds is None else list(np.atleast_1d(worlds).tolist()))
        return vals, rec

    def verify(self, rec):
        """Re-hash a receipt: the same query must return byte-identical evidence."""
        w = None if rec['worlds'] is None else np.array(rec['worlds'])
        vals, _ = self.query(rec['as_of'], rec['lo'], rec['hi'], rec['mode'], w)
        return _sha(vals) == rec['sha']

    def day_hash(self, b, day):
        return _sha(self.v[b, day])


def accepted(window):
    """KPI acceptance: the receipted window must be complete (no unknown day)."""
    return np.isfinite(window).all(axis=1)
