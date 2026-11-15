"""Hidden-truth SaaS business: the thing Steve runs, and the thing its world model must learn.

Post: "Some inputs are directly observable, such as visits, activations, usage and releases. The ones that
are not directly measurable including demand, customer intent and retention risk have to be estimated."

Here those latents are real state variables: x (demand), the activation probability a_t (intent) and the
churn hazards h (retention risk). They are never written to the fact store. Every world draws its own
hidden customer behaviour, so the SAME move is good in one business and bad in another (pack promos are
harmful in most worlds, not all; a +20% price helps where conversion and churn are inelastic). A static
rule ("never promote packs") cannot match a model that learns each business.

Traps that mirror the post, mechanically:
  * credit-pack promo: share kappa of would-be subscribers buy a discounted pack instead, and mature
    subscribers trickle out to packs (delta). Pack cash lands this week, lost subscriptions compound over
    the quarter ("the week read up, the quarter read down").
  * save discount offered to everyone: subscribers fake-cancel to get 50% off (g0, spreading as the
    discounted share grows), as in AlphaSocial's "50 percent discount that was being gamed".
  * OAuth bug: error rate e raises churn and lowers activation; the engineering fix lands after a
    geometric delay (p=1/12 per day) and cuts e by 80% ("OAuth errors per 1,000 calls fell by 80 percent").
  * ads: Gamma-like worlds buy clicks that never convert ("more than 100,000 clicks without a single
    paying customer").
  * price: conversion responds at once, churn responds with a lag through the price-shock stock X.

All noise comes from a pre-drawn tape z[world, day, channel], so forks are exactly paired and any day
rebuilds byte-identically.
"""
import hashlib
import numpy as np
from .schema import init_controls, O, OBS

NCH = 11
H_DAYS, LIVE_DAYS, RUNOFF_DAYS = 365, 182, 90          # 26 weekly decisions, 90-day no-decision runoff
D_TOT = H_DAYS + LIVE_DAYS + RUNOFF_DAYS
POOLS = ['T', 'K', 'Ny', 'Nm', 'Nd']


def sample_params(n_alpha=16, n_gamma=16, seed=0):
    rng = np.random.default_rng(seed)
    B = n_alpha + n_gamma
    al = np.r_[np.ones(n_alpha), np.zeros(n_gamma)].astype(bool)
    U = lambda a, b: rng.uniform(a, b, B)
    full = lambda v: np.full(B, float(v))
    return dict(
        alpha=al * 1., Dbar=np.where(al, 450., 900.), p0=np.where(al, 49., 19.), sigma=full(.025),
        q_ad=np.where(al, U(.5, 2.), U(0, .05)), kappa=U(.1, .7), r_ret=U(.1, .9), buzz=U(0, 1.),
        eps_c=U(.5, 2.), eps_h=U(.005, .06), bundle=U(.9, 1.3), L_new=U(.8, 1.4), g0n=U(0, .004),
        beta_e=U(2e-4, 6e-4), e0=np.where(al & (rng.uniform(size=B) < .7), 12., 1.), act_resp=U(0., 1.5),
        g0=U(5e-4, 4e-3), c0=full(.17), pi0=full(.004), h0y=full(.010), h0m=full(.0045), delta=full(.0015),
        rho=full(.35))


def make_tape(seed, B, D):
    return np.random.default_rng(seed).standard_normal((B, D, NCH))


def draw(mu, z):
    mu = np.maximum(mu, 0.)
    return np.maximum(0., mu + np.sqrt(mu) * z)


def pools(s):
    return sum(s[k] for k in POOLS)


def init_state(p, burn=200, seed=0):
    """Approximate steady state, then burn in on a separate tape so day 0 of history is already 'lived in'."""
    B = len(p['Dbar'])
    s = init_controls(B)
    e = p['e0'].copy()
    S0 = p['sigma'] * p['Dbar']
    hb = p['beta_e'] * e
    Ny = p['c0'] * S0 * (1 - .01 * e) ** .7 / (p['h0y'] + hb + 1 / 60)
    s.update(x=np.zeros(B), T=14 * S0, K=p['pi0'] * 14 * S0 * 45, Ny=Ny, Nm=Ny / 60 / (p['h0m'] + hb),
             Nd=np.zeros(B), P=p['p0'].copy(), X=np.zeros(B), e=e, cum_in=np.zeros(B), cum_out=np.zeros(B))
    z = make_tape(seed + 12345, B, burn)
    for t in range(burn):
        step(s, p, z[:, t], t - burn)
    s['cum_in'][:] = 0; s['cum_out'][:] = 0
    s['pool0'] = pools(s)
    return s


def step(s, p, z, day):
    """Advance every world one day in place. Returns the day's observations (B, len(OBS))."""
    dow = 1 + .15 * np.cos(2 * np.pi * (day % 7) / 7)
    s['x'] = .97 * s['x'] + .05 * z[:, 0]                                  # latent demand
    V = draw(p['Dbar'] * dow * np.exp(s['x']), z[:, 1])
    spend_ads = s['ads'] * (s['ads_end'] > day)
    Vad = spend_ads / .40
    disc = s['promo'] * (s['promo_end'] > day)
    on = (disc > 0) * 1.
    L = np.where(s['listing'] > 0, p['L_new'], 1.)
    So = draw(p['sigma'] * L * V * (1 + p['buzz'] * on), z[:, 2])
    Sa = draw(p['sigma'] * p['q_ad'] * Vad, z[:, 3])
    S = So + Sa
    e = s['e']
    a = np.clip((.35 + .06 * s['cred'] * p['act_resp']) * (1 - .01 * e), .01, .95)   # latent intent
    A = np.minimum(draw(a * S, z[:, 10]), S)
    bun = np.where((s['bundle'] > 0) & (day >= s['bundle_start'] + 10), p['bundle'], 1.)
    c = np.clip(p['c0'] * s['pm'] ** (-p['eps_c']) * bun * (a / .35) ** .7, 0, .9)
    T, K, Ny, Nm, Nd = s['T'], s['K'], s['Ny'], s['Nm'], s['Nd']
    leave = T / 14
    conv = np.minimum(draw(c * leave, z[:, 4]), leave)
    div = p['kappa'] * on * conv                                          # would-be subscribers buy a pack
    C = conv - div
    pkb = np.minimum(draw(p['pi0'] * (1 + 3 * disc) * T, z[:, 5]), T - leave)
    pull = p['delta'] * on * Nm                                           # subscribers leave for packs
    Kout = K / 45
    ret = p['r_ret'] * Kout
    hb = p['beta_e'] * e + p['eps_h'] * s['X']                            # retention risk
    mat = Ny / 60
    Ry = np.minimum(draw((p['h0y'] + hb) * Ny, z[:, 6]), Ny - mat)
    Rm = np.minimum(draw((p['h0m'] + hb) * Nm, z[:, 7]), Nm - pull)
    sy = p['rho'] * Ry * (s['save'] >= 1)
    sm = p['rho'] * Rm * (s['save'] >= 2)
    N = Ny + Nm + Nd
    G = np.where(s['save'] >= 2, np.minimum(draw(p['g0'] * (1 + 20 * Nd / np.maximum(N, 1)) * Nm, z[:, 8]),
                                            Nm - pull - Rm), 0.)            # fake cancellations
    Gy = np.where(s['save'] >= 1, np.minimum(p['g0n'] * Ny, Ny - mat - Ry), 0.)   # newer subs game it too
    Ndo = Nd / 90
    Rd = np.minimum((p['h0m'] + hb) * Nd, Nd - Ndo)
    exits = (leave - conv) + (Kout - ret) + (Ry - sy) + (Rm - sm) + Rd
    s['T'] = T + S + ret - leave - pkb
    s['K'] = K + pkb + div + pull - Kout
    s['Ny'] = Ny + C - Ry - mat - Gy
    s['Nm'] = Nm + mat - Rm - G - pull + Ndo
    s['Nd'] = Nd + sy + sm + G + Gy - Ndo - Rd
    s['cum_in'] = s['cum_in'] + S
    s['cum_out'] = s['cum_out'] + exits
    price = p['p0'] * s['pm']
    dP = (price - s['P']) / 30
    s['X'] = s['X'] + np.maximum(0, dP) / s['P'] - s['X'] / 20               # delayed price-shock churn
    s['P'] = s['P'] + dP
    done = (s['fix_on'] > 0) & (s['fix_done'] == 0) & (z[:, 9] < -1.383)    # Phi(-1.383) = 1/12
    s['e'] = np.where(done, .2 * e, e)
    s['fix_done'] = np.where(done, 1., s['fix_done'])
    mrr = s['P'] * (s['Ny'] + s['Nm']) + .5 * s['P'] * s['Nd']
    packs = pkb + div + pull
    spend = spend_ads + .5 * s['cred'] * S + s['due']
    s['due'] = np.zeros_like(s['due'])
    rev = mrr / 30 + .6 * price * (1 - disc) * packs - spend
    out = np.empty((len(V), len(OBS)))
    for k, v in (('visits', V), ('visits_ad', Vad), ('signups_org', So), ('signups_ad', Sa), ('activations', A),
                 ('new_subs', C), ('cancels', (Ry - sy) + (Rm - sm) + Rd + pull), ('saves', sy + sm + G + Gy),
                 ('packs', packs), ('error_rate', s['e']), ('list_price', price), ('sub_mrr', mrr),
                 ('subs', s['Ny'] + s['Nm'] + s['Nd']), ('spend', spend), ('revenue', rev)):
        out[:, O[k]] = v
    return out


def fork(d, n):
    """Replicate every world n times (world-major), for counterfactual branches."""
    return {k: np.repeat(v, n, axis=0) for k, v in d.items()}


def copy_state(s):
    return {k: v.copy() for k, v in s.items()}


def state_hash(s, b=None):
    h = hashlib.sha256()
    for k in sorted(s):
        h.update(np.ascontiguousarray(s[k] if b is None else s[k][b]).tobytes())
    return h.hexdigest()[:16]
