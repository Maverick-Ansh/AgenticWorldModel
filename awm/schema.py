"""Shared schema: the moves Steve's agents can make and the facts the platform records.

Post: "Those agents work through tools granted to their function. Growth can work on campaigns,
engineering can work in the product repository and support can work in the support desk."

This module is the ONLY thing the simulator, the platform and the world model share. The world model
may know WHAT a move does to the business's controls (that is the platform's own record of what ran:
a price is now 10% higher, ads run for 14 days). It must learn from evidence what the move does to
customers. Hidden customer behaviour lives in sim.py, which worldmodel.py never imports.
"""
import numpy as np

OPTIONS = ['NOOP', 'PRICE+10', 'PRICE-10', 'PRICE+20', 'PRICE-20', 'ADS30', 'ADS100', 'PROMO20', 'PROMO40',
           'CREDITS1', 'CREDITS2', 'BUNDLE', 'SAVE_NEW', 'SAVE_ALL', 'FIX_OAUTH', 'LISTING']
IDX = {o: i for i, o in enumerate(OPTIONS)}
KINDS = ['price', 'ads', 'promo', 'credits', 'bundle', 'save', 'fix', 'listing']
KIND_OF = np.array([-1, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7])
REVERSIBLE = np.array([1, 1, 1, 1, 1, 1, 0, 1], bool)          # a shipped OAuth fix is not rolled back
ROLLBACK = 100                                                  # candidate code 100+k rolls back move kind k
FUNCTION_OF_KIND = ['product', 'growth', 'growth', 'growth', 'product', 'support', 'engineering', 'growth']
MIN_INTENSITY = {IDX['PRICE+20']: IDX['PRICE+10'], IDX['PRICE-20']: IDX['PRICE-10'], IDX['ADS100']: IDX['ADS30'],
                 IDX['PROMO40']: IDX['PROMO20'], IDX['CREDITS2']: IDX['CREDITS1']}
FOUNDER_OPTS = np.array([IDX[o] for o in ['PRICE+10', 'PRICE-10', 'PRICE+20', 'PRICE-20', 'ADS30', 'ADS100',
                                           'PROMO20', 'PROMO40', 'CREDITS1', 'CREDITS2']])
# Founders never tried these four, so the world model starts with prior-width beliefs about them.
UNTRIED_BY_FOUNDERS = ['bundle', 'save', 'fix', 'listing']
LEVERS = ['logpm', 'pup', 'ads', 'promo', 'cred', 'bundle', 'save_new', 'save_all', 'fix', 'listing']
OBS = ['visits', 'visits_ad', 'signups_org', 'signups_ad', 'activations', 'new_subs', 'cancels', 'saves', 'packs',
       'error_rate', 'list_price', 'sub_mrr', 'subs', 'spend', 'revenue']
O = {k: i for i, k in enumerate(OBS)}
CPC = 0.40                                                      # ad platform contract: $ per paid visit


def init_controls(B):
    z = lambda v: np.full(B, float(v))
    return dict(pm=z(1), pm_prev=z(1), ads=z(0), ads_end=z(-1e9), promo=z(0), promo_end=z(-1e9), cred=z(0),
                bundle=z(0), bundle_start=z(-1e9), save=z(0), fix_on=z(0), fix_done=z(0), listing=z(0),
                due=z(0), price_rev=z(1e9), cred_rev=z(1e9))


CONTROL_KEYS = list(init_controls(1))


def apply(c, code, day):
    """Apply candidate codes (B,) to a control dict in place; returns the one-time cost per world.

    Codes < 100 are OPTIONS; 100+k rolls back move kind k. Timed moves (ads, promo) run 14 days.
    """
    code = np.asarray(code)
    cost = np.zeros(code.shape)
    for name, f in (('PRICE+10', 1.1), ('PRICE-10', .9), ('PRICE+20', 1.2), ('PRICE-20', .8)):
        m = code == IDX[name]
        c['pm_prev'][m] = c['pm'][m]; c['pm'][m] *= f; c['price_rev'][m] = 1e9
    for name, v in (('ADS30', 30.), ('ADS100', 100.)):
        m = code == IDX[name]; c['ads'][m] = v; c['ads_end'][m] = day + 14
    for name, v in (('PROMO20', .2), ('PROMO40', .4)):
        m = code == IDX[name]; c['promo'][m] = v; c['promo_end'][m] = day + 14
    for name, v in (('CREDITS1', 1.), ('CREDITS2', 2.)):
        m = code == IDX[name]; c['cred'][m] = v; c['cred_rev'][m] = 1e9
    m = (code == IDX['BUNDLE']) & (c['bundle'] == 0); c['bundle'][m] = 1; c['bundle_start'][m] = day; cost[m] += 400
    c['save'][code == IDX['SAVE_NEW']] = 1
    c['save'][code == IDX['SAVE_ALL']] = 2
    m = (code == IDX['FIX_OAUTH']) & (c['fix_on'] == 0); c['fix_on'][m] = 1; cost[m] += 600
    m = (code == IDX['LISTING']) & (c['listing'] == 0); c['listing'][m] = 1; cost[m] += 200
    k = code - ROLLBACK
    m = k == 0; c['pm'][m] = c['pm_prev'][m]
    m = k == 1; c['ads_end'][m] = np.minimum(c['ads_end'][m], day)
    m = k == 2; c['promo_end'][m] = np.minimum(c['promo_end'][m], day)
    c['cred'][k == 3] = 0; c['bundle'][k == 4] = 0; c['save'][k == 5] = 0; c['listing'][k == 7] = 0
    c['due'] += cost
    return cost


def active(c, day):
    """(B, 8) bool: which move kinds are live right now (and could be rolled back)."""
    return np.stack([np.abs(c['pm'] - c['pm_prev']) > 1e-9, c['ads_end'] > day, c['promo_end'] > day, c['cred'] > 0,
                     c['bundle'] > 0, c['save'] > 0, c['fix_on'] > 0, c['listing'] > 0], 1)


def levers(c, day):
    """(B, 10) lever intensities on `day`, read from the platform's own control record."""
    return np.stack([np.log(c['pm']), np.zeros_like(c['pm']), c['ads'] / 100 * (c['ads_end'] > day),
                     c['promo'] * (c['promo_end'] > day), c['cred'], c['bundle'] * (day >= c['bundle_start'] + 10),
                     (c['save'] == 1) * 1., (c['save'] == 2) * 1., c['fix_done'], c['listing']], -1)
