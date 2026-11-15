"""The CEO agent: an open LLM that reads a context pack and proposes this week's moves.

Post: "A task comes with a context pack containing the goal, the relevant state, lessons and guardrails."
Post: "The CEO agent plans using a world model [...] This results in initiatives which are broken into tasks."

Qwen2.5-3B-Instruct in fp16 on one T4 (no bf16 on compute capability 7.5). All worlds and all LLM arms
are generated in one left-padded batch per simulated week. The pack is built ONLY from strict receipted
facts and the platform's own records: no hidden simulator state, no credentials.
"""
import re
import numpy as np
from .schema import O, OPTIONS, IDX

SYS = ("You are the CEO agent of a small software business. The owner judges you on subscription MRR over "
       "the next quarter. Each week you propose moves; a platform executes one. Answer only in the format asked.")
MENU = """MOVES (code: what it does)
PRICE+10 / PRICE-10 / PRICE+20 / PRICE-20: change the subscription list price
ADS30 / ADS100: paid ads at $30 or $100 per day for 14 days
PROMO20 / PROMO40: 20% or 40% off one-time credit packs for 14 days
CREDITS1 / CREDITS2: more free credits for new signups at activation ($1.5 per signup per level)
BUNDLE: include credits in every subscription ($400 setup)
SAVE_NEW / SAVE_ALL: offer cancelling subscribers 50% off for 90 days (newer subscribers only / everyone)
FIX_OAUTH: engineering fixes the OAuth login failures ($600)
LISTING: rewrite the store listing and landing copy ($200)
NOOP: change nothing this week"""
_PAT = re.compile(r'PRICE[+-](?:10|20)|ADS(?:30|100)|PROMO(?:20|40)|CREDITS[12]|BUNDLE|SAVE_NEW|SAVE_ALL|'
                  r'FIX_OAUTH|LISTING|NOOP')


def _avg(v, k, a, b=None):
    x = v[a:b, O[k]]
    return float(np.nanmean(x)) if np.isfinite(x).any() else float('nan')


def active_text(c, g, H):
    d = lambda day: int(day - H)
    out = []
    if abs(c['pm'] - 1) > 1e-9:
        out.append(f"list price at {100 * c['pm']:.0f}% of the original")
    if c['ads_end'] > g:
        out.append(f"ads ${c['ads']:.0f}/day until day {d(c['ads_end'])}")
    if c['promo_end'] > g:
        out.append(f"credit-pack promo {100 * c['promo']:.0f}% off until day {d(c['promo_end'])}")
    if c['cred'] > 0:
        out.append(f"activation credits level {c['cred']:.0f}")
    if c['bundle'] > 0:
        out.append("credits bundled into subscriptions")
    if c['save'] > 0:
        out.append("save discount for " + ("newer subscribers" if c['save'] == 1 else "everyone"))
    if c['fix_on'] > 0:
        out.append("OAuth fix " + ("shipped" if c['fix_done'] > 0 else "in progress"))
    if c['listing'] > 0:
        out.append("new store listing live")
    return '; '.join(out) or 'none'


def context_pack(v, c, g, H, log, lessons):
    """v: (28, M) strict facts for days g-28..g-1 of one world; c: that world's controls (scalars)."""
    f = lambda k, n=7: _avg(v, k, -n)
    lines = [f"GOAL (owner): grow subscription MRR; judged on days {g - H + 60}-{g - H + 90}. Today is day {g - H}.",
             "STATE (receipted facts; last 7 days, 28-day average in brackets):",
             f"- subscription MRR ${f('sub_mrr'):,.0f} [{f('sub_mrr', 28):,.0f}], paying subscribers "
             f"{f('subs'):.0f}, list price ${f('list_price'):.0f}/mo",
             f"- visits/day {f('visits'):.0f} [{f('visits', 28):.0f}], signups/day {f('signups_org'):.1f} "
             f"[{f('signups_org', 28):.1f}], ad signups/day {f('signups_ad'):.2f}",
             f"- new subscriptions/day {f('new_subs'):.2f} [{f('new_subs', 28):.2f}], cancellations/day "
             f"{f('cancels', 14):.2f} [{f('cancels', 28):.2f}], credit-pack sales/day {f('packs'):.2f}",
             f"- OAuth errors per 1k calls {f('error_rate'):.1f}, spend last 7d ${7 * f('spend'):,.0f}, "
             f"revenue last 7d ${7 * f('revenue'):,.0f}",
             f"ACTIVE MOVES: {active_text(c, g, H)}",
             "RECENT DECISIONS: " + ('; '.join(log[-3:]) or 'none')]
    if lessons:
        lines.append("LESSONS (accepted, ranked by track record): " + ' | '.join(lessons))
    lines += ["GUARDRAILS: one new move per week; credentials stay with the platform.", MENU,
              'Reply with exactly 3 lines, best first: "<CODE> - <reason under 12 words>"']
    return '\n'.join(lines)


def parse(text, K, rng):
    """First K distinct move codes in order of appearance. Returns (codes, failed)."""
    t = text.upper().replace(' ', '')
    seen = []
    for m in _PAT.findall(t):
        c = IDX[m]
        if c not in seen:
            seen.append(c)
    failed = len(seen) < K
    while len(seen) < K:
        c = int(rng.integers(1, len(OPTIONS)))
        if c not in seen:
            seen.append(c)
    return seen[:K], failed


class LLM:
    def __init__(self, path, device='cuda:0'):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(path); self.tok.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(device).eval()

    def generate(self, prompts, seed=0, bs=64, max_new_tokens=110, temperature=0.7):
        out = []
        for i in range(0, len(prompts), bs):
            msgs = [self.tok.apply_chat_template([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': p}],
                                                 tokenize=False, add_generation_prompt=True) for p in prompts[i:i + bs]]
            enc = self.tok(msgs, return_tensors='pt', padding=True).to(self.device)
            self.torch.manual_seed(seed * 1000 + i)
            with self.torch.no_grad():
                g = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                        top_p=0.95, pad_token_id=self.tok.eos_token_id)
            out += self.tok.batch_decode(g[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        return out


class FakeLLM:
    """For smoke tests without a GPU: random valid codes, with some unparseable replies."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def generate(self, prompts, seed=0, **kw):
        out = []
        for _ in prompts:
            if self.rng.random() < .1:
                out.append('I think we should grow.')
            else:
                out.append('\n'.join(f'{OPTIONS[c]} - because' for c in self.rng.choice(np.arange(1, 16), 3, False)))
        return out
