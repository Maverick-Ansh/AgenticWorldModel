# REPORT: Steve, resized

Source: Skyfall AI, [Introducing Steve](https://skyfall.ai/blog/introducing-steve) (Nov 9 2026), a blog post with
no paper and no code. Extended with H-JEPA ([arXiv 2610.06805](https://arxiv.org/abs/2610.06805)) and
SuperMimic-style imagination training ([arXiv 2610.28480](https://arxiv.org/abs/2610.28480)). The experiment
plans were written by Fable 5.2 (a planning agent) and executed here. Hardware: Kaggle 2x Tesla T4, fp16.

All scores are **oracle-normalized**. On each of the 32 businesses we take mean true subscription MRR over a
90-day no-decision runoff after 26 weekly decisions. The score is then
`sum(arm - noop) / sum(oracle - noop)`: 0 means doing nothing, 1 means a planner that runs the TRUE
simulator. Brackets are paired bootstrap 95% CIs over the 32 businesses. The A/A pair (two random-proposer
seeds) differs by **-0.04 [-0.08, -0.00]**, so differences under about 0.08 are noise.

## 1. What was built

- **A hidden-truth business** (`awm/sim.py`): 16 AlphaSocial-like and 16 GammaPromo-like SaaS businesses.
  - Each has latent demand, intent and retention risk, plus per-world hidden customer behaviour.
  - The traps come from the post: a credit-pack promo that lifts the week and cuts the quarter, a save
    discount that gets gamed, an OAuth bug, ads that buy clicks but no customers, and a price move whose
    churn effect lags.
- **Twelve months of founder history**, backfilled into the fact store.
  - Founders tried price, ads, promo and credits.
  - They never tried bundling, save discounts, the OAuth fix or the listing rewrite.
- **The Steve harness:**
  - a bitemporal fact store with receipts that are re-hashed on later reads;
  - KPI acceptance only when a window is complete;
  - scoped short-lived tokens, so the agent never sees a secret;
  - an evidence log with an independent grader;
  - owner vetoes stored as dated facts, where a repeated veto becomes standing policy;
  - lessons with an independent judge and probation;
  - sealed-day replays.
- **The world model of the business** (`awm/worldmodel.py`):
  - The KPI tree's accounting is fixed: subscriptions, ARPU, repricing and the discount bucket.
  - Five Bayesian Poisson rate heads are learned on lever traces, with a Laplace posterior.
  - A latent demand filter tracks demand.
  - The model is deliberately misspecified and learns only from receipted facts.
- **The gate**: a move becomes a task only if its 20th-percentile forecast gain over 90 days beats 0.5% of MRR.
  There is also a small probe budget for moves the model knows it does not know.
- **The CEO agent**: Qwen2.5-3B-Instruct reads a context pack and proposes 3 moves.
- **15 arms run in lockstep** on the same noise tape. Every proposed and executed move is scored by an
  exact counterfactual: fork the true simulator and compare against NOOP on 16 shared tapes.

## 2. Claims and verdicts (main run, 15 arms)

| Arm | Score | 95% CI |
|---|---|---|
| oracle (true simulator, greedy) | 1.000 | |
| **wm_full**: world-model gate over the full menu | **0.414** | [0.26, 0.55] |
| sc_kitchen: static "do everything" schedule | 0.312 | [0.24, 0.39] |
| **llm_gate_k3**: LLM proposes 3, gate picks | **0.300** | [0.14, 0.44] |
| rand_gate / rand_gate_b: random 3 proposals, gate picks | 0.168 / 0.208 | |
| rand_myopic: random move, kept if the next 6 days read better | 0.173 | [0.12, 0.23] |
| **llm_gate_top1**: LLM's first move, gated | **0.150** | [0.01, 0.29] |
| sc_ads100: always run $100/day ads | 0.090 | |
| **llm_myopic_rev**: LLM's first move, judged by 7-day revenue | 0.042 | [-0.02, 0.11] |
| **llm_myopic_goal**: LLM's first move, judged by 7-day sub-MRR | 0.020 | [-0.04, 0.08] |
| noop / founder habit / +10% price once / always promo | 0 / -0.012 / -0.011 / -0.074 | |

The oracle beats noop by 230% of MRR, so there is a lot of headroom.

| # | Claim (post) | Result | Verdict |
|---|---|---|---|
| C1 | "moves like this get priced before they run [...] the mistake is spent in the model" | Same LLM, same first proposal, only the acceptance rule differs: gate - myopic = **+0.131 [0.03, 0.24]** (pre-registered bar 0.10). The gated arm launched 36% fewer harmful moves (244 vs 384). But it rarely rolls back (11 vs 268 rollbacks), so its harmful-move-days were higher (24.6k vs 20.0k). | **Confirmed on outcome. Refuted on exposure.** |
| C2 | "the week read up, the quarter read down" | 83% of the LLM's pack-promo proposals were traps (true 7-day revenue up, true 90-day goal down). The gate admitted 2% of them. Myopic arms kept 54-61% of all harmful moves they launched. Across all trap instances the gate admits 27%, against 43% of good moves. | **Confirmed for the post's anecdote.** Discrimination is weak beyond it. |
| C3 | the world model forecasts lagged effects and "results update the estimates" | Spearman(forecast, true gain): 0.50 on moves the business had tried, about 0 on untried moves (where the 80% intervals are honestly wide, covering 97%). It rises from 0.23 in weeks 0-8 to 0.48 in weeks 17-25. 80% coverage on tried moves is 0.54: overconfident. | **Partly confirmed**: learns with evidence; intervals too narrow. |
| C4 | "doing nothing is a decision" | Share of truly harmful moves among those the gate rejected: 0.57, against a base rate of 0.52 among proposals (+5 pp; the bar was +20). | **Refuted at this model quality.** |
| C5 | the LLM CEO adds value | LLM-3 + gate - random-3 + gate = +0.133 [-0.02, 0.28]. Full menu + gate - LLM-3 + gate = **+0.114 [0.06, 0.18]**. Qwen fixates on PRICE+10 (26-72% of first proposals) and promos. | **Not shown.** A 3B LLM's suggestions narrow the search to worse moves than the model's own sweep. |
| C6 | sealed replays; "the decision where Steve diverged and hindsight sided with Steve" | 127 sealed history days, all rebuilt byte-identically. Steve (model gate, no hindsight) - founders' actual decision = **+171 $MRR [64, 279]**. LLM + gate - founders = +101 [17, 188]. Steve diverged on 116 days; hindsight sided with Steve on 74 of them. Future leakage (guard off) adds +51 [-13, 120]. Late-arriving facts (guard "event" mode) add about 0. | **Confirmed.** The leak effect points the expected way but is not significant. |
| C7 | lessons, judge, probation; "better by being wrong on the record" | 295 lessons proposed, 237 accepted. Later predictions held 91% for accepted lessons vs 74% for rejected ones (+17 pp: the judge discriminates). In paired forks, lessons in the context pack cut harmful LLM proposals from 55% to 47%. The gate's pick did not improve (-79 [-225, +45]). | **Judge confirmed; benefit to decisions not shown.** |
| H | Credentials, evidence, grading, isolation, owner veto | 10 property tests (`tests/test_core.py`): faults the grader catches = all injected; self-report catches none; no secret in any prompt; tokens scoped and expiring; receipts detect tampering; the world model and LLM modules cannot import the simulator. | **Hold by construction** (tested, not a finding). |

## 3. H-JEPA world model (arXiv 2610.06805), trained on GPU

- **Setup:** three action-conditioned JEPA levels (1, 2, 4 weeks; stride 2, window 1; SIGReg; end-to-end).
  - Trained offline on 4,096 OTHER simulated businesses under a random policy.
  - Planned top-down by gradient descent (paper Eq. 4-6).
  - It must infer each held-out business's hidden behaviour from 17 weeks of receipted facts in context.
- **Goal cost:** a frozen-feature KPI probe (MRR at the horizon minus spend), replacing the paper's goal image.

| | Flat JEPA (1 level) | H-JEPA (3 levels) | H - flat |
|---|---|---|---|
| seed 0 | 0.167 | 0.207 | +0.040 [-0.013, 0.093] |
| seed 1 | 0.197 | 0.253 | **+0.055 [0.017, 0.093]** |
| planner time / decision | 8.1 s | **2.7 s** | 3x less compute |

**Selective abstraction reproduces** (paper Fig. 4). Ridge-probe NMSE from each level's latent (1.0 means no
better than the mean):

| Signal | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| visits (fast, noisy) | 0.16-0.47 | 0.79-0.93 | 0.98 |
| signups (fast, noisy) | 0.18 | 0.82-0.87 | 1.02 |
| subscribers (slow) | 0.05 | 0.05 | 0.05 |
| sub-MRR (slow) | 0.08 | 0.16 | 0.26 |

Higher levels drop the day-to-day noise and keep the business state, as the paper reports for ant legs vs
ant position.

**Verdict:**
- Hierarchy beats flat at about a third of the planner compute, on 2 seeds (one CI excludes 0).
- Both JEPA planners (0.17-0.25) trail the Bayesian KPI-tree gate (0.41), which encodes the billing
  accounting by hand.
- They beat random-proposal gates and every myopic arm.

## 4. Imagination-trained policies x LLM (SuperMimic-style, arXiv 2610.28480)

Plan by Fable 5.2 (`awm/imagine.py`).
- **Policy:** a 24-way policy trained only inside the frozen level-1 H-JEPA predictor.
- **Training:** 32 imagined weeks, first-order BPTT, Gumbel straight-through one-hots, Adam(0.9, 0.95)
  with lr 2e-4, gradient clip 1.0, 600 steps x 512 starts drawn from the offline businesses.
- **Loss:** the GD planner's own goal cost.
- **Arms:**
  - pi(z) reads only the world-model latent.
  - pi(z, h) also reads h, Qwen2.5-3B's mean-pooled hidden state of the same text context pack the CEO
    agent reads (PCA-32).
  - pi(z, h-shuffled) rolls h across businesses at evaluation.

**Guard first.** Rolled 32 weeks on logged actions, the frozen model's MRR-probe NMSE is 0.08 / 0.19 /
0.41 / **0.71** at weeks 1 / 8 / 16 / 32. The pre-registered rule says to trust it only up to about
16 weeks. We trained at 32 as planned, and the guard predicted what happened next.

| Arm | Score | Planning per week | Most common moves |
|---|---|---|---|
| GD planner (H-JEPA, top-down) | 0.207 | 2.65 s | NOOP, PRICE+10, ADS30, FIX_OAUTH |
| pi(z), imagination only | **-0.323** | **0.002 s** | PRICE-20 in 83% of moves |
| pi(z, h_LLM) | 0.138 | 3.16 s (Qwen encode) | ADS100, SAVE_NEW, CREDITS2 |
| pi(z, h_LLM shuffled across businesses) | 0.169 | 3.19 s | the same |

- **pi(z) - GD planner = -0.53 [-0.68, -0.40].** The imagination policy found a hole in the learned
  model: the model believes repeated 20% price cuts grow MRR, and the policy drove most businesses below
  doing nothing. This is SuperMimic's known risk (a frozen model, a long horizon, no real-data
  correction) at business scale.
- **pi(z, h) - pi(z) = +0.46 [0.33, 0.61]**: adding the LLM's reading of the context pack rescues the
  policy.
- **pi(z, h) - pi(z, h-shuffled) = -0.03 [-0.05, -0.01]**: the rescue is NOT information. Feeding another
  business's LLM vector works as well or better. The extra input changes which solution BPTT converges to
  and steers it away from the exploit, but it carries nothing about this business that the latent lacks.

**Verdict:**
- The pre-registered LLM claim fails: (a) - (a-shuf) does not favour (a).
- The amortization claim fails: pi(z) lands below the GD planner.
- The interesting emergent result is negative but concrete. Imagination-trained policies collapse onto
  world-model exploits at this model quality. Conditioning on frozen LLM features prevents the collapse
  through a regularization effect, which a shuffle control is needed to see.
- **What would make it work:**
  - Cut the horizon to the guard's 16 weeks.
  - Mix real rollouts into policy training (the paper does this every iteration).
  - Penalize disagreement across an ensemble of world models.

## 5. What broke (the most useful section)

1. **The LLM answered one line, not three.**
   - Qwen2.5-3B, told "reply with exactly 3 lines", wrote one line in 99.3-99.8% of replies. The parser
     filled the rest at random, so the first full run's LLM arms were mostly random.
   - The parse-fail guard caught it. A numbered-list prompt with a prefilled "1." brought parse failures to
     0.6-3.4%.
   - Only the LLM arms changed between runs. llm_gate_top1 went from 0.226 to 0.150 and llm_myopic_goal
     from 0.135 to 0.020: real Qwen proposals are worse than random ones here.
2. **Flat priors exploded untried levers.**
   - An N(0,1) prior on each of three collinear traces gave moves the founders never tried exp(+-1.7)
     lognormal tails. Ranking by mean picked them: BUNDLE was forecast at +$9k against a true +$709.
   - Fixed with scale-aware priors (about +-0.5 log-effect at typical intensity) and ranking by median.
   - Even after the fix, forecasts from founder history alone ranked moves at Spearman 0.01. Live refits
     raise it to 0.48 (C3).
3. **SIGReg had no gradient at init.**
   - The H-JEPA encoders started at a latent std of 0.03, where cos(tP) is flat.
   - A final non-affine LayerNorm fixed it (std 0.97, no collapse).
4. **The environment had dominant static moves** (SAVE_NEW and BUNDLE never harmful; PROMO and ADS100
   almost always harmful).
   - One pre-registered calibration pass changed: ad traffic quality, credits cost, bundle range, SAVE_NEW
     gaming, listing range and promo buzz. The environment was then frozen.
   - SAVE_NEW stayed always good, matching the post's "restricted to newer users" outcome.
   - A static kitchen-sink schedule still scores 0.31: a reminder that a sim can reward doing everything.
5. **Infrastructure.**
   - The Kaggle container was wiped once after about 30 minutes without a cell (a CPU-only smoke run). Its
     state was rebuilt from git.
   - Batched HF generation on a T4 tops out at 64 prompts (batch 128 runs out of memory); 38 s per batch.

## 6. Deviations

| Post / paper | Here | Why it still tests the claim |
|---|---|---|
| 3 real companies, live dashboards | 32 simulated businesses, hidden ground truth | exact counterfactuals for every move; per-world trap signs |
| CEO agent = frontier model | Qwen2.5-3B-Instruct, fp16 | the gate claim is about the acceptance rule, which is held fixed |
| 9 data sources, 5-level goal stack | 15 metrics, one 90-day goal | the claims tested sit at the quarter horizon |
| H-JEPA: images, goal image, ViT | weekly fact vectors, KPI-probe goal, MLP encoders | Eq. (1)-(6) are implemented as written; only the goal cost differs |
| SuperMimic: critic, k=8 autoregressive WM fit | no critic, teacher-forced WM | stated; the open-loop drift guard reports how far the model can be trusted |

## 7. Not tested

- Real businesses, real customers, or anything in the post's absolute numbers.
- The 9-source ingestion, hourly collection and the 12-month backfill on connect.
- Multi-agent task decomposition (Growth / Product / Engineering / Support): one move per week here.
- More than one environment seed.
- LLMs larger than 3B.
- Owner interventions beyond a fixed rule.
- The H-JEPA paper's visual environments.
- Whether Steve's own world model is anything like this one: the post does not say.
