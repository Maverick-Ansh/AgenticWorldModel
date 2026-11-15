# AgenticWorldModel

A resized, from-scratch replication of **Steve**, Skyfall AI's harness for running a business with agents
([Introducing Steve](https://skyfall.ai/blog/introducing-steve), Nov 2026). It adds a second world model
built with **H-JEPA** ([arXiv 2610.06805](https://arxiv.org/abs/2610.06805)), a hierarchy of
action-conditioned JEPAs planned top-down.

Skyfall ran three real software companies. We cannot, so the business is a simulator with hidden ground
truth, which gives us something Skyfall does not have: the **exact counterfactual** of every decision.
Results, claims, deviations and what broke are in **[REPORT.md](REPORT.md)**.

## What is in here

| Post says | Module |
|---|---|
| "The CEO agent plans using a world model that represents the business as code" | `awm/worldmodel.py`: KPI tree with fixed accounting + learned Bayesian Poisson rate heads |
| "only a move that clears the bar becomes tasks, so the mistake [...] is spent in the model" | `awm/runner.py` `gate_decide`: admit a move only if its 20th-percentile forecast gain beats the bar |
| "Credentials / Evidence / Grading / Isolation" | `awm/platform.py`: short-lived scoped tokens, evidence log, independent grader, owner veto as dated fact |
| "typed facts [...] receipt re-hashed on later reads", "both stamps strictly before" the cutoff | `awm/facts.py`: bitemporal store, receipts, strict / event / off guard |
| "A context pack containing the goal, the relevant state, lessons and guardrails" | `awm/llm.py`: Qwen2.5-3B-Instruct CEO agent, fp16, one replica per T4 |
| "There is no staging environment for last March" | `awm/replay.py`: sealed-day replays, Steve vs founders, leakage ablation |
| "Steve gets better by being wrong on the record" | `runner.check_lessons`: lessons with predictions, independent judge, probation ranking |
| The hidden business (AlphaSocial-like and GammaPromo-like) | `awm/sim.py`: 32 worlds; pack-promo cannibalization, discount gaming, OAuth bug, dead ads, lagged price churn |
| H-JEPA, Eq. (1)-(6) | `awm/hjepa.py`: 3 levels (1, 2, 4 weeks), SIGReg, end-to-end training, top-down gradient-descent planning |

## Run

On a GPU box (we used Kaggle 2x T4):

```bash
pip install -r requirements.txt
python -m pytest -q tests                      # simulator invariants + harness property tests
python run.py --stage all --out results        # E0 calibration, 15 arms x 26 weeks, replays, analysis (~25 min)
python hjepa_run.py --out results_hj           # H-JEPA vs flat JEPA (~20 min)
```

`python run.py --stage all --fake --weeks 2 --out results_smoke` runs the whole pipeline in 30 s with a
random stand-in for the LLM.
