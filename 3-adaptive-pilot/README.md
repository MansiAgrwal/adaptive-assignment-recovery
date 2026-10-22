# Step 3: Adaptive pilot

**Question.** With continuous practice and adaptive assignment, which matters more for estimating the joint-minus-single gain: the assignment rule, or whether the analysis model allows for differences between learners?

The design is 2 × 2: assignment (uniform random vs adaptive) × analysis model (one learner type vs two learner types). Each cell has 500 replications at N = 1,000 and N = 300. The main results table is in the [top-level README](../README.md#main-result-step-3).

## Design

- **Learners.** Two hidden rules (*a*, *b*) and activities A, B and AB. Each learner has one unbroken session: a 9-item test every 12 trials (7 tests), 72 practice trials with feedback, 24 hours with possible forgetting, then a delayed test. There is no forgetting within the session, and only practised rules can be learned.
- **Learner types.** Fast (40%), with per-trial gains aA .20, aAB .10, bB .15, bAB .15. Slow (60%), with gains .02, .05, .03, .03. Type persists for the whole session.
- **Measurement.** Starting states .60 / .15 / .15 / .10. Guess rates .25 / .20 / .15 and slip rates .10 / .12 / .15 for A / B / AB items. Forgetting over 24 hours: .15 per rule.
- **Target.** The type-weighted joint-minus-single gain. True values: −0.022 for rule *a* and 0 for rule *b*.
- **Assignment.** Uniform random, or adaptive: choose the activity with the lowest Laplace-smoothed accuracy over the last 18 practice answers, with ε = .15 (every activity has probability ≥ .05 on every trial).
- **Models.** Homogeneous HMM (15 parameters) and two-type mixture HMM (20 parameters). Both are fitted by EM with closed-form updates and 2 or 3 random starts. Apart from learner types, both share the data-generating model's structure, so only heterogeneity is misspecified.
- **Intervals.** Wald intervals from the outer product of scores, using the delta method. Numerical-Hessian standard errors on 69 regenerated adaptive datasets agreed (ratios 1.01 and 0.99; `se_check.py`).

## Files

| File | Contents |
|---|---|
| `ademp_sim.py` | Data generation, models, EM, intervals |
| `run_reps.py` | Runs replications |
| `summarize.py` | Builds `ADEMP_results.csv` (all measures with Monte Carlo SEs) |
| `check1.py` | Likelihood checked against an independent brute-force version |
| `check2.py` | One N = 10,000 dataset per assignment rule (sanity check only) |
| `se_check.py` | Compares interval methods |
| `res_*.jsonl`, `se_*.jsonl` | Raw results for every fit |

## Reproduce

The stored results were split across two workers (`res_*_a.jsonl`, `res_*_b.jsonl`). `summarize.py` reads every `res_*.jsonl` file in this folder, so move the stored files elsewhere before regenerating:

```bash
python run_reps.py 1000 0 500 res_1000.jsonl
python run_reps.py 300 0 500 res_300.jsonl
python summarize.py > summary_table.txt
```

Seeds are fixed per N, replication and assignment rule.
