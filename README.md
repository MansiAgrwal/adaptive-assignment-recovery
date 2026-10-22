# Adaptive assignment and recovery of activity effects

Personalized algorithmic environments, such as adaptive learning platforms, recommender systems and digital health apps, choose what each person encounters next from how they have responded so far. That serves the user, but it also shapes the data researchers later analyse. This project simulates such an environment with a controlled rule-learning task and asks one question:

**When an algorithm picks each person's next activity from their earlier responses, can we still estimate how much each activity changes an ability we cannot observe directly?**

> Everything here is simulated. The learning and measurement parameters are illustrative and have not been calibrated against human data. Nothing here sets a sample size for a real study or describes real learners.

## The setting

Simulated learners practise two rules, *a* and *b*. Each trial uses one of three activities: practise *a* alone (A), *b* alone (B), or both together (AB). Whether a learner has mastered a rule is hidden; we only see right or wrong answers, which include guesses and slips. A hidden Markov model describes how mastery changes from trial to trial.

The quantity of interest is the **joint-minus-single gain**: how much more (or less) often rule *a* is learned after an AB activity than after an A activity.

## Three steps

| Step | Question | Answer in these simulations |
|---|---|---|
| [1. Policy comparison](1-policy-comparison/) | Do two ways of remembering recent answers change which activities learners get, or how much they learn? | Rarely, and only slightly. Effects on delayed accuracy were indistinguishable from zero. |
| [2. Randomized recovery](2-randomized-recovery/) | With one randomly assigned activity per person and tests before and after, can the effect be recovered? | Yes, without detectable bias, but imprecisely at N = 300. An exact example shows why this one-shot result does not carry over to repeated practice. |
| [3. Adaptive pilot](3-adaptive-pilot/) | With repeated practice and adaptive assignment, does the analysis model or the assignment rule matter more? | The model. Ignoring fast/slow learner differences reversed the sign of the effect; allowing for them recovered it under both random and adaptive assignment. |

## Main result (step 3)

500 simulated datasets per row. Learners are 40% fast and 60% slow, and the type persists across the session. The adaptive rule picks the activity with the lowest recent accuracy, with 15% random exploration. True effect for rule *a*: **−0.022**.

| N | Assignment | Analysis model | Mean estimate | 95% interval coverage | Interval width |
|---:|---|---|---:|---:|---:|
| 1,000 | random | one learner type | +0.010 | 0.2% | 0.026 |
| 1,000 | adaptive | one learner type | +0.020 | 0.0% | 0.016 |
| 1,000 | random | two learner types | −0.023 | 95.6% | 0.054 |
| 1,000 | adaptive | two learner types | −0.022 | 94.2% | 0.045 |
| 300 | random | one learner type | +0.010 | 28.8% | 0.049 |
| 300 | adaptive | one learner type | +0.020 | 0.2% | 0.030 |
| 300 | random | two learner types | −0.023 | 94.2% | 0.100 |
| 300 | adaptive | two learner types | −0.024 | 94.2% | 0.089 |

- **The one-type model got the direction wrong** under both assignment rules.
- **Adaptive assignment made that wrong model worse.** Its bias grew and its standard errors became too small; on rule *b*, where the true effect is zero, it falsely rejected zero in 20% of datasets at N = 1,000.
- **The two-type model recovered the effect under adaptive assignment**, with narrower intervals than under random assignment. Adaptive assignment did not, by itself, prevent estimation here.
- **One unexplained detail:** for the two-type model with adaptive assignment at N = 1,000, coverage for rule *b* was 91.8% (Monte Carlo SE 1.2%).

The two learner types are deliberately well separated and the analysis model matches how the data were generated, apart from the learner-type assumption. This is a best case for the two-type model, not evidence about real learners.

## What this does not show

- That the findings hold with continuous individual differences rather than two types.
- That interval coverage is adequate in steps 1 and 2, where it was not evaluated.
- That any particular sample size is enough for a human study.
- That the effect is identified in general. Successful recovery in simulation is not a proof.
- That the numbers transfer to other personalized environments, such as recommender feeds or clinical apps. Those need their own measurement and change models.

## Next steps

1. Compare a rule that uses a learner's full history with one that uses only recent answers, holding everything else fixed.
2. Replace the two learner types with continuous individual differences, and test the model when it is misspecified.
3. Report bias, precision and interval coverage for every condition.
4. Calibrate parameters against pilot data before making any sample-size claim.

## Reproduce

Python 3.12 or later, with NumPy.

```bash
pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
```

Each step folder has its own README with exact commands. The full step 3 run (4,000 model fits) takes several hours on one core.

### Verification

Re-run on 21 October 2026 under Python 3.13.16 and NumPy 2.5.3. The original runs used Python 3.12.14 and NumPy 2.3.5.

| Check | Result |
|---|---|
| Step 1: `policy_gap.py` | Output identical to the stored file |
| Step 2: `heterogeneity_counterexample.py`; both summary scripts | Outputs byte-identical to the stored files |
| Step 2: `design_checks.py` | Identical except one value at floating-point precision (1.6e-15 vs 2.2e-15) |
| Step 3: replications 0–2 at N = 300, both rules and models | Estimates and standard errors identical to the stored fits |
| Step 3: `summarize.py` | Regenerates `ADEMP_results.csv` byte-identically |

The full replication grids were not re-run.

## How this was made

The code was written with help from Claude (Anthropic), an AI model, in October 2026. The research question, design decisions and interpretation are my responsibility. If you find an error, please open an issue.

## Reference

Morris TP, White IR, Crowther MJ. Using simulation studies to evaluate statistical methods. *Statistics in Medicine*. 2019;38(11):2074–2102. [doi:10.1002/sim.8086](https://doi.org/10.1002/sim.8086)

## License

Code: MIT (see `LICENSE`). Results and text: CC BY 4.0.
