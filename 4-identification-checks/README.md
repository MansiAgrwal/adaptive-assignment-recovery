# Step 4: Identification checks

**Question.** How much of the one-type model's error in step 3 follows from which users are still at risk of learning, and from which activities the assignment rule gives them? And which features of the model prevent existing identification theorems from applying?

Everything here is simulated, as in steps 1–3.

## Data

`data/` holds records from a validation grid that extends step 3:

- **Recovery grid.** 3,000 datasets of 120 simulated users: three heterogeneity conditions × random or adaptive assignment × 500 replications. Each dataset was fitted with a one-type and a two-type hidden Markov model, giving 6,000 fits (`recovery_fits.jsonl.gz`, gzipped, one JSON record per line; `recovery_summary.csv`, summaries per condition).
- **History retention.** The assignment rule's memory changed from the last three responses to the complete history (capacity 3, exploration .15): exposure contrast in 3,000 discovery users and 6,000 independent validation users (`history_discovery_validation.json`).
- **Auxiliary measurement.** Interval-width ratios and coverage for paired designs with added state readings (`measurement_extended_ratio_summary.csv`, `measurement_budget_comparison.csv`).

The three heterogeneity conditions share marginal gains (.092, .070, .078, .078) for a|A, a|AB, b|B and b|AB, so the target for attribute *a* is −0.022 in all of them:

- **Activity-by-type:** the two types of step 3 (fast .20, .10, .15, .15; slow .02, .05, .03, .03; weights .4/.6).
- **Common shift:** logit shifts of ±0.8 for all four gains, weights .4/.6.
- **Continuous:** a normal logit effect with standard deviation 1.2.

These records were produced by the validation simulation code, which is not in this repository. That code screened 20 random starts per fit, refined the best three with L-BFGS-B, and bounded probability logits at ±11. The scripts below recompute summaries from the records or simulate the generator directly; none of them refits a model.

## Files

| File | Computes |
|---|---|
| `records_summary.py` | Mixture fits without an interval by condition, the reasons (a gain at the numerical bound, by type; convergence), mean one-type estimates and the adaptive-minus-random shift, history and measurement summaries |
| `oracle.py` | For the activity-by-type condition with observed states: expected at-risk exposures by type and activity, the exposure-weighted limit of the one-type estimate, and its split into a composition term (who is still at risk) and a sorting term (which activities they receive) |
| `families.py` | The same limit for all three conditions and both assignment rules, compared with the mean fitted one-type estimates in `data/recovery_summary.csv` |
| `rank.py` | Ranks of block-emission matrices for the hidden state (user type, mastery state), two attributes, four activity patterns |
| `larger.py` | The same ranks for three structures over three attributes; type composition of at-risk exposures by state depth in a prerequisite chain, and its dependence on session length |
| `outputs/` | Output of each script from the run listed in `outputs/environment.txt` |

## Reproduce

```bash
pip install -r ../requirements.txt scipy
export OPENBLAS_NUM_THREADS=1
python records_summary.py
python rank.py
python larger.py
python oracle.py      # about 10 minutes on one core
python families.py    # about 20 minutes on one core
```

`oracle.py` and `families.py` use fixed seeds; their Monte Carlo error is in the fourth decimal place.

## What this does not show

- The exposure-weighted limit assumes observed mastery states. The fitted models infer them, so the limit is a benchmark for the one-type model, not its exact large-sample value. The gap was at most .002 in this grid and about .004 for adaptive assignment at N = 1,000 in step 3.
- Estimates at the ±11 logit bound are hits of a numerical constraint. They do not show that a true acquisition probability is 0 or 1.
- Rank deficiency rules out applying rank-based identification theorems directly. It does not show that the target contrast is unidentified.
- The three-attribute ranks use one set of randomly drawn parameters per structure. The depth results cover one chain structure under random assignment.
