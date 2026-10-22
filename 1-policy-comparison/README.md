# Step 1: Policy comparison

**Question.** A personalized system decides from a memory of recent responses. Does it matter how that memory is built?

Two rules were compared, each keeping 18 answers:

- **Recency:** the last 18 answers, whatever the activity.
- **Stratified:** up to 6 recent answers per activity type, topped up with the most recent remaining answers.

The selector picks each activity with probability proportional to its Laplace-smoothed error rate in memory, mixed with 10% uniform exploration.

## Results

**`buffer_sim.py`** simulates full sessions: 18 warm-up trials (balanced or skewed across activities), then 72 practice trials. A quarter of the practice trials are scheduled and the rest are chosen by the rule. After 24 hours of possible forgetting comes a 24-item delayed test. The script covers four parameter scenarios.

- Delayed-accuracy differences between the rules ranged from d = −0.027 to +0.026 (Monte Carlo SE 0.018). None is distinguishable from zero.
- Exposure to each activity differed by less than one trial on average.

**`policy_gap.py`** feeds identical histories to both rules and compares their choice probabilities.

- The rules gave identical probabilities on only 7–8% of decisions.
- The differences were small: median largest gap 0.049, 90th percentile 0.12.

## Interpretation

The two rules usually make slightly different decisions, but the differences wash out in learning outcomes. Planning a study around a moderate outcome difference between them (d = 0.30) is not supported by these simulations. This moved the project from comparing outcomes to asking whether activity effects can be estimated at all.

## Reproduce

```bash
python buffer_sim.py 6000 > buffer_sim_output.txt
python policy_gap.py > policy_gap_output.txt
```

All learning parameters are assumptions chosen for illustration.
