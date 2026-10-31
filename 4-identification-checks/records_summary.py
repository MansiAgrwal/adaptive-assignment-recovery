"""Recomputes the counts and means reported from the step 4 records.

1. Two-type (mixture) fits without an observed-information interval, by assignment and heterogeneity condition.
2. Why: a type-specific gain at the fitting code's numerical bound (|logit| >= 10.9; the bounds are +/-11),
   another parameter at the bound, or no boundary flag. Types are ordered by mean acquisition probability,
   so 'slow' is the component with the lower mean gain.
3. Mean homogeneous (one-type) estimates of the skill-a contrast and the adaptive-minus-random shift.
4. History-retention exposure contrast and auxiliary-measurement summaries."""
import json, math, csv, collections, gzip, pathlib
DATA = pathlib.Path(__file__).parent / 'data'
lg = lambda p: math.log(max(p, 1e-300) / max(1 - p, 1e-300))
rows = [json.loads(l) for l in gzip.open(DATA / 'recovery_fits.jsonl.gz', 'rt')]
mix = [r for r in rows if r['model'] == 'mixture']; bad = [r for r in mix if not r['interval_valid']]
print(f'1. Mixture fits without an interval: {len(bad)} of {len(mix)}')
for pol in ['random', 'adaptive']:
    cells = {f: sum(1 for r in bad if r['policy'] == pol and r['family'] == f) for f in ['discrete', 'interaction', 'continuous']}
    n = sum(1 for r in mix if r['policy'] == pol)
    print(f'   {pol:8s}: {sum(cells.values())} of {n}; by condition (common shift, activity-by-type, continuous): '
          f'{cells["discrete"]}, {cells["interaction"]}, {cells["continuous"]} of 500 each')
c = collections.Counter()
for r in bad:
    G = r['parameters']['gains']; slow = min(range(len(G)), key=lambda k: sum(G[k]))
    gain_hit = any(abs(lg(x)) >= 10.9 for g in G for x in g)
    other = [x for k in ['g', 's', 'loss', 'pi', 'p0'] for x in r['parameters'][k]]
    other_hit = any(abs(lg(x)) >= 10.9 for x in other)
    if gain_hit:
        c['gain at bound'] += 1
        if any(lg(x) <= -10.9 for x in G[slow]): c['  slow type at lower bound'] += 1
        elif not any(abs(lg(x)) >= 10.9 for x in G[slow]): c['  fast type only'] += 1
        else: c['  slow type at upper bound only'] += 1
    elif other_hit: c['other parameter at bound'] += 1
    else: c['no parameter at bound (converged=%s)' % r['converged']] += 1
print('2. Reasons:'); [print(f'   {k}: {v}') for k, v in c.items()]
m = {}
for row in csv.DictReader(open(DATA / 'recovery_summary.csv')):
    if row['model'] == 'homogeneous' and row['skill'] == 'a' and row['interval'] == 'hessian':
        m[(row['family'], row['policy'])] = float(row['mean_estimate'])
print('3. Mean homogeneous skill-a estimates (target -.022), 500 datasets of 120 users each:')
for f in ['interaction', 'discrete', 'continuous']:
    print(f'   {f:11s} random {m[(f,"random")]:+.4f}  adaptive {m[(f,"adaptive")]:+.4f}  shift {m[(f,"adaptive")]-m[(f,"random")]:+.4f}')
h = json.load(open(DATA / 'history_discovery_validation.json'))
for k in ['discovery_selected', 'independent_validation']:
    x = h[k]; print(f'4. History ({x["stage"]}, n={x["n"]}, capacity {x["capacity"]}, exploration {x["epsilon"]}): '
                    f'{x["comparison"]} {x["metric"]} exposure {x["mean"]:+.3f} trials (MC SE {x["mcse"]:.3f})')
for row in csv.DictReader(open(DATA / 'measurement_extended_ratio_summary.csv')):
    if row['skill'] == 'a': print(f'   width ratio {row["placement"]} vs {row["reference"]}, skill a: {float(row["mean"]):.3f} (n={row["count"]})')
for row in csv.DictReader(open(DATA / 'measurement_budget_comparison.csv')):
    if row['skill'] == 'a': print(f'   {row["placement"]:18s} skill a: available {row["extended_available"]}/500, conditional coverage {float(row["extended_coverage"]):.3f}')
