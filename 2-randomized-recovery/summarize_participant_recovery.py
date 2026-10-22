"""Export complete exploratory repeated-context diagnostics; no adaptive claim."""
from pathlib import Path
import csv, json
import numpy as np

B = Path(__file__).parent
d = json.loads((B/'participant_recovery_results.json').read_text())
rows = []
for c in d['results']:
    raw = [r for r in d['raw'] if r['scenario']==c['scenario'] and r['people']==c['people'] and r['episodes_per_person']==c['episodes_per_person']]
    for k in range(2):
        err = np.array([r['contrast_error'][k] for r in raw])
        truth = np.array([r['true_contrast'][k] for r in raw])
        assert np.ptp(truth)<1e-12
        bias = err.mean(); sd=err.std(ddof=1); rmse=np.sqrt(np.mean(err*err))
        assert abs(rmse**2-bias**2-(len(err)-1)/len(err)*sd**2)<1e-12
        rows.append(dict(scenario=c['scenario'], independent_people=c['people'],
                         fresh_contexts_per_person=c['episodes_per_person'],
                         replications=len(raw), contrast='joint_minus_single_'+('a' if k==0 else 'b'),
                         true_contrast=float(truth[0]), bias=float(bias), sampling_sd=float(sd),
                         rmse=float(rmse), bias_mcse=float(sd/np.sqrt(len(raw))),
                         interval_coverage='not_evaluated', adaptive_assignment=False,
                         continuing_competence_trajectory=False,
                         actual_counterbalanced_assignment=False))
with (B/'Repeated_Context_Diagnostics.csv').open('w', newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print(json.dumps({'datasets':len(d['raw']), 'cells':len(d['results']), 'exported_rows':len(rows),
                  'bootstrap_replicates':d['bootstrap_replicates'],
                  'scope':'Persistent person type but fresh contexts, independently randomized activities; no counterbalancing implemented, no adaptive assignment and no interval coverage.'},indent=2))
