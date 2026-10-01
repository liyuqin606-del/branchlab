"""Bounded requested 5-alpha x 4-feature DEVELOPMENT diagnostic sweep.

All settings and all outcomes are retained. No confirmation data are accessed.
This probes model capacity, not a new performance claim or confirmatory choice.
"""
import copy
import json
from pathlib import Path

import numpy as np

from branchlab.value_aware import ValueAwareDiagnosticLearner
from branchlab.synthesis import _Ridge

ROOT = Path(__file__).resolve().parents[2]
base = json.loads((ROOT / "artifacts/development_v2/episodes.json").read_text())
assert {e["split"] for e in base} == {"discovery", "development"}
subsets = {"full92": lambda x: x,
           "first14aggregate": lambda x: x[:14],
           "first14_plus_last9candidate": lambda x: x[:14] + x[-9:],
           "last9candidate": lambda x: x[-9:]}
records = []
for subset_name, transform in subsets.items():
    rows = copy.deepcopy(base)
    for e in rows:
        e["log_features"] = transform(e["log_features"])
    discovery = sorted([e for e in rows if e["split"] == "discovery"], key=lambda e: e["id"])
    validation = sorted([e for e in rows if e["split"] == "development"], key=lambda e: e["id"])
    x = np.array([e["log_features"] for e in discovery])
    y0 = np.array([e["budget_losses"]["0"] for e in discovery])
    y2 = np.array([e["budget_losses"]["2"] for e in discovery])
    for alpha in [0.01, 0.1, 1, 10, 100]:
        free = ValueAwareDiagnosticLearner(ridge_alpha=alpha).fit_fixed(rows)
        short = ValueAwareDiagnosticLearner(ridge_alpha=alpha).fit_fixed(rows, budget_cost=2)
        fr = free.evaluate(rows, split="development")
        sr = short.evaluate(rows, split="development")
        free_loss, short_loss = np.array(fr["dev_losses"]), np.array(sr["dev_losses"])
        pred0, pred2 = np.empty_like(y0), np.empty_like(y2)
        for seed in sorted({e["seed"] for e in discovery}):
            mask = np.array([e["seed"] != seed for e in discovery])
            pred0[~mask] = _Ridge.fit(x[mask], y0[mask], alpha).predict(x[~mask])
            pred2[~mask] = _Ridge.fit(x[mask], y2[mask], alpha).predict(x[~mask])
        ix = np.arange(len(discovery))
        oof0 = y0[ix, np.argmin(pred0, axis=1)]
        oof2 = y2[ix, np.argmin(pred2, axis=1)]
        for probe in sorted(rows[0]["probe_features"]):
            gate = ValueAwareDiagnosticLearner(ridge_alpha=alpha, search_budget_cells=1000).fit_fixed(rows, probe)
            head = ValueAwareDiagnosticLearner(ridge_alpha=alpha, search_budget_cells=1000).fit_fixed(rows, probe, always_probe=True)
            gr = gate.evaluate(rows, split="development"); hr = head.evaluate(rows, split="development")
            paid_loss = np.array(hr["dev_losses"])
            gains = np.array(gate.history[0]["crossfit_net_gains"])
            oofpaid = oof0 - gains
            records.append({"subset": subset_name, "features": x.shape[1], "alpha": alpha, "probe": probe,
                            "free_dev_loss": fr["mean_dev_loss"], "short_free_dev_loss": sr["mean_dev_loss"],
                            "paid_head_dev_loss": hr["mean_dev_loss"], "conditional_gate_dev_loss": gr["mean_dev_loss"],
                            "gate_acquisitions": gr["evaluation_revealed_cells"],
                            "oof_discovery_net_gain": float(gains.mean()),
                            "oof_discovery_same_horizon_gain": float((oof2-oofpaid).mean()),
                            "dev_paid_net_gain": float((free_loss-paid_loss).mean()),
                            "dev_paid_same_horizon_gain": float((short_loss-paid_loss).mean()),
                            "dev_conditional_net_gain": fr["mean_dev_loss"]-gr["mean_dev_loss"],
                            "oracle_gate_gain_with_fitted_head": float((free_loss-np.minimum(free_loss,paid_loss)).mean()),
                            "per_dev_seed": [{"seed": seed,
                                              "paid_minus_free_loss": float(np.mean([paid_loss[i]-free_loss[i] for i,e in enumerate(validation) if e['seed']==seed])),
                                              "conditional_minus_free_loss": float(np.mean([gr['dev_losses'][i]-free_loss[i] for i,e in enumerate(validation) if e['seed']==seed]))}
                                             for seed in sorted({e['seed'] for e in validation})]})
settings=[]
for subset_name in subsets:
    for alpha in [0.01,0.1,1,10,100]:
        rs=[r for r in records if r['subset']==subset_name and r['alpha']==alpha]
        settings.append({'subset':subset_name,'alpha':alpha,'free_dev_loss':rs[0]['free_dev_loss'],
                         'short_dev_loss':rs[0]['short_free_dev_loss'],
                         'best_head_dev_loss':min(r['paid_head_dev_loss'] for r in rs),
                         'best_gate_dev_loss':min(r['conditional_gate_dev_loss'] for r in rs),
                         'positive_net_oof_and_dev':sum(r['oof_discovery_net_gain']>0 and r['dev_paid_net_gain']>0 for r in rs),
                         'positive_same_horizon_oof_and_dev':sum(r['oof_discovery_same_horizon_gain']>0 and r['dev_paid_same_horizon_gain']>0 for r in rs)})
out={'scope':'All 160 probe/settings evaluations are development diagnostics only; no confirmation data read; selecting best development row is not a confirmed method.',
     'settings':settings,'records':records}
(ROOT/'notes/goal_v2/postrun_capacity_sweep.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(settings,indent=2))
print('cross-split positive-net heads',sum(r['oof_discovery_net_gain']>0 and r['dev_paid_net_gain']>0 for r in records))
print('best gate',min(records,key=lambda r:r['conditional_gate_dev_loss']))
