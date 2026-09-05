"""Read v2 DEVELOPMENT episodes only; diagnose heads, gates, and headroom."""
from pathlib import Path
import json
from collections import Counter

import numpy as np

from branchlab.value_aware import ValueAwareDiagnosticLearner

ROOT = Path(__file__).resolve().parents[2]
rows = json.loads((ROOT / "artifacts/development_v2/episodes.json").read_text())
assert set(e["split"] for e in rows) == {"discovery", "development"}
ds = sorted([e for e in rows if e["split"] == "discovery"], key=lambda e: e["id"])
vs = sorted([e for e in rows if e["split"] == "development"], key=lambda e: e["id"])
probes = sorted(rows[0]["probe_features"])
y0 = np.array([e["budget_losses"]["0"] for e in vs]); y2 = np.array([e["budget_losses"]["2"] for e in vs])
free = ValueAwareDiagnosticLearner().fit_fixed(rows)
short = ValueAwareDiagnosticLearner().fit_fixed(rows, budget_cost=2)
fr = free.evaluate(rows, split="development"); sr = short.evaluate(rows, split="development")
free_loss = np.array(fr["dev_losses"]); short_loss = np.array(sr["dev_losses"])
entries = []


def correlation(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


for probe in probes:
    conditional = ValueAwareDiagnosticLearner(search_budget_cells=1000).fit_fixed(rows, probe)
    always = ValueAwareDiagnosticLearner(search_budget_cells=1000).fit_fixed(rows, probe, always_probe=True)
    cr = conditional.evaluate(rows, split="development"); ar = always.evaluate(rows, split="development")
    head_loss = np.array(ar["dev_losses"])
    gate_chosen = np.array([p is not None for p in cr["probe_ids"]])
    gains = free_loss - head_loss
    oof_gains = np.array(conditional.history[0]["crossfit_net_gains"])
    action_index = 1 if probe.startswith("lr_half:") else 2
    signal = np.array([e["probe_features"][probe][0] for e in vs])
    ds_signal = np.array([e["probe_features"][probe][0] for e in ds])
    ds_delta = np.array([e["budget_losses"]["2"][action_index] - e["budget_losses"]["2"][0] for e in ds])
    dev_delta = y2[:, action_index] - y2[:, 0]
    x = np.array([e["log_features"] for e in vs]); pred = conditional._free.predict(x)
    margin = np.sort(pred, axis=1)[:, 1] - np.sort(pred, axis=1)[:, 0]
    predicted_gain = conditional._gate.predict(np.column_stack((x, margin)))[:, 0]
    entries.append({"probe": probe, "oof_discovery_mean_net_gain": float(oof_gains.mean()),
                    "oof_discovery_positive_episodes": int(np.sum(oof_gains > 0)),
                    "always_head_development_loss": ar["mean_dev_loss"],
                    "incremental_head_loss_vs_same_horizon_logs": float((head_loss - short_loss).mean()),
                    "net_head_loss_vs_full_logs": float((head_loss - free_loss).mean()),
                    "gate_development_loss": cr["mean_dev_loss"],
                    "gate_acquisitions": int(gate_chosen.sum()),
                    "actual_positive_gain_episodes": int(np.sum(gains > 0)),
                    "gate_true_positive": int(np.sum(gate_chosen & (gains > 0))),
                    "gate_false_positive": int(np.sum(gate_chosen & (gains <= 0))),
                    "gate_false_negative": int(np.sum(~gate_chosen & (gains > 0))),
                    "oracle_gate_with_fitted_head_loss": float(np.minimum(free_loss, head_loss).mean()),
                    "predicted_gain_mean": float(predicted_gain.mean()),
                    "actual_gain_mean": float(gains.mean()),
                    "gain_prediction_corr": correlation(predicted_gain, gains),
                    "signal_future_action_contrast_corr_discovery": correlation(ds_signal, ds_delta),
                    "signal_future_action_contrast_corr_development": correlation(signal, dev_delta),
                    "per_episode": [{"id": e["id"], "free_action": fr["chosen_repairs"][i],
                                     "head_action": ar["chosen_repairs"][i], "actual_net_gain": float(gains[i]),
                                     "predicted_net_gain": float(predicted_gain[i]), "acquired": bool(gate_chosen[i])}
                                    for i, e in enumerate(vs)]})

baseline = {"logs_mean_loss": fr["mean_dev_loss"], "same_horizon_short_logs_mean_loss": sr["mean_dev_loss"],
            "logs_action_counts": dict(Counter(fr["chosen_repairs"])),
            "oracle_action_counts_at_64": dict(Counter(np.argmin(y0, axis=1).tolist())),
            "logs_action_mismatches_at_64": int(np.sum(np.array(fr["chosen_repairs"]) != np.argmin(y0, axis=1))),
            "oracle_loss_at_64": float(y0.min(axis=1).mean()),
            "oracle_loss_at_63_with_paid_probe": float(y2.min(axis=1).mean()),
            "logs_regret_at_64": float((free_loss - y0.min(axis=1)).mean()),
            "oracle_skip_or_paid_action_loss": float(np.minimum(free_loss, y2.min(axis=1)).mean()),
            "oracle_free_action_and_horizon_loss": float(np.minimum(y0.min(axis=1), y2.min(axis=1)).mean()),
            "log_feature_count": len(vs[0]["log_features"]), "discovery_episodes": len(ds),
            "development_episodes": len(vs), "discovery_seed_count": len({e["seed"] for e in ds}),
            "development_seed_count": len({e["seed"] for e in vs})}
by_condition = {}
for condition in sorted({e["condition"] for e in vs}):
    ix = [i for i, e in enumerate(vs) if e["condition"] == condition]
    by_condition[condition] = {"logs_mean_loss": float(free_loss[ix].mean()),
                               "oracle_64_loss": float(y0[ix].min(axis=1).mean()),
                               "logs_regret": float((free_loss[ix]-y0[ix].min(axis=1)).mean()),
                               "logs_actions": [fr["chosen_repairs"][i] for i in ix],
                               "oracle_actions": np.argmin(y0[ix],axis=1).tolist()}
out = {"scope": "Development-only diagnostic analysis; no confirmation states/text read; oracle quantities are nondeployable.",
       "baseline": baseline, "by_condition": by_condition, "probes": entries}
(ROOT / "notes/goal_v2/postrun_synthesis_diagnosis.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"baseline": baseline, "by_condition": by_condition,
                  "probes": [{k:v for k,v in e.items() if k != 'per_episode'} for e in entries]}, indent=2))
