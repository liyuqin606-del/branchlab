"""Audit diagnostic value across repair lifetimes using DEVELOPMENT curves only.

Every requested horizon/regularization setting is retained. This script neither
chooses a winning setting nor evaluates a confirmation gate. The pure function
``analyze_lifetimes`` can be exercised without loading checkpoints or text.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from branchlab.synthesis import _Ridge
from branchlab.value_aware import ValueAwareDiagnosticLearner


ACTIONS = ("keep", "lr_half", "momentum_zero")
DEFAULT_HORIZONS = (4, 8, 16, 32, 64)
DEFAULT_ALPHAS = (1.0, 10.0)


def _validate(episodes: Sequence[Mapping[str, Any]], curves: Mapping[str, Any],
              horizons: Sequence[int], config: Mapping[str, Any]) -> None:
    if not episodes or {e["split"] for e in episodes} != {"discovery", "development"}:
        raise ValueError("Only a nonempty discovery/development collection is allowed; no test/confirmation split")
    ids = [str(e["id"]) for e in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate episode IDs")
    if set(curves) != set(ids):
        raise ValueError("Curve IDs must exactly match the development episode collection")
    actual_seeds = {split: {e["seed"] for e in episodes if e["split"] == split}
                    for split in ("discovery", "development")}
    if actual_seeds["discovery"] & actual_seeds["development"]:
        raise ValueError("Discovery and development training seeds overlap")
    if "seeds" not in config or "conditions" not in config:
        raise ValueError("The development config must declare seeds and conditions")
    if set(config["seeds"]) != {"discovery", "development"}:
        raise ValueError("A confirmation/test seed config is not accepted")
    if any(actual_seeds[s] != set(config["seeds"][s]) for s in actual_seeds):
        raise ValueError("Episode seeds differ from the specified development config")
    expected = {(split, seed, condition) for split, seeds in config["seeds"].items()
                for seed in seeds for condition in config["conditions"]}
    coverage = [(e["split"], e["seed"], e.get("condition")) for e in episodes]
    if len(coverage) != len(expected) or set(coverage) != expected:
        raise ValueError("Episode coverage must contain every configured seed/condition exactly once")
    if any(isinstance(h, bool) or not isinstance(h, int) or h < 2 for h in horizons):
        raise ValueError("Horizons must be integer update counts >= 2")
    if len(horizons) != len(set(horizons)):
        raise ValueError("Duplicate horizons would duplicate evidence")
    needed = {str(h) for end in horizons for h in (end - 1, end)}
    for e in episodes:
        if any(cost != 2 for cost in e["probe_costs"].values()):
            raise ValueError("This audit assumes each purchased pair probe costs exactly 2 units")
        for action in ACTIONS:
            available = curves[e["id"]][action]
            if not needed.issubset(available):
                raise ValueError(f"Missing lifetime curves for {e['id']}/{action}")
            for h in needed:
                value = available[h]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("All supplied dev losses must be finite numeric measurements")


def _scored(name: str, prediction: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
            horizon: int, reserve: int, search_seed: int | None = None) -> dict[str, Any]:
    lookup = {e["id"]: e for e in rows if e["split"] == "development"}
    arrays = [prediction[k] for k in ("episode_ids", "chosen_repairs", "probe_ids", "probe_costs", "budget_costs")]
    if len(arrays[0]) != len(set(arrays[0])):
        raise ValueError("Duplicate prediction episode IDs would duplicate evidence")
    if len({len(a) for a in arrays}) != 1 or set(arrays[0]) != set(lookup):
        raise ValueError("Prediction coverage does not match the entire development split")
    decisions = []
    for eid, action, probe, cost, route in zip(*arrays):
        if route not in (0, 2) or cost not in (0, 2):
            raise ValueError("Only full-horizon/one-shorter routing is supported")
        steps = horizon if route == 0 else horizon - 1
        total = 3 * horizon + reserve
        row = lookup[eid]
        loss = row["budget_losses"][str(route)][action]
        decisions.append({"episode_id": eid, "seed": row["seed"], "condition": row.get("condition"),
                          "action": ACTIONS[action], "probe_id": probe, "probe_cost": cost,
                          "budget_route": route, "total_updates": steps, "development_loss": loss,
                          "unspent_units": total - reserve - 3 * steps - cost})
    per_seed = [{"seed": seed, "development_loss": statistics.mean(d["development_loss"] for d in decisions if d["seed"] == seed),
                 "acquisitions": sum(d["probe_id"] is not None for d in decisions if d["seed"] == seed)}
                for seed in sorted({d["seed"] for d in decisions})]
    return {"name": name, "search_seed": search_seed,
            "mean_development_loss": statistics.mean(s["development_loss"] for s in per_seed),
            "episode_mean_development_loss": statistics.mean(d["development_loss"] for d in decisions),
            "mean_probe_cost": statistics.mean(d["probe_cost"] for d in decisions),
            "acquisition_count": sum(d["probe_id"] is not None for d in decisions),
            "search_revealed_cells": prediction.get("search_revealed_cells", 0),
            "search_replay_probe_cost": prediction.get("search_replay_probe_cost", 0),
            "episodes": len(decisions), "per_seed": per_seed, "decisions": decisions}


def _oracles(rows: Sequence[Mapping[str, Any]], free: Mapping[str, Any]) -> dict[str, Any]:
    dev = sorted([e for e in rows if e["split"] == "development"], key=lambda e: e["id"])
    base = {d["episode_id"]: d["development_loss"] for d in free["decisions"]}
    entries = [{"episode_id": e["id"], "seed": e["seed"],
                "full_horizon_action_oracle": min(e["budget_losses"]["0"]),
                "paid_short_horizon_action_oracle": min(e["budget_losses"]["2"]),
                "free_action_and_horizon_oracle": min(e["budget_losses"]["0"] + e["budget_losses"]["2"]),
                "conditional_paid_or_skip_oracle": min(base[e["id"]], min(e["budget_losses"]["2"])),
                "logs_only_loss": base[e["id"]]} for e in dev]
    keys = ("full_horizon_action_oracle", "paid_short_horizon_action_oracle",
            "free_action_and_horizon_oracle", "conditional_paid_or_skip_oracle", "logs_only_loss")
    per_seed = [{"seed": seed, **{key: statistics.mean(e[key] for e in entries if e["seed"] == seed) for key in keys}}
                for seed in sorted({e["seed"] for e in entries})]
    return {"scope": "Hindsight future dev labels; nondeployable ceilings, not methods",
            "mean_over_seeds": {key: statistics.mean(row[key] for row in per_seed) for key in keys},
            "per_seed": per_seed, "episodes": entries}


def analyze_lifetimes(episodes: Sequence[Mapping[str, Any]], curves: Mapping[str, Any],
                      config: Mapping[str, Any], *, horizons: Sequence[int] = DEFAULT_HORIZONS,
                      alphas: Sequence[float] = DEFAULT_ALPHAS,
                      expert_probe: str = "lr_half:1:next_batch_loss") -> dict[str, Any]:
    """Return all development results and serializable policies, without I/O.

    A horizon h uses budget ``3*h + reserved_final_eval_cost``. Paid probes cost
    2 units and leave h-1 updates plus one unspent unit. The free shorter-horizon
    control acquires no probe and leaves three unspent units.
    """
    _validate(episodes, curves, horizons, config)
    if not alphas or any(not math.isfinite(float(a)) or float(a) <= 0 for a in alphas):
        raise ValueError("Regularization settings must be finite and positive")
    if len(alphas) != len(set(alphas)):
        raise ValueError("Duplicate regularization settings")
    reserve = int(config.get("reserved_final_eval_cost", 32))
    cap = int(config.get("search_budget_cells", 180))
    configurations, states = [], {}
    for h in horizons:
        rows = copy.deepcopy(list(episodes))
        for e in rows:
            e["budget_losses"] = {"0": [curves[e["id"]][a][str(h)] for a in ACTIONS],
                                   "2": [curves[e["id"]][a][str(h - 1)] for a in ACTIONS]}
        ds = sorted([e for e in rows if e["split"] == "discovery"], key=lambda e: e["id"])
        vs = sorted([e for e in rows if e["split"] == "development"], key=lambda e: e["id"])
        for alpha in alphas:
            methods = []
            run_id = f"h{h}-alpha{float(alpha):g}"
            for name, probe, always, budget in (("logs_only", None, False, 0),
                                               ("logs_only_short", None, False, 2),
                                               ("fixed_expert", expert_probe, True, 0),
                                               ("fixed_conditional_expert", expert_probe, False, 0)):
                policy = ValueAwareDiagnosticLearner(ridge_alpha=float(alpha), search_budget_cells=cap).fit_fixed(
                    rows, probe, always_probe=always, budget_cost=budget)
                states[f"{run_id}-{name}"] = policy.state_dict()
                methods.append(_scored(name, policy.predict(rows, split="development"), rows, h, reserve))
            xds = np.asarray([e["log_features"] for e in ds])
            xvs = np.asarray([e["log_features"] for e in vs])
            y = np.asarray([e["budget_losses"]["0"] + e["budget_losses"]["2"] for e in ds])
            joint = _Ridge.fit(xds, y, float(alpha))
            choice = np.argmin(joint.predict(xvs), axis=1)
            prediction = {"episode_ids": [e["id"] for e in vs], "chosen_repairs": (choice % len(ACTIONS)).tolist(),
                          "probe_ids": [None] * len(vs), "probe_costs": [0] * len(vs),
                          "budget_costs": [0 if a < len(ACTIONS) else 2 for a in choice]}
            methods.append(_scored("logs_joint_action_horizon", prediction, rows, h, reserve))
            states[f"{run_id}-logs_joint_action_horizon"] = {"learner": "ridge_joint_action_horizon", "ridge_alpha": float(alpha),
                                                           "actions": list(ACTIONS), "routes": [0, 2], "model": joint.state_dict()}
            for strategy in ("counterexample", "random", "enumeration"):
                for seed in ([0, 1, 2] if strategy == "random" else [0]):
                    policy = ValueAwareDiagnosticLearner(strategy=strategy, ridge_alpha=float(alpha),
                                                        search_budget_cells=cap, seed=seed).fit(rows)
                    states[f"{run_id}-{strategy}-{seed}"] = policy.state_dict()
                    methods.append(_scored(strategy, policy.predict(rows, split="development"), rows, h, reserve, seed))
            baseline = methods[0]
            base_seeds = {s["seed"]: s["development_loss"] for s in baseline["per_seed"]}
            comparisons = [{"method": m["name"], "search_seed": m["search_seed"],
                            "method_minus_logs_mean": m["mean_development_loss"] - baseline["mean_development_loss"],
                            "per_seed": [{"seed": s["seed"], "method_minus_logs": s["development_loss"] - base_seeds[s["seed"]]}
                                         for s in m["per_seed"]]} for m in methods[1:]]
            configurations.append({"run_id": run_id, "horizon": h, "paid_horizon": h - 1,
                                   "ridge_alpha": float(alpha), "free_features": len(ds[0]["log_features"]),
                                   "budget_forward_units": 3 * h + reserve, "reserved_final_eval_cost": reserve,
                                   "search_budget_cells": cap, "methods": methods, "comparisons_to_logs": comparisons,
                                   "oracles": _oracles(rows, baseline)})
    return {"status": "completed", "confirmatory": False, "selection_performed": False,
            "scope": "Development repair-timescale audit; every requested setting retained; no winner or confirmation gate selected",
            "horizons": list(horizons), "alphas": [float(a) for a in alphas],
            "episode_count": len(episodes), "actions": list(ACTIONS), "configurations": configurations,
            "policy_states": states}


def _render(summary: Mapping[str, Any]) -> str:
    lines = ["# Development repair-lifetime audit", "", summary["scope"], "",
             "All losses are macro-averaged over development training seeds. The same seeds recur across rows and horizons. No confirmation gate or winner is selected.", "",
             "| Horizon | Alpha | Method | Search seed | Dev loss | Probe acquisitions | Mean probe cost | Search cells |",
             "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for run in summary["configurations"]:
        for method in run["methods"]:
            lines.append(f"| {run['horizon']} | {run['ridge_alpha']:g} | {method['name']} | {method['search_seed'] if method['search_seed'] is not None else '-'} | {method['mean_development_loss']:.9f} | {method['acquisition_count']} | {method['mean_probe_cost']:.3f} | {method['search_revealed_cells']} |")
    lines += ["", "The h-1 logs control uses no observation and leaves three proxy units unspent; a paid two-unit probe also routes to h-1 and leaves one unit unspent. Joint logs can select either horizon for free. This separates short-training effects from paid information.", "",
              "Per-seed differences, every decision, and nondeployable oracle ceilings are retained in summary.json. Query ledgers and fitted model states are in programs/. Physical curve-collection cost remains separate from this analysis replay.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=Path("artifacts/repair_lifetime_v2/curves.json"))
    parser.add_argument("--episodes", type=Path, default=Path("artifacts/development_v2/episodes.json"))
    parser.add_argument("--config", type=Path, default=Path("artifacts/development_v2/config.json"))
    parser.add_argument("--protocol", type=Path, default=Path("protocols/repair_lifetime_v2.md"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/repair_lifetime_v2_analysis"))
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new empty output directory to retain prior audits")
    sidecars = {}
    for filename in ("config.json", "freeze.json", "parity.json", "collection.json"):
        source = args.curves.parent / filename
        if not source.is_file():
            raise FileNotFoundError(f"Required lifetime completion receipt is missing: {source}")
        sidecars[filename] = json.loads(source.read_text())
    if sidecars["collection.json"].get("status") != "completed":
        raise ValueError("The lifetime collection is not marked completed")
    if sidecars["parity.json"].get("passed") is not True:
        raise ValueError("Final 63/64-step parity did not pass")
    if not args.protocol.is_file():
        raise FileNotFoundError(f"The lifetime protocol must be available for provenance: {args.protocol}")
    summary = analyze_lifetimes(json.loads(args.episodes.read_text()), json.loads(args.curves.read_text()),
                                json.loads(args.config.read_text()), horizons=args.horizons, alphas=args.alphas)
    states = summary.pop("policy_states")
    sources = {"episodes": args.episodes, "curves": args.curves, "development_config": args.config,
               "analysis_script": Path(__file__), "lifetime_protocol": args.protocol}
    sources.update({"lifetime_" + name.removesuffix(".json"): args.curves.parent / name for name in sidecars})
    summary["source_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}
    summary["collection_metadata"] = sidecars
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "programs").mkdir()
    for name, state in states.items():
        (args.output / "programs" / f"{name}.json").write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output / "REPORT.md").write_text(_render(summary))
    print(json.dumps({"status": summary["status"], "confirmatory": False, "selection_performed": False,
                      "configurations": len(summary["configurations"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
