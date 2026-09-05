"""Collect matched real training branches, then replay frozen diagnostic searches.

The complete table is an expensive benchmark construction step. Replay query
budgets never remove that cost. Test text losses live in a separate curve table
and are not passed to the repair predictor or program search.
"""
from __future__ import annotations
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .model import ModelConfig
from .synthesis import (GreedyProgramSynthesizer, candidate_probe_specs,
                        evaluate_no_probe, evaluate_fixed_probes)
from .tokenizer import ByteBPETokenizer
from .training import (train_baseline, capture_state, restore_state, apply_action,
                       evaluation_batches, evaluate_loss, step, json_write)


def probe_cost(horizon):
    return 2 * (3 * horizon + 2)


def repair_budget(total, diagnostic_cost, max_steps):
    """Reserve one dev and one test forward batch after continued training."""
    for name, value in (("total", total), ("diagnostic_cost", diagnostic_cost), ("max_steps", max_steps)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if total < diagnostic_cost + 2:
        raise ValueError("Diagnostic exceeds the per-state budget")
    steps = min(max_steps, int((total - diagnostic_cost - 2) // 3))
    if steps < 1:
        raise ValueError("Budget leaves no repair updates")
    return steps, total - diagnostic_cost - 3 * steps - 2


def validate_config(config):
    """Reject split leakage and configurations incompatible with the fixed DSL."""
    if set(config["seeds"]) != {"discovery", "development", "test"}:
        raise ValueError("seeds must contain discovery, development, and test splits")
    all_seeds = []
    for split, seeds in config["seeds"].items():
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"{split} seeds must be a nonempty list")
        if any(not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32 for seed in seeds):
            raise ValueError("Seeds must be integers in [0, 2**32)")
        all_seeds.extend(seeds)
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("Seeds must be unique within and disjoint across all splits")
    if config["repair_actions"] != ["keep", "lr_half", "momentum_zero"]:
        raise ValueError("repair_actions must preserve fixed DSL order: keep, lr_half, momentum_zero")
    conditions = config["conditions"]
    if (not isinstance(conditions, list) or not conditions or len(set(conditions)) != len(conditions)
            or not set(conditions) <= {"clean", "high_lr", "opposed_momentum"}):
        raise ValueError("conditions must be a nonempty unique subset of the declared perturbations")
    for key in ("baseline_steps", "batch_size", "seq_len", "repair_horizon", "max_repair_steps",
                "audit_budget_forward_batches"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("max_probes", "search_budget_cells"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config["max_probes"] > len(candidate_probe_specs()):
        raise ValueError("max_probes cannot exceed the finite probe catalog")
    if config["max_repair_steps"] < 8 or config["repair_horizon"] > config["max_repair_steps"]:
        raise ValueError("max_repair_steps must cover all probe horizons and repair_horizon")
    search_seeds = config["search_seeds"]
    if (not isinstance(search_seeds, list) or not search_seeds or
            any(not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32 for seed in search_seeds)
            or len(set(search_seeds)) != len(search_seeds)):
        raise ValueError("search_seeds must contain unique nonnegative integers below 2**32")
    if not math.isfinite(config["lr"]) or config["lr"] <= 0:
        raise ValueError("lr must be finite and positive")
    model_config = ModelConfig(**config["model"])
    if config["seq_len"] > model_config.max_seq_len:
        raise ValueError("seq_len exceeds model max_seq_len")
    # Every permitted program, including the fixed expert and direct-trial
    # comparator, must leave at least one repair update at audit time.
    worst_program_cost = sum(sorted((probe_cost(p.steps) for p in candidate_probe_specs()), reverse=True)
                             [:config["max_probes"]])
    repair_budget(config["audit_budget_forward_batches"],
                  max(worst_program_cost, probe_cost(4), 3 * (3 * 4 + 2)), config["max_repair_steps"])


def _flat(values):
    return torch.cat([value.detach().reshape(-1).cpu() for value in values])


def _cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))


def _condition(base, condition):
    state = copy.deepcopy(base)
    if condition == "high_lr":
        for group in state["optimizer"]["param_groups"]:
            group["lr"] *= 4
    elif condition == "opposed_momentum":
        for value in state["optimizer"]["state"].values():
            value["exp_avg"].mul_(-3)
    elif condition != "clean":
        raise ValueError(condition)
    return state


def collect(config, artifact_dir, output_dir, device):
    validate_config(config)
    started = time.perf_counter()
    out, artifacts = Path(output_dir), Path(artifact_dir)
    tokenizer = ByteBPETokenizer.load(artifacts / "tokenizer.json")
    tokens = {s: np.load(artifacts / f"{s}_tokens.npy") for s in ("train", "dev", "test")}
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, **config["model"])
    episodes, curves, failures = [], {}, []
    ledger = {"training_updates": 0, "training_update_attempts": 0,
              "evaluation_batches": 0, "evaluation_batches_completed": 0, "trained_tokens": 0,
              "baseline_training_updates": 0, "branch_training_updates": 0,
              "branch_training_update_attempts": 0,
              "cost_interpretation": "Physical table construction including labels, audit curves and failed attempts. training_updates counts completed optimizer updates; evaluation_batches counts attempted evaluation forwards. Any 3-unit training-attempt cost is an analytic proxy, not exact failed-step FLOPs. Replay savings do not remove construction cost."}
    histories = {}
    # Training documents end with EOS. Reorder whole documents, preserving each
    # document's tokens, without accessing any validation or test document.
    ends = np.flatnonzero(tokens["train"] == tokenizer.eos_id) + 1
    documents = np.split(tokens["train"], ends)
    documents = [doc for doc in documents if len(doc)]
    for split, seeds in config["seeds"].items():
        for seed in seeds:
            order = np.random.default_rng(seed).permutation(len(documents))
            train_tokens = np.concatenate([documents[i] for i in order])
            model, optimizer, stream, base, history = train_baseline(
                model_config, train_tokens, tokens["dev"], device=device, seed=seed,
                steps=config["baseline_steps"], batch_size=config["batch_size"],
                seq_len=config["seq_len"], lr=config["lr"], eval_interval=config["baseline_steps"])
            histories[str(seed)] = history
            ledger["baseline_training_updates"] += config["baseline_steps"]
            ledger["training_update_attempts"] += config["baseline_steps"]
            ledger["evaluation_batches"] += 2 * len(history)
            ledger["evaluation_batches_completed"] += 2 * len(history)
            # Last completed baseline gradient is an available training-log
            # reference, not a recomputed gradient at the branched state.
            gradient = _flat([p.grad for p in model.parameters()])
            initial_weights = _flat(model.parameters())
            checkpoint_dir = out / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"trainer": base, "last_clipped_gradient": gradient, "document_order": order.tolist()},
                       checkpoint_dir / f"seed-{seed}.pt")
            evals = {s: evaluation_batches(tokens[s], config["batch_size"], config["seq_len"],
                                          count=1, device=device) for s in ("dev", "test")}
            for condition in config["conditions"]:
                episode_id = f"{split}-{seed:03d}-{condition}"
                state = _condition(base, condition)
                restore_state(state, model, optimizer, stream)
                initial_loss = history[-1]["dev_loss"]
                moment = _flat([optimizer.state[p]["exp_avg"] for p in model.parameters()])
                logs = [initial_loss, history[-1]["loss"], history[-1]["grad_norm"],
                        math.log(optimizer.param_groups[0]["lr"]), float(moment.norm()),
                        _cosine(moment, gradient)]
                branches, action_curves = {}, {}
                for action in config["repair_actions"]:
                    restore_state(state, model, optimizer, stream)
                    apply_action(optimizer, action)
                    branch, curve, training_losses = {}, {}, []
                    failed = False
                    for h in range(1, config["max_repair_steps"] + 1):
                        if not failed:
                            phase = "training_update"
                            try:
                                ledger["training_update_attempts"] += 1
                                ledger["branch_training_update_attempts"] += 1
                                update = step(model, optimizer, stream, device)
                                ledger["branch_training_updates"] += 1
                                training_losses.append(update["loss"])
                                phase = "dev_evaluation"
                                ledger["evaluation_batches"] += 1
                                dev = evaluate_loss(model, evals["dev"])
                                ledger["evaluation_batches_completed"] += 1
                                if not math.isfinite(dev):
                                    raise FloatingPointError("Nonfinite dev evaluation loss")
                                phase = "test_evaluation"
                                ledger["evaluation_batches"] += 1
                                test = evaluate_loss(model, evals["test"])
                                ledger["evaluation_batches_completed"] += 1
                                if not math.isfinite(test):
                                    raise FloatingPointError("Nonfinite test evaluation loss")
                            except FloatingPointError as error:
                                failures.append({"episode_id": episode_id, "action": action,
                                                 "step": h, "phase": phase, "error": str(error)})
                                failed = True
                        if failed:
                            # Failure penalty keeps the case in every denominator.
                            dev = test = 100.0
                        curve[str(h)] = {"dev_loss": dev, "test_loss": test, "failed": failed}
                        if h in (2, 4, 8):
                            branch[str(h)] = {"dev_loss": dev,
                                "recovery_slope": (training_losses[-1] - training_losses[0]) / max(h-1, 1) if training_losses else 100.0,
                                "grad_alignment": _cosine(_flat(model.parameters()) - initial_weights, -gradient) if not failed else 0.0}
                    branches[action], action_curves[action] = branch, curve
                features = {}
                for probe in candidate_probe_specs():
                    a, b = branches[probe.action][str(probe.steps)], branches["keep"][str(probe.steps)]
                    key = "dev_loss" if probe.readout == "loss_delta" else probe.readout
                    features[probe.id] = [a[key] - b[key]]
                episode = {"id": episode_id, "seed": seed, "split": split, "condition": condition,
                    "log_features": logs, "probe_features": features,
                    "probe_costs": {p.id: probe_cost(p.steps) for p in candidate_probe_specs()},
                    "repair_losses": [action_curves[a][str(config["repair_horizon"])]["dev_loss"] for a in config["repair_actions"]],
                    "direct_trial_choice": int(np.argmin([branches[a]["4"]["dev_loss"] for a in config["repair_actions"]]))}
                episodes.append(episode)
                curves[episode_id] = action_curves
                json_write(out / "episodes.json", episodes)
                json_write(out / "curves.json", curves)
                print(json.dumps({"event": "episode_collected", "id": episode_id,
                                  "repair_losses": episode["repair_losses"]}), flush=True)
    ledger["training_updates"] = ledger["baseline_training_updates"] + ledger["branch_training_updates"]
    ledger["trained_tokens"] = ledger["training_updates"] * config["batch_size"] * config["seq_len"]
    ledger["attempted_training_tokens"] = ledger["training_update_attempts"] * config["batch_size"] * config["seq_len"]
    ledger["failed_training_update_attempts"] = ledger["training_update_attempts"] - ledger["training_updates"]
    ledger["elapsed_seconds"] = time.perf_counter() - started
    json_write(out / "collection.json", ledger)
    json_write(out / "baseline_histories.json", histories)
    json_write(out / "failures.json", failures)
    return episodes, curves, ledger, failures


def summarize_method(name, result, episodes, curves, config, search_seed=None, cost_override=None):
    rows = {e["id"]: e for e in episodes if e["split"] == "test"}
    decisions = []
    probes = result.get("selected_probes", [])
    for episode_id, choice in zip(result["episode_ids"], result["chosen_repairs"]):
        row = rows[episode_id]
        cost = cost_override if cost_override is not None else sum(row["probe_costs"][p] for p in probes)
        steps, unspent = repair_budget(config["audit_budget_forward_batches"], cost, config["max_repair_steps"])
        action = config["repair_actions"][choice]
        decisions.append({"episode_id": episode_id, "seed": row["seed"], "action": action,
                          "repair_steps": steps, "test_loss": curves[episode_id][action][str(steps)]["test_loss"],
                          "regret": row["repair_losses"][choice] - min(row["repair_losses"]),
                          "probe_cost": cost, "unspent_budget": unspent})
    per_seed = [{"seed": seed,
                 "test_loss": float(np.mean([d["test_loss"] for d in decisions if d["seed"] == seed])),
                 "regret": float(np.mean([d["regret"] for d in decisions if d["seed"] == seed]))}
                for seed in sorted({d["seed"] for d in decisions})]
    return {"name": name, "search_seed": search_seed, "selected_probes": probes,
            "mean_test_loss": float(np.mean([d["test_loss"] for d in decisions])),
            "mean_regret": float(np.mean([d["regret"] for d in decisions])),
            "mean_probe_cost": float(np.mean([d["probe_cost"] for d in decisions])),
            "mean_repair_steps": float(np.mean([d["repair_steps"] for d in decisions])),
            "search_revealed_cells": result.get("search_revealed_cells", 0),
            "search_replay_probe_cost": result.get("search_replay_probe_cost", 0),
            "per_seed": per_seed, "decisions": decisions}


def analyze(episodes, curves, config, output_dir, failures=()):
    out = Path(output_dir)
    methods = []
    audit = sorted([e for e in episodes if e["split"] == "test"], key=lambda e:e["id"])
    for name, result, cost in [
        ("passive_keep", {"episode_ids": [e["id"] for e in audit], "chosen_repairs": [0]*len(audit)}, 0),
        ("logs_only", evaluate_no_probe(episodes), None),
        ("fixed_expert", evaluate_fixed_probes(episodes, ["lr_half:4:loss_delta"]), None),
        ("direct_short_trial", {"episode_ids": [e["id"] for e in audit],
                                "chosen_repairs": [e["direct_trial_choice"] for e in audit]}, 3*(3*4+2))]:
        methods.append(summarize_method(name, result, episodes, curves, config, cost_override=cost))
    for strategy in ("counterexample", "random", "enumeration", "full_enumeration"):
        for search_seed in (config["search_seeds"] if strategy != "full_enumeration" else [0]):
            budget = config["search_budget_cells"] if strategy != "full_enumeration" else 100000
            synth = GreedyProgramSynthesizer(strategy="enumeration" if strategy == "full_enumeration" else strategy,
                        max_probes=config["max_probes"], search_budget_cells=budget, seed=search_seed).fit(episodes)
            json_write(out / "programs" / f"{strategy}-{search_seed}.json", synth.state_dict())
            result = synth.evaluate(episodes)
            methods.append(summarize_method(strategy, result, episodes, curves, config, search_seed))
    def seed_average(name, seed):
        return float(np.mean([p["test_loss"] for m in methods if m["name"] == name for p in m["per_seed"] if p["seed"] == seed]))
    comparisons = []
    for name in ("passive_keep", "logs_only", "fixed_expert", "direct_short_trial", "random", "enumeration"):
        diffs = [{"seed": seed, "counterexample_minus_baseline": seed_average("counterexample", seed)-seed_average(name, seed)}
                 for seed in config["seeds"]["test"]]
        comparisons.append({"baseline": name, "per_seed": diffs,
                            "all_audit_seeds_lower": all(d["counterexample_minus_baseline"] < 0 for d in diffs)})
    passed = not failures and all(c["all_audit_seeds_lower"] for c in comparisons)
    gate = {"status": "PASS_EXPLORATORY" if passed else "NOGO",
            "reason": "All paired directions pass the frozen engineering gate; three audit seeds still do not establish RSI or broad superiority."
                      if passed else "The counterexample program does not beat every prespecified comparator in every audit seed, or a branch failed. No diagnostic advantage or RSI claim is admitted.",
            "comparisons": comparisons}
    return methods, gate


def run_pilot(args):
    config_path, out = Path(args.config), Path(args.output)
    config = json.loads(config_path.read_text())
    validate_config(config)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Pilot output contains complete or partial evidence; choose a new empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    freeze_paths = [config_path, Path("protocols/pilot_v1.md"), *Path(__file__).parent.glob("*.py")]
    freeze = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "files": {str(p) if not p.is_absolute() else f"src/branchlab/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in freeze_paths if p.exists()},
              "artifact_sha256": {name: hashlib.sha256((Path(args.artifacts) / name).read_bytes()).hexdigest()
                                  for name in ("tokenizer.json", "train_tokens.npy", "dev_tokens.npy", "test_tokens.npy")}}
    json_write(out / "freeze.json", freeze)
    episodes, curves, collection, failures = collect(config, args.artifacts, out, args.device)
    methods, gate = analyze(episodes, curves, config, out, failures)
    summary = {"status": "completed", "scope": config["scope"], "config": config,
               "episodes": len(episodes), "audit_seeds": config["seeds"]["test"], "collection": collection,
               "methods": methods, "gate": gate, "failures": failures,
               "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                           "device": args.device, "threads": torch.get_num_threads()},
               "limitations": [f"One model size and one small repartitioned corpus; {len(config['seeds']['test'])} held-out seeds",
                    f"One fixed {config['batch_size'] * config['seq_len']}-token dev batch and one fixed {config['batch_size'] * config['seq_len']}-token test batch per branch evaluation",
                    "Synthetic perturbations; not real Marin training failures",
                    "Validation regret selects programs; cost-adjusted test loss is a separate outcome",
                    "All repair labels and probe tables are physically precomputed and charged in collection",
                    "Fixed finite grammar and synthesizer; recursive self-improvement is not tested"]}
    json_write(out / "summary.json", summary)
    print(json.dumps({"event": "pilot_complete", "gate": gate["status"], "episodes": len(episodes)}), flush=True)
