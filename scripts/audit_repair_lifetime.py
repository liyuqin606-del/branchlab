"""Replay frozen development repairs to measure their effect across horizons.

This reads existing v2 checkpoints and dev text only. It preserves the original
NOGO evidence and verifies replay parity at its original 63/64-update endpoints.
It never changes confirmation gates or silently adopts a shorter objective.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from branchlab.experiments_v2 import ACTIONS, variant, validate_development_config
from branchlab.lookahead import prepare_decision, materialize_action
from branchlab.model import ModelConfig, TransformerLM
from branchlab.optim import AdamW
from branchlab.tokenizer import ByteBPETokenizer
from branchlab.training import TokenStream, restore_state, step, json_write


HORIZONS = (1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64)
REPLAY_SOURCES = ("model.py", "optim.py", "tokenizer.py", "training.py", "lookahead.py", "experiments_v2.py")
REPO_ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@torch.no_grad()
def counted_dev_loss(model, batches, ledger):
    """Same reduction as training.evaluate_loss, with per-forward receipts."""
    was_training = model.training
    model.eval()
    total, tokens = 0.0, 0
    try:
        for x, y in batches:
            ledger["development_eval_forward_attempts"] += 1
            logits, _ = model(x)
            ledger["development_eval_forward_batches"] += 1
            value = float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"))
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite repair-lifetime dev loss")
            total += value
            tokens += y.numel()
        if not tokens:
            raise ValueError("Development evaluation requires target tokens")
        return total / tokens
    finally:
        model.train(was_training)


def audit_repair_lifetime(source="artifacts/development_v2", artifacts="artifacts",
                          output="artifacts/repair_lifetime_v2", parity_tolerance=1e-6):
    """Execute a new, separately accounted development-only replay audit."""
    source, artifacts, output = Path(source), Path(artifacts), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve existing complete or partial lifetime evidence; choose a new output")
    if not math.isfinite(parity_tolerance) or not 0 <= parity_tolerance <= 1e-6:
        raise ValueError("parity_tolerance must be finite and between zero and 1e-6")
    config = json.loads((source / "config.json").read_text())
    validate_development_config(config)
    episodes = json.loads((source / "episodes.json").read_text())
    reference_curves = json.loads((source / "curves.json").read_text())
    original_freeze = json.loads((source / "freeze.json").read_text())
    original_collection = json.loads((source / "collection.json").read_text())
    if (source / "run_failure.json").exists() or original_collection["failures"]:
        raise ValueError("Source development run is incomplete or contains failed branches")
    expected_ids = {f"{split}-{seed:03d}-{condition}" for split, seeds in config["seeds"].items()
                    for seed in seeds for condition in config["conditions"]}
    if len(episodes) != len(expected_ids) or {e["id"] for e in episodes} != expected_ids:
        raise ValueError("Source development episodes are incomplete or duplicated")
    rows = {e["id"]: e for e in episodes}
    for row in episodes:
        for action_index, action in enumerate(ACTIONS):
            for cost, horizon in (("0", 64), ("2", 63)):
                if row["budget_losses"][cost][action_index] != reference_curves[row["id"]][action][str(horizon)]:
                    raise ValueError("Source endpoint curves disagree with episode budget labels")

    input_hashes = {}
    for name in ("tokenizer.json", "train_tokens.npy", "dev_tokens.npy"):
        actual = file_hash(artifacts / name)
        if actual != original_freeze["inputs"][name]:
            raise ValueError(f"Input changed since original development freeze: {name}")
        input_hashes[name] = actual
    original_source_checks = {}
    for name in REPLAY_SOURCES:
        relative = f"src/branchlab/{name}"
        actual = file_hash(REPO_ROOT / relative)
        if actual != original_freeze["sources"][relative]:
            raise ValueError(f"Replay source changed since original development freeze: {relative}")
        original_source_checks[relative] = actual

    tokenizer = ByteBPETokenizer.load(artifacts / "tokenizer.json")
    train_tokens = np.load(artifacts / "train_tokens.npy")
    documents = [doc for doc in np.split(train_tokens, np.flatnonzero(train_tokens == tokenizer.eos_id) + 1) if len(doc)]
    dev_tokens = np.load(artifacts / "dev_tokens.npy")[config["eval_offset_tokens"]:]
    if len(dev_tokens) < config["eval_batches"] * config["batch_size"] * config["seq_len"] + 1:
        raise ValueError("Dev window would wrap or repeat target tokens")
    evaluation_stream = TokenStream(dev_tokens, config["batch_size"], config["seq_len"])
    batches = [evaluation_stream.batch("cpu") for _ in range(config["eval_batches"])]

    checkpoint_paths = {seed: source / "checkpoints" / f"seed-{seed}.pt"
                        for seeds in config["seeds"].values() for seed in seeds}
    protocol = REPO_ROOT / "protocols/repair_lifetime_v2.md"
    if not protocol.exists():
        raise FileNotFoundError("Missing prereplay development protocol protocols/repair_lifetime_v2.md")
    source_files = [source / name for name in ("config.json", "episodes.json", "curves.json", "freeze.json", "collection.json")]
    code_files = [Path(__file__).resolve(), protocol, *(REPO_ROOT / "src/branchlab").glob("*.py")]
    freeze = {"created_utc": datetime.now(timezone.utc).isoformat(), "confirmatory": False,
              "purpose": "Development-only repair lifetime audit; original NOGO/gates remain unchanged",
              "source_files": {portable_path(path): file_hash(path) for path in source_files},
              "checkpoints": {str(seed): {"path": portable_path(path), "sha256": file_hash(path)}
                              for seed, path in checkpoint_paths.items()},
              "inputs": input_hashes,
              "code_and_protocol": {portable_path(path): file_hash(path) for path in code_files},
              "original_replay_source_hashes_match": original_source_checks,
              "runtime": {"device": "cpu", "torch_version": str(torch.__version__), "threads": torch.get_num_threads()}}
    plan = {**config, "source_data": portable_path(source),
            "source_episode_sha256": file_hash(source / "episodes.json"),
            "source_curves_sha256": file_hash(source / "curves.json"),
            "horizons": list(HORIZONS), "actions": list(ACTIONS), "parity_tolerance": parity_tolerance,
            "confirmatory": False, "audit_scope": "Replay existing histories; dev text only; no baseline retraining or probe rerun"}
    output.mkdir(parents=True, exist_ok=True)
    json_write(output / "freeze.json", freeze)
    json_write(output / "config.json", plan)

    ledger = {"status": "running", "confirmatory": False, "source_episode_count": len(episodes),
              "completed_episodes": 0, "gradient_batch_attempts": 0, "gradient_batches": 0,
              "shared_gradient_batches": 0, "candidate_optimizer_attempts": 0, "candidate_optimizer_steps": 0,
              "ordinary_optimizer_attempts": 0, "ordinary_optimizer_steps": 0,
              "development_eval_forward_attempts": 0, "development_eval_forward_batches": 0,
              "probe_forward_batches": 0, "baseline_retraining_updates": 0,
              "scope": "Separate lifetime-audit computation, never added to the original development collection. Candidate optimizer arithmetic shares one gradient. Three units per attempted gradient batch remain an approximate proxy, including failed attempts."}
    curves, comparisons = {}, []
    context = {"phase": "starting"}
    started = time.perf_counter()

    def flush(status):
        ledger["status"] = status
        ledger["elapsed_seconds"] = time.perf_counter() - started
        ledger["trained_tokens"] = ledger["gradient_batches"] * config["batch_size"] * config["seq_len"]
        ledger["attempted_training_tokens"] = ledger["gradient_batch_attempts"] * config["batch_size"] * config["seq_len"]
        ledger["approx_forward_batch_units"] = 3 * ledger["gradient_batch_attempts"] + ledger["development_eval_forward_attempts"]
        parity = {"status": status, "tolerance": parity_tolerance, "comparisons": comparisons,
                  "comparison_count": len(comparisons), "expected_comparisons": len(episodes) * len(ACTIONS) * 2,
                  "passed": status == "completed" and len(comparisons) == len(episodes) * len(ACTIONS) * 2
                  and all(row["passed"] for row in comparisons),
                  "all_observed_passed": all(row["passed"] for row in comparisons),
                  "all_exact": all(row["exact"] for row in comparisons),
                  "max_abs_difference": max((row["abs_difference"] for row in comparisons), default=0.0)}
        json_write(output / "curves.json", curves)
        json_write(output / "parity.json", parity)
        json_write(output / "collection.json", ledger)

    try:
        for split, seeds in config["seeds"].items():
            for seed in seeds:
                context = {"phase": "loading_checkpoint", "seed": seed}
                bundle = torch.load(checkpoint_paths[seed], map_location="cpu", weights_only=False)
                order = bundle["document_order"]
                if sorted(order) != list(range(len(documents))) or order != np.random.default_rng(seed).permutation(len(documents)).tolist():
                    raise ValueError("Saved document order does not match source seed/corpus")
                ordered_tokens = np.concatenate([documents[index] for index in order])
                model = TransformerLM(ModelConfig(**config["model"], vocab_size=tokenizer.vocab_size))
                optimizer = AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95), weight_decay=0.01)
                stream = TokenStream(ordered_tokens, config["batch_size"], config["seq_len"])
                for condition in config["conditions"]:
                    episode_id = f"{split}-{seed:03d}-{condition}"
                    origin, metadata = variant(bundle["snapshots"], condition)
                    if metadata != rows[episode_id]["metadata"]:
                        raise ValueError("Restored condition metadata differs from original episode")
                    restore_state(origin, model, optimizer, stream)
                    context = {"phase": "shared_gradient", "episode_id": episode_id}
                    ledger["gradient_batch_attempts"] += 1
                    prepared = prepare_decision(model, optimizer, stream, "cpu", step_number=metadata["origin_step"],
                                                scheduler=origin["scheduler"])
                    ledger["gradient_batches"] += 1
                    ledger["shared_gradient_batches"] += 1
                    curves[episode_id] = {}
                    for action in ACTIONS:
                        context = {"phase": "candidate_materialization", "episode_id": episode_id, "action": action, "horizon": 1}
                        ledger["candidate_optimizer_attempts"] += 1
                        candidate = materialize_action(prepared, model, optimizer, stream, action)
                        ledger["candidate_optimizer_steps"] += 1
                        action_curve = curves[episode_id][action] = {}
                        context["phase"] = "development_evaluation"
                        action_curve["1"] = counted_dev_loss(model, batches, ledger)
                        # Probe/evaluation work never becomes part of the real
                        # continuation's RNG or data cursor.
                        restore_state(candidate, model, optimizer, stream)
                        for horizon in range(2, max(HORIZONS) + 1):
                            context = {"phase": "ordinary_update", "episode_id": episode_id, "action": action, "horizon": horizon}
                            ledger["gradient_batch_attempts"] += 1
                            ledger["ordinary_optimizer_attempts"] += 1
                            step(model, optimizer, stream, "cpu")
                            ledger["gradient_batches"] += 1
                            ledger["ordinary_optimizer_steps"] += 1
                            if horizon in HORIZONS:
                                context["phase"] = "development_evaluation"
                                action_curve[str(horizon)] = counted_dev_loss(model, batches, ledger)
                        for horizon in (63, 64):
                            expected = reference_curves[episode_id][action][str(horizon)]
                            actual = action_curve[str(horizon)]
                            difference = abs(actual - expected)
                            comparisons.append({"episode_id": episode_id, "action": action, "horizon": horizon,
                                                "expected": expected, "actual": actual, "abs_difference": difference,
                                                "exact": actual == expected, "passed": difference <= parity_tolerance})
                        if not all(row["passed"] for row in comparisons[-2:]):
                            context["phase"] = "endpoint_parity"
                            raise ValueError("Replay endpoint failed original 63/64 parity")
                    ledger["completed_episodes"] += 1
                    flush("running")
                    print(json.dumps({"event": "repair_lifetime_episode", "id": episode_id,
                                      "completed": ledger["completed_episodes"], "source_episodes": len(episodes)}), flush=True)
        if len(comparisons) != len(episodes) * len(ACTIONS) * 2:
            raise ValueError("Incomplete endpoint parity coverage")
        expected = {"gradient_batches": len(episodes) * (1 + len(ACTIONS) * 63),
                    "candidate_optimizer_steps": len(episodes) * len(ACTIONS),
                    "ordinary_optimizer_steps": len(episodes) * len(ACTIONS) * 63,
                    "development_eval_forward_batches": len(episodes) * len(ACTIONS) * len(HORIZONS) * len(batches)}
        ledger["expected_counts"] = expected
        if any(ledger[key] != value for key, value in expected.items()):
            raise ValueError("Lifetime audit physical ledger does not match complete replay design")
        flush("completed")
        print(json.dumps({"event": "repair_lifetime_complete", "episodes": len(episodes),
                          "parity_comparisons": len(comparisons), "gradient_batches": ledger["gradient_batches"]}), flush=True)
        return curves, ledger
    except Exception as error:
        flush("FAILED_INCOMPLETE")
        json_write(output / "run_failure.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                   "status": "FAILED_INCOMPLETE", "confirmatory": False, "context": context,
                   "exception_type": type(error).__name__, "error": str(error),
                   "automatic_retry": False, "completion_admitted": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="artifacts/development_v2")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output", default="artifacts/repair_lifetime_v2")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    audit_repair_lifetime(args.source, args.artifacts, args.output)


if __name__ == "__main__":
    main()
