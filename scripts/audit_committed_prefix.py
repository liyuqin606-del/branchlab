"""Development-only four-update branch-prefix observation on existing v2 states.

The selected prefix is restorable without redoing training. All three actions
are collected for offline analysis; a deployed pair may reveal only its two
members and must pay the losing prefix plus both calibration evaluations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
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
from branchlab.training import TokenStream, capture_state, restore_state, step, json_write


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = Path(__file__).resolve().with_name("audit_repair_lifetime.py")
_spec = importlib.util.spec_from_file_location("_committed_prefix_lifetime_helpers", HELPER_PATH)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
file_hash, portable_path = _helpers.file_hash, _helpers.portable_path
counted_dev_loss, REPLAY_SOURCES = _helpers.counted_dev_loss, _helpers.REPLAY_SOURCES
PREFIX_STEPS = 4
CALIBRATION_POSITIONS = (65, 81, 97, 113)
HORIZONS = (58, 61, 64)


def state_hash(value):
    """Hash nested checkpoint values without serialization/storage identity."""
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"torch:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(f"numpy:{item.dtype}:{item.shape}:".encode())
            digest.update(item.tobytes(order="C"))
        elif isinstance(item, dict):
            digest.update(f"dict:{len(item)}:".encode())
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(f"{type(item).__name__}:{len(item)}:".encode())
            for entry in item:
                visit(entry)
        elif item is None or isinstance(item, (str, bool, int, float, np.generic)):
            encoded = f"{type(item).__name__}:{repr(item)}".encode()
            digest.update(str(len(encoded)).encode() + b":" + encoded)
        else:
            raise TypeError(f"Unsupported checkpoint hash value: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def state_signature(state):
    return {"sha256": state_hash(state),
            "components": {key: state_hash(value) for key, value in state.items()}}


def calibration_batches(stream, positions=CALIBRATION_POSITIONS):
    """Origin-relative, nonwrapping batches with disjoint target indices.

    As in ordinary adjacent TokenStream batches, position 65's first context
    token is position 64's last target. Their target sets do not overlap.
    """
    if (not positions or len(positions) != len(set(positions))
            or any(not isinstance(position, int) or isinstance(position, bool) or position <= max(HORIZONS)
                   for position in positions)):
        raise ValueError("Calibration positions must be distinct integers beyond retained training positions 1-64")
    width, cursor = stream.batch_size * stream.seq_len, stream.cursor
    if cursor + max(positions) * width >= len(stream.tokens):
        raise ValueError("Origin-relative training/calibration window would wrap")
    batches, windows = [], []
    for position in positions:
        start = cursor + (position - 1) * width
        block = stream.tokens[start:start + width + 1]
        batches.append((block[:-1].reshape(stream.batch_size, stream.seq_len),
                        block[1:].reshape(stream.batch_size, stream.seq_len)))
        windows.append({"position": position, "input_start": start,
                        "target_start": start + 1, "target_end_exclusive": start + width + 1})
    metadata = {"origin_cursor": cursor, "batch_target_tokens": width,
                "retained_training_target_start": cursor + 1,
                "retained_training_target_end_exclusive": cursor + max(HORIZONS) * width + 1,
                "calibration_windows": windows, "target_indices_disjoint": True, "no_wrap": True,
                "boundary_context_overlap": "Position 65 reuses one context token at the end of training position 64; target indices are disjoint."}
    return batches, metadata


@torch.no_grad()
def counted_calibration_loss(model, batch, ledger):
    was_training = model.training
    model.eval()
    try:
        x, y = batch
        ledger["calibration_forward_attempts"] += 1
        logits, _ = model(x)
        ledger["calibration_forward_batches"] += 1
        value = float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")) / y.numel()
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite committed-prefix calibration loss")
        return value
    finally:
        model.train(was_training)


def parameter_statistics(model, optimizer):
    """Read the already paid step-four gradient and post-step-four moment."""
    result = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise ValueError("A dense prefix parameter has no saved final training gradient")
        gradient = parameter.grad.detach().reshape(-1).cpu()
        moment = optimizer.state[parameter]["exp_avg"].detach().reshape(-1).cpu()
        gradient_norm, moment_norm = gradient.norm(), moment.norm()
        cosine = torch.dot(gradient, moment) / (gradient_norm * moment_norm).clamp_min(1e-12)
        values = (float(gradient_norm), float(moment_norm), float(cosine))
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("Nonfinite prefix parameter statistics")
        result.append(dict(zip(("name", "gradient_norm", "moment_norm", "gradient_moment_cosine"), (name, *values))))
    return result


def audit_committed_prefix(source="artifacts/development_v2", artifacts="artifacts",
                           output="artifacts/committed_prefix_v3", parity_tolerance=1e-6):
    source, artifacts, output = Path(source), Path(artifacts), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve existing complete or partial prefix evidence; choose a new output")
    if not math.isfinite(parity_tolerance) or not 0 <= parity_tolerance <= 1e-6:
        raise ValueError("parity_tolerance must be finite and between zero and 1e-6")
    config = json.loads((source / "config.json").read_text())
    validate_development_config(config)
    if config["audit_budget_forward_batches"] - config["reserved_final_eval_cost"] != 192:
        raise ValueError("Committed-prefix horizon accounting requires 192 non-reserved forward units")
    episodes = json.loads((source / "episodes.json").read_text())
    reference_curves = json.loads((source / "curves.json").read_text())
    original_freeze = json.loads((source / "freeze.json").read_text())
    original_collection = json.loads((source / "collection.json").read_text())
    if (source / "run_failure.json").exists() or original_collection["failures"]:
        raise ValueError("Source development run is incomplete or contains failed branches")
    expected_ids = {f"{split}-{seed:03d}-{condition}" for split, seeds in config["seeds"].items()
                    for seed in seeds for condition in config["conditions"]}
    if len(episodes) != len(expected_ids) or {row["id"] for row in episodes} != expected_ids:
        raise ValueError("Source development episodes are incomplete or duplicated")
    rows = {row["id"]: row for row in episodes}
    for row in episodes:
        for index, action in enumerate(ACTIONS):
            if row["budget_losses"]["0"][index] != reference_curves[row["id"]][action]["64"]:
                raise ValueError("Source endpoint curves disagree with episode budget labels")
    input_hashes = {}
    for name in ("tokenizer.json", "train_tokens.npy", "dev_tokens.npy"):
        actual = file_hash(artifacts / name)
        if actual != original_freeze["inputs"][name]:
            raise ValueError(f"Input changed since original development freeze: {name}")
        input_hashes[name] = actual
    source_checks = {}
    for name in REPLAY_SOURCES:
        relative = f"src/branchlab/{name}"
        actual = file_hash(REPO_ROOT / relative)
        if actual != original_freeze["sources"][relative]:
            raise ValueError(f"Replay source changed since original development freeze: {relative}")
        source_checks[relative] = actual
    tokenizer = ByteBPETokenizer.load(artifacts / "tokenizer.json")
    train_tokens = np.load(artifacts / "train_tokens.npy")
    documents = [doc for doc in np.split(train_tokens, np.flatnonzero(train_tokens == tokenizer.eos_id) + 1) if len(doc)]
    dev_tokens = np.load(artifacts / "dev_tokens.npy")[config["eval_offset_tokens"]:]
    if len(dev_tokens) < config["eval_batches"] * config["batch_size"] * config["seq_len"] + 1:
        raise ValueError("Dev window would wrap or repeat target tokens")
    evaluation_stream = TokenStream(dev_tokens, config["batch_size"], config["seq_len"])
    dev_batches = [evaluation_stream.batch("cpu") for _ in range(config["eval_batches"])]
    checkpoint_paths = {seed: source / "checkpoints" / f"seed-{seed}.pt"
                        for seeds in config["seeds"].values() for seed in seeds}
    protocol = REPO_ROOT / "protocols/committed_prefix_v3.md"
    proposer = REPO_ROOT / "protocols/prefix_proposer_v3.json"
    if not protocol.exists() or not proposer.exists():
        raise FileNotFoundError("Missing pre-run protocol or frozen prefix proposer")
    source_files = [source / name for name in ("config.json", "episodes.json", "curves.json", "freeze.json", "collection.json")]
    code_files = [Path(__file__).resolve(), HELPER_PATH, protocol, proposer, *(REPO_ROOT / "src/branchlab").glob("*.py")]
    analysis_script = REPO_ROOT / "scripts/analyze_committed_prefix.py"
    if analysis_script.exists():
        code_files.append(analysis_script)
    freeze = {"created_utc": datetime.now(timezone.utc).isoformat(), "confirmatory": False,
              "purpose": "Development-only committed-prefix observation; original evidence and confirmation gates unchanged",
              "source_files": {portable_path(path): file_hash(path) for path in source_files},
              "checkpoints": {str(seed): {"path": portable_path(path), "sha256": file_hash(path)}
                              for seed, path in checkpoint_paths.items()},
              "inputs": input_hashes, "code_and_protocol": {portable_path(path): file_hash(path) for path in code_files},
              "original_replay_source_hashes_match": source_checks,
              "runtime": {"device": "cpu", "torch_version": str(torch.__version__), "threads": torch.get_num_threads()}}
    plan = {**config, "source_data": portable_path(source), "source_episode_sha256": file_hash(source / "episodes.json"),
            "source_curves_sha256": file_hash(source / "curves.json"), "horizons": list(HORIZONS),
            "prefix_steps": PREFIX_STEPS, "prefix_len": PREFIX_STEPS,
            "calibration_positions": list(CALIBRATION_POSITIONS), "actions": list(ACTIONS),
            "pair_extra_cost": 17, "pair_retained_updates": 58, "prefix_only_extra_cost": 9,
            "prefix_only_retained_updates": 61, "parity_tolerance": parity_tolerance, "confirmatory": False,
            "gradient_statistics_semantics": "Step-four clipped gradient was computed before update four; the moment was read after update four. No new gradient at the final prefix weights is computed.",
            "audit_scope": "Existing state replay; dev and declared training calibration only; all-action offline table is separate from logical two-action deployment."}
    output.mkdir(parents=True, exist_ok=True)
    json_write(output / "freeze.json", freeze)
    json_write(output / "config.json", plan)
    ledger = {"status": "running", "confirmatory": False, "source_episode_count": len(episodes), "completed_episodes": 0,
              "gradient_batch_attempts": 0, "gradient_batches": 0, "shared_gradient_batches": 0,
              "candidate_optimizer_attempts": 0, "candidate_optimizer_steps": 0,
              "ordinary_optimizer_attempts": 0, "ordinary_optimizer_steps": 0,
              "calibration_forward_attempts": 0, "calibration_forward_batches": 0,
              "development_eval_forward_attempts": 0, "development_eval_forward_batches": 0,
              "baseline_retraining_updates": 0, "test_forward_batches": 0,
              "scope": "Physical offline construction counts; candidate optimizer arithmetic shares only the first gradient. Logical pair cost is 17 extra forward units. Three units per attempted gradient batch are a proxy, not a measured speedup."}
    curves, observations, comparisons, verifications = {}, {}, [], []
    context = {"phase": "starting"}
    started = time.perf_counter()

    def flush(status):
        ledger["status"] = status
        ledger["elapsed_seconds"] = time.perf_counter() - started
        ledger["trained_tokens"] = ledger["gradient_batches"] * config["batch_size"] * config["seq_len"]
        ledger["attempted_training_tokens"] = ledger["gradient_batch_attempts"] * config["batch_size"] * config["seq_len"]
        ledger["approx_forward_batch_units"] = (3 * ledger["gradient_batch_attempts"]
                                                 + ledger["calibration_forward_attempts"] + ledger["development_eval_forward_attempts"])
        expected_comparisons = len(episodes) * len(ACTIONS)
        parity = {"status": status, "tolerance": parity_tolerance, "comparisons": comparisons,
                  "comparison_count": len(comparisons), "expected_comparisons": expected_comparisons,
                  "passed": status == "completed" and len(comparisons) == expected_comparisons and all(row["passed"] for row in comparisons),
                  "all_exact": all(row["exact"] for row in comparisons),
                  "max_abs_difference": max((row["abs_difference"] for row in comparisons), default=0.0)}
        verification = {"status": status, "records": verifications, "count": len(verifications),
                        "expected_count": expected_comparisons,
                        "passed": status == "completed" and len(verifications) == expected_comparisons and all(row["passed"] for row in verifications)}
        for name, value in (("curves.json", curves), ("observations.json", observations), ("parity.json", parity),
                            ("prefix_verification.json", verification), ("collection.json", ledger)):
            json_write(output / name, value)

    def ordinary_update(model, optimizer, stream):
        ledger["gradient_batch_attempts"] += 1
        ledger["ordinary_optimizer_attempts"] += 1
        metrics = step(model, optimizer, stream, "cpu")
        ledger["gradient_batches"] += 1
        ledger["ordinary_optimizer_steps"] += 1
        return metrics

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
                    context = {"phase": "restoring_origin", "episode_id": episode_id}
                    origin, metadata = variant(bundle["snapshots"], condition)
                    if metadata != rows[episode_id]["metadata"]:
                        raise ValueError("Restored condition metadata differs from original episode")
                    restore_state(origin, model, optimizer, stream)
                    calibration, windows = calibration_batches(stream)
                    context["phase"] = "shared_gradient"
                    ledger["gradient_batch_attempts"] += 1
                    prepared = prepare_decision(model, optimizer, stream, "cpu", step_number=metadata["origin_step"], scheduler=origin["scheduler"])
                    ledger["gradient_batches"] += 1
                    ledger["shared_gradient_batches"] += 1
                    curves[episode_id], observations[episode_id] = {}, {}
                    for action in ACTIONS:
                        context = {"phase": "candidate_materialization", "episode_id": episode_id, "action": action, "horizon": 1}
                        ledger["candidate_optimizer_attempts"] += 1
                        materialize_action(prepared, model, optimizer, stream, action)
                        ledger["candidate_optimizer_steps"] += 1
                        observation = observations[episode_id][action] = {
                            "prefix_metrics": [{"update_index": 1, **prepared.metrics}], "calibration_losses": [],
                            "calibration_windows": windows}
                        for horizon in range(2, PREFIX_STEPS + 1):
                            context.update(phase="prefix_update", horizon=horizon)
                            observation["prefix_metrics"].append({"update_index": horizon, **ordinary_update(model, optimizer, stream)})
                        observation["parameter_statistics"] = parameter_statistics(model, optimizer)
                        observation["optimizer_groups"] = [{key: group[key] for key in ("lr", "betas", "eps", "weight_decay")}
                                                           for group in optimizer.param_groups]
                        prefix = capture_state(model, optimizer, stream, metadata["origin_step"] + PREFIX_STEPS, origin["scheduler"])
                        signature = state_signature(prefix)
                        expected_cursor = origin["stream"]["cursor"] + PREFIX_STEPS * config["batch_size"] * config["seq_len"]
                        optimizer_steps = [int(value["step"]) for value in prefix["optimizer"]["state"].values()]
                        if (prefix["stream"]["cursor"] != expected_cursor or not optimizer_steps
                                or any(value != prefix["step"] for value in optimizer_steps)):
                            raise ValueError("Prefix update counter or data cursor is not four committed updates after origin")
                        context["phase"] = "calibration"
                        for batch in calibration:
                            observation["calibration_losses"].append(counted_calibration_loss(model, batch, ledger))
                        after_calibration = capture_state(model, optimizer, stream, prefix["step"], prefix["scheduler"])
                        # Restore the complete retained prefix, including RNG.
                        # No prefix gradient or action is executed again.
                        restore_state(prefix, model, optimizer, stream)
                        restored = capture_state(model, optimizer, stream, prefix["step"], prefix["scheduler"])
                        after_signature, restored_signature = state_signature(after_calibration), state_signature(restored)
                        unchanged_fields = ("model", "optimizer", "stream", "step", "scheduler", "model_training")
                        observation["prefix_state_sha256"] = signature["sha256"]
                        verification = {"episode_id": episode_id, "action": action, "step": prefix["step"],
                                        "expected_step": metadata["origin_step"] + PREFIX_STEPS, "cursor": expected_cursor,
                                        "optimizer_steps": sorted(set(optimizer_steps)), "before_calibration": signature,
                                        "after_calibration": after_signature, "after_restore": restored_signature,
                                        "calibration_preserved_training_state": all(signature["components"][key] == after_signature["components"][key] for key in unchanged_fields),
                                        "restored_exactly": signature == restored_signature}
                        verification["passed"] = verification["calibration_preserved_training_state"] and verification["restored_exactly"]
                        verifications.append(verification)
                        if not verification["passed"]:
                            raise ValueError("Calibration or prefix restoration changed the committed training state")
                        action_curve = curves[episode_id][action] = {}
                        for horizon in range(PREFIX_STEPS + 1, max(HORIZONS) + 1):
                            context.update(phase="ordinary_update", horizon=horizon)
                            ordinary_update(model, optimizer, stream)
                            if horizon in HORIZONS:
                                context["phase"] = "development_evaluation"
                                # Preserve RNG/cursor even if future eval models
                                # have stateful or stochastic evaluation paths.
                                endpoint = capture_state(model, optimizer, stream, metadata["origin_step"] + horizon, origin["scheduler"])
                                action_curve[str(horizon)] = counted_dev_loss(model, dev_batches, ledger)
                                restore_state(endpoint, model, optimizer, stream)
                        expected, actual = reference_curves[episode_id][action]["64"], action_curve["64"]
                        difference = abs(actual - expected)
                        comparisons.append({"episode_id": episode_id, "action": action, "horizon": 64, "expected": expected,
                                            "actual": actual, "abs_difference": difference, "exact": actual == expected,
                                            "passed": difference <= parity_tolerance})
                        if not comparisons[-1]["passed"]:
                            context["phase"] = "endpoint_parity"
                            raise ValueError("Replay endpoint failed original 64-update parity")
                    ledger["completed_episodes"] += 1
                    flush("running")
                    print(json.dumps({"event": "committed_prefix_episode", "id": episode_id,
                                      "completed": ledger["completed_episodes"], "source_episodes": len(episodes)}), flush=True)
        expected = {"gradient_batches": len(episodes) * (1 + len(ACTIONS) * 63),
                    "candidate_optimizer_steps": len(episodes) * len(ACTIONS), "ordinary_optimizer_steps": len(episodes) * len(ACTIONS) * 63,
                    "calibration_forward_batches": len(episodes) * len(ACTIONS) * len(CALIBRATION_POSITIONS),
                    "development_eval_forward_batches": len(episodes) * len(ACTIONS) * len(HORIZONS) * len(dev_batches)}
        ledger["expected_counts"] = expected
        if (len(comparisons) != len(episodes) * len(ACTIONS) or len(verifications) != len(comparisons)
                or any(ledger[key] != value for key, value in expected.items())):
            raise ValueError("Committed-prefix collection coverage or physical ledger is incomplete")
        flush("completed")
        print(json.dumps({"event": "committed_prefix_complete", "episodes": len(episodes), "gradient_batches": ledger["gradient_batches"]}), flush=True)
        return curves, observations, ledger
    except Exception as error:
        flush("FAILED_INCOMPLETE")
        json_write(output / "run_failure.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "FAILED_INCOMPLETE",
                   "confirmatory": False, "context": context, "exception_type": type(error).__name__, "error": str(error),
                   "automatic_retry": False, "completion_admitted": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="artifacts/development_v2")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output", default="artifacts/committed_prefix_v3")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    audit_committed_prefix(args.source, args.artifacts, args.output)


if __name__ == "__main__":
    main()
