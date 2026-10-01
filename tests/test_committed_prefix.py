"""Synthetic checks of retained prefixes, observation boundaries and receipts."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from branchlab.experiments_v2 import collect_development
from branchlab.tokenizer import ByteBPETokenizer
from branchlab.training import TokenStream, apply_action, capture_state, restore_state, step


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_committed_prefix.py"
SPEC = importlib.util.spec_from_file_location("committed_prefix_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture(scope="module")
def source_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("committed-prefix")
    inputs = root / "inputs"
    ByteBPETokenizer().save(inputs / "tokenizer.json")
    for split, first, length in (("train", 32, 128), ("dev", 80, 16)):
        docs = [np.asarray([first + (i + j) % 16 for j in range(length - 1)] + [256]) for i in range(12)]
        np.save(inputs / f"{split}_tokens.npy", np.concatenate(docs))
    config = {"model": {"d_model": 16, "n_layers": 1, "n_heads": 2, "max_seq_len": 16},
              "seeds": {"discovery": [8], "development": [9]}, "baseline_steps": 160,
              "checkpoint_steps": [40, 80, 120, 160], "batch_size": 1, "seq_len": 4, "lr": 0.001,
              "conditions": ["native_80"], "probe_offsets": [1, 2, 4, 8],
              "eval_split": "dev", "eval_offset_tokens": 0, "eval_batches": 2,
              "audit_budget_forward_batches": 224, "reserved_final_eval_cost": 32,
              "search_budget_cells": 180, "search_seeds": [0], "scope": "Synthetic plumbing only"}
    config_path = root / "source_config.json"
    config_path.write_text(json.dumps(config))
    source = root / "development"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        collect_development(config_path, inputs, source)
        yield inputs, source, config
    finally:
        torch.set_num_threads(previous_threads)


def test_retained_prefix_matches_regular_training_and_original_endpoints(source_bundle, tmp_path):
    inputs, source, config = source_bundle
    original_hashes = {name: audit.file_hash(source / name) for name in ("freeze.json", "episodes.json", "curves.json", "collection.json")}
    output = tmp_path / "audit"
    curves, observations, ledger = audit.audit_committed_prefix(source, inputs, output)
    assert ledger["status"] == "completed"
    assert ledger["gradient_batches"] == ledger["gradient_batch_attempts"] == 380
    assert ledger["candidate_optimizer_steps"] == ledger["candidate_optimizer_attempts"] == 6
    assert ledger["ordinary_optimizer_steps"] == ledger["ordinary_optimizer_attempts"] == 378
    assert ledger["calibration_forward_batches"] == ledger["calibration_forward_attempts"] == 24
    assert ledger["development_eval_forward_batches"] == ledger["development_eval_forward_attempts"] == 36
    assert ledger["approx_forward_batch_units"] == 3 * 380 + 24 + 36 == 1200
    assert ledger["baseline_retraining_updates"] == ledger["test_forward_batches"] == 0
    assert all(set(curve) == {"58", "61", "64"} for row in curves.values() for curve in row.values())
    parity = json.loads((output / "parity.json").read_text())
    assert parity["passed"] and parity["all_exact"] and parity["comparison_count"] == 6
    verification = json.loads((output / "prefix_verification.json").read_text())
    assert verification["passed"] and verification["count"] == 6
    assert all(record["before_calibration"] == record["after_calibration"] == record["after_restore"] for record in verification["records"])

    # Independently perform ordinary apply_action+step, without prepare or
    # materialization, and compare every component at the retained boundary.
    bundle = torch.load(source / "checkpoints/seed-8.pt", map_location="cpu", weights_only=False)
    train = np.load(inputs / "train_tokens.npy")
    docs = [doc for doc in np.split(train, np.flatnonzero(train == 256) + 1) if len(doc)]
    ordered = np.concatenate([docs[index] for index in bundle["document_order"]])
    model = audit.TransformerLM(audit.ModelConfig(**config["model"], vocab_size=257))
    optimizer = audit.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95), weight_decay=0.01)
    stream = TokenStream(ordered, 1, 4)
    origin = bundle["snapshots"][80]
    for action in audit.ACTIONS:
        restore_state(origin, model, optimizer, stream)
        apply_action(optimizer, action)
        metrics = [{"update_index": index + 1, **step(model, optimizer, stream, "cpu")} for index in range(4)]
        prefix = capture_state(model, optimizer, stream, 84, origin["scheduler"])
        observed = observations["discovery-008-native_80"][action]
        assert observed["prefix_state_sha256"] == audit.state_hash(prefix)
        assert observed["prefix_metrics"] == metrics
        assert observed["parameter_statistics"] == audit.parameter_statistics(model, optimizer)
        assert len(observed["calibration_losses"]) == 4
        assert np.isfinite(observed["calibration_losses"]).all()
    assert all(audit.file_hash(source / name) == value for name, value in original_hashes.items())
    assert not (inputs / "test_tokens.npy").exists()
    freeze = json.loads((output / "freeze.json").read_text())
    assert "protocols/committed_prefix_v3.md" in freeze["code_and_protocol"]
    assert "protocols/prefix_proposer_v3.json" in freeze["code_and_protocol"]


def test_calibration_positions_match_training_batches_without_cursor_or_rng_changes():
    stream = TokenStream(np.arange(2048), 2, 4, cursor=32)
    before, rng_before = stream.state_dict(), torch.get_rng_state().clone()
    batches, windows = audit.calibration_batches(stream)
    reference = TokenStream(stream.tokens, 2, 4, cursor=32)
    ordinary = [reference.batch("cpu") for _ in range(113)]
    for actual, position in zip(batches, audit.CALIBRATION_POSITIONS):
        assert all(torch.equal(a, b) for a, b in zip(actual, ordinary[position - 1]))
    assert stream.state_dict() == before and torch.equal(torch.get_rng_state(), rng_before)
    assert windows["target_indices_disjoint"] and windows["no_wrap"]
    training_targets = set(range(windows["retained_training_target_start"], windows["retained_training_target_end_exclusive"]))
    seen = set()
    for window in windows["calibration_windows"]:
        targets = set(range(window["target_start"], window["target_end_exclusive"]))
        assert not targets & training_targets and not targets & seen
        seen |= targets
    with pytest.raises(ValueError, match="would wrap"):
        audit.calibration_batches(TokenStream(np.arange(80), 2, 4))
    for positions in ((65, 65), (64,), (True,)):
        with pytest.raises(ValueError, match="distinct integers"):
            audit.calibration_batches(stream, positions)


def test_state_hash_is_storage_independent_and_covers_rng_optimizer_and_cursor():
    state = {"model": {"w": torch.arange(3.0)}, "optimizer": {"m": torch.ones(3), "step": torch.tensor(4)},
             "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}, "stream": {"cursor": 16}}
    assert audit.state_hash(state) == audit.state_hash(copy.deepcopy(state))
    for component in ("model", "optimizer", "rng", "stream"):
        changed = copy.deepcopy(state)
        if component == "model":
            changed[component]["w"][0] += 1
        elif component == "optimizer":
            changed[component]["m"][0] += 1
        elif component == "rng":
            changed[component]["torch"][0] ^= 1
        else:
            changed[component]["cursor"] += 1
        assert audit.state_hash(changed) != audit.state_hash(state)


def test_nonfinite_calibration_preserves_failed_run_and_rejects_overwrite(source_bundle, tmp_path, monkeypatch):
    inputs, source, _ = source_bundle

    class NonfiniteEvaluationModel(audit.TransformerLM):
        def forward(self, *args, **kwargs):
            logits, cache = super().forward(*args, **kwargs)
            return (logits if self.training else logits * float("nan")), cache

    monkeypatch.setattr(audit, "TransformerLM", NonfiniteEvaluationModel)
    output = tmp_path / "failure"
    with pytest.raises(FloatingPointError, match="Nonfinite committed-prefix"):
        audit.audit_committed_prefix(source, inputs, output)
    receipt = (output / "run_failure.json").read_text()
    failure = json.loads(receipt)
    assert failure["context"]["phase"] == "calibration"
    assert failure["completion_admitted"] is False and failure["automatic_retry"] is False
    ledger = json.loads((output / "collection.json").read_text())
    assert ledger["status"] == "FAILED_INCOMPLETE" and ledger["completed_episodes"] == 0
    assert ledger["gradient_batches"] == 4 and ledger["calibration_forward_attempts"] == 1
    assert ledger["development_eval_forward_attempts"] == 0
    assert json.loads((output / "parity.json").read_text())["passed"] is False
    assert json.loads((output / "prefix_verification.json").read_text())["passed"] is False
    with pytest.raises(FileExistsError, match="partial prefix evidence"):
        audit.audit_committed_prefix(source, inputs, output)
    assert (output / "run_failure.json").read_text() == receipt
