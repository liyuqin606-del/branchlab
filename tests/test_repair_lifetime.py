"""Synthetic replay checks mechanism-audit bookkeeping and endpoint parity."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from branchlab.experiments_v2 import collect_development
from branchlab.tokenizer import ByteBPETokenizer


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_repair_lifetime.py"
SPEC = importlib.util.spec_from_file_location("repair_lifetime_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_existing_checkpoint_replay_lifetimes_preserve_original_endpoints(tmp_path):
    inputs = tmp_path / "inputs"
    ByteBPETokenizer().save(inputs / "tokenizer.json")
    for split, first in (("train", 32), ("dev", 80)):
        docs = [np.asarray([first + (i + j) % 16 for j in range(15)] + [256]) for i in range(12)]
        np.save(inputs / f"{split}_tokens.npy", np.concatenate(docs))
    config = {"model": {"d_model": 16, "n_layers": 1, "n_heads": 2, "max_seq_len": 16},
              "seeds": {"discovery": [8], "development": [9]}, "baseline_steps": 160,
              "checkpoint_steps": [40, 80, 120, 160], "batch_size": 1, "seq_len": 4, "lr": 0.001,
              "conditions": ["native_80"], "probe_offsets": [1, 2, 4, 8],
              "eval_split": "dev", "eval_offset_tokens": 0, "eval_batches": 2,
              "audit_budget_forward_batches": 224, "reserved_final_eval_cost": 32,
              "search_budget_cells": 180, "search_seeds": [0], "scope": "Synthetic plumbing only"}
    config_path = tmp_path / "source_config.json"
    config_path.write_text(json.dumps(config))
    source = tmp_path / "development"
    output = tmp_path / "lifetimes"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        collect_development(config_path, inputs, source)
        source_files = [source / name for name in ("freeze.json", "episodes.json", "curves.json", "collection.json")]
        original_hashes = {str(path): audit.file_hash(path) for path in source_files}
        curves, ledger = audit.audit_repair_lifetime(source, inputs, output)
    finally:
        torch.set_num_threads(previous_threads)
    assert ledger["status"] == "completed"
    assert ledger["gradient_batches"] == 2 * (1 + 3 * 63) == 380
    assert ledger["ordinary_optimizer_steps"] == 2 * 3 * 63 == 378
    assert ledger["candidate_optimizer_steps"] == 6
    assert ledger["development_eval_forward_batches"] == 2 * 3 * 12 * 2 == 144
    assert ledger["baseline_retraining_updates"] == ledger["probe_forward_batches"] == 0
    assert ledger["approx_forward_batch_units"] == 3 * 380 + 144
    assert all(set(curve) == set(map(str, audit.HORIZONS)) for actions in curves.values() for curve in actions.values())
    parity = json.loads((output / "parity.json").read_text())
    assert parity["comparison_count"] == parity["expected_comparisons"] == 12
    assert parity["passed"] and parity["all_observed_passed"] and parity["all_exact"]
    assert parity["max_abs_difference"] == 0.0
    assert all(audit.file_hash(path) == value for path, value in original_hashes.items())
    assert not (inputs / "test_tokens.npy").exists()
    freeze = json.loads((output / "freeze.json").read_text())
    assert "protocols/repair_lifetime_v2.md" in freeze["code_and_protocol"]


def test_existing_lifetime_evidence_is_not_overwritten(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    failure = output / "run_failure.json"
    failure.write_text("keep original failure receipt")
    with pytest.raises(FileExistsError, match="partial lifetime evidence"):
        audit.audit_repair_lifetime(tmp_path / "absent", tmp_path / "absent", output)
    assert failure.read_text() == "keep original failure receipt"
