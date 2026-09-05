"""Synthetic tiny training checks pipeline plumbing, never scientific gains."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import branchlab.experiments as experiments
from branchlab.tokenizer import ByteBPETokenizer


def tiny_config():
    return {"model": {"d_model": 16, "n_layers": 1, "n_heads": 2, "max_seq_len": 16},
            "seeds": {"discovery": [0, 1, 2], "development": [3], "test": [4]},
            "baseline_steps": 2, "batch_size": 1, "seq_len": 4, "lr": 0.001,
            "conditions": ["clean", "high_lr", "opposed_momentum"],
            "repair_actions": ["keep", "lr_half", "momentum_zero"], "repair_horizon": 8,
            "audit_budget_forward_batches": 160, "max_repair_steps": 52,
            "search_budget_cells": 30, "max_probes": 1, "search_seeds": [0],
            "scope": "Synthetic plumbing smoke only, not evidence of diagnostic gains"}


@pytest.fixture
def tiny_artifacts(tmp_path):
    artifacts = tmp_path / "inputs"
    ByteBPETokenizer().save(artifacts / "tokenizer.json")
    for split, first in (("train", 32), ("dev", 80), ("test", 128)):
        documents = [np.asarray([first + (j + i) % 16 for j in range(15)] + [256]) for i in range(12)]
        np.save(artifacts / f"{split}_tokens.npy", np.concatenate(documents))
    return artifacts


def test_analytic_budget_conserves_units_and_reserves_evaluations():
    assert [experiments.probe_cost(h) for h in (2, 4, 8)] == [16, 28, 52]
    for cost, expected in ((0, (52, 2)), (42, (38, 2)), (104, (18, 0))):
        steps, unspent = experiments.repair_budget(160, cost, 52)
        assert (steps, unspent) == expected
        assert cost + 3 * steps + 2 + unspent == 160
    for total, cost in ((4, 0), (20, 20)):
        with pytest.raises(ValueError):
            experiments.repair_budget(total, cost, 52)


@pytest.mark.parametrize("mutation", [
    lambda c: c["seeds"]["test"].append(0),
    lambda c: c["seeds"]["discovery"].append(0),
    lambda c: c.update(repair_actions=["lr_half", "keep", "momentum_zero"]),
    lambda c: c.update(max_repair_steps=7),
    lambda c: c.update(repair_horizon=53),
    lambda c: c.update(audit_budget_forward_batches=54, max_probes=2),
])
def test_config_rejects_leakage_and_incompatible_dsl(mutation):
    config = tiny_config()
    mutation(config)
    with pytest.raises(ValueError):
        experiments.validate_config(config)


def test_real_tiny_collection_and_analysis_smoke(tiny_artifacts, tmp_path):
    config = tiny_config()
    out = tmp_path / "real-smoke"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        episodes, curves, ledger, failures = experiments.collect(config, tiny_artifacts, out, "cpu")
        methods, gate = experiments.analyze(episodes, curves, config, out, failures)
    finally:
        torch.set_num_threads(previous_threads)
    assert len(episodes) == 15
    assert not failures
    assert ledger["baseline_training_updates"] == 10
    assert ledger["branch_training_updates"] == 15 * 3 * 52
    assert ledger["training_updates"] == ledger["training_update_attempts"] == 2350
    assert ledger["evaluation_batches"] == ledger["evaluation_batches_completed"] == 20 + 2 * 15 * 3 * 52
    assert all(len(e["probe_features"]) == 18 for e in episodes)
    assert all(len(curves[e["id"]][a]) == 52 for e in episodes for a in config["repair_actions"])
    assert len(list((out / "checkpoints").glob("seed-*.pt"))) == 5
    for method in methods:
        assert len(method["decisions"]) == 3
        assert np.isfinite(method["mean_test_loss"])
        if method["name"] in ("counterexample", "random", "enumeration"):
            assert method["search_revealed_cells"] <= config["search_budget_cells"]
        for decision in method["decisions"]:
            assert decision["probe_cost"] + 3 * decision["repair_steps"] + 2 + decision["unspent_budget"] == 160
    assert gate["status"] in ("NOGO", "PASS_EXPLORATORY")
    # Numeric output here validates computation/serialization only.
    json.dumps({"methods": methods, "gate": gate}, allow_nan=False)


@pytest.mark.parametrize("failure_phase", ["training_update", "test_evaluation"])
def test_failed_attempts_remain_in_physical_ledger(tiny_artifacts, tmp_path, monkeypatch, failure_phase):
    config = tiny_config()
    config["conditions"] = ["clean"]
    config["baseline_steps"] = 1
    if failure_phase == "training_update":
        def failing_update(*args, **kwargs):
            raise FloatingPointError("Synthetic failed forward/update attempt")
        monkeypatch.setattr(experiments, "step", failing_update)
    else:
        real_evaluate = experiments.evaluate_loss
        calls = 0

        def failing_second_evaluation(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls % 2 == 0:
                raise FloatingPointError("Synthetic failed test evaluation")
            return real_evaluate(*args, **kwargs)
        monkeypatch.setattr(experiments, "evaluate_loss", failing_second_evaluation)
    episodes, curves, ledger, failures = experiments.collect(config, tiny_artifacts, tmp_path / failure_phase, "cpu")
    assert len(failures) == 15
    assert {f["phase"] for f in failures} == {failure_phase}
    assert ledger["training_update_attempts"] == 5 + 15
    assert ledger["branch_training_update_attempts"] == 15
    expected_successful = 5 if failure_phase == "training_update" else 20
    assert ledger["training_updates"] == expected_successful
    assert ledger["evaluation_batches"] == (20 if failure_phase == "training_update" else 50)
    assert ledger["evaluation_batches_completed"] == (20 if failure_phase == "training_update" else 35)
    assert all(record["failed"] for actions in curves.values() for curve in actions.values() for record in curve.values())


def test_partial_output_is_never_overwritten(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(tiny_config()))
    output = tmp_path / "interrupted"
    output.mkdir()
    evidence = output / "freeze.json"
    evidence.write_text("preserve this interrupted run")
    args = SimpleNamespace(config=config_path, output=output, artifacts=tmp_path / "absent", device="cpu")
    with pytest.raises(FileExistsError, match="partial evidence"):
        experiments.run_pilot(args)
    assert evidence.read_text() == "preserve this interrupted run"


def test_condition_changes_only_declared_optimizer_state():
    base = {"model": {"weights": torch.ones(2)}, "optimizer": {
        "param_groups": [{"lr": 0.01}], "state": {0: {"exp_avg": torch.tensor([1.0, -2.0]),
                                                      "exp_avg_sq": torch.tensor([3.0, 4.0])}}}}
    original = copy.deepcopy(base)
    high = experiments._condition(base, "high_lr")
    opposed = experiments._condition(base, "opposed_momentum")
    assert high["optimizer"]["param_groups"][0]["lr"] == 0.04
    assert torch.equal(opposed["optimizer"]["state"][0]["exp_avg"], torch.tensor([-3.0, 6.0]))
    for modified in (high, opposed):
        assert torch.equal(modified["model"]["weights"], original["model"]["weights"])
        assert torch.equal(modified["optimizer"]["state"][0]["exp_avg_sq"], original["optimizer"]["state"][0]["exp_avg_sq"])
    assert torch.equal(base["optimizer"]["state"][0]["exp_avg"], original["optimizer"]["state"][0]["exp_avg"])
