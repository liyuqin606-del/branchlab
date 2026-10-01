"""Real tiny development collection verifies plumbing, not scientific gains."""

import copy
import json

import numpy as np
import pytest
import torch

import branchlab.experiments_v2 as experiments_v2
from branchlab.experiments_v2 import collect_development, variant, validate_development_config
from branchlab.lookahead import prepare_decision, materialize_action
from branchlab.model import ModelConfig, TransformerLM
from branchlab.optim import AdamW
from branchlab.training import TokenStream, restore_state
from branchlab.tokenizer import ByteBPETokenizer


def tiny_config():
    return {
        "model": {"d_model": 16, "n_layers": 1, "n_heads": 2, "max_seq_len": 16},
        "seeds": {"discovery": [8], "development": [9]},
        "baseline_steps": 160, "checkpoint_steps": [40, 80, 120, 160],
        "batch_size": 1, "seq_len": 4, "lr": 0.001,
        "conditions": ["native_80", "stale_80"], "probe_offsets": [1, 2, 4, 8],
        "eval_split": "dev", "eval_offset_tokens": 0, "eval_batches": 2,
        "audit_budget_forward_batches": 224, "reserved_final_eval_cost": 32,
        "search_budget_cells": 180, "search_seeds": [0],
        "scope": "Synthetic plumbing only. Two small dev batches retain the declared reserve for this fixture.",
    }


@pytest.fixture
def input_artifacts(tmp_path):
    artifacts = tmp_path / "inputs"
    ByteBPETokenizer().save(artifacts / "tokenizer.json")
    for split, start in (("train", 32), ("dev", 80)):
        documents = [np.asarray([start + (i + j) % 16 for j in range(15)] + [256]) for i in range(12)]
        np.save(artifacts / f"{split}_tokens.npy", np.concatenate(documents))
    # Deliberately do not create test_tokens.npy: this collector may never load it.
    return artifacts


@pytest.fixture
def collected(tmp_path, input_artifacts):
    artifacts = input_artifacts
    config = tiny_config()
    config_path = tmp_path / "tiny.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "collection"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        episodes, curves, ledger = collect_development(config_path, artifacts, output)
    finally:
        torch.set_num_threads(previous_threads)
    return config, artifacts, output, episodes, curves, ledger


def test_real_collector_shared_gradient_costs_and_history(collected):
    config, artifacts, output, episodes, curves, ledger = collected
    assert len(episodes) == 4
    assert ledger["failures"] == []
    # Per state, materialization shares one gradient; all three ordinary branch
    # continuations then perform exactly 63 additional backward passes each.
    assert ledger["gradient_batches"] == 320 + 4 * (1 + 3 * 63) == 1080
    assert ledger["ordinary_optimizer_steps"] == 320 + 4 * 3 * 63 == 1076
    assert ledger["candidate_optimizer_steps"] == 4 * 3 == 12
    assert ledger["probe_forward_batches"] == 4 * 3 * 4 == 48
    assert ledger["development_eval_forward_batches"] == 4 * 3 * 2 * 2 == 48
    assert ledger["trained_tokens"] == 1080 * 4
    assert ledger["approx_forward_batch_units"] == 3 * 1080 + 48 + 48
    for episode in episodes:
        assert np.isfinite(episode["log_features"]).all()
        assert len(episode["probe_features"]) == 8
        assert set(episode["probe_costs"].values()) == {2}
        assert set(episode["budget_losses"]) == {"0", "2"}
        assert all(np.isfinite(values).all() for values in episode["budget_losses"].values())
        for cost, horizon in (("0", 64), ("2", 63)):
            assert episode["budget_losses"][cost] == [curves[episode["id"]][a][str(horizon)]
                for a in ("keep", "lr_half", "momentum_zero")]
            unspent = 224 - 32 - int(cost) - 3 * horizon
            assert 0 <= unspent < 3
        if episode["condition"] == "native_80":
            assert episode["metadata"]["origin_step"] == 80
            assert episode["metadata"]["moment_age"] == 0
        else:
            assert episode["metadata"]["origin_step"] == 160
            assert episode["metadata"]["moment_age"] == 80
    freeze = json.loads((output / "freeze.json").read_text())
    assert freeze["confirmatory"] is False
    assert set(freeze["inputs"]) == {"tokenizer.json", "train_tokens.npy", "dev_tokens.npy"}
    assert not (artifacts / "test_tokens.npy").exists()
    json.dumps({"episodes": episodes, "curves": curves, "ledger": ledger}, allow_nan=False)

    tokens = np.load(artifacts / "train_tokens.npy")
    document_count = int(np.count_nonzero(tokens == 256))
    for seed in (8, 9):
        checkpoint = torch.load(output / "checkpoints" / f"seed-{seed}.pt", map_location="cpu", weights_only=False)
        assert checkpoint["document_order"] == np.random.default_rng(seed).permutation(document_count).tolist()
        snapshots = checkpoint["snapshots"]
        assert set(snapshots) == {40, 80, 120, 160}
        for horizon, state in snapshots.items():
            assert state["step"] == horizon
            assert state["stream"]["cursor"] == (horizon * 4) % len(tokens)
            assert all(int(s["step"].item()) == horizon for s in state["optimizer"]["state"].values())
            assert state["scheduler"] == {"name": "fixed_after_warmup", "lr": 0.001}
        # Actual old moments are transplanted while the current weight tensors,
        # second moments, counters, and stream position remain at step 160.
        stale, metadata = variant(snapshots, "stale_80")
        assert metadata["moment_age"] == 80
        assert stale["stream"] == snapshots[160]["stream"]
        assert all(torch.equal(value, snapshots[160]["model"][key]) for key, value in stale["model"].items())
        for index, state in stale["optimizer"]["state"].items():
            assert torch.equal(state["exp_avg"], snapshots[80]["optimizer"]["state"][index]["exp_avg"])
            assert torch.equal(state["exp_avg_sq"], snapshots[160]["optimizer"]["state"][index]["exp_avg_sq"])
            assert int(state["step"].item()) == 160
        if seed == 8:
            # Recompute all nine cheap features using only the saved state and
            # one current gradient. No probe or continuation losses are loaded.
            documents = [doc for doc in np.split(tokens, np.flatnonzero(tokens == 256) + 1) if len(doc)]
            ordered = np.concatenate([documents[i] for i in checkpoint["document_order"]])
            stream = TokenStream(ordered, 1, 4)
            model = TransformerLM(ModelConfig(**snapshots[80]["model_config"]))
            optimizer = AdamW(model.parameters())
            restore_state(snapshots[80], model, optimizer, stream)
            prepared = prepare_decision(model, optimizer, stream, "cpu", step_number=80,
                                        scheduler=snapshots[80]["scheduler"])
            candidates = {action: materialize_action(prepared, model, optimizer, stream, action)
                          for action in ("keep", "lr_half", "momentum_zero")}
            minimal_candidates = {action: {"model": state["model"]} for action, state in candidates.items()}
            expected = experiments_v2.candidate_first_order_features(prepared, minimal_candidates)
            actual = next(e for e in episodes if e["id"] == "discovery-008-native_80")["log_features"][-9:]
            assert len(expected) == 9
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(eval_split="test"),
    lambda c: c["seeds"].update(test=[10]),
    lambda c: c["seeds"]["development"].append(8),
    lambda c: c["seeds"].update(discovery=[0]),
])
def test_no_confirmation_or_seed_reuse_allowed(tmp_path, mutation):
    config = tiny_config()
    mutation(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        collect_development(path, tmp_path / "absent-inputs", tmp_path / "output")


def test_partial_development_evidence_not_overwritten(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(tiny_config()))
    output = tmp_path / "partial"
    output.mkdir()
    evidence = output / "freeze.json"
    evidence.write_text("preserve incomplete evidence")
    failure = output / "run_failure.json"
    failure.write_text("preserve prior failure")
    with pytest.raises(FileExistsError, match="partial"):
        collect_development(path, tmp_path / "absent-inputs", output)
    assert evidence.read_text() == "preserve incomplete evidence"
    assert failure.read_text() == "preserve prior failure"


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(baseline_steps=159),
    lambda c: c.update(checkpoint_steps=[40, 80, 160]),
    lambda c: c.update(batch_size=0),
    lambda c: c.update(eval_offset_tokens=-1),
    lambda c: c.update(audit_budget_forward_batches=39),
    lambda c: c.update(conditions=["undeclared_failure"]),
])
def test_history_and_budget_preconditions(mutation):
    config = tiny_config()
    mutation(config)
    with pytest.raises(ValueError):
        validate_development_config(config)


def test_nonfinite_probe_records_fatal_failure_without_completion(tmp_path, input_artifacts, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(tiny_config()))
    output = tmp_path / "failed-probe"
    monkeypatch.setattr(experiments_v2, "evaluate_loss", lambda *args, **kwargs: float("nan"))
    with pytest.raises(FloatingPointError, match="Nonfinite probe loss"):
        collect_development(config_path, input_artifacts, output)
    assert (output / "freeze.json").exists()
    receipt = json.loads((output / "run_failure.json").read_text())
    assert receipt["status"] == "FAILED_INCOMPLETE"
    assert receipt["completion_admitted"] is False
    assert receipt["automatic_retry"] is False
    assert not (output / "config.json").exists()


def test_declared_moment_interventions_preserve_snapshot_inputs():
    snapshots = {}
    for horizon in (40, 80, 120, 160):
        snapshots[horizon] = {"optimizer": {"param_groups": [{"lr": 0.001}],
            "state": {0: {"exp_avg": torch.tensor([float(horizon), 1.0]),
                          "exp_avg_sq": torch.tensor([2.0, 3.0]), "step": torch.tensor(float(horizon))}}},
            "model": {"weight": torch.tensor([float(horizon)])}}
    before = copy.deepcopy(snapshots)
    matched, meta = variant(snapshots, "matched_stale_80")
    current_norm = snapshots[160]["optimizer"]["state"][0]["exp_avg"].norm()
    torch.testing.assert_close(matched["optimizer"]["state"][0]["exp_avg"].norm(), current_norm)
    assert meta["moment_age"] == 80 and meta["moment_scale"] > 1
    blend, meta = variant(snapshots, "blend_80")
    torch.testing.assert_close(blend["optimizer"]["state"][0]["exp_avg"], torch.tensor([120.0, 1.0]))
    assert meta["old_moment_fraction"] == 0.5
    for horizon in snapshots:
        assert torch.equal(snapshots[horizon]["optimizer"]["state"][0]["exp_avg"],
                           before[horizon]["optimizer"]["state"][0]["exp_avg"])
