"""Development-only lifetime analysis checks using tiny artificial fixtures."""
import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest


spec = importlib.util.spec_from_file_location("repair_lifetime_analysis", Path(__file__).parents[1] / "scripts/analyze_repair_lifetime.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture():
    probe = "lr_half:1:next_batch_loss"
    rows, curves = [], {}
    for split, seeds in (("discovery", [0, 1, 2]), ("development", [3, 4])):
        for seed in seeds:
            for i in range(4):
                eid = f"{split}-{seed}-{i}"
                bit = i % 2 * 2 - 1
                rows.append({"id": eid, "seed": seed, "split": split, "condition": f"fixture-{i}",
                             "log_features": [1.0], "probe_features": {probe: [bit]},
                             "probe_costs": {probe: 2}, "budget_losses": {"0": [99, 99, 99], "2": [99, 99, 99]}})
                curves[eid] = {action: {str(h): 5 - 0.1 * h + [bit, -bit, 1][a]
                                        for h in (3, 4, 7, 8)}
                               for a, action in enumerate(analysis.ACTIONS)}
    config = {"seeds": {"discovery": [0, 1, 2], "development": [3, 4]},
              "conditions": [f"fixture-{i}" for i in range(4)],
              "search_budget_cells": 100, "reserved_final_eval_cost": 32}
    return rows, curves, config


def test_all_horizons_and_settings_retained_without_selection():
    rows, curves, cfg = fixture()
    before = copy.deepcopy(rows)
    result = analysis.analyze_lifetimes(rows, curves, cfg, horizons=[4, 8], alphas=[1, 10])
    assert rows == before
    assert result["confirmatory"] is False and result["selection_performed"] is False
    assert len(result["configurations"]) == 4
    assert {r["run_id"] for r in result["configurations"]} == {"h4-alpha1", "h4-alpha10", "h8-alpha1", "h8-alpha10"}
    for run in result["configurations"]:
        assert len(run["methods"]) == 10
        assert {m["search_seed"] for m in run["methods"] if m["name"] == "random"} == {0, 1, 2}
        for m in run["methods"]:
            assert len(m["per_seed"]) == 2
            assert m["search_revealed_cells"] <= 100
            for d in m["decisions"]:
                assert d["development_loss"] == curves[d["episode_id"]][d["action"]][str(d["total_updates"])]
                assert 3 * d["total_updates"] + d["probe_cost"] + d["unspent_units"] + 32 == run["budget_forward_units"]
        shorter = next(m for m in run["methods"] if m["name"] == "logs_only_short")
        assert shorter["mean_probe_cost"] == 0
        assert all(d["unspent_units"] == 3 for d in shorter["decisions"])
        assert run["oracles"]["mean_over_seeds"]["free_action_and_horizon_oracle"] <= run["methods"][0]["mean_development_loss"]


def test_incomplete_or_confirmation_collection_is_rejected():
    rows, curves, cfg = fixture()
    del curves[rows[0]["id"]]["keep"]["3"]
    with pytest.raises(ValueError, match="Missing lifetime"):
        analysis.analyze_lifetimes(rows, curves, cfg, horizons=[4], alphas=[1])
    rows, curves, cfg = fixture()
    rows[-1]["split"] = "test"
    with pytest.raises(ValueError, match="no test/confirmation"):
        analysis.analyze_lifetimes(rows, curves, cfg, horizons=[4], alphas=[1])


def test_dropped_condition_and_duplicate_prediction_are_rejected():
    rows, curves, cfg = fixture()
    dropped = rows.pop()
    del curves[dropped["id"]]
    with pytest.raises(ValueError, match="every configured seed/condition"):
        analysis.analyze_lifetimes(rows, curves, cfg, horizons=[4], alphas=[1])
    rows, curves, cfg = fixture()
    ids = [e["id"] for e in rows if e["split"] == "development"]
    ids.append(ids[0])
    pred = {"episode_ids": ids, "chosen_repairs": [0]*len(ids), "probe_ids": [None]*len(ids),
            "probe_costs": [0]*len(ids), "budget_costs": [0]*len(ids)}
    with pytest.raises(ValueError, match="Duplicate prediction"):
        analysis._scored("fixture", pred, rows, 4, 32)


def test_cli_requires_receipts_and_records_analysis_provenance(tmp_path, monkeypatch):
    rows, curves, cfg = fixture()
    lifetime = tmp_path / "lifetime"
    lifetime.mkdir()
    episodes_path, config_path, protocol_path = (tmp_path / p for p in ("episodes.json", "config.json", "protocol.md"))
    episodes_path.write_text(json.dumps(rows))
    config_path.write_text(json.dumps(cfg))
    protocol_path.write_text("Frozen synthetic test protocol\n")
    (lifetime / "curves.json").write_text(json.dumps(curves))
    out = tmp_path / "output"
    out.mkdir()  # An existing empty output directory is valid.
    argv = ["analyze", "--curves", str(lifetime / "curves.json"), "--episodes", str(episodes_path),
            "--config", str(config_path), "--protocol", str(protocol_path), "--output", str(out),
            "--horizons", "4", "--alphas", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileNotFoundError, match="completion receipt"):
        analysis.main()
    receipts = {"config.json": {}, "freeze.json": {}, "parity.json": {"passed": True},
                "collection.json": {"status": "incomplete"}}
    for name, value in receipts.items():
        (lifetime / name).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="not marked completed"):
        analysis.main()
    (lifetime / "collection.json").write_text(json.dumps({"status": "completed"}))
    analysis.main()
    result = json.loads((out / "summary.json").read_text())
    assert result["source_sha256"]["analysis_script"] == hashlib.sha256(Path(analysis.__file__).read_bytes()).hexdigest()
    assert result["source_sha256"]["lifetime_protocol"] == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    assert result["selection_performed"] is False
