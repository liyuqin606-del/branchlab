"""Report integrity tests use small fabricated *fixtures*, never pilot evidence."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from branchlab.reporting import audit_series, build_report, render_report, render_resume


def fixture_summary():
    return {"status": "complete", "scope": "fixture only; equal update-budget proxy", "config": {},
            "episodes": 12, "audit_seeds": [7, 8],
            "collection": {"elapsed_seconds": 4.5, "training_updates": 64,
                           "evaluation_batches": 8, "trained_tokens": 1024},
            "methods": [{"name": "enumeration", "search_seed": None, "selected_probes": [],
                         "mean_test_loss": 2.25, "mean_regret": 0.1, "mean_probe_cost": 0,
                         "mean_repair_steps": 12, "search_revealed_cells": 12,
                         "search_replay_probe_cost": 48,
                         "per_seed": [{"seed": 7, "test_loss": 2.0, "regret": 0.05},
                                      {"seed": 8, "test_loss": 2.5, "regret": 0.15}],
                         "decisions": []}],
            "gate": {"status": "NOGO", "reason": "No demonstrated improvement in this fixture", "comparisons": []},
            "failures": []}


def test_missing_measurements_stay_missing_and_gate_is_preserved():
    text = render_report(fixture_summary())
    assert "**NOGO**" in text
    assert "2.250000" in text
    assert "未提供 showcase/run.json" in text
    assert "未提供 benchmark.json" in text
    assert "不宣称验证递归自我改进" in text
    assert "未提供" in render_report({})
    assert "暂不填写训练效果数字" in render_resume(fixture_summary())
    summary = fixture_summary()
    summary["methods"][0]["mean_test_loss"] = float("nan")
    with pytest.raises(ValueError, match="finite numeric"):
        render_report(summary)


def test_audit_macro_mean_and_search_seeds_stay_separate():
    summary = fixture_summary()
    item = dict(summary["methods"][0])
    item["name"], item["search_seed"] = "random", 1
    other = dict(item)
    other["search_seed"] = 2
    summary["methods"] = [item, other]
    series = audit_series(summary)
    assert len(series) == 2
    assert series[0]["mean"] == 2.25
    assert series[0]["seeds"] == [7, 8]
    assert series[0]["label"] != series[1]["label"]
    item["per_seed"] = [item["per_seed"][0], item["per_seed"][0]]
    with pytest.raises(ValueError, match="Duplicate"):
        audit_series(summary)


def test_build_report_copies_hashes_and_renders_actual_inputs(tmp_path):
    pilot, showcase, output = (tmp_path / n for n in ("pilot", "showcase", "output"))
    pilot.mkdir()
    showcase.mkdir()
    summary = fixture_summary()
    (pilot / "summary.json").write_text(json.dumps(summary))
    run = {"parameters": 1234, "trained_tokens": 1024, "initial_dev_loss": 3.0,
           "final_dev_loss": 2.1, "elapsed_seconds": 2.0, "device": "cpu", "torch_version": "fixture"}
    history = [{"step": 0, "dev_loss": 3.0}, {"step": 8, "dev_loss": 2.1}]
    benchmark = {"cached_decode_tokens_per_second": 12.0, "uncached_decode_tokens_per_second": 10.0,
                 "decode_speedup": 1.2, "prefill_seconds": 0.1, "logits_max_diff": 0.0,
                 "metadata": {"device": "cpu", "dtype": "float32", "batch_size": 1,
                              "prompt_tokens": 16, "decode_tokens_per_sequence": 8, "repeats": 3},
                 "peak_memory": {"available": False, "unavailable_reason": "fixture"}}
    for name, data in (("run.json", run), ("history.json", history), ("benchmark.json", benchmark)):
        (showcase / name).write_text(json.dumps(data))
    result = build_report(SimpleNamespace(pilot=pilot, showcase=showcase, output=output))
    assert Path(result["report"]).is_file()
    assert (output / "summary.json").read_bytes() == (pilot / "summary.json").read_bytes()
    manifest = json.loads((output / "report_manifest.json").read_text())
    for group in ("inputs", "generated"):
        for name, entry in manifest[group].items():
            content = (output / name).read_bytes()
            assert entry["sha256"] == hashlib.sha256(content).hexdigest()
            assert entry["bytes"] == len(content)
    for name in ("training_curve.png", "audit_loss.png"):
        assert (output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    text = (output / "REPORT.md").read_text()
    assert "1,234" in text and "2.100000" in text
    assert "并非置信区间" in text
    assert "NOGO" in (output / "resume_zh.md").read_text()
