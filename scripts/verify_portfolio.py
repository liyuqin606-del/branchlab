#!/usr/bin/env python3
"""Independently check raw portfolio records, budgets, weights and timings.

This verifies stored results; it does not re-run training or language evaluation.
Use branchlab.fit.evaluate_tokens to independently replay final model test loss.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from branchlab.fit import FitConfig
from branchlab.model import ModelConfig, Transformer
from branchlab.portfolio_data import file_sha256


def check_benchmark(benchmark):
    metadata = benchmark["metadata"]
    count = metadata["batch_size"] * metadata["decode_tokens"]
    for mode, row in benchmark["paths"].items():
        assert len(row["decode_seconds_samples"]) == metadata["repeats"]
        assert all(math.isfinite(t) and t > 0 for t in row["decode_seconds_samples"])
        median = statistics.median(row["decode_seconds_samples"])
        assert math.isclose(row["decode_median_seconds"], median, rel_tol=1e-12)
        assert math.isclose(
            row["decode_tokens_per_second"], count / median, rel_tol=1e-12
        )
        reference = statistics.median(
            benchmark["paths"]["uncached"]["decode_seconds_samples"]
        )
        assert math.isclose(
            row["speedup_vs_uncached"], reference / median, rel_tol=1e-12
        )
        if mode != "uncached":
            assert (
                row["parity"]["mean_distribution_kl"] <= row["parity"]["kl_tolerance"]
            )
    return True


def check_graph(record):
    meta = record["metadata"]
    count = meta["batch_size"] * meta["decode_tokens"]
    medians = {}
    for mode in ("dynamic", "cuda_graph"):
        values = record["decode_seconds_samples"][mode]
        assert len(values) == meta["repeats"]
        assert all(math.isfinite(t) and t > 0 for t in values)
        medians[mode] = statistics.median(values)
        assert math.isclose(
            record["decode_tokens_per_second"][mode],
            count / medians[mode],
            rel_tol=1e-12,
        )
    assert math.isclose(
        record["speedup_vs_dynamic"],
        medians["dynamic"] / medians["cuda_graph"],
        rel_tol=1e-12,
    )
    assert (
        0
        <= record["parity"]["mean_distribution_kl"]
        <= record["parity"]["kl_tolerance"]
    )
    assert (
        math.isfinite(record["capture_and_warmup_seconds"])
        and record["capture_and_warmup_seconds"] > 0
    )
    return True


def verify(root, prepared=None):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    completion = json.loads((root / "completion.json").read_text())
    assert completion["status"] == "completed"
    assert set(completion["runs"]) == set(protocol["runs"])
    assert completion["protocol_sha256"] == file_sha256(root / "protocol.json")
    source_checked = all(Path(name).exists() for name in protocol["source_hashes"])
    if source_checked:
        for filename, expected in protocol["source_hashes"].items():
            assert file_sha256(filename) == expected, (
                f"Frozen source mismatch: {filename}"
            )
    counts = {
        "runs": 0,
        "benchmarks": 0,
        "inference_parameters": 0,
        "training_tokens": 0,
    }
    for name, planned in protocol["runs"].items():
        path = root / name
        run = json.loads((path / "run.json").read_text())
        config = FitConfig(**planned)
        assert run["status"] == "completed" and run["config"] == planned
        assert run["trained_tokens"] == config.updates * config.tokens_per_update
        assert run["test"]["target_tokens"] == config.eval_tokens
        assert (
            math.isfinite(run["test"]["cross_entropy"])
            and run["test"]["cross_entropy"] > 0
        )
        assert math.isclose(
            run["test"]["perplexity"],
            math.exp(run["test"]["cross_entropy"]),
            rel_tol=1e-12,
        )
        for filename, expected in run["artifacts"].items():
            assert file_sha256(path / filename) == expected, (
                f"{name}/{filename}: hash mismatch"
            )
        history = json.loads((path / "history.json").read_text())
        assert history[0]["update"] == 0 and history[-1]["update"] == config.updates
        assert history[-1]["dev"] == run["final_dev"]
        assert all(b["update"] > a["update"] for a, b in zip(history, history[1:]))
        assert run["source_hashes"] == {
            Path(k).name: v for k, v in protocol["source_hashes"].items()
        }
        assert run["provenance"]["manifest.json"] == protocol["data_manifest_sha256"]
        if prepared:
            for filename, expected in run["provenance"].items():
                assert file_sha256(Path(prepared) / filename) == expected
        artifact = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model = Transformer(ModelConfig(**artifact["config"]))
        model.load_state_dict(artifact["model"])
        assert model.num_parameters() == run["parameters"]
        assert all(torch.isfinite(p).all() for p in model.parameters())
        counts["inference_parameters"] += model.num_parameters()
        counts["training_tokens"] += run["trained_tokens"]
        counts["runs"] += 1
        for benchmark_path in path.glob("benchmark-*.json"):
            benchmark = json.loads(benchmark_path.read_text())
            check_benchmark(benchmark)
            assert benchmark["metadata"]["model_sha256"] == run["artifacts"]["model.pt"]
            meta = benchmark["metadata"]
            element_size = (
                2 if meta["dtype"] in ("torch.bfloat16", "torch.float16") else 4
            )
            capacity = meta["prompt_tokens"] + meta["decode_tokens"]
            expected_bytes = (
                2
                * model.config.n_layers
                * meta["batch_size"]
                * model.config.kv_heads
                * capacity
                * (model.config.d_model // model.config.n_heads)
                * element_size
            )
            assert (
                benchmark["paths"]["static"]["cache_allocated_bytes"] == expected_bytes
            )
            assert (
                benchmark["paths"]["dynamic"]["cache_allocated_bytes"] == expected_bytes
            )
            counts["benchmarks"] += 1
        del model, artifact
    graph_dir = root / "graph-inference"
    if graph_dir.exists():
        graph_protocol = json.loads((graph_dir / "graph_protocol.json").read_text())
        assert graph_protocol["model_sha256"] == file_sha256(
            root / "showcase-gqa/model.pt"
        )
        assert graph_protocol["script_sha256"] == file_sha256(
            Path(__file__).with_name("cuda_graph_decode.py")
        )
        counts["graph_benchmarks"] = 0
        for spec in graph_protocol["grid"]:
            record = json.loads(
                (
                    graph_dir
                    / f"graph-b{spec['batch_size']}-p{spec['prompt_tokens']}.json"
                ).read_text()
            )
            check_graph(record)
            for key, value in spec.items():
                assert record["metadata"][key] == value
            assert record["metadata"]["model_sha256"] == graph_protocol["model_sha256"]
            assert (
                record["metadata"]["script_sha256"] == graph_protocol["script_sha256"]
            )
            counts["graph_benchmarks"] += 1
    replay_path = root / "evaluation_replay.json"
    if replay_path.exists():
        replays = json.loads(replay_path.read_text())
        assert set(replays) == set(protocol["runs"])
        for name, row in replays.items():
            run = json.loads((root / name / "run.json").read_text())
            assert row["model_sha256"] == run["artifacts"]["model.pt"]
            assert row["recorded_cross_entropy"] == run["test"]["cross_entropy"]
            difference = abs(
                row["replayed_cross_entropy"] - row["recorded_cross_entropy"]
            )
            assert math.isclose(difference, row["absolute_difference"], abs_tol=1e-15)
            assert difference <= row["tolerance"]
        counts["evaluation_replays"] = len(replays)
    return {
        "status": "verified_stored_records",
        "frozen_source_matches_current_checkout": source_checked,
        **counts,
        "scope": "Hashes, budgets, finite weights, parameter counts and independently recomputed timing statistics; no training rerun",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="artifacts/portfolio")
    parser.add_argument("--prepared")
    args = parser.parse_args()
    print(json.dumps(verify(args.results, args.prepared), indent=2))
