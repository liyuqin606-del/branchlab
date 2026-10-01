"""The evidence checker must reject plausible-looking but incorrect summaries."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from branchlab.model import ModelConfig, Transformer
from branchlab.systems import benchmark_paths


def verifier():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_portfolio.py"
    spec = importlib.util.spec_from_file_location("portfolio_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stored_timing_summary_is_independently_recomputed():
    model = Transformer(
        ModelConfig(
            vocab_size=43,
            d_model=16,
            n_heads=2,
            n_kv_heads=1,
            n_layers=1,
            max_seq_len=16,
        )
    )
    record = benchmark_paths(
        model, torch.tensor([[1, 3, 7]]), decode_tokens=3, repeats=2
    )
    checker = verifier()
    assert checker.check_benchmark(record)
    wrong = deepcopy(record)
    wrong["paths"]["static"]["decode_tokens_per_second"] *= 2
    with pytest.raises(AssertionError):
        checker.check_benchmark(wrong)
    wrong = deepcopy(record)
    wrong["paths"]["dynamic"]["parity"]["mean_distribution_kl"] = 1.0
    with pytest.raises(AssertionError):
        checker.check_benchmark(wrong)


def test_graph_speedup_tampering_is_rejected():
    record = {
        "metadata": {"batch_size": 1, "decode_tokens": 10, "repeats": 3},
        "decode_seconds_samples": {"dynamic": [2, 3, 4], "cuda_graph": [1, 1.5, 2]},
        "decode_tokens_per_second": {"dynamic": 10 / 3, "cuda_graph": 10 / 1.5},
        "speedup_vs_dynamic": 2,
        "capture_and_warmup_seconds": 0.5,
        "parity": {"mean_distribution_kl": 0, "kl_tolerance": 0.002},
    }
    assert verifier().check_graph(record)
    record["speedup_vs_dynamic"] = 3
    with pytest.raises(AssertionError):
        verifier().check_graph(record)
