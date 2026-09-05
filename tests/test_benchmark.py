import json

import pytest
import torch

from branchlab.benchmark import benchmark_kv
from branchlab.model import ModelConfig, TransformerLM


def test_kv_benchmark_compares_fixed_tokens_without_mutating_model_or_rng():
    torch.manual_seed(4)
    model = TransformerLM(ModelConfig(vocab_size=43, d_model=16, n_layers=1, n_heads=2, max_seq_len=32))
    prompt = torch.tensor([[1, 3, 7, 2], [4, 9, 1, 8]])
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = benchmark_kv(model, prompt, generated_tokens=4, repeats=2)
    assert result["logits_max_diff"] < 2e-6
    assert result["prefill_seconds"] > 0
    assert result["cached_decode_tokens_per_second"] > 0
    assert result["uncached_decode_tokens_per_second"] > 0
    assert len(result["cached_decode_seconds_samples"]) == 2
    assert result["metadata"]["timed_tokens_per_trial"] == 8
    assert result["metadata"]["batch_size"] == 2
    assert not result["metadata"]["prefill_in_decode_timing"]
    assert result["peak_memory"]["cached_bytes"] is None
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.training
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    # Published results must serialize without NaN/Infinity or tensor objects.
    json.dumps(result, allow_nan=False)


def test_invalid_benchmark_budget_rejected():
    model = TransformerLM(ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, max_seq_len=8))
    prompt = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="positive integer"):
        benchmark_kv(model, prompt, generated_tokens=0)
    with pytest.raises(ValueError, match="max_seq_len"):
        benchmark_kv(model, prompt, generated_tokens=6)
