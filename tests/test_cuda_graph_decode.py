"""Fixed-shape inference must preserve the original model's causal semantics."""

import importlib.util
from pathlib import Path

import pytest
import torch

from branchlab.model import ModelConfig, Transformer

path = Path(__file__).resolve().parents[1] / "scripts/cuda_graph_decode.py"
spec = importlib.util.spec_from_file_location("graph_decode", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fixed_shape_cache_replay_and_reset_match_full_model(kv_heads, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(5)
    model = (
        Transformer(
            ModelConfig(
                vocab_size=43,
                d_model=32,
                n_layers=2,
                n_heads=4,
                n_kv_heads=kv_heads,
                max_seq_len=32,
                attention_backend="sdpa",
            )
        )
        .to(device)
        .eval()
    )
    decoder = module.FixedShapeDecoder(model, 2, 16, capture=device == "cuda")
    with torch.inference_mode():
        # A second prompt checks that no previous request's keys leak into output.
        for _ in range(2):
            tokens = torch.randint(0, 43, (2, 16), device=device)
            expected, _ = model(tokens)
            decoder.reset(tokens[:, :5])
            actual = [decoder.step(tokens[:, i : i + 1]).clone() for i in range(5, 16)]
            torch.testing.assert_close(
                torch.cat(actual, 1), expected[:, 5:], atol=3e-6, rtol=3e-5
            )
            with pytest.raises(ValueError, match="capacity"):
                decoder.step(tokens[:, :1])
