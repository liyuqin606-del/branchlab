"""Independent numerical checks for GQA, fused attention and mutable caches."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from branchlab.model import CausalSelfAttention, ModelConfig, Transformer


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_manual_and_sdpa_outputs_and_gradients(kv_heads):
    torch.manual_seed(81)
    config = ModelConfig(
        vocab_size=43,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=kv_heads,
        max_seq_len=32,
    )
    manual = Transformer(config).double()
    fused = Transformer(replace(config, attention_backend="sdpa")).double()
    fused.load_state_dict(manual.state_dict())
    tokens = torch.randint(0, 43, (2, 13))
    a, _ = manual(tokens)
    b, _ = fused(tokens)
    torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-8)
    a.square().mean().backward()
    b.square().mean().backward()
    for (name, p), (_, q) in zip(manual.named_parameters(), fused.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-9, rtol=1e-8, msg=name)


def test_gqa_matches_independent_per_query_head_reference():
    torch.manual_seed(2)
    config = ModelConfig(d_model=32, n_heads=4, n_kv_heads=2)
    attention = CausalSelfAttention(config).double()
    x = torch.randn(2, 7, 32, dtype=torch.double)
    actual, cache = attention(x, use_cache=True)
    packed = attention.qkv(x)
    q, k, v = packed.split((32, 16, 16), -1)
    q = attention.rope(q.reshape(2, 7, 4, 8).transpose(1, 2))
    k = attention.rope(k.reshape(2, 7, 2, 8).transpose(1, 2))
    v = v.reshape(2, 7, 2, 8).transpose(1, 2)
    # Independent reference loops over query heads; no repeat_interleave.
    heads = [
        F.scaled_dot_product_attention(
            q[:, h], k[:, h // 2], v[:, h // 2], is_causal=True
        )
        for h in range(4)
    ]
    expected = attention.out_proj(torch.stack(heads, dim=2).reshape(2, 7, 32))
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-8)
    assert cache[0].shape == (2, 2, 7, 8)


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_static_cache_chunk_parity_allocation_and_no_overflow_mutation(
    backend, kv_heads
):
    torch.manual_seed(5)
    model = Transformer(
        ModelConfig(
            vocab_size=43,
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=kv_heads,
            max_seq_len=16,
            attention_backend=backend,
        )
    ).eval()
    tokens = torch.randint(0, 43, (2, 11))
    cache = model.allocate_cache(2, capacity=11)
    pointers = [c.key.data_ptr() for c in cache]
    with torch.inference_mode():
        full, _ = model(tokens)
        a, cache = model(tokens[:, :4], cache, use_cache=True)
        b, cache = model(tokens[:, 4:9], cache, use_cache=True)
        c, cache = model(tokens[:, 9:], cache, use_cache=True)
        torch.testing.assert_close(torch.cat([a, b, c], 1), full, atol=2e-6, rtol=2e-5)
        before = [c.key.clone() for c in cache]
        with pytest.raises(ValueError, match="capacity"):
            model(tokens[:, :1], cache, use_cache=True)
        assert [c.length for c in cache] == [11, 11]
        assert pointers == [c.key.data_ptr() for c in cache]
        assert all(torch.equal(c.key, saved) for c, saved in zip(cache, before))
    assert sum(c.allocated_bytes for c in cache) == 2 * 2 * 2 * kv_heads * 11 * 8 * 4
    with pytest.raises(ValueError, match="inference-only"):
        model(tokens[:, :1], model.allocate_cache(2))


def test_greedy_generation_static_dynamic_uncached_and_seeded_sampling():
    torch.manual_seed(17)
    model = Transformer(
        ModelConfig(
            vocab_size=43,
            d_model=32,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
        )
    ).train()
    prompt = torch.tensor([[1, 3, 5]])
    expected = model.generate(prompt, 10, use_cache=False)
    assert torch.equal(expected, model.generate(prompt, 10))
    assert torch.equal(expected, model.generate(prompt, 10, cache_mode="static"))
    a = model.generate(
        prompt, 10, temperature=0.8, top_k=8, generator=torch.Generator().manual_seed(9)
    )
    b = model.generate(
        prompt, 10, temperature=0.8, top_k=8, generator=torch.Generator().manual_seed(9)
    )
    assert torch.equal(a, b)
    assert model.training


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_kv_heads": 3},
        {"n_kv_heads": 0},
        {"n_kv_heads": True},
        {"attention_backend": "magic"},
    ],
)
def test_invalid_attention_configuration(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(d_model=32, n_heads=4, **kwargs)
