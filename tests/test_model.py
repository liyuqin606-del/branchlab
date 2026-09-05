import pytest
import torch
import torch.nn.functional as F

from branchlab.model import ModelConfig, Transformer


def make_model():
    torch.manual_seed(34)
    return Transformer(ModelConfig(vocab_size=61, d_model=32, n_layers=2, n_heads=4, max_seq_len=32))


def test_causal_logits_are_independent_of_future_tokens():
    model = make_model().eval()
    tokens = torch.randint(0, 61, (2, 11))
    altered = tokens.clone()
    altered[:, 5:] = (altered[:, 5:] + 17) % 61
    original, _ = model(tokens)
    changed, _ = model(altered)
    torch.testing.assert_close(original[:, :5], changed[:, :5], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(original[:, 5:], changed[:, 5:])


@pytest.mark.parametrize("chunks", [(3, 4, 4), (1,) * 11, (8, 3)])
def test_cache_equivalence_including_multiple_new_tokens(chunks):
    model = make_model().eval()
    tokens = torch.randint(0, 61, (2, 11))
    with torch.no_grad():
        full, none_cache = model(tokens)
        assert none_cache is None
        cache, position, parts = None, 0, []
        for length in chunks:
            logits, cache = model(tokens[:, position:position + length], cache=cache, use_cache=True)
            position += length
            parts.append(logits)
            assert len(cache) == 2
            assert cache[0][0].shape == (2, 4, position, 8)
        torch.testing.assert_close(torch.cat(parts, dim=1), full, atol=2e-6, rtol=2e-5)


def test_all_parameters_receive_finite_nonzero_gradients():
    model = make_model()
    tokens = torch.randint(0, 61, (3, 13))
    logits, _ = model(tokens[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, 61), tokens[:, 1:].reshape(-1))
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert model.num_parameters() == sum(p.numel() for p in model.parameters())
    assert model.lm_head.weight is model.embedding.weight


def test_cache_preserves_training_gradients():
    model = make_model()
    tokens = torch.randint(0, 61, (2, 9))
    expected, _ = model(tokens)
    expected.square().mean().backward()
    expected_grads = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    first, cache = model(tokens[:, :4], use_cache=True)
    last, _ = model(tokens[:, 4:], cache=cache)
    torch.cat((first, last), dim=1).square().mean().backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, expected_grads[name], atol=1e-6, rtol=2e-5)


def test_generation_matches_with_and_without_cache_and_restores_mode():
    model = make_model().train()
    tokens = torch.randint(0, 61, (2, 5))
    cached = model.generate(tokens, max_new_tokens=8)
    full = model.generate(tokens, max_new_tokens=8, use_cache=False)
    assert torch.equal(cached, full)
    assert model.training
    assert torch.equal(cached[:, :5], tokens)


def test_invalid_context_and_cache_fail_explicitly():
    model = make_model()
    with pytest.raises(ValueError, match="max_seq_len"):
        model(torch.zeros((1, 33), dtype=torch.long))
    with pytest.raises(ValueError, match="per layer"):
        model(torch.zeros((1, 3), dtype=torch.long), cache=[])
    with pytest.raises(ValueError, match="even head"):
        ModelConfig(d_model=12, n_heads=4)
