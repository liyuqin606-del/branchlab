"""Shared-gradient checks establish update semantics, not diagnostic gains."""

import copy

import numpy as np
import pytest
import torch

from branchlab.lookahead import prepare_decision, materialize_action, peek_training_batch, peek_training_batches
from branchlab.model import ModelConfig, TransformerLM
from branchlab.optim import AdamW
from branchlab.training import TokenStream, seed_all, step, capture_state, restore_state, apply_action, evaluate_loss


def assert_nested_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


class DropoutTransformer(TransformerLM):
    """A stochastic fixture tests post-backward RNG restoration explicitly."""

    def forward(self, *args, **kwargs):
        logits, cache = super().forward(*args, **kwargs)
        return torch.nn.functional.dropout(logits, p=0.2, training=self.training), cache


def initialized(stochastic=False):
    seed_all(91)
    cls = DropoutTransformer if stochastic else TransformerLM
    model = cls(ModelConfig(vocab_size=37, d_model=16, n_layers=1, n_heads=2, max_seq_len=16))
    optimizer = AdamW(model.parameters(), lr=0.003, betas=(0.9, 0.95))
    stream = TokenStream((np.arange(213) * 7) % 37, batch_size=2, seq_len=8)
    for _ in range(3):
        step(model, optimizer, stream, "cpu")
    return model, optimizer, stream


@pytest.mark.parametrize("action", ["keep", "lr_half", "momentum_zero"])
@pytest.mark.parametrize("stochastic", [False, True])
def test_shared_gradient_matches_four_regular_updates_exactly(action, stochastic):
    model, optimizer, stream = initialized(stochastic)
    scheduler = {"name": "fixed", "continuation_lr": 0.003}
    prepared = prepare_decision(model, optimizer, stream, "cpu", step_number=3, scheduler=scheduler)
    assert prepared.origin["step"] == prepared.post_gradient_state["step"] == 3
    assert prepared.post_batch_stream["cursor"] == (prepared.origin["stream"]["cursor"] + 16) % len(stream.tokens)
    assert all(int(state["step"].item()) == 3 for state in optimizer.state.values())

    restore_state(prepared.origin, model, optimizer, stream)
    apply_action(optimizer, action)
    expected_metrics = [step(model, optimizer, stream, "cpu") for _ in range(4)]
    expected = capture_state(model, optimizer, stream, 7, scheduler)

    forward_calls = []
    hook = model.register_forward_pre_hook(lambda *_: forward_calls.append(1))
    candidate = materialize_action(prepared, model, optimizer, stream, action)
    hook.remove()
    assert forward_calls == []
    assert candidate["step"] == 4
    assert candidate["stream"] == prepared.post_batch_stream
    assert all(int(state["step"].item()) == 4 for state in optimizer.state.values())
    actual_metrics = [prepared.metrics] + [step(model, optimizer, stream, "cpu") for _ in range(3)]
    actual = capture_state(model, optimizer, stream, 7, scheduler)
    assert actual_metrics == expected_metrics
    assert_nested_equal(actual, expected)


def test_candidate_materialization_is_order_independent_and_does_not_mutate_prepared():
    model, optimizer, stream = initialized()
    prepared = prepare_decision(model, optimizer, stream, "cpu")
    origin_before = copy.deepcopy(prepared.origin)
    grads_before = [g.clone() if g is not None else None for g in prepared.gradients]
    first = materialize_action(prepared, model, optimizer, stream, "keep")
    materialize_action(prepared, model, optimizer, stream, "momentum_zero")
    second = materialize_action(prepared, model, optimizer, stream, "keep")
    assert_nested_equal(first, second)
    assert_nested_equal(prepared.origin, origin_before)
    assert_nested_equal(list(prepared.gradients), grads_before)


def test_probe_batches_and_forward_losses_do_not_advance_optimizer_or_cursor():
    model, optimizer, stream = initialized()
    prepared = prepare_decision(model, optimizer, stream, "cpu")
    candidate = materialize_action(prepared, model, optimizer, stream, "lr_half")
    before = capture_state(model, optimizer, stream, candidate["step"], candidate["scheduler"])
    batches = peek_training_batches(stream, "cpu", offsets=(1, 2, 4, 8))
    reference_stream = TokenStream(stream.tokens, stream.batch_size, stream.seq_len, stream.cursor)
    future = [reference_stream.batch("cpu") for _ in range(8)]
    for actual, index in zip(batches, (0, 1, 3, 7)):
        assert all(torch.equal(a, b) for a, b in zip(actual, future[index]))
    assert np.isfinite(evaluate_loss(model, batches))
    after = capture_state(model, optimizer, stream, candidate["step"], candidate["scheduler"])
    assert_nested_equal(before, after)
    # A frozen candidate remains committable after arbitrary probe evaluations.
    restore_state(candidate, model, optimizer, stream)
    assert stream.cursor == candidate["stream"]["cursor"]
    with pytest.raises(ValueError, match="positive integer"):
        peek_training_batch(stream, "cpu", offset=0)


def test_model_or_stream_mismatch_fails_before_any_mutation():
    model, optimizer, stream = initialized()
    prepared = prepare_decision(model, optimizer, stream, "cpu")
    wrong_stream = TokenStream((np.arange(213) * 5) % 37, batch_size=2, seq_len=8)
    before = capture_state(model, optimizer, wrong_stream)
    with pytest.raises(ValueError, match="tokens_sha256"):
        materialize_action(prepared, model, optimizer, wrong_stream, "keep")
    assert_nested_equal(before, capture_state(model, optimizer, wrong_stream))
    wrong_model = TransformerLM(ModelConfig(vocab_size=37, d_model=32, n_layers=1, n_heads=2, max_seq_len=16))
    wrong_optimizer = AdamW(wrong_model.parameters())
    before = capture_state(wrong_model, wrong_optimizer, stream)
    with pytest.raises(ValueError, match="configuration"):
        materialize_action(prepared, wrong_model, wrong_optimizer, stream, "keep")
    assert_nested_equal(before, capture_state(wrong_model, wrong_optimizer, stream))


def test_wrong_optimizer_or_action_rejected_before_mutation():
    model, optimizer, stream = initialized()
    prepared = prepare_decision(model, optimizer, stream, "cpu")
    before = capture_state(model, optimizer, stream)
    with pytest.raises(ValueError, match="Unsupported action"):
        materialize_action(prepared, model, optimizer, stream, "unknown")
    foreign = AdamW([torch.nn.Parameter(torch.ones(1))])
    with pytest.raises(ValueError, match="each model parameter"):
        materialize_action(prepared, model, foreign, stream, "keep")
    assert_nested_equal(before, capture_state(model, optimizer, stream))


def test_failed_prepare_restores_origin(monkeypatch):
    model, optimizer, stream = initialized()
    before = capture_state(model, optimizer, stream, step_number=3)
    original_forward = model.forward

    def nonfinite(*args, **kwargs):
        logits, cache = original_forward(*args, **kwargs)
        return logits * float("nan"), cache

    monkeypatch.setattr(model, "forward", nonfinite)
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        prepare_decision(model, optimizer, stream, "cpu")
    assert_nested_equal(before, capture_state(model, optimizer, stream, step_number=3))
