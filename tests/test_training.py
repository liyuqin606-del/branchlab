import copy
import random
import numpy as np
import pytest
import torch

from branchlab.model import ModelConfig, TransformerLM
from branchlab.optim import AdamW
from branchlab.training import TokenStream, capture_state, restore_state, seed_all, step, apply_action, evaluate_loss, synchronize


def test_complete_checkpoint_replays_next_batches_and_updates():
    seed_all(4)
    model = TransformerLM(ModelConfig(vocab_size=32,d_model=16,n_layers=1,n_heads=2,max_seq_len=16))
    opt = AdamW(model.parameters(), lr=0.002)
    stream = TokenStream(np.arange(300)%32,2,8)
    step(model,opt,stream,"cpu")
    snapshot = capture_state(model,opt,stream,1,{"counter":1})
    first = [step(model,opt,stream,"cpu") for _ in range(3)]
    expected = copy.deepcopy(model.state_dict())
    number, scheduler = restore_state(snapshot,model,opt,stream)
    assert number == 1 and scheduler == {"counter":1}
    second = [step(model,opt,stream,"cpu") for _ in range(3)]
    assert first == second
    assert all(torch.equal(v,expected[k]) for k,v in model.state_dict().items())
    wrong = TokenStream(np.arange(300)[::-1].copy()%32,2,8)
    with pytest.raises(ValueError):
        wrong.load_state_dict(snapshot["stream"])


def test_moment_action_does_not_edit_weights_or_second_moment():
    p = torch.nn.Parameter(torch.tensor([1.,2.]))
    opt = AdamW([p],lr=0.01)
    p.grad = torch.tensor([.2,.3])
    opt.step()
    weights, second = p.detach().clone(),opt.state[p]["exp_avg_sq"].clone()
    apply_action(opt,"momentum_zero")
    assert torch.equal(weights,p)
    assert torch.equal(second,opt.state[p]["exp_avg_sq"])
    assert torch.count_nonzero(opt.state[p]["exp_avg"]) == 0


def make_training_state():
    seed_all(7)
    model = TransformerLM(ModelConfig(vocab_size=16, d_model=16, n_layers=1, n_heads=2, max_seq_len=16))
    optimizer = AdamW(model.parameters(), lr=0.003)
    stream = TokenStream(np.arange(137) % 16, 2, 8)
    return model, optimizer, stream


def test_disk_checkpoint_restores_rng_optimizer_cursor_and_live_scheduler(tmp_path):
    model, optimizer, stream = make_training_state()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    for _ in range(2):
        step(model, optimizer, stream, "cpu")
        scheduler.step()
    snapshot = capture_state(model, optimizer, stream, 2, scheduler)
    path = tmp_path / "checkpoint.pt"
    torch.save(snapshot, path)
    next_random = (random.random(), np.random.random(), torch.rand(4))
    expected_metrics = step(model, optimizer, stream, "cpu")
    scheduler.step()
    expected = copy.deepcopy(model.state_dict())
    expected_lr = optimizer.param_groups[0]["lr"]
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    restored_step, saved_scheduler = restore_state(loaded, model, optimizer, stream, scheduler)
    assert restored_step == 2
    assert saved_scheduler["last_epoch"] == 2
    assert all(p.grad is None for p in model.parameters())
    actual_random = (random.random(), np.random.random(), torch.rand(4))
    assert actual_random[:2] == next_random[:2]
    assert torch.equal(actual_random[2], next_random[2])
    assert step(model, optimizer, stream, "cpu") == expected_metrics
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == expected_lr
    assert all(torch.equal(value, expected[key]) for key, value in model.state_dict().items())


def test_cpu_checkpoint_does_not_touch_accelerator_rng(monkeypatch):
    model, optimizer, stream = make_training_state()

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU branching must not read or change an accelerator RNG")

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "get_rng_state", forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state", forbidden)
    monkeypatch.setattr(torch.mps, "get_rng_state", forbidden)
    monkeypatch.setattr(torch.mps, "set_rng_state", forbidden)
    snapshot = capture_state(model, optimizer, stream)
    assert "cuda" not in snapshot["rng"] and "mps" not in snapshot["rng"]
    # Loading an accelerator-origin checkpoint onto CPU must stay local too.
    snapshot["rng"]["cuda"] = {0: torch.zeros(3, dtype=torch.uint8)}
    snapshot["rng"]["mps"] = torch.zeros(3, dtype=torch.uint8)
    restore_state(snapshot, model, optimizer, stream)


def test_mismatched_data_rejected_before_model_mutation():
    model, optimizer, stream = make_training_state()
    snapshot = capture_state(model, optimizer, stream)
    step(model, optimizer, stream, "cpu")
    existing = copy.deepcopy(model.state_dict())
    wrong_stream = TokenStream((np.arange(137) + 1) % 16, 2, 8)
    with pytest.raises(ValueError, match="tokens_sha256"):
        restore_state(snapshot, model, optimizer, wrong_stream)
    assert all(torch.equal(value, existing[key]) for key, value in model.state_dict().items())


def test_token_stream_wrap_target_alignment_and_owned_source():
    source = np.arange(9)
    stream = TokenStream(source, batch_size=2, seq_len=4, cursor=6)
    source[:] = 99
    x, y = stream.batch("cpu")
    assert x.tolist() == [[6, 7, 8, 0], [1, 2, 3, 4]]
    assert y.tolist() == [[7, 8, 0, 1], [2, 3, 4, 5]]
    assert stream.cursor == 5
    with pytest.raises(ValueError, match="positive integers"):
        TokenStream(np.arange(30), batch_size=0)


def test_evaluation_restores_mode_when_evaluation_fails():
    model, _, _ = make_training_state()
    model.train()
    with pytest.raises(ValueError, match="at least one target"):
        evaluate_loss(model, [])
    assert model.training


def test_synchronize_recognizes_indexed_mps_and_cuda_devices(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("mps"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(str(device)))
    synchronize(torch.device("mps:0"))
    synchronize("mps")
    synchronize(torch.device("cuda:1"))
    synchronize("cpu")
    assert calls == ["mps", "mps", "cuda:1"]


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_accelerator_checkpoint_rng_replays_when_available(device, tmp_path):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} unavailable")
    model, optimizer, stream = make_training_state()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=0.003)
    step(model, optimizer, stream, device)
    assert optimizer.state
    snapshot = capture_state(model, optimizer, stream, step_number=1)
    path = tmp_path / "device.pt"
    torch.save(snapshot, path)
    expected = torch.rand(7, device=device).cpu()
    # Device map_location also moves RNG tensors, which restore must undo.
    loaded = torch.load(path, map_location=device, weights_only=False)
    restore_state(loaded, model, optimizer, stream)
    assert torch.equal(torch.rand(7, device=device).cpu(), expected)
