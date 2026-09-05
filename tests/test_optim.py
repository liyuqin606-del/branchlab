import copy

import pytest
import torch

from branchlab.optim import AdamW


def test_multiple_steps_match_torch_adamw_with_parameter_groups():
    torch.manual_seed(91)
    ours = [torch.nn.Parameter(torch.randn(4, 3, dtype=torch.float64)),
            torch.nn.Parameter(torch.randn(3, dtype=torch.float64))]
    reference = [torch.nn.Parameter(p.detach().clone()) for p in ours]
    groups = lambda ps: [{"params": [ps[0]], "lr": 0.003, "weight_decay": 0.08},
                         {"params": [ps[1]], "lr": 0.01, "weight_decay": 0.0}]
    opt = AdamW(groups(ours), betas=(0.83, 0.96), eps=1e-7)
    ref = torch.optim.AdamW(groups(reference), betas=(0.83, 0.96), eps=1e-7, foreach=False)
    for step in range(12):
        for index, (parameter, target) in enumerate(zip(ours, reference)):
            grad = torch.randn_like(parameter) if not (step == 3 and index == 1) else None
            parameter.grad = grad
            target.grad = None if grad is None else grad.clone()
        opt.step()
        ref.step()
        for parameter, target in zip(ours, reference):
            torch.testing.assert_close(parameter, target, atol=2e-14, rtol=2e-14)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(opt.state[parameter][key], ref.state[target][key])


def test_checkpoint_resume_matches_uninterrupted_updates():
    parameter = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    opt = AdamW([parameter], lr=0.01)
    for _ in range(4):
        parameter.grad = parameter.detach().square() / 10
        opt.step()
    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = AdamW([resumed_parameter], lr=0.5)
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    for _ in range(5):
        parameter.grad = parameter.detach().square() / 10
        resumed_parameter.grad = resumed_parameter.detach().square() / 10
        opt.step()
        resumed.step()
    torch.testing.assert_close(parameter, resumed_parameter, atol=0, rtol=0)
    assert resumed.param_groups[0]["lr"] == 0.01


def test_closure_computes_gradients_and_missing_gradient_skips_decay():
    active = torch.nn.Parameter(torch.tensor([2.0]))
    unused = torch.nn.Parameter(torch.tensor([3.0]))
    opt = AdamW([active, unused], weight_decay=0.2)

    def closure():
        opt.zero_grad()
        loss = active.square().sum()
        loss.backward()
        return loss

    assert opt.step(closure).item() == 4.0
    assert active.item() < 2.0
    assert unused.item() == 3.0
    assert unused not in opt.state


def test_sparse_gradient_rejected():
    embedding = torch.nn.Embedding(10, 4, sparse=True)
    embedding(torch.tensor([1, 2])).sum().backward()
    with pytest.raises(RuntimeError, match="sparse"):
        AdamW(embedding.parameters()).step()
