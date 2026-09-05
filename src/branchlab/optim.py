"""AdamW with explicit moments, bias correction, and decoupled weight decay."""

from __future__ import annotations

import math
from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(self, params: Iterable[Tensor] | Iterable[dict], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01) -> None:
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and nonnegative")
        if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError("betas must both be in [0, 1)")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("BranchLab AdamW does not support sparse gradients")
                if parameter.is_complex():
                    raise RuntimeError("BranchLab AdamW supports real parameters only")
                state = self.state[parameter]
                if not state:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                state["step"] += 1
                step = int(state["step"].item())
                first_moment, second_moment = state["exp_avg"], state["exp_avg_sq"]
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                first_moment.lerp_(gradient, 1 - beta1)
                second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                correction1, correction2 = 1 - beta1 ** step, 1 - beta2 ** step
                denominator = second_moment.sqrt().div_(math.sqrt(correction2)).add_(group["eps"])
                parameter.addcdiv_(first_moment, denominator, value=-group["lr"] / correction1)
        return loss
