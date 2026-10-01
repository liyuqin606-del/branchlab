"""Share one real training gradient across candidate AdamW decisions.

Preparing a decision performs one ordinary training forward/backward and clips
its gradient.  Materializing candidates performs optimizer math only.  A caller
must charge that common training computation to every method, count all probe
forwards separately, and count the selected candidate as one committed update.

These pending-gradient objects extend, rather than change, v0.1 checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F

from .training import apply_action, capture_state, restore_state


@dataclass(frozen=True)
class PreparedDecision:
    """An origin checkpoint plus one completed but uncommitted gradient.

    ``post_gradient_state`` has the original parameters/optimizer state, but the
    cursor and RNG after the training forward/backward.  It also preserves any
    forward-updated buffers.  Gradient tensors are independent copies aligned
    with ``parameter_names``.  No optimizer step has occurred yet.
    """

    origin: dict
    post_gradient_state: dict
    post_batch_stream: dict
    parameter_names: tuple[str, ...]
    gradients: tuple[torch.Tensor | None, ...]
    loss: float
    grad_norm: float
    tokens: int

    @property
    def metrics(self) -> dict:
        """The same metrics as ``training.step``; grad_norm is before clipping."""
        return {"loss": self.loss, "grad_norm": self.grad_norm, "tokens": self.tokens}


def _validate_optimizer(model, optimizer):
    model_parameters = list(model.parameters())
    optimizer_parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if (len(optimizer_parameters) != len(model_parameters)
            or {id(p) for p in optimizer_parameters} != {id(p) for p in model_parameters}):
        raise ValueError("Optimizer must contain each model parameter exactly once")


def _inferred_step(optimizer):
    # This code targets the dense Transformer whose parameters update together.
    # Callers with an independent schedule/global counter should pass it.
    return max((int(state["step"].item()) if torch.is_tensor(state.get("step")) else int(state.get("step", 0))
                for state in optimizer.state.values()), default=0)


def prepare_decision(model, optimizer, stream, device, *, step_number=None,
                     scheduler=None, max_grad_norm=1.0):
    """Compute exactly one batch gradient without performing optimizer.step.

    All methods, including cheap-log and fresh-gradient baselines, may read the
    returned gradient.  The live model is left after backward, with its stream
    advanced once.  Candidate materialization always restores its own saved
    origin/post-gradient state and is independent of intervening probe work.
    """
    _validate_optimizer(model, optimizer)
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if step_number is None:
        step_number = _inferred_step(optimizer)
    if not isinstance(step_number, int) or isinstance(step_number, bool) or step_number < 0:
        raise ValueError("step_number must be a nonnegative integer")
    origin = capture_state(model, optimizer, stream, step_number, scheduler)
    try:
        model.train()
        x, y = stream.batch(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite decision training loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Nonfinite decision gradient norm")
        names, gradients = [], []
        for name, parameter in model.named_parameters():
            names.append(name)
            gradients.append(None if parameter.grad is None else parameter.grad.detach().clone())
        post_gradient_state = capture_state(model, optimizer, stream, step_number, scheduler)
        return PreparedDecision(origin=origin, post_gradient_state=post_gradient_state,
            post_batch_stream=stream.state_dict(), parameter_names=tuple(names), gradients=tuple(gradients),
            loss=float(loss.detach()), grad_norm=float(grad_norm), tokens=y.numel())
    except Exception:
        # Even a failed diagnostic must not strand the caller at a later cursor
        # or consume an unrecorded RNG transition in its actual training branch.
        restore_state(origin, model, optimizer, stream)
        raise


def _validate_prepared(prepared, model, optimizer, stream, action):
    if action not in ("keep", "lr_half", "momentum_zero"):
        raise ValueError(f"Unsupported action {action}")
    _validate_optimizer(model, optimizer)
    for snapshot in (prepared.origin, prepared.post_gradient_state):
        if snapshot.get("format_version") != 1:
            raise ValueError("Unsupported prepared checkpoint format")
        if snapshot["model_config"] != asdict(model.config):
            raise ValueError("Prepared model configuration does not match the target model")
        stream.validate_state_dict(snapshot["stream"])
    if prepared.post_batch_stream != prepared.post_gradient_state["stream"]:
        raise ValueError("Prepared post-batch cursor metadata is inconsistent")
    parameters = list(model.named_parameters())
    if tuple(name for name, _ in parameters) != prepared.parameter_names or len(parameters) != len(prepared.gradients):
        raise ValueError("Prepared gradients do not match target parameter names")
    for (_, parameter), gradient in zip(parameters, prepared.gradients):
        if gradient is not None and (gradient.shape != parameter.shape or gradient.dtype != parameter.dtype):
            raise ValueError("Prepared gradient shape/dtype does not match target parameter")


def materialize_action(prepared, model, optimizer, stream, action):
    """Return a full candidate checkpoint after exactly one committed update.

    This function makes no model forward/backward calls.  Post-gradient state
    restores origin weights and optimizer moments together with the correct
    post-batch RNG/cursor, then installs independent copies of the shared grads.
    All compatibility checks run before any model/optimizer/cursor mutation.
    The chosen returned state can later be committed with ``restore_state``.
    """
    _validate_prepared(prepared, model, optimizer, stream, action)
    restore_state(prepared.post_gradient_state, model, optimizer, stream)
    for parameter, gradient in zip(model.parameters(), prepared.gradients):
        parameter.grad = None if gradient is None else gradient.to(device=parameter.device).clone()
    apply_action(optimizer, action)
    optimizer.step()
    return capture_state(model, optimizer, stream, prepared.origin["step"] + 1,
                         prepared.post_gradient_state["scheduler"])


def peek_training_batch(stream, device, offset=1):
    """Read an upcoming batch without changing the committed stream or RNG.

    Offset 1 means the next *unconsumed* batch at the stream's current cursor;
    after candidate commitment this is the batch following the shared-gradient
    batch.  No optimizer state or held-out validation/test data is involved.
    """
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
        raise ValueError("offset must be a positive integer")
    width = stream.batch_size * stream.seq_len
    start = (stream.cursor + (offset - 1) * width) % len(stream.tokens)
    indices = (torch.arange(width + 1) + start) % len(stream.tokens)
    block = stream.tokens[indices]
    return (block[:-1].reshape(stream.batch_size, stream.seq_len).to(device),
            block[1:].reshape(stream.batch_size, stream.seq_len).to(device))


def peek_training_batches(stream, device, offsets=(1, 2, 4, 8)):
    """Return declared future training batches with no stream advancement."""
    return [peek_training_batch(stream, device, offset) for offset in offsets]
