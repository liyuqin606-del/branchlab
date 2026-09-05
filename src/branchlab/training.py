"""Training and complete-state branching on CPU, CUDA, or Apple MPS."""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import ModelConfig, TransformerLM
from .optim import AdamW


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class TokenStream:
    """Sequential batches with a saved cursor; no random hidden sampler state."""
    def __init__(self, tokens, batch_size=8, seq_len=64, cursor=0):
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in (batch_size, seq_len)):
            raise ValueError("batch_size and seq_len must be positive integers")
        self.tokens = torch.as_tensor(tokens, dtype=torch.long, device="cpu").clone()
        if self.tokens.ndim != 1:
            raise ValueError("Token stream must be one-dimensional")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.cursor = cursor
        if len(self.tokens) < batch_size * seq_len + 1:
            raise ValueError("Token stream too short for one full batch")
        if not isinstance(cursor, int) or not 0 <= cursor < len(self.tokens):
            raise ValueError("cursor must index the token stream")

    def batch(self, device):
        width = self.batch_size * self.seq_len
        index = (torch.arange(width + 1) + self.cursor) % len(self.tokens)
        block = self.tokens[index]
        self.cursor = (self.cursor + width) % len(self.tokens)
        return block[:-1].reshape(self.batch_size, self.seq_len).to(device), block[1:].reshape(self.batch_size, self.seq_len).to(device)

    def state_dict(self):
        return {"cursor": self.cursor, "batch_size": self.batch_size, "seq_len": self.seq_len,
                "tokens_sha256": hashlib.sha256(self.tokens.numpy().tobytes()).hexdigest()}

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.cursor = state["cursor"]

    def validate_state_dict(self, state):
        """Check compatibility before restoring any part of a training branch."""
        expected = self.state_dict()
        for key in ("batch_size", "seq_len", "tokens_sha256"):
            if state[key] != expected[key]:
                raise ValueError(f"Stream resume mismatch: {key}")
        if not isinstance(state["cursor"], int) or not 0 <= state["cursor"] < len(self.tokens):
            raise ValueError("Stream resume mismatch: cursor")


def pack_texts(texts, tokenizer):
    return np.asarray([token for text in texts for token in (*tokenizer.encode(text), tokenizer.eos_id)], dtype=np.int64)


def evaluation_batches(tokens, batch_size=4, seq_len=64, count=2, device="cpu"):
    stream = TokenStream(tokens, batch_size, seq_len)
    return [stream.batch(device) for _ in range(count)]


@torch.no_grad()
def evaluate_loss(model, batches):
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for x, y in batches:
            logits, _ = model(x)
            total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"))
            count += y.numel()
        if count == 0:
            raise ValueError("Evaluation requires at least one target token")
        return total / count
    finally:
        model.train(was_training)


def step(model, optimizer, stream, device, max_grad_norm=1.0):
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    model.train()
    x, y = stream.batch(device)
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("Nonfinite gradient norm")
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": float(grad_norm), "tokens": y.numel()}


def capture_state(model, optimizer, stream, step_number=0, scheduler=None):
    """Snapshot an optimizer-step boundary, including the exact next input.

    Gradients are transient and cleared on restore.  RNG state is process-local;
    only the model's accelerator generator is touched, even if other devices are
    available.  A scheduler may be a metadata dictionary or a PyTorch scheduler.
    """
    schedule_state = scheduler.state_dict() if hasattr(scheduler, "state_dict") else scheduler
    state = {"format_version": 1, "model_config": asdict(model.config),
             "model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
             "stream": stream.state_dict(), "step": step_number, "scheduler": copy.deepcopy(schedule_state),
             "model_training": model.training,
             "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}}
    device = next(model.parameters()).device
    if device.type == "cuda":
        state["rng"]["cuda"] = {device.index: torch.cuda.get_rng_state(device)}
    if device.type == "mps":
        state["rng"]["mps"] = torch.mps.get_rng_state()
    return state


def restore_state(state, model, optimizer, stream, scheduler=None):
    """Restore a branch; pass a scheduler object to restore it in-place too.

    The returned schedule dictionary supports callers that apply schedules
    directly.  Byte RNG tensors are moved to CPU because ``map_location`` may
    have moved them with the checkpoint's model tensors during deserialization.
    """
    if state.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format_version")
    if state["model_config"] != asdict(model.config):
        raise ValueError("Checkpoint model configuration does not match the target model")
    stream.validate_state_dict(state["stream"])
    if scheduler is not None and state["scheduler"] is None:
        raise ValueError("Checkpoint contains no scheduler state")
    model.load_state_dict(state["model"])
    model.train(state.get("model_training", True))
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    optimizer.zero_grad(set_to_none=True)
    stream.load_state_dict(state["stream"])
    if scheduler is not None:
        scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"].cpu())
    device = next(model.parameters()).device
    if "cuda" in state["rng"] and device.type == "cuda":
        saved = state["rng"]["cuda"]
        # Accept legacy lists while restoring only this model's local device.
        if isinstance(saved, list):
            saved_rng = saved[device.index]
        else:
            saved_rng = saved.get(device.index)
            if saved_rng is None and len(saved) == 1:
                saved_rng = next(iter(saved.values()))
            if saved_rng is None:
                raise ValueError("Checkpoint has no RNG state for the target CUDA device")
        torch.cuda.set_rng_state(saved_rng.cpu(), device=device)
    if "mps" in state["rng"] and device.type == "mps":
        torch.mps.set_rng_state(state["rng"]["mps"].cpu())
    return state["step"], copy.deepcopy(state["scheduler"])


def apply_action(optimizer, action):
    if action == "keep":
        return
    if action == "lr_half":
        for group in optimizer.param_groups:
            group["lr"] *= 0.5
    elif action == "momentum_zero":
        for state in optimizer.state.values():
            if "exp_avg" in state:
                state["exp_avg"].zero_()
    else:
        raise ValueError(f"Unsupported action {action}")


def train_baseline(config, train_tokens, dev_tokens, *, device="auto", seed=0, steps=100,
                   batch_size=8, seq_len=64, lr=0.001, output_dir=None, eval_interval=25):
    seed_all(seed)
    device = resolve_device(device)
    model = TransformerLM(config).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    stream = TokenStream(train_tokens, batch_size, seq_len)
    evals = evaluation_batches(dev_tokens, batch_size, seq_len, count=2, device=device)
    initial_loss = evaluate_loss(model, evals)
    history = [{"step": 0, "dev_loss": initial_loss, "elapsed_seconds": 0.0, "tokens": 0}]
    synchronize(device)
    start = time.perf_counter()
    for t in range(steps):
        # Warmup then cosine schedule; scheduler state accompanies every checkpoint.
        warmup = max(1, min(10, steps // 10))
        factor = (t + 1) / warmup if t < warmup else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (t-warmup) / max(1, steps-warmup)))
        for group in optimizer.param_groups:
            group["lr"] = lr * factor
        record = step(model, optimizer, stream, device)
        if (t+1) % eval_interval == 0 or t+1 == steps:
            record["dev_loss"] = evaluate_loss(model, evals)
            synchronize(device)
            record.update(step=t+1, elapsed_seconds=time.perf_counter()-start,
                          tokens=(t+1)*batch_size*seq_len, lr=optimizer.param_groups[0]["lr"])
            history.append(record)
            print(json.dumps({"event": "train", "seed": seed, **record}), flush=True)
    # Diagnostic continuation starts at a declared fixed learning rate.
    for group in optimizer.param_groups:
        group["lr"] = lr
    state = capture_state(model, optimizer, stream, steps, {"name": "completed_cosine_then_fixed", "continuation_lr": lr})
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(state, out / "checkpoint.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2)+"\n")
    return model, optimizer, stream, state, history


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
