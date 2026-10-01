"""Resumable fixed-budget training for the portfolio experiment.

The research pilot retains its original trainer. This runner adds accumulation,
BF16 autocast, periodic atomic checkpoints, full fixed-window evaluation and
explicit provenance. Test loss is read only after the declared final update.
"""

from __future__ import annotations

import json
import math
import platform
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import ModelConfig, Transformer
from .optim import AdamW
from .portfolio_data import file_sha256
from .tokenizer import ByteBPETokenizer
from .training import (
    TokenStream,
    capture_state,
    json_write,
    resolve_device,
    restore_state,
    seed_all,
    synchronize,
)


@dataclass(frozen=True)
class FitConfig:
    """Budgets are optimizer updates; each update consumes micro_batch * accumulation * seq_len tokens."""

    model: dict
    seed: int = 42
    updates: int = 800
    micro_batch: int = 16
    accumulation: int = 4
    seq_len: int = 256
    lr: float = 0.001
    min_lr_ratio: float = 0.1
    warmup_updates: int = 50
    weight_decay: float = 0.1
    precision: str = "bf16"
    optimizer: str = "torch"
    eval_interval: int = 100
    checkpoint_interval: int = 100
    eval_tokens: int = 65536
    max_grad_norm: float = 1.0

    def __post_init__(self):
        for name in (
            "updates",
            "micro_batch",
            "accumulation",
            "seq_len",
            "eval_interval",
            "checkpoint_interval",
            "eval_tokens",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.warmup_updates < self.updates:
            raise ValueError("warmup_updates must be nonnegative and below updates")
        if self.precision not in ("fp32", "bf16") or self.optimizer not in (
            "torch",
            "educational",
        ):
            raise ValueError(
                "Supported precision: fp32/bf16; optimizer: torch/educational"
            )
        if (
            not math.isfinite(self.lr)
            or self.lr <= 0
            or not 0 <= self.min_lr_ratio <= 1
        ):
            raise ValueError("Invalid learning rate schedule")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")

    @property
    def tokens_per_update(self) -> int:
        return self.micro_batch * self.accumulation * self.seq_len


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    if device.type == "mps":
        raise ValueError("Portfolio BF16 path is tested on CUDA/CPU; use fp32 on MPS")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16; use fp32")
    return torch.autocast(device.type, dtype=torch.bfloat16)


def learning_rate(config: FitConfig, update_index: int) -> float:
    """The schedule uses the original budget even when a run is interrupted."""
    if update_index < config.warmup_updates:
        return config.lr * (update_index + 1) / config.warmup_updates
    progress = (update_index - config.warmup_updates) / max(
        1, config.updates - config.warmup_updates - 1
    )
    factor = (
        config.min_lr_ratio
        + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
    )
    return config.lr * factor


@torch.inference_mode()
def evaluate_tokens(model, tokens, config, device):
    """Evaluate exactly eval_tokens targets, without wrapping or overlapping windows.

    Input windows have seq_len independent contexts. A target at a boundary is
    predicted once, with its immediately preceding input in that window.
    """
    if len(tokens) <= config.eval_tokens:
        raise ValueError(
            "Evaluation split must contain more than eval_tokens tokens; never wrap it"
        )
    was_training = model.training
    model.eval()
    total = 0.0
    evaluated = 0
    try:
        width = config.micro_batch * config.seq_len
        while evaluated < config.eval_tokens:
            count = min(width, config.eval_tokens - evaluated)
            # Only full equal-size rows in a batch; a final partial row is legal.
            if count >= config.seq_len:
                count = (count // config.seq_len) * config.seq_len
                shape = (-1, config.seq_len)
            else:
                shape = (1, count)
            x = torch.as_tensor(
                np.array(tokens[evaluated : evaluated + count]), device=device
            ).reshape(shape)
            y = torch.as_tensor(
                np.array(tokens[evaluated + 1 : evaluated + count + 1]), device=device
            ).reshape(shape)
            with autocast_context(device, config.precision):
                logits, _ = model(x)
                loss_sum = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    reduction="sum",
                )
            total += float(loss_sum)
            evaluated += count
        return {
            "cross_entropy": total / evaluated,
            "perplexity": math.exp(total / evaluated),
            "target_tokens": evaluated,
            "context_length": config.seq_len,
        }
    finally:
        model.train(was_training)


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def train_update(model, optimizer, stream, config, device):
    """Accumulate mean losses, clip once, and apply exactly one AdamW update."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for _ in range(config.accumulation):
        x, y = stream.batch(device)
        with autocast_context(device, config.precision):
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1)
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        (loss / config.accumulation).backward()
        losses.append(float(loss.detach()))
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("Nonfinite gradient norm")
    optimizer.step()
    return {
        "train_loss": sum(losses) / len(losses),
        "grad_norm_before_clip": float(grad_norm),
    }


def fit(
    config: FitConfig, prepared, output, device="auto", resume=None, stop_after=None
):
    """Train or resume. A completed output is immutable; use a new directory."""
    prepared, out = Path(prepared), Path(output)
    if (out / "run.json").exists():
        raise FileExistsError("Completed output already exists; use a new directory")
    if out.exists() and any(out.iterdir()) and resume is None:
        raise FileExistsError("Partial output exists; explicitly resume its checkpoint")
    if stop_after is not None and (not isinstance(stop_after, int) or stop_after < 1):
        raise ValueError("stop_after must be a positive update count")
    tokenizer = ByteBPETokenizer.load(prepared / "tokenizer.json")
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, **config.model)
    if config.seq_len > model_config.max_seq_len:
        raise ValueError("Training seq_len exceeds model context")
    provenance = {
        name: file_sha256(prepared / name)
        for name in (
            "manifest.json",
            "tokenizer.json",
            "train_tokens.npy",
            "dev_tokens.npy",
            "test_tokens.npy",
        )
    }
    manifest = json.loads((prepared / "manifest.json").read_text())
    for split in ("train", "dev", "test"):
        if (
            provenance[f"{split}_tokens.npy"]
            != manifest["splits"][split]["tokens_sha256"]
        ):
            raise ValueError(f"Prepared {split} token array disagrees with manifest")
    if provenance["tokenizer.json"] != manifest["tokenizer_sha256"]:
        raise ValueError("Prepared tokenizer disagrees with manifest")
    tokens = {
        name: np.load(prepared / f"{name}_tokens.npy", mmap_mode="r")
        for name in ("train", "dev", "test")
    }
    if any(len(tokens[name]) <= config.eval_tokens for name in ("dev", "test")):
        raise ValueError(
            "Evaluation budget exceeds available split; no repeated targets allowed"
        )
    device = resolve_device(device)
    seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Transformer(model_config).to(device)
    optimizer_class = torch.optim.AdamW if config.optimizer == "torch" else AdamW
    optimizer_options = (
        {"fused": True} if config.optimizer == "torch" and device.type == "cuda" else {}
    )
    optimizer = optimizer_class(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
        **optimizer_options,
    )
    stream = TokenStream(tokens["train"], config.micro_batch, config.seq_len)
    source_hashes = {
        path.name: file_sha256(path)
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    completed, history, segments, training_seconds = 0, [], [], 0.0
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=False)
        if (
            saved.get("fit_config") != asdict(config)
            or saved.get("provenance") != provenance
            or saved.get("source_hashes") != source_hashes
        ):
            raise ValueError("Resume config/data/tokenizer mismatch")
        completed, _ = restore_state(saved, model, optimizer, stream)
        history, segments = saved["history"], saved["segments"]
        training_seconds = saved["training_seconds"]
    else:
        initial = evaluate_tokens(model, tokens["dev"], config, device)
        history.append({"update": 0, "trained_tokens": 0, "dev": initial})
    out.mkdir(parents=True, exist_ok=True)
    json_write(out / "config.json", asdict(config))
    json_write(out / "data_manifest.json", manifest)
    source_hashes = {
        path.name: file_sha256(path)
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    stop = min(config.updates, completed + stop_after) if stop_after else config.updates
    segment_start_update, segment_started = completed, time.perf_counter()

    def save_checkpoint():
        state = capture_state(
            model,
            optimizer,
            stream,
            completed,
            {"name": "warmup_cosine", "next_update": completed},
        )
        state.update(
            fit_config=asdict(config),
            provenance=provenance,
            history=history,
            segments=segments
            + [
                {
                    "from_update": segment_start_update,
                    "to_update": completed,
                    "wall_seconds": time.perf_counter() - segment_started,
                }
            ],
            training_seconds=training_seconds,
            source_hashes=source_hashes,
        )
        atomic_torch_save(state, out / "checkpoint.pt")
        json_write(out / "history.json", history)
        json_write(
            out / "progress.json",
            {
                "status": "interrupted" if completed < config.updates else "trained",
                "completed_updates": completed,
                "planned_updates": config.updates,
            },
        )

    for index in range(completed, stop):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(config, index)
        synchronize(device)
        started = time.perf_counter()
        metrics = train_update(model, optimizer, stream, config, device)
        synchronize(device)
        training_seconds += time.perf_counter() - started
        completed = index + 1
        if completed % config.eval_interval == 0 or completed == config.updates:
            metrics.update(
                update=completed,
                trained_tokens=completed * config.tokens_per_update,
                lr=optimizer.param_groups[0]["lr"],
                dev=evaluate_tokens(model, tokens["dev"], config, device),
            )
            history.append(metrics)
            print(
                json.dumps({"event": "fit", "seed": config.seed, **metrics}), flush=True
            )
        if completed % config.checkpoint_interval == 0 or completed == stop:
            save_checkpoint()
    if completed < config.updates:
        return {
            "status": "interrupted",
            "completed_updates": completed,
            "checkpoint": str(out / "checkpoint.pt"),
        }
    test = evaluate_tokens(model, tokens["test"], config, device)
    atomic_torch_save(
        {
            "config": asdict(model_config),
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
        },
        out / "model.pt",
    )
    tokenizer.save(out / "tokenizer.json")
    run = {
        "status": "completed",
        "config": asdict(config),
        "model_config": asdict(model_config),
        "parameters": model.num_parameters(),
        "trained_tokens": completed * config.tokens_per_update,
        "initial_dev": history[0]["dev"],
        "final_dev": history[-1]["dev"],
        "test": test,
        "training_seconds": training_seconds,
        "training_tokens_per_second": completed
        * config.tokens_per_update
        / training_seconds,
        "segments": segments
        + [
            {
                "from_update": segment_start_update,
                "to_update": completed,
                "wall_seconds": time.perf_counter() - segment_started,
            }
        ],
        "provenance": provenance,
        "source_hashes": source_hashes,
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
            "optimizer": config.optimizer,
            "fused_optimizer": bool(optimizer_options),
            "precision": config.precision,
            "tf32": False,
        },
        "scope": "Fixed-token-budget subset engineering comparison; not SOTA or a full-data benchmark",
        "artifacts": {
            name: file_sha256(out / name)
            for name in ("model.pt", "tokenizer.json", "checkpoint.pt", "history.json")
        },
    }
    json_write(out / "run.json", run)
    return run
