#!/usr/bin/env python3
"""Recompute final held-out loss from saved inference weights and prepared tokens.

This is an evaluation replay, not retraining or independent scientific validation.
The default window traversal is intentionally independent of fit.evaluate_tokens.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from branchlab.cli import load_inference
from branchlab.fit import FitConfig, autocast_context
from branchlab.portfolio_data import file_sha256
from branchlab.training import json_write, resolve_device


@torch.inference_mode()
def replay(model_dir, prepared, device):
    model_dir, prepared = Path(model_dir), Path(prepared)
    recorded = json.loads((model_dir / "run.json").read_text())
    config = FitConfig(**recorded["config"])
    assert (
        file_sha256(prepared / "test_tokens.npy")
        == recorded["provenance"]["test_tokens.npy"]
    )
    assert file_sha256(model_dir / "model.pt") == recorded["artifacts"]["model.pt"]
    device = resolve_device(device)
    model, _ = load_inference(model_dir, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokens = np.load(prepared / "test_tokens.npy", mmap_mode="r")
    # One window per forward, rather than the trainer's batched evaluation loop.
    # Each target position appears exactly once; context resets at seq_len.
    total, count = 0.0, 0
    for start in range(0, config.eval_tokens, config.seq_len):
        length = min(config.seq_len, config.eval_tokens - start)
        x = torch.tensor(
            np.array(tokens[start : start + length]), device=device
        ).unsqueeze(0)
        y = torch.tensor(
            np.array(tokens[start + 1 : start + length + 1]), device=device
        )
        with autocast_context(device, config.precision):
            logits, _ = model(x)
        total += float(F.cross_entropy(logits.float()[0], y, reduction="sum"))
        count += length
    value = total / count
    difference = abs(value - recorded["test"]["cross_entropy"])
    tolerance = 2e-3 if config.precision == "bf16" else 2e-5
    if not math.isfinite(value) or difference > tolerance:
        raise AssertionError(
            f"Evaluation replay disagrees: {value}, delta={difference}, tolerance={tolerance}"
        )
    return {
        "model_sha256": file_sha256(model_dir / "model.pt"),
        "test_tokens_sha256": file_sha256(prepared / "test_tokens.npy"),
        "replayed_cross_entropy": value,
        "recorded_cross_entropy": recorded["test"]["cross_entropy"],
        "absolute_difference": difference,
        "tolerance": tolerance,
        "target_tokens": count,
        "device": str(device),
        "precision": config.precision,
        "torch": str(torch.__version__),
        "scope": "Independent single-window traversal from saved weights; same held-out text, not retraining",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="artifacts/portfolio")
    parser.add_argument("--prepared", default="artifacts/official-data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="reports/portfolio/evaluation_replay.json")
    args = parser.parse_args()
    result = {
        path.name: replay(path, args.prepared, args.device)
        for path in sorted(Path(args.results).glob("*"))
        if (path / "run.json").exists()
    }
    json_write(args.output, result)
    print(json.dumps(result, indent=2))
