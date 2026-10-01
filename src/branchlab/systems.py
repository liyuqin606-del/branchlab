"""Same-weight, same-token measurements of three decoding implementations.

This module measures uncached, concatenating-cache and preallocated-cache paths.
Raw samples, numerical errors and actual buffer bytes accompany every timing.
SDPA is a library backend, never described as a custom FlashAttention kernel.
"""

from __future__ import annotations

import hashlib
import math
import statistics
import time

import torch
import torch.nn.functional as F

from .model import StaticKVCache
from .training import synchronize


@torch.inference_mode()
def benchmark_paths(model, prompt, decode_tokens=64, repeats=7):
    if prompt.ndim != 2 or min(prompt.shape) < 1:
        raise ValueError("Prompt must be a nonempty batch of token sequences")
    if decode_tokens < 1 or repeats < 1:
        raise ValueError("decode_tokens/repeats must be positive")
    if prompt.shape[1] + decode_tokens > model.config.max_seq_len:
        raise ValueError("Prompt plus decode budget exceeds model context")
    parameter = next(model.parameters())
    device = parameter.device
    if prompt.device != device:
        raise ValueError("Prompt/model device mismatch")
    was_training = model.training
    model.eval()
    modes = ("uncached", "dynamic", "static")
    capacity = prompt.shape[1] + decode_tokens
    continuation = (
        prompt[:, -1:] + torch.arange(1, decode_tokens + 1, device=device)
    ) % model.config.vocab_size
    sequence = torch.cat([prompt, continuation], dim=1)
    results = {
        mode: {
            "prefill_seconds_samples": [],
            "decode_seconds_samples": [],
            "peak_allocated_bytes_samples": [],
            "cache_allocated_bytes": 0,
        }
        for mode in modes
    }

    def prefill(mode):
        cache = (
            model.allocate_cache(prompt.shape[0], capacity)
            if mode == "static"
            else None
        )
        _, cache = model(prompt, cache, use_cache=mode != "uncached")
        return cache

    def cache_bytes(cache):
        if cache is None:
            return 0
        return sum(
            c.allocated_bytes
            if isinstance(c, StaticKVCache)
            else sum(t.numel() * t.element_size() for t in c)
            for c in cache
        )

    def decode(mode, cache, record=False):
        values = []
        for i in range(decode_tokens):
            current = (
                sequence[:, : prompt.shape[1] + i + 1]
                if mode == "uncached"
                else continuation[:, i : i + 1]
            )
            logits, cache = model(current, cache, use_cache=mode != "uncached")
            if record:
                values.append(logits[:, -1].clone())
        return (torch.stack(values, dim=1) if record else None), cache

    def measure(call):
        synchronize(device)
        started = time.perf_counter()
        value = call()
        synchronize(device)
        return time.perf_counter() - started, value

    try:
        # Full path warmup and numerical verification happen before any timings.
        reference, _ = decode("uncached", None, record=True)
        for mode in ("dynamic", "static"):
            actual, _ = decode(mode, prefill(mode), record=True)
            difference = float((reference.float() - actual.float()).abs().max())
            kl = float(
                F.kl_div(
                    actual.float().log_softmax(-1),
                    reference.float().softmax(-1),
                    reduction="none",
                )
                .sum(-1)
                .mean()
                .clamp_min(0)
            )
            argmax_difference = float(
                (reference.argmax(-1) != actual.argmax(-1)).float().mean()
            )
            tolerance = (
                2e-3 if parameter.dtype in (torch.bfloat16, torch.float16) else 1e-6
            )
            if not math.isfinite(difference) or not math.isfinite(kl) or kl > tolerance:
                raise AssertionError(
                    f"{mode} numerical parity failed: max_diff={difference}, mean KL={kl}"
                )
            results[mode]["parity"] = {
                "logits_max_abs_difference": difference,
                "mean_distribution_kl": kl,
                "argmax_disagreement_fraction": argmax_difference,
                "kl_tolerance": tolerance,
            }
        del reference, actual
        synchronize(device)
        for repeat in range(repeats):
            # Rotate all three paths to reduce consistent execution-order bias.
            order = modes[repeat % 3 :] + modes[: repeat % 3]
            for mode in order:
                prefill_seconds, cache = measure(lambda: prefill(mode))
                results[mode]["prefill_seconds_samples"].append(prefill_seconds)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                decode_seconds, (_, cache) = measure(
                    lambda mode=mode, cache=cache: decode(mode, cache)
                )
                results[mode]["decode_seconds_samples"].append(decode_seconds)
                if device.type == "cuda":
                    results[mode]["peak_allocated_bytes_samples"].append(
                        torch.cuda.max_memory_allocated(device)
                    )
                results[mode]["cache_allocated_bytes"] = cache_bytes(cache)
                del cache
        token_count = prompt.shape[0] * decode_tokens
        for mode, values in results.items():
            prefill_median = statistics.median(values["prefill_seconds_samples"])
            decode_median = statistics.median(values["decode_seconds_samples"])
            values.update(
                prefill_median_seconds=prefill_median,
                decode_median_seconds=decode_median,
                decode_tokens_per_second=token_count / decode_median,
                decode_ms_per_batch_step=decode_median * 1000 / decode_tokens,
                prefill_plus_decode_seconds=prefill_median + decode_median
                if mode != "uncached"
                else decode_median,
                peak_allocated_bytes=max(
                    values["peak_allocated_bytes_samples"], default=None
                ),
            )
        for values in results.values():
            values["speedup_vs_uncached"] = (
                results["uncached"]["decode_median_seconds"]
                / values["decode_median_seconds"]
            )
        return {
            "paths": results,
            "metadata": {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
                "torch": str(torch.__version__),
                "dtype": str(parameter.dtype),
                "batch_size": prompt.shape[0],
                "prompt_tokens": prompt.shape[1],
                "decode_tokens": decode_tokens,
                "repeats": repeats,
                "attention_backend": model.config.attention_backend,
                "query_heads": model.config.n_heads,
                "kv_heads": model.config.kv_heads,
                "parameters": model.num_parameters(),
                "prompt_sha256": hashlib.sha256(
                    prompt.cpu().numpy().tobytes()
                ).hexdigest(),
                "continuation": "Fixed arithmetic tokens; same continuation for all paths",
                "timing_scope": "Prefill includes cache allocation; decode excludes prefill; synchronized wall time",
                "memory_scope": "Absolute CUDA live allocation, includes model/inputs and live cache; CPU/MPS unavailable",
                "cache_memory_scope": "Actual persistent K/V tensor storage, including unused static capacity",
                "warmup": "One complete correctness run per path; rotating timing order",
            },
        }
    finally:
        model.train(was_training)
